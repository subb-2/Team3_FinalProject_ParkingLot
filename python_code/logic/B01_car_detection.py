import cv2
import time
import sys
import os
from ultralytics import YOLO
from flask import Flask, Response

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B00_camera_input import get_camera

# 설정 (Configuration)
# 검출 관련 설정의 단일 출처.
CONFIG = {
    # 모델 추론 설정
    "MODEL_PATH": "/home/eojin/parking_manage_python/dataset/best_0811.engine", # 엔진 모델 경로 ('yolov8s.pt'로 변경 시 일반 파이토치 모델 사용)

    # Confidence 임계값을 일부러 낮게 둠.
    # 추적기의 2차 연관(ByteTrack 방식)은 신뢰도가 낮은 박스를 재매칭해서 끊긴
    # 추적을 살리는데, 여기서 미리 걸러버리면 그 박스가 추적기까지 오지 못해
    # 기능이 죽는다.
    #
    # 이 값은 config/ocsort.yaml의 track_low_thresh(0.05)보다
    # 낮거나 같아야 한다. 0.25로 두면 저신뢰 버킷 [0.05, 0.4) 중 실제로는
    # [0.25, 0.4)만 존재하게 되어, 손에 가려져 conf가 0.1~0.2로 떨어진 박스가
    # 추적기에 도달조차 못 한다. (가림 구간 ID 스위치의 주된 원인)
    #
    # 실제 필터링은 추적기 쪽의 track_high_thresh / new_track_thresh(0.4)가 담당.
    "CONF_THRESH": 0.05,            # Confidence 임계값
    "IOU_THRESH": 0.45,             # NMS IoU 임계값

    # YOLO 추론 해상도(정사각 한 변, px). 32의 배수여야 함.
    # 카메라 해상도와는 별개다. 입력 프레임이 얼마든 YOLO는 이 크기로 리사이즈해서 추론.
    #   키우면 : 작은 물체(멀리 있는 차, 장난감 차)를 더 잘 잡지만 FPS가 떨어짐.
    #   줄이면 : 빨라지지만 작은 물체를 놓친다
    #
    # MODEL_PATH가 .engine(TensorRT)이면 이 값을 마음대로 바꿀 수 없다.
    # 엔진은 빌드할 때 입력 크기가 고정되므로, 다른 값을 주면 추론 시 매 프레임 AssertionError가 나고 스트리밍이 끊긴다.
    # 즉, 해상도를 올리려면 엔진을 그 크기로 다시 export해야 한다.
    #   yolo export model=yolov8s.pt format=engine imgsz=1280
    # (.pt 모델은 이 제약이 없지만 훨씬 느리다)
    "IMGSZ": 640,

    # 카메라 설정 (B01 단독 실행에만 사용).
    # 주의: B_main / C_main은 각자의 CONFIG에 있는 CAM_* 값을 쓴다.
    #       여기를 고쳐도 그쪽 해상도는 바뀌지 않는다.
    "CAM_WIDTH": 1280,              # 카메라 가로 해상도 (C_main의 설명 참고)
    "CAM_HEIGHT": 800,              # 카메라 세로 해상도
    "CAM_FPS": 30                   # 카메라 프레임레이트
}

# 검출 클래스.
# 직접 촬영해 라벨링한 데이터셋으로 파인튜닝한 1클래스 모델이라 car 하나뿐이다.
# COCO 사전학습 모델로 되돌리려면 {2: car, 3: motorcycle, 5: bus, 7: truck}으로
# 바꾸고 BOX_COLORS도 함께 늘릴 것.
VEHICLE_CLASS_IDS = {0}
VEHICLE_CLASS_NAMES = {0: "Car"}

# Bounding Box 색상 (BGR)
BOX_COLORS = {
    0: (0, 255, 0),  # Car - 초록색
}
DEFAULT_COLOR = (0, 255, 255)


class CarDetector:
    """
    YOLOv8 기반 차량 검출기.

    검출만 담당한다. 추적은 B02_car_mot이 맡고, 이 클래스는 검출 결과를
    돌려주기만 한다. 모델은 프로젝트 전체에서 여기 한 곳에서만 로드한다.
    """

    def __init__(self, model_path=None, conf=None, iou=None, imgsz=None):
        """
        CarDetector 초기화.

        모든 인자의 기본값은 이 모듈 상단의 CONFIG에서 가져온다.
        같은 값을 두 곳에 적어두면 어느 쪽이 실제로 쓰이는지 알 수 없게 되므로,
        여기에 숫자를 직접 적지 말 것. (MAIN_README 5절)

        Args:
            model_path: YOLOv8 모델 파일 경로 (None이면 CONFIG['MODEL_PATH'])
            conf:       Confidence 임계값      (None이면 CONFIG['CONF_THRESH'])
            iou:        NMS IoU 임계값         (None이면 CONFIG['IOU_THRESH'])
            imgsz:      추론 해상도 (px)       (None이면 CONFIG['IMGSZ'])
        """
        self.model_path = CONFIG['MODEL_PATH'] if model_path is None else model_path
        self.conf = CONFIG['CONF_THRESH'] if conf is None else conf
        self.iou = CONFIG['IOU_THRESH'] if iou is None else iou
        self.imgsz = CONFIG['IMGSZ'] if imgsz is None else imgsz

        print(f"[INFO] YOLOv8s 모델을 로드합니다... ({self.model_path})")
        self.model = YOLO(self.model_path, task='detect')
        print(f"[INFO] 모델 로드 완료. (추론 해상도 {self.imgsz}, "
              f"conf {self.conf}, iou {self.iou})")

    def detect(self, frame):
        """
        프레임에서 차량 객체를 검출.
        
        Args:
            frame: OpenCV BGR 이미지 (numpy array)
        
        Returns:
            검출 결과 리스트. 각 항목은 딕셔너리:
            [
                {
                    "class_id": 2,
                    "class_name": "Car",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.94
                },
                ...
            ]
        """

        # YOLO 추론 실행
        results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])

                # 차량 클래스만 필터링
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class_id": cls_id,
                    "class_name": VEHICLE_CLASS_NAMES.get(cls_id, "Vehicle"),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })

        return detections

    def draw_detections(self, frame, detections):
        """
        검출 결과를 프레임에 Bounding Box와 라벨로 시각화.
        
        Args:
            frame:      OpenCV BGR 이미지 (numpy array, 원본이 수정됨)
            detections: detect() 메서드의 반환 결과 리스트
        
        Returns:
            시각화가 적용된 프레임 (입력 frame과 동일 객체)
        """
        for det in detections:
            cls_id = det["class_id"]
            class_name = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]

            color = BOX_COLORS.get(cls_id, DEFAULT_COLOR)

            # Bounding Box 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 라벨 배경 및 텍스트
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - baseline - 4),
                (x1 + label_w, y1),
                color, -1
            )
            cv2.putText(
                frame, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

        return frame

# 테스트용 메인 (단독 실행 시 웹캠 서버 열기)
if __name__ == '__main__':
    print("==========================================")
    print(" B01 : 차량 객체 검출 (Car Detection)")
    print(f" 모델 : {CONFIG['MODEL_PATH']}")
    print("==========================================")

    # 카메라 열기 (B00_camera_input 모듈 활용)
    print(f"[INFO] B00_camera_input 모듈을 통해 카메라를 엽니다... ({CONFIG['CAM_WIDTH']}x{CONFIG['CAM_HEIGHT']})")
    cap = get_camera(sensor_id=0, width=CONFIG['CAM_WIDTH'], height=CONFIG['CAM_HEIGHT'], framerate=CONFIG['CAM_FPS'])

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        sys.exit(1)

    # 검출기 초기화
    detector = CarDetector(
        model_path=CONFIG['MODEL_PATH'],
        conf=CONFIG['CONF_THRESH'],
        iou=CONFIG['IOU_THRESH'],
        imgsz=CONFIG['IMGSZ']
    )

    app = Flask(__name__)

    def generate_frames_with_detection():
        prev_time = time.time()
        frame_count = 0
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 검출 실행 및 시각화
            detections = detector.detect(frame)
            detector.draw_detections(frame, detections)

            # FPS 계산 (0.5초 간격)
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= 0.5:
                fps = frame_count / (current_time - prev_time)
                prev_time = current_time
                frame_count = 0

            # 화면 텍스트 표시
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Vehicles: {len(detections)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # 터미널에 검출 결과 출력 (FPS 갱신 시점마다)
            if frame_count == 1 and detections:
                print(f"[FPS: {fps:.1f}] 검출 차량 {len(detections)}대:")
                for d in detections:
                    print(f"  {d['class_name']:12s}  conf={d['confidence']:.2f}")

            # JPEG 압축 후 웹 스트리밍 반환
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    @app.route('/')
    def index():
        return f"""
        <html>
            <head><title>Jetson Car Detection</title></head>
            <body style="background-color: #222; color: white; text-align: center;">
                <h2>Jetson Orin Nano - Car Detection (YOLOv8s)</h2>
                <img src="/video_feed" width="{CONFIG['CAM_WIDTH']}" height="{CONFIG['CAM_HEIGHT']}">
            </body>
        </html>
        """

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames_with_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

    print("\n[INFO] Flask 웹 서버를 시작합니다. http://젯슨IP:5000/ 으로 접속하세요.")
    app.run(host='0.0.0.0', port=5000, debug=False)
