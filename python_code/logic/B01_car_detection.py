import cv2
import time
import sys
import os
from ultralytics import YOLO
from flask import Flask, Response

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B00_camera_input import get_camera

# ---------------------------------------------------------
# 설정 (Configuration)
# ---------------------------------------------------------
CONFIG = {
    # 모델 추론 설정
    "MODEL_PATH": "yolov8s.engine", # 엔진 모델 경로 ('yolov8s.pt'로 변경 시 일반 파이토치 모델 사용)
    "CONF_THRESH": 0.5,             # Confidence 임계값
    "IOU_THRESH": 0.45,             # NMS IoU 임계값
    "IMGSZ": 1280,                  # YOLO 추론 해상도 (기본 640 -> 1280으로 상향)

    # 카메라 설정
    "CAM_WIDTH": 1280,              # 카메라 가로 해상도 (기본 640 -> 1280으로 상향)
    "CAM_HEIGHT": 720,              # 카메라 세로 해상도 (기본 480 -> 720으로 상향)
    "CAM_FPS": 30                   # 카메라 프레임레이트
}

# COCO 데이터셋 기준 차량 관련 클래스 ID
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Bounding Box 색상 (BGR)
BOX_COLORS = {
    2: (0, 255, 0),    # Car       - 초록
    3: (0, 165, 255),  # Motorcycle - 주황
    5: (255, 0, 0),    # Bus       - 파랑
    7: (0, 0, 255),    # Truck     - 빨강
}
DEFAULT_COLOR = (0, 255, 255)


# CarDetector 클래스 - 차량 검출 전용
class CarDetector:
    """
    YOLOv8s COCO Pretrained 모델을 사용한 차량 검출 클래스.
    검출 전용이며, 추적(Tracking)은 포함하지 않음.
    """

    def __init__(self, model_path='yolov8s.engine', conf=0.5, iou=0.45, imgsz=1280):
        """
        CarDetector 초기화.
        
        Args:
            model_path: YOLOv8 모델 파일 경로
            conf:       Confidence 임계값
            iou:        NMS IoU 임계값
            imgsz:      추론 해상도 (입력 크기)
        """
        print(f"[INFO] YOLOv8s 모델을 로드합니다... ({model_path})")
        self.model = YOLO(model_path, task='detect')
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        print("[INFO] 모델 로드 완료.")

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
    print(" 모델 : YOLOv8s (COCO Pretrained)")
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
