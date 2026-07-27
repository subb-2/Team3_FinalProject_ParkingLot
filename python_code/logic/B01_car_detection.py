import cv2
import time
import sys
import os
from ultralytics import YOLO

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B00_camera_input import get_camera

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

    def __init__(self, model_path='yolov8s.pt', conf=0.5, iou=0.45):
        """
        CarDetector 초기화.
        
        Args:
            model_path: YOLOv8 모델 파일 경로 (기본값: yolov8s.pt, 첫 실행 시 자동 다운로드)
            conf:       Confidence 임계값 (기본값: 0.5)
            iou:        NMS IoU 임계값 (기본값: 0.45)
        """
        print(f"[INFO] YOLOv8s 모델을 로드합니다... ({model_path})")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
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
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)

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

# 테스트용 메인 (단독 실행 시 카메라 연결하여 검출 확인)
if __name__ == '__main__':
    print("==========================================")
    print(" B01 : 차량 객체 검출 (Car Detection)")
    print(" 모델 : YOLOv8s (COCO Pretrained)")
    print("==========================================")

    # 카메라 열기 (B00_camera_input 모듈 활용)
    print("[INFO] B00_camera_input 모듈을 통해 카메라를 엽니다...")
    cap = get_camera(sensor_id=0, width=640, height=480, framerate=30)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        sys.exit(1)

    print("[INFO] 카메라가 열렸습니다. 차량 검출을 시작합니다.")
    print("[INFO] 종료: 영상 창에서 'q' 키\n")

    # 검출기 초기화
    detector = CarDetector(model_path='yolov8s.pt', conf=0.5, iou=0.45)

    # 메인 루프
    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽어올 수 없습니다.")
            break

        # 검출 실행
        detections = detector.detect(frame)

        # 검출 결과 시각화
        detector.draw_detections(frame, detections)

        # FPS 계산 (0.5초 간격 갱신)
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 0.5:
            fps = frame_count / (current_time - prev_time)
            prev_time = current_time
            frame_count = 0

        # 화면 상단에 FPS 표시
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

        # 검출 차량 수 표시
        cv2.putText(
            frame, f"Vehicles: {len(detections)}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
        )

        # 터미널에 검출 결과 출력 (FPS 갱신 시점마다)
        if frame_count == 1 and detections:
            print(f"[FPS: {fps:.1f}] 검출 차량 {len(detections)}대:")
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                w = x2 - x1
                h = y2 - y1
                print(f"  {d['class_name']:12s}  x={x1:<4d} y={y1:<4d} w={w:<4d} h={h:<4d}  conf={d['confidence']:.2f}")

        # 영상 출력
        cv2.imshow("B01 Car Detection - YOLOv8s", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] 사용자에 의해 종료되었습니다.")
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
