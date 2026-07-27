import cv2
import time
import sys
from ultralytics import YOLO

# ============================================================
# COCO 데이터셋 기준 차량 관련 클래스 ID
# 2: car, 3: motorcycle, 5: bus, 7: truck
# ============================================================
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Bounding Box 색상 (BGR)
BOX_COLORS = {
    2: (0, 255, 0),    # Car       - 초록
    3: (255, 165, 0),  # Motorcycle - 주황
    5: (255, 0, 0),    # Bus       - 파랑
    7: (0, 0, 255),    # Truck     - 빨강
}
DEFAULT_COLOR = (0, 255, 255)


# ============================================================
# GStreamer 파이프라인 (1단계 camera_input.py 와 동일)
# ============================================================
def get_usb_pipeline(sensor_id=0, width=640, height=480, framerate=30):
    """USB 카메라용 GStreamer 파이프라인 (V4L2)"""
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def get_csi_pipeline(sensor_id=0, width=1280, height=720, framerate=30):
    """CSI 카메라용 GStreamer 파이프라인 (nvarguscamerasrc)"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def get_rtsp_pipeline(uri):
    """IP 카메라(RTSP)용 GStreamer 파이프라인"""
    return (
        f"uridecodebin uri={uri} ! "
        f"nvvideoconvert ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )


def open_camera(cam_type='1'):
    """카메라 타입에 따라 VideoCapture 객체 반환"""
    if cam_type == '1':
        print("[INFO] USB 카메라를 설정합니다...")
        pipeline = get_usb_pipeline(sensor_id=0, width=640, height=480, framerate=30)
    elif cam_type == '2':
        print("[INFO] CSI 카메라를 설정합니다...")
        pipeline = get_csi_pipeline(sensor_id=0, width=1280, height=720, framerate=30)
    elif cam_type == '3':
        print("[INFO] IP 카메라(RTSP)를 설정합니다...")
        rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream"  # 실제 URL로 변경 필요
        print(f"[INFO] RTSP URL: {rtsp_url}")
        pipeline = get_rtsp_pipeline(rtsp_url)
    else:
        print("[WARN] 잘못된 입력입니다. 기본값인 USB 카메라로 진행합니다.")
        pipeline = get_usb_pipeline()

    print(f"[INFO] GStreamer 파이프라인:\n  {pipeline}\n")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


def draw_vehicle_boxes(frame, results):
    """
    YOLO 추론 결과에서 차량 클래스만 필터링하여
    Bounding Box와 Confidence를 프레임에 그린다.
    검출된 차량 정보 리스트를 반환한다.
    """
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
            w = x2 - x1
            h = y2 - y1

            class_name = VEHICLE_CLASS_NAMES.get(cls_id, "Vehicle")
            color = BOX_COLORS.get(cls_id, DEFAULT_COLOR)

            # Bounding Box 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 라벨 배경
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - baseline - 4), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            detections.append({
                "class": class_name,
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "confidence": conf,
            })

    return detections


def main():
    print("==========================================")
    print(" 2단계 : 차량 객체 검출 (Vehicle Detection)")
    print("==========================================")
    print("카메라 타입 선택:")
    print("  1: USB 카메라 (기본값)")
    print("  2: CSI 카메라")
    print("  3: IP 카메라 (RTSP)")

    cam_type = sys.argv[1] if len(sys.argv) > 1 else '1'

    # ----------------------------------------------------------
    # YOLOv8n 모델 로드 (COCO Pretrained)
    # ----------------------------------------------------------
    print("\n[INFO] YOLOv8n COCO Pretrained 모델을 로드합니다...")
    model = YOLO("yolov8n.pt")
    print("[INFO] 모델 로드 완료.\n")

    # ----------------------------------------------------------
    # 카메라 열기
    # ----------------------------------------------------------
    cap = open_camera(cam_type)

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        return

    print("[INFO] 카메라가 열렸습니다. 차량 검출을 시작합니다.")
    print("[INFO] 종료: 영상 창에서 'q' 키\n")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽어올 수 없습니다.")
            break

        # ----------------------------------------------------------
        # YOLOv8 추론
        # ----------------------------------------------------------
        results = model(frame, verbose=False)

        # ----------------------------------------------------------
        # 차량만 필터링하여 Bounding Box 그리기
        # ----------------------------------------------------------
        detections = draw_vehicle_boxes(frame, results)

        # ----------------------------------------------------------
        # FPS 계산 (0.5초 간격 갱신)
        # ----------------------------------------------------------
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 0.5:
            fps = frame_count / (current_time - prev_time)
            prev_time = current_time
            frame_count = 0

        # 화면 상단에 FPS 표시
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 검출 차량 수 표시
        cv2.putText(frame, f"Vehicles: {len(detections)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # 터미널에 검출 결과 출력 (5프레임마다)
        if frame_count == 1 and detections:
            print(f"[FPS: {fps:.1f}] 검출 차량 {len(detections)}대:")
            for d in detections:
                print(f"  {d['class']:12s}  x={d['x']:<4d} y={d['y']:<4d} w={d['w']:<4d} h={d['h']:<4d}  conf={d['confidence']:.2f}")

        # ----------------------------------------------------------
        # 영상 출력
        # ----------------------------------------------------------
        cv2.imshow("Vehicle Detection - YOLOv8n", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] 사용자에 의해 종료되었습니다.")
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
