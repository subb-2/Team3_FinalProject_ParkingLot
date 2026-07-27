import cv2
import time

def get_usb_pipeline(sensor_id=0, width=640, height=480, framerate=30):
    """
    USB 카메라용 GStreamer 파이프라인 (V4L2)
    """
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    print("=====================================")
    print("Jetson Orin Nano - USB Camera Input")
    print("=====================================")
    
    print("USB 카메라를 설정합니다...")
    pipeline = get_usb_pipeline(sensor_id=0, width=640, height=480, framerate=30)
    
    print(f"\n적용된 GStreamer 파이프라인:\n{pipeline}\n")

    # OpenCV를 사용해 GStreamer 백엔드로 캡처 객체 생성
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다. 카메라 연결 상태나 파이프라인 설정을 확인하세요.")
        return

    print("카메라가 성공적으로 열렸습니다. 영상을 출력합니다.")
    print("종료하려면 영상 창을 선택하고 'q' 키를 누르세요.\n")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("오류: 프레임을 읽어올 수 없습니다.")
            break

        # FPS 계산
        current_time = time.time()
        frame_count += 1
        
        # 0.5초마다 FPS 갱신
        if current_time - prev_time >= 0.5:
            fps = frame_count / (current_time - prev_time)
            prev_time = current_time
            frame_count = 0
            
        # 프레임에 FPS 출력
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 화면 출력
        cv2.imshow("Jetson Camera Viewer", frame)

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자에 의해 종료되었습니다.")
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
