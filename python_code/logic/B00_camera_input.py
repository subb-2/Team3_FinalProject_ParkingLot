import sys
from flask import Flask, Response
import cv2

app = Flask(__name__)

# 기본 해상도.
#
# 640x480이 아니라 1280x720인 이유는 ArUco 마커 검출이다. 5x5 비트 마커는
# 테두리 포함 7x7 칸이고 OpenCV가 칸마다 4픽셀을 샘플링하므로 한 변이 최소
# 28px은 되어야 비트를 읽는다. 이 목업의 마커는 640x480에서 20px 안팎이라
# 하나도 검출되지 않는다. (실측: 640x480 0개 -> 1280x720 10개)
#
# 화각도 달라진다. 640x480은 4:3, 1280x720은 16:9라 가로로 더 넓게 담긴다.
# 카메라에 따라서는 저해상도에서 센서를 잘라 쓰기도 해서 차이가 더 커진다.
#
# 차량 검출/추적도 같은 프레임을 쓰므로 이 값 하나로 함께 결정된다.
# YOLO의 추론 해상도(B01의 IMGSZ)와는 별개다. 그쪽은 엔진이 640으로 고정이라
# 입력이 얼마든 내부에서 리사이즈한다.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


def get_gstreamer_pipeline(sensor_id=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                           framerate=DEFAULT_FPS):
    """USB 카메라용 GStreamer 파이프라인 (V4L2)"""
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


def get_camera(sensor_id=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
               framerate=DEFAULT_FPS):
    """
    카메라 객체를 생성하여 반환합니다.
    """
    if sys.platform == 'win32':
        # 윈도우 환경에서는 일반 웹캠 사용 (GStreamer 미사용)
        print(f"[INFO] 윈도우 환경 감지됨: 웹캠(장치 {sensor_id}) 연결 시도...")
        cap = cv2.VideoCapture(sensor_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(sensor_id) # DSHOW 실패시 기본 백엔드
        
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        # Jetson / Linux 환경 (GStreamer 파이프라인 사용)
        print(f"[INFO] Linux 환경 감지됨: GStreamer 파이프라인 연결 시도...")
        pipeline = get_gstreamer_pipeline(sensor_id, width, height, framerate)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    return cap

# 카메라 객체
cap = None

def generate_frames():
    global cap
    if cap is None or not cap.isOpened():
        cap = get_camera()
        
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # 프레임을 JPEG 형식으로 압축
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # 웹 스트리밍 형식(MJPEG)으로 변환하여 반환
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # 웹페이지에 띄울 간단한 HTML 구조
    return f"""
    <html>
        <head><title>Jetson Parking Camera</title></head>
        <body style="background-color: #222; color: white; text-align: center;">
            <h2>Jetson Orin Nano - Live Camera Stream</h2>
            <img src="/video_feed" width="{DEFAULT_WIDTH}" height="{DEFAULT_HEIGHT}">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    # 실시간 프레임 스트리밍 라우트
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 0.0.0.0으로 열어야 윈도우 PC(외부)에서 접속 가능
    app.run(host='0.0.0.0', port=5000, debug=False)
