import sys
from flask import Flask, Response
import cv2

app = Flask(__name__)

def get_gstreamer_pipeline(sensor_id=0, width=640, height=480, framerate=30):
    """USB 카메라용 GStreamer 파이프라인 (V4L2)"""
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def get_camera(sensor_id=0, width=640, height=480, framerate=30):
    """카메라 객체를 생성하여 반환합니다."""
    if sys.platform == 'win32':
        # 윈도우 환경에서는 일반 웹캠 사용 (GStreamer 미사용)
        print(f"[INFO] 윈도우 환경 감지됨: 웹캠(장치 {sensor_id}) 연결 시도...")
        cap = cv2.VideoCapture(sensor_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(sensor_id) # DSHOW 실패시 기본 백엔드
        
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap
    else:
        # Jetson / Linux 환경 (GStreamer 파이프라인 사용)
        print(f"[INFO] Linux 환경 감지됨: GStreamer 파이프라인 연결 시도...")
        pipeline = get_gstreamer_pipeline(sensor_id, width, height, framerate)
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

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
    return """
    <html>
        <head><title>Jetson Parking Camera</title></head>
        <body style="background-color: #222; color: white; text-align: center;">
            <h2>Jetson Orin Nano - Live Camera Stream</h2>
            <img src="/video_feed" width="640" height="480">
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
