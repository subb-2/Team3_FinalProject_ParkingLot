from flask import Flask, Response
import cv2

app = Flask(__name__)

# GStreamer 파이프라인 설정 (네가 쓰던 것 그대로 유지)
gstreamer_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=(int)640, height=(int)480, framerate=(fraction)30/1 ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! "
    "appsink"
)

# 카메라 객체 생성
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

def generate_frames():
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
    # 0.0.0.0으로 열어야 윈도우 PC(외부)에서 접속 가능해!
    app.run(host='0.0.0.0', port=5000, debug=False)
