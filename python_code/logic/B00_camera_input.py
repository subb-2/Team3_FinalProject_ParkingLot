import sys
import threading
import time
from flask import Flask, Response
import cv2

app = Flask(__name__)

# 기본 해상도.
#
# 4:3인 이유는 YOLO 학습 데이터가 4:3이기 때문이다. 자세한 근거는
# C_main의 CAM_WIDTH 주석에 적어 두었다. 요약하면, YOLO는 입력을 비율을
# 지킨 채 정사각으로 레터박스하므로 카메라와 학습 데이터의 '비율'이
# 맞아야 모델이 배운 것과 같은 모양의 입력을 받는다.
#
# 화각도 비율을 따라간다. 4:3과 16:9는 담기는 범위가 다르고, 카메라에
# 따라서는 해상도마다 센서를 잘라 쓰기도 한다.
#
# 차량 검출/추적도 같은 프레임을 쓰므로 이 값 하나로 함께 결정된다.
# YOLO의 추론 해상도(B01의 IMGSZ)와는 별개다. 그쪽은 엔진이 640으로 고정이라
# 입력이 얼마든 내부에서 리사이즈한다.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30

# MJPEG 전송 품질. OpenCV 기본값은 95라 1280x720 한 장이 200KB를 넘는다.
# 초당 20장이면 4MB/s로, 인코딩 비용도 네트워크도 이게 병목이 된다.
# 75면 화면상 차이는 거의 없고 용량은 1/3 수준으로 떨어진다.
JPEG_QUALITY = 75
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]


class LatestFrameCamera:
    """
    VideoCapture를 전담 스레드로 계속 비워서 '가장 최신 프레임'만 들고 있는 래퍼.

    왜 필요한가.
      드라이버는 카메라가 만든 프레임을 큐에 쌓아두고, cap.read()는 그중
      '가장 오래된' 것을 꺼낸다. 카메라는 30fps로 밀어넣는데 처리는 24fps면
      매초 6장씩 큐에 남아 지연이 계속 쌓인다. 몇 초 지나면 화면은 과거를
      보여주고, 큐가 찰 때마다 드라이버가 뭉텅이로 버려서 뚝뚝 끊겨 보인다.
      계산 FPS는 24로 멀쩡히 찍히는데 눈으로는 심하게 버벅이는 이유가 이것이다.

      여기서는 읽기 전용 스레드가 최대 속도로 큐를 비우고 최신 한 장만 남긴다.
      처리 쪽은 항상 '지금' 프레임을 받으므로 지연이 쌓이지 않는다.

    read()는 새 프레임이 들어올 때까지 기다렸다가 돌려준다. 같은 프레임을
    두 번 처리하며 CPU를 낭비하지 않게 하기 위해서다. 처리가 카메라보다
    느리면 이미 새 프레임이 있으므로 기다리지 않는다.

    VideoCapture와 같은 인터페이스(read/isOpened/release/set/get)를 제공하므로
    쓰는 쪽 코드는 그대로 두면 된다. 여러 곳(파이프라인, /snapshot)에서 동시에
    read()해도 서로 프레임을 뺏어가지 않는다.
    """

    def __init__(self, cap, read_timeout=2.0):
        self._cap = cap
        self._read_timeout = read_timeout

        self._frame = None
        self._seq = 0               # 프레임 일련번호. 새 프레임인지 판단하는 기준
        self._last_read_seq = 0     # 마지막으로 넘겨준 프레임 번호
        self._alive = True

        self.capture_fps = 0.0      # 카메라가 실제로 주고 있는 fps
        self.dropped = 0            # 처리가 못 따라가 버린 프레임 수(누적)
        self._cond = threading.Condition()
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        count, t0 = 0, time.time()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                with self._cond:
                    self._alive = False
                    self._cond.notify_all()
                break
            with self._cond:
                # 아직 아무도 안 가져간 프레임 위에 덮어쓰면 그건 버려진 것이다.
                # 이 수가 크면 처리 속도가 카메라를 못 따라가고 있다는 뜻.
                if self._seq != self._last_read_seq:
                    self.dropped += 1
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

            # 카메라가 실제로 몇 fps를 주고 있는지. 처리 FPS와 비교하면
            # 화면이 밀리는 원인이 카메라인지 처리인지 바로 갈린다.
            count += 1
            now = time.time()
            if now - t0 >= 1.0:
                self.capture_fps = count / (now - t0)
                count, t0 = 0, now

    def read(self, last_seq=None):
        """
        최신 프레임을 (성공여부, 프레임)으로 반환.

        아직 아무도 가져가지 않은 프레임이 있으면 곧바로 그것을 준다.
        (처리가 카메라보다 느린 정상 상황에서는 대기가 전혀 없다)
        이미 가져간 프레임뿐이면 다음 프레임이 들어올 때까지 기다린다.
        같은 장면을 두 번 처리하며 CPU를 태우지 않기 위해서다.

        last_seq를 주면 그 번호와 다른 프레임이 올 때까지 기다린다.
        (스트리밍 쪽에서 같은 장면을 다시 인코딩하지 않으려고 쓴다)
        """
        ok, frame, _ = self.read_seq(last_seq)
        return ok, frame

    def read_seq(self, last_seq=None):
        """read()와 같지만 프레임 일련번호까지 함께 돌려준다."""
        with self._cond:
            base = self._last_read_seq if last_seq is None else last_seq
            if self._seq == base and self._alive:
                self._cond.wait(self._read_timeout)
            self._last_read_seq = self._seq
            # 카메라가 끊겼는데 새 프레임도 없으면 실패로 알린다.
            # 그러지 않으면 부르는 쪽이 같은 프레임을 무한정 다시 처리한다.
            if self._frame is None or (not self._alive and self._seq == base):
                return False, None, self._seq
            return True, self._frame, self._seq

    def isOpened(self):
        return self._alive and self._cap.isOpened()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()

    def set(self, *args):
        return self._cap.set(*args)

    def get(self, *args):
        return self._cap.get(*args)


def get_gstreamer_pipeline(sensor_id=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                           framerate=DEFAULT_FPS):
    """USB 카메라용 GStreamer 파이프라인 (V4L2)"""
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"videoconvert ! "
        # drop=true max-buffers=1 : 처리가 밀리면 오래된 프레임을 버리고
        # 항상 최신 것만 넘긴다. 이게 없으면 appsink에 프레임이 쌓여
        # 화면이 몇 초씩 뒤처진다. sync=false는 타임스탬프에 맞춰
        # 기다리지 말고 오는 대로 내보내라는 뜻.
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def get_camera(sensor_id=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
               framerate=DEFAULT_FPS, latest_only=True):
    """
    카메라 객체를 생성하여 반환합니다.

    latest_only=True면 LatestFrameCamera로 감싸서 돌려준다. 드라이버 큐에
    프레임이 쌓여 화면이 밀리는 것을 막는다. (기본값)
    """
    if sys.platform == 'win32':
        # 윈도우 환경에서는 일반 웹캠 사용 (GStreamer 미사용)
        print(f"[INFO] 윈도우 환경 감지됨: 웹캠(장치 {sensor_id}) 연결 시도...")
        cap = cv2.VideoCapture(sensor_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(sensor_id) # DSHOW 실패시 기본 백엔드

        # 해상도보다 먼저 코덱을 정한다.
        # 대부분의 USB 웹캠은 무압축(YUY2)으로는 1280x720에서 10fps밖에 못 낸다.
        # USB 2.0 대역폭이 모자라기 때문이다. MJPG로 받으면 같은 해상도에서
        # 30fps가 나온다. 순서가 중요하다. 해상도를 먼저 잡으면 무시되기도 한다.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, framerate)
    else:
        # Jetson / Linux 환경 (GStreamer 파이프라인 사용)
        print(f"[INFO] Linux 환경 감지됨: GStreamer 파이프라인 연결 시도...")
        pipeline = get_gstreamer_pipeline(sensor_id, width, height, framerate)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # 드라이버 큐를 최소로. 백엔드가 지원하지 않으면 무시된다.
    # (그래서 LatestFrameCamera가 따로 필요하다)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if latest_only and cap.isOpened():
        return LatestFrameCamera(cap)
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
            ret, buffer = cv2.imencode('.jpg', frame, JPEG_PARAMS)
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
