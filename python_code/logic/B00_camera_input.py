import sys
import threading
import time
from flask import Flask, Response
import cv2

app = Flask(__name__)

# 기본 해상도.
#
# 두 가지가 서로 당긴다.
#
#   1) 주차장 전체가 화면에 들어와야 한다. (반드시)
#      안 들어오면 기둥을 못 찍어 보정 자체가 안 된다.
#   2) 학습 데이터와 가로세로비가 같아야 좋다. (권장)
#      YOLO는 입력을 비율 그대로 정사각에 레터박스하므로, 비율이 다르면
#      모델이 배운 것과 다른 모양이 들어간다. 학습 데이터는 4:3이다.
#      자세한 근거는 C_main의 CAM_WIDTH 주석 참고.
#
# 실측: 이 목업에서 4:3(640x480)으로 내리니 주차장 좌우가 잘렸다.
# 카메라가 4:3에서 센서를 잘라 쓰기 때문이다. 그래서 1)이 이긴다.
# 4:3보다 넓으면서 4:3에 가장 가까운 비율부터 차례로 시도한다.
#
#   16:10 (1.60) -> 16:9 (1.78) -> 4:3 (1.33)
#
# 차량 검출/추적도 같은 프레임을 쓰므로 이 값 하나로 함께 결정된다.
# YOLO의 추론 해상도(B01의 IMGSZ)와는 별개다. 그쪽은 엔진이 640으로 고정이라
# 입력이 얼마든 내부에서 리사이즈한다.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 800
DEFAULT_FPS = 30

# 위 해상도를 카메라가 받아주지 않을 때 대신 시도할 목록.
#
# 젯슨의 v4l2src는 해상도를 caps로 못 박기 때문에, 지원하지 않는 값을 주면
# 파이프라인이 아예 열리지 않는다. 예전에는 그대로 죽어서 '카메라 없음'으로만
# 보였다. 카메라가 실제로 내주는 모드는 아래로 확인할 수 있다.
#   v4l2-ctl --list-formats-ext -d /dev/video0
FALLBACK_SIZES = [
    (1280, 800),    # 16:10 - 학습(4:3)에 가장 가까우면서 4:3보다 넓다
    (1280, 720),    # 16:9  - 이 목업에서 주차장이 들어오는 것이 확인된 값
    (640, 480),     # 4:3   - 비율은 학습과 같지만 좌우가 잘린다
]

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


def _try_open(sensor_id, width, height, framerate):
    """
    한 해상도로 카메라를 열어 본다.

    '열렸다'로 끝내지 않고 한 장을 실제로 읽어 확인한다. 윈도우 쪽
    VideoCapture.set은 지원하지 않는 해상도를 조용히 무시하고 다른 크기를
    내주기 때문에, 읽어 봐야 실제로 무엇이 오는지 알 수 있다.

    Returns:
        (cap, 실제_가로, 실제_세로). 실패하면 None.
    """
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(sensor_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(sensor_id)   # DSHOW 실패시 기본 백엔드
        if not cap.isOpened():
            return None

        # 해상도보다 먼저 코덱을 정한다.
        # 대부분의 USB 웹캠은 무압축(YUY2)으로는 1280x720에서 10fps밖에 못 낸다.
        # USB 2.0 대역폭이 모자라기 때문이다. MJPG로 받으면 같은 해상도에서
        # 30fps가 나온다. 순서가 중요하다. 해상도를 먼저 잡으면 무시되기도 한다.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, framerate)
    else:
        # Jetson / Linux (GStreamer). v4l2src는 해상도를 caps로 못 박으므로
        # 지원하지 않는 값이면 여기서 아예 열리지 않는다.
        pipeline = get_gstreamer_pipeline(sensor_id, width, height, framerate)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            return None

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap, frame.shape[1], frame.shape[0]


def get_camera(sensor_id=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
               framerate=DEFAULT_FPS, latest_only=True):
    """
    카메라 객체를 생성하여 반환합니다.

    요청한 해상도를 카메라가 받아주지 않으면 FALLBACK_SIZES를 차례로
    시도한다. 예전에는 첫 시도가 실패하면 그대로 '카메라 없음'이 되어,
    카메라는 멀쩡한데 해상도만 안 맞는 경우를 구별할 수 없었다.

    실제로 잡힌 해상도와 가로세로비를 반드시 출력한다. 요청과 다른 값이
    잡히면 화각이 달라져 보정과 검출이 함께 어긋나므로, 조용히 넘어가면
    나중에 원인을 찾기 어렵다.

    latest_only=True면 LatestFrameCamera로 감싸서 돌려준다. 드라이버 큐에
    프레임이 쌓여 화면이 밀리는 것을 막는다. (기본값)
    """
    env = "윈도우 웹캠" if sys.platform == 'win32' else "Linux GStreamer"
    print(f"[INFO] 카메라 연결 시도... ({env}, 장치 {sensor_id})")

    # 요청값을 맨 앞에 두고, 중복 없이 후보를 잇는다
    candidates = [(width, height)]
    candidates += [wh for wh in FALLBACK_SIZES if wh != (width, height)]

    opened = None
    for cand_w, cand_h in candidates:
        opened = _try_open(sensor_id, cand_w, cand_h, framerate)
        if opened is not None:
            if (cand_w, cand_h) != (width, height):
                print(f"[경고] {width}x{height}를 카메라가 받아주지 않아 "
                      f"{cand_w}x{cand_h}로 열었습니다.")
            break
        print(f"       {cand_w}x{cand_h} 실패, 다음 후보로 넘어갑니다.")

    if opened is None:
        print(f"[오류] 카메라를 열지 못했습니다. 지원 해상도를 확인하세요.")
        print(f"       v4l2-ctl --list-formats-ext -d /dev/video{sensor_id}")
        return cv2.VideoCapture()       # isOpened() == False

    cap, real_w, real_h = opened
    ratio = real_w / max(real_h, 1)
    name = {1.333: "4:3", 1.5: "3:2", 1.6: "16:10", 1.778: "16:9"}.get(
        round(ratio, 3), f"{ratio:.2f}:1")
    print(f"[INFO] 카메라 열림. 실제 {real_w}x{real_h} ({name}, {ratio:.3f})")
    if (real_w, real_h) != (width, height):
        print(f"[경고] 요청한 {width}x{height}와 다릅니다. "
              f"화각이 달라지므로 기둥 보정을 다시 해야 합니다.")

    # 드라이버 큐를 최소로. 백엔드가 지원하지 않으면 무시된다.
    # (그래서 LatestFrameCamera가 따로 필요하다)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # 실제로 잡힌 크기를 카메라 객체에 붙여 둔다.
    # 저장된 기둥 보정이 이 해상도에서 찍힌 것인지 확인할 때 쓴다.
    # (좌표가 이미지 픽셀이라 해상도가 다르면 전부 어긋난다)
    cam = LatestFrameCamera(cap) if latest_only else cap
    cam.frame_size = (real_w, real_h)
    return cam

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
