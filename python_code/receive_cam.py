"""
receive_cam.py

젯슨이 송출하는 MJPEG 영상을 다른 PC에서 수신하는 모듈.

젯슨 쪽은 Flask가 이미 MJPEG를 서비스하고 있으므로(D_main / C_main / B_main의
/video_feed), 이 파일은 그 스트림을 받아서 원하는 크기·비율로 가공해 주는 역할만 한다.
크기와 형태는 전부 수신 측에서 결정하므로 젯슨 코드를 고칠 필요가 없다.

[UI에 넣어 쓰는 경우]
    from receive_cam import CamReceiver

    rx = CamReceiver("http://192.168.0.50:5000/video_feed")
    rx.start()
    ...
    frame = rx.get_frame(1150, 300, mode="cover")   # CAM 03 패널 크기에 맞춰서
    if frame is not None:
        ...화면에 그리기...
    ...
    rx.stop()

[단독 실행 - 연결 확인용]
    python receive_cam.py
    python receive_cam.py --url http://192.168.0.50:5000/video_feed --size 960x540
"""

import argparse
import threading
import time

import cv2
import numpy as np

# 젯슨 무선 고정 IP + Flask 포트. A00_uart_rx.py 의 jetson_ip 와 같은 주소.
DEFAULT_URL = "http://192.168.0.50:5000/video_feed"

# 크기를 맞추는 방식
#   stretch : 비율 무시하고 목표 크기로 늘림 (여백 없음, 찌그러질 수 있음)
#   contain : 비율 유지, 남는 부분은 검은 여백 (레터박스)
#   cover   : 비율 유지, 넘치는 부분은 잘라냄 (여백 없음) - 패널 채우기에 적합
FIT_MODES = ("stretch", "contain", "cover")


def _resize_keep(frame, width, height, mode="cover"):
    """frame을 (width, height)에 맞춰 mode 방식으로 변환한 새 프레임을 반환."""
    h, w = frame.shape[:2]

    # 한쪽만 지정하면 비율을 유지해서 나머지를 계산
    if width is None and height is None:
        return frame.copy()
    if width is None:
        width = max(1, int(round(w * height / h)))
    if height is None:
        height = max(1, int(round(h * width / w)))

    def _scaled(sx, sy):
        # 축소는 INTER_AREA, 확대는 INTER_LINEAR 가 화질이 낫다
        interp = cv2.INTER_AREA if (sx < 1.0 or sy < 1.0) else cv2.INTER_LINEAR
        return cv2.resize(frame, (max(1, int(round(w * sx))),
                                  max(1, int(round(h * sy)))),
                          interpolation=interp)

    if mode == "stretch":
        interp = cv2.INTER_AREA if (width < w or height < h) else cv2.INTER_LINEAR
        return cv2.resize(frame, (width, height), interpolation=interp)

    if mode == "contain":
        s = min(width / w, height / h)
        small = _scaled(s, s)
        canvas = np.zeros((height, width, 3), dtype=frame.dtype)
        y0 = (height - small.shape[0]) // 2
        x0 = (width - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        return canvas

    # cover (기본): 목표 영역을 꽉 채우고 넘치는 부분은 가운데 기준으로 잘라냄
    s = max(width / w, height / h)
    big = _scaled(s, s)
    y0 = (big.shape[0] - height) // 2
    x0 = (big.shape[1] - width) // 2
    return big[y0:y0 + height, x0:x0 + width].copy()


def make_placeholder(width, height, text="NO SIGNAL"):
    """연결이 끊겼을 때 UI에 대신 표시할 프레임."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(canvas, text,
                ((width - size[0]) // 2, (height + size[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2, cv2.LINE_AA)
    return canvas


class CamReceiver:
    """
    MJPEG 스트림을 백그라운드 스레드에서 계속 읽어 최신 프레임만 들고 있는다.

    UI 루프에서 직접 cap.read()를 하면 디코딩이 밀리면서 지연이 계속 쌓인다.
    여기서는 읽은 프레임을 버리고 최신 것만 유지하므로 화면이 밀리지 않는다.
    """

    def __init__(self, url=DEFAULT_URL, reconnect_delay=2.0):
        self.url = url
        self.reconnect_delay = reconnect_delay

        self._frame = None          # 최신 프레임 (BGR)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._connected = False
        self._fps = 0.0
        self._last_ts = None

    # ---- 제어 -----------------------------------------------------------
    def start(self):
        """수신 스레드를 시작한다. 이미 돌고 있으면 무시."""
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """수신 스레드를 정지한다."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ---- 상태 -----------------------------------------------------------
    @property
    def connected(self):
        return self._connected

    @property
    def fps(self):
        return self._fps

    # ---- 프레임 획득 ----------------------------------------------------
    def get_frame(self, width=None, height=None, mode="cover"):
        """
        최신 프레임을 원하는 크기로 변환해서 반환. 아직 수신 전이면 None.

        width/height 중 하나만 주면 비율을 유지해 나머지를 계산한다.
        mode 는 "cover"(꽉 채우고 잘라냄), "contain"(여백), "stretch"(찌그러뜨림).
        """
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()

        if frame is None:
            return None
        if width is None and height is None:
            return frame
        return _resize_keep(frame, width, height, mode)

    def get_frame_or_placeholder(self, width, height, mode="cover"):
        """프레임이 없으면 NO SIGNAL 화면을 대신 반환. UI에서 분기 없이 쓰기 좋다."""
        frame = self.get_frame(width, height, mode)
        return make_placeholder(width, height) if frame is None else frame

    # ---- 내부 -----------------------------------------------------------
    def _loop(self):
        cap = None
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                self._connected = False
                print(f"[수신] 접속 시도: {self.url}")
                cap = cv2.VideoCapture(self.url)
                # 버퍼를 최소로 (백엔드가 지원하지 않으면 무시된다)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    if self._stop.wait(self.reconnect_delay):
                        break
                    continue
                print("[수신] 연결됨")

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[수신] 스트림 끊김. 재접속 대기...")
                cap.release()
                cap = None
                self._connected = False
                if self._stop.wait(self.reconnect_delay):
                    break
                continue

            with self._lock:
                self._frame = frame
            self._connected = True

            now = time.time()
            if self._last_ts is not None:
                dt = now - self._last_ts
                if dt > 0:
                    # 지수 이동 평균으로 부드럽게
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
            self._last_ts = now

        if cap is not None:
            cap.release()
        self._connected = False
        print("[수신] 종료")


def _parse_size(text):
    """'960x540' -> (960, 540)"""
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("크기는 960x540 형식으로 입력하세요")


def main():
    parser = argparse.ArgumentParser(description="젯슨 MJPEG 영상 수신 확인")
    parser.add_argument("--url", default=DEFAULT_URL, help="MJPEG 스트림 주소")
    parser.add_argument("--size", type=_parse_size, default=(960, 540),
                        help="표시 크기 (예: 960x540)")
    parser.add_argument("--mode", choices=FIT_MODES, default="contain",
                        help="크기 맞춤 방식")
    args = parser.parse_args()

    width, height = args.size
    mode = args.mode

    print(f"[설정] {args.url}  {width}x{height}  mode={mode}")
    print("[조작] q 종료 / f 맞춤방식 전환 / s 스냅샷 저장")

    win = "Jetson CAM"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # 창 크기 자유 조절
    cv2.resizeWindow(win, width, height)

    rx = CamReceiver(args.url).start()
    try:
        while True:
            frame = rx.get_frame_or_placeholder(width, height, mode)

            status = f"{'LIVE' if rx.connected else 'NO SIGNAL'}  {rx.fps:4.1f} fps  [{mode}]"
            color = (120, 220, 120) if rx.connected else (120, 120, 220)
            cv2.putText(frame, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('f'):
                mode = FIT_MODES[(FIT_MODES.index(mode) + 1) % len(FIT_MODES)]
                print(f"[설정] mode={mode}")
            if key == ord('s'):
                name = time.strftime("snapshot_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, frame)
                print(f"[저장] {name}")
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
