"""
B04_lens_calib : 렌즈 왜곡 보정 (카메라 캘리브레이션)

왜 필요한가
  광각 렌즈는 직선을 휘게 담는다. 목업 매트의 곧은 테두리가 화면에서 활처럼
  굽어 보이는 것이 그 증거다.

  호모그래피는 직선을 직선으로만 보낸다. 그래서 휘어진 화면에는 아무리 잘
  맞춰도 한계가 있다. 실제로 재보면 기둥 10개 평균 오차는 2.3cm인데 최대는
  7.7cm까지 벌어진다. 10cm 칸의 3/4이라 그 근처 자리는 통째로 어긋난다.
  평균만 보면 통과인데 화면은 어긋나는 이유가 이것이다.

  왜곡 계수를 구해 프레임을 미리 펴 놓으면(undistort) 그 뒤로는 직선이
  직선으로 담긴다. 호모그래피의 전제가 성립하므로 오차가 크게 줄어든다.

무엇을 재는가
  체스보드를 여러 각도에서 찍어 카메라의 내부 파라미터(초점거리, 주점)와
  왜곡 계수를 구한다. 카메라와 렌즈가 바뀌지 않으면 한 번만 하면 되고,
  결과는 config/camera_calib.npz에 저장되어 다음 실행부터 자동 적용된다.

  주차장 배치와는 무관하다. 렌즈의 성질만 재는 것이라 목업을 바꿔도
  다시 할 필요가 없다. (호모그래피는 다시 잡아야 한다)

사용법
    python logic/B04_lens_calib.py
    -> http://젯슨IP:5000/ 에서 체스보드를 여러 각도로 찍고 [계산]

!! 중요 !!
  보정을 켜면 화면이 미세하게 달라지므로 호모그래피를 반드시 다시 잡아야 한다.
  기존 캐시는 자동으로 거부된다. (C00이 undistort 여부를 함께 저장한다)
"""

import cv2
import sys
import os
import time
import threading
import numpy as np
from flask import Flask, Response, request

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CONFIG = {
    # 체스보드 '내부 코너' 개수. 칸 개수가 아니라 칸과 칸이 만나는 점의 개수다.
    # 8x8 칸짜리 보드면 내부 코너는 7x7이다. 흔히 쓰는 A4 인쇄물은 9x6이다.
    # 이 값이 실물과 다르면 코너가 하나도 검출되지 않는다.
    "BOARD_COLS": 9,
    "BOARD_ROWS": 6,

    # 체스보드 한 칸의 실제 한 변 길이 (mm).
    # 왜곡 계수만 쓸 거라면 이 값이 틀려도 결과가 같다. 배율에만 영향을 준다.
    # 그래도 맞춰 두면 초점거리가 실제 단위로 나와 나중에 쓸 데가 있다.
    "SQUARE_MM": 25.0,

    # 계산에 필요한 최소 촬영 장수.
    # 적으면 왜곡 계수가 불안정하고, 각도가 다양할수록 좋다.
    # 15장 정도를 서로 다른 위치/기울기로 찍는 것을 권한다.
    "MIN_SHOTS": 10,
    "TARGET_SHOTS": 15,

    # 같은 자세로 연달아 찍히는 것을 막는 최소 간격 (초).
    # 비슷한 장면만 모으면 장수만 늘고 정확도는 안 오른다.
    "MIN_INTERVAL_SEC": 0.7,

    # 결과 저장 경로
    "CALIB_PATH": os.path.join(os.path.dirname(__file__), '..', 'config',
                               'camera_calib.npz'),

    # 보정 후 화면 처리 방식.
    #   0.0 : 왜곡으로 생긴 검은 가장자리를 잘라내 화면을 꽉 채운다. 화각이 줄어든다.
    #   1.0 : 원본 화소를 모두 남긴다. 가장자리에 검은 영역이 생긴다.
    # 마커가 화면 가장자리에 있으면 0.0으로 잘랐을 때 잘려나갈 수 있으므로
    # 화각을 지키는 쪽(1.0)을 기본으로 둔다.
    "ALPHA": 1.0,

    # 캘리브레이션 웹 서버
    "WEB_HOST": "0.0.0.0",
    "WEB_PORT": 5000,
}


# 저장 / 불러오기
def save_calibration(camera_matrix, dist_coeffs, image_size, rms, path=None):
    """왜곡 계수를 파일로 저장한다."""
    path = path or CONFIG['CALIB_PATH']
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs,
                 image_size=np.array(image_size),
                 rms=float(rms))
        print(f"[INFO] 렌즈 보정값을 저장했습니다. {os.path.abspath(path)}")
        return True
    except Exception as e:
        print(f"[경고] 렌즈 보정값 저장 실패: {e}")
        return False


def load_calibration(path=None):
    """
    저장된 왜곡 계수를 불러온다.

    Returns:
        {"camera_matrix", "dist_coeffs", "image_size", "rms"} 또는 None
    """
    path = path or CONFIG['CALIB_PATH']
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        return {
            "camera_matrix": data['camera_matrix'],
            "dist_coeffs": data['dist_coeffs'],
            "image_size": tuple(int(v) for v in data['image_size']),
            "rms": float(data['rms']),
        }
    except Exception as e:
        print(f"[경고] 렌즈 보정값을 읽을 수 없습니다: {e}")
        return None


def has_calibration(path=None):
    """보정값이 준비되어 있는지."""
    return load_calibration(path) is not None


# 프레임 펴기
class Undistorter:
    """
    저장된 왜곡 계수로 프레임을 펴 준다.

    매 프레임 cv2.undistort를 부르면 내부에서 매번 매핑을 다시 만든다.
    해상도가 고정이면 그 매핑은 상수이므로 한 번만 만들어 두고 remap만 한다.
    (1280x720 기준으로 프레임당 수 ms 차이가 난다)
    """

    def __init__(self, path=None, alpha=None):
        self.alpha = CONFIG['ALPHA'] if alpha is None else alpha
        self.calib = load_calibration(path)
        self._map1 = None
        self._map2 = None
        self._size = None
        self.new_matrix = None

        if self.calib is None:
            print("[INFO] 렌즈 보정값이 없습니다. 원본 프레임을 그대로 씁니다. "
                  "(python logic/B04_lens_calib.py 로 만들 수 있습니다)")
        else:
            print(f"[INFO] 렌즈 보정값을 불러왔습니다. "
                  f"(RMS {self.calib['rms']:.3f}px, "
                  f"기준 해상도 {self.calib['image_size'][0]}x{self.calib['image_size'][1]})")

    def is_ready(self):
        return self.calib is not None

    def _build_maps(self, size):
        """해상도가 정해지면 그때 매핑을 만든다."""
        w, h = size
        cam = self.calib['camera_matrix']
        dist = self.calib['dist_coeffs']

        # 보정에 쓴 해상도와 지금 해상도가 다르면 내부 파라미터를 비례 조정한다.
        # 같은 렌즈라도 캡처 해상도가 바뀌면 초점거리와 주점이 화소 단위로 달라진다.
        cal_w, cal_h = self.calib['image_size']
        if (cal_w, cal_h) != (w, h):
            cam = cam.copy()
            cam[0, :] *= w / cal_w
            cam[1, :] *= h / cal_h
            print(f"[경고] 렌즈 보정은 {cal_w}x{cal_h}에서 했는데 지금은 {w}x{h}입니다. "
                  f"비례 조정해서 쓰지만, 같은 해상도로 다시 잡는 편이 정확합니다.")

        self.new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            cam, dist, (w, h), self.alpha, (w, h))
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            cam, dist, None, self.new_matrix, (w, h), cv2.CV_16SC2)
        self._size = (w, h)

    def apply(self, frame):
        """프레임을 펴서 반환. 보정값이 없으면 원본을 그대로 돌려준다."""
        if self.calib is None or frame is None:
            return frame

        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._build_maps((w, h))
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)


# 캘리브레이션
class CameraCalibrator:
    """
    체스보드 사진을 모아 왜곡 계수를 계산한다.

    쓰는 순서
        c = CameraCalibrator()
        c.try_capture(frame)     # 코너가 보이면 저장 (여러 각도로 반복)
        ok, msg = c.compute()    # 충분히 모이면 계산 후 저장
    """

    def __init__(self, cols=None, rows=None, square_mm=None):
        self.cols = CONFIG['BOARD_COLS'] if cols is None else cols
        self.rows = CONFIG['BOARD_ROWS'] if rows is None else rows
        self.square_mm = CONFIG['SQUARE_MM'] if square_mm is None else square_mm

        # 체스보드 한 장의 3D 좌표. 보드가 평면이므로 z는 전부 0이다.
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        self._objp = objp * self.square_mm

        self.obj_points = []      # 각 장의 3D 좌표
        self.img_points = []      # 각 장의 코너 화소 좌표
        self.image_size = None
        self._last_capture = 0.0
        self.result = None        # compute() 결과

        self._lock = threading.Lock()
        print(f"[INFO] 렌즈 캘리브레이션 준비. "
              f"체스보드 내부 코너 {self.cols}x{self.rows}, 칸 {self.square_mm}mm")

    def find_corners(self, frame):
        """
        프레임에서 체스보드 코너를 찾는다.

        Returns:
            (찾았는지, 코너 배열 또는 None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(
            gray, (self.cols, self.rows), flags)
        if not found:
            return False, None

        # 화소 단위로는 부족하다. 부화소까지 다듬어야 왜곡 계수가 안정된다.
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return True, corners

    def try_capture(self, frame):
        """
        코너가 보이면 한 장으로 저장한다.

        Returns:
            (저장했는지, 메시지)
        """
        now = time.time()
        if now - self._last_capture < CONFIG['MIN_INTERVAL_SEC']:
            return False, "너무 빠릅니다. 잠시 뒤에 다시 찍으세요."

        found, corners = self.find_corners(frame)
        if not found:
            return False, "체스보드를 찾지 못했습니다. 보드 전체가 화면에 들어오게 하세요."

        with self._lock:
            self.obj_points.append(self._objp)
            self.img_points.append(corners)
            self.image_size = (frame.shape[1], frame.shape[0])
            self._last_capture = now
            count = len(self.img_points)

        return True, f"{count}장째 저장했습니다."

    def undo(self):
        """마지막 한 장을 취소한다."""
        with self._lock:
            if not self.img_points:
                return False, "취소할 사진이 없습니다."
            self.obj_points.pop()
            self.img_points.pop()
            return True, f"한 장 취소했습니다. (남은 {len(self.img_points)}장)"

    def reset(self):
        with self._lock:
            self.obj_points.clear()
            self.img_points.clear()
            self.result = None
        return True, "모두 지웠습니다."

    def count(self):
        with self._lock:
            return len(self.img_points)

    def compute(self, save=True):
        """
        모은 사진으로 왜곡 계수를 계산하고 저장한다.

        Returns:
            (성공 여부, 메시지)
        """
        with self._lock:
            n = len(self.img_points)
            obj_points = list(self.obj_points)
            img_points = list(self.img_points)
            image_size = self.image_size

        if n < CONFIG['MIN_SHOTS']:
            return False, (f"{n}장뿐입니다. 최소 {CONFIG['MIN_SHOTS']}장이 필요합니다. "
                           f"보드를 여러 각도로 기울여 더 찍으세요.")

        rms, cam, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None)

        # 장마다의 재투영 오차. 유독 큰 장이 있으면 그 사진이 흔들렸거나
        # 코너를 잘못 잡은 것이다.
        per_view = []
        for i in range(n):
            proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], cam, dist)
            err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
            per_view.append(float(err))

        self.result = {
            "rms": float(rms),
            "camera_matrix": cam,
            "dist_coeffs": dist,
            "image_size": image_size,
            "per_view": per_view,
            "shots": n,
        }

        k1, k2, p1, p2, k3 = dist.ravel()[:5]
        print(f"[INFO] 캘리브레이션 완료. {n}장, RMS {rms:.4f}px")
        print(f"       왜곡 계수  k1={k1:+.4f} k2={k2:+.4f} k3={k3:+.4f} "
              f"p1={p1:+.4f} p2={p2:+.4f}")
        print(f"       초점거리   fx={cam[0,0]:.1f} fy={cam[1,1]:.1f}  "
              f"주점 cx={cam[0,2]:.1f} cy={cam[1,2]:.1f}")

        worst = max(per_view)
        if worst > 1.0:
            i = per_view.index(worst)
            print(f"[경고] {i+1}번째 사진의 오차가 큽니다 ({worst:.2f}px). "
                  f"흔들렸을 수 있습니다. 지우고 다시 찍는 편이 좋습니다.")

        msg = f"완료. {n}장, RMS {rms:.4f}px (1.0 아래면 좋음)"
        if rms > 1.0:
            msg += "  주의: RMS가 큽니다. 흔들린 사진이 섞였거나 각도가 단조롭습니다."

        if save:
            save_calibration(cam, dist, image_size, rms)
            msg += "  저장했습니다."

        return True, msg

    def draw_overlay(self, frame):
        """
        현재 프레임에 코너 검출 상태와 진행 상황을 그려 준다.
        (한글은 못 그리므로 영문/숫자만 쓴다)
        """
        canvas = frame.copy()
        found, corners = self.find_corners(frame)
        if found:
            cv2.drawChessboardCorners(canvas, (self.cols, self.rows), corners, True)

        n = self.count()
        target = CONFIG['TARGET_SHOTS']
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(canvas, "BOARD FOUND" if found else "NO BOARD", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"shots: {n}/{target}", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # 이미 찍은 코너들이 화면 어디를 덮었는지 점으로 남긴다.
        # 가운데만 찍으면 가장자리 왜곡이 안 잡히므로, 빈 구석이 보이면
        # 그쪽에서 더 찍으라는 신호가 된다.
        with self._lock:
            for pts in self.img_points:
                for p in pts.reshape(-1, 2)[::6]:
                    cv2.circle(canvas, (int(p[0]), int(p[1])), 1, (255, 160, 0), -1)

        return canvas


# 캘리브레이션 페이지
CALIB_HTML = """
<html><head><meta charset="utf-8"><title>렌즈 왜곡 보정</title>
<style>
 body{background:#1a1a1c;color:#eee;font-family:sans-serif;margin:0;padding:16px}
 h2{margin:0 0 4px}
 .hint{color:#9a9aa2;font-size:14px;line-height:1.7;margin:0 0 14px}
 .hint b{color:#ffd24d}
 img{max-width:100%;border-radius:8px;display:block}
 button{font-size:16px;padding:10px 20px;margin:12px 8px 0 0;cursor:pointer;
        border-radius:6px;border:1px solid #444;background:#2a2a30;color:#eee}
 button.main{background:#2f6fd0;border-color:#2f6fd0;font-weight:700}
 button:hover{filter:brightness(1.2)}
 #msg{margin-top:14px;font-size:15px;white-space:pre-wrap;line-height:1.6}
 ol{color:#9a9aa2;font-size:14px;line-height:1.9}
</style></head><body>
<h2>렌즈 왜곡 보정</h2>
<p class="hint">
  체스보드를 <b>여러 각도·여러 위치</b>에서 찍으세요. 초록 점이 보이면 인식된 겁니다.<br>
  <b>화면 구석에서도 꼭 찍어야 합니다.</b> 왜곡은 가장자리에서 가장 크기 때문에,
  가운데만 찍으면 정작 필요한 곳이 보정되지 않습니다.<br>
  주황 점은 지금까지 찍은 코너 위치입니다. <b>빈 구석이 없도록</b> 채우세요.
</p>

<img id="shot" src="/preview">

<div>
  <button class="main" onclick="act('/shot')">촬영</button>
  <button onclick="act('/undo')">한 장 취소</button>
  <button onclick="act('/reset')">모두 지우기</button>
  <button class="main" onclick="act('/compute')">계산하고 저장</button>
</div>
<div id="msg"></div>

<ol>
  <li>체스보드를 카메라 앞에서 <b>기울여 가며</b> 10~15장 찍습니다.</li>
  <li>가운데 / 네 귀퉁이 / 좌우로 기울인 자세를 골고루 넣습니다.</li>
  <li>[계산하고 저장]을 누릅니다. RMS가 1.0px 아래면 좋습니다.</li>
  <li>끝나면 이 창을 닫고 <b>호모그래피를 다시 잡으세요</b> (/calibrate).</li>
</ol>

<script>
function act(url){
  fetch(url, {method:'POST'}).then(r => r.json()).then(r => {
    const m = document.getElementById('msg');
    m.textContent = r.message;
    m.style.color = r.ok ? '#4ccf6a' : '#ff8a8a';
  });
}
</script></body></html>
"""


# 단독 실행
def main():
    from logic.B00_camera_input import get_camera, DEFAULT_WIDTH, DEFAULT_HEIGHT

    print("=" * 50)
    print(" B04_lens_calib : 렌즈 왜곡 보정")
    print("=" * 50)
    print(f" 체스보드 내부 코너 {CONFIG['BOARD_COLS']}x{CONFIG['BOARD_ROWS']}, "
          f"칸 {CONFIG['SQUARE_MM']}mm")
    print(" 보드 규격이 다르면 CONFIG의 BOARD_COLS/BOARD_ROWS를 고치세요.")
    print(" (칸 개수가 아니라 '내부 코너' 개수입니다. 8x8칸 보드 -> 7x7)")
    print("=" * 50)

    cap = get_camera(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        sys.exit(1)

    calib = CameraCalibrator()
    latest = {"frame": None}

    def grab():
        while True:
            ok, frame = cap.read()
            if ok:
                latest["frame"] = frame
            else:
                time.sleep(0.05)

    threading.Thread(target=grab, daemon=True).start()

    app = Flask(__name__)

    @app.route('/')
    def index():
        return CALIB_HTML

    @app.route('/preview')
    def preview():
        """코너 검출 상태를 겹쳐 그린 실시간 화면."""
        def gen():
            while True:
                frame = latest["frame"]
                if frame is not None:
                    ok, buf = cv2.imencode('.jpg', calib.draw_overlay(frame))
                    if ok:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                               + buf.tobytes() + b'\r\n')
                time.sleep(0.12)
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def _json(result):
        ok, message = result
        return {"ok": ok, "message": message, "shots": calib.count()}

    @app.route('/shot', methods=['POST'])
    def shot():
        frame = latest["frame"]
        if frame is None:
            return {"ok": False, "message": "아직 프레임이 없습니다.", "shots": 0}
        return _json(calib.try_capture(frame))

    @app.route('/undo', methods=['POST'])
    def undo():
        return _json(calib.undo())

    @app.route('/reset', methods=['POST'])
    def reset():
        return _json(calib.reset())

    @app.route('/compute', methods=['POST'])
    def compute():
        return _json(calib.compute())

    print(f"\n[INFO] http://젯슨IP:{CONFIG['WEB_PORT']}/ 으로 접속하세요.")
    try:
        app.run(host=CONFIG['WEB_HOST'], port=CONFIG['WEB_PORT'], debug=False)
    finally:
        cap.release()


if __name__ == '__main__':
    main()
