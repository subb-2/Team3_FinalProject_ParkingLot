import cv2
import os
import json
import numpy as np

CONFIG = {
    "CALIB_PATH": os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_calib.npz'),
    "ALPHA": 1.0,
}

def save_calibration(camera_matrix, dist_coeffs, image_size, rms, path=None):
    path = path or CONFIG['CALIB_PATH']
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                 image_size=np.array(image_size), rms=float(rms))
        print(f"[INFO] 렌즈 보정값을 저장했습니다. {os.path.abspath(path)}")
        return True
    except Exception as e:
        print(f"[경고] 렌즈 보정값 저장 실패: {e}")
        return False

def load_calibration(path=None):
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
    return load_calibration(path) is not None

class Undistorter:
    def __init__(self, path=None, alpha=None):
        self.alpha = CONFIG['ALPHA'] if alpha is None else alpha
        self.calib = load_calibration(path)
        self._map1 = None
        self._map2 = None
        self._size = None
        self.new_matrix = None
        if self.calib is None:
            print("[INFO] 렌즈 보정값이 없습니다. 원본 프레임을 그대로 씁니다.")
        else:
            print(f"[INFO] 렌즈 보정값을 불러왔습니다. (RMS {self.calib['rms']:.3f}px, 기준 해상도 {self.calib['image_size'][0]}x{self.calib['image_size'][1]})")

    def is_ready(self):
        return self.calib is not None

    def _build_maps(self, size):
        w, h = size
        cam = self.calib['camera_matrix']
        dist = self.calib['dist_coeffs']
        cal_w, cal_h = self.calib['image_size']
        if (cal_w, cal_h) != (w, h):
            cam = cam.copy()
            cam[0, :] *= w / cal_w
            cam[1, :] *= h / cal_h
        self.new_matrix, _ = cv2.getOptimalNewCameraMatrix(cam, dist, (w, h), self.alpha, (w, h))
        self._map1, self._map2 = cv2.initUndistortRectifyMap(cam, dist, None, self.new_matrix, (w, h), cv2.CV_16SC2)
        self._size = (w, h)

    def apply(self, frame):
        if self.calib is None or frame is None:
            return frame
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._build_maps((w, h))
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

def solve_radial_from_points(image_points, marker_world_pos, image_size, search=None):
    ids = [int(i) for i in image_points if int(i) in marker_world_pos]
    if len(ids) < 6: return None
    img = np.array([image_points[i] if i in image_points else image_points[str(i)]
                    for i in ids], dtype=np.float64).reshape(-1, 1, 2)
    world = np.array([marker_world_pos[i] for i in ids], dtype=np.float64)
    w, h = image_size
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])

    def fit(k1):
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0])
        und = cv2.undistortPoints(img, K, dist, P=K)
        H, _ = cv2.findHomography(und.astype(np.float32), world.astype(np.float32), 0)
        if H is None: return float('inf'), None
        proj = cv2.perspectiveTransform(und.astype(np.float32), H).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(proj - world, axis=1))), H

    lo, hi, steps = search or (-0.60, 0.30, 61)
    best = (fit(0.0)[0], 0.0)
    error_before = best[0]
    for _ in range(3):
        for k1 in np.linspace(lo, hi, steps):
            err, _ = fit(float(k1))
            if err < best[0]: best = (err, float(k1))
        span = (hi - lo) / (steps - 1) * 2
        lo, hi = best[1] - span, best[1] + span
    error_after, _ = fit(best[1])
    k1 = best[1]
    return {
        "k1": k1, "camera_matrix": K, "dist_coeffs": np.array([[k1, 0.0, 0.0, 0.0, 0.0]]),
        "image_size": image_size, "error_before": error_before, "error_after": error_after,
        "markers": len(ids), "improved": error_before - error_after,
    }

CALIBRATE_HTML = """
<html><head><meta charset="utf-8"><title>주차장 맵 초기 보정</title>
<style>
 body{background:#222;color:#eee;font-family:sans-serif;margin:0;padding:12px}
 #wrap{position:relative;display:inline-block}
 #shot{max-width:100%;cursor:crosshair;display:block}
 .dot{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;
      border-radius:50%;background:#0f0;border:2px solid #000;pointer-events:none}
 .lbl{position:absolute;margin:-28px 0 0 8px;color:#0f0;font-weight:bold;
      text-shadow:1px 1px 2px #000;pointer-events:none}
 #now{font-size:20px;color:#ff0;margin:8px 0}
 button{font-size:15px;padding:8px 16px;margin-right:8px;cursor:pointer;
        background:#444;color:#eee;border:1px solid #666;border-radius:4px}
 button:hover{background:#555}
 button.main{background:#2f6fd0;border-color:#1c4d9b;font-weight:bold}
 button.main:hover{background:#3a82f0}
 #msg{margin-top:10px;font-size:15px;white-space:pre-wrap}
</style></head><body>
<h2>주차장 맵 초기 보정</h2>
<p>카메라 화각 내에 보이는 <b>기둥 윗면 중앙</b>을 순서대로 클릭하세요.<br>
   완료 시 렌즈 왜곡이 자동 보정되고 메인 화면으로 이동합니다.</p>
<div id="now"></div>
<div id="wrap"><img id="shot" src="/snapshot"></div>
<div style="margin-top:10px">
  <button onclick="undo()">한 개 취소</button>
  <button onclick="skip()">이 기둥 건너뛰기</button>
  <button onclick="reload()">사진 다시 찍기</button>
  <button class="main" onclick="save()">저장 및 완료</button>
</div>
<div id="msg"></div>
<script>
const STEPS = __STEPS__;
let i = 0, pts = {};
const wrap = document.getElementById('wrap'), img = document.getElementById('shot');

function show(){
  document.getElementById('now').textContent = i < STEPS.length
    ? `[${i+1}/${STEPS.length}] ${STEPS[i].id}번 기둥 (격자 ${STEPS[i].row},${STEPS[i].col})을 클릭`
    : `모두 지정했습니다. 찍은 기둥 ${Object.keys(pts).length}개 - [저장 및 완료]를 누르세요.`;
}
img.addEventListener('click', e => {
  if (i >= STEPS.length) return;
  const r = img.getBoundingClientRect();
  const sx = img.naturalWidth / r.width, sy = img.naturalHeight / r.height;
  const id = STEPS[i].id;
  pts[id] = [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
  addDot(e.clientX - r.left, e.clientY - r.top, id);
  i++; show();
});
function addDot(x, y, id){
  const d = document.createElement('div'); d.className='dot'; d.dataset.id=id;
  d.style.left = x+'px'; d.style.top = y+'px'; wrap.appendChild(d);
  const l = document.createElement('div'); l.className='lbl'; l.dataset.id=id;
  l.style.left = x+'px'; l.style.top = y+'px'; l.textContent = id; wrap.appendChild(l);
}
function undo(){
  if (i === 0) return;
  i--; const id = STEPS[i].id; delete pts[id];
  document.querySelectorAll(`[data-id="${id}"]`).forEach(e => e.remove());
  show();
}
function skip(){ if (i < STEPS.length) { i++; show(); } }
function reload(){
  img.src = '/snapshot?' + Date.now();
  i = 0; pts = {};
  document.querySelectorAll('.dot, .lbl').forEach(e => e.remove());
  show();
}
function save(){
  fetch('/calibrate/save', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({points: pts})
  }).then(r => r.json()).then(r => {
    document.getElementById('msg').textContent = r.message;
    if(r.ok) {
        document.getElementById('msg').style.color = '#0f0';
        setTimeout(() => { window.location.href = "/"; }, 1500); // 메인으로 자동 복귀
    } else {
        document.getElementById('msg').style.color = '#f00';
    }
  });
}
img.onload = show;
</script></body></html>
"""

def register_map_routes(app, pipeline, cap_module=None):
    from flask import request, Response
    from data.map_data import PILL_MARKER_ID

    @app.route('/calibrate')
    def calibrate():
        order = sorted(PILL_MARKER_ID.items(), key=lambda kv: kv[1])
        steps = [{"id": mid, "row": cell[0], "col": cell[1]} for cell, mid in order]
        return CALIBRATE_HTML.replace("__STEPS__", json.dumps(steps, ensure_ascii=False))

    @app.route('/calibrate/save', methods=['POST'])
    def calibrate_save():
        data = request.get_json(silent=True) or {}
        points = {int(k): v for k, v in (data.get("points") or {}).items()}
        ok, message = pipeline.navigator.mapper.set_homography_from_points(points)

        if ok:
            if cap_module and hasattr(cap_module, "reload_undistort"):
                cap_module.reload_undistort()
            anchored, moved, shift = pipeline.navigator.anchor_spots_to_observed()
            if anchored and moved:
                message += f"  자리 {moved}개를 기둥 기준으로 갱신(최대 {shift:.1f}cm)."
        
        residuals = pipeline.navigator.mapper.marker_residuals
        return {
            "ok": ok,
            "message": message,
            "residuals_cm": {str(k): round(v, 2) for k, v in sorted(residuals.items())},
        }

    @app.route('/snapshot')
    def snapshot():
        ok, frame = pipeline.cap.read()
        if not ok:
            return "카메라 프레임을 읽을 수 없습니다.", 503
        ok, buf = cv2.imencode('.jpg', frame)
        return Response(buf.tobytes(), mimetype='image/jpeg')
