import cv2
import os
import json

PILLAR_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'config', 'pillar_pixels.json')


def save_pillar_pixels(points, frame_size=None):
    """
    기둥 픽셀 좌표를 '찍을 때의 해상도'와 함께 저장한다.

    해상도를 같이 적는 것이 중요하다. 좌표가 이미지 픽셀이라 해상도가
    달라지면 전부 어긋나는데, 기록이 없으면 다음 실행에서 그 사실을 알
    방법이 없어 틀린 좌표를 그대로 불러 쓰게 된다.
    """
    import json as _json
    os.makedirs(os.path.dirname(os.path.abspath(PILLAR_FILE)), exist_ok=True)
    data = {"points": {str(k): list(v) for k, v in points.items()}}
    if frame_size:
        data["width"], data["height"] = int(frame_size[0]), int(frame_size[1])
    with open(PILLAR_FILE, 'w') as f:
        _json.dump(data, f, indent=2)


def load_pillar_pixels():
    """
    저장된 기둥 픽셀 좌표를 불러온다.

    Returns:
        {"points": {번호: (x, y)}, "width": int|None, "height": int|None}
        파일이 없거나 읽을 수 없으면 None.

    해상도를 적기 전에 저장된 옛 파일은 width/height가 None이다.
    그 좌표가 지금 해상도에서 찍힌 것인지 알 수 없으므로, 부르는 쪽이
    쓰지 않고 재보정을 요구해야 한다.
    """
    import json as _json
    if not os.path.exists(PILLAR_FILE):
        return None
    try:
        with open(PILLAR_FILE) as f:
            data = _json.load(f)
        if "points" in data:
            raw = data["points"]
            width, height = data.get("width"), data.get("height")
        else:
            raw, width, height = data, None, None   # 해상도 없던 옛 형식
        return {"points": {int(k): tuple(v) for k, v in raw.items()},
                "width": width, "height": height}
    except Exception as e:
        print(f"[경고] 기둥 픽셀 좌표 로드 실패: {e}")
        return None

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
        """클릭한 기둥 픽셀 좌표를 저장하고 자리 위치를 자동 계산한다."""
        data = request.get_json(silent=True) or {}
        points = {int(k): (float(v[0]), float(v[1]))
                  for k, v in (data.get("points") or {}).items()}

        if len(points) < 4:
            return {"ok": False, "message": f"기둥이 {len(points)}개뿐입니다. 최소 4개가 필요합니다."}

        # 기둥 픽셀 좌표를 지금 해상도와 함께 저장한다.
        save_pillar_pixels(points, getattr(pipeline.cap, "frame_size", None))

        # mapper에 픽셀 좌표 설정
        ok, message = pipeline.navigator.mapper.set_pillar_pixels(points)
        if ok:
            # 자리 좌표가 바뀌었으므로 navigator에도 반영
            pipeline.navigator.update_spot_pixels()

        return {"ok": ok, "message": message}

    @app.route('/snapshot')
    def snapshot():
        ok, frame = pipeline.cap.read()
        if not ok:
            return "카메라 프레임을 읽을 수 없습니다.", 503
        ok, buf = cv2.imencode('.jpg', frame)
        return Response(buf.tobytes(), mimetype='image/jpeg')
