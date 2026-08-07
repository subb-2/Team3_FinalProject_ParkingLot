import cv2
import time
import sys
import os
import threading
import traceback
import numpy as np
from flask import Flask, Response

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.C_main import (
    CONFIG as C_CONFIG, open_camera, build_pipeline,
    setup_car_number_source, register_car_number, build_status,
)
from logic.D00_ui_navi import (
    NavigationView, NavigationMapUI, pick_my_vehicle, CONFIG as D00_CONFIG,
)
from logic.B02_car_mot import car_number_fifo

# 설정 (Configuration)
# 각 모듈의 세부 설정은 해당 모듈의 CONFIG에서 관리.
#   - 카메라 / 검출 / 추적 / 내비게이션 : C_main.py 및 그 하위 모듈
#   - 화면 크기, 표시 옵션              : D00_ui_navi.py
# 여기서는 '세 화면을 동시에 서비스하는 것'에 필요한 설정만 관리.
CONFIG = {
    # 웹 스트리밍 서버 설정
    "WEB_HOST": "0.0.0.0",
    "WEB_PORT": 5000,

    # 각 화면의 전송 프레임레이트 (Hz).
    # 영상 처리는 백그라운드 스레드가 최대 속도로 돌고, 화면은 여기 값으로 내보낸다.
    # 조감도와 차량 시점은 매 프레임 새로 그릴 필요가 없으므로 낮춰서 CPU를 아낀다.
    "VIDEO_FEED_FPS": 20,           # 카메라 원본 + 검출/추적 오버레이
    "NAV_FEED_FPS": 20,             # 차량 시점 내비게이션
    "MAP_FEED_FPS": 15,             # 관제 조감도

    # 내비게이션 화면에 띄울 '내 차'. None이면 자동 선택.
    # 실행 중에는 /follow/<차량번호> 로 바꿀 수 있다.
    "MY_CAR_ID": None,
}


# 파이프라인 실행기 (백그라운드 처리 + 최신 프레임 공유)
class PipelineRunner:
    """
    영상 처리를 백그라운드 스레드에서 돌리고, 그 결과를 여러 화면이 나눠 쓰게 한다.

    C_main의 generate_frames()는 '읽기 -> 처리 -> 전송'을 한 제너레이터 안에서
    수행하므로, 그 스트림을 아무도 보지 않으면 처리 자체가 멈춘다.
    화면이 셋(카메라 / 차량 시점 / 조감도)인 D_main에서는 이 구조가 곤란하다.
      - 조감도만 열어두면 파이프라인이 돌지 않아 화면이 멈춘 것처럼 보이고,
      - 세 화면이 각자 처리를 돌리면 같은 일을 세 번 하게 된다.

    그래서 처리는 이 클래스가 한 번만 수행하고, 각 화면은 최신 결과를
    가져다 그리기만 한다.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._frame = None          # 검출/추적/마커까지 그려진 최신 프레임
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last_error = None

    def start(self):
        """처리 스레드를 시작한다."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[INFO] 파이프라인 처리 스레드를 시작했습니다.")

    def stop(self):
        """처리 스레드를 종료시킨다."""
        self._stop.set()

    def _loop(self):
        """카메라 -> 검출 -> 추적 -> 내비게이션을 최대 속도로 반복."""
        prev_time = time.time()
        frame_count = 0

        while not self._stop.is_set():
            ret, frame = self.pipeline.cap.read()
            if not ret:
                print("[ERROR] 카메라 프레임을 읽을 수 없습니다. 처리 스레드를 종료합니다.")
                break

            # 처리 중 예외가 나도 스레드를 죽이지 않는다.
            # 죽으면 화면이 조용히 멈춰 원인을 알 수 없기 때문이다.
            try:
                self.pipeline.process_frame(frame)
            except Exception as exc:
                self._report_error(exc)
                with self._lock:
                    self._frame = self._error_canvas(frame, exc)
                time.sleep(0.5)
                continue

            # FPS 계산 (0.5초 간격)
            now = time.time()
            frame_count += 1
            if now - prev_time >= 0.5:
                self.pipeline.fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            self.pipeline.draw_status(frame)

            with self._lock:
                self._frame = frame

    def _report_error(self, exc):
        """같은 오류는 한 번만 자세히 출력한다. (매 프레임 도배 방지)"""
        key = f"{type(exc).__name__}: {exc}"
        if key == self._last_error:
            return
        self._last_error = key
        print(f"\n[ERROR] 프레임 처리 실패: {key}")
        traceback.print_exc()

    @staticmethod
    def _error_canvas(frame, exc):
        """오류 내용을 적은 프레임을 만든다."""
        canvas = np.zeros_like(frame)
        msg = f"{type(exc).__name__}: {exc}"
        cv2.putText(canvas, "PIPELINE ERROR", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
        for i in range(0, len(msg), 60):
            cv2.putText(canvas, msg[i:i + 60], (20, 110 + (i // 60) * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "see terminal for full traceback", (20, canvas.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        return canvas

    def latest_frame(self):
        """가장 최근에 처리된 프레임의 복사본. 아직 없으면 None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()


# MJPEG 스트리밍 헬퍼
def mjpeg_response(render_fn, fps):
    """
    매 호출마다 한 장을 그려주는 함수를 MJPEG 스트림으로 만든다.

    Args:
        render_fn: BGR 이미지(numpy array)를 반환하는 함수. None을 반환하면 건너뛴다.
        fps:       초당 전송 프레임 수

    Returns:
        Flask Response
    """
    interval = 1.0 / max(fps, 1)

    def generate():
        while True:
            canvas = render_fn()
            if canvas is not None:
                ok, buf = cv2.imencode('.jpg', canvas)
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(interval)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 메인 (B + C + D 통합 실행)
if __name__ == '__main__':
    print("==========================================")
    print(" D_main : 주차장 내비게이션 통합 실행")
    print(" B00(카메라) -> B01(검출) -> B02(추적)")
    print("   -> C00/C01(위치추정/경로) -> D00(UI)")
    print("==========================================")

    # 카메라 + 파이프라인 (C_main의 구성을 그대로 재사용)
    cap = open_camera()
    if cap is None:
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태와 CAM_SENSOR_ID를 확인하세요.")
        sys.exit(1)

    pipeline = build_pipeline(cap)
    setup_car_number_source()

    # D00 : 화면 두 종류 (차량 시점 / 관제 조감도)
    view = NavigationView(navigator=pipeline.navigator)
    overview = NavigationMapUI(navigator=pipeline.navigator)

    # 영상 처리는 백그라운드에서 한 번만 수행
    runner = PipelineRunner(pipeline)
    runner.start()

    # 내비게이션 화면에 띄울 차량 (/follow 로 변경 가능)
    my_car = {"id": CONFIG['MY_CAR_ID']}

    def extra_info():
        """호모그래피 / 마커 / FIFO 상태를 화면 하단에 표시할 문자열."""
        mapper = pipeline.navigator.mapper
        if not mapper.is_ready():
            state = "NOT READY (show markers)"
        elif mapper.locked:
            state = f"LOCKED {mapper.calibrated_with}pt {mapper.reproj_error:.1f}cm"
        else:
            state = f"PROVISIONAL {mapper.calibrated_with}/{mapper.lock_markers}pt"
        return [
            f"homography: {state}",
            f"markers: {len(pipeline.navigator.latest_markers)}",
            f"fifo: {car_number_fifo.size()} waiting",
        ]

    app = Flask(__name__)

    @app.route('/video_feed')
    def video_feed():
        """카메라 원본 + 검출/추적/마커 오버레이."""
        return mjpeg_response(runner.latest_frame, CONFIG['VIDEO_FEED_FPS'])

    @app.route('/nav_feed')
    def nav_feed():
        """차량 시점 내비게이션 화면."""
        def render():
            nav = pick_my_vehicle(pipeline.latest_nav, my_car["id"])
            return view.render(nav, fps=pipeline.fps, extra_info=extra_info())
        return mjpeg_response(render, CONFIG['NAV_FEED_FPS'])

    @app.route('/map_feed')
    def map_feed():
        """관제용 조감도. 여러 차량을 한 번에 본다."""
        def render():
            return overview.render(pipeline.latest_nav, fps=pipeline.fps,
                                   extra_info=extra_info())
        return mjpeg_response(render, CONFIG['MAP_FEED_FPS'])

    @app.route('/follow/<car_id>')
    def follow(car_id):
        """내비게이션 화면에 띄울 차량을 지정. 'auto'면 자동 선택."""
        my_car["id"] = None if car_id == "auto" else car_id
        return f"내비게이션 대상: {my_car['id'] or '자동 선택'}"

    @app.route('/enqueue/<car_id>')
    def enqueue(car_id):
        """차량번호 수동 등록 (빈자리 배정 + FIFO)."""
        register_car_number(car_id)
        return f"등록: {car_id} (대기 {car_number_fifo.size()}대)"

    @app.route('/status')
    def status():
        """현재 추적/내비게이션 상태를 JSON으로 반환."""
        return build_status(pipeline)

    from logic.B03_map_setting import register_map_routes
    import logic.B00_camera_input as b00_camera_input
    register_map_routes(app, pipeline, cap_module=b00_camera_input)

    @app.route('/')
    def index():
        if not pipeline.navigator.mapper.is_ready():
            from flask import redirect
            return redirect('/calibrate')
        return f"""
        <html>
            <head><title>Parking Navigation - D_main</title></head>
            <body style="background-color:#1a1a1a; color:white; text-align:center;
                         font-family:sans-serif; margin:0; padding:18px;">
                <h2 style="margin:6px 0;">Jetson Orin Nano - Parking Navigation</h2>
                <p style="color:#888; margin:0 0 18px 0;">
                    B00(Camera) -&gt; B01(Detection) -&gt; B02({pipeline.mot.tracker_type.upper()} MOT)
                    -&gt; C00/C01(Navigation) -&gt; D00(UI)
                </p>

                <img src="/nav_feed" style="max-width:100%;"
                     width="{D00_CONFIG['NAV_WIDTH']}" height="{D00_CONFIG['NAV_HEIGHT']}">

                <div style="display:flex; justify-content:center; gap:16px;
                            flex-wrap:wrap; align-items:flex-start; margin-top:18px;">
                    <div>
                        <h3 style="font-weight:normal; color:#aaa;">Camera</h3>
                        <img src="/video_feed" style="max-width:100%;"
                             width="{C_CONFIG['CAM_WIDTH']}" height="{C_CONFIG['CAM_HEIGHT']}">
                    </div>
                    <div>
                        <h3 style="font-weight:normal; color:#aaa;">Overview (관제)</h3>
                        <img src="/map_feed" style="max-width:100%;"
                             width="{D00_CONFIG['MAP_WIDTH']}" height="{D00_CONFIG['MAP_HEIGHT']}">
                    </div>
                </div>

                <p style="color:#888; margin-top:20px;">
                    내비 대상 지정 <code>/follow/1234</code> | 자동 <code>/follow/auto</code>
                </p>
                <p style="color:#888;">
                    차량번호 수동 등록 <code>/enqueue/1234</code> |
                    현재 상태 <code>/status</code> |
                    호모그래피 재계산 <code>/recalibrate</code>
                </p>
            </body>
        </html>
        """

    print(f"\n[INFO] Flask 웹 서버를 시작합니다. "
          f"http://젯슨IP:{CONFIG['WEB_PORT']}/ 으로 접속하세요.")
    try:
        app.run(host=CONFIG['WEB_HOST'], port=CONFIG['WEB_PORT'], debug=False)
    finally:
        runner.stop()
        cap.release()
        print("[INFO] 카메라를 해제하고 종료합니다.")
