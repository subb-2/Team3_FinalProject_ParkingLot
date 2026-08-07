"""
E_main_final : 최종 통합 실행 파일

이것 하나만 실행하면 전체 시스템이 뜬다.

    [1단계] UART   A00_uart_rx  Zybo Wi-Fi(TCP) 수신 서버
    [2단계] 카메라  B00          카메라 열기
    [3단계] MOT    B01/B02/C00  검출 -> 추적 -> 위치추정 -> 경로
    [4단계] 화면    E00          3분할 관제 화면 + 감시 스레드

    python logic/E_main_final.py

왜 이 순서인가
  UART를 먼저 띄운다. 카메라와 모델 로딩에 수 초가 걸리는데, 그 사이에 Zybo가
  접속을 시도하면 연결이 거부된다. 수신 서버가 먼저 서 있으면 언제 들어와도
  받는다. 반대로 카메라를 먼저 열면 그 시간 동안 온 입차를 놓친다.

  카메라는 MOT보다 먼저 연다. 카메라가 없으면 추적기를 만들 이유가 없고,
  YOLO 엔진을 로드한 뒤에 실패를 알게 되면 그만큼 시간을 버린다.

각 단계의 세부 설정은 해당 모듈의 CONFIG에서 관리한다.
  - 카메라 / 검출 / 추적 / 경로 : C_main.py 및 그 하위 모듈
  - Wi-Fi 주소와 포트           : A00_uart_rx.get_wifi_config()
  - 화면 구성과 오주차 판정      : E00_final_ui.py
여기서는 '넷을 순서대로 띄우는 일'에 필요한 설정만 둔다.
"""

import sys
import os
import time
import threading

import cv2
from flask import Flask, Response, request

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logic.C_main import (
    CONFIG as C_CONFIG, CALIBRATE_HTML,
    open_camera, build_pipeline, register_car_number, build_status,
)
from logic.D_main import PipelineRunner, mjpeg_response
from logic.D00_ui_navi import NavigationView, pick_my_vehicle
from logic.E00_final_ui import (
    CONFIG as E00_CONFIG, RxFeed, ParkingWatcher,
    build_lot_layout, build_ui_state, render_page,
)
from logic.B02_car_mot import car_number_fifo

# 설정 (Configuration)
CONFIG = {
    # 웹 서버
    "WEB_HOST": "0.0.0.0",
    "WEB_PORT": 5000,

    # 영상 두 구간의 전송 프레임레이트 (Hz).
    # 영상 처리는 백그라운드 스레드가 최대 속도로 돌고, 화면은 이 값으로 내보낸다.
    "CAM_FEED_FPS": 20,             # 3번 구간 : 카메라 원본 + 검출/추적 오버레이
    "NAV_FEED_FPS": 20,             # 4번 구간 : 차량 시점 안내

    # 3번 구간에 띄울 '내 차'. None이면 안내 중인 차를 자동으로 고른다.
    # 실행 중에는 /follow/<차량번호> 로 바꿀 수 있다.
    "MY_CAR_ID": None,

    # Zybo 수신 서버를 띄울지 여부.
    #
    # 끄면 Zybo 없이 화면만 확인할 수 있다. 그때는 /enqueue/1234 로 번호를
    # 직접 넣으면 1번 구간에 '수동'으로 표시된다.
    # C_main.CONFIG['ENABLE_UART']와 별개로 여기서 정한다. 최종 실행은
    # UART가 기본이어야 하는데, C_main의 값은 단독 테스트용으로 False이기 때문이다.
    "ENABLE_UART": True,
}


def stage(number, title):
    """단계 시작을 알린다. 어디서 멈췄는지 터미널만 보고 알 수 있어야 한다."""
    print(f"\n[{number}/4] {title}")
    print("-" * 46)


def start_uart(rx_feed):
    """
    [1단계] Zybo Wi-Fi(TCP) 수신 서버를 백그라운드로 띄운다.

    A00은 내부에서 자리 배정(handle_car_entry)과 FIFO 등록(enqueue_car_number)을
    모두 처리한다. 여기서는 화면에 띄울 수 있도록 수신 이벤트만 넘겨받는다.

    Returns:
        띄웠으면 True.
    """
    if not CONFIG['ENABLE_UART']:
        print("  UART 수신을 끄고 실행합니다. (CONFIG['ENABLE_UART'] = False)")
        print("  차량번호는 /enqueue/1234 로 직접 넣을 수 있습니다.")
        return False

    from logic.A00_uart_rx import uart_rx_main, add_rx_listener, get_wifi_config

    # 화면(1번 구간)이 수신 값을 받아볼 수 있도록 등록.
    # 서버를 띄우기 전에 등록해야 첫 패킷부터 놓치지 않는다.
    add_rx_listener(rx_feed.on_rx)

    config = get_wifi_config()
    threading.Thread(target=uart_rx_main, daemon=True).start()

    print(f"  Zybo 접속 주소 {config['jetson_ip']}:{config['entry_port']}(입차) "
          f"/ {config['exit_port']}(출차)")
    print("  수신 서버를 띄웠습니다. 언제 접속해도 받습니다.")
    return True


def start_watcher(pipeline, watcher):
    """
    [4단계] 주차 완료와 오주차를 감시하는 스레드를 띄운다.

    파이프라인 스레드 안에서 하지 않는 이유는, 감시에서 예외가 나도 영상
    처리가 멈추면 안 되기 때문이다. 둘을 분리해 두면 서로를 죽이지 않는다.
    """
    interval = E00_CONFIG['WATCH_INTERVAL_SEC']

    # 이미 세워져 있는 차를 '본 것'으로 표시한다. 이게 없으면 첫 tick에서
    # 미리 세워둔 차 전부가 방금 주차한 것으로 잡혀 안내가 쏟아진다.
    watcher.prime()

    def loop():
        last_error = None
        while True:
            try:
                watcher.poll(pipeline)
            except Exception as exc:
                # 같은 오류로 터미널을 도배하지 않는다
                key = f"{type(exc).__name__}: {exc}"
                if key != last_error:
                    last_error = key
                    print(f"[ERROR] 주차 감시 실패: {key}")
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()
    print(f"  주차 완료 / 오주차 감시 스레드를 시작했습니다. ({interval}초 간격)")


def main():
    print("=" * 46)
    print(" E_main_final : 주차 관제 시스템 통합 실행")
    print(" A00(UART) -> B00(카메라) -> B01/B02(MOT)")
    print("   -> C00/C01(위치추정/경로) -> E00(최종 화면)")
    print("=" * 46)

    rx_feed = RxFeed()
    watcher = ParkingWatcher(rx_feed)

    # [1단계] UART -----------------------------------------------------
    # 카메라보다 먼저 띄운다. 로딩 중에 온 입차를 놓치지 않기 위해서다.
    stage(1, "Zybo Wi-Fi(UART) 수신 서버")
    uart_on = start_uart(rx_feed)

    # [2단계] 카메라 ----------------------------------------------------
    stage(2, "카메라")
    cap = open_camera()
    if cap is None:
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태와 CAM_SENSOR_ID를 확인하세요.")
        sys.exit(1)
    print(f"  {C_CONFIG['CAM_WIDTH']}x{C_CONFIG['CAM_HEIGHT']} "
          f"{C_CONFIG['CAM_FPS']}fps 로 열었습니다.")

    # [3단계] MOT -------------------------------------------------------
    # 검출기(YOLO) 로딩이 가장 오래 걸리는 단계다.
    stage(3, "검출 / 추적 / 위치추정")
    pipeline = build_pipeline(cap)

    # UART가 꺼져 있을 때만 테스트 번호를 넣는다.
    # 둘 다 넣으면 실제 입차와 테스트 번호가 섞여 FIFO 순서가 어긋난다.
    if not uart_on:
        from data.car_data import TEST_PRESET_CAR_NUMBERS
        for car_id in TEST_PRESET_CAR_NUMBERS:
            result = register_car_number(car_id)
            rx_feed.push_manual(car_id, result)
        print(f"  테스트 차량번호 {len(TEST_PRESET_CAR_NUMBERS)}개를 등록했습니다.")

    runner = PipelineRunner(pipeline)
    runner.start()

    # [4단계] 화면 ------------------------------------------------------
    stage(4, "최종 관제 화면")
    view = NavigationView(navigator=pipeline.navigator)
    start_watcher(pipeline, watcher)

    my_car = {"id": CONFIG['MY_CAR_ID']}
    app = Flask(__name__)

    # --- 화면 ---------------------------------------------------------
    @app.route('/')
    def index():
        """세로 3분할 최종 화면."""
        return render_page()

    @app.route('/lot_layout')
    def lot_layout():
        """격자 배치. 브라우저가 처음 한 번만 가져간다."""
        return build_lot_layout()

    @app.route('/ui_state')
    def ui_state():
        """세 구간이 그릴 현재 상태 전부."""
        return build_ui_state(pipeline, rx_feed, watcher)

    @app.route('/video_feed')
    def video_feed():
        """3번 구간 : 카메라 원본 + 검출/추적/마커 오버레이 (MJPEG)."""
        return mjpeg_response(runner.latest_frame, CONFIG['CAM_FEED_FPS'])

    @app.route('/nav_feed')
    def nav_feed():
        """4번 구간 : 차량 시점 실시간 안내 (MJPEG)."""
        def render():
            nav = pick_my_vehicle(pipeline.latest_nav, my_car["id"])
            return view.render(nav, fps=pipeline.fps)
        return mjpeg_response(render, CONFIG['NAV_FEED_FPS'])

    # --- 조작 ---------------------------------------------------------
    @app.route('/enqueue/<car_id>')
    def enqueue(car_id):
        """Zybo 없이 차량번호를 등록. (자리 배정 포함, 1번 구간에 '수동'으로 표시)"""
        result = register_car_number(car_id)
        rx_feed.push_manual(car_id, result)
        if result is None:
            return f"등록: {car_id} (대기 {car_number_fifo.size()}대)"
        return {
            "car_id": car_id,
            "assigned": result["success"],
            "spot_id": result["spot_id"],
            "message": result["message"],
        }

    @app.route('/overlay')
    @app.route('/overlay/<state>')
    def overlay(state=None):
        """
        3번 구간(카메라)에 격자 배치를 겹쳐 그릴지 켜고 끈다.

        보정이 맞는지 확인할 때는 켜 두는 것이 좋지만, 시연 중에는 격자가
        차량 위에 겹쳐 검출 박스를 가린다. 그때 꺼서 화면을 정리한다.
        (/overlay/on, /overlay/off, /overlay 로 토글)
        """
        if state == 'on':
            C_CONFIG['DRAW_LAYOUT_OVERLAY'] = True
        elif state == 'off':
            C_CONFIG['DRAW_LAYOUT_OVERLAY'] = False
        else:
            C_CONFIG['DRAW_LAYOUT_OVERLAY'] = not C_CONFIG['DRAW_LAYOUT_OVERLAY']
        return f"배치 오버레이: {'ON' if C_CONFIG['DRAW_LAYOUT_OVERLAY'] else 'OFF'}"

    @app.route('/follow/<car_id>')
    def follow(car_id):
        """3번 구간에 띄울 차량을 지정. 'auto'면 자동 선택."""
        my_car["id"] = None if car_id == "auto" else car_id
        return f"안내 대상: {my_car['id'] or '자동 선택'}"

    @app.route('/status')
    def status():
        """추적/내비게이션 상태 JSON. (점검용)"""
        return build_status(pipeline)

    @app.route('/recalibrate')
    def recalibrate():
        """카메라를 다시 설치했을 때 호모그래피를 재계산."""
        pipeline.navigator.mapper.reset()
        return "호모그래피를 초기화했습니다. 마커가 보이면 자동으로 재계산됩니다."

    # --- 수동 보정 (C_main과 동일) --------------------------------------
    @app.route('/snapshot')
    def snapshot():
        ok, frame = pipeline.cap.read()
        if not ok:
            return "카메라 프레임을 읽을 수 없습니다.", 503
        ok, buf = cv2.imencode('.jpg', frame)
        return Response(buf.tobytes(), mimetype='image/jpeg')

    @app.route('/calibrate')
    def calibrate():
        """기둥을 순서대로 클릭해 호모그래피를 만드는 페이지."""
        import json
        from data.map_data import PILL_MARKER_ID
        order = sorted(PILL_MARKER_ID.items(), key=lambda kv: kv[1])
        steps = [{"id": mid, "row": cell[0], "col": cell[1]} for cell, mid in order]
        return CALIBRATE_HTML.replace("__STEPS__", json.dumps(steps, ensure_ascii=False))

    @app.route('/calibrate/save', methods=['POST'])
    def calibrate_save():
        data = request.get_json(silent=True) or {}
        points = {int(k): v for k, v in (data.get("points") or {}).items()}
        ok, message = pipeline.navigator.mapper.set_homography_from_points(points)
        return {"ok": ok, "message": message}

    print(f"\n[준비 완료] http://젯슨IP:{CONFIG['WEB_PORT']}/ 으로 접속하세요.")
    print(f"  좌표계가 안 잡히면 /calibrate 에서 기둥을 직접 찍으세요.")
    print("=" * 46)

    try:
        app.run(host=CONFIG['WEB_HOST'], port=CONFIG['WEB_PORT'], debug=False)
    finally:
        runner.stop()
        cap.release()
        print("\n[INFO] 카메라를 해제하고 종료합니다.")


if __name__ == '__main__':
    main()
