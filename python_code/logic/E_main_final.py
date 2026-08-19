"""
E_main_final : 최종 통합 실행 파일

이것 하나만 실행하면 전체 시스템이 뜬다.

    [1단계] UART   A00_uart_rx  Zybo Wi-Fi(TCP) 수신 서버
    [2단계] 카메라  B00          카메라 열기
    [3단계] MOT    B01/B02/C00  검출 -> 추적 -> 위치추정 -> 경로
    [4단계] 화면    E00          4구간 관제 화면 + 감시 스레드

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

from flask import Flask

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logic.C_main import (
    CONFIG as C_CONFIG,
    open_camera, build_pipeline, register_car_number, build_status,
)
from logic.D_main import PipelineRunner, mjpeg_response, mjpeg_runner_response
from logic.D00_ui_navi import NavigationView, pick_my_vehicle
from logic.E00_final_ui import (
    CONFIG as E00_CONFIG, RxFeed, ParkingWatcher,
    build_lot_layout, build_ui_state, render_page,
)
from logic.B02_car_mot import car_number_fifo
# 시작 시점에 배정된 자리가 있는지 확인하는 데만 쓴다. (report_assignments)
from data.car_data import cars_info

# 설정 (Configuration)
CONFIG = {
    # 웹 서버
    "WEB_HOST": "0.0.0.0",
    "WEB_PORT": 5000,

    # 영상 두 구간의 전송 프레임레이트 (Hz).
    # 영상 처리는 백그라운드 스레드가 최대 속도로 돌고, 화면은 이 값으로 내보낸다.
    "CAM_FEED_FPS": 20,             # 2번 구간 : 카메라 원본 + 검출/추적 오버레이
    "NAV_FEED_FPS": 20,             # 4번 구간 : 차량 시점 안내

    # 4번 구간(차량 시점 안내)에 띄울 '내 차'. None이면 자동으로 고른다.
    # 3번 격자에 깔리는 안내 경로도 같은 차를 따라간다.
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


def register_test_cars(rx_feed):
    """
    UART 없이 돌릴 때 쓰는 테스트 차량번호를 등록한다.

    UART를 켜면 아무것도 하지 않는다. B_main / C_main과 같은 규칙이다.
    켠 채로 넣으면 프로그램이 뜨자마자 그 번호에 자리가 배정되어, Zybo에서는
    아직 아무 값도 오지 않았는데 화면에는 목적지가 깜빡이고 있게 된다.
    실물 수신을 시험하는 중에 그 자리는 이미 배정 후보에서도 빠져 있다.

    UART를 켠 상태에서 번호를 하나 넣어 보려면 /enqueue/1234 를 쓴다.
    그쪽은 사람이 부른 것이 분명하므로 1번 구간에 '수동'으로 표시된다.
    """
    if CONFIG['ENABLE_UART']:
        return

    from data.car_data import TEST_PRESET_CAR_NUMBERS
    if not TEST_PRESET_CAR_NUMBERS:
        return

    for car_id in TEST_PRESET_CAR_NUMBERS:
        result = register_car_number(car_id)
        rx_feed.push_manual(car_id, result)
    print(f"  미리 지정된 테스트 차량번호 {len(TEST_PRESET_CAR_NUMBERS)}개를 "
          f"등록했습니다. (UART가 꺼져 있을 때만)")


def report_assignments():
    """
    시작 시점에 이미 배정되어 있는 자리를 터미널에 적는다.

    화면에서 깜빡이는 자리는 '배정은 됐는데 아직 도착하지 않은 차'다. 그것이
    어디서 왔는지 화면만 보고는 알 수 없어서, Zybo에서 아무것도 안 왔는데
    자리가 깜빡인다는 이야기가 나온다. 시작할 때 한 줄 적어 두면 그 자리가
    프로그램이 만든 것인지 실제 수신으로 생긴 것인지 바로 갈린다.

    미리 세워둔 차(INITIAL_PARKED)는 여기 해당하지 않는다. 그쪽은 주차까지
    끝난 상태라 깜빡이지 않고 채워진 색으로만 보인다.
    """
    pending = {info["spot_id"]: car_id for car_id, info in cars_info.items()
               if info.get("spot_id") and not info.get("parked")}
    parked = sum(1 for info in cars_info.values() if info.get("parked"))

    print(f"  미리 세워둔 차 {parked}대 (자리 채움, 깜빡이지 않음)")
    if not pending:
        print("  배정 대기 중인 자리 없음. 수신이 와야 자리가 깜빡입니다.")
        return

    where = ", ".join(f"{spot}<-'{car}'" for spot, car in sorted(pending.items()))
    print(f"  [주의] 시작부터 배정된 자리가 있습니다: {where}")
    if CONFIG['ENABLE_UART']:
        print("         UART를 켰는데 이러면 수신 말고 다른 경로로 들어온 것입니다.")
        print("         (data/car_data.py의 TEST_PRESET_CAR_NUMBERS, /enqueue 확인)")


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
    start_uart(rx_feed)

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

    register_test_cars(rx_feed)
    report_assignments()

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
        """최종 관제 화면. 맵 보정이 안 되어 있으면 보정 페이지로 보낸다."""
        if not pipeline.navigator.mapper.is_ready():
            from flask import redirect
            return redirect('/calibrate')
        return render_page()

    @app.route('/lot_layout')
    def lot_layout():
        """격자 배치. 브라우저가 처음 한 번만 가져간다."""
        return build_lot_layout()

    @app.route('/ui_state')
    def ui_state():
        """네 구간이 그릴 현재 상태 전부."""
        # my_car를 함께 넘긴다. 3번 격자에 깔 경로와 4번이 안내하는 차가
        # 같아야 두 화면을 짝지어 읽을 수 있다. (/follow로 지정한 차 포함)
        return build_ui_state(pipeline, rx_feed, watcher, my_car["id"])

    @app.route('/video_feed')
    def video_feed():
        """2번 구간 : 카메라 원본 + 검출/추적/마커 오버레이 (MJPEG)."""
        return mjpeg_runner_response(runner, CONFIG['CAM_FEED_FPS'])

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
        2번 구간(카메라)에 격자 배치를 겹쳐 그릴지 켜고 끈다.

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

    @app.route('/rawdet')
    @app.route('/rawdet/<state>')
    def rawdet(state=None):
        """
        YOLO가 찾은 박스를 신뢰도와 함께 전부 얇게 그린다.

        어떤 차만 초록 박스가 안 뜰 때 여기를 켠다. 원인이 둘이고 손볼 곳이
        완전히 다르다.
          얇은 박스도 없다        -> YOLO가 못 본다. 학습 데이터에 그 차가 없거나
                                    조명/각도 문제. 모델을 다시 학습해야 한다.
          얇은 박스가 'LOW'로 뜬다 -> 신뢰도가 config/ocsort.yaml의
                                    new_track_thresh에 못 미쳐 트랙이 안 생긴 것.
                                    임계값을 내리면 된다.
        (/rawdet/on, /rawdet/off, /rawdet 로 토글)
        """
        if state == 'on':
            C_CONFIG['DRAW_RAW_DETECTIONS'] = True
        elif state == 'off':
            C_CONFIG['DRAW_RAW_DETECTIONS'] = False
        else:
            C_CONFIG['DRAW_RAW_DETECTIONS'] = not C_CONFIG['DRAW_RAW_DETECTIONS']
        return f"원본 검출 표시: {'ON' if C_CONFIG['DRAW_RAW_DETECTIONS'] else 'OFF'}"

    @app.route('/follow/<car_id>')
    def follow(car_id):
        """4번 구간(과 3번 격자의 경로)에 띄울 차량을 지정. 'auto'면 자동 선택."""
        my_car["id"] = None if car_id == "auto" else car_id
        return f"안내 대상: {my_car['id'] or '자동 선택'}"

    @app.route('/status')
    def status():
        """추적/내비게이션 상태 JSON. (점검용)"""
        return build_status(pipeline)

    @app.route('/debug')
    @app.route('/debug/<state>')
    def debug(state=None):
        """
        영상 위 상태 글자(FPS / Tracks / FIFO / Map / 단계별 소요시간)를 켜고 끈다.

        2번 구간의 [디버그] 버튼이 이 주소를 부른다. 기본은 꺼짐이다.
        같은 값을 화면 아래 정보줄이 한글로 이미 보여주므로, 영상 위 글자는
        점검할 때만 필요하고 평소에는 왼쪽 위 차량을 가리기만 한다.
        """
        if state == 'on':
            C_CONFIG['DRAW_STATUS_TEXT'] = True
        elif state == 'off':
            C_CONFIG['DRAW_STATUS_TEXT'] = False
        else:
            C_CONFIG['DRAW_STATUS_TEXT'] = not C_CONFIG['DRAW_STATUS_TEXT']
        return {"debug": C_CONFIG['DRAW_STATUS_TEXT']}

    @app.route('/queue/reset')
    def queue_reset():
        """
        대기 중인 차량번호를 전부 지운다.

        번호판을 잘못 읽어 엉뚱한 번호가 들어왔을 때 쓴다. 2번 구간의
        [대기열 리셋] 버튼이 이 주소를 부른다.

        대기열만 비우면 모자란다. 번호는 받는 즉시 자리를 하나 배정받으므로
        (A00 -> A01.handle_car_entry), 큐에서만 지우면 그 자리는 아무도 오지
        않는 채로 막혀 있고 화면에서도 계속 깜빡인다. 그래서 아직 도착하지
        않은 차의 배정도 함께 되돌린다.

        이미 안내를 받고 있는 차와 주차를 마친 차는 건드리지 않는다. 그쪽은
        실제로 움직이고 있거나 자리에 서 있다.
        """
        from logic.A01_parking_manager import cancel_assignment

        dropped = car_number_fifo.clear()
        released = [spot for spot in
                    (cancel_assignment(car_id) for car_id in dropped) if spot]
        return {"cleared": dropped, "released": released,
                "waiting": car_number_fifo.size()}

    @app.route('/recalibrate')
    def recalibrate():
        """
        저장해 둔 기둥 보정을 지우고 처음부터 다시 찍는다.

        보정은 config/pillar_pixels.json에 저장되고 다음 실행부터 자동으로
        불러온다. 그래서 첫 화면이 더 이상 /calibrate로 넘어가지 않는다.
        카메라를 옮겼거나 해상도를 바꿨으면 저장된 좌표가 어긋나므로
        여기로 지우고 다시 찍어야 한다.
        """
        pipeline.navigator.mapper.reset()
        from flask import redirect
        return redirect('/calibrate')

    from logic.B03_map_setting import register_map_routes
    import logic.B00_camera_input as b00_camera_input
    # runner를 넘겨야 /snapshot이 카메라를 다시 읽지 않는다.
    # 안 넘기면 처리 스레드와 프레임을 놓고 다퉈 보정 사진이 안 뜬다.
    register_map_routes(app, pipeline, cap_module=b00_camera_input, runner=runner)
    print(f"\n[준비 완료] http://젯슨IP:{CONFIG['WEB_PORT']}/ 으로 접속하세요.")
    if pipeline.navigator.mapper.is_ready():
        # 저장된 보정을 이미 불러왔다는 뜻이다. 이때 첫 화면은 보정으로
        # 넘어가지 않으므로, '기둥을 안 찍었는데 왜 보정이 안 뜨지'로 헷갈리기
        # 쉽다. 그래서 그 사실과 다시 찍는 방법을 함께 적는다.
        print("  좌표계가 이미 서 있어 보정 화면으로 넘어가지 않습니다.")
        print("  다시 찍으려면 화면 왼쪽 아래 [기둥 보정] 버튼 또는 /calibrate")
        print("  저장된 좌표까지 지우려면 /recalibrate")
    else:
        print("  좌표계가 아직 없습니다. 접속하면 /calibrate 로 넘어갑니다.")
        print("  기둥 10개를 표에 적힌 순서대로 찍어주세요.")
    print("=" * 46)

    try:
        # threaded=True를 명시한다. 이 화면은 영상 두 개를 MJPEG로 계속
        # 흘려보내는데, 한 번에 하나만 처리하는 서버라면 그 스트림이 서버를
        # 통째로 붙들어 나머지 요청이 영영 응답하지 않는다. 브라우저에서는
        # 페이지가 안 열리고 로딩만 도는 것으로 보인다.
        # (Flask 1.0부터 기본값이 True지만, 젯슨에 apt로 깔린 옛 버전은
        #  False라서 실제로 이 증상이 난다)
        app.run(host=CONFIG['WEB_HOST'], port=CONFIG['WEB_PORT'],
                debug=False, threaded=True)
    finally:
        runner.stop()
        cap.release()
        print("\n[INFO] 카메라를 해제하고 종료합니다.")


if __name__ == '__main__':
    main()
