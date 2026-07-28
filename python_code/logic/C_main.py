import cv2
import time
import sys
import os
import threading
from datetime import datetime
from flask import Flask, Response

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B00_camera_input import get_camera
from logic.B01_car_detection import CarDetector, CONFIG as B01_CONFIG
from logic.B02_car_mot import (
    CarMOT, CONFIG as B02_CONFIG,
    car_number_fifo, enqueue_car_number, simulate_uart_rx
)
from logic.C00_navigation import (
    MarkerMapper, ParkingNavigator, CONFIG as C00_CONFIG, GATE_WORLD_POS
)
from logic.C01_path_planner import (
    ParkingLotMap, RoutePlanner, CONFIG as C01_CONFIG
)

# 설정 (Configuration)
# 각 모듈의 세부 설정은 해당 모듈의 CONFIG에서 관리.
#   - 카메라     : B00_camera_input.py
#   - 검출       : B01_car_detection.py  (모델 경로, conf, imgsz 등)
#   - 추적       : B02_car_mot.py        (ByteTrack, FIFO 매칭 등)
#   - 내비게이션 : C00_navigation.py     (ArUco 마커, 호모그래피, 안내 기준)
# 여기서는 통합 실행에 필요한 설정만 관리.
CONFIG = {
    # 카메라 설정
    "CAM_SENSOR_ID": 0,             # 카메라 장치 번호
    "CAM_WIDTH": 640,               # 카메라 가로 해상도
    "CAM_HEIGHT": 480,              # 카메라 세로 해상도
    "CAM_FPS": 30,                  # 카메라 프레임레이트

    # 웹 스트리밍 서버 설정
    "WEB_HOST": "0.0.0.0",
    "WEB_PORT": 5000,

    # 차량번호 입력 소스 설정
    "ENABLE_UART": False,           # True: A00_uart_rx로 실제 Zybo UART 수신 (하드웨어 필요)
    "TEST_PRESET_CAR_NUMBERS": ["1234", "1998", "0828", "9999"],  # UART 없이 테스트할 차량번호(순서대로 FIFO 등록)
    "TEST_UART_SIMULATOR": False,   # True: 임의의 차량번호를 주기적으로 자동 생성
    "TEST_UART_INTERVAL_SEC": 5.0,  # 자동 생성 주기(초)

    # 내비게이션 연동 설정
    "AUTO_ASSIGN_SPOT": True,       # 차량번호 등록 시 A01_parking_manager로 빈자리를 자동 배정
    "DRAW_MARKERS": True,           # 검출된 ArUco 마커를 화면에 표시
    "DRAW_NAVIGATION": True,        # 목표 지점 안내선을 화면에 표시
}


# 통합 파이프라인 (B + C)
class ParkingNavigationPipeline:
    """
    B00(카메라) -> B01(검출) -> B02(추적) -> C00(내비게이션)을
    하나의 flow로 연결하는 파이프라인.

    각 단계는 독립 모듈로 분리되어 있고, 이 클래스는 연결만 담당한다.
      - B00_camera_input : 프레임 획득
      - B01_car_detection: YOLO 차량 검출 (모델은 여기 한 곳에서만 로드)
      - B02_car_mot      : ByteTrack 추적 + FIFO 차량번호 매칭
      - C00_navigation   : ArUco 마커 기반 실좌표 변환 + 경로 안내
    """

    def __init__(self, cap, detector, mot, navigator):
        self.cap = cap
        self.detector = detector
        self.mot = mot
        self.navigator = navigator

        # 최근 처리 결과 (다른 모듈/모니터링에서 조회 가능)
        self.latest_tracks = []
        self.latest_nav = []
        self.fps = 0.0

    def process_frame(self, frame):
        """
        한 프레임에 대해 검출 -> 추적 -> 내비게이션 -> 시각화를 순서대로 수행.

        Args:
            frame: OpenCV BGR 이미지 (numpy array, 시각화로 원본이 수정됨)

        Returns:
            (tracks, nav_results) 튜플

        참고: 시각화 전에 내비게이션을 먼저 수행한다.
              draw_tracks가 프레임에 박스를 그리고 나면 ArUco 마커 검출이
              방해받을 수 있기 때문이다.
        """
        # 1) B01 : 차량 검출
        detections = self.detector.detect(frame)

        # 2) B02 : ByteTrack 추적 + 차량번호 매칭
        tracks = self.mot.update(detections)

        # 3) C00 : 배정된 목표 구역 동기화 후 위치 추정 및 경로 안내
        self.navigator.sync_targets_from_parking_manager()
        nav_results = self.navigator.update(frame, tracks)

        # 4) 시각화
        if CONFIG['DRAW_MARKERS']:
            self.navigator.mapper.draw_markers(frame, self.navigator.latest_markers)
        self.mot.draw_tracks(frame, tracks)
        if CONFIG['DRAW_NAVIGATION']:
            self.navigator.draw_navigation(frame, nav_results)

        self.latest_tracks = tracks
        self.latest_nav = nav_results
        return tracks, nav_results

    def draw_status(self, frame):
        """FPS, 추적 대수, FIFO 대기 수, 호모그래피 상태를 프레임에 표시."""
        matched = sum(1 for t in self.latest_tracks if t["car_id"])
        mapper = self.navigator.mapper
        ready = mapper.is_ready()

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Tracks: {len(self.latest_tracks)} (Matched: {matched})", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FIFO Waiting: {self.mot.fifo.size()}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # 호모그래피 상태를 품질까지 함께 표시.
        # 확정(LOCKED) 전에는 아직 좌표를 신뢰할 수 없다는 뜻이므로 구분해서 보여준다.
        if not ready:
            text, color = "Homography: NOT READY (show markers)", (0, 0, 255)
        elif mapper.locked:
            text = f"Homography: LOCKED ({mapper.calibrated_with} markers, {mapper.reproj_error:.1f}cm)"
            color = (0, 255, 0)
        else:
            text = (f"Homography: PROVISIONAL ({mapper.calibrated_with}/{mapper.lock_markers} "
                    f"markers, {mapper.reproj_error:.1f}cm)")
            color = (0, 200, 255)
        cv2.putText(frame, text, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return frame

    def generate_frames(self):
        """
        Flask MJPEG 스트리밍용 제너레이터.
        카메라 -> 검출 -> 추적 -> 내비게이션 -> JPEG 인코딩을 반복한다.
        """
        prev_time = time.time()
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] 카메라 프레임을 읽을 수 없습니다. 스트리밍을 종료합니다.")
                break

            tracks, nav_results = self.process_frame(frame)

            # FPS 계산 (0.5초 간격)
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= 0.5:
                self.fps = frame_count / (current_time - prev_time)
                prev_time = current_time
                frame_count = 0

            self.draw_status(frame)

            # 터미널에 안내 상태 출력 (FPS 갱신 시점마다)
            if frame_count == 1 and nav_results:
                print(f"[FPS: {self.fps:.1f}] 내비게이션 {len(nav_results)}대:")
                for n in nav_results:
                    car_str = n["car_id"] if n["car_id"] else "미매칭"
                    wx, wy = n["world_pos"]
                    dist = f"{n['distance_cm']:.0f}cm" if n["distance_cm"] is not None else "-"
                    print(f"  {car_str:8s} 위치=({wx:6.1f},{wy:6.1f})cm "
                          f"목표={str(n['target_spot']):5s} 거리={dist:8s} 안내={n['guide_text']}")

            # JPEG 압축 후 웹 스트리밍 반환
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# 차량번호 입력 소스 준비
def register_car_number(car_id):
    """
    차량번호를 시스템에 등록.

    AUTO_ASSIGN_SPOT이 True면 A01_parking_manager로 빈자리를 먼저 배정한 뒤
    FIFO에 넣는다. 배정 정보(cars_info)가 있어야 C00이 목표 구역을 알 수 있다.

    Args:
        car_id: 차량 번호 4자리 문자열
    """
    if CONFIG['AUTO_ASSIGN_SPOT']:
        from logic.A01_parking_manager import handle_car_entry
        handle_car_entry(car_id, datetime.now())

    enqueue_car_number(car_id)


def setup_car_number_source():
    """
    FIFO에 차량번호를 공급할 소스를 설정에 따라 준비.
      - ENABLE_UART            : A00_uart_rx를 별도 스레드로 실행 (실제 Zybo 연동)
      - TEST_PRESET_CAR_NUMBERS: 지정한 번호를 순서대로 미리 등록
      - TEST_UART_SIMULATOR    : 임의 번호를 주기적으로 자동 생성

    Returns:
        시뮬레이터 종료용 threading.Event
    """
    stop_event = threading.Event()

    # 1) 실제 UART 수신 (A00_uart_rx)
    #    A00은 내부에서 handle_car_entry와 enqueue_car_number를 모두 호출하므로
    #    별도의 빈자리 배정이 필요 없다.
    if CONFIG['ENABLE_UART']:
        from logic.A00_uart_rx import uart_rx_main
        threading.Thread(target=uart_rx_main, daemon=True).start()
        print("[INFO] A00_uart_rx 수신 스레드를 시작했습니다.")

    # 2) 미리 지정해 둔 차량번호를 순서대로 등록 (빈자리 배정 + FIFO)
    for car_id in CONFIG['TEST_PRESET_CAR_NUMBERS']:
        register_car_number(car_id)

    # 3) 가짜 UART 송신기 (임의 번호 자동 생성)
    if CONFIG['TEST_UART_SIMULATOR']:
        threading.Thread(
            target=simulate_uart_rx,
            args=(CONFIG['TEST_UART_INTERVAL_SEC'], stop_event),
            daemon=True
        ).start()

    return stop_event


def build_pipeline(cap):
    """
    B01 검출기, B02 추적기, C00 내비게이터를 생성하고 파이프라인으로 묶는다.

    Args:
        cap: 열려 있는 카메라 객체

    Returns:
        ParkingNavigationPipeline 인스턴스
    """
    # B01 : 차량 검출기 (YOLO 모델은 이 한 곳에서만 로드)
    detector = CarDetector(
        model_path=B01_CONFIG['MODEL_PATH'],
        conf=B01_CONFIG['CONF_THRESH'],
        iou=B01_CONFIG['IOU_THRESH'],
        imgsz=B01_CONFIG['IMGSZ']
    )

    # B02 : ByteTrack 추적기
    mot = CarMOT(
        tracker_cfg=B02_CONFIG['TRACKER_CFG'],
        min_hits=B02_CONFIG['MIN_HITS_FOR_ASSIGN'],
        lost_ttl=B02_CONFIG['LOST_TTL_FRAMES'],
        trajectory_maxlen=B02_CONFIG['TRAJECTORY_MAXLEN']
    )

    # C00 : ArUco 마커 매퍼 + 내비게이터
    mapper = MarkerMapper(
        aruco_dict_name=C00_CONFIG['ARUCO_DICT'],
        min_markers=C00_CONFIG['MIN_MARKERS_FOR_HOMOGRAPHY'],
        lock_homography=C00_CONFIG['LOCK_HOMOGRAPHY'],
        lock_markers=C00_CONFIG['MARKERS_FOR_LOCK'],
        max_error=C00_CONFIG['MAX_REPROJ_ERROR_CM'],
        min_spread=C00_CONFIG['MIN_MARKER_SPREAD'],
        ransac_thresh_cm=C00_CONFIG['RANSAC_THRESH_CM']
    )
    navigator = ParkingNavigator(
        mapper=mapper,
        arrival_threshold=C00_CONFIG['ARRIVAL_THRESHOLD_CM'],
        turn_threshold=C00_CONFIG['TURN_ANGLE_THRESHOLD_DEG'],
        uturn_threshold=C00_CONFIG['UTURN_ANGLE_THRESHOLD_DEG'],
        min_move_for_heading=C00_CONFIG['MIN_MOVE_CM_FOR_HEADING'],
        heading_window=C00_CONFIG['HEADING_WINDOW'],
        history_maxlen=C00_CONFIG['HISTORY_MAXLEN'],
        replan_tolerance=C01_CONFIG['REPLAN_TOLERANCE_CM']
    )

    # C01 : 경로 계획기 (주차 구역을 장애물로 두고 통로를 따라 경로 생성)
    lot_map = ParkingLotMap(
        navigator.spot_world_pos, GATE_WORLD_POS,
        resolution=C01_CONFIG['GRID_RESOLUTION_CM'],
        spot_w=C01_CONFIG['SPOT_W_CM'],
        spot_h=C01_CONFIG['SPOT_H_CM'],
        clearance=C01_CONFIG['VEHICLE_CLEARANCE_CM'],
        lot_margin=C01_CONFIG['LOT_MARGIN_CM'],
    )
    navigator.planner = RoutePlanner(lot_map, simplify=C01_CONFIG['SIMPLIFY_PATH'])

    return ParkingNavigationPipeline(cap, detector, mot, navigator)


def open_camera():
    """설정에 따라 카메라를 열고 반환. 실패 시 None."""
    print(f"[INFO] 카메라를 엽니다... ({CONFIG['CAM_WIDTH']}x{CONFIG['CAM_HEIGHT']})")
    cap = get_camera(
        sensor_id=CONFIG['CAM_SENSOR_ID'],
        width=CONFIG['CAM_WIDTH'],
        height=CONFIG['CAM_HEIGHT'],
        framerate=CONFIG['CAM_FPS']
    )
    return cap if cap.isOpened() else None


def build_status(pipeline):
    """현재 추적/내비게이션 상태를 JSON 직렬화 가능한 딕셔너리로 반환."""
    mapper = pipeline.navigator.mapper
    return {
        "fps": round(pipeline.fps, 1),
        "homography_ready": mapper.is_ready(),
        "homography_locked": mapper.locked,
        "homography_markers": mapper.calibrated_with,
        "homography_error_cm": (round(mapper.reproj_error, 2)
                                if mapper.is_ready() else None),
        "markers_detected": sorted(pipeline.navigator.latest_markers.keys()),
        "fifo_waiting": car_number_fifo.snapshot(),
        "vehicles": [
            {
                "track_id": n["track_id"],
                "car_id": n["car_id"],
                "world_pos_cm": [round(n["world_pos"][0], 1), round(n["world_pos"][1], 1)],
                "heading_deg": round(n["heading_deg"], 1) if n["heading_deg"] is not None else None,
                "target_spot": n["target_spot"],
                "distance_cm": round(n["distance_cm"], 1) if n["distance_cm"] is not None else None,
                "guide": n["guide"],
                "route": [[round(p[0], 1), round(p[1], 1)] for p in n["route"]] if n.get("route") else None,
                "route_index": n.get("route_index"),
            }
            for n in pipeline.latest_nav
        ],
    }


# 메인 (B00 + B01 + B02 + C00 통합 실행)
if __name__ == '__main__':
    print("==========================================")
    print(" C_main : 주차장 비전 + 내비게이션 통합 실행")
    print(" B00(카메라) -> B01(검출) -> B02(추적) -> C00(내비)")
    print("==========================================")

    cap = open_camera()
    if cap is None:
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        sys.exit(1)

    pipeline = build_pipeline(cap)

    # 차량번호 입력 소스 준비 (UART 또는 테스트용)
    uart_sim_stop = setup_car_number_source()

    app = Flask(__name__)

    @app.route('/')
    def index():
        return f"""
        <html>
            <head><title>Jetson Parking Navigation (C_main)</title></head>
            <body style="background-color: #222; color: white; text-align: center;">
                <h2>Jetson Orin Nano - Parking Navigation Pipeline</h2>
                <p>B00(Camera) -&gt; B01(Detection) -&gt; B02(ByteTrack MOT) -&gt; C00(Navigation)</p>
                <img src="/video_feed" width="{CONFIG['CAM_WIDTH']}" height="{CONFIG['CAM_HEIGHT']}">
                <p>차량번호 수동 등록: <code>/enqueue/1234</code> | 현재 상태: <code>/status</code></p>
                <p>호모그래피 재계산: <code>/recalibrate</code></p>
            </body>
        </html>
        """

    @app.route('/video_feed')
    def video_feed():
        return Response(pipeline.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/enqueue/<car_id>')
    def enqueue(car_id):
        """UART 없이 차량번호를 등록하기 위한 라우트. (빈자리 배정 포함)"""
        register_car_number(car_id)
        return f"등록: {car_id} (대기 {car_number_fifo.size()}대) / 현재 큐: {car_number_fifo.snapshot()}"

    @app.route('/status')
    def status():
        """현재 추적/내비게이션 상태와 FIFO 대기열을 조회."""
        return build_status(pipeline)

    @app.route('/recalibrate')
    def recalibrate():
        """카메라를 다시 설치했을 때 호모그래피를 재계산."""
        pipeline.navigator.mapper.reset()
        return "호모그래피를 초기화했습니다. 마커가 보이면 자동으로 재계산됩니다."

    print(f"\n[INFO] Flask 웹 서버를 시작합니다. http://젯슨IP:{CONFIG['WEB_PORT']}/ 으로 접속하세요.")
    try:
        app.run(host=CONFIG['WEB_HOST'], port=CONFIG['WEB_PORT'], debug=False)
    finally:
        uart_sim_stop.set()
        cap.release()
        print("[INFO] 카메라를 해제하고 종료합니다.")
