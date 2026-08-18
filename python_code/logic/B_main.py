import cv2
import time
import sys
import os
import threading
from flask import Flask, Response

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B00_camera_input import get_camera
from logic.B01_car_detection import CarDetector, CONFIG as B01_CONFIG
from logic.B02_car_mot import (
    CarMOT, CONFIG as B02_CONFIG,
    car_number_fifo, enqueue_car_number
)

# 설정 (Configuration)
# 각 모듈의 세부 설정은 해당 모듈의 CONFIG에서 관리.
#   - 카메라 : B00_camera_input.py
#   - 검출   : B01_car_detection.py  (모델 경로, conf, imgsz 등)
#   - 추적   : B02_car_mot.py        (추적기 선택, FIFO 매칭 등)
#              실제 추적기는 config/*.yaml의 tracker_type이 결정한다.
#              현재 기본값: OC-SORT + ByteTrack 저신뢰 2차 연관 (config/ocsort.yaml)
# 여기서는 통합 실행에 필요한 설정만 관리.
CONFIG = {
    # 카메라 설정
    "CAM_SENSOR_ID": 0,             # 카메라 장치 번호
    # C_main / E_main_final과 반드시 같은 값을 쓴다.
    #
    # 여기만 다르게 두면 검출/추적을 확인한 화각과 실제로 운용하는 화각이
    # 달라진다. 담기는 범위가 다르고, 카메라에 따라서는 해상도마다 센서를
    # 잘라 쓰기도 한다. 여기서 잘 잡히던 차가 최종 실행에서 다르게 보이면
    # 원인을 찾기 어렵다.
    #
    # 해상도를 고른 이유는 C_main의 CAM_WIDTH 주석 참고. (학습 데이터가 4:3)
    "CAM_WIDTH": 1280,              # 카메라 가로 해상도
    "CAM_HEIGHT": 800,              # 카메라 세로 해상도
    "CAM_FPS": 30,                  # 카메라 프레임레이트

    # 웹 스트리밍 서버 설정
    "WEB_HOST": "0.0.0.0",
    "WEB_PORT": 5000,

    # 차량번호 입력 소스 설정
    "ENABLE_UART": False,           # True: A00_uart_rx로 실제 Zybo UART 수신 (하드웨어 필요)
    # UART 없이 테스트할 차량번호는 data/car_data.py의 TEST_PRESET_CAR_NUMBERS에서 관리한다.
    # 차량 종류 등록부(car_types)와 같은 곳에 두어야 어느 자리로 갈지 함께 볼 수 있다.
}

# 통합 파이프라인
class ParkingVisionPipeline:
    """
    B00(카메라) -> B01(검출) -> B02(추적/번호매칭)을 하나의 flow로 연결하는 파이프라인.

    각 단계는 독립 모듈로 분리되어 있고, 이 클래스는 연결만 담당한다.
      - B00_camera_input : 프레임 획득
      - B01_car_detection: YOLO 차량 검출 (모델은 여기 한 곳에서만 로드)
      - B02_car_mot      : MOT 추적(OC-SORT) + FIFO 차량번호 매칭
    """

    def __init__(self, cap, detector, mot):
        self.cap = cap
        self.detector = detector
        self.mot = mot

        # 최근 처리 결과 (다른 모듈/모니터링에서 조회 가능)
        self.latest_tracks = []
        self.fps = 0.0

    def process_frame(self, frame):
        """
        한 프레임에 대해 검출 -> 추적 -> 시각화를 순서대로 수행.

        Args:
            frame: OpenCV BGR 이미지 (numpy array, 시각화로 원본이 수정됨)

        Returns:
            추적 결과 리스트 (B02_car_mot.CarMOT.update()의 반환 형식)
        """
        # 1) B01 : 차량 검출
        detections = self.detector.detect(frame)

        # 2) B02 : MOT 추적 + 차량번호 매칭
        #    B_main에는 호모그래피도 목표 구역 정보도 없다. 따라서
        #      - 이동/거리 판정은 cm가 아니라 픽셀 기준으로 동작하고
        #      - 주차 완료(도착) 판정을 할 수 없어 활성 차량은
        #        SINGLE_ACTIVE['STUCK_RELEASE_SEC'](정지 시간)으로만 해제된다.
        #    전체 시나리오 확인은 C_main으로 할 것. 여기는 검출/추적 확인용이다.
        tracks = self.mot.update(detections)

        # 3) 시각화
        self.mot.draw_tracks(frame, tracks)

        self.latest_tracks = tracks
        return tracks

    def draw_status(self, frame, detections_count):
        """FPS, 추적 대수, FIFO 대기 수를 프레임에 표시."""
        matched = sum(1 for t in self.latest_tracks if t["car_id"])
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Tracks: {len(self.latest_tracks)} (Matched: {matched})", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FIFO Waiting: {self.mot.fifo.size()}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def generate_frames(self):
        """
        Flask MJPEG 스트리밍용 제너레이터.
        카메라 -> 검출 -> 추적 -> JPEG 인코딩을 반복한다.
        """
        prev_time = time.time()
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] 카메라 프레임을 읽을 수 없습니다. 스트리밍을 종료합니다.")
                break

            tracks = self.process_frame(frame)

            # FPS 계산 (0.5초 간격)
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= 0.5:
                self.fps = frame_count / (current_time - prev_time)
                prev_time = current_time
                frame_count = 0

            self.draw_status(frame, len(tracks))

            # 터미널에 추적 결과 출력 (FPS 갱신 시점마다)
            if frame_count == 1 and tracks:
                print(f"[FPS: {self.fps:.1f}] 추적 차량 {len(tracks)}대:")
                for t in tracks:
                    car_str = t["car_id"] if t["car_id"] else "미매칭"
                    print(f"  ID:{t['track_id']:<3d} {t['class_name']:12s} 번호={car_str:8s} conf={t['confidence']:.2f}")

            # JPEG 압축 후 웹 스트리밍 반환
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# 차량번호 입력 소스 준비
def setup_car_number_source():
    """
    FIFO에 차량번호를 공급할 소스를 설정에 따라 준비한다.
      - ENABLE_UART : A00_uart_rx를 별도 스레드로 실행 (실제 Zybo 연동)
      - 그 외       : car_data.TEST_PRESET_CAR_NUMBERS를 순서대로 등록

    UART를 켜면 테스트 번호는 넣지 않는다. 둘 다 넣으면 실제 입차와 테스트
    번호가 뒤섞여 FIFO 순서가 어긋난다.
    """
    if CONFIG['ENABLE_UART']:
        from logic.A00_uart_rx import uart_rx_main
        threading.Thread(target=uart_rx_main, daemon=True).start()
        print("[INFO] A00_uart_rx 수신 스레드를 시작했습니다.")
        return

    # 미리 지정해 둔 차량번호를 순서대로 FIFO에 등록
    # (B_main은 자리 배정 없이 추적만 확인하는 용도라 FIFO에만 넣는다)
    from data.car_data import TEST_PRESET_CAR_NUMBERS
    for car_id in TEST_PRESET_CAR_NUMBERS:
        enqueue_car_number(car_id)


# 메인 (B00 + B01 + B02 통합 실행)
if __name__ == '__main__':
    print("==========================================")
    print(" B_main : 주차장 비전 파이프라인 통합 실행")
    print(" B00(카메라) -> B01(검출) -> B02(추적)")
    print("==========================================")

    # B00 : 카메라 열기
    print(f"[INFO] 카메라를 엽니다... ({CONFIG['CAM_WIDTH']}x{CONFIG['CAM_HEIGHT']})")
    cap = get_camera(
        sensor_id=CONFIG['CAM_SENSOR_ID'],
        width=CONFIG['CAM_WIDTH'],
        height=CONFIG['CAM_HEIGHT'],
        framerate=CONFIG['CAM_FPS']
    )

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        sys.exit(1)

    # B01 : 차량 검출기 초기화 (YOLO 모델은 이 한 곳에서만 로드)
    detector = CarDetector(
        model_path=B01_CONFIG['MODEL_PATH'],
        conf=B01_CONFIG['CONF_THRESH'],
        iou=B01_CONFIG['IOU_THRESH'],
        imgsz=B01_CONFIG['IMGSZ']
    )

    # B02 : 추적기 초기화 (어떤 추적기인지는 TRACKER_CFG의 tracker_type이 결정)
    mot = CarMOT(
        tracker_cfg=B02_CONFIG['TRACKER_CFG'],
        min_hits=B02_CONFIG['MIN_HITS_FOR_ASSIGN'],
        trajectory_maxlen=B02_CONFIG['TRAJECTORY_MAXLEN']
    )

    # 차량번호 입력 소스 준비 (UART 또는 테스트용)
    setup_car_number_source()

    # 파이프라인 구성
    pipeline = ParkingVisionPipeline(cap, detector, mot)

    app = Flask(__name__)

    @app.route('/')
    def index():
        return f"""
        <html>
            <head><title>Jetson Parking Vision (B_main)</title></head>
            <body style="background-color: #222; color: white; text-align: center;">
                <h2>Jetson Orin Nano - Parking Vision Pipeline</h2>
                <p>B00(Camera) -&gt; B01(YOLOv8 Detection) -&gt; B02({pipeline.mot.tracker_type.upper()} MOT)</p>
                <img src="/video_feed" width="{CONFIG['CAM_WIDTH']}" height="{CONFIG['CAM_HEIGHT']}">
                <p>차량번호 수동 등록: <code>/enqueue/1234</code> | 현재 상태: <code>/status</code></p>
            </body>
        </html>
        """

    @app.route('/video_feed')
    def video_feed():
        return Response(pipeline.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/enqueue/<car_id>')
    def enqueue(car_id):
        """UART 없이 차량번호를 FIFO에 수동 등록하기 위한 라우트."""
        enqueue_car_number(car_id)
        return f"FIFO 등록: {car_id} (대기 {car_number_fifo.size()}대) / 현재 큐: {car_number_fifo.snapshot()}"

    @app.route('/status')
    def status():
        """현재 추적 상태와 FIFO 대기열을 조회."""
        return {
            "fps": round(pipeline.fps, 1),
            # 추적기 A/B 비교용. total_ids가 실제 차량 대수보다 크면 그만큼 ID가 바뀐 것.
            "tracker": pipeline.mot.track_id_stats(),
            "fifo_waiting": car_number_fifo.snapshot(),
            "tracks": [
                {
                    "track_id": t["track_id"],
                    "car_id": t["car_id"],
                    "class_name": t["class_name"],
                    "bbox": t["bbox"],
                }
                for t in pipeline.latest_tracks
            ],
        }

    print(f"\n[INFO] Flask 웹 서버를 시작합니다. http://젯슨IP:{CONFIG['WEB_PORT']}/ 으로 접속하세요.")
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
        cap.release()
        print("[INFO] 카메라를 해제하고 종료합니다.")
