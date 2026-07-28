import cv2
import time
import sys
import os
import random
import threading
from collections import deque
from ultralytics import YOLO
from flask import Flask, Response

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B00_camera_input import get_camera
from logic.B01_car_detection import VEHICLE_CLASS_IDS, VEHICLE_CLASS_NAMES

# ---------------------------------------------------------
# 설정 (Configuration)
# ---------------------------------------------------------
CONFIG = {
    # 모델 추론 설정
    "MODEL_PATH": "yolov8s.engine", # 엔진 모델 경로 ('yolov8s.pt'로 변경 시 일반 파이토치 모델 사용)
    "TRACKER_CFG": "bytetrack.yaml", # ByteTrack 설정 (ultralytics 내장)
    "CONF_THRESH": 0.5,             # Confidence 임계값
    "IOU_THRESH": 0.45,             # NMS IoU 임계값
    "IMGSZ": 640,                  # YOLO 추론 해상도

    # 카메라 설정
    "CAM_WIDTH": 1280,              # 카메라 가로 해상도
    "CAM_HEIGHT": 720,              # 카메라 세로 해상도
    "CAM_FPS": 30,                  # 카메라 프레임레이트

    # 추적 <-> 차량번호 매칭 설정
    "MIN_HITS_FOR_ASSIGN": 3,       # 이 프레임 수 이상 연속 추적되어야 차량번호를 부여 (오검출 방지)
    "LOST_TTL_FRAMES": 90,          # 추적이 끊긴 뒤 매칭 정보를 유지할 프레임 수 (30fps 기준 약 3초)
    "TRAJECTORY_MAXLEN": 64,        # 궤적(Trajectory) 저장 최대 길이

    # UART 미연결 상태에서 FIFO 매칭 로직을 테스트하기 위한 설정
    "TEST_PRESET_CAR_NUMBERS": ["1234", "1998", "0828"],  # 여기에 원하는 차량번호를 순서대로 적으면 실행 시작 시 FIFO에 그대로 등록됨. 예: ["1234", "5678", "9012"]
    "TEST_UART_SIMULATOR": False,   # True로 설정 시 임의의 차량번호를 주기적으로 FIFO에 추가 생성
    "TEST_UART_INTERVAL_SEC": 5.0,  # 가짜 차량번호 생성 주기(초)
}

# 차량번호가 부여된 트랙 / 대기 중인 트랙 색상 (BGR)
COLOR_MATCHED = (0, 255, 0)     # 번호 매칭 완료 - 초록
COLOR_PENDING = (0, 165, 255)   # 번호 대기 중   - 주황


# =====================================================================
# 차량번호 FIFO 큐
# =====================================================================
class CarNumberFIFO:
    """
    UART로 수신된 차량 번호를 들어온 순서대로 관리하는 FIFO 큐.

    입구에서 차량이 검출되면 Zybo가 UART로 차량 번호를 송신하고,
    Jetson은 그 번호를 이 큐에 순서대로 적재한다.

    이후 카메라에 새로운 차량이 검출(추적 시작)되면
    큐의 가장 앞(가장 먼저 들어온) 번호를 꺼내어 해당 트랙에 부여한다.

    UART 수신 스레드와 영상 처리 스레드가 동시에 접근하므로
    모든 연산은 Lock으로 보호한다.
    """

    def __init__(self):
        self._queue = deque()
        self._lock = threading.Lock()

    def push(self, car_id):
        """
        차량 번호를 큐의 뒤에 추가. (UART 수신 시 호출)

        Args:
            car_id: 차량 번호 4자리 문자열 (예: "1234")
        """
        with self._lock:
            self._queue.append(car_id)
            print(f"[FIFO] 차량번호 '{car_id}' 등록 (대기 {len(self._queue)}대)")

    def pop(self):
        """
        큐의 가장 앞 차량 번호를 꺼내고 큐에서 삭제.

        Returns:
            차량 번호 문자열. 큐가 비어있으면 None.
        """
        with self._lock:
            if not self._queue:
                return None
            car_id = self._queue.popleft()
            print(f"[FIFO] 차량번호 '{car_id}' 출고 (잔여 {len(self._queue)}대)")
            return car_id

    def peek(self):
        """큐의 가장 앞 차량 번호를 삭제하지 않고 조회. 비어있으면 None."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def push_front(self, car_id):
        """
        차량 번호를 큐의 맨 앞으로 되돌림.
        (부여했던 트랙이 유실되어 번호를 회수할 때 사용)
        """
        with self._lock:
            self._queue.appendleft(car_id)
            print(f"[FIFO] 차량번호 '{car_id}' 반환 (대기 {len(self._queue)}대)")

    def size(self):
        """큐에 대기 중인 차량 번호 개수."""
        with self._lock:
            return len(self._queue)

    def snapshot(self):
        """현재 대기 중인 차량 번호 목록을 리스트로 반환. (모니터링용)"""
        with self._lock:
            return list(self._queue)

    def clear(self):
        """큐를 비움."""
        with self._lock:
            self._queue.clear()


# ByteTrack 기반 차량 추적 + 차량번호 매칭
class CarMOT:
    """
    ByteTrack을 이용한 다중 객체 추적(MOT) 및 차량번호 매칭 클래스.

    동작 흐름:
      1. UART로 수신된 차량 번호가 CarNumberFIFO에 순서대로 쌓인다.
      2. YOLO + ByteTrack이 프레임마다 차량에 Track ID를 부여한다.
      3. 새로운 Track ID가 MIN_HITS_FOR_ASSIGN 프레임 이상 안정적으로
         추적되면, FIFO에서 가장 앞 번호를 꺼내(pop) 해당 트랙에 매칭한다.
      4. 이후 그 Track ID는 계속 같은 차량 번호로 추적된다.

    참고: 추적이 완전히 끊긴 뒤 같은 차량이 새로운 Track ID로 다시
    잡히면 FIFO의 다음 번호를 소비하게 된다. ByteTrack의 track_buffer와
    LOST_TTL_FRAMES가 짧은 가려짐(Occlusion) 구간을 보완한다.
    """

    def __init__(self, model_path='yolov8s.engine', tracker_cfg='bytetrack.yaml',
                 conf=0.5, iou=0.45, imgsz=1280,
                 min_hits=3, lost_ttl=90, trajectory_maxlen=64, fifo=None):
        """
        CarMOT 초기화.

        Args:
            model_path:        YOLOv8 모델 파일 경로
            tracker_cfg:       ByteTrack 설정 파일 (ultralytics 내장 'bytetrack.yaml')
            conf:              Confidence 임계값
            iou:               NMS IoU 임계값
            imgsz:             추론 해상도 (입력 크기)
            min_hits:          차량번호 부여에 필요한 최소 연속 추적 프레임 수
            lost_ttl:          추적 소실 후 매칭 정보를 유지할 프레임 수
            trajectory_maxlen: 트랙별 궤적 저장 최대 길이
            fifo:              사용할 CarNumberFIFO 인스턴스 (None이면 모듈 전역 큐 사용)
        """
        print(f"[INFO] YOLOv8 + ByteTrack 모델을 로드합니다... ({model_path})")
        self.model = YOLO(model_path, task='detect')
        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.min_hits = min_hits
        self.lost_ttl = lost_ttl
        self.trajectory_maxlen = trajectory_maxlen

        self.fifo = fifo if fifo is not None else car_number_fifo

        # Track ID -> 차량 번호 매핑 (예: {5: "1234"})
        self.track_to_car = {}
        # Track ID -> 연속 검출 프레임 수 (오검출로 FIFO가 소모되는 것 방지)
        self.hit_counts = {}
        # Track ID -> 추적이 끊긴 뒤 경과한 프레임 수
        self.lost_counts = {}
        # Track ID -> 이동 궤적 deque([(cx, cy, timestamp), ...])
        self.trajectories = {}

        print("[INFO] 모델 로드 완료. (Tracker: ByteTrack)")

    def update(self, frame):
        """
        프레임에서 차량을 추적하고, 신규 트랙에 FIFO의 차량 번호를 매칭.

        Args:
            frame: OpenCV BGR 이미지 (numpy array)

        Returns:
            추적 결과 리스트. 각 항목은 딕셔너리:
            [
                {
                    "track_id": 5,
                    "car_id": "1234",          # 아직 매칭 전이면 None
                    "class_id": 2,
                    "class_name": "Car",
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),        # 하단 중앙 기준점
                    "confidence": 0.94
                },
                ...
            ]
        """
        # ByteTrack 추적 실행 (persist=True로 프레임 간 Track ID 유지)
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=list(VEHICLE_CLASS_IDS),
            verbose=False
        )

        tracks = []
        alive_ids = set()
        now = time.time()

        for result in results:
            boxes = result.boxes

            # 추적된 객체가 없으면 boxes.id가 None
            if boxes is None or boxes.id is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])

                # 차량 클래스만 필터링
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                track_id = int(box.id[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                alive_ids.add(track_id)
                self.hit_counts[track_id] = self.hit_counts.get(track_id, 0) + 1
                self.lost_counts.pop(track_id, None)

                # 신규 트랙이 안정적으로 잡히면 FIFO에서 차량 번호를 꺼내 매칭
                if track_id not in self.track_to_car and self.hit_counts[track_id] >= self.min_hits:
                    self._assign_car_id(track_id)

                # 하단 중앙(Bottom-Center) 기준점 - 노면 위치에 가장 가까운 좌표
                cx, cy = (x1 + x2) // 2, y2
                traj = self.trajectories.setdefault(
                    track_id, deque(maxlen=self.trajectory_maxlen)
                )
                traj.append((cx, cy, now))

                tracks.append({
                    "track_id": track_id,
                    "car_id": self.track_to_car.get(track_id),
                    "class_id": cls_id,
                    "class_name": VEHICLE_CLASS_NAMES.get(cls_id, "Vehicle"),
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "confidence": conf
                })

        # 이번 프레임에 잡히지 않은 트랙 정리
        self._cleanup_lost(alive_ids)

        return tracks

    def _assign_car_id(self, track_id):
        """
        FIFO에서 가장 먼저 들어온 차량 번호를 꺼내 트랙에 부여.
        큐가 비어있으면(UART 수신이 아직 늦은 경우) 부여하지 않고
        다음 프레임에 다시 시도한다.
        """
        car_id = self.fifo.pop()
        if car_id is None:
            return False

        self.track_to_car[track_id] = car_id
        print(f"[매칭] Track ID {track_id} <- 차량번호 '{car_id}'")
        return True

    def _cleanup_lost(self, alive_ids):
        """
        이번 프레임에 검출되지 않은 트랙의 소실 카운트를 증가시키고,
        LOST_TTL_FRAMES를 초과하면 매칭 정보와 궤적을 제거한다.
        """
        for track_id in list(self.hit_counts.keys()):
            if track_id in alive_ids:
                continue

            self.lost_counts[track_id] = self.lost_counts.get(track_id, 0) + 1
            if self.lost_counts[track_id] < self.lost_ttl:
                continue

            car_id = self.track_to_car.pop(track_id, None)
            self.hit_counts.pop(track_id, None)
            self.lost_counts.pop(track_id, None)
            self.trajectories.pop(track_id, None)

            if car_id:
                print(f"[소실] Track ID {track_id} (차량번호 '{car_id}') 추적 종료")

    def get_car_id(self, track_id):
        """Track ID에 매칭된 차량 번호를 반환. 없으면 None."""
        return self.track_to_car.get(track_id)

    def get_track_id(self, car_id):
        """차량 번호에 매칭된 Track ID를 반환. 없으면 None."""
        for t_id, c_id in self.track_to_car.items():
            if c_id == car_id:
                return t_id
        return None

    def get_trajectory(self, track_id):
        """Track ID의 이동 궤적 리스트 [(cx, cy, timestamp), ...]를 반환."""
        return list(self.trajectories.get(track_id, []))

    def draw_tracks(self, frame, tracks, draw_trajectory=True):
        """
        추적 결과를 프레임에 Bounding Box, Track ID, 차량번호로 시각화.

        Args:
            frame:           OpenCV BGR 이미지 (numpy array, 원본이 수정됨)
            tracks:          update() 메서드의 반환 결과 리스트
            draw_trajectory: 이동 궤적 선 표시 여부

        Returns:
            시각화가 적용된 프레임 (입력 frame과 동일 객체)
        """
        for trk in tracks:
            track_id = trk["track_id"]
            car_id = trk["car_id"]
            x1, y1, x2, y2 = trk["bbox"]

            # 번호 매칭 여부에 따라 색상 구분
            color = COLOR_MATCHED if car_id else COLOR_PENDING
            label = f"ID:{track_id} {car_id}" if car_id else f"ID:{track_id} WAIT"

            # Bounding Box 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 라벨 배경 및 텍스트
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - baseline - 4),
                (x1 + label_w, y1),
                color, -1
            )
            cv2.putText(
                frame, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

            # 이동 궤적 표시
            if draw_trajectory:
                points = self.trajectories.get(track_id)
                if points and len(points) >= 2:
                    for i in range(1, len(points)):
                        p1 = (points[i - 1][0], points[i - 1][1])
                        p2 = (points[i][0], points[i][1])
                        cv2.line(frame, p1, p2, color, 2)

        return frame


# 모듈 전역 FIFO (UART 수신 모듈에서 바로 사용)
car_number_fifo = CarNumberFIFO()


def enqueue_car_number(car_id):
    """
    UART로 수신한 차량 번호를 전역 FIFO에 등록.

    A00_uart_rx.py의 입차 수신 처리에서 호출:
        from logic.B03_car_mot import enqueue_car_number
        enqueue_car_number(car_id)
    """
    car_number_fifo.push(car_id)


def simulate_uart_rx(interval_sec=5.0, stop_event=None):
    """
    실제 UART(A00_uart_rx.py) 없이 FIFO 매칭 로직을 테스트하기 위한 가짜 송신기.
    Zybo가 보내는 것과 동일하게 4자리 차량번호를 주기적으로 생성해 FIFO에 등록한다.
    별도 스레드에서 실행해야 카메라 스트리밍을 막지 않는다.

    Args:
        interval_sec: 차량번호 생성 주기(초)
        stop_event:   threading.Event. set()되면 루프를 종료
    """
    print(f"[TEST] 가짜 UART 송신 시작 ({interval_sec}초 간격으로 임의 차량번호 생성)")
    while stop_event is None or not stop_event.is_set():
        car_id = f"{random.randint(0, 9999):04d}"
        enqueue_car_number(car_id)
        time.sleep(interval_sec)


# 테스트용 메인 (단독 실행 시 웹 스트리밍 서버 열기)
if __name__ == '__main__':
    print("==========================================")
    print(" B03 : 차량 다중 객체 추적 (ByteTrack MOT)")
    print(" 모델 : YOLOv8s + ByteTrack")
    print("==========================================")

    # 카메라 열기 (B00_camera_input 모듈 활용)
    print(f"[INFO] B00_camera_input 모듈을 통해 카메라를 엽니다... ({CONFIG['CAM_WIDTH']}x{CONFIG['CAM_HEIGHT']})")
    cap = get_camera(sensor_id=0, width=CONFIG['CAM_WIDTH'], height=CONFIG['CAM_HEIGHT'], framerate=CONFIG['CAM_FPS'])

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        sys.exit(1)

    # 추적기 초기화
    mot = CarMOT(
        model_path=CONFIG['MODEL_PATH'],
        tracker_cfg=CONFIG['TRACKER_CFG'],
        conf=CONFIG['CONF_THRESH'],
        iou=CONFIG['IOU_THRESH'],
        imgsz=CONFIG['IMGSZ'],
        min_hits=CONFIG['MIN_HITS_FOR_ASSIGN'],
        lost_ttl=CONFIG['LOST_TTL_FRAMES'],
        trajectory_maxlen=CONFIG['TRAJECTORY_MAXLEN']
    )

    # 미리 지정해 둔 차량번호를 순서대로 FIFO에 등록 (CONFIG['TEST_PRESET_CAR_NUMBERS'])
    for car_id in CONFIG['TEST_PRESET_CAR_NUMBERS']:
        enqueue_car_number(car_id)

    # UART 미연결 환경을 위한 가짜 차량번호 송신기 (설정 시에만 동작)
    uart_sim_stop = threading.Event()
    if CONFIG['TEST_UART_SIMULATOR']:
        threading.Thread(
            target=simulate_uart_rx,
            args=(CONFIG['TEST_UART_INTERVAL_SEC'], uart_sim_stop),
            daemon=True
        ).start()

    app = Flask(__name__)

    def generate_frames_with_tracking():
        prev_time = time.time()
        frame_count = 0
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 추적 실행 및 시각화
            tracks = mot.update(frame)
            mot.draw_tracks(frame, tracks)

            # FPS 계산 (0.5초 간격)
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= 0.5:
                fps = frame_count / (current_time - prev_time)
                prev_time = current_time
                frame_count = 0

            # 화면 텍스트 표시
            matched = sum(1 for t in tracks if t["car_id"])
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Tracks: {len(tracks)} (Matched: {matched})", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FIFO Waiting: {mot.fifo.size()}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # 터미널에 추적 결과 출력 (FPS 갱신 시점마다)
            if frame_count == 1 and tracks:
                print(f"[FPS: {fps:.1f}] 추적 차량 {len(tracks)}대:")
                for t in tracks:
                    car_str = t["car_id"] if t["car_id"] else "미매칭"
                    print(f"  ID:{t['track_id']:<3d} {t['class_name']:12s} 번호={car_str:8s} conf={t['confidence']:.2f}")

            # JPEG 압축 후 웹 스트리밍 반환
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    @app.route('/')
    def index():
        return f"""
        <html>
            <head><title>Jetson Car Tracking (ByteTrack)</title></head>
            <body style="background-color: #222; color: white; text-align: center;">
                <h2>Jetson Orin Nano - Car MOT (YOLOv8s + ByteTrack)</h2>
                <img src="/video_feed" width="{CONFIG['CAM_WIDTH']}" height="{CONFIG['CAM_HEIGHT']}">
                <p>UART 없이 테스트: <code>/enqueue/1234</code> 로 차량번호를 직접 FIFO에 넣거나,
                CONFIG['TEST_UART_SIMULATOR']=True로 설정해 자동 생성기를 사용하세요.</p>
            </body>
        </html>
        """

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames_with_tracking(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/enqueue/<car_id>')
    def enqueue(car_id):
        """UART 없이 차량번호 매칭을 테스트하기 위한 수동 등록 라우트."""
        enqueue_car_number(car_id)
        return f"FIFO 등록: {car_id} (대기 {car_number_fifo.size()}대) / 현재 큐: {car_number_fifo.snapshot()}"

    print("\n[INFO] Flask 웹 서버를 시작합니다. http://젯슨IP:5000/ 으로 접속하세요.")
    app.run(host='0.0.0.0', port=5000, debug=False)
