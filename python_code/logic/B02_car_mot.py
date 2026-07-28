import cv2
import time
import sys
import os
import random
import threading
import numpy as np
from collections import deque
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B01_car_detection import VEHICLE_CLASS_NAMES

# ---------------------------------------------------------
# 설정 (Configuration)
# ---------------------------------------------------------
# 이 모듈은 '추적'만 담당한다. 검출 관련 설정(모델 경로, conf, imgsz 등)은
# B01_car_detection.py의 CONFIG에서 관리한다.
CONFIG = {
    # ByteTrack 설정
    "TRACKER_CFG": "bytetrack.yaml", # ByteTrack 설정 파일 (ultralytics 내장)

    # 추적 <-> 차량번호 매칭 설정
    "MIN_HITS_FOR_ASSIGN": 3,       # 이 프레임 수 이상 연속 추적되어야 차량번호를 부여 (오검출 방지)
    "LOST_TTL_FRAMES": 90,          # 추적이 끊긴 뒤 매칭 정보를 유지할 프레임 수 (30fps 기준 약 3초)
    "TRAJECTORY_MAXLEN": 64,        # 궤적(Trajectory) 저장 최대 길이
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


# =====================================================================
# 검출 결과 -> ByteTrack 입력 어댑터
# =====================================================================
class DetectionResults:
    """
    B01_car_detection의 검출 결과(list of dict)를 ByteTrack이 요구하는
    형식으로 변환하는 어댑터.

    BYTETracker.update()는 아래 인터페이스를 요구한다.
      - xywh : (N, 4) 중심좌표 기반 박스 배열
      - conf : (N,) Confidence 배열
      - cls  : (N,) 클래스 ID 배열
      - len() 및 boolean 마스크 인덱싱 지원
    """

    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_detections(cls, detections):
        """
        B01_car_detection.CarDetector.detect()의 반환 결과를 변환.

        Args:
            detections: [{"class_id":.., "bbox":[x1,y1,x2,y2], "confidence":..}, ...]
        """
        if not detections:
            return cls(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        xywh = np.array(
            [
                [(d["bbox"][0] + d["bbox"][2]) / 2,   # 중심 x
                 (d["bbox"][1] + d["bbox"][3]) / 2,   # 중심 y
                 d["bbox"][2] - d["bbox"][0],         # 폭
                 d["bbox"][3] - d["bbox"][1]]         # 높이
                for d in detections
            ],
            dtype=np.float32
        )
        conf = np.array([d["confidence"] for d in detections], dtype=np.float32)
        cls_arr = np.array([d["class_id"] for d in detections], dtype=np.float32)
        return cls(xywh, conf, cls_arr)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        """boolean 마스크로 고신뢰/저신뢰 검출을 분리할 때 사용."""
        return DetectionResults(self.xywh[mask], self.conf[mask], self.cls[mask])


# =====================================================================
# ByteTrack 기반 차량 추적 + 차량번호 매칭
# =====================================================================
class CarMOT:
    """
    ByteTrack을 이용한 다중 객체 추적(MOT) 및 차량번호 매칭 클래스.

    이 클래스는 '추적'만 담당하며 YOLO 모델을 직접 들고 있지 않는다.
    검출은 B01_car_detection.CarDetector가 수행하고, 그 결과를
    update()의 인자로 전달받는다. (검출/추적 책임 분리)

    동작 흐름:
      1. UART로 수신된 차량 번호가 CarNumberFIFO에 순서대로 쌓인다.
      2. B01이 검출한 결과를 ByteTrack에 넣어 Track ID를 부여한다.
      3. 새로운 Track ID가 MIN_HITS_FOR_ASSIGN 프레임 이상 안정적으로
         추적되면, FIFO에서 가장 앞 번호를 꺼내(pop) 해당 트랙에 매칭한다.
      4. 이후 그 Track ID는 계속 같은 차량 번호로 추적된다.

    참고: 추적이 완전히 끊긴 뒤 같은 차량이 새로운 Track ID로 다시
    잡히면 FIFO의 다음 번호를 소비하게 된다. ByteTrack의 track_buffer와
    LOST_TTL_FRAMES가 짧은 가려짐(Occlusion) 구간을 보완한다.
    """

    def __init__(self, tracker_cfg='bytetrack.yaml',
                 min_hits=3, lost_ttl=90, trajectory_maxlen=64, fifo=None):
        """
        CarMOT 초기화.

        Args:
            tracker_cfg:       ByteTrack 설정 파일 (ultralytics 내장 'bytetrack.yaml')
            min_hits:          차량번호 부여에 필요한 최소 연속 추적 프레임 수
            lost_ttl:          추적 소실 후 매칭 정보를 유지할 프레임 수
            trajectory_maxlen: 트랙별 궤적 저장 최대 길이
            fifo:              사용할 CarNumberFIFO 인스턴스 (None이면 모듈 전역 큐 사용)
        """
        print(f"[INFO] ByteTrack 추적기를 초기화합니다... ({tracker_cfg})")
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_cfg)))
        self.tracker = BYTETracker(cfg)

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

        print("[INFO] 추적기 초기화 완료. (Tracker: ByteTrack)")

    def update(self, detections):
        """
        검출 결과를 추적하고, 신규 트랙에 FIFO의 차량 번호를 매칭.

        Args:
            detections: B01_car_detection.CarDetector.detect()의 반환 결과 리스트
                        [{"class_id":.., "class_name":.., "bbox":[..], "confidence":..}, ...]

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
        # 검출 결과를 ByteTrack 입력 형식으로 변환 후 연관(association) 수행
        results = DetectionResults.from_detections(detections)
        outputs = self.tracker.update(results)

        tracks = []
        alive_ids = set()
        now = time.time()

        # outputs 각 행: [x1, y1, x2, y2, track_id, score, cls, det_idx]
        for row in outputs:
            x1, y1, x2, y2 = map(int, row[:4])
            track_id = int(row[4])
            conf = float(row[5])
            cls_id = int(row[6])

            alive_ids.add(track_id)
            self.hit_counts[track_id] = self.hit_counts.get(track_id, 0) + 1
            self.lost_counts.pop(track_id, None)

            # 신규 트랙이 안정적으로 잡히면 FIFO에서 차량 번호를 꺼내 매칭
            if track_id not in self.track_to_car and self.hit_counts[track_id] >= self.min_hits:
                self._assign_car_id(track_id)

            # 박스 중앙(Center) 기준점
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
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
        from logic.B02_car_mot import enqueue_car_number
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


# =====================================================================
# 테스트용 메인 (단독 실행 시 합성 검출 데이터로 추적/매칭 로직 검증)
# =====================================================================
# 이 모듈은 추적 전용이므로 카메라와 YOLO 모델 없이도 단독 검증이 가능하다.
# 카메라 + 검출 + 추적 통합 실행은 B_main.py를 사용할 것.
if __name__ == '__main__':
    print("==========================================")
    print(" B02 : 차량 다중 객체 추적 (ByteTrack MOT)")
    print(" 단독 테스트 : 합성 검출 데이터로 로직 검증")
    print("==========================================")

    # 테스트용 차량번호를 FIFO에 미리 등록 (원하는 번호로 수정 가능)
    TEST_CAR_NUMBERS = ["1234", "5678", "9012"]
    for num in TEST_CAR_NUMBERS:
        enqueue_car_number(num)

    mot = CarMOT(
        tracker_cfg=CONFIG['TRACKER_CFG'],
        min_hits=CONFIG['MIN_HITS_FOR_ASSIGN'],
        lost_ttl=CONFIG['LOST_TTL_FRAMES'],
        trajectory_maxlen=CONFIG['TRAJECTORY_MAXLEN']
    )

    # 차량 3대가 순차적으로 등장하여 오른쪽으로 이동하는 시나리오
    # (등장 프레임, 시작 x좌표, y좌표)
    scenario = [(0, 50, 200), (12, 50, 320), (24, 50, 440)]
    TOTAL_FRAMES = 40

    print(f"\n[TEST] 합성 시나리오 시작: 차량 {len(scenario)}대가 순차 등장\n")

    for frame_idx in range(TOTAL_FRAMES):
        # 이번 프레임의 합성 검출 결과 생성 (B01의 detect() 반환 형식과 동일)
        detections = []
        for appear_at, start_x, y in scenario:
            if frame_idx < appear_at:
                continue
            x = start_x + (frame_idx - appear_at) * 12  # 프레임당 12px 이동
            detections.append({
                "class_id": 2,
                "class_name": "Car",
                "bbox": [x, y, x + 100, y + 80],
                "confidence": 0.9
            })

        tracks = mot.update(detections)

        # 매칭 상태가 바뀌는 시점만 출력
        if frame_idx in (2, 3, 14, 15, 26, 27, TOTAL_FRAMES - 1):
            print(f"--- frame {frame_idx:2d} | 추적 {len(tracks)}대 | FIFO 잔여 {mot.fifo.size()}대")
            for t in tracks:
                car_str = t["car_id"] if t["car_id"] else "미매칭"
                print(f"      ID:{t['track_id']:<3d} 번호={car_str:8s} bbox={t['bbox']}")

    print(f"\n[TEST] 최종 Track ID -> 차량번호 매핑: {mot.track_to_car}")
    print("[TEST] 검출된 순서대로 FIFO 번호가 부여되었는지 확인하세요.")
