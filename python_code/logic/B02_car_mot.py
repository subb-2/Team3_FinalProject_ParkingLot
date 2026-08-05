import cv2
import time
import sys
import os
import math
import threading
import numpy as np
from collections import deque
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.oc_sort import OCSORT
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.B01_car_detection import VEHICLE_CLASS_NAMES

# 설정 (Configuration)
# 이 모듈은 '추적'만 담당. 검출 관련 설정(모델 경로, conf, imgsz 등)은 B01_car_detection.py의 CONFIG에서 관리.

# 프로젝트 전용 추적기 설정 파일 경로.
# ultralytics 내장 파일을 쓰면 패키지 재설치 시 설정이 날아가므로 python_code/config/ 아래에 복사해 두고 그것을 사용함.
#
# 어떤 추적기를 쓸지는 이 yaml 안의 tracker_type이 결정한다. (아래 TRACKERS 참고)
# 파일을 바꿔 끼우는 것만으로 추적기가 교체되므로 성능 비교(A/B)가 쉽다.
#   ocsort.yaml    : OC-SORT + ByteTrack 저신뢰 2차 연관 (use_byte: True)  <- 현재 사용
#   bytetrack.yaml : ByteTrack 단독 (비교용으로 남겨둠)
TRACKER_CFG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'config', 'ocsort.yaml')
)

# tracker_type -> 추적기 클래스.
#
# OCSORT는 BYTETracker의 서브클래스이고 update()의 입출력 형식이 완전히 동일하다.
# 따라서 이 모듈의 나머지 코드(DetectionResults 어댑터, outputs 파싱)는
# 어느 쪽을 쓰든 손댈 필요가 없다.
TRACKERS = {
    "bytetrack": BYTETracker,
    "ocsort": OCSORT,
}

CONFIG = {
    # 추적기 설정 (세부 파라미터는 위 경로의 yaml 참고)
    "TRACKER_CFG": TRACKER_CFG_PATH,

    # 추적 <-> 차량번호 매칭 설정
    "MIN_HITS_FOR_ASSIGN": 30,      # 이 프레임 수 이상 '연속' 추적되어야 차량번호를 부여 (30fps 기준 1초)
    "TRAJECTORY_MAXLEN": 64,        # 궤적(Trajectory) 저장 최대 길이

    # ---------------------------------------------------------------------
    # 재바인딩(Re-binding) 레이어
    #
    # 추적기가 아무리 좋아도 ID 스위치를 0으로 만들 수는 없다. (config/ocsort.yaml의
    # 실측표 참고: 가려진 동안 차가 방향을 바꾸면 OC-SORT도 ID를 놓친다)
    #
    # 그래서 발상을 바꾼다. ID 스위치를 막는 대신 '스위치가 나도 상관없게' 만든다.
    # 하위 모듈(A01 주차관리, C00 내비, D00 UI)이 실제로 쓰는 것은 track_id가 아니라
    # car_id다. 그러니 track_id가 바뀌어도 car_id만 같은 차를 따라가면 된다.
    #
    # 동작: 번호를 가진 트랙이 사라지면 마지막 상태를 '유령(ghost)'으로 남겨두고,
    #       새 track_id가 뜨면 FIFO에서 번호를 꺼내기 '전에' 유령과 먼저 대조한다.
    #       조건을 만족하면 FIFO를 소비하지 않고 옛 번호를 그대로 승계한다.
    #
    # 판정 조건 (전부 통과해야 승계):
    #   1) 시간   : 사라진 지 MAX_GAP_SEC 이내
    #   2) 거리   : 그 시간 안에 물리적으로 도달 가능한 위치인가
    #   3) 외형   : 색 히스토그램이 비슷한가 (프레임이 주어졌을 때만)
    # ---------------------------------------------------------------------
    "REBIND": {
        "ENABLE": True,

        # 재바인딩 시도에 필요한 최소 연속 추적 프레임.
        # MIN_HITS_FOR_ASSIGN(30)보다 훨씬 짧게 둔다. 재바인딩은 거리/외형으로
        # 이미 검증된 승계라 FIFO를 새로 소비하는 것보다 안전하고, 되돌아온 차가
        # 1초씩 'WAIT'로 떠 있으면 내비게이션이 그만큼 끊기기 때문이다.
        "MIN_HITS": 3,

        # 유령을 유지할 시간(초). 이보다 오래 안 보이면 다른 차로 간주한다.
        # 추적기의 track_buffer(60프레임 = 30fps에서 2초)보다 길게 두어야
        # 추적기가 포기한 뒤의 구간을 이 레이어가 이어받는다.
        "MAX_GAP_SEC": 3.0,

        # 도달 가능 거리 판정. 실좌표(cm)를 쓸 수 있으면 그쪽을 쓴다.
        # (C_main처럼 C00의 호모그래피가 준비된 경우. B_main 단독이면 픽셀 기준)
        "MAX_SPEED_CM_S": 30.0,     # RC카 최고 속도 가정
        "MARGIN_CM": 10.0,          # 위치 추정 오차 여유
        "MAX_SPEED_PX_S": 250.0,    # 실좌표가 없을 때 픽셀 기준 속도
        "MARGIN_PX": 60.0,

        # 도달 반경 상한. 시간에 비례해 반경을 넓히기만 하면, 오래 안 보인
        # 차일수록 판정이 헐거워져 결국 아무 차나 붙는다.
        # 현재 주차장 목업이 40 x 140cm이므로 3초 gap의 100cm는 사실상
        # 무제한이다. 그래서 gap과 무관한 상한을 따로 둔다.
        # 주차장을 키우면 C02의 CELL_W/H_CM과 함께 이 값도 조정할 것.
        "MAX_REACH_CM": 60.0,
        "MAX_REACH_PX": 400.0,

        # 색 히스토그램(외형) 판정
        "HIST_MAX_DIST": 0.45,      # Bhattacharyya 거리 상한 (작을수록 엄격)
        "HIST_MIN_CONF": 0.6,       # 이 conf 이상일 때만 히스토그램을 갱신
                                    # (반쯤 가려진 프레임으로 갱신하면 손 색이 섞인다)
        "HIST_EMA": 0.9,            # 기존 히스토그램에 주는 가중치
    },
}

# 트랙이 같은 ID로 되살아날 수 있는 기간은 전적으로 추적기가 결정(track_buffer),

# 차량번호가 부여된 트랙 / 대기 중인 트랙 색상 (BGR)
COLOR_MATCHED = (0, 255, 0)     # 번호 매칭 완료 - 초록
COLOR_PENDING = (0, 165, 255)   # 번호 대기 중   - 주황
COLOR_REBOUND = (255, 0, 255)   # 재바인딩 직후  - 자홍 (승계가 실제로 도는지 눈으로 확인용)

# 재바인딩 직후 자홍색으로 강조할 시간(초).
REBOUND_HIGHLIGHT_SEC = 2.0


def color_histogram(frame, bbox):
    """
    박스 영역의 HSV 색 히스토그램을 계산. (경량 외형 특징)

    딥러닝 ReID 모델 대신 색 히스토그램을 쓰는 이유:
      - 젯슨에서 별도 신경망을 더 돌릴 여유가 없다. 이 함수는 박스당 0.1ms 수준.
      - 탑뷰의 작은 RC카는 형태 정보가 거의 없어 ReID 임베딩의 변별력이 떨어진다.
        반대로 '차 색깔'은 이 목업 환경에서 가장 확실한 구분 단서다.

    Args:
        frame: OpenCV BGR 이미지. 시각화가 그려지기 '전'의 원본이어야 한다.
        bbox:  [x1, y1, x2, y2]

    Returns:
        (256,) float32 히스토그램. 계산할 수 없으면 None.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    # 박스 가장자리는 배경(바닥)이 섞이므로 중앙 60%만 사용한다.
    mx, my = int(w * 0.2), int(h * 0.2)
    fh, fw = frame.shape[:2]
    xa, xb = max(0, x1 + mx), min(fw, x2 - mx)
    ya, yb = max(0, y1 + my), min(fh, y2 - my)
    if xb <= xa or yb <= ya:
        return None

    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    # 색상(H) x 채도(S) 2D 히스토그램. 명도(V)는 조명 변화에 약해 제외한다.
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


# 차량번호 FIFO 큐
class CarNumberFIFO:
    """
    UART로 수신된 차량 번호를 들어온 순서대로 관리하는 FIFO 큐.

    입구에서 차량이 검출되면 Zybo가 UART로 차량 번호를 송신하고,
    Jetson은 그 번호를 이 큐에 순서대로 적재.

    이후 카메라에 새로운 차량이 검출(추적 시작)되면
    큐의 가장 앞(가장 먼저 들어온) 번호를 꺼내어 해당 트랙에 부여.

    UART 수신 스레드와 영상 처리 스레드가 동시에 접근하므로
    모든 연산은 Lock으로 보호.
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

# 검출 결과 -> 추적기 입력 어댑터
class DetectionResults:
    """
    B01_car_detection의 검출 결과(list of dict)를 추적기가 요구하는 형식으로 변환하는 어댑터.

    BYTETracker.update() / OCSORT.update()는 아래 인터페이스를 요구한다.
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


# 다중 객체 추적 + 차량번호 매칭
class CarMOT:
    """
    다중 객체 추적(MOT) 및 차량번호 매칭 클래스.

    이 클래스는 '추적'만 담당하며 YOLO 모델을 직접 들고 있지 않음.
    검출은 B01_car_detection.CarDetector가 수행하고,
    그 결과를 update()의 인자로 전달받는다. (검출/추적 책임 분리)

    추적기는 설정 파일의 tracker_type이 결정한다. (모듈 상단 TRACKERS 참고)
    현재 기본값은 OC-SORT이며 use_byte: True로 ByteTrack의 저신뢰 2차 연관도
    함께 사용한다. 두 알고리즘은 서로 다른 실패를 보완하므로 배타적이지 않다.
      - ByteTrack : 가림으로 conf가 '떨어진' 박스를 2차로 재매칭
      - OC-SORT   : 가림으로 '사라졌다 돌아온' 트랙을 마지막 관측 위치로 복구(OCR)
                    + 복귀 시 칼만 상태 재계산(ORU)

    동작 흐름:
      1. UART로 수신된 차량 번호가 CarNumberFIFO에 순서대로 쌓인다.
      2. B01이 검출한 결과를 추적기에 넣어 Track ID를 부여한다.
      3. 새로운 Track ID가 MIN_HITS_FOR_ASSIGN 프레임 이상 '연속으로'
         추적되면, FIFO에서 가장 앞 번호를 꺼내(pop) 해당 트랙에 매칭한다.
         중간에 한 프레임이라도 끊기면 카운트는 0으로 초기화된다.
      4. 이후 그 Track ID는 계속 같은 차량 번호로 추적된다.

    3번에서 FIFO를 소비하기 전에 '재바인딩'을 먼저 시도한다. 추적이 끊겨 새 Track
    ID로 다시 잡힌 것뿐이라면, 사라진 트랙의 유령(마지막 위치/외형)과 대조해
    옛 차량번호를 그대로 승계한다. 덕분에 추적기가 ID를 바꿔도 car_id는 물리적인
    차를 계속 따라간다. 설정은 CONFIG['REBIND'] 참고.

    C00의 world_history / routes / targets가 전부 car_id로 키를 잡고 있으므로,
    car_id만 유지되면 ID 스위치가 나도 경로 안내 상태가 그대로 살아남는다.

    한계: 두 트랙이 '살아있는 채로' 서로 ID를 맞바꾸는 경우(교차 중 스왑)는
          유령이 생기지 않으므로 이 레이어가 잡지 못한다. 이 환경에서는 차가
          서로 겹쳐 지나갈 일이 거의 없어 실용상 문제되지 않는다.

    추적기 성능 비교는 track_id_stats()로 측정할 것.
    """

    def __init__(self, tracker_cfg=TRACKER_CFG_PATH,
                 min_hits=30, lost_ttl=None, trajectory_maxlen=64, fifo=None,
                 rebind=None):
        """
        CarMOT 초기화.

        Args:
            tracker_cfg:       추적기 설정 파일 경로. 이 yaml의 tracker_type이
                               실제로 사용할 추적기를 결정한다. (모듈 상단 TRACKERS)
            min_hits:          차량번호 부여에 필요한 최소 '연속' 추적 프레임 수
            lost_ttl:          추적 소실 후 매칭 정보를 유지할 프레임 수.
                               None이면 tracker_cfg의 track_buffer를 그대로 사용한다.
                               추적기가 트랙을 버리는 시점과 일치시키는 것이 기본 동작이므로
                               특별한 이유가 없으면 None으로 둘 것.
            trajectory_maxlen: 트랙별 궤적 저장 최대 길이
            fifo:              사용할 CarNumberFIFO 인스턴스 (None이면 모듈 전역 큐 사용)
            rebind:            재바인딩 설정 dict (None이면 CONFIG['REBIND'])
        """
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_cfg)))

        tracker_cls = TRACKERS.get(cfg.tracker_type)
        if tracker_cls is None:
            raise ValueError(
                f"지원하지 않는 tracker_type '{cfg.tracker_type}'. "
                f"사용 가능: {sorted(TRACKERS)} ({tracker_cfg})"
            )

        print(f"[INFO] {cfg.tracker_type} 추적기를 초기화합니다... ({tracker_cfg})")
        self.tracker_type = cfg.tracker_type
        self.tracker = tracker_cls(cfg)

        # 저신뢰(track_low_thresh) 2차 연관이 실제로 도는지 여부.
        # ByteTrack은 그 단계가 알고리즘 자체라 항상 켜져 있고,
        # OC-SORT는 use_byte로 켜고 끈다. ultralytics 기본값이 False라서,
        # 그대로 두면 oc_sort.py의 _second_association이 통째로 건너뛰어지고
        # (unmatched 트랙은 바로 mark_lost) 지금까지 쓰던 2차 연관이 '꺼진다'.
        # 조용히 성능이 나빠지는 함정이라 경고를 남긴다.
        self.use_byte = (self.tracker_type == 'bytetrack'
                         or bool(getattr(cfg, 'use_byte', False)))
        if self.tracker_type == 'ocsort' and not self.use_byte:
            print("[경고] use_byte=False. OC-SORT가 ByteTrack의 저신뢰 2차 연관을 "
                  "건너뜁니다. 의도한 설정이 아니면 yaml에서 True로 바꾸세요.")

        self.min_hits = min_hits
        # 추적기는 track_buffer 프레임을 넘겨 소실된 트랙을 버리고, 같은 차량이
        # 다시 잡혀도 새 ID를 발급한다. 그 시점 이후의 매칭 정보는 복구에 쓰일 수 없다.
        self.lost_ttl = cfg.track_buffer if lost_ttl is None else lost_ttl
        self.trajectory_maxlen = trajectory_maxlen

        self.fifo = fifo if fifo is not None else car_number_fifo

        # Track ID -> 차량 번호 매핑 (예: {5: "1234"})
        self.track_to_car = {}
        # Track ID -> 연속 검출 프레임 수 (오검출로 FIFO가 소모되는 것 방지)
        # 한 프레임이라도 관측되지 않으면 0으로 리셋된다.
        self.hit_counts = {}
        # Track ID -> 추적이 끊긴 뒤 경과한 프레임 수
        self.lost_counts = {}
        # Track ID -> 이동 궤적 deque([(cx, cy, timestamp), ...])
        self.trajectories = {}

        # 추적기 성능 측정용: 지금까지 등장한 '서로 다른' Track ID 전체.
        # 실제 차가 3대인데 이 집합의 크기가 12라면 ID 스위치가 9번 난 것이다.
        # 같은 영상으로 ocsort.yaml / bytetrack.yaml을 각각 돌려 이 수치를 비교할 것.
        # (ultralytics의 STrack ID 카운터는 클래스 전역이라 max값 대신 집합으로 센다)
        self.seen_track_ids = set()

        # --- 재바인딩 레이어 상태 ---
        self.rebind = dict(CONFIG['REBIND'] if rebind is None else rebind)
        self.rebind_enable = bool(self.rebind.get('ENABLE', True))

        # Track ID -> 마지막으로 관측된 상태
        #   {"center": (cx,cy), "bbox": [...], "world": (x_cm,y_cm)|None,
        #    "hist": ndarray|None, "time": float}
        self.last_states = {}
        # 사라진 Track ID -> 유령 기록 (차량번호 승계 후보)
        self.ghosts = {}
        # Track ID -> 재바인딩된 시각 (시각화 강조용)
        self.rebound_at = {}
        self.rebind_count = 0

        print(f"[INFO] 추적기 초기화 완료. "
              f"(Tracker: {self.tracker_type}, use_byte: {self.use_byte}, "
              f"track_buffer: {cfg.track_buffer}, "
              f"rebind: {'ON' if self.rebind_enable else 'OFF'})")

    def update(self, detections, frame=None, to_world=None):
        """
        검출 결과를 추적하고, 신규 트랙에 차량 번호를 매칭.

        번호 부여는 두 경로가 있고 재바인딩이 항상 우선한다.
          1) 재바인딩 : 사라진 트랙의 유령과 대조해 옛 번호를 승계 (FIFO 소비 없음)
          2) 신규 부여: 승계할 유령이 없으면 FIFO에서 다음 번호를 꺼낸다

        Args:
            detections: B01_car_detection.CarDetector.detect()의 반환 결과 리스트
                        [{"class_id":.., "class_name":.., "bbox":[..], "confidence":..}, ...]
            frame:      원본 BGR 프레임. 주면 색 히스토그램을 뽑아 재바인딩 판정에
                        외형 조건을 추가한다. 반드시 시각화가 그려지기 '전'의
                        프레임이어야 한다. (박스를 그린 프레임을 넣으면 초록 테두리
                        색이 히스토그램에 섞인다) None이면 위치/시간만으로 판정.
            to_world:   (x, y) 이미지 좌표 -> (x_cm, y_cm) 실좌표 변환 함수.
                        C00 MarkerMapper.image_to_world를 넘기면 된다.
                        주면 거리 판정이 픽셀 대신 cm 기준이 되어 훨씬 정확해진다.
                        (픽셀 거리는 원근 때문에 화면 위치마다 실제 거리가 다르다)
                        None이거나 호모그래피가 없으면 픽셀 기준으로 자동 대체.

        Returns:
            추적 결과 리스트. 각 항목은 딕셔너리:
            [
                {
                    "track_id": 5,
                    "car_id": "1234",          # 아직 매칭 전이면 None
                    "class_id": 2,
                    "class_name": "Car",
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),        # 박스 정중앙 기준점
                    "confidence": 0.94
                },
                ...
            ]
        """
        # 검출 결과를 추적기 입력 형식으로 변환 후 연관(association) 수행
        results = DetectionResults.from_detections(detections)
        outputs = self.tracker.update(results)

        tracks = []
        alive_ids = set()
        now = time.time()

        # 1) 추적기 출력을 파싱하고 트랙별 상태(위치/외형)를 갱신
        # outputs 각 행: [x1, y1, x2, y2, track_id, score, cls, det_idx]
        for row in outputs:
            x1, y1, x2, y2 = map(int, row[:4])
            track_id = int(row[4])
            conf = float(row[5])
            cls_id = int(row[6])

            alive_ids.add(track_id)
            self.seen_track_ids.add(track_id)
            self.hit_counts[track_id] = self.hit_counts.get(track_id, 0) + 1
            self.lost_counts.pop(track_id, None)

            # 박스 중앙(Center) 기준점
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            traj = self.trajectories.setdefault(
                track_id, deque(maxlen=self.trajectory_maxlen)
            )
            traj.append((cx, cy, now))

            bbox = [x1, y1, x2, y2]
            self._update_state(track_id, (cx, cy), bbox, conf, frame, to_world, now)

            tracks.append({
                "track_id": track_id,
                "car_id": None,             # 2)에서 확정한 뒤 3)에서 채운다
                "class_id": cls_id,
                "class_name": VEHICLE_CLASS_NAMES.get(cls_id, "Vehicle"),
                "bbox": bbox,
                "center": (cx, cy),
                "confidence": conf
            })

        # 2) 번호가 없는 트랙에 번호를 부여한다. 재바인딩(승계)이 FIFO보다 우선.
        #
        #    이 작업을 1)의 루프 안이 아니라 여기서 하는 이유:
        #    유령의 원래 트랙이 '이번 프레임에 살아있는지'를 알아야 하는데,
        #    루프 도중에는 alive_ids가 아직 완성되지 않았다. 미완성 상태로
        #    판단하면 멀쩡히 살아있는 트랙의 차량번호를 뺏어올 수 있다.
        for trk in tracks:
            track_id = trk["track_id"]
            if track_id in self.track_to_car:
                continue

            hits = self.hit_counts[track_id]
            if (self.rebind_enable and hits >= self.rebind['MIN_HITS']
                    and self._try_rebind(track_id, alive_ids, now)):
                continue

            # 승계할 유령이 없으면 신규 차량으로 보고 FIFO에서 번호를 꺼낸다
            if hits >= self.min_hits:
                self._assign_car_id(track_id)

        # 3) 확정된 차량번호를 결과에 반영
        for trk in tracks:
            trk["car_id"] = self.track_to_car.get(trk["track_id"])

        # 4) 이번 프레임에 잡히지 않은 트랙 정리 (+ 유령 생성/만료)
        self._cleanup_lost(alive_ids, now)

        return tracks

    def _update_state(self, track_id, center, bbox, conf, frame, to_world, now):
        """
        트랙의 마지막 관측 상태를 갱신. 재바인딩 판정의 비교 기준이 된다.
        """
        state = self.last_states.setdefault(track_id, {})
        state["center"] = center
        state["bbox"] = bbox
        state["time"] = now

        if to_world is not None:
            world = to_world(center)
            if world is not None:      # 호모그래피가 아직 준비 안 됐으면 None
                state["world"] = world

        # 외형은 '깨끗하게 보일 때'만 갱신한다.
        # 손에 반쯤 가려진 프레임(conf가 떨어진 프레임)으로 갱신하면 손 색이
        # 히스토그램에 섞여 들어가, 정작 복귀할 때 자기 자신과 안 닮게 된다.
        if frame is not None and conf >= self.rebind['HIST_MIN_CONF']:
            hist = color_histogram(frame, bbox)
            if hist is not None:
                prev = state.get("hist")
                if prev is None:
                    state["hist"] = hist
                else:
                    a = self.rebind['HIST_EMA']
                    # compareHist는 CV_32F를 요구한다. float 곱셈은 float64로
                    # 승격되므로 반드시 float32로 되돌려야 한다.
                    state["hist"] = (a * prev + (1.0 - a) * hist).astype(np.float32)

    def _try_rebind(self, track_id, alive_ids, now):
        """
        새로 생긴 트랙을 사라진 트랙의 유령과 대조해 차량번호를 승계한다.

        Args:
            track_id:  번호가 없는 신규 트랙
            alive_ids: 이번 프레임에 관측된 모든 Track ID
            now:       현재 시각

        Returns:
            승계에 성공하면 True.
        """
        state = self.last_states.get(track_id)
        if state is None or not self.ghosts:
            return False

        assigned = set(self.track_to_car.values())
        best_id, best_cost = None, None

        for ghost_id, ghost in self.ghosts.items():
            # 원래 트랙이 이번 프레임에 살아있으면 그 번호를 뺏으면 안 된다.
            # (추적기가 스스로 복구한 경우. 유령은 4)에서 폐기된다)
            if ghost_id == track_id or ghost_id in alive_ids:
                continue
            # 이미 다른 트랙이 쓰고 있는 번호도 제외
            if ghost["car_id"] in assigned:
                continue

            gap = now - ghost["lost_at"]
            if gap > self.rebind['MAX_GAP_SEC']:
                continue

            reachable, dist_cost = self._reachable(state, ghost, gap)
            if not reachable:
                continue
            similar, hist_cost = self._appearance_matches(state, ghost)
            if not similar:
                continue

            cost = dist_cost + hist_cost
            if best_cost is None or cost < best_cost:
                best_id, best_cost = ghost_id, cost

        if best_id is None:
            return False

        self._inherit_car_id(track_id, best_id, now)
        return True

    def _reachable(self, state, ghost, gap_sec):
        """
        사라진 위치에서 현재 위치까지, 그 시간 안에 물리적으로 갈 수 있는가.

        실좌표(cm)를 쓸 수 있으면 그쪽을 쓴다. 픽셀 거리는 원근 때문에 화면
        위치마다 같은 값이 다른 실제 거리를 뜻해서 임계값을 정할 수가 없다.
        cm 기준이면 'RC카가 1.2초 동안 최대 46cm'처럼 물리적으로 말이 된다.

        Returns:
            (도달 가능 여부, 정규화된 거리 비용 0~1)
        """
        w_now, w_old = state.get("world"), ghost.get("world")
        if w_now is not None and w_old is not None:
            limit = min(self.rebind['MAX_SPEED_CM_S'] * gap_sec + self.rebind['MARGIN_CM'],
                        self.rebind['MAX_REACH_CM'])
            dist = math.hypot(w_now[0] - w_old[0], w_now[1] - w_old[1])
        else:
            limit = min(self.rebind['MAX_SPEED_PX_S'] * gap_sec + self.rebind['MARGIN_PX'],
                        self.rebind['MAX_REACH_PX'])
            dist = math.hypot(state["center"][0] - ghost["center"][0],
                              state["center"][1] - ghost["center"][1])

        if dist > limit:
            return False, 1.0
        return True, dist / limit

    def _appearance_matches(self, state, ghost):
        """
        색 히스토그램이 충분히 비슷한가.

        한쪽이라도 히스토그램이 없으면(프레임을 안 넘겼거나 계산 실패)
        외형 조건은 건너뛰고 위치/시간만으로 판정한다.

        Returns:
            (유사 여부, 히스토그램 거리 0~1)
        """
        h_now, h_old = state.get("hist"), ghost.get("hist")
        if h_now is None or h_old is None:
            return True, 0.0

        dist = float(cv2.compareHist(h_now, h_old, cv2.HISTCMP_BHATTACHARYYA))
        return dist <= self.rebind['HIST_MAX_DIST'], dist

    def _inherit_car_id(self, new_id, ghost_id, now):
        """유령의 차량번호와 궤적을 새 트랙으로 옮기고 옛 트랙의 잔여 정보를 지운다."""
        ghost = self.ghosts.pop(ghost_id)
        car_id = ghost["car_id"]

        self.track_to_car[new_id] = car_id
        self.rebound_at[new_id] = now
        self.rebind_count += 1

        # 끊기기 전 궤적을 앞에 이어 붙인다.
        # 궤적선이 끊겨 보이지 않게 하는 목적도 있지만, C00이 최근 이동 궤적으로
        # 진행 방향(heading)을 계산하므로 여기서 이어주면 복귀 직후에도
        # 방향 안내가 UNKNOWN으로 떨어지지 않는다.
        old_traj = self.trajectories.pop(ghost_id, None)
        if old_traj:
            merged = deque(old_traj, maxlen=self.trajectory_maxlen)
            merged.extend(self.trajectories.get(new_id, ()))
            self.trajectories[new_id] = merged

        self.track_to_car.pop(ghost_id, None)
        self.hit_counts.pop(ghost_id, None)
        self.lost_counts.pop(ghost_id, None)
        self.last_states.pop(ghost_id, None)
        self.rebound_at.pop(ghost_id, None)

        print(f"[재바인딩] Track ID {ghost_id} -> {new_id} 승계 "
              f"(차량번호 '{car_id}', FIFO 소비 없음)")

    def _make_ghost(self, track_id, now):
        """
        사라진 트랙의 마지막 상태를 유령으로 남긴다.

        번호를 아직 못 받은 트랙은 승계할 것이 없으므로 유령을 만들지 않는다.
        """
        car_id = self.track_to_car.get(track_id)
        state = self.last_states.get(track_id)
        if car_id is None or state is None:
            return

        self.ghosts[track_id] = {
            "car_id": car_id,
            "center": state["center"],
            "world": state.get("world"),
            "hist": state.get("hist"),
            "lost_at": now,
        }

    def _expire_ghosts(self, now):
        """MAX_GAP_SEC이 지난 유령을 폐기. 그 뒤에 잡히는 차는 신규로 취급된다."""
        stale = [gid for gid, g in self.ghosts.items()
                 if now - g["lost_at"] > self.rebind['MAX_GAP_SEC']]
        for gid in stale:
            ghost = self.ghosts.pop(gid)
            print(f"[재바인딩] Track ID {gid} 유령 만료 "
                  f"(차량번호 '{ghost['car_id']}', 이후 재등장은 신규 취급)")

    def _assign_car_id(self, track_id):
        """
        FIFO에서 가장 먼저 들어온 차량 번호를 꺼내 트랙에 부여.
        큐가 비어있으면(UART 수신이 아직 늦은 경우) 부여하지 않고
        다음 프레임에 다시 시도.
        """
        car_id = self.fifo.pop()
        if car_id is None:
            return False

        self.track_to_car[track_id] = car_id
        print(f"[매칭] Track ID {track_id} <- 차량번호 '{car_id}'")
        return True

    def _cleanup_lost(self, alive_ids, now):
        """
        이번 프레임에 검출되지 않은 트랙의 소실 카운트를 증가시키고,
        lost_ttl(기본값: 추적기 설정 yaml의 track_buffer)을 초과하면
        매칭 정보와 궤적을 제거한다.

        아직 번호를 부여받지 못한 트랙은 이 시점에 연속 카운트(hit_counts)를
        0으로 되돌린다. min_hits는 '연속' 관측을 요구하므로, 끊긴 적이 있는
        트랙은 처음부터 다시 세어야 한다.

        재바인딩을 위한 유령도 여기서 만들고 폐기한다.
        """
        for track_id in list(self.hit_counts.keys()):
            if track_id in alive_ids:
                # 같은 ID로 돌아왔다. 추적기가 스스로 복구했으므로 유령은 불필요.
                self.ghosts.pop(track_id, None)
                continue

            # 연속성이 깨졌으므로 매칭 대기 중인 트랙의 카운트를 리셋
            if track_id not in self.track_to_car:
                self.hit_counts[track_id] = 0

            self.lost_counts[track_id] = self.lost_counts.get(track_id, 0) + 1

            # 사라진 '첫' 프레임에 유령을 남긴다.
            #
            # 이 시점이 중요하다. 트랙을 실제로 버리는 것은 lost_ttl(2초) 뒤인데,
            # 같은 차가 새 ID로 다시 잡히는 것은 보통 그보다 훨씬 이르다.
            # 버릴 때 유령을 만들면 정작 필요한 순간에 비교 대상이 없다.
            #
            # 유령을 만들어도 track_to_car는 그대로 두므로, 원래 ID로 복귀하는
            # 경우는 지금까지와 똑같이 동작한다. (위 alive_ids 분기에서 폐기)
            if self.rebind_enable and self.lost_counts[track_id] == 1:
                self._make_ghost(track_id, now)

            if self.lost_counts[track_id] < self.lost_ttl:
                continue

            car_id = self.track_to_car.pop(track_id, None)
            self.hit_counts.pop(track_id, None)
            self.lost_counts.pop(track_id, None)
            self.trajectories.pop(track_id, None)
            self.last_states.pop(track_id, None)
            self.rebound_at.pop(track_id, None)

            if car_id:
                # 유령은 여기서 지우지 않는다. 추적기가 트랙을 버린 뒤부터가
                # 오히려 재바인딩이 필요한 구간이다. (만료는 _expire_ghosts가 담당)
                held = " (유령 보존 중)" if track_id in self.ghosts else ""
                print(f"[소실] Track ID {track_id} (차량번호 '{car_id}') 추적 종료{held}")

        if self.rebind_enable:
            self._expire_ghosts(now)

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

    def track_id_stats(self, expected_cars=None):
        """
        추적기 성능(ID 스위치) 측정용 통계.

        정답을 아는 영상 - 예를 들어 차 3대를 넣고 손으로 5번, 기둥으로 3번
        가리는 클립 - 을 같은 조건으로 두 번 돌려서 비교한다.
        차가 3대면 이상적인 total_ids는 3이다. 12가 나왔다면 ID 스위치 9번.

            config/ocsort.yaml    -> total_ids: ?
            config/bytetrack.yaml -> total_ids: ?

        재바인딩 레이어가 켜져 있으면 total_ids가 커도 서비스에는 영향이 없을 수
        있다. ID가 바뀌어도 rebinds만큼은 car_id가 승계되었다는 뜻이기 때문이다.
        그래서 실제로 봐야 할 값은 uncovered다.

            uncovered = id_switches - rebinds

        이것이 '차량번호를 놓친 횟수'다. 0이면 ID가 아무리 바뀌었어도 하위
        모듈(주차관리/내비/UI)은 전혀 영향을 받지 않았다는 뜻이다.

        Args:
            expected_cars: 실제 차량 대수(정답). 주면 초과 발급된 ID 수까지 계산.

        Returns:
            {"tracker": "ocsort", "use_byte": True, "total_ids": 5, "rebinds": 2,
             "expected_cars": 3, "id_switches": 2, "uncovered": 0}
        """
        stats = {
            "tracker": self.tracker_type,
            "use_byte": self.use_byte,
            "total_ids": len(self.seen_track_ids),
            "rebinds": self.rebind_count,
            "ghosts_alive": len(self.ghosts),
        }
        if expected_cars is not None:
            switches = max(0, len(self.seen_track_ids) - expected_cars)
            stats["expected_cars"] = expected_cars
            stats["id_switches"] = switches
            stats["uncovered"] = max(0, switches - self.rebind_count)
        return stats

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

            # 방금 번호를 승계받은 트랙은 잠시 자홍색으로 강조한다.
            # 재바인딩이 실제로 도는지, 엉뚱한 차에 붙지는 않는지를
            # 영상만 보고 확인할 수 있어야 튜닝이 가능하다.
            rebound = self.rebound_at.get(track_id)
            if rebound is not None and time.time() - rebound < REBOUND_HIGHLIGHT_SEC:
                color = COLOR_REBOUND
                label = f"ID:{track_id} {car_id} REBIND"

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


# 테스트용 메인 (단독 실행 시 합성 검출 데이터로 추적/매칭 로직 검증)
# 이 모듈은 추적 전용이므로 카메라와 YOLO 모델 없이도 단독 검증이 가능.
# 카메라 + 검출 + 추적 통합 실행은 B_main.py를 사용할 것.
if __name__ == '__main__':
    print("==========================================")
    print(" B02 : 차량 다중 객체 추적 (MOT)")
    print(" 단독 테스트 : 합성 검출 데이터로 로직 검증")
    print(f" 추적기 설정 : {CONFIG['TRACKER_CFG']}")
    print("==========================================")

    # 테스트용 차량번호를 FIFO에 미리 등록 (원하는 번호로 수정 가능)
    TEST_CAR_NUMBERS = ["1234", "5678", "9012"]
    for num in TEST_CAR_NUMBERS:
        enqueue_car_number(num)

    mot = CarMOT(
        tracker_cfg=CONFIG['TRACKER_CFG'],
        min_hits=CONFIG['MIN_HITS_FOR_ASSIGN'],
        trajectory_maxlen=CONFIG['TRAJECTORY_MAXLEN']
    )

    # 차량 3대가 순차적으로 등장하여 오른쪽으로 이동하는 시나리오
    # (등장 프레임, 시작 x좌표, y좌표)
    # min_hits(30프레임 연속)를 채우려면 등장 간격도 그만큼 벌려야 한다.
    scenario = [(0, 50, 200), (40, 50, 320), (80, 50, 440)]
    TOTAL_FRAMES = 130
    MOVE_PER_FRAME = 4  # 프레임당 이동량(px)

    # 연속 카운트 리셋 검증용: 2번째 차량(index 1)을 잠시 검출에서 누락시킨다.
    # 이 구간 때문에 2번 차량은 카운트가 0으로 리셋되고 번호 부여가 그만큼 늦어져야 한다.
    DROPOUT = {1: range(55, 58)}

    print(f"\n[TEST] 합성 시나리오 시작: 차량 {len(scenario)}대 순차 등장")
    print(f"[TEST] min_hits={mot.min_hits} (연속), lost_ttl={mot.lost_ttl} (track_buffer에서 자동 적용)")
    print(f"[TEST] 2번 차량은 frame {DROPOUT[1].start}~{DROPOUT[1].stop - 1} 구간에서 검출 누락\n")

    for frame_idx in range(TOTAL_FRAMES):
        # 이번 프레임의 합성 검출 결과 생성 (B01의 detect() 반환 형식과 동일)
        detections = []
        for car_idx, (appear_at, start_x, y) in enumerate(scenario):
            if frame_idx < appear_at:
                continue
            if frame_idx in DROPOUT.get(car_idx, ()):
                continue
            x = start_x + (frame_idx - appear_at) * MOVE_PER_FRAME
            detections.append({
                "class_id": 2,
                "class_name": "Car",
                "bbox": [x, y, x + 100, y + 80],
                "confidence": 0.9
            })

        tracks = mot.update(detections)

        # 매칭 상태가 바뀌는 시점 위주로 출력
        if frame_idx in (29, 30, 54, 58, 70, 86, 87, 109, 110, TOTAL_FRAMES - 1):
            print(f"--- frame {frame_idx:3d} | 추적 {len(tracks)}대 | FIFO 잔여 {mot.fifo.size()}대")
            for t in tracks:
                car_str = t["car_id"] if t["car_id"] else "미매칭"
                hits = mot.hit_counts.get(t["track_id"], 0)
                print(f"      ID:{t['track_id']:<3d} 번호={car_str:8s} 연속={hits:<3d} bbox={t['bbox']}")

    print(f"\n[TEST] 최종 Track ID -> 차량번호 매핑: {mot.track_to_car}")
    print(f"[TEST] ID 통계: {mot.track_id_stats(expected_cars=len(scenario))}")
    print("[TEST] 확인 사항")
    print("       1) 검출된 순서대로 FIFO 번호가 부여되었는가")
    print(f"       2) 각 차량이 등장 후 정확히 {mot.min_hits}프레임째에 번호를 받았는가")
    print("       3) 검출이 누락된 2번 차량은 그만큼 번호 부여가 밀렸는가")
    print(f"       4) id_switches가 0인가 (차 {len(scenario)}대 -> Track ID {len(scenario)}개)")

    # ---------------------------------------------------------------------
    # 시나리오 2 : 재바인딩 레이어 검증
    #
    # 차가 오래 사라져 추적기가 트랙을 완전히 버린 뒤(새 Track ID 발급),
    # 같은 자리에 다시 나타났을 때 차량번호가 승계되는가.
    #
    # 색까지 검증하기 위해 합성 프레임을 만들어 넘긴다.
    #   - 같은 색으로 돌아오면 -> 승계 (FIFO 소비 없음)
    #   - 다른 색으로 돌아오면 -> 외형 조건에서 걸러져 신규 취급 (FIFO 소비)
    # ---------------------------------------------------------------------
    RED, BLUE = (0, 0, 255), (255, 0, 0)

    def synth_frame(boxes):
        """(bbox, BGR색) 목록으로 합성 프레임을 만든다."""
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)   # 어두운 배경
        for (bx1, by1, bx2, by2), col in boxes:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, -1)
        return frame

    def run_rebind_case(return_color, label):
        """차가 사라졌다가 return_color 색으로 되돌아오는 시나리오를 돌린다."""
        fifo = CarNumberFIFO()
        fifo.push("1234")     # 처음 등장한 차가 받을 번호
        fifo.push("5678")     # 승계에 실패하면 이 번호가 소비된다

        m = CarMOT(
            tracker_cfg=CONFIG['TRACKER_CFG'],
            min_hits=CONFIG['MIN_HITS_FOR_ASSIGN'],
            trajectory_maxlen=CONFIG['TRAJECTORY_MAXLEN'],
            fifo=fifo
        )

        GONE = range(60, 130)          # lost_ttl(60)을 넘겨 트랙이 완전히 버려지는 구간
        for f in range(180):
            if f in GONE:
                boxes, dets = [], []
            else:
                # 사라지기 전엔 빨강, 돌아올 때는 return_color.
                # 위치도 20px만 옮겨 '같은 자리에 다시 나타난' 상황을 만든다.
                col = RED if f < GONE.start else return_color
                x = 100 if f < GONE.start else 120
                box = (x, 100, x + 90, 160)
                boxes = [(box, col)]
                dets = [{"class_id": 2, "class_name": "Car",
                         "bbox": list(box), "confidence": 0.9}]
            m.update(dets, frame=synth_frame(boxes))

        stats = m.track_id_stats(expected_cars=1)
        print(f"\n  [{label}]")
        print(f"    최종 매핑   : {m.track_to_car}")
        print(f"    FIFO 잔여   : {fifo.snapshot()}")
        print(f"    통계        : {stats}")
        return m, fifo

    print("\n" + "=" * 60)
    print(" 시나리오 2 : 재바인딩 (차가 사라진 뒤 새 Track ID로 복귀)")
    print("=" * 60)
    print(f" 설정: MIN_HITS={CONFIG['REBIND']['MIN_HITS']}, "
          f"MAX_GAP_SEC={CONFIG['REBIND']['MAX_GAP_SEC']}, "
          f"HIST_MAX_DIST={CONFIG['REBIND']['HIST_MAX_DIST']}")
    print(" 주의: 이 루프는 실시간이 아니라 즉시 돌기 때문에 MAX_GAP_SEC(시간)"
          " 조건은 사실상 항상 통과한다. 여기서 검증되는 것은 거리/외형 조건이다.")

    same, same_fifo = run_rebind_case(RED,  "같은 색으로 복귀 -> 승계 기대")
    diff, diff_fifo = run_rebind_case(BLUE, "다른 색으로 복귀 -> 신규 취급 기대")

    print("\n[TEST] 확인 사항")
    ok_same = (same.rebind_count == 1 and "5678" in same_fifo.snapshot()
               and list(same.track_to_car.values()) == ["1234"])
    ok_diff = (diff.rebind_count == 0 and "5678" not in diff_fifo.snapshot())
    print(f"       5) 같은 색 복귀 시 '1234'를 승계하고 FIFO의 '5678'은 그대로인가"
          f"  -> {'OK' if ok_same else 'FAIL'}")
    print(f"       6) 다른 색 복귀 시 승계를 거부하고 '5678'을 새로 소비했는가"
          f"  -> {'OK' if ok_diff else 'FAIL'}")
