import cv2
import time
import sys
import os
import io
import math
import contextlib
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

    # 지나온 경로를 화면에 선으로 그릴지. 궤적 자체는 진행 방향 계산에 계속
    # 쓰이므로 저장은 유지하고, 화면 표시만 끈다.
    "DRAW_TRAJECTORY": False,

    # =====================================================================
    # 단일 활성 차량(Single Active Vehicle) 모드
    #
    # 운영 시나리오상 '동시에 움직이는 차는 1대'로 고정한다.
    # 차가 한 대씩 들어와 배정된 구역까지 이동해 주차하고, 그다음 차가 들어온다.
    #
    # 이 제약은 어떤 추적 알고리즘보다 강한 단서다.
    #   "지금 움직이고 있는 차 = 지금 안내 중인 차"
    # 이 한 줄이면 ID가 바뀌어도 다음 프레임에 곧바로 복구된다.
    # 색/거리/외형을 따질 필요가 없다.
    #
    # 동작:
    #   1) 활성 차량이 없고 움직이는 트랙이 나타나면 FIFO에서 번호를 꺼내 붙인다.
    #   2) 한 번 붙으면 트랙이 살아 있는 동안 유지된다. (sticky)
    #      중간에 차가 멈춰 서도 결합은 풀리지 않는다.
    #   3) 트랙이 끊기면, 다시 움직이는 트랙이 나타나는 즉시 같은 번호를 붙인다.
    #   4) 목표 구역 ARRIVAL_RADIUS_CM 이내에 ARRIVAL_HOLD_SEC 이상 머무르면
    #      주차 완료로 보고 활성 차량을 해제한다. -> 다음 차 차례
    #
    # 주차가 끝난 차는 번호를 그대로 유지한 채 정지 상태로 남는다. 그 차의 트랙이
    # 끊겼다 되살아나는 경우는 아래 재바인딩 레이어(위치 매칭)가 담당한다.
    # =====================================================================
    "SINGLE_ACTIVE": {
        "ENABLE": True,

        # --- '움직이는 중' 판정 ---
        # 최근 MOVE_WINDOW_SEC 동안의 이동량이 임계값을 넘으면 움직이는 것으로 본다.
        # 실좌표(cm)를 쓸 수 있으면 cm 기준, 아니면 픽셀 기준으로 자동 전환.
        "MOVE_WINDOW_SEC": 0.5,
        "MOVE_MIN_CM": 2.0,
        "MOVE_MIN_PX": 12.0,

        # 활성 차량으로 인정할 최소 연속 추적 프레임.
        # '일정 시간 일관되게 움직였다'는 조건 자체가 오검출을 걸러주므로
        # MIN_HITS_FOR_ASSIGN(30)만큼 길게 볼 필요가 없다.
        "MIN_HITS": 5,

        # --- 주차 완료(안내 종료) 판정 ---
        # 목표 구역 중심에서 이 거리 이내로 들어오면 주차한 것으로 본다.
        # 목업 주차장은 자리 간격이 35cm, 통로 폭이 10cm이므로 그보다 작아야 한다.
        # 너무 작으면(예: 3cm) 차가 살짝 못 미쳐 멈췄을 때 영영 해제되지 않고,
        # 너무 크면 지나가는 중에 주차한 것으로 오판한다.
        # 주차장을 키우면 C02의 CELL_W/H_CM과 함께 조정할 것.
        "ARRIVAL_RADIUS_CM": 8.0,

        # 위 반경 안에 이 시간 이상 머물러야 확정한다.
        # 목표 구역을 스쳐 지나가는 순간을 주차로 오판하지 않기 위한 조건.
        "ARRIVAL_HOLD_SEC": 1.0,

        # --- 안전장치 ---
        # 도착 판정이 되지 않는 상태로 이 시간 이상 정지해 있으면 강제 해제한다.
        # 호모그래피가 아직 안 잡혔거나, 차가 목표 반경에 살짝 못 미친 곳에
        # 멈춰버린 경우에 대비한 것이다. 이게 없으면 활성 차량이 영영 해제되지
        # 않아 다음 차가 번호를 못 받는다.
        # 0으로 두면 비활성화되고 오로지 도착 판정으로만 해제된다.
        "STUCK_RELEASE_SEC": 15.0,

        # 이미 주차된 차를 그 자리의 트랙에 묶을 때 쓰는 반경(cm).
        #
        # INITIAL_PARKED로 미리 세워둔 차나 안내를 마친 차는 cars_info에만
        # 기록되어 있을 뿐, 화면의 어느 트랙이 그 차인지 알 수 없다.
        # 그대로 두면 'WAIT'로 뜨고, 검출이 흔들려 조금이라도 움직이면
        # 활성 차량으로 잡혀 엉뚱하게 안내가 시작된다.
        # 그래서 배정된 자리 근처에 서 있는 트랙을 찾아 차량번호를 묶어준다.
        #
        # ARRIVAL_RADIUS_CM보다 넉넉하게 둔다. 실제로 세워둔 위치가 자리
        # 중심에서 조금 벗어나 있어도 잡아야 하기 때문이다.
        "PARKED_BIND_RADIUS_CM": 15.0,
    },

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
    # 동작: 번호를 가진 트랙이 사라지면 마지막 위치를 '유령(ghost)'으로 남겨두고,
    #       새 track_id가 뜨면 FIFO에서 번호를 꺼내기 '전에' 유령과 먼저 대조한다.
    #       조건을 만족하면 FIFO를 소비하지 않고 옛 번호를 그대로 승계한다.
    #
    # 판정 조건 (둘 다 통과해야 승계):
    #   1) 시간 : 사라진 지 MAX_GAP_SEC 이내
    #   2) 거리 : 그 시간 안에 물리적으로 도달 가능한 위치인가
    #
    # 단일 활성 차량 모드에서 이 레이어가 담당하는 것은 '이미 주차를 마친 차'다.
    # 움직이는 차는 위의 SINGLE_ACTIVE 규칙이 훨씬 확실하게 잡아준다.
    # 주차된 차는 정지해 있으므로 위치만으로 충분히 구분된다.
    # (색 히스토그램도 썼었지만, 위 규칙이 생기면서 불필요해져 제거했다)
    # ---------------------------------------------------------------------
    "REBIND": {
        "ENABLE": True,

        # 재바인딩 시도에 필요한 최소 연속 추적 프레임.
        # MIN_HITS_FOR_ASSIGN(30)보다 훨씬 짧게 둔다. 재바인딩은 거리로 이미
        # 검증된 승계라 FIFO를 새로 소비하는 것보다 안전하고, 되돌아온 차가
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
    },
}

# 트랙이 같은 ID로 되살아날 수 있는 기간은 전적으로 추적기가 결정(track_buffer),

# 차량번호가 부여된 트랙 / 대기 중인 트랙 색상 (BGR)
COLOR_MATCHED = (0, 255, 0)     # 번호 매칭 완료 - 초록
COLOR_PENDING = (0, 165, 255)   # 번호 대기 중   - 주황
COLOR_REBOUND = (255, 0, 255)   # 번호 승계 직후 - 자홍 (승계가 실제로 도는지 눈으로 확인용)
COLOR_ACTIVE  = (0, 255, 255)   # 현재 안내 중인 활성 차량 - 노랑

# 번호 승계 직후 자홍색으로 강조할 시간(초).
REBOUND_HIGHLIGHT_SEC = 2.0


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

    def push_front(self, car_id):
        """
        잘못 꺼낸 번호를 큐의 '앞'으로 되돌린다.

        뒤에 붙이면 안 된다. 이 번호는 원래 다음 차례였으므로, 뒤로 보내면
        나중에 들어온 차가 먼저 안내를 받는다. 입구 통과 순서와 안내 순서가
        어긋나면 FIFO를 쓰는 의미가 없어진다.
        """
        with self._lock:
            if car_id in self._queue:
                return
            self._queue.appendleft(car_id)
            print(f"[FIFO] 차량번호 '{car_id}' 반납 (대기 {len(self._queue)}대)")


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

    동작 흐름 (단일 활성 차량 모드, CONFIG['SINGLE_ACTIVE']):
      1. UART로 수신된 차량 번호가 CarNumberFIFO에 순서대로 쌓인다.
      2. B01이 검출한 결과를 추적기에 넣어 Track ID를 부여한다.
      3. '움직이는' 트랙이 나타나면 FIFO에서 가장 앞 번호를 꺼내 붙인다.
         이 차가 활성 차량(active_car_id), 즉 지금 안내 중인 차다.
      4. 결합은 sticky다. 트랙이 살아 있는 동안 유지되므로 도중에 차가 멈춰
         서도 번호가 떨어지지 않는다.
      5. 추적이 끊겨 Track ID가 새로 발급되면, 다시 움직이는 트랙이 나타나는
         즉시 같은 번호를 붙인다. (동시에 움직이는 차가 1대뿐이므로 확실하다)
      6. 목표 구역 ARRIVAL_RADIUS_CM 이내에 ARRIVAL_HOLD_SEC 이상 머무르면
         주차 완료로 보고 활성 차량을 해제한다. -> 다음 차 차례

    "지금 움직이는 차 = 지금 안내 중인 차"라는 운영 제약이 어떤 외형/거리
    매칭보다 강한 단서다. ID가 바뀌어도 다음 프레임에 곧바로 복구된다.

    주차를 마친 차는 정지한 채 번호를 유지한다. 그 차의 트랙이 끊겼다 되살아나는
    경우는 재바인딩 레이어(CONFIG['REBIND'])가 마지막 위치로 이어받는다.
    정지해 있으므로 위치만으로 충분히 구분된다.

    C00의 world_history / routes / targets가 전부 car_id로 키를 잡고 있으므로,
    car_id만 유지되면 ID 스위치가 나도 경로 안내 상태가 그대로 살아남는다.

    SINGLE_ACTIVE를 끄면 예전 방식(MIN_HITS_FOR_ASSIGN 연속 관측 후 FIFO 소비)
    으로 돌아간다. 여러 대가 동시에 움직이는 시나리오로 바뀌면 그쪽을 쓸 것.

    추적기 성능 비교는 track_id_stats()로 측정할 것.
    """

    def __init__(self, tracker_cfg=TRACKER_CFG_PATH,
                 min_hits=30, lost_ttl=None, trajectory_maxlen=64, fifo=None,
                 rebind=None, single_active=None, on_parked=None):
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
            single_active:     단일 활성 차량 모드 설정 dict (None이면 CONFIG['SINGLE_ACTIVE'])
            on_parked:         차량이 목표 구역에 도착해 안내가 끝났을 때 호출할
                               콜백. car_id 하나를 인자로 받는다.
                               A01.mark_parked를 넘기면 cars_info가 갱신된다.
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
        #   {"center": (cx,cy), "bbox": [...], "world": (x_cm,y_cm)|None, "time": float}
        self.last_states = {}
        # 사라진 Track ID -> 유령 기록 (차량번호 승계 후보)
        self.ghosts = {}
        # Track ID -> 번호를 승계받은 시각 (시각화 강조용)
        self.rebound_at = {}
        self.rebind_count = 0

        # --- 단일 활성 차량 모드 상태 ---
        self.single = dict(CONFIG['SINGLE_ACTIVE'] if single_active is None else single_active)
        self.single_enable = bool(self.single.get('ENABLE', True))

        # 안내가 끝났을 때 알려줄 콜백 (A01.mark_parked)
        self.on_parked = on_parked

        # 지금 안내 중인 차량번호. 주차를 마치면 None으로 돌아간다.
        self.active_car_id = None
        # 목표 반경 안에 처음 들어온 시각 (ARRIVAL_HOLD_SEC 판정용)
        self._arrived_since = None
        # 활성 차량이 마지막으로 '움직인' 시각 (STUCK_RELEASE_SEC 판정용)
        self._active_moved_at = None
        # Track ID -> 실좌표 궤적 deque([(x_cm, y_cm, t), ...])
        # 이동량을 cm로 재기 위한 것. to_world가 주어질 때만 쌓인다.
        self.world_trajectories = {}
        # Track ID -> 그 트랙의 '첫' 실좌표.
        # 궤적(world_trajectories)은 maxlen이 있어 오래된 점이 밀려 나가므로,
        # '처음 잡힌 자리에서 얼마나 벗어났는지'는 따로 붙들어야 한다.
        # 이미 세워둔 차인지(제자리에서 떨린 것뿐인지) 가리는 데 쓴다.
        self.first_world = {}

        # 이번 프레임에 좌표계를 쓸 수 있는가.
        # 보정(/calibrate) 전에는 to_world가 좌표를 못 주므로 False다.
        self._world_ready = False
        # 호출자가 '이미 세워둔 차' 목록을 넘겨주는가. (parked_of 제공 여부)
        self._parked_known = False

        print(f"[INFO] 추적기 초기화 완료. "
              f"(Tracker: {self.tracker_type}, use_byte: {self.use_byte}, "
              f"track_buffer: {cfg.track_buffer}, "
              f"rebind: {'ON' if self.rebind_enable else 'OFF'}, "
              f"single_active: {'ON' if self.single_enable else 'OFF'})")
        if self.single_enable:
            print(f"[INFO] 단일 활성 차량 모드. "
                  f"주차 판정 반경 {self.single['ARRIVAL_RADIUS_CM']}cm / "
                  f"{self.single['ARRIVAL_HOLD_SEC']}초 유지")

    def update(self, detections, to_world=None, target_of=None, parked_of=None):
        """
        검출 결과를 추적하고, 트랙에 차량 번호를 매칭.

        번호 부여 경로는 우선순위 순으로 넷이다.
          1) 주차 완료  : 이미 그 자리에 세워져 있는 차 (parked_of)
          2) 활성 차량  : 움직이는 트랙 = 지금 안내 중인 차 (SINGLE_ACTIVE)
          3) 재바인딩   : 사라진 트랙의 유령과 위치를 대조해 옛 번호를 승계
          4) 신규 부여  : 위 셋 다 아니면 FIFO에서 다음 번호를 꺼낸다

        Args:
            detections: B01_car_detection.CarDetector.detect()의 반환 결과 리스트
                        [{"class_id":.., "class_name":.., "bbox":[..], "confidence":..}, ...]
            to_world:   (x, y) 이미지 좌표 -> (x_cm, y_cm) 실좌표 변환 함수.
                        C_main이 보정 좌표계에 맞춰 만들어 넘긴다.
                        주면 이동량/거리 판정이 픽셀 대신 cm 기준이 되어 정확해진다.
                        (픽셀 거리는 원근 때문에 화면 위치마다 실제 거리가 다르다)
                        None이거나 호모그래피가 없으면 픽셀 기준으로 자동 대체.
            target_of:  car_id -> (x_cm, y_cm) 목표 구역 실좌표를 돌려주는 함수.
                        주차 완료(안내 종료) 판정에 쓴다. C_main이 C00의
                        targets/spot_world_pos를 엮어서 넘긴다.
                        None이면 도착 판정을 할 수 없으므로 STUCK_RELEASE_SEC
                        (정지 시간) 기준으로만 활성 차량이 해제된다.
            parked_of:  {차량번호: (x_cm, y_cm)}를 돌려주는 함수.
                        '이미 그 자리에 세워져 있는 차'의 목록이다.
                        A01.get_parked_world_positions를 넘기면 된다.
                        그 자리 근처에 서 있는 트랙을 찾아 차량번호를 묶는다.
                        None이면 이 단계를 건너뛴다.

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
            self._update_state(track_id, (cx, cy), bbox, to_world, now)

            tracks.append({
                "track_id": track_id,
                "car_id": None,             # 2)에서 확정한 뒤 3)에서 채운다
                "class_id": cls_id,
                "class_name": VEHICLE_CLASS_NAMES.get(cls_id, "Vehicle"),
                "bbox": bbox,
                "center": (cx, cy),
                "confidence": conf
            })

        # 좌표계가 준비됐는지 확인한다. 보정 전에는 to_world가 좌표를 주지
        # 못하므로 어느 트랙에도 world가 없다. 이 상태에서는 '자리에 서 있는
        # 차'와 '움직이는 차'를 구별할 방법이 없다. (아래 3) 주석 참고)
        self._parked_known = parked_of is not None
        self._world_ready = any(
            self.last_states.get(trk["track_id"], {}).get("world") is not None
            for trk in tracks)

        # 2) 이미 주차된 차를 그 자리의 트랙에 묶는다.
        #    활성 차량 결합보다 먼저 해야, 주차된 차가 검출 흔들림 때문에
        #    '움직이는 차'로 오인되어 안내 대상이 되는 것을 막을 수 있다.
        if parked_of is not None:
            self._bind_parked_cars(tracks, parked_of(), now)

        # 3) 활성 차량(움직이는 차) 결합 및 주차 완료 판정.
        #    이 규칙이 가장 강하므로 재바인딩/FIFO보다 먼저 처리한다.
        if self.single_enable:
            self._update_active_binding(tracks, alive_ids, now, target_of)

        # 4) 나머지 트랙에 번호를 부여한다. 재바인딩(승계)이 FIFO보다 우선.
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

            # 단일 활성 차량 모드에서는 FIFO를 여기서 소비하지 않는다.
            # 번호는 '움직이는 차'에게만 나가야 하고 그 판단은 2)가 담당한다.
            # 여기서도 꺼내면 주차장에 서 있던 차나 오검출이 번호를 채간다.
            if self.single_enable:
                continue

            # 좌표계가 없으면 세워둔 차를 가려낼 수 없다. (3)의 주석 참고)
            if self._parked_known and not self._world_ready:
                continue

            if hits >= self.min_hits:
                self._assign_car_id(track_id)

        # 5) 확정된 차량번호를 결과에 반영
        for trk in tracks:
            trk["car_id"] = self.track_to_car.get(trk["track_id"])

        # 6) 이번 프레임에 잡히지 않은 트랙 정리 (+ 유령 생성/만료)
        self._cleanup_lost(alive_ids, now)

        return tracks

    def _bind_parked_cars(self, tracks, parked_positions, now):
        """
        이미 주차된 차를 그 자리에 서 있는 트랙에 묶는다.

        cars_info에는 '차량번호 -> 배정된 자리'가 있지만, 화면의 어느 트랙이
        그 차인지는 알 수 없다. 그래서 자리의 실좌표 근처에 서 있는 트랙을
        찾아 차량번호를 붙여준다.

        이걸 하지 않으면 미리 세워둔 차가 계속 'WAIT'로 뜨고, 검출이 흔들려
        조금만 움직여도 활성 차량으로 잡혀 엉뚱한 안내가 시작된다.

        Args:
            tracks:           이번 프레임 트랙 목록
            parked_positions: {차량번호: (x_cm, y_cm)} 주차 완료된 차의 자리 좌표
            now:              현재 시각
        """
        if not parked_positions:
            return

        radius = self.single['PARKED_BIND_RADIUS_CM']
        taken = set(self.track_to_car.values())

        for trk in tracks:
            track_id = trk["track_id"]
            held = self.track_to_car.get(track_id)
            if held is not None and held in parked_positions:
                continue        # 이미 제 번호로 묶여 있다

            world = self.last_states.get(track_id, {}).get("world")
            if world is None:
                continue        # 좌표계가 아직 없으면 판단 불가

            # 반경 안에서 가장 가까운 '아직 안 묶인' 주차 차량
            best_car, best_dist = None, None
            for car_id, pos in parked_positions.items():
                if car_id in taken:
                    continue
                dist = math.hypot(world[0] - pos[0], world[1] - pos[1])
                if dist > radius:
                    continue
                if best_dist is None or dist < best_dist:
                    best_car, best_dist = car_id, dist

            if best_car is None:
                continue

            # 이미 다른 번호를 달고 있는 트랙이라면, 그 번호가 잘못 나간 것이다.
            #
            # 이런 일이 생기는 순서는 이렇다. 프로그램이 뜨면 검출이 먼저 돌기
            # 시작하는데, 그때는 아직 /calibrate 전이라 좌표계가 없다. 자리
            # 위치를 모르니 아래 주차 결합을 할 수 없고, 그 사이 세워둔 차의
            # 검출 상자가 몇 픽셀 흔들린 것을 '움직이는 차'로 보고 FIFO의
            # 다음 번호를 내준다. 보정을 마쳐도 그 트랙은 이미 번호를 달고
            # 있어서 여기까지 오지 못했다.
            #
            # 그래서 '처음 잡힌 자리에서 한 번도 벗어난 적 없는' 트랙에 한해
            # 바로잡는다. 안내를 받고 들어와 주차한 차는 주차장을 가로질러
            # 왔으므로 이 조건에 걸리지 않는다. 옆자리에 세우려고 잠깐 멈춘
            # 차를 빼앗지 않기 위한 구분이다.
            if held is not None:
                if not self._stayed_in_place(track_id, radius):
                    continue
                self.fifo.push_front(held)
                if self.active_car_id == held:
                    self.active_car_id = None
                    self._arrived_since = None
                print(f"[정정] Track ID {track_id}는 처음부터 자리에 서 있던 차입니다. "
                      f"'{held}' -> '{best_car}' (번호는 FIFO로 돌려줌)")

            self.track_to_car[track_id] = best_car
            self.rebound_at[track_id] = now
            taken.discard(held)
            taken.add(best_car)
            print(f"[주차 결합] Track ID {track_id} <- 차량번호 '{best_car}' "
                  f"(자리에서 {best_dist:.1f}cm)")

    def _stayed_in_place(self, track_id, radius):
        """
        트랙이 처음 잡힌 자리를 벗어난 적이 없는지.

        검출 상자는 제자리에서도 늘 조금씩 흔들리므로 '한 프레임 이동량'으로는
        서 있는 차와 기어가는 차를 가를 수 없다. 대신 처음 위치에서의 총
        벗어남을 본다. 세워둔 차는 떨림 범위를 넘지 않고, 주차장을 가로질러 온
        차는 훨씬 크게 벗어난다.

        Returns:
            벗어난 적이 없으면 True. 판단할 좌표가 없으면 False. (모르면 건드리지 않는다)
        """
        start = self.first_world.get(track_id)
        world = self.last_states.get(track_id, {}).get("world")
        if start is None or world is None:
            return False

        traj = self.world_trajectories.get(track_id) or ()
        points = [(x, y) for x, y, _ in traj] + [(world[0], world[1])]
        return all(math.hypot(x - start[0], y - start[1]) <= radius
                   for x, y in points)

    def _update_state(self, track_id, center, bbox, to_world, now):
        """
        트랙의 마지막 관측 상태를 갱신. 재바인딩/이동 판정의 비교 기준이 된다.
        """
        state = self.last_states.setdefault(track_id, {})
        state["center"] = center
        state["bbox"] = bbox
        state["time"] = now

        if to_world is not None:
            world = to_world(center)
            if world is not None:      # 호모그래피가 아직 준비 안 됐으면 None
                state["world"] = world
                self.first_world.setdefault(track_id, (world[0], world[1]))
                wtraj = self.world_trajectories.setdefault(
                    track_id, deque(maxlen=self.trajectory_maxlen)
                )
                wtraj.append((world[0], world[1], now))

    # --- 단일 활성 차량 모드 -------------------------------------------------

    def _update_active_binding(self, tracks, alive_ids, now, target_of):
        """
        '움직이는 트랙 = 지금 안내 중인 차'라는 규칙으로 활성 차량을 관리한다.

        결합은 sticky다. 한 번 붙으면 트랙이 살아 있는 동안 유지되므로,
        차가 도중에 멈춰 서도 번호가 떨어지지 않는다. 이동 판정은 결합이
        끊겼을 때 '다시 붙일 대상'을 고르는 데만 쓴다.
        """
        bound = self._track_of_car(self.active_car_id) if self.active_car_id else None

        if bound is not None and bound in alive_ids:
            # 이미 붙어 있다. 주차를 마쳤는지만 확인한다.
            self._check_parked(bound, now, target_of)
            return

        # 좌표계가 아직 없으면 번호를 내주지 않는다.
        #
        # 보정(/calibrate) 전에는 자리가 화면 어디인지 모르므로 위의
        # _bind_parked_cars가 동작하지 못한다. 그 상태에서 이동 판정은 픽셀
        # 기준으로 떨어지는데(MOVE_MIN_PX), 세워둔 차도 검출 상자가 그만큼은
        # 흔들린다. 그래서 가만히 서 있는 차가 '움직이는 차'로 잡혀 FIFO의
        # 다음 번호를 채갔다. 어차피 보정 전에는 목표 좌표도 경로도 없어
        # 안내를 시작할 수 없으니, 좌표계가 설 때까지 기다린다.
        if self._parked_known and not self._world_ready:
            return

        # 결합이 없다(아직 시작 전이거나 트랙이 끊겼다). 움직이는 트랙을 찾는다.
        moving = self._find_moving_track(tracks, now)
        if moving is None:
            return

        if self.active_car_id is None:
            car_id = self.fifo.pop()
            if car_id is None:
                return          # 아직 UART로 번호가 안 들어왔다. 다음 프레임에 재시도.
            self.active_car_id = car_id
            self._active_moved_at = now
            print(f"[활성] 차량번호 '{car_id}' 안내 시작 (Track ID {moving})")
        else:
            # 같은 차인데 추적이 끊겨 ID만 새로 발급된 경우. 번호를 그대로 옮긴다.
            self.rebind_count += 1
            print(f"[활성] 차량번호 '{self.active_car_id}' 재결합 "
                  f"(Track ID {moving}, FIFO 소비 없음)")

        self._bind_active(moving, now)

    def _bind_active(self, track_id, now):
        """활성 차량번호를 트랙에 붙이고, 이전 트랙에 남아 있던 흔적을 지운다."""
        for tid, cid in list(self.track_to_car.items()):
            if cid == self.active_car_id and tid != track_id:
                self.track_to_car.pop(tid, None)
                self.ghosts.pop(tid, None)

        self.track_to_car[track_id] = self.active_car_id
        self.rebound_at[track_id] = now
        self._arrived_since = None

    def _track_of_car(self, car_id):
        """차량번호가 붙어 있는 Track ID. 없으면 None."""
        for tid, cid in self.track_to_car.items():
            if cid == car_id:
                return tid
        return None

    def _find_moving_track(self, tracks, now):
        """
        가장 많이 움직인 트랙을 고른다. 임계값 미만이면 None.

        이미 다른 차량번호가 붙은 트랙(= 주차를 마친 차)은 후보에서 제외한다.
        """
        best, best_move = None, 0.0
        for trk in tracks:
            track_id = trk["track_id"]
            if self.hit_counts.get(track_id, 0) < self.single['MIN_HITS']:
                continue
            bound_car = self.track_to_car.get(track_id)
            if bound_car is not None and bound_car != self.active_car_id:
                continue

            moved, threshold = self._recent_movement(track_id, now)
            if moved is None or moved < threshold:
                continue
            if moved > best_move:
                best, best_move = track_id, moved
        return best

    def _recent_movement(self, track_id, now):
        """
        최근 MOVE_WINDOW_SEC 동안의 이동 거리와 그 판정 임계값을 돌려준다.

        실좌표 궤적이 쌓여 있으면 cm, 아니면 픽셀 기준으로 자동 전환한다.

        Returns:
            (이동거리, 임계값). 계산할 수 없으면 (None, 0).
        """
        window = self.single['MOVE_WINDOW_SEC']

        wtraj = self.world_trajectories.get(track_id)
        if wtraj and len(wtraj) >= 2:
            points, threshold = wtraj, self.single['MOVE_MIN_CM']
        else:
            points, threshold = self.trajectories.get(track_id), self.single['MOVE_MIN_PX']
            if not points or len(points) < 2:
                return None, 0.0

        # 윈도우 안에서 가장 오래된 점을 찾는다
        oldest = None
        for x, y, t in points:
            if now - t <= window:
                oldest = (x, y)
                break
        if oldest is None:
            return None, 0.0

        last = points[-1]
        return math.hypot(last[0] - oldest[0], last[1] - oldest[1]), threshold

    @staticmethod
    def _distance_to_target(world, target):
        """
        차량 위치에서 목표까지의 거리(cm).

        target이 (x, y, 반폭, 반높이)면 '사각형까지의 거리'를 잰다.
        자리 안에 들어와 있으면 0이 되고, 밖에 있으면 가장 가까운 변까지의
        거리가 나온다. 이렇게 해야 자리 크기가 판정에 반영된다.

        target이 (x, y)면 중심까지의 직선거리를 잰다. (예전 방식)
        """
        if len(target) >= 4:
            dx = max(abs(world[0] - target[0]) - target[2], 0.0)
            dy = max(abs(world[1] - target[1]) - target[3], 0.0)
            return math.hypot(dx, dy)
        return math.hypot(world[0] - target[0], world[1] - target[1])

    def _check_parked(self, track_id, now, target_of):
        """
        활성 차량이 목표 구역에 도착해 주차를 마쳤는지 판정한다.

        목표 구역까지의 거리가 ARRIVAL_RADIUS_CM 이내인 상태로
        ARRIVAL_HOLD_SEC 이상 머무르면 주차 완료로 보고 활성 차량을 해제한다.

        target_of가 (x, y, 반폭, 반높이)를 주면 '자리 사각형까지의 거리'로 잰다.
        중심까지의 거리로 재면 자리 크기를 무시하게 되어, 자리에 제대로 세워도
        중심에서 그만큼 떨어져 있으면 도착으로 안 잡힌다.
        (x, y)만 주면 예전처럼 중심까지의 거리로 잰다.
        """
        # 정지 감시 (안전장치용). 도착 판정과 무관하게 항상 갱신한다.
        moved, threshold = self._recent_movement(track_id, now)
        if moved is not None and moved >= threshold:
            self._active_moved_at = now

        target = target_of(self.active_car_id) if target_of else None
        world = self.last_states.get(track_id, {}).get("world")

        if target is None or world is None:
            # 호모그래피가 아직 없거나 목표 구역이 배정되지 않았다.
            # 도착 판정을 할 수 없으므로 안전장치에 맡긴다.
            self._arrived_since = None
            self._check_stuck(now)
            return

        distance = self._distance_to_target(world, target)
        if distance > self.single['ARRIVAL_RADIUS_CM']:
            self._arrived_since = None      # 반경을 벗어났으면 처음부터 다시
            self._check_stuck(now)
            return

        if self._arrived_since is None:
            self._arrived_since = now
        elif now - self._arrived_since >= self.single['ARRIVAL_HOLD_SEC']:
            self._release_active(f"목표 구역 {distance:.1f}cm 이내 도착")

    def _check_stuck(self, now):
        """
        도착 판정이 되지 않는 채로 오래 정지해 있으면 강제로 해제한다.

        차가 목표 반경에 살짝 못 미친 곳에 멈춰버리거나 호모그래피가 잡히지
        않으면, 이 장치가 없을 때 활성 차량이 영영 해제되지 않아 다음 차가
        번호를 받지 못한다. SINGLE_ACTIVE['STUCK_RELEASE_SEC']를 0으로 두면
        비활성화되고 오로지 도착 판정으로만 해제된다.
        """
        limit = self.single.get('STUCK_RELEASE_SEC', 0)
        if not limit or self._active_moved_at is None:
            return
        stopped_for = now - self._active_moved_at
        if stopped_for >= limit:
            self._release_active(f"{stopped_for:.0f}초간 정지 (도착 판정 실패)")

    def _release_active(self, reason):
        """
        활성 차량을 해제한다. 다음 차가 FIFO에서 번호를 받을 수 있게 된다.

        주차를 마친 차의 track_to_car 결합은 그대로 남겨둔다. 그 차가 화면에
        계속 보여야 하고, 트랙이 끊겼다 되살아나는 경우는 재바인딩 레이어가
        위치로 이어받기 때문이다.

        on_parked 콜백으로 '주차 완료'를 알린다. A01이 cars_info의 parked를
        True로 바꾸면, 이후 그 차는 _bind_parked_cars의 대상이 되어 트랙이
        끊겨도 자리 위치로 다시 묶인다.
        """
        car_id = self.active_car_id
        print(f"[활성] 차량번호 '{car_id}' 안내 종료 - {reason}")
        self.active_car_id = None
        self._arrived_since = None
        self._active_moved_at = None

        if car_id is not None and self.on_parked is not None:
            self.on_parked(car_id)

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

            reachable, cost = self._reachable(state, ghost, gap)
            if not reachable:
                continue

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
        for store in (self.trajectories, self.world_trajectories):
            old_traj = store.pop(ghost_id, None)
            if not old_traj:
                continue
            merged = deque(old_traj, maxlen=self.trajectory_maxlen)
            merged.extend(store.get(new_id, ()))
            store[new_id] = merged

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
            self.world_trajectories.pop(track_id, None)
            self.first_world.pop(track_id, None)
            self.last_states.pop(track_id, None)
            self.rebound_at.pop(track_id, None)

            if car_id:
                # 유령은 여기서 지우지 않는다. 추적기가 트랙을 버린 뒤부터가
                # 오히려 재바인딩이 필요한 구간이다. (만료는 _expire_ghosts가 담당)
                held = " (유령 보존 중)" if track_id in self.ghosts else ""
                print(f"[소실] Track ID {track_id} (차량번호 '{car_id}') 추적 종료{held}")

        if self.rebind_enable:
            self._expire_ghosts(now)


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
            "active_car_id": self.active_car_id,   # 지금 안내 중인 차 (없으면 None)
        }
        if expected_cars is not None:
            switches = max(0, len(self.seen_track_ids) - expected_cars)
            stats["expected_cars"] = expected_cars
            stats["id_switches"] = switches
            stats["uncovered"] = max(0, switches - self.rebind_count)
        return stats

    def draw_tracks(self, frame, tracks, draw_trajectory=None):
        """
        추적 결과를 프레임에 Bounding Box, Track ID, 차량번호로 시각화.

        Args:
            frame:           OpenCV BGR 이미지 (numpy array, 원본이 수정됨)
            tracks:          update() 메서드의 반환 결과 리스트
            draw_trajectory: 이동 궤적 선 표시 여부.
                             None이면 CONFIG['DRAW_TRAJECTORY']를 따른다.

        Returns:
            시각화가 적용된 프레임 (입력 frame과 동일 객체)
        """
        if draw_trajectory is None:
            draw_trajectory = CONFIG['DRAW_TRAJECTORY']

        for trk in tracks:
            track_id = trk["track_id"]
            car_id = trk["car_id"]
            x1, y1, x2, y2 = trk["bbox"]

            # 번호 매칭 여부에 따라 색상 구분
            color = COLOR_MATCHED if car_id else COLOR_PENDING
            label = f"ID:{track_id} {car_id}" if car_id else f"ID:{track_id} WAIT"

            # 지금 안내 중인 차(활성 차량)를 노랑으로 구분한다.
            # 주차를 마친 차들(초록)과 한눈에 구분되어야 시나리오 진행 상황을
            # 영상만 보고 따라갈 수 있다.
            if car_id is not None and car_id == self.active_car_id:
                color = COLOR_ACTIVE
                label = f"ID:{track_id} {car_id} ACTIVE"

            # 방금 번호를 승계받은 트랙은 잠시 자홍색으로 강조한다.
            # 승계가 실제로 도는지, 엉뚱한 차에 붙지는 않는지를
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
# 카메라 + 검출 + 추적 통합 실행은 B_main.py / C_main.py를 사용할 것.
if __name__ == '__main__':
    print("=" * 62)
    print(" B02 : 차량 다중 객체 추적 (MOT) - 단일 활성 차량 모드")
    print(" 단독 테스트 : 합성 검출 데이터로 로직 검증")
    print(f" 추적기 설정 : {CONFIG['TRACKER_CFG']}")
    print("=" * 62)

    # 합성 좌표계: 10px = 1cm. C00 호모그래피 대신 쓰는 가짜 변환.
    PX_PER_CM = 10.0

    def to_world(point):
        return point[0] / PX_PER_CM, point[1] / PX_PER_CM

    def det(cx, cy):
        """중심이 (cx, cy)인 합성 검출 하나. (B01의 detect() 반환 형식)"""
        return {"class_id": 0, "class_name": "Car",
                "bbox": [cx - 20, cy - 15, cx + 20, cy + 15], "confidence": 0.9}

    # 목표 구역 실좌표 (cm). C00의 get_target_world 대신 쓰는 가짜 목표.
    TARGETS = {"1234": (52.0, 30.0), "5678": (20.0, 30.0)}

    # 이 루프는 실시간이 아니라 즉시 돌기 때문에 '초' 단위 조건이 의미를 갖지
    # 못한다. 시간 조건만 무력화하고 거리/이동 조건은 실제 설정을 그대로 쓴다.
    TEST_SINGLE = dict(CONFIG['SINGLE_ACTIVE'],
                       ARRIVAL_HOLD_SEC=0.0,     # 반경에 들어오면 다음 프레임에 확정
                       STUCK_RELEASE_SEC=0.0)    # 정지 기반 안전장치는 끄고 도착만 본다

    def run(script, frames, label, quiet=True):
        """
        script(frame_idx) -> [(cx, cy), ...] 형태의 시나리오를 돌린다.

        Returns:
            (CarMOT, CarNumberFIFO, 로그 문자열)
        """
        fifo = CarNumberFIFO()
        for num in ("1234", "5678", "9012"):
            fifo.push(num)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            mot = CarMOT(
                tracker_cfg=CONFIG['TRACKER_CFG'],
                min_hits=CONFIG['MIN_HITS_FOR_ASSIGN'],
                trajectory_maxlen=CONFIG['TRAJECTORY_MAXLEN'],
                fifo=fifo,
                single_active=TEST_SINGLE
            )
            for f in range(frames):
                dets = [det(cx, cy) for cx, cy in script(f)]
                mot.update(dets, to_world=to_world, target_of=TARGETS.get)

        print(f"\n[{label}]")
        for line in buf.getvalue().splitlines():
            if any(tag in line for tag in ("[활성]", "[재바인딩]", "[매칭]", "[소실]")):
                print("   ", line)
        print(f"    최종 매핑     : {mot.track_to_car}")
        print(f"    FIFO 잔여     : {fifo.snapshot()}")
        print(f"    활성 차량     : {mot.active_car_id}")
        return mot, fifo

    # -------------------------------------------------------------------
    # 시나리오 1 : 기본 흐름 - 한 대씩 들어와 주차하고 다음 차로 넘어간다
    #
    #   차 A : frame 0~59  (100 -> 520px) 이동 후 정지. 목표 (52.0, 30.0)cm
    #          = (520, 300)px 에 도착하므로 주차 완료 판정을 받아야 한다.
    #   차 B : frame 130~  등장해 이동. A가 해제된 뒤이므로 '5678'을 받아야 한다.
    # -------------------------------------------------------------------
    def scenario_basic(f):
        cars = []
        # 차 A: 60프레임 동안 7px씩 전진, 이후 목표 지점에 정지
        cars.append((min(100 + f * 7, 520), 300))
        # 차 B: frame 130부터 다른 통로에서 등장해 이동
        if f >= 130:
            cars.append((min(100 + (f - 130) * 7, 200), 500))
        return cars

    mot1, fifo1 = run(scenario_basic, 220, "시나리오 1 : 기본 흐름 (A 주차 -> B 안내)")
    ok1 = (sorted(mot1.track_to_car.values()) == ["1234", "5678"]
           and fifo1.snapshot() == ["9012"])
    print(f"    -> 순서대로 1234, 5678이 부여되고 9012는 남았는가 : "
          f"{'OK' if ok1 else 'FAIL'}")

    # -------------------------------------------------------------------
    # 시나리오 2 : 이동 중 추적이 끊겨도 같은 번호로 즉시 재결합
    #
    #   frame 50~119 완전 소실(lost_ttl 60 초과 -> 트랙 폐기, 새 ID 발급).
    #   목표에 도착하기 '전'에 끊기므로 활성 차량은 유지된 상태다.
    #   다시 움직이는 트랙이 나타나면 FIFO 소비 없이 '1234'가 다시 붙어야 한다.
    # -------------------------------------------------------------------
    def scenario_broken(f):
        if 50 <= f < 120:
            return []
        cx = 100 + f * 3 if f < 50 else 100 + (f - 70) * 3
        return [(min(cx, 500), 300)]

    mot2, fifo2 = run(scenario_broken, 200, "시나리오 2 : 추적 끊김 -> 즉시 재결합")
    ok2 = (list(mot2.track_to_car.values()) == ["1234"]
           and fifo2.snapshot() == ["5678", "9012"])
    print(f"    -> FIFO 소비 없이 '1234'를 되찾았는가 : {'OK' if ok2 else 'FAIL'}")

    # -------------------------------------------------------------------
    # 시나리오 3 : 주차를 마친 차는 번호를 뺏기지 않는다
    #
    #   A가 주차를 마친 뒤 B가 A의 바로 옆을 지나간다.
    #   B는 '5678'을 새로 받아야 하고, A는 '1234'를 그대로 유지해야 한다.
    # -------------------------------------------------------------------
    def scenario_passby(f):
        cars = [(min(100 + f * 7, 520), 300)]          # 차 A (주차 완료 후 정지)
        if f >= 130:
            cars.append((min(120 + (f - 130) * 7, 480), 340))   # 차 B가 옆을 통과
        return cars

    mot3, fifo3 = run(scenario_passby, 220, "시나리오 3 : 주차 차량 옆 통과")
    ok3 = (sorted(mot3.track_to_car.values()) == ["1234", "5678"]
           and fifo3.snapshot() == ["9012"])
    print(f"    -> A는 1234 유지, B는 5678 신규 취득인가 : {'OK' if ok3 else 'FAIL'}")

    # -------------------------------------------------------------------
    # 시나리오 4 : 유령(위치 매칭)이 못 잡는 구간을 활성 차량 규칙이 잡는다
    #
    #   오래 가려진 사이 차가 멀리(80cm) 이동해 버린 경우.
    #   REBIND['MAX_REACH_CM'](60cm)를 넘으므로 위치 매칭은 실패해야 정상이다.
    #   그래도 '지금 움직이는 차는 안내 중인 그 차'이므로 번호를 되찾아야 한다.
    #
    #   이 시나리오가 SINGLE_ACTIVE의 존재 이유다. 규칙을 끄면 엉뚱하게 FIFO의
    #   다음 번호('5678')를 소비해 차량번호가 어긋난다.
    # -------------------------------------------------------------------
    def scenario_far_return(f):
        if 50 <= f < 120:
            return []                                   # 완전 소실
        if f < 50:
            return [(100 + f * 3, 300)]                 # 10 -> 25cm 부근
        return [(min(1050 + (f - 120) * 3, 1200), 300)] # 105cm 지점에서 재등장

    mot4, fifo4 = run(scenario_far_return, 200,
                      "시나리오 4 : 먼 곳에서 재등장 (유령 도달 반경 밖)")
    ok4 = (list(mot4.track_to_car.values()) == ["1234"]
           and "5678" in fifo4.snapshot())
    print(f"    -> 위치 매칭이 실패해도 '1234'를 되찾았는가 : {'OK' if ok4 else 'FAIL'}")
    print("       (SINGLE_ACTIVE를 끄면 여기서 '5678'을 잘못 소비한다)")

    print("\n" + "=" * 62)
    print(f" 결과 : {'전부 OK' if all((ok1, ok2, ok3, ok4)) else '실패 항목 있음'}")
    print("=" * 62)


