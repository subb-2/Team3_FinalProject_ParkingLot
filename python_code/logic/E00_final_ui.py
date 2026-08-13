"""
E00_final_ui : 최종 통합 관제 화면

위에 1번 구간을 가로 띠로 깔고, 그 아래를 2·3·4번이 똑같이 3등분한다.

    +------------------------------------------------------------------+
    | 1) Zybo 수신 & 차량번호 FIFO                                        |
    |   [수신] -push-> [FIFO 대기열 front→back] -pop-> [안내 중] [주차완료]  |
    +--------------------+-------------------+-------------------------+
    | 2) 주차장 CCTV       | 3) 주차장 상태      | 4) 실시간 주차 안내        |
    |                    |                   |                         |
    |  카메라 원본 +       |  전체 자리 현황 +    |  차량 시점 내비게이션       |
    |  검출/추적 오버레이   |  일방통행 화살표 +   |  (D00 재사용)            |
    |  (B01/B02)         |  안내 경로(파란 선)  |                         |
    +--------------------+-------------------+-------------------------+

1번을 가로 띠로 올린 이유는 이 구간이 보여주는 것이 '한 대의 차가 수신에서
안내, 주차 완료까지 지나가는 흐름'이기 때문이다. B02의 CarNumberFIFO가 실제로
하는 일(뒤로 쌓고 앞에서 꺼낸다)을 왼쪽에서 오른쪽으로 그대로 늘어놓는다.
숫자 하나('대기 3대')로 줄이면 어느 차가 다음 차례인지가 화면에서 사라진다.

3번의 안내 경로는 4번이 그리는 것과 같은 경로다. 4번은 차 기준으로 돌려 원근을
입힐 뿐이라, 일방통행을 지키는지는 화살표와 같은 평면인 3번에서 확인한다.

왜 OpenCV 캔버스가 아니라 HTML인가
  1번과 3번 구간은 한글 안내문이 핵심이다. OpenCV의 putText는 한글을 못 그린다.
  (D00의 MANEUVER_LABEL이 전부 영문인 것도 그 때문이다)
  3, 4번만 영상이므로 그 둘은 MJPEG로 넣고 나머지는 브라우저가 그리게 했다.
  깜빡이는 점도 CSS 애니메이션이 프레임을 새로 그리는 것보다 싸다.

이 모듈이 하는 일
  - 수신 이벤트 수집    : A00의 리스너로 등록해 받은 값을 쌓아둔다
  - 주차 완료 감시      : cars_info의 parked 전이를 보고 안내문을 만든다
  - 오주차 판정과 정정  : 실제 위치가 배정된 자리와 다르면 기록을 실물에 맞춘다
  - 상태 JSON 제공      : 브라우저가 0.5초마다 가져간다
  - 페이지 HTML 제공

실행은 E_main_final.py가 한다. 이 파일은 화면과 상태만 담당한다.
"""

import sys
import os
import threading
from collections import deque
from datetime import datetime

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.map_data import (
    grid_map, coord_to_spot, PILL_MARKER_ID, spot_type as SPOT_TYPE_OF,
    SPOT_TYPE_NAME, SPOT_CELLS, ROAD, PILL, GATE1, GATE2,
    SPOT1, SPOT2, SPOT3, SPOT4, get_rows, get_cols, one_way_segments,
)
from data.car_data import cars_info, get_car_type
from logic.B02_car_mot import CONFIG as B02_CONFIG
# 영상 위 상태 글자 on/off는 C_main이 들고 있다. 화면은 그 값을 읽기만 한다.
from logic.C_main import CONFIG as C_MAIN_CONFIG
from logic.C02_lot_layout import CONFIG as C02_CONFIG, SPOT_WORLD_POS


# 설정 (Configuration)
CONFIG = {
    # 브라우저가 상태를 가져가는 주기 (ms). 짧을수록 반응이 빠르고 부하가 는다.
    "POLL_INTERVAL_MS": 400,

    # 서버 쪽에서 주차 완료 여부를 확인하는 주기 (초).
    # 폴링보다 촘촘해야 화면이 갱신되기 전에 판정이 끝나 있다.
    "WATCH_INTERVAL_SEC": 0.2,

    # 수신 목록과 안내문 로그에 남길 최대 개수
    "RX_LOG_MAX": 60,
    "EVENT_LOG_MAX": 40,

    # 3번 구간 격자를 어떤 가로세로비로 그릴지.
    #
    #   "grid" : 칸을 정사각형으로 그린다. 열(13)이 행(9)보다 많으므로
    #            가로가 긴 화면이 된다. 실제 매트가 가로로 긴 형태라
    #            눈으로 보는 배치와 맞다. (기본값)
    #   "cm"   : C02의 CELL_W_CM / CELL_H_CM 비율 그대로 그린다.
    #            세로 칸 크기(CELL_H_CM)를 실측해서 넣었다면 이쪽이 정확하다.
    #            지금 값(17.5)은 미실측 추정치라 화면이 세로로 길쭉해진다.
    #   숫자    : 가로/세로 비를 직접 지정 (예: 1.8)
    "LOT_ASPECT": "grid",

    # --- 오주차 판정 ---------------------------------------------------
    # 배정된 자리에서 이만큼 넘게 떨어져 멈추면 '다른 곳에 주차했다'고 본다.
    #
    # B02의 도착 판정 반경과 같은 값을 쓴다. B02는 이 반경 안에 들어와야
    # '도착'으로 안내를 끝내므로, 그보다 멀리서 끝났다면 도착이 아니라
    # 안전장치(STUCK_RELEASE_SEC)가 끊은 것이다. 즉 엉뚱한 데 세운 것이다.
    # 두 값을 따로 두면 어느 쪽도 아닌 회색지대가 생긴다.
    "MISPARK_TOLERANCE_CM": B02_CONFIG['SINGLE_ACTIVE']['ARRIVAL_RADIUS_CM'],

    # 실제 위치에서 이 거리 안에 자리가 있어야 '그 자리에 주차했다'고 인정한다.
    #
    # 이 목업은 폭이 120cm뿐이라 자리가 촘촘하다. 통로 한가운데(60,60)에서도
    # 가장 가까운 자리까지 5cm밖에 안 된다. 값을 키우면 통로에 멈춘 차까지
    # 아무 자리에나 배정해 버린다. 그래서 도착 반경과 같은 수준으로 좁게 둔다.
    # 어느 자리도 이 안에 없으면 자리를 바꾸지 않고 경고만 띄운다.
    "MISPARK_MAX_DIST_CM": B02_CONFIG['SINGLE_ACTIVE']['ARRIVAL_RADIUS_CM'],
}


# 구역 종류별 색 (CSS). D00의 COLOR_SPOT_BY_TYPE와 같은 구분을 쓴다.
SPOT_TYPE_CSS = {
    SPOT1: "#8a8a8a",   # 일반   - 회색
    SPOT2: "#2f8fff",   # 장애인 - 파랑
    SPOT3: "#ffc83c",   # 대형   - 노랑
    SPOT4: "#78dc78",   # 전기차 - 초록
}

# 격자 칸 종류 -> 브라우저가 쓸 짧은 이름
CELL_KIND = {
    ROAD: "road",
    PILL: "pill",
    GATE1: "gate1",
    GATE2: "gate2",
}


# 안내문
# 사용자가 지정한 문구를 그대로 상수로 둔다. 화면에 나가는 말이 코드
# 여기저기에 흩어져 있으면 문구를 바꿀 때 빠뜨리는 곳이 생긴다.
MSG_PARKED_OK = ["지정 자리 주차 완료", "주차장 자리 정보 업데이트"]
MSG_MISPARKED = ["운전자가 다른 위치에 주차하였습니다",
                 "주차 자리 파악 후 해당 자리를 업데이트 합니다"]
MSG_MISPARK_UNKNOWN = ["운전자가 다른 위치에 주차하였습니다",
                       "주차 자리를 파악하지 못했습니다. 확인이 필요합니다"]


# 수신 값 수집 (1번 구간)
class RxFeed:
    """
    Zybo에서 받은 값을 쌓아두는 곳.

    A00_uart_rx가 수신할 때마다 on_rx를 부른다. 화면은 snapshot을 가져간다.
    수신 스레드와 웹 요청 스레드가 동시에 건드리므로 잠금이 필요하다.
    """

    def __init__(self, maxlen=None):
        self._items = deque(maxlen=maxlen or CONFIG['RX_LOG_MAX'])
        self._lock = threading.Lock()
        self._total = {"entry": 0, "exit": 0}

    def on_rx(self, event):
        """A00_uart_rx.add_rx_listener에 등록할 콜백."""
        result = event.get("result") or {}
        role = event.get("role", "entry")

        item = {
            "time": event["time"].strftime("%H:%M:%S"),
            "role": role,
            "role_label": "입차" if role == "entry" else "출차",
            "car_id": event["car_id"],
            "raw_hex": event.get("raw_hex", ""),
            "car_type": result.get("car_type") or get_car_type(event["car_id"]),
            "spot_id": result.get("spot_id"),
            "ok": result.get("success", True),
            "message": result.get("message", ""),
            # 출차일 때만 채워진다 (A01.remove_car의 계산 결과)
            "fee": result.get("fee"),
            "minutes": result.get("minutes"),
            "source": event.get("source", "wifi"),
        }

        with self._lock:
            self._items.appendleft(item)
            if role in self._total:
                self._total[role] += 1

    def push_manual(self, car_id, result=None, role="entry"):
        """
        Zybo 없이 수동으로 등록했을 때도 화면에 남긴다.

        시연이나 점검에서 /enqueue로 넣는 경우다. 화면에 아무것도 안 뜨면
        등록이 됐는지 알 수 없어 확인이 어렵다. 실제 수신과 구분되도록
        source를 'manual'로 표시한다.
        """
        self.on_rx({
            "role": role,
            "car_id": car_id,
            "raw_hex": "-",
            "time": datetime.now(),
            "result": result,
            "source": "manual",
        })

    def snapshot(self):
        with self._lock:
            return list(self._items), dict(self._total)


# 주차 완료 / 오주차 감시 (3번 구간)
class ParkingWatcher:
    """
    차량이 실제로 어디에 섰는지 보고 안내문과 기록을 정리한다.

    B02는 두 가지 경우에 안내를 끝낸다.
      1. 목표 구역 반경 안에 들어와 머물렀다  -> 제대로 주차
      2. 반경 밖에서 오래 멈춰 있었다          -> 엉뚱한 데 주차

    두 경우 모두 A01.mark_parked가 불려 cars_info의 parked가 True가 된다.
    콜백만으로는 둘을 구분할 수 없으므로, 여기서 '차가 실제로 있던 좌표'를
    따로 들고 있다가 전이 시점에 배정된 자리와 비교한다.

    좌표를 미리 캐시해 두는 이유는 전이를 감지한 순간에는 이미 안내가 끝나
    latest_nav에서 그 차가 빠져 있을 수 있기 때문이다. 매 tick마다 마지막
    위치를 갱신해 두면 그 문제가 없다.
    """

    def __init__(self, rx_feed=None):
        self.rx_feed = rx_feed
        self._lock = threading.Lock()

        self._last_world = {}      # {차량번호: (x_cm, y_cm)} 마지막으로 본 위치
        self._parked_seen = {}     # {차량번호: bool} 직전 tick의 parked 값
        self._events = deque(maxlen=CONFIG['EVENT_LOG_MAX'])
        self._latest = None        # 화면 하단에 크게 띄울 가장 최근 안내

        self.prime()

    def prime(self):
        """
        지금 이미 주차되어 있는 차를 '본 것'으로 표시해 둔다.

        이게 없으면 기동 직후 첫 poll에서 미리 세워둔 차(car_data의
        INITIAL_PARKED) 전부가 False -> True 전이로 잡혀 "지정 자리 주차 완료"가
        한꺼번에 쏟아진다. 그 차들은 시스템이 켜지기 전부터 서 있던 것이라
        방금 주차한 것이 아니다.

        E_main_final이 감시 스레드를 띄우기 직전에 한 번 더 부른다.
        생성 시점과 감시 시작 시점 사이에 입차가 들어와도 어긋나지 않게 하기
        위해서다.
        """
        for car_id, info in cars_info.items():
            self._parked_seen[car_id] = bool(info.get("parked"))

    # --- 감시 루프 -------------------------------------------------------
    def poll(self, pipeline):
        """
        한 tick 분량의 감시. E_main_final의 감시 스레드가 반복 호출한다.

        Args:
            pipeline: ParkingNavigationPipeline (latest_nav를 읽는다)
        """
        # 1) 보이는 차량의 위치를 갱신해 둔다
        #
        #    픽셀 모드에서는 world_pos가 cm가 아니라 이미지 픽셀이다
        #    (C00_navigation.update의 is_pixel_mode 분기). 아래 판정은 전부
        #    cm 기준이므로 여기서 한 번 cm로 옮겨 두어야 한다. 그러지 않으면
        #    배정 자리에 제대로 세운 차도 수백 cm 떨어진 것으로 계산되어
        #    "운전자가 다른 위치에 주차하였습니다"가 뜬다.
        mapper = pipeline.navigator.mapper
        for nav in pipeline.latest_nav:
            car_id = nav.get("car_id")
            pos = nav.get("world_pos")
            if not car_id or pos is None:
                continue
            pos = mapper.pixel_to_cm(pos) or pos
            with self._lock:
                self._last_world[car_id] = (pos[0], pos[1])

        # 2) parked가 False -> True로 바뀐 차를 찾는다
        for car_id, info in list(cars_info.items()):
            parked = bool(info.get("parked"))
            was_parked = self._parked_seen.get(car_id, False)
            if parked and not was_parked:
                self._handle_parked(car_id, info)
            self._parked_seen[car_id] = parked

        # 3) 출차한 차는 기억에서 지운다
        for car_id in list(self._parked_seen):
            if car_id not in cars_info:
                del self._parked_seen[car_id]
                with self._lock:
                    self._last_world.pop(car_id, None)

    def _handle_parked(self, car_id, info):
        """주차 완료 전이를 처리한다. 정상인지 오주차인지 판정하고 기록한다."""
        from logic.A01_parking_manager import (
            distance_to_spot, find_nearest_spot, relocate_car,
        )

        assigned = info.get("spot_id")
        with self._lock:
            world = self._last_world.get(car_id)

        # 위치를 한 번도 못 봤다. 호모그래피가 없거나 검출이 안 된 경우다.
        # 확인할 방법이 없으므로 안내대로 세웠다고 보고 그 사실을 남긴다.
        if world is None:
            self._emit("parked_ok", car_id, assigned, MSG_PARKED_OK,
                       note="차량 위치를 확인하지 못해 배정 자리 기준으로 기록했습니다.")
            return

        distance = distance_to_spot(world, assigned) if assigned else float('inf')

        # 배정된 자리에 제대로 섰다
        if distance <= CONFIG['MISPARK_TOLERANCE_CM']:
            self._emit("parked_ok", car_id, assigned, MSG_PARKED_OK,
                       world=world, distance=distance)
            return

        # 배정된 자리가 아니다. 실제로 어느 자리인지 찾는다.
        actual, actual_dist = find_nearest_spot(
            world,
            exclude={assigned} if assigned else None,
            max_distance_cm=CONFIG['MISPARK_MAX_DIST_CM'],
        )

        # 어느 자리로도 보기 어려운 위치(통로 한가운데 등)
        if actual is None:
            self._emit("mispark_unknown", car_id, assigned, MSG_MISPARK_UNKNOWN,
                       world=world, distance=distance,
                       note=f"가장 가까운 자리도 {actual_dist:.0f}cm 떨어져 있습니다.")
            return

        # 기록을 실제 자리로 옮긴다
        result = relocate_car(car_id, actual)
        if not result["success"]:
            self._emit("relocate_failed", car_id, assigned, MSG_MISPARK_UNKNOWN,
                       world=world, distance=distance, note=result["message"])
            return

        note = f"{assigned} 안내 -> 실제 {actual}에 주차 (배정 자리에서 {distance:.0f}cm)"
        if result["type_mismatch"]:
            note += f" [경고: {get_car_type(car_id)} 차량인데 " \
                    f"{SPOT_TYPE_NAME.get(SPOT_TYPE_OF.get(actual), '?')} 구역입니다]"

        self._emit("misparked", car_id, actual, MSG_MISPARKED,
                   world=world, distance=distance, note=note,
                   assigned_spot=assigned)

    def _emit(self, kind, car_id, spot_id, lines, world=None, distance=None,
              note=None, assigned_spot=None):
        """안내문 하나를 기록하고 화면에 띄울 최신 항목으로 삼는다."""
        event = {
            "kind": kind,
            "time": datetime.now().strftime("%H:%M:%S"),
            "car_id": car_id,
            "spot_id": spot_id,
            "assigned_spot": assigned_spot,
            "lines": list(lines),
            "note": note,
            "world": [round(world[0], 1), round(world[1], 1)] if world else None,
            "distance_cm": round(distance, 1) if distance is not None else None,
        }
        with self._lock:
            self._events.appendleft(event)
            self._latest = event

        head = " / ".join(lines)
        print(f"[화면안내] {car_id} {spot_id or '-'} : {head}"
              + (f"  ({note})" if note else ""))

    def snapshot(self):
        with self._lock:
            return list(self._events), self._latest


# 주차장 배치 (브라우저가 격자를 그릴 때 한 번만 가져간다)
def lot_aspect(rows, cols):
    """
    3번 구간 격자의 가로/세로 비를 CONFIG['LOT_ASPECT']에 따라 계산.

    1보다 크면 가로가 긴 화면이다. 자세한 설명은 CONFIG의 주석 참고.
    """
    mode = CONFIG['LOT_ASPECT']
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return float(mode)
    if mode == "cm":
        return ((cols * C02_CONFIG['CELL_W_CM']) /
                (rows * C02_CONFIG['CELL_H_CM']))
    return cols / max(rows, 1)


def build_lot_layout():
    """
    격자 배치를 브라우저가 그릴 수 있는 형태로 변환.

    자리 상태처럼 매번 바뀌는 값은 넣지 않는다. 이건 한 번만 가져가고,
    변하는 것은 build_ui_state가 준다.

    Returns:
        {"rows", "cols", "cells": [[...]], "legend": {...}}
    """
    rows, cols = get_rows(), get_cols()
    cells = []

    for r in range(rows):
        row_out = []
        for c in range(cols):
            cell_type = grid_map[r][c]
            entry = {"kind": CELL_KIND.get(cell_type, "road")}

            if cell_type in SPOT_CELLS:
                entry["kind"] = "spot"
                entry["spot"] = coord_to_spot.get((r, c))
                entry["type_name"] = SPOT_TYPE_NAME.get(cell_type, "?")
                entry["color"] = SPOT_TYPE_CSS.get(cell_type, "#8a8a8a")
            elif cell_type == PILL:
                entry["marker"] = PILL_MARKER_ID.get((r, c))

            row_out.append(entry)
        cells.append(row_out)

    return {
        "rows": rows,
        "cols": cols,
        "cells": cells,
        # 일방통행 순환선. 화면이 격자 위에 화살표로 깔아 준다.
        # 안내 경로가 왜 한 바퀴 도는지를 이것 없이는 설명할 수 없다.
        "one_way": [
            {"r1": a[0], "c1": a[1], "r2": b[0], "c2": b[1]}
            for a, b in one_way_segments()
        ],
        # 격자 한 칸의 실제 크기. 참고용으로 함께 보낸다.
        "cell_w_cm": C02_CONFIG['CELL_W_CM'],
        "cell_h_cm": C02_CONFIG['CELL_H_CM'],
        # 격자 전체의 가로/세로 비. 계산은 여기서 한다. 화면 쪽에 비율을
        # 박아두면 CONFIG를 고쳤을 때 화면만 옛 값으로 남는다.
        "aspect": lot_aspect(rows, cols),
        "legend": [
            {"name": name, "color": SPOT_TYPE_CSS.get(t, "#8a8a8a")}
            for t, name in SPOT_TYPE_NAME.items()
        ],
    }


# 카메라 구간의 트랙 목록
# 영상 위 라벨(B02.draw_tracks가 그리는 "ID:5 1234 ACTIVE")과 같은 정보를
# 글로 뽑는다. 색으로만 구분되는 상태를 한글로 풀어 준다.
TRACK_STATE = {
    "active":  ("안내중",   "active"),   # 지금 목적지로 안내하는 차 (영상에서 노랑)
    "parked":  ("주차완료", "parked"),   # 배정 자리에 세운 차       (영상에서 초록)
    "guided":  ("배정됨",   "guided"),   # 번호는 붙었지만 안내 대상은 아님
    "waiting": ("번호대기", "waiting"),  # 검출은 됐는데 번호 미매칭 (영상에서 주황)
}


def _build_track_items(pipeline):
    """
    현재 추적 중인 차량 목록. 화면 표시용.

    Returns:
        [{"track_id", "car_id", "state", "state_label", "conf", "spot_id"}, ...]
        안내 중인 차를 맨 위로, 그다음 번호가 붙은 차, 미매칭 순으로 정렬한다.
        관제하는 사람이 가장 먼저 볼 것이 지금 움직이는 차이기 때문이다.
    """
    active_car = getattr(pipeline.mot, "active_car_id", None)
    items = []

    for trk in pipeline.latest_tracks:
        car_id = trk.get("car_id")
        info = cars_info.get(car_id) if car_id else None

        if car_id and car_id == active_car:
            state = "active"
        elif info and info.get("parked"):
            state = "parked"
        elif car_id:
            state = "guided"
        else:
            state = "waiting"

        label, css = TRACK_STATE[state]
        items.append({
            "track_id": trk.get("track_id"),
            "car_id": car_id,
            "state": css,
            "state_label": label,
            "conf": round(trk.get("confidence") or 0.0, 2),
            "spot_id": info.get("spot_id") if info else None,
        })

    # 추적이 끊긴 주차 차량도 목록에 남긴다.
    #
    # 세워둔 차가 잠깐 안 잡혔다고 목록에서 사라지면, 차가 없어진 것인지
    # 검출만 놓친 것인지 알 수 없다. 자리에 있다는 것은 배정 기록이 알고
    # 있으므로 그대로 두고, 추적이 없다는 사실만 track_id를 비워 표시한다.
    seen = {t["car_id"] for t in items if t["car_id"]}
    for car_id, info in cars_info.items():
        if car_id in seen or not info.get("parked"):
            continue
        label, css = TRACK_STATE["parked"]
        items.append({
            "track_id": None,           # 지금 추적 중이 아니다
            "car_id": car_id,
            "state": css,
            "state_label": label,
            "conf": 0.0,
            "spot_id": info.get("spot_id"),
        })

    order = {"active": 0, "guided": 1, "parked": 2, "waiting": 3}
    items.sort(key=lambda t: (order.get(t["state"], 9), t["track_id"] or 0))
    return items


def world_to_cell(world_cm):
    """
    cm 좌표를 격자 좌표(행, 열)로. 소수까지 그대로 돌려준다.

    3번 구간은 격자를 칸 단위로 그리므로, 차량 점을 찍으려면 cm가 아니라
    '몇 번째 칸의 어디쯤'인지가 필요하다. cell_to_world의 역변환이다.

    Returns:
        (row, col) 실수. (0.0, 0.0)이 ORIGIN_CELL 칸의 중심이다.
    """
    origin_row, origin_col = C02_CONFIG['ORIGIN_CELL']
    return (world_cm[1] / C02_CONFIG['CELL_H_CM'] + origin_row,
            world_cm[0] / C02_CONFIG['CELL_W_CM'] + origin_col)


def _build_cars_on_map(pipeline):
    """
    3번 구간 격자 위에 찍을 차량 점 목록.

    픽셀 모드의 world_pos는 이미지 픽셀이므로 cm로 옮긴 뒤 격자 좌표로 바꾼다.
    (PillarMapper.pixel_to_cm)

    Returns:
        [{"key", "car_id", "row", "col", "parked", "target_spot"}, ...]
    """
    mapper = pipeline.navigator.mapper
    cars = []

    for nav in pipeline.latest_nav:
        pos = nav.get("world_pos")
        if pos is None:
            continue
        world = mapper.pixel_to_cm(pos) or pos
        row, col = world_to_cell(world)

        # 격자 밖에 찍히는 점은 버린다.
        # 바닥에 비친 그림자나 삼각대 같은 오검출이 주차장 바깥에서 잡히면
        # 격자를 벗어난 자리에 점이 찍혀 화면만 어지럽다. 한 칸 정도는
        # 넘어가도 봐준다. 입출구가 격자 맨 아랫줄에 걸쳐 있기 때문이다.
        if not (-1 <= row <= get_rows() and -1 <= col <= get_cols()):
            continue

        car_id = nav.get("car_id")
        info = cars_info.get(car_id) if car_id else None
        cars.append({
            # 번호가 아직 안 붙은 차도 점은 찍는다. 화면에서 사라졌다 나타나면
            # 오히려 헷갈리기 때문이다. 대신 색을 달리한다.
            "key": car_id or f"#{nav.get('track_id')}",
            "car_id": car_id,
            "row": round(row, 3),
            "col": round(col, 3),
            "parked": bool(info and info.get("parked")),
            "target_spot": nav.get("target_spot"),
            "tracked": True,
        })

    # 주차를 마친 차는 추적이 끊겨도 점을 유지한다.
    #
    # 세워둔 차의 위치는 추적이 아니라 배정 기록이 정한다. 이미 그 자리에
    # 있다는 것을 아는데 굳이 매 프레임 다시 찾아낼 이유가 없다. 검출이
    # 한 번 흔들릴 때마다 점이 사라지면 화면만 깜빡인다.
    # (자리를 뜨면 감시 스레드가 parked를 풀어 주므로 점도 같이 사라진다)
    seen = {c["car_id"] for c in cars if c["car_id"]}
    for car_id, info in cars_info.items():
        if car_id in seen or not info.get("parked"):
            continue
        spot_pos = SPOT_WORLD_POS.get(info.get("spot_id"))
        if spot_pos is None:
            continue
        row, col = world_to_cell(spot_pos)
        cars.append({
            "key": car_id,
            "car_id": car_id,
            "row": round(row, 3),
            "col": round(col, 3),
            "parked": True,
            "target_spot": None,
            "tracked": False,       # 배정 기록에서 온 점
        })
    return cars


def _build_fifo_view(pipeline):
    """
    1번 구간이 그릴 FIFO 큐의 현재 모습.

    화면을 로직 그대로 세운다. 왼쪽에서 번호가 들어와(push) 큐에 줄을 서고,
    카메라가 움직이는 차를 찾으면 맨 앞에서 하나 꺼내(pop) 안내가 시작된다.
    큐 내용을 숫자 하나('대기 3대')로만 보여주면 순서가 안 보여서, 어느 차가
    다음 차례인지도 왜 이 차가 먼저 안내받는지도 화면에서 읽을 수 없다.

    Returns:
        {"waiting": [앞에서부터 차량번호], "active": {...}|None}
    """
    fifo = getattr(pipeline.mot, "fifo", None)
    waiting = fifo.snapshot() if fifo is not None else []

    active_id = getattr(pipeline.mot, "active_car_id", None)
    active = None
    if active_id:
        info = cars_info.get(active_id) or {}
        active = {
            "car_id": active_id,
            "spot_id": info.get("spot_id"),
            "car_type": info.get("car_type") or get_car_type(active_id),
        }

    return {
        "waiting": waiting,
        "active": active,
        # 큐에 들어오기를 기다리는 것이 아니라 '이미 꺼내 간' 수. 화면 설명용.
        "size": len(waiting),
    }


def _build_route_on_map(pipeline, follow_car=None):
    """
    3번 구간 격자 위에 그릴 안내 경로.

    4번 구간(차량 시점 3D)이 그리는 것과 '같은 경로, 같은 차'다. 4번은 이
    경로를 차 기준으로 돌려서 원근을 입힐 뿐이므로, 여기 격자에 그려지는
    선이 곧 4번이 안내하는 길이다.

    이 선을 3번에 그리는 이유:
      3D 화면은 차를 따라 돌아가고 원근이 걸려 있어서, 그려진 선이 일방통행을
      지키는지 눈으로 판정할 수가 없다. 격자는 바닥 화살표(one_way)와 같은
      평면에 같은 방향으로 서 있으므로, 두 선을 겹쳐 보면 역주행인지 아닌지가
      바로 보인다.

    좌표는 C00이 계획한 cm 원본(route_cm)을 쓴다. 화면용 픽셀 사본을 다시
    cm로 되돌리면 변환을 왕복하게 되고, 되돌리지 못한 점이 섞이면 선이 튄다.

    Returns:
        {"car_id", "target_spot", "wrong_way", "points": [{"row","col"}, ...]}
        그릴 것이 없으면 None.
    """
    from logic.D00_ui_navi import pick_my_vehicle

    nav = pick_my_vehicle(pipeline.latest_nav, follow_car)
    if nav is None:
        return None

    route_cm = nav.get("route_cm")
    if not route_cm or len(route_cm) < 2:
        return None

    # 남은 구간만 그린다. 이미 지나온 경유점까지 그리면 차 뒤로 선이 남는다.
    idx = min(nav.get("route_index", 1), len(route_cm) - 1)
    points = list(route_cm[idx:])

    # 선은 차의 현재 위치에서 시작해야 한다. 다음 경유점부터 그리면
    # 차와 선이 떨어져 있어 어디로 가라는 것인지 읽히지 않는다.
    pos = nav.get("world_pos")
    if pos is not None:
        start = pipeline.navigator.mapper.pixel_to_cm(pos) or pos
        points.insert(0, start)

    cells = []
    for pt in points:
        row, col = world_to_cell(pt)
        cells.append({"row": round(row, 3), "col": round(col, 3)})

    return {
        "car_id": nav.get("car_id"),
        "target_spot": nav.get("target_spot"),
        # 계획기가 역주행 구간을 물고 나왔다는 표시. 화면이 붉게 그린다.
        # (C00._check_one_way가 판정하고 터미널에도 한 번 남긴다)
        "wrong_way": bool(nav.get("route_wrong_way")),
        "points": cells,
    }


# 화면 상태 (0.5초마다 브라우저가 가져간다)
def build_ui_state(pipeline, rx_feed, watcher, follow_car=None):
    """
    세 구간이 그릴 내용을 한 번에 담아 반환.

    한 번의 요청으로 전부 주는 이유는 세 구간이 같은 시점의 상태를 보여야
    하기 때문이다. 따로 가져가면 자리 상태와 안내문이 어긋난 순간이 화면에
    남는다.

    Args:
        pipeline: ParkingNavigationPipeline
        rx_feed:  RxFeed
        watcher:  ParkingWatcher

    Returns:
        JSON 직렬화 가능한 딕셔너리
    """
    from data.map_data import spot_status
    from logic.A01_parking_manager import get_availability_by_type

    rx_items, rx_total = rx_feed.snapshot()
    events, latest = watcher.snapshot()

    # 배정은 됐지만 아직 도착하지 않은 자리. 여기가 깜빡인다.
    assigned_pending = {}
    for car_id, info in cars_info.items():
        if not info.get("parked") and info.get("spot_id"):
            assigned_pending[info["spot_id"]] = car_id

    # 자리별 상태
    spots = {}
    for spot_id, status in spot_status.items():
        occupant = next((cid for cid, i in cars_info.items()
                         if i.get("spot_id") == spot_id), None)
        spots[spot_id] = {
            "status": status,
            "car_id": occupant,
            # 배정만 된 상태(가는 중)와 실제로 주차를 마친 상태를 구분한다.
            # 둘 다 spot_status는 full이지만 화면에서는 달리 보여야 한다.
            "parked": bool(occupant and cars_info[occupant].get("parked")),
            "pending": spot_id in assigned_pending,
            "type_name": SPOT_TYPE_NAME.get(SPOT_TYPE_OF.get(spot_id), "?"),
        }

    # 보정 상태. 기둥을 찍었으면 좌표계가 선 것이다.
    mapper = pipeline.navigator.mapper
    if mapper.is_ready():
        homography = {"state": "locked",
                      "text": f"기둥 {len(mapper.pillar_pixels)}개"}
    else:
        homography = {"state": "not_ready", "text": "보정 필요"}

    return {
        # --- 1번 구간 ---
        "rx": {"items": rx_items, "total": rx_total},
        "fifo": _build_fifo_view(pipeline),

        # --- 3번 구간 ---
        "spots": spots,
        # 격자 위에 실시간으로 움직이는 차량 점
        "cars_on_map": _build_cars_on_map(pipeline),
        # 4번 구간이 안내하는 것과 같은 경로. 격자 위에 파란 선으로 깐다.
        "route_on_map": _build_route_on_map(pipeline, follow_car),
        "assigned_pending": assigned_pending,
        "availability": {
            info["name"]: {"empty": info["empty"], "total": info["total"]}
            for info in get_availability_by_type().values()
        },
        "latest_event": latest,
        "events": events,

        # --- 2번 구간 + 공통 ---
        "vehicles": [
            {
                "car_id": n.get("car_id"),
                "track_id": n.get("track_id"),
                "target_spot": n.get("target_spot"),
                "distance_cm": (round(n["distance_cm"], 1)
                                if n.get("distance_cm") is not None else None),
                "guide_text": n.get("guide_text"),
                "world": [round(n["world_pos"][0], 1), round(n["world_pos"][1], 1)],
            }
            for n in pipeline.latest_nav
        ],
        # 카메라 구간이 쓰는 값. nav_results가 아니라 추적 결과를 센다.
        # nav_results는 실좌표가 나온 차만 들어 있어, 호모그래피가 없으면
        # 화면에 박스가 보이는데도 0으로 나온다.
        "tracks": {
            # detected = YOLO가 찾은 수, total = 그중 추적 중인 수.
            # 둘이 다르면 추적기가 트랙을 못 만들고 있다는 뜻이다.
            "detected": getattr(pipeline, "latest_detections", None),
            "total": len(pipeline.latest_tracks),
            "matched": sum(1 for t in pipeline.latest_tracks if t.get("car_id")),
            # 영상에 그려진 박스 라벨과 같은 내용을 글로도 준다.
            # 라벨은 화면에 작게 박혀 있어 관제 화면에서 읽기 어렵다.
            "items": _build_track_items(pipeline),
        },
        "stage_ms": pipeline.stage_ms,
        "fps": round(pipeline.fps, 1),
        # 카메라가 주는 fps와 처리가 못 따라가 버린 프레임 수.
        # 예전에는 여기에 검출된 마커 수를 넣었는데 마커 검출을 걷어내면서
        # 늘 비어 있게 됐다. 대신 실제로 봐야 하는 값을 넣는다.
        # camera_fps가 fps보다 크면 그 차이만큼 프레임을 버리고 있다는 뜻이다.
        "camera_fps": round(getattr(pipeline.cap, "capture_fps", 0.0), 1),
        "dropped_frames": getattr(pipeline.cap, "dropped", 0),
        "homography": homography,
        # 영상 위 상태 글자가 켜져 있는지. 버튼이 이 값을 따라간다.
        # 창을 여러 개 열어도 서버 값 하나만 보므로 어긋나지 않는다.
        "debug_overlay": bool(C_MAIN_CONFIG['DRAW_STATUS_TEXT']),
        "time": datetime.now().strftime("%H:%M:%S"),
    }


# 페이지
# __POLL_MS__ 자리에 폴링 주기가 채워진다.
FINAL_UI_HTML = r"""
<html><head><meta charset="utf-8">
<title>주차 관제 - 최종 화면</title>
<style>
 :root{
   --bg:#141416; --panel:#1c1c20; --line:#2e2e34;
   --text:#e8e8ea; --dim:#8a8a93; --accent:#3fa9ff;
   --ok:#4ccf6a; --warn:#ffb03a; --bad:#ff4d4d;
 }
 *{box-sizing:border-box}
 body{margin:0;height:100vh;overflow:hidden;background:var(--bg);color:var(--text);
      font-family:"Malgun Gothic","맑은 고딕","Noto Sans KR",sans-serif}

 /* 위에 1번 가로 띠, 아래에 2·3·4번 3등분.
    +-------------------------------------------------------+
    | 1 수신 -> push -> FIFO 큐 -> pop -> 안내 중 -> 주차 완료  |
    +---------------+---------------+-----------------------+
    | 2 주차장 CCTV  | 3 주차장 상태   | 4 실시간 주차 안내       |
    +---------------+---------------+-----------------------+

    1번을 가로 띠로 올린 이유: 이 구간이 보여주는 것은 '한 대의 차가 수신에서
    안내까지 지나가는 흐름'이라 왼쪽에서 오른쪽으로 읽혀야 한다. 세로 열에
    넣으면 그 흐름이 그냥 목록으로만 보인다.

    아래 셋은 1fr씩 똑같이. 퍼센트로 주면 33%x3 + gap이 100%를 넘어
    오른쪽이 잘린다. */
 #app{display:grid;grid-template-rows:auto 1fr;
      gap:10px;padding:10px;height:100vh}
 #grid{display:grid;grid-template-columns:1fr 1fr 1fr;
       gap:10px;min-height:0;min-width:0}
 .col{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}
 .col > h2{margin:0;padding:11px 14px;font-size:15px;font-weight:600;
           border-bottom:1px solid var(--line);display:flex;
           justify-content:space-between;align-items:center}
 .num{display:inline-block;width:20px;height:20px;line-height:20px;
      text-align:center;border-radius:5px;background:var(--accent);
      color:#06121e;font-size:12px;font-weight:700;margin-right:8px}
 .sub{font-size:11px;color:var(--dim);font-weight:400}
 .body{flex:1;overflow:auto;padding:10px 12px;min-height:0}

 /* ---- 1번 구간 : 수신 -> FIFO -> 안내 (가로 흐름) ----
    로직 순서 그대로 왼쪽에서 오른쪽으로 늘어놓는다.
      수신(A00) -> push -> 대기열(CarNumberFIFO) -> pop -> 안내 중(B02) -> 주차 완료 */
 /* 세로를 넉넉히 준다. 좁으면 대기열 카드와 안내문이 눌려 읽기 어렵다. */
 #pipe{display:flex;align-items:stretch;gap:0;padding:12px;
       min-height:200px;overflow:hidden}
 .stg{background:#232329;border:1px solid var(--line);border-radius:8px;
      padding:10px 13px;display:flex;flex-direction:column;min-width:0;
      justify-content:flex-start}
 .stg > .cap{font-size:11px;color:var(--dim);margin-bottom:8px;white-space:nowrap;
             display:flex;justify-content:space-between;gap:8px;align-items:baseline}
 .stg.rxbox{flex:0 0 230px}
 .stg.queue{flex:1;min-width:0;border-color:#3a5a72;background:#1e242b}
 /* '안내 중'과 '주차 완료'는 글이 길어 좁으면 줄이 잘린다.
    안내문은 두 줄(자리 + 사유)이 다 보여야 한다. */
 .stg.act  {flex:0 0 250px}
 .stg.done {flex:0 0 420px}

 /* 단계 사이 화살표. push / pop이 어느 쪽으로 도는지가 이 화면의 핵심이다. */
 .flowarw{flex:0 0 62px;display:flex;flex-direction:column;align-items:center;
          justify-content:center;color:var(--dim);font-size:10px;gap:2px}
 .flowarw b{font-size:11px;color:var(--accent);font-weight:700}
 .flowarw .ln{width:100%;height:0;border-top:1px solid #3f3f48;position:relative}
 .flowarw .ln::after{content:"";position:absolute;right:0;top:-4px;
                     border-left:7px solid #3f3f48;
                     border-top:4px solid transparent;border-bottom:4px solid transparent}

 /* 대기열 카드. 왼쪽 끝이 front(다음에 매칭될 차)다. */
 .qrow{display:flex;align-items:center;gap:8px;overflow-x:auto;flex:1;min-height:0}
 .qcard{flex:0 0 auto;border:1px solid #3f6f92;border-radius:8px;background:#22303b;
        padding:9px 15px;text-align:center}
 .qcard.front{border-color:var(--accent);background:#1d3d55;
              box-shadow:0 0 0 2px rgba(63,169,255,.22)}
 .qcard .no{font-size:22px;font-weight:700;letter-spacing:2px;
            font-variant-numeric:tabular-nums;line-height:1.25}
 .qcard .mk{font-size:10px;color:var(--accent);letter-spacing:0;margin-top:2px}
 .qempty{color:var(--dim);font-size:13px;padding:16px 4px}

 /* 최근 수신 (가로 띠라 최근 두 건만 보인다) */
 .rx{border:1px solid var(--line);border-left:3px solid var(--accent);
     border-radius:6px;padding:5px 8px;margin-bottom:5px;background:#1e1e23}
 .rx.exit{border-left-color:var(--warn)}
 .rx.fail{border-left-color:var(--bad)}
 .rx-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
 .rx-car{font-size:16px;font-weight:700;letter-spacing:1.5px;font-variant-numeric:tabular-nums}
 .rx-time{font-size:10px;color:var(--dim);font-variant-numeric:tabular-nums}
 .rx-line{font-size:11px;color:var(--dim);margin-top:2px;white-space:nowrap;
          overflow:hidden;text-overflow:ellipsis}
 .rx-spot{color:var(--ok);font-weight:600}
 .rx-fail{color:var(--bad)}
 .empty{color:var(--dim);font-size:12px;text-align:center;padding:14px 8px;line-height:1.6}

 /* 안내 중인 차 */
 .actcar{font-size:26px;font-weight:700;letter-spacing:2px;color:#ffd964;
         font-variant-numeric:tabular-nums;line-height:1.3}
 .actto{font-size:13px;color:var(--dim);margin-top:5px}
 .actto b{color:var(--ok)}

 /* ---- 3번 구간 : 주차장 상태 ---- */
 #lotwrap{flex:1;display:flex;align-items:center;justify-content:center;
          padding:8px;min-height:0;min-width:0}
 /* 크기는 fitLot()이 픽셀로 넣는다.
    aspect-ratio + max-width/height 조합은 폭도 높이도 auto인 flex 자식에서
    0으로 무너진다. 남는 공간을 재서 직접 계산하는 편이 확실하다. */
 #lot{display:grid;gap:2px;position:relative}
 .cell{border-radius:2px;position:relative;min-width:0;min-height:0}

 /* 격자 위를 실시간으로 움직이는 차량 점.
    칸(그리드 아이템)이 아니라 격자 전체를 덮는 한 겹 위에 올린다.
    transition 시간을 폴링 주기에 맞춰 두면 0.4초마다 오는 위치가
    뚝뚝 끊기지 않고 이어져 보인다. */
 /* 일방통행 화살표. 바닥 표시이므로 차량 점보다 아래, 칸보다 위에 깐다. */
 /* 안내 경로. 일방통행 화살표 위, 차량 점 아래에 깐다.
    4번 구간이 그리는 것과 같은 선이므로 색도 그쪽 파랑에 맞춘다. */
 #route{position:absolute;inset:0;pointer-events:none;overflow:visible}
 #route .rt{stroke:#42b2ff;stroke-width:5;fill:none;
            stroke-linejoin:round;stroke-linecap:round;opacity:.9}
 #route .rt.bad{stroke:#ff5a5a}
 #route .goal{fill:#42b2ff}
 #route .goal.bad{fill:#ff5a5a}
 /* 3번 제목줄에 지금 그려진 경로가 누구 것인지 적는다.
    4번과 같은 차를 고르므로 두 화면을 짝지어 읽을 수 있다. */
 #lotroute{color:#42b2ff;margin-right:8px}
 #lotroute.bad{color:#ff5a5a}
 #cars{position:absolute;inset:0;pointer-events:none}
 .car{position:absolute;width:11px;height:11px;margin:-5.5px 0 0 -5.5px;
      border-radius:50%;background:var(--accent);border:2px solid #0b0b0d;
      box-shadow:0 0 0 2px rgba(63,169,255,.35);
      transition:left .4s linear, top .4s linear}
 .car.parked{background:var(--ok);box-shadow:0 0 0 2px rgba(76,207,106,.3)}
 .car.unknown{background:var(--warn);box-shadow:0 0 0 2px rgba(255,176,58,.3)}
 .car > span{position:absolute;left:13px;top:-5px;font-size:10px;font-weight:700;
             color:var(--text);text-shadow:0 0 4px #000,0 0 4px #000;
             white-space:nowrap;font-variant-numeric:tabular-nums}
 .cell.road{background:#212125}
 .cell.gate1,.cell.gate2{background:#2b3b2b;border:1px solid #4f7a4f}
 .cell.pill{background:#3a3f49;border:1px solid #58606f}
 .cell.gate1::after,.cell.gate2::after,.cell.pill::after{
   content:attr(data-lbl);position:absolute;inset:0;display:flex;
   align-items:center;justify-content:center;font-size:9px;color:#9aa6b8}
 .cell.spot{border:1.5px solid var(--c,#8a8a8a);background:#1e1e22;
            display:flex;align-items:center;justify-content:center;
            font-size:clamp(6px,1.3vh,11px);color:var(--c,#8a8a8a);
            font-weight:600;overflow:hidden}
 .cell.spot.full{background:rgba(255,77,77,.20);border-color:#ff6b6b;color:#ffbcbc}
 .cell.spot.pending{border-color:var(--bad);border-width:2px}
 /* 배정된 자리에서 깜빡이는 빨간 점 */
 .dot{position:absolute;top:50%;left:50%;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;
      border-radius:50%;background:var(--bad);animation:blink 1s infinite}
 @keyframes blink{
   0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(255,77,77,.75)}
   50%    {opacity:.25;box-shadow:0 0 0 7px rgba(255,77,77,0)}
 }
 /* 여러 칸짜리 자리(대형)는 위쪽 칸에만 이름을 쓴다 */
 .cell.spot.cont{border-top:none;border-top-left-radius:0;border-top-right-radius:0}
 .cell.spot.head{border-bottom:none;border-bottom-left-radius:0;border-bottom-right-radius:0}

 #avail{display:flex;gap:6px;flex-wrap:wrap;padding:0 12px 8px}
 .av{flex:1;min-width:62px;background:#232329;border:1px solid var(--line);
     border-radius:6px;padding:6px 8px;text-align:center}
 .av .n{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums}
 .av .l{font-size:10px;color:var(--dim);margin-top:1px}

 /* 주차 완료 안내문. 흐름의 마지막 칸이라 1번 띠의 오른쪽 끝에 붙는다.
    수신 -> 큐 -> 안내 -> 완료가 한 줄로 읽힌다. */
 #notice{overflow:hidden;min-width:0}
 #notice.ok  {border-color:var(--ok)}
 #notice.bad {border-color:var(--bad)}
 #notice.warn{border-color:var(--warn)}
 .nt-head{display:flex;justify-content:space-between;font-size:11px;
          color:var(--dim);margin-bottom:6px;gap:10px}
 .nt-line{font-size:14px;font-weight:600;line-height:1.5}
 #notice.ok .nt-line{color:var(--ok)}
 #notice.bad .nt-line{color:#ff9b9b}
 #notice.warn .nt-line{color:var(--warn)}
 .nt-note{font-size:11px;color:var(--dim);margin-top:5px;line-height:1.5}
 .nt-idle{font-size:13px;color:var(--dim)}

 /* ---- 2번 구간 : 카메라 (차량 검출) ---- */
 /* 영상 두 개 모두 남는 공간에 맞춰 비율을 지키며 들어간다.
    min-height:0이 없으면 flex 자식이 콘텐츠 크기만큼 밀어내 패널이 넘친다. */
 #camwrap{flex:1;display:flex;align-items:center;justify-content:center;
          background:#0e0e10;min-height:0;min-width:0;padding:6px;position:relative}
 #camimg{max-width:100%;max-height:100%;object-fit:contain;border-radius:6px}
 #caminfo{border-top:1px solid var(--line);padding:9px 14px;font-size:12px;
          color:var(--dim);display:flex;justify-content:space-between;gap:10px;
          align-items:center}
 #caminfo > span{display:flex;align-items:center;gap:8px;min-width:0}
 /* 영상 위 상태 글자 토글. 평소에는 꺼져 있어야 시연 화면이 깨끗하다. */
 #dbgbtn{background:#2a2a31;color:var(--dim);border:1px solid var(--line);
         border-radius:5px;padding:3px 9px;font-size:11px;cursor:pointer;
         font-family:inherit;white-space:nowrap;flex:0 0 auto}
 #dbgbtn:hover{background:#34343d;color:var(--text)}
 #dbgbtn.on{background:#1d3d55;border-color:var(--accent);color:var(--accent)}

 /* 트랙 목록. 영상 위에 겹친다.
    영상에도 박스와 라벨이 그려지지만 화면에서 작아 읽기 어렵다.
    같은 내용을 읽을 수 있는 크기로 다시 보여주는 것이다. */
 #tracks{position:absolute;top:12px;left:12px;display:flex;flex-direction:column;
         gap:4px;pointer-events:none;max-height:calc(100% - 24px);overflow:hidden}
 .trk{display:flex;align-items:center;gap:7px;padding:4px 9px;border-radius:6px;
      background:rgba(14,14,16,.82);border:1px solid #33333b;
      font-size:12px;white-space:nowrap;backdrop-filter:blur(2px)}
 .trk .tid{color:var(--dim);font-variant-numeric:tabular-nums}
 .trk .car{font-weight:700;letter-spacing:1px;font-variant-numeric:tabular-nums}
 .trk .st{font-size:10px;padding:1px 6px;border-radius:4px}
 /* 영상 위 박스 색과 맞춘다. B02의 COLOR_* 와 같은 구분이라
    영상에서 노란 박스를 찾으면 여기 '안내중'과 짝이 맞는다. */
 .trk.active  {border-color:#c8c800}
 .trk.active  .st{background:#4a4a00;color:#ffff64}   /* 영상: 노랑 */
 .trk.parked  .st{background:#0e3a17;color:#4ccf6a}   /* 영상: 초록 */
 .trk.guided  .st{background:#0e2e3a;color:#5cc8ff}
 .trk.waiting .st{background:#3d2a0a;color:#ffa53c}   /* 영상: 주황 */
 .trk .sp{color:var(--dim);font-size:11px}

 /* ---- 4번 구간 : 실시간 안내 ---- */
 #navwrap{flex:1;display:flex;align-items:center;justify-content:center;
          background:#0e0e10;min-height:0;min-width:0;padding:6px}
 #navimg{max-width:100%;max-height:100%;object-fit:contain;border-radius:6px}
 #navinfo{border-top:1px solid var(--line);padding:9px 14px;font-size:12px;
          color:var(--dim);display:flex;justify-content:space-between;gap:10px}
 .pill{padding:2px 7px;border-radius:4px;background:#232329;font-size:11px}
 .pill.ok{color:var(--ok)} .pill.bad{color:var(--bad)} .pill.warn{color:var(--warn)}
</style></head><body>
<div id="app">

  <!-- 1번 구간 : Zybo 수신 -> FIFO 대기열 -> 안내 -> 주차 완료 (상단 가로 띠).
       B02의 CarNumberFIFO가 실제로 하는 일을 그대로 늘어놓은 것이다. -->
  <div class="col" id="top">
    <h2><span><span class="num">1</span>Zybo 수신 &amp; 차량번호 FIFO</span>
        <span class="sub" id="rxcount">-</span></h2>
    <div id="pipe">

      <div class="stg rxbox">
        <div class="cap"><span>수신 (A00)</span><span>2바이트</span></div>
        <div id="rxlist">
          <div class="empty">Zybo 수신 대기 중</div>
        </div>
      </div>

      <div class="flowarw"><b>push</b><div class="ln"></div><span>배정 성공만</span></div>

      <div class="stg queue">
        <div class="cap"><span>FIFO 대기열 (front → back)</span>
                         <span id="qcount">대기 0대</span></div>
        <div class="qrow" id="queue">
          <div class="qempty">대기 중인 차량번호가 없습니다.</div>
        </div>
      </div>

      <div class="flowarw"><b>pop</b><div class="ln"></div><span>움직이는 차</span></div>

      <div class="stg act">
        <div class="cap"><span>안내 중 (B02)</span></div>
        <div id="active"><div class="empty">없음</div></div>
      </div>

      <div class="flowarw"><b>도착</b><div class="ln"></div><span>반경 __ARRIVAL_CM__cm</span></div>

      <div class="stg done" id="notice">
        <div class="cap"><span>주차 완료 / 오주차</span></div>
        <div id="noticebody"><div class="nt-idle">주차 완료를 기다리는 중입니다.</div></div>
      </div>

    </div>
  </div>

  <!-- 아래 3등분 : 2 CCTV / 3 주차장 상태 / 4 실시간 안내 -->
  <div id="grid">

    <!-- 2번 구간 : 주차장 CCTV (차량 검출/추적 오버레이) -->
    <div class="col">
      <h2><span><span class="num">2</span>주차장 CCTV 화면</span>
          <span class="sub" id="camsub">-</span></h2>
      <div id="camwrap">
        <img id="camimg" src="/video_feed">
        <div id="tracks"></div>
      </div>
      <div id="caminfo">
        <span><button id="dbgbtn" onclick="toggleDebug()">디버그</button>
              <span id="camdet">검출 대기 중</span></span>
        <span><span class="pill" id="cammark">-</span>
              <span class="pill" id="camfps">-</span></span>
      </div>
    </div>

    <!-- 3번 구간 : 주차장 상태 -->
    <div class="col">
      <h2><span><span class="num">3</span>주차장 상태</span>
          <span class="sub"><span id="lotroute"></span>
          <span id="lottime">-</span></span></h2>
      <div id="lotwrap"><div id="lot"></div></div>
      <div id="avail"></div>
    </div>

    <!-- 4번 구간 : 실시간 주차 안내 -->
    <div class="col">
      <h2><span><span class="num">4</span>실시간 주차 안내</span>
          <span class="sub" id="navsub">-</span></h2>
      <div id="navwrap"><img id="navimg" src="/nav_feed"></div>
      <div id="navinfo">
        <span id="navguide">대기 중</span>
        <span><span class="pill" id="navhomo">-</span>
              <span class="pill" id="navfps">-</span></span>
      </div>
    </div>

  </div>

</div>
<script>
const POLL_MS = __POLL_MS__;
let layout = null;
let lotAspect = 1;       // 격자 전체의 가로/세로 비
let lotObserver = null;  // 아래 fitLot 주석 참고. 참조를 반드시 붙들어야 한다.
const cellEls = {};      // 자리ID -> [칸 엘리먼트, ...]

// 남는 공간에 맞춰 격자 크기를 정한다.
// 비율(lotAspect)은 서버가 CONFIG['LOT_ASPECT']를 보고 정해서 보낸다.
// 폭에만 맞추면 아래가 잘리므로 가로/세로 중 좁은 쪽에 맞춘다.
function fitLot(){
  const wrap = document.getElementById('lotwrap');
  const lot = document.getElementById('lot');
  const pad = 16;
  const availW = wrap.clientWidth - pad, availH = wrap.clientHeight - pad;
  // 아직 배치 전이라 크기가 0이다. 지금 계산하면 0이 박히므로 그냥 넘긴다.
  // 크기가 잡히면 아래 감시자들이 다시 부른다.
  if (availW <= 0 || availH <= 0 || !(lotAspect > 0)) return;

  const w = Math.min(availW, availH * lotAspect);
  lot.style.width  = Math.floor(w) + 'px';
  lot.style.height = Math.floor(w / lotAspect) + 'px';
  drawRoute();
}

// 격자 좌표(행/열) -> 격자 안 픽셀. 칸 사이 gap까지 넣는다.
// drawRoute / renderCars가 같은 규칙을 써야 선과 점이 맞는다.
function lotMetrics(){
  const lot = document.getElementById('lot');
  if (!layout) return null;
  const w = lot.clientWidth, h = lot.clientHeight;
  if (!w || !h) return null;
  const cw = (w - (layout.cols - 1) * LOT_GAP) / layout.cols;
  const ch = (h - (layout.rows - 1) * LOT_GAP) / layout.rows;
  return {
    w, h, cw, ch,
    cx: c => c * (cw + LOT_GAP) + cw / 2,
    cy: r => r * (ch + LOT_GAP) + ch / 2,
  };
}

// 일방통행 화살표는 격자에 그리지 않는다.
//
// 목업 바닥에 화살표가 이미 붙어 있고, 2번 CCTV 화면에도 그 실물이 보인다.
// 격자에 또 그리면 같은 정보가 두 겹이 되고, 안내 경로(파란 선)와 겹쳐서
// 어느 것이 지금 가야 할 길인지 읽기 어려워진다.
// 방향 데이터(layout.one_way)는 그대로 내려오므로 필요하면 다시 그릴 수 있다.

// 안내 경로. 4번 구간(차량 시점)이 그리는 것과 같은 선을 격자 위에 그대로
// 깐다. 격자는 실제 주차장과 같은 방향으로 서 있으므로 어느 길로 가는지가
// 그대로 읽힌다.
//
// 격자 크기가 바뀔 때(fitLot)와 상태가 올 때(renderRoute) 모두 다시 그려야
// 하므로 마지막 상태를 붙들어 둔다.
let lastRoute = null;

function drawRoute(){
  const svg = document.getElementById('route');
  const m = lotMetrics();
  if (!svg || !m) return;

  svg.setAttribute('viewBox', `0 0 ${m.w} ${m.h}`);
  svg.setAttribute('width', m.w);
  svg.setAttribute('height', m.h);

  const rt = lastRoute;
  if (!rt || !rt.points || rt.points.length < 2){
    svg.innerHTML = '';
    return;
  }

  // 역주행 구간을 물고 나온 경로는 붉게 그린다. 잘못된 안내를 파란 선으로
  // 그리면 화면만 보고는 정상과 구별할 수 없다.
  const bad = rt.wrong_way ? ' bad' : '';
  const pts = rt.points
    .map(p => `${m.cx(p.col).toFixed(1)},${m.cy(p.row).toFixed(1)}`).join(' ');
  const goal = rt.points[rt.points.length - 1];

  svg.innerHTML =
    `<polyline class="rt${bad}" points="${pts}"/>` +
    `<circle class="goal${bad}" cx="${m.cx(goal.col).toFixed(1)}" `
    + `cy="${m.cy(goal.row).toFixed(1)}" r="4"/>`;
}

function renderRoute(st){
  lastRoute = st.route_on_map || null;
  drawRoute();

  const sub = document.getElementById('lotroute');
  if (!sub) return;
  if (!lastRoute){
    sub.textContent = '';
    return;
  }
  sub.textContent = `${lastRoute.car_id || '?'} → ${lastRoute.target_spot || '?'}`
    + (lastRoute.wrong_way ? ' (역주행 경고)' : '');
  sub.className = lastRoute.wrong_way ? 'bad' : '';
}

// 크기가 잡히는 시점이 제각각이라 세 군데서 부른다.
//   ResizeObserver : 패널 크기가 바뀔 때 (주 경로)
//   rAF            : 첫 배치 직후. buildLot 시점에는 컨테이너가 0일 수 있다
//   window resize  : 관찰자가 놓치는 경우의 최후 보루
// lotObserver를 전역에 붙들어 두는 것이 중요하다. 지역 변수로 두면 참조가
// 사라진 관찰자가 수거되어 콜백이 한 번도 오지 않는 일이 실제로 있었다.
function watchLotSize(){
  if (lotObserver) return;
  lotObserver = new ResizeObserver(fitLot);
  lotObserver.observe(document.getElementById('lotwrap'));
  window.addEventListener('resize', fitLot);
}

// 격자를 한 번만 만든다. 이후에는 색과 점만 바꾼다.
// 매번 다시 만들면 CSS 깜빡임 애니메이션이 폴링 주기마다 처음으로 돌아간다.
function buildLot(lay){
  layout = lay;
  const lot = document.getElementById('lot');
  lot.style.gridTemplateColumns = `repeat(${lay.cols}, 1fr)`;
  lot.style.gridTemplateRows    = `repeat(${lay.rows}, 1fr)`;
  // 전체 비율만 잡는다. 칸마다 aspect-ratio를 걸면 행 높이가 제각각
  // 계산되어 격자가 컨테이너를 넘는다.
  lotAspect = lay.aspect > 0 ? lay.aspect : lay.cols / lay.rows;
  lot.innerHTML = '';

  for (let r = 0; r < lay.rows; r++){
    for (let c = 0; c < lay.cols; c++){
      const info = lay.cells[r][c];
      const el = document.createElement('div');
      el.className = 'cell ' + info.kind;

      if (info.kind === 'spot' && info.spot){
        el.style.setProperty('--c', info.color);
        // 같은 자리가 세로로 이어지는지 보고 테두리를 붙인다 (대형 = 2칸)
        const above = r > 0 ? lay.cells[r-1][c] : null;
        const below = r < lay.rows-1 ? lay.cells[r+1][c] : null;
        const contUp   = above && above.spot === info.spot;
        const contDown = below && below.spot === info.spot;
        if (contUp)   el.classList.add('cont');
        if (contDown) el.classList.add('head');
        if (!contUp)  el.textContent = info.spot;   // 이름은 첫 칸에만
        (cellEls[info.spot] = cellEls[info.spot] || []).push(el);
      } else if (info.kind === 'pill'){
        el.dataset.lbl = info.marker != null ? info.marker : '';
      } else if (info.kind === 'gate1'){
        el.dataset.lbl = '입구';
      } else if (info.kind === 'gate2'){
        el.dataset.lbl = '출구';
      }
      lot.appendChild(el);
    }
  }

  // 안내 경로 겹. 칸 위, 차량 점 아래.
  const route = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  route.id = 'route';
  lot.appendChild(route);

  // 차량 점을 올릴 겹. 칸을 다 만든 뒤에 붙인다 (innerHTML='' 로 지워지므로)
  const cars = document.createElement('div');
  cars.id = 'cars';
  lot.appendChild(cars);

  const av = document.getElementById('avail');
  av.dataset.legend = JSON.stringify(lay.legend);

  fitLot();
  requestAnimationFrame(fitLot);   // 첫 배치가 끝난 뒤 한 번 더
  watchLotSize();
}

function esc(s){
  return String(s == null ? '' : s).replace(/[&<>"]/g,
    m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));
}

// ---- 1번 구간 : 수신 -> FIFO -> 안내 ----
// 가로 띠라 높이가 좁다. 수신 목록은 최근 두 건만 보여준다.
// (전체 이력이 필요하면 /status를 볼 것)
const RX_SHOWN = 2;

function renderRx(rx){
  document.getElementById('rxcount').textContent =
    `입차 ${rx.total.entry} · 출차 ${rx.total.exit}`;

  const box = document.getElementById('rxlist');
  if (!rx.items.length){
    box.innerHTML = '<div class="empty">Zybo 수신 대기 중</div>';
    return;
  }

  box.innerHTML = rx.items.slice(0, RX_SHOWN).map(it => {
    const cls = !it.ok ? 'fail' : (it.role === 'exit' ? 'exit' : '');
    let detail;
    if (it.role === 'exit'){
      // 출차는 '어느 자리에서 나갔고, 얼마나 있었고, 얼마인지'가 전부다.
      detail = it.ok
        ? `<span class="rx-spot">${esc(it.spot_id)} 출차</span> · ` +
          `${it.minutes}분 · <b>${(it.fee || 0).toLocaleString()}원</b>`
        : `<span class="rx-fail">${esc(it.message) || '출차 실패'}</span>`;
    } else if (it.ok && it.spot_id){
      detail = `${esc(it.car_type)} · <span class="rx-spot">${esc(it.spot_id)} 배정</span>`;
    } else {
      detail = `<span class="rx-fail">${esc(it.message) || '배정 실패'}</span>`;
    }
    const src = it.source === 'manual' ? '(수동)' : '';
    return `<div class="rx ${cls}">
      <div class="rx-top">
        <span class="rx-car">${esc(it.car_id)}</span>
        <span class="rx-time">${esc(it.role_label)}${src} ${esc(it.time)}</span>
      </div>
      <div class="rx-line">${detail}</div>
    </div>`;
  }).join('');
}

// FIFO 대기열. 왼쪽 끝이 front(다음에 매칭될 차)다.
// 이 순서가 곧 안내를 받는 순서라, 숫자 하나로 줄이면 화면에서 그 사실이 사라진다.
function renderFifo(st){
  const f = st.fifo || {waiting: [], active: null};

  document.getElementById('qcount').textContent = `대기 ${f.waiting.length}대`;

  const q = document.getElementById('queue');
  q.innerHTML = f.waiting.length
    ? f.waiting.map((car, i) =>
        `<div class="qcard${i === 0 ? ' front' : ''}">
           <div class="no">${esc(car)}</div>
           <div class="mk">${i === 0 ? 'front' : '&nbsp;'}</div>
         </div>`).join('')
    : '<div class="qempty">대기 중인 차량번호가 없습니다.</div>';

  const a = document.getElementById('active');
  a.innerHTML = f.active
    ? `<div class="actcar">${esc(f.active.car_id)}</div>
       <div class="actto">→ <b>${esc(f.active.spot_id || '?')}</b>
          <span>${esc(f.active.car_type || '')}</span></div>`
    : '<div class="empty">없음</div>';
}

// ---- 3번 구간 ----
function renderLot(st){
  for (const [spotId, els] of Object.entries(cellEls)){
    const s = st.spots[spotId];
    if (!s) continue;
    const full    = s.status === 'full' && s.parked;
    const pending = s.pending;

    els.forEach((el, idx) => {
      el.classList.toggle('full', full);
      el.classList.toggle('pending', pending);

      // 깜빡이는 점은 자리마다 하나만. 이미 있으면 그대로 두어야
      // 애니메이션이 끊기지 않는다.
      const has = el.querySelector('.dot');
      if (pending && idx === 0 && !has){
        const d = document.createElement('div');
        d.className = 'dot';
        el.appendChild(d);
      } else if ((!pending || idx !== 0) && has){
        has.remove();
      }
    });
  }

  const legend = JSON.parse(document.getElementById('avail').dataset.legend || '[]');
  document.getElementById('avail').innerHTML = legend.map(g => {
    const a = st.availability[g.name];
    if (!a) return '';
    return `<div class="av">
      <div class="n" style="color:${g.color}">${a.empty}<span
        style="font-size:11px;color:var(--dim)">/${a.total}</span></div>
      <div class="l">${esc(g.name)}</div></div>`;
  }).join('');

  document.getElementById('lottime').textContent = st.time;

  // 범례(#avail)를 채운 뒤에 맞춘다. 그 칸의 높이가 정해져야 격자에
  // 남는 높이가 확정되기 때문이다. 먼저 재면 아래가 컨테이너를 넘친다.
  // 값이 그대로면 같은 값을 다시 쓸 뿐이라 매 폴링마다 불러도 부담이 없다.
  // (#lotwrap 높이는 flex:1로 패널이 정하므로 여기서 되먹임이 생기지 않는다)
  fitLot();
}

// 격자 위 차량 점.
// 칸 사이 간격(CSS의 gap)까지 계산에 넣는다. 퍼센트로만 두면 간격이 쌓여
// 오른쪽/아래로 갈수록 점이 실제 칸에서 밀린다.
const LOT_GAP = 2;
const carEls = {};

function renderCars(st){
  const lot = document.getElementById('lot');
  const box = document.getElementById('cars');
  if (!layout || !box) return;

  const w = lot.clientWidth, h = lot.clientHeight;
  if (!w || !h) return;
  const cw = (w - (layout.cols - 1) * LOT_GAP) / layout.cols;
  const ch = (h - (layout.rows - 1) * LOT_GAP) / layout.rows;

  const seen = new Set();
  (st.cars_on_map || []).forEach(c => {
    // 격자 밖으로 크게 벗어난 값은 좌표가 튄 것이다. 찍지 않는다.
    if (c.row < -1 || c.row > layout.rows || c.col < -1 || c.col > layout.cols) return;
    seen.add(c.key);

    let el = carEls[c.key];
    if (!el){
      el = document.createElement('div');
      el.innerHTML = '<span></span>';
      box.appendChild(el);
      carEls[c.key] = el;
      el.querySelector('span').textContent = c.key;
    }
    el.className = 'car' + (c.parked ? ' parked' : '') + (c.car_id ? '' : ' unknown');
    el.style.left = (c.col * (cw + LOT_GAP) + cw / 2) + 'px';
    el.style.top  = (c.row * (ch + LOT_GAP) + ch / 2) + 'px';
  });

  // 사라진 차의 점은 지운다
  for (const key of Object.keys(carEls)){
    if (!seen.has(key)){ carEls[key].remove(); delete carEls[key]; }
  }
}

const NOTICE_CLASS = {
  parked_ok:        'ok',
  misparked:        'bad',
  mispark_unknown:  'warn',
  relocate_failed:  'warn',
};

function renderNotice(ev){
  // 겉칸(#notice)은 1번 띠의 카드라서 stg/done 클래스를 잃으면 안 된다.
  // 상태 색만 갈아 끼우고 내용은 안쪽(#noticebody)에 쓴다.
  const card = document.getElementById('notice');
  const box = document.getElementById('noticebody');
  if (!ev){
    card.className = 'stg done';
    box.innerHTML = '<div class="nt-idle">주차 완료를 기다리는 중입니다.</div>';
    return;
  }
  card.className = 'stg done ' + (NOTICE_CLASS[ev.kind] || '');
  const where = ev.spot_id ? `${esc(ev.spot_id)}` : '위치 미확인';
  box.innerHTML =
    `<div class="nt-head"><span>차량 ${esc(ev.car_id)} · ${where}</span>
       <span>${esc(ev.time)}</span></div>` +
    ev.lines.map(l => `<div class="nt-line">${esc(l)}</div>`).join('') +
    (ev.note ? `<div class="nt-note">${esc(ev.note)}</div>` : '');
}

// ---- 2번 구간 : 카메라 ----
function renderCam(st){
  const t = st.tracks;
  document.getElementById('camsub').textContent =
    t.total ? `추적 ${t.total}대` : '추적 없음';
  const det = t.detected;
  const gap = (det != null && det !== t.total) ? ` (검출 ${det})` : '';
  document.getElementById('camdet').textContent = t.total
    ? `추적 ${t.total}대${gap} · 번호 매칭 ${t.matched}대`
    : (det ? `검출 ${det}대 · 추적 0대` : '검출 대기 중');

  // 카메라가 주는 fps와 처리 fps의 차이. 벌어져 있으면 프레임을 버리는 중이다.
  const m = document.getElementById('cammark');
  const cam = st.camera_fps || 0, drop = st.dropped_frames || 0;
  m.textContent = cam ? `카메라 ${cam} / 처리 ${st.fps} · 버림 ${drop}`
                      : `처리 ${st.fps} FPS`;
  m.className = 'pill ' + (cam && cam - st.fps > cam * 0.3 ? 'warn' : '');

  const d = st.stage_ms || {};
  document.getElementById('camfps').textContent =
    d.detect != null ? `검출 ${d.detect}ms` : `${st.fps} FPS`;

  // 영상 위에 겹치는 트랙 목록.
  //
  // 지금 안내 중인 차만 띄운다. 주차를 마친 차까지 전부 띄우면 목록이
  // 화면 절반을 덮어 정작 봐야 할 영상을 가린다. 세워둔 차가 어디에 있는지는
  // 3번 격자가 점으로 보여주고, 영상에도 초록 박스로 이미 표시된다.
  // 여기서 알고 싶은 것은 '지금 누구를 어디로 보내는 중인가' 하나다.
  document.getElementById('tracks').innerHTML = (t.items || [])
    .filter(it => it.state === 'active')
    .map(it => `
    <div class="trk ${it.state}">
      <span class="tid">#${it.track_id}</span>
      <span class="car">${esc(it.car_id) || '번호 미상'}</span>
      <span class="st">${esc(it.state_label)}</span>
      ${it.spot_id ? `<span class="sp">→ ${esc(it.spot_id)}</span>` : ''}
    </div>`).join('');

  // 디버그 버튼 상태 (영상 위 상태 글자 on/off)
  const b = document.getElementById('dbgbtn');
  if (b){
    b.classList.toggle('on', !!st.debug_overlay);
    b.textContent = st.debug_overlay ? '디버그 끄기' : '디버그';
  }
}

// 영상 위 상태 글자를 켜고 끈다. 서버가 실제 상태를 돌려주므로
// 화면은 그것만 반영한다. (여러 창을 열어도 어긋나지 않는다)
async function toggleDebug(){
  try {
    const r = await (await fetch('/debug')).json();
    const b = document.getElementById('dbgbtn');
    b.classList.toggle('on', !!r.debug);
    b.textContent = r.debug ? '디버그 끄기' : '디버그';
  } catch (e) {
    console.error('디버그 토글 실패', e);
  }
}

// ---- 4번 구간 ----
function renderNav(st){
  const guided = st.vehicles.find(v => v.car_id) || st.vehicles[0];
  document.getElementById('navguide').textContent = guided
    ? `${guided.car_id || '번호 미매칭'} → ${guided.target_spot || '목표 없음'}` +
      (guided.distance_cm != null ? `  ${guided.distance_cm}cm` : '')
    : '안내 중인 차량 없음';

  document.getElementById('navsub').textContent = `추적 ${st.vehicles.length}대`;

  const h = document.getElementById('navhomo');
  h.textContent = '좌표계 ' + st.homography.text;
  h.className = 'pill ' + ({locked:'ok', provisional:'warn', not_ready:'bad'}
                            [st.homography.state] || '');
  document.getElementById('navfps').textContent = `${st.fps} FPS`;
}

async function tick(){
  try {
    if (!layout){
      buildLot(await (await fetch('/lot_layout')).json());
    }
    const st = await (await fetch('/ui_state')).json();
    renderRx(st.rx);
    renderFifo(st);
    renderLot(st);
    renderCars(st);
    renderRoute(st);
    renderNotice(st.latest_event);
    renderCam(st);
    renderNav(st);
  } catch (e) {
    console.error('상태 갱신 실패', e);
  }
}

tick();
setInterval(tick, POLL_MS);
</script></body></html>
"""


def render_page():
    """
    최종 화면 HTML을 반환.

    설정에서 온 값은 HTML에 박아두지 않고 여기서 채운다. 박아두면 CONFIG를
    고쳤을 때 화면만 옛 값으로 남아, 화면에 적힌 숫자와 실제 판정이 어긋난다.
    """
    return (FINAL_UI_HTML
            .replace("__POLL_MS__", str(CONFIG['POLL_INTERVAL_MS']))
            .replace("__ARRIVAL_CM__",
                     f"{B02_CONFIG['SINGLE_ACTIVE']['ARRIVAL_RADIUS_CM']:g}"))


# 단독 실행 : 배치 JSON 확인
if __name__ == "__main__":
    layout = build_lot_layout()
    print(f"격자 {layout['rows']}행 x {layout['cols']}열")

    spots = {}
    for row in layout["cells"]:
        for cell in row:
            if cell.get("spot"):
                spots.setdefault(cell["spot"], cell["type_name"])
    print(f"주차 자리 {len(spots)}개")
    for spot_id, name in sorted(spots.items()):
        print(f"  {spot_id:5s} {name}")

    print("\n안내 문구")
    for label, lines in [("정상 주차", MSG_PARKED_OK),
                         ("오주차", MSG_MISPARKED),
                         ("오주차(자리 불명)", MSG_MISPARK_UNKNOWN)]:
        print(f"  {label:16s} " + " / ".join(lines))

    print(f"\n오주차 판정 기준  허용 {CONFIG['MISPARK_TOLERANCE_CM']}cm / "
          f"탐색 {CONFIG['MISPARK_MAX_DIST_CM']}cm")
