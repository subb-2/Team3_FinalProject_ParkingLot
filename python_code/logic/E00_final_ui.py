"""
E00_final_ui : 최종 통합 관제 화면

화면을 세로로 3분할한다.

    +----------------+----------------+----------------+
    |   1) 수신       |  2) 주차장 상태  |  3) 실시간 안내  |
    |                |                |                |
    |  Zybo에서 온    |  전체 자리 현황  |  차량 시점      |
    |  Wi-Fi UART    |  + 배정 자리     |  내비게이션     |
    |  수신 값 목록    |    깜빡임       |  (D00 재사용)   |
    |                |  + 하단 안내문    |                |
    +----------------+----------------+----------------+

왜 OpenCV 캔버스가 아니라 HTML인가
  1번과 2번 구간은 한글 안내문이 핵심이다. OpenCV의 putText는 한글을 못 그린다.
  (D00의 MANEUVER_LABEL이 전부 영문인 것도 그 때문이다)
  3번 구간만 영상이므로, 영상은 MJPEG로 넣고 나머지는 브라우저가 그리게 했다.
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
import json
import threading
from collections import deque
from datetime import datetime

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.map_data import (
    grid_map, coord_to_spot, PILL_MARKER_ID, spot_type as SPOT_TYPE_OF,
    SPOT_TYPE_NAME, SPOT_CELLS, ROAD, PILL, GATE1, GATE2,
    SPOT1, SPOT2, SPOT3, SPOT4, get_rows, get_cols,
)
from data.car_data import cars_info, get_car_type
from logic.B02_car_mot import CONFIG as B02_CONFIG


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


# 주차 완료 / 오주차 감시 (2번 구간)
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
        for nav in pipeline.latest_nav:
            car_id = nav.get("car_id")
            pos = nav.get("world_pos")
            if car_id and pos is not None:
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
        "legend": [
            {"name": name, "color": SPOT_TYPE_CSS.get(t, "#8a8a8a")}
            for t, name in SPOT_TYPE_NAME.items()
        ],
    }


# 화면 상태 (0.5초마다 브라우저가 가져간다)
def build_ui_state(pipeline, rx_feed, watcher):
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

    mapper = pipeline.navigator.mapper
    if not mapper.is_ready():
        homography = {"state": "not_ready", "text": "마커 대기 중"}
    elif mapper.locked:
        homography = {"state": "locked",
                      "text": f"확정 {mapper.calibrated_with}점 "
                              f"오차 {mapper.reproj_error:.1f}cm"}
    else:
        homography = {"state": "provisional",
                      "text": f"임시 {mapper.calibrated_with}/{mapper.lock_markers}점"}

    return {
        # --- 1번 구간 ---
        "rx": {"items": rx_items, "total": rx_total},

        # --- 2번 구간 ---
        "spots": spots,
        "assigned_pending": assigned_pending,
        "availability": {
            info["name"]: {"empty": info["empty"], "total": info["total"]}
            for info in get_availability_by_type().values()
        },
        "latest_event": latest,
        "events": events,

        # --- 3번 구간 + 공통 ---
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
        "fps": round(pipeline.fps, 1),
        "homography": homography,
        "markers": len(pipeline.navigator.latest_markers),
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

 /* 세로 3분할 */
 #app{display:grid;grid-template-columns:1fr 1.25fr 1.35fr;
      gap:10px;padding:10px;height:100vh}
 .col{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      display:flex;flex-direction:column;min-width:0;overflow:hidden}
 .col > h2{margin:0;padding:11px 14px;font-size:15px;font-weight:600;
           border-bottom:1px solid var(--line);display:flex;
           justify-content:space-between;align-items:center}
 .num{display:inline-block;width:20px;height:20px;line-height:20px;
      text-align:center;border-radius:5px;background:var(--accent);
      color:#06121e;font-size:12px;font-weight:700;margin-right:8px}
 .sub{font-size:11px;color:var(--dim);font-weight:400}
 .body{flex:1;overflow:auto;padding:10px 12px;min-height:0}

 /* ---- 1번 구간 : 수신 목록 ---- */
 .rx{border:1px solid var(--line);border-left:3px solid var(--accent);
     border-radius:7px;padding:8px 10px;margin-bottom:7px;background:#232329}
 .rx.exit{border-left-color:var(--warn)}
 .rx.fail{border-left-color:var(--bad)}
 .rx-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
 .rx-car{font-size:21px;font-weight:700;letter-spacing:2px;font-variant-numeric:tabular-nums}
 .rx-time{font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums}
 .rx-tag{font-size:10px;padding:2px 6px;border-radius:4px;background:#33333b;color:var(--dim)}
 .rx-line{font-size:12px;color:var(--dim);margin-top:4px;word-break:break-all}
 .rx-spot{color:var(--ok);font-weight:600}
 .rx-fail{color:var(--bad)}
 .empty{color:var(--dim);font-size:13px;text-align:center;padding:26px 10px;line-height:1.7}

 /* ---- 2번 구간 : 주차장 상태 ---- */
 #lotwrap{flex:1;display:flex;align-items:center;justify-content:center;
          padding:10px;min-height:0}
 #lot{display:grid;gap:2px;width:100%;max-width:100%}
 .cell{aspect-ratio:10/17.5;border-radius:2px;position:relative}
 .cell.road{background:#212125}
 .cell.gate1,.cell.gate2{background:#2b3b2b;border:1px solid #4f7a4f}
 .cell.pill{background:#3a3f49;border:1px solid #58606f}
 .cell.gate1::after,.cell.gate2::after,.cell.pill::after{
   content:attr(data-lbl);position:absolute;inset:0;display:flex;
   align-items:center;justify-content:center;font-size:9px;color:#9aa6b8}
 .cell.spot{border:1.5px solid var(--c,#8a8a8a);background:#1e1e22;
            display:flex;align-items:center;justify-content:center;
            font-size:9px;color:var(--c,#8a8a8a);font-weight:600;overflow:hidden}
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

 /* 하단 안내문 */
 #notice{border-top:1px solid var(--line);padding:11px 14px;min-height:96px;
         background:#1a1a1e}
 #notice.ok{border-top:2px solid var(--ok)}
 #notice.bad{border-top:2px solid var(--bad)}
 #notice.warn{border-top:2px solid var(--warn)}
 .nt-head{display:flex;justify-content:space-between;font-size:11px;
          color:var(--dim);margin-bottom:6px}
 .nt-line{font-size:15px;font-weight:600;line-height:1.55}
 #notice.ok .nt-line{color:var(--ok)}
 #notice.bad .nt-line{color:#ff9b9b}
 #notice.warn .nt-line{color:var(--warn)}
 .nt-note{font-size:11px;color:var(--dim);margin-top:5px;line-height:1.5}
 .nt-idle{font-size:13px;color:var(--dim)}

 /* ---- 3번 구간 : 실시간 안내 ---- */
 #navwrap{flex:1;display:flex;align-items:center;justify-content:center;
          background:#0e0e10;min-height:0;padding:6px}
 #navimg{max-width:100%;max-height:100%;object-fit:contain;border-radius:6px}
 #navinfo{border-top:1px solid var(--line);padding:9px 14px;font-size:12px;
          color:var(--dim);display:flex;justify-content:space-between;gap:10px}
 .pill{padding:2px 7px;border-radius:4px;background:#232329;font-size:11px}
 .pill.ok{color:var(--ok)} .pill.bad{color:var(--bad)} .pill.warn{color:var(--warn)}
</style></head><body>
<div id="app">

  <!-- 1번 구간 : Zybo 수신 -->
  <div class="col">
    <h2><span><span class="num">1</span>Zybo 수신</span>
        <span class="sub" id="rxcount">-</span></h2>
    <div class="body" id="rxlist">
      <div class="empty">Zybo에서 보낸 차량번호를 기다리는 중입니다.</div>
    </div>
  </div>

  <!-- 2번 구간 : 주차장 상태 -->
  <div class="col">
    <h2><span><span class="num">2</span>주차장 상태</span>
        <span class="sub" id="lottime">-</span></h2>
    <div id="lotwrap"><div id="lot"></div></div>
    <div id="avail"></div>
    <div id="notice"><div class="nt-idle">주차 완료를 기다리는 중입니다.</div></div>
  </div>

  <!-- 3번 구간 : 실시간 안내 -->
  <div class="col">
    <h2><span><span class="num">3</span>실시간 주차 안내</span>
        <span class="sub" id="navsub">-</span></h2>
    <div id="navwrap"><img id="navimg" src="/nav_feed"></div>
    <div id="navinfo">
      <span id="navguide">대기 중</span>
      <span><span class="pill" id="navhomo">-</span>
            <span class="pill" id="navfps">-</span></span>
    </div>
  </div>

</div>
<script>
const POLL_MS = __POLL_MS__;
let layout = null;
const cellEls = {};      // 자리ID -> [칸 엘리먼트, ...]

// 격자를 한 번만 만든다. 이후에는 색과 점만 바꾼다.
// 매번 다시 만들면 CSS 깜빡임 애니메이션이 폴링 주기마다 처음으로 돌아간다.
function buildLot(lay){
  layout = lay;
  const lot = document.getElementById('lot');
  lot.style.gridTemplateColumns = `repeat(${lay.cols}, 1fr)`;
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

  const av = document.getElementById('avail');
  av.dataset.legend = JSON.stringify(lay.legend);
}

function esc(s){
  return String(s == null ? '' : s).replace(/[&<>"]/g,
    m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));
}

// ---- 1번 구간 ----
function renderRx(rx){
  document.getElementById('rxcount').textContent =
    `입차 ${rx.total.entry} · 출차 ${rx.total.exit}`;

  const box = document.getElementById('rxlist');
  if (!rx.items.length){
    box.innerHTML = '<div class="empty">Zybo에서 보낸 차량번호를 기다리는 중입니다.</div>';
    return;
  }

  box.innerHTML = rx.items.map(it => {
    const cls = !it.ok ? 'fail' : (it.role === 'exit' ? 'exit' : '');
    let detail;
    if (it.role === 'exit'){
      detail = '출차 처리';
    } else if (it.ok && it.spot_id){
      detail = `${esc(it.car_type)} · <span class="rx-spot">${esc(it.spot_id)} 배정</span>`;
    } else {
      detail = `<span class="rx-fail">${esc(it.message) || '배정 실패'}</span>`;
    }
    const src = it.source === 'manual' ? '<span class="rx-tag">수동</span>' : '';
    return `<div class="rx ${cls}">
      <div class="rx-top">
        <span class="rx-car">${esc(it.car_id)}</span>
        <span class="rx-time">${esc(it.role_label)} ${src} ${esc(it.time)}</span>
      </div>
      <div class="rx-line">${detail}</div>
      <div class="rx-line">수신 ${esc(it.raw_hex)}</div>
    </div>`;
  }).join('');
}

// ---- 2번 구간 ----
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
}

const NOTICE_CLASS = {
  parked_ok:        'ok',
  misparked:        'bad',
  mispark_unknown:  'warn',
  relocate_failed:  'warn',
};

function renderNotice(ev){
  const box = document.getElementById('notice');
  if (!ev){
    box.className = '';
    box.innerHTML = '<div class="nt-idle">주차 완료를 기다리는 중입니다.</div>';
    return;
  }
  box.className = NOTICE_CLASS[ev.kind] || '';
  const where = ev.spot_id ? `${esc(ev.spot_id)}` : '위치 미확인';
  box.innerHTML =
    `<div class="nt-head"><span>차량 ${esc(ev.car_id)} · ${where}</span>
       <span>${esc(ev.time)}</span></div>` +
    ev.lines.map(l => `<div class="nt-line">${esc(l)}</div>`).join('') +
    (ev.note ? `<div class="nt-note">${esc(ev.note)}</div>` : '');
}

// ---- 3번 구간 ----
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
    renderLot(st);
    renderNotice(st.latest_event);
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
    """최종 화면 HTML을 반환."""
    return FINAL_UI_HTML.replace("__POLL_MS__", str(CONFIG['POLL_INTERVAL_MS']))


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
