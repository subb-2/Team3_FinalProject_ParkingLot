import sys
import os
import math
# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from data.car_data import (
    cars_info, spot_status, get_empty_spots,
    get_car_type, get_required_spot_type, describe_car,
)
from data.map_data import (
    spot_type as SPOT_TYPE_OF, SPOT_TYPE_NAME, SPOT_PRIORITY, get_spot_ids_by_type,
)
from logic.C02_lot_layout import SPOT_WORLD_POS, GATE1_WORLD_POS
from logic.fee_calculator import calculate_fee

# 배정 실패 사유 상수 (나중에 UI가 이 값으로 분기한다)
REASON_OK           = "ok"            # 배정 성공
REASON_FULL         = "full"          # 그 종류의 자리가 전부 찼음
REASON_NO_SPOT_TYPE = "no_spot_type"  # 그 종류의 자리가 주차장에 아예 없음
REASON_UNKNOWN_TYPE = "unknown_type"  # 차량 종류 -> 구역 종류 매핑이 없음
REASON_ALREADY      = "already"       # 이미 주차 중인 차량번호

# 가장 최근 입차 시도 결과. (UI/모니터링이 읽어가는 자리)
# handle_car_entry가 매번 갱신한다. "자리 없음" 안내 화면은 이 값을 보면 된다.
last_entry_result = None


def _distance_from_gate(spot_id):
    """입구(GATE1)에서 해당 구역까지의 직선 거리(cm). 좌표가 없으면 무한대."""
    pos = SPOT_WORLD_POS.get(spot_id)
    if pos is None:
        return float('inf')
    return math.hypot(pos[0] - GATE1_WORLD_POS[0], pos[1] - GATE1_WORLD_POS[1])


# 통합 입차 처리
def get_assign_order(cell_type, distance_fn=None):
    """
    구역 종류의 배정 순서를 반환.

    map_data.SPOT_PRIORITY에 적어둔 고정 순서가 우선이고, 거기에 없는 자리는
    뒤에 붙인다(입구 직선거리 순). 배치를 바꿔 새 자리가 생겨도 배정이 멈추지
    않게 하기 위한 안전장치다.

    Args:
        cell_type:   구역 종류 상수 (SPOT1~SPOT4)
        distance_fn: 목록에 없는 자리를 정렬할 거리 함수

    Returns:
        구역 ID 리스트 (배정할 순서대로)
    """
    distance_fn = _distance_from_gate if distance_fn is None else distance_fn

    actual = set(get_spot_ids_by_type(cell_type))
    ordered = [s for s in SPOT_PRIORITY.get(cell_type, []) if s in actual]
    leftover = sorted(actual - set(ordered), key=lambda s: (distance_fn(s), s))
    return ordered + leftover


def find_spot_for_car(car_id, distance_fn=None):
    """
    차량 번호에 맞는 종류의 빈자리 중 우선순위가 가장 앞선 자리를 찾는다.

    번호판별 차량 종류는 data/car_data.py의 car_types에 미리 등록해 둔다.
    등록되지 않은 번호는 DEFAULT_CAR_TYPE(일반)으로 처리된다.

    배정 순서는 data/map_data.py의 SPOT_PRIORITY에 종류별로 고정해 두었다.
    입구에서의 직선거리로 자동 정렬하지 않는 이유는 그 주석에 적어두었다.

    종류가 맞지 않는 자리에는 배정하지 않는다. 대형차를 일반 자리에 넣거나
    일반차를 장애인 자리에 넣으면 안 되기 때문이다. 그래서 해당 종류가 다 차면
    다른 종류로 대체하지 않고 실패를 돌려준다.

    Args:
        car_id:      차량 번호 4자리 문자열
        distance_fn: SPOT_PRIORITY에 없는 자리를 정렬할 때 쓸 거리 함수.
                     None이면 입구에서의 직선거리.

    Returns:
        (spot_id, reason) 튜플. 실패하면 spot_id가 None이고 reason이 사유.
    """
    required = get_required_spot_type(car_id)
    if required is None:
        return None, REASON_UNKNOWN_TYPE

    # 이 종류로 지정된 구역이 주차장에 하나라도 있는가
    order = get_assign_order(required, distance_fn)
    if not order:
        return None, REASON_NO_SPOT_TYPE

    empty = get_empty_spots()
    for spot_id in order:
        if spot_id in empty:
            return spot_id, REASON_OK

    return None, REASON_FULL


def handle_car_entry(car_id, receive_time):
    """
    차량 번호 수신 시 차량 종류에 맞는 가장 가까운 빈자리를 찾아 입차를 처리.

    Args:
        car_id:       차량 번호 4자리 문자열
        receive_time: 번호를 수신한 시각

    Returns:
        결과 딕셔너리. 호출 측은 "success"로 성공 여부를 판단하고,
        실패 시 "reason"으로 안내 문구를 고르면 된다.
        {
            "success": False,
            "car_id": "9999",
            "car_type": "대형",
            "spot_id": None,
            "reason": "full",
            "message": "대형 구역이 모두 찼습니다. (6/6)",
        }
    """
    global last_entry_result

    car_type = get_car_type(car_id)
    print(f"\n[입차 요청] 차량 번호: {car_id} 수신됨  ({describe_car(car_id)})")

    def _result(success, spot_id, reason, message):
        global last_entry_result
        last_entry_result = {
            "success": success,
            "car_id": car_id,
            "car_type": car_type,
            "spot_id": spot_id,
            "reason": reason,
            "message": message,
            "time": receive_time,
        }
        if not success:
            print(f"[입차 거부] {message}")
        return last_entry_result

    if car_id in cars_info:
        return _result(False, cars_info[car_id]["spot_id"], REASON_ALREADY,
                       f"차량 '{car_id}'는 이미 {cars_info[car_id]['spot_id']}에 주차 중입니다.")

    spot_id, reason = find_spot_for_car(car_id)

    if reason == REASON_UNKNOWN_TYPE:
        return _result(False, None, reason,
                       f"차량 종류 '{car_type}'에 해당하는 구역 종류가 없습니다. "
                       f"car_data.CAR_TYPE_TO_SPOT를 확인하세요.")

    if reason == REASON_NO_SPOT_TYPE:
        return _result(False, None, reason,
                       f"{car_type} 차량이 댈 수 있는 구역이 주차장에 없습니다.")

    if reason == REASON_FULL:
        required = get_required_spot_type(car_id)
        total = len(get_spot_ids_by_type(required))
        kind = SPOT_TYPE_NAME.get(required, car_type)
        return _result(False, None, reason,
                       f"{kind} 구역이 모두 찼습니다. ({total}/{total})")

    park_car(spot_id, car_id, entry_time=receive_time)
    distance = _distance_from_gate(spot_id)
    return _result(True, spot_id, REASON_OK,
                   f"{car_type} 차량 '{car_id}' -> {spot_id} 배정 (입구에서 {distance:.0f}cm)")


def get_availability_by_type():
    """
    구역 종류별 빈자리/전체 수를 반환. ("자리 없음" 안내 화면용)

    Returns:
        {구역종류상수: {"name": "대형", "empty": 2, "total": 6, "spots": [...]}}
    """
    empty = get_empty_spots()
    summary = {}
    for cell_type, name in SPOT_TYPE_NAME.items():
        ids = get_spot_ids_by_type(cell_type)
        if not ids:
            continue
        empty_ids = sorted(s for s in ids if s in empty)
        summary[cell_type] = {
            "name": name,
            "empty": len(empty_ids),
            "total": len(ids),
            "spots": empty_ids,
        }
    return summary

# 개별 입차 처리
def park_car(spot_id, car_id, entry_time=None):
    """
    지정된 주차 구역에 차량을 입차 처리. (순수 데이터 업데이트 로직)
    
    Args:
        spot_id: 주차 구역 ID (예: "A-1")
        car_id:  차량 번호 4자리 문자열 (예: "1234")
        entry_time: 외부에서 전달받은 정확한 수신 시간
    """
    if entry_time is None:
        entry_time = datetime.now()

    # data -> car_data -> car_info 안에 차량 번호를 메인 키로 정보 저장
    cars_info[car_id] = {
        "spot_id": spot_id,
        "entry_time": entry_time,
        # 차량 종류도 함께 남긴다. 출차/요금/UI에서 다시 조회할 필요가 없어진다.
        "car_type": get_car_type(car_id),
    }
    # 해당 구역 상태 업데이트
    spot_status[spot_id] = "full"

    kind = SPOT_TYPE_NAME.get(SPOT_TYPE_OF.get(spot_id), "?")
    print(f"[입차 완료] 구역: {spot_id}({kind}) | 차량번호: {car_id}({get_car_type(car_id)}) "
          f"| 입차시간: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return True

# 출차 처리 및 요금 계산
def remove_car(car_id):
    """
    차량 번호로 출차 처리를 하고, 주차 요금을 계산하여 반환.
    
    Args:
        car_id: 차량 번호 4자리 문자열 (예: "1234")
    
    Returns:
        성공 시 (spot_id, fee, duration_minutes) 튜플 반환.
        해당 차량이 없으면 None 반환.
    """
    if car_id not in cars_info:
        print(f"[경고] 차량번호 '{car_id}'에 해당하는 주차 정보가 없습니다.")
        return None
    
    info = cars_info[car_id]
    spot_id = info["spot_id"]
    entry_time = info["entry_time"]
    exit_time = datetime.now()
    
    # 주차 시간 계산 (분 단위)
    duration = exit_time - entry_time
    duration_minutes = int(duration.total_seconds() / 60)
    
    # 요금 계산
    fee = calculate_fee(duration_minutes)
    
    print(f"[출차 완료] 구역: {spot_id} | 차량번호: {car_id}")
    print(f"           입차: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"           출차: {exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"           주차시간: {duration_minutes}분 | 요금: {fee:,}원")
    
    # 저장소에서 제거 및 상태 변경
    del cars_info[car_id]
    spot_status[spot_id] = "empty"
    
    return spot_id, fee, duration_minutes

def print_all_parked():
    """현재 주차 중인 모든 차량 정보를 출력합니다."""
    
    if not cars_info:
        print("[알림] 현재 주차된 차량이 없습니다.")
        return
    
    print("\n========== 주차 현황 ==========")
    print(f"{'구역':^6} | {'차량번호':^8} | {'입차시간':^20}")
    print("-" * 42)
    # 구역 이름순 정렬을 위해 리스트 생성 후 출력
    sorted_cars = sorted(cars_info.items(), key=lambda x: x[1]['spot_id'])
    for car_id, info in sorted_cars:
        spot_id = info['spot_id']
        entry_str = info['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"{spot_id:^6} | {car_id:^8} | {entry_str}")
    print("================================\n")


# =====================================================================
# 테스트 (단독 실행 시)
# =====================================================================
if __name__ == "__main__":
    from data.car_data import get_occupied_spots, get_car_info
    
    # 입차 테스트
    park_car("A-1", "1234")
    park_car("C-2", "5678")
    
    # 현황 출력
    print_all_parked()
    
    # 주차 중인 구역 확인
    print(f"주차 중인 구역: {get_occupied_spots()}")
    
    # 특정 차량 정보 조회
    info = get_car_info("1234")
    if info:
        print(f"차량 '1234' 정보: 구역={info['spot_id']}, 입차시간={info['entry_time']}")
    
    # 요금 계산 테스트 (60분 주차 시)
    test_fee = calculate_fee(60)
    print(f"\n60분 주차 요금 테스트: {test_fee:,}원")
