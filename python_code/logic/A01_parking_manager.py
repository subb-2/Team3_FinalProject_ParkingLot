import sys
import os
# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from data.car_data import cars_info, spot_status, get_empty_spots
from logic.fee_calculator import calculate_fee

# 통합 입차 처리
def find_sequential_empty_spot():
    """알파벳 및 숫자 오름차순으로 가장 앞선 빈자리를 반환."""
    # "A-1", "A-2"... "B-1", "B-2"... "C-1", "C-2"... "D-1", "D-2"... 순으로 정렬 후 첫 번째 반환

    # 빈자리 우선 탐색
    empty_spots = get_empty_spots()
    if not empty_spots:
        return None
    
    # 빈자리 안에서 "A-1", "A-2"... 순으로 정렬 후 첫 번째 반환
    return sorted(list(empty_spots))[0]

def handle_car_entry(car_id, receive_time):
    """
    차량 번호 수신 시 자동으로 빈자리를 찾아 입차를 처리.
    """
    print(f"\n[입차 요청] 차량 번호: {car_id} 수신됨")
    spot_id = find_sequential_empty_spot()
    if not spot_id:
        print("[경고] 주차장이 만차입니다. 입차할 수 없습니다.")
        return False
    
    return park_car(spot_id, car_id, entry_time=receive_time)

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
        "entry_time": entry_time
    }
    # 해당 구역 상태 업데이트
    spot_status[spot_id] = "full"
    
    print(f"[입차 완료] 구역: {spot_id} | 차량번호: {car_id} | 입차시간: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
