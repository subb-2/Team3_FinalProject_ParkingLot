from datetime import datetime

# =====================================================================
# 요금 설정
# =====================================================================
def get_fee_config():
    """
    주차 요금 관련 설정값을 반환합니다.
    필요에 따라 여기서 요금 체계를 수정하세요.
    """
    return {
        'base_minutes': 30,       # 기본 요금 적용 시간 (분)
        'base_fee': 1000,         # 기본 요금 (원)
        'extra_per_minutes': 10,  # 추가 요금 단위 시간 (분)
        'extra_fee': 500,         # 추가 요금 단위 금액 (원)
    }

# =====================================================================
# 차량 정보 저장소
# =====================================================================
# 주차 구역 ID -> 차량 상세 정보 딕셔너리
# {
#     "A-1": {"car_id": "1234", "entry_time": datetime 객체},
#     ...
# }
parked_cars = {}

# 차량 번호 -> 주차 구역 ID (역방향 검색용)
car_to_spot = {}

# =====================================================================
# 입차 처리
# =====================================================================
def park_car(spot_id, car_id):
    """
    지정된 주차 구역에 차량을 입차 처리합니다.
    
    Args:
        spot_id: 주차 구역 ID (예: "A-1")
        car_id:  차량 번호 4자리 문자열 (예: "1234")
    
    Returns:
        성공 시 True, 이미 주차된 구역이면 False
    """
    if spot_id in parked_cars:
        print(f"[경고] {spot_id} 구역은 이미 주차 중입니다. (차량: {parked_cars[spot_id]['car_id']})")
        return False
    
    entry_time = datetime.now()
    parked_cars[spot_id] = {
        "car_id": car_id,
        "entry_time": entry_time
    }
    car_to_spot[car_id] = spot_id
    
    print(f"[입차 완료] 구역: {spot_id} | 차량번호: {car_id} | 입차시간: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return True

# =====================================================================
# 출차 처리 및 요금 계산
# =====================================================================
def remove_car(car_id):
    """
    차량 번호로 출차 처리를 하고, 주차 요금을 계산하여 반환합니다.
    
    Args:
        car_id: 차량 번호 4자리 문자열 (예: "1234")
    
    Returns:
        성공 시 (spot_id, fee, duration_minutes) 튜플 반환.
        해당 차량이 없으면 None 반환.
    """
    if car_id not in car_to_spot:
        print(f"[경고] 차량번호 '{car_id}'에 해당하는 주차 정보가 없습니다.")
        return None
    
    spot_id = car_to_spot[car_id]
    entry_time = parked_cars[spot_id]["entry_time"]
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
    
    # 저장소에서 제거
    del parked_cars[spot_id]
    del car_to_spot[car_id]
    
    return spot_id, fee, duration_minutes

def calculate_fee(duration_minutes):
    """
    주차 시간(분)을 기반으로 요금을 계산합니다.
    
    Args:
        duration_minutes: 주차 시간 (분 단위, 정수)
    
    Returns:
        계산된 주차 요금 (원, 정수)
    """
    config = get_fee_config()
    
    if duration_minutes <= config['base_minutes']:
        return config['base_fee']
    
    # 기본 시간 초과분에 대한 추가 요금 계산
    extra_minutes = duration_minutes - config['base_minutes']
    # 올림 처리: 10분 단위로 올림하여 추가 요금 부과
    extra_units = (extra_minutes + config['extra_per_minutes'] - 1) // config['extra_per_minutes']
    total_fee = config['base_fee'] + (extra_units * config['extra_fee'])
    
    return total_fee

# =====================================================================
# 조회 함수
# =====================================================================
def get_occupied_spots():
    """현재 주차 중인 구역 ID의 set을 반환합니다."""
    return set(parked_cars.keys())

def get_car_info(car_id):
    """
    차량 번호로 주차 정보를 조회합니다.
    
    Args:
        car_id: 차량 번호 4자리 문자열
    
    Returns:
        주차 정보 딕셔너리 또는 None
    """
    if car_id in car_to_spot:
        spot_id = car_to_spot[car_id]
        info = parked_cars[spot_id].copy()
        info["spot_id"] = spot_id
        return info
    return None

def get_spot_info(spot_id):
    """
    주차 구역 ID로 해당 구역의 정보를 조회합니다.
    
    Args:
        spot_id: 주차 구역 ID (예: "A-1")
    
    Returns:
        주차 정보 딕셔너리 또는 None (비어있는 경우)
    """
    return parked_cars.get(spot_id, None)

def print_all_parked():
    """현재 주차 중인 모든 차량 정보를 출력합니다."""
    if not parked_cars:
        print("[알림] 현재 주차된 차량이 없습니다.")
        return
    
    print("\n========== 주차 현황 ==========")
    print(f"{'구역':^6} | {'차량번호':^8} | {'입차시간':^20}")
    print("-" * 42)
    for spot_id in sorted(parked_cars.keys()):
        info = parked_cars[spot_id]
        entry_str = info['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"{spot_id:^6} | {info['car_id']:^8} | {entry_str}")
    print("================================\n")


# =====================================================================
# 테스트 (단독 실행 시)
# =====================================================================
if __name__ == "__main__":
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
