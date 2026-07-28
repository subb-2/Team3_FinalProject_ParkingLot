from data.map_data import spot_status

# 주차 차량 데이터 관리
# 순수 데이터 저장소(차량 정보, 요금 설정)만 관리합니다.

# 입출차 처리 로직은 logic/parking_manager.py에서 처리.
# 요금 계산 로직은 logic/fee_calculator.py에서 처리.

# 요금 설정
def get_fee_config():
    """
    주차 요금 관련 설정값을 반환.
    """
    return {
        'base_minutes': 30,       # 기본 요금 적용 시간 (분)
        'base_fee': 1000,         # 기본 요금 (원)
        'extra_per_minutes': 10,  # 추가 요금 단위 시간 (분)
        'extra_fee': 500,         # 추가 요금 단위 금액 (원)
        'min_time' : 10,          # 기본 시간
        'max_fee' : 50_000        # 하루 최대 요금
    }

# 차량 정보 저장소
# 차량 번호 -> 차량 관리 정보 (딕셔너리 구조)
# 예시
# {
#     "1234": {"spot_id": "A-1", "entry_time": datetime 객체},
#     "5678": {"spot_id": "B-2", "entry_time": datetime 객체},
#     ...
# }
cars_info = {}

# 조회 함수
def get_occupied_spots():
    """현재 주차 중인 구역 ID의 set을 반환합니다."""
    return {spot_id for spot_id, status in spot_status.items() if status == "full"}

def get_empty_spots():
    """현재 비어있는 구역 ID의 set을 반환합니다."""
    return {spot_id for spot_id, status in spot_status.items() if status == "empty"}

def get_car_info(car_id):
    """
    차량 번호로 주차 정보를 조회합니다.
    
    Args:
        car_id: 차량 번호 4자리 문자열
    
    Returns:
        주차 정보 딕셔너리 또는 None
    """
    return cars_info.get(car_id, None)

def get_spot_info(spot_id):
    """
    주차 구역 ID로 해당 구역의 정보를 조회합니다.
    (공간 중심으로 조회할 때 사용)
    
    Args:
        spot_id: 주차 구역 ID (예: "A-1")
    
    Returns:
        주차 정보 딕셔너리 ({"car_id": "1234", "entry_time": ...}) 또는 None
    """
    if spot_status.get(spot_id) == "full":
        for c_id, info in cars_info.items():
            if info["spot_id"] == spot_id:
                return {"car_id": c_id, "entry_time": info["entry_time"]}
    return None
