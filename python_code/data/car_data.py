from data.map_data import spot_status, SPOT1, SPOT2, SPOT3, SPOT4, SPOT_TYPE_NAME

# 주차 차량 데이터 관리
# 순수 데이터 저장소(차량 정보, 차량 종류 등록부, 요금 설정)만 관리합니다.

# 입출차 처리 로직은 logic/parking_manager.py에서 처리.
# 요금 계산 로직은 logic/fee_calculator.py에서 처리.

# =====================================================================
# 차량 종류 등록부
# =====================================================================
# 번호판마다 차량 종류를 미리 등록해 둔다.
# 입구에서 번호가 넘어오면 이 표를 보고 '그 종류가 댈 수 있는 구역' 중에서
# 입구에 가장 가까운 빈자리를 배정한다. (logic/A01_parking_manager.py)
#
# 예) "9999"(대형차)가 인식되면 -> 대형 구역(SPOT3) 중 가장 가까운 빈자리로 안내
#     대형 구역이 전부 차 있으면 -> 배정 실패. "자리 없음"으로 안내한다.
#     (일반 자리로 대신 보내지 않는다. 종류가 맞아야 하는 자리들이기 때문)

# 차량 종류 상수.
# 값이 그대로 화면/로그에 찍히므로 한글 문자열을 쓴다.
CAR_NORMAL   = "일반"
CAR_HANDICAP = "장애인"
CAR_LARGE    = "대형"
CAR_EV       = "전기차"

# 차량 종류 -> 그 차가 댈 수 있는 주차 구역 종류
# 한 종류가 여러 구역에 댈 수 있게 하려면 값을 리스트로 바꾸고
# A01의 find_spot_for_car도 함께 고칠 것. 지금은 1:1이다.
CAR_TYPE_TO_SPOT = {
    CAR_NORMAL:   SPOT1,
    CAR_HANDICAP: SPOT2,
    CAR_LARGE:    SPOT3,
    CAR_EV:       SPOT4,
}

# 등록되지 않은 번호판이 들어왔을 때 적용할 기본 종류.
DEFAULT_CAR_TYPE = CAR_NORMAL

# 번호판 -> 차량 종류.
# 여기에 없는 번호는 DEFAULT_CAR_TYPE(일반)으로 처리된다.
# 차량 번호는 반드시 4자리 '문자열'이어야 한다. (앞자리 0이 사라지므로 정수 금지)
car_types = {
    "1234": CAR_EV,
    "9999": CAR_LARGE,
    "0828": CAR_HANDICAP,
    "1998": CAR_NORMAL,
}

# 등록부 검증 (오타로 조용히 일반 차량이 되는 것을 막는다)
for _car_id, _type in car_types.items():
    if not isinstance(_car_id, str):
        print(f"[경고] car_types의 번호판 {_car_id!r}이 문자열이 아닙니다. "
              f"4자리 문자열로 적을 것. (예: \"0828\")")
    if _type not in CAR_TYPE_TO_SPOT:
        print(f"[경고] car_types['{_car_id}']의 종류 '{_type}'을 알 수 없습니다. "
              f"사용 가능: {list(CAR_TYPE_TO_SPOT)}")


def get_car_type(car_id):
    """
    번호판에 등록된 차량 종류를 반환. 등록되지 않았으면 DEFAULT_CAR_TYPE.

    Args:
        car_id: 차량 번호 4자리 문자열

    Returns:
        차량 종류 문자열 (예: "전기차")
    """
    return car_types.get(car_id, DEFAULT_CAR_TYPE)


def get_required_spot_type(car_id):
    """
    번호판에 맞는 주차 구역 종류(SPOT1~SPOT4)를 반환.

    Args:
        car_id: 차량 번호 4자리 문자열

    Returns:
        구역 종류 상수. 매핑이 없으면 None.
    """
    return CAR_TYPE_TO_SPOT.get(get_car_type(car_id))


def describe_car(car_id):
    """차량 종류와 필요한 구역 종류를 사람이 읽을 문자열로 반환. (로그/UI용)"""
    car_type = get_car_type(car_id)
    spot_type = CAR_TYPE_TO_SPOT.get(car_type)
    registered = "등록" if car_id in car_types else "미등록->기본"
    return f"{car_type}({registered}) / {SPOT_TYPE_NAME.get(spot_type, '?')} 구역"

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
