from data.map_data import spot_status, SPOT1, SPOT2, SPOT3, SPOT4, SPOT_TYPE_NAME

# 주차 차량 데이터 관리
# 순수 데이터 저장소(차량 정보, 차량 종류 등록부, 요금 설정)만 관리합니다.

# 입출차 처리 로직은 logic/parking_manager.py에서 처리.
# 요금 계산 로직은 logic/fee_calculator.py에서 처리.

# =====================================================================
# 시연 전에 여기만 고치면 된다 (차량 관련 초기 설정 3종)
# =====================================================================
#   1) car_types               번호판 -> 차량 종류      (어느 구역에 댈지 결정)
#   2) INITIAL_PARKED          시작부터 세워둘 차량      (자리 + 차량정보 + 상태)
#   3) TEST_PRESET_CAR_NUMBERS UART 없이 테스트할 번호   (순서대로 입차 처리)
#
# 주차장 배치(자리 위치/종류/배정 순서)는 data/map_data.py에서 관리한다.
# =====================================================================

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
    "1998": CAR_EV,
    "0828": CAR_HANDICAP,
    "1111": CAR_NORMAL,
    "2222": CAR_NORMAL,
    "3333": CAR_NORMAL,
    "4444": CAR_NORMAL,
    "5555": CAR_NORMAL,
    "6666": CAR_NORMAL,
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


# =====================================================================
# 미리 주차되어 있는 차량 (시작 상태 세팅)
# =====================================================================
# 시연을 시작할 때 이미 몇 자리에 차를 세워둔 상태로 두려면 여기에 적는다.
# 프로그램이 뜰 때 아래 세 가지를 한 번에 채운다.
#   1) spot_status[구역] = "full"   -> 그 자리가 배정 후보에서 빠진다
#   2) cars_info[차량번호]          -> 출차/요금 계산이 가능해진다
#   3) 차량 종류                    -> car_types에 등록된 값을 따른다
#
# map_data.INITIAL_OCCUPIED와는 다르다. 그쪽은 '차 없이 자리만 막아두는' 용도라
# 차량 정보가 없어서 출차가 되지 않는다. 실제로 차를 세워둔 거라면 여기에 적을 것.
#
# 적는 방법 두 가지:
#   "A-1": "1234"          입차 시각 = 프로그램 시작 시각
#   "D-1": ("9999", 45)    45분 전부터 주차 중이던 것으로 처리 (요금 테스트용)
#
# 구역 종류와 차량 종류가 맞지 않으면 경고를 띄운다. (예: 일반 자리에 대형차)
# 실물로는 세워둘 수 있으므로 막지는 않고 알려만 준다.
#
# ---------------------------------------------------------------------
# 자리 목록 (data/map_data.py의 grid_map에서 자동 생성. 배치를 바꾸면 함께 바뀐다)
#
#   구역ID   격자(row,col)      실좌표(cm)      종류      배정순위
#   ------------------------------------------------------------------
#   "A-1"    (1, 0)            (  0.0,  17.5)  일반       7
#   "A-2"    (2, 0)            (  0.0,  35.0)  일반       6
#   "A-3"    (4, 0)+(5, 0)     (  0.0,  78.8)  대형       1   <- 2칸이 1자리
#   "A-4"    (7, 0)            (  0.0, 122.5)  장애인     1
#   "B-1"    (3, 5)            ( 50.0,  52.5)  전기차     4
#   "B-2"    (4, 5)            ( 50.0,  70.0)  전기차     2
#   "C-1"    (3, 7)            ( 70.0,  52.5)  전기차     3
#   "C-2"    (4, 7)            ( 70.0,  70.0)  전기차     1
#   "D-1"    (1,12)            (120.0,  17.5)  일반       5
#   "D-2"    (2,12)            (120.0,  35.0)  일반       4
#   "D-3"    (4,12)            (120.0,  70.0)  일반       3
#   "D-4"    (5,12)            (120.0,  87.5)  일반       2
#   "D-5"    (7,12)            (120.0, 122.5)  일반       1
#
#   배정순위 = 같은 종류 안에서 빈자리를 고르는 순서 (map_data.SPOT_PRIORITY)
#   입구 GATE1 = 격자(8, 8) = (80.0, 140.0)cm  /  출구 GATE2 = 격자(8, 3) = (30.0, 140.0)cm
#
#   격자 그림 (A=왼쪽열 c0, B=중앙섬 c5, C=중앙섬 c7, D=오른쪽열 c12)
#         c0    c5    c7   c12
#    r1  A-1               D-1
#    r2  A-2               D-2
#    r3        B-1   C-1
#    r4  A-3 ┐ B-2   C-2   D-3
#    r5  A-3 ┘             D-4      A-3 = 대형 1자리 (2칸)
#    r7  A-4(장애인)        D-5
#    r8         출구 c3        입구 c8
# ---------------------------------------------------------------------
INITIAL_PARKED = {
    "A-1": "1111",
    "A-2": "2222",
    "B-1": "1234",
    "B-2": "1998",
}


# =====================================================================
# 테스트용 입차 차량 (UART 없이 돌릴 때)
# =====================================================================
# Zybo UART를 연결하지 않고 B_main / C_main / D_main을 돌릴 때, 여기 적은
# 번호가 순서대로 입차 처리된다. 실제 UART로 번호가 들어오는 것과 같은 경로를
# 타므로(차량 종류 조회 -> 자리 배정 -> FIFO 등록) 동작 확인에 그대로 쓸 수 있다.
#
# 각 main의 ENABLE_UART가 False일 때만 쓰인다.
#
# 주의: 여기 적은 번호는 위 car_types에도 등록해 두는 것이 좋다.
#       등록하지 않으면 전부 DEFAULT_CAR_TYPE(일반)으로 처리되어
#       대형/전기차 자리로 가는 시나리오를 확인할 수 없다.
#       (등록되지 않은 번호가 있으면 아래에서 경고를 출력한다)
TEST_PRESET_CAR_NUMBERS = ["1234", "1998", "1111", "2222", "3333", "4444"]

for _car_id in TEST_PRESET_CAR_NUMBERS:
    if _car_id not in car_types:
        print(f"[경고] TEST_PRESET_CAR_NUMBERS의 '{_car_id}'가 car_types에 없습니다. "
              f"'{DEFAULT_CAR_TYPE}' 차량으로 처리됩니다.")


def _apply_initial_parked():
    """INITIAL_PARKED를 cars_info와 spot_status에 반영. (import 시 1회)"""
    from datetime import datetime, timedelta
    from data.map_data import spot_map, spot_type as spot_type_of

    now = datetime.now()
    for spot_id, entry in INITIAL_PARKED.items():
        # ("9999", 45) 형태면 경과 시간(분)을 분리
        if isinstance(entry, (tuple, list)):
            car_id, elapsed_min = entry[0], entry[1]
        else:
            car_id, elapsed_min = entry, 0

        if spot_id not in spot_map:
            print(f"[경고] INITIAL_PARKED의 '{spot_id}'는 현재 배치에 없는 구역입니다. (무시)")
            continue
        if not isinstance(car_id, str):
            print(f"[경고] INITIAL_PARKED['{spot_id}']의 차량번호 {car_id!r}가 문자열이 아닙니다. "
                  f"4자리 문자열로 적을 것.")
            continue
        if car_id in cars_info:
            print(f"[경고] 차량번호 '{car_id}'가 INITIAL_PARKED에 중복되었습니다. (무시)")
            continue

        # 구역 종류와 차량 종류가 맞는지 확인 (막지는 않고 경고만)
        required = get_required_spot_type(car_id)
        actual = spot_type_of.get(spot_id)
        if required is not None and actual is not None and required != actual:
            print(f"[경고] '{spot_id}'는 {SPOT_TYPE_NAME.get(actual)} 구역인데 "
                  f"'{car_id}'는 {get_car_type(car_id)} 차량입니다. "
                  f"(그대로 진행하지만 배정 규칙과 어긋납니다)")

        cars_info[car_id] = {
            "spot_id": spot_id,
            "entry_time": now - timedelta(minutes=elapsed_min),
            "car_type": get_car_type(car_id),
            # 이미 자리에 세워져 있는 차다. 안내 대상이 아니며, B02가 그 자리에
            # 서 있는 트랙을 찾아 이 차량번호로 묶는다. (아래 parked 필드 설명 참고)
            "parked": True,
        }
        spot_status[spot_id] = "full"

        elapsed_note = f", {elapsed_min}분 경과" if elapsed_min else ""
        print(f"[초기 주차] {spot_id}({SPOT_TYPE_NAME.get(actual, '?')}) "
              f"<- '{car_id}'({get_car_type(car_id)}){elapsed_note}")

# 실제 적용은 cars_info가 만들어진 뒤(파일 맨 아래)에 한다.

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
#     "1234": {"spot_id": "A-1", "entry_time": ..., "car_type": "전기차", "parked": True},
#     "5678": {"spot_id": "B-2", "entry_time": ..., "car_type": "일반",  "parked": False},
# }
#
# parked 필드의 의미:
#   True  = 그 자리에 실제로 세워져 있다. 안내가 끝난 차 또는 INITIAL_PARKED로
#           미리 세워둔 차. B02가 그 자리에 있는 트랙을 찾아 차량번호를 묶는다.
#   False = 자리는 배정됐지만 아직 가는 중이다. (입구에서 번호만 받은 상태)
#
# 이 구분이 없으면 '이미 주차된 차'와 '안내 중인 차'를 구별할 수 없어서,
# 미리 세워둔 차가 안내 대상으로 잡히거나 그 반대가 된다.
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


# 미리 주차되어 있는 차량을 반영한다.
# cars_info가 만들어진 뒤여야 하므로 파일 맨 아래에서 호출한다.
_apply_initial_parked()
