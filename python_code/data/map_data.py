# 주차장 맵 데이터 정의
# 순수 데이터(그리드, 좌표, 상수)만 관리.
#   - 실좌표(cm) 변환 : logic/C02_lot_layout.py
#   - 경로 계산       : logic/C01_path_planner.py
#
# =====================================================================
# 이 파일에서 사람이 손으로 고치는 것은 딱 두 가지다.
#   1) grid_map      : 주차장 배치 (설계도)
#   2) PILL_MARKER_ID: 기둥에 실제로 붙인 ArUco 마커 ID (물리 정보라 자동화 불가)
#
# spot_map / coord_to_spot / spot_type / spot_status / GATE1_POS / GATE2_POS는
# 전부 grid_map에서 자동으로 만들어진다. 예전에는 이것들을 따로 적어두었는데,
# grid_map만 고치고 나머지를 안 고쳐서 조용히 어긋나는 사고가 났다.
# 배치를 바꾸려면 grid_map만 고치고 C02_lot_layout.py를 단독 실행해 검증할 것.
# =====================================================================

# 셀 타입 상수 정의
ROAD  = 0  # 도로 (차량이 이동 가능한 통로)
PILL = 1 #기둥
GATE1  = 2  # 입구
GATE2 = 3 # 출구
SPOT1  = 4  # 일반 주차 구역 (빈자리/주차중 구분)
SPOT2  = 5  # 장애인 주차 구역 (빈자리/주차중 구분)
SPOT3  = 6  # 대형 주차 구역 (빈자리/주차중 구분)
SPOT4  = 7  # 전기차 주차 구역 (빈자리/주차중 구분)


# 주차 구역으로 취급할 셀 타입 전체.
# 주차 구역 종류가 늘어나면 상수를 추가하고 이 집합과 SPOT_TYPE_NAME에도 넣을 것.
# 코드에서 '주차 구역인가'를 판정할 때는 반드시 이 집합을 쓴다.
# (cell == SPOT1 처럼 하나만 비교하면 나머지 종류가 통째로 누락된다)
SPOT_CELLS = {SPOT1, SPOT2, SPOT3, SPOT4}

# 주차 구역 종류 이름 (UI 표시 / 로그용)
SPOT_TYPE_NAME = {
    SPOT1: "일반",
    SPOT2: "장애인",
    SPOT3: "대형",
    SPOT4: "전기차",
}

# 주차 구역 '한 자리'가 차지하는 격자 칸 수 (세로 방향).
#
# 대형차는 차체가 길어서 한 자리가 세로 2칸을 차지한다. 격자에는 SPOT3가
# 2칸으로 그려져 있지만 실제로는 '1자리'다.
# 이 표에 따라 같은 열에서 연속된 같은 종류 칸들을 자리 하나로 묶는다.
#
# 예) col 12의 row 1, 2가 모두 SPOT3 -> 두 칸을 묶어 대형 1자리
#     col 0의 row 1, 2가 모두 SPOT1  -> span이 1이므로 각각 별개 자리
#
# 종류를 추가하면 여기에도 넣을 것. 없으면 1로 취급한다.
SPOT_CELL_SPAN = {
    SPOT1: 1,
    SPOT2: 1,
    SPOT3: 2,   # 대형차 1자리 = 세로 2칸
    SPOT4: 1,
}


# 주차장 그리드 맵 (9행 x 13열)
# 구조:
#   - 바깥 양쪽 열(col 0 = 왼쪽, col 12 = 오른쪽)에 주차 구역이 늘어서 있다.
#   - 가운데 col 5 / col 7은 중앙 섬 두 개. 각각 기둥 2개 사이에 전기차 구역 2칸.
#   - 그 사이(col 1~3, 5~6, 8~10)와 맨 아랫줄(row 8)이 전부 도로다.
#   - 입구(GATE1)는 아래 오른쪽(row 8, col 8), 출구(GATE2)는 아래 왼쪽(row 8, col 3).
#     입구로 들어와 안쪽을 돌고 왼쪽 아래 출구로 빠지는 흐름이 된다.
#
# 기둥과 주차 구역의 관계:
#   자리의 실좌표는 '같은 열에 있는 기둥들'을 기준으로 보간해서 구한다.
#   따라서 주차 구역이 있는 열에는 기둥이 최소 2개 있어야 한다.
#   기둥 사이에 자리가 몇 칸이 들어가든(현재는 2칸씩) 상관없고,
#   마지막 기둥 바깥의 자리(예: row 7)는 외삽으로 처리된다.
#   자세한 계산은 logic/C02_lot_layout.py의 build_spot_world_pos 참고.
# 열 배분은 실측에 맞춘 것이다. (기둥 중심 사이 거리, CELL_W_CM = 10cm 기준)
#   왼쪽 열(c0) <-> 왼쪽 섬(c5)    = 5칸 = 50cm
#   왼쪽 섬(c5) <-> 오른쪽 섬(c7)  = 2칸 = 20cm
#   오른쪽 섬(c7) <-> 오른쪽 열(c12) = 5칸 = 50cm
#   전체 폭 (c0 <-> c12)           = 12칸 = 120cm
# 실물 치수가 바뀌면 이 배분과 C02의 CELL_W_CM을 함께 고칠 것.
grid_map = [
    # col: 0       1       2       3       4       5       6       7       8       9      10      11      12
        [PILL   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,PILL],  # row 0
        [SPOT1  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,SPOT3],  # row 1
        [SPOT1  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,PILL   ,ROAD   ,PILL   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,SPOT3],  # row 2
        [PILL   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,SPOT4  ,ROAD   ,SPOT4  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,PILL],  # row 3
        [SPOT1  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,SPOT4  ,ROAD   ,SPOT4  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,SPOT1],  # row 4
        [SPOT1  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,PILL   ,ROAD   ,PILL   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,SPOT1],  # row 5
        [PILL   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,PILL],  # row 6
        [SPOT2  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,SPOT1],  # row 7
        [ROAD   ,ROAD   ,ROAD   ,GATE2  ,ROAD   ,ROAD   ,ROAD   ,ROAD   ,GATE1  ,ROAD   ,ROAD   ,ROAD   ,ROAD],  # row 8
]

# 기둥(PILL) -> ArUco 마커 ID 매핑
#
# !! 물리 정보 !! 기둥 칸에 '실제로 붙여 놓은' ArUco 마커의 ID다.
# 이 표만은 자동으로 만들 수 없다. 목업의 어느 기둥에 몇 번 마커를 붙였는지는
# 코드가 알 수 없기 때문이다. 마커를 옮기거나 다시 인쇄하면 반드시 함께 고칠 것.
#
# 이 표가 격자(설계도)와 실제 카메라 영상을 잇는 유일한 연결 고리다.
# 마커의 실좌표(cm)는 logic/C02_lot_layout.py가 이 표를 보고 만든다.
#
# 배정 규칙: '왼쪽 열 -> 중앙 섬 좌 -> 중앙 섬 우 -> 오른쪽 열',
#            각 열 안에서는 입출구에서 '먼' 쪽(row 0)부터 1, 2, 3...
#
# ---------------------------------------------------------------------
# 실물 부착표 (이 표대로 기둥에 마커를 붙일 것)
#
#   방향 기준: row 8이 입출구가 있는 줄이다. 즉 row 0이 입출구에서 가장 먼 쪽.
#             카메라가 입출구 쪽에서 본다면 화면 '위'가 row 0이다.
#
#   마커ID   기둥 격자     실좌표(cm)      위치
#   ------------------------------------------------------
#     1     (0,  0)      (  0.0,   0.0)   왼쪽 열   · 맨 위
#     2     (3,  0)      (  0.0,  52.5)   왼쪽 열   · 가운데
#     3     (6,  0)      (  0.0, 105.0)   왼쪽 열   · 맨 아래
#     4     (2,  5)      ( 50.0,  35.0)   중앙 섬 좌 · 위
#     5     (5,  5)      ( 50.0,  87.5)   중앙 섬 좌 · 아래
#     6     (2,  7)      ( 70.0,  35.0)   중앙 섬 우 · 위
#     7     (5,  7)      ( 70.0,  87.5)   중앙 섬 우 · 아래
#     8     (0, 12)      (120.0,   0.0)   오른쪽 열 · 맨 위
#     9     (3, 12)      (120.0,  52.5)   오른쪽 열 · 가운데
#    10     (6, 12)      (120.0, 105.0)   오른쪽 열 · 맨 아래
#
#          c0          c5      c7            c12
#    r0   (1)                                (8)      <- 입출구에서 먼 쪽
#    r2              (4)     (6)
#    r3   (2)                                (9)
#    r5              (5)     (7)
#    r6   (3)                                (10)
#    r8         출구 c3            입구 c8            <- 입출구가 있는 줄
#
#   실측 (기둥 중심 사이)
#     1 <-> 4   50cm    1 <-> 8   120cm
#     4 <-> 6   20cm    1 <-> 3   105cm
#     6 <-> 8   50cm
#
#   실좌표는 CELL_W_CM / CELL_H_CM(C02_lot_layout.CONFIG)에서 계산된 값이다.
#   10.0 x 17.5cm 기준으로 주차장이 120 x 105cm가 된다.
#   실제 목업 치수가 다르면 그 CONFIG와 위 열 배분을 함께 고칠 것.
#
#   11~13번 등 이 표에 없는 마커는 주차장에서 치울 것. 남아 있으면 검출은
#   되지만 호모그래피에 쓰이지 않아 화면에 빨간 점 + "11?"로 계속 뜬다.
# ---------------------------------------------------------------------
PILL_MARKER_ID = {
    (0, 0): 1,  (3, 0): 2,  (6, 0): 3,
    (2, 5): 4,  (5, 5): 5,
    (2, 7): 6,  (5, 7): 7,
    (0, 12): 8, (3, 12): 9, (6, 12): 10,
}

# 역방향 매핑: 마커 ID -> 격자 좌표
MARKER_ID_CELL = {mid: cell for cell, mid in PILL_MARKER_ID.items()}


# =====================================================================
# 주차 구역 자동 생성 (grid_map에서 유도)
# =====================================================================
# 구역 ID 규칙: "<구역문자>-<번호>"
#   구역문자 : 주차 구역이 있는 열을 왼쪽부터 A, B, C, D ... 로 매긴다.
#   번호     : 같은 열 안에서 위(row 0)부터 1, 2, 3 ...
#
# 현재 배치에서는 이렇게 나온다.
#   A = col 0 (왼쪽)   B = col 5 (중앙 섬 좌)
#   C = col 7 (중앙 섬 우)   D = col 12 (오른쪽)
#
# 주의: 열을 추가/삭제하면 구역 문자가 통째로 밀린다. (예전 A-1이 B-1이 된다)
#       주차 중인 차가 있는 상태로 배치를 바꾸지 말 것.
_ZONE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _build_spot_tables():
    """
    grid_map을 훑어 주차 구역 관련 표를 한 번에 만든다.

    Returns:
        (spot_map, coord_to_spot, spot_type)
        spot_map     : {구역ID: [(row, col), ...]}
        coord_to_spot: {(row, col): 구역ID}
        spot_type    : {구역ID: SPOT1..SPOT4}
    """
    # 주차 구역이 있는 열을 왼쪽부터 모은다
    spot_cols = sorted({
        col
        for row in range(len(grid_map))
        for col in range(len(grid_map[0]))
        if grid_map[row][col] in SPOT_CELLS
    })

    rows_n = len(grid_map)
    spot_map, coord_to_spot, spot_type = {}, {}, {}

    for zone_index, col in enumerate(spot_cols):
        letter = _ZONE_LETTERS[zone_index % len(_ZONE_LETTERS)]
        number = 0
        row = 0
        while row < rows_n:
            cell = grid_map[row][col]
            if cell not in SPOT_CELLS:
                row += 1
                continue

            # SPOT_CELL_SPAN만큼 같은 종류가 연속되면 한 자리로 묶는다.
            # (대형차는 2칸이 1자리)
            span = SPOT_CELL_SPAN.get(cell, 1)
            cells = [(row, col)]
            next_row = row + 1
            while len(cells) < span and next_row < rows_n and grid_map[next_row][col] == cell:
                cells.append((next_row, col))
                next_row += 1

            if len(cells) < span:
                print(f"[경고] ({row}, {col})의 {SPOT_TYPE_NAME.get(cell, cell)} 구역은 "
                      f"{span}칸이 필요한데 {len(cells)}칸만 연속됩니다. "
                      f"grid_map을 확인하세요.")

            number += 1
            spot_id = f"{letter}-{number}"
            spot_map[spot_id] = cells
            for coord in cells:
                coord_to_spot[coord] = spot_id
            spot_type[spot_id] = cell

            row = next_row      # 묶은 칸들은 건너뛴다

    return spot_map, coord_to_spot, spot_type


def _find_cell(cell_type):
    """grid_map에서 해당 타입의 첫 칸 좌표를 찾는다. 없으면 None."""
    for row in range(len(grid_map)):
        for col in range(len(grid_map[0])):
            if grid_map[row][col] == cell_type:
                return (row, col)
    return None


# 구역ID -> [(row, col)]  /  (row, col) -> 구역ID  /  구역ID -> 셀 타입
spot_map, coord_to_spot, spot_type = _build_spot_tables()

# 입출구 좌표 (grid_map의 GATE1 / GATE2 칸에서 자동 추출)
GATE1_POS = _find_cell(GATE1)   # 입구
GATE2_POS = _find_cell(GATE2)   # 출구


# =====================================================================
# 구역 종류별 배정 우선순위 (고정)
# =====================================================================
# 차량 종류에 맞는 구역 중 '이 순서대로' 첫 번째 빈자리를 배정한다.
# (logic/A01_parking_manager.py의 find_spot_for_car)
#
# 입구에서의 직선거리로 자동 정렬하지 않고 손으로 적어둔 이유:
#   직선거리는 실제 주행거리와 다르다. 예를 들어 A열(왼쪽)은 입구에서 직선으로는
#   가까워 보여도 주차장을 가로질러 가야 한다. 사람이 보기에 자연스러운 순서는
#   '입구가 있는 오른쪽 아래에서 시작해 위로, 그다음 반대편'이다.
#
# 순서 규칙: 입구(오른쪽 아래)에서 가까운 쪽부터 아래 -> 위.
#
# 목록에 없는 자리는 맨 뒤로 밀린다(입구 직선거리 순). 종류 자체가 여기 없으면
# 전부 직선거리 순으로 배정된다. 즉 이 표는 '덮어쓰기'이지 필수가 아니다.
# 배치를 바꾸면 구역 ID가 밀릴 수 있으므로 아래 검증 경고를 꼭 확인할 것.
SPOT_PRIORITY = {
    # 오른쪽 열 아래 -> 위, 그다음 왼쪽 열 아래 -> 위
    SPOT1: ["D-4", "D-3", "D-2", "A-4", "A-3", "A-2", "A-1"],

    # 장애인 구역은 현재 1자리뿐
    SPOT2: ["A-5"],

    # 대형 구역은 2칸을 묶어 1자리뿐 (오른쪽 열 위쪽)
    SPOT3: ["D-1"],

    # 중앙 섬. 입구에 가까운 아래쪽(row 4) 두 자리를 먼저 쓴다.
    SPOT4: ["C-2", "B-2", "C-1", "B-1"],
}


# =====================================================================
# 주차 구역 상태
# =====================================================================
# 모든 구역을 "empty"로 만들어 두고, 아래 표에 적은 것만 덮어쓴다.
#
# !! 주의 !! 여기는 '차량 정보 없이 자리만 막아두는' 용도다. (공사중, 사용 불가 등)
# 차량번호가 없으므로 출차(remove_car)도 요금 계산도 되지 않는다.
#
#   실제로 차를 미리 세워둘 거라면 data/car_data.py의 INITIAL_PARKED에 적을 것.
#   그쪽은 spot_status와 cars_info를 함께 채워서 출차/요금까지 정상 동작한다.
#
#   INITIAL_OCCUPIED = {"A-1": "full"}          <- 자리만 막음 (차 정보 없음)
#   INITIAL_PARKED   = {"A-1": "1234"}          <- 차를 세워둠 (car_data.py)
#
# 구역 ID는 grid_map에서 자동 생성되므로, 배치를 바꾸면 여기 적어둔 ID가
# 사라질 수 있다. 없는 ID를 적으면 아래에서 경고를 출력한다.
INITIAL_OCCUPIED = {}

spot_status = {spot_id: "empty" for spot_id in spot_map}
for _spot_id, _status in INITIAL_OCCUPIED.items():
    if _spot_id not in spot_status:
        print(f"[경고] INITIAL_OCCUPIED의 '{_spot_id}'는 현재 배치에 없는 구역입니다.")
        continue
    spot_status[_spot_id] = _status


# =====================================================================
# 맵 관련 유틸 함수
# =====================================================================
def get_rows():
    """그리드 맵의 행 수를 반환합니다."""
    return len(grid_map)

def get_cols():
    """그리드 맵의 열 수를 반환합니다."""
    return len(grid_map[0])

# 차량이 항상 지나갈 수 있는 셀 타입
DRIVABLE_CELLS = {ROAD, GATE1, GATE2}

def is_valid_pos(row, col, open_spot=None):
    """
    주어진 좌표가 맵 범위 내이고 차량이 지나갈 수 있는 위치인지 확인합니다.

    PILL(기둥)은 언제나 통과할 수 없습니다.
    주차 구역(SPOT1~4)은 기본적으로 통과할 수 없고, 목적지로 지정된 구역만
    예외적으로 열어줍니다. (통로를 따라 이동하도록 강제하기 위함)

    Args:
        row, col:  확인할 격자 좌표
        open_spot: 진입을 허용할 주차 구역 ID (예: "A-1"). None이면 모두 차단.
    """
    if not (0 <= row < get_rows() and 0 <= col < get_cols()):
        return False

    cell = grid_map[row][col]
    if cell in DRIVABLE_CELLS:
        return True
    if cell in SPOT_CELLS and open_spot is not None:
        return coord_to_spot.get((row, col)) == open_spot
    return False

def get_all_spot_ids():
    """정의된 모든 주차 구역 ID 리스트를 반환합니다."""
    return list(spot_map.keys())

def get_spot_entry_coord(spot_id):
    """
    주차 구역의 진입 좌표(도로와 인접한 첫 번째 칸)를 반환합니다.
    BFS 경로 탐색 시 목적지로 사용됩니다.
    """
    if spot_id in spot_map:
        return spot_map[spot_id][0]
    return None

def get_spot_type(spot_id):
    """주차 구역의 셀 타입(SPOT1~SPOT4)을 반환합니다. 없으면 None."""
    return spot_type.get(spot_id)

def get_spot_type_name(spot_id):
    """주차 구역 종류의 한글 이름을 반환합니다. (예: "장애인")"""
    return SPOT_TYPE_NAME.get(spot_type.get(spot_id), "알 수 없음")

def get_spot_ids_by_type(cell_type):
    """
    특정 종류의 주차 구역 ID 리스트를 반환합니다.

    예: 장애인 차량 전용 배정
        get_spot_ids_by_type(SPOT2)  ->  ["A-5"]
    """
    return [s for s, t in spot_type.items() if t == cell_type]

def get_spot_cell_count(spot_id):
    """주차 구역이 차지하는 격자 칸 수. (대형 구역은 2)"""
    return len(spot_map.get(spot_id, ()))


# SPOT_PRIORITY 검증.
# 배치를 바꾸면 구역 ID가 밀리므로, 여기서 걸러주지 않으면 우선순위가 조용히
# 무시된 채 엉뚱한 자리로 배정된다.
for _cell_type, _order in SPOT_PRIORITY.items():
    _actual = set(get_spot_ids_by_type(_cell_type))
    _listed = set(_order)

    if len(_order) != len(_listed):
        print(f"[경고] SPOT_PRIORITY[{SPOT_TYPE_NAME.get(_cell_type, _cell_type)}]에 "
              f"중복된 구역 ID가 있습니다.")

    for _unknown in sorted(_listed - _actual):
        print(f"[경고] SPOT_PRIORITY의 '{_unknown}'은 "
              f"{SPOT_TYPE_NAME.get(_cell_type, _cell_type)} 구역이 아니거나 "
              f"현재 배치에 없습니다. (무시됨)")

    for _missing in sorted(_actual - _listed):
        print(f"[경고] {SPOT_TYPE_NAME.get(_cell_type, _cell_type)} 구역 "
              f"'{_missing}'이 SPOT_PRIORITY에 없습니다. (맨 뒤로 밀립니다)")
