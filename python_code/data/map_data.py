# 주차장 맵 데이터 정의
# 순수 데이터(그리드, 좌표, 상수)만 관리.
#   - 실좌표(cm) 변환 : logic/C02_lot_layout.py
#   - 경로 계산       : logic/C01_path_planner.py

# 셀 타입 상수 정의
WALL  = 0  # 벽 (이동 불가 영역, 경계선, 중앙 섬 내부)
ROAD  = 1  # 도로 (차량이 이동 가능한 통로)
SPOT  = 2  # 주차 구역 (빈자리/주차중 구분)
GATE1  = 3  # 입구
GATE2 = 4 # 출구
PILL = 5 #기둥

# =====================================================================
# 주차장 그리드 맵 (11행 × 7열)
# =====================================================================
# 구조:
#   - 세로 통로 2개(col 2 = 왼쪽, col 4 = 오른쪽)가 위(row 1~2)와
#     아래(row 8~9)에서 이어지는 ㅁ자 순환로.
#   - 가운데 col 3의 row 3~7은 중앙 섬 (기둥 3 + 주차 구역 2).
#   - 입구(GATE1)는 왼쪽 통로 위, 출구(GATE2)는 오른쪽 통로 위.
#     입구로 들어와 왼쪽으로 내려가고, 아래에서 돌아 오른쪽으로 올라가
#     출구로 나가는 흐름이 자연스럽게 만들어진다.
#
# 규칙: 주차 구역(SPOT)은 반드시 '같은 열의 위아래 기둥(PILL) 사이'에 놓인다.
#       이 규칙 덕분에 자리의 실좌표를 두 기둥 마커의 중점으로 계산할 수 있다.
#       (logic/C02_lot_layout.py 참고) 배치를 바꿀 때 이 규칙을 깨지 말 것.
grid_map = [
    # col: 0     1     2     3     4     5     6
        [WALL, WALL, GATE1, WALL, GATE2, WALL, WALL],  # row 0  
        [WALL, PILL, ROAD, ROAD, ROAD, PILL, WALL],  # row 1  
        [WALL, SPOT, ROAD, ROAD, ROAD, SPOT, WALL],  # row 2  
        [WALL, PILL, ROAD, PILL, ROAD, PILL, WALL],  # row 3  
        [WALL, SPOT, ROAD, SPOT, ROAD, SPOT, WALL],  # row 4  
        [WALL, PILL, ROAD, PILL, ROAD, PILL, WALL],  # row 5  
        [WALL, SPOT, ROAD, SPOT, ROAD, SPOT, WALL],  # row 6  
        [WALL, PILL, ROAD, PILL, ROAD, PILL, WALL],  # row 7  
        [WALL, SPOT, ROAD, ROAD, ROAD, SPOT, WALL],  # row 8  
        [WALL, PILL, ROAD, ROAD, ROAD, PILL, WALL],  # row 9  
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL],  # row 10 
]

# =====================================================================
# 기둥(PILL) -> ArUco 마커 ID 매핑
# =====================================================================
# 기둥 칸에 실제로 붙여 놓은 ArUco 마커의 ID.
# 마커를 옮기거나 다시 인쇄하면 이 표를 반드시 함께 고칠 것.
#
# 이 표가 격자(설계도)와 실제 카메라 영상을 잇는 유일한 연결 고리다.
# 마커의 실좌표(cm)는 logic/C02_lot_layout.py가 이 표를 보고 만든다.
#
#   왼쪽 열(col 1)  중앙 섬(col 3)  오른쪽 열(col 5)
#     row 1 : id1                     row 1 : id6
#     row 3 : id2     row 3 : id11    row 3 : id7
#     row 5 : id3     row 5 : id12    row 5 : id8
#     row 7 : id4     row 7 : id13    row 7 : id9
#     row 9 : id5                     row 9 : id10
PILL_MARKER_ID = {
    (1, 1): 1,  (3, 1): 2,  (5, 1): 3,  (7, 1): 4,  (9, 1): 5,
    (1, 5): 6,  (3, 5): 7,  (5, 5): 8,  (7, 5): 9,  (9, 5): 10,
                (3, 3): 11, (5, 3): 12, (7, 3): 13,
}

# 역방향 매핑: 마커 ID -> 격자 좌표
MARKER_ID_CELL = {mid: cell for cell, mid in PILL_MARKER_ID.items()}

# =====================================================================
# 주차 구역 좌표 매핑
# =====================================================================
# (row, col) 좌표 리스트 -> 주차 구역 ID
# 각 주차 구역은 1칸이며, 같은 열의 위아래 기둥 사이에 놓인다.
#   A : 왼쪽 열(col 1)   B : 오른쪽 열(col 5)   C : 중앙 섬(col 3)
# 번호는 입구(row 0)에서 가까운 순.
spot_map = {
    "A-1": [(2, 1)],
    "A-2": [(4, 1)],
    "A-3": [(6, 1)],
    "A-4": [(8, 1)],

    "B-1": [(2, 5)],
    "B-2": [(4, 5)],
    "B-3": [(6, 5)],
    "B-4": [(8, 5)],

    "C-1": [(4, 3)],
    "C-2": [(6, 3)],
}

# 역방향 매핑: 좌표 -> 주차 구역 ID (빠른 검색용)
coord_to_spot = {}
for spot_id, coords in spot_map.items():
    for coord in coords:
        coord_to_spot[coord] = spot_id

# 입출구 좌표 (입구와 출구가 분리되어 있음)
GATE1_POS = (0, 2)   # 입구 : 왼쪽 세로 통로(col 2) 위
GATE2_POS = (0, 4)   # 출구 : 오른쪽 세로 통로(col 4) 위

# =====================================================================
# 주차 구역 상태 (초기 상태를 직접 설정)
# =====================================================================
# 각 구역의 점유 여부를 "empty" 또는 "full" 로 직접 지정하세요.
# 예: 이미 주차되어 있는 자리는 "full", 비어있는 자리는 "empty"
# {
#     "A-1": "empty",
#     "A-2": "full",
#     ...
# }
# 키 목록은 위 spot_map과 정확히 일치해야 한다. (A01_parking_manager가 이 키로 배정)
spot_status = {
    "A-1": "empty",
    "A-2": "empty",
    "A-3": "empty",
    "A-4": "empty",

    "B-1": "empty",
    "B-2": "empty",
    "B-3": "empty",
    "B-4": "empty",

    "C-1": "empty",
    "C-2": "empty",
}

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

    WALL과 PILL(기둥)은 언제나 통과할 수 없습니다.
    주차 구역(SPOT)은 기본적으로 통과할 수 없고, 목적지로 지정된 구역만
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
    if cell == SPOT and open_spot is not None:
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
