# 주차장 맵 데이터 정의
# 순수 데이터(그리드, 좌표, 상수)만 관리.
# 로직(BFS, 경로 탐색 등)은 logic/path_finder.py에서 처리.

# 셀 타입 상수 정의
WALL  = 0  # 벽 (이동 불가 영역, 경계선, 중앙 섬 내부)
ROAD  = 1  # 도로 (차량이 이동 가능한 통로)
SPOT  = 2  # 주차 구역 (빈자리/주차중 구분)
GATE  = 3  # 입출구 (입구와 출구가 같은 위치)

# 주차장 그리드 맵 정의 (15행 × 13열)
# 맵 구조 설명:
# - row  0       : 상단 외벽
# - row  1       : A구역 주차칸 (A-1 ~ A-5)
# - row  2       : A구역 앞 도로
# - row  3 ~ 4   : 상단 도로 (입출구 포함)
# - row  5       : 중앙 섬 상단 도로
# - row  6       : C구역 주차칸 (C-1 ~ C-3)
# - row  7       : 중앙 섬 내부 벽
# - row  8       : D구역 주차칸 (D-1 ~ D-3)
# - row  9       : 중앙 섬 하단 도로
# - row 10 ~ 11  : 하단 도로
# - row 12       : B구역 앞 도로
# - row 13       : B구역 주차칸 (B-1 ~ B-5)
# - row 14       : 하단 외벽
# =====================================================================
grid_map = [
    # col: 0     1     2     3     4     5     6     7     8     9    10    11    12
    [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],  # row 0  : 상단 외벽
    [WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL],  # row 1  : A구역 (A-1 ~ A-5, 각 2칸 폭)
    [WALL, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 2  : A구역 앞 도로
    [GATE, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 3  : 입출구 + 도로
    [WALL, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 4  : 도로
    [WALL, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 5  : 중앙 섬 상단 도로
    [WALL, ROAD, ROAD, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL],  # row 6  : C구역 (C-1 ~ C-3, 각 2칸 폭)
    [WALL, ROAD, ROAD, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],  # row 7  : 중앙 섬 내부 벽
    [WALL, ROAD, ROAD, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL],  # row 8  : D구역 (D-1 ~ D-3, 각 2칸 폭)
    [WALL, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 9  : 중앙 섬 하단 도로
    [WALL, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 10 : 도로
    [WALL, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 11 : 도로
    [WALL, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, ROAD, WALL],  # row 12 : B구역 앞 도로
    [WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL, SPOT, SPOT, WALL],  # row 13 : B구역 (B-1 ~ B-5, 각 2칸 폭)
    [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],  # row 14 : 하단 외벽
]

# =====================================================================
# 주차 구역 좌표 매핑
# (row, col) 좌표 리스트 -> 주차 구역 ID
# 각 주차 구역은 2칸 폭으로 구성됨
# =====================================================================
spot_map = {
    "A-1": [(1, 1),  (1, 2)],
    "A-2": [(1, 4),  (1, 5)],
    "A-3": [(1, 7),  (1, 8)],
    "A-4": [(1, 10), (1, 11)],
    # A-5는 맵 크기 제한으로 A-4까지만 반영 (필요시 열 확장)
    
    "B-1": [(13, 1),  (13, 2)],
    "B-2": [(13, 4),  (13, 5)],
    "B-3": [(13, 7),  (13, 8)],
    "B-4": [(13, 10), (13, 11)],
    # B-5도 마찬가지
    
    "C-1": [(6, 4),  (6, 5)],
    "C-2": [(6, 7),  (6, 8)],
    "C-3": [(6, 10), (6, 11)],
    
    "D-1": [(8, 4),  (8, 5)],
    "D-2": [(8, 7),  (8, 8)],
    "D-3": [(8, 10), (8, 11)],
}

# 역방향 매핑: 좌표 -> 주차 구역 ID (빠른 검색용)
coord_to_spot = {}
for spot_id, coords in spot_map.items():
    for coord in coords:
        coord_to_spot[coord] = spot_id

# 입출구 좌표
GATE_POS = (3, 0)

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
spot_status = {
    "A-1": "empty",
    "A-2": "empty",
    "A-3": "empty",
    "A-4": "empty",
    "A-5": "empty",

    "B-1": "empty",
    "B-2": "empty",
    "B-3": "empty",
    "B-4": "empty",
    "B-5": "empty",

    "C-1": "empty",
    "C-2": "empty",
    "C-3": "empty",
    "C-4": "empty",
    "C-5": "empty",

    "D-1": "empty",
    "D-2": "empty",
    "D-3": "empty",
    "D-4": "empty",
    "D-5": "empty",
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

def is_valid_pos(row, col):
    """주어진 좌표가 맵 범위 내이고, 이동 가능한(벽이 아닌) 위치인지 확인합니다."""
    if 0 <= row < get_rows() and 0 <= col < get_cols():
        return grid_map[row][col] != WALL
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
