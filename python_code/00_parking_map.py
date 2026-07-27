from collections import deque

# =====================================================================
# 셀 타입 상수 정의
# =====================================================================
WALL  = 0  # 벽 (이동 불가 영역, 경계선, 중앙 섬 내부)
ROAD  = 1  # 도로 (차량이 이동 가능한 통로)
SPOT  = 2  # 주차 구역 (빈자리/주차중 구분)
GATE  = 3  # 입출구 (입구와 출구가 같은 위치)

# =====================================================================
# 주차장 그리드 맵 정의 (15행 × 13열)
# =====================================================================
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

# =====================================================================
# BFS 기반 최단경로 탐색
# =====================================================================
def find_shortest_path(start, end):
    """
    BFS를 사용하여 start 좌표에서 end 좌표까지의 최단경로를 탐색합니다.
    
    Args:
        start: 시작 좌표 (row, col) 튜플
        end:   도착 좌표 (row, col) 튜플
    
    Returns:
        경로가 존재하면 좌표 리스트 [(row, col), ...] 반환.
        경로가 없으면 None 반환.
    """
    # 상하좌우 4방향 이동
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = deque()
    queue.append(start)
    
    # 방문 여부 및 이전 좌표(경로 역추적용) 기록
    visited = {start: None}
    
    while queue:
        current = queue.popleft()
        
        # 도착 지점에 도달한 경우
        if current == end:
            # 경로 역추적
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            path.reverse()
            return path
        
        # 인접한 4방향 탐색
        for dr, dc in directions:
            next_row = current[0] + dr
            next_col = current[1] + dc
            next_pos = (next_row, next_col)
            
            if next_pos not in visited and is_valid_pos(next_row, next_col):
                visited[next_pos] = current
                queue.append(next_pos)
    
    # 경로를 찾지 못한 경우
    return None

def find_nearest_empty_spot(occupied_spots):
    """
    입출구(GATE)에서 가장 가까운 빈 주차 구역을 찾아 반환합니다.
    
    Args:
        occupied_spots: 현재 주차 중인 구역 ID의 set 또는 list
                        (예: {"A-1", "C-2"})
    
    Returns:
        (spot_id, path) 튜플. 빈자리가 없으면 (None, None) 반환.
        - spot_id: 추천된 주차 구역 ID (예: "A-1")
        - path: 입구에서 해당 구역까지의 좌표 경로 리스트
    """
    best_spot = None
    best_path = None
    best_distance = float('inf')
    
    for spot_id in get_all_spot_ids():
        # 이미 주차된 구역은 건너뛰기
        if spot_id in occupied_spots:
            continue
        
        # 해당 주차 구역의 진입 좌표
        target = get_spot_entry_coord(spot_id)
        if target is None:
            continue
        
        # BFS로 최단경로 탐색
        path = find_shortest_path(GATE_POS, target)
        if path is not None and len(path) < best_distance:
            best_distance = len(path)
            best_spot = spot_id
            best_path = path
    
    return best_spot, best_path

# =====================================================================
# 맵 시각화 (디버깅/터미널 출력용)
# =====================================================================
def print_map(occupied_spots=None):
    """
    현재 주차장 맵 상태를 터미널에 출력합니다.
    
    Args:
        occupied_spots: 현재 주차 중인 구역 ID의 set 또는 list
    """
    if occupied_spots is None:
        occupied_spots = set()
    
    # 셀 타입별 출력 문자
    cell_chars = {
        WALL: "██",
        ROAD: "  ",
        GATE: "🚪",
    }
    
    print("\n========== 주차장 현황 ==========")
    for row in range(get_rows()):
        line = ""
        for col in range(get_cols()):
            cell = grid_map[row][col]
            pos = (row, col)
            
            if cell == SPOT:
                # 주차 구역인 경우: 점유 여부에 따라 다르게 표시
                spot_id = coord_to_spot.get(pos, None)
                if spot_id and spot_id in occupied_spots:
                    line += "🚗"  # 주차된 차량
                else:
                    line += "🅿️"  # 빈자리
            else:
                line += cell_chars.get(cell, "??")
        print(line)
    print("================================\n")


# =====================================================================
# 테스트 (단독 실행 시)
# =====================================================================
if __name__ == "__main__":
    print("--- 주차장 맵 초기 상태 ---")
    print_map()
    
    # 테스트: A-1, C-2가 주차 중인 상태에서 가장 가까운 빈자리 찾기
    test_occupied = {"A-1", "C-2"}
    print(f"현재 주차 중인 구역: {test_occupied}")
    print_map(test_occupied)
    
    spot, path = find_nearest_empty_spot(test_occupied)
    if spot:
        print(f"[추천 빈자리] : {spot}")
        print(f"[최단 경로]   : {' -> '.join(str(p) for p in path)}")
        print(f"[이동 거리]   : {len(path) - 1} 칸")
    else:
        print("[알림] 빈자리가 없습니다.")
