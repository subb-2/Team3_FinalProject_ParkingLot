import sys
import os
# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import deque
from data.map_data import (
    grid_map, spot_map, coord_to_spot, GATE_POS,
    WALL, ROAD, SPOT, GATE,
    get_rows, get_cols, is_valid_pos, get_all_spot_ids, get_spot_entry_coord
)

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

def find_nearest_empty_spot(empty_spots):
    """
    입출구(GATE)에서 가장 가까운 빈 주차 구역을 찾아 반환합니다.
    
    Args:
        empty_spots: 현재 비어있는 구역 ID의 set 또는 list
                     (예: {"A-2", "A-3", ...})
    
    Returns:
        (spot_id, path) 튜플. 빈자리가 없으면 (None, None) 반환.
        - spot_id: 추천된 주차 구역 ID (예: "A-1")
        - path: 입구에서 해당 구역까지의 좌표 경로 리스트
    """
    best_spot = None
    best_path = None
    best_distance = float('inf')
    
    for spot_id in empty_spots:
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
    all_spots = set(get_all_spot_ids())
    test_empty = all_spots - test_occupied
    
    print(f"현재 주차 중인 구역: {test_occupied}")
    print_map(test_occupied)
    
    spot, path = find_nearest_empty_spot(test_empty)
    if spot:
        print(f"[추천 빈자리] : {spot}")
        print(f"[최단 경로]   : {' -> '.join(str(p) for p in path)}")
        print(f"[이동 거리]   : {len(path) - 1} 칸")
    else:
        print("[알림] 빈자리가 없습니다.")
