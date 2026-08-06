import sys
import os
import math
import heapq
import numpy as np

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.map_data import (
    grid_map, coord_to_spot, DRIVABLE_CELLS, SPOT_CELLS, get_rows, get_cols,
)
from logic.C02_lot_layout import (
    SPOT_WORLD_POS, GATE1_WORLD_POS, cell_to_world, CONFIG as C02_CONFIG,
)

# 설정 (Configuration)
# 이 모듈은 '경로 계산'만 담당.
#   - 주차장 배치   : data/map_data.py (격자) + C02_lot_layout.py (실좌표 변환)
#   - 위치 추정/안내 : C00_navigation.py
#
# 주차장 크기와 구역 위치는 여기서 정하지 않는다. 격자에서 자동으로 온다.
CONFIG = {
    # 격자 해상도 (cm). 작을수록 정밀하지만 계산량이 늘어난다.
    # 통로 폭(격자 한 칸)보다 충분히 작아야 한다.
    "GRID_RESOLUTION_CM": 1.0,

    # 차량이 장애물에 얼마나 가까이 갈 수 있는지 (cm).
    # 벽/기둥/주차구역을 이 값만큼 부풀려서 경로가 붙지 않게 한다.
    # 주의: 통로 폭이 좁으므로 이 값을 키우면 통로가 막혀 경로를 못 찾는다.
    #       통로 폭 = C02의 CELL_W_CM (현재 10cm) 이고, 양쪽에서 부풀리므로
    #       실제로 남는 폭은 (CELL_W_CM - 2 x VEHICLE_CLEARANCE_CM) 이다.
    "VEHICLE_CLEARANCE_CM": 2.0,

    # 경로 단순화: 직선으로 갈 수 있는 구간을 하나의 경유점으로 합친다.
    "SIMPLIFY_PATH": True,

    # 경로에서 이 거리 이상 벗어나면 다시 계획한다 (cm).
    "REPLAN_TOLERANCE_CM": 12.0,
}


# 주차장 점유 격자
class ParkingLotMap:
    """
    data/map_data.py의 격자를 실좌표(cm) 점유 격자로 바꾸고, 주행 가능 여부를 판정하는 지도.

    격자의 셀 타입을 그대로 따른다.
      - 기둥(PILL)             : 항상 주행 불가
      - 도로(ROAD), 입출구     : 주행 가능
      - 주차 구역(SPOT_CELLS)  : 기본 주행 불가. 목적지로 지정된 구역만 열어준다.
        (차량이 통로를 따라 이동하도록 강제하기 위함)
        구역 종류(일반/장애인/대형/전기차)는 주행 판정에 영향을 주지 않는다.

    장애물은 clearance만큼 부풀려서(inflation) 표시하므로, 경로 계산 시
    차량을 점으로 취급해도 실제로는 여유가 확보된다.
    """

    def __init__(self, spot_world_pos=None, resolution=1.0, clearance=2.0,
                 cell_w=None, cell_h=None):
        """
        ParkingLotMap 초기화.

        Args:
            spot_world_pos: {구역ID: (x_cm, y_cm)}. None이면 C02의 기본 배치 사용
            resolution:     점유 격자 한 칸 크기 (cm)
            clearance:      장애물을 부풀릴 거리 (cm)
            cell_w/cell_h:  map_data 격자 한 칸의 실제 크기 (cm). None이면 C02 CONFIG
        """
        self.spot_world_pos = dict(
            spot_world_pos if spot_world_pos is not None else SPOT_WORLD_POS
        )
        self.resolution = resolution
        self.clearance = clearance
        self.cell_w = C02_CONFIG['CELL_W_CM'] if cell_w is None else cell_w
        self.cell_h = C02_CONFIG['CELL_H_CM'] if cell_h is None else cell_h

        self._compute_bounds()

        # 현재 열려 있는 목적지 구역 (재구축 판단용)
        self._open_spot = None
        self.grid = None
        self.rebuild()

    def _compute_bounds(self):
        """map_data 격자 전체를 덮는 범위를 계산."""
        top_left = cell_to_world((0, 0))
        bottom_right = cell_to_world((get_rows() - 1, get_cols() - 1))

        self.min_x = top_left[0] - self.cell_w / 2
        self.max_x = bottom_right[0] + self.cell_w / 2
        self.min_y = top_left[1] - self.cell_h / 2
        self.max_y = bottom_right[1] + self.cell_h / 2

        self.cols = max(int(math.ceil((self.max_x - self.min_x) / self.resolution)), 1)
        self.rows = max(int(math.ceil((self.max_y - self.min_y) / self.resolution)), 1)

    def _block_cell(self, grid, map_cell):
        """map_data 격자 한 칸을 clearance만큼 부풀려 주행 불가로 표시."""
        cx, cy = cell_to_world(map_cell)
        pad = self.clearance
        c1, r1 = self.world_to_cell(
            (cx - self.cell_w / 2 - pad, cy - self.cell_h / 2 - pad), clamp=True)
        c2, r2 = self.world_to_cell(
            (cx + self.cell_w / 2 + pad, cy + self.cell_h / 2 + pad), clamp=True)
        grid[r1:r2 + 1, c1:c2 + 1] = False

    def rebuild(self, spot_status=None, open_spot=None):
        """
        점유 상태에 맞춰 격자를 다시 만든다.

        Args:
            spot_status: {구역ID: "empty"|"full"} 점유 상태.
                         지금은 모든 구역을 주행 불가로 두므로 표시용으로만 쓰이지만,
                         나중에 빈 구역을 통과 가능하게 바꿀 때를 위해 받아둔다.
            open_spot:   주행 가능하게 열어둘 구역 ID (목적지)
        """
        self._open_spot = open_spot

        # True = 주행 가능
        grid = np.ones((self.rows, self.cols), dtype=bool)

        for row in range(get_rows()):
            for col in range(get_cols()):
                cell_type = grid_map[row][col]
                if cell_type in DRIVABLE_CELLS:
                    continue
                # 목적지 구역은 진입해야 하므로 열어둔다
                if cell_type in SPOT_CELLS and coord_to_spot.get((row, col)) == open_spot:
                    continue
                self._block_cell(grid, (row, col))

        self.grid = grid

    def world_to_cell(self, point, clamp=False):
        """실좌표(cm)를 격자 좌표 (col, row)로 변환."""
        col = int((point[0] - self.min_x) / self.resolution)
        row = int((point[1] - self.min_y) / self.resolution)
        if clamp:
            col = min(max(col, 0), self.cols - 1)
            row = min(max(row, 0), self.rows - 1)
        return col, row

    def cell_to_world(self, cell):
        """격자 좌표 (col, row)를 해당 칸 중심의 실좌표(cm)로 변환."""
        col, row = cell
        return (self.min_x + (col + 0.5) * self.resolution,
                self.min_y + (row + 0.5) * self.resolution)

    def in_bounds(self, cell):
        """격자 범위 안인지 확인."""
        col, row = cell
        return 0 <= col < self.cols and 0 <= row < self.rows

    def is_free(self, cell):
        """해당 칸이 주행 가능한지 확인."""
        if not self.in_bounds(cell):
            return False
        return bool(self.grid[cell[1], cell[0]])

    def nearest_free_cell(self, cell, max_radius=25):
        """
        주어진 칸이 막혀 있으면 가장 가까운 주행 가능 칸을 찾는다.

        차량이 주차 구역 위에 있거나 여유 영역 안에 있을 때
        출발점을 잡지 못하는 상황을 막기 위한 보정이다.
        """
        if self.is_free(cell):
            return cell

        col, row = cell
        for r in range(1, max_radius + 1):
            best, best_d = None, None
            for dc in range(-r, r + 1):
                for dr in range(-r, r + 1):
                    # 정사각 테두리만 검사
                    if max(abs(dc), abs(dr)) != r:
                        continue
                    cand = (col + dc, row + dr)
                    if not self.is_free(cand):
                        continue
                    d = dc * dc + dr * dr
                    if best_d is None or d < best_d:
                        best, best_d = cand, d
            if best is not None:
                return best
        return None

    def line_of_sight(self, cell_a, cell_b):
        """
        두 칸 사이를 직선으로 이동할 수 있는지 확인. (경로 단순화에 사용)
        Bresenham 직선을 따라가며 막힌 칸이 있는지 검사한다.
        """
        x0, y0 = cell_a
        x1, y1 = cell_b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if not self.is_free((x0, y0)):
                return False
            if (x0, y0) == (x1, y1):
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy


# A* 기반 경로 계획
class RoutePlanner:
    """
    현재 차량 위치에서 배정된 주차 구역까지의 경로를 계산.

    주차 구역이 장애물로 막혀 있으므로 경로는 자연스럽게 통로를 따라
    형성되며, 목적지 구역만 열려 있어 그 앞에서 진입하는 형태가 된다.

    격자 경로를 그대로 쓰면 계단 모양이 되므로, 직선으로 갈 수 있는
    구간을 합쳐 최소한의 경유점만 남긴다.
    """

    # 8방향 이동 (대각선 포함)
    _MOVES = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def __init__(self, lot_map, simplify=True):
        """
        RoutePlanner 초기화.

        Args:
            lot_map:  ParkingLotMap 인스턴스
            simplify: True면 직선 구간을 합쳐 경유점을 줄인다
        """
        self.lot_map = lot_map
        self.simplify = simplify

    def plan(self, start_world, goal_spot_id, spot_status=None):
        """
        출발 위치에서 목표 주차 구역까지의 경로를 계산.

        Args:
            start_world:  차량의 현재 실좌표 (x_cm, y_cm)
            goal_spot_id: 목표 주차 구역 ID
            spot_status:  {구역ID: "empty"|"full"} 점유 상태 (선택)

        Returns:
            경유점 리스트 [(x_cm, y_cm), ...]. 출발점을 포함하고 목적지로 끝난다.
            경로를 찾지 못하면 None.
        """
        goal_world = self.lot_map.spot_world_pos.get(goal_spot_id)
        if goal_world is None:
            return None

        # 목적지 구역만 열어둔 상태로 격자를 준비
        self.lot_map.rebuild(spot_status=spot_status, open_spot=goal_spot_id)

        start_cell = self.lot_map.nearest_free_cell(
            self.lot_map.world_to_cell(start_world, clamp=True))
        goal_cell = self.lot_map.nearest_free_cell(
            self.lot_map.world_to_cell(goal_world, clamp=True))

        if start_cell is None or goal_cell is None:
            return None

        cells = self._astar(start_cell, goal_cell)
        if cells is None:
            return None

        if self.simplify:
            cells = self._simplify(cells)

        # 격자 중심 좌표로 변환하되, 시작점과 끝점은 실제 좌표를 그대로 사용
        route = [self.lot_map.cell_to_world(c) for c in cells]
        route[0] = (float(start_world[0]), float(start_world[1]))
        route[-1] = (float(goal_world[0]), float(goal_world[1]))
        return route

    def _astar(self, start, goal):
        """A*로 격자 경로를 탐색. 경로가 없으면 None."""
        def h(c):
            return math.hypot(c[0] - goal[0], c[1] - goal[1])

        open_heap = [(h(start), 0.0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0.0}

        while open_heap:
            _, cost, current = heapq.heappop(open_heap)

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            # 이미 더 좋은 경로로 처리된 칸이면 건너뛴다
            if cost > cost_so_far.get(current, float('inf')):
                continue

            for dc, dr, step in self._MOVES:
                nxt = (current[0] + dc, current[1] + dr)
                if not self.lot_map.is_free(nxt):
                    continue

                # 대각선 이동 시 모서리를 뚫고 지나가지 않도록 확인
                if dc and dr:
                    if not (self.lot_map.is_free((current[0] + dc, current[1]))
                            and self.lot_map.is_free((current[0], current[1] + dr))):
                        continue

                new_cost = cost + step
                if new_cost < cost_so_far.get(nxt, float('inf')):
                    cost_so_far[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(open_heap, (new_cost + h(nxt), new_cost, nxt))

        return None

    def _simplify(self, cells):
        """
        직선으로 갈 수 있는 구간을 하나로 합쳐 경유점을 줄인다.
        (격자 경로의 계단 모양을 제거)
        """
        if len(cells) <= 2:
            return cells

        simplified = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            # 현재 지점에서 직선으로 도달 가능한 가장 먼 지점을 찾는다
            j = len(cells) - 1
            while j > i + 1 and not self.lot_map.line_of_sight(cells[i], cells[j]):
                j -= 1
            simplified.append(cells[j])
            i = j

        return simplified


# 경로 유틸리티
def route_length(route, from_index=0, current_pos=None):
    """
    경로의 남은 총 길이(cm)를 계산.

    Args:
        route:       경유점 리스트 [(x, y), ...]
        from_index:  현재 향하고 있는 경유점의 인덱스
        current_pos: 차량의 현재 위치. 주면 현재 위치부터의 거리로 계산한다.

    Returns:
        남은 거리 (cm)
    """
    if not route or from_index >= len(route):
        return 0.0

    total = 0.0
    prev = current_pos if current_pos is not None else route[from_index]
    for pt in route[from_index:]:
        total += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
        prev = pt
    return total


def distance_to_route(route, point, from_index=0):
    """
    현재 위치가 경로에서 얼마나 벗어나 있는지 계산 (cm).
    재계획 필요 여부를 판단하는 데 사용한다.
    """
    if not route or len(route) < 2:
        if route:
            return math.hypot(point[0] - route[0][0], point[1] - route[0][1])
        return float('inf')

    best = float('inf')
    start = max(from_index - 1, 0)
    for a, b in zip(route[start:-1], route[start + 1:]):
        best = min(best, _point_segment_distance(point, a, b))
    return best


def _point_segment_distance(p, a, b):
    """점 p와 선분 ab 사이의 최단 거리."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# =====================================================================
# 테스트용 메인 (단독 실행 시 경로 계산 검증)
# =====================================================================
# 카메라 없이 동작한다. C00의 마커 배치를 그대로 읽어 경로를 계산하고
# 터미널에 주차장 격자와 경로를 그려서 보여준다.
if __name__ == '__main__':
    print("==========================================")
    print(" C01 : 주차장 경로 계획 (A* + 경로 단순화)")
    print(" 단독 테스트 : 통로를 따라가는 경로 검증")
    print("==========================================")

    lot = ParkingLotMap(
        resolution=CONFIG['GRID_RESOLUTION_CM'],
        clearance=CONFIG['VEHICLE_CLEARANCE_CM'],
    )
    planner = RoutePlanner(lot, simplify=CONFIG['SIMPLIFY_PATH'])

    print(f"\n[INFO] 점유격자 {lot.cols}x{lot.rows} "
          f"({lot.resolution}cm/칸), 범위 x[{lot.min_x:.0f},{lot.max_x:.0f}] "
          f"y[{lot.min_y:.0f},{lot.max_y:.0f}]")
    print(f"[INFO] 통로 폭 {lot.cell_w:.0f}cm, 여유 {lot.clearance:.0f}cm "
          f"-> 실제 주행 가능 폭 {lot.cell_w - 2 * lot.clearance:.0f}cm")

    # 왼쪽 열 / 중앙 섬 / 오른쪽 열을 하나씩 확인
    for goal in ("A-1", "C-2", "B-4"):
        print(f"\n{'='*60}")
        print(f"[TEST] 입구 {GATE1_WORLD_POS} -> {goal} 경로")
        route = planner.plan(GATE1_WORLD_POS, goal)

        if route is None:
            print("  경로를 찾지 못했습니다.")
            continue

        print(f"  경유점 {len(route)}개, 총 거리 {route_length(route):.1f}cm")
        for i, (x, y) in enumerate(route):
            print(f"    {i}: ({x:6.1f}, {y:6.1f})")

        # 터미널에 격자 시각화 (해상도가 높으므로 몇 칸씩 건너뛰어 그린다)
        print("\n  [격자]  #=벽/기둥/주차구역  .=주행가능  *=경로  G=입구  T=목적지")
        route_cells = set()
        for a, b in zip(route[:-1], route[1:]):
            ca = lot.world_to_cell(a, clamp=True)
            cb = lot.world_to_cell(b, clamp=True)
            steps = max(abs(cb[0] - ca[0]), abs(cb[1] - ca[1]), 1)
            for s in range(steps + 1):
                route_cells.add((round(ca[0] + (cb[0] - ca[0]) * s / steps),
                                 round(ca[1] + (cb[1] - ca[1]) * s / steps)))

        gate_cell = lot.world_to_cell(GATE1_WORLD_POS, clamp=True)
        goal_cell = lot.world_to_cell(lot.spot_world_pos[goal], clamp=True)

        # 화면에 들어오도록 축소 표시
        step_c = max(lot.cols // 60, 1)
        step_r = max(lot.rows // 40, 1)
        for r in range(0, lot.rows, step_r):
            line = "  "
            for c in range(0, lot.cols, step_c):
                block_r = range(r, min(r + step_r, lot.rows))
                block_c = range(c, min(c + step_c, lot.cols))
                cells = {(cc, rr) for rr in block_r for cc in block_c}

                if gate_cell in cells:
                    line += "G"
                elif goal_cell in cells:
                    line += "T"
                elif cells & route_cells:
                    line += "*"
                elif lot.grid[r, c]:
                    line += "."
                else:
                    line += "#"
            print(line)

    print(f"\n{'='*60}")
    print("[TEST] 경로가 벽/기둥/중앙 섬을 통과하지 않고 통로를 따라가는지 확인하세요.")
