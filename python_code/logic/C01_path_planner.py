import sys
import os
import math
import heapq
import numpy as np

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 설정 (Configuration)
# 이 모듈은 '경로 계산'만 담당.
#   - 위치 추정 / 안내 : C00_navigation.py
#   - 마커 실좌표 정의 : C00_navigation.py의 MARKER_WORLD_POS
CONFIG = {
    # 격자 해상도 (cm). 작을수록 정밀하지만 계산량이 늘어난다.
    "GRID_RESOLUTION_CM": 4.0,

    # 주차 구역 크기 (cm). D00_ui_navi의 표시 크기와 맞추는 것을 권장.
    "SPOT_W_CM": 22.0,
    "SPOT_H_CM": 16.0,

    # 차량이 장애물에 얼마나 가까이 갈 수 있는지 (cm).
    # 장애물을 이 값만큼 부풀려서 경로가 벽에 붙지 않게 한다.
    "VEHICLE_CLEARANCE_CM": 7.0,

    # 주차 구역 바깥으로 확보할 주행 가능 영역 (cm).
    # 이 값이 차량 폭보다 크면 주차 구역 뒤로 돌아가는 경로가 생길 수 있다.
    "LOT_MARGIN_CM": 10.0,

    # 경로 단순화: 직선으로 갈 수 있는 구간을 하나의 경유점으로 합친다.
    "SIMPLIFY_PATH": True,

    # 경로에서 이 거리 이상 벗어나면 다시 계획한다 (cm).
    "REPLAN_TOLERANCE_CM": 25.0,
}


# 주차장 점유 격자
class ParkingLotMap:
    """
    주차장 실좌표(cm)를 격자로 변환하고, 주행 가능 여부를 판정하는 지도.

    주차 구역은 기본적으로 주행 불가 영역이다. 차량이 통로를 따라
    이동하도록 강제하기 위함이며, 목적지로 지정된 구역만 예외적으로
    진입 가능하게 열어준다.

    장애물은 VEHICLE_CLEARANCE_CM만큼 부풀려서(inflation) 표시하므로,
    경로 계산 시 차량을 점으로 취급해도 실제로는 여유가 확보된다.
    """

    def __init__(self, spot_world_pos, gate_world_pos=None,
                 resolution=4.0, spot_w=22.0, spot_h=16.0,
                 clearance=7.0, lot_margin=10.0):
        """
        ParkingLotMap 초기화.

        Args:
            spot_world_pos: {구역ID: (x_cm, y_cm)} 주차 구역 중심 좌표
            gate_world_pos: 입출구 좌표 (x_cm, y_cm). 주행 영역에 포함시킨다.
            resolution:     격자 한 칸 크기 (cm)
            spot_w/spot_h:  주차 구역 크기 (cm)
            clearance:      장애물을 부풀릴 거리 (cm)
            lot_margin:     주차 구역 바깥 주행 가능 여유 (cm)
        """
        self.spot_world_pos = dict(spot_world_pos)
        self.gate_world_pos = gate_world_pos
        self.resolution = resolution
        self.spot_w = spot_w
        self.spot_h = spot_h
        self.clearance = clearance
        self.lot_margin = lot_margin

        self._compute_bounds()

        # 현재 열려 있는 목적지 구역과 점유 상태 (재구축 판단용)
        self._open_spot = None
        self._blocked_key = None
        self.grid = None
        self.rebuild()

    def _compute_bounds(self):
        """주차 구역과 입출구를 모두 포함하는 격자 범위를 계산."""
        xs, ys = [], []
        for x, y in self.spot_world_pos.values():
            xs += [x - self.spot_w / 2, x + self.spot_w / 2]
            ys += [y - self.spot_h / 2, y + self.spot_h / 2]

        if self.gate_world_pos is not None:
            xs.append(self.gate_world_pos[0])
            ys.append(self.gate_world_pos[1])

        if not xs:
            xs, ys = [0.0], [0.0]

        m = self.lot_margin
        self.min_x, self.max_x = min(xs) - m, max(xs) + m
        self.min_y, self.max_y = min(ys) - m, max(ys) + m

        self.cols = max(int(math.ceil((self.max_x - self.min_x) / self.resolution)), 1)
        self.rows = max(int(math.ceil((self.max_y - self.min_y) / self.resolution)), 1)

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

        # 장애물을 부풀릴 반경(격자 칸 수)
        pad = self.clearance

        for spot_id, (cx, cy) in self.spot_world_pos.items():
            if spot_id == open_spot:
                continue    # 목적지 구역은 진입해야 하므로 열어둔다

            x1 = cx - self.spot_w / 2 - pad
            x2 = cx + self.spot_w / 2 + pad
            y1 = cy - self.spot_h / 2 - pad
            y2 = cy + self.spot_h / 2 + pad

            c1, r1 = self.world_to_cell((x1, y1), clamp=True)
            c2, r2 = self.world_to_cell((x2, y2), clamp=True)
            grid[r1:r2 + 1, c1:c2 + 1] = False

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
    from logic.C00_navigation import MARKER_WORLD_POS, MARKER_TO_SPOT, GATE_WORLD_POS

    print("==========================================")
    print(" C01 : 주차장 경로 계획 (A* + 경로 단순화)")
    print(" 단독 테스트 : 통로를 따라가는 경로 검증")
    print("==========================================")

    spot_world_pos = {
        spot_id: MARKER_WORLD_POS[marker_id]
        for marker_id, spot_id in MARKER_TO_SPOT.items()
        if marker_id in MARKER_WORLD_POS
    }

    lot = ParkingLotMap(
        spot_world_pos, GATE_WORLD_POS,
        resolution=CONFIG['GRID_RESOLUTION_CM'],
        spot_w=CONFIG['SPOT_W_CM'], spot_h=CONFIG['SPOT_H_CM'],
        clearance=CONFIG['VEHICLE_CLEARANCE_CM'],
        lot_margin=CONFIG['LOT_MARGIN_CM'],
    )
    planner = RoutePlanner(lot, simplify=CONFIG['SIMPLIFY_PATH'])

    print(f"\n[INFO] 격자 {lot.cols}x{lot.rows} "
          f"({lot.resolution}cm/칸), 범위 x[{lot.min_x:.0f},{lot.max_x:.0f}] "
          f"y[{lot.min_y:.0f},{lot.max_y:.0f}]")

    for goal in ("A-1", "B-3"):
        print(f"\n{'='*60}")
        print(f"[TEST] 입출구 {GATE_WORLD_POS} -> {goal} 경로")
        route = planner.plan(GATE_WORLD_POS, goal)

        if route is None:
            print("  경로를 찾지 못했습니다.")
            continue

        print(f"  경유점 {len(route)}개, 총 거리 {route_length(route):.1f}cm")
        for i, (x, y) in enumerate(route):
            print(f"    {i}: ({x:6.1f}, {y:6.1f})")

        # 터미널에 격자 시각화
        print("\n  [격자]  #=주차구역/장애물  .=주행가능  *=경로  G=입출구  T=목적지")
        route_cells = set()
        for a, b in zip(route[:-1], route[1:]):
            ca = lot.world_to_cell(a, clamp=True)
            cb = lot.world_to_cell(b, clamp=True)
            steps = max(abs(cb[0] - ca[0]), abs(cb[1] - ca[1]), 1)
            for s in range(steps + 1):
                route_cells.add((round(ca[0] + (cb[0] - ca[0]) * s / steps),
                                 round(ca[1] + (cb[1] - ca[1]) * s / steps)))

        gate_cell = lot.world_to_cell(GATE_WORLD_POS, clamp=True)
        goal_cell = lot.world_to_cell(spot_world_pos[goal], clamp=True)

        for r in range(lot.rows):
            line = "  "
            for c in range(lot.cols):
                if (c, r) == gate_cell:
                    line += "G"
                elif (c, r) == goal_cell:
                    line += "T"
                elif (c, r) in route_cells:
                    line += "*"
                elif lot.grid[r, c]:
                    line += "."
                else:
                    line += "#"
            print(line)

    print(f"\n{'='*60}")
    print("[TEST] 경로가 주차 구역(#)을 통과하지 않고 통로를 따라가는지 확인하세요.")
