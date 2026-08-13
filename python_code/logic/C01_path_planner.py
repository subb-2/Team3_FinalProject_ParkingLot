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
    SPOT_WORLD_POS, GATE1_WORLD_POS, cell_to_world,
    ONE_WAY_SEGMENTS_WORLD, CONFIG as C02_CONFIG,
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

    # -----------------------------------------------------------------
    # 일방통행
    # -----------------------------------------------------------------
    # 통로가 한 방향으로만 도는 주차장이다. 방향은 data/map_data.py의
    # ONE_WAY_LOOP가 정하고, 여기서는 '그 방향을 얼마나 강하게 지킬지'만 정한다.
    #
    # 판정은 '이 칸에서 가장 가까운 순환선 구간의 방향'과 지금 가려는 방향의
    # cos으로 한다.
    #   cos > 0  : 순방향        -> 그대로
    #   cos ~ 0  : 옆으로 빠짐    -> 그대로 (자리로 들어가거나 차선을 옮기는 동작)
    #   cos < 0  : 역주행        -> 아래 비용을 물린다
    "ONE_WAY": {
        "ENABLE": True,

        # 역주행 한 칸마다 얹을 벌점 (cm 환산).
        #
        # 하드 금지(None)로 두지 않는 이유: 금지하면 조건이 하나라도 어긋났을 때
        # (차가 차선 밖에 서 있다거나 목적지가 역주행 쪽에만 있다거나) 경로가
        # 아예 안 나와서 안내가 멈춘다. 시연 중에 화면이 비는 것이 가장 나쁘다.
        # 벌점이 충분히 크면 '한 바퀴 도는 쪽'이 늘 싸므로 실질적으로 금지와
        # 같고, 정 그것도 막혔을 때만 돌아가는 길이라도 나온다.
        #
        # 배수가 아니라 '한 칸당 정액'인 이유: 배수로 두면 짧은 역주행이 싸진다.
        # 배수 20에서 1cm만 거꾸로 가는 것은 20cm어치라, 한 바퀴(둘레 약 480cm)
        # 도는 것보다 훨씬 싸서 A*가 그냥 골라버린다. 실제로 '차가 자리를 조금
        # 지나쳤을 때 살짝 후진하는' 경로가 나왔다.
        #
        # 정액이면 한 칸을 거꾸로 가는 순간 이미 한 바퀴보다 비싸다.
        # 주차장 둘레(약 480cm)보다 크게 잡는다.
        "WRONG_WAY_PENALTY_CM": 1000.0,

        # 이 cos 미만이면 역주행으로 본다.
        #
        # 0.0으로 두면 안 된다. '자리로 들어가는' 이동은 통행 방향과 정확히
        # 수직(cos 0)인데, 좌표가 조금만 어긋나도 cos이 -0.01쯤으로 떨어져
        # 멀쩡한 진입이 역주행으로 잡힌다. 수직 근처는 봐줘야 한다.
        #
        # -0.3이면 수직에서 약 17도까지 봐준다. 넉넉해 보이지만 A*의 판정은
        # 실제로 무뎌지지 않는다. 8방향 이동에서 흐름을 거스르는 방향의 cos은
        # 정면 -1.0, 대각 -0.707뿐이라 -0.707과 0.0 사이 어디에 두든 같기
        # 때문이다. 이 값이 실제로 쓰이는 곳은 임의 방향을 다루는 경로 단순화와
        # 검증(wrong_way_legs)이고, 거기서는 경계에 딱 걸린 차선 변경이
        # 역주행으로 뒤집히지 않도록 여유가 있어야 한다.
        "REVERSE_COS": -0.3,

        # 순환선에서 이 거리(cm) 안에서만 방향을 강제한다.
        # None이면 주차장 전체에 적용한다. (칸마다 가장 가까운 구간을 따른다)
        "INFLUENCE_CM": None,

        # -----------------------------------------------------------------
        # 차선 따라가기 (순환선에서 벗어난 만큼 무는 비용)
        # -----------------------------------------------------------------
        # 위의 역주행 벌점만으로는 '통로를 따라 돌기'가 되지 않는다. 그 판정은
        # 흐름을 거스르는 이동(cos < 0)만 막을 뿐, 흐름에 수직인 이동은 공짜로
        # 둔다. 그런데 이 주차장은 가운데가 통째로 빈 도로라, 흐름에 수직으로
        # 가로지른 다음 방향이 같은 곳에서 위로 올라가는 길이 늘 더 짧다.
        # 그래서 입구에서 A-2로 갈 때 D열 통로(col 11)를 타지 않고 주차장
        # 한복판(col 8)을 곧장 가로질러 올라가는 경로가 나왔다.
        #
        # 판정만 놓고 보면 역주행이 아니지만, 바닥 화살표는 순환 차선 위에만
        # 붙어 있다. 차선을 벗어나 가로지르면 사람 눈에는 화살표를 무시하고
        # 역주행하는 것으로 보이고, 실제로 그 자리에서 맞은편 차와 마주친다.
        #
        # 그래서 '차선에서 벗어난 거리'에 비례하는 비용을 한 칸마다 얹는다.
        # 벌점이 아니라 인력(引力)이다. 차선 위가 가장 싸고, 멀어질수록
        # 비싸진다. 자리로 들어가는 마지막 구간처럼 벗어날 수밖에 없는 곳은
        # 대안이 없으므로 그대로 지나간다.
        #
        # 이 값이 0이면 예전 동작(가로지르기 허용)으로 돌아간다.
        "LANE_PULL_PER_CM": 0.12,

        # 차선에도 폭이 있다. 이 거리 안은 벗어난 것으로 치지 않는다.
        # 통로 한 칸이 10cm이므로 그 절반보다 조금 넉넉하게 둔다.
        "LANE_FREE_CM": 6.0,

        # 경로 단순화가 지름길을 낼 때 봐줄 비용 차이 (cm).
        #
        # 단순화는 격자의 계단 모양을 없애려는 것이지 더 싼 길을 찾는 것이
        # 아니다. 그래서 '원래 구간보다 비싸지지 않는' 지름길만 쓰는데, 계단을
        # 펴는 정상적인 단순화도 반올림 때문에 몇 cm 오르내린다. 그 정도는
        # 봐준다. 0으로 두면 계단이 그대로 남아 경유점이 수십 개가 된다.
        "LANE_SHORTCUT_TOLERANCE_CM": 4.0,
    },
}


# A*가 쓰는 8방향 이동과 그 비용 (dc, dr, 한 칸 비용).
# dc가 +x(오른쪽), dr이 +y(아래쪽)이다.
# 일방통행 방향장(OneWayField)이 이 목록 그대로 판정표를 구워두므로,
# 여기를 고치면 그쪽도 자동으로 따라간다.
MOVES = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def bresenham_cells(cell_a, cell_b):
    """
    두 격자 칸을 잇는 직선 위의 칸들을 차례로 낸다. (양 끝 포함)

    Args:
        cell_a, cell_b: 격자 좌표 (col, row)
    """
    x0, y0 = cell_a
    x1, y1 = cell_b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        yield (x0, y0)
        if (x0, y0) == (x1, y1):
            return
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


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

    def line_of_sight(self, cell_a, cell_b, allow=None):
        """
        두 칸 사이를 직선으로 이동할 수 있는지 확인. (경로 단순화에 사용)
        Bresenham 직선을 따라가며 막힌 칸이 있는지 검사한다.

        Args:
            cell_a, cell_b: 격자 좌표 (col, row)
            allow: 칸마다 추가로 검사할 조건. allow(cell) -> bool.
                   일방통행 검사를 끼워 넣는 데 쓴다. None이면 점유만 본다.
        """
        for cell in bresenham_cells(cell_a, cell_b):
            if not self.is_free(cell):
                return False
            if allow is not None and not allow(cell):
                return False
        return True


# 일방통행 방향장
class OneWayField:
    """
    주차장의 통행 방향을 격자 한 칸 단위로 들고 있는 방향장.

    data/map_data.py의 ONE_WAY_LOOP가 '순환 차선의 중심선'을 모서리 점들로
    정의한다. 어떤 지점의 통행 방향은 '그 지점에서 가장 가까운 순환선 구간의
    방향'이다. 사각형 고리라면 결국 '가장 가까운 변의 방향'이 되어,
      오른쪽 통로에 있으면 위로, 윗줄에 있으면 왼쪽으로,
      왼쪽 통로에 있으면 아래로, 아랫줄에 있으면 오른쪽으로
    가 된다. 바닥에 붙여 둔 화살표와 같은 규칙이다.

    이 방향장은 '역주행 금지'만 뜻한다. 방향에 수직으로 움직이는 것
    (자리로 들어가기, 차선 바꾸기, 중앙 섬 쪽으로 빠지기)은 막지 않는다.
    수직 이동까지 막으면 중앙 섬(B/C열)에 아예 들어갈 수 없다.

    비용은 칸과 이동 방향만으로 정해지므로(지나온 경로와 무관) A*의 전제가
    깨지지 않는다. 휴리스틱(직선거리)도 그대로 하한이다.
    """

    # 벌점을 '주지 않음'과 '설정에서 가져옴'을 구분하기 위한 표식.
    # penalty=None이 '역주행 금지'라는 뜻이라 기본값으로 쓸 수 없다.
    _DEFAULT = object()

    def __init__(self, lot_map, segments=None, penalty_cm=_DEFAULT,
                 reverse_cos=None, influence_cm=None,
                 lane_pull=None, lane_free_cm=None):
        """
        Args:
            lot_map:      ParkingLotMap. 방향장을 이 격자 크기에 맞춰 만든다.
            segments:     [((x1,y1),(x2,y2)), ...] 순환선 구간 (cm).
                          None이면 C02의 ONE_WAY_SEGMENTS_WORLD.
            penalty_cm:   역주행 한 칸당 벌점 (cm 환산). None이면 아예 막는다.
            reverse_cos:  이 cos 미만이면 역주행으로 본다.
            influence_cm: 순환선에서 이 거리 안에서만 적용. None이면 전체.
            lane_pull:    차선에서 1cm 벗어날 때마다 얹을 비용. 0이면 끈다.
            lane_free_cm: 이 거리 안은 벗어난 것으로 치지 않는다. (차선 폭)
        """
        one_way_cfg = CONFIG['ONE_WAY']

        self.segments = list(
            ONE_WAY_SEGMENTS_WORLD if segments is None else segments)
        self.penalty_cm = (one_way_cfg['WRONG_WAY_PENALTY_CM']
                           if penalty_cm is self._DEFAULT else penalty_cm)
        self.reverse_cos = (one_way_cfg['REVERSE_COS']
                            if reverse_cos is None else reverse_cos)
        self.influence_cm = (one_way_cfg['INFLUENCE_CM']
                             if influence_cm is None else influence_cm)
        self.lane_pull = (one_way_cfg['LANE_PULL_PER_CM']
                          if lane_pull is None else lane_pull)
        self.lane_free_cm = (one_way_cfg['LANE_FREE_CM']
                             if lane_free_cm is None else lane_free_cm)

        self.rows = 0
        self.cols = 0
        self.flow_x = None
        self.flow_y = None
        # 칸마다 '순환선까지의 거리(cm)'. 차선 따라가기 비용의 재료다.
        self.lane_dist = None
        self._lane_cost = None
        self._masks = {}
        self.build(lot_map)

    def build(self, lot_map):
        """
        격자 칸마다 통행 방향 단위벡터를 계산해 둔다.

        칸마다 4개 구간을 다시 재면 계획 한 번에 수십만 번 계산하게 되므로,
        격자 전체를 numpy로 한 번에 구해 캐시한다. 순환선과 격자 크기는
        실행 중에 바뀌지 않으므로 이 계산은 시작할 때 한 번이면 된다.
        """
        rows, cols = lot_map.rows, lot_map.cols
        self.rows, self.cols = rows, cols
        self.flow_x = np.zeros((rows, cols), dtype=np.float32)
        self.flow_y = np.zeros((rows, cols), dtype=np.float32)
        self.lane_dist = np.zeros((rows, cols), dtype=np.float32)
        self._lane_cost = np.zeros((rows, cols), dtype=np.float32)
        self._masks = {}

        if not self.segments:
            return

        xs = lot_map.min_x + (np.arange(cols) + 0.5) * lot_map.resolution
        ys = lot_map.min_y + (np.arange(rows) + 0.5) * lot_map.resolution
        gx, gy = np.meshgrid(xs, ys)

        best = np.full((rows, cols), np.inf, dtype=np.float32)

        for (ax, ay), (bx, by) in self.segments:
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            if length == 0:
                continue

            # 점에서 선분에 내린 수선의 발까지의 거리 (선분 밖이면 끝점까지)
            t = np.clip(((gx - ax) * dx + (gy - ay) * dy) / (length * length), 0.0, 1.0)
            dist = np.hypot(gx - (ax + t * dx), gy - (ay + t * dy))

            closer = dist < best
            best = np.where(closer, dist, best)
            self.flow_x = np.where(closer, dx / length, self.flow_x)
            self.flow_y = np.where(closer, dy / length, self.flow_y)

        # 순환선까지의 거리. 차선 따라가기 비용의 재료.
        self.lane_dist = np.where(np.isfinite(best), best, 0.0).astype(np.float32)
        self._lane_cost = (
            np.maximum(self.lane_dist - self.lane_free_cm, 0.0) * self.lane_pull
        ).astype(np.float32)

        # 순환선에서 먼 곳은 방향을 강제하지 않는다 (0벡터 = 제약 없음)
        if self.influence_cm is not None:
            far = best > self.influence_cm
            self.flow_x[far] = 0.0
            self.flow_y[far] = 0.0

        self._build_masks()

    def _build_masks(self):
        """
        A*가 쓰는 8방향에 대해 '이 칸에서 그쪽으로 가면 역주행인가'를
        미리 구워 둔다.

        A*는 한 번 계획할 때 이 판정을 수십만 번 한다. 그때마다 numpy 배열에서
        실수 두 개를 꺼내 내적을 계산하면 계획 시간이 배로 늘어난다. 방향이
        8가지뿐이므로 칸마다 0/1로 미리 계산해 bytes에 담아두면, 판정이
        바이트 하나 읽는 것으로 끝난다.
        """
        free = (self.flow_x == 0.0) & (self.flow_y == 0.0)
        for dc, dr, _ in MOVES:
            norm = math.hypot(dc, dr)
            cos = (dc * self.flow_x + dr * self.flow_y) / norm
            wrong = (cos < self.reverse_cos) & ~free
            self._masks[(dc, dr)] = wrong.astype(np.uint8).tobytes()

    def direction_at(self, cell):
        """
        해당 칸의 통행 방향 단위벡터. 제약이 없으면 (0.0, 0.0).

        Args:
            cell: 격자 좌표 (col, row)
        """
        col, row = cell
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return (0.0, 0.0)
        return (float(self.flow_x[row, col]), float(self.flow_y[row, col]))

    def is_wrong_way(self, cell, dc, dr):
        """
        이 칸에서 (dc, dr) 방향으로 가는 것이 역주행인지.

        Args:
            cell:   출발 칸 (col, row)
            dc, dr: 이동 방향. dc가 +x(오른쪽), dr이 +y(아래쪽)다.
                    A*의 8방향이면 미리 구워둔 표를 쓰고, 그 밖의 방향
                    (경로 단순화나 검증에서 오는 임의 방향)은 그때 계산한다.
        """
        col, row = cell
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False

        mask = self._masks.get((dc, dr))
        if mask is not None:
            return mask[row * self.cols + col] != 0

        fx = float(self.flow_x[row, col])
        fy = float(self.flow_y[row, col])
        if fx == 0.0 and fy == 0.0:
            return False        # 제약이 없는 칸

        norm = math.hypot(dc, dr)
        if norm == 0:
            return False

        return (dc * fx + dr * fy) / norm < self.reverse_cos

    def lane_distance(self, cell):
        """이 칸이 순환 차선에서 얼마나 떨어져 있는지 (cm). 격자 밖이면 0."""
        col, row = cell
        if self.lane_dist is None or not (0 <= row < self.rows and 0 <= col < self.cols):
            return 0.0
        return float(self.lane_dist[row, col])

    def lane_cost(self, cell):
        """
        이 칸을 1cm 지날 때마다 얹을 '차선을 벗어난 값' (cm 환산).

        차선 위(폭 안)면 0. 멀어질수록 커진다. 이것이 있어야 A*가 통로를
        따라 돌고, 없으면 가운데를 가로지르는 쪽이 늘 짧아서 그쪽을 고른다.
        (CONFIG['ONE_WAY']['LANE_PULL_PER_CM'] 주석 참고)
        """
        col, row = cell
        if self._lane_cost is None or not (0 <= row < self.rows and 0 <= col < self.cols):
            return 0.0
        return float(self._lane_cost[row, col])

    def step_penalty(self, cell, dc, dr):
        """
        이 칸에서 (dc, dr)로 한 칸 갈 때 얹을 벌점 (cm).

        Returns:
            순방향이면 0.0. 역주행이면 penalty_cm.
            penalty_cm이 None이면 None을 돌려준다. (= 갈 수 없음)
        """
        if not self.is_wrong_way(cell, dc, dr):
            return 0.0
        return self.penalty_cm


# A* 기반 경로 계획
class RoutePlanner:
    """
    현재 차량 위치에서 배정된 주차 구역까지의 경로를 계산.

    주차 구역이 장애물로 막혀 있으므로 경로는 자연스럽게 통로를 따라
    형성되며, 목적지 구역만 열려 있어 그 앞에서 진입하는 형태가 된다.

    격자 경로를 그대로 쓰면 계단 모양이 되므로, 직선으로 갈 수 있는
    구간을 합쳐 최소한의 경유점만 남긴다.
    """

    # 8방향 이동 (대각선 포함). 모듈 상단의 MOVES와 같은 것을 쓴다.
    _MOVES = MOVES

    def __init__(self, lot_map, simplify=True, one_way=None):
        """
        RoutePlanner 초기화.

        Args:
            lot_map:  ParkingLotMap 인스턴스
            simplify: True면 직선 구간을 합쳐 경유점을 줄인다
            one_way:  OneWayField 인스턴스. None이면 CONFIG['ONE_WAY']에 따라
                      만든다. False를 주면 일방통행을 끈다. (비교 시험용)
        """
        self.lot_map = lot_map
        self.simplify = simplify

        if one_way is None:
            one_way = (OneWayField(lot_map)
                       if CONFIG['ONE_WAY']['ENABLE'] else False)
        self.one_way = one_way or None

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

                # 일방통행. 역주행은 벌점을 크게 물리거나(기본) 아예 막는다.
                # 그 위에 '차선에서 벗어난 만큼'을 거리에 비례해 더 얹는다.
                # 벌점이 아니라 인력이라 값이 늘 0 이상이므로, 휴리스틱
                # (직선거리)은 그대로 하한으로 남는다. (A* 전제 유지)
                penalty = 0.0
                if self.one_way is not None:
                    penalty = self.one_way.step_penalty(current, dc, dr)
                    if penalty is None:
                        continue
                    penalty += self.one_way.lane_cost(nxt) * step

                new_cost = cost + step + penalty
                if new_cost < cost_so_far.get(nxt, float('inf')):
                    cost_so_far[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(open_heap, (new_cost + h(nxt), new_cost, nxt))

        return None

    def _simplify(self, cells):
        """
        직선으로 갈 수 있는 구간을 하나로 합쳐 경유점을 줄인다.
        (격자 경로의 계단 모양을 제거)

        지름길은 점유뿐 아니라 통행 방향도 지켜야 한다. A*가 통로를 따라
        돌아온 경로라도, 두 점을 직선으로 이으면 그 직선이 역주행 구간을
        가로지를 수 있다. 그러면 계산은 일방통행인데 화면에 그려지는 선과
        차에 주는 안내만 역주행이 된다.
        """
        if len(cells) <= 2:
            return cells

        # 칸까지의 누적 차선 비용. 아래에서 '원래 경로 구간의 값'을 O(1)로
        # 꺼내기 위한 것이다. 매번 더하면 단순화가 경로 길이의 제곱이 된다.
        if self.one_way is not None:
            prefix = [0.0]
            for cell in cells[1:]:
                prefix.append(prefix[-1] + self.one_way.lane_cost(cell))
        else:
            prefix = None

        simplified = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            # 현재 지점에서 직선으로 도달 가능한 가장 먼 지점을 찾는다
            j = len(cells) - 1
            while j > i + 1 and not self._can_shortcut(cells, i, j, prefix):
                j -= 1
            simplified.append(cells[j])
            i = j

        return simplified

    def wrong_way_legs(self, route):
        """
        경로에서 역주행하는 구간을 찾는다. (검증 / 자체 시험용)

        Args:
            route: 경유점 리스트 [(x_cm, y_cm), ...]

        Returns:
            [(구간 인덱스, 시작점, 끝점), ...]. 비어 있으면 일방통행을 지킨 것.
        """
        if self.one_way is None or not route or len(route) < 2:
            return []

        # 차가 주행 불가 칸(주차 구역이나 여유 영역) 안에 있으면 plan()이
        # 출발점을 가장 가까운 통로 칸으로 옮긴다. 그 결과 첫 구간은 '차의 실제
        # 위치 -> 옮겨진 출발점'이라는, 계획하지 않은 탈출 구간이 된다.
        # 계획기가 정할 수 없는 구간이므로 역주행으로 세지 않는다.
        skip_first = not self.lot_map.is_free(
            self.lot_map.world_to_cell(route[0], clamp=True))

        # 점유는 보지 않는다. 경로의 끝 점은 자리 중심이라 여유 영역 안에
        # 있을 수 있는데, 그걸 역주행으로 세면 안 된다.
        bad = []
        for i, (a, b) in enumerate(zip(route[:-1], route[1:])):
            if i == 0 and skip_first:
                continue

            # 격자 한 칸도 안 되는 구간은 방향이 수치 잡음이라 판정하지 않는다.
            # (A*의 시작점은 차의 실제 위치라 첫 경유점과 겹칠 수 있다)
            if math.hypot(b[0] - a[0], b[1] - a[1]) < self.lot_map.resolution:
                continue

            cell_a = self.lot_map.world_to_cell(a, clamp=True)
            cell_b = self.lot_map.world_to_cell(b, clamp=True)

            # 방향은 실좌표 차이가 아니라 '칸 차이'로 잰다.
            # 계획기가 판정한 것이 칸 단위이기 때문이다. plan()이 첫/끝 점을
            # 칸 중심에서 차의 실제 위치와 자리 중심으로 바꿔치기하므로,
            # 실좌표로 재면 방향이 1도 안쪽으로 틀어진다. 그 차이가 임계값
            # 근처에 걸리면 계획기는 통과시킨 구간을 검증기만 역주행이라고
            # 하는 일이 생긴다. 같은 것을 재야 한다.
            dc = cell_b[0] - cell_a[0]
            dr = cell_b[1] - cell_a[1]

            # 마지막 칸은 빼고 본다. 판정은 '이 칸에서 저 방향으로 나가도 되는가'
            # 이므로, 도착해서 멈추는(또는 방향을 바꾸는) 칸은 대상이 아니다.
            # 넣으면 차선이 바뀌는 경계에서 멀쩡한 차선 변경이 역주행으로 잡힌다.
            cells = list(bresenham_cells(cell_a, cell_b))[:-1]
            if any(self.one_way.is_wrong_way(cell, dc, dr) for cell in cells):
                bad.append((i, a, b))
        return bad

    def _can_shortcut(self, cells, i, j, prefix=None):
        """
        경로의 i번째 칸에서 j번째 칸까지를 직선 하나로 대신할 수 있는지.

        점유와 일방통행은 예전과 같다. 여기에 '차선 비용이 더 비싸지지 않을
        것'을 더한다.

        차선 조건이 필요한 이유: A*가 통로를 따라 돌아온 경로라도, 두 점을
        직선으로 이으면 그 직선이 주차장 한복판을 가로지를 수 있다. 그러면
        계산은 차선을 따라갔는데 화면에 그려지는 선과 차에 주는 안내만
        가로지르기가 된다. 실제로 중앙 섬으로 갈 때 이 일이 났다. 마지막
        구간이 차선 밖이라는 이유로, 출발점에서 거기까지가 통째로 직선이 되어
        주차장을 대각선으로 질러가는 안내가 나왔다.

        단순화는 격자의 계단 모양을 없애려는 것이지 더 싼 길을 찾는 것이
        아니다. 그러니 '원래 구간보다 비싸지지 않을 때만' 편다. 자리로 들어가는
        마지막 구간처럼 원래도 차선 밖이던 곳은 값이 비슷하므로 그대로 펴진다.
        """
        cell_a, cell_b = cells[i], cells[j]
        if self.one_way is None:
            return self.lot_map.line_of_sight(cell_a, cell_b)

        dc = cell_b[0] - cell_a[0]
        dr = cell_b[1] - cell_a[1]

        line = []
        for cell in bresenham_cells(cell_a, cell_b):
            if not self.lot_map.is_free(cell):
                return False
            if self.one_way.is_wrong_way(cell, dc, dr):
                return False
            line.append(cell)

        # 원래 구간의 비용 (한 칸 = 1cm 이동 + 그 칸의 차선 비용)
        if prefix is not None:
            lane_before = prefix[j] - prefix[i]
        else:
            lane_before = sum(self.one_way.lane_cost(c) for c in cells[i + 1:j + 1])
        before = (j - i) + lane_before

        # 지름길의 비용. 직선 길이를 지나온 칸 수로 나눠 칸마다 배분한다.
        length = math.hypot(dc, dr)
        per_cell = length / max(len(line) - 1, 1)
        after = length + sum(self.one_way.lane_cost(c) for c in line[1:]) * per_cell

        return after <= before + CONFIG['ONE_WAY']['LANE_SHORTCUT_TOLERANCE_CM']


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
        best = min(best, point_segment_distance(point, a, b))
    return best


def point_segment_distance(p, a, b):
    """
    점 p와 선분 ab 사이의 최단 거리.

    C00이 '차가 경로의 어느 구간에 와 있는지' 찾을 때도 쓰므로 공개 이름이다.
    """
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
    print(" 단독 테스트 : 통로 + 일방통행을 지키는지 검증")
    print("==========================================")

    lot = ParkingLotMap(
        resolution=CONFIG['GRID_RESOLUTION_CM'],
        clearance=CONFIG['VEHICLE_CLEARANCE_CM'],
    )
    planner = RoutePlanner(lot, simplify=CONFIG['SIMPLIFY_PATH'])
    # 비교용. 일방통행을 끈 계획기로도 같은 경로를 뽑아 차이를 본다.
    free_planner = RoutePlanner(lot, simplify=CONFIG['SIMPLIFY_PATH'], one_way=False)

    print(f"\n[INFO] 점유격자 {lot.cols}x{lot.rows} "
          f"({lot.resolution}cm/칸), 범위 x[{lot.min_x:.0f},{lot.max_x:.0f}] "
          f"y[{lot.min_y:.0f},{lot.max_y:.0f}]")
    print(f"[INFO] 통로 폭 {lot.cell_w:.0f}cm, 여유 {lot.clearance:.0f}cm "
          f"-> 실제 주행 가능 폭 {lot.cell_w - 2 * lot.clearance:.0f}cm")
    print(f"[INFO] 일방통행 {'켜짐' if planner.one_way else '꺼짐'}"
          + (f" (순환선 {len(planner.one_way.segments)}구간, "
             f"역주행 벌점 {planner.one_way.penalty_cm}cm/칸)"
             if planner.one_way else ""))

    def draw_grid(route, goal_cell, extra=None):
        """터미널에 격자와 경로를 그린다. (해상도가 높으므로 축소 표시)"""
        route_cells = set()
        for a, b in zip(route[:-1], route[1:]):
            ca = lot.world_to_cell(a, clamp=True)
            cb = lot.world_to_cell(b, clamp=True)
            route_cells.update(bresenham_cells(ca, cb))

        gate_cell = lot.world_to_cell(GATE1_WORLD_POS, clamp=True)
        step_c = max(lot.cols // 60, 1)
        step_r = max(lot.rows // 40, 1)

        for r in range(0, lot.rows, step_r):
            line = "  "
            for c in range(0, lot.cols, step_c):
                cells = {(cc, rr)
                         for rr in range(r, min(r + step_r, lot.rows))
                         for cc in range(c, min(c + step_c, lot.cols))}

                if gate_cell in cells:
                    line += "G"
                elif goal_cell in cells:
                    line += "T"
                elif cells & route_cells:
                    line += "*"
                elif extra and cells & extra:
                    line += "o"
                elif lot.grid[r, c]:
                    line += "."
                else:
                    line += "#"
            print(line)

    # 일방통행 방향장이 제대로 섰는지 먼저 확인한다.
    # 오른쪽 통로는 위로, 윗줄은 왼쪽으로, 왼쪽 통로는 아래로, 아랫줄은 오른쪽.
    if planner.one_way:
        print(f"\n{'='*60}")
        print("[TEST] 통행 방향장 (자리 위치에서 본 방향)")
        arrow = {(1, 0): "-> 오른쪽", (-1, 0): "<- 왼쪽",
                 (0, 1): "v  아래", (0, -1): "^  위"}
        for spot_id in sorted(lot.spot_world_pos):
            cell = lot.world_to_cell(lot.spot_world_pos[spot_id], clamp=True)
            fx, fy = planner.one_way.direction_at(cell)
            key = (round(fx), round(fy))
            print(f"    {spot_id:5s} 흐름 {arrow.get(key, f'({fx:+.2f},{fy:+.2f})')}")

    # 왼쪽 열 / 중앙 섬 / 오른쪽 열 / 가장 먼 자리를 하나씩 확인.
    # A-4는 입구 바로 왼쪽이라, 일방통행이면 한 바퀴 돌아야 한다.
    failures = 0
    for goal in ("D-4", "C-2", "A-1", "A-4"):
        print(f"\n{'='*60}")
        print(f"[TEST] 입구 {GATE1_WORLD_POS} -> {goal} 경로")
        route = planner.plan(GATE1_WORLD_POS, goal)

        if route is None:
            print("  경로를 찾지 못했습니다.")
            failures += 1
            continue

        bad = planner.wrong_way_legs(route)
        free_route = free_planner.plan(GATE1_WORLD_POS, goal)
        free_len = route_length(free_route) if free_route else float('nan')

        print(f"  경유점 {len(route)}개, 총 거리 {route_length(route):.1f}cm "
              f"(일방통행 무시하면 {free_len:.1f}cm)")
        for i, (x, y) in enumerate(route):
            print(f"    {i}: ({x:6.1f}, {y:6.1f})")

        if bad:
            failures += 1
            print(f"  [실패] 역주행 구간 {len(bad)}개")
            for i, a, b in bad:
                print(f"    구간 {i}: ({a[0]:.1f},{a[1]:.1f}) -> ({b[0]:.1f},{b[1]:.1f})")
        else:
            print("  [OK] 역주행 구간 없음")

        print("\n  [격자]  #=벽/기둥/주차구역  .=주행가능  *=경로  G=입구  T=목적지")
        draw_grid(route, lot.world_to_cell(lot.spot_world_pos[goal], clamp=True))

    print(f"\n{'='*60}")
    if failures:
        print(f"[TEST] 실패 {failures}건. 위의 역주행 구간을 확인할 것.")
    else:
        print("[TEST] 모든 경로가 통로와 일방통행을 지켰습니다.")
    print("       A-4는 입구 바로 옆이지만 한 바퀴 돌아가야 정상이다.")
    print("       (아랫줄은 오른쪽으로만 갈 수 있으므로)")
