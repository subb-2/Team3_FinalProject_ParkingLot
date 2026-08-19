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

    # 경로를 가로/세로 구간으로만 낸다.
    #
    # A*와 단순화는 이미 4방향만 쓰므로 격자 경로 자체는 가로/세로뿐이다.
    # 그런데 마지막에 첫 점을 '차의 실제 위치'로, 끝 점을 '자리 중심'으로
    # 바꿔치기하기 때문에, 차가 차선 한가운데에서 몇 cm만 벗어나 있어도
    # 그 구간만 비스듬한 선이 된다. 화면에서는 통로를 대각선으로 가로지르는
    # 것처럼 보이고, 차가 움직일 때마다 그 선의 기울기가 계속 흔들린다.
    #
    # 켜면 계획할 때 그 두 구간을 모서리 경유점을 옮겨서 편다. 경유점 개수는
    # 그대로다. 옮기는 것은 계획하는 순간뿐이고, 그 뒤로 꼭지점은 고정이다.
    #
    # 화면에 그릴 때도 같은 값을 본다. 차를 지금 달리는 구간 위에 내려 찍어
    # 선을 시작하므로(route_from_position) 꼭지점을 건드리지 않고도 첫 구간이
    # 가로/세로로 떨어진다.
    "STRAIGHT_LEGS_ONLY": True,

    # 꼭지점을 격자 칸 중심에 맞춘다.
    #
    # A*는 1cm 격자에서 최단거리를 찾으므로 통로 한가운데가 아니라 안쪽
    # 가장자리에 붙어서 돈다. 차선 인력에는 폭(LANE_FREE_CM)이 있어서
    # 그 안이면 어디에 있든 공짜이기 때문이다. 실제로 오른쪽 통로를 지날 때
    # 경로가 통로 중심(x=110)이 아니라 x=104.5에 섰다.
    #
    # 몇 cm 차이지만 화면에서는 그 칸의 한복판이 아니라 칸 경계에 걸친 선으로
    # 보인다. 화면이 격자로 그려져 있으니 선도 칸을 따라야 어느 통로로
    # 가라는 것인지 읽힌다.
    #
    # 켜면 구간마다 축을 하나씩 맞춘다. 세로 구간은 x를 그 칸 열의 중심으로,
    # 가로 구간은 y를 그 칸 행의 중심으로 옮긴다. 꼭지점의 x는 세로 구간이,
    # y는 가로 구간이 정하므로 둘이 다툴 일이 없고 직각도 그대로 유지된다.
    #
    # 차와 자리에 닿는 양 끝 구간은 건드리지 않는다. 그 좌표는 차의 실제
    # 위치와 자리 중심이라 옮길 수 없다.
    "SNAP_TO_CELL_CENTERS": True,

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

        # -----------------------------------------------------------------
        # 목적지 앞 통로에서는 일방통행을 풀어준다
        # -----------------------------------------------------------------
        # 목적지가 있는 열에서 통로 쪽으로 이 칸 수만큼이 예외 구역이다.
        # 그 안에서는 역주행 벌점과 차선 인력을 모두 끈다.
        #
        # 자리를 조금만 지나쳐도 규칙만 지켜서는 주차장을 한 바퀴 돌아 다시
        # 들어오는 경로가 나온다. 계산은 맞지만 눈앞의 자리를 두고 한 바퀴
        # 도는 안내라 사람이 따르지 않는다. 실제 주차장에서도 자리 앞
        # 통로에서는 조금 물러서거나 비스듬히 들어가는 것이 허용된다.
        #
        # 왜 원이 아니라 '열 묶음'인가: 자리로 들어가려면 그 자리가 면한
        # 통로에 서야 하는데, 원으로 자르면 통로를 따라 얼마나 지나쳤는지에
        # 따라 예외가 됐다 말았다 한다. 통로 한 줄을 통째로 예외로 두면
        # 그 통로 안에서는 어디에 있든 바로 자리로 들어갈 수 있다.
        #
        # 어느 쪽인지는 자동으로 정해진다. 그 자리에서 가장 가까운 세로
        # 차선 쪽이다. A열(col 0)은 오른쪽 1~4, B열(col 5)은 왼쪽 4~1,
        # C열(col 7)은 오른쪽 8~11, D열(col 12)은 왼쪽 11~8이 된다.
        # 네 줄이면 그 자리가 면한 통로 하나가 통째로 예외가 된다.
        #
        # 통로 폭(네 칸)보다 크게 잡으면 건너편 통로까지 풀려서 진짜 역주행이
        # 나온다. 0으로 두면 주차장 전체에 규칙을 적용한다.
        "GOAL_RELAX_COLUMNS": 4,

        # 계획이 끝난 경로를 차선 위로 붙일 때, 이 거리(cm) 안의 차선까지
        # 끌어당긴다. (SNAP_TO_CELL_CENTERS)
        #
        # 위의 LANE_FREE_CM(차선 폭)보다 훨씬 넉넉해야 한다. 그쪽은 A*가
        # 계획할 때 '이 정도는 벗어나도 공짜'라고 봐주는 폭이고, 이쪽은 그렇게
        # 벗어나서 나온 선을 화면에 그리기 전에 되돌리는 거리다. A*는 모서리를
        # 안쪽으로 질러 가므로 코너 부근에서 한두 칸까지 벌어진다.
        #
        # 너무 키우면 통로 하나를 건너뛰어 엉뚱한 차선에 붙으므로, 통로 폭
        # (네 칸, 40cm)의 절반쯤인 20cm로 둔다.
        "LANE_SNAP_CM": 20.0,

        # 경로 단순화가 지름길을 낼 때 봐줄 비용 차이 (cm).
        #
        # 단순화는 격자의 계단 모양을 없애려는 것이지 더 싼 길을 찾는 것이
        # 아니다. 그래서 '원래 구간보다 비싸지지 않는' 지름길만 쓰는데, 계단을
        # 펴는 정상적인 단순화도 반올림 때문에 몇 cm 오르내린다. 그 정도는
        # 봐준다. 0으로 두면 계단이 그대로 남아 경유점이 수십 개가 된다.
        "LANE_SHORTCUT_TOLERANCE_CM": 4.0,
    },
}


# A*가 쓰는 이동 방향과 그 비용 (dc, dr, 한 칸 비용).
# dc가 +x(오른쪽), dr이 +y(아래쪽)이다.
# 일방통행 방향장(OneWayField)이 이 목록 그대로 판정표를 구워두므로,
# 여기를 고치면 그쪽도 자동으로 따라간다.
#
# 대각선을 뺐다. 4방향만 쓴다.
#
# 대각선을 넣으면 통로를 비스듬히 가로지르는 경로가 나온다. 계산상으로는
# 그쪽이 짧지만, 실제 주차장에서 사람이 그렇게 몰지 않고 화면에 그려지는
# 선도 '어디서 꺾으라는 것인지' 읽히지 않는다. 4방향이면 가로/세로 구간만
# 남아서 "직진 후 좌회전" 형태로 떨어진다.
#
# 대각선을 다시 넣으려면 아래 두 곳이 함께 따라와야 한다.
#   - RoutePlanner._astar의 휴리스틱 (지금은 맨해튼 거리)
#   - RoutePlanner._can_shortcut의 가로/세로 검사
MOVES = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
]

# 위 목록에 대각선이 들어 있는가. 휴리스틱과 경로 단순화가 이 값을 본다.
HAS_DIAGONAL = any(dc and dr for dc, dr, _ in MOVES)


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

    def __init__(self, lot_map, simplify=True, one_way=None,
                 goal_relax_columns=None):
        """
        RoutePlanner 초기화.

        Args:
            lot_map:  ParkingLotMap 인스턴스
            simplify: True면 직선 구간을 합쳐 경유점을 줄인다
            one_way:  OneWayField 인스턴스. None이면 CONFIG['ONE_WAY']에 따라
                      만든다. False를 주면 일방통행을 끈다. (비교 시험용)
            goal_relax_columns: 목적지가 면한 통로 몇 줄까지 일방통행을
                      풀어줄지. None이면 CONFIG['ONE_WAY']['GOAL_RELAX_COLUMNS'].
        """
        self.lot_map = lot_map
        self.simplify = simplify

        if one_way is None:
            one_way = (OneWayField(lot_map)
                       if CONFIG['ONE_WAY']['ENABLE'] else False)
        self.one_way = one_way or None

        self.goal_relax_columns = (CONFIG['ONE_WAY']['GOAL_RELAX_COLUMNS']
                                   if goal_relax_columns is None
                                   else goal_relax_columns)

    def _relax_side(self, goal_world):
        """
        목적지에서 어느 쪽으로 예외 구역을 펼칠지. (+1 오른쪽, -1 왼쪽)

        그 자리에서 가장 가까운 세로 차선 쪽이다. 자리로 들어가려면 결국 그
        차선을 타고 와야 하므로, 예외를 둘 곳도 그쪽이다. 중앙 섬처럼 양옆이
        모두 통로인 자리도 이 기준이면 한쪽으로 정해진다.
        """
        best, best_d = None, None
        for a, b in (self.one_way.segments if self.one_way else ()):
            if abs(a[0] - b[0]) > self.lot_map.resolution:
                continue        # 가로 차선. 열을 정하는 데는 쓰지 않는다.
            d = abs(a[0] - goal_world[0])
            if best_d is None or d < best_d:
                best, best_d = a[0], d
        if best is None:
            return 1
        return 1 if best >= goal_world[0] else -1

    def _make_relaxed(self, goal_world):
        """
        '이 칸은 목적지 앞 통로라 일방통행을 묻지 않는다'를 판정하는 함수.

        목적지가 있는 열부터 통로 쪽으로 GOAL_RELAX_COLUMNS칸까지가 예외
        구역이다. 세로로는 자르지 않는다. 그 통로 안이면 자리를 지나쳤든
        아직 못 미쳤든 바로 들어갈 수 있어야 하기 때문이다.

        예외를 두지 않을 때는 None을 돌려주어, 호출부가 판정 자체를 건너뛰게
        한다. (A* 안쪽 반복문이라 함수 호출 한 번도 아깝다)
        """
        columns = self.goal_relax_columns
        if not columns or self.one_way is None:
            return None

        cell_w = self.lot_map.cell_w
        side = self._relax_side(goal_world)
        # 자리 칸의 중심에서 반 칸 바깥부터, 통로 쪽으로 columns칸까지.
        near = goal_world[0] - side * cell_w / 2.0
        far = goal_world[0] + side * (columns + 0.5) * cell_w

        lo, hi = sorted((near, far))
        c_lo, _ = self.lot_map.world_to_cell((lo, goal_world[1]), clamp=True)
        c_hi, _ = self.lot_map.world_to_cell((hi, goal_world[1]), clamp=True)

        def relaxed(cell):
            return c_lo <= cell[0] <= c_hi

        return relaxed

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

        # 목적지가 면한 통로에서는 일방통행을 풀어준다.
        # (CONFIG['ONE_WAY']['GOAL_RELAX_COLUMNS'] 주석 참고)
        relaxed = self._make_relaxed(goal_world)

        cells = self._astar(start_cell, goal_cell, relaxed)
        if cells is None:
            return None

        if self.simplify:
            cells = self._simplify(cells, relaxed)

        # 격자 중심 좌표로 변환하되, 시작점과 끝점은 실제 좌표를 그대로 사용
        route = [self.lot_map.cell_to_world(c) for c in cells]
        route[0] = (float(start_world[0]), float(start_world[1]))
        route[-1] = (float(goal_world[0]), float(goal_world[1]))

        if CONFIG['STRAIGHT_LEGS_ONLY']:
            route = self._straighten(route)
        if CONFIG['SNAP_TO_CELL_CENTERS']:
            route = self._snap_to_cells(route)
        return route

    def _astar(self, start, goal, relaxed=None):
        """
        A*로 격자 경로를 탐색. 경로가 없으면 None.

        Args:
            relaxed: relaxed(cell) -> bool. True인 칸에서는 일방통행 비용을
                     묻지 않는다. None이면 주차장 전체에 규칙을 적용한다.
        """
        if HAS_DIAGONAL:
            def h(c):
                return math.hypot(c[0] - goal[0], c[1] - goal[1])
        else:
            # 4방향만 쓰면 맨해튼 거리가 정확한 하한이다. 직선거리를 쓰면
            # 하한이 느슨해져 A*가 훨씬 넓게 탐색한다. (계획 시간이 몇 배)
            # 차선 비용은 늘 0 이상이라 이 하한은 그대로 유효하다.
            def h(c):
                return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

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

                # 대각선 이동 시 모서리를 뚫고 지나가지 않도록 확인.
                # (지금 MOVES는 4방향이라 걸리지 않는다. 대각선을 되살릴 때 필요)
                if dc and dr:
                    if not (self.lot_map.is_free((current[0] + dc, current[1]))
                            and self.lot_map.is_free((current[0], current[1] + dr))):
                        continue

                # 일방통행. 역주행은 벌점을 크게 물리거나(기본) 아예 막는다.
                # 그 위에 '차선에서 벗어난 만큼'을 거리에 비례해 더 얹는다.
                # 벌점이 아니라 인력이라 값이 늘 0 이상이므로, 휴리스틱
                # (직선거리)은 그대로 하한으로 남는다. (A* 전제 유지)
                #
                # 목적지가 면한 통로에서는 역주행 벌점만 빼준다. 차선 인력은
                # 거기서도 그대로 둔다. 둘 다 빼면 그 통로 안에서 끌어당기는
                # 것이 없어져, 자리로 갈 때 차선을 버리고 아무 데로나 질러
                # 간다. 실제로 D열로 갈 때 통로 한가운데(col 10)가 아니라
                # 입구에서 그대로 직진하는 경로가 나왔다. 역주행이 아니라고
                # 해서 차선을 안 타도 되는 것은 아니다.
                penalty = 0.0
                if self.one_way is not None:
                    if not (relaxed and relaxed(current)):
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

    def _simplify(self, cells, relaxed=None):
        """
        직선으로 갈 수 있는 구간을 하나로 합쳐 경유점을 줄인다.
        (격자 경로의 계단 모양을 제거)

        지름길은 점유뿐 아니라 통행 방향도 지켜야 한다. A*가 통로를 따라
        돌아온 경로라도, 두 점을 직선으로 이으면 그 직선이 역주행 구간을
        가로지를 수 있다. 그러면 계산은 일방통행인데 화면에 그려지는 선과
        차에 주는 안내만 역주행이 된다.

        Args:
            relaxed: relaxed(cell) -> bool. A*에 넘긴 것과 같은 함수여야 한다.
                     A*가 예외로 지나온 칸을 여기서 다시 따지면 그 구간만
                     계단으로 남는다.
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
            while j > i + 1 and not self._can_shortcut(cells, i, j, prefix, relaxed):
                j -= 1
            simplified.append(cells[j])
            i = j

        return simplified

    def _straighten(self, route):
        """
        비스듬한 구간을 없애고 가로/세로 구간만 남긴다.

        격자 경로는 원래 가로/세로뿐인데, plan()이 첫 점을 차의 실제 위치로,
        끝 점을 자리 중심으로 바꿔치기하면서 그 두 구간만 비스듬해진다.
        차가 차선 한가운데에서 벗어난 만큼 기울기가 생기고, 차가 움직일
        때마다 그 값이 바뀌므로 화면의 선이 계속 흔들린다.

        경유점을 새로 끼우지 않고 '모서리를 옆으로 옮겨서' 편다. 끼우면
        3cm짜리 구간이 하나 생기고, C00이 그것도 하나의 구간으로 세어
        "3cm 앞에서 좌회전" 같은 안내가 나온다. 모서리를 옮기면 구간 수가
        그대로라 안내도 그대로다.

        옮긴 모서리는 차선 중심에서 차가 벗어난 만큼(보통 몇 cm) 어긋나지만,
        그 자리는 차가 실제로 지나온 곳이므로 통로 안이다. 그래도 옮긴 두
        구간이 막히지 않는지는 확인하고, 막히면 원래대로 둔다.
        """
        if len(route) < 3:
            # 모서리가 없으면 옮길 것도 없다. 이런 경로는 자리 코앞에서
            # 시작한 경우뿐이라 비스듬해도 몇 cm짜리다.
            return route

        r = [(float(x), float(y)) for x, y in route]

        if len(r) == 3:
            # 모서리가 하나뿐이면 양 끝이 그 하나를 같이 쓴다.
            # 한쪽을 맞추면 다른 쪽이 틀어지므로 따로 다룬다.
            r[1] = self._corner_between(r[0], r[2], r[1])
        else:
            r[1] = self._snap_corner(r[0], r[1], r[2])
            r[-2] = self._snap_corner(r[-1], r[-2], r[-3])

        return self._drop_tiny(r)

    def _snap_to_cells(self, route):
        """
        구간을 격자 칸의 중심선에 맞춘다.

        세로 구간은 x를 그 칸 열의 중심으로, 가로 구간은 y를 그 칸 행의
        중심으로 옮긴다. 꼭지점 하나를 놓고 보면 x는 세로 구간이, y는 가로
        구간이 정하는 셈이라 두 구간이 서로 다툴 일이 없고, 옮긴 뒤에도
        직각이 그대로 남는다. (CONFIG['SNAP_TO_CELL_CENTERS'] 주석 참고)

        양 끝 구간은 건드리지 않는다. 차의 실제 위치와 자리 중심은 옮길 수
        없는 값이고, 그 구간의 축을 맞추려면 그것부터 옮겨야 하기 때문이다.

        구간 하나를 옮길 때마다 그 구간과 양옆 구간이 여전히 뚫려 있는지
        확인하고, 막히면 그 구간만 되돌린다. 통로를 벗어나는 자리로 옮기느니
        몇 cm 어긋난 채로 두는 편이 낫다.
        """
        if len(route) < 3:
            return route

        eps = self.lot_map.resolution
        base = cell_to_world((0, 0))
        cell = (self.lot_map.cell_w, self.lot_map.cell_h)
        pts = self._drop_inside_spot([list(p) for p in route])
        # 층계는 옮기기 전에 편다. A*가 남긴 한 칸짜리 층계를 그대로 두고
        # 구간을 차선에 붙이면, 층계 양쪽이 서로 다른 줄에 앉아 오히려
        # 눈에 띄는 계단이 된다.
        pts = self._merge_stairs(pts, len(self.wrong_way_legs(
            [tuple(q) for q in pts])))
        last = len(pts) - 1

        def snapped(a, b, axis):
            """
            구간을 붙일 자리.

            1) 나란한 순환 차선이 가까이 있으면 그 위. 차선은 통로 한가운데를
               지나므로(data/map_data.py의 ONE_WAY_LOOP) 이것이 곧 길 중앙이다.
               A*는 최단거리를 따라 모서리를 안쪽으로 질러 가므로 차선에서
               한두 칸 벗어난 채로 나오는데, 그 어긋남을 여기서 되돌린다.
            2) 차선이 없는 곳(섬으로 들어가는 길 등)은 그 구간이 지나는
               통로의 한가운데 칸.
            3) 둘 다 아니면 가장 가까운 칸 중심.
            """
            lane = self._lane_coord(a, b, axis)
            if lane is not None:
                return lane
            middle = self._corridor_center(a, b, axis)
            if middle is not None:
                return middle
            size = cell[axis]
            value = (a[axis] + b[axis]) / 2.0
            return base[axis] + round((value - base[axis]) / size) * size

        def leg_ok(i):
            """i번 구간이 뚫려 있는지. 범위 밖이면 볼 것이 없다."""
            if i < 0 or i >= last:
                return True
            return self._clear(pts[i], pts[i + 1])

        # 옮기다가 역주행을 만들면 안 된다. 차선을 통로 한가운데로 옮겼으므로
        # 자리로 들어가는 구간이 그만큼 길어졌고, 그 구간이 목적지 예외 구역
        # (GOAL_RELAX_COLUMNS)을 벗어나면 역주행으로 잡힌다. 계획 결과보다 나빠지면
        # 그 구간만 되돌린다.
        wrong_before = len(self.wrong_way_legs([tuple(p) for p in pts]))

        # 구간은 0번부터 last-1번까지다. 0번은 차에, last-1번은 자리에 닿아
        # 있으므로 건드리지 않는다. 그 사이만 옮긴다.
        for i in range(1, last - 1):
            a, b = pts[i], pts[i + 1]
            vertical = abs(b[0] - a[0]) < eps
            horizontal = abs(b[1] - a[1]) < eps
            # 축이 하나로 정해지는 구간만 맞춘다. 점에 가까운 구간(둘 다 참)은
            # 어느 쪽으로 옮겨야 할지 정할 수 없다.
            if vertical == horizontal:
                continue

            axis = 0 if vertical else 1
            value = snapped(a, b, axis)
            before = (a[axis], b[axis])
            if abs(value - before[0]) < eps and abs(value - before[1]) < eps:
                continue

            a[axis] = b[axis] = value
            ok = (leg_ok(i - 1) and leg_ok(i) and leg_ok(i + 1)
                  and len(self.wrong_way_legs([tuple(p) for p in pts]))
                  <= wrong_before)
            if not ok:
                a[axis], b[axis] = before      # 되돌린다

        pts = self._align_goal_leg(pts)
        return self._drop_straight([tuple(p) for p in pts])

    def _merge_stairs(self, pts, wrong_before):
        """
        계단으로 남은 세 구간을 'ㄱ'자 두 구간으로 합친다.

        A*가 통로에서 자리 쪽으로 빠질 때 한 칸짜리 층계가 섞여 나오는 일이
        있다. 단순화가 그것을 못 펴는 이유는 차선 비용 때문인데(지름길이
        차선에서 멀어지면 비싸다고 본다), 칸 중심으로 옮기고 나면 그 층계가
        '통로를 따라가다 한 칸 옆으로 비켰다가 다시 가는' 눈에 띄는 군더더기가
        된다. C-1로 갈 때 오른쪽 통로에서 한 칸 왼쪽으로 비켰다가 올라가는
        경로가 그것이었다.

        양쪽 끝점은 그대로 두고 가운데 두 점을 모서리 하나로 바꾼다. 끝점이
        칸 중심이므로 새 모서리도 칸 중심이다.

        '차선에서 더 멀어지지 않을 것'을 조건으로 단다. 이것이 없으면 층계를
        편다면서 통로 자체를 질러 버린다. 실제로 입구에서 A-1까지가 통로를
        도는 대신 주차장 한복판을 곧장 가로지르는 두 구간으로 접혔다.
        방향장으로는 역주행이 아니라서(가운데도 오른쪽 통로가 제일 가깝다)
        그 검사만으로는 걸리지 않는다. A*가 그 길을 안 고른 이유가 차선
        비용이었으므로, 같은 잣대를 여기서도 쓴다.
        """
        tolerance = CONFIG['ONE_WAY']['LANE_SHORTCUT_TOLERANCE_CM']
        i = 0
        while i + 3 < len(pts):
            p0, p1, p2, p3 = pts[i:i + 4]
            before = (self._lane_cost_of(p0, p1) + self._lane_cost_of(p1, p2)
                      + self._lane_cost_of(p2, p3))
            for corner in ([p3[0], p0[1]], [p0[0], p3[1]]):
                if not (self._clear(p0, corner) and self._clear(corner, p3)):
                    continue
                after = (self._lane_cost_of(p0, corner)
                         + self._lane_cost_of(corner, p3))
                if after > before + tolerance:
                    continue
                trial = pts[:i + 1] + [corner] + pts[i + 3:]
                if len(self.wrong_way_legs([tuple(p) for p in trial])) > wrong_before:
                    continue
                pts = trial
                break
            else:
                i += 1
        return pts

    def _lane_cost_of(self, p, q):
        """구간 하나가 차선에서 벗어난 값 (cm 환산). A*가 매기는 것과 같다."""
        if self.one_way is None:
            return 0.0
        cells = list(bresenham_cells(self.lot_map.world_to_cell(p, clamp=True),
                                     self.lot_map.world_to_cell(q, clamp=True)))
        if len(cells) < 2:
            return 0.0
        per_cell = math.hypot(q[0] - p[0], q[1] - p[1]) / (len(cells) - 1)
        return sum(self.one_way.lane_cost(c) for c in cells[1:]) * per_cell

    def _in_spot_cell(self, point):
        """이 점이 주차 구역 칸 안에 있는지. (목적지 구역도 포함)"""
        col = int(round((point[0] - cell_to_world((0, 0))[0]) / self.lot_map.cell_w))
        row = int(round((point[1] - cell_to_world((0, 0))[1]) / self.lot_map.cell_h))
        if not (0 <= row < get_rows() and 0 <= col < get_cols()):
            return False
        return grid_map[row][col] in SPOT_CELLS

    def _drop_inside_spot(self, pts):
        """
        자리 안에서 몇 cm 움직이는 토막을 지운다.

        A*는 자리 칸에 닿자마자 멈추므로 진입이 자리 중심이 아니라 칸 모서리
        쪽에서 끝난다. 그래서 '자리 앞에서 꺾어 들어간 다음 자리 안에서 한 번
        더 꺾어 중심으로 가는' 토막이 남는다. 화면에서는 자리 앞에서 두 번
        꺾이는 모양이라 어디에 세우라는 것인지 흐려진다.

        지우고 나면 마지막 구간이 '통로에서 자리 중심으로'가 되어, 아래
        _align_goal_leg가 그것을 자리 중심에 맞출 수 있다.

        통로에서 들어오는 구간 하나는 반드시 남긴다. 목적지와 차 위치만
        남으면 방향을 알 수 없다.
        """
        while len(pts) > 3 and self._in_spot_cell(pts[-2]):
            del pts[-2]
        return pts

    def _align_goal_leg(self, pts):
        """
        자리로 들어가는 마지막 구간을 자리 중심에 맞춘다.

        자리 중심은 그 자체로 칸 중심이므로, 진입 구간을 거기에 맞추면
        '통로를 따라가다 자리 앞에서 한 번 꺾어 그대로 들어간다'가 된다.

        통로 구간을 차선 위로 옮긴 뒤에 부른다. 어느 축으로 들어가는지를
        구간의 긴 쪽으로 정하는데, 옮기기 전에는 그 길이가 A*가 자리 칸
        모서리에 걸쳐 놓은 값이라 축이 뒤집혀 나온다.
        """
        if len(pts) < 3:
            return pts

        goal, corner = pts[-1], pts[-2]
        dx, dy = goal[0] - corner[0], goal[1] - corner[1]
        axis = 1 if abs(dx) >= abs(dy) else 0    # 긴 쪽이 진행 방향
        before = corner[axis]
        if goal[axis] == before:
            return pts          # 이미 맞다

        # 여기서는 격자 한 칸(1cm)을 봐주지 않는다. 0.2cm만 어긋나도 자리
        # 앞에서 선이 살짝 기운 것이 보이고, 그 구간이 짧아서 더 도드라진다.

        corner[axis] = goal[axis]
        if not (self._clear(corner, goal) and self._clear(pts[-3], corner)
                and not self.wrong_way_legs([tuple(p) for p in pts])):
            corner[axis] = before
        return pts

    def _lane_coord(self, a, b, axis):
        """
        이 구간과 나란히 붙어 있는 순환 차선의 좌표. 없으면 None.

        차선은 통로 한가운데를 지나므로 여기에 붙이는 것이 곧 '길 중앙으로
        붙이기'다. 찾는 조건은 둘이다.
          - 구간과 나란할 것 (세로 구간이면 세로 차선)
          - 구간이 그 차선을 옆에서 지나가는 중일 것. 차선의 시작과 끝
            사이를 지나야 하고, 옆으로 벌어진 거리가 LANE_SNAP_CM 안이어야
            한다. 이 조건이 없으면 저 멀리 있는 차선에 끌려간다.

        Args:
            a, b: 구간의 두 끝점 (실좌표)
            axis: 0이면 세로 구간(x를 옮긴다), 1이면 가로 구간(y를 옮긴다)
        """
        if self.one_way is None:
            return None

        along = 1 - axis            # 구간이 뻗어 있는 축
        lo, hi = sorted((a[along], b[along]))
        best, best_d = None, CONFIG['ONE_WAY']['LANE_SNAP_CM']

        for p, q in self.one_way.segments:
            if abs(p[axis] - q[axis]) > self.lot_map.resolution:
                continue        # 나란하지 않다
            # 겹치는 구간이 있어야 '옆에 나란히 있는' 차선이다
            s_lo, s_hi = sorted((p[along], q[along]))
            if min(hi, s_hi) < max(lo, s_lo):
                continue
            d = abs(p[axis] - (a[axis] + b[axis]) / 2.0)
            if d <= best_d:
                best, best_d = p[axis], d
        return best

    def _corridor_center(self, a, b, axis):
        """
        이 구간이 지나는 통로의 한가운데 칸. 옮길 곳이 없으면 None.

        구간을 옆 칸으로 통째로 밀어 보면서, 끝에서 끝까지 뚫려 있는 칸이
        어디까지인지 양쪽으로 넓혀 본다. 그렇게 나온 칸의 범위가 지금 지나는
        통로이고, 그 한가운데로 옮긴다.

        통로 폭을 미리 표로 두지 않고 매번 재는 이유: 이 주차장은 자리
        배치에 따라 통로 폭이 자리마다 다르다. 오른쪽은 섬과 D열 사이 네 칸,
        섬 사이는 한 칸이다. 격자를 고치면 폭도 따라 바뀌어야 한다.

        Args:
            a, b: 구간의 두 끝점 (실좌표)
            axis: 0이면 세로 구간(x를 옮긴다), 1이면 가로 구간(y를 옮긴다)
        """
        base = cell_to_world((0, 0))[axis]
        size = (self.lot_map.cell_w, self.lot_map.cell_h)[axis]
        limit = get_cols() if axis == 0 else get_rows()

        def clear_at(k):
            """구간을 k번째 칸의 중심으로 옮겨도 뚫려 있는지."""
            p, q = list(a), list(b)
            p[axis] = q[axis] = base + k * size
            return self._clear(p, q)

        here = int(round((a[axis] - base) / size))
        if not clear_at(here):
            return None         # 칸 중심으로는 아예 못 옮기는 구간

        lo = hi = here
        while lo - 1 >= 0 and clear_at(lo - 1):
            lo -= 1
        while hi + 1 < limit and clear_at(hi + 1):
            hi += 1

        # 통로가 짝수 칸이면 한가운데가 두 칸 사이에 있다. 주차장 중심에
        # 가까운 쪽을 고른다. 바깥쪽을 고르면 자리에 바짝 붙어 달리게 된다.
        center = (limit - 1) / 2.0
        middle = (lo + hi) / 2.0

        # 옮기는 폭은 한 칸까지다. 짧은 구간은 옆으로 넓게 열려 있는 일이
        # 흔해서(아랫줄 한 토막은 위로 주차장 끝까지 뚫려 있다) 폭을 두지
        # 않으면 '통로의 한가운데'가 주차장 한복판이 되어, 계획한 길과 전혀
        # 다른 곳으로 선이 날아간다. 한 칸이면 가장자리에서 떼어 놓기에
        # 충분하고 경로 모양은 그대로 남는다.
        near = [k for k in (here - 1, here, here + 1)
                if 0 <= k < limit and clear_at(k)]
        k = min(near, key=lambda c: (abs(c - middle), abs(c - center)))
        return base + k * size

    def _drop_straight(self, route):
        """
        옮기면서 한 직선이 된 꼭지점을 지운다.

        자리로 들어가는 마지막 구간에서 생긴다. 앞 구간을 행 중심으로 맞추면
        그 행에 있던 자리 중심과 같은 높이가 되어, 가운데 꼭지점이 직선 위의
        한 점이 된다. 남겨 두면 화면에 꺾이지 않는 꺾임점이 찍힌다.
        """
        eps = self.lot_map.resolution
        out = [route[0]]
        for cur, nxt in zip(route[1:-1], route[2:]):
            prev = out[-1]
            # 앞뒤가 같은 가로줄이거나 같은 세로줄이면 가운데는 필요 없다
            same_row = abs(prev[1] - cur[1]) < eps and abs(cur[1] - nxt[1]) < eps
            same_col = abs(prev[0] - cur[0]) < eps and abs(cur[0] - nxt[0]) < eps
            if same_row or same_col:
                continue
            out.append(cur)
        out.append(route[-1])
        return self._drop_tiny(out)

    def _clear(self, a, b):
        """두 실좌표 사이가 뚫려 있는지. (격자 점유만 본다)"""
        return self.lot_map.line_of_sight(
            self.lot_map.world_to_cell(a, clamp=True),
            self.lot_map.world_to_cell(b, clamp=True))

    def _snap_corner(self, anchor, corner, other):
        """
        anchor -> corner 구간이 가로나 세로가 되도록 corner를 옮긴다.

        Args:
            anchor: 옮길 수 없는 끝점 (차의 위치나 자리 중심)
            corner: 옮길 모서리 경유점
            other:  모서리 반대쪽 경유점. 이쪽 구간의 방향은 유지해야 한다.

        Returns:
            옮긴 모서리. 옮길 수 없으면 원래 값.
        """
        eps = self.lot_map.resolution
        dx = corner[0] - anchor[0]
        dy = corner[1] - anchor[1]
        if abs(dx) < eps or abs(dy) < eps:
            return corner       # 이미 가로/세로다

        # 반대쪽 구간이 세로면 모서리의 x를, 가로면 y를 지켜야 한다.
        # 그래야 그 구간까지 같이 틀어지지 않는다.
        keep_x = abs(other[0] - corner[0]) < eps
        keep_y = abs(other[1] - corner[1]) < eps
        if keep_x and not keep_y:
            first = (corner[0], anchor[1])
        elif keep_y and not keep_x:
            first = (anchor[0], corner[1])
        else:
            # 반대쪽도 비스듬하면(있을 수 없지만) 긴 축을 진행 방향으로 본다.
            first = ((corner[0], anchor[1]) if abs(dx) >= abs(dy)
                     else (anchor[0], corner[1]))

        # 원래 구간이 이미 막혀 있었다면(차가 자리나 여유 영역 안에 서 있는
        # 경우다) 뚫림을 따지지 않는다. 어차피 더 나빠질 것이 없다.
        strict = self._clear(anchor, corner)
        second = ((anchor[0], corner[1]) if first[0] == corner[0]
                  else (corner[0], anchor[1]))
        for cand in (first, second):
            if not strict or (self._clear(anchor, cand) and self._clear(cand, other)):
                return cand
        return corner

    def _corner_between(self, a, b, prefer):
        """
        a와 b를 가로 한 번, 세로 한 번으로 잇는 모서리 점.
        원래 모서리(prefer)에 가까운 쪽을 먼저 쓰고, 막혀 있으면 반대쪽.
        """
        eps = self.lot_map.resolution
        if abs(b[0] - a[0]) < eps or abs(b[1] - a[1]) < eps:
            return prefer       # a-b가 이미 한 직선이면 모서리를 건드리지 않는다

        cands = [(b[0], a[1]), (a[0], b[1])]
        cands.sort(key=lambda c: math.hypot(c[0] - prefer[0], c[1] - prefer[1]))

        strict = self._clear(a, prefer) and self._clear(prefer, b)
        for cand in cands:
            if not strict or (self._clear(a, cand) and self._clear(cand, b)):
                return cand
        return prefer

    def _drop_tiny(self, route):
        """
        모서리를 옮기면서 길이가 0에 가까워진 구간을 지운다.
        양 끝점은 남긴다. (차의 위치와 자리 중심이라 바꿀 수 없다)
        """
        eps = self.lot_map.resolution
        out = [route[0]]
        for pt in route[1:-1]:
            if math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) >= eps:
                out.append(pt)
        last = route[-1]
        while len(out) > 1 and math.hypot(last[0] - out[-1][0],
                                          last[1] - out[-1][1]) < eps:
            out.pop()
        out.append(last)
        return out

    def wrong_way_legs(self, route):
        """
        경로에서 역주행하는 구간을 찾는다. (검증 / 자체 시험용)

        Args:
            route: 경유점 리스트 [(x_cm, y_cm), ...]

        Returns:
            [(구간 인덱스, 시작점, 끝점), ...]. 비어 있으면 일방통행을 지킨 것.
            목적지 예외 구역(GOAL_RELAX_COLUMNS) 안은 세지 않는다. 계획기가 일부러
            봐준 곳이므로, 여기서 세면 정상 경로가 실패로 나온다.
        """
        if self.one_way is None or not route or len(route) < 2:
            return []

        # 경로의 끝점이 목적지다. 계획기가 쓴 것과 같은 예외 구역을 만든다.
        relaxed = self._make_relaxed(route[-1])

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
            if relaxed is not None:
                cells = [c for c in cells if not relaxed(c)]
            if any(self.one_way.is_wrong_way(cell, dc, dr) for cell in cells):
                bad.append((i, a, b))
        return bad

    def _can_shortcut(self, cells, i, j, prefix=None, relaxed=None):
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
        dc = cell_b[0] - cell_a[0]
        dr = cell_b[1] - cell_a[1]

        # 4방향으로 계획했으면 지름길도 가로/세로여야 한다.
        # 이걸 빼면 계단 모양을 편다면서 대각선 구간을 만들어, 대각선을
        # 없앤 의미가 사라진다. (MOVES 주석 참고)
        if not HAS_DIAGONAL and dc and dr:
            return False

        if self.one_way is None:
            return self.lot_map.line_of_sight(cell_a, cell_b)

        line = []
        for cell in bresenham_cells(cell_a, cell_b):
            if not self.lot_map.is_free(cell):
                return False
            if not (relaxed and relaxed(cell)) and self.one_way.is_wrong_way(cell, dc, dr):
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
def route_from_position(route, index, position, straight=None):
    """
    차의 현재 위치에서 시작하는 '남은 경로'를 만든다. (화면에 그릴 선)

    이미 지나온 경유점을 빼고 남은 것만 이어야 차 뒤로 선이 남지 않는다.
    그런데 그냥 앞에 차 위치를 끼우면, 차가 차선 한가운데에서 벗어난 만큼
    첫 구간이 비스듬해진다.

    경유점(꼭지점)은 한 번 계획한 뒤로는 건드리지 않는다. 차에 맞춰 모서리를
    옮기면 선이 가로/세로로 떨어지기는 하지만, 차가 조금 움직일 때마다 그
    모서리가 따라 움직여서 선 전체가 계속 흔들린다. 화면에서는 그것이
    '직선으로 안 움직인다'로 보인다.

    대신 차를 선 위에 세운다. 지금 달리고 있는 구간(index-1 -> index)에
    차의 위치를 수직으로 내려 찍고, 그 점에서 선을 시작한다. 구간이 가로나
    세로이므로 첫 구간도 반드시 가로나 세로가 되고, 꼭지점은 계획된 자리에
    그대로 박혀 있다. 실제 자동차 내비게이션이 하는 것과 같다. 차가 차선을
    조금 벗어나 있으면 선이 차 옆을 지나가는데, 벗어난 거리는 재계획 기준
    (REPLAN_TOLERANCE_CM)보다 작으므로 몇 cm다.

    Args:
        route:    경유점 리스트 (cm)
        index:    지금 향하고 있는 경유점 인덱스
        position: 차의 현재 실좌표 (cm). None이면 경로만 잘라서 돌려준다.
        straight: 선을 구간 위에 세울지 여부. None이면 CONFIG를 따른다.
                  끄면 예전처럼 차 위치를 그대로 앞에 붙인다.

    Returns:
        [(x, y), ...] 차 앞의 선. 목적지로 끝난다.
    """
    if straight is None:
        straight = CONFIG['STRAIGHT_LEGS_ONLY']

    if not route:
        return [] if position is None else [(float(position[0]), float(position[1]))]

    idx = min(max(index, 0), len(route) - 1)
    rest = [(float(p[0]), float(p[1])) for p in route[idx:]]
    if position is None:
        return rest

    pos = (float(position[0]), float(position[1]))
    if not straight:
        return [pos] + rest

    # 달리고 있는 구간에 차를 내려 찍는다. 첫 경유점을 향하는 중이라
    # 이전 구간이 없으면(index 0) 차 위치를 그대로 쓴다.
    start = pos if idx == 0 else _project_on_segment(pos, route[idx - 1], rest[0])

    # 그 점이 다음 경유점에 거의 닿아 있으면 길이 0인 구간이 생긴다.
    # C00이 구간마다 방향을 재므로 그런 구간은 방향이 수치 잡음이 된다.
    if math.hypot(rest[0][0] - start[0], rest[0][1] - start[1]) < \
            CONFIG['GRID_RESOLUTION_CM']:
        return rest
    return [start] + rest


def _project_on_segment(p, a, b):
    """점 p에서 선분 ab에 내린 수선의 발. 선분 밖이면 가까운 끝점."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return (ax, ay)

    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom))
    return (ax + t * dx, ay + t * dy)


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
