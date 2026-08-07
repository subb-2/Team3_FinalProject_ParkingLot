import cv2
import sys
import os
import math
import numpy as np

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.C00_navigation import (
    SPOT_WORLD_POS, GATE1_WORLD_POS, GATE2_WORLD_POS,
    GUIDE_ARRIVED, GUIDE_UNKNOWN, GUIDE_STRAIGHT, GUIDE_LEFT,
    GUIDE_RIGHT, GUIDE_UTURN,
)
from logic.C02_lot_layout import cell_to_world, CONFIG as C02_CONFIG
from data.map_data import (
    grid_map, coord_to_spot, PILL_MARKER_ID, spot_type,
    ROAD, SPOT_CELLS, SPOT_TYPE_NAME, SPOT1, SPOT2, SPOT3, SPOT4,
    GATE1, GATE2, PILL, get_rows, get_cols, get_spot_cell_count,
)

# 설정 (Configuration)
# 이 모듈은 '화면 표시'만 담당.
#   - 위치 추정 / 경로 안내 : C00_navigation.py
#   - 통합 실행             : C_main.py
CONFIG = {
    # 맵 화면 크기 (px)
    "MAP_WIDTH": 1000,
    "MAP_HEIGHT": 620,
    "PANEL_RATIO": 0.30,        # 오른쪽 안내 패널이 차지하는 가로 비율

    # 맵 여백 및 축척
    "MARGIN_PX": 50,            # 맵 영역 바깥 여백
    "PAD_CM": 8.0,              # 격자 바깥으로 확보할 여유 공간 (주차장 크기에 맞게 조정)

    # 주차 구역 크기 (cm).
    # 격자 한 칸(C02의 CELL_W/H_CM)에서 자동으로 가져오되, 칸을 꽉 채우면
    # 옆 칸과 붙어 보이므로 아래 비율만큼 줄여서 그린다.
    "SPOT_FILL_RATIO": 0.86,

    # 맵에 격자 배경(벽/도로/기둥)을 그릴지 여부
    "SHOW_LAYOUT": True,
    "SHOW_PILLAR_ID": True,     # 기둥에 ArUco 마커 ID 표시

    # 표시 옵션
    "SHOW_GRID": True,          # 배경 격자 표시
    "GRID_STEP_CM": 20.0,       # 격자 간격 (cm)
    "SHOW_TRAJECTORY": True,    # 차량 이동 궤적 표시
    "TRAJECTORY_MAX_POINTS": 60,
    "VEHICLE_RADIUS_PX": 11,

    # ---------------------------------------------------------------
    # 차량 시점 내비게이션 화면 (NavigationView) 설정
    # ---------------------------------------------------------------
    "NAV_WIDTH": 900,           # 화면 크기 (px)
    "NAV_HEIGHT": 620,
    "NAV_PX_PER_CM": 7.0,       # 확대 배율. 클수록 차량 주변만 크게 보인다
    "NAV_CAR_Y_RATIO": 0.74,    # 화면에서 내 차량이 놓이는 세로 위치 비율
    "NAV_HORIZON_RATIO": 0.30,  # 지평선 높이 비율 (작을수록 멀리까지 보임)
    "NAV_TOP_WIDTH_RATIO": 0.30, # 원근 상단 폭 비율 (작을수록 원근감이 강함)
    "NAV_BANNER_H": 118,        # 상단 턴 안내 배너 높이
    "NAV_HEADING_SMOOTH": 0.25, # 화면 회전 감속 계수 (0~1, 작을수록 부드러움)

    # 화면을 차량 진행 방향에 맞춰 돌릴지 여부.
    #
    #   False (기본) : 방위 고정. 2번 주차장 상태 화면과 같은 방향으로 서 있고,
    #                  차가 움직여도 위아래가 바뀌지 않는다. 목업처럼 주차장
    #                  전체가 한눈에 들어오는 크기에서는 화면이 계속 도는 것보다
    #                  어디가 어디인지 알아보기 쉽다.
    #   True         : 헤딩업. 진행 방향이 항상 화면 위쪽이 된다. (자동차 내비 방식)
    #
    # 어느 쪽이든 내 차 아이콘은 실제 진행 방향을 가리킨다.
    "NAV_HEADING_UP": False,
    "NAV_SHOW_MINIMAP": True,   # 좌하단 전체 조감도 표시
    "NAV_MINIMAP_W": 210,
    "NAV_MINIMAP_H": 168,
}

# 색상 (BGR)
COLOR_BG          = (34, 34, 34)     # 배경
COLOR_PANEL_BG    = (24, 24, 24)     # 패널 배경
COLOR_GRID        = (54, 54, 54)     # 격자
COLOR_TEXT        = (235, 235, 235)  # 기본 텍스트
COLOR_TEXT_DIM    = (150, 150, 150)  # 보조 텍스트
COLOR_SPOT_EMPTY  = (130, 130, 130)  # 빈자리 테두리
COLOR_SPOT_FULL   = (70, 70, 200)    # 주차중 (붉은 계열)
COLOR_SPOT_TARGET = (255, 0, 255)    # 목표 구역 (자홍)
COLOR_GATE        = (0, 200, 255)    # 입출구 (주황)
# 주차 구역 종류별 빈자리 테두리 색 (BGR).
# 실제 주차장의 노면 표시 관례를 따라 구분한다.
COLOR_SPOT_BY_TYPE = {
    SPOT1: (130, 130, 130),   # 일반   - 회색
    SPOT2: (255, 150, 0),     # 장애인 - 파랑
    SPOT3: (60, 200, 255),    # 대형   - 노랑
    SPOT4: (120, 220, 120),   # 전기차 - 초록
}
COLOR_ROAD        = (44, 44, 46)     # 도로 (통로)
COLOR_PILL        = (74, 80, 92)     # 기둥 (ArUco 마커 위치) - 콘크리트 느낌
COLOR_PILL_EDGE   = (105, 118, 138)  # 기둥 테두리
COLOR_PILL_TEXT   = (150, 175, 205)  # 기둥의 마커 ID
COLOR_VEHICLE     = (0, 230, 0)      # 차량 (번호 매칭됨)
COLOR_VEHICLE_UNK = (0, 165, 255)    # 차량 (번호 미매칭)
COLOR_TRAJECTORY  = (0, 220, 220)    # 이동 궤적
COLOR_GUIDE_LINE  = (0, 255, 255)    # 안내선
COLOR_ARRIVED     = (0, 255, 0)      # 도착 표시

# 차량 시점 화면 색상 (BGR)
COLOR_NAV_SKY     = (48, 40, 34)     # 지평선 위 (원근 바깥 영역)
COLOR_NAV_GROUND  = (40, 40, 42)     # 노면
COLOR_NAV_BANNER  = (26, 26, 28)     # 상단 배너 배경
COLOR_NAV_ROUTE   = (255, 200, 40)   # 주행 경로 (하늘색 계열)
COLOR_NAV_ROUTE_E = (255, 235, 150)  # 경로 테두리
COLOR_NAV_CAR     = (80, 230, 80)    # 내 차량
COLOR_NAV_ACCENT  = (60, 220, 255)   # 강조 (거리, 화살표)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# 방위 고정 화면에서 쓰는 시선 방향.
# _world_to_flat의 각도 규약(x축 오른쪽 0도, y축 아래로 +90도)에서 -90도가
# '월드의 위쪽(-y)이 화면 위쪽'이다. 즉 2번 주차장 상태 화면과 같은 방향.
FIXED_VIEW_HEADING = -90.0


# 좌표계 해석
#
# 이 시스템은 좌표계가 두 가지다. 화면을 그리는 쪽이 둘을 섞으면
# 차는 이쪽, 주차 구역은 저쪽에 그려져 아무것도 맞지 않는다.
#
#   1) 호모그래피 모드 : 마커로 호모그래피를 잡은 경우.
#      C00이 주는 world_pos가 실제 cm다. 배치도 cm(SPOT_WORLD_POS)다.
#   2) 픽셀 모드       : /calibrate에서 기둥을 직접 찍은 경우.
#      호모그래피가 없다. C00은 world_pos에 '이미지 픽셀 좌표'를 그대로
#      담아 주고(C00_navigation.update의 is_pixel_mode 분기), 자리 좌표도
#      navigator.spot_pixels의 픽셀 좌표다.
#
# 아래 함수가 '지금 어느 쪽인지'와 '그 좌표계에서의 배치'를 한 곳에서 정한다.
def _median(values):
    """중앙값. 비어 있으면 None. (원근 때문에 평균보다 중앙값이 안전하다)"""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def cell_size_from_pillars(pillar_pixels):
    """
    클릭한 기둥 픽셀 좌표에서 격자 한 칸의 픽셀 크기를 추정한다.

    픽셀 모드에는 cm 환산이 없다. CELL_W_CM(10cm)에 임의의 상수(8px/cm)를
    곱해 쓰면 가로세로 비율이 실제 화면과 어긋난다. 카메라가 비스듬히 보고
    있어서 가로 1cm와 세로 1cm의 픽셀 수가 애초에 다르기 때문이다.
    (이 목업에서는 세로가 가로의 절반도 안 된다)

    같은 행에 있는 기둥 쌍에서 가로 간격을, 같은 열에 있는 쌍에서 세로
    간격을 재면 화면에 실제로 보이는 칸 크기가 그대로 나온다.

    Returns:
        (cell_w_px, cell_h_px). 잴 수 없으면 (None, None).
    """
    cell_of = {mid: cell for cell, mid in PILL_MARKER_ID.items()}
    known = [(cell_of[mid], pos) for mid, pos in pillar_pixels.items()
             if mid in cell_of]

    dxs, dys = [], []
    for i, ((r1, c1), p1) in enumerate(known):
        for (r2, c2), p2 in known[i + 1:]:
            if r1 == r2 and c1 != c2:
                dxs.append(abs(p2[0] - p1[0]) / abs(c2 - c1))
            elif c1 == c2 and r1 != r2:
                dys.append(abs(p2[1] - p1[1]) / abs(r2 - r1))

    cell_w, cell_h = _median(dxs), _median(dys)
    if cell_w is None and cell_h is None:
        return None, None

    # 한쪽만 재졌으면 설계상의 가로세로 비로 나머지를 추정한다
    ratio = C02_CONFIG['CELL_H_CM'] / C02_CONFIG['CELL_W_CM']
    if cell_w is None:
        cell_w = cell_h / ratio
    if cell_h is None:
        cell_h = cell_w * ratio
    return cell_w, cell_h


def resolve_layout(navigator):
    """
    지금 좌표계에 맞는 배치를 돌려준다.

    Args:
        navigator: ParkingNavigator 인스턴스 (None이면 cm 기본 배치)

    Returns:
        {
          "pixel_mode": bool,
          "spot_pos":   {구역ID: (x, y)},   # 모드에 맞는 단위 (cm 또는 px)
          "gate_pos":   (x, y) 또는 None,
          "cell_w":     격자 한 칸의 가로 크기 (같은 단위),
          "cell_h":     격자 한 칸의 세로 크기 (같은 단위),
        }
    """
    mapper = getattr(navigator, 'mapper', None)
    pillar_pixels = getattr(mapper, 'pillar_pixels', None)

    if pillar_pixels:
        spot_pos = dict(getattr(navigator, 'spot_pixels', None) or mapper.spot_pixels)
        gate_pos = mapper.gate_pixels[0] if mapper.gate_pixels else None
        cell_w, cell_h = cell_size_from_pillars(pillar_pixels)
        if cell_w is None:
            # 기둥이 한 줄로만 찍힌 경우. 자리 간격으로라도 크기를 잡는다.
            cell_w = cell_h = 60.0
        return {"pixel_mode": True, "spot_pos": spot_pos, "gate_pos": gate_pos,
                "cell_w": cell_w, "cell_h": cell_h}

    spot_pos = dict(getattr(navigator, 'spot_world_pos', None) or SPOT_WORLD_POS)
    return {"pixel_mode": False, "spot_pos": spot_pos, "gate_pos": GATE1_WORLD_POS,
            "cell_w": C02_CONFIG['CELL_W_CM'], "cell_h": C02_CONFIG['CELL_H_CM']}

# 안내 상수 -> 화면에 띄울 짧은 영문 (OpenCV는 한글 렌더링 불가)
MANEUVER_LABEL = {
    GUIDE_STRAIGHT: "GO STRAIGHT",
    GUIDE_LEFT:     "TURN LEFT",
    GUIDE_RIGHT:    "TURN RIGHT",
    GUIDE_UTURN:    "MAKE U-TURN",
    GUIDE_ARRIVED:  "ARRIVING",
    GUIDE_UNKNOWN:  "SEARCHING",
}


# 차량 시점 내비게이션 화면
class NavigationView:
    """
    자동차 내부 내비게이션처럼 보이는 화면.

    NavigationMapUI가 주차장을 천장에서 내려다보는 관제 화면이라면,
    이 클래스는 운전자 시점이다. 세 가지가 다르다.

      1. 헤딩업(Heading-up): 차량의 진행 방향이 항상 화면 위쪽이 되도록
         지도 전체를 회전시킨다. 내 차는 화면 아래쪽에 고정된다.
      2. 원근 시점: 노면 레이어에 사다리꼴 변환을 걸어, 먼 곳은 좁고
         가까운 곳은 넓게 보이게 한다.
      3. 턴 안내 배너: 다음에 무엇을 해야 하는지("30cm 앞 우회전")를
         화면 상단에 크게 띄운다.

    한 대의 차량('내 차')만 표시한다. 여러 대를 한 번에 보려면
    NavigationMapUI를 쓸 것.
    """

    def __init__(self, navigator=None, width=None, height=None,
                 spot_world_pos=None, gate_world_pos=None):
        """
        NavigationView 초기화.

        Args:
            navigator:      ParkingNavigator 인스턴스 (구역 좌표 조회용)
            width/height:   화면 크기 (px)
            spot_world_pos: {구역ID: (x_cm, y_cm)} 매핑
            gate_world_pos: 입출구 실좌표
        """
        self.navigator = navigator
        self.width = width or CONFIG['NAV_WIDTH']
        self.height = height or CONFIG['NAV_HEIGHT']

        # 호출자가 직접 지정한 배치는 navigator보다 우선한다 (단독 테스트용)
        self._spot_override = dict(spot_world_pos) if spot_world_pos is not None else None
        self._gate_override = gate_world_pos

        self.car_y = int(self.height * CONFIG['NAV_CAR_Y_RATIO'])
        self.banner_h = CONFIG['NAV_BANNER_H']
        self._display_heading = None        # 지도 회전 (헤딩업일 때만 쓴다)
        self._display_car_heading = None    # 내 차 아이콘이 가리키는 방향

        self._layout = None
        self._sync_layout()
        self._build_perspective()

    def _sync_layout(self):
        """
        navigator의 보정 상태를 읽어 좌표계를 맞춘다. (매 프레임 호출)

        생성 시점에 한 번만 읽으면 안 된다. 보정(/calibrate)은 프로그램이
        뜬 뒤에 하므로, 그때는 자리 좌표가 아직 비어 있고 픽셀 모드인지도
        알 수 없다. 그 상태로 굳으면 차 위치는 픽셀(예: 332,260), 배치는
        cm(0~120) 좌표가 되어 주차장이 화면 밖 저 멀리 그려진다.
        안내 화면에 노면만 덩그러니 보이던 원인이 이것이다.
        """
        layout = resolve_layout(self.navigator)
        if self._spot_override is not None:
            layout['spot_pos'] = dict(self._spot_override)
        if self._gate_override is not None:
            layout['gate_pos'] = self._gate_override
        if layout == self._layout:
            return

        self._layout = layout
        self.is_pixel_mode = layout['pixel_mode']
        self.spot_world_pos = layout['spot_pos']
        self.gate_world_pos = layout['gate_pos']
        self.cell_w = layout['cell_w']
        self.cell_h = layout['cell_h']

        # 확대 배율. 두 좌표계에서 '화면에 보이는 한 칸의 크기'가 같아지도록
        # 맞춘다. cm 모드면 NAV_PX_PER_CM 그대로다.
        self.scale = (CONFIG['NAV_PX_PER_CM'] * C02_CONFIG['CELL_W_CM']) / self.cell_w

        unit = "px" if self.is_pixel_mode else "cm"
        print(f"[INFO] 차량 시점 내비게이션 화면 좌표계 설정. "
              f"({self.width}x{self.height}, {'픽셀' if self.is_pixel_mode else '호모그래피'} 모드, "
              f"자리 {len(self.spot_world_pos)}개, 한 칸 {self.cell_w:.0f}x{self.cell_h:.0f}{unit}, "
              f"배율 {self.scale:.2f})")

    def _build_perspective(self):
        """
        노면 레이어(위에서 본 그림)를 원근 시점으로 바꾸는 변환을 준비.

        위쪽 변을 좁히고 아래로 내려서, 먼 곳이 지평선으로 수렴하는
        자동차 내비게이션 특유의 시점을 만든다.
        """
        w, h = self.width, self.height
        top_w = w * CONFIG['NAV_TOP_WIDTH_RATIO']
        hz = h * CONFIG['NAV_HORIZON_RATIO']

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [w / 2 - top_w / 2, hz],
            [w / 2 + top_w / 2, hz],
            [w, h],
            [0, h],
        ])
        self._warp = cv2.getPerspectiveTransform(src, dst)

    # ---------- 좌표 변환 ----------
    def _world_to_flat(self, point, car_pos, heading_deg):
        """
        실좌표를 '차량 기준 위에서 본' 화면 좌표로 변환.
        (원근 변환 전 단계)

        차량을 원점으로 옮기고, 진행 방향이 화면 위쪽이 되도록 회전한다.
        """
        rad = math.radians(heading_deg)
        # y축이 아래 방향인 좌표계에서의 전방/우측 단위 벡터
        fx, fy = math.cos(rad), math.sin(rad)
        rx, ry = -math.sin(rad), math.cos(rad)

        dx = point[0] - car_pos[0]
        dy = point[1] - car_pos[1]

        ahead = dx * fx + dy * fy      # 전방 거리
        right = dx * rx + dy * ry      # 우측 거리

        return (self.width / 2 + right * self.scale,
                self.car_y - ahead * self.scale)

    def _flat_to_screen(self, pt):
        """원근 변환을 적용해 최종 화면 좌표를 얻는다. (텍스트 배치용)"""
        src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self._warp)
        return int(dst[0][0][0]), int(dst[0][0][1])

    def _view_heading(self, nav):
        """
        지도를 어느 방향으로 세울지 결정.

        NAV_HEADING_UP이 꺼져 있으면 방위를 고정한다. 차가 움직일 때마다
        화면 전체가 도는 것을 막기 위해서다.
        """
        if not CONFIG['NAV_HEADING_UP']:
            return FIXED_VIEW_HEADING
        return self._update_heading(nav)

    def _car_heading(self, nav, view_heading):
        """
        내 차 아이콘이 가리킬 방향(화면 기준 각도, 위쪽이 0).

        진행 방향을 아직 모르면(정지 등) 직전 값을 유지하고, 그것도 없으면
        화면 위쪽을 가리킨다. 매 프레임 튀지 않게 지도 회전과 같은 계수로
        완만하게 따라간다.
        """
        heading = nav.get("heading_deg")
        if heading is None:
            wp = nav.get("next_waypoint") or nav.get("target_world")
            pos = nav.get("world_pos")
            if wp is not None and pos is not None:
                heading = math.degrees(math.atan2(wp[1] - pos[1], wp[0] - pos[0]))
        if heading is None:
            heading = self._display_car_heading
        if heading is None:
            return 0.0

        if self._display_car_heading is None:
            self._display_car_heading = heading
        else:
            diff = (heading - self._display_car_heading + 180) % 360 - 180
            self._display_car_heading += diff * CONFIG['NAV_HEADING_SMOOTH']

        # 지도가 view_heading 방향으로 서 있으므로, 그 차이만큼만 돌리면 된다
        return (self._display_car_heading - view_heading + 180) % 360 - 180

    def _update_heading(self, nav):
        """
        화면 회전에 쓸 진행 방향을 결정.

        정지 상태라 진행 방향을 모를 때는 다음 경유점 쪽을 바라보게 하고,
        그것도 없으면 직전 값을 유지한다. 그대로 두면 차가 멈출 때마다
        화면이 튀어서 보기 어렵다. 급격한 회전도 완만하게 보간한다.
        """
        heading = nav.get("heading_deg")

        if heading is None:
            wp = nav.get("next_waypoint")
            pos = nav.get("world_pos")
            if wp is not None and pos is not None:
                heading = math.degrees(math.atan2(wp[1] - pos[1], wp[0] - pos[0]))

        if heading is None:
            heading = self._display_heading if self._display_heading is not None else 0.0

        if self._display_heading is None:
            self._display_heading = heading
        else:
            # 각도는 순환하므로 -180~180 범위의 최단 차이로 보간
            diff = (heading - self._display_heading + 180) % 360 - 180
            self._display_heading += diff * CONFIG['NAV_HEADING_SMOOTH']

        return self._display_heading

    # ---------- 렌더링 ----------
    def render(self, nav, spot_status=None, fps=None, extra_info=None):
        """
        차량 시점 내비게이션 화면 한 장을 그린다.

        Args:
            nav:         C00_navigation의 nav 결과 딕셔너리 하나 ('내 차').
                         None이면 대기 화면을 보여준다.
            spot_status: {구역ID: "empty"|"full"} 점유 상태
            fps:         표시할 FPS
            extra_info:  하단에 추가로 표시할 문자열 리스트

        Returns:
            렌더링된 BGR 이미지
        """
        if spot_status is None:
            from data.map_data import spot_status as live_status
            spot_status = live_status

        # 보정이 실행 중에 바뀔 수 있으므로 매번 좌표계를 확인한다
        self._sync_layout()

        if nav is None or nav.get("world_pos") is None:
            return self._render_waiting(fps, extra_info)

        car_pos = nav["world_pos"]
        heading = self._view_heading(nav)

        # 1) 노면 레이어를 '위에서 본' 상태로 그린 뒤 원근 변환
        ground = np.full((self.height, self.width, 3), COLOR_NAV_GROUND, dtype=np.uint8)
        self._draw_ground_grid(ground, car_pos, heading)
        self._draw_ground_spots(ground, car_pos, heading, spot_status, nav)
        self._draw_ground_route(ground, car_pos, heading, nav)

        canvas = cv2.warpPerspective(
            ground, self._warp, (self.width, self.height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=COLOR_NAV_SKY)

        # 2) 원근 영향을 받지 않아야 하는 요소는 그 위에 직접 그린다
        self._draw_spot_labels(canvas, car_pos, heading, nav)
        self._draw_car(canvas, self._car_heading(nav, heading))
        self._draw_banner(canvas, nav)
        self._draw_bottom_bar(canvas, nav, fps, extra_info)
        if CONFIG['NAV_SHOW_MINIMAP']:
            self._draw_minimap(canvas, car_pos, nav, spot_status)

        return canvas

    def _render_waiting(self, fps, extra_info):
        """추적 중인 차량이 없을 때의 대기 화면."""
        canvas = np.full((self.height, self.width, 3), COLOR_NAV_GROUND, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.width, self.banner_h), COLOR_NAV_BANNER, -1)
        cv2.putText(canvas, "WAITING FOR VEHICLE", (28, 68),
                    FONT, 1.0, COLOR_TEXT_DIM, 2, cv2.LINE_AA)
        msg = "no tracked vehicle with an assigned spot"
        cv2.putText(canvas, msg, (28, self.height // 2),
                    FONT, 0.6, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        self._draw_bottom_bar(canvas, None, fps, extra_info)
        return canvas

    def _draw_ground_grid(self, layer, car_pos, heading):
        """
        노면 격자. 차량이 움직이는 느낌(속도감)을 준다.

        간격은 주차장 한 칸으로 잡는다. 고정 cm 값을 쓰면 픽셀 모드에서
        간격이 수십 배 촘촘해져 화면이 회색으로 덮인다.
        """
        if not CONFIG['SHOW_GRID']:
            return

        step_x, step_y = self.cell_w, self.cell_h
        # 화면을 덮을 만큼만 그린다. 회전을 감안해 대각선 길이를 쓴다.
        span = math.hypot(self.width, self.height) / self.scale

        # 차량 위치를 격자 간격에 맞춰 내림해 기준점을 잡는다
        bx = math.floor(car_pos[0] / step_x) * step_x
        by = math.floor(car_pos[1] / step_y) * step_y

        for i in range(-int(span / step_x) - 1, int(span / step_x) + 2):
            x = bx + i * step_x
            p1 = self._world_to_flat((x, by - span), car_pos, heading)
            p2 = self._world_to_flat((x, by + span), car_pos, heading)
            cv2.line(layer, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     COLOR_GRID, 1, cv2.LINE_AA)

        for i in range(-int(span / step_y) - 1, int(span / step_y) + 2):
            y = by + i * step_y
            p3 = self._world_to_flat((bx - span, y), car_pos, heading)
            p4 = self._world_to_flat((bx + span, y), car_pos, heading)
            cv2.line(layer, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])),
                     COLOR_GRID, 1, cv2.LINE_AA)

    def _draw_ground_spots(self, layer, car_pos, heading, spot_status, nav):
        """주차 구역을 회전된 사각형으로 그린다."""
        ratio = CONFIG['SPOT_FILL_RATIO']
        hw = self.cell_w * ratio / 2
        target = nav.get("target_spot")

        for spot_id, (sx, sy) in self.spot_world_pos.items():
            # 대형 구역은 한 자리가 세로 2칸이므로 그만큼 길게 그린다
            hh = self.cell_h * get_spot_cell_count(spot_id) * ratio / 2
            corners = [(sx - hw, sy - hh), (sx + hw, sy - hh),
                       (sx + hw, sy + hh), (sx - hw, sy + hh)]
            pts = np.array([self._world_to_flat(c, car_pos, heading) for c in corners],
                           dtype=np.int32)

            if spot_id == target:
                cv2.fillPoly(layer, [pts], (90, 40, 90))
                cv2.polylines(layer, [pts], True, COLOR_SPOT_TARGET, 3, cv2.LINE_AA)
            elif spot_status.get(spot_id) == "full":
                cv2.fillPoly(layer, [pts], COLOR_SPOT_FULL)
                cv2.polylines(layer, [pts], True, (110, 110, 160), 1, cv2.LINE_AA)
            else:
                # 빈자리는 구역 종류(일반/장애인/대형/전기차)에 따라 색을 나눈다
                color = COLOR_SPOT_BY_TYPE.get(spot_type.get(spot_id), COLOR_SPOT_EMPTY)
                cv2.polylines(layer, [pts], True, color, 2, cv2.LINE_AA)

        # 입출구
        if self.gate_world_pos is not None:
            g = self._world_to_flat(self.gate_world_pos, car_pos, heading)
            cv2.circle(layer, (int(g[0]), int(g[1])),
                       max(8, int(self.cell_w * self.scale * 0.4)),
                       COLOR_GATE, 2, cv2.LINE_AA)

    def _route_points(self, nav):
        """
        내 차에서 목적지까지 이어 그릴 점들. 없으면 None.

        경로 계획(C01)이 있으면 남은 경유점을 따라가고, 없으면 목적지까지
        직선으로 잇는다. 픽셀 모드에는 경로 계획이 없어 목적지가 정해져 있는데도
        아무 선도 그리지 않던 것을 메운다.
        """
        route = nav.get("route")
        if route and len(route) >= 2:
            idx = min(nav.get("route_index", 1), len(route) - 1)
            return list(route[idx:])

        target = nav.get("target_world")
        return [target] if target is not None else None

    def _draw_ground_route(self, layer, car_pos, heading, nav):
        """
        주행 경로를 굵은 띠로 그린다.
        내비게이션의 파란 경로선처럼 보이도록 테두리를 덧그린다.
        """
        rest = self._route_points(nav)
        if not rest:
            return

        pts = [self._world_to_flat(car_pos, car_pos, heading)]
        pts += [self._world_to_flat(p, car_pos, heading) for p in rest]
        arr = np.array(pts, dtype=np.int32)

        cv2.polylines(layer, [arr], False, COLOR_NAV_ROUTE_E, 26, cv2.LINE_AA)
        cv2.polylines(layer, [arr], False, COLOR_NAV_ROUTE, 18, cv2.LINE_AA)

        # 목적지 지점 표시
        end = arr[-1]
        cv2.circle(layer, tuple(end), 13, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(layer, tuple(end), 13, COLOR_SPOT_TARGET, 3, cv2.LINE_AA)

    def _draw_spot_labels(self, canvas, car_pos, heading, nav):
        """
        구역 이름은 원근 변환 후 좌표에 똑바로 그린다.
        노면과 함께 변환하면 기울어져서 읽기 어렵다.
        """
        target = nav.get("target_spot")
        for spot_id, pos in self.spot_world_pos.items():
            flat = self._world_to_flat(pos, car_pos, heading)
            # 화면 밖이면 건너뛴다
            if not (-200 < flat[0] < self.width + 200 and -200 < flat[1] < self.height + 200):
                continue
            x, y = self._flat_to_screen(flat)
            if not (0 < x < self.width and self.banner_h < y < self.height):
                continue

            color = COLOR_SPOT_TARGET if spot_id == target else COLOR_TEXT_DIM
            scale = 0.5 if spot_id == target else 0.42
            (tw, _), _ = cv2.getTextSize(spot_id, FONT, scale, 1)
            cv2.putText(canvas, spot_id, (x - tw // 2, y + 4),
                        FONT, scale, color, 1, cv2.LINE_AA)

    def _draw_car(self, canvas, screen_heading_deg=0.0):
        """
        내 차량. 화면 고정 위치에 삼각형으로 그린다.

        방위 고정 화면에서는 지도가 돌지 않으므로 이 아이콘이 진행 방향을
        가리켜야 한다. screen_heading_deg는 화면 위쪽을 0으로 한 각도다.
        (헤딩업 화면에서는 항상 0이라 예전과 똑같이 위를 본다)
        """
        cx = self.width // 2
        cy = self._flat_to_screen((self.width / 2, self.car_y))[1]

        shape = [(0, -20), (-15, 16), (0, 8), (15, 16)]
        rad = math.radians(screen_heading_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        pts = np.array(
            [[int(cx + x * cos_a - y * sin_a), int(cy + x * sin_a + y * cos_a)]
             for x, y in shape], dtype=np.int32)

        cv2.fillPoly(canvas, [pts], COLOR_NAV_CAR, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_banner(self, canvas, nav):
        """
        상단 턴 안내 배너.
        "무엇을, 얼마 앞에서" 해야 하는지가 내비게이션의 핵심 정보다.
        """
        w = self.width
        cv2.rectangle(canvas, (0, 0), (w, self.banner_h), COLOR_NAV_BANNER, -1)
        cv2.line(canvas, (0, self.banner_h), (w, self.banner_h), (70, 70, 74), 1)

        # 다음 동작. maneuver는 경로 계획이 있을 때만 채워진다.
        # 픽셀 모드에는 경로 계획이 없어 maneuver가 늘 UNKNOWN이므로,
        # 그때는 C00이 목표 방향으로 계산해 둔 guide를 쓴다.
        # (이걸 안 보고 있어서 안내가 항상 SEARCHING으로만 떴다)
        maneuver = nav.get("maneuver")
        if not maneuver or maneuver == GUIDE_UNKNOWN:
            maneuver = nav.get("guide") or GUIDE_UNKNOWN

        # 다음 동작까지의 거리도 마찬가지. 없으면 목적지까지 남은 거리를 쓴다.
        dist = nav.get("maneuver_distance_cm")
        if dist is None:
            dist = nav.get("distance_cm")

        # 좌측: 방향 화살표 아이콘
        self._draw_maneuver_icon(canvas, (66, self.banner_h // 2), maneuver)

        # 중앙: 거리 + 지시 문구
        # 도착 단계에서도 남은 거리를 보여준다. 거리 없이 문구만 뜨면
        # 얼마나 더 가야 하는지 알 수 없다.
        tx = 132
        color = COLOR_ARRIVED if maneuver == GUIDE_ARRIVED else COLOR_NAV_ACCENT
        if dist is not None:
            cv2.putText(canvas, f"{dist:.0f}", (tx, 56),
                        FONT, 1.5, color, 3, cv2.LINE_AA)
            (dw, _), _ = cv2.getTextSize(f"{dist:.0f}", FONT, 1.5, 3)
            cv2.putText(canvas, "cm", (tx + dw + 8, 56),
                        FONT, 0.7, color, 2, cv2.LINE_AA)

        cv2.putText(canvas, MANEUVER_LABEL.get(maneuver, maneuver), (tx, 96),
                    FONT, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)

        # 우측: 목적지와 남은 총 거리
        target = nav.get("target_spot") or "-"
        remain = nav.get("distance_cm")
        cv2.putText(canvas, "DESTINATION", (w - 250, 40),
                    FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(target), (w - 250, 78),
                    FONT, 1.1, COLOR_SPOT_TARGET, 2, cv2.LINE_AA)
        if remain is not None:
            cv2.putText(canvas, f"{remain:.0f}cm left", (w - 250, 102),
                        FONT, 0.5, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_maneuver_icon(self, canvas, center, maneuver):
        """방향 지시 화살표를 그린다."""
        cx, cy = center
        c = COLOR_NAV_ACCENT if maneuver != GUIDE_ARRIVED else COLOR_ARRIVED
        t = 7

        if maneuver == GUIDE_LEFT:
            cv2.line(canvas, (cx + 16, cy + 26), (cx + 16, cy - 4), c, t, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx + 16, cy - 4), (cx - 20, cy - 4), c, t,
                            cv2.LINE_AA, tipLength=0.45)
        elif maneuver == GUIDE_RIGHT:
            cv2.line(canvas, (cx - 16, cy + 26), (cx - 16, cy - 4), c, t, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx - 16, cy - 4), (cx + 20, cy - 4), c, t,
                            cv2.LINE_AA, tipLength=0.45)
        elif maneuver == GUIDE_UTURN:
            cv2.ellipse(canvas, (cx, cy), (16, 16), 0, 180, 360, c, t, cv2.LINE_AA)
            cv2.line(canvas, (cx - 16, cy), (cx - 16, cy + 22), c, t, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx + 16, cy), (cx + 16, cy + 24), c, t,
                            cv2.LINE_AA, tipLength=0.5)
        elif maneuver == GUIDE_ARRIVED:
            cv2.circle(canvas, (cx, cy + 6), 17, c, 3, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy + 6), 6, c, -1, cv2.LINE_AA)
        else:
            cv2.arrowedLine(canvas, (cx, cy + 26), (cx, cy - 20), c, t,
                            cv2.LINE_AA, tipLength=0.4)

    def _draw_bottom_bar(self, canvas, nav, fps, extra_info):
        """하단 상태 표시줄."""
        h, w = self.height, self.width
        bar = 34
        cv2.rectangle(canvas, (0, h - bar), (w, h), COLOR_NAV_BANNER, -1)

        parts = []
        if nav is not None:
            car_id = nav.get("car_id") or f"#{nav.get('track_id')}"
            wx, wy = nav["world_pos"]
            parts.append(f"CAR {car_id}")
            # 픽셀 모드의 좌표는 cm가 아니라 이미지 픽셀이다. cm로 적어두면
            # 값이 이상해 보여 좌표계를 의심하게 된다.
            parts.append(f"({wx:.0f},{wy:.0f}){'px' if self.is_pixel_mode else 'cm'}")
        if fps is not None:
            parts.append(f"{fps:.1f} fps")
        if extra_info:
            parts += list(extra_info)

        cv2.putText(canvas, "   |   ".join(parts), (14, h - 11),
                    FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_minimap(self, canvas, car_pos, nav, spot_status):
        """
        좌하단 전체 조감도.
        확대된 시점만 보면 주차장 전체에서 어디쯤인지 알기 어렵다.
        """
        mw, mh = CONFIG['NAV_MINIMAP_W'], CONFIG['NAV_MINIMAP_H']
        x0, y0 = 14, self.height - mh - 46

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + mw, y0 + mh), (20, 20, 22), -1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + mw, y0 + mh), (80, 80, 84), 1)

        # 전체 구역이 들어가도록 축척 계산
        xs = [p[0] for p in self.spot_world_pos.values()] + [car_pos[0]]
        ys = [p[1] for p in self.spot_world_pos.values()] + [car_pos[1]]
        if self.gate_world_pos is not None:
            xs.append(self.gate_world_pos[0])
            ys.append(self.gate_world_pos[1])
        # 여백은 한 칸의 크기에 비례해서 준다 (모드마다 단위가 다르므로)
        min_x, max_x = min(xs) - self.cell_w, max(xs) + self.cell_w
        min_y, max_y = min(ys) - self.cell_h, max(ys) + self.cell_h
        s = min((mw - 16) / max(max_x - min_x, 1e-6), (mh - 16) / max(max_y - min_y, 1e-6))

        def to_mini(p):
            return (int(x0 + 8 + (p[0] - min_x) * s), int(y0 + 8 + (p[1] - min_y) * s))

        target = nav.get("target_spot")
        ratio = CONFIG['SPOT_FILL_RATIO']
        hw = self.cell_w * ratio / 2
        for spot_id, pos in self.spot_world_pos.items():
            hh = self.cell_h * get_spot_cell_count(spot_id) * ratio / 2
            p1 = to_mini((pos[0] - hw, pos[1] - hh))
            p2 = to_mini((pos[0] + hw, pos[1] + hh))
            if spot_id == target:
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_TARGET, -1)
            elif spot_status.get(spot_id) == "full":
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_FULL, -1)
            else:
                color = COLOR_SPOT_BY_TYPE.get(spot_type.get(spot_id), (100, 100, 100))
                cv2.rectangle(canvas, p1, p2, color, 1)

        rest = self._route_points(nav)
        if rest:
            pts = np.array([to_mini(car_pos)] + [to_mini(p) for p in rest],
                           dtype=np.int32)
            cv2.polylines(canvas, [pts], False, COLOR_NAV_ROUTE, 2, cv2.LINE_AA)

        cv2.circle(canvas, to_mini(car_pos), 5, COLOR_NAV_CAR, -1, cv2.LINE_AA)
        cv2.circle(canvas, to_mini(car_pos), 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "OVERVIEW", (x0 + 6, y0 + mh - 6),
                    FONT, 0.36, COLOR_TEXT_DIM, 1, cv2.LINE_AA)


def pick_my_vehicle(nav_results, car_id=None):
    """
    여러 추적 결과 중 내비게이션 화면에 띄울 '내 차'를 고른다.

    Args:
        nav_results: C00_navigation.ParkingNavigator.update()의 반환 결과
        car_id:      특정 차량번호를 지정 (None이면 자동 선택)

    Returns:
        선택된 nav 딕셔너리. 없으면 None.
    """
    if not nav_results:
        return None

    if car_id is not None:
        for n in nav_results:
            if n.get("car_id") == car_id:
                return n
        return None

    # 목표 구역이 배정된 차량을 우선, 그중 목적지가 가장 가까운 차량
    with_target = [n for n in nav_results if n.get("target_spot")]
    if with_target:
        return min(with_target, key=lambda n: n.get("distance_cm") or float('inf'))
    return nav_results[0]


# 내비게이션 맵 UI
class NavigationMapUI:
    """
    차량의 실좌표(cm)와 주차 구역 정보를 위에서 내려다본 형태의
    2D 맵으로 렌더링하는 UI.

    카메라 영상이 아니라 C00_navigation이 계산한 실좌표를 그리므로,
    카메라 각도와 무관하게 항상 정면에서 본 주차장 배치로 표시된다.

    이 클래스는 순수 렌더러다. 검출/추적/위치추정을 하지 않고,
    이미 계산된 결과(nav_results)를 받아 그림만 그린다.
    """

    def __init__(self, navigator=None, width=None, height=None,
                 spot_world_pos=None, gate_world_pos=None):
        """
        NavigationMapUI 초기화.

        Args:
            navigator:      ParkingNavigator 인스턴스 (궤적/목표 조회에 사용, 없어도 동작)
            width:          맵 화면 가로 크기 (px)
            height:         맵 화면 세로 크기 (px)
            spot_world_pos: {구역ID: (x_cm, y_cm)} 매핑.
                            None이면 navigator 또는 C00의 기본 마커 배치에서 가져온다.
            gate_world_pos: 입출구 실좌표 (x_cm, y_cm)
        """
        self.navigator = navigator
        self.width = width or CONFIG['MAP_WIDTH']
        self.height = height or CONFIG['MAP_HEIGHT']

        # 이 조감도는 항상 cm 좌표계로 그린다. 격자(벽/도로/기둥)를
        # cell_to_world로 배치하기 때문이다. 픽셀 모드일 때 들어오는
        # 픽셀 좌표는 _to_world가 cm로 바꿔서 넘긴다.
        self.cell_scale = 1.0
        self._px_to_cm = None       # 픽셀 모드일 때만 채워진다 (MarkerMapper)

        # 주차 구역 실좌표 확보 (우선순위: 인자 > navigator > C00 기본값)
        if spot_world_pos is not None:
            self.spot_world_pos = dict(spot_world_pos)
        elif navigator is not None and navigator.spot_world_pos:
            self.spot_world_pos = dict(navigator.spot_world_pos)
        else:
            self.spot_world_pos = dict(SPOT_WORLD_POS)

        self.gate_world_pos = gate_world_pos if gate_world_pos is not None else GATE1_WORLD_POS

        # 맵 영역과 패널 영역의 가로 크기
        self.panel_w = int(self.width * CONFIG['PANEL_RATIO'])
        self.map_w = self.width - self.panel_w

        # 실좌표 -> 화면좌표 변환 파라미터 계산
        self._compute_transform()

        print(f"[INFO] 내비게이션 맵 UI 초기화 완료. "
              f"({self.width}x{self.height}, 구역 {len(self.spot_world_pos)}개, "
              f"축척 {self.scale:.2f} px/cm)")

    def _compute_transform(self):
        """
        주차장 격자 전체(벽 포함)가 화면에 들어오도록
        실좌표 -> 화면좌표 변환(축척과 원점)을 자동 계산.
        """
        # 격자 바깥 테두리(벽)까지 포함한 범위
        cell_w = (C02_CONFIG['CELL_W_CM'] * self.cell_scale)
        cell_h = (C02_CONFIG['CELL_H_CM'] * self.cell_scale)
        top_left = cell_to_world((0, 0))
        bottom_right = cell_to_world((get_rows() - 1, get_cols() - 1))

        xs = [top_left[0] - cell_w / 2, bottom_right[0] + cell_w / 2]
        ys = [top_left[1] - cell_h / 2, bottom_right[1] + cell_h / 2]

        # 격자 바깥에 있는 구역/입출구가 있어도 잘리지 않도록 포함시킨다
        xs += [p[0] for p in self.spot_world_pos.values()]
        ys += [p[1] for p in self.spot_world_pos.values()]
        if self.gate_world_pos is not None:
            xs.append(self.gate_world_pos[0])
            ys.append(self.gate_world_pos[1])

        pad = CONFIG['PAD_CM']
        self.min_x, self.max_x = min(xs) - pad, max(xs) + pad
        self.min_y, self.max_y = min(ys) - pad, max(ys) + pad

        span_x = max(self.max_x - self.min_x, 1e-6)
        span_y = max(self.max_y - self.min_y, 1e-6)

        margin = CONFIG['MARGIN_PX']
        usable_w = max(self.map_w - 2 * margin, 1)
        usable_h = max(self.height - 2 * margin, 1)

        # 가로/세로 비율을 유지하기 위해 더 빡빡한 쪽에 맞춘다
        self.scale = min(usable_w / span_x, usable_h / span_y)

        # 남는 공간만큼 가운데 정렬
        self.offset_x = margin + (usable_w - span_x * self.scale) / 2
        self.offset_y = margin + (usable_h - span_y * self.scale) / 2

    def _refresh_px_to_cm(self):
        """
        픽셀 모드일 때 쓸 '이미지 픽셀 -> cm' 변환을 준비한다.

        이 조감도의 배경(격자/기둥/도로)은 cm 좌표로 그려진다. 그런데
        픽셀 모드에서는 C00이 차량 위치를 이미지 픽셀로 준다. 그대로 찍으면
        차가 주차장 밖 엉뚱한 곳에 나타난다.

        변환 자체는 MarkerMapper.pixel_to_cm이 갖고 있다. 보정 때 찍은 기둥이
        (픽셀, cm) 대응쌍이라 거기서 만들 수 있다. 여기서는 쓸 수 있는지만 본다.
        """
        mapper = getattr(self.navigator, 'mapper', None)
        self._px_to_cm = mapper if getattr(mapper, 'pillar_pixels', None) else None

    def _to_cm(self, pt):
        """픽셀 모드로 들어온 좌표를 cm로 바꾼다. cm 모드면 그대로 돌려준다."""
        if pt is None or self._px_to_cm is None:
            return pt
        return self._px_to_cm.pixel_to_cm(pt) or pt

    def _to_cm_results(self, nav_results):
        """nav 결과의 좌표들을 이 화면의 좌표계(cm)로 옮긴 사본을 만든다."""
        if self._px_to_cm is None:
            return nav_results
        out = []
        for nav in nav_results:
            n = dict(nav)
            n["world_pos"] = self._to_cm(nav.get("world_pos"))
            n["target_world"] = self._to_cm(nav.get("target_world"))
            if nav.get("route"):
                n["route"] = [self._to_cm(p) for p in nav["route"]]
            out.append(n)
        return out

    def world_to_map(self, world_pt):
        """
        주차장 실좌표(cm)를 맵 화면 좌표(px)로 변환.

        Args:
            world_pt: (x_cm, y_cm)

        Returns:
            (px, py) 정수 좌표
        """
        px = self.offset_x + (world_pt[0] - self.min_x) * self.scale
        py = self.offset_y + (world_pt[1] - self.min_y) * self.scale
        return int(px), int(py)

    def render(self, nav_results, spot_status=None, fps=None, extra_info=None):
        """
        내비게이션 맵 한 장을 그려서 반환.

        Args:
            nav_results: C00_navigation.ParkingNavigator.update()의 반환 결과
            spot_status: {구역ID: "empty"|"full"} 점유 상태.
                         None이면 data.map_data의 실제 상태를 사용한다.
            fps:         화면에 표시할 FPS (없으면 생략)
            extra_info:  패널 하단에 추가로 표시할 문자열 리스트

        Returns:
            렌더링된 BGR 이미지 (numpy array)
        """
        if spot_status is None:
            from data.map_data import spot_status as live_status
            spot_status = live_status

        canvas = np.full((self.height, self.width, 3), COLOR_BG, dtype=np.uint8)

        # 픽셀 모드면 차량 좌표를 cm로 옮겨서 그린다 (배경이 cm 좌표계다)
        self._refresh_px_to_cm()
        nav_results = self._to_cm_results(nav_results)

        # 현재 목표로 지정된 구역들 (강조 표시용)
        target_spots = {n["target_spot"] for n in nav_results if n.get("target_spot")}

        self._draw_grid(canvas)
        self._draw_layout(canvas)
        self._draw_gate(canvas)
        self._draw_spots(canvas, spot_status, target_spots)
        self._draw_guide_lines(canvas, nav_results)
        self._draw_trajectories(canvas, nav_results)
        self._draw_vehicles(canvas, nav_results)
        self._draw_panel(canvas, nav_results, fps, extra_info)

        return canvas

    def _draw_grid(self, canvas):
        """배경 격자와 축척 기준선을 그린다."""
        if not CONFIG['SHOW_GRID']:
            return

        step = CONFIG['GRID_STEP_CM']

        # 세로선 (x = 일정 간격)
        x = math.ceil(self.min_x / step) * step
        while x <= self.max_x:
            px, _ = self.world_to_map((x, self.min_y))
            _, py2 = self.world_to_map((x, self.max_y))
            _, py1 = self.world_to_map((x, self.min_y))
            if 0 <= px < self.map_w:
                cv2.line(canvas, (px, py1), (px, py2), COLOR_GRID, 1)
            x += step

        # 가로선 (y = 일정 간격)
        y = math.ceil(self.min_y / step) * step
        while y <= self.max_y:
            px1, py = self.world_to_map((self.min_x, y))
            px2, _ = self.world_to_map((self.max_x, y))
            px2 = min(px2, self.map_w - 1)
            cv2.line(canvas, (px1, py), (px2, py), COLOR_GRID, 1)
            y += step

        # 축척 안내
        cv2.putText(canvas, f"grid {step:.0f}cm", (10, self.height - 12),
                    FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def _cell_rect(self, cell, fill_ratio=1.0):
        """map_data 격자 한 칸의 화면 사각형 좌표 (p1, p2)를 계산."""
        cx, cy = cell_to_world(cell)
        half_w = (C02_CONFIG['CELL_W_CM'] * self.cell_scale) * fill_ratio * self.scale / 2
        half_h = (C02_CONFIG['CELL_H_CM'] * self.cell_scale) * fill_ratio * self.scale / 2
        px, py = self.world_to_map((cx, cy))
        return ((int(px - half_w), int(py - half_h)),
                (int(px + half_w), int(py + half_h)))

    def _draw_layout(self, canvas):
        """
        주차장 구조(벽 / 도로 / 기둥)를 격자에서 읽어 그린다.

        주차 구역은 점유 상태/종류에 따라 색이 달라지므로 _draw_spots가 따로 그리고,
        여기서는 배경 구조만 그린다. 중앙 섬도 이 함수가 그린 기둥으로 드러난다.
        """
        if not CONFIG['SHOW_LAYOUT']:
            return

        for row in range(get_rows()):
            for col in range(get_cols()):
                cell_type = grid_map[row][col]
                if cell_type in SPOT_CELLS:
                    continue    # _draw_spots가 담당

                p1, p2 = self._cell_rect((row, col))

                if cell_type in (ROAD, GATE1, GATE2):
                    cv2.rectangle(canvas, p1, p2, COLOR_ROAD, -1)
                elif cell_type == PILL:
                    cv2.rectangle(canvas, p1, p2, COLOR_PILL, -1)
                    cv2.rectangle(canvas, p1, p2, COLOR_PILL_EDGE, 1)

                    if CONFIG['SHOW_PILLAR_ID']:
                        marker_id = PILL_MARKER_ID.get((row, col))
                        if marker_id is not None:
                            text = str(marker_id)
                            (tw, th), _ = cv2.getTextSize(text, FONT, 0.4, 1)
                            cx = (p1[0] + p2[0]) // 2
                            cy = (p1[1] + p2[1]) // 2
                            cv2.putText(canvas, text,
                                        (cx - tw // 2, cy + th // 2),
                                        FONT, 0.4, COLOR_PILL_TEXT, 1, cv2.LINE_AA)

    def _draw_gate(self, canvas):
        """입구(GATE1)와 출구(GATE2)를 구분해서 표시."""
        for world_pt, label, color in (
            (GATE1_WORLD_POS, "IN", COLOR_GATE),
            (GATE2_WORLD_POS, "OUT", COLOR_ARRIVED),
        ):
            if world_pt is None:
                continue
            gx, gy = self.world_to_map(world_pt)
            cv2.circle(canvas, (gx, gy), 11, color, 2)
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.45, 2)
            cv2.putText(canvas, label, (gx - tw // 2, gy - 16),
                        FONT, 0.45, color, 2, cv2.LINE_AA)

    def _draw_spots(self, canvas, spot_status, target_spots):
        """주차 구역을 점유 상태에 따라 색을 달리하여 그린다."""
        ratio = CONFIG['SPOT_FILL_RATIO']
        half_w = (C02_CONFIG['CELL_W_CM'] * self.cell_scale) * ratio * self.scale / 2
        half_h = (C02_CONFIG['CELL_H_CM'] * self.cell_scale) * ratio * self.scale / 2

        for spot_id, world_pt in sorted(self.spot_world_pos.items()):
            cx, cy = self.world_to_map(world_pt)
            p1 = (int(cx - half_w), int(cy - half_h))
            p2 = (int(cx + half_w), int(cy + half_h))

            is_full = spot_status.get(spot_id) == "full"
            is_target = spot_id in target_spots

            # 주차중인 구역은 채워서, 빈 구역은 어둡게 깔고 테두리만
            if is_full:
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_FULL, -1)
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_EMPTY, 1)
            else:
                cv2.rectangle(canvas, p1, p2, (30, 34, 30), -1)
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_EMPTY, 1)

            # 목표 구역은 굵은 자홍 테두리로 강조
            if is_target:
                cv2.rectangle(canvas,
                              (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3),
                              COLOR_SPOT_TARGET, 2)

            # 칸이 좁으므로 라벨은 칸 폭에 맞춰 축소한다
            label_color = COLOR_SPOT_TARGET if is_target else COLOR_TEXT
            font_scale = 0.4
            (tw, th), _ = cv2.getTextSize(spot_id, FONT, font_scale, 1)
            while tw > (p2[0] - p1[0]) and font_scale > 0.25:
                font_scale -= 0.05
                (tw, th), _ = cv2.getTextSize(spot_id, FONT, font_scale, 1)
            cv2.putText(canvas, spot_id, (cx - tw // 2, cy + th // 2),
                        FONT, font_scale, label_color, 1, cv2.LINE_AA)

    def _draw_guide_lines(self, canvas, nav_results):
        """
        차량에서 목표 구역까지의 주행 경로를 그린다.

        직선이 아니라 C01_path_planner가 계산한 경유점을 따라 그리므로,
        주차 구역을 가로지르지 않고 통로를 따라가는 실제 경로가 보인다.
        """
        for nav in nav_results:
            if nav.get("target_world") is None:
                continue

            start = self.world_to_map(nav["world_pos"])
            end = self.world_to_map(nav["target_world"])

            # 도착한 차량은 안내선 대신 도착 표시
            if nav.get("guide") == GUIDE_ARRIVED:
                cv2.circle(canvas, end, 16, COLOR_ARRIVED, 2)
                continue

            route = nav.get("route")
            if route and len(route) >= 2:
                # 현재 위치에서 남은 경유점까지만 이어서 그린다
                idx = min(nav.get("route_index", 1), len(route) - 1)
                pts = [start] + [self.world_to_map(p) for p in route[idx:]]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, a, b, COLOR_GUIDE_LINE, 2, cv2.LINE_AA)
                # 경유점 표시 (목적지 제외)
                for p in pts[1:-1]:
                    cv2.circle(canvas, p, 4, COLOR_GUIDE_LINE, -1)
                cv2.arrowedLine(canvas, pts[-2], pts[-1],
                                COLOR_GUIDE_LINE, 2, tipLength=0.25)
                mid = pts[len(pts) // 2]
            else:
                # 경로를 찾지 못한 경우에만 직선으로 대체
                cv2.arrowedLine(canvas, start, end, COLOR_GUIDE_LINE, 2, tipLength=0.06)
                mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)

            # 남은 거리 표시 (경로를 따라간 거리)
            if nav.get("distance_cm") is not None:
                cv2.putText(canvas, f"{nav['distance_cm']:.0f}cm", (mid[0] + 6, mid[1] - 6),
                            FONT, 0.45, COLOR_GUIDE_LINE, 1, cv2.LINE_AA)

    def _draw_trajectories(self, canvas, nav_results):
        """차량이 지나온 경로를 선으로 그린다."""
        if not CONFIG['SHOW_TRAJECTORY'] or self.navigator is None:
            return

        for nav in nav_results:
            key = nav["car_id"] if nav["car_id"] else f"track_{nav['track_id']}"
            history = self.navigator.get_world_trajectory(key)
            if len(history) < 2:
                continue

            points = history[-CONFIG['TRAJECTORY_MAX_POINTS']:]
            pts = np.array([self.world_to_map(self._to_cm(p)) for p in points],
                           dtype=np.int32)
            cv2.polylines(canvas, [pts], False, COLOR_TRAJECTORY, 1, cv2.LINE_AA)

    def _draw_vehicles(self, canvas, nav_results):
        """차량 위치, 진행 방향, 차량번호를 표시."""
        radius = CONFIG['VEHICLE_RADIUS_PX']

        for nav in nav_results:
            cx, cy = self.world_to_map(nav["world_pos"])
            color = COLOR_VEHICLE if nav["car_id"] else COLOR_VEHICLE_UNK

            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1)

            # 진행 방향 화살표 (정지 상태면 생략)
            heading = nav.get("heading_deg")
            if heading is not None:
                rad = math.radians(heading)
                hx = int(cx + math.cos(rad) * radius * 2.4)
                hy = int(cy + math.sin(rad) * radius * 2.4)
                cv2.arrowedLine(canvas, (cx, cy), (hx, hy), (255, 255, 255), 2, tipLength=0.35)

            label = nav["car_id"] if nav["car_id"] else f"#{nav['track_id']}"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 2)
            cv2.putText(canvas, label, (cx - tw // 2, cy - radius - 7),
                        FONT, 0.5, color, 2, cv2.LINE_AA)

    def _draw_panel(self, canvas, nav_results, fps, extra_info):
        """오른쪽에 차량별 안내 정보를 표 형태로 표시."""
        x0 = self.map_w
        cv2.rectangle(canvas, (x0, 0), (self.width, self.height), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (x0, 0), (x0, self.height), (70, 70, 70), 1)

        pad = 14
        y = 32
        cv2.putText(canvas, "NAVIGATION", (x0 + pad, y), FONT, 0.62, COLOR_TEXT, 2, cv2.LINE_AA)

        y += 22
        if fps is not None:
            cv2.putText(canvas, f"FPS {fps:.1f}", (x0 + pad, y),
                        FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y += 14
        cv2.line(canvas, (x0 + pad, y), (self.width - pad, y), (70, 70, 70), 1)
        y += 22

        if not nav_results:
            cv2.putText(canvas, "no vehicle", (x0 + pad, y),
                        FONT, 0.5, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            y += 24
        else:
            for nav in nav_results:
                if y > self.height - 70:
                    cv2.putText(canvas, "...", (x0 + pad, y),
                                FONT, 0.5, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                    break

                car_label = nav["car_id"] if nav["car_id"] else f"#{nav['track_id']}"
                color = COLOR_VEHICLE if nav["car_id"] else COLOR_VEHICLE_UNK

                # 1행: 차량번호 -> 목표 구역
                target = nav["target_spot"] if nav["target_spot"] else "-"
                cv2.putText(canvas, f"{car_label} > {target}", (x0 + pad, y),
                            FONT, 0.52, color, 2, cv2.LINE_AA)
                y += 20

                # 2행: 안내 방향과 남은 거리
                guide = nav.get("guide", GUIDE_UNKNOWN)
                dist = f"{nav['distance_cm']:.0f}cm" if nav.get("distance_cm") is not None else "-"
                guide_color = COLOR_ARRIVED if guide == GUIDE_ARRIVED else COLOR_GUIDE_LINE
                cv2.putText(canvas, f"  {guide}  {dist}", (x0 + pad, y),
                            FONT, 0.48, guide_color, 1, cv2.LINE_AA)
                y += 18

                # 3행: 현재 실좌표
                wx, wy = nav["world_pos"]
                cv2.putText(canvas, f"  ({wx:.0f}, {wy:.0f})cm", (x0 + pad, y),
                            FONT, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                y += 24

        # 추가 정보 (FIFO 대기열, 호모그래피 상태 등)
        if extra_info:
            y = min(y + 6, self.height - 20)
            cv2.line(canvas, (x0 + pad, y), (self.width - pad, y), (70, 70, 70), 1)
            y += 20
            for line in extra_info:
                if y > self.height - 12:
                    break
                cv2.putText(canvas, line, (x0 + pad, y),
                            FONT, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                y += 17

        return canvas
