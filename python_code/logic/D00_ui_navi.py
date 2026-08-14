import cv2
import sys
import os
import math
import numpy as np
from collections import deque

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# 안내 방향 상수는 C00이, 배치 실좌표는 C02가 원본이다.
# C00을 거쳐 좌표를 가져오지 않는다. 재수출은 출처를 흐린다.
from logic.C00_navigation import (
    GUIDE_ARRIVED, GUIDE_UNKNOWN, GUIDE_STRAIGHT, GUIDE_LEFT,
    GUIDE_RIGHT, GUIDE_UTURN,
)
from logic.C02_lot_layout import (
    SPOT_WORLD_POS, GATE1_WORLD_POS, GATE2_WORLD_POS,
    cell_to_world, ONE_WAY_SEGMENTS_WORLD, CONFIG as C02_CONFIG,
)
# 차 위치에서 시작하는 남은 경로. 첫 구간을 가로/세로로 펴는 규칙이
# 화면마다 달라지면 안 되므로 계산은 C01 한 곳에 둔다.
from logic.C01_path_planner import route_from_position


def _lot_map_of(navigator):
    """
    계획기가 쓰는 점유 격자. 없으면 None.

    경로를 펼 때 '옮긴 모서리가 벽을 뚫지 않는지' 확인하는 데만 쓴다.
    단독 테스트처럼 navigator가 없으면 확인 없이 편다.
    """
    return getattr(getattr(navigator, "planner", None), "lot_map", None)
from data.map_data import (
    grid_map, PILL_MARKER_ID, spot_type,
    ROAD, SPOT_CELLS, SPOT1, SPOT2, SPOT3, SPOT4,
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
    "SHOW_PILLAR_ID": True,     # 기둥에 번호 표시 (보정에서 클릭하는 순서)

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
    "NAV_CAR_Y_RATIO": 0.76,    # 화면에서 내 차량이 놓이는 세로 위치 비율
    "NAV_HORIZON_RATIO": 0.34,  # 지평선 높이 비율 (작을수록 멀리까지 보임)
    "NAV_TOP_WIDTH_RATIO": 0.34, # 원근 상단 폭 비율 (작을수록 원근감이 강함)
    "NAV_HORIZON_FADE_PX": 96,  # 지평선 아래로 안개처럼 흐려지는 구간 높이
    "NAV_BANNER_H": 112,        # 상단 턴 안내 배너 높이

    # ---------------------------------------------------------------
    # 화면 회전 규칙
    #
    # 화면이 계속 돌면 어지럽다. 원인은 '멈춰 있을 때도 방향을 계산한다'는
    # 데 있다. 정지한 차의 진행 방향은 검출 상자가 몇 픽셀 떨리는 것으로
    # 정해지므로, 실제로는 서 있는데 방향만 사방으로 튄다. 특히 180도
    # 뒤집히면 화면 위아래가 통째로 바뀐다.
    #
    # 그래서 방향은 '실제로 이동한 거리'가 있을 때만 갱신한다.
    # 멈춰 있으면 마지막 방향 그대로 한 도도 돌지 않는다. 실제 자동차
    # 내비게이션도 정차 중에는 지도를 돌리지 않는다.
    # ---------------------------------------------------------------
    "NAV_HEADING_UP": True,     # 진행 방향이 화면 위쪽 (끄면 방위 고정)
    "NAV_MOVE_WINDOW": 8,       # 이동량을 재는 최근 위치 개수
    "NAV_MOVE_MIN_CM": 2.5,     # 이만큼은 움직여야 '주행 중'으로 본다
    "NAV_STILL_FRAMES": 8,      # 새 위치가 이만큼 안 들어오면 정지로 본다
    "NAV_TURN_DEADBAND_DEG": 10.0,  # 이 안쪽의 방향 변화는 무시 (미세 떨림)
    "NAV_TURN_RATE_DEG": 2.5,   # 한 프레임에 돌릴 수 있는 최대 각도
    "NAV_HEADING_SMOOTH": 0.18, # 목표 각도로 다가가는 비율 (작을수록 부드러움)
    "NAV_FLIP_HOLD": 10,        # 180도 급반전은 이만큼 이어져야 인정한다

    "NAV_LABEL_RANGE_CM": 60.0, # 이 거리 안쪽 구역만 이름을 적는다
    "NAV_SHOW_MINIMAP": True,   # 우하단 전체 조감도 표시
    "NAV_MINIMAP_W": 196,
    "NAV_MINIMAP_H": 160,
    "NAV_SHOW_COMPASS": True,   # 방위 나침반 (화면이 돌기 때문에 필요하다)
}

# 색상 (BGR)
#
# 밝은 화면이다. 바탕을 흰색에 가깝게 깔고, 눈이 먼저 가야 하는 것
# (글자, 기둥, 내 차, 경로)만 검정에 가깝게 눌러서 대비로 읽히게 한다.
# 어두운 화면에서는 밝은 것이 튀지만 여기서는 반대다. 색을 밝게 하면
# 바탕에 묻히므로, 강조는 '더 밝게'가 아니라 '더 진하게'로 준다.
#
# 조명이 밝은 곳에서 프로젝터나 노트북 화면으로 볼 때 어두운 화면은
# 반사가 심해 아무것도 안 보인다. 시연장이 그런 곳이다.
COLOR_BG          = (248, 245, 243)  # 배경
COLOR_PANEL_BG    = (255, 255, 255)  # 패널 배경
COLOR_GRID        = (231, 226, 221)  # 격자
COLOR_TEXT        = (31, 26, 24)     # 기본 텍스트 (거의 검정)
COLOR_TEXT_DIM    = (125, 115, 110)  # 보조 텍스트
COLOR_TEXT_FAINT  = (170, 162, 158)  # 더 옅은 글자 (값이 없을 때)
COLOR_SPOT_EMPTY  = (132, 124, 120)  # 빈자리 테두리
COLOR_SPOT_FULL   = (72, 72, 214)    # 주차중 (붉은 계열)
COLOR_SPOT_TARGET = (170, 0, 198)    # 목표 구역 (자홍)
COLOR_GATE        = (0, 140, 214)    # 입출구 (주황)
# 주차 구역 종류별 빈자리 테두리 색 (BGR).
# 실제 주차장의 노면 표시 관례를 따라 구분한다.
# 흰 바닥 위라서 어두운 화면에서 쓰던 파스텔로는 테두리가 보이지 않는다.
COLOR_SPOT_BY_TYPE = {
    SPOT1: (110, 99, 92),     # 일반   - 회색
    SPOT2: (214, 98, 15),     # 장애인 - 파랑
    SPOT3: (0, 90, 138),      # 대형   - 노랑
    SPOT4: (67, 127, 26),     # 전기차 - 초록
}
COLOR_ROAD        = (240, 235, 232)  # 도로 (통로)
COLOR_PILL        = (70, 62, 58)     # 기둥 - 바닥에서 유일하게 검은 덩어리
COLOR_PILL_EDGE   = (44, 38, 35)     # 기둥 테두리
COLOR_PILL_TEXT   = (242, 236, 232)  # 기둥의 마커 ID (기둥이 어두우므로 흰 글자)
COLOR_VEHICLE     = (60, 150, 20)    # 차량 (번호 매칭됨)
COLOR_VEHICLE_UNK = (0, 120, 230)    # 차량 (번호 미매칭)
COLOR_TRAJECTORY  = (150, 150, 0)    # 이동 궤적
COLOR_GUIDE_LINE  = (220, 110, 20)   # 안내선
COLOR_ARRIVED     = (60, 140, 16)    # 도착 표시
COLOR_ONE_WAY     = (120, 150, 165)  # 일방통행 방향 (바닥 표시라 옅게)

# 일방통행 화살표 간격 (cm). 촘촘하면 배경이 지저분해진다.
ONE_WAY_ARROW_STEP_CM = 25.0

# 차량 시점 화면 색상 (BGR)
#
# 이 화면은 관제 화면(1000x620)과 달리 대시보드 오른쪽 아래 칸에 작게
# 들어간다. 색을 많이 쓰면 축소했을 때 그냥 얼룩으로 보인다. 그래서
# 바탕은 무채색 한 계열로 눌러 두고, 색은 '경로'와 '목적지'에만 쓴다.
#
# 밝기 순서가 뜻을 나른다. 주차장 바깥 < 통로 < 빈자리 순으로 밝아지고,
# 갈 수 없는 곳(주차중인 자리)과 기둥만 어둡다.
COLOR_NAV_SKY     = (245, 241, 238)  # 지평선 위 (원근 바깥 영역)
COLOR_NAV_GROUND  = (236, 231, 228)  # 주차장 바깥 바닥
COLOR_NAV_ROAD    = (221, 213, 208)  # 통로 (아스팔트)
COLOR_NAV_PILL    = (84, 75, 70)     # 기둥 (솟아 있는 장애물 - 어둡게)
COLOR_NAV_BANNER  = (255, 255, 255)  # 상단 배너 배경
COLOR_NAV_CARD    = (253, 252, 252)  # 떠 있는 카드 (조감도/상태)
COLOR_NAV_LINE    = (218, 210, 205)  # 카드 테두리
COLOR_NAV_TILE    = (240, 236, 233)  # 배너의 방향 아이콘 타일 바탕
COLOR_NAV_ROUTE   = (225, 110, 20)   # 주행 경로 (내비 특유의 파랑)
COLOR_NAV_ROUTE_E = (150, 70, 10)    # 경로 테두리 (짙은 파랑)
# 계획된 경로가 아니라 '목적지 방향'만 가리키는 직선. 통로도 일방통행도
# 지키지 않은 선이므로 경로와 같은 색으로 그리면 안 된다. (_route_points)
COLOR_NAV_ROUTE_HINT = (162, 155, 150)
COLOR_NAV_CAR     = (33, 28, 26)     # 내 차량 (검은 화살표)
COLOR_NAV_CAR_E   = (255, 255, 255)  # 내 차량 테두리 (바닥에서 떼어 놓는다)
COLOR_NAV_HALO    = (240, 180, 120)  # 내 차량 주변 후광
COLOR_NAV_ACCENT  = (200, 90, 10)    # 강조 (거리 숫자)
COLOR_NAV_NORTH   = (60, 60, 220)    # 나침반 바늘 (북쪽)

# 노면에 그리는 자리 사각형의 채움색.
# 테두리 색은 위의 COLOR_SPOT_* 를 그대로 쓰고, 채움만 여기서 정한다.
COLOR_NAV_SPOT_FILL      = (252, 250, 250)  # 빈자리 (통로보다 밝게)
COLOR_NAV_SPOT_FULL      = (196, 188, 186)  # 주차중 (통로보다 어둡게)
COLOR_NAV_SPOT_FULL_EDGE = (160, 145, 150)
COLOR_NAV_SPOT_TARGET    = (246, 228, 250)  # 목적지 (자홍 테두리 안쪽)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# 방위 고정 화면에서 쓰는 시선 방향.
# _world_to_flat의 각도 규약(x축 오른쪽 0도, y축 아래로 +90도)에서 -90도가
# '월드의 위쪽(-y)이 화면 위쪽'이다. 즉 3번 주차장 상태 화면과 같은 방향.
FIXED_VIEW_HEADING = -90.0


# 그리기 도우미
# OpenCV에는 둥근 모서리도, 반투명 채우기도 없다. 실제 내비게이션 UI는
# 거의 전부 '둥근 카드'라서 이 둘이 없으면 어떻게 그려도 각진 관제 화면이
# 된다. 아래 세 함수가 그 셋을 메운다.
def _dashed_polyline(img, pts, color, thickness, dash=16, gap=12):
    """
    점선 꺾은선. OpenCV에는 점선이 없어서 구간을 잘라 그린다.

    '계획된 경로가 아니다'를 선 모양만으로 알리려고 만들었다. 색만 바꾸면
    작은 화면에서 구분이 안 된다.
    """
    period = max(dash + gap, 1)
    for a, b in zip(pts[:-1], pts[1:]):
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            continue
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        t = 0.0
        while t < length:
            e = min(t + dash, length)
            cv2.line(img,
                     (int(x1 + ux * t), int(y1 + uy * t)),
                     (int(x1 + ux * e), int(y1 + uy * e)),
                     color, thickness, cv2.LINE_AA)
            t += period


def _rounded_rect(img, p1, p2, radius, color, thickness=-1):
    """모서리가 둥근 사각형."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    # 네 모서리의 (중심, 시작각). OpenCV의 각도는 x축 오른쪽이 0도, 시계 방향.
    corners = ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
               (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90))

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy, start in corners:
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90,
                        color, -1, cv2.LINE_AA)
        return

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    for cx, cy, start in corners:
        cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90,
                    color, thickness, cv2.LINE_AA)


def _card(canvas, p1, p2, color, alpha=0.9, radius=14, border=None):
    """반투명 둥근 카드. 아래 그림이 살짝 비쳐 화면에 '떠 있게' 보인다."""
    h, w = canvas.shape[:2]
    x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
    x2, y2 = min(w, int(p2[0])), min(h, int(p2[1]))
    if x2 <= x1 or y2 <= y1:
        return

    # 전체 캔버스를 복사하면 매 프레임 낭비다. 카드 영역만 합성한다.
    roi = canvas[y1:y2, x1:x2]
    layer = roi.copy()
    _rounded_rect(layer, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), radius, color, -1)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)
    if border is not None:
        _rounded_rect(canvas, (x1, y1), (x2 - 1, y2 - 1), radius, border, 1)


def _ascii_label(s):
    """
    OpenCV가 그릴 수 있는 글자만 남긴다.

    Hershey 폰트에는 한글이 없다. '12가3456'을 그대로 넘기면 물음표가 섞여
    '12???3456'이 되어 번호를 잘못 읽게 된다. 못 그리는 구간은 가운뎃점
    하나로 접어 '12·3456'처럼 자리만 표시한다. (전체 번호는 최종 화면이
    영상 아래 줄에 HTML로 제대로 적는다)
    """
    out = []
    for ch in str(s):
        if 32 <= ord(ch) < 127:
            out.append(ch)
        elif not out or out[-1] != '.':
            out.append('.')
    return "".join(out)


def _text(img, s, org, scale, color, thickness=1, anchor="l"):
    """정렬을 지정할 수 있는 putText. anchor: l(왼쪽) c(가운데) r(오른쪽)."""
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    if anchor == "c":
        x -= tw // 2
    elif anchor == "r":
        x -= tw
    cv2.putText(img, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return tw, th


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
    GUIDE_UTURN:    "U-TURN",
    GUIDE_ARRIVED:  "ARRIVING",
    GUIDE_UNKNOWN:  "SEARCHING",
}


# 차량 시점 내비게이션 화면
class NavigationView:
    """
    운전자 시점 안내 화면. 자동차 내비게이션과 같은 방식으로 그린다.

    NavigationMapUI가 주차장 전체를 천장에서 내려다보는 관제 화면이라면,
    이쪽은 '내 차' 한 대만 따라간다. 셋이 다르다.

      1. 헤딩업: 진행 방향이 늘 화면 위쪽이고 내 차는 아래쪽에 고정된다.
         단, 화면을 돌리는 것은 차가 실제로 움직였을 때뿐이다.
         (아래 '화면이 돌아가는 규칙' 참고)
      2. 원근 시점: 노면 레이어에 사다리꼴 변환을 걸어 먼 곳이 지평선으로
         수렴하게 만든다.
      3. 턴 안내 배너: 다음에 무엇을 얼마 앞에서 해야 하는지를 크게 띄운다.

    화면이 돌아가는 규칙
      진행 방향은 위치 이력에서 나온다. 그런데 멈춰 있는 차의 위치도 검출
      상자가 흔들리는 만큼 매 프레임 조금씩 바뀐다. 그 미세한 흔들림으로
      방향을 계산하면 서 있는 차의 방향이 사방으로 튀고, 180도 뒤집히는
      순간 화면 위아래가 통째로 바뀐다. 눈이 아픈 원인이 이것이다.

      그래서 방향을 갱신하는 조건을 네 겹으로 걸었다.
        - 최근 NAV_MOVE_WINDOW개 위치의 총 이동량이 NAV_MOVE_MIN_CM 이상
        - 새 위치가 계속 들어오는 중 (NAV_STILL_FRAMES 안에)
        - 방향 변화가 NAV_TURN_DEADBAND_DEG를 넘음
        - 180도 급반전은 NAV_FLIP_HOLD 프레임 이어질 때만 인정
      전부 통과해도 한 프레임에 NAV_TURN_RATE_DEG까지만 돈다.
      멈춰 있으면 한 도도 돌지 않는다. 실제 내비게이션도 정차 중에는
      지도를 돌리지 않는다.

    왜 cm 좌표계로 그리는가
      픽셀 모드(기둥을 직접 찍어 보정한 경우)의 좌표는 이미지 픽셀이다.
      카메라가 비스듬히 보고 있어서 가로 1cm와 세로 1cm의 픽셀 수가 달라,
      픽셀 그대로 그리면 주차장이 납작하게 눌린다. 기둥 대응쌍으로 만든
      pixel_to_cm이 있으면 cm로 옮겨서 그린다. 그러면 두 모드가 같은
      그림이 되고, 통로와 기둥도 설계 배치대로 깔 수 있다.
      (변환을 만들 수 없을 때만 예전처럼 픽셀 좌표로 그린다)

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

        # 화면 회전 상태
        self._display_heading = None    # 지금 지도가 서 있는 방향
        self._car_deg = None            # 마지막으로 확인된 실제 진행 방향
        self._path = deque(maxlen=CONFIG['NAV_MOVE_WINDOW'])
        self._track_key = None          # 지금 따라가는 차 (바뀌면 이력을 버린다)
        self._still = 0                 # 새 위치 없이 지나간 프레임 수
        self._flip = 0                  # 180도 반전이 이어진 프레임 수

        self._layout = None
        self._px_to_cm = None
        self._move_min = CONFIG['NAV_MOVE_MIN_CM']
        self._sync_layout()
        self._build_perspective()

    def _sync_layout(self):
        """
        navigator의 보정 상태를 읽어 좌표계를 맞춘다. (매 프레임 호출)

        생성 시점에 한 번만 읽으면 안 된다. 보정(/calibrate)은 프로그램이
        뜬 뒤에 하므로, 그때는 자리 좌표가 아직 비어 있고 픽셀 모드인지도
        알 수 없다. 그 상태로 굳으면 차 위치는 픽셀(예: 332,260), 배치는
        cm(0~120) 좌표가 되어 주차장이 화면 밖 저 멀리 그려진다.
        """
        layout = resolve_layout(self.navigator)
        mapper = getattr(self.navigator, 'mapper', None)

        # 픽셀 모드라도 픽셀->cm 변환을 만들 수 있으면 cm로 옮겨 그린다.
        # (클래스 주석의 '왜 cm 좌표계로 그리는가' 참고)
        px_to_cm = None
        if layout['pixel_mode'] and mapper is not None \
                and mapper.pixel_to_cm((0.0, 0.0)) is not None:
            px_to_cm = mapper
            layout = {
                "pixel_mode": False,
                "spot_pos": dict(getattr(self.navigator, 'spot_world_pos', None)
                                 or SPOT_WORLD_POS),
                "gate_pos": GATE1_WORLD_POS,
                "cell_w": C02_CONFIG['CELL_W_CM'],
                "cell_h": C02_CONFIG['CELL_H_CM'],
            }

        if self._spot_override is not None:
            layout['spot_pos'] = dict(self._spot_override)
        if self._gate_override is not None:
            layout['gate_pos'] = self._gate_override

        self._px_to_cm = px_to_cm   # 보정 도중에도 바뀔 수 있으므로 매번 갱신
        if layout == self._layout:
            return

        self._layout = layout
        self.is_pixel_mode = layout['pixel_mode']
        self.spot_world_pos = layout['spot_pos']
        self.gate_world_pos = layout['gate_pos']
        self.cell_w = layout['cell_w']
        self.cell_h = layout['cell_h']

        # 통로/기둥 배경은 cell_to_world가 주는 cm 좌표로 깐다.
        # 픽셀 좌표계로 떨어졌을 때는 격자를 놓을 자리를 알 수 없어 생략한다.
        self.draw_lot = not self.is_pixel_mode

        # 확대 배율. 두 좌표계에서 '화면에 보이는 한 칸의 크기'가 같아지도록
        # 맞춘다. cm 모드면 NAV_PX_PER_CM 그대로다.
        self.scale = (CONFIG['NAV_PX_PER_CM'] * C02_CONFIG['CELL_W_CM']) / self.cell_w
        # '움직였다'고 볼 최소 이동량. 픽셀 좌표계에는 cm가 없으므로
        # 칸 크기에 견주어 같은 비율로 환산한다.
        self._move_min = (CONFIG['NAV_MOVE_MIN_CM']
                          * self.cell_w / C02_CONFIG['CELL_W_CM'])
        self._path.clear()

        unit = "px" if self.is_pixel_mode else "cm"
        print(f"[INFO] 차량 시점 내비게이션 화면 좌표계 설정. "
              f"({self.width}x{self.height}, "
              f"{'픽셀' if self.is_pixel_mode else 'cm'} 좌표계"
              f"{' (픽셀->cm 변환)' if px_to_cm is not None else ''}, "
              f"자리 {len(self.spot_world_pos)}개, "
              f"한 칸 {self.cell_w:.0f}x{self.cell_h:.0f}{unit}, "
              f"배율 {self.scale:.2f})")

    def _build_perspective(self):
        """
        노면 레이어(위에서 본 그림)를 원근 시점으로 바꾸는 변환을 준비.

        위쪽 변을 좁히고 아래로 내려서, 먼 곳이 지평선으로 수렴하는
        자동차 내비게이션 특유의 시점을 만든다.
        """
        w, h = self.width, self.height
        top_w = w * CONFIG['NAV_TOP_WIDTH_RATIO']
        self.horizon_y = int(h * CONFIG['NAV_HORIZON_RATIO'])

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [w / 2 - top_w / 2, self.horizon_y],
            [w / 2 + top_w / 2, self.horizon_y],
            [w, h],
            [0, h],
        ])
        self._warp = cv2.getPerspectiveTransform(src, dst)

        # 지평선 부근을 하늘색으로 녹이는 그라데이션.
        # 원근 변환만 걸면 먼 곳이 칼로 자른 듯 끊겨 종이 모형처럼 보인다.
        fade = max(1, CONFIG['NAV_HORIZON_FADE_PX'])
        self._fade_h = min(fade, h - self.horizon_y)
        self._fade_a = np.linspace(0.94, 0.0, self._fade_h,
                                   dtype=np.float32)[:, None, None]
        self._fade_sky = np.full((self._fade_h, w, 3), COLOR_NAV_SKY, dtype=np.float32)

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

    def _to_cm_nav(self, nav):
        """픽셀 좌표로 들어온 안내 결과를 cm 좌표계 사본으로 옮긴다."""
        if nav is None or self._px_to_cm is None:
            return nav

        conv = self._px_to_cm.pixel_to_cm
        out = dict(nav)
        for key in ("world_pos", "target_world", "next_waypoint"):
            if nav.get(key) is not None:
                out[key] = conv(nav[key]) or nav[key]
        if nav.get("route"):
            out["route"] = [conv(p) or p for p in nav["route"]]
        return out

    # ---------- 화면 방향 ----------
    def _track_motion(self, nav):
        """
        실제로 움직였을 때만 진행 방향(도)을 돌려준다. 서 있으면 None.

        MJPEG 피드는 파이프라인보다 빠르게 프레임을 뽑으므로 같은 위치가
        여러 번 들어온다. 그대로 쌓으면 창이 같은 점으로만 채워져 항상
        '정지'가 된다. 새 위치일 때만 쌓고, 새 위치가 한동안 없으면
        그때 정지로 판정한다.
        """
        key = nav.get("car_id") or nav.get("track_id")
        if key != self._track_key:
            # 다른 차로 바뀌었는데 이전 차의 궤적으로 방향을 잡으면
            # 화면이 엉뚱한 쪽으로 홱 돈다. 이력만 버린다.
            # 화면 각도는 그대로 두고, 새 차가 움직이기 시작하면 평소처럼
            # 초당 몇 도씩 돌아서 따라간다. 여기서 각도를 초기화하면
            # 차가 바뀌는 순간 화면이 통째로 튄다.
            self._track_key = key
            self._path.clear()
            self._still = 0
            self._flip = 0

        pos = (float(nav["world_pos"][0]), float(nav["world_pos"][1]))
        if self._path and pos == self._path[-1]:
            self._still += 1
        else:
            self._path.append(pos)
            self._still = 0

        if self._still > CONFIG['NAV_STILL_FRAMES'] or len(self._path) < 2:
            return None

        start, end = self._path[0], self._path[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        if math.hypot(dx, dy) < self._move_min:
            return None

        self._car_deg = math.degrees(math.atan2(dy, dx))
        return self._car_deg

    def _view_heading(self, moving_deg):
        """
        지도를 어느 방향으로 세울지 결정. 정지 중에는 직전 값을 그대로 쓴다.
        """
        if self._display_heading is None:
            # 아직 한 번도 안 움직인 차를 임의 방향으로 돌려놓으면 어디가
            # 어디인지 알 수 없다. 3번 주차장 상태 화면과 같은 방향에서 시작한다.
            self._display_heading = (FIXED_VIEW_HEADING if moving_deg is None
                                     else moving_deg)
            return self._display_heading

        if not CONFIG['NAV_HEADING_UP'] or moving_deg is None:
            return self._display_heading

        diff = (moving_deg - self._display_heading + 180) % 360 - 180

        # 180도 급반전. 후진이나 유턴이면 계속 이어지고, 검출 떨림이면
        # 다음 프레임에 사라진다. 이어질 때만 받아들인다.
        if abs(diff) > 135:
            self._flip += 1
            if self._flip < CONFIG['NAV_FLIP_HOLD']:
                return self._display_heading
        else:
            self._flip = 0

        if abs(diff) < CONFIG['NAV_TURN_DEADBAND_DEG']:
            return self._display_heading

        rate = CONFIG['NAV_TURN_RATE_DEG']
        step = max(-rate, min(rate, diff * CONFIG['NAV_HEADING_SMOOTH']))
        self._display_heading = (self._display_heading + step + 180) % 360 - 180
        return self._display_heading

    def _car_screen_deg(self, view_heading):
        """
        내 차 아이콘이 화면에서 가리킬 각도 (화면 위쪽이 0).

        헤딩업에서는 지도가 도니까 아이콘은 늘 위를 본다. 실제 내비게이션도
        화살표를 고정해 두고 지도를 돌린다. 방위 고정으로 쓸 때만 아이콘이 돈다.
        """
        if CONFIG['NAV_HEADING_UP']:
            return 0.0
        if self._car_deg is None:
            return 0.0
        return (self._car_deg - view_heading + 180) % 360 - 180

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
        nav = self._to_cm_nav(nav)

        if nav is None or nav.get("world_pos") is None:
            return self._render_waiting(fps, extra_info)

        car_pos = nav["world_pos"]
        heading = self._view_heading(self._track_motion(nav))
        target = nav.get("target_spot")

        # 1) 노면 레이어를 '위에서 본' 상태로 그린 뒤 원근 변환
        ground = np.full((self.height, self.width, 3), COLOR_NAV_GROUND, dtype=np.uint8)
        self._draw_ground_lot(ground, car_pos, heading)
        self._draw_ground_spots(ground, car_pos, heading, spot_status, target)
        self._draw_ground_route(ground, car_pos, heading, nav)

        canvas = cv2.warpPerspective(
            ground, self._warp, (self.width, self.height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=COLOR_NAV_SKY)
        self._fade_horizon(canvas)

        # 2) 원근에 눕히면 안 되는 것들은 변환 후에 똑바로 올린다
        self._draw_spot_labels(canvas, car_pos, heading, target)
        self._draw_destination_pin(canvas, car_pos, heading, nav)
        self._draw_car(canvas, self._car_screen_deg(heading))
        self._draw_banner(canvas, nav)
        if CONFIG['NAV_SHOW_COMPASS']:
            self._draw_compass(canvas, heading)
        if CONFIG['NAV_SHOW_MINIMAP']:
            self._draw_minimap(canvas, car_pos, nav, spot_status)
        self._draw_status(canvas, nav, fps, extra_info)

        return canvas

    def _render_waiting(self, fps, extra_info):
        """추적 중인 차량이 없을 때의 대기 화면."""
        canvas = np.full((self.height, self.width, 3), COLOR_NAV_SKY, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.width, self.banner_h),
                      COLOR_NAV_BANNER, -1)
        cv2.line(canvas, (0, self.banner_h), (self.width, self.banner_h),
                 COLOR_NAV_LINE, 1)

        # 안내 중일 때와 같은 자리에 같은 크기로 둔다. 대기 화면만 배치가
        # 다르면 차가 잡히는 순간 화면이 통째로 갈아 끼워진 것처럼 보인다.
        tile = self.banner_h - 32
        _rounded_rect(canvas, (16, 16), (16 + tile, 16 + tile), 14, COLOR_NAV_TILE, -1)
        self._draw_maneuver_icon(canvas, (16 + tile // 2, 16 + tile // 2),
                                 GUIDE_UNKNOWN, COLOR_NAV_ACCENT, tile)
        _text(canvas, "SEARCHING", (16 + tile + 22, 74), 0.82, COLOR_TEXT_DIM, 2)
        _text(canvas, "no vehicle to guide", (self.width // 2, self.height // 2),
              0.7, COLOR_TEXT_DIM, 1, anchor="c")
        self._draw_status(canvas, None, fps, extra_info)
        return canvas

    def _fade_horizon(self, canvas):
        """지평선 아래를 하늘색으로 녹여 먼 곳이 흐려 보이게 한다."""
        y0, fh = self.horizon_y, self._fade_h
        if fh <= 0 or y0 >= self.height:
            return
        band = canvas[y0:y0 + fh].astype(np.float32)
        canvas[y0:y0 + fh] = (band * (1.0 - self._fade_a)
                              + self._fade_sky * self._fade_a).astype(np.uint8)

    # ---------- 노면 (원근 변환 대상) ----------
    def _quad(self, cx, cy, hw, hh, car_pos, heading):
        """(cx, cy)를 중심으로 한 실좌표 사각형의 노면 레이어 폴리곤."""
        corners = ((cx - hw, cy - hh), (cx + hw, cy - hh),
                   (cx + hw, cy + hh), (cx - hw, cy + hh))
        return np.array([self._world_to_flat(p, car_pos, heading) for p in corners],
                        dtype=np.int32)

    def _draw_ground_lot(self, layer, car_pos, heading):
        """
        통로와 기둥.

        예전에는 여기에 격자선을 촘촘히 그었다. 속도감을 주려던 것인데
        축소해서 보면 선만 잔뜩 보여 화면이 복잡해진다. 대신 주차장 바닥
        전체를 아스팔트 색 한 장으로 깔고 기둥만 얹는다. 실제 내비게이션도
        격자가 아니라 도로 모양을 그린다.
        """
        if not self.draw_lot:
            return

        hw, hh = self.cell_w / 2.0, self.cell_h / 2.0
        tl = cell_to_world((0, 0))
        br = cell_to_world((get_rows() - 1, get_cols() - 1))
        cv2.fillPoly(layer, [self._quad((tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2,
                                        (br[0] - tl[0]) / 2 + hw,
                                        (br[1] - tl[1]) / 2 + hh,
                                        car_pos, heading)], COLOR_NAV_ROAD)

        for row in range(get_rows()):
            for col in range(get_cols()):
                if grid_map[row][col] != PILL:
                    continue
                cx, cy = cell_to_world((row, col))
                cv2.fillPoly(layer, [self._quad(cx, cy, hw, hh, car_pos, heading)],
                             COLOR_NAV_PILL)

    def _draw_ground_spots(self, layer, car_pos, heading, spot_status, target):
        """주차 구역. 빈자리/주차중/목적지 셋이 한눈에 갈리게 그린다."""
        ratio = CONFIG['SPOT_FILL_RATIO']
        hw = self.cell_w * ratio / 2

        for spot_id, (sx, sy) in self.spot_world_pos.items():
            # 대형 구역은 한 자리가 세로 2칸이므로 그만큼 길게 그린다
            hh = self.cell_h * get_spot_cell_count(spot_id) * ratio / 2
            pts = self._quad(sx, sy, hw, hh, car_pos, heading)

            if spot_id == target:
                cv2.fillPoly(layer, [pts], COLOR_NAV_SPOT_TARGET)
                cv2.polylines(layer, [pts], True, COLOR_SPOT_TARGET, 3, cv2.LINE_AA)
            elif spot_status.get(spot_id) == "full":
                # 이미 차 있는 자리. 갈 수 없는 곳이니 통로보다 어둡게 눌러
                # 둔다. 빈자리와의 차이가 밝기로 먼저 보여야 한눈에 갈린다.
                cv2.fillPoly(layer, [pts], COLOR_NAV_SPOT_FULL)
                cv2.polylines(layer, [pts], True, COLOR_NAV_SPOT_FULL_EDGE, 1, cv2.LINE_AA)
            else:
                # 빈자리는 구역 종류(일반/장애인/대형/전기차)에 따라 색을 나눈다
                color = COLOR_SPOT_BY_TYPE.get(spot_type.get(spot_id), COLOR_SPOT_EMPTY)
                cv2.fillPoly(layer, [pts], COLOR_NAV_SPOT_FILL)
                cv2.polylines(layer, [pts], True, color, 2, cv2.LINE_AA)

    def _route_points(self, nav, car_pos):
        """
        내 차에서 목적지까지 이어 그릴 점들과 그것이 '계획된 경로'인지 여부.
        (차의 현재 위치가 첫 점이다)

        경로 계획(C01)이 있으면 남은 경유점을 따라가고, 없으면 목적지까지
        직선으로 잇는다. 픽셀 모드에는 경로 계획이 없어 목적지가 정해져 있는데도
        아무 선도 그리지 않던 것을 메운다.

        둘을 구분해서 돌려주는 이유: 직선은 통로도 일방통행도 전혀 고려하지
        않은 '방향 표시'일 뿐인데, 계획된 경로와 똑같은 파란 띠로 그리면
        화면만 보고는 구별할 수 없다. 주차 구역을 관통하고 역주행으로 보이는
        선이 계획기가 낸 것으로 오해된다. 그래서 호출부가 다르게 그릴 수 있게
        사실을 함께 넘긴다.

        Returns:
            (점 목록, planned). 그릴 것이 없으면 (None, False).
        """
        route = nav.get("route")
        if route and len(route) >= 2:
            # 첫 구간이 비스듬해지지 않도록 C01이 모서리를 차에 맞춰 준다.
            return route_from_position(
                route, nav.get("route_index", 1), car_pos,
                lot_map=_lot_map_of(self.navigator)), True

        target = nav.get("target_world")
        return ([car_pos, target], False) if target is not None else (None, False)

    def _draw_ground_route(self, layer, car_pos, heading, nav):
        """
        주행 경로를 굵은 띠로 그린다.

        polylines는 꺾이는 곳이 뾰족하게 잘린다. 이음매마다 원을 찍어
        둥글게 만들면 내비게이션의 그 파란 경로선이 된다.

        계획된 경로가 아닐 때(목적지까지의 직선)는 얇은 회색 점선으로 그린다.
        같은 파란 띠로 그리면 통로도 일방통행도 안 지킨 선이 계획 결과처럼
        보인다. (_route_points 주석 참고)
        """
        rest, planned = self._route_points(nav, car_pos)
        if not rest:
            return

        arr = np.array([self._world_to_flat(p, car_pos, heading) for p in rest],
                       dtype=np.int32)

        if not planned:
            _dashed_polyline(layer, arr, COLOR_NAV_ROUTE_HINT, 4, dash=18, gap=14)
            return

        for color, w in ((COLOR_NAV_ROUTE_E, 30), (COLOR_NAV_ROUTE, 22)):
            cv2.polylines(layer, [arr], False, color, w, cv2.LINE_AA)
            for p in arr:
                cv2.circle(layer, (int(p[0]), int(p[1])), w // 2, color, -1, cv2.LINE_AA)

    # ---------- 원근 변환 뒤 (똑바로 서는 것들) ----------
    def _screen_of(self, world_pt, car_pos, heading, margin=400, top=None):
        """
        실좌표의 최종 화면 위치. 화면 밖이면 None.

        top을 주면 그보다 위(먼 곳)는 버린다. 지평선 부근은 안개로 흐려 놓은
        구간이라 거기에 글자를 얹으면 읽을 수도 없으면서 화면만 지저분해진다.
        """
        flat = self._world_to_flat(world_pt, car_pos, heading)
        if not (-margin < flat[0] < self.width + margin
                and -margin < flat[1] < self.height + margin):
            return None
        x, y = self._flat_to_screen(flat)
        if top is None:
            top = self.banner_h
        if not (0 < x < self.width and top < y < self.height):
            return None
        return x, y

    def _draw_spot_labels(self, canvas, car_pos, heading, target):
        """
        구역 이름. 원근 변환 후 좌표에 똑바로 그린다.
        노면과 함께 변환하면 기울어져서 읽기 어렵다.

        전부 적으면 먼 곳까지 글씨로 뒤덮인다. 지금 지나가는 근처만 적는다.
        """
        limit = CONFIG['NAV_LABEL_RANGE_CM'] * self.cell_w / C02_CONFIG['CELL_W_CM']
        top = self.horizon_y + self._fade_h // 2

        for spot_id, pos in self.spot_world_pos.items():
            if spot_id == target:
                continue    # 목적지는 핀이 대신한다
            if math.dist(pos, car_pos) > limit:
                continue
            xy = self._screen_of(pos, car_pos, heading, top=top)
            if xy is None:
                continue
            _text(canvas, spot_id, (xy[0], xy[1] + 4), 0.46, COLOR_TEXT_DIM,
                  1, anchor="c")

    def _draw_destination_pin(self, canvas, car_pos, heading, nav):
        """
        목적지 핀. 지도 위에 눕히지 않고 세워서 꽂는다.
        실제 내비게이션의 도착 지점 표시와 같은 모양이다.
        """
        target_world = nav.get("target_world")
        if target_world is None:
            return
        xy = self._screen_of(target_world, car_pos, heading)
        if xy is None:
            return

        x, y = xy
        tail = np.array([[x - 9, y - 20], [x + 9, y - 20], [x, y]], dtype=np.int32)
        cv2.fillPoly(canvas, [tail], COLOR_SPOT_TARGET, cv2.LINE_AA)
        cv2.circle(canvas, (x, y - 28), 14, COLOR_SPOT_TARGET, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y - 28), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (x, y - 28), 5, (255, 255, 255), -1, cv2.LINE_AA)

        spot = nav.get("target_spot")
        if spot:
            _text(canvas, str(spot), (x, y - 46), 0.5, COLOR_TEXT, 2, anchor="c")

    def _draw_car(self, canvas, screen_heading_deg=0.0):
        """
        내 차량. 화면 고정 위치에 화살촉으로 그린다.

        헤딩업에서는 screen_heading_deg가 늘 0이라 화살촉이 위를 본다.
        방위 고정으로 쓸 때만 실제 진행 방향으로 돈다.
        """
        cx = self.width // 2
        cy = self._flat_to_screen((self.width / 2, self.car_y))[1]

        # 후광. 내 위치를 부드럽게 드러낸다.
        r = 40
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(self.width, cx + r), min(self.height, cy + r)
        roi = canvas[y1:y2, x1:x2]
        glow = roi.copy()
        cv2.circle(glow, (cx - x1, cy - y1), r, COLOR_NAV_HALO, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.25, roi, 0.75, 0, roi)

        shape = ((0, -23), (17, 18), (0, 8), (-17, 18))
        rad = math.radians(screen_heading_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        pts = np.array(
            [[int(cx + x * cos_a - y * sin_a), int(cy + x * sin_a + y * cos_a)]
             for x, y in shape], dtype=np.int32)

        cv2.fillPoly(canvas, [pts], COLOR_NAV_CAR, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, COLOR_NAV_CAR_E, 2, cv2.LINE_AA)

    # ---------- 상단 배너 ----------
    def _draw_banner(self, canvas, nav):
        """
        상단 턴 안내 배너.

        내비게이션이 알려 줘야 하는 건 결국 '무엇을, 얼마 앞에서'다.
        왼쪽에 큰 화살표, 가운데에 거리와 지시, 오른쪽에 목적지를 놓는다.
        이 화면은 대시보드에서 작게 보이므로 글자를 큼직하게 잡았다.
        """
        w, bh = self.width, self.banner_h
        cv2.rectangle(canvas, (0, 0), (w, bh), COLOR_NAV_BANNER, -1)
        cv2.line(canvas, (0, bh), (w, bh), COLOR_NAV_LINE, 1)

        # 다음 동작. maneuver는 경로 계획이 있을 때만 채워진다.
        # 픽셀 모드에는 경로 계획이 없어 maneuver가 늘 UNKNOWN이므로,
        # 그때는 C00이 목표 방향으로 계산해 둔 guide를 쓴다.
        maneuver = nav.get("maneuver")
        if not maneuver or maneuver == GUIDE_UNKNOWN:
            maneuver = nav.get("guide") or GUIDE_UNKNOWN

        # 다음 동작까지의 거리도 마찬가지. 없으면 목적지까지 남은 거리를 쓴다.
        dist = nav.get("maneuver_distance_cm")
        remain = nav.get("distance_cm")
        if dist is None:
            dist = remain

        # 이미 주차를 마친 차를 비추는 중이면 '안내'가 아니라 '완료'다.
        # 안내 대상이 아니라 목표가 없을 뿐인데, 그대로 두면 SEARCHING /
        # NOT ASSIGNED로 떠서 안내가 고장난 것처럼 보인다.
        parked_spot = nav.get("parked_spot")
        if parked_spot:
            maneuver = GUIDE_ARRIVED
            dist = remain = None

        arrived = maneuver == GUIDE_ARRIVED
        accent = COLOR_ARRIVED if arrived else COLOR_NAV_ACCENT

        # 왼쪽 : 방향 아이콘을 둥근 타일에 넣는다
        tile = bh - 32
        _rounded_rect(canvas, (16, 16), (16 + tile, 16 + tile), 14, COLOR_NAV_TILE, -1)
        self._draw_maneuver_icon(canvas, (16 + tile // 2, 16 + tile // 2), maneuver,
                                 accent, tile)

        # 가운데 : 거리(크게) + 지시 문구
        tx = 16 + tile + 22
        if dist is not None:
            num = f"{dist:.0f}"
            tw, _ = _text(canvas, num, (tx, 62), 1.7, accent, 4)
            _text(canvas, "cm", (tx + tw + 9, 62), 0.75, accent, 2)
        label = "PARKED" if parked_spot else MANEUVER_LABEL.get(maneuver, maneuver)
        _text(canvas, label, (tx, 96), 0.82, COLOR_TEXT, 2)

        # 오른쪽 : 목적지(또는 세워진 자리)와 남은 총 거리
        rx = w - 20
        spot = nav.get("target_spot") or parked_spot
        head = "PARKED AT" if parked_spot else "DESTINATION"
        _text(canvas, head, (rx, 36), 0.42, COLOR_TEXT_DIM, 1, anchor="r")
        _text(canvas, str(spot) if spot else "NOT ASSIGNED", (rx, 74),
              1.1 if spot else 0.6,
              (COLOR_ARRIVED if parked_spot else COLOR_SPOT_TARGET) if spot
              else COLOR_TEXT_FAINT, 2, anchor="r")
        # 남은 거리는 배너 왼쪽 숫자와 다를 때만 적는다.
        # 같은 값을 두 번 적으면 어느 쪽이 무엇인지 헷갈린다.
        if remain is not None and (dist is None or abs(remain - dist) >= 1.0):
            _text(canvas, f"{remain:.0f}cm left", (rx, 98), 0.5, COLOR_TEXT_DIM,
                  1, anchor="r")

    def _draw_maneuver_icon(self, canvas, center, maneuver, color, size):
        """방향 지시 화살표. 타일 크기에 맞춰 굵기와 길이를 정한다."""
        cx, cy = center
        u = size / 80.0                 # 기준 크기(80px) 대비 배율
        t = max(4, int(8 * u))
        a = int(20 * u)                 # 화살표 팔 길이
        b = int(26 * u)                 # 세로 기둥 길이

        if maneuver == GUIDE_LEFT:
            cv2.line(canvas, (cx + a, cy + b), (cx + a, cy - 4), color, t, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx + a, cy - 4), (cx - a - 4, cy - 4), color, t,
                            cv2.LINE_AA, tipLength=0.45)
        elif maneuver == GUIDE_RIGHT:
            cv2.line(canvas, (cx - a, cy + b), (cx - a, cy - 4), color, t, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx - a, cy - 4), (cx + a + 4, cy - 4), color, t,
                            cv2.LINE_AA, tipLength=0.45)
        elif maneuver == GUIDE_UTURN:
            r = int(16 * u)
            cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, 360, color, t, cv2.LINE_AA)
            cv2.line(canvas, (cx - r, cy), (cx - r, cy + b), color, t, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (cx + r, cy), (cx + r, cy + b), color, t,
                            cv2.LINE_AA, tipLength=0.5)
        elif maneuver == GUIDE_ARRIVED:
            cv2.circle(canvas, (cx, cy), int(20 * u), color, max(3, int(4 * u)),
                       cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), int(8 * u), color, -1, cv2.LINE_AA)
        elif maneuver == GUIDE_UNKNOWN:
            # 아직 목표를 못 잡은 상태. 방향을 단정하면 안 된다.
            cv2.circle(canvas, (cx, cy), int(20 * u), COLOR_TEXT_FAINT,
                       max(3, int(4 * u)), cv2.LINE_AA)
            _text(canvas, "?", (cx, cy + int(11 * u)), 0.9 * u * 1.4,
                  COLOR_TEXT_DIM, 2, anchor="c")
        else:
            cv2.arrowedLine(canvas, (cx, cy + b), (cx, cy - b), color, t,
                            cv2.LINE_AA, tipLength=0.4)

    # ---------- 떠 있는 요소 ----------
    def _draw_compass(self, canvas, heading):
        """
        방위 나침반.

        화면이 진행 방향으로 도니까 북쪽이 어디인지 알 수 없다. 실제
        내비게이션에도 같은 이유로 나침반이 붙어 있다. 3번 주차장 상태
        화면이 '북쪽 위'로 서 있으므로, 이 바늘이 그 화면과의 대조표가 된다.
        """
        r = 24
        cx, cy = self.width - r - 18, self.banner_h + r + 16
        _card(canvas, (cx - r, cy - r), (cx + r, cy + r), COLOR_NAV_CARD,
              0.82, r, COLOR_NAV_LINE)

        # 화면 위쪽은 heading 방향이다. 북(월드 -y, 즉 -90도)이 화면에서
        # 어느 쪽으로 보이는지는 그 차이만큼 돌린 방향이다.
        rad = math.radians(FIXED_VIEW_HEADING - heading)
        dx, dy = math.sin(rad), -math.cos(rad)
        tip = (int(cx + dx * (r - 7)), int(cy + dy * (r - 7)))
        tail = (int(cx - dx * (r - 11)), int(cy - dy * (r - 11)))
        cv2.line(canvas, tail, (cx, cy), COLOR_TEXT_FAINT, 3, cv2.LINE_AA)
        cv2.arrowedLine(canvas, (cx, cy), tip, COLOR_NAV_NORTH, 3, cv2.LINE_AA,
                        tipLength=0.55)
        _text(canvas, "N", (int(cx + dx * (r + 9)), int(cy + dy * (r + 9)) + 4),
              0.4, COLOR_TEXT_DIM, 1, anchor="c")

    def _draw_status(self, canvas, nav, fps, extra_info):
        """
        좌하단 상태 알약. 지금 누구를 안내하는 중인지만 크게 보이면 된다.

        FPS와 좌표계 상태는 최종 화면(E00)이 영상 아래 줄에 이미 적고 있다.
        같은 걸 영상 안에 또 적으면 글자만 는다. 여기서는 호출자가 굳이
        넘겨준 값만 작게 덧붙인다.
        """
        label = None
        if nav is not None:
            car = nav.get("car_id") or f"#{nav.get('track_id')}"
            label = f"CAR {_ascii_label(car)}"

        sub = list(extra_info) if extra_info else []
        # 안내가 끝났거나 추적이 끊겨 '마지막 모습'을 붙들고 있는 중이라면
        # 그 사실을 적는다. 안 적으면 멈춘 화면을 실시간으로 착각한다.
        if nav is not None and nav.get("stale"):
            sub.append("last seen")
        if fps is not None:
            sub.append(f"{fps:.0f} fps")
        sub_text = "  ".join(sub)
        if label is None and not sub_text:
            return

        pad = 14
        lw = cv2.getTextSize(label, FONT, 0.62, 2)[0][0] if label else 0
        sw = cv2.getTextSize(sub_text, FONT, 0.44, 1)[0][0] if sub_text else 0
        box_w = pad * 2 + max(lw, sw)
        x1, y2 = 16, self.height - 16
        y1 = y2 - (54 if (label and sub_text) else 34)

        _card(canvas, (x1, y1), (x1 + box_w, y2), COLOR_NAV_CARD, 0.85, 12,
              COLOR_NAV_LINE)
        if label:
            _text(canvas, label, (x1 + pad, y1 + 26), 0.62, COLOR_TEXT, 2)
            if sub_text:
                _text(canvas, sub_text, (x1 + pad, y1 + 46), 0.44, COLOR_TEXT_DIM, 1)
        else:
            _text(canvas, sub_text, (x1 + pad, y1 + 23), 0.44, COLOR_TEXT_DIM, 1)

    def _draw_minimap(self, canvas, car_pos, nav, spot_status):
        """
        우하단 전체 조감도.

        확대된 시점만 보면 주차장 전체에서 어디쯤인지 알기 어렵다.
        이쪽은 화면과 달리 늘 북쪽이 위라서, 3번 주차장 상태 화면과
        같은 방향으로 읽힌다.
        """
        mw, mh = CONFIG['NAV_MINIMAP_W'], CONFIG['NAV_MINIMAP_H']
        x0, y0 = self.width - mw - 16, self.height - mh - 16
        _card(canvas, (x0, y0), (x0 + mw, y0 + mh), COLOR_NAV_CARD, 0.88, 12,
              COLOR_NAV_LINE)

        # 전체 구역이 들어가도록 축척 계산
        xs = [p[0] for p in self.spot_world_pos.values()] + [car_pos[0]]
        ys = [p[1] for p in self.spot_world_pos.values()] + [car_pos[1]]
        if self.gate_world_pos is not None:
            xs.append(self.gate_world_pos[0])
            ys.append(self.gate_world_pos[1])
        # 여백은 한 칸의 크기에 비례해서 준다 (모드마다 단위가 다르므로)
        min_x, max_x = min(xs) - self.cell_w, max(xs) + self.cell_w
        min_y, max_y = min(ys) - self.cell_h, max(ys) + self.cell_h
        pad = 12
        s = min((mw - pad * 2) / max(max_x - min_x, 1e-6),
                (mh - pad * 2) / max(max_y - min_y, 1e-6))
        # 남는 쪽은 가운데로 모은다
        ox = x0 + (mw - (max_x - min_x) * s) / 2
        oy = y0 + (mh - (max_y - min_y) * s) / 2

        def to_mini(p):
            return (int(ox + (p[0] - min_x) * s), int(oy + (p[1] - min_y) * s))

        # 주차장 바닥부터 깔아 준다. 자리 사각형만 띄엄띄엄 찍어 두면
        # 무엇을 줄여 놓은 그림인지 알 수 없다.
        if self.draw_lot:
            hw, hh = self.cell_w / 2.0, self.cell_h / 2.0
            tl = cell_to_world((0, 0))
            br = cell_to_world((get_rows() - 1, get_cols() - 1))
            cv2.rectangle(canvas, to_mini((tl[0] - hw, tl[1] - hh)),
                          to_mini((br[0] + hw, br[1] + hh)), COLOR_NAV_ROAD, -1)

        target = nav.get("target_spot")
        ratio = CONFIG['SPOT_FILL_RATIO']
        hw = self.cell_w * ratio / 2
        for spot_id, pos in self.spot_world_pos.items():
            hh = self.cell_h * get_spot_cell_count(spot_id) * ratio / 2
            p1 = to_mini((pos[0] - hw, pos[1] - hh))
            p2 = to_mini((pos[0] + hw, pos[1] + hh))
            # 조감도는 아주 작다. 선 한 겹으로는 안 보이므로 전부 채운다.
            if spot_id == target:
                color = COLOR_SPOT_TARGET
            elif spot_status.get(spot_id) == "full":
                color = COLOR_NAV_SPOT_FULL_EDGE
            else:
                color = COLOR_SPOT_EMPTY
            cv2.rectangle(canvas, p1, p2, color, -1)

        rest, planned = self._route_points(nav, car_pos)
        if rest:
            pts = np.array([to_mini(p) for p in rest], dtype=np.int32)
            cv2.polylines(canvas, [pts], False,
                          COLOR_NAV_ROUTE if planned else COLOR_NAV_ROUTE_HINT,
                          2, cv2.LINE_AA)

        mc = to_mini(car_pos)
        cv2.circle(canvas, mc, 6, COLOR_NAV_ROUTE_E, -1, cv2.LINE_AA)
        cv2.circle(canvas, mc, 4, (255, 255, 255), -1, cv2.LINE_AA)


# 자동 선택으로 마지막에 고른 차. 아래 pick_my_vehicle이 쓴다.
_LAST_PICK = None
# 마지막으로 내보낸 안내 결과 그대로. 안내가 끝났거나 추적이 끊겨 고를 차가
# 없을 때 이것을 다시 내보내 화면을 마지막 모습에 붙들어 둔다.
_LAST_NAV = None


def pick_my_vehicle(nav_results, car_id=None, sticky=True):
    """
    여러 추적 결과 중 내비게이션 화면에 띄울 '내 차'를 고른다.

    자동 선택은 한 번 고른 차를 계속 따라간다(sticky). 매 프레임 '목적지가
    가장 가까운 차'로 다시 고르면, 두 대가 안내를 받는 동안 남은 거리가
    엇갈릴 때마다 화면이 다른 차로 갈아탄다. 차가 바뀌면 서 있는 방향도
    있는 자리도 달라지니 화면이 통째로 뒤집힌 것처럼 보인다.

    안내가 끝나면(주차 완료) 그 차의 목표가 지워진다. 예전에는 그 순간
    '번호가 붙은 아무 차'로 넘어가서, 방금까지 안내하던 화면이 엉뚱한
    주차 차량과 그 자리(D-3 같은)로 튀었다. 추적이 잠깐 끊겨도 마찬가지였다.
    지금은 그러지 않고 **마지막 모습 그대로 붙들어 둔다.** 안내가 끝난 차의
    마지막 위치가 화면에 남는 편이, 관계없는 차로 갈아타는 것보다 낫다.

    Args:
        nav_results: C00_navigation.ParkingNavigator.update()의 반환 결과
        car_id:      특정 차량번호를 지정 (None이면 자동 선택)
        sticky:      자동 선택일 때 직전에 고른 차를 계속 따라갈지 여부

    Returns:
        선택된 nav 딕셔너리. 없으면 None.
        붙들어 둔 결과를 내보낼 때는 nav["stale"]이 True다.
    """
    global _LAST_PICK, _LAST_NAV

    def remember(nav):
        """고른 결과를 기억하고 그대로 돌려준다."""
        global _LAST_PICK, _LAST_NAV
        _LAST_PICK = key_of(nav)
        _LAST_NAV = dict(nav, stale=False)
        return nav

    def key_of(n):
        return n.get("car_id") or f"#{n.get('track_id')}"

    def frozen():
        """마지막 모습 그대로. 살아 있는 결과가 아님을 표시해 둔다."""
        if _LAST_NAV is None:
            return None
        return dict(_LAST_NAV, stale=True)

    if car_id is not None:
        for n in nav_results or ():
            if n.get("car_id") == car_id:
                return remember(n)
        # 지정한 차가 화면에서 사라졌다. 다른 차로 갈아타지 않는다.
        return frozen() if _LAST_PICK == car_id else None

    if not nav_results:
        return frozen()

    if sticky and _LAST_PICK is not None:
        for n in nav_results:
            if key_of(n) == _LAST_PICK and n.get("target_spot"):
                return remember(n)

    # 고르는 순서
    #   1) 안내 중인 차 (목표 구역이 있다). 그중 목적지가 가장 가까운 차.
    #   2) 직전에 보던 차. 안내는 끝났어도 아직 화면에 있으면 계속 본다.
    #   3) 마지막 모습 그대로 붙들어 둔다.
    #
    # '번호가 붙은 아무 차'를 고르던 단계는 뺐다. 안내가 끝나는 순간 화면이
    # 관계없는 주차 차량으로 튀는 원인이었고, 그 차는 어차피 갈 곳이 없어
    # 안내 화면에 띄울 내용도 없다. 세워둔 차들의 위치는 3번 격자가 보여준다.
    with_target = [n for n in nav_results if n.get("target_spot")]
    if with_target:
        picked = min(with_target, key=lambda n: n.get("distance_cm") or float('inf'))
        return remember(picked)

    if _LAST_PICK is not None:
        for n in nav_results:
            if key_of(n) == _LAST_PICK:
                return remember(n)

    return frozen()


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
        self._px_to_cm = None       # 픽셀 모드일 때만 채워진다 (PillarMapper)

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

        변환 자체는 PillarMapper.pixel_to_cm이 갖고 있다. 보정 때 찍은 기둥이
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
        # 통행 방향은 바닥 표시다. 배치 위, 자리/경로 아래에 깐다.
        self._draw_one_way(canvas)
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

    def _draw_one_way(self, canvas):
        """
        일방통행 순환 방향을 바닥 화살표로 그린다.

        안내선이 왜 한 바퀴 도는지가 이 화살표로 설명된다. 없으면 경로가
        엉뚱하게 도는 것처럼 보인다.
        """
        for (ax, ay), (bx, by) in ONE_WAY_SEGMENTS_WORLD:
            length = math.hypot(bx - ax, by - ay)
            if length == 0:
                continue

            steps = max(int(length / ONE_WAY_ARROW_STEP_CM), 1)
            for i in range(steps):
                t0 = i / steps
                t1 = (i + 1) / steps
                p0 = self.world_to_map((ax + (bx - ax) * t0, ay + (by - ay) * t0))
                p1 = self.world_to_map((ax + (bx - ax) * t1, ay + (by - ay) * t1))
                cv2.arrowedLine(canvas, p0, p1, COLOR_ONE_WAY, 1,
                                cv2.LINE_AA, tipLength=0.3)

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
                cv2.rectangle(canvas, p1, p2, COLOR_PANEL_BG, -1)
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
                # 현재 위치에서 남은 경유점까지만 이어서 그린다.
                # (첫 구간이 비스듬해지지 않게 C01이 모서리를 맞춰 준다)
                pts = [self.world_to_map(p) for p in route_from_position(
                    route, nav.get("route_index", 1), nav["world_pos"],
                    lot_map=_lot_map_of(self.navigator))]
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
            cv2.circle(canvas, (cx, cy), radius, COLOR_TEXT, 1)

            # 진행 방향 화살표 (정지 상태면 생략)
            heading = nav.get("heading_deg")
            if heading is not None:
                rad = math.radians(heading)
                hx = int(cx + math.cos(rad) * radius * 2.4)
                hy = int(cy + math.sin(rad) * radius * 2.4)
                cv2.arrowedLine(canvas, (cx, cy), (hx, hy), COLOR_TEXT, 2, tipLength=0.35)

            label = nav["car_id"] if nav["car_id"] else f"#{nav['track_id']}"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 2)
            cv2.putText(canvas, label, (cx - tw // 2, cy - radius - 7),
                        FONT, 0.5, color, 2, cv2.LINE_AA)

    def _draw_panel(self, canvas, nav_results, fps, extra_info):
        """오른쪽에 차량별 안내 정보를 표 형태로 표시."""
        x0 = self.map_w
        cv2.rectangle(canvas, (x0, 0), (self.width, self.height), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (x0, 0), (x0, self.height), COLOR_GRID, 1)

        pad = 14
        y = 32
        cv2.putText(canvas, "NAVIGATION", (x0 + pad, y), FONT, 0.62, COLOR_TEXT, 2, cv2.LINE_AA)

        y += 22
        if fps is not None:
            cv2.putText(canvas, f"FPS {fps:.1f}", (x0 + pad, y),
                        FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y += 14
        cv2.line(canvas, (x0 + pad, y), (self.width - pad, y), COLOR_GRID, 1)
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
            cv2.line(canvas, (x0 + pad, y), (self.width - pad, y), COLOR_GRID, 1)
            y += 20
            for line in extra_info:
                if y > self.height - 12:
                    break
                cv2.putText(canvas, line, (x0 + pad, y),
                            FONT, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                y += 17

        return canvas
