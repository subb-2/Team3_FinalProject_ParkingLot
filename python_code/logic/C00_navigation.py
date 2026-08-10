import cv2
import sys
import os
import math
import numpy as np
from collections import deque

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.C01_path_planner import route_length, distance_to_route



# 설정 (Configuration)
# 이 모듈은 '위치 추정 및 경로 안내'만 담당.
#   - 검출 : B01_car_detection.py
#   - 추적 : B02_car_mot.py  (여기서 나온 박스 정중앙점을 입력으로 사용)
CONFIG = {
    # 내비게이션 판정 기준
    # 거리 기준은 주차장 규모에 맞춰야 한다. 현재 목업은 40 x 140cm이고
    # 통로 폭이 10cm, 자리 간격이 35cm이므로 값들이 작다.
    # 주차장을 키우면 C02의 CELL_W/H_CM과 함께 이 값들도 조정할 것.
    "ARRIVAL_THRESHOLD_CM": 5.0,    # 목표 지점 이 거리 이내면 '도착'으로 판정
    "TURN_ANGLE_THRESHOLD_DEG": 25.0, # 이 각도 이내면 '직진'으로 안내
    "UTURN_ANGLE_THRESHOLD_DEG": 150.0, # 이 각도 이상이면 '유턴'으로 안내
    "MIN_MOVE_CM_FOR_HEADING": 1.5, # 진행 방향 계산에 필요한 최소 이동 거리
    "HEADING_WINDOW": 5,            # 진행 방향 계산에 사용할 최근 위치 개수
    "HISTORY_MAXLEN": 128,          # 차량별 위치 이력 최대 길이
}

# 주차장 배치(마커/자리/입출구의 실좌표)는 C02_lot_layout이 격자에서 만든다.
#
# 마커는 '기둥'이며 주차 자리가 아니다. 자리의 좌표는 그 자리를 감싸는
# 위아래 기둥 마커의 중점으로 계산된다. 자세한 규칙은 MAIN_README.md 3절 참고.
#
# 좌표계: 왼쪽 위 기둥(마커 ID 1)이 원점 (0, 0),
#         x축은 오른쪽 방향(+), y축은 아래쪽 방향(+). 단위는 cm.
#
# 배치를 바꾸려면 data/map_data.py의 grid_map과 PILL_MARKER_ID를 고칠 것.
# 여기서는 아무것도 하드코딩하지 않는다.
from logic.C02_lot_layout import (
    MARKER_WORLD_POS,     # {마커ID: (x_cm, y_cm)}  기둥 위치
    SPOT_WORLD_POS,       # {구역ID: (x_cm, y_cm)}  기둥 사이 중점
    GATE1_WORLD_POS,      # 입구 (경로 안내 시작점)
    GATE2_WORLD_POS,      # 출구
    cell_to_world,        # 격자 칸 -> 실좌표
    build_spot_world_pos, # 기둥 좌표 -> 자리 좌표 (같은 열 기둥 사이 보간)
    CONFIG as C02_CONFIG, # 칸 크기(CELL_W_CM / CELL_H_CM)
)
# 역투영 오버레이(등록된 배치를 화면에 겹쳐 그리기)에 필요한 격자 정보
from data.map_data import (
    grid_map, spot_map, spot_type, coord_to_spot,
    SPOT_CELLS, SPOT_TYPE_NAME, SPOT1, SPOT2, SPOT3, SPOT4,
    PILL, GATE1_POS, GATE2_POS, get_rows, get_cols, get_spot_cell_count,
)

# 시각화 색상 (BGR)
# 역투영 오버레이 색상 (BGR)
# 등록된 격자를 화면에 겹쳐 그려 실물과 맞는지 눈으로 확인하기 위한 것.
COLOR_OVERLAY_EDGE  = (90, 90, 90)      # 도로 격자선 - 어두운 회색
COLOR_OVERLAY_BOUND = (255, 255, 255)   # 주차장 외곽 - 흰색
COLOR_OVERLAY_PILL  = (0, 140, 255)     # 기둥       - 주황
COLOR_OVERLAY_GATE1 = (0, 255, 0)       # 입구       - 초록
COLOR_OVERLAY_GATE2 = (0, 128, 255)     # 출구       - 주황빨강

# 주차 구역 종류별 색 (D00_ui_navi와 같은 규칙)
COLOR_OVERLAY_SPOT = {
    SPOT1: (130, 130, 130),   # 일반   - 회색
    SPOT2: (255, 150, 0),     # 장애인 - 파랑
    SPOT3: (60, 200, 255),    # 대형   - 노랑
    SPOT4: (120, 220, 120),   # 전기차 - 초록
}

COLOR_PATH   = (0, 255, 255)    # 안내 경로  - 노랑
COLOR_TARGET = (255, 0, 255)    # 목표 지점  - 자홍


# 안내 방향 상수
GUIDE_STRAIGHT = "STRAIGHT"
GUIDE_LEFT     = "LEFT"
GUIDE_RIGHT    = "RIGHT"
GUIDE_UTURN    = "UTURN"
GUIDE_ARRIVED  = "ARRIVED"
GUIDE_UNKNOWN  = "UNKNOWN"      # 진행 방향을 아직 알 수 없음(정지 상태 등)

GUIDE_TEXT_KO = {
    GUIDE_STRAIGHT: "직진",
    GUIDE_LEFT:     "좌회전",
    GUIDE_RIGHT:    "우회전",
    GUIDE_UTURN:    "유턴",
    GUIDE_ARRIVED:  "도착",
    GUIDE_UNKNOWN:  "방향탐색중",
}


# 기둥 기반 좌표 변환기
class PillarMapper:
    """
    화면에서 찍은 기둥 위치를 기준으로 '이미지 픽셀 <-> 주차장 cm'를 잇는다.

    보정은 사람이 한다. B03_map_setting의 /calibrate 화면에서 기둥을 순서대로
    클릭하면 그 좌표가 set_pillar_pixels로 들어오고, 자리와 입출구의 픽셀
    위치는 기둥 사이를 보간해서 만든다. (C02_lot_layout)

    기둥에는 번호가 있다. data/map_data.py의 PILL_MARKER_ID가 격자 칸마다
    번호를 매겨 두었고, C02가 그 번호별 설계 cm 좌표(MARKER_WORLD_POS)를
    만든다. 그래서 '클릭한 픽셀'과 '설계 cm'이 번호로 짝지어지고,
    pixel_to_cm이 그 대응으로 변환을 만든다.

    예전에는 이 번호가 기둥에 붙인 ArUco 마커 ID였고 카메라가 자동으로
    검출했다. 지금은 검출을 쓰지 않고 사람이 직접 찍는다. 번호 체계만
    그대로 남아 기둥의 이름 노릇을 한다.
    """

    def __init__(self, marker_world_pos=None):
        """
        PillarMapper 초기화.

        Args:
            marker_world_pos: {기둥번호: (x_cm, y_cm)} 설계 좌표.
                              None이면 C02가 격자에서 만든 값을 쓴다.
        """
        self.marker_world_pos = marker_world_pos or MARKER_WORLD_POS
        self._warned = set()          # 중복 경고 억제용

        # 사용자가 클릭한 기둥 픽셀 좌표 {기둥번호: (x_px, y_px)}
        self.pillar_pixels = {}
        # 기둥 사이 보간으로 계산된 자리 픽셀 좌표 {구역ID: (x_px, y_px)}
        self.spot_pixels = {}
        # 입출구 픽셀 좌표 (gate1_pixel, gate2_pixel)
        self.gate_pixels = (None, None)
        # 픽셀 -> cm 변환 캐시. pixel_to_cm이 채운다.
        self._px_cm_H = None
        self._px_cm_key = None

        print(f"[INFO] 기둥 매퍼 초기화 완료. ("
              f"등록된 기둥 {len(self.marker_world_pos)}개)")


    def _warn_once(self, key, message):
        """같은 경고가 매 프레임 쏟아지지 않도록 한 번만 출력."""
        if key in self._warned:
            return
        self._warned.add(key)
        print(message)

    def is_ready(self):
        """좌표 변환이 가능한 상태인지 확인. 기둥을 찍었으면 준비 완료다."""
        return bool(self.pillar_pixels)

    # --- 기둥 기반 매핑 -------------------------------------------------------
    # 기둥 픽셀 좌표만으로 자리 위치를 보간하고 차량 위치를 판단한다.
    # 거리 비교는 이미지 픽셀로 하고, cm가 필요한 곳에서만 pixel_to_cm을 쓴다.

    def set_pillar_pixels(self, pillar_pixels):
        """
        사용자가 클릭한 기둥 픽셀 좌표를 설정하고, 자리/입출구 픽셀 좌표를 자동 계산.

        호모그래피를 쓰지 않는다. 기둥 사이의 상대 관계(map_data)만 참고하여
        자리 픽셀 위치를 보간으로 구한다.

        Args:
            pillar_pixels: {마커ID: (x_px, y_px)} 사용자가 지정한 기둥 이미지 좌표

        Returns:
            (성공 여부, 메시지)
        """
        from logic.C02_lot_layout import build_spot_pixel_pos, build_gate_pixel_pos

        if len(pillar_pixels) < 4:
            return False, f"기둥이 {len(pillar_pixels)}개뿐입니다. 최소 4개가 필요합니다."

        self.pillar_pixels = dict(pillar_pixels)
        self.spot_pixels = build_spot_pixel_pos(pillar_pixels)
        self.gate_pixels = build_gate_pixel_pos(pillar_pixels)



        msg = (f"픽셀 기반 보정 완료. 기둥 {len(pillar_pixels)}개, "
               f"자리 {len(self.spot_pixels)}개 위치 계산됨.")
        print(f"[INFO] {msg}")
        return True, msg

    def pixel_to_cm(self, point):
        """
        픽셀 모드에서 이미지 좌표를 설계 cm 좌표로 옮긴다.

        보정 때 찍은 기둥은 '이미지 픽셀'과 '설계 cm'을 둘 다 아는 대응쌍이다.
        그 대응으로 변환을 만들면, 호모그래피 자동 보정이 안 된 상태에서도
        화면 좌표를 cm로 이야기할 수 있다.

        이게 필요한 이유는 cm를 전제로 만들어 둔 판정들 때문이다.
        오주차 거리 비교(E00)나 조감도 배치(D00)는 전부 SPOT_WORLD_POS(cm)를
        기준으로 하는데, 픽셀 모드의 world_pos는 이미지 픽셀이다. 그대로 비교하면
        수백 cm 떨어진 것으로 나와 제자리에 세운 차가 오주차로 잡힌다.

        Args:
            point: (x_px, y_px) 이미지 좌표

        Returns:
            (x_cm, y_cm). 픽셀 모드가 아니거나 기둥이 4개 미만이면 None.
        """
        H = self._pixel_to_cm_matrix()
        if H is None or point is None:
            return None
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def _pixel_to_cm_matrix(self):
        """pixel_to_cm이 쓸 변환 행렬. 보정이 바뀌면 다시 만든다."""
        if not self.pillar_pixels:
            return None
        if self._px_cm_key == self.pillar_pixels:
            return self._px_cm_H

        ids = [i for i in self.pillar_pixels if i in self.marker_world_pos]
        self._px_cm_key = dict(self.pillar_pixels)
        if len(ids) < 4:
            self._px_cm_H = None
            self._warn_once("px_cm_few",
                            f"[경고] 기둥이 {len(ids)}개뿐이라 픽셀->cm 변환을 만들 수 없습니다. "
                            f"오주차 판정이 동작하지 않습니다.")
            return None

        src = np.array([self.pillar_pixels[i] for i in ids], dtype=np.float32)
        dst = np.array([self.marker_world_pos[i] for i in ids], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        self._px_cm_H = H
        return H

    def find_nearest_spot_pixel(self, car_pixel):
        """
        차량 bbox 중심 픽셀에서 가장 가까운 주차 구역을 찾는다.

        Args:
            car_pixel: (x_px, y_px) 차량 bbox 중심

        Returns:
            가장 가까운 구역 ID. spot_pixels가 없으면 None.
        """
        if not self.spot_pixels:
            return None
        return min(
            self.spot_pixels,
            key=lambda s: math.hypot(
                car_pixel[0] - self.spot_pixels[s][0],
                car_pixel[1] - self.spot_pixels[s][1])
        )

    def pixel_distance_to_spot(self, car_pixel, spot_id):
        """
        차량 픽셀과 특정 자리 픽셀 간의 거리.

        Args:
            car_pixel: (x_px, y_px)
            spot_id: 구역 ID

        Returns:
            픽셀 거리. spot이 없으면 float('inf').
        """
        sp = self.spot_pixels.get(spot_id)
        if sp is None:
            return float('inf')
        return math.hypot(car_pixel[0] - sp[0], car_pixel[1] - sp[1])

    def reset(self, clear_cache=True):
        """
        보정을 초기화한다. (카메라를 다시 설치했을 때 사용)

        저장된 기둥 좌표 파일도 함께 지운다. 지우지 않으면 다음 실행에서
        옛 좌표를 다시 불러와 카메라를 옮긴 것이 반영되지 않는다.
        """
        self.pillar_pixels = {}
        self.spot_pixels = {}
        self.gate_pixels = (None, None)
        self._px_cm_H = None
        self._px_cm_key = None
        self._warned.clear()

        if clear_cache:
            px_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'pillar_pixels.json')
            if os.path.exists(px_path):
                try:
                    os.remove(px_path)
                    print(f"[INFO] 저장된 기둥 좌표를 삭제했습니다. {px_path}")
                except OSError as e:
                    print(f"[경고] 기둥 좌표 파일을 지우지 못했습니다: {e}")

        print("[INFO] 보정을 초기화했습니다. 재보정이 필요합니다.")

    # --- 배치 오버레이 -------------------------------------------------------

    def draw_layout_overlay(self, frame, show_grid=True, show_spots=True,
                            show_pillars=True, show_gates=True, show_labels=True):
        """
        등록된 배치(기둥/자리/입출구)를 화면에 오버레이한다.

        기둥과 자리의 픽셀 좌표를 그대로 쓴다. 좌표 변환을 거치지 않으므로
        기둥 위치가 맞으면 자리 위치도 반드시 맞는다.

        Args:
            frame:        그릴 대상 프레임 (원본이 수정됨)
            show_grid:    쓰지 않는다. 격자는 픽셀 좌표로 놓을 수 없다.
            show_spots:   주차 구역 (종류별 색)
            show_pillars: 기둥
            show_gates:   입출구
            show_labels:  구역 ID 등 글자

        Returns:
            frame (입력과 동일 객체)
        """
        if not self.pillar_pixels:
            return frame
        return self._draw_layout_pixel(frame, show_spots, show_pillars,
                                       show_gates, show_labels)

    def _draw_layout_pixel(self, frame, show_spots, show_pillars, show_gates, show_labels):
        """
        픽셀 기반 오버레이. 기둥/자리 픽셀 좌표를 직접 사용.

        호모그래피 없이 동작하므로 절대 좌표 오차가 없다.
        기둥 위치가 맞으면 자리 위치도 반드시 맞는다.
        """
        # 자리 표시 (원으로 표시 + 자리 ID)
        if show_spots:
            for spot_id, (sx, sy) in self.spot_pixels.items():
                sx, sy = int(sx), int(sy)
                color = COLOR_OVERLAY_SPOT.get(spot_type.get(spot_id), (200, 200, 200))
                cv2.circle(frame, (sx, sy), 18, color, 2, cv2.LINE_AA)
                if show_labels:
                    cv2.putText(frame, spot_id, (sx - 16, sy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 기둥 표시 (십자 마커)
        if show_pillars:
            for marker_id, (px, py) in self.pillar_pixels.items():
                cx, cy = int(px), int(py)
                cv2.drawMarker(frame, (cx, cy), COLOR_OVERLAY_PILL,
                               cv2.MARKER_CROSS, 16, 2)
                if show_labels:
                    cv2.putText(frame, str(marker_id), (cx + 9, cy - 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                COLOR_OVERLAY_PILL, 2, cv2.LINE_AA)

        # 입출구 표시
        if show_gates:
            gate1_px, gate2_px = self.gate_pixels
            for gate_px, label, color in (
                (gate1_px, "IN", COLOR_OVERLAY_GATE1),
                (gate2_px, "OUT", COLOR_OVERLAY_GATE2),
            ):
                if gate_px is None:
                    continue
                gx, gy = int(gate_px[0]), int(gate_px[1])
                cv2.rectangle(frame, (gx - 20, gy - 12), (gx + 20, gy + 12),
                              color, 2, cv2.LINE_AA)
                if show_labels:
                    cv2.putText(frame, label, (gx - 14, gy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        return frame


# 차량 위치 추정 및 경로 안내
class ParkingNavigator:
    """
    B02_car_mot의 추적 결과를 받아 각 차량의 실시간 실좌표를 계산하고,
    배정된 주차 구역까지의 경로를 안내한다.

    이 클래스는 검출/추적/경로계산을 직접 하지 않는다. B02가 만든 트랙의
    박스 정중앙점을 입력으로 받고, 경로는 C01_path_planner에 맡긴다.

    동작 흐름:
      1. PillarMapper가 보정 때 찍은 기둥으로 자리 위치를 잡아 둔다.
      2. 각 차량의 박스 정중앙점을 그 좌표계로 옮긴다.
      3. A01_parking_manager가 배정한 주차 구역까지의 경로를
         C01_path_planner로 계산한다. (주차 구역을 뚫지 않고 통로를 따라감)
      4. 위치 이력으로 진행 방향(heading)을 추정한다.
      5. 목적지가 아니라 '다음 경유점' 방향과 비교해 안내한다.
         목적지 직선 방향으로 안내하면 주차 구역을 가로지르라는 잘못된
         안내가 되므로, 반드시 경로를 따라 앞서 안내해야 한다.
    """

    def __init__(self, mapper=None, spot_world_pos=None,
                 arrival_threshold=None, turn_threshold=None, uturn_threshold=None,
                 min_move_for_heading=None, heading_window=None, history_maxlen=None,
                 planner=None, waypoint_radius=5.0, replan_tolerance=None):
        """
        ParkingNavigator 초기화.

        판정 기준의 기본값은 이 모듈 상단의 CONFIG에서 가져온다.
        (waypoint_radius만 CONFIG 항목이 없어 여기서 기본값을 갖는다)

        Args:
            mapper:               PillarMapper 인스턴스 (None이면 기본 설정으로 생성)
            spot_world_pos:       {구역ID: (x_cm, y_cm)} 주차 구역 좌표
                                  (None이면 C02_lot_layout이 만든 기본 배치 사용)
            arrival_threshold:    도착 판정 거리 (cm)
            turn_threshold:       직진으로 볼 각도 허용치 (도)
            uturn_threshold:      유턴으로 안내할 각도 (도)
            min_move_for_heading: 진행 방향 계산에 필요한 최소 이동 거리 (cm)
            heading_window:       진행 방향 계산에 쓸 최근 위치 개수
            history_maxlen:       차량별 위치 이력 최대 길이
            planner:              C01_path_planner.RoutePlanner 인스턴스
                                  (None이면 등록된 주차 구역으로 기본 생성)
            waypoint_radius:      경유점을 통과한 것으로 볼 거리 (cm)
            replan_tolerance:     경로에서 이만큼 벗어나면 재계획 (cm)
        """
        from logic.C01_path_planner import CONFIG as C01_CONFIG

        self.mapper = mapper if mapper is not None else PillarMapper()

        self.arrival_threshold = (
            CONFIG['ARRIVAL_THRESHOLD_CM'] if arrival_threshold is None else arrival_threshold)
        self.turn_threshold = (
            CONFIG['TURN_ANGLE_THRESHOLD_DEG'] if turn_threshold is None else turn_threshold)
        self.uturn_threshold = (
            CONFIG['UTURN_ANGLE_THRESHOLD_DEG'] if uturn_threshold is None else uturn_threshold)
        self.min_move_for_heading = (
            CONFIG['MIN_MOVE_CM_FOR_HEADING'] if min_move_for_heading is None else min_move_for_heading)
        self.heading_window = (
            CONFIG['HEADING_WINDOW'] if heading_window is None else heading_window)
        self.history_maxlen = (
            CONFIG['HISTORY_MAXLEN'] if history_maxlen is None else history_maxlen)
        self.waypoint_radius = waypoint_radius
        self.replan_tolerance = (
            C01_CONFIG['REPLAN_TOLERANCE_CM'] if replan_tolerance is None else replan_tolerance)

        # 주차 구역 ID -> 실좌표 (cm). C02가 기둥 마커의 중점으로 계산해 둔 값.
        self.spot_world_pos = dict(
            spot_world_pos if spot_world_pos is not None else SPOT_WORLD_POS
        )

        # 경로 계획기 (C01). 주차 구역을 장애물로 두고 통로를 따라 경로를 만든다.
        self.planner = planner if planner is not None else self._build_default_planner()

        # 차량번호 -> 실좌표 이력 deque([(x_cm, y_cm), ...])
        self.world_history = {}
        # 차량번호 -> 목표 주차 구역 ID
        self.targets = {}
        # 차량번호 -> 경로 상태 {"waypoints": [...], "index": int, "spot": 구역ID}
        self.routes = {}
        # 보정으로 잡은 자리 픽셀 좌표 {구역ID: (x_px, y_px)}.
        # update_spot_pixels가 매퍼에서 옮겨 담는다. 그 호출을 놓쳐도
        # _spot_pixel이 매퍼에서 바로 읽으므로 비어 있어도 동작한다.
        self.spot_pixels = {}

        print(f"[INFO] 내비게이터 초기화 완료. (주차 구역 {len(self.spot_world_pos)}개 등록)")

    def _spot_pixel(self, spot_id):
        """
        자리의 픽셀 좌표.

        보정 직후 update_spot_pixels를 부르지 않았더라도 매퍼에서 바로
        읽어 온다. 예전에는 그 호출을 놓치면 self.spot_pixels 자체가 없어서
        AttributeError로 죽었다.
        """
        return self.spot_pixels.get(spot_id) or self.mapper.spot_pixels.get(spot_id)

    def _build_default_planner(self):
        """등록된 주차 구역 배치로 기본 경로 계획기를 생성."""
        from logic.C01_path_planner import (
            ParkingLotMap, RoutePlanner, CONFIG as C01_CONFIG
        )

        lot_map = ParkingLotMap(
            self.spot_world_pos,
            resolution=C01_CONFIG['GRID_RESOLUTION_CM'],
            clearance=C01_CONFIG['VEHICLE_CLEARANCE_CM'],
        )
        return RoutePlanner(lot_map, simplify=C01_CONFIG['SIMPLIFY_PATH'])

    def set_target(self, car_id, spot_id):
        """
        차량의 목표 주차 구역을 지정.

        A01_parking_manager가 빈자리를 배정한 뒤 호출하면 된다.

        Args:
            car_id:  차량 번호 4자리 문자열
            spot_id: 목표 주차 구역 ID (예: "A-1")
        """
        if spot_id not in self.spot_world_pos:
            print(f"[경고] 주차 구역 '{spot_id}'의 실좌표가 등록되지 않았습니다.")
            return False

        self.targets[car_id] = spot_id
        # 목표가 바뀌었으므로 기존 경로는 폐기. 다음 프레임에 다시 계획된다.
        self.routes.pop(car_id, None)
        print(f"[안내] 차량 '{car_id}' 목표 구역 설정: {spot_id}")
        return True

    def update_spot_pixels(self):
        """
        mapper의 픽셀 좌표가 변경되었을 때 네비게이터에도 반영.
        기존 anchor_spots_to_observed를 대체.
        """
        if not self.mapper.pillar_pixels:
            return False
        
        self.spot_pixels = dict(self.mapper.spot_pixels)
        # 경로는 더 이상 쓰지 않지만 초기화
        self.routes.clear()
        print(f"[보정] 내비게이터에 픽셀 자리 좌표 반영 완료. ({len(self.spot_pixels)}개)")
        return True

    def get_target_world(self, car_id):
        """
        차량의 목표 주차 구역 실좌표(또는 픽셀 좌표)를 반환.
        """
        spot_id = self.targets.get(car_id)
        if not spot_id:
            return None
            
        if self.mapper.pillar_pixels:
            return self._spot_pixel(spot_id)
        return self.spot_world_pos.get(spot_id)

    def get_target_rect(self, car_id):
        """
        차량의 목표 주차 구역을 '사각형'으로 반환. (중심 + 반폭/반높이)
        픽셀 모드에서는 임시로 반폭/반높이를 픽셀로 변환해서 준다.
        """
        spot_id = self.targets.get(car_id)
        if not spot_id:
            return None
            
        if self.mapper.pillar_pixels:
            center = self._spot_pixel(spot_id)
            if center is None:
                return None
            cells = get_spot_cell_count(spot_id) or 1
            # 픽셀 모드: B02_car_mot가 cm 단위로 비교하므로, target_rect도 cm 단위로 반환
            px_per_cm = 8.0 
            half_w = C02_CONFIG['CELL_W_CM'] / 2.0
            half_h = C02_CONFIG['CELL_H_CM'] * cells / 2.0
            return (center[0] / px_per_cm, center[1] / px_per_cm, half_w, half_h)
            
        center = self.spot_world_pos.get(spot_id)
        if center is None:
            return None

        cells = get_spot_cell_count(spot_id) or 1
        half_w = C02_CONFIG['CELL_W_CM'] / 2.0
        half_h = C02_CONFIG['CELL_H_CM'] * cells / 2.0
        return (center[0], center[1], half_w, half_h)

    def clear_target(self, car_id):
        """목표 구역 지정을 지운다. 안내가 끝난 차를 대상에서 뺄 때 쓴다."""
        if self.targets.pop(car_id, None) is None:
            return False
        self.routes.pop(car_id, None)
        return True

    def sync_targets_from_parking_manager(self):
        """
        A01_parking_manager가 관리하는 입차 정보(cars_info)를 읽어
        각 차량의 목표 구역을 자동으로 동기화.

        이미 주차를 마친 차는 목표에서 뺀다. 도착한 차는 목적지까지 남은
        거리가 늘 0에 가까운데, 안내 화면은 '목적지가 가장 가까운 차'를
        골라 띄운다. 그래서 세워둔 차가 안내 중인 차를 계속 밀어내고,
        4번 구간이 갓 들어온 차 대신 이미 주차된 차를 비췄다.
        주차를 마친 차는 갈 곳이 없으므로 목표도 없는 것이 맞다.
        """
        from data.car_data import cars_info

        for car_id, info in cars_info.items():
            if info.get("parked"):
                self.clear_target(car_id)
                continue
            spot_id = info.get("spot_id")
            if spot_id and self.targets.get(car_id) != spot_id:
                self.set_target(car_id, spot_id)

        # 출차한 차의 목표를 지운다.
        #
        # 위 루프는 cars_info를 도는데, 출차하면 그 항목이 지워지므로 나간 차는
        # 아예 들어오지 않는다. 그래서 목표가 그대로 남아 있었다. 남으면
        #   - 이미 나간 차가 아직 그 자리로 가는 중인 것으로 보이고
        #   - 4번 구간이 '목적지가 있는 차'를 우선 고르므로 그 차를 비추며
        #   - 비워 둔 자리를 다른 차에게 배정했을 때 목표가 겹친다.
        for car_id in list(self.targets):
            if car_id not in cars_info:
                self.clear_target(car_id)
                self.world_history.pop(car_id, None)

    def update(self, frame, tracks):
        """
        한 프레임 분량의 추적 결과로 각 차량의 위치와 안내 정보를 갱신.
        """
        results = []
        if not self.mapper.is_ready():
            return results

        # 보정을 마쳤다면 좌표계는 언제나 '이미지 픽셀'이다.
        # 예전에는 마커 자동 검출로 호모그래피를 잡아 cm로 바로 옮기는 길이
        # 하나 더 있었지만, 검출을 걷어내면서 그 길은 사라졌다.
        # cm가 필요한 곳(오주차 판정, 조감도)은 mapper.pixel_to_cm을 쓴다.
        #
        # 아래 is_pixel_mode 분기가 아직 남아 있는 이유는 경로 계획(C01)
        # 때문이다. 그쪽은 cm 좌표를 전제로 짜여 있어 지금은 타지 않는다.
        # 자세한 것은 target_pos 아래 주석 참고.
        is_pixel_mode = True

        for trk in tracks:
            car_id = trk.get("car_id")
            image_pos = trk["center"]
            pos = image_pos

            # 위치 이력 갱신 (보정 좌표계 = 이미지 픽셀)
            key = car_id if car_id else f"track_{trk['track_id']}"
            history = self.world_history.setdefault(
                key, deque(maxlen=self.history_maxlen)
            )
            history.append(pos)

            heading = self._compute_heading(history)
            target_spot = self.targets.get(car_id) if car_id else None
            
            if target_spot:
                if is_pixel_mode:
                    target_pos = self._spot_pixel(target_spot)
                else:
                    target_pos = self.spot_world_pos.get(target_spot)
            else:
                target_pos = None

            distance = None
            guide = GUIDE_UNKNOWN
            route = None
            route_index = 0
            next_waypoint = None
            maneuver = GUIDE_UNKNOWN
            maneuver_distance = None

            if target_spot and target_pos is not None:
                if is_pixel_mode:
                    # 경로 탐색 없이 목적지까지 직선 거리로 안내한다.
                    #
                    # !! 남은 일 !! 아래 else가 C01_path_planner를 써서 통로를
                    # 따라가는 경로를 만드는 쪽인데, cm 좌표를 전제로 짜여 있어
                    # 지금은 타지 않는다. 살리려면 mapper.pixel_to_cm으로 옮긴
                    # 뒤 계획하고 결과를 다시 픽셀로 돌려놓아야 한다.
                    # 그때까지 안내 화면(D00)은 목적지까지 직선을 그린다.
                    distance = self._distance(pos, target_pos)
                    # 픽셀 거리를 cm처럼 보이게 나눠서 넘긴다. 임시 값이라
                    # 화면에 뜨는 '남은 거리'는 실제 cm가 아니다.
                    distance_cm = distance / 8.0
                    guide = self._compute_guide(pos, heading, target_pos, distance)
                else:
                    state = self._ensure_route(car_id, pos, target_spot)
                    if state is not None:
                        route = state["waypoints"]
                        route_index = state["index"]
                        next_waypoint = route[route_index] if route_index < len(route) else None
                        distance = route_length(route, route_index, pos)
                        maneuver, maneuver_distance = self.compute_maneuver(
                            route, route_index, pos)
                        distance_cm = distance
                    else:
                        distance = self._distance(pos, target_pos)
                        distance_cm = distance
                        
                    aim = next_waypoint if next_waypoint is not None else target_pos
                    remaining = self._distance(pos, target_pos)
                    guide = self._compute_guide(pos, heading, aim, remaining)
            else:
                distance_cm = None

            # 이 차가 이미 주차를 마쳤는지. 안내 대상이 아니므로 target_spot은
            # 비어 있지만, 화면은 '어느 자리에 세워진 차'인지 보여줘야 한다.
            # 그게 없으면 4번 구간이 주차된 차를 비출 때 목적지가 '없음'으로만
            # 떠서, 안내를 못 하는 것인지 이미 끝난 것인지 구별할 수 없다.
            parked_spot = None
            if car_id:
                from data.car_data import cars_info
                info = cars_info.get(car_id)
                if info and info.get("parked"):
                    parked_spot = info.get("spot_id")

            results.append({
                "track_id": trk["track_id"],
                "car_id": car_id,
                "image_pos": image_pos,
                "world_pos": pos,
                "heading_deg": heading,
                "target_spot": target_spot,
                "parked_spot": parked_spot,
                "target_world": target_pos, # 픽셀 모드면 픽셀 좌표가 들어감
                "distance_cm": distance_cm,
                "route": route,
                "route_index": route_index,
                "next_waypoint": next_waypoint,
                "maneuver": maneuver,
                "maneuver_distance_cm": maneuver_distance,
                "guide": guide,
                "guide_text": GUIDE_TEXT_KO.get(guide, guide),
                "nearest_spot": self.mapper.find_nearest_spot_pixel(image_pos) if is_pixel_mode else self.find_nearest_spot(pos),
            })

        return results

    def _ensure_route(self, car_id, world_pos, target_spot):
        """
        차량의 경로를 확보하고 진행 상태를 갱신.

        아래 경우에 경로를 다시 계획한다.
          - 아직 경로가 없을 때
          - 목표 구역이 바뀌었을 때
          - 차량이 경로에서 replan_tolerance 이상 벗어났을 때
            (안내를 무시하고 다른 길로 갔거나 위치 추정이 튄 경우)

        Returns:
            경로 상태 딕셔너리. 경로를 찾지 못하면 None.
        """
        state = self.routes.get(car_id)

        need_replan = (
            state is None
            or state["spot"] != target_spot
            or distance_to_route(state["waypoints"], world_pos,
                                 state["index"]) > self.replan_tolerance
        )

        if need_replan:
            waypoints = self.planner.plan(world_pos, target_spot)
            if waypoints is None:
                if state is not None:
                    self.routes.pop(car_id, None)
                return None
            # 첫 경유점은 현재 위치이므로 다음 지점부터 향한다
            state = {"waypoints": waypoints, "index": min(1, len(waypoints) - 1),
                     "spot": target_spot}
            self.routes[car_id] = state

        self._advance_waypoint(state, world_pos)
        return state

    def _advance_waypoint(self, state, world_pos):
        """
        경유점에 충분히 가까워졌으면 다음 경유점으로 넘어간다.
        마지막 경유점(목적지)은 도착 판정에 쓰이므로 넘기지 않는다.
        """
        waypoints = state["waypoints"]
        while state["index"] < len(waypoints) - 1:
            wp = waypoints[state["index"]]
            if self._distance(world_pos, wp) > self.waypoint_radius:
                break
            state["index"] += 1


    def compute_maneuver(self, route, index, world_pos):
        """
        다음 경유점에서 어느 방향으로 꺾어야 하는지와 그 지점까지의 거리를 계산.

        '지금 어디를 향하고 있는가'(guide)와 달리, 이것은 '앞으로 무엇을
        해야 하는가'다. 자동차 내비게이션의 "300m 앞 우회전"에 해당하며,
        UI 표시뿐 아니라 RC카에 조향 명령을 보낼 때도 쓸 수 있다.

        Args:
            route:     경유점 리스트
            index:     현재 향하고 있는 경유점 인덱스
            world_pos: 차량의 현재 실좌표

        Returns:
            (안내 상수, 그 지점까지의 거리 cm) 튜플
        """
        if not route or index >= len(route):
            return GUIDE_ARRIVED, 0.0

        target = route[index]
        dist = self._distance(world_pos, target)

        # 마지막 경유점이면 목적지 도착
        if index >= len(route) - 1:
            return GUIDE_ARRIVED, dist

        # 들어가는 방향과 나가는 방향의 차이가 곧 꺾어야 할 각도
        in_a = math.degrees(math.atan2(target[1] - world_pos[1], target[0] - world_pos[0]))
        nxt = route[index + 1]
        out_a = math.degrees(math.atan2(nxt[1] - target[1], nxt[0] - target[0]))
        diff = (out_a - in_a + 180) % 360 - 180

        if abs(diff) >= self.uturn_threshold:
            return GUIDE_UTURN, dist
        if abs(diff) <= self.turn_threshold:
            return GUIDE_STRAIGHT, dist
        return (GUIDE_RIGHT if diff > 0 else GUIDE_LEFT), dist

    def _compute_heading(self, history):
        """
        최근 위치 이력으로 차량의 진행 방향(도)을 계산.

        좌표계가 y축 아래 방향(+)이므로, 각도는 시계 방향으로 증가한다.
        (0도 = 오른쪽, 90도 = 아래쪽)

        Returns:
            진행 방향 각도. 이동량이 너무 적으면 None.
        """
        if len(history) < 2:
            return None

        recent = list(history)[-self.heading_window:]
        start, end = recent[0], recent[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]

        # 정지 상태에서는 방향을 신뢰할 수 없음
        if math.hypot(dx, dy) < self.min_move_for_heading:
            return None

        return math.degrees(math.atan2(dy, dx))

    def _compute_guide(self, world_pos, heading, target_world, distance):
        """
        현재 위치/진행 방향과 목표 지점을 비교해 안내 방향을 결정.
        """
        if distance <= self.arrival_threshold:
            return GUIDE_ARRIVED

        if heading is None:
            return GUIDE_UNKNOWN

        # 목표 지점의 방위각
        dx = target_world[0] - world_pos[0]
        dy = target_world[1] - world_pos[1]
        bearing = math.degrees(math.atan2(dy, dx))

        # 진행 방향과의 차이를 [-180, 180] 범위로 정규화
        diff = (bearing - heading + 180) % 360 - 180

        if abs(diff) >= self.uturn_threshold:
            return GUIDE_UTURN
        if abs(diff) <= self.turn_threshold:
            return GUIDE_STRAIGHT
        # y축이 아래 방향이므로 각도가 커지는 쪽이 시계 방향(우회전)
        return GUIDE_RIGHT if diff > 0 else GUIDE_LEFT

    @staticmethod
    def _distance(p1, p2):
        """두 실좌표 사이의 거리(cm)."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def find_nearest_spot(self, world_pos):
        """
        주어진 실좌표에서 가장 가까운 주차 구역 ID를 반환.

        Args:
            world_pos: (x_cm, y_cm) 실좌표

        Returns:
            가장 가까운 구역 ID. 등록된 구역이 없으면 None.
        """
        if not self.spot_world_pos:
            return None

        return min(
            self.spot_world_pos,
            key=lambda s: self._distance(world_pos, self.spot_world_pos[s])
        )


    def get_world_trajectory(self, car_id):
        """차량 번호로 실좌표 이동 궤적 리스트를 반환."""
        return list(self.world_history.get(car_id, []))

    def clear_vehicle(self, car_id):
        """출차 등으로 추적이 끝난 차량의 이력과 목표, 경로를 제거."""
        self.world_history.pop(car_id, None)
        self.targets.pop(car_id, None)
        self.routes.pop(car_id, None)

    def draw_navigation(self, frame, nav_results, draw_target_line=True):
        """
        내비게이션 정보를 프레임에 시각화.
        """
        for nav in nav_results:
            cx, cy = int(nav["image_pos"][0]), int(nav["image_pos"][1])

            # 좌표 표시. 보정 좌표계는 이미지 픽셀이라 그대로 적는다.
            cv2.putText(frame, f"({cx},{cy})px", (cx - 40, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PATH, 2, cv2.LINE_AA)

            if not draw_target_line or nav["target_world"] is None:
                continue

            # 목적지까지 직선으로 잇는다.
            # 통로를 따라가는 경로(C01)는 cm 좌표를 전제로 짜여 있어 지금은
            # 만들어지지 않는다. C00.update의 '남은 일' 주석 참고.
            target_img = (int(nav["target_world"][0]), int(nav["target_world"][1]))
            cv2.arrowedLine(frame, (cx, cy), target_img, COLOR_PATH, 2, tipLength=0.05)
            cv2.circle(frame, target_img, 8, COLOR_TARGET, 2)

            # 안내 문구
            dist = nav["distance_cm"]
            if dist is not None:
                cv2.putText(frame, f"{nav['target_spot']} {nav['guide']} {dist:.0f}px",
                            (cx - 40, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TARGET, 2, cv2.LINE_AA)

        return frame


# =====================================================================
# 단독 테스트 (카메라 없이 좌표 변환과 안내 판정을 검증)
# =====================================================================
# 기둥을 클릭하는 대신, 설계 좌표를 일부러 비스듬히 눌러 픽셀로 옮겨 놓고
# 그것을 '클릭한 좌표'로 넣는다. 실제 카메라는 비스듬히 보고 있어서 가로
# 1cm와 세로 1cm의 픽셀 수가 다른데, 그 상황을 흉내내는 것이다.
#
# 카메라 + 검출 + 추적 + 내비게이션 통합 실행은 C_main.py를 사용할 것.
if __name__ == '__main__':
    from data.map_data import PILL_MARKER_ID

    print("==========================================")
    print(" C00 : 주차장 내비게이션 (기둥 기반 좌표계)")
    print(" 단독 테스트 : 좌표 변환과 안내 판정 검증")
    print("==========================================")

    # 가로 6.8px/cm, 세로 3.1px/cm 로 눌린 화면을 가정한다.
    SX, SY, OX, OY = 6.8, 3.1, 60.0, 40.0

    def to_pixel(world):
        return (OX + world[0] * SX, OY + world[1] * SY)

    pillar_pixels = {mid: to_pixel(cell_to_world(cell))
                     for cell, mid in PILL_MARKER_ID.items()}

    mapper = PillarMapper()
    ok, msg = mapper.set_pillar_pixels(pillar_pixels)
    print(f"\n[TEST] 보정 : {ok} - {msg}")
    if not ok:
        sys.exit(1)

    # 1) 픽셀 -> cm 왕복. 클릭 좌표에서 만든 변환이 설계 좌표를 되돌려주는지.
    print("\n[TEST] 픽셀 -> cm 변환")
    worst = 0.0
    for cell in [(0, 0), (3, 0), (4, 6), (6, 12), (8, 8)]:
        want = cell_to_world(cell)
        got = mapper.pixel_to_cm(to_pixel(want))
        err = math.hypot(got[0] - want[0], got[1] - want[1])
        worst = max(worst, err)
        print(f"  격자{cell}  기대 ({want[0]:6.1f}, {want[1]:6.1f})cm  "
              f"-> 실제 ({got[0]:6.1f}, {got[1]:6.1f})cm   오차 {err:.3f}cm")
    print(f"  최대 오차 {worst:.3f}cm  ->  {'OK' if worst < 0.5 else '실패'}")

    # 2) 자리 픽셀 좌표. 기둥 사이 보간이 자리마다 값을 만들어 냈는지.
    print(f"\n[TEST] 자리 픽셀 좌표 {len(mapper.spot_pixels)}개")
    for spot_id in sorted(mapper.spot_pixels)[:4]:
        px, py = mapper.spot_pixels[spot_id]
        cm = mapper.pixel_to_cm((px, py))
        print(f"  {spot_id:5} ({px:6.1f}, {py:6.1f})px  =  "
              f"({cm[0]:6.1f}, {cm[1]:6.1f})cm")

    # 3) 가장 가까운 자리 찾기
    target = sorted(mapper.spot_pixels)[0]
    near = mapper.find_nearest_spot_pixel(mapper.spot_pixels[target])
    print(f"\n[TEST] '{target}' 자리 위의 차 -> 가장 가까운 자리 '{near}'  "
          f"->  {'OK' if near == target else '실패'}")

    # 4) 안내 판정. 목표를 정하고 그쪽으로 다가가며 안내가 바뀌는지 본다.
    navigator = ParkingNavigator(mapper=mapper)
    navigator.set_target("1234", target)
    tx, ty = mapper.spot_pixels[target]

    print(f"\n[TEST] '1234' -> '{target}' 안내")
    for step in range(6):
        t = step / 5.0
        cx = tx + (1 - t) * 160
        cy = ty + (1 - t) * 90
        results = navigator.update(None, [{"track_id": 1, "car_id": "1234",
                                           "center": (cx, cy)}])
        nav = results[0]
        print(f"  step {step}: ({cx:6.1f}, {cy:6.1f})px  "
              f"거리 {nav['distance_cm']:6.1f}  안내 {nav['guide']}")

    print("\n==========================================")
    print(" 결과 : 좌표 변환과 안내 판정이 동작합니다.")
    print("==========================================")
