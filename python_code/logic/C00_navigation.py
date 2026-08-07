import cv2
import sys
import os
import math
import numpy as np
import cv2.aruco as aruco
from collections import deque

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.C01_path_planner import route_length, distance_to_route



# 설정 (Configuration)
# 이 모듈은 '위치 추정 및 경로 안내'만 담당.
#   - 검출 : B01_car_detection.py
#   - 추적 : B02_car_mot.py  (여기서 나온 박스 정중앙점을 입력으로 사용)
CONFIG = {
    "MIN_MARKERS_FOR_HOMOGRAPHY": 4, # 호모그래피 계산을 시도할 최소 마커 수

    # 카메라가 고정 설치된 경우 True 권장.
    # 품질 기준을 만족한 호모그래피를 고정해 계속 재사용하므로,
    # 차량이 마커를 가려도 위치 추정이 끊기지 않는다.
    # 카메라가 움직이면 False로 둘 것.
    "LOCK_HOMOGRAPHY": True,

    # 호모그래피 품질 기준
    # 마커가 정확히 4개면 대응이 틀려 있어도 그 4점에는 항상 오차 0으로 맞춰지므로 검증이 불가능
    # 따라서 최소 개수로 성급히 고정하지 않고, 아래 개수 이상이 보일 때만 확정.
    "MARKERS_FOR_LOCK": 6,          # 이 개수 이상 보일 때만 호모그래피를 확정
    "MAX_REPROJ_ERROR_CM": 10.0,    # 평균 재투영 오차가 이보다 크면 채택하지 않음 (수동 클릭을 위해 기준 완화)
    "MIN_MARKER_SPREAD": 0.02,      # 마커가 한 줄에 몰려 있으면 거부 (0에 가까울수록 일직선)
    "RANSAC_THRESH_CM": 5.0,        # RANSAC 이상치 판정 임계값 (실좌표 cm 기준)

    # ---------------------------------------------------------------------
    # 호모그래피 캐시
    #
    # 카메라가 고정 설치되어 있으면 호모그래피는 '상수'다. 그런데 지금까지는
    # 실행할 때마다 마커 6개를 새로 검출해야 했고, 검출 하나가 흔들리면
    # 전체가 NOT READY로 멈췄다.
    #
    # 한 번 품질 기준을 만족하면 파일로 저장하고, 다음 실행부터는 그것을
    # 그대로 불러 쓴다. 마커가 하나도 안 보여도 좌표 변환이 된다.
    #
    # 카메라를 옮겼으면 반드시 다시 잡아야 한다.
    #   - C_main 실행 중이면 /recalibrate 호출
    #   - 또는 캐시 파일을 지우고 재실행
    # ---------------------------------------------------------------------
    "USE_HOMOGRAPHY_CACHE": True,
    "HOMOGRAPHY_CACHE": os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'config', 'homography.npz')
    ),

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

COLOR_MARKER = (255, 200, 0)    # 마커       - 하늘색
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


# 마커 기반 좌표 변환기
class MarkerMapper:
    """
    ArUco 마커를 검출하여 이미지 좌표 <-> 주차장 실좌표(cm) 변환을 담당.

    카메라가 비스듬히 설치되어 있어도, 바닥 평면 위의 마커 4개 이상이
    보이면 호모그래피(Homography)로 정확한 평면 좌표를 복원할 수 있다.

    LOCK_HOMOGRAPHY가 True면 한 번 계산한 변환 행렬을 계속 재사용하므로,
    이후 차량이 마커를 가려도 위치 추정이 끊기지 않는다.
    """

    def __init__(self, marker_world_pos=None,
                 min_markers=None, lock_homography=None, lock_markers=None,
                 max_error=None, min_spread=None, ransac_thresh_cm=None):
        """
        MarkerMapper 초기화.

        모든 인자의 기본값은 이 모듈 상단의 CONFIG에서 가져온다.
        여기에 값을 직접 적으면 CONFIG를 고쳐도 반영되지 않는 경로가 생기므로
        숫자나 사전 이름을 하드코딩하지 말 것. (MAIN_README 5절)

        Args:
            marker_world_pos: {마커ID: (x_cm, y_cm)} 형태의 실좌표 매핑
            min_markers:      호모그래피 계산을 시도할 최소 마커 수
            lock_homography:  True면 품질 기준 충족 시 호모그래피를 고정
            lock_markers:     고정을 확정하기 위해 필요한 마커 수
            max_error:        허용할 최대 평균 재투영 오차 (cm)
            min_spread:       마커 배치의 최소 퍼짐 정도 (일직선 배치 거부)
            ransac_thresh_cm: RANSAC 이상치 판정 임계값 (cm, 실좌표 기준)
        """
        min_markers = CONFIG['MIN_MARKERS_FOR_HOMOGRAPHY'] if min_markers is None else min_markers
        lock_homography = CONFIG['LOCK_HOMOGRAPHY'] if lock_homography is None else lock_homography
        lock_markers = CONFIG['MARKERS_FOR_LOCK'] if lock_markers is None else lock_markers
        max_error = CONFIG['MAX_REPROJ_ERROR_CM'] if max_error is None else max_error
        min_spread = CONFIG['MIN_MARKER_SPREAD'] if min_spread is None else min_spread
        ransac_thresh_cm = CONFIG['RANSAC_THRESH_CM'] if ransac_thresh_cm is None else ransac_thresh_cm

        self.marker_world_pos = marker_world_pos or MARKER_WORLD_POS
        self.min_markers = min_markers
        self.lock_homography = lock_homography
        self.lock_markers = max(lock_markers, min_markers)
        self.max_error = max_error
        self.min_spread = min_spread
        self.ransac_thresh_cm = ransac_thresh_cm

        # 이미지 -> 실좌표 변환 행렬 및 그 역행렬
        self.H = None
        self.H_inv = None
        # 현재 호모그래피의 품질 지표
        self.calibrated_with = 0      # 계산에 사용된 마커 개수
        self.reproj_error = float('inf')  # 평균 재투영 오차 (cm)
        self.locked = False           # 품질 기준을 만족해 고정되었는지
        self._warned = set()          # 중복 경고 억제용

        # 보정에 실제로 쓴 기둥의 이미지 좌표 {마커ID: (x_px, y_px)}.
        self.marker_image_pos = {}
        # 기둥별 재투영 오차 {마커ID: cm}.
        self.marker_residuals = {}

        # --- 픽셀 기반 매핑 데이터 ---
        # 사용자가 클릭한 기둥 픽셀 좌표 {마커ID: (x_px, y_px)}
        self.pillar_pixels = {}
        # 기둥 사이 보간으로 계산된 자리 픽셀 좌표 {구역ID: (x_px, y_px)}
        self.spot_pixels = {}
        # 입출구 픽셀 좌표 (gate1_pixel, gate2_pixel)
        self.gate_pixels = (None, None)
        # 픽셀 -> cm 변환 캐시. pixel_to_cm이 채운다.
        self._px_cm_H = None
        self._px_cm_key = None

        print(f"[INFO] 마커 매퍼 초기화 완료. ("
              f"등록된 마커 {len(self.marker_world_pos)}개)")




    # --- 호모그래피 저장/복원 -----------------------------------------------
    # (_lens_calibrated는 이 클래스 밖, 모듈 끝에 정의되어 있다)

    def save_homography(self, path=None):
        """
        확정된 호모그래피를 파일로 저장한다.

        카메라가 고정 설치되어 있으면 이 행렬은 변하지 않으므로, 다음 실행부터는
        마커를 다시 찾지 않아도 된다. 마커가 하나도 안 보여도 좌표 변환이 된다.
        """
        if self.H is None:
            return False
        path = CONFIG['HOMOGRAPHY_CACHE'] if path is None else path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ids = sorted(self.marker_world_pos)
            np.savez(
                path,
                H=self.H,
                markers=self.calibrated_with,
                error=self.reproj_error,
                # 배치가 바뀌면 옛 행렬을 쓰면 안 되므로 함께 남긴다.
                # ID만으로는 부족하다. 격자 열 배분을 바꾸면 ID는 그대로인데
                # 실좌표만 달라지므로, 좌표까지 저장해 두고 대조해야 한다.
                marker_ids=np.array(ids, dtype=np.int32),
                marker_world=np.array([self.marker_world_pos[i] for i in ids],
                                      dtype=np.float64),
            )
            print(f"[INFO] 호모그래피를 저장했습니다. {path}")
            print(f"       다음 실행부터는 마커가 안 보여도 이 값을 씁니다. "
                  f"카메라를 옮기면 /recalibrate 하거나 이 파일을 지우세요.")
            return True
        except OSError as e:
            print(f"[경고] 호모그래피 저장 실패: {e}")
            return False

    def set_homography_from_points(self, image_points):
        """
        마커 검출 없이, 지정된 기둥 이미지 좌표로 호모그래피를 직접 계산한다.

        마커가 작거나 흐려서 자동 검출이 잘 안 되어도, 사람이 화면에서 기둥
        위치를 찍어주면 좌표계를 확정할 수 있다. 카메라가 고정 설치되어 있으면
        한 번만 하면 되고, 결과는 캐시에 저장되어 다음 실행부터 재사용된다.

        마커의 '무늬'는 못 읽어도 '어느 기둥인지'만 알면 대응점으로 충분하다.
        기둥의 실좌표는 격자에서 이미 알고 있기 때문이다.

        Args:
            image_points: {마커ID: (x_px, y_px)} 사람이 지정한 기둥 이미지 좌표.
                          4개 이상이어야 하고, 많을수록 정확하다.

        Returns:
            (성공 여부, 메시지)
        """
        img_pts, world_pts, used = [], [], []
        for marker_id, pt in image_points.items():
            world_pt = self.marker_world_pos.get(int(marker_id))
            if world_pt is None:
                continue
            img_pts.append((float(pt[0]), float(pt[1])))
            world_pts.append(world_pt)
            used.append(int(marker_id))

        if len(img_pts) < 4:
            return False, f"기둥이 {len(img_pts)}개뿐입니다. 최소 4개가 필요합니다."

        spread = self._spread_ratio(img_pts)
        if spread < self.min_spread:
            return False, (f"찍은 점들이 한 줄에 몰려 있습니다 (퍼짐 {spread:.4f}). "
                           f"위아래로 골고루 찍어야 합니다.")

        H, _ = cv2.findHomography(
            np.array(img_pts, np.float32), np.array(world_pts, np.float32),
            cv2.RANSAC if len(img_pts) > 4 else 0, self.ransac_thresh_cm)
        if H is None:
            return False, "호모그래피 계산에 실패했습니다. 점 위치를 확인하세요."

        error = self._reprojection_error(H, img_pts, world_pts)
        if error > self.max_error:
            print(f"[경고] 수동 보정 재투영 오차가 큽니다 ({error:.1f}cm > {self.max_error}cm). "
                  f"하지만 수동 클릭을 최우선 적용하여 호모그래피를 강제 확정합니다.")

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.calibrated_with = len(img_pts)
        self.reproj_error = error
        self.locked = True

        # 찍은 위치와 기둥별 오차를 남긴다.
        self.marker_image_pos = {mid: pt for mid, pt in zip(used, img_pts)}
        per_point = self._per_point_error(H, img_pts, world_pts)
        self.marker_residuals = {mid: float(e) for mid, e in zip(used, per_point)}
        self._report_residuals()

        msg = f"수동 보정 완료. 기둥 {len(img_pts)}개 {sorted(used)}, 오차 {error:.2f}cm"
        if len(img_pts) == 4:
            # 4점은 항상 오차 0으로 맞춰진다. 즉 이 숫자로는 옳은지 알 수 없다.
            msg += ("  (주의: 4점은 오차가 항상 0으로 나와 검증이 불가능합니다. "
                    "5개 이상 찍으면 잘못 찍은 것을 오차로 걸러낼 수 있습니다)")
        print(f"[INFO] {msg}")
        if CONFIG['USE_HOMOGRAPHY_CACHE']:
            self.save_homography()
        return True, msg

    def load_homography(self, path=None):
        """저장해 둔 호모그래피를 불러온다. 배치가 바뀌었으면 무시한다."""
        path = CONFIG['HOMOGRAPHY_CACHE'] if path is None else path
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            saved_ids = set(int(i) for i in data['marker_ids'])
        except Exception as e:
            print(f"[경고] 호모그래피 파일을 읽을 수 없습니다: {e}")
            return False

        if saved_ids != set(self.marker_world_pos):
            print(f"[경고] 저장된 호모그래피는 다른 마커 배치로 만든 것입니다. 무시합니다.")
            print(f"       저장됨: {sorted(saved_ids)}")
            print(f"       현재  : {sorted(self.marker_world_pos)}")
            return False

        # 기둥의 실좌표가 바뀌었는지 확인한다.
        # 격자 열 배분을 고치면 마커 ID는 그대로인데 실좌표만 달라진다.
        # 이걸 안 보면 옛 행렬을 그대로 불러와 좌표가 통째로 어긋난다.
        saved_world = data.get('marker_world') if hasattr(data, 'get') else None
        if saved_world is None and 'marker_world' in getattr(data, 'files', []):
            saved_world = data['marker_world']
        ids = sorted(self.marker_world_pos)
        current_world = np.array([self.marker_world_pos[i] for i in ids], dtype=np.float64)

        if saved_world is None:
            print("[경고] 저장된 호모그래피에 기둥 실좌표가 없습니다. (구버전 파일)")
            print("       격자가 그때와 같은지 확인할 수 없으므로 무시합니다. 다시 보정하세요.")
            return False
        if not np.allclose(np.asarray(saved_world, dtype=np.float64), current_world, atol=0.1):
            print("[경고] 저장된 호모그래피는 지금과 다른 격자 배치로 만든 것입니다. 무시합니다.")
            print("       (grid_map의 열 배분이나 CELL_W_CM / CELL_H_CM이 바뀌었습니다)")
            print("       /calibrate 에서 다시 잡으세요.")
            return False



        self.H = data['H']
        self.H_inv = np.linalg.inv(self.H)
        self.calibrated_with = int(data['markers'])
        self.reproj_error = float(data['error'])
        self.locked = bool(self.lock_homography)

        print(f"[INFO] 저장된 호모그래피를 불러왔습니다. "
              f"(마커 {self.calibrated_with}개, 오차 {self.reproj_error:.2f}cm)")
        print(f"       카메라를 옮겼다면 /recalibrate로 다시 잡으세요.")
        return True

    @staticmethod
    def _spread_ratio(points):
        """
        점들이 얼마나 넓게 퍼져 있는지(일직선에 가깝지 않은지) 측정.

        공분산 행렬의 최소/최대 고윳값 비율을 반환한다. 0에 가까울수록
        한 줄에 몰려 있다는 뜻이며, 그런 배치로 만든 호모그래피는 그 줄에서
        떨어진 지점의 좌표가 폭주한다.
        """
        p = np.asarray(points, dtype=np.float64)
        centered = p - p.mean(axis=0)
        eig = np.linalg.eigvalsh(centered.T @ centered)
        if eig[1] <= 0:
            return 0.0
        return float(eig[0] / eig[1])

    def _reprojection_error(self, H, img_pts, world_pts):
        """
        주어진 호모그래피로 마커를 실좌표로 되돌렸을 때의 평균 오차(cm).

        주의: 계산에 사용한 점이 정확히 4개면 그 4점은 항상 오차 0으로
        맞춰지므로, 이 값만으로는 잘못된 대응을 걸러낼 수 없다.
        그래서 채택 조건에 마커 개수(_lock_markers)를 함께 둔다.
        """
        src = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        truth = np.array(world_pts, dtype=np.float32)
        return float(np.mean(np.linalg.norm(dst - truth, axis=1)))

    def _report_residuals(self):
        """
        기둥별 재투영 오차를 표로 출력한다.

        평균 오차는 한 기둥이 크게 틀어져도 묻힌다. 화면에서 특정 구역만
        어긋나 보일 때, 이 표에서 유독 큰 값을 가진 기둥이 범인이다.

        큰 값이 나오는 원인은 대개 셋 중 하나다.
          1. 그 기둥을 잘못 찍었다 (다른 기둥을 찍었거나 순서가 밀렸다)
          2. PILL_MARKER_ID의 마커 ID <-> 격자 칸 대응이 실제와 다르다
          3. 렌즈 왜곡. 호모그래피는 직선을 직선으로만 보내므로 화면 가장자리의
             휘어짐을 표현하지 못한다. 가장자리 기둥만 크게 나오면 이쪽이다.
        """
        if not self.marker_residuals:
            return

        ordered = sorted(self.marker_residuals.items(), key=lambda kv: -kv[1])
        worst_id, worst = ordered[0]
        mean = sum(self.marker_residuals.values()) / len(self.marker_residuals)

        print(f"[보정] 기둥별 재투영 오차 (평균 {mean:.2f}cm / 최대 {worst:.2f}cm)")
        for marker_id, err in sorted(self.marker_residuals.items()):
            bar = '#' * min(int(err * 4), 30)
            flag = '  <-- 확인 필요' if err > mean * 2 and err > 1.0 else ''
            print(f"       마커 {marker_id:2d}  {err:5.2f}cm  {bar}{flag}")

        # 한 기둥만 유독 크면 평균이 기준을 통과해도 그 근처 자리는 어긋난다.
        if worst > mean * 2 and worst > 1.0:
            print(f"[경고] 마커 {worst_id}번이 평균의 2배 넘게 틀어져 있습니다 "
                  f"({worst:.2f}cm). 그 기둥을 잘못 찍었거나 PILL_MARKER_ID의 "
                  f"대응이 실제와 다를 수 있습니다.")

    def observed_marker_world(self):
        """
        보정 때 찍은 기둥의 이미지 좌표를 실좌표로 되돌린 값.

        등록된 격자에서 계산한 marker_world_pos와 달리, 이쪽은 '실제로 화면에서
        본 위치'다. 호모그래피가 완벽하면 둘은 같고, 어긋난 만큼 차이가 난다.

        Returns:
            {마커ID: (x_cm, y_cm)}. 보정 정보가 없으면 빈 dict.
        """
        if self.H is None or not self.marker_image_pos:
            return {}
        ids = list(self.marker_image_pos)
        pts = np.array([self.marker_image_pos[i] for i in ids],
                       dtype=np.float32).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return {mid: (float(w[0]), float(w[1])) for mid, w in zip(ids, world)}

    @staticmethod
    def _per_point_error(H, img_pts, world_pts):
        """점마다의 재투영 오차(cm) 배열. 어느 기둥이 범인인지 찾는 용도."""
        src = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        truth = np.array(world_pts, dtype=np.float32)
        return np.linalg.norm(dst - truth, axis=1)

    def _warn_once(self, key, message):
        """같은 경고가 매 프레임 쏟아지지 않도록 한 번만 출력."""
        if key in self._warned:
            return
        self._warned.add(key)
        print(message)

    def is_ready(self):
        """좌표 변환이 가능한 상태인지 확인."""
        # 픽셀 기반 모드: 기둥 픽셀이 설정되었으면 준비 완료
        if self.pillar_pixels:
            return True
        return self.H is not None

    def image_to_world(self, point):
        """
        이미지 좌표를 주차장 실좌표(cm)로 변환.

        Args:
            point: (x, y) 이미지 좌표

        Returns:
            (x_cm, y_cm) 실좌표. 호모그래피가 없으면 None.
        """
        if self.H is None:
            return None
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def world_to_image(self, point):
        """
        주차장 실좌표(cm)를 이미지 좌표로 변환. (경로 시각화용)

        Args:
            point: (x_cm, y_cm) 실좌표

        Returns:
            (x, y) 이미지 좌표. 호모그래피가 없으면 None.
        """
        if self.H_inv is None:
            return None
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H_inv)
        return int(dst[0][0][0]), int(dst[0][0][1])

    # --- 픽셀 기반 매핑 (호모그래피 없이 동작) --------------------------------
    # 기둥 픽셀 좌표만으로 자리 위치를 보간하고 차량 위치를 판단한다.
    # cm 좌표 변환 없이 이미지 픽셀 거리로 직접 비교한다.

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
        호모그래피를 초기화. (카메라를 다시 설치했을 때 사용)

        저장된 캐시 파일도 함께 지운다. 지우지 않으면 다음 실행에서 옛 행렬을
        다시 불러와 카메라를 옮긴 것이 반영되지 않는다.
        """
        self.H = None
        self.H_inv = None
        self.calibrated_with = 0
        self.reproj_error = float('inf')
        self.locked = False
        self.pillar_pixels = {}
        self.spot_pixels = {}
        self.gate_pixels = (None, None)
        self._px_cm_H = None
        self._px_cm_key = None
        self._warned.clear()

        if clear_cache:
            path = CONFIG['HOMOGRAPHY_CACHE']
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[INFO] 저장된 호모그래피를 삭제했습니다. {path}")
                except OSError as e:
                    print(f"[경고] 캐시 파일을 지우지 못했습니다: {e}")
            # 기둥 픽셀 캐시도 삭제
            px_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'pillar_pixels.json')
            if os.path.exists(px_path):
                try:
                    os.remove(px_path)
                except OSError:
                    pass

        print("[INFO] 보정을 초기화했습니다. 재보정이 필요합니다.")

    # --- 역투영 오버레이 -----------------------------------------------------

    def _cells_to_image(self, cells):
        """
        격자 칸들을 감싸는 사각형을 이미지 좌표 폴리곤으로 바꾼다.

        칸 하나든 여러 칸이든(대형 구역은 2칸) 하나의 사각형으로 묶어 그린다.

        Args:
            cells: [(row, col), ...] 같은 열에 붙어 있는 칸들

        Returns:
            (4, 2) int32 폴리곤. 변환할 수 없으면 None.
        """
        if self.H_inv is None or not cells:
            return None

        hw = C02_CONFIG['CELL_W_CM'] / 2.0
        hh = C02_CONFIG['CELL_H_CM'] / 2.0
        xs, ys = [], []
        for row, col in cells:
            cx, cy = cell_to_world((row, col))
            xs += [cx - hw, cx + hw]
            ys += [cy - hh, cy + hh]

        corners = [(min(xs), min(ys)), (max(xs), min(ys)),
                   (max(xs), max(ys)), (min(xs), max(ys))]
        pts = cv2.perspectiveTransform(
            np.array([corners], dtype=np.float32), self.H_inv)[0]
        return pts.astype(np.int32)

    def draw_layout_overlay(self, frame, show_grid=True, show_spots=True,
                            show_pillars=True, show_gates=True, show_labels=True):
        """
        등록된 배치(기둥/자리/입출구)를 화면에 오버레이한다.

        픽셀 기반 모드에서는 기둥/자리 픽셀 좌표를 직접 사용한다.
        호모그래피 모드에서는 기존처럼 역투영으로 그린다.

        Args:
            frame:        그릴 대상 프레임 (원본이 수정됨)
            show_grid:    (픽셀 모드에서는 무시)
            show_spots:   주차 구역 (종류별 색)
            show_pillars: 기둥
            show_gates:   입출구
            show_labels:  구역 ID 등 글자

        Returns:
            frame (입력과 동일 객체)
        """
        # 픽셀 기반 모드: 기둥과 자리 픽셀 좌표를 직접 사용
        if self.pillar_pixels:
            return self._draw_layout_pixel(frame, show_spots, show_pillars,
                                           show_gates, show_labels)

        # 기존 호모그래피 모드 (fallback)
        if self.H_inv is None:
            return frame

        rows, cols = get_rows(), get_cols()

        if show_grid:
            for row in range(rows):
                for col in range(cols):
                    poly = self._cells_to_image([(row, col)])
                    if poly is not None:
                        cv2.polylines(frame, [poly], True, COLOR_OVERLAY_EDGE, 1, cv2.LINE_AA)
            outline = self._cells_to_image(
                [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)])
            if outline is not None:
                cv2.polylines(frame, [outline], True, COLOR_OVERLAY_BOUND, 2, cv2.LINE_AA)

        if show_spots:
            for spot_id, cells in spot_map.items():
                poly = self._cells_to_image(cells)
                if poly is None:
                    continue
                color = COLOR_OVERLAY_SPOT.get(spot_type.get(spot_id), (200, 200, 200))
                cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
                if show_labels:
                    cx = int(poly[:, 0].mean())
                    cy = int(poly[:, 1].mean())
                    cv2.putText(frame, spot_id, (cx - 16, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if show_pillars:
            for marker_id, (wx, wy) in self.marker_world_pos.items():
                pt = cv2.perspectiveTransform(
                    np.array([[[float(wx), float(wy)]]], np.float32), self.H_inv)[0][0]
                cx, cy = int(pt[0]), int(pt[1])
                cv2.drawMarker(frame, (cx, cy), COLOR_OVERLAY_PILL,
                               cv2.MARKER_CROSS, 16, 2)
                if show_labels:
                    cv2.putText(frame, str(marker_id), (cx + 9, cy - 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                COLOR_OVERLAY_PILL, 2, cv2.LINE_AA)

        if show_gates:
            for pos, label, color in (
                (GATE1_POS, "IN", COLOR_OVERLAY_GATE1),
                (GATE2_POS, "OUT", COLOR_OVERLAY_GATE2),
            ):
                if pos is None:
                    continue
                poly = self._cells_to_image([pos])
                if poly is None:
                    continue
                cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
                if show_labels:
                    cv2.putText(frame, label,
                                (int(poly[:, 0].mean()) - 14, int(poly[:, 1].mean()) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        return frame

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

    def draw_markers(self, frame, markers):
        """
        검출된 마커(기둥)의 위치와 ID를 프레임에 표시.
        호모그래피가 고정(LOCKED)된 상태라면, 현재 화면에 보이지 않는 기둥들도
        변환 행렬을 역산하여 이론적인(수동으로 픽스된) 위치에 계속 파란색으로 그려준다.
        """
        import numpy as np

        # 1. LOCKED 상태라면, 모든 기둥의 '이론적(가상) 위치'를 계산해서 먼저 그린다.
        if self.locked and self.H_inv is not None:
            # H_inv는 [월드 -> 이미지] 변환 행렬
            world_pts = []
            m_ids = []
            for m_id, (wx, wy) in self.marker_world_pos.items():
                world_pts.append([wx, wy])
                m_ids.append(m_id)
            
            if world_pts:
                world_pts = np.array([world_pts], dtype=np.float32)
                img_pts = cv2.perspectiveTransform(world_pts, self.H_inv)[0]
                
                for i, (cx, cy) in enumerate(img_pts):
                    m_id = m_ids[i]
                    cx, cy = int(cx), int(cy)
                    # 수동으로 찍어둔 가상의 마커 위치를 그려줌 (cyan-ish blue)
                    cv2.circle(frame, (cx, cy), 6, COLOR_MARKER, -1)
                    cv2.putText(frame, str(m_id), (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MARKER, 2, cv2.LINE_AA)

        # 2. 실시간으로 검출된(물리적으로 무늬를 읽어낸) 마커들은 좀 더 굵게 또는 겹쳐 그린다.
        # (LOCKED 상태에서는 1번에서 이미 그렸으므로 위치가 거의 겹침)
        for marker_id, (cx, cy) in markers.items():
            cx, cy = int(cx), int(cy)
            known = marker_id in self.marker_world_pos
            if not known:
                # 모르는 마커(오인식)는 빨간색
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"{marker_id}?", (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            elif not self.locked:
                # 아직 잠기기 전이면 실시간 검출 마커만 그림
                cv2.circle(frame, (cx, cy), 6, COLOR_MARKER, -1)
                cv2.putText(frame, str(marker_id), (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MARKER, 2, cv2.LINE_AA)

        return frame


# 차량 위치 추정 및 경로 안내
class ParkingNavigator:
    """
    B02_car_mot의 추적 결과를 받아 각 차량의 실시간 실좌표를 계산하고,
    배정된 주차 구역까지의 경로를 안내한다.

    이 클래스는 검출/추적/경로계산을 직접 하지 않는다. B02가 만든 트랙의
    박스 정중앙점을 입력으로 받고, 경로는 C01_path_planner에 맡긴다.

    동작 흐름:
      1. MarkerMapper가 ArUco 마커로 호모그래피를 계산한다.
      2. 각 차량의 박스 정중앙점을 실좌표(cm)로 변환한다.
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
            mapper:               MarkerMapper 인스턴스 (None이면 기본 설정으로 생성)
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

        self.mapper = mapper if mapper is not None else MarkerMapper()

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
        # 가장 최근 프레임에서 검출된 마커 {마커ID: (cx, cy)} (시각화 재사용용)
        self.latest_markers = {}

        print(f"[INFO] 내비게이터 초기화 완료. (주차 구역 {len(self.spot_world_pos)}개 등록)")

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
            return self.spot_pixels.get(spot_id)
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
            center = self.spot_pixels.get(spot_id)
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

    def sync_targets_from_parking_manager(self):
        """
        A01_parking_manager가 관리하는 입차 정보(cars_info)를 읽어
        각 차량의 목표 구역을 자동으로 동기화.
        """
        from data.car_data import cars_info

        for car_id, info in cars_info.items():
            spot_id = info.get("spot_id")
            if spot_id and self.targets.get(car_id) != spot_id:
                self.set_target(car_id, spot_id)

    def update(self, frame, tracks):
        """
        한 프레임 분량의 추적 결과로 각 차량의 위치와 안내 정보를 갱신.
        """
        results = []
        if not self.mapper.is_ready():
            return results

        is_pixel_mode = bool(self.mapper.pillar_pixels)

        for trk in tracks:
            car_id = trk.get("car_id")
            image_pos = trk["center"]

            if is_pixel_mode:
                pos = image_pos
            else:
                pos = self.mapper.image_to_world(image_pos)
                if pos is None:
                    continue

            # 위치 이력 갱신 (픽셀 모드면 픽셀, 호모그래피 모드면 cm)
            key = car_id if car_id else f"track_{trk['track_id']}"
            history = self.world_history.setdefault(
                key, deque(maxlen=self.history_maxlen)
            )
            history.append(pos)

            heading = self._compute_heading(history)
            target_spot = self.targets.get(car_id) if car_id else None
            
            if target_spot:
                if is_pixel_mode:
                    target_pos = self.spot_pixels.get(target_spot)
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
                    # 픽셀 모드: 경로 탐색 없이 직선 거리로 안내
                    distance = self._distance(pos, target_pos)
                    # cm 단위로 보정해서 넘겨준다 (임시)
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

            results.append({
                "track_id": trk["track_id"],
                "car_id": car_id,
                "image_pos": image_pos,
                "world_pos": pos,
                "heading_deg": heading,
                "target_spot": target_spot,
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
        is_pixel_mode = bool(self.mapper.pillar_pixels)

        for nav in nav_results:
            cx, cy = int(nav["image_pos"][0]), int(nav["image_pos"][1])
            wx, wy = nav["world_pos"]

            # 좌표 표시
            if is_pixel_mode:
                cv2.putText(frame, f"({cx},{cy})px", (cx - 40, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PATH, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"({wx:.0f},{wy:.0f})cm", (cx - 40, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PATH, 2, cv2.LINE_AA)

            if not draw_target_line or nav["target_world"] is None:
                continue

            # 경로를 따라 안내선을 그린다.
            route = nav.get("route")
            if route and not is_pixel_mode:
                pts = [self.mapper.world_to_image(p) for p in route]
                pts = [p for p in pts if p is not None]
                # 남은 구간만 강조하기 위해 현재 위치에서 이어서 그린다
                if len(pts) >= 2:
                    idx = min(nav.get("route_index", 1), len(pts) - 1)
                    remaining = [(cx, cy)] + pts[idx:]
                    for a, b in zip(remaining[:-1], remaining[1:]):
                        cv2.line(frame, a, b, COLOR_PATH, 2, cv2.LINE_AA)
                    for p in pts[idx:-1]:
                        cv2.circle(frame, p, 4, COLOR_PATH, -1)
                    cv2.arrowedLine(frame, remaining[-2], remaining[-1],
                                    COLOR_PATH, 2, tipLength=0.15)

            if is_pixel_mode:
                target_img = (int(nav["target_world"][0]), int(nav["target_world"][1]))
            else:
                target_img = self.mapper.world_to_image(nav["target_world"])
                
            if target_img is not None:
                if not route or is_pixel_mode:
                    cv2.arrowedLine(frame, (cx, cy), target_img, COLOR_PATH, 2, tipLength=0.05)
                cv2.circle(frame, target_img, 8, COLOR_TARGET, 2)

            # 안내 문구
            dist = nav["distance_cm"]
            if dist is not None:
                label = f"{nav['target_spot']} {nav['guide']} {dist:.0f}"
                label += "px" if is_pixel_mode else "cm"
                cv2.putText(frame, label, (cx - 40, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TARGET, 2, cv2.LINE_AA)

        return frame


def build_test_board(px_per_cm=8.0, margin_px=90, marker_px=64, tilt=0.06):
    """
    MARKER_WORLD_POS의 배치대로 ArUco 마커를 배치한 테스트용 이미지를 생성.

    외부 사진 파일 없이도 마커 검출과 좌표 변환 정확도를 검증할 수 있다.
    카메라가 비스듬히 내려다보는 상황을 재현하기 위해 원근 왜곡을 적용한다.

    Args:
        px_per_cm:  1cm를 몇 픽셀로 그릴지
        margin_px:  이미지 가장자리 여백
        marker_px:  마커 한 변의 크기 (픽셀)
        tilt:       원근 왜곡 강도 (0이면 왜곡 없음)

    Returns:
        생성된 BGR 이미지 (numpy array)
    """
    dict_id = getattr(aruco, CONFIG['ARUCO_DICT'])
    aruco_dict = aruco.getPredefinedDictionary(dict_id)

    xs = [p[0] for p in MARKER_WORLD_POS.values()]
    ys = [p[1] for p in MARKER_WORLD_POS.values()]
    min_x, min_y = min(xs), min(ys)

    width = int((max(xs) - min_x) * px_per_cm) + margin_px * 2
    height = int((max(ys) - min_y) * px_per_cm) + margin_px * 2
    board = np.full((height, width, 3), 235, dtype=np.uint8)

    half = marker_px // 2
    for marker_id, (wx, wy) in MARKER_WORLD_POS.items():
        cx = int((wx - min_x) * px_per_cm) + margin_px
        cy = int((wy - min_y) * px_per_cm) + margin_px

        img = aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        board[cy - half:cy + half, cx - half:cx + half] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if tilt <= 0:
        return board

    # 위쪽이 좁아지는 사다리꼴로 왜곡 (비스듬히 설치된 카메라 재현)
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = np.float32([
        [width * tilt, height * tilt * 0.5],
        [width * (1 - tilt), height * tilt * 0.5],
        [width, height],
        [0, height],
    ])
    warp = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board, warp, (width, height), borderValue=(235, 235, 235))


# =====================================================================
# 테스트용 메인 (단독 실행 시 마커 검출 및 좌표 변환 정확도 검증)
# =====================================================================
# 카메라 없이 동작한다. 외부 이미지가 없으면 마커 보드를 직접 생성해서 쓰므로
# 어떤 환경에서든 그대로 실행된다.
# 카메라 + 검출 + 추적 + 내비게이션 통합 실행은 C_main.py를 사용할 것.
if __name__ == '__main__':
    print("==========================================")
    print(" C00 : 주차장 내비게이션 (ArUco + Homography)")
    print(" 단독 테스트 : 좌표 변환 정확도 검증")
    print("==========================================")

    # MARKER_WORLD_POS 배치대로 마커 보드를 생성해서 검증한다.
    # 외부 사진을 쓰지 않는 이유: 사진 속 마커 배치가 현재 MARKER_WORLD_POS와
    # 다르면(마커를 재배치한 경우) 테스트가 실패하는데, 그건 코드 문제가 아니라
    # 촬영 시점의 배치가 다른 것뿐이라 원인 파악만 어려워진다.
    frame = build_test_board()
    print(f"[INFO] MARKER_WORLD_POS 배치로 마커 보드를 생성했습니다. {frame.shape}")

    mapper = MarkerMapper(
        aruco_dict_name=CONFIG['ARUCO_DICT'],
        min_markers=CONFIG['MIN_MARKERS_FOR_HOMOGRAPHY'],
        lock_homography=CONFIG['LOCK_HOMOGRAPHY'],
        lock_markers=CONFIG['MARKERS_FOR_LOCK'],
        max_error=CONFIG['MAX_REPROJ_ERROR_CM'],
        min_spread=CONFIG['MIN_MARKER_SPREAD'],
        ransac_thresh_cm=CONFIG['RANSAC_THRESH_CM']
    )

    # 1) 마커 검출
    markers = mapper.detect_markers(frame)
    print(f"\n[TEST] 검출된 마커 {len(markers)}개: {sorted(markers.keys())}")
    for marker_id in sorted(markers):
        cx, cy = markers[marker_id]
        print(f"  ID {marker_id}: 이미지 좌표 ({cx:7.1f}, {cy:7.1f})")

    # 2) 호모그래피 계산
    if not mapper.update_homography(markers):
        print("[ERROR] 호모그래피를 계산할 수 없습니다. 마커가 충분히 보이는지 확인하세요.")
        sys.exit(1)

    # 3) 역변환 정확도 검증
    #    검출된 마커를 실좌표로 되돌렸을 때 등록값과 얼마나 일치하는지 확인한다.
    print(f"\n[TEST] 좌표 변환 정확도 검증 (등록값 대비 오차)")
    print(f"{'ID':>4} {'등록 좌표(cm)':>18} {'변환 결과(cm)':>18} {'오차(cm)':>10}")
    errors = []
    for marker_id in sorted(markers):
        expected = MARKER_WORLD_POS.get(marker_id)
        if expected is None:
            continue
        actual = mapper.image_to_world(markers[marker_id])
        err = math.hypot(actual[0] - expected[0], actual[1] - expected[1])
        errors.append(err)
        print(f"{marker_id:>4} {str(expected):>18} "
              f"{f'({actual[0]:.1f}, {actual[1]:.1f})':>18} {err:>10.2f}")

    if errors:
        print(f"\n  평균 오차: {sum(errors)/len(errors):.2f} cm | 최대 오차: {max(errors):.2f} cm")

    # 4) 내비게이션 안내 시뮬레이션
    #    차량이 화면 오른쪽에서 왼쪽 아래로 이동하는 상황을 합성 트랙으로 재현
    print(f"\n[TEST] 내비게이션 안내 시뮬레이션")
    navigator = ParkingNavigator(
        mapper=mapper,
        arrival_threshold=CONFIG['ARRIVAL_THRESHOLD_CM'],
        turn_threshold=CONFIG['TURN_ANGLE_THRESHOLD_DEG'],
        uturn_threshold=CONFIG['UTURN_ANGLE_THRESHOLD_DEG'],
        min_move_for_heading=CONFIG['MIN_MOVE_CM_FOR_HEADING'],
        heading_window=CONFIG['HEADING_WINDOW']
    )
    navigator.set_target("1234", "B-1")

    # 차량이 오른쪽에서 왼쪽으로 이동하는 상황을 합성 트랙으로 재현.
    # 시작 위치를 프레임 크기 기준으로 잡아 이미지 종류와 무관하게 동작한다.
    h, w = frame.shape[:2]
    start_x, start_y = int(w * 0.70), int(h * 0.60)
    for step in range(6):
        fake_track = {
            "track_id": 12,
            "car_id": "1234",
            "center": (start_x - step * int(w * 0.03), start_y + step * int(h * 0.025)),
        }
        nav_results = navigator.update(frame, [fake_track])
        if not nav_results:
            continue
        nav = nav_results[0]
        heading_str = f"{nav['heading_deg']:6.1f}도" if nav['heading_deg'] is not None else "  탐색중"
        dist_str = f"{nav['distance_cm']:6.1f}cm" if nav['distance_cm'] is not None else "     -"
        print(f"  step {step}: 실좌표=({nav['world_pos'][0]:6.1f}, {nav['world_pos'][1]:6.1f})cm "
              f"| 방향={heading_str} | 목표까지={dist_str} | 안내={nav['guide_text']} "
              f"| 최근접={nav['nearest_spot']}")

    print(f"\n[TEST] 완료. MARKER_WORLD_POS를 실제 측정값으로 수정해야 정확한 좌표가 나옵니다.")
