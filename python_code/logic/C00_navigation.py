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
from logic.B03_marker_detect import MarkerDetector

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
)

# 시각화 색상 (BGR)
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

        self.detector = MarkerDetector()
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

        print(f"[INFO] ArUco 마커 검출기 초기화 완료. ("
              f"등록된 마커 {len(self.marker_world_pos)}개)")

        # 카메라가 고정이면 호모그래피는 상수다. 저장해 둔 것이 있으면 불러 쓴다.
        if CONFIG['USE_HOMOGRAPHY_CACHE']:
            self.load_homography()

    # --- 호모그래피 저장/복원 -----------------------------------------------

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
            np.savez(
                path,
                H=self.H,
                markers=self.calibrated_with,
                error=self.reproj_error,
                # 배치가 바뀌면 옛 행렬을 쓰면 안 되므로 함께 남긴다
                marker_ids=np.array(sorted(self.marker_world_pos), dtype=np.int32),
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
            return False, (f"재투영 오차가 큽니다 ({error:.1f}cm > {self.max_error}cm). "
                           f"기둥 순서를 잘못 찍었거나, C02의 CELL_W_CM/CELL_H_CM이 "
                           f"실제 목업 치수와 다릅니다.")

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.calibrated_with = len(img_pts)
        self.reproj_error = error
        self.locked = True

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

        self.H = data['H']
        self.H_inv = np.linalg.inv(self.H)
        self.calibrated_with = int(data['markers'])
        self.reproj_error = float(data['error'])
        self.locked = bool(self.lock_homography)

        print(f"[INFO] 저장된 호모그래피를 불러왔습니다. "
              f"(마커 {self.calibrated_with}개, 오차 {self.reproj_error:.2f}cm)")
        print(f"       카메라를 옮겼다면 /recalibrate로 다시 잡으세요.")
        return True

    def detect_markers(self, frame):
        """
        프레임에서 ArUco 마커를 검출.

        Args:
            frame: OpenCV BGR 이미지 (numpy array)

        Returns:
            {마커ID: (cx, cy)} 형태의 딕셔너리. cx, cy는 마커 중심의 이미지 좌표.
        """
        return self.detector.detect(frame)

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

    def update_homography(self, markers):
        """
        검출된 마커로 이미지 -> 실좌표 변환 행렬을 계산.

        첫 성공을 그대로 고정하지 않고, 더 많은 마커로 더 정확한 해가
        나오면 갈아탄다. 충분한 품질(마커 수와 재투영 오차)에 도달했을 때만
        고정하며, 그 전까지는 매 프레임 다시 시도한다.

        이렇게 하는 이유: 마커가 정확히 4개면 대응이 틀려 있어도 그 4점에는
        항상 오차 0으로 맞춰진다. 즉 4점 해는 검증 자체가 불가능하므로,
        최소 개수로 성급히 고정하면 잘못된 좌표계가 영구히 굳어버린다.

        Args:
            markers: detect_markers()의 반환 결과 {마커ID: (cx, cy)}

        Returns:
            호모그래피가 유효하면 True, 아니면 False.
        """
        # 품질 기준을 만족해 고정된 상태면 재계산하지 않음
        # (차량이 마커를 가려도 좌표 추정이 끊기지 않도록)
        if self.locked:
            return True

        # 실좌표가 등록된 마커만 사용
        img_pts, world_pts, unknown = [], [], []
        for marker_id, img_pt in markers.items():
            world_pt = self.marker_world_pos.get(marker_id)
            if world_pt is None:
                unknown.append(marker_id)
                continue
            img_pts.append(img_pt)
            world_pts.append(world_pt)

        # 검출은 됐는데 좌표표에 없는 마커를 알려준다.
        # 이게 있으면 화면에는 마커가 여러 개 보이는데 'N/6 markers'는 안 올라가는,
        # 원인을 찾기 어려운 상태가 된다. (화면에는 빨간 점 + "11?"로 표시된다)
        if unknown:
            self._warn_once(
                "unknown_markers",
                f"[경고] 검출됐지만 좌표가 등록되지 않은 마커: {sorted(unknown)}\n"
                f"        data/map_data.py의 PILL_MARKER_ID에 없는 ID입니다. "
                f"호모그래피에 쓰이지 않습니다.\n"
                f"        현재 등록된 ID: {sorted(self.marker_world_pos)}\n"
                f"        실제 기둥에 붙인 마커와 표가 일치하는지 확인하세요."
            )

        n = len(img_pts)
        if n < self.min_markers:
            return self.H is not None

        # 마커가 한 줄에 몰려 있으면 그 줄 밖의 좌표를 신뢰할 수 없다
        spread = self._spread_ratio(img_pts)
        if spread < self.min_spread:
            self._warn_once("spread",
                            f"[경고] 마커가 한 줄에 몰려 있어 호모그래피를 건너뜁니다. "
                            f"(퍼짐 정도 {spread:.4f} < {self.min_spread}) "
                            f"윗줄과 아랫줄 마커가 함께 보이도록 하세요.")
            return self.H is not None

        img_arr = np.array(img_pts, dtype=np.float32)
        world_arr = np.array(world_pts, dtype=np.float32)

        # 마커가 4개를 넘으면 RANSAC으로 이상치(ID 오인식 등)를 배제
        method = cv2.RANSAC if n > 4 else 0
        H, _ = cv2.findHomography(img_arr, world_arr, method, self.ransac_thresh_cm)
        if H is None:
            return self.H is not None

        error = self._reprojection_error(H, img_pts, world_pts)

        # 오차가 큰 해는 채택하지 않음 (마커 배치와 등록 좌표가 어긋난 상태)
        if error > self.max_error:
            self._warn_once("error",
                            f"[경고] 호모그래피 오차가 큽니다 ({error:.1f}cm > {self.max_error}cm). "
                            f"MARKER_WORLD_POS가 실제 마커 배치와 일치하는지 확인하세요.")
            return self.H is not None

        # 기존 해보다 나을 때만 교체 (마커가 더 많거나, 같은 수면 오차가 더 작을 때)
        better = (
            self.H is None
            or n > self.calibrated_with
            or (n == self.calibrated_with and error < self.reproj_error)
        )
        if not better:
            return True

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.calibrated_with = n
        self.reproj_error = error

        # 품질이 충분하면 고정. 그 전까지는 계속 더 나은 해를 찾는다.
        if self.lock_homography and n >= self.lock_markers:
            self.locked = True
            print(f"[INFO] 호모그래피 확정. (마커 {n}개, 평균 오차 {error:.2f}cm)")
            # 카메라가 고정이므로 다음 실행에서 재사용할 수 있게 저장해 둔다
            if CONFIG['USE_HOMOGRAPHY_CACHE']:
                self.save_homography()
        else:
            need = self.lock_markers if self.lock_homography else n
            print(f"[INFO] 호모그래피 갱신. (마커 {n}개, 평균 오차 {error:.2f}cm) "
                  f"- 마커 {need}개 이상 보이면 확정합니다.")
        return True

    def _warn_once(self, key, message):
        """같은 경고가 매 프레임 쏟아지지 않도록 한 번만 출력."""
        if key in self._warned:
            return
        self._warned.add(key)
        print(message)

    def is_ready(self):
        """좌표 변환이 가능한 상태인지 확인."""
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
        self._warned.clear()

        if clear_cache:
            path = CONFIG['HOMOGRAPHY_CACHE']
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[INFO] 저장된 호모그래피를 삭제했습니다. {path}")
                except OSError as e:
                    print(f"[경고] 캐시 파일을 지우지 못했습니다: {e}")

        print("[INFO] 호모그래피를 초기화했습니다. 재계산이 필요합니다.")

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

    def get_target_world(self, car_id):
        """
        차량의 목표 주차 구역 실좌표를 반환. 목표가 없으면 None.

        B02가 '주차 완료' 판정에 쓴다. (활성 차량이 목표 구역 반경 안에
        들어왔는지 확인) C_main이 이 메서드를 CarMOT.update의 target_of로 넘긴다.

        Args:
            car_id: 차량 번호 4자리 문자열

        Returns:
            (x_cm, y_cm) 목표 구역 실좌표. 목표가 없으면 None.
        """
        spot_id = self.targets.get(car_id)
        return self.spot_world_pos.get(spot_id) if spot_id else None

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

        Args:
            frame:  OpenCV BGR 이미지 (마커 검출용)
            tracks: B02_car_mot.CarMOT.update()의 반환 결과 리스트

        Returns:
            내비게이션 결과 리스트. 각 항목은 딕셔너리:
            [
                {
                    "track_id": 5,
                    "car_id": "1234",
                    "image_pos": (cx, cy),          # 이미지 좌표
                    "world_pos": (x_cm, y_cm),      # 주차장 실좌표
                    "heading_deg": 92.5,            # 진행 방향 (없으면 None)
                    "target_spot": "A-1",           # 목표 구역 (없으면 None)
                    "target_world": (0.0, 0.0),     # 목표 실좌표 (없으면 None)
                    "distance_cm": 45.2,            # 경로를 따라 남은 거리
                    "guide": "LEFT",                # 안내 방향 상수
                    "guide_text": "좌회전",          # 한글 안내
                    "nearest_spot": "A-2",          # 현재 가장 가까운 구역
                    "route": [(x, y), ...],         # 목적지까지의 경유점 (없으면 None)
                    "route_index": 1,               # 현재 향하고 있는 경유점 번호
                    "next_waypoint": (x, y),        # 다음 경유점 (없으면 None)
                },
                ...
            ]
        """
        # 1) 마커 검출 및 호모그래피 갱신
        markers = self.mapper.detect_markers(frame)
        self.latest_markers = markers
        self.mapper.update_homography(markers)

        results = []
        if not self.mapper.is_ready():
            # 아직 좌표 변환이 불가능한 상태 (마커가 충분히 보이지 않음)
            return results

        for trk in tracks:
            car_id = trk.get("car_id")
            image_pos = trk["center"]

            world_pos = self.mapper.image_to_world(image_pos)
            if world_pos is None:
                continue

            # 위치 이력 갱신 (차량번호가 없으면 track_id로 임시 키 사용)
            key = car_id if car_id else f"track_{trk['track_id']}"
            history = self.world_history.setdefault(
                key, deque(maxlen=self.history_maxlen)
            )
            history.append(world_pos)

            heading = self._compute_heading(history)
            target_spot = self.targets.get(car_id) if car_id else None
            target_world = self.spot_world_pos.get(target_spot) if target_spot else None

            distance = None
            guide = GUIDE_UNKNOWN
            route = None
            route_index = 0
            next_waypoint = None
            maneuver = GUIDE_UNKNOWN
            maneuver_distance = None

            if target_spot and target_world is not None:
                # 경로를 확보하고 통과한 경유점을 넘긴다
                state = self._ensure_route(car_id, world_pos, target_spot)
                if state is not None:
                    route = state["waypoints"]
                    route_index = state["index"]
                    next_waypoint = route[route_index] if route_index < len(route) else None
                    distance = route_length(route, route_index, world_pos)
                    maneuver, maneuver_distance = self.compute_maneuver(
                        route, route_index, world_pos)
                else:
                    # 경로를 찾지 못한 경우에도 직선 거리는 알려준다
                    distance = self._distance(world_pos, target_world)

                # 안내는 목적지가 아니라 '다음 경유점' 기준으로 한다.
                # 목적지 직선 방향으로 안내하면 주차 구역을 가로지르게 된다.
                aim = next_waypoint if next_waypoint is not None else target_world
                remaining = self._distance(world_pos, target_world)
                guide = self._compute_guide(world_pos, heading, aim, remaining)

            results.append({
                "track_id": trk["track_id"],
                "car_id": car_id,
                "image_pos": image_pos,
                "world_pos": world_pos,
                "heading_deg": heading,
                "target_spot": target_spot,
                "target_world": target_world,
                "distance_cm": distance,
                "route": route,
                "route_index": route_index,
                "next_waypoint": next_waypoint,
                "maneuver": maneuver,
                "maneuver_distance_cm": maneuver_distance,
                "guide": guide,
                "guide_text": GUIDE_TEXT_KO.get(guide, guide),
                "nearest_spot": self.find_nearest_spot(world_pos),
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

    def get_route(self, car_id):
        """차량의 현재 경로 경유점 리스트를 반환. 없으면 None."""
        state = self.routes.get(car_id)
        return list(state["waypoints"]) if state else None

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

    def get_world_position(self, car_id):
        """차량 번호로 현재 실좌표를 조회. 이력이 없으면 None."""
        history = self.world_history.get(car_id)
        return history[-1] if history else None

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

        Args:
            frame:            OpenCV BGR 이미지 (원본이 수정됨)
            nav_results:      update() 메서드의 반환 결과 리스트
            draw_target_line: 목표 지점까지 안내선을 그릴지 여부

        Returns:
            시각화가 적용된 프레임
        """
        for nav in nav_results:
            cx, cy = int(nav["image_pos"][0]), int(nav["image_pos"][1])
            wx, wy = nav["world_pos"]

            # 실좌표 표시
            cv2.putText(frame, f"({wx:.0f},{wy:.0f})cm", (cx - 40, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PATH, 2, cv2.LINE_AA)

            if not draw_target_line or nav["target_world"] is None:
                continue

            # 경로를 따라 안내선을 그린다. (목적지 직선이 아니라 실제 주행 경로)
            route = nav.get("route")
            if route:
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

            target_img = self.mapper.world_to_image(nav["target_world"])
            if target_img is not None:
                if not route:
                    cv2.arrowedLine(frame, (cx, cy), target_img, COLOR_PATH, 2, tipLength=0.05)
                cv2.circle(frame, target_img, 8, COLOR_TARGET, 2)

            # 안내 문구 (한글은 OpenCV에서 렌더링되지 않으므로 영문 상수 사용)
            dist = nav["distance_cm"]
            label = f"{nav['target_spot']} {nav['guide']} {dist:.0f}cm"
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
