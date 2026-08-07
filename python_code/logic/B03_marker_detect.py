import cv2
import numpy as np
import cv2.aruco as aruco

# =====================================================================
# B03 : ArUco 마커 검출 모듈
# =====================================================================
# C00_navigation.py의 MarkerMapper가 이 모듈의 MarkerDetector를 사용한다.
# C03_marker_calib.py의 진단 도구들도 이 모듈을 통해 검출한다.
#
# 마커 검출 고도화는 이 파일 안에서만 진행한다.
# 다른 모듈은 detect() 인터페이스만 사용하므로 내부 구현이 바뀌어도
# 외부에 영향이 없다.
# =====================================================================

# 설정
CONFIG = {
    # ArUco 사전 설정
    # 사전이 실제 인쇄물과 다르면 마커가 '하나도' 검출되지 않는다.
    # 현재 인쇄물: 4x4 비트 마커 ID 1~10, 한 변 50mm.
    "ARUCO_DICT": "DICT_4X4_50",

    # ArUco 검출 파라미터 덮어쓰기. 여기 없는 값은 OpenCV 기본값을 그대로 쓴다.
    # 바꾸기 전에 반드시 C03_marker_calib.py --sweep 으로 측정할 것.
    "ARUCO_PARAMS": {
        "cornerRefinementMethod": aruco.CORNER_REFINE_SUBPIX,
    },
}


def build_aruco_params(overrides=None):
    """
    ArUco 검출 파라미터를 만든다. OpenCV 기본값에서 시작해 필요한 것만 덮어쓴다.

    Args:
        overrides: {속성명: 값}. None이면 CONFIG['ARUCO_PARAMS'].

    Returns:
        cv2.aruco.DetectorParameters
    """
    params = aruco.DetectorParameters()
    overrides = CONFIG['ARUCO_PARAMS'] if overrides is None else overrides
    for name, value in overrides.items():
        if not hasattr(params, name):
            print(f"[경고] ArUco 파라미터 '{name}'은 존재하지 않습니다. 무시합니다.")
            continue
        setattr(params, name, value)
    return params


class MarkerDetector:
    """
    ArUco 마커 검출기.

    detect()  : {마커ID: (cx, cy)} 반환 — C00_navigation에서 사용
    detect_raw() : (corners, ids, rejected) 반환 — C03 진단 도구에서 사용
    """

    def __init__(self, aruco_dict_name=None, params_overrides=None):
        """
        MarkerDetector 초기화.

        Args:
            aruco_dict_name:   ArUco 사전 이름 (None이면 CONFIG['ARUCO_DICT'])
            params_overrides:  {속성명: 값} 파라미터 덮어쓰기 (None이면 CONFIG['ARUCO_PARAMS'])
        """
        aruco_dict_name = CONFIG['ARUCO_DICT'] if aruco_dict_name is None else aruco_dict_name

        dict_id = getattr(aruco, aruco_dict_name)
        self._detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(dict_id),
            build_aruco_params(params_overrides)
        )
        self._dict_name = aruco_dict_name

        print(f"[INFO] MarkerDetector 초기화 완료. (사전: {aruco_dict_name})")

    def detect(self, frame):
        """
        프레임에서 ArUco 마커를 검출.

        Args:
            frame: OpenCV BGR 이미지 (numpy array)

        Returns:
            {마커ID: (cx, cy)} 형태의 딕셔너리. cx, cy는 마커 중심의 이미지 좌표.
        """
        corners, ids, _ = self.detect_raw(frame)

        found = {}
        if ids is None:
            return found

        for corner, marker_id in zip(corners, ids.flatten()):
            center = corner[0].mean(axis=0)
            found[int(marker_id)] = (float(center[0]), float(center[1]))

        return found

    def detect_raw(self, frame):
        """
        프레임에서 ArUco 마커를 검출하고 원시 결과를 반환.

        Args:
            frame: OpenCV BGR 이미지 (numpy array)

        Returns:
            (corners, ids, rejected) — cv2.aruco.ArucoDetector.detectMarkers()의 반환값 그대로.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detector.detectMarkers(gray)
        return corners, ids, rejected

