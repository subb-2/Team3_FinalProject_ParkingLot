import sys
import os
import cv2
import numpy as np
import cv2.aruco as aruco
from collections import defaultdict

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.map_data import grid_map, PILL, PILL_MARKER_ID, get_rows, get_cols
from logic.C00_navigation import CONFIG as C00_CONFIG, MarkerMapper, build_aruco_params
from logic.B00_camera_input import get_camera

# =====================================================================
# C03 : ArUco 마커 ID 캘리브레이션 도구
# =====================================================================
# 기둥에 '실제로' 붙어 있는 마커 ID를 읽어서, data/map_data.py에 붙여넣을
# PILL_MARKER_ID 표를 만들어 준다.
#
# 왜 필요한가:
#   PILL_MARKER_ID는 격자(설계도)와 실제 카메라 영상을 잇는 유일한 연결 고리인데,
#   '어느 기둥에 몇 번 마커를 붙였는지'는 코드가 알 수 없는 물리 정보다.
#   이 표가 틀리면 호모그래피 재투영 오차가 커져 확정(LOCK)되지 않고,
#   PROVISIONAL 상태로 매 프레임 흔들리며 실좌표도 부정확해진다.
#   더 나쁜 경우, 틀린 대응으로 계산된 호모그래피가 그대로 고정될 수 있다.
#
# 어떻게 맞추는가:
#   기둥은 격자에서 몇 개의 '열'을 이루고 있고, 카메라가 정면에서 보면
#   이미지에서도 같은 열끼리 x좌표가 비슷하게 모인다.
#   그래서 검출된 마커를 x로 나눠 열을 만들고, 각 열 안에서 y로 정렬하면
#   격자의 (row, col)과 순서대로 짝지을 수 있다.
#
# 사용법 (젯슨에서):
#   python logic/C03_marker_calib.py              # 카메라로 측정
#   python logic/C03_marker_calib.py 사진.jpg     # 저장된 이미지로 측정
#   python logic/C03_marker_calib.py --sweep      # 검출 파라미터 비교 측정
#   python logic/C03_marker_calib.py --sweep 사진.jpg
#
# --sweep는 여러 파라미터 조합으로 같은 화면을 돌려 '몇 개 잡히는지'를
# 비교해 준다. 검출 파라미터는 추측으로 건드리면 오히려 나빠지므로
# (실제로 polygonalApproxAccuracyRate를 올렸다가 검출이 거의 죽은 적이 있다)
# C00의 ARUCO_PARAMS를 바꾸기 전에 이걸로 먼저 측정할 것.
#
# 주의: 카메라를 옮기거나 마커를 다시 붙이면 반드시 다시 돌릴 것.
# =====================================================================

CONFIG = {
    # 카메라 설정 (C_main과 맞춰 둘 것)
    "CAM_SENSOR_ID": 0,
    "CAM_WIDTH": 1280,
    "CAM_HEIGHT": 720,
    "CAM_FPS": 30,

    # 여러 프레임을 모아 평균낸다. 한 프레임만 보면 흔들린 검출에 속는다.
    "SAMPLE_FRAMES": 40,

    # 이 비율 이상의 프레임에서 잡힌 마커만 신뢰한다.
    # 가끔 튀는 오검출(다른 사전의 무늬가 우연히 읽히는 경우)을 걸러낸다.
    "MIN_SEEN_RATIO": 0.5,
}


# --sweep에서 비교해 볼 파라미터 조합.
# 이름 -> {속성명: 값}. 빈 dict는 OpenCV 기본값을 뜻한다.
SWEEP_PRESETS = {
    "기본값(OpenCV)": {},
    "코너정밀화만 (현재 설정)": {
        "cornerRefinementMethod": aruco.CORNER_REFINE_SUBPIX,
    },
    "이진화 촘촘히": {
        "cornerRefinementMethod": aruco.CORNER_REFINE_SUBPIX,
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 33,
        "adaptiveThreshWinSizeStep": 4,
    },
    "작은 마커 허용": {
        "cornerRefinementMethod": aruco.CORNER_REFINE_SUBPIX,
        "minMarkerPerimeterRate": 0.01,
    },
    "비트 판정 완화": {
        "cornerRefinementMethod": aruco.CORNER_REFINE_SUBPIX,
        "errorCorrectionRate": 0.8,
        "maxErroneousBitsInBorderRate": 0.5,
    },
    "샘플링 촘촘히(작은 마커용)": {
        "cornerRefinementMethod": aruco.CORNER_REFINE_SUBPIX,
        "perspectiveRemovePixelPerCell": 8,
        "minMarkerPerimeterRate": 0.01,
    },
    "코너근사 완화 (역효과 사례)": {
        "polygonalApproxAccuracyRate": 0.05,
    },
}


def measure_marker_size(frame, dict_name):
    """
    검출된 마커가 화면에서 몇 픽셀인지 잰다. (해상도가 충분한지 판단용)

    Returns:
        (검출 개수, 평균 한 변 픽셀). 검출이 없으면 (0, 0.0).
    """
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(getattr(aruco, dict_name)),
        build_aruco_params({})
    )
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) == 0:
        return 0, 0.0
    sides = []
    for c in corners:
        pts = c[0]
        sides.extend(float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4))
    return len(ids), float(np.mean(sides))


def run_sweep(frame, dict_name):
    """파라미터 조합별로 검출 개수를 비교 출력한다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))

    print(f"\n{'조합':30s}{'검출':>6s}   검출된 ID")
    print("-" * 66)
    best_name, best_count = None, -1
    for name, overrides in SWEEP_PRESETS.items():
        detector = aruco.ArucoDetector(dictionary, build_aruco_params(overrides))
        _, ids, _ = detector.detectMarkers(gray)
        found = sorted(int(i) for i in ids.flatten()) if ids is not None else []
        print(f"{name:30s}{len(found):>6d}   {found}")
        if len(found) > best_count:
            best_name, best_count = name, len(found)

    print(f"\n가장 많이 잡은 조합: '{best_name}' ({best_count}개)")
    if best_count:
        overrides = SWEEP_PRESETS[best_name]
        print("C00_navigation.CONFIG['ARUCO_PARAMS']에 넣을 값:")
        print("    " + (repr(overrides) if overrides else "{}  # 기본값 그대로"))


def report_marker_size(count, side_px):
    """
    마커가 화면에서 몇 픽셀인지 보고하고, 해상도가 부족하면 알려준다.

    5x5 비트 마커(DICT_ARUCO_ORIGINAL)는 테두리를 포함해 7x7 칸이다.
    OpenCV는 칸마다 perspectiveRemovePixelPerCell(기본 4)픽셀을 샘플링하므로
    한 변이 최소 7 x 4 = 28px은 되어야 비트를 안정적으로 읽는다.
    실전에서는 그 두 배(약 50px)는 되어야 흔들리지 않는다.
    """
    MIN_PX, COMFORTABLE_PX = 28, 50

    if count == 0:
        print("\n[크기] 마커를 검출하지 못해 크기를 잴 수 없습니다.")
        return

    print(f"\n[크기] 마커 {count}개, 한 변 평균 {side_px:.0f}px")
    if side_px < MIN_PX:
        print(f"        !! 너무 작습니다 (최소 {MIN_PX}px 필요). "
              f"이 크기로는 비트를 못 읽어 검출이 들쭉날쭉합니다.")
        print(f"        해상도를 올리거나(예: 640x480 -> 1280x720) "
              f"마커를 크게 인쇄하세요.")
        print(f"        해상도를 2배로 올리면 한 변도 약 2배 "
              f"({side_px:.0f} -> {side_px*2:.0f}px)가 됩니다.")
    elif side_px < COMFORTABLE_PX:
        print(f"        아슬아슬합니다 (여유 있으려면 {COMFORTABLE_PX}px 권장). "
              f"조명이나 각도가 나빠지면 놓칠 수 있습니다.")
    else:
        print(f"        충분합니다.")


def expected_pillar_cells():
    """
    격자에서 기둥 칸을 열별로 모아 반환.

    Returns:
        {col: [row, ...]}  각 열의 기둥 행 번호 (위에서 아래 순)
    """
    cells = defaultdict(list)
    for row in range(get_rows()):
        for col in range(get_cols()):
            if grid_map[row][col] == PILL:
                cells[col].append(row)
    return {col: sorted(rows) for col, rows in sorted(cells.items())}


def collect_markers(source):
    """
    여러 프레임에서 마커를 검출해 평균 중심 좌표를 구한다.

    Args:
        source: 카메라 객체 또는 단일 이미지(numpy array)

    Returns:
        {마커ID: (cx, cy)}  안정적으로 검출된 마커만
    """
    mapper = MarkerMapper()          # 검출 파라미터를 C00과 동일하게 쓰기 위함

    if isinstance(source, np.ndarray):
        return mapper.detect_markers(source)

    total = CONFIG['SAMPLE_FRAMES']
    seen = defaultdict(list)
    grabbed = 0

    print(f"[INFO] {total}프레임을 모으는 중...")
    while grabbed < total:
        ret, frame = source.read()
        if not ret:
            break
        grabbed += 1
        for marker_id, center in mapper.detect_markers(frame).items():
            seen[marker_id].append(center)

    if grabbed == 0:
        return {}

    result = {}
    for marker_id, points in seen.items():
        ratio = len(points) / grabbed
        if ratio < CONFIG['MIN_SEEN_RATIO']:
            print(f"[건너뜀] 마커 {marker_id}: {len(points)}/{grabbed} 프레임에서만 검출 "
                  f"({ratio*100:.0f}%). 흔들리는 검출이라 제외합니다.")
            continue
        arr = np.array(points)
        result[marker_id] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return result


def assign_by_position(markers, pillar_cells):
    """
    검출된 마커를 이미지 위치로 격자의 기둥 칸에 짝지어 준다.

    1) 마커를 x좌표 순으로 늘어놓고, 각 열의 기둥 개수만큼 잘라 열을 만든다.
    2) 각 열 안에서 y좌표 순(위 -> 아래)으로 정렬해 격자의 행 순서와 맞춘다.

    Args:
        markers:      {마커ID: (cx, cy)}
        pillar_cells: {col: [row, ...]}

    Returns:
        ({(row, col): 마커ID}, 문제 설명 리스트)
    """
    problems = []
    expected_total = sum(len(rows) for rows in pillar_cells.values())

    if len(markers) != expected_total:
        problems.append(
            f"검출된 마커 {len(markers)}개 != 격자의 기둥 {expected_total}개. "
            f"모든 기둥이 화면에 보이는지, 가려진 것은 없는지 확인하세요."
        )
        return {}, problems

    # x 순으로 정렬한 뒤 열별 개수만큼 끊는다
    by_x = sorted(markers.items(), key=lambda kv: kv[1][0])

    assignment = {}
    index = 0
    for col, rows in pillar_cells.items():
        group = by_x[index:index + len(rows)]
        index += len(rows)

        # 같은 열이면 x가 서로 비슷해야 한다. 많이 벌어지면 열 구분이 틀린 것이다.
        xs = [pt[0] for _, pt in group]
        if len(xs) > 1 and (max(xs) - min(xs)) > CONFIG['CAM_WIDTH'] * 0.25:
            problems.append(
                f"열 {col}으로 묶인 마커들의 x가 너무 벌어져 있습니다 "
                f"({min(xs):.0f}~{max(xs):.0f}px). 카메라가 기울었거나 "
                f"기둥이 가려졌을 수 있습니다."
            )

        # 열 안에서는 y 순(위 -> 아래)이 곧 행 순서
        for (marker_id, _), row in zip(sorted(group, key=lambda kv: kv[1][1]), rows):
            assignment[(row, col)] = marker_id

    return assignment, problems


def format_table(assignment, pillar_cells):
    """map_data.py에 그대로 붙여넣을 수 있는 PILL_MARKER_ID 문자열을 만든다."""
    lines = ["PILL_MARKER_ID = {"]
    for col, rows in pillar_cells.items():
        entries = []
        for row in rows:
            marker_id = assignment.get((row, col))
            if marker_id is not None:
                entries.append(f"({row}, {col}): {marker_id},")
        if entries:
            lines.append("    " + "  ".join(entries))
    lines.append("}")
    return "\n".join(lines)


if __name__ == '__main__':
    print("=" * 62)
    print(" C03 : ArUco 마커 ID 캘리브레이션")
    print(" 기둥에 실제로 붙은 마커를 읽어 PILL_MARKER_ID 표를 만듭니다.")
    print("=" * 62)

    pillar_cells = expected_pillar_cells()
    total = sum(len(r) for r in pillar_cells.values())
    print(f"\n[격자] 기둥 {total}개, 열 {len(pillar_cells)}개")
    for col, rows in pillar_cells.items():
        print(f"   열 {col:2d} : 행 {rows}")

    print(f"\n[사전] {C00_CONFIG['ARUCO_DICT']} "
          f"(인쇄물과 다르면 마커가 '하나도' 검출되지 않습니다)")

    args = [a for a in sys.argv[1:] if a != '--sweep']
    sweep = '--sweep' in sys.argv

    # 이미지 파일이 주어지면 그것으로, 아니면 카메라로
    if args:
        path = args[0]
        image = cv2.imread(path)
        if image is None:
            print(f"\n[ERROR] 이미지를 열 수 없습니다: {path}")
            sys.exit(1)
        print(f"\n[INFO] 이미지에서 검출: {path}  ({image.shape[1]}x{image.shape[0]})")

        count, side = measure_marker_size(image, C00_CONFIG['ARUCO_DICT'])
        report_marker_size(count, side)
        if sweep:
            run_sweep(image, C00_CONFIG['ARUCO_DICT'])
            sys.exit(0)
        markers = collect_markers(image)
    elif sweep:
        print(f"\n[INFO] 카메라를 엽니다... "
              f"({CONFIG['CAM_WIDTH']}x{CONFIG['CAM_HEIGHT']})")
        cap = get_camera(sensor_id=CONFIG['CAM_SENSOR_ID'],
                         width=CONFIG['CAM_WIDTH'],
                         height=CONFIG['CAM_HEIGHT'],
                         framerate=CONFIG['CAM_FPS'])
        if not cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다.")
            sys.exit(1)
        try:
            for _ in range(10):        # 노출이 안정될 때까지 몇 장 버린다
                ret, frame = cap.read()
        finally:
            cap.release()
        if not ret:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            sys.exit(1)
        print(f"[INFO] 실제 프레임 크기: {frame.shape[1]}x{frame.shape[0]}")
        count, side = measure_marker_size(frame, C00_CONFIG['ARUCO_DICT'])
        report_marker_size(count, side)
        run_sweep(frame, C00_CONFIG['ARUCO_DICT'])
        sys.exit(0)
    else:
        print(f"\n[INFO] 카메라를 엽니다... "
              f"({CONFIG['CAM_WIDTH']}x{CONFIG['CAM_HEIGHT']})")
        cap = get_camera(sensor_id=CONFIG['CAM_SENSOR_ID'],
                         width=CONFIG['CAM_WIDTH'],
                         height=CONFIG['CAM_HEIGHT'],
                         framerate=CONFIG['CAM_FPS'])
        if not cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다.")
            sys.exit(1)
        try:
            markers = collect_markers(cap)
        finally:
            cap.release()

    if not markers:
        print("\n[ERROR] 마커를 하나도 검출하지 못했습니다.")
        print("        - ARUCO_DICT가 인쇄물과 같은지 확인하세요.")
        print("        - 조명과 초점, 마커가 화면 안에 있는지 확인하세요.")
        sys.exit(1)

    print(f"\n--- 검출된 마커 {len(markers)}개 (x 순) ---")
    for marker_id, (cx, cy) in sorted(markers.items(), key=lambda kv: kv[1][0]):
        registered = "" if marker_id in PILL_MARKER_ID.values() else "  <- 현재 표에 없음"
        print(f"   id{marker_id:<3d}  이미지({cx:7.1f}, {cy:7.1f}){registered}")

    assignment, problems = assign_by_position(markers, pillar_cells)

    if problems:
        print(f"\n[경고] {len(problems)}건")
        for p in problems:
            print(f"   - {p}")

    if not assignment:
        print("\n표를 만들지 못했습니다. 위 경고를 해결한 뒤 다시 실행하세요.")
        sys.exit(1)

    print("\n--- 격자 칸 <-> 마커 짝짓기 ---")
    for (row, col), marker_id in sorted(assignment.items()):
        cx, cy = markers[marker_id]
        print(f"   기둥({row}, {col})  <-  id{marker_id:<3d}  이미지({cx:7.1f}, {cy:7.1f})")

    changed = assignment != PILL_MARKER_ID
    print(f"\n--- data/map_data.py에 붙여넣을 표 "
          f"({'현재와 다름' if changed else '현재와 동일'}) ---\n")
    print(format_table(assignment, pillar_cells))

    print("\n[다음 단계]")
    print("   1) 위 표를 data/map_data.py의 PILL_MARKER_ID에 붙여넣기")
    print("   2) python logic/C02_lot_layout.py 로 배치 검사")
    print("   3) C_main 실행 후 화면에 'Homography: LOCKED'가 뜨는지 확인")
    print("      (빨간 점 + '숫자?'로 표시되는 마커가 없어야 합니다)")
