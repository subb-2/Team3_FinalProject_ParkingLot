import sys
import os

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.map_data import (
    grid_map, spot_map, coord_to_spot, spot_type, PILL_MARKER_ID, MARKER_ID_CELL,
    GATE1_POS, GATE2_POS, PILL, SPOT_CELLS, SPOT_TYPE_NAME, get_rows, get_cols,
    one_way_segments, validate_one_way_loop,
)

# 설정 (Configuration)
# 이 모듈은 '격자(설계도) <-> 실좌표(cm) 변환'만 담당.
#   - 격자 정의   : data/map_data.py
#   - 위치 추정   : C00_navigation.py
#   - 경로 계산   : C01_path_planner.py
#
# 여기서 만드는 것:
#   MARKER_WORLD_POS : 마커 ID -> 실좌표 (cm).  기둥의 위치
#   SPOT_WORLD_POS   : 구역 ID -> 실좌표 (cm).  같은 열 기둥 사이를 보간한 위치
#   GATE1/2_WORLD_POS: 입출구 실좌표 (cm)
CONFIG = {
    # 격자 한 칸의 실제 크기 (cm).
    #
    # 실측으로 확인한 값이다. (기둥 중심 사이를 자로 잰 것)
    #       왼쪽 열 <-> 왼쪽 섬   (c0 <-> c5)   5칸  = 50cm   -> CELL_W_CM 10.0
    #       섬 <-> 섬             (c5 <-> c7)   2칸  = 20cm   -> CELL_W_CM 10.0
    #       오른쪽 섬 <-> 오른쪽 열 (c7 <-> c12)  5칸  = 50cm   -> CELL_W_CM 10.0
    #       전체 폭               (c0 <-> c12) 12칸 = 120cm
    #
    # 세로는 아직 실측 전이다. 지금 값(17.5)이면 이래야 한다.
    #       세로 기둥 간격 (row 0 <-> row 3) = 3칸 = 52.5cm
    #       왼쪽 열 전체   (row 0 <-> row 6) = 6칸 = 105cm
    # 다르면 CELL_H_CM = 실측값 / 6 으로 고칠 것. (긴 거리로 재야 오차가 작다)
    #
    # 이 값이 틀리면 호모그래피 재투영 오차가 커지고 안내 거리도 전부 어긋난다.
    # 열 배분 자체가 실물과 다르면 data/map_data.py의 grid_map을 고쳐야 한다.
    "CELL_W_CM": 10.0,
    "CELL_H_CM": 17.5,

    # 실좌표 원점으로 삼을 격자 칸.
    # 왼쪽 위 기둥(현재 배치에서 (0,0), 마커 ID 1)을 원점으로 둔다.
    # MAIN_README의 좌표계 규칙과 일치시킬 것.
    "ORIGIN_CELL": (0, 0),
}


def cell_to_world(cell, cell_w=None, cell_h=None, origin_cell=None):
    """
    격자 좌표 (row, col)를 해당 칸 중심의 실좌표(cm)로 변환.

    좌표계는 프로젝트 규칙을 따른다. x는 오른쪽 +, y는 아래쪽 +.

    Args:
        cell:        (row, col) 격자 좌표
        cell_w/h:    격자 한 칸의 가로/세로 크기 (cm). None이면 CONFIG 값
        origin_cell: 실좌표 (0, 0)으로 둘 격자 칸. None이면 CONFIG 값

    Returns:
        (x_cm, y_cm) 튜플
    """
    cell_w = CONFIG['CELL_W_CM'] if cell_w is None else cell_w
    cell_h = CONFIG['CELL_H_CM'] if cell_h is None else cell_h
    origin_cell = CONFIG['ORIGIN_CELL'] if origin_cell is None else origin_cell

    row, col = cell
    return ((col - origin_cell[1]) * cell_w,
            (row - origin_cell[0]) * cell_h)


def uniform_marker_world_pos(cell_w=None, cell_h=None, origin_cell=None):
    """
    기둥이 격자대로 반듯하게 놓여 있다고 가정하고 마커 실좌표를 생성.

    실측 없이 시작할 때 쓰는 초기값이다. 이 값으로 호모그래피를 계산했을 때
    기둥 보정으로 만든 격자가 실물과 겹쳐 보이면 그대로 쓰면 되고,
    넘으면 실제 배치가 격자와 다르다는 뜻이므로 실측값으로 교체해야 한다.

    Returns:
        {마커ID: (x_cm, y_cm)}
    """
    return {
        marker_id: cell_to_world(cell, cell_w, cell_h, origin_cell)
        for cell, marker_id in PILL_MARKER_ID.items()
    }


def column_pillars(col):
    """
    해당 열에 있는 기둥을 위에서 아래 순서로 반환.

    Returns:
        [(row, 마커ID), ...] 행 오름차순
    """
    return sorted(
        (row, marker_id)
        for (row, c), marker_id in PILL_MARKER_ID.items()
        if c == col
    )


def get_spot_anchors(spot_id):
    """
    주차 구역의 실좌표를 계산할 기준 기둥 두 개와 보간 비율을 반환.

    예전 배치는 '자리 한 칸의 위아래가 항상 기둥'이어서 두 기둥의 중점을 쓰면
    됐다. 지금 배치는 그 규칙이 깨졌다.
      - 기둥 사이에 자리가 2칸씩 들어간다 (예: col 0의 row 1, 2)
      - 마지막 기둥보다 아래에 있는 자리도 있다 (예: col 0의 row 7)

    그래서 '중점'이 아니라 '같은 열 기둥들 사이의 보간'으로 일반화한다.
    자리를 감싸는 기둥 쌍이 있으면 그 사이를 행 번호 비율로 보간하고,
    기둥 바깥에 있으면 가장 가까운 기둥 쌍의 간격으로 외삽한다.

    이렇게 해도 '마커 실좌표가 정확하면 자리 좌표도 정확해진다'는 원래 성질은
    그대로 유지된다. 기둥이 격자대로 반듯하지 않아도 된다.

    Args:
        spot_id: 주차 구역 ID (예: "A-1")

    Returns:
        (위쪽 마커ID, 아래쪽 마커ID, 보간비율 t) 튜플.
        t는 0이면 위쪽 기둥 위치, 1이면 아래쪽 기둥 위치. 범위를 벗어나면 외삽.
        기준으로 삼을 기둥이 부족하면 None.
    """
    cells = spot_map.get(spot_id)
    if not cells:
        return None

    # 한 자리가 여러 칸을 차지하는 경우(대형 구역 = 2칸)가 있으므로
    # 대표 행은 칸들의 '가운데'로 잡는다. 2칸이면 row 1.5처럼 소수가 나오는데,
    # 아래 보간이 실수 행을 그대로 다룬다.
    col = cells[0][1]
    row = sum(r for r, _ in cells) / len(cells)

    pillars = column_pillars(col)
    if len(pillars) < 2:
        return None

    # 1) 자리를 감싸는 기둥 쌍을 찾는다 (보간)
    for i in range(len(pillars) - 1):
        row_a, marker_a = pillars[i]
        row_b, marker_b = pillars[i + 1]
        if row_a <= row <= row_b:
            return marker_a, marker_b, (row - row_a) / (row_b - row_a)

    # 2) 기둥 바깥이면 가장 가까운 쌍으로 외삽한다
    if row < pillars[0][0]:
        (row_a, marker_a), (row_b, marker_b) = pillars[0], pillars[1]
    else:
        (row_a, marker_a), (row_b, marker_b) = pillars[-2], pillars[-1]
    return marker_a, marker_b, (row - row_a) / (row_b - row_a)


def build_spot_world_pos(marker_world_pos):
    """
    마커 실좌표로부터 각 주차 구역의 실좌표를 계산.

    같은 열의 기둥 두 개를 기준으로 자리의 행 위치만큼 보간(또는 외삽)한다.
    자세한 규칙은 get_spot_anchors 참고.

    Args:
        marker_world_pos: {마커ID: (x_cm, y_cm)}

    Returns:
        {구역ID: (x_cm, y_cm)}
    """
    spots = {}
    for spot_id in spot_map:
        anchors = get_spot_anchors(spot_id)
        if anchors is None:
            print(f"[경고] '{spot_id}'의 기준 기둥을 찾지 못했습니다. "
                  f"자리가 있는 열에는 기둥이 최소 2개 있어야 합니다. "
                  f"grid_map과 PILL_MARKER_ID를 확인하세요.")
            continue

        marker_a, marker_b, t = anchors
        p1 = marker_world_pos.get(marker_a)
        p2 = marker_world_pos.get(marker_b)
        if p1 is None or p2 is None:
            print(f"[경고] '{spot_id}'의 기준 마커({marker_a}, {marker_b}) 실좌표가 없습니다.")
            continue

        spots[spot_id] = (p1[0] + t * (p2[0] - p1[0]),
                          p1[1] + t * (p2[1] - p1[1]))

    return spots


def validate_layout():
    """
    격자와 마커 표가 서로 맞는지 검사. (설정을 바꾼 뒤 한 번 돌려볼 것)

    Returns:
        문제 설명 문자열 리스트. 비어 있으면 정상.
    """
    problems = []

    # 1) grid_map의 모든 PILL 칸에 마커 ID가 지정되어 있는가
    for row in range(get_rows()):
        for col in range(get_cols()):
            if grid_map[row][col] == PILL and (row, col) not in PILL_MARKER_ID:
                problems.append(f"기둥 칸 ({row}, {col})에 마커 ID가 없습니다.")

    # 2) PILL_MARKER_ID가 실제 PILL 칸을 가리키는가
    for (row, col), marker_id in PILL_MARKER_ID.items():
        if grid_map[row][col] != PILL:
            problems.append(f"마커 {marker_id}의 칸 ({row}, {col})은 기둥이 아닙니다.")

    # 3) 마커 ID가 중복되지 않는가
    if len(MARKER_ID_CELL) != len(PILL_MARKER_ID):
        problems.append("마커 ID가 중복되었습니다.")

    # 4) grid_map의 모든 SPOT 칸이 spot_map에 등록되어 있는가
    #    (spot_map은 grid_map에서 자동 생성되므로 정상이라면 항상 통과한다.
    #     실패한다면 map_data의 자동 생성 로직이 깨진 것이다)
    for row in range(get_rows()):
        for col in range(get_cols()):
            if grid_map[row][col] in SPOT_CELLS and (row, col) not in coord_to_spot:
                problems.append(f"주차 구역 칸 ({row}, {col})이 spot_map에 없습니다.")

    # 5) 모든 자리가 기준 기둥을 찾을 수 있는가
    #    (자리가 있는 열에는 기둥이 최소 2개 있어야 보간이 가능하다)
    for spot_id in spot_map:
        if get_spot_anchors(spot_id) is None:
            col = spot_map[spot_id][0][1]
            problems.append(
                f"'{spot_id}'(열 {col})의 기준 기둥이 부족합니다. "
                f"그 열의 기둥 수: {len(column_pillars(col))}개 (최소 2개 필요)"
            )

    # 6) 입출구가 grid_map에 있는가
    if GATE1_POS is None:
        problems.append("grid_map에 입구(GATE1) 칸이 없습니다.")
    if GATE2_POS is None:
        problems.append("grid_map에 출구(GATE2) 칸이 없습니다.")

    # 7) 일방통행 순환선이 실제 통로 위를 지나는가
    problems.extend(validate_one_way_loop())

    return problems


# 모듈 로드 시 기본 배치를 만들어 둔다.
# 실측값이 생기면 uniform_marker_world_pos() 대신 그 표를 넣고
# build_spot_world_pos()를 다시 호출하면 자리 좌표가 자동으로 갱신된다.
MARKER_WORLD_POS = uniform_marker_world_pos()
SPOT_WORLD_POS = build_spot_world_pos(MARKER_WORLD_POS)
GATE1_WORLD_POS = cell_to_world(GATE1_POS)   # 입구 (경로 시작점)
GATE2_WORLD_POS = cell_to_world(GATE2_POS)   # 출구

# 일방통행 순환선을 실좌표(cm) 선분 목록으로. [((x1,y1), (x2,y2)), ...]
# 마지막 점에서 첫 점으로 닫힌다.
# C01_path_planner가 이걸로 통행 방향장을 만들고, C00과 D00이 화살표로 그린다.
ONE_WAY_SEGMENTS_WORLD = [
    (cell_to_world(a), cell_to_world(b)) for a, b in one_way_segments()
]


def build_spot_pixel_pos(pillar_pixels):
    """
    기둥 픽셀 좌표로부터 각 주차 구역의 픽셀 좌표를 계산.

    build_spot_world_pos과 동일한 보간 로직을 사용한다.
    입력이 cm든 pixel이든 '같은 열 기둥 사이를 행 비율로 보간'하는
    원리는 바뀌지 않기 때문이다.

    Args:
        pillar_pixels: {마커ID: (x_px, y_px)} 사용자가 클릭한 기둥 픽셀 좌표

    Returns:
        {구역ID: (x_px, y_px)}
    """
    return build_spot_world_pos(pillar_pixels)


def build_gate_pixel_pos(pillar_pixels):
    """
    기둥 픽셀 좌표로부터 입출구의 픽셀 좌표를 보간으로 추정.

    입출구는 격자 맨 아래줄(row 8)에 있으므로, 같은 열 또는 가장 가까운
    기둥들로부터 외삽한다.

    Args:
        pillar_pixels: {마커ID: (x_px, y_px)} 사용자가 클릭한 기둥 픽셀 좌표

    Returns:
        (gate1_pixel, gate2_pixel) 튜플. 각각 (x_px, y_px) 또는 None.
    """
    from data.map_data import GATE1_POS, GATE2_POS, PILL_MARKER_ID

    def _interpolate_gate(gate_pos):
        if gate_pos is None:
            return None
        gate_row, gate_col = gate_pos
        # 입출구와 같은 열에 기둥이 없으므로, 가장 가까운 열의 기둥 쌍을 찾아
        # 열 방향으로도 보간한다.
        # 모든 기둥의 위치를 열 기준으로 모은다
        col_pillars = {}  # {col: [(row, marker_id), ...]}
        for (r, c), mid in PILL_MARKER_ID.items():
            col_pillars.setdefault(c, []).append((r, mid))

        # 입출구 열에 가장 가까운 양쪽 기둥 열을 찾는다
        cols = sorted(col_pillars.keys())
        left_col = max((c for c in cols if c <= gate_col), default=None)
        right_col = min((c for c in cols if c >= gate_col), default=None)

        if left_col is None or right_col is None:
            return None

        def _col_gate_y(col):
            """해당 열의 기둥들로 gate_row에 해당하는 y 픽셀을 외삽."""
            pillars = sorted(col_pillars[col])  # (row, marker_id) 오름차순
            if len(pillars) < 2:
                px = pillar_pixels.get(pillars[0][1])
                return px[1] if px else None
            # 가장 아래 기둥 쌍으로 외삽
            r_a, mid_a = pillars[-2]
            r_b, mid_b = pillars[-1]
            px_a = pillar_pixels.get(mid_a)
            px_b = pillar_pixels.get(mid_b)
            if px_a is None or px_b is None:
                return None
            if r_b == r_a:
                return px_b[1]
            t = (gate_row - r_a) / (r_b - r_a)
            return px_a[1] + t * (px_b[1] - px_a[1])

        def _col_gate_x(col):
            """해당 열의 기둥들의 평균 x 픽셀."""
            xs = [pillar_pixels[mid][0] for _, mid in col_pillars[col]
                  if mid in pillar_pixels]
            return sum(xs) / len(xs) if xs else None

        if left_col == right_col:
            x = _col_gate_x(left_col)
            y = _col_gate_y(left_col)
        else:
            x_l, x_r = _col_gate_x(left_col), _col_gate_x(right_col)
            y_l, y_r = _col_gate_y(left_col), _col_gate_y(right_col)
            if any(v is None for v in (x_l, x_r, y_l, y_r)):
                return None
            t = (gate_col - left_col) / (right_col - left_col)
            x = x_l + t * (x_r - x_l)
            y = y_l + t * (y_r - y_l)

        if x is None or y is None:
            return None
        return (float(x), float(y))

    return _interpolate_gate(GATE1_POS), _interpolate_gate(GATE2_POS)


# =====================================================================
# 테스트용 메인 (단독 실행 시 격자 -> 실좌표 변환 검증)
# =====================================================================
# 카메라 없이 동작한다.
if __name__ == '__main__':
    print("==========================================")
    print(" C02 : 주차장 배치 (격자 -> 실좌표 변환)")
    print(" 단독 테스트 : 마커/자리 좌표 생성 검증")
    print("==========================================")

    print(f"\n[INFO] 격자 {get_rows()}행 x {get_cols()}열, "
          f"칸 크기 {CONFIG['CELL_W_CM']} x {CONFIG['CELL_H_CM']}cm, "
          f"원점 칸 {CONFIG['ORIGIN_CELL']}")

    problems = validate_layout()
    if problems:
        print(f"\n[경고] 배치 검사에서 {len(problems)}건의 문제를 발견했습니다.")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\n[OK] 배치 검사 통과 (기둥/자리/마커 ID 모두 정합)")

    print(f"\n--- 기둥 마커 {len(MARKER_WORLD_POS)}개 ---")
    for marker_id in sorted(MARKER_WORLD_POS):
        x, y = MARKER_WORLD_POS[marker_id]
        row, col = MARKER_ID_CELL[marker_id]
        print(f"  id{marker_id:<3d} 격자({row},{col})  ->  ({x:7.1f}, {y:7.1f}) cm")

    print(f"\n--- 주차 구역 {len(SPOT_WORLD_POS)}개 ---")
    for spot_id in sorted(SPOT_WORLD_POS):
        x, y = SPOT_WORLD_POS[spot_id]
        marker_a, marker_b, t = get_spot_anchors(spot_id)
        kind = SPOT_TYPE_NAME.get(spot_type[spot_id], "?")
        note = "보간" if 0.0 <= t <= 1.0 else "외삽"
        print(f"  {spot_id}  {kind:4s} ->  ({x:7.1f}, {y:7.1f}) cm   "
              f"(기둥 id{marker_a} / id{marker_b} 사이 {t:.2f} 지점, {note})")

    print("\n--- 입출구 ---")
    print(f"  GATE1(입구) 격자{GATE1_POS}  ->  "
          f"({GATE1_WORLD_POS[0]:.1f}, {GATE1_WORLD_POS[1]:.1f}) cm")
    print(f"  GATE2(출구) 격자{GATE2_POS}  ->  "
          f"({GATE2_WORLD_POS[0]:.1f}, {GATE2_WORLD_POS[1]:.1f}) cm")

    # 격자 시각화
    print("\n[격자]  O=기둥(숫자=마커ID)  .=도로  1=입구  2=출구  나머지=구역ID")
    from data.map_data import ROAD, GATE1, GATE2
    for row in range(get_rows()):
        line = f"  r{row} "
        for col in range(get_cols()):
            cell = grid_map[row][col]
            if cell == ROAD:
                line += "  .  "
            elif cell == GATE1:
                line += "  1  "
            elif cell == GATE2:
                line += "  2  "
            elif cell == PILL:
                line += f" O{PILL_MARKER_ID[(row, col)]:<3d}"
            elif cell in SPOT_CELLS:
                line += f"{coord_to_spot[(row, col)]:>5s}"
        print(line)

    # 구역 종류별 요약
    print("\n--- 구역 종류별 ---")
    by_kind = {}
    for spot_id, cell in spot_type.items():
        by_kind.setdefault(cell, []).append(spot_id)
    for cell in sorted(by_kind):
        ids = sorted(by_kind[cell])
        print(f"  {SPOT_TYPE_NAME.get(cell, '?'):6s} {len(ids):2d}개 : {', '.join(ids)}")

    print("\n[TEST] 확인 사항")
    print("       1) 배치 검사가 통과했는가")
    print("       2) 같은 열의 자리들이 기둥 사이에 균등하게 배치되었는가")
    print("       3) CELL_W_CM / CELL_H_CM이 실제 목업 치수와 맞는가 (CONFIG 주석 참고)")
