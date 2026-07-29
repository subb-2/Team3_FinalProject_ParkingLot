import sys
import os

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.map_data import (
    grid_map, spot_map, coord_to_spot, PILL_MARKER_ID, MARKER_ID_CELL,
    GATE1_POS, GATE2_POS, PILL, SPOT, get_rows, get_cols,
)

# 설정 (Configuration)
# 이 모듈은 '격자(설계도) <-> 실좌표(cm) 변환'만 담당.
#   - 격자 정의   : data/map_data.py
#   - 위치 추정   : C00_navigation.py
#   - 경로 계산   : C01_path_planner.py
#
# 여기서 만드는 것:
#   MARKER_WORLD_POS : 마커 ID -> 실좌표 (cm).  기둥의 위치
#   SPOT_WORLD_POS   : 구역 ID -> 실좌표 (cm).  위아래 기둥의 중점
#   GATE1/2_WORLD_POS: 입출구 실좌표 (cm)
CONFIG = {
    # 격자 한 칸의 실제 크기 (cm).
    # 주의: 격자 칸과 기둥 간격은 다르다. 기둥 사이에는 칸이 하나씩 끼어 있다.
    #       세로 기둥 간격        = CELL_H_CM x 2
    #       좌우 기둥 열(col 1 <-> col 5) 간격 = CELL_W_CM x 4
    #
    # 실측 (목업 기준)
    #       좌우 기둥 열 간격 40cm  ->  CELL_W_CM = 40 / 4 = 10.0
    #       세로 기둥 간격    35cm  ->  CELL_H_CM = 35 / 2 = 17.5
    "CELL_W_CM": 10.0,
    "CELL_H_CM": 17.5,

    # 실좌표 원점으로 삼을 격자 칸. 기본값은 왼쪽 위 기둥(마커 ID 1).
    "ORIGIN_CELL": (1, 1),
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
    재투영 오차가 기준(C00의 MAX_REPROJ_ERROR_CM) 안에 들어오면 그대로 쓰면 되고,
    넘으면 실제 배치가 격자와 다르다는 뜻이므로 실측값으로 교체해야 한다.

    Returns:
        {마커ID: (x_cm, y_cm)}
    """
    return {
        marker_id: cell_to_world(cell, cell_w, cell_h, origin_cell)
        for cell, marker_id in PILL_MARKER_ID.items()
    }


def get_spot_pillars(spot_id):
    """
    주차 구역의 위/아래를 감싸는 기둥의 마커 ID를 반환.

    격자 규칙상 모든 주차 구역은 같은 열의 기둥 사이에 놓이므로,
    한 칸 위와 한 칸 아래를 보면 항상 기둥이 나온다.

    Args:
        spot_id: 주차 구역 ID (예: "A-1")

    Returns:
        (위쪽 마커ID, 아래쪽 마커ID) 튜플. 규칙에 어긋나면 None.
    """
    cells = spot_map.get(spot_id)
    if not cells:
        return None

    row, col = cells[0]
    upper = PILL_MARKER_ID.get((row - 1, col))
    lower = PILL_MARKER_ID.get((row + 1, col))
    if upper is None or lower is None:
        return None
    return (upper, lower)


def build_spot_world_pos(marker_world_pos):
    """
    마커 실좌표로부터 각 주차 구역의 실좌표를 계산.

    자리의 중심은 그 자리를 감싸는 두 기둥의 중점으로 둔다.
    마커가 격자대로 반듯하게 놓여 있지 않아도, 각 마커의 실좌표만
    정확하면 자리 좌표도 따라서 정확해진다.

    Args:
        marker_world_pos: {마커ID: (x_cm, y_cm)}

    Returns:
        {구역ID: (x_cm, y_cm)}
    """
    spots = {}
    for spot_id in spot_map:
        pillars = get_spot_pillars(spot_id)
        if pillars is None:
            print(f"[경고] '{spot_id}'를 감싸는 기둥을 찾지 못했습니다. "
                  f"grid_map의 기둥/자리 배치 규칙을 확인하세요.")
            continue

        upper, lower = pillars
        p1 = marker_world_pos.get(upper)
        p2 = marker_world_pos.get(lower)
        if p1 is None or p2 is None:
            print(f"[경고] '{spot_id}'의 기둥 마커({upper}, {lower}) 실좌표가 없습니다.")
            continue

        spots[spot_id] = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

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
    for row in range(get_rows()):
        for col in range(get_cols()):
            if grid_map[row][col] == SPOT and (row, col) not in coord_to_spot:
                problems.append(f"주차 구역 칸 ({row}, {col})이 spot_map에 없습니다.")

    # 5) 모든 자리가 위아래 기둥 사이에 있는가
    for spot_id in spot_map:
        if get_spot_pillars(spot_id) is None:
            problems.append(f"'{spot_id}'가 기둥 사이에 있지 않습니다.")

    return problems


# 모듈 로드 시 기본 배치를 만들어 둔다.
# 실측값이 생기면 uniform_marker_world_pos() 대신 그 표를 넣고
# build_spot_world_pos()를 다시 호출하면 자리 좌표가 자동으로 갱신된다.
MARKER_WORLD_POS = uniform_marker_world_pos()
SPOT_WORLD_POS = build_spot_world_pos(MARKER_WORLD_POS)
GATE1_WORLD_POS = cell_to_world(GATE1_POS)   # 입구 (경로 시작점)
GATE2_WORLD_POS = cell_to_world(GATE2_POS)   # 출구


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
        upper, lower = get_spot_pillars(spot_id)
        print(f"  {spot_id}  ->  ({x:7.1f}, {y:7.1f}) cm   "
              f"(기둥 id{upper} / id{lower} 의 중점)")

    print(f"\n--- 입출구 ---")
    print(f"  GATE1(입구) 격자{GATE1_POS}  ->  "
          f"({GATE1_WORLD_POS[0]:.1f}, {GATE1_WORLD_POS[1]:.1f}) cm")
    print(f"  GATE2(출구) 격자{GATE2_POS}  ->  "
          f"({GATE2_WORLD_POS[0]:.1f}, {GATE2_WORLD_POS[1]:.1f}) cm")

    # 격자 시각화
    print("\n[격자]  #=벽  O=기둥(숫자=마커ID)  P=주차구역  .=도로  1=입구  2=출구")
    from data.map_data import WALL, ROAD, GATE1, GATE2
    for row in range(get_rows()):
        line = "  "
        for col in range(get_cols()):
            cell = grid_map[row][col]
            if cell == WALL:
                line += "  # "
            elif cell == ROAD:
                line += "  . "
            elif cell == GATE1:
                line += "  1 "
            elif cell == GATE2:
                line += "  2 "
            elif cell == PILL:
                line += f" O{PILL_MARKER_ID[(row, col)]:<2d}"
            elif cell == SPOT:
                line += f"{coord_to_spot[(row, col)]:>4s}"
        print(line)

    print("\n[TEST] 자리 좌표가 위아래 기둥의 정확히 가운데인지 확인하세요.")
