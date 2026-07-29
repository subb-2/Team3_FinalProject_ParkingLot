# 주차장 맵 관리 모듈 명세서 (README)
# map_data.py

## 개요

주차장 전체를 **2D 그리드(11행 × 7열)** 격자판으로 관리하는 핵심 파일.
입출구, 도로, 벽, 기둥, 주차 구역을 하나의 격자판 위에서 관리하며,
각 자리의 status(`empty` / `full`)도 함께 관리한다.

이 격자는 **설계도**이고, 실좌표(cm) 변환은 `logic/C02_lot_layout.py`가 담당한다.
경로 계산은 `logic/C01_path_planner.py`(A\*)가 이 격자를 그대로 읽어서 수행한다.

## 셀 타입 상수

| 코드 | 상수명 | 의미 | 통행 |
|:---:|:---:|:---:|:---|
| `0` | `WALL`  | 벽 | 불가 |
| `1` | `ROAD`  | 도로 | 가능 |
| `2` | `SPOT`  | 주차 구역 | 목적지로 지정된 구역만 가능 |
| `3` | `GATE1` | 입구 | 가능 |
| `4` | `GATE2` | 출구 | 가능 |
| `5` | `PILL`  | 기둥 (ArUco 마커 부착 위치) | 불가 |

## 그리드 맵 구조 (11행 × 7열)

```
col:    0     1     2     3     4     5     6
row 0  WALL  WALL  GATE1 WALL  GATE2 WALL  WALL
row 1  WALL  PILL  ROAD  ROAD  ROAD  PILL  WALL
row 2  WALL  SPOT  ROAD  ROAD  ROAD  SPOT  WALL
row 3  WALL  PILL  ROAD  PILL  ROAD  PILL  WALL
row 4  WALL  SPOT  ROAD  SPOT  ROAD  SPOT  WALL
row 5  WALL  PILL  ROAD  PILL  ROAD  PILL  WALL
row 6  WALL  SPOT  ROAD  SPOT  ROAD  SPOT  WALL
row 7  WALL  PILL  ROAD  PILL  ROAD  PILL  WALL
row 8  WALL  SPOT  ROAD  ROAD  ROAD  SPOT  WALL
row 9  WALL  PILL  ROAD  ROAD  ROAD  PILL  WALL
row 10 WALL  WALL  WALL  WALL  WALL  WALL  WALL
```

- 세로 통로 2개(**col 2** = 왼쪽, **col 4** = 오른쪽)가 위(row 1\~2)와 아래(row 8\~9)에서
  이어지는 **ㅁ자 순환로**.
- 가운데 **col 3의 row 3\~7**은 중앙 섬 (기둥 3개 + 주차 구역 2개).
- 입구는 왼쪽 통로 위, 출구는 오른쪽 통로 위 → 입구로 들어와 왼쪽으로 내려가고
  아래에서 돌아 오른쪽으로 올라가 출구로 나가는 **일방통행 흐름**.

### 배치 규칙 (중요)

> 주차 구역(SPOT)은 반드시 **같은 열의 위아래 기둥(PILL) 사이**에 놓인다.

이 규칙 덕분에 자리의 실좌표를 **두 기둥 마커의 중점**으로 계산할 수 있다.
배치를 바꿀 때 이 규칙을 깨지 말 것. `C02_lot_layout.validate_layout()`이 검사한다.

## 기둥 → ArUco 마커 ID (PILL_MARKER_ID)

각 기둥 칸에 실제로 붙여 놓은 마커 ID. **격자와 카메라 영상을 잇는 유일한 연결 고리**이며,
자동 생성이 불가능한 유일한 표다. 마커를 옮기면 반드시 함께 고칠 것.

| 격자 (row, col) | 마커 ID | | 격자 (row, col) | 마커 ID |
|:---:|:---:|:---:|:---:|:---:|
| (1, 1) | 1  | | (1, 5) | 6  |
| (3, 1) | 2  | | (3, 5) | 7  |
| (5, 1) | 3  | | (5, 5) | 8  |
| (7, 1) | 4  | | (7, 5) | 9  |
| (9, 1) | 5  | | (9, 5) | 10 |
| (3, 3) | 11 | | (5, 3) | 12 |
| (7, 3) | 13 | | | |

역방향 매핑은 `MARKER_ID_CELL`.

## 주차 구역 좌표 매핑 (spot_map)

각 주차 구역은 **1칸**이며 좌표는 `(row, col)` 형식.
A = 왼쪽 열, B = 오른쪽 열, C = 중앙 섬. 번호는 **입구에서 가까운 순**.

| 구역 ID | 좌표 | | 구역 ID | 좌표 | | 구역 ID | 좌표 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A-1 | (2, 1) | | B-1 | (2, 5) | | C-1 | (4, 3) |
| A-2 | (4, 1) | | B-2 | (4, 5) | | C-2 | (6, 3) |
| A-3 | (6, 1) | | B-3 | (6, 5) | | | |
| A-4 | (8, 1) | | B-4 | (8, 5) | | | |

역방향 매핑은 `coord_to_spot`. `spot_status`의 키도 이 목록과 정확히 일치해야 한다.

## 입출구 좌표

- **GATE1_POS**: `(0, 2)` — 입구. 왼쪽 세로 통로 위. 경로 안내의 시작점
- **GATE2_POS**: `(0, 4)` — 출구. 오른쪽 세로 통로 위

## 주요 함수

### `get_rows()` / `get_cols()`
그리드 맵의 행 수 / 열 수를 반환.

### `is_valid_pos(row, col, open_spot=None)`
해당 좌표가 맵 범위 안이고 차량이 지나갈 수 있는지 확인.
`WALL`과 `PILL`은 항상 통과 불가. `SPOT`은 `open_spot`으로 지정된 구역만 통과 가능.

### `get_all_spot_ids()`
정의된 모든 주차 구역 ID 리스트를 반환.

### `get_spot_entry_coord(spot_id)`
주차 구역의 진입 좌표를 반환. 경로 탐색 시 목적지로 사용.

## 관련 모듈

| 모듈 | 역할 |
|---|---|
| `logic/C02_lot_layout.py` | 이 격자를 실좌표(cm)로 변환. 마커·자리·입출구 좌표 생성 |
| `logic/C01_path_planner.py` | 이 격자를 점유 격자로 바꿔 A\* 경로 계산 |
| `data/car_data.py` | `spot_status`를 읽어 빈자리/점유 조회 |
| `logic/A01_parking_manager.py` | 빈자리 배정 및 입출차 처리 |
