# 주차 차량 관리 모듈 명세서 (README)
# car_data.py

## 개요

차량의 **입출차 처리**, **주차 시간 기록**, **요금 계산**, **차량 정보 조회** 등을 담당하는 차량 관리 파일.

## 요금 체계

`get_fee_config()` 함수에서 설정값을 관리합니다.

| 항목                  | 기본값     | 설명 |
|:---                   |:---:      |:---|
| `min_time`            | 10분      | 무료 요금이 적용되는 시간 |
| `base_minutes`        | 30분      | 기본 요금이 적용되는 시간 |
| `extra_per_minutes`   | 10분      | 추가 요금 단위 시간 |
| `base_fee`            | 1,000원   | 기본 요금 단위 금액|
| `extra_fee`           | 500원     | 추가 요금 단위 금액 |
| `max_fee`             | 50,000원  | 최대 요금 단위 금액 |

## 데이터 구조

### cars_info (차량 번호 기준 메인 저장소)
```python
{
    "1234": {"spot_id": "A-1", "entry_time": datetime(...)},
    "5678": {"spot_id": "C-2", "entry_time": datetime(...)},
}
```

### spot_status (주차 구역 상태, 빠른 검색용 보조 데이터)
`map_data.py`에서 상태를 관리하며 `car_data.py`에서 import하여 사용.
```python
{
    "A-1": "full",
    "A-2": "empty",
    # ... 전체 주차 구역
}
```

## 주요 함수 (데이터 조회용)

### `get_occupied_spots()`
현재 `"full"` 상태인 주차 중인 구역 ID의 set을 반환.

### `get_empty_spots()`
현재 `"empty"` 상태인 비어있는 구역 ID의 set을 반환. `logic.A01_parking_manager`의 빈자리 배정에 사용.

### `get_car_info(car_id)`
차량 번호로 주차 정보를 조회. (`cars_info` 직접 조회)

### `get_spot_info(spot_id)`
구역 ID로 주차 정보를 조회. (`spot_status` 확인 후 `cars_info` 역방향 검색)
