import sys
import os

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.car_data import get_fee_config

# 요금 계산 로직
def calculate_fee(duration_minutes):
    """
    주차 시간(분)을 기반으로 요금을 계산.
    
    Args:
        duration_minutes: 주차 시간 (분 단위, 정수)
    
    Returns:
        계산된 주차 요금 (원, 정수)
    """
    config = get_fee_config()

    # 10분 이하일 시 0원 (회차 무료)
    if duration_minutes <= config['min_time']:
        return 0
    elif duration_minutes <= config['base_minutes']:
        return config['base_fee']
    
    # 기본 시간 초과분에 대한 추가 요금 계산
    extra_minutes = duration_minutes - config['base_minutes']

    # 올림 처리: 10분 단위로 올림하여 추가 요금 부과
    extra_units = (extra_minutes + config['extra_per_minutes'] - 1) // config['extra_per_minutes']
    total_fee = config['base_fee'] + (extra_units * config['extra_fee'])

    # 일일 최대 요금(상한선) 적용
    if total_fee >= config['max_fee']:
        return config['max_fee'] 

    return total_fee


# 테스트
# if __name__ == "__main__":
    # test_cases = [5, 10, 20, 30, 40, 60, 90, 120, 1000, 2100]
    # print("========== 요금 계산 테스트 ==========")
    # for minutes in test_cases:
        # fee = calculate_fee(minutes)
        # print(f"  {minutes:>4d}분 주차 → {fee:,}원")
    # print("=====================================") 