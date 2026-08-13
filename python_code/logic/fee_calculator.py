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


# 테스트용 메인 (단독 실행 시 구간별 요금 확인)
if __name__ == "__main__":
    cfg = get_fee_config()
    print("=" * 40)
    print(" 요금 계산")
    print(f"  {cfg['min_time']}분 이하 무료 / "
          f"{cfg['base_minutes']}분까지 {cfg['base_fee']:,}원 / "
          f"이후 {cfg['extra_per_minutes']}분마다 {cfg['extra_fee']:,}원 / "
          f"상한 {cfg['max_fee']:,}원")
    print("=" * 40)

    for minutes in (5, 10, 11, 30, 31, 40, 60, 90, 120, 1000, 2100):
        print(f"  {minutes:>4d}분  ->  {calculate_fee(minutes):>7,}원") 