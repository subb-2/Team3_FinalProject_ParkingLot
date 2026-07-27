import serial
import time
import sys
import threading

# ============================================================
# 주차장 설정
# ============================================================
TOTAL_PARKING_SPOTS = 10  # 전체 주차 자리 수 (필요에 따라 조정)

# UART 설정
UART_PORT = "/dev/ttyTHS1"  # Jetson Orin Nano 기본 UART 포트
UART_BAUDRATE = 115200
UART_TIMEOUT = 1  # 초


class ParkingManager:
    """주차장 빈자리 관리 및 차량 ID 매칭 클래스"""

    def __init__(self, total_spots=TOTAL_PARKING_SPOTS):
        self.total_spots = total_spots
        # 주차 자리 상태: {spot_id: car_info or None}
        # spot_id는 1부터 시작
        self.spots = {i: None for i in range(1, total_spots + 1)}
        # 차량 번호 → 할당된 자리 매핑
        self.car_to_spot = {}
        # Track ID 카운터
        self.next_track_id = 1
        # 차량 정보 저장: {track_id: {car_num, spot_id, ...}}
        self.tracked_cars = {}
        # 스레드 락 (동시 접근 보호)
        self.lock = threading.Lock()

    def get_empty_spots(self):
        """빈자리 목록 반환 (번호 오름차순 = 가장 가까운 자리부터)"""
        with self.lock:
            return [spot_id for spot_id, car in self.spots.items() if car is None]

    def get_occupied_spots(self):
        """사용 중인 자리 목록 반환"""
        with self.lock:
            return {spot_id: car for spot_id, car in self.spots.items() if car is not None}

    def assign_spot(self, car_num):
        """
        차량에 가장 가까운 빈자리를 매칭하고 고유 Track ID를 부여한다.

        Parameters:
            car_num (int): 차량 번호 (하위 4비트에서 추출된 값)

        Returns:
            dict: 매칭 결과 {car_num, id, spot_id} 또는 None (빈자리 없음)
        """
        with self.lock:
            # 이미 등록된 차량인지 확인
            if car_num in self.car_to_spot:
                spot_id = self.car_to_spot[car_num]
                track_id = self.tracked_cars[car_num]["id"]
                print(f"[INFO] 차량 {car_num} 은(는) 이미 자리 {spot_id}에 배정됨 (ID: {track_id})")
                return self.tracked_cars[car_num]

            # 빈자리 찾기 (가장 가까운 = 번호가 작은 자리)
            empty = [s for s, c in self.spots.items() if c is None]
            if not empty:
                print("[WARN] 빈자리가 없습니다!")
                return None

            # 가장 가까운 자리 배정
            spot_id = min(empty)
            track_id = self.next_track_id
            self.next_track_id += 1

            car_info = {
                "car_num": car_num,
                "car_num_bin": format(car_num, '08b'),
                "id": track_id,
                "spot_id": spot_id,
                "entry_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            self.spots[spot_id] = car_info
            self.car_to_spot[car_num] = spot_id
            self.tracked_cars[car_num] = car_info

            return car_info

    def release_spot(self, car_num):
        """
        차량 출차 처리: 자리를 비우고 매핑 정보를 제거한다.

        Parameters:
            car_num (int): 출차할 차량 번호

        Returns:
            bool: 성공 여부
        """
        with self.lock:
            if car_num not in self.car_to_spot:
                print(f"[WARN] 차량 {car_num} 은(는) 등록되어 있지 않습니다.")
                return False

            spot_id = self.car_to_spot[car_num]
            self.spots[spot_id] = None
            del self.car_to_spot[car_num]
            del self.tracked_cars[car_num]

            print(f"[INFO] 차량 {car_num} 출차 완료. 자리 {spot_id} 반납.")
            return True

    def get_status(self):
        """현재 주차장 상태 출력"""
        with self.lock:
            occupied = sum(1 for c in self.spots.values() if c is not None)
            empty = self.total_spots - occupied
            return {
                "total": self.total_spots,
                "occupied": occupied,
                "empty": empty,
                "spots": dict(self.spots),
            }

    def print_status(self):
        """현재 주차장 상태를 터미널에 출력"""
        status = self.get_status()
        print("\n========== 주차장 현황 ==========")
        print(f"  전체: {status['total']}  |  사용: {status['occupied']}  |  빈자리: {status['empty']}")
        print("-" * 40)
        for spot_id in range(1, self.total_spots + 1):
            car = status["spots"][spot_id]
            if car:
                print(f"  자리 {spot_id:>2d}: 🚗 차량 {car['car_num']} (ID: {car['id']}, 입차: {car['entry_time']})")
            else:
                print(f"  자리 {spot_id:>2d}: ⬜ 빈자리")
        print("=" * 40 + "\n")


def parse_uart_data(raw_byte):
    """
    UART로 수신된 8비트 데이터에서 차량 번호를 추출한다.

    입력 형식: 8비트 (예: 0000_1234)
    - 하위 [3]~[0] 비트 = 차량 번호

    Parameters:
        raw_byte (int): UART 수신 바이트 (0~255)

    Returns:
        int: 차량 번호 (하위 4비트)
    """
    car_num = raw_byte & 0x0F  # 하위 4비트 추출
    print(f"[UART] 수신 원본: {format(raw_byte, '08b')} (0x{raw_byte:02X})")
    print(f"[UART] 추출 차량 번호: {car_num} (하위 4비트: {format(car_num, '04b')})")
    return car_num


def uart_receive_loop(ser, parking_manager):
    """
    UART 수신 루프: ZYBO로부터 데이터를 수신하여 차량 매칭을 수행한다.
    """
    print("[INFO] UART 수신 대기 중...")

    while True:
        try:
            if ser.in_waiting > 0:
                raw_data = ser.read(1)
                if raw_data:
                    raw_byte = raw_data[0]

                    # 차량 번호 추출
                    car_num = parse_uart_data(raw_byte)

                    if car_num == 0:
                        print("[INFO] 차량 번호 0: 무시 (유효하지 않은 데이터)")
                        continue

                    # 빈자리 매칭
                    result = parking_manager.assign_spot(car_num)

                    if result:
                        print(f"\n[결과] 차량 매칭 완료:")
                        print(f"  car_num : {result['car_num_bin']}")
                        print(f"  id      : {result['id']}")
                        print(f"  spot    : {result['spot_id']}")

                    # 주차장 현황 출력
                    parking_manager.print_status()

            time.sleep(0.01)  # CPU 과부하 방지

        except KeyboardInterrupt:
            print("\n[INFO] UART 수신 종료.")
            break
        except Exception as e:
            print(f"[ERROR] UART 수신 오류: {e}")
            time.sleep(1)


def demo_mode(parking_manager):
    """
    UART 없이 동작을 테스트하기 위한 데모 모드.
    키보드로 차량 번호를 입력받아 시뮬레이션한다.
    """
    print("\n===== 데모 모드 (UART 없이 테스트) =====")
    print("명령어:")
    print("  [숫자]    : 해당 번호의 차량 입차 (1~15)")
    print("  r [숫자]  : 해당 번호의 차량 출차")
    print("  s         : 주차장 현황 보기")
    print("  q         : 종료\n")

    while True:
        try:
            user_input = input("입력> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'q':
                print("[INFO] 데모 모드를 종료합니다.")
                break

            if user_input.lower() == 's':
                parking_manager.print_status()
                continue

            if user_input.lower().startswith('r '):
                # 출차 처리
                try:
                    car_num = int(user_input.split()[1])
                    parking_manager.release_spot(car_num)
                    parking_manager.print_status()
                except (ValueError, IndexError):
                    print("[WARN] 형식: r [차량번호]")
                continue

            # 입차 처리 - 8비트 시뮬레이션
            try:
                raw_value = int(user_input)
                if raw_value < 0 or raw_value > 255:
                    print("[WARN] 0~255 범위의 값을 입력하세요.")
                    continue

                car_num = parse_uart_data(raw_value)

                if car_num == 0:
                    print("[INFO] 차량 번호 0: 무시")
                    continue

                result = parking_manager.assign_spot(car_num)

                if result:
                    print(f"\n[결과] 차량 매칭 완료:")
                    print(f"  car_num : {result['car_num_bin']}")
                    print(f"  id      : {result['id']}")
                    print(f"  spot    : {result['spot_id']}")

                parking_manager.print_status()

            except ValueError:
                print("[WARN] 올바른 숫자를 입력하세요.")

        except KeyboardInterrupt:
            print("\n[INFO] 데모 모드를 종료합니다.")
            break


def main():
    print("==============================================")
    print(" 2.5단계 : 고유 ID 부여 및 빈자리 매칭")
    print("==============================================")
    print("모드 선택:")
    print("  1: UART 수신 모드 (ZYBO 연결)")
    print("  2: 데모 모드 (키보드 입력 테스트)")

    mode = sys.argv[1] if len(sys.argv) > 1 else '2'

    # 주차장 매니저 초기화
    parking_manager = ParkingManager(total_spots=TOTAL_PARKING_SPOTS)
    print(f"\n[INFO] 주차장 초기화 완료 (총 {TOTAL_PARKING_SPOTS}자리)")

    if mode == '1':
        # UART 모드
        print(f"[INFO] UART 포트: {UART_PORT}, 속도: {UART_BAUDRATE} bps")
        try:
            ser = serial.Serial(
                port=UART_PORT,
                baudrate=UART_BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=UART_TIMEOUT,
            )
            print(f"[INFO] UART 포트 열림: {ser.name}")
            uart_receive_loop(ser, parking_manager)
            ser.close()
        except serial.SerialException as e:
            print(f"[ERROR] UART 포트를 열 수 없습니다: {e}")
            print("[INFO] 데모 모드로 전환합니다.\n")
            demo_mode(parking_manager)
    else:
        # 데모 모드
        demo_mode(parking_manager)


if __name__ == "__main__":
    main()
