import serial
import time
import datetime
from logic.A01_parking_manager import handle_car_entry, remove_car

def get_uart_config():
    """
    UART 통신에 필요한 설정값을 반환. 
    - 윈도우 환경: 'COM3', 'COM4' 등
    - 리눅스/젯슨 환경: '/dev/ttyUSB0', '/dev/ttyTHS1', '/dev/ttyACM0' 등
    """
    return {
        'entry_port': '/dev/ttyTHS1',   # 입차(Zybo1) 포트 
        'exit_port': '/dev/ttyTHS2',    # 출차(Zybo2) 포트 (필요 시 실제 포트로 수정)
        'baud_rate': 115200,            # 통신 속도    
        'timeout': 1                    # 응답 없으면 종료 하는 시간
    }

def parse_car_id(raw_data):
    """2바이트 바이너리 데이터에서 4자리 차량 번호를 추출합니다."""
    byte1 = raw_data[0]
    byte2 = raw_data[1]
    
    digit1 = (byte1 >> 4) & 0x0F
    digit2 = byte1 & 0x0F 
    digit3 = (byte2 >> 4) & 0x0F
    digit4 = byte2 & 0x0F
    
    return f"{digit1}{digit2}{digit3}{digit4}", byte1, byte2

def uart_rx_main():
    """
    Zybo 보드(입차/출차)와 UART 통신을 통해 데이터를 수신하고 파싱하는 모듈.
    """
    config = get_uart_config()
    ENTRY_PORT = config['entry_port']
    EXIT_PORT = config['exit_port']
    BAUD_RATE = config['baud_rate']
    TIMEOUT = config['timeout']
    
    ser_entry = None
    ser_exit = None

    # 입차 포트 연결 시도
    try:
        ser_entry = serial.Serial(ENTRY_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"[입차] UART 연결 성공: {ENTRY_PORT} ({BAUD_RATE} bps)")
    except serial.SerialException as e:
        print(f"[경고] 입차 포트({ENTRY_PORT})를 열 수 없습니다. 입차 처리가 비활성화됩니다.")

    # 출차 포트 연결 시도
    try:
        ser_exit = serial.Serial(EXIT_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"[출차] UART 연결 성공: {EXIT_PORT} ({BAUD_RATE} bps)")
    except serial.SerialException as e:
        print(f"[경고] 출차 포트({EXIT_PORT})를 열 수 없습니다. 출차 처리가 비활성화됩니다.")

    if not ser_entry and not ser_exit:
        print("[에러] 연결된 UART 포트가 하나도 없습니다. 프로그램을 종료합니다.")
        return

    print("Zybo로부터 데이터 수신 대기 중 (Ctrl+C를 누르면 종료)...")

    try:
        while True:
            # 1. 입차(Entry) 데이터 확인
            if ser_entry and ser_entry.in_waiting >= 1:
                raw_data = ser_entry.read(2)
                if len(raw_data) == 2:
                    car_id, b1, b2 = parse_car_id(raw_data)
                    receive_time = datetime.datetime.now().replace(second=0, microsecond=0)
                    print(f"\n[입차수신(Dec)] (Hex: 0x{b1:02X} 0x{b2:02X}) -> {car_id}")
                    
                    # 입차 처리 호출
                    handle_car_entry(car_id, receive_time)

            # 2. 출차(Exit) 데이터 확인
            if ser_exit and ser_exit.in_waiting >= 1:
                raw_data = ser_exit.read(2)
                if len(raw_data) == 2:
                    car_id, b1, b2 = parse_car_id(raw_data)
                    print(f"\n[출차수신(Dec)] (Hex: 0x{b1:02X} 0x{b2:02X}) -> {car_id}")
                    
                    # 출차 및 요금 계산 호출
                    remove_car(car_id)
                    
            # CPU 과부하 방지를 위한 짧은 대기
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n수신 프로그램을 강제 종료합니다.")
    finally:
        if ser_entry and ser_entry.is_open:
            ser_entry.close()
            print("입차 시리얼 포트가 닫혔습니다.")
        if ser_exit and ser_exit.is_open:
            ser_exit.close()
            print("출차 시리얼 포트가 닫혔습니다.")

if __name__ == "__main__":
    uart_rx_main()
