import serial
import time

def get_uart_config():
    """
    UART 통신에 필요한 설정값을 반환합니다.
    환경에 맞게 포트 번호와 보드레이트를 수정하세요.
    - 윈도우 환경: 'COM3', 'COM4' 등
    - 리눅스/젯슨 환경: '/dev/ttyUSB0', '/dev/ttyTHS1', '/dev/ttyACM0' 등
    """
    return {
        'port': '/dev/ttyUSB0',
        'baud_rate': 115200,
        'timeout': 1
    }

def uart_rx_main():
    """
    Zybo 보드와 UART 통신을 통해 16비트(2바이트) 원시 데이터를 수신하고 파싱하는 모듈입니다.
    """
    # 외부 config 함수에서 설정값 불러오기
    config = get_uart_config()
    PORT = config['port']
    BAUD_RATE = config['baud_rate']
    TIMEOUT = config['timeout']
    
    try:
        # 시리얼 포트 열기
        ser = serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"UART 연결 성공: {PORT} ({BAUD_RATE} bps)")
        print("Zybo로부터 데이터 수신 대기 중 (Ctrl+C를 누르면 종료됩니다)...")

        while True:
            # 2바이트(16비트) 이상 수신되었을 때 처리
            if ser.in_waiting >= 2:
                # 버퍼에서 2바이트를 순수 바이너리 형태로 읽어오기
                raw_data = ser.read(2)
                
                # 각각의 바이트 분리
                byte1 = raw_data[0] # 첫 번째 수신 바이트
                byte2 = raw_data[1] # 두 번째 수신 바이트
                
                # 비트 연산(Shift 및 AND)을 통해 4비트씩 잘라내어 숫자 추출   
                digit1 = (byte1 >> 4) & 0x0F
                digit2 = byte1 & 0x0F 
                digit3 = (byte2 >> 4) & 0x0F
                digit4 = byte2 & 0x0F
                
                # 추출한 4개의 숫자를 문자열로 합치기
                received_number_str = f"{digit1}{digit2}{digit3}{digit4}"

                # debug output 
                print(f"[수신 원본(Dec)] : (Hex: 0x{byte1:02X} 0x{byte2:02X}) -> [파싱 결과] : {received_number_str}")
                
                # TODO: 여기서 수신된 데이터를 파싱하거나 다른 함수(예: id_manage)로 전달하는 로직 추가
                    
            # CPU 과부하 방지를 위한 짧은 대기
            time.sleep(0.01)

    except serial.SerialException as e:
        print(f"[에러] 시리얼 포트를 열 수 없습니다. 포트 이름과 권한을 확인하세요.\n상세: {e}")
    except KeyboardInterrupt:
        print("\n수신 프로그램을 강제 종료합니다.")
    finally:
        # 프로그램 종료 시 포트 안전하게 닫기
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("시리얼 포트가 닫혔습니다.")

if __name__ == "__main__":
    uart_rx_main()



0001230000000 
0000000000000 
0000000000000 
0000000000000 
