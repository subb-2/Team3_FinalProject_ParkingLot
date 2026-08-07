"""
A00_uart_rx : Zybo(FPGA) → Jetson 차량번호 수신 모듈  [Wi-Fi(TCP) 버전]

기존 UART(시리얼) 방식을 Wi-Fi 소켓 통신으로 대체한 버전.
- Jetson  : TCP 서버 (입차/출차 포트를 각각 열고 대기)
- Zybo    : TCP 클라이언트 (Wi-Fi 모듈로 AP에 접속 후 Jetson IP:PORT로 접속)

전송 데이터 포맷은 UART 버전과 동일한 2바이트 바이너리를 유지한다.
  byte1 상위4비트 = 1번째 자리, byte1 하위4비트 = 2번째 자리
  byte2 상위4비트 = 3번째 자리, byte2 하위4비트 = 4번째 자리
  예) 0x12 0x34  ->  "1234"

기존 UART 코드는 아래쪽 '[UART 버전 원본 코드]' 블록에 주석으로 보존해 두었다.
"""

# 입차 중복 막기
# 입차 안했는데 출차 하는 경우 (정보없음 처리)


import socket
import threading
import time
import datetime
import sys
import os

# import serial   # [UART 버전] pyserial - Wi-Fi 버전에서는 사용하지 않음

# 상위 디렉토리(python_code)를 import 경로에 추가하여 'logic' 패키지를 인식할 수 있도록 함
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logic.A01_parking_manager import handle_car_entry, remove_car
from logic.B02_car_mot import enqueue_car_number


def get_wifi_config():
    """
    Wi-Fi(TCP) 통신에 필요한 설정값을 반환.

    [주의] wifi_ssid / wifi_password 는 Zybo의 Wi-Fi 모듈이 AP에 접속할 때 쓰는 값이다.
           Jetson 본체는 OS(nmcli 등)로 이미 같은 공유기에 붙어 있어야 하며,
           이 파이썬 코드가 SSID/비밀번호로 직접 Wi-Fi에 접속하지는 않는다.
           (Zybo 펌웨어에 값을 넣을 때 참조하도록 여기 기록해 둔 것)
    """
    return {
        # --- 접속할 Wi-Fi AP 정보 (팀 ipTIME 공유기 / Zybo 측 설정용 참조값) ---
        # PC(유선 LAN 포트) - 공유기 - 젯슨/Zybo(무선) 가 모두 192.168.0.x 한 대역.
        'wifi_ssid': 'kcci603_2.4g',
        'wifi_password': 'kcci603_2.4g',

        # Zybo가 접속할 젯슨의 무선 IP (고정 설정)
        'jetson_ip': '192.168.0.50',

        # --- Jetson TCP 서버 설정 ---
        'host': '0.0.0.0',      # 0.0.0.0 = 모든 네트워크 인터페이스에서 수신
        'entry_port': 5001,     # 입차(Zybo1)가 접속할 포트
        'exit_port': 5002,      # 출차(Zybo2)가 접속할 포트
        'timeout': 1,           # accept/recv 대기 시간(초). 종료 응답성에 사용

        # --- [UART 버전 설정] 보존 ---
        # 'entry_port': '/dev/ttyTHS1',   # 입차(Zybo1) 포트
        # 'exit_port':  '/dev/ttyTHS2',   # 출차(Zybo2) 포트
        # 'baud_rate':  115200,           # 통신 속도
        # 'timeout':    1,                # 응답 없으면 종료 하는 시간
    }


# 기존 이름으로 호출하던 코드 호환용 별칭
get_uart_config = get_wifi_config


def parse_car_id(raw_data):
    """2바이트 바이너리 데이터에서 4자리 차량 번호를 추출합니다."""
    byte1 = raw_data[0]
    byte2 = raw_data[1]

    digit1 = (byte1 >> 4) & 0x0F
    digit2 = byte1 & 0x0F
    digit3 = (byte2 >> 4) & 0x0F
    digit4 = byte2 & 0x0F

    return f"{digit1}{digit2}{digit3}{digit4}", byte1, byte2


def build_car_packet(car_id):
    """4자리 차량번호 문자열 -> 2바이트 패킷. (테스트 송신용 / parse_car_id의 역연산)"""
    d = f"{int(car_id):04d}"
    byte1 = (int(d[0]) << 4) | int(d[1])
    byte2 = (int(d[2]) << 4) | int(d[3])
    return bytes([byte1, byte2])


def _recv_exact(conn, size):
    """소켓에서 정확히 size 바이트를 모을 때까지 읽는다. 연결이 끊기면 None 반환."""
    buf = b''
    while len(buf) < size:
        try:
            chunk = conn.recv(size - len(buf))
        except socket.timeout:
            continue          # 아직 안 왔을 뿐이므로 계속 대기
        except OSError:
            return None       # 소켓이 닫힘
        if not chunk:
            return None       # 상대방이 연결을 종료함
        buf += chunk
    return buf


# 수신 이벤트 구독
# UI가 '방금 무엇을 받았는지'를 알아야 하는데, 지금까지는 print로만 남았다.
# 화면에 띄우려면 그 사실이 프로그램 안에 남아야 하므로 구독 창구를 둔다.
#
# 통신 코드가 UI를 직접 부르지 않게 하려고 콜백 방식으로 만들었다.
# A00은 '누가 듣는지' 모른 채 사실만 알리고, 듣는 쪽이 알아서 처리한다.
_rx_listeners = []


def add_rx_listener(callback):
    """
    패킷을 수신할 때마다 호출될 함수를 등록한다.

    Args:
        callback: callback(event_dict) 형태. event_dict의 내용은
                  _notify_rx의 docstring 참고.
    """
    _rx_listeners.append(callback)
    return callback


def _notify_rx(role, car_id, byte1, byte2, result=None):
    """
    수신 사실을 등록된 리스너 전원에게 알린다.

    리스너에서 예외가 나도 수신 루프를 멈추지 않는다. 화면 갱신이 실패했다고
    입출차 처리까지 죽으면 안 되기 때문이다.

    전달되는 event_dict:
        role     : 'entry'(입차) 또는 'exit'(출차)
        car_id   : 4자리 차량번호 문자열
        raw_hex  : "0x12 0x34" 형태의 원본 바이트 표기
        time     : 수신 시각 (datetime)
        result   : 입차면 handle_car_entry의 결과 dict, 출차면 None
    """
    event = {
        "role": role,
        "car_id": car_id,
        "raw_hex": f"0x{byte1:02X} 0x{byte2:02X}",
        "time": datetime.datetime.now(),
        "result": result,
    }
    for callback in _rx_listeners:
        try:
            callback(event)
        except Exception as e:
            print(f"[경고] 수신 리스너에서 오류가 발생했습니다: {e}")


def _process_packet(role, raw_data):
    """수신한 2바이트를 파싱해서 입차/출차 로직으로 넘긴다."""
    car_id, b1, b2 = parse_car_id(raw_data)

    if role == 'entry':
        receive_time = datetime.datetime.now().replace(second=0, microsecond=0)
        print(f"\n[입차수신(Dec)] (Hex: 0x{b1:02X} 0x{b2:02X}) -> {car_id}")

        # 입차 처리 호출 (차량 종류에 맞는 가장 가까운 빈자리를 배정)
        result = handle_car_entry(car_id, receive_time)

        # 배정에 실패했으면(해당 종류 자리가 다 참 등) 추적 대기열에 넣지 않는다.
        # 목표 구역이 없으면 C00이 안내할 곳도 없기 때문이다.
        # result["reason"]으로 "자리 없음" 안내 화면을 띄우면 된다.
        if result["success"]:
            enqueue_car_number(car_id)
        else:
            print(f"[입차수신] 안내를 시작하지 않습니다. ({result['reason']})")

        # 배정 결과까지 확정된 뒤에 알린다. UI가 "1234 -> A-1 배정"까지
        # 한 번에 띄울 수 있어야 하기 때문이다.
        _notify_rx(role, car_id, b1, b2, result)
    else:
        print(f"\n[출차수신(Dec)] (Hex: 0x{b1:02X} 0x{b2:02X}) -> {car_id}")

        # 출차 및 요금 계산 호출
        remove_car(car_id)
        _notify_rx(role, car_id, b1, b2, None)


def _serve_role(role, label, host, port, timeout, stop_event):
    """
    한 역할(입차 또는 출차)에 대한 TCP 서버 루프.
    Zybo가 끊었다 다시 붙어도 계속 받을 수 있도록 accept를 반복한다.
    """
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        srv.settimeout(timeout)
    except OSError as e:
        print(f"[에러] {label} 포트({port})를 열 수 없습니다: {e}")
        return

    print(f"[{label}] TCP 서버 대기 중: {host}:{port}")

    while not stop_event.is_set():
        # 1) Zybo 접속 대기
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        print(f"[{label}] Zybo 접속됨: {addr[0]}:{addr[1]}")
        conn.settimeout(timeout)

        # 2) 연결이 유지되는 동안 2바이트씩 계속 수신
        try:
            while not stop_event.is_set():
                raw_data = _recv_exact(conn, 2)
                if raw_data is None:
                    break
                _process_packet(role, raw_data)
        except Exception as e:
            print(f"[{label}] 수신 중 오류: {e}")
        finally:
            conn.close()
            print(f"[{label}] Zybo 연결이 끊어졌습니다. 재접속 대기...")

    srv.close()
    print(f"[{label}] 서버 소켓을 닫았습니다.")


def uart_rx_main():
    """
    Zybo 보드(입차/출차)와 Wi-Fi(TCP) 통신을 통해 데이터를 수신하고 파싱하는 모듈.

    함수 이름은 기존 호출부(B_main / C_main의 ENABLE_UART) 호환을 위해 그대로 둔다.
    입차/출차를 각각 별도 스레드에서 받으므로 한쪽이 다른 쪽을 막지 않는다.
    """
    config = get_wifi_config()
    HOST = config['host']
    ENTRY_PORT = config['entry_port']
    EXIT_PORT = config['exit_port']
    TIMEOUT = config['timeout']

    print(f"[INFO] Wi-Fi AP: {config['wifi_ssid']} (Zybo가 이 AP에 접속되어 있어야 함)")
    print(f"[INFO] Zybo 접속 주소: {config['jetson_ip']}:{ENTRY_PORT}(입차) / {EXIT_PORT}(출차)")

    stop_event = threading.Event()

    threads = [
        threading.Thread(target=_serve_role,
                         args=('entry', '입차', HOST, ENTRY_PORT, TIMEOUT, stop_event),
                         daemon=True),
        threading.Thread(target=_serve_role,
                         args=('exit', '출차', HOST, EXIT_PORT, TIMEOUT, stop_event),
                         daemon=True),
    ]
    for t in threads:
        t.start()

    print("Zybo로부터 데이터 수신 대기 중 (Ctrl+C를 누르면 종료)...")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n수신 프로그램을 강제 종료합니다.")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=TIMEOUT + 1)


def send_test_packet(role, car_id, host='127.0.0.1'):
    """
    Zybo 없이 테스트할 때 쓰는 송신 헬퍼.
    사용법:  python A00_uart_rx.py send entry 1234
    """
    config = get_wifi_config()
    port = config['entry_port'] if role == 'entry' else config['exit_port']
    packet = build_car_packet(car_id)

    with socket.create_connection((host, port), timeout=3) as sock:
        sock.sendall(packet)
    print(f"[테스트송신] {role} {car_id} -> {host}:{port} "
          f"(Hex: 0x{packet[0]:02X} 0x{packet[1]:02X})")


if __name__ == "__main__":
    # python A00_uart_rx.py                    -> 수신 서버 실행
    # python A00_uart_rx.py send entry 1234    -> 테스트 패킷 송신
    if len(sys.argv) >= 4 and sys.argv[1] == 'send':
        send_test_packet(sys.argv[2], sys.argv[3])
    else:
        uart_rx_main()


# =============================================================================
# [UART 버전 원본 코드] - Wi-Fi 전환 전 시리얼 통신 구현. 필요 시 참고/복구용.
# =============================================================================
# import serial
#
# def get_uart_config():
#     """
#     UART 통신에 필요한 설정값을 반환.
#     - 윈도우 환경: 'COM3', 'COM4' 등
#     - 리눅스/젯슨 환경: '/dev/ttyUSB0', '/dev/ttyTHS1', '/dev/ttyACM0' 등
#     """
#     return {
#         'entry_port': '/dev/ttyTHS1',   # 입차(Zybo1) 포트
#         'exit_port': '/dev/ttyTHS2',    # 출차(Zybo2) 포트 (필요 시 실제 포트로 수정)
#         'baud_rate': 115200,            # 통신 속도
#         'timeout': 1                    # 응답 없으면 종료 하는 시간
#     }
#
# def uart_rx_main():
#     """
#     Zybo 보드(입차/출차)와 UART 통신을 통해 데이터를 수신하고 파싱하는 모듈.
#     """
#     config = get_uart_config()
#     ENTRY_PORT = config['entry_port']
#     EXIT_PORT = config['exit_port']
#     BAUD_RATE = config['baud_rate']
#     TIMEOUT = config['timeout']
#
#     ser_entry = None
#     ser_exit = None
#
#     # 입차 포트 연결 시도
#     try:
#         ser_entry = serial.Serial(ENTRY_PORT, BAUD_RATE, timeout=TIMEOUT)
#         print(f"[입차] UART 연결 성공: {ENTRY_PORT} ({BAUD_RATE} bps)")
#     except serial.SerialException as e:
#         print(f"[경고] 입차 포트({ENTRY_PORT})를 열 수 없습니다. 입차 처리가 비활성화됩니다.")
#
#     # 출차 포트 연결 시도
#     try:
#         ser_exit = serial.Serial(EXIT_PORT, BAUD_RATE, timeout=TIMEOUT)
#         print(f"[출차] UART 연결 성공: {EXIT_PORT} ({BAUD_RATE} bps)")
#     except serial.SerialException as e:
#         print(f"[경고] 출차 포트({EXIT_PORT})를 열 수 없습니다. 출차 처리가 비활성화됩니다.")
#
#     if not ser_entry and not ser_exit:
#         print("[에러] 연결된 UART 포트가 하나도 없습니다. 프로그램을 종료합니다.")
#         return
#
#     print("Zybo로부터 데이터 수신 대기 중 (Ctrl+C를 누르면 종료)...")
#
#     try:
#         while True:
#             # 1. 입차(Entry) 데이터 확인
#             if ser_entry and ser_entry.in_waiting >= 1:
#                 raw_data = ser_entry.read(2)
#                 if len(raw_data) == 2:
#                     car_id, b1, b2 = parse_car_id(raw_data)
#                     receive_time = datetime.datetime.now().replace(second=0, microsecond=0)
#                     print(f"\n[입차수신(Dec)] (Hex: 0x{b1:02X} 0x{b2:02X}) -> {car_id}")
#
#                     # 입차 처리 호출
#                     handle_car_entry(car_id, receive_time)
#                     enqueue_car_number(car_id)
#
#             # 2. 출차(Exit) 데이터 확인
#             if ser_exit and ser_exit.in_waiting >= 1:
#                 raw_data = ser_exit.read(2)
#                 if len(raw_data) == 2:
#                     car_id, b1, b2 = parse_car_id(raw_data)
#                     print(f"\n[출차수신(Dec)] (Hex: 0x{b1:02X} 0x{b2:02X}) -> {car_id}")
#
#                     # 출차 및 요금 계산 호출
#                     remove_car(car_id)
#
#             # CPU 과부하 방지를 위한 짧은 대기
#             time.sleep(0.01)
#
#     except KeyboardInterrupt:
#         print("\n수신 프로그램을 강제 종료합니다.")
#     finally:
#         if ser_entry and ser_entry.is_open:
#             ser_entry.close()
#             print("입차 시리얼 포트가 닫혔습니다.")
#         if ser_exit and ser_exit.is_open:
#             ser_exit.close()
#             print("출차 시리얼 포트가 닫혔습니다.")
