"""
A00_uart_rx : Zybo(FPGA) → Jetson 차량번호 수신 모듈  [Wi-Fi(TCP) 버전]

기존 UART(시리얼) 방식을 Wi-Fi 소켓 통신으로 대체한 버전.
- Jetson  : TCP 서버 (입차/출차 포트를 각각 열고 대기)
- Zybo    : TCP 클라이언트 (Wi-Fi 모듈로 AP에 접속 후 Jetson IP:PORT로 접속)

전송 데이터 포맷은 UART 버전과 동일한 2바이트 바이너리를 유지한다.
한 바이트에 두 자리를 담고, 한 바이트 안에서는 상위4비트가 앞자리다.
  0x12 -> "12"

두 바이트 중 어느 쪽이 먼저 오는지는 보드마다 다르다. 입차와 출차 Zybo의
펌웨어가 서로 반대라, 아래 BYTE_ORDER에 역할별로 적어 둔다.

시리얼 구현이 필요하면 git 이력에서 꺼낼 것. (Wi-Fi 전환 전 버전)

포트가 곧 역할이지만 예외를 하나 둔다. 이미 주차장 안에 있는 차의 번호가
입차 포트로 다시 들어오면 그 차는 나가는 중이므로 출차로 처리한다.
(아래 ROLE_RESOLVE 주석 참고)

TODO
  - 입차 없이 출차만 들어오면 '정보 없음'으로 처리
"""

import socket
import select
import threading
import time
import errno
import datetime
import sys
import os

# 상위 디렉토리(python_code)를 import 경로에 추가하여 'logic' 패키지를 인식할 수 있도록 함
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logic.A01_parking_manager import handle_car_entry, remove_car
from logic.B02_car_mot import enqueue_car_number
# 들어온 번호가 이미 주차장 안에 있는 차인지 보는 데 쓴다. (resolve_packet)
from data.car_data import cars_info


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
    }


# 역할별 바이트 순서. 실물로 확인한 값이다.
#   'low_first'  : 뒤 두 자리를 담은 바이트가 먼저 온다.  0x34 0x12 -> "1234"
#   'high_first' : 앞 두 자리를 담은 바이트가 먼저 온다.  0x12 0x34 -> "1234"
#
# 두 보드의 순서가 서로 다르다. 실물 번호판 8935로 양쪽을 다 확인했다.
#   입차 : 8935를 high_first로 읽어야 8935가 나온다.
#   출차 : 8935를 보냈는데 high_first로 읽으니 3589가 나왔다. 앞뒤 두 자리가
#          뒤집힌 값이므로 이 보드는 뒤 두 자리를 먼저 보낸다(low_first).
#
# 한동안 둘 다 high_first로 두었던 것은 실측이 아니었다. 시연 번호판이
# 1111, 2222처럼 앞뒤 두 자리가 같은 것뿐이라 뒤집혀도 같은 값이 나와
# 드러나지 않았을 뿐이다. 실물 번호판을 등록하고 나서야 보였다.
#
# 뒤집히면 등록되지 않은 번호가 된다. 입차에서는 조용히 일반 차량으로
# 처리되고(8935는 전기차인데 3589로 읽혀 일반 자리를 받았다), 출차에서는
# 그런 번호의 입차 기록이 없으므로 '입차 기록이 없습니다'로 거부된다.
# 한쪽 펌웨어가 순서를 바꾸면 그 역할만 여기서 되돌릴 것.
BYTE_ORDER = {
    'entry': 'high_first',      # 입차 Zybo (앞 두 자리를 먼저 보낸다)
    'exit': 'low_first',        # 출차 Zybo (뒤 두 자리를 먼저 보낸다)
}


# 들어온 패킷을 어느 쪽으로 처리할지 정하는 규칙.
#
# 포트만으로 정하면 안 되는 경우가 있다. 주차를 마친 차가 나가려고 움직일 때
# 입구 카메라가 그 번호판을 다시 읽어 입차 포트로 보내는 일이 그것이다.
# 그대로 입차로 처리하면 "이미 주차 중입니다"로 거부되고, 화면에는 붉은 실패
# 기록만 남는다. 정작 하려던 출차는 어디에도 뜨지 않는다.
#
# 이미 주차장 안에 있는 차의 번호가 또 들어왔다면 그 차는 나가는 중이다.
# 그렇게 보고 출차(5번 칸)로 넘긴다.
ROLE_RESOLVE = {
    # 위 규칙을 쓸지 여부. 끄면 포트가 곧 역할이다.
    'KNOWN_CAR_IS_EXIT': True,

    # 보드의 IP를 역할에 못박아 둔다. 두 보드가 같은 포트로 붙는 경우에 쓴다.
    # 터미널의 "[입차] Zybo 접속됨: 192.168.0.31:xxxxx" 줄에서 IP를 확인해
    # 여기 적으면, 그 IP에서 온 것은 어느 포트로 왔든 적힌 역할로 처리된다.
    #   'ROLE_BY_IP': {'192.168.0.31': 'exit'},
    'ROLE_BY_IP': {},
}


def parse_car_id(raw_data, role='entry'):
    """
    2바이트 바이너리 데이터에서 4자리 차량 번호를 추출합니다.

    Args:
        raw_data: 받은 2바이트
        role:     'entry' 또는 'exit'. 바이트 순서가 역할마다 다르다.
                  (BYTE_ORDER 주석 참고)

    Returns:
        (차량번호 4자리 문자열, 먼저 온 바이트, 나중에 온 바이트)
    """
    first, second = raw_data[0], raw_data[1]
    if BYTE_ORDER.get(role, 'low_first') == 'low_first':
        high, low = second, first
    else:
        high, low = first, second

    digit1 = (high >> 4) & 0x0F
    digit2 = high & 0x0F
    digit3 = (low >> 4) & 0x0F
    digit4 = low & 0x0F

    return f"{digit1}{digit2}{digit3}{digit4}", first, second


def build_car_packet(car_id, role='entry'):
    """
    4자리 차량번호 문자열 -> 2바이트 패킷. (테스트 송신용 / parse_car_id의 역연산)

    역할에 맞는 순서로 만든다. 그래야 그 역할의 Zybo가 보내는 것과 같은
    바이트가 나가고, 테스트 송신으로 실제 동작을 그대로 흉내 낼 수 있다.
    """
    d = f"{int(car_id):04d}"
    high = (int(d[0]) << 4) | int(d[1])     # 앞 두 자리
    low = (int(d[2]) << 4) | int(d[3])      # 뒤 두 자리
    if BYTE_ORDER.get(role, 'low_first') == 'low_first':
        return bytes([low, high])
    return bytes([high, low])


def _is_inside(car_id):
    """
    그 번호가 지금 자리에 세워져 있는가.

    주차를 마친 차(parked=True)만 해당한다. 두 경우를 일부러 뺐다.

      안내를 받는 중인 차 (자리는 배정됐고 parked=False)
        입구를 막 지난 차다. 번호가 한 번 더 들어오면 같은 입차가 두 번 온
        것이지 나가는 것이 아니다.

      자리를 잃은 기록 (spot_id가 None)
        카메라가 그 자리를 비었다고 보면 기록에서 자리만 떨어져 나간다.
        (sync_spot_occupancy) 실물이 이미 나갔는데 출차 수신을 놓친 경우가
        대부분이라, 그 번호가 다시 들어오면 새 입차로 보는 편이 맞다.
        handle_car_entry가 남은 기록을 지우고 자리를 새로 배정한다.
    """
    info = cars_info.get(car_id)
    return bool(info and info.get("parked"))


def resolve_packet(port_role, raw_data, peer_ip=None):
    """
    받은 2바이트를 어느 역할로 처리할지 정하고 번호를 읽는다.

    Args:
        port_role: 패킷이 들어온 포트의 역할 ('entry' / 'exit')
        raw_data:  받은 2바이트
        peer_ip:   보낸 쪽 IP (ROLE_RESOLVE['ROLE_BY_IP'] 조회용)

    Returns:
        (role, car_id, 먼저 온 바이트, 나중에 온 바이트, rerouted)
        rerouted는 포트가 말하는 역할과 다르게 처리한다는 뜻이다.
    """
    pinned = ROLE_RESOLVE['ROLE_BY_IP'].get(peer_ip)
    role = pinned or port_role
    car_id, first, second = parse_car_id(raw_data, role)

    if pinned or not ROLE_RESOLVE['KNOWN_CAR_IS_EXIT'] or role != 'entry':
        return role, car_id, first, second, role != port_role

    # 이미 주차장 안에 있는 차의 번호다. 나가는 중으로 본다.
    if _is_inside(car_id):
        return 'exit', car_id, first, second, True

    # 출차 보드가 입차 포트로 붙은 경우까지 본다. 두 보드는 바이트 순서가
    # 반대라(BYTE_ORDER) 입차 순서로 읽으면 앞뒤 두 자리가 뒤집힌, 등록되지
    # 않은 번호가 나온다. 반대 순서로 읽었을 때만 주차장 안의 차와 맞는다면
    # 그쪽이 진짜 번호다.
    other, first_o, second_o = parse_car_id(raw_data, 'exit')
    if other != car_id and _is_inside(other):
        return 'exit', other, first_o, second_o, True

    return 'entry', car_id, first, second, False


def _enable_keepalive(sock, idle=10, interval=5, count=3):
    """
    죽은 상대를 스스로 알아채도록 TCP keepalive를 켠다.

    Zybo의 전원이 꺼지거나 Wi-Fi가 끊기면 FIN이 오지 않는다. 그러면 소켓은
    멀쩡한 것처럼 남아서, 서버는 영영 오지 않을 2바이트를 기다리며 그 자리를
    붙들고 있게 된다. Zybo를 다시 켜서 접속해도 서버는 이미 붙들려 있으니
    받아주지 못한다. keepalive를 켜 두면 약 25초 안에 소켓이 오류로 끊겨
    자리가 풀린다.

    설정 이름은 리눅스 것이다. 없는 플랫폼에서는 조용히 건너뛴다.
    (그래도 SO_KEEPALIVE 자체는 켜지므로 기본 주기로는 동작한다)
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        return
    for name, value in (('TCP_KEEPIDLE', idle),
                        ('TCP_KEEPINTVL', interval),
                        ('TCP_KEEPCNT', count)):
        option = getattr(socket, name, None)
        if option is None:
            continue
        try:
            sock.setsockopt(socket.IPPROTO_TCP, option, value)
        except OSError:
            pass


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


def _notify_rx(role, car_id, first_byte, second_byte, result=None,
               port_role=None, rerouted=False):
    """
    수신 사실을 등록된 리스너 전원에게 알린다.

    리스너에서 예외가 나도 수신 루프를 멈추지 않는다. 화면 갱신이 실패했다고
    입출차 처리까지 죽으면 안 되기 때문이다.

    전달되는 event_dict:
        role      : 'entry'(입차) 또는 'exit'(출차). 실제로 처리한 역할이다.
        car_id    : 4자리 차량번호 문자열
        raw_hex   : "0x34 0x12" 형태의 원본 바이트 표기 (받은 순서 그대로)
        time      : 수신 시각 (datetime)
        result    : 입차면 handle_car_entry, 출차면 remove_car의 결과 dict
        port_role : 패킷이 들어온 포트의 역할
        rerouted  : 포트와 다른 역할로 처리했는가 (resolve_packet 참고)
    """
    event = {
        "role": role,
        "car_id": car_id,
        "raw_hex": f"0x{first_byte:02X} 0x{second_byte:02X}",
        "time": datetime.datetime.now(),
        "result": result,
        "port_role": port_role or role,
        "rerouted": rerouted,
    }
    for callback in _rx_listeners:
        try:
            callback(event)
        except Exception as e:
            print(f"[경고] 수신 리스너에서 오류가 발생했습니다: {e}")


def _process_packet(port_role, raw_data, peer_ip=None):
    """수신한 2바이트를 파싱해서 입차/출차 로직으로 넘긴다."""
    role, car_id, b1, b2, rerouted = resolve_packet(port_role, raw_data, peer_ip)

    if rerouted:
        label = '입차' if port_role == 'entry' else '출차'
        moved = '출차' if role == 'exit' else '입차'
        print(f"\n[{label}포트] 번호 {car_id}는 이미 주차장 안에 있는 차입니다. "
              f"{moved}로 처리합니다.")

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
        _notify_rx(role, car_id, b1, b2, result,
                   port_role=port_role, rerouted=rerouted)
    else:
        print(f"\n[출차수신(Dec)] (Hex: 0x{b1:02X} 0x{b2:02X}) -> {car_id}")

        # 출차 및 요금 계산 호출.
        # 반환값을 화면까지 실어 보낸다. 예전에는 여기서 버려서 요금과
        # 주차 시간이 터미널에만 찍히고 관제 화면에는 '출차 처리'로만 떴다.
        removed = remove_car(car_id)
        if removed is None:
            result = {"success": False, "message": "입차 기록이 없습니다"}
        else:
            spot_id, fee, minutes = removed
            result = {"success": True, "spot_id": spot_id,
                      "fee": fee, "minutes": minutes}

        _notify_rx(role, car_id, b1, b2, result,
                   port_role=port_role, rerouted=rerouted)


def _serve_role(role, label, host, port, timeout, stop_event):
    """
    한 역할(입차 또는 출차)에 대한 TCP 서버 루프.
    Zybo가 끊었다 다시 붙어도 계속 받을 수 있도록 accept를 반복한다.
    """
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        # 대기열을 1로 두면, 한 자리가 붙들려 있을 때 들어오는 접속이 커널
        # 단계에서 거절된다. Zybo 쪽에서는 '접속이 안 된다'로만 보인다.
        srv.listen(4)
        srv.settimeout(timeout)
    except OSError as e:
        print(f"[에러] {label} 포트({port})를 열 수 없습니다: {e}")

        # 이 오류는 거의 항상 '이미 실행 중인 프로그램이 있다'는 뜻이다.
        # SO_REUSEADDR을 켜 두었으므로 종료 직후의 TIME_WAIT 때문은 아니다.
        #
        # 이걸 짚어주지 않으면 원인을 엉뚱한 데서 찾게 된다. 같은 이유로
        # 카메라(/dev/video0)도 그 프로세스가 잡고 있어서 몇 줄 뒤에
        # "카메라를 열 수 없습니다"가 뜨는데, 그쪽이 훨씬 눈에 띄기 때문이다.
        if e.errno == errno.EADDRINUSE:
            print(f"       이미 실행 중인 프로그램이 이 포트를 쓰고 있습니다.")
            print(f"       확인 : ss -ltnp | grep {port}")
            print(f"       정리 : pkill -f E_main_final")
        return

    print(f"[{label}] TCP 서버 대기 중: {host}:{port}")

    # 접속을 받은 뒤에도 서버 소켓을 계속 지켜본다.
    #
    # 예전에는 한 번 받으면 그 연결이 끊길 때까지 accept로 돌아가지 않았다.
    # Zybo가 리셋되거나 Wi-Fi가 끊겨 FIN 없이 사라지면 그 연결은 소켓 위에
    # 그대로 남고, 서버는 오지 않을 2바이트를 기다리며 자리를 붙들고 있는다.
    # 그 상태에서 Zybo가 다시 붙으려 하면 대기열이 차 있어 거절된다.
    # 한쪽(입차)만 접속이 안 되는 증상이 이것이다.
    #
    # 접속을 IP별로 하나씩 들고 있는다. {소켓: [IP, 남은 바이트]}
    #
    # 예전에는 한 포트에 연결 하나만 두고, 새 접속이 오면 이전 것을 닫았다.
    # 죽은 Zybo가 자리를 붙들지 않게 하려던 것인데, 보드 두 대가 같은 포트로
    # 붙으면 서로를 계속 끊어 둘 다 못 쓰게 된다. 같은 IP에서 다시 붙을 때만
    # 이전 연결을 닫으면 그 목적은 그대로 두고 두 대를 함께 받을 수 있다.
    conns = {}

    while not stop_event.is_set():
        try:
            ready, _, _ = select.select([srv] + list(conns), [], [], timeout)
        except OSError:
            break

        if srv in ready:
            try:
                new_conn, addr = srv.accept()
            except OSError:
                break
            _enable_keepalive(new_conn)
            for old, (old_ip, _) in list(conns.items()):
                if old_ip == addr[0]:
                    print(f"[{label}] {old_ip}가 다시 접속해 이전 연결을 닫습니다.")
                    old.close()
                    del conns[old]
            conns[new_conn] = [addr[0], b'']
            print(f"[{label}] Zybo 접속됨: {addr[0]}:{addr[1]}")

        for conn in [c for c in ready if c is not srv and c in conns]:
            peer_ip, buf = conns[conn]

            # 2바이트씩 끊어 읽는다. 한 번에 여러 대가 몰려와도, 반대로 한
            # 바이트씩 쪼개져 와도 남는 것은 buf에 두고 다음에 이어 붙인다.
            try:
                chunk = conn.recv(64)
            except OSError as e:
                print(f"[{label}] 수신 중 오류: {e}")
                chunk = b''

            if not chunk:
                conn.close()
                del conns[conn]
                print(f"[{label}] {peer_ip} 연결이 끊어졌습니다. 재접속 대기...")
                continue

            buf += chunk
            while len(buf) >= 2:
                _process_packet(role, buf[:2], peer_ip)
                buf = buf[2:]
            conns[conn][1] = buf

    for conn in conns:
        conn.close()
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
    packet = build_car_packet(car_id, role)

    with socket.create_connection((host, port), timeout=3) as sock:
        sock.sendall(packet)
    print(f"[테스트송신] {role} {car_id} -> {host}:{port} "
          f"(Hex: 0x{packet[0]:02X} 0x{packet[1]:02X})")


if __name__ == "__main__":
    # python A00_uart_rx.py                                 -> 수신 서버 실행
    # python A00_uart_rx.py send entry 1234                 -> 자기 자신에게 송신
    # python A00_uart_rx.py send entry 1234 192.168.0.50    -> 젯슨으로 송신
    #
    # 마지막 형태가 배선 점검용이다. PC에서 젯슨으로 쏴 보면 Zybo가 붙지
    # 못하는 것이 네트워크 문제인지 Zybo 쪽 문제인지 바로 갈린다.
    if len(sys.argv) >= 4 and sys.argv[1] == 'send':
        host = sys.argv[4] if len(sys.argv) >= 5 else '127.0.0.1'
        send_test_packet(sys.argv[2], sys.argv[3], host=host)
    else:
        uart_rx_main()
