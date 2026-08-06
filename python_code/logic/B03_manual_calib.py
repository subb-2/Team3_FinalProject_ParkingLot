import cv2
import sys
import os
import numpy as np

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logic.C00_navigation import MarkerMapper, CONFIG as C00_CONFIG
from logic.C02_lot_layout import MARKER_WORLD_POS
from logic.B00_camera_input import get_camera

# 전역 변수 설정
points_dict = {}  # {marker_id: (x_px, y_px)}
current_index = 0
marker_ids = sorted(list(MARKER_WORLD_POS.keys()))
image_display = None
image_clean = None
window_name = "Manual Calibration"

def draw_instructions(img):
    """현재 상태와 안내 메시지를 화면에 그린다."""
    global current_index
    
    # 상단 정보 바 (배경)
    cv2.rectangle(img, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
    
    if current_index < len(marker_ids):
        target_id = marker_ids[current_index]
        msg1 = f"[{current_index+1}/{len(marker_ids)}] Please click the center of PILLAR (Marker ID: {target_id})"
        msg2 = "Controls: [Click] Select | [z] Undo | [q] Quit"
        color = (0, 255, 255)
    else:
        msg1 = "All pillars selected!"
        msg2 = "Press [Enter] to save and exit, or [z] to undo."
        color = (0, 255, 0)
        
    cv2.putText(img, msg1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, msg2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 찍힌 점들 그리기
    for m_id, pt in points_dict.items():
        x, y = int(pt[0]), int(pt[1])
        cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, f"ID:{m_id}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
def mouse_callback(event, x, y, flags, param):
    global current_index, image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_index < len(marker_ids):
            target_id = marker_ids[current_index]
            points_dict[target_id] = (x, y)
            print(f"[선택] 기둥 ID {target_id} -> 이미지 좌표 ({x}, {y})")
            current_index += 1
            
            # 화면 갱신
            image_display = image_clean.copy()
            draw_instructions(image_display)
            cv2.imshow(window_name, image_display)

def main():
    global image_display, image_clean, current_index
    
    print("=" * 60)
    print(" B03_manual_calib : 수동 마커(기둥) 좌표 지정 툴")
    print("=" * 60)
    print("카메라를 켜서 화면을 불러오는 중입니다...")
    
    # 이미지 파일이 인자로 주어지면 그걸 쓰고, 아니면 카메라를 켠다.
    args = sys.argv[1:]
    
    if '--capture' in args:
        print("[INFO] 캡처 모드로 실행합니다. GUI 창을 띄우지 않고 카메라 사진만 저장합니다.")
        cap = get_camera(sensor_id=0, width=1280, height=720, framerate=30)
        if not cap.isOpened():
            print("[오류] 카메라를 열 수 없습니다.")
            sys.exit(1)
            
        print("[INFO] 카메라 안정화를 위해 프레임을 읽습니다...")
        for _ in range(10):
            ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[오류] 카메라 프레임을 읽을 수 없습니다.")
            sys.exit(1)
            
        cv2.imwrite("capture.jpg", frame)
        print("[성공] 카메라 프레임을 'capture.jpg' 파일로 저장했습니다.")
        print("이 파일을 복사해서 GUI를 띄울 수 있는 컴퓨터에서 실행하세요:")
        print("예) python B03_manual_calib.py capture.jpg")
        sys.exit(0)
        
    if args and not args[0].startswith('--'):
        path = args[0]
        frame = cv2.imread(path)
        if frame is None:
            print(f"[오류] 이미지를 열 수 없습니다: {path}")
            sys.exit(1)
        print(f"[INFO] 이미지 로드 완료: {path}")
    else:
        # 카메라 켜기
        cap = get_camera(sensor_id=0, width=1280, height=720, framerate=30)
        if not cap.isOpened():
            print("[오류] 카메라를 열 수 없습니다.")
            sys.exit(1)
            
        print("[INFO] 카메라 안정화를 위해 프레임을 읽습니다...")
        for _ in range(10):
            ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[오류] 카메라 프레임을 읽을 수 없습니다.")
            sys.exit(1)
        print("[INFO] 카메라 프레임 촬영 완료.")
        
    image_clean = frame.copy()
    image_display = image_clean.copy()
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    draw_instructions(image_display)
    cv2.imshow(window_name, image_display)
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q') or key == 27: # q 또는 ESC
            print("[종료] 사용자가 취소했습니다. 저장되지 않았습니다.")
            break
            
        elif key == ord('z'): # Undo
            if current_index > 0:
                current_index -= 1
                target_id = marker_ids[current_index]
                if target_id in points_dict:
                    del points_dict[target_id]
                print(f"[취소] 기둥 ID {target_id}의 좌표를 지웠습니다.")
                
                image_display = image_clean.copy()
                draw_instructions(image_display)
                cv2.imshow(window_name, image_display)
                
        elif key == 13: # Enter
            if current_index >= 4: # 최소 4점은 되어야 호모그래피 가능
                if current_index < len(marker_ids):
                    print(f"\n[알림] 전체 {len(marker_ids)}개의 기둥 중 {current_index}개만 선택했습니다.")
                    val = input("이대로 저장을 진행하시겠습니까? (y/n): ")
                    if val.lower() != 'y':
                        continue
                
                print("\n[INFO] 호모그래피 계산을 시도합니다...")
                mapper = MarkerMapper()
                success, msg = mapper.set_homography_from_points(points_dict)
                if success:
                    print(f"[성공] 변환 행렬이 확정되었습니다.")
                    print("이제 C_main.py 등 메인 시스템을 실행하면 이 좌표계를 사용합니다.")
                    break
                else:
                    print(f"[실패] {msg}")
                    print("점의 위치가 올바르지 않습니다. 'z'를 눌러 수정하거나 다시 시도하세요.")
            else:
                print(f"[불가] 최소 4개의 기둥을 찍어야 합니다. (현재 {current_index}개)")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
