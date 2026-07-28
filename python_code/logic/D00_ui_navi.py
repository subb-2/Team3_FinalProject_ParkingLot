import cv2
import sys
import os
import math
import numpy as np

# 상위 디렉토리(python_code)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from logic.C00_navigation import (
    MARKER_TO_SPOT, MARKER_WORLD_POS, GATE_WORLD_POS,
    GUIDE_ARRIVED, GUIDE_UNKNOWN,
)

# 설정 (Configuration)
# 이 모듈은 '화면 표시'만 담당.
#   - 위치 추정 / 경로 안내 : C00_navigation.py
#   - 통합 실행             : C_main.py
CONFIG = {
    # 맵 화면 크기 (px)
    "MAP_WIDTH": 1000,
    "MAP_HEIGHT": 620,
    "PANEL_RATIO": 0.30,        # 오른쪽 안내 패널이 차지하는 가로 비율

    # 맵 여백 및 축척
    "MARGIN_PX": 50,            # 맵 영역 바깥 여백
    "PAD_CM": 25.0,             # 주차 구역 바깥으로 확보할 여유 공간

    # 주차 구역 크기 (cm) - 실제 목업 치수에 맞게 조정
    "SPOT_W_CM": 22.0,
    "SPOT_H_CM": 16.0,

    # 표시 옵션
    "SHOW_GRID": True,          # 배경 격자 표시
    "GRID_STEP_CM": 20.0,       # 격자 간격 (cm)
    "SHOW_TRAJECTORY": True,    # 차량 이동 궤적 표시
    "TRAJECTORY_MAX_POINTS": 60,
    "VEHICLE_RADIUS_PX": 11,
}

# 색상 (BGR)
COLOR_BG          = (34, 34, 34)     # 배경
COLOR_PANEL_BG    = (24, 24, 24)     # 패널 배경
COLOR_GRID        = (54, 54, 54)     # 격자
COLOR_TEXT        = (235, 235, 235)  # 기본 텍스트
COLOR_TEXT_DIM    = (150, 150, 150)  # 보조 텍스트
COLOR_SPOT_EMPTY  = (130, 130, 130)  # 빈자리 테두리
COLOR_SPOT_FULL   = (70, 70, 200)    # 주차중 (붉은 계열)
COLOR_SPOT_TARGET = (255, 0, 255)    # 목표 구역 (자홍)
COLOR_GATE        = (0, 200, 255)    # 입출구 (주황)
COLOR_VEHICLE     = (0, 230, 0)      # 차량 (번호 매칭됨)
COLOR_VEHICLE_UNK = (0, 165, 255)    # 차량 (번호 미매칭)
COLOR_TRAJECTORY  = (0, 220, 220)    # 이동 궤적
COLOR_GUIDE_LINE  = (0, 255, 255)    # 안내선
COLOR_ARRIVED     = (0, 255, 0)      # 도착 표시

FONT = cv2.FONT_HERSHEY_SIMPLEX


# 내비게이션 맵 UI
class NavigationMapUI:
    """
    차량의 실좌표(cm)와 주차 구역 정보를 위에서 내려다본 형태의
    2D 맵으로 렌더링하는 UI.

    카메라 영상이 아니라 C00_navigation이 계산한 실좌표를 그리므로,
    카메라 각도와 무관하게 항상 정면에서 본 주차장 배치로 표시된다.

    이 클래스는 순수 렌더러다. 검출/추적/위치추정을 하지 않고,
    이미 계산된 결과(nav_results)를 받아 그림만 그린다.
    """

    def __init__(self, navigator=None, width=None, height=None,
                 spot_world_pos=None, gate_world_pos=None):
        """
        NavigationMapUI 초기화.

        Args:
            navigator:      ParkingNavigator 인스턴스 (궤적/목표 조회에 사용, 없어도 동작)
            width:          맵 화면 가로 크기 (px)
            height:         맵 화면 세로 크기 (px)
            spot_world_pos: {구역ID: (x_cm, y_cm)} 매핑.
                            None이면 navigator 또는 C00의 기본 마커 배치에서 가져온다.
            gate_world_pos: 입출구 실좌표 (x_cm, y_cm)
        """
        self.navigator = navigator
        self.width = width or CONFIG['MAP_WIDTH']
        self.height = height or CONFIG['MAP_HEIGHT']

        # 주차 구역 실좌표 확보 (우선순위: 인자 > navigator > C00 기본값)
        if spot_world_pos is not None:
            self.spot_world_pos = dict(spot_world_pos)
        elif navigator is not None and navigator.spot_world_pos:
            self.spot_world_pos = dict(navigator.spot_world_pos)
        else:
            self.spot_world_pos = {
                spot_id: MARKER_WORLD_POS[marker_id]
                for marker_id, spot_id in MARKER_TO_SPOT.items()
                if marker_id in MARKER_WORLD_POS
            }

        self.gate_world_pos = gate_world_pos if gate_world_pos is not None else GATE_WORLD_POS

        # 맵 영역과 패널 영역의 가로 크기
        self.panel_w = int(self.width * CONFIG['PANEL_RATIO'])
        self.map_w = self.width - self.panel_w

        # 실좌표 -> 화면좌표 변환 파라미터 계산
        self._compute_transform()

        print(f"[INFO] 내비게이션 맵 UI 초기화 완료. "
              f"({self.width}x{self.height}, 구역 {len(self.spot_world_pos)}개, "
              f"축척 {self.scale:.2f} px/cm)")

    def _compute_transform(self):
        """
        등록된 주차 구역과 입출구가 모두 화면에 들어오도록
        실좌표 -> 화면좌표 변환(축척과 원점)을 자동 계산.
        """
        xs = [p[0] for p in self.spot_world_pos.values()]
        ys = [p[1] for p in self.spot_world_pos.values()]
        if self.gate_world_pos is not None:
            xs.append(self.gate_world_pos[0])
            ys.append(self.gate_world_pos[1])

        if not xs:
            # 등록된 좌표가 하나도 없을 때의 안전한 기본값
            self.min_x, self.min_y = 0.0, 0.0
            self.scale = 1.0
            return

        pad = CONFIG['PAD_CM']
        self.min_x, self.max_x = min(xs) - pad, max(xs) + pad
        self.min_y, self.max_y = min(ys) - pad, max(ys) + pad

        span_x = max(self.max_x - self.min_x, 1e-6)
        span_y = max(self.max_y - self.min_y, 1e-6)

        margin = CONFIG['MARGIN_PX']
        usable_w = max(self.map_w - 2 * margin, 1)
        usable_h = max(self.height - 2 * margin, 1)

        # 가로/세로 비율을 유지하기 위해 더 빡빡한 쪽에 맞춘다
        self.scale = min(usable_w / span_x, usable_h / span_y)

        # 남는 공간만큼 가운데 정렬
        self.offset_x = margin + (usable_w - span_x * self.scale) / 2
        self.offset_y = margin + (usable_h - span_y * self.scale) / 2

    def world_to_map(self, world_pt):
        """
        주차장 실좌표(cm)를 맵 화면 좌표(px)로 변환.

        Args:
            world_pt: (x_cm, y_cm)

        Returns:
            (px, py) 정수 좌표
        """
        px = self.offset_x + (world_pt[0] - self.min_x) * self.scale
        py = self.offset_y + (world_pt[1] - self.min_y) * self.scale
        return int(px), int(py)

    def render(self, nav_results, spot_status=None, fps=None, extra_info=None):
        """
        내비게이션 맵 한 장을 그려서 반환.

        Args:
            nav_results: C00_navigation.ParkingNavigator.update()의 반환 결과
            spot_status: {구역ID: "empty"|"full"} 점유 상태.
                         None이면 data.map_data의 실제 상태를 사용한다.
            fps:         화면에 표시할 FPS (없으면 생략)
            extra_info:  패널 하단에 추가로 표시할 문자열 리스트

        Returns:
            렌더링된 BGR 이미지 (numpy array)
        """
        if spot_status is None:
            from data.map_data import spot_status as live_status
            spot_status = live_status

        canvas = np.full((self.height, self.width, 3), COLOR_BG, dtype=np.uint8)

        # 현재 목표로 지정된 구역들 (강조 표시용)
        target_spots = {n["target_spot"] for n in nav_results if n.get("target_spot")}

        self._draw_grid(canvas)
        self._draw_gate(canvas)
        self._draw_spots(canvas, spot_status, target_spots)
        self._draw_guide_lines(canvas, nav_results)
        self._draw_trajectories(canvas, nav_results)
        self._draw_vehicles(canvas, nav_results)
        self._draw_panel(canvas, nav_results, fps, extra_info)

        return canvas

    def _draw_grid(self, canvas):
        """배경 격자와 축척 기준선을 그린다."""
        if not CONFIG['SHOW_GRID']:
            return

        step = CONFIG['GRID_STEP_CM']

        # 세로선 (x = 일정 간격)
        x = math.ceil(self.min_x / step) * step
        while x <= self.max_x:
            px, _ = self.world_to_map((x, self.min_y))
            _, py2 = self.world_to_map((x, self.max_y))
            _, py1 = self.world_to_map((x, self.min_y))
            if 0 <= px < self.map_w:
                cv2.line(canvas, (px, py1), (px, py2), COLOR_GRID, 1)
            x += step

        # 가로선 (y = 일정 간격)
        y = math.ceil(self.min_y / step) * step
        while y <= self.max_y:
            px1, py = self.world_to_map((self.min_x, y))
            px2, _ = self.world_to_map((self.max_x, y))
            px2 = min(px2, self.map_w - 1)
            cv2.line(canvas, (px1, py), (px2, py), COLOR_GRID, 1)
            y += step

        # 축척 안내
        cv2.putText(canvas, f"grid {step:.0f}cm", (10, self.height - 12),
                    FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_gate(self, canvas):
        """입출구 위치를 표시."""
        if self.gate_world_pos is None:
            return

        gx, gy = self.world_to_map(self.gate_world_pos)
        cv2.circle(canvas, (gx, gy), 13, COLOR_GATE, 2)
        cv2.putText(canvas, "GATE", (gx - 20, gy - 20),
                    FONT, 0.5, COLOR_GATE, 2, cv2.LINE_AA)

    def _draw_spots(self, canvas, spot_status, target_spots):
        """주차 구역을 점유 상태에 따라 색을 달리하여 그린다."""
        half_w = CONFIG['SPOT_W_CM'] * self.scale / 2
        half_h = CONFIG['SPOT_H_CM'] * self.scale / 2

        for spot_id, world_pt in sorted(self.spot_world_pos.items()):
            cx, cy = self.world_to_map(world_pt)
            p1 = (int(cx - half_w), int(cy - half_h))
            p2 = (int(cx + half_w), int(cy + half_h))

            is_full = spot_status.get(spot_id) == "full"
            is_target = spot_id in target_spots

            # 주차중인 구역은 채워서, 빈 구역은 테두리만
            if is_full:
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_FULL, -1)
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_EMPTY, 1)
            else:
                cv2.rectangle(canvas, p1, p2, COLOR_SPOT_EMPTY, 1)

            # 목표 구역은 굵은 자홍 테두리로 강조
            if is_target:
                cv2.rectangle(canvas,
                              (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3),
                              COLOR_SPOT_TARGET, 2)

            label_color = COLOR_SPOT_TARGET if is_target else COLOR_TEXT
            (tw, _), _ = cv2.getTextSize(spot_id, FONT, 0.45, 1)
            cv2.putText(canvas, spot_id, (cx - tw // 2, cy + 5),
                        FONT, 0.45, label_color, 1, cv2.LINE_AA)

    def _draw_guide_lines(self, canvas, nav_results):
        """
        차량에서 목표 구역까지의 주행 경로를 그린다.

        직선이 아니라 C01_path_planner가 계산한 경유점을 따라 그리므로,
        주차 구역을 가로지르지 않고 통로를 따라가는 실제 경로가 보인다.
        """
        for nav in nav_results:
            if nav.get("target_world") is None:
                continue

            start = self.world_to_map(nav["world_pos"])
            end = self.world_to_map(nav["target_world"])

            # 도착한 차량은 안내선 대신 도착 표시
            if nav.get("guide") == GUIDE_ARRIVED:
                cv2.circle(canvas, end, 16, COLOR_ARRIVED, 2)
                continue

            route = nav.get("route")
            if route and len(route) >= 2:
                # 현재 위치에서 남은 경유점까지만 이어서 그린다
                idx = min(nav.get("route_index", 1), len(route) - 1)
                pts = [start] + [self.world_to_map(p) for p in route[idx:]]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, a, b, COLOR_GUIDE_LINE, 2, cv2.LINE_AA)
                # 경유점 표시 (목적지 제외)
                for p in pts[1:-1]:
                    cv2.circle(canvas, p, 4, COLOR_GUIDE_LINE, -1)
                cv2.arrowedLine(canvas, pts[-2], pts[-1],
                                COLOR_GUIDE_LINE, 2, tipLength=0.25)
                mid = pts[len(pts) // 2]
            else:
                # 경로를 찾지 못한 경우에만 직선으로 대체
                cv2.arrowedLine(canvas, start, end, COLOR_GUIDE_LINE, 2, tipLength=0.06)
                mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)

            # 남은 거리 표시 (경로를 따라간 거리)
            if nav.get("distance_cm") is not None:
                cv2.putText(canvas, f"{nav['distance_cm']:.0f}cm", (mid[0] + 6, mid[1] - 6),
                            FONT, 0.45, COLOR_GUIDE_LINE, 1, cv2.LINE_AA)

    def _draw_trajectories(self, canvas, nav_results):
        """차량이 지나온 경로를 선으로 그린다."""
        if not CONFIG['SHOW_TRAJECTORY'] or self.navigator is None:
            return

        for nav in nav_results:
            key = nav["car_id"] if nav["car_id"] else f"track_{nav['track_id']}"
            history = self.navigator.get_world_trajectory(key)
            if len(history) < 2:
                continue

            points = history[-CONFIG['TRAJECTORY_MAX_POINTS']:]
            pts = np.array([self.world_to_map(p) for p in points], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, COLOR_TRAJECTORY, 1, cv2.LINE_AA)

    def _draw_vehicles(self, canvas, nav_results):
        """차량 위치, 진행 방향, 차량번호를 표시."""
        radius = CONFIG['VEHICLE_RADIUS_PX']

        for nav in nav_results:
            cx, cy = self.world_to_map(nav["world_pos"])
            color = COLOR_VEHICLE if nav["car_id"] else COLOR_VEHICLE_UNK

            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1)

            # 진행 방향 화살표 (정지 상태면 생략)
            heading = nav.get("heading_deg")
            if heading is not None:
                rad = math.radians(heading)
                hx = int(cx + math.cos(rad) * radius * 2.4)
                hy = int(cy + math.sin(rad) * radius * 2.4)
                cv2.arrowedLine(canvas, (cx, cy), (hx, hy), (255, 255, 255), 2, tipLength=0.35)

            label = nav["car_id"] if nav["car_id"] else f"#{nav['track_id']}"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 2)
            cv2.putText(canvas, label, (cx - tw // 2, cy - radius - 7),
                        FONT, 0.5, color, 2, cv2.LINE_AA)

    def _draw_panel(self, canvas, nav_results, fps, extra_info):
        """오른쪽에 차량별 안내 정보를 표 형태로 표시."""
        x0 = self.map_w
        cv2.rectangle(canvas, (x0, 0), (self.width, self.height), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (x0, 0), (x0, self.height), (70, 70, 70), 1)

        pad = 14
        y = 32
        cv2.putText(canvas, "NAVIGATION", (x0 + pad, y), FONT, 0.62, COLOR_TEXT, 2, cv2.LINE_AA)

        y += 22
        if fps is not None:
            cv2.putText(canvas, f"FPS {fps:.1f}", (x0 + pad, y),
                        FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y += 14
        cv2.line(canvas, (x0 + pad, y), (self.width - pad, y), (70, 70, 70), 1)
        y += 22

        if not nav_results:
            cv2.putText(canvas, "no vehicle", (x0 + pad, y),
                        FONT, 0.5, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            y += 24
        else:
            for nav in nav_results:
                if y > self.height - 70:
                    cv2.putText(canvas, "...", (x0 + pad, y),
                                FONT, 0.5, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                    break

                car_label = nav["car_id"] if nav["car_id"] else f"#{nav['track_id']}"
                color = COLOR_VEHICLE if nav["car_id"] else COLOR_VEHICLE_UNK

                # 1행: 차량번호 -> 목표 구역
                target = nav["target_spot"] if nav["target_spot"] else "-"
                cv2.putText(canvas, f"{car_label} > {target}", (x0 + pad, y),
                            FONT, 0.52, color, 2, cv2.LINE_AA)
                y += 20

                # 2행: 안내 방향과 남은 거리
                guide = nav.get("guide", GUIDE_UNKNOWN)
                dist = f"{nav['distance_cm']:.0f}cm" if nav.get("distance_cm") is not None else "-"
                guide_color = COLOR_ARRIVED if guide == GUIDE_ARRIVED else COLOR_GUIDE_LINE
                cv2.putText(canvas, f"  {guide}  {dist}", (x0 + pad, y),
                            FONT, 0.48, guide_color, 1, cv2.LINE_AA)
                y += 18

                # 3행: 현재 실좌표
                wx, wy = nav["world_pos"]
                cv2.putText(canvas, f"  ({wx:.0f}, {wy:.0f})cm", (x0 + pad, y),
                            FONT, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                y += 24

        # 추가 정보 (FIFO 대기열, 호모그래피 상태 등)
        if extra_info:
            y = min(y + 6, self.height - 20)
            cv2.line(canvas, (x0 + pad, y), (self.width - pad, y), (70, 70, 70), 1)
            y += 20
            for line in extra_info:
                if y > self.height - 12:
                    break
                cv2.putText(canvas, line, (x0 + pad, y),
                            FONT, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                y += 17

        return canvas


# =====================================================================
# 테스트용 메인
# =====================================================================
# DEMO_MODE = True  : 카메라 없이 합성 차량으로 맵 UI만 확인 (윈도우에서 바로 실행 가능)
# DEMO_MODE = False : C_main의 파이프라인을 실행하여 카메라 영상 + 실시간 맵을 함께 스트리밍
DEMO_MODE = True

if __name__ == '__main__':
    from flask import Flask, Response

    print("==========================================")
    print(" D00 : 주차장 내비게이션 UI")
    print(f" 모드 : {'DEMO (카메라 없이 합성 데이터)' if DEMO_MODE else 'LIVE (C_main 파이프라인)'}")
    print("==========================================")

    app = Flask(__name__)

    if DEMO_MODE:
        # ---------------------------------------------------------
        # 데모 모드: 합성 차량 2대가 각자 목표 구역으로 이동하는 장면
        # ---------------------------------------------------------
        import time
        from logic.C00_navigation import GUIDE_STRAIGHT, GUIDE_RIGHT

        ui = NavigationMapUI()

        # 데모용 점유 상태 (실제 spot_status 대신 사용)
        demo_status = {spot_id: "empty" for spot_id in ui.spot_world_pos}
        demo_status["A-3"] = "full"
        demo_status["B-2"] = "full"

        def demo_frames():
            step = 0
            while True:
                t = (step % 120) / 120.0    # 0.0 ~ 1.0 반복

                # 차량 1: 입출구 -> A-1 로 이동
                start1, goal1 = (-30.0, 30.0), ui.spot_world_pos["A-1"]
                pos1 = (start1[0] + (goal1[0] - start1[0]) * t,
                        start1[1] + (goal1[1] - start1[1]) * t)
                dist1 = math.hypot(goal1[0] - pos1[0], goal1[1] - pos1[1])

                # 차량 2: 오른쪽에서 B-4 로 이동
                start2, goal2 = (110.0, 10.0), ui.spot_world_pos["B-4"]
                pos2 = (start2[0] + (goal2[0] - start2[0]) * t,
                        start2[1] + (goal2[1] - start2[1]) * t)
                dist2 = math.hypot(goal2[0] - pos2[0], goal2[1] - pos2[1])

                nav_results = [
                    {
                        "track_id": 1, "car_id": "1234",
                        "image_pos": (0, 0), "world_pos": pos1,
                        "heading_deg": math.degrees(math.atan2(goal1[1] - start1[1],
                                                               goal1[0] - start1[0])),
                        "target_spot": "A-1", "target_world": goal1,
                        "distance_cm": dist1,
                        "guide": GUIDE_ARRIVED if dist1 < 15 else GUIDE_STRAIGHT,
                        "guide_text": "", "nearest_spot": "A-1",
                    },
                    {
                        "track_id": 2, "car_id": "1998",
                        "image_pos": (0, 0), "world_pos": pos2,
                        "heading_deg": math.degrees(math.atan2(goal2[1] - start2[1],
                                                               goal2[0] - start2[0])),
                        "target_spot": "B-4", "target_world": goal2,
                        "distance_cm": dist2,
                        "guide": GUIDE_ARRIVED if dist2 < 15 else GUIDE_RIGHT,
                        "guide_text": "", "nearest_spot": "B-4",
                    },
                ]

                canvas = ui.render(
                    nav_results,
                    spot_status=demo_status,
                    fps=30.0,
                    extra_info=["DEMO MODE", "no camera / no UART"]
                )

                ret, buffer = cv2.imencode('.jpg', canvas)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                step += 1
                time.sleep(1 / 30)

        @app.route('/map_feed')
        def map_feed():
            return Response(demo_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/')
        def index():
            return f"""
            <html>
                <head><title>Parking Navigation UI (DEMO)</title></head>
                <body style="background-color:#222; color:white; text-align:center;">
                    <h2>Parking Navigation UI - DEMO MODE</h2>
                    <p>합성 데이터로 맵 UI만 확인하는 모드입니다.
                       실제 연동은 DEMO_MODE = False 로 변경하세요.</p>
                    <img src="/map_feed" width="{CONFIG['MAP_WIDTH']}" height="{CONFIG['MAP_HEIGHT']}">
                </body>
            </html>
            """

    else:
        # ---------------------------------------------------------
        # 실시간 모드: C_main 파이프라인 + 맵 UI
        # ---------------------------------------------------------
        from logic.C_main import (
            CONFIG as C_CONFIG, open_camera, build_pipeline,
            setup_car_number_source, register_car_number, build_status,
        )
        from logic.B02_car_mot import car_number_fifo

        cap = open_camera()
        if cap is None:
            print("[ERROR] 카메라를 열 수 없습니다. DEMO_MODE = True 로 두면 카메라 없이 확인할 수 있습니다.")
            sys.exit(1)

        pipeline = build_pipeline(cap)
        uart_sim_stop = setup_car_number_source()
        ui = NavigationMapUI(navigator=pipeline.navigator)

        def map_frames():
            """
            파이프라인이 갱신한 최신 결과로 맵을 그린다.
            영상 처리는 /video_feed 쪽 제너레이터가 수행하므로,
            여기서는 최신 상태를 읽어 그리기만 한다.
            """
            import time
            while True:
                mapper = pipeline.navigator.mapper
                if not mapper.is_ready():
                    state = "NOT READY"
                elif mapper.locked:
                    state = f"LOCKED {mapper.calibrated_with}pt {mapper.reproj_error:.1f}cm"
                else:
                    state = f"PROVISIONAL {mapper.calibrated_with}/{mapper.lock_markers}pt"
                extra = [
                    f"homography: {state}",
                    f"markers: {len(pipeline.navigator.latest_markers)}",
                    f"fifo: {car_number_fifo.size()} waiting",
                ]
                canvas = ui.render(pipeline.latest_nav, fps=pipeline.fps, extra_info=extra)

                ret, buffer = cv2.imencode('.jpg', canvas)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1 / 15)   # 맵은 15fps면 충분

        @app.route('/video_feed')
        def video_feed():
            return Response(pipeline.generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/map_feed')
        def map_feed():
            return Response(map_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/enqueue/<car_id>')
        def enqueue(car_id):
            register_car_number(car_id)
            return f"등록: {car_id} (대기 {car_number_fifo.size()}대)"

        @app.route('/status')
        def status():
            return build_status(pipeline)

        @app.route('/')
        def index():
            return f"""
            <html>
                <head><title>Parking Navigation UI</title></head>
                <body style="background-color:#222; color:white; text-align:center;">
                    <h2>Jetson Orin Nano - Parking Navigation UI</h2>
                    <div style="display:flex; justify-content:center; gap:16px;
                                flex-wrap:wrap; align-items:flex-start;">
                        <div>
                            <h3 style="font-weight:normal;">Camera</h3>
                            <img src="/video_feed"
                                 width="{C_CONFIG['CAM_WIDTH']}" height="{C_CONFIG['CAM_HEIGHT']}">
                        </div>
                        <div>
                            <h3 style="font-weight:normal;">Navigation Map</h3>
                            <img src="/map_feed"
                                 width="{CONFIG['MAP_WIDTH']}" height="{CONFIG['MAP_HEIGHT']}">
                        </div>
                    </div>
                    <p>차량번호 수동 등록: <code>/enqueue/1234</code> | 상태: <code>/status</code></p>
                </body>
            </html>
            """

    print("\n[INFO] Flask 웹 서버를 시작합니다. http://젯슨IP:5000/ 으로 접속하세요.")
    app.run(host='0.0.0.0', port=5000, debug=False)
