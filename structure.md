# Jetson Orin Nano 기반 지능형 주차 관리 및 안내 시스템 Structure

본 문서는 **NVIDIA Jetson Orin Nano**를 활용하여 입차 차량의 객체 인식, 다중 객체 추적, FPGA(Zybo) 연동 번호판 매칭, 실시간 빈자리 배치 및 주차 안내 시스템을 구축하기 위한 **상세 아키텍처 및 단계별 구현 기술 명세서**입니다.

---

## 1. 시스템 전체 데이터 흐름 (System Data Flow)

```
[ CCTV / IP Camera (1080p, 30fps) ]
              │
              ▼
  ┌───────────────────────────────┐
  │ ① 영상 입력 (GStreamer/OpenCV)│
  └──────────────┬────────────────┘
                 │
                 ▼
  ┌───────────────────────────────┐
  │ ② YOLOv8n 객체 검출 (TensorRT)│ ◄── [ 차량 Bounding Box (x,y,w,h) ]
  └──────────────┬────────────────┘
                 │
                 ├───────────────────────────────────────┐
                 ▼                                       ▼
  ┌───────────────────────────────┐     ┌─────────────────────────────────┐
  │ ③ ByteTrack 다중 객체 추적     │     │ 2.5 FPGA (Zybo) UART 통신       │
  │   (Track ID 부여 및 유지)      │     │ (하위 비트 차량 번호 0000_1234)  │
  └──────────────┬────────────────┘     └────────────────┬────────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │ Track ID ↔ 번호판 정보 매칭 │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ④ 차량 중심 좌표 계산 (cx, cy)                             │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ⑤ Homography (IPM) 실제 주차장 좌표계 변환 (X_m, Y_m)       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ⑥ 주차장 지도 매핑 & ⑦ 최적 빈자리 배치 및 경로 저장        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ⑧ 다중 CCTV Handover (카메라 영역 간 Track ID 인계)          │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ⑨ 주차장 실시간 네비게이션 & 모니터링 대시보드              │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 2. 하드웨어 및 소프트웨어 개발 환경 (Tech Stack)

| 구분 | 사양 및 기술 스택 | 비고 |
| :--- | :--- | :--- |
| **Main Hardware** | NVIDIA Jetson Orin Nano (8GB / 4GB) | Deep Learning Inference & Control |
| **Co-Hardware** | Zybo Z7 (FPGA) | CNN 기반 번호판 실시간 인식 & UART 전송 |
| **OS Environment** | JetPack 5.x / 6.x (Ubuntu 22.04 LTS) | CUDA 11.8 / 12.2, TensorRT 지원 |
| **Vision Framework** | NVIDIA DeepStream SDK / OpenCV 4.x / GStreamer | 실시간 RTSP/CCTV 스트림 디코딩 |
| **Object Detection** | YOLOv8n (FP16 / INT8 TensorRT Engine) | 실시간 차량 검출 (Car, Bus, Truck) |
| **Multi-Object Tracking**| ByteTrack (C++ / Python Wrapper) | 가려짐(Occlusion) 시 Track ID 유지 |
| **Communication** | PySerial (UART - Baudrate 115200) | FPGA ↔ Jetson 인터페이스 |
| **GUI / Monitoring** | PySide6 (Qt) / Streamlit / Web Socket Dashboard | 실시간 주차장 Map 및 경로 시각화 |

---

## 📋 3. 단계별 구체화 상세 설계 명세서

### 🔹 1단계: 카메라 영상 입력 및 전처리 (Camera Input Pipeline)
* **목적**: Jetson Orin Nano의 HW 디코더를 활용하여 CCTV/IP Camera 영상을 최소 Latency로 수신하고 실시간 프레임을 확보.
* **입력**: RTSP 스트림 또는 USB/IP 카메라 (1920x1080 @ 30FPS).
* **출력**: BGR/RGB Image Frame (`nvmm` 메모리 연동 가능).
* **구현 세부사항**:
  - GStreamer 파이프라인 적용: `nvv4l2decoder` → `nvvideoconvert` → OpenCV/DeepStream 전달.
  - Frame Drop 최소화를 위한 링 버퍼(Ring Buffer) 구성 및 30FPS 유지 검증.
* **검증 지표**: Latency < 30ms, FPS >= 30, CPU 점유율 < 20%.

---

### 🔹 2단계: YOLOv8n 기반 차량 객체 검출 (Vehicle Detection)
* **목적**: 입력 프레임에서 차량(Car, Van, SUV 등)의 Bounding Box 위치 및 Confidence 추출.
* **입력**: 1080p Image Frame.
* **출력**: `[x1, y1, x2, y2, confidence, class_id]` 리스트.
* **구현 세부사항**:
  - PyTorch `.pt` 모델을 Jetson Orin Nano 전용 **TensorRT (.engine)** 모델로 변환 (FP16 양자화).
  - ROI(Region of Interest) 설정을 통해 주차장 외 구역(도로, 건물 등) 오검출 차단.
* **검증 지표**: 차량 mAP@0.5 > 92%, Orin Nano 추론 속도 < 10ms/frame.

---

### 🔹 2.5단계: FPGA(Zybo) UART 통신 & 번호판 - Track ID 매칭
* **목적**: 입구 게이트의 Zybo FPGA에서 넘겨주는 8비트 차량 번호 데이터 수신 및 신규 입차 차량 Track ID 연동.
* **입력**:
  - UART 8비트 패킷 (상위 비트: 상태 제어 / 하위 4비트: 차량 식별 번호 예: `0000_1234`).
  - Jetson 입구 카메라 영역에 새로 생성된 YOLO Bounding Box / Track ID.
* **출력**: 차량 매칭 객체 `{ "track_id": 5, "car_num": "1234", "entry_time": "21:40:00", "assigned_spot": "P-13" }`.
* **구현 세부사항**:
  - 입구 Trigger Zone에 차량 진입 시 FPGA UART 데이터 읽기 동기화.
  - 최적 주차 공간(입구와 가깝거나 동선이 짧은 구역) 즉시 계산 및 매칭.
* **검증 지표**: UART 수신 성공률 99.9%, 매칭 지연시간 < 50ms.

---

### 🔹 3단계: ByteTrack 다중 객체 추적 (Multi-Object Tracking)
* **목적**: 차량 이동 중 가려짐, 조명 변화, 타 차량과의 교차 상황에서도 동일한 `Track ID`를 지속 유지.
* **입력**: YOLOv8 Bounding Box + Confidence.
* **출력**: Tracked Bounding Box (Track ID 포함).
* **구현 세부사항**:
  - ByteTrack 알고리즘 적용: 낮은 Confidence의 Bounding Box도 2차 매칭하여 프레임 이탈 방지.
  - Kalman Filter 기반 위치 예측으로 빠른 이동 시 ID Swapping(아이디 바뀜) 방지.
* **검증 지표**: ID Switch 횟수 < 1회/분, 추적 유지율 > 98%.

---

### 🔹 4단계: 차량 중심 좌표 계산 (Center Point Extraction)
* **목적**: 2D Bounding Box를 주차장 바닥면 기준의 단일 점(Point) 좌표로 정밀 환산.
* **입력**: Bounding Box `(x1, y1, x2, y2)`.
* **출력**: 중심 바닥 좌표 $(c_x, c_y) = (x1 + \frac{w}{2}, y2)$ *(차량 하단 중앙점).*
* **구현 세부사항**:
  - 차량의 바운딩 박스 중앙 상단 대신 **하단 바깥쪽 중심점 $(x_{center}, y_{bottom})$**을 채택하여 Ground Contact Point 정확도 확보.

---

### 🔹 5단계: 픽셀 좌표 → 실제 주차장 좌표 변환 (Homography / IPM)
* **목적**: 카메라 영상의 픽셀 좌표 $(c_x, c_y)$를 실제 미터(m) 단위 주차장 2D 평면 좌표 $(X_m, Y_m)$로 변환.
* **입력**: 픽셀 좌표 $(c_x, c_y)$, $3 \times 3$ Homography Matrix $H$.
* **출력**: 실제 좌표 $(X_m, Y_m)$.
* **구현 세부사항**:
  - Calibration: 주차장 바닥 4개 이상의 기준점(Reference Points)을 측정하여 호모그래피 행렬 $H$ 산출.
  - 변환식:
    $$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \cdot \begin{bmatrix} c_x \\ c_y \\ 1 \end{bmatrix}, \quad X_m = \frac{x'}{w'}, \; Y_m = \frac{y'}{w'}$$
* **검증 지표**: 실제 측정 거리 대비 좌표 오차 < 15cm 이내.

---

### 🔹 6단계: 주차장 벡터 지도(Parking Map) 생성 및 주차구역 데이터베이스
* **목적**: 주차 구역(Parking Slot) 및 주차장 내 도로 구역(Road Way)을 2D Grid/Vector 지도 데이터로 정의.
* **구현 세부사항**:
  - 주차면 좌표 미리 정의: 예) `P1: (2.0m, 4.0m, w=2.3m, h=5.0m)`, `P2: (5.0m, 4.0m, ...)`
  - 구역 상태 Management (`EMPTY`, `OCCUPIED`, `RESERVED`).

---

### 🔹 7단계: 차량 위치 매핑 & 빈자리 배치 알고리즘
* **목적**: 차량 좌표 $(X_m, Y_m)$가 주차 구역 내에 3초 이상 체류 시 주차 완료(`OCCUPIED`) 처리 및 빈자리 상태 자동 업데이트.
* **구현 세부사항**:
  - **Point-in-Polygon** 알고리즘으로 차량 중심점이 어느 주차면에 포함되는지 연산.
  - **빈자리 추천 알고리즘**: 입구에서 가장 인접하고 동선 충돌이 적은 Slot 자동 지정 및 안내.

---

### 🔹 8단계: 차량 이동 경로 저장 및 속도/방향 분석
* **목적**: 차량별 이동 궤적(Trajectory Queue)을 저장하여 주차 진행 여부, 불법 주정차, 逆주행 감지.
* **데이터 구조**: `Track_History[track_id] = [(X_0, Y_0, t_0), (X_1, Y_1, t_1), ...]`
* **활용**:
  - 주차장 내 시각화 궤적 표시 (Path Line).
  - 평균 이동 속도 및 주차 완료 예측 시간 계산.

---

### 🔹 9단계: 다중 CCTV 연동 및 Cross-Camera Handover
* **목적**: 사각지대 해소를 위해 2대 이상의 CCTV를 사용할 때 카메라 음영 지역 진출입 시 동일 차량 추적 연속성 유지.
* **구현 세부사항**:
  - **Global World Coordinates**: 각 카메라의 호모그래피 결과를 하나의 공통 주차장 지도 좌표계로 통합.
  - **Handover Zone**: Camera 1의 경계 영역에서 사라진 `Track ID`와 Camera 2의 경계 영역에 출현한 `Vehicle Feature / 좌표`를 시공간(Spatio-Temporal) 조건으로 바인딩.

---

### 🔹 10단계: 실시간 주차 네비게이션 & 통합 모니터링 UI
* **목적**: 운전자 및 관리자에게 실시간 주차장 현황, 추천 빈자리 경로, 차량 위치 모니터링 제공.
* **주요 기능**:
  1. **실시간 2D 주차장 맵**: 차량 아이콘 실시간 이동 시각화 + 주차 구역 색상 변경 (초록: 빈자리 / 빨강: 주차됨 / 노랑: 안내중).
  2. **입차 안내 전광판**: 입차 차량 번호판(`1234`)과 배정된 주차 구역(`P-13`) 및 최단 경로 표시.
  3. **차량 위치 검색**: 차량 번호 4자리 입력 시 현재 주차 위치 및 층/구역 시각화.
  4. **통계 및 모니터링**: 전체 주차면수, 현재 주차율, 평균 주차 소요 시간 산출.

---

## 🔄 4. 최종 데이터 파이프라인 종합 예시

```python
# Jetson Orin Nano 메인 루프 pseudo code 

def main_pipeline():
    # 1. Hardware & Stream Initialization
    cap = init_gstreamer_rtsp_stream(cctv_url)
    fpga_uart = init_uart_serial(port='/dev/ttyTHS1', baudrate=115200)
    yolo_model = load_tensorrt_engine('yolov8n_fp16.engine')
    tracker = ByteTracker()
    homography_matrix = load_calibration_matrix()
    parking_map = LoadParkingMapConfig('parking_slots.json')
    
    while True:
        # 1단계: Frame Read
        frame = cap.read()
        
        # 2.5단계: FPGA UART Signal Check (Non-blocking)
        if fpga_uart.in_waiting:
            car_num_4digit = fpga_uart.read_car_number() # e.g. "1234"
            assigned_slot = parking_map.allocate_nearest_empty_slot()
            
        # 2단계 & 3단계: Object Detection & Tracking
        detections = yolo_model.infer(frame)
        tracks = tracker.update(detections) # returns list of [x1, y1, x2, y2, track_id]
        
        for track in tracks:
            # FPGA 차량번호 ↔ Track ID 바인딩
            bind_fpga_data_if_in_entry_zone(track, car_num_4digit)
            
            # 4단계: Center Point (Bottom-Center)
            cx, cy = (track.x1 + track.x2)/2, track.y2
            
            # 5단계: Homography Transform to Real World Meter (X, Y)
            real_x, real_y = transform_to_real_coords(cx, cy, homography_matrix)
            
            # 6단계 & 7단계: Slot Mapping
            current_slot = parking_map.find_slot(real_x, real_y)
            parking_map.update_vehicle_location(track.id, real_x, real_y, current_slot)
            
            # 8단계: Path History Logging
            save_trajectory(track.id, real_x, real_y)
            
        # 9단계: Cross-Camera Handover Check
        perform_camera_handover(tracks)
        
        # 10단계: Update UI / Dashboard / Display
        render_monitoring_dashboard(parking_map, tracks)
```

---

## 📌 5. 프로젝트 향후 발전 및 검토 과제
1. **야간/조명 변화 대응**: HDR 카메라 설정 및 주간/야간 Homography 파라미터 이원화.
2. **소형/대형 차량 파티션**: Bounding Box 크기 기반 SUV/승용차 구분 주차면 배치.
3. **Jetson Orin Nano NVDLA / TensorRT 최적화**: GPU 사용률 모니터링을 통한 FPS 안정성(30FPS 이상) 확보.
