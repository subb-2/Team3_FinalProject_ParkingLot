# 2단계 : 차량 객체 검출 (Vehicle Detection)

## 목적

입력된 카메라 영상에서 **차량만 검출**하여 Bounding Box와 Confidence를 실시간으로 표시합니다.

---

## 개요

`vehicle_detection.py`는 **YOLOv8n COCO Pretrained 모델**을 사용하여 카메라 영상에서 차량 객체를 실시간으로 검출하는 스크립트입니다.  
COCO 데이터셋의 80개 클래스 중 **차량 관련 클래스(Car, Motorcycle, Bus, Truck)**만 필터링하여 검출하므로 별도의 모델 학습이 필요하지 않습니다.

---

## 검출 대상 클래스

| COCO Class ID | 클래스명 | Bounding Box 색상 |
|:---:|---|---|
| 2 | **Car** (승용차) | 🟩 초록 |
| 3 | **Motorcycle** (오토바이) | 🟧 주황 |
| 5 | **Bus** (버스) | 🟦 파랑 |
| 7 | **Truck** (트럭) | 🟥 빨강 |

> 위 4개 클래스 외의 객체(사람, 자전거, 신호등 등)는 자동으로 필터링되어 표시되지 않습니다.

---

## 환경 요구사항

### 하드웨어
- NVIDIA Jetson Orin Nano
- USB 웹캠 / CSI 카메라 / IP 카메라 중 하나 이상

### 소프트웨어
- **JetPack SDK** (권장: 5.x 이상)
- **Python 3.8+**
- **OpenCV** (GStreamer 지원 빌드, JetPack 기본 포함)
- **Ultralytics (YOLOv8)**

### 패키지 설치

```bash
pip install ultralytics
```

> **참고:** `yolov8n.pt` 모델 파일은 첫 실행 시 자동으로 다운로드됩니다.

---

## 기본 설정값

| 항목 | 값 |
|------|---|
| **모델** | YOLOv8n (Nano) |
| **Pretrained 데이터셋** | COCO (80 클래스) |
| **필터링 클래스** | Car, Motorcycle, Bus, Truck |
| **USB 카메라 해상도** | 640 x 480 |
| **CSI 카메라 해상도** | 1280 x 720 |
| **FPS 목표** | 30 FPS |

---

## 실행 방법

### USB 카메라 (기본값)
```bash
python3 code/vehicle_detection.py
```

### CSI 카메라
```bash
python3 code/vehicle_detection.py 2
```

### IP 카메라 (RTSP)
```bash
python3 code/vehicle_detection.py 3
```

> **주의:** IP 카메라 사용 시 코드 내 `rtsp_url` 변수를 실제 CCTV 주소로 변경해야 합니다.

---

## 출력

### 화면 출력
- **Bounding Box**: 검출된 차량마다 클래스별 색상의 사각형 표시
- **라벨**: 각 박스 상단에 `클래스명 + Confidence` 표시 (예: `Car 0.94`)
- **FPS**: 화면 좌측 상단에 초록색 텍스트
- **검출 수**: 화면 좌측에 현재 프레임의 차량 검출 수 표시

### 터미널 출력 (예시)

```
[FPS: 28.5] 검출 차량 2대:
  Car           x=320  y=180  w=160  h=90   conf=0.94
  Truck         x=480  y=200  w=200  h=120  conf=0.87
```

| 필드 | 설명 |
|------|------|
| `class` | 차량 클래스명 (Car / Motorcycle / Bus / Truck) |
| `x`, `y` | Bounding Box 좌측 상단 좌표 (px) |
| `w`, `h` | Bounding Box 너비, 높이 (px) |
| `confidence` | 검출 신뢰도 (0.0 ~ 1.0) |

### 종료 방법
영상 창을 선택한 상태에서 **`q` 키**를 누르면 종료됩니다.

---

## 확인 사항 체크리스트

| 확인 항목 | 확인 방법 |
|----------|----------|
| **차량을 놓치지 않는가** | 영상 내 모든 차량에 Bounding Box가 그려지는지 확인 |
| **차량 이외의 객체를 잘못 검출하지 않는지** | 사람, 자전거 등 비차량 객체에 박스가 없는지 확인 |
| **실시간 처리가 가능한 FPS가 나오는지** | 화면 상단 FPS 수치 확인 (Jetson Orin Nano 기준 25~40 FPS 기대) |
| **Confidence가 적절한가** | 확실한 차량의 conf > 0.5, 애매한 경우 conf < 0.5 |

---

## 파일 구조

```
final_team/
├── camera_input.py                    # 1단계 - 카메라 영상 입력 확인
├── camera_input_readme.md             # 1단계 README
└── code/
    ├── vehicle_detection.py           # 2단계 - 차량 객체 검출
    └── vehicle_detection_readme.md    # 2단계 README (본 파일)
```

---

## 코드 구조

```
vehicle_detection.py
│
├── 상수 정의
│   ├── VEHICLE_CLASS_IDS      # 차량 클래스 ID 집합 {2, 3, 5, 7}
│   ├── VEHICLE_CLASS_NAMES    # ID → 이름 매핑
│   └── BOX_COLORS             # 클래스별 Bounding Box 색상
│
├── GStreamer 파이프라인 함수
│   ├── get_usb_pipeline()     # USB 카메라
│   ├── get_csi_pipeline()     # CSI 카메라
│   └── get_rtsp_pipeline()    # IP 카메라 (RTSP)
│
├── open_camera()              # 카메라 타입에 따른 VideoCapture 생성
├── draw_vehicle_boxes()       # YOLO 결과에서 차량 필터링 및 시각화
└── main()                     # 메인 루프 (추론 → 시각화 → 출력)
```

---

## 트러블슈팅

| 증상 | 원인 및 해결 |
|------|-------------|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` 실행 |
| 모델 다운로드 실패 | 인터넷 연결 확인. 수동 다운로드 후 같은 디렉토리에 `yolov8n.pt` 배치 |
| FPS가 매우 낮음 (< 10) | 입력 해상도를 낮추거나, `model(frame, imgsz=320)`으로 추론 해상도 축소 |
| 차량을 잘 검출하지 못함 | Confidence 임계값 조정: `model(frame, conf=0.3)` |
| 카메라를 열 수 없음 | 1단계 `camera_input.py`로 카메라 정상 동작 먼저 확인 |

---

## 성능 참고 (Jetson Orin Nano 기준)

| 모델 | 입력 해상도 | 예상 FPS |
|------|-----------|---------|
| YOLOv8n | 640 x 480 | ~30-40 FPS |
| YOLOv8n | 1280 x 720 | ~15-25 FPS |
| YOLOv8s | 640 x 480 | ~15-20 FPS |

> YOLOv8n(Nano)은 경량 모델로, Jetson Orin Nano에서 실시간 처리에 적합합니다.

---

## 다음 단계

✅ 차량 검출이 정상적으로 확인되면 **3단계**로 진행합니다.
