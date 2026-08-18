#include <WiFi.h>

// ===== [사용자 설정 영역] =====
const char* ssid = "kcci603_2.4g";        // Wi-Fi 공유기 이름
const char* password = "kcci603_2.4g";    // Wi-Fi 공유기 비밀번호

// 접속할 Jetson TCP 서버 설정
const char* server_ip = "192.168.0.50"; // Jetson IP 주소로 수정 필요
const int server_port = 5001;           // Jetson TCP 서버 포트 번호

WiFiClient client;
// ============================

void setup() {
  // 1. PC 디버깅용 시리얼 (USB 케이블)
  Serial.begin(115200);
  
  // 2. Board 수신용 HardwareSerial2 (RX2=GPIO 4, TX2=GPIO 17)
  Serial2.begin(115200, SERIAL_8N1, 4, 17); 
  
  // 3. Wi-Fi 연결 시작
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected.");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());
}

// Jetson 서버 재연결 전담 함수
void connectToJetsonServer() {
  while (!client.connected()) {
    Serial.printf("Connecting to Jetson Server (%s:%d)...\n", server_ip, server_port);
    
    if (client.connect(server_ip, server_port)) {
      Serial.println("[Info] Connected to Jetson Server!");
    } else {
      Serial.println("[Error] Connection failed. Retrying in 2 seconds...");
      delay(2000);
    }
  }
}

void loop() {
  // 1. Jetson 서버와의 연결 상태 유지 (끊어지면 자동 재연결)
  if (!client.connected()) {
    connectToJetsonServer();
  }

  // 2. Board로부터 UART 데이터가 2바이트 이상 수신되었는지 확인
  if (Serial2.available() >= 2) {
    uint8_t data1 = Serial2.read(); // 첫 번째 8-bit 데이터 (MSB)
    uint8_t data2 = Serial2.read(); // 두 번째 8-bit 데이터 (LSB)

    // 2바이트 합성 (16비트 값)
    uint16_t combined_data = ((uint16_t)data1 << 8) | data2;

    // 시리얼 모니터 디버깅 출력
    Serial.printf("Received from Basys3 -> 0x%02X, 0x%02X | Combined: 0x%04X (%d)\n", 
                  data1, data2, combined_data, combined_data);

    // 3. Jetson TCP 서버로 8-bit 데이터 2번 연속 전송 (바이너리 Raw 데이터)
    uint8_t payload[2] = { data1, data2 };
    size_t sent_bytes = client.write(payload, 2);

    if (sent_bytes == 2) {
      Serial.println(" -> Successfully sent 2 bytes to Jetson.");
    } else {
      Serial.println(" -> Send failed or partial send.");
    }
  }
}