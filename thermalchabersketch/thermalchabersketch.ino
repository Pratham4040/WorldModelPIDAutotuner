#include <WiFi.h>
#include <WebServer.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <esp_arduino_version.h>

// ===========================
// User config
// ===========================
static const char* SSID     = "Airtel_1 New virus found";
static const char* PASSWORD = "2829@504FiWi";

// Pin mapping (set these to your wiring)
static const int ONE_WIRE_BUS_PIN = 4;   // DS18B20 data pin
static const int MOSFET_PIN       = 26;  // Heater MOSFET gate pin
static const int LED_PIN          = 2;   // Status LED

// Safety
static const float MAX_TEMP_C = 39.0f;        // Hardware safety cutoff
static const uint32_t CMD_TIMEOUT_MS = 5000;  // If no PWM command, force heater OFF

// PWM config
static const uint32_t PWM_FREQ_HZ = 2000;
static const uint8_t PWM_RES_BITS = 8; // 0..255
static const uint8_t PWM_CHANNEL  = 0; // Used on Arduino-ESP32 v2.x APIs

// ===========================
// Globals
// ===========================
WebServer server(80);
OneWire oneWire(ONE_WIRE_BUS_PIN);
DallasTemperature sensors(&oneWire);

int currentPWM = 0;
uint32_t lastPwmCommandMs = 0;

// ===========================
// PWM abstraction for core v2/v3
// ===========================
void pwmInit() {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcAttach(MOSFET_PIN, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcWrite(MOSFET_PIN, 0);
#else
  ledcSetup(PWM_CHANNEL, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcAttachPin(MOSFET_PIN, PWM_CHANNEL);
  ledcWrite(PWM_CHANNEL, 0);
#endif
}

void pwmWriteValue(int pwm) {
  pwm = constrain(pwm, 0, 255);
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  ledcWrite(MOSFET_PIN, pwm);
#else
  ledcWrite(PWM_CHANNEL, pwm);
#endif
}

// ===========================
// Helpers
// ===========================
float readTemperatureC() {
  sensors.requestTemperatures();
  float t = sensors.getTempCByIndex(0);
  return t;
}

bool isTempValid(float t) {
  return (t != DEVICE_DISCONNECTED_C) && !isnan(t) && (t > -80.0f) && (t < 150.0f);
}

bool isSafeTemp(float t) {
  return isTempValid(t) && (t < MAX_TEMP_C);
}

void forceHeaterOff() {
  currentPWM = 0;
  pwmWriteValue(0);
}

// ===========================
// HTTP handlers
// ===========================
void handleTemp() {
  float t = readTemperatureC();

  if (!isTempValid(t)) {
    forceHeaterOff();
    server.send(500, "text/plain", "nan,SENSOR_ERROR");
    return;
  }

  if (t >= MAX_TEMP_C) {
    forceHeaterOff();
    server.send(200, "text/plain", String(t, 4) + ",SAFETY");
    return;
  }

  server.send(200, "text/plain", String(t, 4));
  Serial.println(t);
}

void handleSetPWM() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "No PWM value");
    return;
  }

  int pwm = server.arg("plain").toInt();
  pwm = constrain(pwm, 0, 255);

  float t = readTemperatureC();
  if (!isTempValid(t) || t >= MAX_TEMP_C) {
    pwm = 0;
  }

  currentPWM = pwm;
  pwmWriteValue(currentPWM);
  lastPwmCommandMs = millis();

  server.send(200, "text/plain", "OK:" + String(currentPWM));
}

void handleStatus() {
  float t = readTemperatureC();

  bool safe = isSafeTemp(t);
  if (!safe) {
    forceHeaterOff();
  }

  String json = "{\"temp\":";
  if (isTempValid(t)) {
    json += String(t, 4);
  } else {
    json += "null";
  }
  json += ",\"pwm\":" + String(currentPWM);
  json += ",\"safe\":" + String(safe ? "true" : "false") + "}";

  server.send(200, "application/json", json);
}

void handleRoot() {
  String msg = "ESP32 Thermal Control Ready\n";
  msg += "GET  /temp\n";
  msg += "POST /pwm (plain body: 0..255)\n";
  msg += "GET  /status\n";
  server.send(200, "text/plain", msg);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// ===========================
// Setup / Loop
// ===========================
void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  sensors.begin();
  pwmInit();
  forceHeaterOff();

  Serial.printf("\nConnecting to %s", SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
  digitalWrite(LED_PIN, HIGH);

  server.on("/", HTTP_GET, handleRoot);
  server.on("/temp", HTTP_GET, handleTemp);
  server.on("/pwm", HTTP_POST, handleSetPWM);
  server.on("/status", HTTP_GET, handleStatus);
  server.onNotFound(handleNotFound);
  server.begin();

  lastPwmCommandMs = millis();

  Serial.println("\n\n=== ESP32 READY ===");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println("Endpoints:");
  Serial.println("  GET  /temp");
  Serial.println("  POST /pwm");
  Serial.println("  GET  /status");
  Serial.println("Use this IP in your PC script --esp-ip");
}

void loop() {
  server.handleClient();

  // Command-timeout failsafe
  if (millis() - lastPwmCommandMs > CMD_TIMEOUT_MS) {
    forceHeaterOff();
  }

  // LED heartbeat:
  // slow blink when receiving commands recently, fast blink otherwise
  static uint32_t lastBlinkMs = 0;
  uint32_t interval = (millis() - lastPwmCommandMs <= 3000) ? 1000 : 250;
  if (millis() - lastBlinkMs >= interval) {
    lastBlinkMs = millis();
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
}