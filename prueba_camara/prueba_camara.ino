#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <lwip/sockets.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ----------------- DATOS DE TU WI-FI -----------------
const char* ssid     = "TU_SSID_AQUI";
const char* password = "TU_PASSWORD_AQUI";

// ----------------- PINES PANTALLA OLED I2C -----------------
#define OLED_SDA_PIN   14
#define OLED_SCL_PIN   15

#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT  64
#define OLED_ADDRESS   0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
bool oledDisponible = false;

// ----------------- PINES DE AI-THINKER ESP32-CAM -----------------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_PART = "\r\n--" PART_BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t control_httpd = NULL;
httpd_handle_t stream_httpd  = NULL;

String estadoActual = "ESPERA";
String ipLocalStr   = "0.0.0.0";

// ----------------- FUNCION DE PANTALLA OLED -----------------
void actualizarOLED(const String& estado) {
  if (estado == estadoActual && estado != "ESPERA") return;
  estadoActual = estado;

  if (!oledDisponible) return;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  if (estado == "AUTORIZADO") {
    display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(16, 12);
    display.println("ACCESO AUTORIZADO");
    display.drawFastHLine(10, 26, 108, SSD1306_WHITE);
    display.setTextSize(2);
    display.setCursor(6, 36);
    display.println("BIENVENIDO");
  } else if (estado == "DENEGADO") {
    display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(20, 12);
    display.println("ACCESO DENEGADO");
    display.drawFastHLine(10, 26, 108, SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(16, 36);
    display.println("ROSTRO NO AUTORIZADO");
    display.setCursor(28, 49);
    display.println("PUERTA CERRADA");
  } else {
    display.setTextSize(1);
    display.setCursor(10, 6);
    display.println("CONTROL BIOMETRICO");
    display.drawFastHLine(0, 18, 128, SSD1306_WHITE);
    display.setCursor(12, 28);
    display.println("ESPERANDO ROSTRO...");
    display.setCursor(6, 50);
    display.print("IP: ");
    display.println(ipLocalStr);
  }

  display.display();
}

// ----------------- HANDLER DE CONTROL OLED (/acceso) -----------------
static esp_err_t acceso_handler(httpd_req_t *req) {
  char query[128];
  char valEstado[32] = "ESPERA";

  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    if (httpd_query_key_value(query, "estado", valEstado, sizeof(valEstado)) == ESP_OK) {
      String est = String(valEstado);
      est.toUpperCase();
      if (est == "AUTORIZADO" || est == "1") {
        actualizarOLED("AUTORIZADO");
      } else if (est == "DENEGADO" || est == "0") {
        actualizarOLED("DENEGADO");
      } else {
        actualizarOLED("ESPERA");
      }
    }
  }

  String resp = "{\"ok\":true,\"estado\":\"" + estadoActual + "\"}";
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, resp.c_str(), resp.length());
}

// ----------------- HANDLER DE SNAPSHOT (/capture) -----------------
static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// ----------------- HANDLER DE VIDEO FLUIDO (STREAM MJPEG) -----------------
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];

  int sockfd = httpd_req_to_sockfd(req);
  if (sockfd >= 0) {
    int nodelay = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  }

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Fallo al capturar frame de la camara");
      res = ESP_FAIL;
    } else {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
      if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      }
      esp_camera_fb_return(fb);
      fb = NULL;
    }
    if (res != ESP_OK) {
      break;
    }
  }
  return res;
}

void startCameraServer() {
  // 1. Servidor de Control OLED (/acceso) y Snapshot (/capture) en Puerto 80
  httpd_config_t config_ctrl = HTTPD_DEFAULT_CONFIG();
  config_ctrl.server_port = 80;
  config_ctrl.ctrl_port   = 32768;
  config_ctrl.lru_purge_enable = true;

  httpd_uri_t uri_acceso = {
    .uri       = "/acceso",
    .method    = HTTP_GET,
    .handler   = acceso_handler,
    .user_ctx  = NULL
  };
  httpd_uri_t uri_capture = {
    .uri       = "/capture",
    .method    = HTTP_GET,
    .handler   = capture_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&control_httpd, &config_ctrl) == ESP_OK) {
    httpd_register_uri_handler(control_httpd, &uri_acceso);
    httpd_register_uri_handler(control_httpd, &uri_capture);
    Serial.println("Servidor de control OLED iniciado en puerto 80 (/acceso)");
  }

  // 2. Servidor dedicado al Stream de Video de Alta Velocidad en Puerto 81
  httpd_config_t config_stream = HTTPD_DEFAULT_CONFIG();
  config_stream.server_port = 81;
  config_stream.ctrl_port   = 32769;
  config_stream.lru_purge_enable = true;
  config_stream.core_id = 1;
  config_stream.task_priority = tskIDLE_PRIORITY + 6;
  config_stream.stack_size = 8192;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };
  httpd_uri_t root_stream_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config_stream) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &root_stream_uri);
    Serial.println("Servidor de video fluido iniciado en puerto 81 (/stream)!");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println("\n--- SISTEMA BIOMETRICO ESP32-CAM + OLED (FISEI UTA) ---");

  // Inicializar Pantalla OLED I2C en SDA=14, SCL=15 (400 kHz Fast I2C para no frenar camara)
  Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  Wire.setClock(400000);
  if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    oledDisponible = true;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(10, 20);
    display.println("Conectando Wi-Fi...");
    display.display();
    Serial.println("Pantalla OLED 0.96 inicializada OK!");
  } else {
    Serial.println("Aviso: Pantalla OLED no detectada en 0x3C.");
  }

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Resolucion CIF (400x296): alta fluidez (~25 FPS) y baja temperatura sin throttling
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_CIF;
    config.jpeg_quality = 14;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size   = FRAMESIZE_CIF;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Error al iniciar camara: 0x%x\n", err);
    if (oledDisponible) {
      display.clearDisplay();
      display.setCursor(10, 24);
      display.println("ERROR CAMARA");
      display.display();
    }
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_ae_level(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
  }
  Serial.println("Camara OV2640 inicializada OK!");

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_17dBm);
  Serial.print("Conectando a Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  WiFi.setSleep(false);
  ipLocalStr = WiFi.localIP().toString();
  Serial.println("\nWiFi Conectado!");
  Serial.print("Stream de video : http://");
  Serial.print(ipLocalStr);
  Serial.println(":81/stream");
  Serial.print("Control OLED    : http://");
  Serial.print(ipLocalStr);
  Serial.println("/acceso?estado=AUTORIZADO");

  startCameraServer();
  actualizarOLED("ESPERA");
}

void loop() {
  delay(10000);
}
