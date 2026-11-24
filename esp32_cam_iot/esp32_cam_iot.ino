/*
 * =============================================================================
 * Dispositivo IoT ESP32-CAM (AI-Thinker) + Pantalla OLED 0.96" I2C + LEDs
 * Proyecto Final - Inteligencia Artificial (FISEI - UTA)
 * Autores: Sebastián Ortiz & Ángel Ayuquina
 * =============================================================================
 *
 * ARQUITECTURA IoT + EDGE AI:
 *   1. El ESP32-CAM captura video en vivo (640x480 VGA) y lo transmite por Wi-Fi
 *      a la laptop en:
 *         - Stream MJPEG:  http://<IP_ESP32>:81/stream
 *         - Snapshot JPEG: http://<IP_ESP32>/capture
 *   2. La laptop ejecuta todo el pipeline PDI + PCA + ML/CNN Multi-Rostro.
 *   3. La laptop devuelve la etiqueta de decisión al ESP32-CAM en:
 *         - http://<IP_ESP32>/acceso?estado=AUTORIZADO
 *         - http://<IP_ESP32>/acceso?estado=DENEGADO
 *         - http://<IP_ESP32>/acceso?estado=ESPERA
 *   4. El ESP32-CAM muestra en la pantalla OLED "ACCESO AUTORIZADO / BIENVENIDO"
 *      (sin nombres personales) o "ACCESO DENEGADO" y enciende el LED verde/rojo.
 *
 * CONEXIONES FÍSICAS (NO insertar tarjeta microSD):
 *   - Pantalla OLED 0.96" I2C (SSD1306, dirección 0x3C):
 *       VCC -> 3.3V (o 5V si tu módulo OLED soporta 3.3V-5V)
 *       GND -> GND
 *       SDA -> GPIO 14
 *       SCL -> GPIO 15
 *   - LED Verde (Autorizado):
 *       GPIO 12 -> Resistencia 220 ohm -> Ánodo (+) LED Verde -> Cátodo (-) -> GND
 *   - LED Rojo (Denegado / Cerrado):
 *       GPIO 13 -> Resistencia 220 ohm -> Ánodo (+) LED Rojo  -> Cátodo (-) -> GND
 *
 * LIBRERÍAS REQUERIDAS EN ARDUINO IDE (Gestor de Bibliotecas):
 *   - "Adafruit SSD1306"
 *   - "Adafruit GFX Library"
 * Placa en Arduino IDE: "AI Thinker ESP32-CAM" (paquete esp32 by Espressif)
 * =============================================================================
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ===================== CONFIGURACIÓN WI-FI =====================
// Red Wi-Fi configurada
const char* WIFI_SSID     = "infinix";
const char* WIFI_PASSWORD = "12345678@";

// ===================== PINES DE HARDWARE =======================
#define OLED_SDA_PIN   14
#define OLED_SCL_PIN   15
#define LED_VERDE_PIN  12
#define LED_ROJO_PIN   13

#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT  64
#define OLED_ADDRESS   0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
bool oledDisponible = false;

// Pines de cámara del modelo AI-THINKER ESP32-CAM
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

httpd_handle_t control_httpd = NULL;
httpd_handle_t stream_httpd  = NULL;

String estadoActual = "ESPERA";
String ipLocalStr   = "0.0.0.0";

// =============================================================================
// FUNCIONES DE PANTALLA OLED Y ACTUADORES LED
// =============================================================================
void actualizarOLEDyLEDs(const String& estado) {
  estadoActual = estado;

  if (estado == "AUTORIZADO") {
    digitalWrite(LED_VERDE_PIN, HIGH);
    digitalWrite(LED_ROJO_PIN, LOW);
  } else if (estado == "DENEGADO") {
    digitalWrite(LED_VERDE_PIN, LOW);
    digitalWrite(LED_ROJO_PIN, HIGH);
  } else {
    // ESPERA
    digitalWrite(LED_VERDE_PIN, LOW);
    digitalWrite(LED_ROJO_PIN, LOW);
  }

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

// =============================================================================
// HANDLERS HTTP (CONTROL / CAPTURA / STREAM MJPEG)
// =============================================================================
static esp_err_t acceso_handler(httpd_req_t *req) {
  char query[128];
  char valEstado[32] = "ESPERA";

  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    if (httpd_query_key_value(query, "estado", valEstado, sizeof(valEstado)) == ESP_OK) {
      String est = String(valEstado);
      est.toUpperCase();
      if (est == "AUTORIZADO" || est == "1") {
        actualizarOLEDyLEDs("AUTORIZADO");
      } else if (est == "DENEGADO" || est == "0") {
        actualizarOLEDyLEDs("DENEGADO");
      } else {
        actualizarOLEDyLEDs("ESPERA");
      }
    }
  }

  String resp = "{\"ok\":true,\"estado\":\"" + estadoActual + "\"}";
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, resp.c_str(), resp.length());
}

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

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len);
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      esp_camera_fb_return(fb);
      fb = NULL;
    }
    if (res != ESP_OK) break;
  }
  return res;
}

void iniciarServidoresHTTP() {
  // Servidor de Control y Snapshot en Puerto 80
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port   = 32768;

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

  if (httpd_start(&control_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(control_httpd, &uri_acceso);
    httpd_register_uri_handler(control_httpd, &uri_capture);
  }

  // Servidor de Stream MJPEG en Puerto 81
  config.server_port = 81;
  config.ctrl_port   = 32769;
  httpd_uri_t uri_stream = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &uri_stream);
  }
}

bool inicializarCamaraOV2640() {
  camera_config_t config;
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

  // Resolución VGA (640x480) equilibrada para multi-rostro + buena velocidad Wi-Fi
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_HVGA; // 480x320: ~14KB/frame -> 22-28 FPS estables sin lag por Wi-Fi
    config.jpeg_quality = 16;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size   = FRAMESIZE_CIF;
    config.jpeg_quality = 16;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Fallo inicializando camara: 0x%x\n", err);
    return false;
  }

  // Calibración óptica del sensor OV2640 para rostros en interiores
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 0);
    s->set_contrast(s, 1);         // Ligero refuerzo de contraste facial
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);         // Balance de blancos automático
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);    // Exposición automática
    s->set_aec2(s, 1);             // DSP auto-exposición mejorada
    s->set_ae_level(s, -1);        // Evita sobreexposición blanca (quemado) en rostros bajo luz intensa
    s->set_gain_ctrl(s, 1);
    s->set_gainceiling(s, (gainceiling_t)2); // Limitar ganancia máxima para reducir ruido ISO
    s->set_bpc(s, 1);              // Corrección de píxeles negros
    s->set_wpc(s, 1);              // Corrección de píxeles blancos
    s->set_lenc(s, 1);             // Corrección de viñeteo de lente
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== INICIANDO ESP32-CAM IoT (FISEI - UTA) ===");

  pinMode(LED_VERDE_PIN, OUTPUT);
  pinMode(LED_ROJO_PIN, OUTPUT);
  digitalWrite(LED_VERDE_PIN, LOW);
  digitalWrite(LED_ROJO_PIN, HIGH); // Rojo mientras conecta

  // Inicializar I2C en pines libres (SDA=14, SCL=15) para OLED 0.96"
  Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    oledDisponible = true;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(10, 20);
    display.println("Conectando Wi-Fi...");
    display.display();
  } else {
    Serial.println("[AVISO] Pantalla OLED 0x3C no detectada. Continuando solo con LEDs.");
  }

  if (!inicializarCamaraOV2640()) {
    if (oledDisponible) {
      display.clearDisplay();
      display.setCursor(10, 20);
      display.println("ERROR CAMARA OV2640");
      display.display();
    }
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);  // Evitar latencia Wi-Fi en el stream de video

  Serial.print("Conectando a Wi-Fi");
  int intentos = 0;
  while (WiFi.status() != WL_CONNECTED && intentos < 40) {
    delay(400);
    Serial.print(".");
    intentos++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    ipLocalStr = WiFi.localIP().toString();
    Serial.println("\n[OK] Wi-Fi Conectado!");
    Serial.print("IP del ESP32-CAM: ");
    Serial.println(ipLocalStr);
    Serial.println("Stream MJPEG : http://" + ipLocalStr + ":81/stream");
    Serial.println("Control OLED : http://" + ipLocalStr + "/acceso?estado=AUTORIZADO");
    iniciarServidoresHTTP();
    actualizarOLEDyLEDs("ESPERA");
  } else {
    Serial.println("\n[ERROR] No se pudo conectar al Wi-Fi.");
    if (oledDisponible) {
      display.clearDisplay();
      display.setCursor(8, 24);
      display.println("ERROR WI-FI");
      display.display();
    }
  }
}

void loop() {
  delay(100);
}
