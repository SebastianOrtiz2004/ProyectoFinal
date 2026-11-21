/*
 * =============================================================================
 * EJEMPLO DE PRUEBA: ESP32-CAM (AI-Thinker) Web Server de Video
 * =============================================================================
 * Este sketch simple inicia la camara OV2640, se conecta a tu Wi-Fi
 * e inicia un servidor web para que puedas ver el video en vivo
 * desde cualquier navegador web en tu laptop o celular.
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include <lwip/sockets.h>

// ----------------- DATOS DE TU WI-FI -----------------
const char* ssid     = "NETLIFE-ambwportizo1";
const char* password = "1802891919";

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

httpd_handle_t stream_httpd = NULL;

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];

  // Desactivar algoritmo de Nagle (TCP_NODELAY) para eliminar micro-pausas entre frames
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
      // Enviar boundary + cabecera juntos (2 chunks por frame en lugar de 3 -> menos overhead TCP)
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
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.lru_purge_enable = true;
  config.core_id = 1;                    // Ejecutar servidor HTTP en Core 1 (Wi-Fi/lwIP corre en Core 0)
  config.task_priority = tskIDLE_PRIORITY + 6;
  config.stack_size = 8192;

  httpd_uri_t stream_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("Servidor de video iniciado correctamente en el puerto 80!");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println();
  Serial.println("--- PRUEBA DE CAMARA ESP32-CAM ---");

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

  if (psramFound()) {
    config.frame_size   = FRAMESIZE_CIF;  // 400x296: ~8KB/frame -> 28-35 FPS muy fluidos
    config.jpeg_quality = 14;             // Buena nitidez facial y peso muy liviano
    config.fb_count     = 2;              // Más estable en bus PSRAM que 3
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size   = FRAMESIZE_CIF;
    config.jpeg_quality = 14;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  // Inicializar camara
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Error al iniciar camara: 0x%x\n", err);
    return;
  }
  // Exposición automática estándar OV2640 (misma fotometría con la que se capturó el dataset)
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_ae_level(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
  }
  Serial.println("Camara OV2640 inicializada OK!");

  // Conectar a Wi-Fi (sin modo ahorro para 0 lag y potencia estable para evitar reinicios por USB)
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_17dBm);
  Serial.print("Conectando a Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  WiFi.setSleep(false); // Reafirmar sin ahorro de energía tras conectar
  Serial.println("\nWiFi Conectado!");
  Serial.print("Abre tu navegador y entra a: http://");
  Serial.println(WiFi.localIP());

  startCameraServer();
}

void loop() {
  delay(10000);
}
