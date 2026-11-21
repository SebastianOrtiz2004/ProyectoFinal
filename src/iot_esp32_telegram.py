"""
Módulo de Integración IoT (ESP32-CAM + Pantalla OLED + LEDs) y Notificador Bot de Telegram
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Universidad Técnica de Ambato - FISEI - Software

Sin dependencias externas adicionales (usa únicamente librería estándar urllib + threading + OpenCV).
"""

import os
import csv
import json
import time
import uuid
import socket
import http.client
import threading
import urllib.request
import urllib.parse
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CONFIG_IOT = os.path.join(BASE_DIR, "config_iot.json")
RUTA_BITACORA_CSV = os.path.join(BASE_DIR, "figuras_y_reportes", "bitacora_accesos_iot.csv")


def cargar_config_iot() -> dict:
    """Carga config_iot.json desde la raíz del proyecto con valores seguros por defecto."""
    defecto = {
        "esp32": {
            "habilitado": False,
            "ip": "192.168.43.100",
            "puerto_control": 80,
            "puerto_stream": 81,
            "usar_camara_esp32": False,
            "rotar_180": False,
            "voltear_horizontal": True,
        },
        "telegram": {
            "habilitado": False,
            "bot_token": "",
            "chat_id": "",
            "nombre_lugar": "Laboratorio FISEI - UTA",
            "cooldown_autorizado_seg": 15,
            "cooldown_alerta_seg": 15,
            "enviar_alerta_desconocido": True,
        },
    }
    if not os.path.exists(RUTA_CONFIG_IOT):
        return defecto
    try:
        with open(RUTA_CONFIG_IOT, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sec in ("esp32", "telegram"):
            if sec in data and isinstance(data[sec], dict):
                defecto[sec].update(data[sec])
    except Exception as e:
        print(f"[AVISO IoT] No se pudo leer config_iot.json ({e}). Usando valores por defecto.")
    return defecto


class LectorCamaraThreaded:
    """
    Lector de video en hilo en segundo plano de ultra-baja latencia (Zero-Buffer).
    Compatible con:
      - Webcam local USB (fuente=0, 1) vía DirectShow (cv2.CAP_DSHOW)
      - Stream MJPEG del ESP32-CAM (fuente='http://<IP>:81/stream' o 'http://<IP>/')
      - Snapshot JPEG por HTTP como respaldo automático ('http://<IP>/capture')

    Para streams de red del ESP32-CAM:
      - Lee directamente el flujo de bytes HTTP (multipart MJPEG) sin el buffer acumulativo
        de FFmpeg que causa retrasos de 3-5 segundos y congelamiento.
      - Si se acumulan fotogramas en red por demoras de Wi-Fi, descarta automáticamente
        los paquetes viejos y decodifica EXCLUSIVAMENTE el último JPEG completo (latencia < 25ms).
      - Tolera micro-cortes de red Wi-Fi sin destruir la conexión ni trabar la interfaz.
    """

    def __init__(self, fuente=0, url_capture_respaldo: str = None, rotar_180: bool = False):
        self.fuente = fuente
        self.url_capture_respaldo = url_capture_respaldo
        self.rotar_180 = rotar_180
        self.es_red = isinstance(fuente, str) and fuente.lower().startswith("http")

        self._cap = None
        self._lock = threading.Lock()
        self._ultimo_frame = None
        self._ultimo_ok = False
        self._activo = False
        self._hilo = None
        self._nuevo_frame = False
        self._frame_id = 0
        self.fps_camara = 0.0
        self._t_ultimo_frame = 0.0

    def iniciar(self) -> bool:
        self._activo = True
        if self.es_red:
            self._hilo = threading.Thread(target=self._bucle_mjpeg_directo, daemon=True)
            self._hilo.start()
            t0 = time.time()
            while not self._ultimo_ok and (time.time() - t0) < 6.5:
                time.sleep(0.05)
            return self._ultimo_ok
        else:
            self._cap = cv2.VideoCapture(self.fuente, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.fuente)
            if not self._cap.isOpened():
                self._activo = False
                return False
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            for _ in range(5):
                ret, fr = self._cap.read()
                if ret and fr is not None:
                    self._ultimo_frame = self._ajustar_orientacion(fr)
                    self._ultimo_ok = True
                    break
                time.sleep(0.05)
            self._hilo = threading.Thread(target=self._bucle_webcam_local, daemon=True)
            self._hilo.start()
            return self._ultimo_ok

    def _ajustar_orientacion(self, frame: np.ndarray) -> np.ndarray:
        if self.rotar_180 and frame is not None:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _leer_snapshot_http(self):
        if not self.url_capture_respaldo:
            return None
        try:
            req = urllib.request.Request(self.url_capture_respaldo, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                datos = resp.read()
            arr = np.frombuffer(datos, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _publicar_frame(self, fr: np.ndarray):
        fr_adj = self._ajustar_orientacion(fr)
        # Normalizar resoluciones ligeras del ESP32 (ej. HVGA 480x320 / CIF 400x296) a 640x480
        # PRESERVANDO la relación de aspecto 1:1 (escala uniforme + recorte central) para NO
        # estirar ni deformar la geometría facial al pasar de 3:2 (HVGA) a 4:3 (640x480).
        if self.es_red and fr_adj is not None and (fr_adj.shape[1] < 600 or fr_adj.shape[0] < 440):
            h0, w0 = fr_adj.shape[:2]
            escala = max(640.0 / max(1, w0), 480.0 / max(1, h0))
            nw = int(round(w0 * escala))
            nh = int(round(h0 * escala))
            fr_esc = cv2.resize(fr_adj, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x0 = max(0, (nw - 640) // 2)
            y0 = max(0, (nh - 480) // 2)
            fr_adj = fr_esc[y0 : y0 + 480, x0 : x0 + 640]
        ahora = time.time()
        with self._lock:
            if self._t_ultimo_frame > 0.0:
                dt_c = max(1e-3, ahora - self._t_ultimo_frame)
                fps_inst = min(60.0, 1.0 / dt_c)
                self.fps_camara = fps_inst if self.fps_camara <= 0.1 else (0.82 * self.fps_camara + 0.18 * fps_inst)
            self._t_ultimo_frame = ahora
            self._ultimo_frame = fr_adj
            self._ultimo_ok = True
            self._nuevo_frame = True
            self._frame_id += 1

    def _bucle_mjpeg_directo(self):
        """
        Lector HTTP de socket directo (TCP_NODELAY + SO_RCVBUF acotado + read1 sin bloqueo)
        con búfer bytearray y purga automática para latencia cero (<25ms) y recuperación rápida.
        """
        parsed = urllib.parse.urlparse(self.fuente)
        host = parsed.hostname or "192.168.100.77"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        while self._activo:
            conn = None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=6.0)
                conn.connect()
                # Optimizar socket TCP: TCP_NODELAY + búfer amplio (512KB como navegador) para evitar
                # que Windows active TCP Zero-Window y congele la transmisión lwIP del ESP32-CAM.
                if conn.sock is not None:
                    try:
                        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 524288)
                    except Exception:
                        pass

                conn.request("GET", path, headers={"User-Agent": "Mozilla/5.0", "Connection": "keep-alive"})
                resp = conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")

                bytes_buf = bytearray()

                while self._activo:
                    # read1 retorna inmediatamente con los bytes ya llegados sin bloquear llenando el búfer
                    chunk = resp.read1(32768) if hasattr(resp, "read1") else resp.read(8192)
                    if not chunk:
                        break
                    bytes_buf.extend(chunk)

                    # Si el búfer acumuló > 160KB (varios JPEGs encolados por Wi-Fi), purgar hasta el último SOI
                    if len(bytes_buf) > 163840:
                        last_soi = bytes_buf.rfind(b"\xff\xd8")
                        if last_soi > 0:
                            del bytes_buf[:last_soi]

                    a = bytes_buf.find(b"\xff\xd8")
                    if a == -1:
                        if len(bytes_buf) > 4096:
                            bytes_buf.clear()
                        continue

                    b = bytes_buf.find(b"\xff\xd9", a + 2)
                    if b == -1:
                        if a > 0:
                            del bytes_buf[:a]
                        continue

                    # Extraer el JPEG más reciente completo dentro del búfer actual (descarta intermedios viejos)
                    jpg_bytes = bytes(bytes_buf[a : b + 2])
                    del bytes_buf[: b + 2]

                    while True:
                        a2 = bytes_buf.find(b"\xff\xd8")
                        if a2 == -1:
                            break
                        b2 = bytes_buf.find(b"\xff\xd9", a2 + 2)
                        if b2 == -1:
                            if a2 > 0:
                                del bytes_buf[:a2]
                            break
                        jpg_bytes = bytes(bytes_buf[a2 : b2 + 2])
                        del bytes_buf[: b2 + 2]

                    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    fr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if fr is not None:
                        self._publicar_frame(fr)

            except Exception:
                # Respaldo instantáneo por /capture si existe antes de reintentar stream
                if self.url_capture_respaldo:
                    fr = self._leer_snapshot_http()
                    if fr is not None:
                        self._publicar_frame(fr)
                time.sleep(0.20)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def _bucle_webcam_local(self):
        while self._activo:
            if self._cap is not None and self._cap.isOpened():
                ret, fr = self._cap.read()
                if ret and fr is not None:
                    self._publicar_frame(fr)
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def leer(self):
        with self._lock:
            if not self._ultimo_ok or self._ultimo_frame is None:
                return False, None
            self._nuevo_frame = False
            return True, self._ultimo_frame.copy()

    def leer_con_id(self):
        """Retorna (ok, frame_copy, frame_id) para saber si llegó un fotograma nuevo desde la cámara."""
        with self._lock:
            if not self._ultimo_ok or self._ultimo_frame is None:
                return False, None, 0
            self._nuevo_frame = False
            return True, self._ultimo_frame.copy(), self._frame_id

    def liberar(self):
        self._activo = False
        if self._hilo is not None:
            self._hilo.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass


class ClienteIoTESP32:
    """
    Envía las etiquetas de decisión ('AUTORIZADO', 'DENEGADO', 'ESPERA') por Wi-Fi HTTP
    al dispositivo IoT ESP32-CAM para actualizar su pantalla OLED 0.96" y LEDs físicos.
    Se ejecuta de forma asíncrona para no bloquear nunca los FPS del reconocimiento.
    """

    def __init__(self, habilitado: bool, ip: str, puerto_control: int = 80, puerto_stream: int = 80):
        # Si el stream y el control usan el MISMO puerto 80 (firmware simple de solo video como prueba_camara.ino),
        # NO mandar /acceso por ese puerto porque satura y bloquea el streaming de video del ESP32.
        self.ip = str(ip).strip()
        self.puerto = int(puerto_control)
        self.puerto_stream = int(puerto_stream)
        self.puerto_compartido = (self.puerto == self.puerto_stream)
        self.habilitado = bool(habilitado and ip and (not self.puerto_compartido))
        self._ultimo_estado = None
        self._ultimo_envio = 0.0
        self._fallos_consecutivos = 0
        self.conectado = False

    def enviar_estado(self, estado: str, total_rostros: int = 0, forzar: bool = False):
        """
        estado ∈ {'AUTORIZADO', 'DENEGADO', 'ESPERA'}
        En la pantalla OLED del ESP32 solo se mostrará 'ACCESO AUTORIZADO / BIENVENIDO'
        (sin nombres personales), 'ACCESO DENEGADO / NO AUTORIZADO' o 'ESPERANDO ROSTRO'.
        """
        if not self.habilitado:
            return
        ahora = time.time()
        # Si el endpoint /acceso no está activo (ej. sketch de solo cámara en puerto 80), espaciar reintentos
        intervalo_min = 20.0 if self._fallos_consecutivos >= 2 else 4.0
        if (not forzar) and (estado == self._ultimo_estado) and ((ahora - self._ultimo_envio) < intervalo_min):
            return
        if self._fallos_consecutivos >= 2 and ((ahora - self._ultimo_envio) < intervalo_min):
            return

        self._ultimo_estado = estado
        self._ultimo_envio = ahora

        url = f"http://{self.ip}:{self.puerto}/acceso?estado={urllib.parse.quote(estado)}&rostros={int(total_rostros)}"
        hilo = threading.Thread(target=self._hacer_peticion, args=(url,), daemon=True)
        hilo.start()

    def _hacer_peticion(self, url: str):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                self.conectado = (resp.status == 200)
                if self.conectado:
                    self._fallos_consecutivos = 0
        except Exception:
            self.conectado = False
            self._fallos_consecutivos += 1


class NotificadorTelegram:
    """
    Envía alertas y avisos de ingreso en tiempo real al Bot de Telegram con fotografía
    adjunta capturada desde el dispositivo IoT, más registro automático en bitácora CSV.
    """

    def __init__(
        self,
        habilitado: bool,
        bot_token: str,
        chat_id: str,
        nombre_lugar: str = "Laboratorio FISEI - UTA",
        cooldown_autorizado_seg: float = 15.0,
        cooldown_alerta_seg: float = 15.0,
        enviar_alerta_desconocido: bool = True,
    ):
        token_valido = bool(bot_token) and ("PON_AQUI" not in bot_token)
        chat_valido = bool(chat_id) and ("PON_AQUI" not in str(chat_id))
        self.habilitado = bool(habilitado and token_valido and chat_valido)
        self.bot_token = str(bot_token).strip()
        self.chat_id = str(chat_id).strip()
        self.nombre_lugar = nombre_lugar
        self.cooldown_autorizado = float(cooldown_autorizado_seg)
        self.cooldown_alerta = float(cooldown_alerta_seg)
        self.enviar_alerta_desconocido = bool(enviar_alerta_desconocido)

        self._ultimo_envio_aut = 0.0
        self._ultimo_envio_alerta = 0.0
        self._asegurar_bitacora()

    def _asegurar_bitacora(self):
        os.makedirs(os.path.dirname(RUTA_BITACORA_CSV), exist_ok=True)
        if not os.path.exists(RUTA_BITACORA_CSV):
            try:
                with open(RUTA_BITACORA_CSV, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["fecha_hora", "evento", "lugar", "total_rostros", "autorizados", "desconocidos", "detalle"])
            except Exception:
                pass

    def _registrar_bitacora(self, evento: str, total: int, aut: int, desc: int, detalle: str):
        try:
            with open(RUTA_BITACORA_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), evento, self.nombre_lugar, total, aut, desc, detalle])
        except Exception:
            pass

    def notificar_ingreso_autorizado(self, frame_bgr: np.ndarray, total_rostros: int, num_autorizados: int, nombres_aut: list):
        ahora = time.time()
        if (ahora - self._ultimo_envio_aut) < self.cooldown_autorizado:
            return
        self._ultimo_envio_aut = ahora

        fecha_str = time.strftime("%Y-%m-%d %H:%M:%S")
        detalle = ", ".join(nombres_aut) if nombres_aut else "Usuario autorizado"
        self._registrar_bitacora("ACCESO_AUTORIZADO", total_rostros, num_autorizados, total_rostros - num_autorizados, detalle)

        if not self.habilitado:
            return

        mensaje = (
            f"ACCESO AUTORIZADO\n"
            f"Lugar: {self.nombre_lugar}\n"
            f"Fecha y hora: {fecha_str}\n"
            f"Rostros en camara: {total_rostros} (Autorizados: {num_autorizados})\n"
            f"Estado: Ingreso permitido por el sistema biometrico IoT."
        )
        frame_copia = frame_bgr.copy() if frame_bgr is not None else None
        threading.Thread(target=self._enviar_telegram, args=(mensaje, frame_copia), daemon=True).start()

    def notificar_alerta_desconocido(self, frame_bgr: np.ndarray, total_rostros: int, num_desconocidos: int):
        if not self.enviar_alerta_desconocido:
            return
        ahora = time.time()
        if (ahora - self._ultimo_envio_alerta) < self.cooldown_alerta:
            return
        self._ultimo_envio_alerta = ahora

        fecha_str = time.strftime("%Y-%m-%d %H:%M:%S")
        self._registrar_bitacora("ACCESO_DENEGADO", total_rostros, 0, num_desconocidos, "Rostro desconocido")

        if not self.habilitado:
            return

        mensaje = (
            f"ALERTA: INTENTO DE ACCESO NO AUTORIZADO\n"
            f"Lugar: {self.nombre_lugar}\n"
            f"Fecha y hora: {fecha_str}\n"
            f"Rostros desconocidos detectados: {num_desconocidos}\n"
            f"Estado: Puerta bloqueada. Se adjunta evidencia capturada."
        )
        frame_copia = frame_bgr.copy() if frame_bgr is not None else None
        threading.Thread(target=self._enviar_telegram, args=(mensaje, frame_copia), daemon=True).start()

    def _enviar_telegram(self, caption: str, frame_bgr):
        try:
            if frame_bgr is not None:
                ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    self._post_multipart_foto(caption, buf.tobytes())
                    return
            # Respaldo solo texto
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": caption}).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=5.0):
                pass
        except Exception as e:
            print(f"[AVISO TELEGRAM] No se pudo enviar notificación: {e}")

    def _post_multipart_foto(self, caption: str, jpeg_bytes: bytes):
        boundary = f"----IoTBioBoundary{uuid.uuid4().hex}"
        partes = []

        for campo, valor in [("chat_id", self.chat_id), ("caption", caption)]:
            partes.append(f"--{boundary}\r\n".encode("utf-8"))
            partes.append(f'Content-Disposition: form-data; name="{campo}"\r\n\r\n'.encode("utf-8"))
            partes.append(f"{valor}\r\n".encode("utf-8"))

        partes.append(f"--{boundary}\r\n".encode("utf-8"))
        partes.append(
            b'Content-Disposition: form-data; name="photo"; filename="captura_iot.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n"
        )
        partes.append(jpeg_bytes)
        partes.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        cuerpo = b"".join(partes)

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        req = urllib.request.Request(
            url,
            data=cuerpo,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=6.0):
            pass
