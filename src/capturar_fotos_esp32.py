"""
=============================================================================
CAPTURA DE DATASET FACIAL CON ESP32-CAM (STREAMING AP DIRECTO)
Proyecto Final - Inteligencia Artificial (FISEI - UTA)
Autores: Sebastian Ortiz & Angel Ayuquina
=============================================================================
Instrucciones:
  1. Conecta tu laptop a la red Wi-Fi que creo el ESP32:
     - SSID: ESP32-CAM-TEST
     - Clave: 12345678
  2. Ejecuta este script con:
     python src/capturar_fotos_esp32.py
  3. Ingresa tu ID (0: Sebastian, 1: Angel) o un nuevo ID.
  4. Controles en la ventana de video:
     - [ESPACIO]: Guarda foto actual (solo si pasa el filtro de nitidez).
     - [A]: Modo Rafaga Automatica (guarda fotos nitidas cada ~400ms).
     - [ESC] o [Q]: Salir y guardar.
=============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np

# URL del streaming del ESP32-CAM en la red de tu casa
STREAM_URL = "http://192.168.100.77/"

# Carpeta base donde se guardan las imagenes del dataset
CARPETA_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def calcular_nitidez(imagen_bgr):
    """Calcula la varianza del Laplaciano para evitar fotos borrosas."""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gris, cv2.CV_64F).var()

def main():
    print("=" * 70)
    print(" CAPTURA DE FOTOS CON ESP32-CAM (http://192.168.4.1/)")
    print("=" * 70)
    print("IMPORTANTE: Asegurate de que tu laptop este conectada a la red Wi-Fi:")
    print("  SSID:  ESP32-CAM-TEST")
    print("  Clave: 12345678\n")

    # Seleccionar Sujeto
    print("Elige a quien le vas a tomar fotos:")
    print("  [0] Sebastian Ortiz (id_0_sebastian)")
    print("  [1] Angel Ayuquina  (id_1_angel)")
    print("  [2] Nueva Persona / Personalizado")
    opcion = input("Selecciona (0, 1 o 2) [Default 0]: ").strip()

    if opcion == "1":
        nombre_carpeta = "id_1_angel"
        nombre_persona = "Angel Ayuquina"
    elif opcion == "2":
        nombre_persona = input("Nombre completo de la persona: ").strip()
        slug = nombre_persona.lower().replace(" ", "_")
        nombre_carpeta = f"id_custom_{slug}"
    else:
        nombre_carpeta = "id_0_sebastian"
        nombre_persona = "Sebastian Ortiz"

    ruta_destino = os.path.join(CARPETA_DATA, nombre_carpeta)
    os.makedirs(ruta_destino, exist_ok=True)

    # Contar fotos existentes para no sobreescribir
    existentes = [f for f in os.listdir(ruta_destino) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    contador = len(existentes)
    print(f"\n[INFO] Guardando fotos en: {ruta_destino}")
    print(f"[INFO] Fotos ya existentes en esta carpeta: {contador}")

    print("\n[INFO] Conectando a la camara ESP32 en:", STREAM_URL)
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("\n[ERROR] No se pudo abrir el stream de la camara.")
        print("Revisa:")
        print("  1. Que tu laptop este conectada a la red 'ESP32-CAM-TEST'.")
        print("  2. Que puedas ver la camara en tu navegador entrando a: http://192.168.4.1/")
        return

    print("\n" + "=" * 50)
    print(" ¡CAMARA CONECTADA CON EXITO!")
    print(" CONTROLES DE CAPTURA:")
    print("   [ESPACIO] : Tomar 1 foto")
    print("   [A]       : Activar/Desactivar Rafaga Automatica")
    print("   [Q] o [ESC]: Terminar y Salir")
    print("=" * 50 + "\n")

    modo_rafaga = False
    ultimo_disparo = 0
    INTERVALO_RAFAGA_SEG = 0.35  # guarda cada ~350 ms si el rostro esta nitido
    fotos_sesion = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[AVISO] Esperando frame de la camara...")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        nitidez = calcular_nitidez(frame)
        esta_nitida = nitidez >= 18.0

        # Copia para visualizacion con textos de ayuda
        display = frame.copy()

        # Cuadro guia central (para sugerir centrar el rostro)
        ch, cw = int(h * 0.75), int(w * 0.50)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - cw // 2, cy - ch // 2
        x2, y2 = cx + cw // 2, cy + ch // 2
        color_guia = (0, 255, 0) if esta_nitida else (0, 0, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color_guia, 2)

        # Barra superior con estado
        cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(display, f"SUJETO: {nombre_persona} | TOTAL: {contador} (+{fotos_sesion})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        estado_rafaga = "ACTIVA [A]" if modo_rafaga else "INACTIVA"
        color_rafaga = (0, 255, 255) if modo_rafaga else (180, 180, 180)
        cv2.putText(display, f"Nitidez: {nitidez:.1f} | Rafaga: {estado_rafaga}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_rafaga, 1)

        cv2.putText(display, "[ESPACIO] Foto  [A] Rafaga  [Q] Salir",
                    (w - 280, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow("Captura de Dataset - ESP32-CAM", display)

        # Logica de guardado automatico en modo rafaga
        ahora = time.time()
        debe_guardar = False

        if modo_rafaga and (ahora - ultimo_disparo >= INTERVALO_RAFAGA_SEG):
            if esta_nitida:
                debe_guardar = True

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):  # Espacio -> guardar manual
            if esta_nitida:
                debe_guardar = True
            else:
                print(f"[AVISO] Imagen algo borrosa (Nitidez: {nitidez:.1f} < 18.0). Mantente quieto y vuelve a presionar.")
                debe_guardar = True  # Aun asi permitimos si el usuario insiste

        elif tecla in (ord('a'), ord('A')):
            modo_rafaga = not modo_rafaga
            estado_txt = "ACTIVADA" if modo_rafaga else "DESACTIVADA"
            print(f"[INFO] Rafaga automatica {estado_txt}")

        elif tecla in (ord('q'), ord('Q'), 27):  # 27 = ESC
            print("\n[INFO] Finalizando captura...")
            break

        if debe_guardar:
            contador += 1
            fotos_sesion += 1
            nombre_archivo = f"{nombre_carpeta}_{contador:04d}.jpg"
            ruta_archivo = os.path.join(ruta_destino, nombre_archivo)
            
            # Guardamos con imencode + open() para ser inmunes a caracteres especiales/tildes en Windows (ej. TÉCNICA)
            exito, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if exito:
                with open(ruta_archivo, 'wb') as f:
                    f.write(buffer)
                print(f" -> Guardada [{fotos_sesion}]: {nombre_archivo} (Nitidez: {nitidez:.1f})")
            else:
                print(f"[ERROR] No se pudo codificar el frame {nombre_archivo}")
            ultimo_disparo = ahora

            # Efecto visual de flash
            cv2.rectangle(display, (0, 0), (w, h), (255, 255, 255), 8)
            cv2.imshow("Captura de Dataset - ESP32-CAM", display)
            cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 50)
    print(f" CAPTURA COMPLETADA: {fotos_sesion} fotos nuevas guardadas.")
    print(f" Total en {nombre_carpeta}: {contador} fotos.")
    print("=" * 50)

if __name__ == "__main__":
    main()
