"""
Módulo 1: Adquisición de Datos y Técnica de Distribución Automática por Ventanas
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Docente: Ing. Rubén Nogales
Universidad Técnica de Ambato - FISEI - Software
"""

import os
import sys
import glob
import time
import shutil
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Importar funciones canónicas de PDI e IoT
from utils_pdi import (
    descomponer_rgb_a_double,
    calcular_cuadrado_envolvente,
    calcular_iou,
    imfill_holes,
    extraer_regionprops_basico,
    preprocesar_ventana_64x64,
    extraer_ventanas_negativas_esquinas,
    obtener_rejilla_ventanas_geometrica,
    analizar_morfologia_ventana_otsu,
    detectar_rostro_principal,
    proponer_ventanas_rostros_pdi,
    verificar_estructura_facial_pdi,
    extraer_descriptor_espacial_bloques,
    extraer_puntos_faciales_landmarks_pdi,
    aplicar_nms_cajas,
    calcular_nitidez_laplaciano,
    cv2_imread_unicode,
    cv2_imwrite_unicode,
)
from iot_esp32_telegram import cargar_config_iot, LectorCamaraThreaded

CARPETAS_PERSONAS_DEFECTO = {
    0: ("id_0_sebastian", "Sebastián Ortiz"),
    1: ("id_1_angel", "Ángel Ayuquina"),
    2: ("id_2_persona3", "Persona 3 (Opcional)"),
    3: ("id_3_persona4", "Persona 4 (Opcional)"),
    4: ("clase_0_no_persona", "Clase 0: Objetos Inanimados (Botella, Celular, Manos, Fondo)"),
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(BASE_DIR, "data")
DIR_FOTOS_CRUDAS = os.path.join(DIR_DATA, "fotos_crudas_celular")
DIR_DATASET_64 = os.path.join(DIR_DATA, "dataset_procesado_64x64")
DIR_CLASE_0 = os.path.join(DIR_DATASET_64, "clase_0_no_persona")
DIR_MODELOS = os.path.join(BASE_DIR, "modelos_entrenados")
DIR_FIGURAS = os.path.join(BASE_DIR, "figuras_y_reportes")


def asegurar_directorios():
    os.makedirs(DIR_CLASE_0, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    for id_p, (carpeta, _) in CARPETAS_PERSONAS_DEFECTO.items():
        os.makedirs(os.path.join(DIR_FOTOS_CRUDAS, carpeta), exist_ok=True)
        os.makedirs(os.path.join(DIR_DATASET_64, carpeta), exist_ok=True)

    # Limpieza automática rigurosa de datos sintéticos o archivos seed
    archivos_seed = glob.glob(os.path.join(DIR_CLASE_0, "objeto_real_seed_*.png"))
    for f in archivos_seed:
        try:
            os.remove(f)
        except Exception:
            pass
    if archivos_seed:
        print(f"[LIMPIEZA] Eliminados {len(archivos_seed)} archivos sintéticos/seed de clase_0_no_persona.")


def _archivar_fotos_sin_mascara_si_usuario_acepta(dir_carpeta: str, nombre_carpeta: str):
    """
    Detecta fotos antiguas SIN máscara elíptica (esquinas != 107) que degradan
    la coherencia de encuadre entre identidades y ofrece archivarlas de forma segura
    en data/backup_antiguas/ sin borrar nada permanentemente.
    """
    archivos = sorted(glob.glob(os.path.join(dir_carpeta, "*.png")))
    if not archivos:
        return
    sin_mascara = []
    for f in archivos:
        im = cv2_imread_unicode(f, cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (64, 64):
            continue
        esquinas = [int(im[0, 0]), int(im[0, 63]), int(im[63, 0]), int(im[63, 63])]
        if not all(abs(c - 107) <= 4 for c in esquinas):
            sin_mascara.append(f)

    if sin_mascara:
        print(f"\n[AVISO CALIDAD DATASET] Se detectaron {len(sin_mascara)} fotos antiguas SIN máscara elíptica en '{nombre_carpeta}'.")
        print("  Archivarlas evita confusiones entre personas y falsos positivos en objetos.")
        resp = input("  ¿Desea mover esas fotos antiguas a 'data/backup_antiguas/' ahora? [s/N]: ").strip().lower()
        if resp in ("s", "si", "sí", "y"):
            dir_bkp = os.path.join(DIR_DATA, "backup_antiguas", nombre_carpeta)
            os.makedirs(dir_bkp, exist_ok=True)
            for f in sin_mascara:
                try:
                    shutil.move(f, os.path.join(dir_bkp, os.path.basename(f)))
                except Exception:
                    pass
            print(f"  [OK] {len(sin_mascara)} fotos antiguas archivadas en: {dir_bkp}")


def opcion_captura_webcam():
    """
    Opción 1: Captura Interactiva con ESP32-CAM (IoT Wi-Fi) o Cámara Web Laptop.
    Incluye Auto-Recorte Anatómico PDI unificado con el detector multi-rostro en vivo,
    filtro de nitidez Laplaciana (para descartar fotos borrosas del OV2640) y vista previa 64x64.
    """
    asegurar_directorios()
    cfg_iot = cargar_config_iot()
    cfg_esp = cfg_iot["esp32"]

    print("\n" + "=" * 70)
    print("OPCIÓN 1: CAPTURA DE DATASET (ESP32-CAM IoT / CÁMARA WEB LAPTOP)")
    print("=" * 70)
    ip_predeterminada = cfg_esp.get("ip", "10.198.13.94")
    puerto_ctrl = int(cfg_esp.get("puerto_control", 80))
    puerto_str = int(cfg_esp.get("puerto_stream", 81))
    print("Seleccione la fuente de cámara:")
    print(f"  [1] Cámara IoT ESP32-CAM por Wi-Fi (IP en config_iot.json: {ip_predeterminada}:{puerto_str})")
    print("  [2] Cámara Web de la Laptop (índice 0)")
    opc_cam = input("Fuente [1=ESP32-CAM, 2=Laptop, Enter=2]: ").strip()

    usar_esp32 = (opc_cam == "1")
    if usar_esp32:
        ip_in = input(f"IP del ESP32-CAM [Enter para usar {ip_predeterminada}]: ").strip()
        ip_esp = ip_in if ip_in else ip_predeterminada
        # Usa la misma configuración de puertos de config_iot.json que 3_reconocimiento_en_vivo.py:
        #   - puerto_stream == 80 -> http://<ip>/
        #   - puerto_stream == 81 -> http://<ip>:81/stream (+ respaldo /capture en puerto_control)
        if puerto_str == 80:
            url_stream = f"http://{ip_esp}/"
            url_capture = None
        else:
            url_stream = f"http://{ip_esp}:{puerto_str}/stream"
            url_capture = f"http://{ip_esp}:{puerto_ctrl}/capture"
        fuente = url_stream
        prefijo_cam = "esp32"
        voltear_h = bool(cfg_esp.get("voltear_horizontal", False))
        rotar_180 = bool(cfg_esp.get("rotar_180", False))
        print(f"Conectando al ESP32-CAM en {url_stream} ...")
    else:
        fuente = 0
        url_capture = None
        prefijo_cam = "rostro"
        voltear_h = True
        rotar_180 = False

    print("\nSeleccione la persona / clase a capturar:")
    for id_p, (carpeta, nombre) in CARPETAS_PERSONAS_DEFECTO.items():
        print(f"  [{id_p}] {nombre} ({carpeta})")

    entrada_id = input("\nIngrese el ID [0 por defecto]: ").strip()
    id_sel = int(entrada_id) if entrada_id.isdigit() and int(entrada_id) in CARPETAS_PERSONAS_DEFECTO else 0
    carpeta_sel, nombre_sel = CARPETAS_PERSONAS_DEFECTO[id_sel]
    dir_destino_rostro = os.path.join(DIR_DATASET_64, carpeta_sel)

    if id_sel != 4:
        _archivar_fotos_sin_mascara_si_usuario_acepta(dir_destino_rostro, carpeta_sel)

    meta_nuevas = 350 if usar_esp32 else 250

    # Iniciar lector de video en hilo de ultra-baja latencia (Zero-Buffer)
    print(f"Abriendo stream de cámara ({'ESP32-CAM IoT' if usar_esp32 else 'Laptop'}): {fuente} ...")
    lector = LectorCamaraThreaded(fuente=fuente, url_capture_respaldo=url_capture, rotar_180=rotar_180)
    if not lector.iniciar():
        print(f"[ERROR] No se pudo conectar a la cámara ({fuente}). Verifique conexión Wi-Fi / red.")
        return

    contador_rostros = len(glob.glob(os.path.join(dir_destino_rostro, "*.png")))
    contador_fondo = len(glob.glob(os.path.join(DIR_CLASE_0, "*.png")))
    capturadas_sesion = 0

    # Cargar clasificador supervisado de Etapa 1 para garantizar que el auto-recorte PDI
    # encuadre exactamente el rostro humano y bloquee cuellos, capuchas o ropa cálida.
    mod_e1_acq = None
    pca_e1_acq = None
    try:
        p_pca = os.path.join(DIR_MODELOS, "etapa1_pca_humano.pkl")
        p_mod = os.path.join(DIR_MODELOS, "etapa1_mejor_modelo.pkl")
        if os.path.exists(p_pca) and os.path.exists(p_mod):
            pca_e1_acq = joblib.load(p_pca)
            mod_e1_acq = joblib.load(p_mod).get("info", None)
            if mod_e1_acq:
                print("[PDI] Validador biométrico de Etapa 1 cargado para auto-recorte anti-cuello.")
    except Exception:
        pass

    modo_auto = False
    # Para rostros: Auto-Recorte PDI activado por defecto (idéntico al detector multi-rostro en vivo)
    modo_auto_recorte = (id_sel != 4)
    ultimo_tiempo_auto = 0.0
    img_anterior_64 = None
    UMBRAL_NITIDEZ_MIN = 15.0  # Filtra frames movidos/borrosos del ESP32-CAM
    caja_suave_acq = None
    frames_sin_rostro_acq = 0
    factor_fijo = 0.46
    nombre_limpio = (
        nombre_sel.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        .replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
        .replace("ñ", "n").replace("Ñ", "N")
    )

    def _evaluar_svm_etapa1_acq(vec_c: np.ndarray):
        """Evalúa el mismo vector completo [PCA + MSE + PDI + Landmarks] que 3_reconocimiento_en_vivo.py."""
        if mod_e1_acq is None or pca_e1_acq is None:
            return 0.70, 0.0015
        x = vec_c.reshape(1, -1)
        z = pca_e1_acq.transform(x)
        x_hat = pca_e1_acq.inverse_transform(z)
        mse_v = float(np.mean((x - x_hat) ** 2))
        im = vec_c.reshape(64, 64)
        gy = np.abs(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3))
        gx = np.abs(cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3))
        peso = gy + np.maximum(0.0, 0.45 - im)
        cx_f = float(np.sum(np.arange(64) * np.sum(peso, axis=0)) / (np.sum(peso) + 1e-6))
        cx = int(np.clip(round(cx_f), 24, 40))
        w_sim = min(cx, 63 - cx)
        s_cx = 1.0 - float(np.mean(np.abs(im[:, cx - w_sim : cx] - np.fliplr(im[:, cx : cx + w_sim]))))
        s_cent = 1.0 - float(np.mean(np.abs(im[:, :32] - np.fliplr(im[:, 32:]))))
        sim = max(s_cent, s_cx)
        ratio_hv = float(np.mean(gy) / (np.mean(gx) + 1e-6))
        desc_b = extraer_descriptor_espacial_bloques(im)
        land_12, _ = extraer_puntos_faciales_landmarks_pdi(im)
        pdi_feats = np.concatenate([[sim, ratio_hv], desc_b, land_12]).reshape(1, -1)
        f = np.hstack([z, np.array([[mse_v]]), pdi_feats])
        sc_e1 = mod_e1_acq.get("scaler", None)
        if sc_e1 is not None:
            f = sc_e1.transform(f)
        mod_obj = mod_e1_acq["modelo"]
        if hasattr(mod_obj, "predict_proba"):
            prob_h = float(mod_obj.predict_proba(f)[0, 1])
        else:
            prob_h = 0.85 if int(mod_obj.predict(f)[0]) == 1 else 0.15
        return prob_h, mse_v

    print(f"\nCámara iniciada ({'ESP32-CAM IoT' if usar_esp32 else 'Laptop'}) para: {nombre_sel}")
    print(f"Meta sugerida en esta sesión: +{meta_nuevas} muestras nuevas (varía distancia: cerca, media, lejos y leves giros).")
    print("Controles:")
    print("  - [ESPACIO]: Captura manual inmediata")
    print("  - [A]: Alternar ráfaga automática (cada 0.22s con filtro anti-borroso y anti-duplicado)")
    print("  - [T]: Alternar encuadre: AUTO-SEGUIMIENTO PDI vs RECUADRO FIJO CENTRAL (estable)")
    print("  - [+] / [-]: Agrandar o reducir el recuadro en modo FIJO")
    print("  - [F]: Capturar directamente OBJETO / CHAQUETA / FONDO en Clase 0")
    print("  - [Q / ESC]: Terminar captura\n")

    while True:
        ret, frame = lector.leer()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        if voltear_h:
            frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        frame_vis = frame.copy()

        cuadrado_fijo, esquinas_fondo = obtener_rejilla_ventanas_geometrica(W, H, factor_central=factor_fijo)
        caja_activa = cuadrado_fijo
        rostro_pdi_valido = (id_sel == 4)

        if modo_auto_recorte and id_sel != 4:
            # Usar el MISMO generador multi-rostro PDI + validador completo SVM/MSE de Etapa 1
            propuestas = proponer_ventanas_rostros_pdi(frame, max_candidatos=16, incluir_rejilla_respaldo=False)
            cajas_ok_pdi = []
            scores_ok_pdi = []
            min_lado_rostro = max(105, int(min(W, H) * 0.22))
            for cand in propuestas:
                cx_c, cy_c, cl_c, _ = cand
                if cl_c < min_lado_rostro:
                    continue  # Ignorar objetos pequeños / lámparas del techo
                parche_c = frame[cy_c : cy_c + cl_c, cx_c : cx_c + cl_c]
                if parche_c.shape[0] < min_lado_rostro or parche_c.shape[1] < min_lado_rostro:
                    continue
                im_c, vec_c = preprocesar_ventana_64x64(parche_c)
                ok_pdi, met_pdi = verificar_estructura_facial_pdi(parche_c, im_c)
                if ok_pdi:
                    try:
                        prob_humano, mse_e1_c = _evaluar_svm_etapa1_acq(vec_c)
                    except Exception:
                        prob_humano, mse_e1_c = 0.0, 1.0

                    if prob_humano < 0.65 or mse_e1_c > 0.0036:
                        continue  # Rechazar lámparas, techo, cuellos o fondos iluminados

                    dist_centro = np.hypot((cx_c + cl_c * 0.5) - W * 0.5, (cy_c + cl_c * 0.45) - H * 0.42) / float(W)
                    score = 2.5 * prob_humano + float(met_pdi.get("sim_bilateral", 0.0)) - 0.8 * dist_centro
                    cajas_ok_pdi.append(cand)
                    scores_ok_pdi.append(score)
            idx_nms = aplicar_nms_cajas(cajas_ok_pdi, scores_ok_pdi, umbral_iou=0.25, umbral_contencion=0.45)
            if idx_nms:
                det_cand = cajas_ok_pdi[idx_nms[0]]
                if caja_suave_acq is None:
                    caja_suave_acq = [float(v) for v in det_cand]
                else:
                    # Suavizado exponencial EMA anti-parpadeo y anti-saltos bruscos
                    alpha_s = 0.28
                    for k_i in range(4):
                        caja_suave_acq[k_i] = alpha_s * float(det_cand[k_i]) + (1.0 - alpha_s) * caja_suave_acq[k_i]
                ql_s = max(min_lado_rostro, min(int(round(caja_suave_acq[2])), min(W, H)))
                qx_s = int(np.clip(round(caja_suave_acq[0]), 0, max(0, W - ql_s)))
                qy_s = int(np.clip(round(caja_suave_acq[1]), 0, max(0, H - ql_s)))
                caja_activa = [qx_s, qy_s, ql_s, ql_s]
                rostro_pdi_valido = True
                frames_sin_rostro_acq = 0
            else:
                frames_sin_rostro_acq += 1
                if caja_suave_acq is not None and frames_sin_rostro_acq <= 6:
                    # Persistencia breve de 6 frames para evitar parpadeo momentáneo
                    ql_s = max(min_lado_rostro, min(int(round(caja_suave_acq[2])), min(W, H)))
                    qx_s = int(np.clip(round(caja_suave_acq[0]), 0, max(0, W - ql_s)))
                    qy_s = int(np.clip(round(caja_suave_acq[1]), 0, max(0, H - ql_s)))
                    caja_activa = [qx_s, qy_s, ql_s, ql_s]
                    rostro_pdi_valido = True
                else:
                    caja_suave_acq = None
                    caja_activa = cuadrado_fijo
                    rostro_pdi_valido = False
        else:
            caja_suave_acq = None
            rostro_pdi_valido = True

        qx, qy, qlado, _ = caja_activa
        parche_activo = frame[qy : qy + qlado, qx : qx + qlado]
        nitidez = calcular_nitidez_laplaciano(parche_activo)
        es_nitido = nitidez >= UMBRAL_NITIDEZ_MIN

        # Vista previa exacta de la matriz 64x64 preprocesada (esquina superior derecha)
        img_preview_64, _ = preprocesar_ventana_64x64(parche_activo)
        prev_u8 = np.clip(img_preview_64 * 255.0, 0, 255).astype(np.uint8)
        prev_bgr = cv2.cvtColor(cv2.resize(prev_u8, (96, 96), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        if W >= 120 and H >= 120:
            frame_vis[8:104, W - 104 : W - 8] = prev_bgr
            cv2.rectangle(frame_vis, (W - 105, 7), (W - 7, 105), (0, 255, 200), 1)
            cv2.putText(frame_vis, "Entrada 64x64", (W - 104, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 200), 1)

        color_caja = (0, 165, 255) if id_sel == 4 else ((0, 255, 0) if (rostro_pdi_valido and es_nitido) else (0, 200, 255))
        cv2.rectangle(frame_vis, (qx, qy), (qx + qlado, qy + qlado), color_caja, 2)
        cv2.ellipse(
            frame_vis,
            (qx + qlado // 2, qy + qlado // 2),
            (int(qlado * 0.35), int(qlado * 0.46)),
            0,
            0,
            360,
            (0, 255, 200) if id_sel != 4 else (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

        modo_txt = "AUTO-PDI" if (modo_auto_recorte and id_sel != 4) else "FIJO"
        estado_nit = f"Nitidez:{nitidez:.0f} ({'OK' if es_nitido else 'BORROSA'})"
        etiqueta_caja = (
            f"OBJETO/CLASE 0 [{qlado}x{qlado}]"
            if id_sel == 4
            else (
                f"{modo_txt}: {qlado}x{qlado} | {estado_nit}"
                if rostro_pdi_valido
                else "BUSCANDO ROSTRO... (centre su cara o pulse [T]=FIJO)"
            )
        )
        cv2.putText(frame_vis, etiqueta_caja, (qx, max(20, qy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color_caja, 1, cv2.LINE_AA)

        # Filtrar esquinas de fondo con IoU == 0 respecto a la caja activa
        esquinas_validas = [e for e in esquinas_fondo if calcular_iou(e, caja_activa) == 0.0]
        for vx, vy, vw, vh in esquinas_validas:
            cv2.rectangle(frame_vis, (vx, vy), (vx + vw, vy + vh), (255, 120, 0), 1)

        # Panel inferior
        cv2.rectangle(frame_vis, (8, H - 72), (W - 8, H - 8), (20, 20, 20), -1)
        cv2.putText(
            frame_vis,
            f"Cam: {'ESP32-CAM' if usar_esp32 else 'Laptop'} | {nombre_limpio[:18]} | Sesion: +{capturadas_sesion}/{meta_nuevas} (Total:{contador_rostros}) | C0:{contador_fondo}",
            (14, H - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
        )
        estado_auto = "AUTO ON" if modo_auto else "AUTO OFF [A]"
        cv2.putText(
            frame_vis,
            f"{estado_auto} | [ESP]=Foto | [T]=Modo({modo_txt}) | [+/-]=Tam | [Q]=Salir",
            (14, H - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (0, 220, 255),
            1,
        )

        cv2.imshow("Adquisicion PDI (ESP32-CAM / Webcam) - FISEI UTA", frame_vis)
        tecla = cv2.waitKey(1) & 0xFF

        capturar_ahora = False
        if tecla in [ord("q"), ord("Q"), 27]:
            print("\nCaptura finalizada por el usuario.")
            break
        elif tecla == ord(" "):
            capturar_ahora = True
        elif tecla in [ord("a"), ord("A")]:
            modo_auto = not modo_auto
            print(f"\n[MODO AUTO] {'Activado' if modo_auto else 'Desactivado'}")
        elif tecla in [ord("t"), ord("T")] and id_sel != 4:
            modo_auto_recorte = not modo_auto_recorte
            caja_suave_acq = None
            print(f"\n[ENCUADRE] Cambiado a: {'AUTO-SEGUIMIENTO PDI' if modo_auto_recorte else 'RECUADRO FIJO CENTRAL'}")
        elif tecla in [ord("+"), ord("=")] and (not modo_auto_recorte or id_sel == 4):
            factor_fijo = min(0.66, factor_fijo + 0.04)
        elif tecla in [ord("-"), ord("_")] and (not modo_auto_recorte or id_sel == 4):
            factor_fijo = max(0.26, factor_fijo - 0.04)
        elif tecla in [ord("f"), ord("F")]:
            if parche_activo.shape[0] >= 32 and parche_activo.shape[1] >= 32:
                img_f_64, _ = preprocesar_ventana_64x64(parche_activo)
                img_save_f = np.clip(img_f_64 * 255.0, 0, 255).astype(np.uint8)
                ruta_f = os.path.join(DIR_CLASE_0, f"objeto_{prefijo_cam}_{contador_fondo:05d}.png")
                cv2_imwrite_unicode(ruta_f, img_save_f)
                contador_fondo += 1
                print(f"\r-> [TECLA F] Objeto/Fondo guardado ({prefijo_cam}): #{contador_fondo:04d}", end="")

        if modo_auto and (time.time() - ultimo_tiempo_auto >= 0.22):
            capturar_ahora = True
            ultimo_tiempo_auto = time.time()

        if capturar_ahora and (parche_activo.shape[0] >= 32 and parche_activo.shape[1] >= 32):
            if id_sel == 4:
                img_obj_64, _ = preprocesar_ventana_64x64(parche_activo)
                img_save_obj = np.clip(img_obj_64 * 255.0, 0, 255).astype(np.uint8)
                ruta_obj = os.path.join(DIR_CLASE_0, f"objeto_{prefijo_cam}_{contador_fondo:05d}.png")
                cv2_imwrite_unicode(ruta_obj, img_save_obj)
                contador_fondo += 1
                capturadas_sesion += 1
                print(f"\r-> Guardada muestra Clase 0 ({prefijo_cam}): #{contador_fondo:04d}", end="")
            else:
                if modo_auto and ((not rostro_pdi_valido) or (not es_nitido)):
                    continue  # Saltar frame sin rostro claro o movido

                img_rostro_64 = img_preview_64
                if img_anterior_64 is not None and modo_auto:
                    dif_img = float(np.mean(np.abs(img_rostro_64 - img_anterior_64)))
                    if dif_img < 0.018:
                        continue  # Esperar ligera variación de pose/posición

                img_anterior_64 = img_rostro_64.copy()
                img_save_rostro = np.clip(img_rostro_64 * 255.0, 0, 255).astype(np.uint8)
                ts_ms = int(time.time() * 1000) % 100000
                ruta_rostro = os.path.join(dir_destino_rostro, f"{prefijo_cam}_{contador_rostros:04d}_{ts_ms}.png")
                cv2_imwrite_unicode(ruta_rostro, img_save_rostro)
                contador_rostros += 1
                capturadas_sesion += 1

                # Guardar 1 ventana de fondo (IoU == 0) cada 2 fotos faciales para nutrir Clase 0 del mismo sensor
                if esquinas_validas and (capturadas_sesion % 2 == 0):
                    vx, vy, vw, vh = esquinas_validas[contador_rostros % len(esquinas_validas)]
                    parche_esq = frame[vy : vy + vh, vx : vx + vw]
                    if parche_esq.shape[0] >= 16 and parche_esq.shape[1] >= 16:
                        img_fondo_64, _ = preprocesar_ventana_64x64(parche_esq)
                        img_save_fondo = np.clip(img_fondo_64 * 255.0, 0, 255).astype(np.uint8)
                        ruta_fondo = os.path.join(DIR_CLASE_0, f"fondo_{prefijo_cam}_{contador_fondo:05d}.png")
                        cv2_imwrite_unicode(ruta_fondo, img_save_fondo)
                        contador_fondo += 1

                print(
                    f"\r-> Guardada +{capturadas_sesion:03d} ({prefijo_cam}, nitidez={nitidez:.0f}) | Total {carpeta_sel}: {contador_rostros} | C0: {contador_fondo}",
                    end="",
                )

            if id_sel != 4 and capturadas_sesion >= meta_nuevas and modo_auto:
                print(f"\n[OK] Se alcanzó la meta de +{meta_nuevas} muestras nuevas en esta sesión.")
                modo_auto = False

    if lector is not None:
        lector.liberar()
    cv2.destroyAllWindows()
    print(f"\nResumen: +{capturadas_sesion} nuevas (total {contador_rostros} en '{carpeta_sel}', {contador_fondo} en 'clase_0_no_persona').\n")


def opcion_procesar_fotos_cruda_lotes():
    """
    Opción 2: Procesa por lotes las fotografías crudas tomadas con el ESP32-CAM o celular
    guardadas en data/id_0_sebastian, data/id_1_angel, etc.
    Aplica el algoritmo del cuadrado envolvente, PDI de cátedra (mediana, Gauss, Otsu, ecualización),
    extrae el rostro normalizado a 64x64 y 3 ventanas de fondo para clase_0_no_persona.
    """
    asegurar_directorios()
    print("\n" + "=" * 70)
    print("OPCIÓN 2: INGESTA Y PROCESAMIENTO PDI POR LOTES (ESP32-CAM / CRUDAS)")
    print("=" * 70)

    total_rostros_proc = 0
    total_fondos_proc = len(glob.glob(os.path.join(DIR_CLASE_0, "*.png")))

    for id_p, (carpeta, nombre) in CARPETAS_PERSONAS_DEFECTO.items():
        if id_p == 4:
            continue  # Clase 0 es solo de fondo

        # Buscar imágenes tanto en data/id_X como en data/fotos_crudas_celular/id_X
        posibles_dirs = [
            os.path.join(DIR_DATA, carpeta),
            os.path.join(DIR_FOTOS_CRUDAS, carpeta),
        ]

        archivos = []
        dir_cruda_encontrada = ""
        patrones = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for d in posibles_dirs:
            if os.path.exists(d):
                for pat in patrones:
                    encontrados = glob.glob(os.path.join(d, pat))
                    if encontrados:
                        archivos.extend(encontrados)
                        dir_cruda_encontrada = d

        dir_salida = os.path.join(DIR_DATASET_64, carpeta)
        os.makedirs(dir_salida, exist_ok=True)

        print(f"\nRevisando {carpeta} ({nombre}): {len(archivos)} imágenes crudas encontradas.")
        if not archivos:
            print(f"  [AVISO] No hay imágenes crudas para {nombre}.")
            continue

        procesadas_sujeto = 0
        for ruta_img in archivos:
            frame = cv2_imread_unicode(ruta_img)
            if frame is None:
                continue

            H, W = frame.shape[:2]
            # Localización Anatómica Facial Canónica
            cuadrado_rostro = detectar_rostro_principal(frame, factor_central=0.48)
            ventanas_fondo = extraer_ventanas_negativas_esquinas(frame, cuadrado_rostro, num_ventanas=3)

            qx, qy, qlado, _ = cuadrado_rostro
            parche_rostro = frame[qy : qy + qlado, qx : qx + qlado]

            if parche_rostro.shape[0] >= 32 and parche_rostro.shape[1] >= 32:
                img_rostro_64, _ = preprocesar_ventana_64x64(parche_rostro)
                img_save = np.clip(img_rostro_64 * 255.0, 0, 255).astype(np.uint8)
                nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
                ruta_guardar = os.path.join(dir_salida, f"proc_{nombre_base}.png")
                cv2_imwrite_unicode(ruta_guardar, img_save)
                procesadas_sujeto += 1
                total_rostros_proc += 1

                # Ventanas negativas de fondo calculadas por extraer_ventanas_negativas_esquinas
                for img_f_64, _, _ in ventanas_fondo:
                    img_f_save = np.clip(img_f_64 * 255.0, 0, 255).astype(np.uint8)
                    ruta_f = os.path.join(DIR_CLASE_0, f"fondo_auto_{total_fondos_proc:05d}.png")
                    cv2_imwrite_unicode(ruta_f, img_f_save)
                    total_fondos_proc += 1

        print(f"  -> {procesadas_sujeto} rostros preprocesados PDI a 64x64 guardados en '{dir_salida}'.")

    print(f"\n[OK] Proceso por lotes finalizado. Total rostros: {total_rostros_proc}. Total fondos: {total_fondos_proc}\n")


def opcion_generar_figura1_diagnostico():
    """
    Opción 3: Genera la lámina diagnóstica oficial de PDI (Figura 1) con formato 2x3:
        Panel 1: Imagen RGB original con cuadrado envolvente y ventanas de esquina.
        Panel 2: Plano en escala de grises en precisión continua double [0.0, 1.0].
        Panel 3: Mitigación de ruido de cátedra (Filtro Mediana 3x3 + Suavizado Gaussiano).
        Panel 4: Compensación de iluminación mediante Ecualización de Histograma.
        Panel 5: Binarización de Otsu + sellado morfológico imfill + Centroide continuo (regionprops).
        Panel 6: Entrada estándar de la red/clasificador (64x64 en float64).
    """
    asegurar_directorios()
    print("\n" + "=" * 70)
    print("OPCIÓN 3: GENERAR FIGURA 1 DIAGNÓSTICA DEL PIPELINE DE PDI COMPLETO")
    print("=" * 70)

    # Buscar una imagen candidata en fotos crudas o capturar de cámara o generar prototipo
    archivos_muestra = glob.glob(os.path.join(DIR_FOTOS_CRUDAS, "*", "*.jpg")) + glob.glob(
        os.path.join(DIR_FOTOS_CRUDAS, "*", "*.png")
    )

    frame_muestra = None
    if archivos_muestra:
        for f in archivos_muestra:
            tmp = cv2_imread_unicode(f)
            if tmp is not None:
                frame_muestra = tmp
                print(f"Usando muestra de fotografía: {f}")
                break

    if frame_muestra is None:
        # Intentar captura instantánea de cámara web
        print("Intentando capturar muestra instantánea desde la cámara web...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            for _ in range(15):  # descartar primeros fotogramas para balance de exposición
                ret, frame_cam = cap.read()
            cap.release()
            if ret and frame_cam is not None:
                frame_muestra = cv2.flip(frame_cam, 1)

    if frame_muestra is None:
        # Si no hay cámara ni fotos, generar un rostro canónico sintético bien estructurado
        print("[AVISO] No se detectó cámara web ni fotografías crudas.")
        print("Generando fotograma canónico de calibración con geometría facial de cátedra...")
        frame_muestra = np.ones((480, 640, 3), dtype=np.uint8) * 220
        # Dibujar fondo con texturas
        cv2.rectangle(frame_muestra, (0, 0), (200, 480), (190, 180, 170), -1)
        cv2.rectangle(frame_muestra, (440, 0), (640, 480), (200, 210, 205), -1)
        # Rostro elíptico canónico
        centro_cara = (320, 240)
        cv2.ellipse(frame_muestra, centro_cara, (110, 150), 0, 0, 360, (180, 200, 235), -1)
        # Ojos
        cv2.circle(frame_muestra, (280, 210), 18, (70, 60, 50), -1)
        cv2.circle(frame_muestra, (360, 210), 18, (70, 60, 50), -1)
        # Pupilas
        cv2.circle(frame_muestra, (280, 210), 8, (20, 20, 20), -1)
        cv2.circle(frame_muestra, (360, 210), 8, (20, 20, 20), -1)
        # Nariz y Boca
        cv2.line(frame_muestra, (320, 220), (320, 260), (120, 130, 160), 3)
        cv2.ellipse(frame_muestra, (320, 300), (38, 16), 0, 0, 180, (70, 80, 180), -1)
        # Agregar ruido impulsivo leve simulado
        ruido = np.random.randint(0, 100, (480, 640), dtype=np.uint8)
        frame_muestra[ruido < 3] = 255
        frame_muestra[ruido > 97] = 0

    H, W = frame_muestra.shape[:2]
    cuadrado, esquinas_fondo = obtener_rejilla_ventanas_geometrica(W, H, factor_central=0.55)
    qx, qy, qlado, _ = cuadrado
    parche_rgb = cv2.cvtColor(frame_muestra[qy : qy + qlado, qx : qx + qlado], cv2.COLOR_BGR2RGB)

    # 1. Panel 1: RGB original con cuadrado simétrico central y esquinas de fondo
    panel1_rgb = cv2.cvtColor(frame_muestra.copy(), cv2.COLOR_BGR2RGB)
    cv2.rectangle(panel1_rgb, (qx, qy), (qx + qlado, qy + qlado), (0, 255, 0), 3)
    cv2.circle(panel1_rgb, (qx + qlado // 2, qy + qlado // 2), 5, (0, 255, 0), -1)
    for vx, vy, vw, vh in esquinas_fondo:
        cv2.rectangle(panel1_rgb, (vx, vy), (vx + vw, vy + vh), (255, 140, 0), 2)

    # 2. Panel 2: Gris double [0.0, 1.0]
    pdi_dict = descomponer_rgb_a_double(cv2.cvtColor(parche_rgb, cv2.COLOR_RGB2BGR))
    gris_double = pdi_dict["gris_double"]

    # 3. Panel 3: Mitigación de ruido (Mediana 3x3 + Gaussiano)
    gris_uint8 = np.clip(gris_double * 255.0, 0, 255).astype(np.uint8)
    mediana_uint8 = cv2.medianBlur(gris_uint8, 3)
    filtrada_gauss = cv2.GaussianBlur(mediana_uint8, (3, 3), 1.0)
    filtrada_double = filtrada_gauss.astype(np.float64) / 255.0

    # 4. Panel 4: Ecualización de histograma
    ecualizada_uint8 = cv2.equalizeHist(filtrada_gauss)
    ecualizada_double = ecualizada_uint8.astype(np.float64) / 255.0

    # 5. Panel 5: Otsu + imfill + regionprops
    _, mascara_otsu = cv2.threshold(ecualizada_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara_rellena = imfill_holes(mascara_otsu)
    rprops = extraer_regionprops_basico(mascara_rellena)

    # 6. Panel 6: Entrada final 64x64
    final_64_double, _ = preprocesar_ventana_64x64(parche_rgb)

    # Construir lámina en Matplotlib
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Figura 1: Pipeline Canónico de Preprocesamiento Digital de Imágenes (PDI de Cátedra)\n"
        "Sistema de Reconocimiento Facial Inteligente - Ortiz & Ayuquina (FISEI - UTA)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Subplot 1
    axes[0, 0].imshow(panel1_rgb)
    axes[0, 0].set_title("(1) Fotograma RGB Original\nCuadrado Simétrico max(w,h) y Ventanas IoU=0", fontsize=10)
    axes[0, 0].axis("off")

    # Subplot 2
    im2 = axes[0, 1].imshow(gris_double, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(
        "(2) Región Cuadrada en Double [0.0, 1.0]\n"
        r"$I_{gris} = 0.299 R_d + 0.587 G_d + 0.114 B_d$",
        fontsize=10,
    )
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Subplot 3
    im3 = axes[0, 2].imshow(filtrada_double, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(
        "(3) Mitigación de Ruido (Cátedra)\nFiltro Mediana 3x3 + Suavizado Gaussiano (σ=1.0)",
        fontsize=10,
    )
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Subplot 4
    im4 = axes[1, 0].imshow(ecualizada_double, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(
        "(4) Ecualización del Histograma\nCompensación de Sombras y Contraluz",
        fontsize=10,
    )
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Subplot 5
    axes[1, 1].imshow(mascara_rellena, cmap="gray")
    cx, cy = rprops["centroide"]
    axes[1, 1].plot(cx, cy, "ro", markersize=7, label=f"Centroide ({cx:.1f}, {cy:.1f})")
    freeman_txt = "".join(str(c) for c in rprops["freeman_sample"][:12])
    axes[1, 1].set_title(
        f"(5) Binarización Otsu + imfill + regionprops\n"
        f"Área={rprops['area']:.0f} px, Perím={rprops['perimetro']:.1f} px\n"
        f"Freeman-8: {freeman_txt}...",
        fontsize=10,
    )
    axes[1, 1].legend(loc="lower right", fontsize=8)

    # Subplot 6
    im6 = axes[1, 2].imshow(final_64_double, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title(
        "(6) Entrada Reducida por Mediana a 64x64\nVector x ∈ R^4096 (Sin desenfoque de orillas)",
        fontsize=10,
    )
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ruta_fig1 = os.path.join(DIR_FIGURAS, "figura1_pipeline_pdi_completo.png")
    plt.savefig(ruta_fig1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Lámina diagnóstica guardada con éxito en:\n     {ruta_fig1}\n")


def menu_principal():
    asegurar_directorios()
    while True:
        print("\n" + "=" * 70)
        print("SISTEMA DE ADQUISICIÓN Y PDI - RECONOCIMIENTO FACIAL IA (FISEI UTA)")
        print("Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)")
        print("=" * 70)
        print("  1. Captura interactiva con cámara web (Distribución por Ventanas en vivo)")
        print("  2. Procesar fotografías crudas por lotes (ESP32-CAM / data/id_X/)")
        print("  3. Generar Figura 1 diagnóstica de PDI (figura1_pipeline_pdi_completo.png)")
        print("  4. Salir")
        print("=" * 70)

        opc = input("Seleccione una opción [1-4]: ").strip()
        if opc == "1":
            opcion_captura_webcam()
        elif opc == "2":
            opcion_procesar_fotos_cruda_lotes()
        elif opc == "3":
            opcion_generar_figura1_diagnostico()
        elif opc == "4":
            print("\nSaliendo del módulo de adquisición. ¡Hasta luego!")
            break
        else:
            print("[AVISO] Opción no válida. Ingrese un número entre 1 y 4.")


if __name__ == "__main__":
    menu_principal()
