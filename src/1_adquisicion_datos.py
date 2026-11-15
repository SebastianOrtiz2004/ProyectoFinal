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
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importar funciones canónicas de PDI
from utils_pdi import (
    descomponer_rgb_a_double,
    calcular_cuadrado_envolvente,
    imfill_holes,
    extraer_regionprops_basico,
    preprocesar_ventana_64x64,
    extraer_ventanas_negativas_esquinas,
    obtener_rejilla_ventanas_geometrica,
    analizar_morfologia_ventana_otsu,
    detectar_rostro_principal,
    cv2_imread_unicode,
    cv2_imwrite_unicode,
)

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


def opcion_captura_webcam():
    """
    Opción 1: Captura Interactiva con Cámara Web con Distribución por Ventanas en Vivo.
    Guarda muestras de rostro (Clase 1 / id_X) y ventanas periféricas (Clase 0: No Persona).
    """
    asegurar_directorios()
    print("\n" + "=" * 70)
    print("OPCIÓN 1: CAPTURA INTERACTIVA CON CÁMARA WEB (DISTRIBUCIÓN POR VENTANAS)")
    print("=" * 70)
    print("Seleccione la persona a capturar:")
    for id_p, (carpeta, nombre) in CARPETAS_PERSONAS_DEFECTO.items():
        print(f"  [{id_p}] {nombre} ({carpeta})")

    entrada_id = input("\nIngrese el ID de la persona [0 por defecto]: ").strip()
    id_sel = int(entrada_id) if entrada_id.isdigit() and int(entrada_id) in CARPETAS_PERSONAS_DEFECTO else 0
    carpeta_sel, nombre_sel = CARPETAS_PERSONAS_DEFECTO[id_sel]
    dir_destino_rostro = os.path.join(DIR_DATASET_64, carpeta_sel)

    meta_muestras = 60
    print(f"\nIniciando cámara para: {nombre_sel}")
    print(f"Meta: {meta_muestras} muestras faciales.")
    print("Controles:")
    print("  - [ESPACIO]: Captura manual de muestra (rostro + 1 ventana de fondo)")
    print("  - [A]: Alternar modo de captura continua automática (cada 0.45s con filtro de variación)")
    print("  - [F]: Capturar directamente OBJETO / FONDO en la ventana central")
    print("  - [Q / ESC]: Terminar captura y volver al menú principal\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara web (índice 0). Verifique su conexión.")
        return

    contador_rostros = len(glob.glob(os.path.join(dir_destino_rostro, "*.png")))
    contador_fondo = len(glob.glob(os.path.join(DIR_CLASE_0, "*.png")))
    modo_auto = False
    ultimo_tiempo_auto = 0.0
    img_anterior_64 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Error al leer fotograma de la cámara.")
            break

        # Espejo horizontal para visualización natural del usuario
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        frame_vis = frame.copy()

        # REJILLA MATEMÁTICA PURA (Sin clasificadores preentrenados)
        # Ventana Central Anatómica (factor_central=0.38, ~180x180 px en 640x480)
        cuadrado_rostro, esquinas_fondo = obtener_rejilla_ventanas_geometrica(W, H, factor_central=0.38)
        qx, qy, qlado, _ = cuadrado_rostro

        # Análisis morfológico puro con Binarización de Otsu + imfill + regionprops
        parche_central = frame[qy : qy + qlado, qx : qx + qlado]
        info_morf = analizar_morfologia_ventana_otsu(parche_central)

        # Centroide geométrico continuo y caja ajustada dentro de la ventana
        cx_local, cy_local = info_morf["centroide_local"]
        cx_global = int(qx + cx_local)
        cy_global = int(qy + cy_local)

        # Dibujar ventana central simétrica (Verde para rostro, Naranja para Objetos)
        color_caja = (0, 165, 255) if id_sel == 4 else (0, 255, 0)
        cv2.rectangle(frame_vis, (qx, qy), (qx + qlado, qy + qlado), color_caja, 2)
        cv2.circle(frame_vis, (cx_global, cy_global), 4, color_caja, -1)

        # Elipse anatómica guía punteada en el centro de la caja
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

        # Guías visuales de alineación de cátedra
        etiqueta_caja = (
            f"OBJETO / NO-PERSONA: Botellas, celulares, manos [{qlado}x{qlado}]"
            if id_sel == 4
            else f"Ventana Facial: {nombre_sel} [{qlado}x{qlado}]"
        )
        cv2.putText(
            frame_vis,
            etiqueta_caja,
            (qx, max(20, qy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color_caja,
            2,
        )
        freeman_preview = "".join(str(c) for c in info_morf["freeman_sample"][:8])
        cv2.putText(
            frame_vis,
            f"Centroide: ({cx_global}, {cy_global}) | Otsu Area: {info_morf['area']:.0f}px | Freeman8: {freeman_preview}",
            (qx, qy + qlado + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 230, 255),
            1,
        )

        # Dibujar las 4 ventanas periféricas de fondo (Clase 0)
        for vx, vy, vw, vh in esquinas_fondo:
            cv2.rectangle(frame_vis, (vx, vy), (vx + vw, vy + vh), (255, 120, 0), 1)
            cv2.putText(
                frame_vis,
                "Fondo",
                (vx + 4, vy + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (255, 140, 0),
                1,
            )

        # Panel de estado en pantalla
        cv2.rectangle(frame_vis, (10, H - 75), (W - 10, H - 10), (20, 20, 20), -1)
        cv2.putText(
            frame_vis,
            f"Persona: {nombre_sel} | Rostros: {contador_rostros}/{meta_muestras} | Fondo: {contador_fondo}",
            (20, H - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        estado_auto = "ACTIVADO (0.45s)" if modo_auto else "DESACTIVADO [Presione 'A']"
        cv2.putText(
            frame_vis,
            f"Modo Auto: {estado_auto} | [ESPACIO]=Capturar | [A]=Auto | [F]=Objeto | [Q]=Salir",
            (20, H - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
        )

        cv2.imshow("Adquisicion Hibrida PDI - Distribucion por Ventanas", frame_vis)
        tecla = cv2.waitKey(1) & 0xFF

        capturar_ahora = False
        if tecla in [ord("q"), ord("Q"), 27]:
            print("\nCaptura finalizada por el usuario.")
            break
        elif tecla == ord(" "):
            capturar_ahora = True
        elif tecla in [ord("a"), ord("A")]:
            modo_auto = not modo_auto
            print(f"[MODO AUTO] {'Activado' if modo_auto else 'Desactivado'}")
        elif tecla in [ord("f"), ord("F")]:
            # Captura inmediata de objeto/fondo central
            if parche_central.shape[0] >= 32 and parche_central.shape[1] >= 32:
                img_f_64, _ = preprocesar_ventana_64x64(parche_central)
                img_save_f = np.clip(img_f_64 * 255.0, 0, 255).astype(np.uint8)
                ruta_f = os.path.join(DIR_CLASE_0, f"objeto_manual_{contador_fondo:05d}.png")
                cv2_imwrite_unicode(ruta_f, img_save_f)
                contador_fondo += 1
                print(f"\r-> [TECLA F] Objeto/Fondo manual guardado: #{contador_fondo:04d}", end="")

        if modo_auto and (time.time() - ultimo_tiempo_auto >= 0.45):
            capturar_ahora = True
            ultimo_tiempo_auto = time.time()

        if capturar_ahora and (parche_central.shape[0] >= 32 and parche_central.shape[1] >= 32):
            if id_sel == 4:
                # Captura de Objeto / Fondo para enriquecer Clase 0
                img_obj_64, _ = preprocesar_ventana_64x64(parche_central)
                img_save_obj = np.clip(img_obj_64 * 255.0, 0, 255).astype(np.uint8)
                ruta_obj = os.path.join(DIR_CLASE_0, f"objeto_central_{contador_fondo:05d}.png")
                cv2_imwrite_unicode(ruta_obj, img_save_obj)
                contador_fondo += 1
                print(
                    f"\r-> Guardada muestra de OBJETOS/FONDO Clase 0: #{contador_fondo:04d}",
                    end="",
                )
            else:
                # Preprocesar rostro central con PDI completo de cátedra
                img_rostro_64, _ = preprocesar_ventana_64x64(parche_central)

                # Control anti-leakage: verificar que haya movimiento respecto al fotograma anterior
                if img_anterior_64 is not None and modo_auto:
                    dif_img = float(np.mean(np.abs(img_rostro_64 - img_anterior_64)))
                    if dif_img < 0.020:
                        continue  # Fotograma estático/idéntico; esperar a que el usuario se mueva

                img_anterior_64 = img_rostro_64.copy()
                img_save_rostro = np.clip(img_rostro_64 * 255.0, 0, 255).astype(np.uint8)
                ruta_rostro = os.path.join(
                    dir_destino_rostro, f"rostro_{contador_rostros:04d}.png"
                )
                cv2_imwrite_unicode(ruta_rostro, img_save_rostro)
                contador_rostros += 1

                # Guardar como máximo 1 ventana de fondo por foto de rostro (alternando esquinas)
                if len(esquinas_fondo) > 0:
                    vx, vy, vw, vh = esquinas_fondo[contador_rostros % len(esquinas_fondo)]
                    parche_esq = frame[vy : vy + vh, vx : vx + vw]
                    if parche_esq.shape[0] >= 16 and parche_esq.shape[1] >= 16:
                        img_fondo_64, _ = preprocesar_ventana_64x64(parche_esq)
                        img_save_fondo = np.clip(img_fondo_64 * 255.0, 0, 255).astype(np.uint8)
                        ruta_fondo = os.path.join(
                            DIR_CLASE_0, f"fondo_auto_{contador_fondo:05d}.png"
                        )
                        cv2_imwrite_unicode(ruta_fondo, img_save_fondo)
                        contador_fondo += 1

                print(
                    f"\r-> Guardada muestra facial #{contador_rostros:03d} | Ventanas fondo Clase 0: #{contador_fondo:04d}",
                    end="",
                )

            if id_sel != 4 and contador_rostros >= meta_muestras and modo_auto:
                print(f"\n[OK] Se ha alcanzado la meta de {meta_muestras} muestras.")
                modo_auto = False
                modo_auto = False

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nResumen: {contador_rostros} rostros en '{carpeta_sel}' y {contador_fondo} fondos en 'clase_0_no_persona'.\n")


def opcion_procesar_fotos_celular():
    """
    Opción 2: Procesa fotografías tomadas con el teléfono celular colocadas en data/fotos_crudas_celular/
    Aplica el algoritmo del cuadrado envolvente, extrae el rostro a 64x64 y 3 ventanas de fondo.
    """
    asegurar_directorios()
    print("\n" + "=" * 70)
    print("OPCIÓN 2: INGESTA Y PROCESAMIENTO POR LOTES DE FOTOS DEL CELULAR")
    print("=" * 70)

    total_rostros_proc = 0
    total_fondos_proc = len(glob.glob(os.path.join(DIR_CLASE_0, "*.png")))

    for id_p, (carpeta, nombre) in CARPETAS_PERSONAS_DEFECTO.items():
        dir_cruda = os.path.join(DIR_FOTOS_CRUDAS, carpeta)
        dir_salida = os.path.join(DIR_DATASET_64, carpeta)

        patrones = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        archivos = []
        for pat in patrones:
            archivos.extend(glob.glob(os.path.join(dir_cruda, pat)))

        print(f"\nRevisando {carpeta} ({nombre}): {len(archivos)} imágenes crudas encontradas.")
        if not archivos:
            print(f"  [AVISO] No hay imágenes en '{dir_cruda}'. Copie aquí fotos tomadas con su celular.")
            continue

        procesadas_sujeto = 0
        for ruta_img in archivos:
            frame = cv2_imread_unicode(ruta_img)
            if frame is None:
                continue

            H, W = frame.shape[:2]
            # En fotos del celular: Usar rejilla geométrica central o recorte proporcional
            cuadrado_rostro, esquinas_fondo = obtener_rejilla_ventanas_geometrica(W, H, factor_central=0.55)
            qx, qy, qlado, _ = cuadrado_rostro
            parche_rostro = frame[qy : qy + qlado, qx : qx + qlado]

            if parche_rostro.shape[0] >= 32 and parche_rostro.shape[1] >= 32:
                img_rostro_64, _ = preprocesar_ventana_64x64(parche_rostro)
                img_save = np.clip(img_rostro_64 * 255.0, 0, 255).astype(np.uint8)
                nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
                ruta_guardar = os.path.join(dir_salida, f"cel_{nombre_base}_{procesadas_sujeto:03d}.png")
                cv2_imwrite_unicode(ruta_guardar, img_save)
                procesadas_sujeto += 1
                total_rostros_proc += 1

                # Ventanas negativas de fondo
                for vx, vy, vw, vh in esquinas_fondo:
                    parche_esq = frame[vy : vy + vh, vx : vx + vw]
                    if parche_esq.shape[0] >= 16 and parche_esq.shape[1] >= 16:
                        img_f_64, _ = preprocesar_ventana_64x64(parche_esq)
                        img_f_save = np.clip(img_f_64 * 255.0, 0, 255).astype(np.uint8)
                        ruta_f = os.path.join(DIR_CLASE_0, f"fondo_cel_{total_fondos_proc:05d}.png")
                        cv2_imwrite_unicode(ruta_f, img_f_save)
                        total_fondos_proc += 1

        print(f"  -> {procesadas_sujeto} rostros extraídos y guardados en '{dir_salida}'.")

    print(f"\n[OK] Proceso por lotes finalizado. Total rostros nuevos: {total_rostros_proc}. Total fondos: {total_fondos_proc}\n")


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
        print("  2. Procesar fotografías del celular por lotes (fotos_crudas_celular/)")
        print("  3. Generar Figura 1 diagnóstica de PDI (figura1_pipeline_pdi_completo.png)")
        print("  4. Salir")
        print("=" * 70)

        opc = input("Seleccione una opción [1-4]: ").strip()
        if opc == "1":
            opcion_captura_webcam()
        elif opc == "2":
            opcion_procesar_fotos_celular()
        elif opc == "3":
            opcion_generar_figura1_diagnostico()
        elif opc == "4":
            print("\nSaliendo del módulo de adquisición. ¡Hasta luego!")
            break
        else:
            print("[AVISO] Opción no válida. Ingrese un número entre 1 y 4.")


if __name__ == "__main__":
    menu_principal()
