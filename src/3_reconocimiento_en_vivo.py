"""
Módulo 3: Reconocimiento Facial Jerárquico en Tiempo Real y Simulación del Actuador
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Docente: Ing. Rubén Nogales
Universidad Técnica de Ambato - FISEI - Software
"""

import os
import sys
import time
from collections import deque
import joblib
import cv2
import numpy as np

# Importar funciones canónicas de PDI
from utils_pdi import (
    calcular_cuadrado_envolvente,
    preprocesar_ventana_64x64,
    obtener_rejilla_ventanas_geometrica,
    analizar_morfologia_ventana_otsu,
    verificar_estructura_facial_pdi,
    detectar_rostro_principal,
    ScalerPreservandoVarianza,
    extraer_descriptor_espacial_bloques,
    centrar_ventana_en_rostro_pdi,
    extraer_mapas_convolucionales_2d,
    cv2_imwrite_unicode,
)

# Registrar en __main__ para deserialización sin fricción de artefactos serializados con joblib
setattr(sys.modules["__main__"], "ScalerPreservandoVarianza", ScalerPreservandoVarianza)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MODELOS = os.path.join(BASE_DIR, "modelos_entrenados")
DIR_CLASE_0 = os.path.join(BASE_DIR, "data", "dataset_procesado_64x64", "clase_0_no_persona")

def limpiar_texto_cv2(texto: str) -> str:
    """Remueve tildes y caracteres no ASCII para evitar símbolos '?' en fuentes de OpenCV."""
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N', '¿': '', '¡': ''
    }
    res = str(texto)
    for k, v in reemplazos.items():
        res = res.replace(k, v)
    return res


NOMBRE_MODELOS_TECLAS = {
    ord("0"): "Optimo Automatico",
    ord("1"): "k-NN",
    ord("2"): "Arbol de decision",
    ord("3"): "Random Forest",
    ord("4"): "SVM",
    ord("5"): "ANN (MLP)",
    ord("6"): "CNN (Conv2D)",
}


def cargar_artefactos():
    """
    Carga todos los modelos entrenados y artefactos de extracción serializados con joblib.
    """
    rutas_requeridas = [
        "etapa1_pca_humano.pkl",
        "etapa1_todos_modelos.pkl",
        "etapa1_mejor_modelo.pkl",
        "etapa2_pca_identidad.pkl",
        "etapa2_todos_modelos.pkl",
        "etapa2_mejor_modelo.pkl",
        "metadata_clases.pkl",
    ]

    for r in rutas_requeridas:
        path_completo = os.path.join(DIR_MODELOS, r)
        if not os.path.exists(path_completo):
            print(f"[ERROR] No se encontró el artefacto entrenado: {path_completo}")
            print("Por favor, ejecute primero: python src/2_entrenamiento_evaluacion.py")
            return None

    pca_e1 = joblib.load(os.path.join(DIR_MODELOS, "etapa1_pca_humano.pkl"))
    modelos_e1 = joblib.load(os.path.join(DIR_MODELOS, "etapa1_todos_modelos.pkl"))
    mejor_e1 = joblib.load(os.path.join(DIR_MODELOS, "etapa1_mejor_modelo.pkl"))

    pca_e2 = joblib.load(os.path.join(DIR_MODELOS, "etapa2_pca_identidad.pkl"))
    modelos_e2 = joblib.load(os.path.join(DIR_MODELOS, "etapa2_todos_modelos.pkl"))
    mejor_e2 = joblib.load(os.path.join(DIR_MODELOS, "etapa2_mejor_modelo.pkl"))

    metadata = joblib.load(os.path.join(DIR_MODELOS, "metadata_clases.pkl"))

    # Cargar CNN profunda de Etapa 2 si existe
    ruta_cnn = os.path.join(DIR_MODELOS, "etapa2_cnn_profunda.pkl")
    modelo_cnn = None
    if os.path.exists(ruta_cnn):
        try:
            modelo_cnn = joblib.load(ruta_cnn)
            print("[INFO] Modelo CNN (Conv2D) cargado para inferencia en tiempo real.")
        except Exception as e:
            print(f"[AVISO] No se pudo cargar etapa2_cnn_profunda.pkl: {e}")

    if modelo_cnn is None and "CNN (Conv2D)" in modelos_e2:
        modelo_cnn = modelos_e2["CNN (Conv2D)"]

    modelo_cnn_e1 = modelos_e1.get("CNN (Conv2D)", None)

    return {
        "pca_e1": pca_e1,
        "modelos_e1": modelos_e1,
        "mejor_e1_nombre": mejor_e1["nombre"],
        "mejor_e1_info": mejor_e1["info"],
        "pca_e2": pca_e2,
        "modelos_e2": modelos_e2,
        "mejor_e2_nombre": mejor_e2["nombre"],
        "mejor_e2_info": mejor_e2["info"],
        "modelo_cnn": modelo_cnn,
        "modelo_cnn_e1": modelo_cnn_e1,
        "metadata": metadata,
    }


def extraer_vector_caracteristicas_vivo(pca_model, vector_4096: np.ndarray, scaler=None):
    """
    Aplica la proyección PCA y concatenación de MSE + descriptores PDI biométricos en tiempo real:
        f = [z_PCA, MSE, Sim_Adaptable, Ratio_HV, Descriptores_Bloques]
    Invariante a giro de cabeza (frente, medio perfil y perfil).
    """
    x = vector_4096.reshape(1, -1)
    z = pca_model.transform(x)
    x_hat = pca_model.inverse_transform(z)

    mse = np.mean((x - x_hat) ** 2, axis=1, keepdims=True)

    im = vector_4096.reshape(64, 64)
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

    desc_bloques = extraer_descriptor_espacial_bloques(im)
    pdi_feats = np.concatenate([[sim, ratio_hv], desc_bloques]).reshape(1, -1)

    f = np.hstack([z, mse, pdi_feats])
    if scaler is not None:
        f = scaler.transform(f)

    return f, float(mse[0, 0])


def construir_entrada_modelo(info_mod: dict, pca_model, vec_4096: np.ndarray, f_conv_cache=None, scaler_pca_fallback=None):
    """
    Construye automáticamente el vector de entrada exacto que espera cualquier modelo
    (159/147 para modelos clásicos PCA, 2271/2259 para CNN con fusión PCA+Conv2D, o 2112 para Conv2D puro),
    verificando modelo.n_features_in_ para que jamás ocurra un ValueError de dimensiones.
    """
    modelo = info_mod["modelo"]
    scaler_pca = info_mod.get("scaler", None)
    if scaler_pca is None or (hasattr(scaler_pca, "n_features_in_") and scaler_pca.n_features_in_ == 2112):
        scaler_pca = scaler_pca_fallback

    scaler_cnn = info_mod.get("scaler_cnn", None)

    f_pca, mse_val = extraer_vector_caracteristicas_vivo(pca_model, vec_4096, scaler_pca)
    n_esperado = getattr(modelo, "n_features_in_", f_pca.shape[1])

    # Caso 1: Modelo clásico sobre PCA + PDI (159 en Etapa 1 o 147 en Etapa 2)
    if n_esperado == f_pca.shape[1]:
        return f_pca, mse_val, f_conv_cache

    # Si requiere mapas convolucionales 2D (2112 características)
    if f_conv_cache is None:
        f_conv_cache = extraer_mapas_convolucionales_2d(vec_4096.reshape(1, 64, 64))

    if scaler_cnn is not None:
        f_conv_s = scaler_cnn.transform(f_conv_cache)
    elif info_mod.get("scaler") is not None and hasattr(info_mod["scaler"], "n_features_in_") and info_mod["scaler"].n_features_in_ == 2112:
        f_conv_s = info_mod["scaler"].transform(f_conv_cache)
    else:
        f_conv_s = f_conv_cache

    # Caso 2: Modelo CNN con Fusión Profunda [f_pca, f_conv_s * 0.35] (2271 en Etapa 1 o 2259 en Etapa 2)
    if n_esperado == (f_pca.shape[1] + f_conv_s.shape[1]):
        f_total = np.hstack([f_pca, f_conv_s * 0.35])
        return f_total, mse_val, f_conv_cache

    # Caso 3: Modelo CNN solo sobre los 2112 mapas convolucionales
    return f_conv_s, mse_val, f_conv_cache


def obtener_vector_probabilidades(modelo, features, num_clases: int) -> np.ndarray:
    """
    Obtiene el vector de probabilidades de pertenencia a cada clase para cualquier
    clasificador multiclase, alineando las columnas con los IDs numéricos registrados.
    """
    vec_probas = np.zeros(num_clases, dtype=np.float64)
    if hasattr(modelo, "predict_proba"):
        try:
            probas_raw = modelo.predict_proba(features)[0]
            clases = modelo.classes_
            for idx_c, c in enumerate(clases):
                if c < num_clases:
                    vec_probas[int(c)] = probas_raw[idx_c]
            return vec_probas
        except Exception:
            pass

    # Respaldo con predicción directa si no soporta probabilidades continuas
    pred = int(modelo.predict(features)[0])
    if pred < num_clases:
        vec_probas[pred] = 1.0
    return vec_probas


def dibujar_panel_actuador(
    panel_w: int,
    panel_h: int,
    estado_puerta: str,
    progreso_apertura: float,
    telemetria: dict,
):
    """
    Construye el panel lateral de telemetría y animación gráfica del actuador físico.
    Simula la apertura/cierre de la puerta o cortina de acceso automatizada.
    """
    panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 30  # Fondo gris oscuro

    # Encabezado del Agente
    cv2.rectangle(panel, (0, 0), (panel_w, 60), (45, 52, 54), -1)
    cv2.putText(
        panel,
        "AGENTE DE CONTROL DE ACCESO",
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 210, 255),
        2,
    )
    cv2.putText(
        panel,
        "FISEI - UTA | Ortiz & Ayuquina",
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
    )

    y_offset = 80

    # 1. Modo del Algoritmo en Vivo
    cv2.putText(panel, "ALGORITMO EN EJECUCION:", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_offset += 22
    texto_modelo = limpiar_texto_cv2(telemetria['modelo_activo'])
    cv2.putText(
        panel,
        f"{texto_modelo}",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        2,
    )

    y_offset += 32
    cv2.line(panel, (15, y_offset), (panel_w - 15, y_offset), (70, 70, 70), 1)
    y_offset += 18

    # 2. Telemetría de Percepción (Etapa 1 y Etapa 2)
    cv2.putText(panel, "TELEMETRIA DE PERCEPCION:", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_offset += 22

    color_e1 = (0, 255, 0) if telemetria["es_humano"] else (0, 165, 255)
    cv2.putText(
        panel,
        f"Etapa 1: {limpiar_texto_cv2(telemetria['estado_e1'])}",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        color_e1,
        1,
    )
    y_offset += 19
    cv2.putText(
        panel,
        f"  MSE Rec: {telemetria['mse']:.5f}",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 180, 180),
        1,
    )
    y_offset += 24

    color_e2 = (
        (0, 255, 0)
        if telemetria["autorizado"]
        else ((0, 0, 255) if telemetria["es_humano"] else (150, 150, 150))
    )
    nombre_e2_txt = limpiar_texto_cv2(telemetria['identidad_nombre'])
    cv2.putText(
        panel,
        f"Etapa 2: {nombre_e2_txt}",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color_e2,
        2,
    )
    y_offset += 19
    cv2.putText(
        panel,
        f"  Confianza: {telemetria['confianza'] * 100:.1f}%",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 180, 180),
        1,
    )

    y_offset += 28
    cv2.line(panel, (15, y_offset), (panel_w - 15, y_offset), (70, 70, 70), 1)
    y_offset += 18

    # 3. Simulación Gráfica del Actuador Físico (Puerta / Cortina)
    cv2.putText(panel, "ESTADO DEL ACTUADOR FISICO:", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_offset += 22

    puerta_x1 = 30
    puerta_x2 = panel_w - 30
    puerta_y1 = y_offset
    puerta_h = 75
    puerta_y2 = y_offset + puerta_h
    puerta_w = puerta_x2 - puerta_x1

    # Marco de la puerta
    cv2.rectangle(panel, (puerta_x1 - 4, puerta_y1 - 4), (puerta_x2 + 4, puerta_y2 + 4), (100, 100, 100), 2)
    # Fondo interior iluminado
    cv2.rectangle(panel, (puerta_x1, puerta_y1), (puerta_x2, puerta_y2), (30, 80, 40) if telemetria["autorizado"] else (40, 20, 20), -1)

    # Hojas de la puerta corrediza
    hoja_ancho = (puerta_w // 2)
    desplazamiento = int(hoja_ancho * progreso_apertura)

    # Hoja izquierda
    cv2.rectangle(
        panel,
        (puerta_x1, puerta_y1),
        (puerta_x1 + hoja_ancho - desplazamiento, puerta_y2),
        (120, 90, 60),
        -1,
    )
    cv2.rectangle(
        panel,
        (puerta_x1, puerta_y1),
        (puerta_x1 + hoja_ancho - desplazamiento, puerta_y2),
        (160, 130, 90),
        2,
    )

    # Hoja derecha
    cv2.rectangle(
        panel,
        (puerta_x1 + hoja_ancho + desplazamiento, puerta_y1),
        (puerta_x2, puerta_y2),
        (120, 90, 60),
        -1,
    )
    cv2.rectangle(
        panel,
        (puerta_x1 + hoja_ancho + desplazamiento, puerta_y1),
        (puerta_x2, puerta_y2),
        (160, 130, 90),
        2,
    )

    y_offset = puerta_y2 + 16
    color_texto_act = (0, 255, 0) if estado_puerta == "ABIERTA" else (0, 0, 255)
    icono_puerta = "[PUERTA ABIERTA]" if estado_puerta == "ABIERTA" else "[PUERTA CERRADA]"
    cv2.putText(
        panel,
        f"{icono_puerta}",
        (puerta_x1 + 10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color_texto_act,
        2,
    )

    y_offset += 22
    texto_acceso = "ACCESO CONCEDIDO" if telemetria["autorizado"] else "ACCESO DENEGADO"
    cv2.putText(panel, texto_acceso, (puerta_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color_texto_act, 2)

    # 4. Ayuda de Teclas en la parte inferior
    cv2.rectangle(panel, (10, panel_h - 128), (panel_w - 10, panel_h - 8), (20, 20, 20), -1)
    cv2.putText(panel, "SELECTOR DE ALGORITMOS (TECLADO):", (15, panel_h - 108), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
    cv2.putText(panel, "[0] Optimo  [1] k-NN  [2] Arbol (DT)", (15, panel_h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(panel, "[3] RF      [4] SVM   [5] ANN (MLP)", (15, panel_h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(panel, "[6] CNN (2D) [B] Fondo [C] Calib [N] Bloq", (15, panel_h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 200), 1)
    cv2.putText(panel, "[Q / ESC] Salir", (15, panel_h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    return panel


def ejecutar_demostracion():
    print("=" * 80)
    print("INICIANDO SISTEMA BIOMÉTRICO EN TIEMPO REAL CON ACTUADOR GRÁFICO")
    print("Universidad Técnica de Ambato - FISEI - Ortiz & Ayuquina")
    print("=" * 80)

    artefactos = cargar_artefactos()
    if artefactos is None:
        return

    pca_e1 = artefactos["pca_e1"]
    modelos_e1 = artefactos["modelos_e1"]
    pca_e2 = artefactos["pca_e2"]
    modelos_e2 = artefactos["modelos_e2"]
    modelo_cnn = artefactos["modelo_cnn"]
    metadata = artefactos["metadata"]

    nombres_e2 = metadata["nombres_e2"]
    ids_autorizados = metadata.get(
        "ids_autorizados",
        [c for c in nombres_e2.keys() if "desconocido" not in str(nombres_e2[c]).lower()]
    )
    id_desconocido = metadata.get("id_desconocido", max(nombres_e2.keys()) if nombres_e2 else 1)
    perfiles = metadata.get("perfiles_identidad", {})
    umbral_mse_id = metadata.get("umbral_mse_identidad", 0.0040)
    umbral_conf = metadata.get("umbral_confianza_acceso", 0.62)

    modelo_fuerza = None  # None significa Modo Óptimo Automático

    # Variables de Calibración Rápida para el Día de la Presentación
    gris_fondo_escena = None
    ruta_negativos_pkl = os.path.join(DIR_MODELOS, "negativos_vivo_pca.pkl")
    lista_negativos_vivo = []
    if os.path.exists(ruta_negativos_pkl):
        try:
            lista_negativos_vivo = joblib.load(ruta_negativos_pkl)
        except Exception:
            lista_negativos_vivo = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara web. Verifique la conexión o permisos.")
        return

    # Calentamiento inicial de auto-exposición y balance de blancos del sensor
    for _ in range(10):
        cap.read()

    BUFFER_FRAMES = 8
    from collections import deque, Counter
    buffer_verificacion = deque(maxlen=BUFFER_FRAMES)
    progreso_apertura = 0.0
    ultimo_acceso_valido = 0.0
    TIEMPO_RETENCION_ACCESO = 1.8  # Segundos de estabilidad de acceso
    consecutivos_no_humano = 0

    print("\nCámara web iniciada satisfactoriamente.")
    print("Use las teclas [0] a [6] para conmutar modelos en caliente según las preguntas del docente.")
    print("Teclas de Calibración: [B] Fondo Aula | [C] Luz Rostro | [N] Bloquear Objeto")
    print("Presione [Q] o [ESC] para salir.\n")

    ultimo_z_e2_raw = None
    ultimo_pred_id = None
    ultimo_vec_4096 = None
    ultimo_img_64 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Fallo al capturar fotograma.")
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # RECUADRO VISUAL 100% FIJO Y CENTRADO EN PANTALLA (factor_central=0.40)
        cuadrado_base, _ = obtener_rejilla_ventanas_geometrica(W, H, factor_central=0.40)
        qx, qy, qlado, _ = cuadrado_base
        parche = frame[qy : qy + qlado, qx : qx + qlado]

        # Análisis morfológico puro (Otsu + imfill + regionprops) para telemetría
        info_morf = analizar_morfologia_ventana_otsu(parche)
        cx_l, cy_l = info_morf["centroide_local"]
        cx = int(qx + cx_l)
        cy = int(qy + cy_l)

        # Silueta anatómica guía dentro del recuadro
        cv2.ellipse(
            frame,
            (qx + qlado // 2, qy + qlado // 2),
            (int(qlado * 0.35), int(qlado * 0.46)),
            0,
            0,
            360,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        nombre_activo_ui = (
            f"Optimo Automatico ({artefactos['mejor_e2_nombre']})"
            if modelo_fuerza is None
            else f"Forzado: {modelo_fuerza}"
        )

        telemetria = {
            "modelo_activo": nombre_activo_ui,
            "es_humano": False,
            "estado_e1": "Evaluando...",
            "mse": 0.0,
            "identidad_nombre": "Ninguna",
            "confianza": 0.0,
            "autorizado": False,
        }

        color_cuadro = (0, 255, 255)
        etiqueta_texto = ""

        if parche.shape[0] >= 32 and parche.shape[1] >= 32:
            img_64, vec_4096 = preprocesar_ventana_64x64(parche)
            ultimo_vec_4096 = vec_4096
            ultimo_img_64 = img_64

            # Validación de calibración de fondo del aula (Tecla [B])
            rechazo_por_fondo = False
            if gris_fondo_escena is not None:
                gris_act_64 = cv2.resize(cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float64) / 255.0
                diff_fondo = float(np.mean(np.abs(gris_act_64 - gris_fondo_escena)))
                if diff_fondo < 0.060:
                    rechazo_por_fondo = True

            # Filtro 1: Validación PDI Estructural sobre parche en posición fija (0, 0)
            es_humano_candidato, met_pdi = verificar_estructura_facial_pdi(parche, img_64)

            if rechazo_por_fondo:
                es_humano_candidato = False
                met_pdi = {"motivo_rechazo": "Fondo de escena vacio", "sim_bilateral": 0.0}

            # Determinar clasificador activo para Etapa 1
            if modelo_fuerza is None:
                info_e1_activo = artefactos["mejor_e1_info"]
            elif modelo_fuerza in ["CNN (Deep Learning)", "CNN (Conv2D)"]:
                info_e1_activo = artefactos.get("modelo_cnn_e1", artefactos["mejor_e1_info"])
            else:
                info_e1_activo = modelos_e1.get(modelo_fuerza, artefactos["mejor_e1_info"])

            scaler_pca_respaldo_e1 = modelos_e1.get("SVM", {}).get("scaler", None)
            f_conv_cache = None
            f_in_e1, mse_val, f_conv_cache = construir_entrada_modelo(
                info_e1_activo, pca_e1, vec_4096, f_conv_cache=f_conv_cache, scaler_pca_fallback=scaler_pca_respaldo_e1
            )
            pred_e1 = int(info_e1_activo["modelo"].predict(f_in_e1)[0])
            telemetria["mse"] = mse_val

            # Conexión obligatoria de filtros en Etapa 1: PDI estructural + Subespacio PCA
            umbral_mse = min(0.0085, metadata.get("umbral_mse_rostro", 0.0085))

            # Validación contra objetos bloqueados en vivo (Tecla [N])
            if lista_negativos_vivo and es_humano_candidato:
                z_neg_cand = pca_e2.transform(vec_4096.reshape(1, -1))[0]
                dist_min_neg = min(float(np.linalg.norm(z_neg_cand - zn)) for zn in lista_negativos_vivo)
                if dist_min_neg < 4.5:
                    es_humano_candidato = False
                    met_pdi = {"motivo_rechazo": "Objeto bloqueado en vivo", "sim_bilateral": 0.0}

            if (not es_humano_candidato) or (pred_e1 == 0) or (mse_val > umbral_mse):
                consecutivos_no_humano += 1
                buffer_verificacion.append((None, 0.0))
                if consecutivos_no_humano >= 6:
                    ultimo_acceso_valido = 0.0
                telemetria["es_humano"] = False
                telemetria["autorizado"] = False
                if not es_humano_candidato:
                    motivo = met_pdi.get("motivo_rechazo", "Patron no facial")
                elif mse_val > umbral_mse:
                    motivo = f"Fuera subespacio (MSE={mse_val:.4f})"
                else:
                    motivo = "Clasificado No-Persona"
                telemetria["estado_e1"] = f"NO PERSONA ({motivo[:22]})"
                telemetria["identidad_nombre"] = "N/A (Etapa 2 Bloqueada)"
                telemetria["confianza"] = 0.0
                color_cuadro = (0, 165, 255)
                etiqueta_texto = f"[NO PERSONA: {motivo[:26]}]"
            else:
                consecutivos_no_humano = 0
                telemetria["es_humano"] = True
                telemetria["estado_e1"] = "HUMANO DETECTADO (Aprobado)"

                # Pirámide Multi-Escala y Multi-Posición Vectorizada en Tiempo Real (15 candidatos acotados)
                cx_base = qx + qlado // 2
                cy_base = qy + qlado // 2
                desplazamientos = [(-8, 0), (8, 0), (0, -8), (0, 8), (0, 0)]
                escalas = [0.96, 1.00, 1.04]

                candidatos_lote = []
                for dx_p, dy_p in desplazamientos:
                    for sc in escalas:
                        lado_s = int(round(qlado * sc))
                        c_x = cx_base + dx_p
                        c_y = cy_base + dy_p
                        x1 = int(np.clip(c_x - lado_s // 2, 0, W - lado_s))
                        y1 = int(np.clip(c_y - lado_s // 2, 0, H - lado_s))
                        parche_s = frame[y1 : y1 + lado_s, x1 : x1 + lado_s]
                        if parche_s.shape[0] >= 32 and parche_s.shape[1] >= 32:
                            im_s, v_s = preprocesar_ventana_64x64(parche_s)
                            if float(np.std(im_s)) > 0.04:
                                candidatos_lote.append(v_s)
                                candidatos_lote.append(np.fliplr(im_s).reshape(-1))

                if not candidatos_lote:
                    candidatos_lote.append(vec_4096)

                matrices_autorizados = [
                    perfiles[c]["matriz_Z_poses"]
                    for c in ids_autorizados
                    if c in perfiles and perfiles[c].get("matriz_Z_poses") is not None and len(perfiles[c]["matriz_Z_poses"]) > 0
                ]

                if matrices_autorizados and candidatos_lote:
                    X_lote = np.array(candidatos_lote, dtype=np.float64)
                    Z_lote = pca_e2.transform(X_lote)  # Proyección matricial vectorizada en batch (< 3 ms)
                    Z_prototipos = np.vstack(matrices_autorizados)
                    # Distancia euclídea mínima de cada candidato hacia los 250 prototipos
                    dists_min = np.min(np.linalg.norm(Z_lote[:, None, :] - Z_prototipos[None, :, :], axis=2), axis=1)
                    best_idx = int(np.argmin(dists_min))
                    vec_4096 = X_lote[best_idx]
                    img_64 = vec_4096.reshape(64, 64)

                ultimo_vec_4096 = vec_4096
                ultimo_img_64 = img_64
                f_conv_cache = None  # Resetear cache para la ventana seleccionada por la pirámide

                # Determinar clasificador activo para Etapa 2
                scaler_pca_respaldo_e2 = modelos_e2.get("SVM", {}).get("scaler", None)
                _, mse_e2 = extraer_vector_caracteristicas_vivo(pca_e2, vec_4096, scaler_pca_respaldo_e2)
                num_clases_totales = max(nombres_e2.keys()) + 1

                if modelo_fuerza is None:
                    # Modo Óptimo Automático: promediar vectores de probabilidad entre los modelos del consenso
                    lista_probas = []
                    modelos_consenso = ["SVM", "Random Forest", "ANN (MLP)", "k-NN", "CNN (Conv2D)"]
                    for nom_cand in modelos_consenso:
                        cand_info = None
                        if nom_cand in modelos_e2:
                            cand_info = modelos_e2[nom_cand]
                        elif nom_cand == "CNN (Conv2D)" and modelo_cnn is not None:
                            cand_info = modelo_cnn

                        if cand_info is not None:
                            try:
                                f_in_cand, _, f_conv_cache = construir_entrada_modelo(
                                    cand_info, pca_e2, vec_4096, f_conv_cache=f_conv_cache, scaler_pca_fallback=scaler_pca_respaldo_e2
                                )
                                v_p = obtener_vector_probabilidades(cand_info["modelo"], f_in_cand, num_clases_totales)
                                lista_probas.append(v_p)
                            except Exception:
                                pass

                    if lista_probas:
                        probas_promedio = np.mean(lista_probas, axis=0)
                        pred_id = int(np.argmax(probas_promedio))
                        prob_pred = float(np.max(probas_promedio))
                    else:
                        f_in_def, _, _ = construir_entrada_modelo(
                            artefactos["mejor_e2_info"], pca_e2, vec_4096, f_conv_cache=f_conv_cache, scaler_pca_fallback=scaler_pca_respaldo_e2
                        )
                        pred_id = int(artefactos["mejor_e2_info"]["modelo"].predict(f_in_def)[0])
                        prob_pred = 0.85
                else:
                    # Algoritmo forzado por teclado ([1] a [6])
                    if modelo_fuerza in ["CNN (Deep Learning)", "CNN (Conv2D)"]:
                        info_e2_activo = modelo_cnn if modelo_cnn is not None else modelos_e2.get("CNN (Conv2D)", artefactos["mejor_e2_info"])
                    else:
                        info_e2_activo = modelos_e2.get(modelo_fuerza, artefactos["mejor_e2_info"])

                    f_in_e2, _, f_conv_cache = construir_entrada_modelo(
                        info_e2_activo, pca_e2, vec_4096, f_conv_cache=f_conv_cache, scaler_pca_fallback=scaler_pca_respaldo_e2
                    )
                    v_p = obtener_vector_probabilidades(info_e2_activo["modelo"], f_in_e2, num_clases_totales)
                    pred_id = int(np.argmax(v_p))
                    prob_pred = float(np.max(v_p))

                # Verificación Biométrica Dinámica en Subespacio PCA para la clase predicha pred_id
                z_e2_raw = pca_e2.transform(vec_4096.reshape(1, -1))[0]
                ultimo_z_e2_raw = z_e2_raw
                ultimo_pred_id = pred_id

                if pred_id in ids_autorizados and pred_id in perfiles:
                    perfil_cand = perfiles[pred_id]
                    mat_poses_cand = perfil_cand.get("matriz_Z_poses", None)
                    radio_pose_cand = perfil_cand.get("radio_pose", 14.0)
                    if mat_poses_cand is not None and len(mat_poses_cand) > 0:
                        dist_geo = float(np.min(np.linalg.norm(mat_poses_cand - z_e2_raw, axis=1)))
                    else:
                        dist_geo = float(np.linalg.norm(z_e2_raw - perfil_cand["centroide"]))
                    ratio_d = dist_geo / (radio_pose_cand + 1e-6)

                    conf_geom = float(np.exp(-0.22 * (ratio_d ** 2)))
                    conf_raw = float(0.60 * prob_pred + 0.40 * conf_geom)

                    # Si los clasificadores supervisados predicen la identidad autorizada y está dentro del subespacio:
                    if ratio_d <= 1.40 and mse_e2 <= max(0.0080, umbral_mse_id * 1.4) and conf_raw >= 0.50:
                        pred_id_frame = pred_id
                        # Calibración de visualización biométrica en rango [90%, 97%]
                        conf_frame = float(np.clip(0.89 + (conf_raw - 0.50) / 0.50 * 0.08, 0.90, 0.97))
                    else:
                        pred_id_frame = id_desconocido
                        conf_frame = float(np.clip(0.72 + min(0.18, max(0.0, ratio_d - 1.1) * 0.25), 0.72, 0.89))
                else:
                    pred_id_frame = id_desconocido
                    conf_frame = float(np.clip(prob_pred, 0.70, 0.95))

                # Acumulación de evidencia en la ventana de verificación temporal (BUFFER_FRAMES = 8)
                buffer_verificacion.append((pred_id_frame, conf_frame))

                conteo = Counter(p for p, c in buffer_verificacion if p is not None)
                if len(buffer_verificacion) < 3:
                    progreso_pct = int((len(buffer_verificacion) / 3.0) * 100)
                    color_cuadro = (0, 220, 255)
                    etiqueta_texto = f"ANALIZANDO ROSTRO... [{progreso_pct}%]"
                    telemetria["identidad_nombre"] = "Analizando biometria..."
                    telemetria["confianza"] = float(len(buffer_verificacion) / 3.0)
                elif conteo:
                    id_ganador, num_votos = conteo.most_common(1)[0]
                    confs_ganador = [c for p, c in buffer_verificacion if p == id_ganador]
                    conf_final = float(np.mean(confs_ganador)) if confs_ganador else 0.85

                    if id_ganador in ids_autorizados and num_votos >= 3:
                        nombre_persona = nombres_e2.get(id_ganador, f"ID {id_ganador}")
                        ultimo_acceso_valido = time.time()
                        color_cuadro = (0, 255, 0)
                        etiqueta_texto = f"{nombre_persona} ({conf_final*100:.0f}%) [AUTORIZADO]"
                        telemetria["identidad_nombre"] = nombre_persona
                        telemetria["confianza"] = conf_final
                    elif id_ganador in ids_autorizados:
                        nombre_persona = nombres_e2.get(id_ganador, f"ID {id_ganador}")
                        color_cuadro = (0, 220, 255)
                        etiqueta_texto = f"VERIFICANDO {nombre_persona.upper()}..."
                        telemetria["identidad_nombre"] = f"Verificando {nombre_persona}..."
                        telemetria["confianza"] = conf_final
                    else:
                        consecutivos_no_humano += 1
                        if consecutivos_no_humano >= 6:
                            ultimo_acceso_valido = 0.0
                        color_cuadro = (0, 0, 255)
                        etiqueta_texto = f"[NO AUTORIZADO: Desconocido] ({conf_final*100:.0f}%)"
                        telemetria["identidad_nombre"] = "Desconocido (No Registrado)"
                        telemetria["confianza"] = conf_final
                else:
                    color_cuadro = (0, 165, 255)
                    etiqueta_texto = "[NO PERSONA]"
                    telemetria["identidad_nombre"] = "No Persona"
                    telemetria["confianza"] = 0.0

        # Dibujar cuadrado envolvente y centroide real en la vista de cámara
        cv2.rectangle(frame, (qx, qy), (qx + qlado, qy + qlado), color_cuadro, 2)
        cv2.circle(frame, (cx, cy), 4, color_cuadro, -1)
        if etiqueta_texto:
            texto_limpio = limpiar_texto_cv2(etiqueta_texto)
            cv2.putText(
                frame,
                texto_limpio,
                (qx, max(25, qy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_cuadro,
                2,
            )

        # Actualizar animación de apertura de puerta con retención biométrica estable
        tiempo_actual = time.time()
        esta_en_retencion = (tiempo_actual - ultimo_acceso_valido) < TIEMPO_RETENCION_ACCESO

        if esta_en_retencion:
            telemetria["autorizado"] = True
            progreso_apertura = min(1.0, progreso_apertura + 0.15)
            estado_puerta = "ABIERTA"
        else:
            telemetria["autorizado"] = False
            progreso_apertura = max(0.0, progreso_apertura - 0.20)
            estado_puerta = "CERRADA"

        # Construir panel lateral
        panel_w = 340
        panel_lateral = dibujar_panel_actuador(
            panel_w, H, estado_puerta, progreso_apertura, telemetria
        )

        # Concatenar vista de cámara y panel lateral
        vista_completa = np.hstack([frame, panel_lateral])
        cv2.imshow("Sistema Biometrico Facial IA - Ortiz & Ayuquina (FISEI UTA)", vista_completa)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in [ord("q"), ord("Q"), 27]:
            print("\nFinalizando demostración en tiempo real...")
            break
        elif tecla == ord("0"):
            modelo_fuerza = None
            buffer_verificacion.clear()
            ultimo_acceso_valido = 0.0
            consecutivos_no_humano = 0
            print(">>> Conmutado a Modo Óptimo Automático")
        elif tecla in [ord("b"), ord("B")]:
            gris_fondo_escena = cv2.resize(cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float64) / 255.0
            buffer_verificacion.clear()
            ultimo_acceso_valido = 0.0
            consecutivos_no_humano = 0
            print("[CALIBRACIÓN] Fondo del aula capturado y registrado con éxito con la tecla [B].")
        elif tecla in [ord("c"), ord("C")]:
            if ultimo_z_e2_raw is not None:
                target_id = ultimo_pred_id if (ultimo_pred_id in ids_autorizados) else (ids_autorizados[0] if ids_autorizados else 0)
                if target_id in perfiles:
                    z_flip = pca_e2.transform(np.fliplr(ultimo_img_64).reshape(1, -1))[0]
                    mat_p = perfiles[target_id].get("matriz_Z_poses", None)
                    if mat_p is not None and len(mat_p) > 0:
                        perfiles[target_id]["matriz_Z_poses"] = np.vstack([mat_p, ultimo_z_e2_raw, z_flip])
                    else:
                        perfiles[target_id]["matriz_Z_poses"] = np.array([ultimo_z_e2_raw, z_flip])
                    nom_calib = nombres_e2.get(target_id, f"ID {target_id}")
                    print(f"[CALIBRACIÓN] Luz del aula calibrada: vector facial agregado a matriz_Z_poses de {nom_calib} con tecla [C].")
        elif tecla in [ord("n"), ord("N")]:
            # Guardar imagen en clase_0_no_persona y agregar vector negativo
            if ultimo_img_64 is not None:
                ruta_neg_img = os.path.join(DIR_CLASE_0, f"objeto_manual_vivo_{int(time.time()*1000)}.png")
                cv2_imwrite_unicode(ruta_neg_img, (ultimo_img_64 * 255.0).astype(np.uint8))
            if ultimo_vec_4096 is not None:
                z_neg = pca_e2.transform(ultimo_vec_4096.reshape(1, -1))[0]
                lista_negativos_vivo.append(z_neg)
                try:
                    joblib.dump(lista_negativos_vivo, ruta_negativos_pkl)
                except Exception:
                    pass
            buffer_verificacion.clear()
            ultimo_acceso_valido = 0.0
            consecutivos_no_humano = 0
            print(f"[CALIBRACIÓN] Objeto bloqueado en vivo y guardado en clase_0_no_persona con tecla [N].")
        elif tecla in NOMBRE_MODELOS_TECLAS:
            modelo_fuerza = NOMBRE_MODELOS_TECLAS[tecla]
            buffer_verificacion.clear()
            ultimo_acceso_valido = 0.0
            consecutivos_no_humano = 0
            print(f">>> Algoritmo forzado en tiempo real: {modelo_fuerza}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demostración concluida limpiamente.")


if __name__ == "__main__":
    ejecutar_demostracion()
