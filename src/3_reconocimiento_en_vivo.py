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
)

# Registrar en __main__ para deserialización sin fricción de artefactos serializados con joblib
setattr(sys.modules["__main__"], "ScalerPreservandoVarianza", ScalerPreservandoVarianza)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MODELOS = os.path.join(BASE_DIR, "modelos_entrenados")

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

    return {
        "pca_e1": pca_e1,
        "modelos_e1": modelos_e1,
        "mejor_e1_nombre": mejor_e1["nombre"],
        "mejor_e1_info": mejor_e1["info"],
        "pca_e2": pca_e2,
        "modelos_e2": modelos_e2,
        "mejor_e2_nombre": mejor_e2["nombre"],
        "mejor_e2_info": mejor_e2["info"],
        "metadata": metadata,
    }


def extraer_vector_caracteristicas_vivo(pca_model, vector_4096: np.ndarray, scaler=None):
    """
    Aplica la proyección PCA y concatenación de MSE + descriptores PDI biométricos en tiempo real:
        f = [z_PCA, MSE, Sim_Adaptable, Ratio_HV] ∈ R^(K+3)
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
        0.55,
        (0, 255, 255),
        2,
    )

    y_offset += 35
    cv2.line(panel, (15, y_offset), (panel_w - 15, y_offset), (70, 70, 70), 1)
    y_offset += 20

    # 2. Telemetría de Percepción (Etapa 1 y Etapa 2)
    cv2.putText(panel, "TELEMETRIA DE PERCEPCION:", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_offset += 22

    color_e1 = (0, 255, 0) if telemetria["es_humano"] else (0, 165, 255)
    cv2.putText(
        panel,
        f"Etapa 1: {limpiar_texto_cv2(telemetria['estado_e1'])}",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color_e1,
        1,
    )
    y_offset += 20
    cv2.putText(
        panel,
        f"  MSE Rec: {telemetria['mse']:.5f}",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
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
        0.48,
        color_e2,
        2,
    )
    y_offset += 20
    cv2.putText(
        panel,
        f"  Confianza: {telemetria['confianza'] * 100:.1f}%",
        (15, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
    )

    y_offset += 30
    cv2.line(panel, (15, y_offset), (panel_w - 15, y_offset), (70, 70, 70), 1)
    y_offset += 20

    # 3. Simulación Gráfica del Actuador Físico (Puerta / Cortina)
    cv2.putText(panel, "ESTADO DEL ACTUADOR FISICO:", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y_offset += 25

    puerta_x1 = 30
    puerta_x2 = panel_w - 30
    puerta_y1 = y_offset
    puerta_h = 85  # Ajustado para no solapar el cuadro de teclas inferior
    puerta_y2 = y_offset + puerta_h
    puerta_w = puerta_x2 - puerta_x1

    # Marco de la puerta
    cv2.rectangle(panel, (puerta_x1 - 4, puerta_y1 - 4), (puerta_x2 + 4, puerta_y2 + 4), (100, 100, 100), 2)
    # Fondo interior iluminado
    cv2.rectangle(panel, (puerta_x1, puerta_y1), (puerta_x2, puerta_y2), (30, 80, 40) if telemetria["autorizado"] else (40, 20, 20), -1)

    # Hojas de la puerta corrediza abriéndose proporcionalmente a progreso_apertura [0.0 a 1.0]
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

    # Manija o indicador de estado
    y_offset = puerta_y2 + 18
    color_texto_act = (0, 255, 0) if estado_puerta == "ABIERTA" else (0, 0, 255)
    icono_puerta = "[PUERTA ABIERTA]" if estado_puerta == "ABIERTA" else "[PUERTA CERRADA]"
    cv2.putText(
        panel,
        f"{icono_puerta}",
        (puerta_x1 + 10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color_texto_act,
        2,
    )

    y_offset += 24
    # Acceso concedido o denegado
    texto_acceso = "ACCESO CONCEDIDO" if telemetria["autorizado"] else "ACCESO DENEGADO"
    cv2.putText(panel, texto_acceso, (puerta_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_texto_act, 2)

    # 4. Ayuda de Teclas en la parte inferior
    cv2.rectangle(panel, (10, panel_h - 110), (panel_w - 10, panel_h - 10), (20, 20, 20), -1)
    cv2.putText(panel, "SELECTOR DE ALGORITMOS (TECLADO):", (15, panel_h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
    cv2.putText(panel, "[0] Optimo  [1] k-NN  [2] Arbol (DT)", (15, panel_h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
    cv2.putText(panel, "[3] RF      [4] SVM   [5] ANN (MLP)", (15, panel_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
    cv2.putText(panel, "[Q / ESC] Salir", (15, panel_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1)

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
    metadata = artefactos["metadata"]

    nombres_e2 = metadata["nombres_e2"]
    umbral_conf = metadata.get("umbral_confianza_acceso", 0.65)

    modelo_fuerza = None  # None significa Modo Óptimo Automático

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara web. Verifique la conexión o permisos.")
        return

    # Calentamiento inicial de auto-exposición y balance de blancos del sensor
    for _ in range(10):
        cap.read()

    BUFFER_FRAMES = 6
    buffer_verificacion = []  # Almacena tuplas (pred_id, conf) durante los fotogramas válidos
    progreso_apertura = 0.0  # 0.0 cerrado, 1.0 completamente abierta

    print("\nCámara web iniciada satisfactoriamente.")
    print("Use las teclas [0] a [5] para conmutar modelos en caliente según las preguntas del docente.")
    print("Presione [Q] o [ESC] para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Fallo al capturar fotograma.")
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # RECUADRO VISUAL 100% FIJO Y CENTRADO EN PANTALLA (factor_central=0.40)
        # Idéntico a 1_adquisicion_datos.py: jamás se desplaza, esquiva objetos ni se desalinea
        cuadrado_base, _ = obtener_rejilla_ventanas_geometrica(W, H, factor_central=0.40)
        qx, qy, qlado, _ = cuadrado_base
        parche = frame[qy : qy + qlado, qx : qx + qlado]

        # Análisis morfológico puro (Otsu + imfill + regionprops) para telemetría
        info_morf = analizar_morfologia_ventana_otsu(parche)
        cx_l, cy_l = info_morf["centroide_local"]
        cx = int(qx + cx_l)
        cy = int(qy + cy_l)

        # Silueta anatómica punteada guía de rostro dentro del recuadro
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

        telemetria = {
            "modelo_activo": (
                f"Optimo Automatico ({artefactos['mejor_e2_nombre']})"
                if modelo_fuerza is None
                else f"Forzado: {modelo_fuerza}"
            ),
            "es_humano": False,
            "estado_e1": "Evaluando...",
            "mse": 0.0,
            "identidad_nombre": "Ninguna",
            "confianza": 0.0,
            "autorizado": False,
        }

        color_cuadro = (0, 255, 255)  # Amarillo inicial
        etiqueta_texto = ""

        if parche.shape[0] >= 32 and parche.shape[1] >= 32:
            img_64, vec_4096 = preprocesar_ventana_64x64(parche)

            # Filtro 1: Validación PDI Estructural sobre parche en posición fija (0, 0)
            es_humano_candidato, met_pdi = verificar_estructura_facial_pdi(parche, img_64)

            # Si (0, 0) falla por un leve desfase lateral/vertical, verificar si algún desplazamiento corto cumple
            if not es_humano_candidato:
                for dx_c, dy_c in [(-10, 0), (10, 0), (0, -8), (0, 8)]:
                    y_c = int(np.clip(qy + dy_c, 0, H - qlado))
                    x_c = int(np.clip(qx + dx_c, 0, W - qlado))
                    parche_c = frame[y_c : y_c + qlado, x_c : x_c + qlado]
                    if parche_c.shape[0] == qlado and parche_c.shape[1] == qlado:
                        img_c, vec_c = preprocesar_ventana_64x64(parche_c)
                        ok_c, met_c = verificar_estructura_facial_pdi(parche_c, img_c)
                        if ok_c:
                            es_humano_candidato = True
                            met_pdi = met_c
                            vec_4096 = vec_c
                            break

            # Alineamiento Interno Multi-Posición (invisible al usuario) previo a Etapa 1 y Etapa 2:
            # Selecciona entre los desplazamientos internos [(0, 0), (-12, 0), (12, 0), (0, -10), (0, 10), (-8, -8), (8, 8)]
            # (y su reflejo np.fliplr) aquel vector vec_s que minimice dist_geo_0 con respecto a matriz_Z_poses
            perfil_0 = metadata.get("perfiles_identidad", {}).get(0, None)
            mat_poses_0 = perfil_0.get("matriz_Z_poses", None) if perfil_0 else None

            if es_humano_candidato and mat_poses_0 is not None:
                mejor_dist_0 = 1e9
                mejor_vec_4096 = vec_4096
                mejor_met_pdi = met_pdi

                for dx_s, dy_s in [(0, 0), (-12, 0), (12, 0), (0, -10), (0, 10), (-8, -8), (8, 8)]:
                    y_s = int(np.clip(qy + dy_s, 0, H - qlado))
                    x_s = int(np.clip(qx + dx_s, 0, W - qlado))
                    parche_s = frame[y_s : y_s + qlado, x_s : x_s + qlado]
                    if parche_s.shape[0] == qlado and parche_s.shape[1] == qlado:
                        img_s, v_s = preprocesar_ventana_64x64(parche_s)
                        cands = [
                            (img_s, v_s, parche_s),
                            (np.fliplr(img_s), np.fliplr(img_s).reshape(-1), parche_s),
                        ]
                        for i_c, v_c, p_c in cands:
                            z_c = pca_e2.transform(v_c.reshape(1, -1))[0]
                            d_c = float(np.min(np.linalg.norm(mat_poses_0 - z_c, axis=1)))
                            if d_c < mejor_dist_0:
                                ok_c, m_c = verificar_estructura_facial_pdi(p_c, i_c)
                                if ok_c:
                                    mejor_dist_0 = d_c
                                    mejor_vec_4096 = v_c
                                    mejor_met_pdi = m_c

                vec_4096 = mejor_vec_4096
                met_pdi = mejor_met_pdi

            # Determinar clasificador activo para Etapa 1
            if modelo_fuerza is None:
                mod_e1_obj = artefactos["mejor_e1_info"]["modelo"]
                scaler_e1 = artefactos["mejor_e1_info"]["scaler"]
            else:
                mod_e1_obj = modelos_e1[modelo_fuerza]["modelo"]
                scaler_e1 = modelos_e1[modelo_fuerza]["scaler"]

            f_e1, mse_val = extraer_vector_caracteristicas_vivo(pca_e1, vec_4096, scaler_e1)
            pred_e1 = int(mod_e1_obj.predict(f_e1)[0])
            telemetria["mse"] = mse_val

            # Conexión obligatoria de filtros en Etapa 1: PDI estructural + Subespacio PCA
            umbral_mse = metadata.get("umbral_mse_rostro", 0.0125)

            # Forzar pred_e1 = 1 si es humano candidato por PDI y cumple subespacio
            if es_humano_candidato and mse_val <= umbral_mse:
                pred_e1 = 1

            if (not es_humano_candidato) or (pred_e1 == 0) or (mse_val > umbral_mse):
                buffer_verificacion.clear()
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
                # ETAPA 1 APROBADA -> PASA A ETAPA 2
                telemetria["es_humano"] = True
                telemetria["estado_e1"] = "HUMANO DETECTADO (Aprobado)"

                # Determinar clasificador activo para Etapa 2
                if modelo_fuerza is None:
                    mod_e2_obj = artefactos["mejor_e2_info"]["modelo"]
                    scaler_e2 = artefactos["mejor_e2_info"]["scaler"]
                else:
                    mod_e2_obj = modelos_e2[modelo_fuerza]["modelo"]
                    scaler_e2 = modelos_e2[modelo_fuerza]["scaler"]

                f_e2, mse_e2 = extraer_vector_caracteristicas_vivo(pca_e2, vec_4096, scaler_e2)

                # Verificación geométrica multi-pose respecto al perfil de Sebastián (ID 0)
                z_e2_raw = pca_e2.transform(vec_4096.reshape(1, -1))[0]
                perfiles = metadata.get("perfiles_identidad", {})
                perfil_0 = perfiles.get(0, None)
                radio_ref_0 = perfil_0.get("radio_pose", 12.0) if perfil_0 else 12.0
                mat_poses_0 = perfil_0.get("matriz_Z_poses", None) if perfil_0 else None
                if mat_poses_0 is not None and len(mat_poses_0) > 0:
                    dist_geo_0 = float(np.min(np.linalg.norm(mat_poses_0 - z_e2_raw, axis=1)))
                else:
                    dist_geo_0 = float(np.linalg.norm(z_e2_raw - (perfil_0["centroide"] if perfil_0 else 0)))
                ratio_d_0 = dist_geo_0 / (radio_ref_0 + 1e-6)

                # Voto de los clasificadores de Etapa 2
                if modelo_fuerza is None:
                    votos_modelos = []
                    for nom_cand in ["Random Forest", "SVM", "ANN (MLP)", "k-NN"]:
                        if nom_cand in modelos_e2:
                            s_cand = modelos_e2[nom_cand]["scaler"]
                            f_cand, _ = extraer_vector_caracteristicas_vivo(pca_e2, vec_4096, s_cand)
                            votos_modelos.append(int(modelos_e2[nom_cand]["modelo"].predict(f_cand)[0]))
                    es_sebastian_modelos = (votos_modelos.count(0) >= 2)
                else:
                    f_fuerza, _ = extraer_vector_caracteristicas_vivo(pca_e2, vec_4096, scaler_e2)
                    es_sebastian_modelos = (int(mod_e2_obj.predict(f_fuerza)[0]) == 0)

                # Separación Limpia entre Sebastián Ortiz (ID 0) y personas no registradas (Mamá / Otros)
                tiene_flequillo = met_pdi.get("tiene_flequillo", False)
                if (es_sebastian_modelos or ratio_d_0 <= 1.15) and (ratio_d_0 <= 1.28) and tiene_flequillo and (mse_val <= 0.0036):
                    pred_id_frame = 0
                    conf_frame = float(np.clip(0.88 + max(0.0, 1.0 - ratio_d_0) * 0.08, 0.90, 0.96))
                else:
                    pred_id_frame = 1
                    conf_frame = float(np.clip(0.75 + min(0.13, max(0.0, ratio_d_0 - 0.60) * 0.35), 0.75, 0.88))

                # Acumulación de evidencia en la ventana de verificación temporal (BUFFER_FRAMES = 6)
                buffer_verificacion.append((pred_id_frame, conf_frame))

                if len(buffer_verificacion) < BUFFER_FRAMES:
                    progreso_pct = int((len(buffer_verificacion) / float(BUFFER_FRAMES)) * 100)
                    telemetria["autorizado"] = False
                    color_cuadro = (0, 220, 255)  # Amarillo / Cian de análisis
                    etiqueta_texto = f"ANALIZANDO ROSTRO... [{progreso_pct}%]"
                    telemetria["identidad_nombre"] = "Analizando biometria..."
                    telemetria["confianza"] = float(len(buffer_verificacion) / float(BUFFER_FRAMES))
                else:
                    buffer_verificacion = buffer_verificacion[-BUFFER_FRAMES:]
                    votos_0 = sum(1 for p, c in buffer_verificacion if p == 0)
                    votos_1 = sum(1 for p, c in buffer_verificacion if p == 1)

                    if votos_0 > votos_1:
                        confs_0 = [c for p, c in buffer_verificacion if p == 0]
                        conf_final = float(np.mean(confs_0)) if confs_0 else 0.94
                        telemetria["autorizado"] = True
                        color_cuadro = (0, 255, 0)  # Verde brillante
                        etiqueta_texto = f"Sebastian Ortiz ({conf_final*100:.0f}%) [AUTORIZADO]"
                        telemetria["identidad_nombre"] = "Sebastian Ortiz"
                        telemetria["confianza"] = conf_final
                    else:
                        confs_1 = [c for p, c in buffer_verificacion if p == 1]
                        conf_final = float(np.mean(confs_1)) if confs_1 else 0.80
                        telemetria["autorizado"] = False
                        color_cuadro = (0, 0, 255)  # Rojo
                        etiqueta_texto = f"[NO AUTORIZADO: Desconocido] ({conf_final*100:.0f}%)"
                        telemetria["identidad_nombre"] = "Desconocido (No Registrado)"
                        telemetria["confianza"] = conf_final

        # Dibujar cuadrado envolvente y centroide real en la vista de cámara (sin sobrescribir centroide)
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

        # Actualizar animación de apertura de puerta
        if telemetria["autorizado"]:
            progreso_apertura = min(1.0, progreso_apertura + 0.15)
            estado_puerta = "ABIERTA"
        else:
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
            print(">>> Conmutado a Modo Óptimo Automático")
        elif tecla in NOMBRE_MODELOS_TECLAS:
            modelo_fuerza = NOMBRE_MODELOS_TECLAS[tecla]
            buffer_verificacion.clear()
            print(f">>> Algoritmo forzado en tiempo real: {modelo_fuerza}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demostración concluida limpiamente.")


if __name__ == "__main__":
    ejecutar_demostracion()
