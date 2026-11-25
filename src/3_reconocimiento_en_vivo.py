"""
Módulo 3: Reconocimiento Facial Multi-Rostro Jerárquico en Tiempo Real y Simulación del Actuador
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Docente: Ing. Rubén Nogales
Universidad Técnica de Ambato - FISEI - Software
"""

import os
import sys
import time
import copy
import threading
from collections import deque, Counter
import joblib
import cv2
import numpy as np

# Importar funciones canónicas de PDI
from utils_pdi import (
    calcular_cuadrado_envolvente,
    calcular_iou,
    aplicar_nms_cajas,
    proponer_ventanas_rostros_pdi,
    preprocesar_ventana_64x64,
    aplicar_mascara_eliptica_a_img64,
    obtener_rejilla_ventanas_geometrica,
    analizar_morfologia_ventana_otsu,
    verificar_estructura_facial_pdi,
    detectar_rostro_principal,
    ScalerPreservandoVarianza,
    extraer_descriptor_espacial_bloques,
    extraer_puntos_faciales_landmarks_pdi,
    centrar_ventana_en_rostro_pdi,
    extraer_mapas_convolucionales_2d,
    cv2_imwrite_unicode,
)
from iot_esp32_telegram import (
    cargar_config_iot,
    LectorCamaraThreaded,
    ClienteIoTESP32,
    NotificadorTelegram,
)

# Registrar en __main__ para deserialización sin fricción de artefactos serializados con joblib
setattr(sys.modules["__main__"], "ScalerPreservandoVarianza", ScalerPreservandoVarianza)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MODELOS = os.path.join(BASE_DIR, "modelos_entrenados")
DIR_CLASE_0 = os.path.join(BASE_DIR, "data", "dataset_procesado_64x64", "clase_0_no_persona")

MAX_ROSTROS_SIMULTANEOS = 5


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


def obtener_modelo_por_nombre(modelos_dict: dict, nombre_fuerza: str, fallback_info: dict):
    """Busca un modelo por nombre tolerando diferencias de tildes (ej. Arbol de decision)."""
    if nombre_fuerza in modelos_dict:
        return modelos_dict[nombre_fuerza]
    nombre_limpio = limpiar_texto_cv2(nombre_fuerza).lower()
    for k, v in modelos_dict.items():
        if limpiar_texto_cv2(k).lower() == nombre_limpio:
            return v
    return fallback_info


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

    # Desactivar paralelismo joblib en modelos basados en árboles/vecinos para inferencia de 1 muestra
    # (elimina ~8ms de overhead de creación de hilos en cada fotograma en vivo)
    for dic_m in (modelos_e1, modelos_e2):
        for info_m in dic_m.values():
            m_obj = info_m.get("modelo") if isinstance(info_m, dict) else None
            if m_obj is not None and hasattr(m_obj, "n_jobs"):
                try:
                    m_obj.n_jobs = 1
                except Exception:
                    pass

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
    Aplica la proyección PCA y concatenación de MSE + descriptores PDI biométricos + 12 Landmarks:
        f = [z_PCA, MSE, Sim_Adaptable, Ratio_HV, Descriptores_Bloques, Landmarks_12]
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
    landmarks_12, _ = extraer_puntos_faciales_landmarks_pdi(im)
    pdi_feats = np.concatenate([[sim, ratio_hv], desc_bloques, landmarks_12]).reshape(1, -1)

    f = np.hstack([z, mse, pdi_feats])
    if scaler is not None:
        f = scaler.transform(f)

    return f, float(mse[0, 0])


def construir_entrada_modelo(
    info_mod: dict,
    pca_model,
    vec_4096: np.ndarray,
    f_conv_cache=None,
    scaler_pca_fallback=None,
    f_pca_raw_cache=None,
):
    """
    Construye automáticamente el vector de entrada exacto que espera cualquier modelo
    (modelos clásicos PCA, CNN con fusión PCA+Conv2D, o Conv2D puro),
    verificando modelo.n_features_in_ para que jamás ocurra un ValueError de dimensiones.
    """
    modelo = info_mod["modelo"]
    scaler_pca = info_mod.get("scaler", None)
    if scaler_pca is None or (hasattr(scaler_pca, "n_features_in_") and scaler_pca.n_features_in_ == 2112):
        scaler_pca = scaler_pca_fallback

    scaler_cnn = info_mod.get("scaler_cnn", None)

    if f_pca_raw_cache is not None:
        f_raw, mse_val = f_pca_raw_cache
        f_pca = scaler_pca.transform(f_raw) if scaler_pca is not None else f_raw
    else:
        f_pca, mse_val = extraer_vector_caracteristicas_vivo(pca_model, vec_4096, scaler_pca)

    n_esperado = getattr(modelo, "n_features_in_", f_pca.shape[1])

    # Caso 1: Modelo clásico sobre PCA + PDI + Landmarks
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

    # Caso 2: Modelo CNN con Fusión Profunda [f_pca, f_conv_s] (compatible con BottleneckConvolucional2D y StandardScaler)
    if n_esperado == (f_pca.shape[1] + f_conv_s.shape[1]):
        escala_conv = 1.0 if getattr(scaler_cnn, "es_bottleneck", False) else 0.12
        f_total = np.hstack([f_pca, f_conv_s * escala_conv])
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


# =============================================================================
# GESTIÓN DE TRACKING MULTI-ROSTRO (BUFFER INDEPENDIENTE POR PERSONA)
# =============================================================================
class RostroTrack:
    """
    Representa un rostro humano individual seguido entre fotogramas consecutivos.
    Cada rostro mantiene su propia caja suavizada (EMA) y su propio buffer de
    verificación temporal para evitar que votos de desconocidos contaminen a
    una persona autorizada (y viceversa).
    """

    def __init__(self, track_id: int, caja: list, buffer_len: int = 10):
        self.track_id = track_id
        self.caja = [float(v) for v in caja]  # [x, y, lado, lado]
        self.buffer_verificacion = deque(maxlen=buffer_len)
        self.frames_sin_ver = 0
        self.frames_vistos = 1
        self.t_creacion = time.time()

        self.mse_e1 = 0.0
        self.mse_e2 = 0.0
        self.identidad_id = None
        self.identidad_nombre = "Analizando..."
        self.confianza = 0.0
        self.progreso_verificacion = 0.0
        self.autorizado = False
        self.etiqueta_texto = "VERIFICANDO..."
        self.color_cuadro = (0, 220, 255)
        self.landmarks_px = None

        self.ultimo_v_id_vivo = None
        self.ultimo_vec_4096 = None
        self.ultimo_img_64 = None

    def caja_int(self, W: int = None, H: int = None) -> list:
        qlado = max(32, int(round(self.caja[2])))
        qx = int(round(self.caja[0]))
        qy = int(round(self.caja[1]))
        if W is not None and H is not None:
            qlado = min(qlado, min(W, H))
            qx = int(np.clip(qx, 0, max(0, W - qlado)))
            qy = int(np.clip(qy, 0, max(0, H - qlado)))
        return [qx, qy, qlado, qlado]

    def actualizar_caja(self, nueva_caja: list, W: int, H: int, alpha: float = 0.45):
        nx, ny, nl, _ = [float(v) for v in nueva_caja]
        self.caja[0] = alpha * nx + (1.0 - alpha) * self.caja[0]
        self.caja[1] = alpha * ny + (1.0 - alpha) * self.caja[1]
        self.caja[2] = alpha * nl + (1.0 - alpha) * self.caja[2]
        self.caja[3] = self.caja[2]
        # Mantener dentro de los bordes del fotograma
        ql = min(self.caja[2], float(min(W, H)))
        self.caja[2] = ql
        self.caja[3] = ql
        self.caja[0] = float(np.clip(self.caja[0], 0.0, max(0.0, W - ql)))
        self.caja[1] = float(np.clip(self.caja[1], 0.0, max(0.0, H - ql)))

    def actualizar_decision_etapa2(self, res_e2: dict, ids_autorizados: list, id_desconocido: int, nombres_e2: dict):
        self.mse_e2 = res_e2["mse_e2"]
        self.ultimo_v_id_vivo = res_e2["v_id_vivo"]
        self.ultimo_vec_4096 = res_e2["vec_4096"]
        self.ultimo_img_64 = res_e2["img_64"]
        self.landmarks_px = res_e2["landmarks_px"]

        self.buffer_verificacion.append((res_e2["pred_id_frame"], res_e2["conf_frame"]))

        # Verificación biométrica rigurosa basada en tiempo real + consenso de votos:
        # Durante los primeros 1.8 segundos (mientras el usuario se acomoda y el auto-exposición
        # AEC del sensor OV2640 se estabiliza) SIEMPRE permanece en estado amarillo VERIFICANDO...
        # con barra de progreso suave. Solo después de ese periodo emite veredicto firme:
        #   - AUTORIZADO (verde): >= 4 votos consistentes del mismo usuario autorizado.
        #   - DESCONOCIDO (rojo): >= 6 votos de Desconocido y >= 1.8 s transcurridos.
        VOTOS_CONFIRMACION_AUT = 4
        VOTOS_CONFIRMACION_DESC = 6
        TIEMPO_MIN_VERIFICACION_SEG = 1.80
        conteo = Counter(p for p, _ in self.buffer_verificacion if p is not None)
        edad_track = max(0.0, time.time() - getattr(self, "t_creacion", 0.0))

        votos_aut_max = 0
        id_aut_mejor = None
        for cid_a in ids_autorizados:
            va = conteo.get(cid_a, 0)
            if va > votos_aut_max:
                votos_aut_max = va
                id_aut_mejor = cid_a

        if (not self.autorizado) and (edad_track < TIEMPO_MIN_VERIFICACION_SEG or len(self.buffer_verificacion) < 5):
            # Si ya alcanzó 4 votos claros de usuario autorizado después de 1.0s, puede autorizar sin esperar de más
            if id_aut_mejor is not None and votos_aut_max >= VOTOS_CONFIRMACION_AUT and edad_track >= 1.0:
                pass
            else:
                prog_t = edad_track / TIEMPO_MIN_VERIFICACION_SEG
                prog_v = float(len(self.buffer_verificacion)) / 5.0
                prog = float(np.clip(min(0.95, 0.55 * prog_t + 0.45 * prog_v), 0.12, 0.95))
                self.progreso_verificacion = prog
                self.autorizado = False
                self.identidad_id = None
                self.identidad_nombre = "Verificando..."
                self.confianza = prog
                self.color_cuadro = (0, 220, 255)
                self.etiqueta_texto = f"VERIFICANDO... [{int(prog * 100)}%]"
                return

        if not conteo:
            self.progreso_verificacion = 0.0
            self.autorizado = False
            self.identidad_id = id_desconocido
            self.identidad_nombre = "Desconocido"
            self.confianza = 0.75
            self.color_cuadro = (0, 0, 255)
            self.etiqueta_texto = "[NO AUTORIZADO: Desconocido]"
            return

        id_ganador, num_votos = conteo.most_common(1)[0]
        confs_ganador = [c for p, c in self.buffer_verificacion if p == id_ganador]
        conf_final = float(np.mean(confs_ganador)) if confs_ganador else 0.85

        # Histéresis biométrica en track ya verificado: evita que un parpadeo, micro-movimiento o
        # destello de luz de 2-3 frames haga oscilar "Autorizado -> Desconocido -> Autorizado".
        # Mientras conserve al menos 2 votos del usuario en la ventana de 8 (y no exista mayoría
        # abrumadora >=6 de Desconocido o de otro usuario), mantiene la autorización estable.
        if (
            self.autorizado
            and self.identidad_id in ids_autorizados
            and conteo.get(self.identidad_id, 0) >= 2
            and conteo.get(id_desconocido, 0) < 6
        ):
            id_mant = self.identidad_id
            confs_m = [c for p, c in self.buffer_verificacion if p == id_mant]
            conf_m = float(np.mean(confs_m)) if confs_m else self.confianza
            nombre_persona = nombres_e2.get(id_mant, f"ID {id_mant}")
            self.progreso_verificacion = 1.0
            self.autorizado = True
            self.identidad_nombre = nombre_persona
            self.confianza = conf_m
            self.color_cuadro = (0, 255, 0)
            self.etiqueta_texto = f"{nombre_persona} ({conf_m * 100:.0f}%) [AUTORIZADO]"
            return

        votos_aut_max = 0
        id_aut_mejor = None
        for cid_a in ids_autorizados:
            va = conteo.get(cid_a, 0)
            if va > votos_aut_max:
                votos_aut_max = va
                id_aut_mejor = cid_a

        if (
            id_ganador in ids_autorizados
            and (
                num_votos >= VOTOS_CONFIRMACION_AUT
                or (num_votos >= 3 and edad_track >= TIEMPO_MIN_VERIFICACION_SEG and conteo.get(id_desconocido, 0) < 4)
            )
        ):
            nombre_persona = nombres_e2.get(id_ganador, f"ID {id_ganador}")
            self.progreso_verificacion = 1.0
            self.autorizado = True
            self.identidad_id = id_ganador
            self.identidad_nombre = nombre_persona
            self.confianza = conf_final
            self.color_cuadro = (0, 255, 0)
            self.etiqueta_texto = f"{nombre_persona} ({conf_final * 100:.0f}%) [AUTORIZADO]"
        elif id_aut_mejor is not None and votos_aut_max >= 2 and conteo.get(id_desconocido, 0) < VOTOS_CONFIRMACION_DESC:
            # Evidencia parcial de usuario autorizado (ej. mientras el recuadro se termina de ajustar):
            # mantener en estado amarillo VERIFICANDO sin etiquetar falsamente como Desconocido.
            nombre_persona = nombres_e2.get(id_aut_mejor, f"ID {id_aut_mejor}")
            prog = min(0.95, float(votos_aut_max) / float(VOTOS_CONFIRMACION_AUT))
            self.progreso_verificacion = prog
            self.autorizado = False
            self.identidad_id = None
            self.identidad_nombre = f"Verificando {nombre_persona}..."
            self.confianza = conf_final
            self.color_cuadro = (0, 220, 255)
            self.etiqueta_texto = f"VERIFICANDO... [{int(prog * 100)}%]"
        elif id_ganador == id_desconocido and num_votos >= VOTOS_CONFIRMACION_DESC:
            self.progreso_verificacion = 0.0
            self.autorizado = False
            self.identidad_id = id_desconocido
            self.identidad_nombre = "Desconocido"
            self.confianza = conf_final
            self.color_cuadro = (0, 0, 255)
            self.etiqueta_texto = f"[NO AUTORIZADO: Desconocido] ({conf_final * 100:.0f}%)"
        else:
            prog = min(0.92, float(num_votos) / float(VOTOS_CONFIRMACION_DESC))
            self.progreso_verificacion = prog
            self.autorizado = False
            self.identidad_id = None
            self.identidad_nombre = "Verificando..."
            self.confianza = prog
            self.color_cuadro = (0, 220, 255)
            self.etiqueta_texto = f"VERIFICANDO... [{int(prog * 100)}%]"


class GestorTracksFaciales:
    """
    Asocia detecciones multi-rostro entre fotogramas usando IoU y proximidad de centroides.
    Mantiene hasta max_rostros tracks simultáneos con expiración limpia cuando salen de escena.
    """

    def __init__(self, max_rostros: int = MAX_ROSTROS_SIMULTANEOS, max_frames_perdidos: int = 4):
        self.tracks = []
        self.next_id = 1
        self.max_rostros = max_rostros
        self.max_frames_perdidos = max_frames_perdidos

    def limpiar_buffers(self):
        for t in self.tracks:
            t.buffer_verificacion.clear()
            t.autorizado = False
            t.identidad_id = None
            t.etiqueta_texto = "RE-EVALUANDO..."
            t.color_cuadro = (0, 220, 255)

    def asociar_detecciones(self, detecciones: list, W: int, H: int) -> list:
        """
        Empareja tracks existentes con detecciones actuales.
        Retorna una lista de tuplas (track, det) correspondientes a rostros activos en este frame.
        """
        emparejados = []
        tracks_usados = set()
        dets_usadas = set()

        if self.tracks and detecciones:
            pares_candidatos = []
            for ti, tr in enumerate(self.tracks):
                ct = tr.caja_int(W, H)
                cx_t = ct[0] + ct[2] * 0.5
                cy_t = ct[1] + ct[3] * 0.5
                for di, det in enumerate(detecciones):
                    cd = det["caja"]
                    cx_d = cd[0] + cd[2] * 0.5
                    cy_d = cd[1] + cd[3] * 0.5
                    iou = calcular_iou(ct, cd)
                    lado_ref = max(float(ct[2]), float(cd[2]), 1.0)
                    dist_norm = float(np.hypot(cx_t - cx_d, cy_t - cy_d)) / (0.65 * lado_ref)
                    if iou >= 0.16 or dist_norm <= 0.90:
                        afinidad = iou + max(0.0, 1.0 - dist_norm)
                        pares_candidatos.append((afinidad, ti, di))

            pares_candidatos.sort(key=lambda x: x[0], reverse=True)
            for _, ti, di in pares_candidatos:
                if ti in tracks_usados or di in dets_usadas:
                    continue
                tracks_usados.add(ti)
                dets_usadas.add(di)
                tr = self.tracks[ti]
                det = detecciones[di]
                tr.actualizar_caja(det["caja"], W, H)
                tr.frames_sin_ver = 0
                tr.frames_vistos += 1
                tr.mse_e1 = det["mse_e1"]
                emparejados.append((tr, det))

        # Envejecer tracks no vistos en este fotograma y podar los expirados.
        # Si un track deja de ser detectado en Etapa 1 durante 2 frames seguidos (ej. la persona
        # salió del cuadro o giró la cámara al techo), expirarlo de inmediato para nunca dejar
        # cuadros "fantasma" flotando sobre paredes/techos vacíos.
        tracks_sobrevivientes = []
        for ti, tr in enumerate(self.tracks):
            if ti not in tracks_usados:
                tr.frames_sin_ver += 1
                if tr.frames_sin_ver > self.max_frames_perdidos:
                    continue
            tracks_sobrevivientes.append(tr)
        self.tracks = tracks_sobrevivientes

        # Crear nuevos tracks para detecciones no emparejadas (sin superar max_rostros ni duplicar)
        for di, det in enumerate(detecciones):
            if di in dets_usadas:
                continue
            if len(self.tracks) >= self.max_rostros:
                break
            cd = det["caja"]
            duplicado = any(calcular_iou(tr.caja_int(W, H), cd) >= 0.35 for tr in self.tracks)
            if duplicado:
                continue
            nuevo = RostroTrack(self.next_id, cd)
            nuevo.mse_e1 = det["mse_e1"]
            self.next_id += 1
            self.tracks.append(nuevo)
            emparejados.append((nuevo, det))

        return emparejados

    def obtener_tracks_visibles(self) -> list:
        """Retorna únicamente tracks presentes en escena (frames_sin_ver <= 1) con NMS anti sub-cajas."""
        visibles = [t for t in self.tracks if t.frames_sin_ver <= 1]
        if len(visibles) <= 1:
            return visibles
        cajas_v = [t.caja_int() for t in visibles]
        puntajes_v = [
            (1000.0 if t.autorizado else 0.0) + float(t.frames_vistos) * 5.0 + float(t.caja[2])
            for t in visibles
        ]
        keep_idx = aplicar_nms_cajas(cajas_v, puntajes_v, umbral_iou=0.22, umbral_contencion=0.40)
        filtrados = [visibles[i] for i in keep_idx]
        filtrados.sort(key=lambda t: (not t.autorizado, -t.caja[2]))
        return filtrados


# =============================================================================
# PIPELINE MULTI-ROSTRO: ETAPA 1 (BATCH + NMS), ETAPA 2 Y COMPUERTA LÓGICA OR
# =============================================================================
def detectar_rostros_etapa1(
    frame: np.ndarray,
    pca_e1,
    info_e1_activo: dict,
    scaler_pca_respaldo_e1,
    umbral_mse_e1: float,
    tracks_previos: list,
    gris_fondo_completo,
    lista_negativos_vivo: list,
    pca_e2,
    max_rostros: int = MAX_ROSTROS_SIMULTANEOS,
    modelos_e1: dict = None,
) -> list:
    """
    Etapa 1 Multi-Rostro Blindada Anti-Objetos:
    1. Genera ventanas candidatas con PDI puro (sin rejilla ciega sobre camas/paredes si ya hay piel).
    2. Filtra fondo estático [B] y estructura antropométrica facial (verificar_estructura_facial_pdi).
    3. Proyecta en lote sobre pca_e1 y exige MSE facial estricto (<= 0.00180: bloquea cobijas, paredes y ropa).
    4. Verifica consenso supervisado de Etapa 1 (SVM + CNN + RF) para garantizar que SOLO pasen rostros humanos reales.
    5. Aplica NMS (aplicar_nms_cajas) priorizando encuadre facial completo de menor residuo MSE.
    """
    H, W = frame.shape[:2]
    # Incluir ventanas de respaldo multi-altura (izquierda/centro/derecha) para detectar simultáneamente
    # a todas las personas en escena sin importar su altura o si hay lentes/fondo beige.
    propuestas = proponer_ventanas_rostros_pdi(frame, max_candidatos=32, incluir_rejilla_respaldo=True)

    # Reinyectar posiciones activas previas para continuidad fluida del tracking
    for tr in tracks_previos:
        ct = tr.caja_int(W, H)
        if not any(calcular_iou(ct, p) >= 0.80 for p in propuestas):
            propuestas.insert(0, ct)

    if not propuestas:
        return []

    gris_frame_actual = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gris_fondo_completo is not None else None

    candidatos_pdi = []
    for caja in propuestas:
        qx, qy, qlado, _ = caja
        parche = frame[qy : qy + qlado, qx : qx + qlado]
        if parche.shape[0] < 32 or parche.shape[1] < 32:
            continue

        # Validación de fondo de escena completo (Tecla [B])
        if gris_fondo_completo is not None and gris_frame_actual is not None:
            rec_act = gris_frame_actual[qy : qy + qlado, qx : qx + qlado]
            rec_fnd = gris_fondo_completo[qy : qy + qlado, qx : qx + qlado]
            if rec_act.shape == rec_fnd.shape:
                diff_f = float(np.mean(np.abs(rec_act.astype(np.float64) - rec_fnd.astype(np.float64)))) / 255.0
                if diff_f < 0.055:
                    continue

        img_64, vec_4096 = preprocesar_ventana_64x64(parche)
        ok_pdi, met_pdi = verificar_estructura_facial_pdi(parche, img_64)
        if not ok_pdi:
            continue

        candidatos_pdi.append(
            {
                "caja": [qx, qy, qlado, qlado],
                "parche": parche,
                "img_64": img_64,
                "vec_4096": vec_4096,
                "sim_bilateral": float(met_pdi.get("sim_bilateral", 0.0)),
            }
        )

    if not candidatos_pdi:
        return []

    # Proyección PCA en lote para calcular MSE de Etapa 1 a máxima velocidad
    X_lote_e1 = np.array([c["vec_4096"] for c in candidatos_pdi], dtype=np.float64)
    Z_lote_e1 = pca_e1.transform(X_lote_e1)
    X_hat_e1 = pca_e1.inverse_transform(Z_lote_e1)
    mse_lote_e1 = np.mean((X_lote_e1 - X_hat_e1) ** 2, axis=1)

    # Umbral MSE adaptativo: rostros sin gafas tienen MSE ~0.0003-0.0018; rostros con lentes oscuros/montura
    # gruesa alcanzan MSE ~0.0020-0.0030 (permitidos SOLO con triple consenso estricto SVM+CNN+RF más abajo).
    umbral_mse_estricto = min(float(umbral_mse_e1), 0.00305)
    info_cnn_e1 = modelos_e1.get("CNN (Conv2D)") if isinstance(modelos_e1, dict) else None
    info_rf_e1 = modelos_e1.get("Random Forest") if isinstance(modelos_e1, dict) else None

    aprobados = []
    cajas_ok = []
    scores_ok = []

    # Evaluar todos los candidatos con MSE facial válido priorizando encuadres craneofaciales
    # completos (evita que recortes diminutos de piel pura desplacen al rostro completo con cabello).
    min_dim_f = float(max(1, min(H, W)))
    indices_validos = [
        i for i in range(len(candidatos_pdi)) if float(mse_lote_e1[i]) <= umbral_mse_estricto
    ]
    indices_validos.sort(
        key=lambda i: float(mse_lote_e1[i]) - 0.00140 * (float(candidatos_pdi[i]["caja"][2]) / min_dim_f)
    )

    for idx in indices_validos[:24]:
        cand = candidatos_pdi[idx]
        mse_val = float(mse_lote_e1[idx])

        # Validación contra objetos bloqueados en vivo (Tecla [N])
        if lista_negativos_vivo:
            z_neg_cand = pca_e2.transform(cand["vec_4096"].reshape(1, -1))[0]
            dist_min_neg = min(float(np.linalg.norm(z_neg_cand - zn)) for zn in lista_negativos_vivo)
            if dist_min_neg < 4.5:
                continue

        # Extraer vector crudo PCA+PDI+Landmarks UNA sola vez y reutilizarlo en SVM, CNN y RF
        f_pca_raw_e1 = extraer_vector_caracteristicas_vivo(pca_e1, cand["vec_4096"], scaler=None)
        f_conv_cache_e1 = None

        # Clasificador supervisado principal de Etapa 1
        f_in_e1, _, f_conv_cache_e1 = construir_entrada_modelo(
            info_e1_activo,
            pca_e1,
            cand["vec_4096"],
            f_conv_cache=f_conv_cache_e1,
            scaler_pca_fallback=scaler_pca_respaldo_e1,
            f_pca_raw_cache=f_pca_raw_e1,
        )
        probas_e1 = obtener_vector_probabilidades(info_e1_activo["modelo"], f_in_e1, 2)
        prob_humano_e1 = float(probas_e1[1])

        if prob_humano_e1 < 0.52:
            continue

        # Doble verificación anti-falsos positivos (CNN + Random Forest de Etapa 1):
        # Bloquea paredes/sombras/objetos que logren engañar a un solo modelo.
        prob_ens_e1 = prob_humano_e1
        p_cnn_e1 = prob_humano_e1
        if info_cnn_e1 is not None and info_cnn_e1 is not info_e1_activo:
            try:
                f_cnn_e1, _, f_conv_cache_e1 = construir_entrada_modelo(
                    info_cnn_e1, pca_e1, cand["vec_4096"],
                    f_conv_cache=f_conv_cache_e1, scaler_pca_fallback=scaler_pca_respaldo_e1,
                    f_pca_raw_cache=f_pca_raw_e1,
                )
                p_cnn_e1 = float(obtener_vector_probabilidades(info_cnn_e1["modelo"], f_cnn_e1, 2)[1])
                if p_cnn_e1 < 0.50:
                    continue
                prob_ens_e1 = 0.55 * prob_humano_e1 + 0.45 * p_cnn_e1
            except Exception:
                pass

        # Caso especial rostro humano lejano (pequeño): exige que no sea un micro-borde (<86px) de mueble
        # y que PCA + CNN + SVM coincidan sólidamente.
        es_tamano_rostro_seguro = (cand["caja"][2] >= 86)
        rostro_lejano_limpio = (
            (es_tamano_rostro_seguro and mse_val <= 0.00090 and p_cnn_e1 >= 0.70 and prob_humano_e1 >= 0.56)
            or (es_tamano_rostro_seguro and mse_val <= 0.00145 and p_cnn_e1 >= 0.80 and prob_humano_e1 >= 0.54)
            or (es_tamano_rostro_seguro and mse_val <= 0.00240 and p_cnn_e1 >= 0.90 and prob_humano_e1 >= 0.75)
        )

        # Si MSE > 0.00180 (lentes oscuros / montura gruesa / sombra fuerte), exigir consenso sólido de SVM+CNN
        # y, si MSE > 0.00240, exigir además triple consenso con Random Forest (los objetos jamás pasan esto).
        if mse_val > 0.00240 and (prob_humano_e1 < 0.78 or p_cnn_e1 < 0.85):
            continue
        if mse_val > 0.00180 and (prob_humano_e1 < 0.64 or p_cnn_e1 < 0.72):
            continue

        if not rostro_lejano_limpio and prob_humano_e1 < 0.60:
            continue

        if info_rf_e1 is not None and info_rf_e1 is not info_e1_activo:
            try:
                f_rf_e1, _, f_conv_cache_e1 = construir_entrada_modelo(
                    info_rf_e1, pca_e1, cand["vec_4096"],
                    f_conv_cache=f_conv_cache_e1, scaler_pca_fallback=scaler_pca_respaldo_e1,
                    f_pca_raw_cache=f_pca_raw_e1,
                )
                p_rf_e1 = float(obtener_vector_probabilidades(info_rf_e1["modelo"], f_rf_e1, 2)[1])
                umbral_rf_dyn = 0.60 if mse_val > 0.00240 else 0.52
                if not rostro_lejano_limpio and p_rf_e1 < umbral_rf_dyn:
                    continue
                prob_ens_e1 = 0.75 * prob_ens_e1 + 0.25 * p_rf_e1
            except Exception:
                pass

        # Puntuación que prioriza encuadre craneofacial completo sobre recortes parciales (ej. solo ojos/nariz)
        rel_lado = float(cand["caja"][2]) / float(max(1, min(H, W)))
        score_e1 = 2.8 * prob_ens_e1 + 1.55 * rel_lado + 0.6 * cand["sim_bilateral"] - 180.0 * mse_val
        cand["mse_e1"] = mse_val
        cand["score_e1"] = score_e1
        aprobados.append(cand)
        cajas_ok.append(cand["caja"])
        scores_ok.append(score_e1)

    if not aprobados:
        return []

    # Supresión de No-Máximos (NMS) estricta para obtener exactamente 1 caja limpia por cada rostro humano
    indices_nms = aplicar_nms_cajas(cajas_ok, scores_ok, umbral_iou=0.20, umbral_contencion=0.38)
    detecciones_finales = [aprobados[i] for i in indices_nms[:max_rostros]]
    return detecciones_finales


def evaluar_identidad_etapa2(
    frame: np.ndarray,
    caja_rostro: list,
    pca_e2,
    modelos_e2: dict,
    modelo_cnn,
    artefactos: dict,
    modelo_fuerza,
    nombres_e2: dict,
    ids_autorizados: list,
    id_desconocido: int,
    perfiles: dict,
    umbral_mse_id: float,
    modo_compacto: bool = False,
) -> dict:
    """
    Etapa 2 Individual por Rostro:
    1. Pirámide de centrado por minimización de residuo ortogonal en pca_e2 (evita deformación
       de visitantes/impostores).
    2. Extracción del vector biométrico completo (PCA + MSE + Bloques + 12 Landmarks).
    3. Consenso de modelos ML (o modelo forzado por teclado).
    4. Verificación biométrica estricta en F_Identity (f_pca_e2[0, 2:], libre de luz global).
    5. Mapeo exacto de los 5 Landmarks faciales sobre la ventana ganadora.
    """
    H, W = frame.shape[:2]
    qx, qy, qlado, _ = caja_rostro
    cx_base = qx + qlado // 2
    cy_base = qy + qlado // 2

    # En el 1er frame de vida del track o con 1 rostro, usar pirámide equilibrada alrededor de la caja
    # detectada en Etapa 1 (0.72 a 1.12), evitando tanto el colapso (<0.70) como el exceso de fondo.
    if modo_compacto:
        fracs_xy = [(0.0, 0.0), (-0.04, 0.0), (0.04, 0.0), (0.0, -0.05), (0.0, 0.06)]
        escalas = [0.74, 0.86, 1.00, 1.12]
    else:
        fracs_xy = [
            (0.0, 0.0), (-0.04, 0.0), (0.04, 0.0),
            (0.0, -0.05), (0.0, 0.06),
        ]
        escalas = [0.72, 0.85, 1.00, 1.12]

    candidatos_lote = []
    coords_lote = []
    for fx_p, fy_p in fracs_xy:
        dx_p = int(round(fx_p * qlado))
        dy_p = int(round(fy_p * qlado))
        for sc in escalas:
            lado_s = int(round(qlado * sc))
            lado_s = max(32, min(lado_s, min(W, H)))
            c_x = cx_base + dx_p
            c_y = cy_base + dy_p
            x1 = int(np.clip(c_x - lado_s // 2, 0, max(0, W - lado_s)))
            y1 = int(np.clip(c_y - lado_s // 2, 0, max(0, H - lado_s)))
            parche_s = frame[y1 : y1 + lado_s, x1 : x1 + lado_s]
            if parche_s.shape[0] >= 32 and parche_s.shape[1] >= 32:
                im_s, v_s = preprocesar_ventana_64x64(parche_s)
                if float(np.std(im_s)) > 0.04:
                    candidatos_lote.append(v_s)
                    coords_lote.append((x1, y1, lado_s))

    if not candidatos_lote:
        parche_def = frame[qy : qy + qlado, qx : qx + qlado]
        _, v_def = preprocesar_ventana_64x64(parche_def)
        candidatos_lote.append(v_def)
        coords_lote.append((qx, qy, qlado))

    scaler_pca_respaldo_e2 = modelos_e2.get("SVM", {}).get("scaler", None)
    scale_pca_val = getattr(scaler_pca_respaldo_e2, "scale_pca", 1.0) if scaler_pca_respaldo_e2 is not None else 1.0

    X_lote = np.array(candidatos_lote, dtype=np.float64)
    Z_lote = pca_e2.transform(X_lote)
    X_hat_lote = pca_e2.inverse_transform(Z_lote)
    mse_lote = np.mean((X_lote - X_hat_lote) ** 2, axis=1)

    # Selección de encuadre óptimo combinando residuo ortogonal (MSE) y alineación en subespacio de identidad (Z[:, 2:])
    K_e2 = pca_e2.n_components_
    Z_id_lote = Z_lote[:, 2:] / max(1e-6, float(scale_pca_val))
    protos_pca = [
        perfiles[c]["matriz_Z_poses"][:, : K_e2 - 2]
        for c in ids_autorizados
        if (c in perfiles and perfiles[c].get("matriz_Z_poses") is not None and perfiles[c]["matriz_Z_poses"].shape[1] >= K_e2 - 2)
    ]
    if protos_pca:
        mejor_score_aln = -1e9
        best_idx = int(np.argmin(mse_lote))
        for idx_c in range(len(X_lote)):
            if mse_lote[idx_c] > 0.0042:
                continue
            z_cand = Z_id_lote[idx_c]
            d_min_proto = min(
                float(np.mean(np.sort(np.linalg.norm(mp - z_cand, axis=1))[:5]))
                for mp in protos_pca
            )
            score_c = -d_min_proto - 120.0 * float(mse_lote[idx_c])
            if score_c > mejor_score_aln:
                mejor_score_aln = score_c
                best_idx = idx_c
    else:
        best_idx = int(np.argmin(mse_lote))

    vec_4096 = X_lote[best_idx]
    img_64 = vec_4096.reshape(64, 64)
    best_x1, best_y1, best_lado = coords_lote[best_idx]

    # Extraer vector crudo una sola vez para acelerar el ensamble completo de modelos
    f_pca_raw, mse_e2 = extraer_vector_caracteristicas_vivo(pca_e2, vec_4096, scaler=None)
    f_pca_raw_cache = (f_pca_raw, mse_e2)
    f_pca_e2 = scaler_pca_respaldo_e2.transform(f_pca_raw) if scaler_pca_respaldo_e2 is not None else f_pca_raw

    # Vector de identidad libre de las 2 primeras componentes de iluminación global del cuarto
    v_id_vivo = f_pca_e2[0, 2:]

    num_clases_totales = max(nombres_e2.keys()) + 1
    f_conv_cache = None

    probas_svm = None
    probas_ann = None
    probas_cnn = None
    probas_knn = None
    if modelo_fuerza is None:
        # Consenso ponderado completo de los 5 modelos supervisados (SVM + ANN + k-NN + RF + CNN)
        # k-NN en F_Identity tiene peso alto (0.28) por su excelente rechazo Open-Set frente a familiares/desconocidos
        pesos_modelos = [
            ("SVM", 0.24),
            ("ANN (MLP)", 0.16),
            ("k-NN", 0.28),
            ("Random Forest", 0.14),
            ("CNN (Conv2D)", 0.18),
        ]
        probas_acum = np.zeros(num_clases_totales, dtype=np.float64)
        peso_total = 0.0

        for nom_cand, peso_m in pesos_modelos:
            cand_info = None
            if nom_cand in modelos_e2:
                cand_info = modelos_e2[nom_cand]
            elif nom_cand == "CNN (Conv2D)" and modelo_cnn is not None:
                cand_info = modelo_cnn

            if cand_info is not None:
                try:
                    f_in_cand, _, f_conv_cache = construir_entrada_modelo(
                        cand_info,
                        pca_e2,
                        vec_4096,
                        f_conv_cache=f_conv_cache,
                        scaler_pca_fallback=scaler_pca_respaldo_e2,
                        f_pca_raw_cache=f_pca_raw_cache,
                    )
                    v_p = obtener_vector_probabilidades(cand_info["modelo"], f_in_cand, num_clases_totales)
                    if nom_cand == "SVM":
                        probas_svm = v_p
                    elif nom_cand == "ANN (MLP)":
                        probas_ann = v_p
                    elif nom_cand == "CNN (Conv2D)":
                        probas_cnn = v_p
                    elif nom_cand == "k-NN":
                        probas_knn = v_p
                    probas_acum += peso_m * v_p
                    peso_total += peso_m
                except Exception:
                    pass

        if peso_total > 0.0:
            probas_vector = probas_acum / peso_total
        else:
            f_in_def, _, _ = construir_entrada_modelo(
                artefactos["mejor_e2_info"],
                pca_e2,
                vec_4096,
                f_conv_cache=f_conv_cache,
                scaler_pca_fallback=scaler_pca_respaldo_e2,
                f_pca_raw_cache=f_pca_raw_cache,
            )
            pred_def = int(artefactos["mejor_e2_info"]["modelo"].predict(f_in_def)[0])
            probas_vector = np.zeros(num_clases_totales, dtype=np.float64)
            probas_vector[pred_def] = 0.85
    else:
        if modelo_fuerza in ["CNN (Deep Learning)", "CNN (Conv2D)"]:
            info_e2_activo = modelo_cnn if modelo_cnn is not None else modelos_e2.get("CNN (Conv2D)", artefactos["mejor_e2_info"])
        else:
            info_e2_activo = obtener_modelo_por_nombre(modelos_e2, modelo_fuerza, artefactos["mejor_e2_info"])

        f_in_e2, _, f_conv_cache = construir_entrada_modelo(
            info_e2_activo,
            pca_e2,
            vec_4096,
            f_conv_cache=f_conv_cache,
            scaler_pca_fallback=scaler_pca_respaldo_e2,
            f_pca_raw_cache=f_pca_raw_cache,
        )
        probas_vector = obtener_vector_probabilidades(info_e2_activo["modelo"], f_in_e2, num_clases_totales)

    # Verificación por Consenso Real del Ensamble ML + DL (SVM + ANN + k-NN + RF + CNN) + MSE PCA:
    # Principio científico limpio (sin parches manuales ni umbrales ad-hoc de coseno que anulen al modelo):
    #   1. Se elige la clase autorizada con mayor probabilidad del ensamble supervisado ML+DL.
    #   2. Un rostro pertenece a esa clase autorizada si y solo si existe acuerdo real entre:
    #        - El modelo neuronal convolucional 2D (CNN) y los clasificadores clásicos (SVM, ANN, RF)
    #        - La vecindad local en el subespacio Eigenfaces (k-NN) o la dirección angular (cos_id)
    #        - El residuo de reconstrucción PCA (MSE dentro del subespacio facial válido).
    #   3. Si los modelos supervisados discrepan (ej. SVM/CNN dicen una clase pero k-NN local dice otra,
    #      o viceversa), se clasifica de forma segura como Desconocido (Open-Set).
    mejor_c_aut = None
    mejor_prob_ens = -1.0
    mejor_ratio_d = 999.0
    mejor_cos_id = 0.0
    mejor_umbral_mse = umbral_mse_id
    prob_c_ganador = 0.0
    knn_prob_ganador = 1.0
    v_id_norm = v_id_vivo / (float(np.linalg.norm(v_id_vivo)) + 1e-8)

    cos_por_clase = {}
    for c_aut in ids_autorizados:
        if c_aut in perfiles:
            p_cand = perfiles[c_aut]
            mat_p = p_cand.get("matriz_Z_poses", None)
            rad_p = p_cand.get("radio_pose", 3.5)
            umb_m = p_cand.get("umbral_mse_clase", 0.0030)
            if mat_p is not None and len(mat_p) > 0 and mat_p.shape[1] == len(v_id_vivo):
                d_all = np.linalg.norm(mat_p - v_id_vivo, axis=1)
                d_geo = float(np.mean(np.sort(d_all)[:5]))
                mat_n = mat_p / (np.linalg.norm(mat_p, axis=1, keepdims=True) + 1e-8)
                cos_all = mat_n @ v_id_norm
                cos_id = float(np.mean(np.sort(cos_all)[-5:]))
            else:
                d_geo = float(np.linalg.norm(v_id_vivo - p_cand["centroide"]))
                cen = p_cand["centroide"]
                cos_id = float(np.dot(cen / (float(np.linalg.norm(cen)) + 1e-8), v_id_norm))
            cos_por_clase[c_aut] = cos_id
            r_d = d_geo / (rad_p + 1e-6)
            p_knn_c = float(probas_knn[c_aut]) if (probas_knn is not None and c_aut < len(probas_knn)) else 0.65
            prob_c = float(probas_vector[c_aut]) if c_aut < len(probas_vector) else 0.0
            if prob_c > mejor_prob_ens:
                mejor_prob_ens = prob_c
                mejor_c_aut = c_aut
                mejor_ratio_d = r_d
                mejor_cos_id = cos_id
                mejor_umbral_mse = umb_m
                prob_c_ganador = prob_c
                knn_prob_ganador = p_knn_c

    umbral_mse_vivo = min(max(float(mejor_umbral_mse), 0.0020), 0.0024)
    if modelo_fuerza is None:
        # Modo [0] Óptimo: Consenso genuino del Ensamble (ML clásico + Deep Learning CNN)
        p_svm_aut = float(probas_svm[mejor_c_aut]) if (probas_svm is not None and mejor_c_aut is not None and mejor_c_aut < len(probas_svm)) else prob_c_ganador
        p_ann_aut = float(probas_ann[mejor_c_aut]) if (probas_ann is not None and mejor_c_aut is not None and mejor_c_aut < len(probas_ann)) else prob_c_ganador
        p_cnn_aut = float(probas_cnn[mejor_c_aut]) if (probas_cnn is not None and mejor_c_aut is not None and mejor_c_aut < len(probas_cnn)) else prob_c_ganador
        p_knn_aut = knn_prob_ganador if probas_knn is not None else prob_c_ganador
        p_desc_ens = float(probas_vector[id_desconocido]) if id_desconocido < len(probas_vector) else 0.0

        # Coherencia angular Open-Set: la clase ganadora en el ensamble supervisado debe ser también
        # la dirección más afín en el subespacio Eigenfaces F_Identity (rechaza impostores/familiares
        # que engañan a SVM/CNN hacia una clase mientras su geometría angular apunta a otra).
        cos_mejor_otra = max((v for k, v in cos_por_clase.items() if k != mejor_c_aut), default=-1.0)
        es_ganador_angular = (mejor_cos_id >= cos_mejor_otra - 0.01)

        # 1) Acuerdo unánime fuerte de los 5 modelos (ML + DL + k-NN):
        acuerdo_ensamble_completo = (
            prob_c_ganador >= 0.48
            and p_cnn_aut >= 0.42
            and p_knn_aut >= 0.50
            and p_svm_aut >= 0.30
            and mejor_ratio_d <= 0.98
        )
        # 2) Consenso dominante SVM + ANN + CNN con coherencia angular (cuando el fondo blanco del aula
        #    desplaza la distancia L2 de k-NN pero los 3 clasificadores no lineales coinciden con >90-95%):
        acuerdo_svm_cnn_dominante = (
            prob_c_ganador >= 0.65
            and p_svm_aut >= 0.92
            and p_ann_aut >= 0.90
            and p_cnn_aut >= 0.95
            and p_desc_ens <= 0.12
            and mejor_ratio_d <= 0.88
            and mejor_cos_id >= 0.32
            and es_ganador_angular
        )
        # 3) Consenso k-NN + Coseno en subespacio facial (luz fría fluorescente de aula donde k-NN=100%
        #    y la dirección angular en F_Identity confirma inequívocamente al usuario autorizado):
        acuerdo_knn_coseno = (
            prob_c_ganador >= 0.43
            and p_knn_aut >= 0.85
            and p_cnn_aut >= 0.22
            and p_desc_ens <= 0.18
            and mejor_cos_id >= 0.43
            and mejor_ratio_d <= 0.92
            and es_ganador_angular
        )
        # 4) Acuerdo alternativo CNN + Coseno angular cuando la caja está levemente desplazada:
        acuerdo_cnn_geometrico = (
            prob_c_ganador >= 0.43
            and p_cnn_aut >= 0.65
            and p_svm_aut >= 0.25
            and mejor_cos_id >= 0.48
            and mejor_ratio_d <= 1.00
            and es_ganador_angular
        )
        es_autorizado_frame = (
            mejor_c_aut is not None
            and mse_e2 <= umbral_mse_vivo
            and (
                acuerdo_ensamble_completo
                or acuerdo_svm_cnn_dominante
                or acuerdo_knn_coseno
                or acuerdo_cnn_geometrico
            )
        )
    else:
        # Modo de evaluación individual por teclado [1]-[6]: decide directamente el modelo seleccionado
        # con verificación mínima del subespacio facial (MSE + radio).
        es_autorizado_frame = (
            mejor_c_aut is not None
            and prob_c_ganador >= 0.52
            and mse_e2 <= umbral_mse_vivo
            and mejor_ratio_d <= 0.96
        )

    if es_autorizado_frame:
        pred_id_frame = mejor_c_aut
        conf_frame = float(np.clip(0.88 + max(0.0, prob_c_ganador - 0.43) / 0.55 * 0.10, 0.89, 0.98))
    else:
        pred_id_frame = id_desconocido
        conf_frame = float(np.clip(0.75 + min(0.15, max(0.0, 0.50 - prob_c_ganador) * 0.35), 0.75, 0.90))

    # Extraer y mapear los 5 Landmarks faciales sobre las coordenadas de la ventana ganadora
    landmarks_px = None
    try:
        _, pts_landmarks = extraer_puntos_faciales_landmarks_pdi(img_64)
        landmarks_px = {
            k_lm: (int(best_x1 + lx * best_lado / 64.0), int(best_y1 + ly * best_lado / 64.0))
            for k_lm, (lx, ly) in pts_landmarks.items()
        }
    except Exception:
        landmarks_px = None

    return {
        "pred_id_frame": pred_id_frame,
        "conf_frame": conf_frame,
        "mse_e2": mse_e2,
        "v_id_vivo": v_id_vivo,
        "vec_4096": vec_4096,
        "img_64": img_64,
        "landmarks_px": landmarks_px,
        "caja_ajustada": [best_x1, best_y1, best_lado, best_lado],
    }


def evaluar_acceso_global_or(tracks_visibles: list, ids_autorizados: list, id_desconocido: int) -> dict:
    """
    Regla Lógica OR Multi-Rostro:
    Autoriza el acceso si y solo si EXISTE AL MENOS UN rostro humano confirmado
    cuya identidad pertenece a ids_autorizados.
    """
    autorizados = [t for t in tracks_visibles if t.autorizado and (t.identidad_id in ids_autorizados)]
    desconocidos = [t for t in tracks_visibles if (not t.autorizado and t.identidad_id == id_desconocido)]
    verificando = [t for t in tracks_visibles if (not t.autorizado and t.identidad_id != id_desconocido)]

    nombres_aut = sorted(set(t.identidad_nombre for t in autorizados))
    resumen = []
    for t in tracks_visibles:
        resumen.append(
            {
                "track_id": t.track_id,
                "nombre": t.identidad_nombre,
                "confianza": t.confianza,
                "autorizado": t.autorizado,
                "desconocido": (not t.autorizado and t.identidad_id == id_desconocido),
                "mse": t.mse_e2 if t.mse_e2 > 0 else t.mse_e1,
            }
        )

    return {
        "hay_autorizado": len(autorizados) >= 1,
        "total_rostros": len(tracks_visibles),
        "num_autorizados": len(autorizados),
        "num_desconocidos": len(desconocidos),
        "num_verificando": len(verificando),
        "nombres_autorizados": nombres_aut,
        "resumen_rostros": resumen,
    }


def dibujar_panel_actuador(
    panel_w: int,
    panel_h: int,
    estado_puerta: str,
    progreso_apertura: float,
    telemetria: dict,
):
    """
    Construye el panel lateral de telemetría multi-rostro y animación gráfica del actuador físico.
    Muestra el conteo de rostros, el resumen por persona y el estado global (Compuerta OR).
    """
    panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 30  # Fondo gris oscuro

    # Encabezado del Agente
    cv2.rectangle(panel, (0, 0), (panel_w, 54), (45, 52, 54), -1)
    cv2.putText(
        panel,
        "AGENTE CONTROL ACCESO (MULTI)",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 210, 255),
        2,
    )
    iot_txt = telemetria.get("estado_iot", "Local")
    cv2.putText(
        panel,
        f"FISEI-UTA | {iot_txt} | FPS: {telemetria.get('fps', 0.0):.1f}",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
    )

    y_offset = 67

    # 1. Algoritmos Activos Explícitos por Fase (Etapa 1: Rostro vs Objeto | Etapa 2: Autorizado vs No)
    cv2.putText(panel, "ALGORITMOS ACTIVOS POR FASE:", (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y_offset += 16
    txt_f1 = limpiar_texto_cv2(telemetria.get("alg_fase1", "ML: SVM"))
    txt_f2 = limpiar_texto_cv2(telemetria.get("alg_fase2", "ML: SVM"))
    cv2.putText(panel, f"E1 (Rostro/Obj): {txt_f1[:23]}", (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
    y_offset += 15
    cv2.putText(panel, f"E2 (Autorizado): {txt_f2[:23]}", (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 160), 1)

    y_offset += 10
    cv2.line(panel, (12, y_offset), (panel_w - 12, y_offset), (70, 70, 70), 1)
    y_offset += 15

    # 2. Telemetría Multi-Rostro (Etapa 1 + Etapa 2)
    total_r = telemetria.get("total_rostros", 0)
    num_aut = telemetria.get("num_autorizados", 0)
    num_desc = telemetria.get("num_desconocidos", 0)
    cv2.putText(panel, "PERCEPCION MULTI-ROSTRO:", (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    y_offset += 18

    color_conteo = (0, 255, 0) if num_aut > 0 else ((0, 165, 255) if total_r == 0 else (0, 100, 255))
    cv2.putText(
        panel,
        f"Rostros: {total_r} | Aut: {num_aut} | Desc: {num_desc}",
        (12, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        color_conteo,
        1,
    )
    y_offset += 17
    cv2.putText(
        panel,
        f"MSE Principal: {telemetria.get('mse', 0.0):.5f}",
        (12, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
    )
    y_offset += 18

    # Lista compacta de hasta 3 rostros activos
    resumen = telemetria.get("resumen_rostros", [])
    if not resumen:
        cv2.putText(
            panel,
            "  Sin rostros humanos en escena",
            (12, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (150, 150, 150),
            1,
        )
        y_offset += 46
    else:
        for item in resumen[:3]:
            nom_l = limpiar_texto_cv2(item["nombre"])[:18]
            if item["autorizado"]:
                col_item = (0, 255, 0)
                txt_item = f" [OK] {nom_l} ({item['confianza']*100:.0f}%)"
            elif item["desconocido"]:
                col_item = (0, 80, 255)
                txt_item = f" [X]  Desconocido ({item['confianza']*100:.0f}%)"
            else:
                col_item = (0, 220, 255)
                txt_item = f" [..] {nom_l}"
            cv2.putText(panel, txt_item, (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col_item, 1)
            y_offset += 16
        if len(resumen) < 3:
            y_offset += 16 * (3 - len(resumen))

    cv2.line(panel, (12, y_offset), (panel_w - 12, y_offset), (70, 70, 70), 1)
    y_offset += 16

    # 3. Simulación Gráfica del Actuador Físico (Puerta Corrediza)
    cv2.putText(panel, "ESTADO DEL ACTUADOR FISICO:", (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    y_offset += 12

    puerta_x1 = 30
    puerta_x2 = panel_w - 30
    puerta_y1 = y_offset
    puerta_h = 62
    puerta_y2 = y_offset + puerta_h
    puerta_w = puerta_x2 - puerta_x1

    cv2.rectangle(panel, (puerta_x1 - 4, puerta_y1 - 4), (puerta_x2 + 4, puerta_y2 + 4), (100, 100, 100), 2)
    cv2.rectangle(
        panel,
        (puerta_x1, puerta_y1),
        (puerta_x2, puerta_y2),
        (30, 80, 40) if telemetria["autorizado"] else (40, 20, 20),
        -1,
    )

    hoja_ancho = puerta_w // 2
    desplazamiento = int(hoja_ancho * progreso_apertura)

    cv2.rectangle(panel, (puerta_x1, puerta_y1), (puerta_x1 + hoja_ancho - desplazamiento, puerta_y2), (120, 90, 60), -1)
    cv2.rectangle(panel, (puerta_x1, puerta_y1), (puerta_x1 + hoja_ancho - desplazamiento, puerta_y2), (160, 130, 90), 2)

    cv2.rectangle(panel, (puerta_x1 + hoja_ancho + desplazamiento, puerta_y1), (puerta_x2, puerta_y2), (120, 90, 60), -1)
    cv2.rectangle(panel, (puerta_x1 + hoja_ancho + desplazamiento, puerta_y1), (puerta_x2, puerta_y2), (160, 130, 90), 2)

    y_offset = puerta_y2 + 18
    color_texto_act = (0, 255, 0) if estado_puerta == "ABIERTA" else (0, 0, 255)
    icono_puerta = "[PUERTA ABIERTA]" if estado_puerta == "ABIERTA" else "[PUERTA CERRADA]"
    cv2.putText(panel, icono_puerta, (puerta_x1 + 12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_texto_act, 2)

    y_offset += 18
    if telemetria["autorizado"]:
        noms = telemetria.get("nombres_autorizados", [])
        if noms:
            txt_acc = f"AUTORIZADO: {limpiar_texto_cv2(', '.join(noms))[:22]}"
        else:
            txt_acc = "ACCESO CONCEDIDO (OR)"
    else:
        txt_acc = "ACCESO DENEGADO"
    cv2.putText(panel, txt_acc, (18, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_texto_act, 1)

    # 4. Ayuda de Teclas en la parte inferior
    box_top = max(y_offset + 10, panel_h - 114)
    cv2.rectangle(panel, (10, box_top), (panel_w - 10, panel_h - 8), (20, 20, 20), -1)
    cv2.putText(panel, "SELECTOR DE ALGORITMOS (TECLADO):", (15, box_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 200, 255), 1)
    cv2.putText(panel, "[0] Optimo  [1] k-NN  [2] Arbol (DT)", (15, box_top + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
    cv2.putText(panel, "[3] RF      [4] SVM   [5] ANN (MLP)", (15, box_top + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
    cv2.putText(panel, "[6] CNN (2D) [B] Fondo [C] Calib [N] Bloq", (15, box_top + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 200), 1)
    cv2.putText(panel, "[Q / ESC] Salir", (15, box_top + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 120), 1)

    return panel


def dibujar_rostro_y_landmarks(frame: np.ndarray, track: RostroTrack, W: int, H: int):
    """Dibuja el cuadro delimitador individual, su etiqueta y la malla cian de 5 Landmarks."""
    qx, qy, qlado, _ = track.caja_int(W, H)
    color = track.color_cuadro

    cv2.rectangle(frame, (qx, qy), (qx + qlado, qy + qlado), color, 2)

    # Dibujar malla triangular de 5 Landmarks anatómicos
    if track.landmarks_px:
        pts_px = track.landmarks_px
        color_malla = (255, 255, 0)  # Cian en BGR
        conexiones = [
            ("ojo_izq", "ojo_der"),
            ("ojo_izq", "nariz"),
            ("ojo_der", "nariz"),
            ("nariz", "boca_izq"),
            ("nariz", "boca_der"),
            ("boca_izq", "boca_der"),
        ]
        for p1_k, p2_k in conexiones:
            if p1_k in pts_px and p2_k in pts_px:
                cv2.line(frame, pts_px[p1_k], pts_px[p2_k], color_malla, 1, cv2.LINE_AA)
        for pt_k in ["ojo_izq", "ojo_der", "nariz", "boca_izq", "boca_der"]:
            if pt_k in pts_px:
                cv2.circle(frame, pts_px[pt_k], 3, color_malla, -1, cv2.LINE_AA)

    # Barra de progreso biométrica bajo el rostro mientras verifica (0% -> 100%)
    prog_v = float(getattr(track, "progreso_verificacion", 0.0))
    if (not track.autorizado) and (0.0 < prog_v < 1.0):
        by1 = min(H - 10, qy + qlado + 4)
        by2 = min(H - 3, by1 + 7)
        cv2.rectangle(frame, (qx, by1), (qx + qlado, by2), (35, 35, 35), -1)
        ancho_lleno = int(round(qlado * prog_v))
        if ancho_lleno > 0:
            cv2.rectangle(frame, (qx, by1), (qx + ancho_lleno, by2), (0, 220, 255), -1)
        cv2.rectangle(frame, (qx, by1), (qx + qlado, by2), (0, 180, 220), 1)

    # Etiqueta limpia con fondo semioscuro para alta legibilidad
    if track.etiqueta_texto:
        texto_limpio = limpiar_texto_cv2(track.etiqueta_texto)
        escala_f = 0.48
        (tw, th), _ = cv2.getTextSize(texto_limpio, cv2.FONT_HERSHEY_SIMPLEX, escala_f, 1)
        tx = int(np.clip(qx, 4, max(4, W - tw - 8)))
        ty = max(th + 8, qy - 6)
        cv2.rectangle(frame, (tx - 3, ty - th - 4), (tx + tw + 4, ty + 4), (20, 20, 20), -1)
        cv2.putText(
            frame,
            texto_limpio,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_f,
            color,
            1,
            cv2.LINE_AA,
        )


def dibujar_animacion_autorizado(
    frame: np.ndarray,
    t_transcurrido: float,
    duracion: float,
    nombres_aut: list,
    telegram_enviado: bool,
):
    """
    Animación visual de confirmación de acceso autorizado (durante ~2.8 s):
    - Borde verde pulsante alrededor del video.
    - Tarjeta inferior semitransparente con checkmark animado, nombre del usuario y estado Telegram.
    """
    if t_transcurrido < 0.0 or t_transcurrido > duracion:
        return
    H, W = frame.shape[:2]

    # Envolvente suave fade-in (0..0.25s) y fade-out (últimos 0.55s)
    if t_transcurrido < 0.25:
        alpha_env = t_transcurrido / 0.25
    elif t_transcurrido > (duracion - 0.55):
        alpha_env = max(0.0, (duracion - t_transcurrido) / 0.55)
    else:
        alpha_env = 1.0

    # 1. Marco verde pulsante alrededor de la cámara
    pulso = 0.65 + 0.35 * float(np.sin(t_transcurrido * 8.0))
    grosor = max(2, int(round(5 * alpha_env * pulso)))
    cv2.rectangle(frame, (2, 30), (W - 3, H - 3), (0, int(235 * alpha_env), 60), grosor)

    # 2. Tarjeta semitransparente en la franja inferior del video
    card_h = 74
    y1 = H - card_h - 14
    y2 = H - 14
    x1 = 22
    x2 = W - 22

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 52, 24), -1)
    cv2.addWeighted(overlay, 0.78 * alpha_env, frame, 1.0 - 0.78 * alpha_env, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 90), 2)

    # 3. Círculo con Checkmark animado (se dibuja en los primeros 0.45 s)
    cx = x1 + 38
    cy = (y1 + y2) // 2
    r_chk = int(round(22 * min(1.0, t_transcurrido / 0.22)))
    if r_chk >= 4:
        cv2.circle(frame, (cx, cy), r_chk, (0, 210, 75), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r_chk, (190, 255, 200), 2, cv2.LINE_AA)

    p_chk = min(1.0, max(0.0, (t_transcurrido - 0.12) / 0.35))
    if p_chk > 0.0:
        pt1 = (cx - 10, cy + 1)
        pt2 = (cx - 3, cy + 9)
        pt3 = (cx + 12, cy - 8)
        if p_chk <= 0.45:
            t1 = p_chk / 0.45
            pm = (int(pt1[0] + (pt2[0] - pt1[0]) * t1), int(pt1[1] + (pt2[1] - pt1[1]) * t1))
            cv2.line(frame, pt1, pm, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            cv2.line(frame, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
            t2 = (p_chk - 0.45) / 0.55
            pm2 = (int(pt2[0] + (pt3[0] - pt2[0]) * t2), int(pt2[1] + (pt3[1] - pt2[1]) * t2))
            cv2.line(frame, pt2, pm2, (255, 255, 255), 3, cv2.LINE_AA)

    # 4. Textos de bienvenida y estado Telegram
    tx_base = x1 + 74
    nombre_str = limpiar_texto_cv2(", ".join(nombres_aut)) if nombres_aut else "Usuario Autorizado"
    cv2.putText(
        frame,
        "ACCESO AUTORIZADO - BIENVENIDO",
        (tx_base, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (120, 255, 140),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Persona: {nombre_str[:34]}",
        (tx_base, y1 + 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    sub_tg = "Telegram: Notificacion con foto enviada" if telegram_enviado else "Bitacora: Acceso verificado y registrado"
    cv2.putText(
        frame,
        sub_tg,
        (tx_base, y1 + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (160, 240, 255) if telegram_enviado else (180, 220, 180),
        1,
        cv2.LINE_AA,
    )


def ejecutar_demostracion():
    print("=" * 80)
    print("INICIANDO SISTEMA BIOMÉTRICO MULTI-ROSTRO EN TIEMPO REAL (REGLA OR)")
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
        [c for c in nombres_e2.keys() if "desconocido" not in str(nombres_e2[c]).lower()],
    )
    id_desconocido = metadata.get("id_desconocido", max(nombres_e2.keys()) if nombres_e2 else 1)
    perfiles = metadata.get("perfiles_identidad", {})
    umbral_mse_id = metadata.get("umbral_mse_identidad", 0.0030)
    umbral_mse_e1 = min(0.0085, metadata.get("umbral_mse_rostro", 0.0085))

    scaler_pca_respaldo_e1 = modelos_e1.get("SVM", {}).get("scaler", None)
    modelo_fuerza = None  # None significa Modo Óptimo Automático

    # Variables de Calibración Rápida en Vivo
    gris_fondo_completo = None
    ruta_negativos_pkl = os.path.join(DIR_MODELOS, "negativos_vivo_pca.pkl")
    lista_negativos_vivo = []
    if os.path.exists(ruta_negativos_pkl):
        try:
            lista_negativos_vivo = joblib.load(ruta_negativos_pkl)
        except Exception:
            lista_negativos_vivo = []

    # Cargar configuración IoT (ESP32-CAM + OLED + LEDs) y Bot de Telegram desde config_iot.json
    cfg_iot = cargar_config_iot()
    cfg_esp = cfg_iot["esp32"]
    cfg_tg = cfg_iot["telegram"]

    puerto_ctrl = int(cfg_esp.get("puerto_control", 80))
    puerto_str = int(cfg_esp.get("puerto_stream", 80))
    # Habilitar control IoT asíncrono para enviar etiquetas al ESP32 (OLED y actuadores)
    habilitar_ctrl_iot = bool(cfg_esp.get("habilitado", False))

    cliente_iot = ClienteIoTESP32(
        habilitado=habilitar_ctrl_iot,
        ip=cfg_esp.get("ip", ""),
        puerto_control=puerto_ctrl,
        puerto_stream=puerto_str,
    )
    notificador_tg = NotificadorTelegram(
        habilitado=cfg_tg.get("habilitado", False),
        bot_token=cfg_tg.get("bot_token", ""),
        chat_id=cfg_tg.get("chat_id", ""),
        nombre_lugar=cfg_tg.get("nombre_lugar", "Laboratorio FISEI - UTA"),
        cooldown_autorizado_seg=cfg_tg.get("cooldown_autorizado_seg", 15.0),
        cooldown_alerta_seg=cfg_tg.get("cooldown_alerta_seg", 15.0),
        enviar_alerta_desconocido=cfg_tg.get("enviar_alerta_desconocido", True),
    )

    usar_cam_esp32 = bool(cfg_esp.get("usar_camara_esp32", False))
    voltear_h = bool(cfg_esp.get("voltear_horizontal", False))
    rotar_180 = bool(cfg_esp.get("rotar_180", False))

    if usar_cam_esp32:
        ip_esp = cfg_esp.get("ip", "192.168.100.77")
        if puerto_str == 80:
            url_stream = f"http://{ip_esp}/"
            url_capture = None
        else:
            url_stream = f"http://{ip_esp}:{puerto_str}/stream"
            url_capture = f"http://{ip_esp}:{puerto_ctrl}/capture"
        print(f"[IoT] Conectando a cámara ESP32-CAM en {url_stream} ...")
        lector = LectorCamaraThreaded(fuente=url_stream, url_capture_respaldo=url_capture, rotar_180=rotar_180)
        if not lector.iniciar():
            print(f"[AVISO IoT] No se pudo conectar al ESP32-CAM en {url_stream}.")
            print("            Consejo: cierre cualquier pestaña del navegador abierta en esa IP y verifique que el ESP32 esté encendido.")
            print("            Cambiando automáticamente a Webcam Laptop (0)...")
            usar_cam_esp32 = False
            # IMPORTANTE: NO invertir horizontalmente (voltear_h = False) para respetar la misma
            # orientación lateral izquierda/derecha con la que se capturó el dataset biométrico.
            voltear_h = False
            lector = LectorCamaraThreaded(fuente=0, rotar_180=False)
            if not lector.iniciar():
                print("[ERROR] Tampoco se pudo abrir la cámara web local.")
                return
    else:
        lector = LectorCamaraThreaded(fuente=0, rotar_180=False)
        if not lector.iniciar():
            print("[ERROR] No se pudo acceder a la cámara web. Verifique la conexión o permisos.")
            return

    gestor_tracks = GestorTracksFaciales(max_rostros=MAX_ROSTROS_SIMULTANEOS, max_frames_perdidos=3)
    progreso_apertura = 0.0
    ultimo_acceso_valido = 0.0
    TIEMPO_RETENCION_ACCESO = 2.6  # Segundos de estabilidad de puerta abierta
    inicio_desconocido_continuo = None
    ultimos_nombres_autorizados = []
    # Estado de la animación de confirmación de Acceso Autorizado + sincronización con Telegram
    t_inicio_anim_aut = -10.0
    DURACION_ANIM_AUT = 2.8
    anim_aut_nombres = []
    anim_tg_enviado = False
    estaba_autorizado_prev = False
    toast_msg = ""
    toast_color = (0, 230, 255)
    toast_hasta = 0.0
    fps_suave = 15.0
    t_prev = time.time()

    etiqueta_cam = "ESP32-CAM IoT" if usar_cam_esp32 else "Webcam Laptop"
    print(f"\nSistema iniciado ({etiqueta_cam}) en MODO MULTI-ROSTRO (Regla de acceso OR).")
    print(f"  - Dispositivo IoT ESP32 OLED/LEDs: {'ACTIVO (' + cfg_esp['ip'] + ')' if cliente_iot.habilitado else 'DESACTIVADO (config_iot.json)'}")
    print(f"  - Bot de Telegram (Avisos + Foto): {'ACTIVO' if notificador_tg.habilitado else 'DESACTIVADO (config_iot.json)'}")
    print("Use las teclas [0] a [6] para conmutar modelos en caliente.")
    print("Teclas de Calibración: [B] Fondo Escena | [C] Luz Rostro Principal | [N] Bloquear Objeto")
    print("Presione [Q] o [ESC] para salir.\n")

    cliente_iot.enviar_estado("ESPERA", total_rostros=0, forzar=True)

    # =========================================================================
    # ARQUITECTURA DESACOPLADA (3 HILOS: CAMARA + IA BIOMETRICA + INTERFAZ GUI)
    # Garantiza video 100% fluido sin congelamientos incluso con multiples rostros o CNN.
    # =========================================================================
    lock_ia = threading.Lock()
    estado_ia = {
        "tracks_visibles": [],
        "estado_or": {
            "hay_autorizado": False,
            "total_rostros": 0,
            "num_autorizados": 0,
            "num_desconocidos": 0,
            "nombres_autorizados": [],
            "resumen_rostros": [],
        },
        "fps_ia": 0.0,
        "t_ultimo_frame_ia": time.time(),
    }
    ctrl_ia = {
        "activo": True,
        "modelo_fuerza": modelo_fuerza,
        "gris_fondo": gris_fondo_completo,
        "limpiar_buffers": False,
        "limpiar_tracks": False,
    }

    def _bucle_worker_ia():
        ultimo_fid_proc = -1
        t_ia_prev = time.time()
        fps_ia_loc = 0.0
        INTERVALO_IA_SEG = 0.14  # ~6-7 evaluaciones IA/seg: libera >70% del GIL para video fluido sin congelamientos

        while ctrl_ia["activo"]:
            t_ciclo_ini = time.time()
            ret_w, fr_w, fid_w = lector.leer_con_id()
            if (not ret_w) or (fr_w is None) or (fid_w == ultimo_fid_proc):
                time.sleep(0.015)
                continue
            ultimo_fid_proc = fid_w

            if voltear_h:
                fr_w = cv2.flip(fr_w, 1)
            Hw, Ww = fr_w.shape[:2]

            with lock_ia:
                if ctrl_ia["limpiar_tracks"]:
                    gestor_tracks.tracks.clear()
                    ctrl_ia["limpiar_tracks"] = False
                if ctrl_ia["limpiar_buffers"]:
                    gestor_tracks.limpiar_buffers()
                    ctrl_ia["limpiar_buffers"] = False
                mod_f = ctrl_ia["modelo_fuerza"]
                gris_fnd = ctrl_ia["gris_fondo"]

            if mod_f is None:
                info_e1_act = modelos_e1.get("SVM", artefactos["mejor_e1_info"])
            elif mod_f in ["CNN (Deep Learning)", "CNN (Conv2D)"]:
                info_e1_act = artefactos.get("modelo_cnn_e1", artefactos["mejor_e1_info"])
            else:
                info_e1_act = obtener_modelo_por_nombre(modelos_e1, mod_f, artefactos["mejor_e1_info"])

            try:
                # 1. ETAPA 1 MULTI-ROSTRO
                dets_e1 = detectar_rostros_etapa1(
                    frame=fr_w,
                    pca_e1=pca_e1,
                    info_e1_activo=info_e1_act,
                    scaler_pca_respaldo_e1=scaler_pca_respaldo_e1,
                    umbral_mse_e1=umbral_mse_e1,
                    tracks_previos=gestor_tracks.tracks,
                    gris_fondo_completo=gris_fnd,
                    lista_negativos_vivo=lista_negativos_vivo,
                    pca_e2=pca_e2,
                    max_rostros=MAX_ROSTROS_SIMULTANEOS,
                    modelos_e1=(modelos_e1 if mod_f is None else None),
                )

                # 2. ASOCIACIÓN DE TRACKS MULTI-ROSTRO
                pares_act = gestor_tracks.asociar_detecciones(dets_e1, Ww, Hw)

                # 3. ETAPA 2 INDEPENDIENTE PARA CADA ROSTRO HUMANO CONFIRMADO
                for tr_k, det_k in pares_act:
                    modo_comp = (len(pares_act) >= 2) and (tr_k.frames_vistos > 3)
                    # Evaluar usando la caja suavizada del track (evita saltos bruscos de E1 frame-a-frame)
                    caja_eval_e2 = tr_k.caja_int(Ww, Hw)
                    res_e2_k = evaluar_identidad_etapa2(
                        frame=fr_w,
                        caja_rostro=caja_eval_e2,
                        pca_e2=pca_e2,
                        modelos_e2=modelos_e2,
                        modelo_cnn=modelo_cnn,
                        artefactos=artefactos,
                        modelo_fuerza=mod_f,
                        nombres_e2=nombres_e2,
                        ids_autorizados=ids_autorizados,
                        id_desconocido=id_desconocido,
                        perfiles=perfiles,
                        umbral_mse_id=umbral_mse_id,
                        modo_compacto=modo_comp,
                    )
                    # Solo retroalimentar centrado cuando el encuadre ajustado mantiene una escala facial
                    # saludable (>= 85% de la detección de Etapa 1) para evitar colapso progresivo de la caja.
                    c_adj = res_e2_k.get("caja_ajustada")
                    if c_adj is not None and c_adj[2] >= int(det_k["caja"][2] * 0.85):
                        tr_k.actualizar_caja(c_adj, Ww, Hw, alpha=0.18)
                    tr_k.actualizar_decision_etapa2(res_e2_k, ids_autorizados, id_desconocido, nombres_e2)

                # 4. UNICIDAD BIOMÉTRICA 1-A-1 + BARRERA DE VERIFICACIÓN SIMULTÁNEA MULTI-ROSTRO
                # Si hay múltiples rostros en escena y uno todavía está en fase amarilla "VERIFICANDO..."
                # (aún no termina sus ~1.8s iniciales), mantener el veredicto en espera sincronizada
                # hasta que TODOS los rostros visibles completen su verificación biométrica, para
                # revelar todos los resultados (Autorizado / Desconocido) al mismo tiempo de forma limpia.
                tr_vis = gestor_tracks.obtener_tracks_visibles()
                tr_snap = [copy.copy(t) for t in tr_vis]
                ganadores_por_id = {}
                for t_s in tr_snap:
                    if t_s.autorizado and (t_s.identidad_id in ids_autorizados):
                        prev_t = ganadores_por_id.get(t_s.identidad_id)
                        if prev_t is None or t_s.confianza > prev_t.confianza:
                            ganadores_por_id[t_s.identidad_id] = t_s
                for t_s in tr_snap:
                    if t_s.autorizado and (t_s.identidad_id in ids_autorizados):
                        if ganadores_por_id.get(t_s.identidad_id) is not t_s:
                            t_s.autorizado = False
                            t_s.identidad_id = id_desconocido
                            t_s.identidad_nombre = "Desconocido"
                            t_s.confianza = 0.78
                            t_s.color_cuadro = (0, 0, 255)
                            t_s.etiqueta_texto = "[NO AUTORIZADO: Desconocido] (78%)"

                # Barrera sincronizada multi-rostro: si hay >= 2 rostros y alguno aún está verificando
                # (edad < 4.5 s para evitar bloqueos eternos), mostrar la barra amarilla sincronizada en todos
                # hasta que todos los rostros de la escena tengan su veredicto biométrico listo a la vez.
                ahora_barrera = time.time()
                alguno_verificando = any(
                    (not t_s.autorizado)
                    and (t_s.identidad_id is None)
                    and ((ahora_barrera - getattr(t_s, "t_creacion", 0.0)) < 4.5)
                    for t_s in tr_snap
                )
                if len(tr_snap) >= 2 and alguno_verificando:
                    progs_sync = [
                        (1.0 if (t_s.autorizado or t_s.identidad_id == id_desconocido) else float(getattr(t_s, "progreso_verificacion", 0.5)))
                        for t_s in tr_snap
                    ]
                    prog_global = float(np.clip(min(progs_sync), 0.15, 0.95))
                    for t_s in tr_snap:
                        t_s.autorizado = False
                        t_s.identidad_id = None
                        t_s.progreso_verificacion = prog_global
                        t_s.identidad_nombre = "Verificando escena..."
                        t_s.color_cuadro = (0, 220, 255)
                        t_s.etiqueta_texto = f"VERIFICANDO ESCENA... [{int(prog_global * 100)}%]"

                est_or = evaluar_acceso_global_or(tr_snap, ids_autorizados, id_desconocido)

                t_ia_now = time.time()
                dt_ia = max(1e-3, t_ia_now - t_ia_prev)
                t_ia_prev = t_ia_now
                fps_ia_loc = (1.0 / dt_ia) if fps_ia_loc <= 0.1 else (0.80 * fps_ia_loc + 0.20 * (1.0 / dt_ia))

                with lock_ia:
                    estado_ia["tracks_visibles"] = tr_snap
                    estado_ia["estado_or"] = est_or
                    estado_ia["fps_ia"] = fps_ia_loc
                    estado_ia["t_ultimo_frame_ia"] = time.time()
            except Exception:
                time.sleep(0.03)

            # Ceder explícitamente el GIL al hilo de red del ESP32-CAM y a la ventana GUI
            dt_ciclo = time.time() - t_ciclo_ini
            pausa_gil = max(0.05, INTERVALO_IA_SEG - dt_ciclo)
            time.sleep(pausa_gil)

    hilo_ia = threading.Thread(target=_bucle_worker_ia, daemon=True)
    hilo_ia.start()

    while True:
        ret, frame = lector.leer()
        if not ret or frame is None:
            time.sleep(0.015)
            continue

        t_now = time.time()
        dt = max(1e-4, t_now - t_prev)
        t_prev = t_now
        fps_suave = 0.85 * fps_suave + 0.15 * (1.0 / dt)

        if voltear_h:
            frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        with lock_ia:
            tracks_visibles = list(estado_ia["tracks_visibles"])
            estado_or = dict(estado_ia["estado_or"])
            fps_ia_ui = float(estado_ia["fps_ia"])
            t_ultimo_frame_ia = float(estado_ia["t_ultimo_frame_ia"])

        tiempo_actual = time.time()

        # Fallo seguro ante pérdida de video: si el hilo IA no procesa frames
        # nuevos en > 2.5 s (stream cortado, cámara desconectada, etc.), expirar
        # el estado a ESPERA / PUERTA CERRADA para nunca mantener AUTORIZADO
        # indefinidamente sobre un fotograma obsoleto.
        TIMEOUT_SIN_FRAMES_SEG = 2.5
        stream_vivo = (tiempo_actual - t_ultimo_frame_ia) < TIMEOUT_SIN_FRAMES_SEG
        if not stream_vivo:
            estado_or["hay_autorizado"] = False
            estado_or["total_rostros"] = 0
            estado_or["num_autorizados"] = 0
            estado_or["num_desconocidos"] = 0
            estado_or["nombres_autorizados"] = []
            estado_or["resumen_rostros"] = []
            tracks_visibles = []

        if estado_or["hay_autorizado"]:
            ultimo_acceso_valido = tiempo_actual
            ultimos_nombres_autorizados = estado_or["nombres_autorizados"]
            inicio_desconocido_continuo = None

        esta_en_retencion = (tiempo_actual - ultimo_acceso_valido) < TIEMPO_RETENCION_ACCESO
        if esta_en_retencion:
            progreso_apertura = min(1.0, progreso_apertura + 0.15)
            estado_puerta = "ABIERTA"
            nombres_mostrar_aut = (
                estado_or["nombres_autorizados"]
                if estado_or["nombres_autorizados"]
                else ultimos_nombres_autorizados
            )
            # Enviar etiqueta AUTORIZADO al ESP32-CAM (OLED muestra "ACCESO AUTORIZADO / BIENVENIDO" sin nombre)
            cliente_iot.enviar_estado("AUTORIZADO", total_rostros=estado_or["total_rostros"])
        else:
            progreso_apertura = max(0.0, progreso_apertura - 0.20)
            estado_puerta = "CERRADA"
            nombres_mostrar_aut = []
            if estado_or["num_desconocidos"] > 0:
                cliente_iot.enviar_estado("DENEGADO", total_rostros=estado_or["total_rostros"])
                if inicio_desconocido_continuo is None:
                    inicio_desconocido_continuo = tiempo_actual
            else:
                cliente_iot.enviar_estado("ESPERA", total_rostros=0)
                inicio_desconocido_continuo = None

        # 5. DIBUJO SOBRE EL VIDEO EN VIVO (TODOS LOS ROSTROS + LANDMARKS + ANIMACIÓN DE AUTORIZACIÓN)
        for tr in tracks_visibles:
            dibujar_rostro_y_landmarks(frame, tr, W, H)

        # Detectar el instante exacto en que se confirma la autorización (100% de verificación):
        # dispara simultáneamente la animación visual y el aviso de Telegram con el nombre y foto.
        if estado_or["hay_autorizado"]:
            es_nueva_confirmacion = (not estaba_autorizado_prev) or ((tiempo_actual - t_inicio_anim_aut) > 12.0)
            if es_nueva_confirmacion:
                t_inicio_anim_aut = tiempo_actual
                anim_aut_nombres = list(estado_or["nombres_autorizados"])
                anim_tg_enviado = notificador_tg.notificar_ingreso_autorizado(
                    frame_bgr=frame,
                    total_rostros=estado_or["total_rostros"],
                    num_autorizados=estado_or["num_autorizados"],
                    nombres_aut=estado_or["nombres_autorizados"],
                )
        elif (
            (not esta_en_retencion)
            and inicio_desconocido_continuo is not None
            and (tiempo_actual - inicio_desconocido_continuo) >= 2.2
        ):
            notificador_tg.notificar_alerta_desconocido(
                frame_bgr=frame,
                total_rostros=estado_or["total_rostros"],
                num_desconocidos=estado_or["num_desconocidos"],
            )
        estaba_autorizado_prev = bool(estado_or["hay_autorizado"])

        # Superponer animación elegante de "ACCESO AUTORIZADO" mientras esté activa
        t_anim = tiempo_actual - t_inicio_anim_aut
        if 0.0 <= t_anim <= DURACION_ANIM_AUT:
            dibujar_animacion_autorizado(frame, t_anim, DURACION_ANIM_AUT, anim_aut_nombres, anim_tg_enviado)

        # Superponer aviso visual (toast) para confirmar pulsaciones de teclado [C], [B], [N], [0]-[6]
        if toast_msg and tiempo_actual < toast_hasta:
            txt_toast = limpiar_texto_cv2(toast_msg)[:64]
            cv2.rectangle(frame, (8, H - 36), (W - 8, H - 8), (20, 24, 30), -1)
            cv2.rectangle(frame, (8, H - 36), (W - 8, H - 8), toast_color, 2)
            cv2.putText(frame, txt_toast, (16, H - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, toast_color, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (W, 28), (22, 24, 28), -1)
        hud_color = (0, 255, 0) if esta_en_retencion else (0, 210, 255)
        fps_cam_real = lector.fps_camara if getattr(lector, "fps_camara", 0.0) > 0.5 else fps_suave
        if estado_or["total_rostros"] == 0:
            hud_txt = f"{etiqueta_cam} | Escaneando... (0 rostros) | Cam:{fps_cam_real:.0f} IA:{fps_ia_ui:.0f} FPS"
        else:
            hud_txt = (
                f"{etiqueta_cam} (OR) | Rostros:{estado_or['total_rostros']} "
                f"Aut:{estado_or['num_autorizados']} Desc:{estado_or['num_desconocidos']} | "
                f"Cam:{fps_cam_real:.0f} IA:{fps_ia_ui:.0f}"
            )
        cv2.putText(frame, hud_txt, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, hud_color, 1, cv2.LINE_AA)

        # 6. PANEL LATERAL DE TELEMETRÍA Y ACTUADOR
        def _etiqueta_tipo_ia(nom_mod: str) -> str:
            nl = str(nom_mod).lower()
            return "DL" if ("cnn" in nl or "ann" in nl or "mlp" in nl or "deep" in nl) else "ML"

        if modelo_fuerza is None:
            alg_f1_ui = "Ensamble (SVM+CNN+RF)"
            alg_f2_ui = "Ensamble (SVM+ML+CNN)"
            nombre_activo_ui = "Optimo (Ensamble ML+DL)"
        else:
            t_f = _etiqueta_tipo_ia(modelo_fuerza)
            alg_f1_ui = f"{t_f}: {modelo_fuerza}"
            alg_f2_ui = f"{t_f}: {modelo_fuerza}"
            nombre_activo_ui = f"Forzado: {modelo_fuerza}"
        mse_principal = tracks_visibles[0].mse_e2 if tracks_visibles else 0.0
        iot_badge = f"IoT:{'ON' if cliente_iot.habilitado else 'OFF'} TG:{'ON' if notificador_tg.habilitado else 'OFF'}"
        telemetria = {
            "modelo_activo": nombre_activo_ui,
            "alg_fase1": alg_f1_ui,
            "alg_fase2": alg_f2_ui,
            "estado_iot": iot_badge,
            "fps": fps_cam_real,
            "total_rostros": estado_or["total_rostros"],
            "num_autorizados": estado_or["num_autorizados"],
            "num_desconocidos": estado_or["num_desconocidos"],
            "resumen_rostros": estado_or["resumen_rostros"],
            "nombres_autorizados": nombres_mostrar_aut,
            "mse": mse_principal,
            "autorizado": esta_en_retencion,
        }

        panel_w = 340
        panel_lateral = dibujar_panel_actuador(panel_w, H, estado_puerta, progreso_apertura, telemetria)
        vista_completa = np.hstack([frame, panel_lateral])
        cv2.imshow("Sistema Biometrico Multi-Rostro IA - Ortiz & Ayuquina (FISEI UTA)", vista_completa)

        # 7. CONTROLES DE TECLADO EN TIEMPO REAL (waitKey(15) libera CPU para hilos de Red e IA)
        # Además lee la consola de Windows (msvcrt) por si el usuario pulsa [C]/[B]/[0-6] con foco en la terminal.
        tecla = cv2.waitKey(15) & 0xFF
        if tecla == 255 and os.name == "nt":
            try:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch and len(ch) == 1:
                        tecla = ord(ch)
            except Exception:
                pass

        if tecla in [ord("q"), ord("Q"), 27]:
            print("\nFinalizando demostración en tiempo real...", flush=True)
            break
        elif tecla == ord("0"):
            modelo_fuerza = None
            with lock_ia:
                ctrl_ia["modelo_fuerza"] = None
                ctrl_ia["limpiar_buffers"] = True
            ultimo_acceso_valido = 0.0
            toast_msg = "[0] MODO OPTIMO AUTOMATICO (Ensamble ML + CNN)"
            toast_color = (0, 255, 160)
            toast_hasta = time.time() + 2.5
            print(">>> Conmutado a Modo Óptimo Automático", flush=True)
        elif tecla in [ord("b"), ord("B")]:
            gris_fondo_completo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()
            with lock_ia:
                ctrl_ia["gris_fondo"] = gris_fondo_completo
                ctrl_ia["limpiar_tracks"] = True
            ultimo_acceso_valido = 0.0
            toast_msg = "[B] FONDO DE ESCENA CALIBRADO CORRECTAMENTE"
            toast_color = (0, 230, 255)
            toast_hasta = time.time() + 2.5
            print("[CALIBRACIÓN] Fondo completo de la escena registrado con tecla [B].", flush=True)
        elif tecla in [ord("c"), ord("C")]:
            calibrados = []
            with lock_ia:
                tracks_cal = list(gestor_tracks.tracks)
            usados_cal = set()
            for tr_p in tracks_cal:
                if tr_p.ultimo_v_id_vivo is not None:
                    v_cal = tr_p.ultimo_v_id_vivo
                    v_cal_n = v_cal / (float(np.linalg.norm(v_cal)) + 1e-8)
                    if tr_p.identidad_id in ids_autorizados and tr_p.identidad_id not in usados_cal:
                        target_id = tr_p.identidad_id
                    else:
                        # Elegir el perfil autorizado con mayor afinidad angular en F_Identity (asignación 1-a-1)
                        cands_id = [cid for cid in ids_autorizados if cid not in usados_cal and cid in perfiles]
                        if not cands_id:
                            cands_id = [cid for cid in ids_autorizados if cid in perfiles]
                        target_id = None
                        mejor_cos_cal = -2.0
                        for cid in cands_id:
                            mat_p = perfiles[cid].get("matriz_Z_poses", None)
                            if mat_p is not None and len(mat_p) > 0 and mat_p.shape[1] == len(v_cal):
                                mat_n = mat_p / (np.linalg.norm(mat_p, axis=1, keepdims=True) + 1e-8)
                                cos_c = float(np.mean(np.sort(mat_n @ v_cal_n)[-5:]))
                            else:
                                cen = perfiles[cid]["centroide"]
                                cos_c = float(np.dot(cen / (float(np.linalg.norm(cen)) + 1e-8), v_cal_n))
                            if cos_c > mejor_cos_cal:
                                mejor_cos_cal = cos_c
                                target_id = cid
                    if target_id is not None and target_id in perfiles:
                        usados_cal.add(target_id)
                        mat_p = perfiles[target_id].get("matriz_Z_poses", None)
                        if mat_p is not None and len(mat_p) > 0 and mat_p.shape[1] == len(v_cal):
                            perfiles[target_id]["matriz_Z_poses"] = np.vstack([mat_p, v_cal])
                        else:
                            perfiles[target_id]["matriz_Z_poses"] = np.array([v_cal])
                        calibrados.append(nombres_e2.get(target_id, f"ID {target_id}"))
            if calibrados:
                with lock_ia:
                    ctrl_ia["limpiar_buffers"] = True
                toast_msg = f"[C] LUZ CALIBRADA: {', '.join(calibrados)}"
                toast_color = (0, 255, 120)
                toast_hasta = time.time() + 3.0
                print(f"[CALIBRACIÓN] Luz del aula calibrada con tecla [C] para: {', '.join(calibrados)}", flush=True)
            else:
                toast_msg = "[C] Sin rostro detectado para calibrar luz"
                toast_color = (0, 165, 255)
                toast_hasta = time.time() + 2.5
                print("[CALIBRACIÓN] Presione [C] con un rostro visible dentro de la ventana de video.", flush=True)
        elif tecla in [ord("n"), ord("N")]:
            if tracks_visibles:
                candidatos_bloq = [t for t in tracks_visibles if not t.autorizado] or tracks_visibles
                tr_n = candidatos_bloq[0]
                if tr_n.ultimo_img_64 is not None:
                    ruta_neg_img = os.path.join(DIR_CLASE_0, f"objeto_manual_vivo_{int(time.time()*1000)}.png")
                    cv2_imwrite_unicode(ruta_neg_img, (tr_n.ultimo_img_64 * 255.0).astype(np.uint8))
                if tr_n.ultimo_vec_4096 is not None:
                    z_neg = pca_e2.transform(tr_n.ultimo_vec_4096.reshape(1, -1))[0]
                    lista_negativos_vivo.append(z_neg)
                    try:
                        joblib.dump(lista_negativos_vivo, ruta_negativos_pkl)
                    except Exception:
                        pass
                with lock_ia:
                    ctrl_ia["limpiar_tracks"] = True
                ultimo_acceso_valido = 0.0
                toast_msg = "[N] OBJETO DEL FONDO BLOQUEADO EN VIVO"
                toast_color = (0, 120, 255)
                toast_hasta = time.time() + 2.5
                print("[CALIBRACIÓN] Objeto bloqueado en vivo y guardado en clase_0_no_persona con tecla [N].", flush=True)
        elif tecla in NOMBRE_MODELOS_TECLAS:
            modelo_fuerza = NOMBRE_MODELOS_TECLAS[tecla]
            with lock_ia:
                ctrl_ia["modelo_fuerza"] = modelo_fuerza
                ctrl_ia["limpiar_buffers"] = True
            ultimo_acceso_valido = 0.0
            toast_msg = f"ALGORITMO FORZADO: {modelo_fuerza}"
            toast_color = (0, 220, 255)
            toast_hasta = time.time() + 2.5
            print(f">>> Algoritmo forzado en tiempo real: {modelo_fuerza}", flush=True)

    ctrl_ia["activo"] = False
    hilo_ia.join(timeout=1.5)
    cliente_iot.enviar_estado("ESPERA", total_rostros=0, forzar=True)
    lector.liberar()
    cv2.destroyAllWindows()
    print("Demostración concluida limpiamente.")


if __name__ == "__main__":
    ejecutar_demostracion()
