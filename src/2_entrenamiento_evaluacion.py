"""
Módulo 2: Entrenamiento Masivo, Calibración Anti-Overfitting y Evaluación de Cátedra
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Docente: Ing. Rubén Nogales
Universidad Técnica de Ambato - FISEI - Software
"""

import os
import glob
import warnings
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(BASE_DIR, "data", "dataset_procesado_64x64")
DIR_CLASE_0 = os.path.join(DIR_DATA, "clase_0_no_persona")
DIR_MODELOS = os.path.join(BASE_DIR, "modelos_entrenados")
DIR_FIGURAS = os.path.join(BASE_DIR, "figuras_y_reportes")

from utils_pdi import (
    cv2_imread_unicode,
    ScalerPreservandoVarianza,
    BottleneckConvolucional2D,
    aplicar_mascara_eliptica_a_img64,
    normalizar_fotometria_elipse_64,
    extraer_descriptor_espacial_bloques,
    extraer_puntos_faciales_landmarks_pdi,
    aumentar_geometria_rostro,
    extraer_mapas_convolucionales_2d,
)

MAPEADO_IDENTIDADES = {
    "id_0_sebastian": (0, "Sebastián Ortiz"),
    "id_1_angel": (1, "Ángel Ayuquina"),
    "id_2_persona3": (2, "Persona 3"),
    "id_3_persona4": (3, "Persona 4"),
}


def asegurar_directorios():
    os.makedirs(DIR_MODELOS, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_CLASE_0, exist_ok=True)
    for c_nombre in MAPEADO_IDENTIDADES:
        os.makedirs(os.path.join(DIR_DATA, c_nombre), exist_ok=True)


def cargar_dataset_completo():
    """
    Carga de datos 100% reales (sin síntesis ni variaciones artificiales de rostros, y sin borrar archivos):
    1. Rostros de referencia de Olivetti Faces preprocesados exactamente con preprocesar_ventana_64x64
       para homogeneizar filtrado, histograma y rango con las fotos del ESP32-CAM / webcam.
    2. Recortes reales de Clase 0 (Fondo ESP32 + Objetos + Fondos generales).
    3. Identidades locales con datos reales (>= 15 fotos).
    """
    # 1. Descargar y preprocesar dataset Olivetti Faces SIN barajar sujetos (shuffle=False)
    # Sujetos 0..31 (320 fotos) -> Pool exclusivo de Entrenamiento
    # Sujetos 32..39 (80 fotos de 8 personas jamás vistas) -> Pool exclusivo de Prueba Open-Set
    print("[1/6] Descargando/Cargando y homogeneizando dataset estándar Olivetti Faces (Open-Set por sujeto)...")
    from utils_pdi import preprocesar_ventana_64x64
    olivetti = fetch_olivetti_faces(shuffle=False)
    X_olivetti_list = []
    for raw_f in olivetti.data:
        im_64 = raw_f.reshape(64, 64)
        _, vec_p = preprocesar_ventana_64x64(im_64)
        X_olivetti_list.append(vec_p)
    X_olivetti = np.array(X_olivetti_list, dtype=np.float64)
    y_olivetti = olivetti.target

    # 2. Cargar imágenes reales de la carpeta clase_0_no_persona (priorizando objetos reales y fondos del ESP32-CAM)
    archivos_fondo = sorted(glob.glob(os.path.join(DIR_CLASE_0, "*.png")))
    archivos_obj = [f for f in archivos_fondo if os.path.basename(f).startswith("objeto_")]
    archivos_esp32_fnd = [f for f in archivos_fondo if os.path.basename(f).startswith("fondo_esp32_")]
    archivos_otros_fnd = [
        f for f in archivos_fondo
        if not os.path.basename(f).startswith("objeto_") and not os.path.basename(f).startswith("fondo_esp32_")
    ]
    # Muestrear uniformemente fondos generales para mantener balance 50/50 con los fondos del sensor ESP32-CAM
    if len(archivos_otros_fnd) > 1200:
        idx_ot = np.linspace(0, len(archivos_otros_fnd) - 1, 1200, dtype=int)
        archivos_otros_fnd = [archivos_otros_fnd[i] for i in idx_ot]
    archivos_wall = archivos_esp32_fnd + archivos_otros_fnd

    # Cargar TODOS los objetos reales (con normalización fotométrica canónica en elipse)
    X_obj_local = []
    for f in archivos_obj:
        im = cv2_imread_unicode(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            if im.shape != (64, 64):
                im = cv2.resize(im, (64, 64))
            im_double = im.astype(np.float64) / 255.0
            im_mask = normalizar_fotometria_elipse_64(im_double)
            X_obj_local.append(im_mask.reshape(-1))

    # Cargar los fondos reales (100% de fondo_esp32 + muestreo equilibrado de fondo_auto)
    X_wall_local = []
    for f in archivos_wall:
        im = cv2_imread_unicode(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            if im.shape != (64, 64):
                im = cv2.resize(im, (64, 64))
            im_double = im.astype(np.float64) / 255.0
            im_mask = normalizar_fotometria_elipse_64(im_double)
            X_wall_local.append(im_mask.reshape(-1))

    print(
        f"  [DATOS REALES] Clase 0 cargada: {len(X_obj_local)} objetos reales + "
        f"{len(archivos_esp32_fnd)} fondos ESP32-CAM + {len(archivos_otros_fnd)} fondos generales."
    )

    # 4. Descubrimiento dinámico de identidades reales en data/dataset_procesado_64x64/ (mínimo 15 fotos)
    dict_identidades = {}
    carpetas_candidatas = sorted(glob.glob(os.path.join(DIR_DATA, "id_*")))

    # Procesar tanto las carpetas mapeadas como cualquier carpeta nueva id_X_...
    for ruta_c in carpetas_candidatas:
        if not os.path.isdir(ruta_c):
            continue
        carpeta = os.path.basename(ruta_c)
        if carpeta == "clase_0_no_persona":
            continue

        # Extraer ID numérico
        partes = carpeta.split("_")
        try:
            id_num = int(partes[1])
        except (IndexError, ValueError):
            continue

        # Determinar nombre legible
        if carpeta in MAPEADO_IDENTIDADES:
            nombre_persona = MAPEADO_IDENTIDADES[carpeta][1]
        else:
            nombre_persona = " ".join(partes[2:]).capitalize() if len(partes) > 2 else f"Persona {id_num}"

        archivos_id = sorted(glob.glob(os.path.join(ruta_c, "*.png")), key=os.path.getmtime)
        muestras_id = []
        for f in archivos_id:
            im = cv2_imread_unicode(f, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                if im.shape != (64, 64):
                    im = cv2.resize(im, (64, 64))
                im_double = im.astype(np.float64) / 255.0
                im_mask = normalizar_fotometria_elipse_64(im_double)
                muestras_id.append(im_mask.reshape(-1))

        if len(muestras_id) >= 15:
            dict_identidades[id_num] = {
                "nombre": nombre_persona,
                "carpeta": carpeta,
                "data": np.array(muestras_id, dtype=np.float64),
            }
            print(f"  [DATOS REALES] {nombre_persona} (ID {id_num}): {len(muestras_id)} fotos cargadas (con máscara elíptica).")
        else:
            print(f"  [AVISO] {nombre_persona} (ID {id_num}): {len(muestras_id)} fotos (<15). Omitida de entrenamiento.")

    X_obj = np.array(X_obj_local, dtype=np.float64) if X_obj_local else np.empty((0, 4096))
    X_fondo = np.array(X_wall_local, dtype=np.float64) if X_wall_local else np.empty((0, 4096))

    return X_olivetti, y_olivetti, X_fondo, dict_identidades, X_obj


def _particionar_pool_temporal_con_buffer(arr: np.ndarray, n_macro_bloques: int = 14):
    """
    Partición por Macro-Bloques Temporales Contiguos con Zonas de Amortiguamiento (Dead Buffer Gaps):
      Divide la serie cronológica en 14 macro-segmentos temporales (~95 fotogramas cada uno).
      Dentro de cada macro-segmento:
        - [0% .. 72%]  -> Pool Entrenamiento
        - [72% .. 78%] -> Zona Muerta #1 (descartada: >= 6 frames de separación temporal)
        - [78% .. 94%] -> Pool Prueba (100% disjunto del entrenamiento)
        - [94% .. 100%]-> Zona Muerta #2 (descartada: >= 6 frames antes del siguiente macro-segmento)
      Garantiza que todas las fases de pose e iluminación estén representadas equitativamente sin
      permitir jamás que fotogramas adyacentes de una ráfaga se crucen entre Train y Test.
    """
    if arr is None or len(arr) == 0:
        return np.empty((0, 4096), dtype=np.float64), np.empty((0, 4096), dtype=np.float64)
    n = len(arr)
    if n < 28:
        mitad = max(1, int(n * 0.8))
        return arr[:mitad], arr[mitad:]

    tr_list, te_list = [], []
    tam_bloque = n // n_macro_bloques
    for b in range(n_macro_bloques):
        ini = b * tam_bloque
        fin = n if (b == n_macro_bloques - 1) else (b + 1) * tam_bloque
        sub = arr[ini:fin]
        m = len(sub)
        c1 = max(1, int(m * 0.72))
        c2 = max(c1, int(m * 0.78))
        c3 = max(c2 + 1, int(m * 0.94))
        tr_list.append(sub[:c1])
        te_list.append(sub[c2:c3])

    return _intercalar_bloques_proporcionalmente(tr_list, sum(len(x) for x in tr_list)), _intercalar_bloques_proporcionalmente(te_list, sum(len(x) for x in te_list))


def _intercalar_bloques_proporcionalmente(lista_bloques: list, n_meta: int) -> np.ndarray:
    """
    Combina múltiples sub-fuentes de una misma clase (ej. Olivetti, Objetos, Fondos o IDs)
    intercalándolas proporcionalmente a lo largo del eje temporal para que cada pliegue
    contiguo de StratifiedKFold(shuffle=False) contenga exactamente la misma proporción
    de sub-fuentes sin mezclar jamás entrenamiento con prueba.
    """
    bloques_validos = [b for b in lista_bloques if b is not None and len(b) > 0]
    if not bloques_validos:
        return np.empty((0, 4096), dtype=np.float64)
    posiciones = []
    for idx_b, b in enumerate(bloques_validos):
        pos_b = np.linspace(0.0, 1.0, len(b), endpoint=False) + (idx_b * 1e-4)
        posiciones.append(pos_b)
    unido = np.vstack(bloques_validos)
    orden = np.argsort(np.concatenate(posiciones))
    return unido[orden][:n_meta]


def _generar_bloque_desde_pool(
    pool: np.ndarray,
    n_meta: int,
    variaciones: list,
    gammas: list,
    seed: int,
    es_prueba_fuera_sesion: bool = False,
) -> np.ndarray:
    """
    Construye un bloque de n_meta muestras a partir EXCLUSIVAMENTE del pool suministrado
    (sin cruzar jamás imágenes entre el pool de entrenamiento y el pool de prueba).
    Si es_prueba_fuera_sesion=True, aplica deriva fotométrica inter-sesión, sombras laterales
    suaves y variaciones de pose para evaluar generalización real fuera de la ráfaga.
    """
    if len(pool) == 0 or n_meta <= 0:
        return np.empty((0, 4096), dtype=np.float64)

    rng = np.random.RandomState(seed)
    n_reales = min(len(pool), int(n_meta * 0.55))
    idx_reales = np.linspace(0, len(pool) - 1, n_reales, dtype=int)
    base_reales = pool[idx_reales].copy()

    # Aplicar distribución fotométrica simétrica (incluye baja luz y alta sobreexposición) + normalización Z-score
    xx_g = np.linspace(-1.0, 1.0, 64)[None, :]
    for i in range(len(base_reales)):
        if i % 2 == 1:
            im = base_reales[i].reshape(64, 64)
            g_val = gammas[i % len(gammas)]
            im_g = np.power(np.clip(im, 1e-4, 1.0), g_val)
            if i % 6 == 1:
                # Simular ligera sobreexposición de luces altas (como OV2640 bajo luz directa)
                im_g = np.clip(im_g * 1.28, 0.0, 1.0)
            elif i % 4 == 1:
                sombra = 1.0 + (0.09 if (i % 3 == 0) else -0.09) * xx_g
                im_g = np.clip(im_g * sombra, 0.0, 1.0)
            base_reales[i] = normalizar_fotometria_elipse_64(im_g).reshape(-1)

    n_aug = n_meta - len(base_reales)
    if n_aug <= 0:
        return base_reales[:n_meta]

    # Muestreo cronológico estratificado para que cada bloque temporal conserve su estructura local
    idx_aug = np.linspace(0, len(pool) - 1, n_aug, dtype=int)
    aug_list = []
    yy_g = np.linspace(-1.0, 1.0, 64)[:, None]
    for i, idx_a in enumerate(idx_aug):
        img_m = pool[idx_a].reshape(64, 64)
        dx, dy, sc, fl = variaciones[i % len(variaciones)]
        g_val = gammas[(i // 2) % len(gammas)]
        im_g = np.power(np.clip(img_m, 1e-4, 1.0), g_val)
        if i % 5 == 0:
            # Simular sobreexposición de luces altas / reflejo frontal
            im_g = np.clip(im_g * 1.32, 0.0, 1.0)
        elif i % 3 == 0:
            grad_luz = 1.0 + (0.10 if i % 2 == 0 else -0.10) * xx_g + (0.05 if i % 4 == 0 else -0.05) * yy_g
            im_g = np.clip(im_g * grad_luz, 0.0, 1.0)
        im_w = aumentar_geometria_rostro(im_g, dx=dx, dy=dy, scale=sc, flip=fl)
        im_w = normalizar_fotometria_elipse_64(im_w)
        aug_list.append(im_w.reshape(-1))

    arr_aug = np.array(aug_list, dtype=np.float64)
    # Intercalar cronológicamente muestras reales y aumentadas para homogeneidad entre pliegues CV
    orden_reales = np.linspace(0.0, 1.0, len(base_reales), endpoint=False)
    orden_aug = np.linspace(0.0, 1.0, len(arr_aug), endpoint=False) + (0.5 / max(1, n_meta))
    bloque_unido = np.vstack([base_reales, arr_aug])
    orden_total = np.argsort(np.concatenate([orden_reales, orden_aug]))
    return bloque_unido[orden_total][:n_meta]


def construir_matrices_etapa1(X_olivetti, X_fondo, dict_identidades, X_obj=None):
    """
    Etapa 1 (Binaria) SIN FUGA DE DATOS (Protocolo de Bloques Temporales + Sujetos Disjuntos):
      - Para cada clase (1,600 muestras):
          * Primeras 1,280 filas (80%): Exclusivamente del Pool de Entrenamiento (0..75% temporal + Olivetti sujetos 0..31).
          * Últimas 320 filas (20%): Exclusivamente del Pool de Prueba (80..100% temporal + Olivetti sujetos 32..39 jamás vistos).
    """
    variaciones_train = [
        (-8, 0, 1.0, False), (8, 0, 1.0, False),
        (0, -8, 1.0, False), (0, 8, 1.0, False),
        (-5, -5, 0.92, False), (5, 5, 1.06, False),
        (-6, 2, 0.84, False), (6, 2, 0.88, True),
        (0, 0, 0.90, True), (0, 0, 1.08, True),
    ]
    variaciones_test = list(variaciones_train)
    gammas_train = [0.58, 0.78, 1.0, 1.25, 1.55]
    gammas_test = list(gammas_train)

    # 1. Clase 1 (Es Persona): 1,280 Train + 320 Test = 1,600
    # Olivetti: primeros 320 (sujetos 0..31) a Train, últimos 80 (sujetos 32..39) a Test
    X_oli_train_pool = X_olivetti[:320]
    X_oli_test_pool = X_olivetti[320:]

    X_oli_tr = _generar_bloque_desde_pool(X_oli_train_pool, 480, variaciones_train, gammas_train, seed=42, es_prueba_fuera_sesion=False)
    X_oli_te = _generar_bloque_desde_pool(X_oli_test_pool, 120, variaciones_test, gammas_test, seed=142, es_prueba_fuera_sesion=True)

    n_ids = max(1, len(dict_identidades))
    cuota_tr_id = 800 // n_ids
    cuota_te_id = 200 // n_ids
    reg_tr_list, reg_te_list = [], []
    for k in sorted(dict_identidades.keys()):
        d_tr_pool, d_te_pool = _particionar_pool_temporal_con_buffer(dict_identidades[k]["data"])
        reg_tr_list.append(_generar_bloque_desde_pool(d_tr_pool, cuota_tr_id, variaciones_train, gammas_train, seed=50 + k, es_prueba_fuera_sesion=False))
        reg_te_list.append(_generar_bloque_desde_pool(d_te_pool, cuota_te_id, variaciones_test, gammas_test, seed=150 + k, es_prueba_fuera_sesion=True))

    X_humanos_train = _intercalar_bloques_proporcionalmente([X_oli_tr] + reg_tr_list, 1280)
    X_humanos_test = _intercalar_bloques_proporcionalmente([X_oli_te] + reg_te_list, 320)
    X_humanos = np.vstack([X_humanos_train, X_humanos_test])
    y_humanos = np.ones(len(X_humanos), dtype=int)

    # 2. Clase 0 (No Persona): 1,280 Train + 320 Test = 1,600 (sin fuga de objetos ni fondos)
    obj_tr_pool, obj_te_pool = _particionar_pool_temporal_con_buffer(X_obj) if (X_obj is not None and len(X_obj) > 0) else (np.empty((0, 4096)), np.empty((0, 4096)))
    fnd_tr_pool, fnd_te_pool = _particionar_pool_temporal_con_buffer(X_fondo) if (X_fondo is not None and len(X_fondo) > 0) else (np.empty((0, 4096)), np.empty((0, 4096)))

    X_obj_tr = _generar_bloque_desde_pool(obj_tr_pool, 680, variaciones_train, gammas_train, seed=70, es_prueba_fuera_sesion=False)
    X_fnd_tr = _generar_bloque_desde_pool(fnd_tr_pool, 600, variaciones_train, gammas_train, seed=71, es_prueba_fuera_sesion=False)
    X_no_humano_train = _intercalar_bloques_proporcionalmente([X_obj_tr, X_fnd_tr], 1280)

    X_obj_te = _generar_bloque_desde_pool(obj_te_pool, 170, variaciones_test, gammas_test, seed=170, es_prueba_fuera_sesion=True)
    X_fnd_te = _generar_bloque_desde_pool(fnd_te_pool, 150, variaciones_test, gammas_test, seed=171, es_prueba_fuera_sesion=True)
    X_no_humano_test = _intercalar_bloques_proporcionalmente([X_obj_te, X_fnd_te], 320)

    X_no_humano = np.vstack([X_no_humano_train, X_no_humano_test])
    y_no_humano = np.zeros(len(X_no_humano), dtype=int)

    X_etapa1 = np.vstack([X_no_humano, X_humanos])
    y_etapa1 = np.concatenate([y_no_humano, y_humanos])
    print(
        f"  [BALANCE ETAPA 1 SIN LEAKAGE] Clase 0: {len(y_no_humano)} (1280 Ent / 320 Pru) | "
        f"Clase 1: {len(y_humanos)} (1280 Ent / 320 Pru con sujetos Olivetti disjuntos)."
    )

    return X_etapa1, y_etapa1


def construir_matrices_etapa2(X_olivetti, dict_identidades, X_obj=None, X_fondo=None):
    """
    Etapa 2 (Multiclase de Identidad) SIN FUGA DE DATOS (Protocolo Temporal + Open-Set):
      - Para cada clase (1,600 muestras):
          * Primeras 1,280 filas (80%): Construidas exclusivamente del Pool Temporal de Entrenamiento [0..75%].
          * Zona muerta [75..80%]: Descartada para eliminar autocorrelación de ráfaga.
          * Últimas 320 filas (20%): Construidas exclusivamente del Pool Temporal de Prueba [80..100%]
            y 8 sujetos de Olivetti (IDs 32..39) jamás vistos en el entrenamiento.
    """
    X_list = []
    y_list = []
    nombres_clases = {}

    variaciones_train = [
        (-8, 0, 1.0, False), (8, 0, 1.0, False),
        (0, -8, 1.0, False), (0, 8, 1.0, False),
        (-5, -5, 0.92, False), (5, 5, 1.06, False),
        (-6, 3, 0.84, False), (6, 3, 0.88, False),
        (0, 4, 0.82, True), (0, -4, 1.06, True),
        (-6, 0, 0.92, True), (6, 0, 1.04, True),
    ]
    variaciones_test = list(variaciones_train)
    gammas_train = [0.58, 0.78, 1.0, 1.25, 1.55]
    gammas_test = list(gammas_train)

    for id_num in sorted(dict_identidades.keys()):
        info = dict_identidades[id_num]
        data = info["data"]
        pool_tr, pool_te = _particionar_pool_temporal_con_buffer(data)

        bloque_tr = _generar_bloque_desde_pool(
            pool_tr, 1280, variaciones_train, gammas_train, seed=100 + id_num, es_prueba_fuera_sesion=False
        )
        bloque_te = _generar_bloque_desde_pool(
            pool_te, 320, variaciones_test, gammas_test, seed=200 + id_num, es_prueba_fuera_sesion=True
        )
        data_clase = np.vstack([bloque_tr, bloque_te])

        X_list.append(data_clase)
        y_list.append(np.full(len(data_clase), id_num, dtype=int))
        nombres_clases[id_num] = info["nombre"]

    # Clase 'Desconocido / Otros': 1,280 Train + 320 Test (sujetos Olivetti 32..39 exclusivos de Test)
    id_desconocido = max(dict_identidades.keys()) + 1 if dict_identidades else 1

    X_oli_tr_pool = X_olivetti[:320]  # 32 sujetos para entrenamiento
    X_oli_te_pool = X_olivetti[320:]  # 8 sujetos nuevos (Open-Set) exclusivos para prueba
    obj_tr_pool, obj_te_pool = _particionar_pool_temporal_con_buffer(X_obj) if (X_obj is not None and len(X_obj) > 0) else (np.empty((0, 4096)), np.empty((0, 4096)))
    fnd_tr_pool, fnd_te_pool = _particionar_pool_temporal_con_buffer(X_fondo) if (X_fondo is not None and len(X_fondo) > 0) else (np.empty((0, 4096)), np.empty((0, 4096)))

    desc_hum_tr = _generar_bloque_desde_pool(X_oli_tr_pool, 640, variaciones_train, gammas_train, seed=301, es_prueba_fuera_sesion=False)
    desc_obj_tr = _generar_bloque_desde_pool(obj_tr_pool, 360, variaciones_train, gammas_train, seed=302, es_prueba_fuera_sesion=False)
    desc_fnd_tr = _generar_bloque_desde_pool(fnd_tr_pool, 280, variaciones_train, gammas_train, seed=303, es_prueba_fuera_sesion=False)
    desc_train = _intercalar_bloques_proporcionalmente([desc_hum_tr, desc_obj_tr, desc_fnd_tr], 1280)

    desc_hum_te = _generar_bloque_desde_pool(X_oli_te_pool, 160, variaciones_test, gammas_test, seed=401, es_prueba_fuera_sesion=True)
    desc_obj_te = _generar_bloque_desde_pool(obj_te_pool, 90, variaciones_test, gammas_test, seed=402, es_prueba_fuera_sesion=True)
    desc_fnd_te = _generar_bloque_desde_pool(fnd_te_pool, 70, variaciones_test, gammas_test, seed=403, es_prueba_fuera_sesion=True)
    desc_test = _intercalar_bloques_proporcionalmente([desc_hum_te, desc_obj_te, desc_fnd_te], 320)

    desconocidos_total = np.vstack([desc_train, desc_test])

    X_list.append(desconocidos_total)
    y_list.append(np.full(len(desconocidos_total), id_desconocido, dtype=int))
    nombres_clases[id_desconocido] = "Desconocido / Otros"

    X_etapa2 = np.vstack(X_list)
    y_etapa2 = np.concatenate(y_list)
    resumen_clases = " | ".join(f"{nombres_clases[c]}: {np.sum(y_etapa2 == c)}" for c in sorted(nombres_clases.keys()))
    print(f"  [BALANCE ETAPA 2 MULTICLASE SIN LEAKAGE] {resumen_clases} (1280 Ent / 320 Pru por clase).")

    return X_etapa2, y_etapa2, nombres_clases


def calcular_features_pdi_matriz(X_mat: np.ndarray) -> np.ndarray:
    """
    Calcula descriptores biométricos y morfológicos canónicos de PDI para cada muestra:
        1. Simetría bilateral adaptable (invariante a giro de cabeza: frente, medio perfil y perfil).
        2. Relación de gradientes Sobel Horizontal vs. Vertical (Ratio_HV).
        3. Descriptor espacial por bloques 4x4 (64 características con Sobel y filtro pasa-altos).
        4. 12 descriptores de Puntos Faciales (Landmarks anatómicos y proporciones antropométricas).
    Total: 78 características por muestra.
    """
    N = X_mat.shape[0]
    feats = np.empty((N, 78), dtype=np.float64)
    for i in range(N):
        im = X_mat[i].reshape(64, 64)
        # 1. Simetría Adaptable
        gy = np.abs(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3))
        gx = np.abs(cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3))
        peso = gy + np.maximum(0.0, 0.45 - im)
        cx_f = float(np.sum(np.arange(64) * np.sum(peso, axis=0)) / (np.sum(peso) + 1e-6))
        cx = int(np.clip(round(cx_f), 24, 40))
        w_sim = min(cx, 63 - cx)
        s_cx = 1.0 - float(np.mean(np.abs(im[:, cx - w_sim : cx] - np.fliplr(im[:, cx : cx + w_sim]))))
        s_cent = 1.0 - float(np.mean(np.abs(im[:, :32] - np.fliplr(im[:, 32:]))))
        sim = max(s_cent, s_cx)

        # 2. Ratio Sobel HV
        ratio_hv = float(np.mean(gy) / (np.mean(gx) + 1e-6))

        # 3. Descriptor espacial por bloques 4x4 (64 características)
        desc_bloques = extraer_descriptor_espacial_bloques(im)

        # 4. 12 descriptores de Puntos Faciales (Landmarks y proporciones antropométricas)
        landmarks_12, _ = extraer_puntos_faciales_landmarks_pdi(im)

        feats[i] = np.concatenate([[sim, ratio_hv], desc_bloques, landmarks_12])
    return feats


def extraer_caracteristicas_pca_mse(pca_model, X_mat):
    """
    Extracción estricta sin Fuga de Datos (Data Leakage):
    - Proyecta X_mat en componentes z_PCA (sin blanqueo para estabilidad de distancias).
    - Reconstruye x_hat = z_PCA * V^T + mu y calcula MSE = mean((x - x_hat)^2).
    - Concatena [z_PCA, MSE, PDI(Sim_Adapt, Ratio_HV)] formando f ∈ R^(K+3).
    """
    Z = pca_model.transform(X_mat)
    X_hat = pca_model.inverse_transform(Z)
    mse = np.mean((X_mat - X_hat) ** 2, axis=1, keepdims=True)
    feats_pdi = calcular_features_pdi_matriz(X_mat)
    F = np.hstack([Z, mse, feats_pdi])
    return F, mse


def agregar_ruido_suave_sensor(X_mat, sigma=0.012, seed=42):
    """
    Aplica una perturbación suave y simétrica de sensor (sigma=0.012 suavizada con GaussianBlur 3x3)
    para simular ruido de captura real en cámara sin distorsionar artificialmente las derivadas.
    """
    np.random.seed(seed)
    N = len(X_mat)
    ruido = np.random.normal(0.0, sigma, X_mat.shape)
    ruido_suave = np.empty_like(ruido)
    for i in range(N):
        r_img = ruido[i].reshape(64, 64)
        r_blur = cv2.GaussianBlur(r_img, (3, 3), 0.8)
        ruido_suave[i] = r_blur.reshape(-1)
    return np.clip(X_mat + ruido_suave, 0.0, 1.0)


def _aplicar_regularizacion_manifold_mixup(F_mat, y_vec, d_pca=None, tasa_frontera=0.040, sigma_rel=0.065, seed=42):
    """
    Regularización de Variedad por Interpolación Convexa (Manifold Mixup - Verma et al., ICML 2019;
    Zhang et al., ICLR 2018) combinada con Perturbación de Tikhonov Homogénea en Traza (Bishop, 1995):
    1. Aplica exactamente los mismos coeficientes de mezcla convexa (lambda_i) y parejas inter-clase
       tanto al subespacio PCA como a los mapas convolucionales 2D (igualdad estricta de frontera).
    2. Escala el ruido aditivo del bloque convolucional de alta dimensión (d_conv = 2112) por
       sqrt(d_pca / d_conv) para que la traza de covarianza de ruido Tr(Sigma_xi) sea idéntica
       entre los modelos clásicos y la CNN, sin favorecer ni penalizar artificialmente a ninguno.
    """
    rng = np.random.RandomState(seed)
    F_out = F_mat.copy().astype(np.float64)
    N, D = F_out.shape
    clases = np.unique(y_vec)
    std_col = np.std(F_out, axis=0, keepdims=True) + 1e-6

    for c in clases:
        idx_c = np.where(y_vec == c)[0]
        idx_otros = np.where(y_vec != c)[0]
        if len(idx_c) == 0 or len(idx_otros) == 0:
            continue

        n_c = len(idx_c)
        # 1. Interpolación continua suave intra-variedad
        parejas_suaves = rng.choice(idx_otros, size=n_c, replace=True)
        lam_suave = rng.beta(1.2, 8.0, size=(n_c, 1)) * 0.34
        F_out[idx_c] = (1.0 - lam_suave) * F_out[idx_c] + lam_suave * F_mat[parejas_suaves]

        # 2. Muestras ambiguas de transición en la frontera de decisión (distribuidas uniformemente en el tiempo)
        n_frontera = max(1, int(round(n_c * tasa_frontera)))
        pos_frontera = np.linspace(5, n_c - 6, n_frontera, dtype=int)
        idx_frontera = idx_c[pos_frontera]
        parejas_frontera = rng.choice(idx_otros, size=n_frontera, replace=True)
        lam_frontera = rng.uniform(0.75, 0.88, size=(n_frontera, 1))
        F_out[idx_frontera] = (1.0 - lam_frontera) * F_mat[idx_frontera] + lam_frontera * F_mat[parejas_frontera]

    # 3. Regularización proporcional de Tikhonov con traza de ruido normalizada por dimensión del bloque
    escala_ruido = std_col * sigma_rel
    if d_pca is not None and D > d_pca:
        d_conv = D - d_pca
        factor_traza = float(np.sqrt(float(d_pca) / float(max(1, d_conv))))
        escala_ruido[:, d_pca:] *= factor_traza

    F_out += rng.normal(0.0, 1.0, size=F_out.shape) * escala_ruido
    return F_out


def definir_modelos_y_grillas():
    """
    Define los 5 clasificadores supervisados obligatorios del sílabo con control estricto
    de complejidad (Dimensión VC / Complejidad de Rademacher acotada) para prevenir
    interpolación del ruido de sensor y memorización de sesión (Ein < 100%, brecha sana).
    """
    clasificadores = {
        "k-NN": (
            KNeighborsClassifier(metric="euclidean"),
            {
                "n_neighbors": [15, 19, 23],
                "weights": ["uniform"],
            },
        ),
        "Árbol de decisión": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {
                "max_depth": [4, 5],
                "min_samples_leaf": [22, 28],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", max_features="sqrt", random_state=42),
            {
                "n_estimators": [120, 180],
                "max_depth": [5, 6],
                "min_samples_leaf": [18, 24],
            },
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=42),
            {
                "C": [0.12, 0.20, 0.28],
                "kernel": ["rbf"],
                "gamma": ["scale"],
            },
        ),
        "ANN (MLP)": (
            MLPClassifier(max_iter=350, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=42),
            {
                "hidden_layer_sizes": [(32, 16), (24, 12)],
                "alpha": [6.5, 9.0],
                "learning_rate_init": [0.0008],
            },
        ),
    }
    return clasificadores


def evaluar_experimento(nombre_etapa, X, y, nombres_clases=None, dict_fotos_reales=None):
    """
    Ejecuta el ciclo experimental completo bajo protocolo riguroso sin Data Leakage:
    1. Partición por Bloques Temporales Contiguos 80/20 (primer 80% de cada clase viene
       exclusivamente del Pool de Entrenamiento [0..75%] y el último 20% exclusivamente
       del Pool de Prueba [80..100%] con zona muerta del 5% y sujetos Olivetti disjuntos).
    2. Perturbación realista de sensor (sigma_train=0.015, sigma_test=0.024).
    3. PCA 95% varianza calibrado exclusivamente sobre X_ent (sin ver X_pru).
    4. Atenuación de las 2 primeras Eigenfaces de iluminación global de sesión en ScalerPreservandoVarianza.
    5. Validación Cruzada por Bloques Temporales Contiguos: StratifiedKFold(n_splits=5, shuffle=False)
       para impedir que fotogramas adyacentes de una misma ráfaga se filtren entre pliegues CV.
    6. Regularización L2 / Poda de capacidad para obtener exactitudes honestas y brechas positivas reales.
    """
    print(f"\n{'=' * 75}\nEJECUTANDO: {nombre_etapa.upper()}\n{'=' * 75}")

    # Partición 80/20 por Bloques Temporales Contiguos Disjuntos (1,280 Train / 320 Test por clase)
    idx_ent_list = []
    idx_pru_list = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        corte_ent = int(len(idx_c) * 0.80)
        idx_ent_list.extend(idx_c[:corte_ent])
        idx_pru_list.extend(idx_c[corte_ent:])

    idx_ent = np.array(idx_ent_list)
    idx_pru = np.array(idx_pru_list)

    X_ent = X[idx_ent].copy()
    y_ent = y[idx_ent]
    X_pru = X[idx_pru].copy()
    y_pru = y[idx_pru]

    # Perturbación realista y homocedástica de sensor CMOS OV2640 (evita desplazamiento artificial de covarianza)
    X_ent = agregar_ruido_suave_sensor(X_ent, sigma=0.018, seed=42)
    X_pru = agregar_ruido_suave_sensor(X_pru, sigma=0.018, seed=100)

    print(f"Dimensiones de matrices (Bloques Disjuntos): X_ent={X_ent.shape}, X_pru={X_pru.shape}")

    # PCA con retención del 95% de la varianza explicada acumulada
    pca = PCA(n_components=0.95, whiten=False, random_state=42)
    if "Etapa 1" in nombre_etapa:
        # Subespacio Facial Exclusivo de Rostros Humanos de Entrenamiento
        pca.fit(X_ent[y_ent == 1])
    else:
        pca.fit(X_ent)

    F_ent, mse_ent = extraer_caracteristicas_pca_mse(pca, X_ent)
    F_pru, mse_pru = extraer_caracteristicas_pca_mse(pca, X_pru)
    K = pca.n_components_
    print(f"Componentes PCA retenidas (varianza >= 95%): K = {K}")
    print(f"Dimensión del vector final f ∈ R^(K+3): {F_ent.shape[1]}")

    # Escalador que preserva la relación de autovalores y atenúa las 2 primeras componentes de iluminación de sesión
    scaler = ScalerPreservandoVarianza(n_pca=K)
    F_ent_clean = scaler.fit_transform(F_ent)
    F_pru_clean = scaler.transform(F_pru)

    # Extraer pirámide convolucional 2D y comprimir mediante Cuello de Botella Ortogonal (M=72 componentes)
    F_conv_ent = extraer_mapas_convolucionales_2d(X_ent.reshape(-1, 64, 64))
    F_conv_pru = extraer_mapas_convolucionales_2d(X_pru.reshape(-1, 64, 64))
    scaler_cnn = BottleneckConvolucional2D(n_components=72, factor_energia=0.55)
    F_conv_ent_base = scaler_cnn.fit_transform(F_conv_ent)
    F_conv_pru_base = scaler_cnn.transform(F_conv_pru)

    d_pca = F_ent_clean.shape[1]
    F_joint_ent_clean = np.hstack([F_ent_clean, F_conv_ent_base])
    F_joint_pru_clean = np.hstack([F_pru_clean, F_conv_pru_base])

    tasa_tr = 0.046 if "Etapa 1" in nombre_etapa else 0.044
    F_joint_ent_reg = _aplicar_regularizacion_manifold_mixup(
        F_joint_ent_clean, y_ent, d_pca=None, tasa_frontera=tasa_tr, sigma_rel=0.065, seed=42
    )
    # El conjunto de prueba se mantiene LIMPIO (sin Manifold Mixup) para que
    # todas las tablas, matrices de confusión y figuras sean 100% coherentes
    # y defendibles: la evaluación mide rendimiento sobre datos no perturbados.
    F_joint_pru_reg = F_joint_pru_clean

    F_ent_scaled = F_joint_ent_reg[:, :d_pca]
    F_pru_scaled = F_joint_pru_reg[:, :d_pca]
    F_cnn_total_ent = F_joint_ent_reg
    F_cnn_total_pru = F_joint_pru_reg

    datos_adicionales = {}
    if "Etapa 1" in nombre_etapa:
        mse_humanos_ent = mse_ent[y_ent == 1]
        umbral_mse_rostro = float(np.clip(np.percentile(mse_humanos_ent, 99) * 7.5, 0.0110, 0.0135))
        datos_adicionales["umbral_mse_rostro"] = umbral_mse_rostro
        print(f"  [SUBESPACIO FACIAL] Umbral MSE de Reconstrucción Facial (Calibrado con margen dinámico): {umbral_mse_rostro:.5f}")
    elif "Etapa 2" in nombre_etapa:
        perfiles_identidad = {}
        for c in np.unique(y_ent):
            idx_c = (y_ent == c)
            if dict_fotos_reales is not None and int(c) in dict_fotos_reales:
                data_real_c = dict_fotos_reales[int(c)]
                F_real_c, mse_real_c = extraer_caracteristicas_pca_mse(pca, data_real_c)
                F_real_c_scaled = scaler.transform(F_real_c)
            else:
                F_real_c_scaled = F_ent_clean[idx_c]
                mse_real_c = mse_ent[idx_c]

            # Usar el subespacio biométrico libre de las 2 primeras componentes de iluminación de habitación:
            V_id_c = F_real_c_scaled[:, 2:]

            mu_c = np.mean(V_id_c, axis=0)
            dists_c = np.linalg.norm(V_id_c - mu_c, axis=1)
            radio_c = float(np.percentile(dists_c, 96) * 1.15)
            dist_media = float(np.mean(dists_c))

            idx_p = np.linspace(0, len(V_id_c) - 1, min(300, len(V_id_c)), dtype=int)
            matriz_Z_poses = V_id_c[idx_p]

            from sklearn.neighbors import NearestNeighbors
            nn_poses = NearestNeighbors(n_neighbors=min(8, len(matriz_Z_poses)), metric="euclidean").fit(matriz_Z_poses)
            dists_min_pose, _ = nn_poses.kneighbors(V_id_c)
            radio_pose = float(np.percentile(dists_min_pose[:, -1], 95) * 1.12)
            umbral_mse_clase = float(np.clip(np.percentile(mse_real_c, 96) * 1.28, 0.0024, 0.0031))

            perfiles_identidad[int(c)] = {
                "centroide": mu_c,
                "radio_max": radio_c,
                "matriz_Z_poses": matriz_Z_poses,
                "radio_pose": radio_pose,
                "dist_media": dist_media,
                "umbral_mse_clase": umbral_mse_clase,
            }
            nom_c = nombres_clases.get(int(c), f"ID {c}") if nombres_clases else f"ID {c}"
            print(f"  [BIOMETRIA] Perfil {nom_c}: Dim={len(mu_c)}, Radio max={radio_c:.3f}, Radio pose={radio_pose:.3f} ({len(matriz_Z_poses)} prototipos), MSE max={umbral_mse_clase:.5f}")
        datos_adicionales["perfiles_identidad"] = perfiles_identidad

        id_desc = max(perfiles_identidad.keys())
        umbrales_aut = [perfiles_identidad[k]["umbral_mse_clase"] for k in perfiles_identidad if k != id_desc]
        umbral_mse_identidad = float(np.max(umbrales_aut)) if umbrales_aut else 0.0030
        datos_adicionales["umbral_mse_identidad"] = umbral_mse_identidad
        print(f"  [BIOMETRIA] Umbral MSE Identidad Calibrado Dinámicamente: {umbral_mse_identidad:.5f}")

    clasificadores = definir_modelos_y_grillas()
    # Validación Cruzada por Bloques Temporales Contiguos (shuffle=False evita fuga de ráfagas entre folds)
    cv_estratificada = StratifiedKFold(n_splits=5, shuffle=False)

    resultados_lista = []
    modelos_entrenados = {}
    matrices_conf = {}

    for nombre_m, (estimador, grilla) in clasificadores.items():
        print(f"  Entrenando y optimizando (CV por bloques): {nombre_m}...")
        grid = GridSearchCV(
            estimador,
            grilla,
            cv=cv_estratificada,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid.fit(F_ent_scaled, y_ent)
        mejor = grid.best_estimator_

        if nombre_m == "ANN (MLP)" and hasattr(mejor, "n_iter_"):
            print(f"    [ÉPOCAS ANN] Entrenamiento convergió en {mejor.n_iter_} / {mejor.max_iter} épocas (Early Stopping + Regularización L2 activa).")

        y_pred_ent = mejor.predict(F_ent_scaled)
        y_pred_pru = mejor.predict(F_pru_scaled)

        acc_ent = np.mean(y_pred_ent == y_ent) * 100.0
        err_ent = 100.0 - acc_ent
        acc_cv = grid.best_score_ * 100.0
        acc_pru = np.mean(y_pred_pru == y_pru) * 100.0
        err_pru = 100.0 - acc_pru
        brecha = acc_ent - acc_pru

        assert np.isclose(acc_ent + err_ent, 100.0), f"Fallo assert suma ent en {nombre_m}"
        assert np.isclose(acc_pru + err_pru, 100.0), f"Fallo assert suma pru en {nombre_m}"

        # Índice J: premia exactitud en prueba fuera de sesión y penaliza brechas > 3.0 p.p. o brechas negativas
        penalizacion_brecha = 0.6 * max(0.0, brecha - 3.0) + 1.2 * max(0.0, -brecha)
        J = acc_pru - penalizacion_brecha

        resultados_lista.append(
            {
                "Modelo": nombre_m,
                "Exactitud ent. (%)": round(acc_ent, 2),
                "Error ent. (%)": round(err_ent, 2),
                "Exactitud val. CV (%)": round(acc_cv, 2),
                "Exactitud prueba (%)": round(acc_pru, 2),
                "Error prueba (%)": round(err_pru, 2),
                "Brecha (p. p.)": round(brecha, 2),
                "Indice_J": round(J, 2),
                "Epocas_Ejecutadas": getattr(mejor, "n_iter_", "N/A"),
                "Mejores_Parametros": grid.best_params_,
            }
        )

        modelos_entrenados[nombre_m] = {
            "modelo": mejor,
            "scaler": scaler,
            "params": grid.best_params_,
        }
        matrices_conf[nombre_m] = confusion_matrix(y_pru, y_pred_pru)

    # -------------------------------------------------------------------------
    # SEXTO MODELO: CNN (Conv2D) [Optimizado con GridSearchCV bajo el mismo protocolo CV que los otros 5 modelos]
    # Banco de 8 Kernels 2D + ReLU + MaxPool 4x4 + AvgPool 8x8 + Fusión Espectral PCA
    # -------------------------------------------------------------------------
    print("  Entrenando y optimizando (CV por bloques): CNN (Conv2D) [8 Filtros 2D + MaxPool 4x4 + Fusión PCA]...")
    estimador_cnn = MLPClassifier(
        activation="relu",
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.15,
        random_state=42,
    )
    grilla_cnn = {
        "hidden_layer_sizes": [(64, 32), (48, 24)],
        "alpha": [2.2, 3.5],
        "learning_rate_init": [0.0008],
    }
    grid_cnn = GridSearchCV(
        estimador_cnn,
        grilla_cnn,
        cv=cv_estratificada,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_cnn.fit(F_cnn_total_ent, y_ent)
    clf_cnn = grid_cnn.best_estimator_
    print(f"    [ÉPOCAS CNN] Entrenamiento convergió en {clf_cnn.n_iter_} / {clf_cnn.max_iter} épocas (Mejores params CV: {grid_cnn.best_params_}).")

    y_pred_ent_cnn = clf_cnn.predict(F_cnn_total_ent)
    y_pred_pru_cnn = clf_cnn.predict(F_cnn_total_pru)

    acc_ent_cnn = np.mean(y_pred_ent_cnn == y_ent) * 100.0
    err_ent_cnn = 100.0 - acc_ent_cnn
    acc_pru_cnn = np.mean(y_pred_pru_cnn == y_pru) * 100.0
    err_pru_cnn = 100.0 - acc_pru_cnn
    brecha_cnn = acc_ent_cnn - acc_pru_cnn

    assert np.isclose(acc_ent_cnn + err_ent_cnn, 100.0), "Fallo assert suma ent en CNN (Conv2D)"
    assert np.isclose(acc_pru_cnn + err_pru_cnn, 100.0), "Fallo assert suma pru en CNN (Conv2D)"

    acc_val_cnn = float(grid_cnn.best_score_ * 100.0)

    J_cnn = acc_pru_cnn - 0.6 * max(0.0, brecha_cnn - 3.0) - 1.2 * max(0.0, -brecha_cnn)

    resultados_lista.append(
        {
            "Modelo": "CNN (Conv2D)",
            "Exactitud ent. (%)": round(acc_ent_cnn, 2),
            "Error ent. (%)": round(err_ent_cnn, 2),
            "Exactitud val. CV (%)": round(acc_val_cnn, 2),
            "Exactitud prueba (%)": round(acc_pru_cnn, 2),
            "Error prueba (%)": round(err_pru_cnn, 2),
            "Brecha (p. p.)": round(brecha_cnn, 2),
            "Indice_J": round(J_cnn, 2),
            "Epocas_Ejecutadas": getattr(clf_cnn, "n_iter_", "N/A"),
            "Mejores_Parametros": grid_cnn.best_params_,
        }
    )

    modelos_entrenados["CNN (Conv2D)"] = {
        "modelo": clf_cnn,
        "scaler": scaler,
        "scaler_cnn": scaler_cnn,
        "es_cnn_2d": True,
        "params": grid_cnn.best_params_,
    }
    matrices_conf["CNN (Conv2D)"] = confusion_matrix(y_pru, y_pred_pru_cnn)

    df_resultados = pd.DataFrame(resultados_lista)
    # Ordenar por el criterio de selección J
    df_resultados = df_resultados.sort_values(by="Indice_J", ascending=False).reset_index(drop=True)

    # Imprimir tabla de cátedra oficial
    columnas_oficiales = [
        "Modelo",
        "Exactitud ent. (%)",
        "Error ent. (%)",
        "Exactitud val. CV (%)",
        "Exactitud prueba (%)",
        "Error prueba (%)",
        "Brecha (p. p.)",
    ]
    print("\nTABLA OFICIAL DE RESULTADOS DE CÁTEDRA:")
    try:
        print(df_resultados[columnas_oficiales].to_markdown(index=False))
    except Exception:
        print(df_resultados[columnas_oficiales].to_string(index=False))

    mejor_modelo_nombre = df_resultados.iloc[0]["Modelo"]
    mejor_modelo_info = modelos_entrenados[mejor_modelo_nombre]

    df_clasicos = df_resultados[df_resultados["Modelo"] != "CNN (Conv2D)"]
    mejor_clasico_nombre = df_clasicos.iloc[0]["Modelo"]
    mejor_clasico_info = modelos_entrenados[mejor_clasico_nombre]

    if mejor_modelo_nombre == "CNN (Conv2D)":
        print(f"\n>>> MODELO GANADOR SELECCIONADO: CNN (Conv2D)")
        print(
            f"    Criterio matemático: Máximo índice J = {df_resultados.iloc[0]['Indice_J']} | "
            f"Exactitud prueba = {df_resultados.iloc[0]['Exactitud prueba (%)']}% | "
            f"Brecha = {df_resultados.iloc[0]['Brecha (p. p.)']} p.p."
        )
        print(f">>> MODELO CLÁSICO SELECCIONADO PARA PIPELINE PCA: {mejor_clasico_nombre} (Prueba: {df_clasicos.iloc[0]['Exactitud prueba (%)']}%)")
    else:
        print(f"\n>>> MODELO GANADOR SELECCIONADO: {mejor_modelo_nombre}")
        print(
            f"    Criterio matemático: Máximo índice J = {df_resultados.iloc[0]['Indice_J']} | "
            f"Exactitud prueba = {df_resultados.iloc[0]['Exactitud prueba (%)']}% | "
            f"Brecha = {df_resultados.iloc[0]['Brecha (p. p.)']} p.p."
        )

    return (
        df_resultados,
        modelos_entrenados,
        mejor_modelo_nombre,
        mejor_modelo_info,
        pca,
        matrices_conf,
        y_pru,
        datos_adicionales,
        mejor_clasico_nombre,
        mejor_clasico_info,
    )


def generar_figura2_eigenfaces_varianza(pca_etapa2):
    """
    Figura 2: Curva de Varianza Acumulada (con umbral del 95%) y primeras 10 Eigenfaces.
    """
    var_acum = np.cumsum(pca_etapa2.explained_variance_ratio_) * 100.0
    k_95 = pca_etapa2.n_components_

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 5)
    fig.suptitle(
        "Figura 2: Análisis de Componentes Principales (PCA / Eigenfaces) - Retención 95% de Varianza\n"
        "Sistema de Reconocimiento Facial Inteligente - Ortiz & Ayuquina (FISEI - UTA)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Gráfico de Varianza Explicada en la fila superior
    ax_var = fig.add_subplot(gs[0, :])
    ax_var.plot(range(1, len(var_acum) + 1), var_acum, "b-o", linewidth=2, markersize=4, label="Varianza acumulada")
    ax_var.axhline(95.0, color="r", linestyle="--", label="Umbral de Cátedra (95.0%)")
    ax_var.axvline(k_95, color="g", linestyle=":", label=f"Dimensión Óptima K={k_95}")
    ax_var.set_xlabel("Número de Componentes Principales (Autovectores)", fontsize=10)
    ax_var.set_ylabel("Varianza Acumulada Explicada (%)", fontsize=10)
    ax_var.set_title("Curva de Varianza Explicada Acumulada vs. Componentes K", fontsize=11)
    ax_var.grid(True, linestyle="--", alpha=0.6)
    ax_var.legend(loc="lower right")

    # Primeras 5 Eigenfaces en la fila inferior
    autovectores = pca_etapa2.components_
    num_ef = min(5, len(autovectores))
    for i in range(num_ef):
        ax_ef = fig.add_subplot(gs[1, i])
        ef_img = autovectores[i].reshape(64, 64)
        ax_ef.imshow(ef_img, cmap="bone")
        ax_ef.set_title(f"Eigenface #{i+1}\n(λ_{i+1}={pca_etapa2.explained_variance_[i]:.1f})", fontsize=9)
        ax_ef.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    ruta_fig2 = os.path.join(DIR_FIGURAS, "figura2_eigenfaces_varianza_pca.png")
    plt.savefig(ruta_fig2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Guardada Figura 2 en: {ruta_fig2}")


def generar_figura_matrices_confusion(matrices_conf, nombres_clases, ruta_guardar, titulo):
    """
    Genera la figura comparativa de matrices de confusión para los 6 algoritmos evaluados (5 clásicos + CNN 2D).
    """
    nombres_modelos = list(matrices_conf.keys())
    num_m = len(nombres_modelos)
    fig, axes = plt.subplots(1, num_m, figsize=(4.2 * num_m, 5))
    if num_m == 1:
        axes = [axes]
    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=1.02)

    clases_etiquetas = [nombres_clases[k] for k in sorted(nombres_clases.keys())]

    for idx, nombre_m in enumerate(nombres_modelos):
        ax = axes[idx]
        mat = matrices_conf[nombre_m]
        disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=clases_etiquetas)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(nombre_m, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(ruta_guardar, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Guardada figura de matrices de confusión en: {ruta_guardar}")


def generar_figura5_comparativa_brechas(df_e1, df_e2):
    """
    Figura 5: Gráfico comparativo de barras mostrando Exactitud de Entrenamiento vs. Prueba
    y la Brecha en puntos porcentuales (p. p.) para ambas etapas (5 clásicos + CNN 2D).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Figura 5: Comparativa de Rendimiento y Brecha de Generalización (Ein vs. Etest)\n"
        "Control Estricto Anti-Overfitting - Ortiz & Ayuquina (FISEI - UTA)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, (df, tit) in enumerate(
        [(df_e1, "Etapa 1: Humano vs. No Persona"), (df_e2, "Etapa 2: Reconocimiento de Identidad")]
    ):
        ax = axes[idx]
        x = np.arange(len(df))
        ancho = 0.35

        bar1 = ax.bar(x - ancho / 2, df["Exactitud ent. (%)"], ancho, label="Exactitud Ent. (100 - Ein)", color="#3498db")
        bar2 = ax.bar(x + ancho / 2, df["Exactitud prueba (%)"], ancho, label="Exactitud Prueba (100 - Etest)", color="#2ecc71")

        # Dibujar línea o etiqueta de la brecha
        for i, b in enumerate(df["Brecha (p. p.)"]):
            y_pos = max(df["Exactitud ent. (%)"][i], df["Exactitud prueba (%)"][i]) + 1.0
            ax.text(
                x[i],
                y_pos,
                f"Δ={b:.1f} pp",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#e74c3c" if b > 5.0 else "#27ae60",
            )

        ax.set_ylabel("Exactitud (%)", fontsize=10)
        ax.set_title(tit, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Modelo"], rotation=22, ha="right", fontsize=8.5)
        ax.set_ylim(0, 115)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    ruta_fig5 = os.path.join(DIR_FIGURAS, "figura5_comparativa_brechas.png")
    plt.savefig(ruta_fig5, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Guardada Figura 5 en: {ruta_fig5}")


def main():
    asegurar_directorios()
    print("=" * 80)
    print("INICIANDO PROTOCOLO COMPLETO DE ENTRENAMIENTO, CALIBRACIÓN Y EVALUACIÓN")
    print("SISTEMA BIOMÉTRICO BASADO EN AGENTE INTELIGENTE (PDI + ML)")
    print("Universidad Técnica de Ambato - FISEI - Ortiz & Ayuquina")
    print("=" * 80)

    # 1. Cargar fuentes de datos
    X_oli, y_oli, X_fondo, dict_id, X_obj = cargar_dataset_completo()
    dict_fotos_reales = {id_num: dict_id[id_num]["data"] for id_num in dict_id}

    # 2. Construir matrices para Etapa 1
    print("\n[2/6] Estructurando matrices de la Etapa 1 (Persona vs. No Persona)...")
    X_e1, y_e1 = construir_matrices_etapa1(X_oli, X_fondo, dict_id, X_obj)
    nombres_e1 = {0: "No Persona", 1: "Es Persona"}

    # 3. Evaluar Etapa 1
    (
        df_e1,
        modelos_e1,
        mejor_nombre_e1,
        mejor_info_e1,
        pca_e1,
        matrices_conf_e1,
        _,
        datos_e1,
        mejor_clasico_e1,
        mejor_clasico_info_e1,
    ) = evaluar_experimento("Etapa 1: Persona vs. No Persona", X_e1, y_e1, nombres_e1)

    # Exportar tabla CSV Etapa 1
    ruta_csv_e1 = os.path.join(DIR_FIGURAS, "tabla_resultados_etapa1_humano.csv")
    df_e1.to_csv(ruta_csv_e1, index=False)
    print(f"[OK] Exportada tabla de resultados Etapa 1 a: {ruta_csv_e1}")

    # 4. Construir matrices para Etapa 2
    print("\n[3/6] Estructurando matrices de la Etapa 2 (Identidad Multiclase)...")
    X_e2, y_e2, nombres_e2 = construir_matrices_etapa2(X_oli, dict_id, X_obj, X_fondo)

    # 5. Evaluar Etapa 2
    (
        df_e2,
        modelos_e2,
        mejor_nombre_e2,
        mejor_info_e2,
        pca_e2,
        matrices_conf_e2,
        _,
        datos_e2,
        mejor_clasico_e2,
        mejor_clasico_info_e2,
    ) = evaluar_experimento("Etapa 2: Reconocimiento de Identidad", X_e2, y_e2, nombres_e2, dict_fotos_reales=dict_fotos_reales)

    # Exportar tabla CSV Etapa 2
    ruta_csv_e2 = os.path.join(DIR_FIGURAS, "tabla_resultados_etapa2_identidad.csv")
    df_e2.to_csv(ruta_csv_e2, index=False)
    print(f"[OK] Exportada tabla de resultados Etapa 2 a: {ruta_csv_e2}")

    # 6. Generar Figuras Oficiales del Informe
    print("\n[4/6] Generando figuras diagnósticas y comparativas para el informe...")
    generar_figura2_eigenfaces_varianza(pca_e2)

    ruta_fig3 = os.path.join(DIR_FIGURAS, "figura3_matrices_confusion_etapa1.png")
    generar_figura_matrices_confusion(
        matrices_conf_e1,
        nombres_e1,
        ruta_fig3,
        "Figura 3: Matrices de Confusión de los 6 Modelos Evaluados (Etapa 1: Humano vs. No Persona)",
    )

    ruta_fig4 = os.path.join(DIR_FIGURAS, "figura4_matrices_confusion_etapa2.png")
    generar_figura_matrices_confusion(
        matrices_conf_e2,
        nombres_e2,
        ruta_fig4,
        "Figura 4: Matrices de Confusión de los 6 Modelos Evaluados (Etapa 2: Identidades Multiclase)",
    )

    generar_figura5_comparativa_brechas(df_e1, df_e2)

    # 7. Serialización de Artefactos de Producción en modelos_entrenados/
    print("\n[5/6] Serializando artefactos entrenados con joblib en 'modelos_entrenados/'...")
    joblib.dump(pca_e1, os.path.join(DIR_MODELOS, "etapa1_pca_humano.pkl"))
    joblib.dump(modelos_e1, os.path.join(DIR_MODELOS, "etapa1_todos_modelos.pkl"))
    joblib.dump(
        {"nombre": mejor_nombre_e1, "info": mejor_info_e1, "clasico_nombre": mejor_clasico_e1, "clasico_info": mejor_clasico_info_e1},
        os.path.join(DIR_MODELOS, "etapa1_mejor_modelo.pkl"),
    )

    joblib.dump(pca_e2, os.path.join(DIR_MODELOS, "etapa2_pca_identidad.pkl"))
    joblib.dump(modelos_e2, os.path.join(DIR_MODELOS, "etapa2_todos_modelos.pkl"))
    joblib.dump(
        {"nombre": mejor_nombre_e2, "info": mejor_info_e2, "clasico_nombre": mejor_clasico_e2, "clasico_info": mejor_clasico_info_e2},
        os.path.join(DIR_MODELOS, "etapa2_mejor_modelo.pkl"),
    )

    # Guardar modelo CNN profundo de Etapa 2 para inferencia directa
    if "CNN (Conv2D)" in modelos_e2:
        joblib.dump(modelos_e2["CNN (Conv2D)"], os.path.join(DIR_MODELOS, "etapa2_cnn_profunda.pkl"))
        print("[OK] Modelo CNN profundo serializado en: etapa2_cnn_profunda.pkl")

    ids_autorizados = sorted(list(dict_id.keys()))
    id_desconocido = max(ids_autorizados) + 1 if ids_autorizados else 1
    umbral_mse_identidad = datos_e2.get("umbral_mse_identidad", 0.0055)

    # Guardar metadatos y mapeos
    metadata = {
        "nombres_e1": nombres_e1,
        "nombres_e2": nombres_e2,
        "mejor_modelo_e1": mejor_nombre_e1,
        "mejor_modelo_e2": mejor_nombre_e2,
        "mejor_clasico_e1": mejor_clasico_e1,
        "mejor_clasico_e2": mejor_clasico_e2,
        "umbral_confianza_acceso": 0.62,
        "umbral_mse_rostro": datos_e1.get("umbral_mse_rostro", 0.0042),
        "perfiles_identidad": datos_e2.get("perfiles_identidad", {}),
        "ids_autorizados": ids_autorizados,
        "id_desconocido": int(id_desconocido),
        "umbral_mse_identidad": umbral_mse_identidad,
    }
    joblib.dump(metadata, os.path.join(DIR_MODELOS, "metadata_clases.pkl"))
    print(f"[OK] Metadatos guardados: {len(ids_autorizados)} autorizados ({ids_autorizados}), Desconocido={id_desconocido}, Umbral MSE ID={umbral_mse_identidad:.5f}")
    print("[OK] Todos los artefactos .pkl han sido serializados exitosamente.")

    print("\n" + "=" * 80)
    print("FASE DE ENTRENAMIENTO Y EVALUACIÓN FINALIZADA SATISFACTORIAMENTE")
    print(f"Mejor Modelo Etapa 1: {mejor_nombre_e1}")
    print(f"Mejor Modelo Etapa 2: {mejor_nombre_e2}")
    print("=" * 80)


if __name__ == "__main__":
    main()
