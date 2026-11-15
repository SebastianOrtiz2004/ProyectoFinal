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
    aplicar_mascara_eliptica_a_img64,
    extraer_descriptor_espacial_bloques,
    aumentar_geometria_rostro,
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
    Carga de datos 100% reales (sin síntesis ni variaciones artificiales de rostros):
    1. Rostros de referencia de Olivetti Faces preprocesados exactamente con preprocesar_ventana_64x64
       para homogeneizar filtrado, histograma y rango con las fotos de la webcam.
    2. Recortes reales de Clase 0 (Fondo / Objetos) sin archivos sintéticos seed.
    3. Identidades locales con datos reales (>= 15 fotos).
    """
    # 1. Limpieza rigurosa de cualquier archivo sintético/seed residual
    archivos_seed = glob.glob(os.path.join(DIR_CLASE_0, "objeto_real_seed_*.png"))
    for f in archivos_seed:
        try:
            os.remove(f)
        except Exception:
            pass
    if archivos_seed:
        print(f"  [LIMPIEZA] Eliminados {len(archivos_seed)} archivos sintéticos residuales de clase_0_no_persona.")

    # 2. Descargar y preprocesar dataset Olivetti Faces (preprocesar_ventana_64x64 ya aplica la máscara elíptica)
    print("[1/6] Descargando/Cargando y homogeneizando dataset estándar Olivetti Faces...")
    from utils_pdi import preprocesar_ventana_64x64
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
    X_olivetti_list = []
    for raw_f in olivetti.data:
        im_64 = raw_f.reshape(64, 64)
        _, vec_p = preprocesar_ventana_64x64(im_64)
        X_olivetti_list.append(vec_p)
    X_olivetti = np.array(X_olivetti_list, dtype=np.float64)
    y_olivetti = olivetti.target

    # 3. Cargar imágenes reales de la carpeta clase_0_no_persona (balanceando fondos y objetos)
    archivos_fondo = glob.glob(os.path.join(DIR_CLASE_0, "*.png"))
    archivos_obj = [f for f in archivos_fondo if os.path.basename(f).startswith("objeto_")]
    archivos_wall = [f for f in archivos_fondo if not os.path.basename(f).startswith("objeto_")]

    # Cargar TODOS los objetos reales (incluyendo objeto_manual_*.png recién capturados con la tecla [F])
    X_obj_local = []
    for f in archivos_obj:
        im = cv2_imread_unicode(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            if im.shape != (64, 64):
                im = cv2.resize(im, (64, 64))
            im_double = im.astype(np.float64) / 255.0
            im_mask = aplicar_mascara_eliptica_a_img64(im_double)
            X_obj_local.append(im_mask.reshape(-1))

    # Cargar TODOS los fondos/paredes reales disponibles (sin recortar a 250)
    X_wall_local = []
    for f in archivos_wall:
        im = cv2_imread_unicode(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            if im.shape != (64, 64):
                im = cv2.resize(im, (64, 64))
            im_double = im.astype(np.float64) / 255.0
            im_mask = aplicar_mascara_eliptica_a_img64(im_double)
            X_wall_local.append(im_mask.reshape(-1))

    print(f"  [DATOS REALES] Clase 0 cargada: {len(X_obj_local)} objetos reales + {len(X_wall_local)} fondos/paredes (100% muestras reales sin descarte).")

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
                im_mask = aplicar_mascara_eliptica_a_img64(im_double)
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


def construir_matrices_etapa1(X_olivetti, X_fondo, dict_identidades, X_obj=None):
    """
    Etapa 1 (Binaria):
        Clase 0: No Persona (1,280 muestras: todos los objetos reales + fliplr + completado con fondos)
        Clase 1: Es Persona (1,280 muestras: 100% fotos reales de Sebastián + 350 aumentación + Olivetti)
    Balanceo estricto 50% / 50% (1,280 vs 1,280).
    """
    rostros_list = []
    variaciones = [
        (-8, 0, 1.0, False), (8, 0, 1.0, False),
        (0, -8, 1.0, False), (0, 8, 1.0, False),
        (-6, -6, 1.0, False), (6, 6, 1.0, False),
        (-5, 0, 0.95, False), (5, 0, 1.05, False),
        (0, -5, 0.95, False), (0, 5, 1.05, False),
        (0, 0, 0.90, False), (0, 0, 0.95, False),
        (0, 0, 1.05, False), (0, 0, 1.10, False),
        (0, 0, 1.0, True), (5, 0, 1.0, True), (-5, 0, 1.0, True),
        (0, -5, 1.0, True), (0, 5, 1.0, True),
        (-6, -6, 1.05, True), (6, 6, 0.95, True),
        (-4, 0, 0.90, True), (4, 0, 1.10, True),
    ]

    for k in dict_identidades:
        data_k = dict_identidades[k]["data"]  # 100% de fotos reales de cada identidad
        rostros_list.append(data_k)

        # 350 muestras con aumentación geométrica multi-escala y traslación
        np.random.seed(42 + k)
        idx_aug = np.random.choice(len(data_k), size=350, replace=(len(data_k) < 350))
        muestras_aug = []
        for i, idx_a in enumerate(idx_aug):
            img_m = data_k[idx_a].reshape(64, 64)
            dx, dy, sc, fl = variaciones[i % len(variaciones)]
            im_warp = aumentar_geometria_rostro(img_m, dx=dx, dy=dy, scale=sc, flip=fl)
            muestras_aug.append(im_warp.reshape(-1))
        rostros_list.append(np.array(muestras_aug))

    rostros_humanos_totales = np.vstack(rostros_list)
    if len(rostros_humanos_totales) < 1280:
        faltan_h = 1280 - len(rostros_humanos_totales)
        np.random.seed(42)
        idx_oli = np.random.choice(len(X_olivetti), size=min(faltan_h, len(X_olivetti)), replace=False)
        rostros_humanos_totales = np.vstack([rostros_humanos_totales, X_olivetti[idx_oli]])

    if len(rostros_humanos_totales) > 1280:
        np.random.seed(42)
        idx_h_sel = np.random.choice(len(rostros_humanos_totales), size=1280, replace=False)
        X_humanos = rostros_humanos_totales[idx_h_sel]
    else:
        X_humanos = rostros_humanos_totales

    y_humanos = np.ones(len(X_humanos), dtype=int)
    num_humanos = 1280

    # Clase 0: No Persona (1,280 muestras con todos los objetos + fondos)
    no_humano_list = []
    if X_obj is not None and len(X_obj) > 0:
        no_humano_list.append(X_obj)
        obj_flip = np.array([np.fliplr(im.reshape(64, 64)).reshape(-1) for im in X_obj])
        no_humano_list.append(obj_flip)

    X_objs_aug = np.vstack(no_humano_list) if no_humano_list else np.empty((0, 4096))
    if len(X_objs_aug) >= num_humanos:
        np.random.seed(42)
        idx_sel = np.random.choice(len(X_objs_aug), size=num_humanos, replace=False)
        X_no_humano = X_objs_aug[idx_sel]
    else:
        faltantes = num_humanos - len(X_objs_aug)
        np.random.seed(42)
        idx_f = np.random.choice(len(X_fondo), size=min(faltantes, len(X_fondo)), replace=(faltantes > len(X_fondo)))
        X_no_humano = np.vstack([X_objs_aug, X_fondo[idx_f]]) if len(X_objs_aug) > 0 else X_fondo[idx_f]

    if len(X_no_humano) > num_humanos:
        X_no_humano = X_no_humano[:num_humanos]

    y_no_humano = np.zeros(len(X_no_humano), dtype=int)

    X_etapa1 = np.vstack([X_no_humano, X_humanos])
    y_etapa1 = np.concatenate([y_no_humano, y_humanos])
    print(f"  [BALANCE ETAPA 1] Clase 0 (No Persona): {len(y_no_humano)} | Clase 1 (Persona): {len(y_humanos)} (Balance 50/50 estricto: 1280 vs 1280).")

    return X_etapa1, y_etapa1


def construir_matrices_etapa2(X_olivetti, dict_identidades, X_obj=None, X_fondo=None):
    """
    Etapa 2 (Multiclase de Identidad):
        Para cada ID registrado: 100% fotos reales (~933) + 350 aumentación geométrica = 1,280 muestras.
        Clase Desconocido / Otros: Olivetti + Objetos reales + fliplr + Fondos = 1,280 muestras.
    """
    X_list = []
    y_list = []
    nombres_clases = {}

    variaciones = [
        (-8, 0, 1.0, False), (8, 0, 1.0, False),
        (0, -8, 1.0, False), (0, 8, 1.0, False),
        (-6, -6, 1.0, False), (6, 6, 1.0, False),
        (-5, 0, 0.95, False), (5, 0, 1.05, False),
        (0, -5, 0.95, False), (0, 5, 1.05, False),
        (0, 0, 0.90, False), (0, 0, 0.95, False),
        (0, 0, 1.05, False), (0, 0, 1.10, False),
        (0, 0, 1.0, True), (5, 0, 1.0, True), (-5, 0, 1.0, True),
        (0, -5, 1.0, True), (0, 5, 1.0, True),
        (-6, -6, 1.05, True), (6, 6, 0.95, True),
        (-4, 0, 0.90, True), (4, 0, 1.10, True),
    ]

    for id_num in sorted(dict_identidades.keys()):
        info = dict_identidades[id_num]
        data = info["data"]  # 100% de fotos reales (sin descarte a 250!)

        # 350 muestras con aumentación geométrica
        np.random.seed(42 + id_num)
        idx_aug = np.random.choice(len(data), size=350, replace=(len(data) < 350))
        muestras_aug = []
        for i, idx_a in enumerate(idx_aug):
            img_m = data[idx_a].reshape(64, 64)
            dx, dy, sc, fl = variaciones[i % len(variaciones)]
            im_warp = aumentar_geometria_rostro(img_m, dx=dx, dy=dy, scale=sc, flip=fl)
            muestras_aug.append(im_warp.reshape(-1))

        data_clase = np.vstack([data, np.array(muestras_aug)])
        if len(data_clase) > 1280:
            data_clase = data_clase[:1280]
        elif len(data_clase) < 1280:
            faltan = 1280 - len(data_clase)
            idx_rep = np.random.choice(len(data_clase), size=faltan, replace=True)
            data_clase = np.vstack([data_clase, data_clase[idx_rep]])

        X_list.append(data_clase)
        y_list.append(np.full(len(data_clase), id_num, dtype=int))
        nombres_clases[id_num] = info["nombre"]

    # Clase 'Desconocido / Otros': 1,280 muestras (Olivetti + Objetos + Fondos)
    id_desconocido = max(dict_identidades.keys()) + 1 if dict_identidades else 1
    desc_componentes = []

    # 1. Olivetti Faces completos
    desc_componentes.append(X_olivetti)

    # 2. Objetos reales + fliplr
    if X_obj is not None and len(X_obj) > 0:
        desc_componentes.append(X_obj)
        obj_fl = np.array([np.fliplr(im.reshape(64, 64)).reshape(-1) for im in X_obj])
        desc_componentes.append(obj_fl)

    # 3. Fondos de la habitación
    if X_fondo is not None and len(X_fondo) > 0:
        desc_componentes.append(X_fondo)

    desconocidos_total = np.vstack(desc_componentes)
    if len(desconocidos_total) > 1280:
        np.random.seed(42)
        idx_desc_sel = np.random.choice(len(desconocidos_total), size=1280, replace=False)
        desconocidos_total = desconocidos_total[idx_desc_sel]
    elif len(desconocidos_total) < 1280:
        faltan_desc = 1280 - len(desconocidos_total)
        idx_extra_oli = np.random.choice(len(X_olivetti), size=faltan_desc, replace=True)
        desconocidos_total = np.vstack([desconocidos_total, X_olivetti[idx_extra_oli]])

    X_list.append(desconocidos_total)
    y_list.append(np.full(len(desconocidos_total), id_desconocido, dtype=int))
    nombres_clases[id_desconocido] = "Desconocido / Otros"

    X_etapa2 = np.vstack(X_list)
    y_etapa2 = np.concatenate(y_list)
    resumen_clases = " | ".join(f"{nombres_clases[c]}: {np.sum(y_etapa2 == c)}" for c in sorted(nombres_clases.keys()))
    print(f"  [BALANCE ETAPA 2 MULTICLASE] {resumen_clases} (1,280 muestras por clase).")

    return X_etapa2, y_etapa2, nombres_clases


def calcular_features_pdi_matriz(X_mat: np.ndarray) -> np.ndarray:
    """
    Calcula descriptores biométricos y morfológicos canónicos de PDI para cada muestra:
        1. Simetría bilateral adaptable (invariante a giro de cabeza: frente, medio perfil y perfil).
        2. Relación de gradientes Sobel Horizontal vs. Vertical (Ratio_HV).
        3. Descriptor espacial por bloques 4x4 (64 características con Sobel y filtro pasa-altos).
    Total: 66 características por muestra.
    """
    N = X_mat.shape[0]
    feats = np.empty((N, 66), dtype=np.float64)
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

        feats[i] = np.concatenate([[sim, ratio_hv], desc_bloques])
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


def definir_modelos_y_grillas():
    """
    Define los 5 clasificadores supervisados obligatorios del sílabo
    con hiperparámetros ampliados para aprovechar la máxima capacidad computacional.
    """
    clasificadores = {
        "k-NN": (
            KNeighborsClassifier(metric="euclidean"),
            {
                "n_neighbors": [5, 7, 9],
                "weights": ["uniform"],
            },
        ),
        "Árbol de decisión": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {
                "max_depth": [4, 6],
                "min_samples_leaf": [4, 8],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "n_estimators": [150, 250],
                "max_depth": [8, 12],
                "min_samples_leaf": [2, 4],
            },
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=42),
            {
                "C": [1.0, 3.0, 5.0],
                "kernel": ["rbf"],
                "gamma": ["scale"],
            },
        ),
        "ANN (MLP)": (
            MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
            {
                "hidden_layer_sizes": [(128, 64), (128,)],
                "alpha": [0.1, 0.2],
                "learning_rate_init": [0.001],
            },
        ),
    }
    return clasificadores


def evaluar_experimento(nombre_etapa, X, y, nombres_clases=None):
    """
    Ejecuta el ciclo experimental completo para una etapa:
    1. Partición 80/20 en bloques de 10 imágenes consecutivas (8 a entrenamiento y 2 a prueba).
    2. Perturbación suave y simétrica de sensor (sigma = 0.012 con filtro gaussiano 3x3).
    3. PCA 97% varianza calibrado:
       - En Etapa 1: calibrado EXCLUSIVAMENTE sobre rostros humanos (y_ent == 1).
       - En Etapa 2: calibrado sobre identidades de entrenamiento.
    4. Extracción de f = [z_PCA, MSE, Sim_Adapt, Ratio_HV] en R^(K+3).
    5. Escalado que preserva la varianza relativa de las Eigenfaces para todos los modelos.
    6. GridSearchCV con StratifiedKFold(n_splits=5).
    7. Cálculo de umbral_mse_rostro y perfiles multi-pose de identidad (250 prototipos en matriz_Z_poses).
    """
    print(f"\n{'=' * 75}\nEJECUTANDO: {nombre_etapa.upper()}\n{'=' * 75}")

    # Partición 80/20 en bloques de 10 imágenes consecutivas (8 a entrenamiento y 2 a prueba)
    idx_ent_list = []
    idx_pru_list = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        mask_pru_c = (np.arange(len(idx_c)) % 10 >= 8)
        idx_ent_list.extend(idx_c[~mask_pru_c])
        idx_pru_list.extend(idx_c[mask_pru_c])

    idx_ent = np.array(idx_ent_list)
    idx_pru = np.array(idx_pru_list)

    X_ent = X[idx_ent]
    y_ent = y[idx_ent]
    X_pru = X[idx_pru].copy()
    y_pru = y[idx_pru]

    # Perturbación suave y simétrica de sensor (sigma = 0.012 suavizada con GaussianBlur 3x3)
    X_ent = agregar_ruido_suave_sensor(X_ent, sigma=0.012, seed=42)
    X_pru = agregar_ruido_suave_sensor(X_pru, sigma=0.012, seed=100)

    print(f"Dimensiones de matrices: X_ent={X_ent.shape}, X_pru={X_pru.shape}")

    # PCA con retención del 97% de la varianza explicada acumulada
    pca = PCA(n_components=0.97, whiten=False, random_state=42)
    if "Etapa 1" in nombre_etapa:
        # Subespacio Facial Exclusivo de Rostros Humanos
        pca.fit(X_ent[y_ent == 1])
    else:
        pca.fit(X_ent)

    F_ent, mse_ent = extraer_caracteristicas_pca_mse(pca, X_ent)
    F_pru, mse_pru = extraer_caracteristicas_pca_mse(pca, X_pru)
    K = pca.n_components_
    print(f"Componentes PCA retenidas (varianza >= 97%): K = {K}")
    print(f"Dimensión del vector final f ∈ R^(K+3): {F_ent.shape[1]}")

    # Escalador que preserva la relación de autovalores de Eigenfaces
    scaler = ScalerPreservandoVarianza(n_pca=K)
    F_ent_scaled = scaler.fit_transform(F_ent)
    F_pru_scaled = scaler.transform(F_pru)

    datos_adicionales = {}
    if "Etapa 1" in nombre_etapa:
        mse_humanos_ent = mse_ent[y_ent == 1]
        umbral_mse_rostro = float(np.clip(np.percentile(mse_humanos_ent, 99) * 7.5, 0.0110, 0.0135))
        datos_adicionales["umbral_mse_rostro"] = umbral_mse_rostro
        print(f"  [SUBESPACIO FACIAL] Umbral MSE de Reconstrucción Facial (Calibrado con margen dinámico): {umbral_mse_rostro:.5f}")
    elif "Etapa 2" in nombre_etapa:
        perfiles_identidad = {}
        Z_ent_pca = pca.transform(X_ent)
        for c in np.unique(y_ent):
            idx_c = (y_ent == c)
            Z_c = Z_ent_pca[idx_c]
            mu_c = np.mean(Z_c, axis=0)
            dists_c = np.linalg.norm(Z_c - mu_c, axis=1)
            radio_c = float(np.percentile(dists_c, 98) * 1.40)
            dist_media = float(np.mean(dists_c))

            # Matriz de referencia matriz_Z_poses con 250 prototipos representativos multi-pose
            idx_p = np.linspace(0, len(Z_c) - 1, min(250, len(Z_c)), dtype=int)
            matriz_Z_poses = Z_c[idx_p]

            from sklearn.neighbors import NearestNeighbors
            nn_poses = NearestNeighbors(n_neighbors=min(12, len(matriz_Z_poses)), metric="euclidean").fit(matriz_Z_poses)
            dists_min_pose, _ = nn_poses.kneighbors(Z_c)
            radio_pose = float(np.percentile(dists_min_pose[:, -1], 95) * 1.25)

            perfiles_identidad[int(c)] = {
                "centroide": mu_c,
                "radio_max": radio_c,
                "matriz_Z_poses": matriz_Z_poses,
                "radio_pose": radio_pose,
                "dist_media": dist_media,
            }
            nom_c = nombres_clases.get(int(c), f"ID {c}") if nombres_clases else f"ID {c}"
            print(f"  [BIOMETRIA] Perfil {nom_c}: Centroide K={len(mu_c)}, Radio max={radio_c:.3f}, Radio pose={radio_pose:.3f} (250 prototipos), Dist media={dist_media:.3f}")
        datos_adicionales["perfiles_identidad"] = perfiles_identidad

        # Umbral de reconstrucción MSE para identidades autorizadas (excluyendo id_desconocido)
        id_desc = max(perfiles_identidad.keys())
        mask_autorizados = (y_ent != id_desc)
        if np.any(mask_autorizados):
            umbral_mse_identidad = float(max(0.0040, np.percentile(mse_ent[mask_autorizados], 96) * 1.35))
        else:
            umbral_mse_identidad = float(max(0.0040, np.percentile(mse_ent, 96) * 1.35))
        datos_adicionales["umbral_mse_identidad"] = umbral_mse_identidad
        print(f"  [BIOMETRIA] Umbral MSE Identidad Calibrado Dinámicamente: {umbral_mse_identidad:.5f}")

    clasificadores = definir_modelos_y_grillas()
    cv_estratificada = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    resultados_lista = []
    modelos_entrenados = {}
    matrices_conf = {}

    for nombre_m, (estimador, grilla) in clasificadores.items():
        print(f"  Entrenando y optimizando: {nombre_m}...")
        grid = GridSearchCV(
            estimador,
            grilla,
            cv=cv_estratificada,
            scoring="accuracy",
            n_jobs=-1,
        )

        # Todos los modelos entrenan con F_ent_scaled para igualdad de condiciones y generalización óptima
        grid.fit(F_ent_scaled, y_ent)
        mejor = grid.best_estimator_

        if nombre_m == "ANN (MLP)" and hasattr(mejor, "n_iter_"):
            print(f"    [ÉPOCAS ANN] Entrenamiento convergió en {mejor.n_iter_} / {mejor.max_iter} épocas (Early Stopping activo).")

        y_pred_ent = mejor.predict(F_ent_scaled)
        y_pred_pru = mejor.predict(F_pru_scaled)

        acc_ent = np.mean(y_pred_ent == y_ent) * 100.0
        err_ent = 100.0 - acc_ent
        acc_cv = grid.best_score_ * 100.0
        acc_pru = np.mean(y_pred_pru == y_pru) * 100.0
        err_pru = 100.0 - acc_pru
        brecha = acc_ent - acc_pru

        # Comprobación formal obligatoria requerida por el docente
        assert np.isclose(acc_ent + err_ent, 100.0), f"Fallo assert suma ent en {nombre_m}"
        assert np.isclose(acc_pru + err_pru, 100.0), f"Fallo assert suma pru en {nombre_m}"

        # Función de utilidad anti-overfitting (penaliza levemente extrapolación excesiva de MLP para producción)
        penalizacion_mlp = 2.0 if nombre_m == "ANN (MLP)" else 0.0
        J = acc_pru - penalizacion_mlp - 0.5 * max(0.0, brecha - 3.0)

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
    Genera la figura comparativa de matrices de confusión para los 5 algoritmos evaluados.
    """
    nombres_modelos = list(matrices_conf.keys())
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
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
    y la Brecha en puntos porcentuales (p. p.) para ambas etapas.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
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
                fontsize=9,
                fontweight="bold",
                color="#e74c3c" if b > 5.0 else "#27ae60",
            )

        ax.set_ylabel("Exactitud (%)", fontsize=10)
        ax.set_title(tit, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Modelo"], rotation=20, ha="right", fontsize=9)
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
    ) = evaluar_experimento("Etapa 2: Reconocimiento de Identidad", X_e2, y_e2, nombres_e2)

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
        "Figura 3: Matrices de Confusión de los 5 Modelos Evaluados (Etapa 1: Humano vs. No Persona)",
    )

    ruta_fig4 = os.path.join(DIR_FIGURAS, "figura4_matrices_confusion_etapa2.png")
    generar_figura_matrices_confusion(
        matrices_conf_e2,
        nombres_e2,
        ruta_fig4,
        "Figura 4: Matrices de Confusión de los 5 Modelos Evaluados (Etapa 2: Identidades Multiclase)",
    )

    generar_figura5_comparativa_brechas(df_e1, df_e2)

    # 7. Serialización de Artefactos de Producción en modelos_entrenados/
    print("\n[5/6] Serializando artefactos entrenados con joblib en 'modelos_entrenados/'...")
    joblib.dump(pca_e1, os.path.join(DIR_MODELOS, "etapa1_pca_humano.pkl"))
    joblib.dump(modelos_e1, os.path.join(DIR_MODELOS, "etapa1_todos_modelos.pkl"))
    joblib.dump(
        {"nombre": mejor_nombre_e1, "info": mejor_info_e1},
        os.path.join(DIR_MODELOS, "etapa1_mejor_modelo.pkl"),
    )

    joblib.dump(pca_e2, os.path.join(DIR_MODELOS, "etapa2_pca_identidad.pkl"))
    joblib.dump(modelos_e2, os.path.join(DIR_MODELOS, "etapa2_todos_modelos.pkl"))
    joblib.dump(
        {"nombre": mejor_nombre_e2, "info": mejor_info_e2},
        os.path.join(DIR_MODELOS, "etapa2_mejor_modelo.pkl"),
    )

    ids_autorizados = sorted(list(dict_id.keys()))
    id_desconocido = max(ids_autorizados) + 1 if ids_autorizados else 1
    umbral_mse_identidad = datos_e2.get("umbral_mse_identidad", 0.0040)

    # Guardar metadatos y mapeos
    metadata = {
        "nombres_e1": nombres_e1,
        "nombres_e2": nombres_e2,
        "mejor_modelo_e1": mejor_nombre_e1,
        "mejor_modelo_e2": mejor_nombre_e2,
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
