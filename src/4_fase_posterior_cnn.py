"""
Módulo 4: Fase Posterior - Red Neuronal Convolucional (CNN) y Comparativa Global
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Docente: Ing. Rubén Nogales
Universidad Técnica de Ambato - FISEI - Software
"""

import os
import sys
import warnings
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MODELOS = os.path.join(BASE_DIR, "modelos_entrenados")
DIR_FIGURAS = os.path.join(BASE_DIR, "figuras_y_reportes")

sys.path.append(os.path.dirname(__file__))
from utils_pdi import extraer_mapas_convolucionales_2d


def entrenar_evaluar_cnn():
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_MODELOS, exist_ok=True)
    print("=" * 80)
    print("FASE POSTERIOR: RED NEURONAL CONVOLUCIONAL (CNN) vs. PIPELINE CLÁSICO (PCA)")
    print("Universidad Técnica de Ambato - FISEI - Ortiz & Ayuquina")
    print("=" * 80)

    # Cargar datos idénticos de la Etapa 2
    import importlib
    modulo_ent = importlib.import_module("2_entrenamiento_evaluacion")

    print("\nCargando datos homogeneizados de la Etapa 2 (Multiclase de Identidad)...")
    X_oli, y_oli, X_fondo, dict_id, X_obj = modulo_ent.cargar_dataset_completo()
    X_e2, y_e2, nombres_e2 = modulo_ent.construir_matrices_etapa2(X_oli, dict_id, X_obj, X_fondo)
    X_raw = X_e2.reshape(-1, 64, 64).astype(np.float64)
    y = y_e2

    # Partición 80/20 por Bloques Temporales Contiguos Disjuntos (idéntico a 2_entrenamiento_evaluacion.py)
    idx_ent_list, idx_pru_list = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        corte_ent = int(len(idx_c) * 0.80)
        idx_ent_list.extend(idx_c[:corte_ent])
        idx_pru_list.extend(idx_c[corte_ent:])

    X_ent_flat = modulo_ent.agregar_ruido_suave_sensor(X_e2[idx_ent_list].copy(), sigma=0.018, seed=42)
    X_pru_flat = modulo_ent.agregar_ruido_suave_sensor(X_e2[idx_pru_list].copy(), sigma=0.018, seed=100)
    X_ent = X_ent_flat.reshape(-1, 64, 64)
    X_pru = X_pru_flat.reshape(-1, 64, 64)
    y_ent = y[idx_ent_list]
    y_pru = y[idx_pru_list]

    print(f"Conjunto de entrenamiento: {X_ent.shape[0]} imágenes de 64x64x1")
    print(f"Conjunto de prueba: {X_pru.shape[0]} imágenes de 64x64x1")

    print("\nAplicando Bloque Convolucional 2D [8 Filtros + ReLU + MaxPool2D 4x4 + AvgPool2D 8x8]...")
    F_conv_ent = extraer_mapas_convolucionales_2d(X_ent)
    F_conv_pru = extraer_mapas_convolucionales_2d(X_pru)
    print(f"Dimensión del mapa de características convolucional: {F_conv_ent.shape[1]}")

    print("Cargando proyección espectral PCA y fusionando con mapas convolucionales 2D...")
    from utils_pdi import ScalerPreservandoVarianza
    ruta_pca_e2 = os.path.join(DIR_MODELOS, "etapa2_pca_identidad.pkl")
    if os.path.exists(ruta_pca_e2):
        datos_pca = joblib.load(ruta_pca_e2)
        if isinstance(datos_pca, dict):
            pca_e2 = datos_pca["pca"]
            scaler_pca = datos_pca.get("scaler", ScalerPreservandoVarianza(n_pca=pca_e2.n_components_))
        else:
            pca_e2 = datos_pca
            scaler_pca = ScalerPreservandoVarianza(n_pca=pca_e2.n_components_)
    else:
        from sklearn.decomposition import PCA
        pca_e2 = PCA(n_components=0.95, whiten=False, random_state=42).fit(X_ent.reshape(-1, 4096))
        scaler_pca = ScalerPreservandoVarianza(n_pca=pca_e2.n_components_)

    F_ent, _ = modulo_ent.extraer_caracteristicas_pca_mse(pca_e2, X_ent.reshape(-1, 4096))
    F_pru, _ = modulo_ent.extraer_caracteristicas_pca_mse(pca_e2, X_pru.reshape(-1, 4096))
    F_ent_clean = scaler_pca.fit_transform(F_ent) if hasattr(scaler_pca, "fit_transform") else scaler_pca.transform(F_ent)
    F_pru_clean = scaler_pca.transform(F_pru) if hasattr(scaler_pca, "transform") else F_pru

    print("Aplicando Pirámide Jerárquica (MaxPool 2x2/4x4) + Cuello de Botella Convolucional Ortogonal (M=72)...")
    from utils_pdi import BottleneckConvolucional2D
    scaler_cnn = BottleneckConvolucional2D(n_components=72, factor_energia=0.55)
    F_conv_ent_base = scaler_cnn.fit_transform(F_conv_ent)
    F_conv_pru_base = scaler_cnn.transform(F_conv_pru)

    F_joint_ent_clean = np.hstack([F_ent_clean, F_conv_ent_base])
    F_joint_pru_clean = np.hstack([F_pru_clean, F_conv_pru_base])

    F_cnn_total_ent = modulo_ent._aplicar_regularizacion_manifold_mixup(
        F_joint_ent_clean, y_ent, d_pca=None, tasa_frontera=0.044, sigma_rel=0.065, seed=42
    )
    # El conjunto de prueba se mantiene LIMPIO (sin Manifold Mixup) para
    # coherencia con la evaluación de 2_entrenamiento_evaluacion.py.
    F_cnn_total_pru = F_joint_pru_clean
    print(f"Dimensión del vector fusionado [PCA + Bottleneck Conv2D]: {F_cnn_total_ent.shape[1]}")

    print("Optimizando clasificador denso profundo con GridSearchCV (5 pliegues por bloques contiguos) y Early Stopping...")
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    cv_estratificada = StratifiedKFold(n_splits=5, shuffle=False)
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
    grid_cnn = GridSearchCV(estimador_cnn, grilla_cnn, cv=cv_estratificada, scoring="accuracy", n_jobs=-1)
    grid_cnn.fit(F_cnn_total_ent, y_ent)
    clf_cnn = grid_cnn.best_estimator_
    print(f"  [ÉPOCAS CNN] Clasificador denso convergió en {clf_cnn.n_iter_} / {clf_cnn.max_iter} épocas (Mejores params: {grid_cnn.best_params_}).")

    y_pred_ent = clf_cnn.predict(F_cnn_total_ent)
    y_pred_pru = clf_cnn.predict(F_cnn_total_pru)

    acc_ent = accuracy_score(y_ent, y_pred_ent) * 100.0
    err_ent = 100.0 - acc_ent
    acc_pru = accuracy_score(y_pru, y_pred_pru) * 100.0
    err_pru = 100.0 - acc_pru
    brecha = acc_ent - acc_pru

    # Verificación estricta de cátedra
    assert np.isclose(acc_ent + err_ent, 100.0)
    assert np.isclose(acc_pru + err_pru, 100.0)

    # Serializar el modelo CNN entrenado para su uso en tiempo real en 3_reconocimiento_en_vivo.py
    ruta_modelo_cnn = os.path.join(DIR_MODELOS, "etapa2_cnn_profunda.pkl")
    joblib.dump(
        {
            "modelo": clf_cnn,
            "scaler": scaler_pca,
            "scaler_cnn": scaler_cnn,
            "es_cnn_2d": True,
            "params": grid_cnn.best_params_,
        },
        ruta_modelo_cnn,
    )
    print(f"[OK] Modelo CNN serializado exitosamente en: {ruta_modelo_cnn}")

    # Cargar tabla previa de la Etapa 2 si existe para comparación directa (excluyendo CNN de la lista de clásicos)
    ruta_csv_e2 = os.path.join(DIR_FIGURAS, "tabla_resultados_etapa2_identidad.csv")
    filas_comparativa = []

    if os.path.exists(ruta_csv_e2):
        df_previo = pd.read_csv(ruta_csv_e2)
        for _, row in df_previo.iterrows():
            if "CNN" in str(row["Modelo"]):
                continue
            filas_comparativa.append(
                {
                    "Enfoque": "Clásico (PDI + PCA)",
                    "Modelo": row["Modelo"],
                    "Exactitud ent. (%)": row["Exactitud ent. (%)"],
                    "Error ent. (%)": row["Error ent. (%)"],
                    "Exactitud prueba (%)": row["Exactitud prueba (%)"],
                    "Error prueba (%)": row["Error prueba (%)"],
                    "Brecha (p. p.)": row["Brecha (p. p.)"],
                }
            )

    filas_comparativa.append(
        {
            "Enfoque": "Deep Learning (Fase Posterior)",
            "Modelo": "CNN (Conv2D + MaxPool + FC)",
            "Exactitud ent. (%)": round(acc_ent, 2),
            "Error ent. (%)": round(err_ent, 2),
            "Exactitud prueba (%)": round(acc_pru, 2),
            "Error prueba (%)": round(err_pru, 2),
            "Brecha (p. p.)": round(brecha, 2),
        }
    )

    df_comp = pd.DataFrame(filas_comparativa)
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA GLOBAL: ENFOQUE CLÁSICO (PCA) vs. RED CONVOLUCIONAL (CNN)")
    print("=" * 80)
    try:
        print(df_comp.to_markdown(index=False))
    except Exception:
        print(df_comp.to_string(index=False))

    ruta_csv_cnn = os.path.join(DIR_FIGURAS, "tabla_comparativa_cnn_vs_clasicos.csv")
    df_comp.to_csv(ruta_csv_cnn, index=False)
    print(f"\n[OK] Tabla comparativa exportada a: {ruta_csv_cnn}")

    # Gráfico comparativo
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_comp))
    ancho = 0.35
    ax.bar(x - ancho / 2, df_comp["Exactitud ent. (%)"], ancho, label="Exactitud Entrenamiento", color="#2980b9")
    ax.bar(x + ancho / 2, df_comp["Exactitud prueba (%)"], ancho, label="Exactitud Prueba", color="#27ae60")

    for i, b in enumerate(df_comp["Brecha (p. p.)"]):
        ax.text(
            x[i],
            max(df_comp["Exactitud ent. (%)"][i], df_comp["Exactitud prueba (%)"][i]) + 2,
            f"Δ={b:.1f} pp",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylabel("Exactitud (%)")
    ax.set_title("Comparativa de Generalización: Enfoque Clásico (PCA) vs. Convolucional (CNN)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["Modelo"], rotation=25, ha="right")
    ax.set_ylim(0, 118)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="lower right")

    plt.tight_layout()
    ruta_fig_cnn = os.path.join(DIR_FIGURAS, "figura6_comparativa_cnn.png")
    plt.savefig(ruta_fig_cnn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfico comparativo guardado en: {ruta_fig_cnn}")


if __name__ == "__main__":
    entrenar_evaluar_cnn()
