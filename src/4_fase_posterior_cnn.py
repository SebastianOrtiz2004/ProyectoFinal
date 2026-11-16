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

    # Dividir estratificadamente 80% train / 20% test (idéntico a etapa clásica)
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=42
    )

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
    F_ent_scaled = scaler_pca.fit_transform(F_ent) if hasattr(scaler_pca, "fit_transform") else scaler_pca.transform(F_ent)
    F_pru_scaled = scaler_pca.transform(F_pru) if hasattr(scaler_pca, "transform") else F_pru

    print("Estandarizando características convolucionales con StandardScaler (factor 0.35)...")
    scaler_cnn = StandardScaler()
    F_conv_ent_s = scaler_cnn.fit_transform(F_conv_ent) * 0.35
    F_conv_pru_s = scaler_cnn.transform(F_conv_pru) * 0.35

    F_cnn_total_ent = np.hstack([F_ent_scaled, F_conv_ent_s])
    F_cnn_total_pru = np.hstack([F_pru_scaled, F_conv_pru_s])
    print(f"Dimensión del vector fusionado [PCA + Conv2D]: {F_cnn_total_ent.shape[1]}")

    print("Entrenando clasificador denso profundo (256, 128) con regularización L2 y Early Stopping...")
    # Capa Densa Final (Softmax / Cross-Entropy)
    clf_cnn = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=0.08,  # Regularización anti-overfitting
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    clf_cnn.fit(F_cnn_total_ent, y_ent)
    print(f"  [ÉPOCAS CNN] Clasificador denso convergió en {clf_cnn.n_iter_} / {clf_cnn.max_iter} épocas.")

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
            "params": {"hidden_layers": (256, 128), "alpha": 0.08},
        },
        ruta_modelo_cnn,
    )
    print(f"[OK] Modelo CNN serializado exitosamente en: {ruta_modelo_cnn}")

    # Cargar tabla previa de la Etapa 2 si existe para comparación directa
    ruta_csv_e2 = os.path.join(DIR_FIGURAS, "tabla_resultados_etapa2_identidad.csv")
    filas_comparativa = []

    if os.path.exists(ruta_csv_e2):
        df_previo = pd.read_csv(ruta_csv_e2)
        for _, row in df_previo.iterrows():
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
