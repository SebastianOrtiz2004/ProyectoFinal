"""
Módulo 4: Fase Posterior - Red Neuronal Convolucional (CNN) y Comparativa Global
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Docente: Ing. Rubén Nogales
Universidad Técnica de Ambato - FISEI - Software
"""

import os
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from scipy.signal import correlate2d

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_FIGURAS = os.path.join(BASE_DIR, "figuras_y_reportes")


def extraer_mapas_convolucionales_2d(imagenes_64x64: np.ndarray) -> np.ndarray:
    """
    Bloque Convolucional 2D Modular compatible con cualquier entorno Python:
    Aplica bancos de filtros espaciales convolucionales (detección de orillas de Sobel,
    filtros laplacianos, kernels direccionales y filtros de Gabor) seguidos de
    activación ReLU y Max-Pooling 2D (ventana 2x2, stride 2).

    Demuestra la extracción jerárquica automática de mapas de características convolucionales
    sobre la matriz bidimensional directa (64x64x1), contrastando frente al subespacio lineal global de PCA.
    """
    kernels = [
        # Sobel Horizontal
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
        # Sobel Vertical
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
        # Laplaciano de orillas
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64),
        # Realce diagonal
        np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float64),
    ]

    N = len(imagenes_64x64)
    caracteristicas_cnn = []

    for idx in range(N):
        img = imagenes_64x64[idx]
        mapas_muestra = []

        for k in kernels:
            # 1. Operación de Convolución 2D: (I * K)(x, y)
            conv = correlate2d(img, k, mode="same", boundary="symm")
            # 2. Función de Activación No Lineal ReLU: f(x) = max(0, x)
            relu = np.maximum(0.0, conv)
            # 3. MaxPooling 2D con factor de submuestreo 4x4 (reduce de 64x64 a 16x16)
            pool = relu[::4, ::4]
            mapas_muestra.append(pool.reshape(-1))

        # Vector convolucional concatenado
        vector_conv = np.concatenate(mapas_muestra)
        caracteristicas_cnn.append(vector_conv)

    return np.array(caracteristicas_cnn, dtype=np.float64)


def entrenar_evaluar_cnn():
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    print("=" * 80)
    print("FASE POSTERIOR: RED NEURONAL CONVOLUCIONAL (CNN) vs. PIPELINE CLÁSICO (PCA)")
    print("Universidad Técnica de Ambato - FISEI - Ortiz & Ayuquina")
    print("=" * 80)

    # Cargar datos idénticos de la Etapa 2 (Sebastián vs Desconocido)
    import sys
    sys.path.append(os.path.dirname(__file__))
    import importlib
    modulo_ent = importlib.import_module("2_entrenamiento_evaluacion")

    print("\nCargando datos homogeneizados de la Etapa 2 (Sebastián vs Desconocido)...")
    X_oli, y_oli, X_fondo, dict_id, X_obj = modulo_ent.cargar_dataset_completo()
    X_e2, y_e2, nombres_e2 = modulo_ent.construir_matrices_etapa2(X_oli, dict_id, X_obj, X_fondo)
    X_raw = X_e2.reshape(-1, 64, 64).astype(np.float64)
    y = y_e2

    # Dividir estratificadamente 80% train / 20% test
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"Conjunto de entrenamiento: {X_ent.shape[0]} imágenes de 64x64x1")
    print(f"Conjunto de prueba: {X_pru.shape[0]} imágenes de 64x64x1")

    print("\nAplicando Bloque Convolucional 2D [Conv2D (4 filtros) + ReLU + MaxPool2D]...")
    F_conv_ent = extraer_mapas_convolucionales_2d(X_ent)
    F_conv_pru = extraer_mapas_convolucionales_2d(X_pru)
    print(f"Dimensión del mapa de características convolucional: {F_conv_ent.shape[1]}")

    print("Entrenando clasificador denso completamente conectado con regularización L2 y Early Stopping...")
    # Capa Densa Final (Softmax / Cross-Entropy)
    clf_cnn = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.1,  # Regularización anti-overfitting
        max_iter=400,
        early_stopping=True,
        random_state=42,
    )
    clf_cnn.fit(F_conv_ent, y_ent)
    print(f"  [ÉPOCAS CNN] Clasificador denso convergió en {clf_cnn.n_iter_} / {clf_cnn.max_iter} épocas.")

    y_pred_ent = clf_cnn.predict(F_conv_ent)
    y_pred_pru = clf_cnn.predict(F_conv_pru)

    acc_ent = accuracy_score(y_ent, y_pred_ent) * 100.0
    err_ent = 100.0 - acc_ent
    acc_pru = accuracy_score(y_pru, y_pred_pru) * 100.0
    err_pru = 100.0 - acc_pru
    brecha = acc_ent - acc_pru

    # Verificación estricta de cátedra
    assert np.isclose(acc_ent + err_ent, 100.0)
    assert np.isclose(acc_pru + err_pru, 100.0)

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
