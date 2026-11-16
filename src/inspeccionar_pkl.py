"""
Utilidad para Inspeccionar y Explorar Archivos .pkl del Proyecto
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
"""

import os
import sys
import pprint
import joblib
import numpy as np

# Añadir 'src' al path para poder deserializar clases personalizadas como ScalerPreservandoVarianza
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

try:
    from utils_pdi import ScalerPreservandoVarianza
    setattr(sys.modules["__main__"], "ScalerPreservandoVarianza", ScalerPreservandoVarianza)
except ImportError:
    pass

DIR_MODELOS = os.path.join(BASE_DIR, "modelos_entrenados")


def describir_objeto(obj, indent=0):
    prefijo = "  " * indent
    if isinstance(obj, dict):
        print(f"{prefijo}[Diccionario con {len(obj)} claves]:")
        for k, v in obj.items():
            tipo = type(v).__name__
            if isinstance(v, (int, float, str, bool)):
                print(f"{prefijo}  • {k}: {v} ({tipo})")
            elif isinstance(v, np.ndarray):
                print(f"{prefijo}  • {k}: ndarray shape={v.shape}, dtype={v.dtype}, min={np.min(v):.4f}, max={np.max(v):.4f}")
            elif isinstance(v, dict):
                print(f"{prefijo}  • {k}:")
                describir_objeto(v, indent + 2)
            elif isinstance(v, list):
                print(f"{prefijo}  • {k}: [Lista de {len(v)} elementos (tipo: {type(v[0]).__name__ if v else 'vacía'})]")
            else:
                print(f"{prefijo}  • {k}: objeto de clase {tipo}")
                if hasattr(v, "get_params"):
                    print(f"{prefijo}    - Parámetros: {v.get_params(deep=False)}")
                if hasattr(v, "classes_"):
                    print(f"{prefijo}    - Clases aprendidas: {v.classes_}")
    elif isinstance(obj, np.ndarray):
        print(f"{prefijo}ndarray shape={obj.shape}, dtype={obj.dtype}, min={np.min(obj):.4f}, max={np.max(obj):.4f}")
    else:
        tipo = type(obj).__name__
        print(f"{prefijo}Objeto de tipo: {tipo}")
        if hasattr(obj, "get_params"):
            print(f"{prefijo}Parámetros:")
            pprint.pprint(obj.get_params(deep=False))
        if hasattr(obj, "n_components_"):
            print(f"{prefijo}Componentes PCA aprendidas: {obj.n_components_}")
            if hasattr(obj, "explained_variance_ratio_"):
                var_total = np.sum(obj.explained_variance_ratio_) * 100.0
                print(f"{prefijo}Varianza total explicada: {var_total:.2f}%")


def inspeccionar_archivo(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        print(f"[ERROR] El archivo no existe: {ruta_archivo}")
        return

    print("=" * 80)
    print(f"ARCHIVO: {os.path.basename(ruta_archivo)}")
    print(f"Ruta: {ruta_archivo}")
    tam_kb = os.path.getsize(ruta_archivo) / 1024.0
    print(f"Tamaño: {tam_kb:.2f} KB ({tam_kb/1024.0:.2f} MB)" if tam_kb >= 1024 else f"Tamaño: {tam_kb:.2f} KB")
    print("-" * 80)

    try:
        contenido = joblib.load(ruta_archivo)
        describir_objeto(contenido)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el archivo: {e}")
    print("=" * 80 + "\n")


def main():
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
        if not os.path.isabs(ruta):
            if os.path.exists(os.path.join(DIR_MODELOS, ruta)):
                ruta = os.path.join(DIR_MODELOS, ruta)
        inspeccionar_archivo(ruta)
    else:
        print("\n" + "=" * 80)
        print("INSPECTOR DE ARCHIVOS SERIALIZADOS (.pkl) - SISTEMA BIOMÉTRICO")
        print("=" * 80)
        archivos = [f for f in os.listdir(DIR_MODELOS) if f.endswith(".pkl")]
        if not archivos:
            print(f"No se encontraron archivos .pkl en: {DIR_MODELOS}")
            return

        print(f"Archivos disponibles en '{DIR_MODELOS}':")
        for i, a in enumerate(archivos, 1):
            tam_kb = os.path.getsize(os.path.join(DIR_MODELOS, a)) / 1024.0
            print(f"  [{i}] {a} ({tam_kb:.1f} KB)")

        print("\nPuedes ejecutar:")
        print("  python src/inspeccionar_pkl.py <nombre_archivo.pkl>")
        print("Ejemplo:")
        print("  python src/inspeccionar_pkl.py metadata_clases.pkl")
        print("\n--- MOSTRANDO CONTENIDO DE: metadata_clases.pkl ---")
        inspeccionar_archivo(os.path.join(DIR_MODELOS, "metadata_clases.pkl"))


if __name__ == "__main__":
    main()
