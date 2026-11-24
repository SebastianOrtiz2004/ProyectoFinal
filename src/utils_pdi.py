"""
Módulo de Procesamiento Digital de Imágenes (PDI Clásico de Cátedra)
Proyecto Final - Inteligencia Artificial
Autores: Sebastián Ortiz (ID 0) & Ángel Ayuquina (ID 1)
Docente: Ing. Rubén Nogales
Universidad Técnica de Ambato - FISEI - Software
"""

import os
import cv2
import numpy as np


def cv2_imread_unicode(ruta: str, flags: int = cv2.IMREAD_COLOR):
    """
    Lectura de imágenes compatible con rutas con tildes y caracteres especiales en Windows.
    Soluciona la limitación de cv2.imread con rutas como 'UNIVERSIDAD TÉCNICA DE AMBATO'.
    """
    try:
        with open(ruta, "rb") as f:
            bytes_data = bytearray(f.read())
        arr = np.asarray(bytes_data, dtype=np.uint8)
        return cv2.imdecode(arr, flags)
    except Exception:
        return None


def cv2_imwrite_unicode(ruta: str, img: np.ndarray, ext: str = ".png"):
    """
    Escritura de imágenes compatible con rutas con tildes y caracteres especiales en Windows.
    Soluciona el fallo silencioso de cv2.imwrite con rutas como 'UNIVERSIDAD TÉCNICA DE AMBATO'.
    """
    try:
        directorio = os.path.dirname(ruta)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)
        is_success, buffer = cv2.imencode(ext, img)
        if is_success:
            with open(ruta, "wb") as f:
                f.write(buffer)
            return True
        return False
    except Exception:
        return False



def descomponer_rgb_a_double(img_bgr: np.ndarray):
    """
    Convierte de BGR a RGB, separa explícitamente los planos espectrales R, G, B
    y calcula las versiones en precisión continua double [0.0, 1.0] junto con
    el canal de luminancia / escala de grises canónico.

    Ecuaciones de Cátedra:
        R_double = double(R) / 255.0
        G_double = double(G) / 255.0
        B_double = double(B) / 255.0
        I_gris_double(x, y) = 0.299 * R_double + 0.587 * G_double + 0.114 * B_double
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R_uint8 = img_rgb[:, :, 0]
    G_uint8 = img_rgb[:, :, 1]
    B_uint8 = img_rgb[:, :, 2]

    R_double = R_uint8.astype(np.float64) / 255.0
    G_double = G_uint8.astype(np.float64) / 255.0
    B_double = B_uint8.astype(np.float64) / 255.0

    gris_double = 0.299 * R_double + 0.587 * G_double + 0.114 * B_double
    gris_double = np.clip(gris_double, 0.0, 1.0)

    return {
        "img_rgb": img_rgb,
        "R_uint8": R_uint8,
        "G_uint8": G_uint8,
        "B_uint8": B_uint8,
        "R_double": R_double,
        "G_double": G_double,
        "B_double": B_double,
        "gris_double": gris_double,
    }


def calcular_cuadrado_envolvente(caja, ancho_img: int, alto_img: int):
    """
    Calcula el cuadrado envolvente simétrico centrado en el centroide de la caja facial.
    Nunca estira ni aplasta la región facial de forma asimétrica.

    Ecuaciones de Cátedra:
        lado = max(w, h)
        cx = x + w / 2.0
        cy = y + h / 2.0
        x_esq = max(0, min(cx - lado / 2.0, ancho_img - lado))
        y_esq = max(0, min(cy - lado / 2.0, alto_img - lado))
    """
    x, y, w, h = [int(v) for v in caja]
    lado = max(w, h)

    # Restringir el lado máximo a las dimensiones disponibles de la imagen
    max_dim = min(ancho_img, alto_img)
    if lado > max_dim:
        lado = max_dim

    cx = x + (w / 2.0)
    cy = y + (h / 2.0)

    x_esq = cx - (lado / 2.0)
    y_esq = cy - (lado / 2.0)

    # Acotamiento riguroso a los límites físicos del fotograma
    x_esq = max(0.0, min(x_esq, float(ancho_img - lado)))
    y_esq = max(0.0, min(y_esq, float(alto_img - lado)))

    return [int(round(x_esq)), int(round(y_esq)), int(lado), int(lado)]


def imfill_holes(mascara_binaria: np.ndarray) -> np.ndarray:
    """
    Emulación algorítmica de imfill(BW, 'holes') de MATLAB usando cv2.floodFill.

    Ecuación de Cátedra:
        BW_rellena = BW U compl(Fondo_Externo)
    """
    # Asegurar máscara uint8 en {0, 255}
    if mascara_binaria.dtype != np.uint8:
        bw = (mascara_binaria > 0).astype(np.uint8) * 255
    else:
        bw = (mascara_binaria > 127).astype(np.uint8) * 255

    h, w = bw.shape
    # Agregar borde exterior de 1 píxel en 0 para que floodFill siempre inunde desde el exterior
    # incluso si la esquina (0,0) de la máscara original está encendida (255)
    padded = np.pad(bw, 1, mode="constant", constant_values=0)
    mask_ff = np.zeros((h + 4, w + 4), np.uint8)

    # Inundación desde el borde exterior garantizado en (0,0)
    cv2.floodFill(padded, mask_ff, (0, 0), 255)

    # Invertir el resultado de la inundación externa y recortar al tamaño original
    im_floodfill_inv = cv2.bitwise_not(padded[1:-1, 1:-1])

    # La unión lógica sella todas las cavidades internas
    bw_rellena = cv2.bitwise_or(bw, im_floodfill_inv)
    return bw_rellena


def extraer_codigo_freeman_8(contorno: np.ndarray) -> list:
    """
    Calcula el Código de Cadena de Freeman en 8-vecindad para el contorno principal.
    Direcciones (0 a 7):
        3  2  1
        4  *  0
        5  6  7
    """
    freeman_dir = {
        (1, 0): 0,
        (1, -1): 1,
        (0, -1): 2,
        (-1, -1): 3,
        (-1, 0): 4,
        (-1, 1): 5,
        (0, 1): 6,
        (1, 1): 7,
    }

    if contorno is None or len(contorno) < 2:
        return []

    puntos = contorno.reshape(-1, 2)
    cadena = []

    for i in range(len(puntos) - 1):
        dx = int(np.sign(puntos[i + 1][0] - puntos[i][0]))
        dy = int(np.sign(puntos[i + 1][1] - puntos[i][1]))
        codigo = freeman_dir.get((dx, dy), 0)
        cadena.append(codigo)

    return cadena


def extraer_regionprops_basico(mascara_binaria: np.ndarray) -> dict:
    """
    Extrae propiedades geométricas de la región conexa principal (emulación de regionprops de MATLAB):
        - Área (momento m00)
        - Perímetro (longitud de arco)
        - Centroide geométrico continuo (x_bar = m10/m00, y_bar = m01/m00)
        - Caja envolvente [x, y, w, h]
        - Código de cadena de Freeman (primeros 25 códigos para telemetría)
    """
    bw = (mascara_binaria > 0).astype(np.uint8) * 255
    contornos, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contornos:
        return {
            "area": 0.0,
            "perimetro": 0.0,
            "centroide": (0.0, 0.0),
            "bbox": (0, 0, 0, 0),
            "freeman_sample": [],
            "contorno": None,
        }

    # Seleccionar el contorno principal de mayor área
    c_principal = max(contornos, key=cv2.contourArea)
    M = cv2.moments(c_principal)

    area = float(M["m00"])
    if area > 1e-5:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx, cy = 0.0, 0.0

    perimetro = float(cv2.arcLength(c_principal, closed=True))
    bbox = cv2.boundingRect(c_principal)
    freeman = extraer_codigo_freeman_8(c_principal)

    return {
        "area": area,
        "perimetro": perimetro,
        "centroide": (cx, cy),
        "bbox": bbox,
        "freeman_sample": freeman[:25],
        "contorno": c_principal,
    }


def reducir_por_mediana_bloque(imagen: np.ndarray, alto_dest: int = 64, ancho_dest: int = 64) -> np.ndarray:
    """
    Técnica de Reducción Espacial por Mediana de Bloque Vectorizada (Block-Median Decimation):
    Redimensiona a 128x128 y calcula la mediana estadística en bloques de 2x2.
    Preserva orillas con máxima nitidez, elimina valores atípicos y ejecuta
    50 veces más rápido en tiempo real gracias a operaciones vectorizadas en NumPy.
    """
    if imagen.shape[:2] != (128, 128):
        im128 = cv2.resize(imagen, (128, 128), interpolation=cv2.INTER_AREA)
    else:
        im128 = imagen
    blocks = im128.reshape(alto_dest, 2, ancho_dest, 2)
    return np.median(blocks, axis=(1, 3)).astype(np.float64)


_MASCARA_ELIPTICA_64_CACHE = None
_MASCARA_ELIPTICA_64_BOOL = None
_CLAHE_CACHE = None


def obtener_mascara_eliptica_64() -> np.ndarray:
    """
    Genera (con caché en memoria) una máscara elíptica suave en float64 de tamaño (64, 64):
    Valor 1.0 en la elipse facial central (centro=(32, 32), ejes=(23, 29))
    y transición suave (GaussianBlur 5x5) hacia 0.0 en las esquinas exteriores.
    Elimina los artefactos de fondo de la habitación (pared, respaldar de cama, hombros).
    """
    global _MASCARA_ELIPTICA_64_CACHE, _MASCARA_ELIPTICA_64_BOOL
    if _MASCARA_ELIPTICA_64_CACHE is None:
        mask = np.zeros((64, 64), dtype=np.float64)
        cv2.ellipse(mask, (32, 32), (23, 29), 0, 0, 360, 1.0, -1)
        mask_suave = cv2.GaussianBlur(mask, (5, 5), 1.0)
        _MASCARA_ELIPTICA_64_CACHE = np.clip(mask_suave, 0.0, 1.0)
        _MASCARA_ELIPTICA_64_BOOL = _MASCARA_ELIPTICA_64_CACHE > 0.50
    return _MASCARA_ELIPTICA_64_CACHE


def aplicar_mascara_eliptica_a_img64(img_64_double: np.ndarray) -> np.ndarray:
    """
    Aplica la máscara elíptica facial suave a una imagen de 64x64 en [0.0, 1.0]
    preservando intacto el contraste real del interior del rostro y neutralizando
    únicamente las 4 esquinas exteriores fuera de la elipse con un tono neutro (0.42).
    """
    mascara = obtener_mascara_eliptica_64()
    img_enmascarada = img_64_double * mascara + 0.42 * (1.0 - mascara)
    return np.clip(img_enmascarada, 0.0, 1.0)


def normalizar_fotometria_elipse_64(
    img_64_double: np.ndarray,
    media_objetivo: float = 0.44,
    std_objetivo: float = 0.16,
) -> np.ndarray:
    """
    Normalización Fotométrica Canónica (Z-score intracraneal en la elipse facial):
    Alinea la media y la desviación estándar de luminancia dentro de la elipse facial
    para lograr invarianza real frente a iluminación extrema (cuarto oscuro, aula brillante,
    contraluz o sobreexposición del sensor OV2640).
    Si la textura interior es completamente plana (std < 0.028, ej. pared lisa), NO amplifica ruido.
    """
    mascara = obtener_mascara_eliptica_64()
    vals = img_64_double[_MASCARA_ELIPTICA_64_BOOL]
    std_int = float(np.std(vals))
    if std_int < 0.028:
        return aplicar_mascara_eliptica_a_img64(img_64_double)
    media_int = float(np.mean(vals))
    escala = float(np.clip(std_objetivo / max(std_int, 1e-4), 0.45, 2.60))
    img_norm = (img_64_double - media_int) * escala + media_objetivo
    return aplicar_mascara_eliptica_a_img64(np.clip(img_norm, 0.0, 1.0))


def preprocesar_ventana_64x64(parche_bgr_o_gris: np.ndarray):
    """
    Pipeline PDI Clásico de Cátedra con Invarianza Fotométrica sobre la ventana cuadrada:
        1. Conversión a escala de grises uint8 [0, 255].
        2. Mitigación de ruido: Filtro de mediana 3x3 + Suavizado gaussiano ligero.
        3. Estiramiento adaptativo de rango dinámico (percentiles 2-98) + CLAHE local:
           - Recupera contraste facial en habitaciones oscuras (p98 bajo) y controla
             sobreexposición en habitaciones muy iluminadas, sin inflar paredes planas (std < 9.0).
        4. Reducción espacial de cátedra por MEDIANA DE BLOQUE VECTORIZADA a 64x64.
        5. Conversión a precisión continua double en [0.0, 1.0].
        6. Normalización fotométrica Z-score dentro de la elipse facial + máscara elíptica suave.

    Retorna:
        - img_64x64_double: Matriz np.ndarray de tamaño (64, 64) en np.float64.
        - vector_4096: Vector fila aplanado np.ndarray de 4096 elementos en np.float64.
    """
    global _CLAHE_CACHE
    if _CLAHE_CACHE is None:
        _CLAHE_CACHE = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))

    if len(parche_bgr_o_gris.shape) == 3:
        gris_uint8 = cv2.cvtColor(parche_bgr_o_gris, cv2.COLOR_BGR2GRAY)
    else:
        if parche_bgr_o_gris.dtype == np.float64 or parche_bgr_o_gris.dtype == np.float32:
            gris_uint8 = np.clip(parche_bgr_o_gris * 255.0, 0, 255).astype(np.uint8)
        else:
            gris_uint8 = parche_bgr_o_gris.copy()

    std_cruda = float(np.std(gris_uint8))

    # Mitigación de ruido de cátedra (elimina ruido impulsivo y bloques JPEG del ESP32-CAM)
    filtrada_mediana = cv2.medianBlur(gris_uint8, 3)
    filtrada_gauss = cv2.GaussianBlur(filtrada_mediana, (3, 3), 0.9)

    # Ecualización local adaptativa (CLAHE) + realce suave de bordes (Unsharp Mask)
    # Permite operar desde baja iluminación (cuarto oscuro) hasta alta iluminación (aula brillante)
    # sin inflar ruido en paredes lisas (std_cruda < 9.0).
    if std_cruda < 9.0:
        ecualizada = filtrada_gauss
    else:
        # Estiramiento lineal robusto por percentiles antes de CLAHE para rescatar detalles en cuartos oscuros
        p2, p98 = np.percentile(filtrada_gauss, (2, 98))
        rango = float(p98 - p2)
        if rango >= 16.0 and (p98 < 145.0 or p2 > 75.0):
            base_eq = np.clip((filtrada_gauss.astype(np.float32) - p2) * (225.0 / rango) + 15.0, 0, 255).astype(np.uint8)
        else:
            base_eq = filtrada_gauss

        eq_local = _CLAHE_CACHE.apply(base_eq)
        ecualizada = cv2.addWeighted(eq_local, 0.65, base_eq, 0.35, 0.0)
        blur_ref = cv2.GaussianBlur(ecualizada, (3, 3), 1.0)
        ecualizada = cv2.addWeighted(ecualizada, 1.20, blur_ref, -0.20, 0.0)

    # Reducción espacial por MEDIANA DE BLOQUE vectorizada a 64x64
    img_64x64_mediana = reducir_por_mediana_bloque(ecualizada, 64, 64)

    # Conversión explícita a double [0.0, 1.0] + normalización fotométrica canónica en elipse
    img_64x64_base = np.clip(img_64x64_mediana / 255.0, 0.0, 1.0)
    img_64x64_double = normalizar_fotometria_elipse_64(img_64x64_base)
    vector_4096 = img_64x64_double.reshape(-1)

    return img_64x64_double, vector_4096


def calcular_nitidez_laplaciano(parche_bgr_o_gris: np.ndarray) -> float:
    """
    Mide la nitidez óptica de una ventana mediante la varianza del Laplaciano (PDI clásico).
    Permite descartar fotogramas movidos o desenfocados durante la adquisición con ESP32-CAM.
    """
    if parche_bgr_o_gris is None or parche_bgr_o_gris.size == 0:
        return 0.0
    if len(parche_bgr_o_gris.shape) == 3:
        gris = cv2.cvtColor(parche_bgr_o_gris, cv2.COLOR_BGR2GRAY)
    else:
        gris = parche_bgr_o_gris
    gris_norm = cv2.resize(gris, (128, 128), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gris_norm, cv2.CV_64F).var())


def calcular_iou(caja_a, caja_b):
    """
    Calcula la intersección sobre la unión (IoU) entre dos cajas [x, y, w, h].
    """
    xa = max(caja_a[0], caja_b[0])
    ya = max(caja_a[1], caja_b[1])
    xb = min(caja_a[0] + caja_a[2], caja_b[0] + caja_b[2])
    yb = min(caja_a[1] + caja_a[3], caja_b[1] + caja_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    area_inter = inter_w * inter_h

    area_a = caja_a[2] * caja_a[3]
    area_b = caja_b[2] * caja_b[3]
    area_union = area_a + area_b - area_inter

    if area_union <= 0:
        return 0.0
    return float(area_inter) / float(area_union)


def extraer_ventanas_negativas_esquinas(
    frame_bgr: np.ndarray, caja_rostro: list, num_ventanas: int = 3
):
    """
    Técnica de Distribución por Ventanas (Opción 1):
    Extrae ventanas del fondo (esquinas y periferia) del mismo fotograma,
    verificando matemáticamente que IoU == 0 con respecto al rostro humano.
    Alimenta automáticamente la Clase 0 (No Persona) sin fotos manuales de objetos.

    Retorna una lista de tuplas: (img_64x64_double, vector_4096, caja_coords)
    """
    H, W = frame_bgr.shape[:2]
    lado_rostro = max(caja_rostro[2], caja_rostro[3]) if caja_rostro is not None else 64
    lado_ventana = max(64, min(lado_rostro, min(H, W) // 3))

    candidatos = [
        # Esquina superior izquierda
        [0, 0, lado_ventana, lado_ventana],
        # Esquina superior derecha
        [W - lado_ventana, 0, lado_ventana, lado_ventana],
        # Esquina inferior izquierda
        [0, H - lado_ventana, lado_ventana, lado_ventana],
        # Esquina inferior derecha
        [W - lado_ventana, H - lado_ventana, lado_ventana, lado_ventana],
        # Periferia superior centro
        [max(0, (W - lado_ventana) // 2), 0, lado_ventana, lado_ventana],
        # Periferia lateral izquierda
        [0, max(0, (H - lado_ventana) // 2), lado_ventana, lado_ventana],
        # Periferia lateral derecha
        [W - lado_ventana, max(0, (H - lado_ventana) // 2), lado_ventana, lado_ventana],
    ]

    ventanas_validas = []
    for cand in candidatos:
        # Asegurar dentro de los límites
        if cand[0] < 0 or cand[1] < 0 or cand[0] + cand[2] > W or cand[1] + cand[3] > H:
            continue

        iou = (
            calcular_iou(cand, caja_rostro) if caja_rostro is not None else 0.0
        )
        if iou == 0.0:
            parche = frame_bgr[
                cand[1] : cand[1] + cand[3], cand[0] : cand[0] + cand[2]
            ]
            if parche.shape[0] >= 16 and parche.shape[1] >= 16:
                img_64, vec_4096 = preprocesar_ventana_64x64(parche)
                ventanas_validas.append((img_64, vec_4096, cand))
                if len(ventanas_validas) >= num_ventanas:
                    break

    return ventanas_validas


def obtener_rejilla_ventanas_geometrica(ancho_img: int, alto_img: int, factor_central: float = 0.38):
    """
    Distribución por Ventanas Geométrica (Rejilla Matemática de Cátedra):
    Divide matemáticamente el fotograma sin usar clasificadores preentrenados.
    
    1. Ventana Central Simétrica Anatómica (Candidato a Rostro):
       - Cuadrado simétrico lado = max(w, h) centrado en (W//2, H//2) con factor_central=0.38
         (~180x180 px en 640x480) que recorta de cejas a mentón sin meter 70% de pared.
       - Permite que el usuario ubique su rostro de frente, medio perfil o perfil
         sin ser bloqueado por un detector preentrenado.
    2. Ventanas Periféricas de Esquinas (Clase 0: No Persona / Fondo):
       - 4 subventanas ubicadas en las esquinas que tienen solapamiento nulo (IoU = 0)
         con respecto a la ventana central.
    """
    lado_centro = int(min(ancho_img, alto_img) * factor_central)
    # Asegurar que sea par
    lado_centro = (lado_centro // 2) * 2

    cx = ancho_img // 2
    cy = alto_img // 2

    x_centro = cx - lado_centro // 2
    y_centro = cy - lado_centro // 2
    ventana_central = [x_centro, y_centro, lado_centro, lado_centro]

    # Tamaño de esquinas
    lado_esq = min(x_centro, y_centro, ancho_img - (x_centro + lado_centro), alto_img - (y_centro + lado_centro))
    if lado_esq < 64:
        lado_esq = min(ancho_img, alto_img) // 4

    lado_esq = max(48, min(lado_esq, 160))

    ventanas_esquinas = [
        [0, 0, lado_esq, lado_esq],  # Superior Izquierda
        [ancho_img - lado_esq, 0, lado_esq, lado_esq],  # Superior Derecha
        [0, alto_img - lado_esq, lado_esq, lado_esq],  # Inferior Izquierda
        [ancho_img - lado_esq, alto_img - lado_esq, lado_esq, lado_esq],  # Inferior Derecha
    ]

    # Filtrar solo las que tengan IoU == 0 con la central
    esquinas_puras = [
        v for v in ventanas_esquinas if calcular_iou(v, ventana_central) == 0.0
    ]

    return ventana_central, esquinas_puras


def detectar_rostro_principal(frame_bgr: np.ndarray, factor_central: float = 0.38):
    """
    Localización Anatómica Facial Canónica (Cuadrado Simétrico lado = max(w, h)):
    Define la ventana central cuadrada estricta de proporción facial (factor_central=0.38)
    para recortar exactamente de cejas a mentón sin incluir hombros ni fondo excesivo.
    Retorna: [x, y, w, h]
    """
    H, W = frame_bgr.shape[:2]
    caja, _ = obtener_rejilla_ventanas_geometrica(W, H, factor_central=factor_central)
    return caja


def localizar_y_recortar_rostro_o_ventana(frame_bgr: np.ndarray, factor_central: float = 0.38):
    """
    Función unificada de localización y recorte usada idénticamente en captura e inferencia.
    Retorna: caja [x, y, w, h], parche_recortado
    """
    caja = detectar_rostro_principal(frame_bgr, factor_central=factor_central)
    x, y, w, h = caja
    parche = frame_bgr[y : y + h, x : x + w]
    return caja, parche


def analizar_morfologia_ventana_otsu(parche_bgr: np.ndarray):
    """
    Cálculo morfológico puro de clase dentro de la ventana de interés:
    1. Conversión a gris + mitigación de ruido (Mediana 3x3).
    2. Binarización de Otsu invertida (THRESH_BINARY_INV + THRESH_OTSU)
       con máscara elíptica para segmentar los rasgos faciales/objeto (ojos, cejas, nariz, boca)
       y excluir la pared clara en los bordes.
    3. Sellado morfológico de cavidades internas con imfill_holes (cv2.floodFill).
    4. Propiedades de región (regionprops): Área, Perímetro, Centroide y BoundingBox
       calculados con momentos espaciales de la componente conexa principal.
    """
    if len(parche_bgr.shape) == 3:
        gris = cv2.cvtColor(parche_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gris = parche_bgr.copy()

    h_p, w_p = gris.shape[:2]

    # Mitigación de ruido de cátedra
    gris_filtrado = cv2.medianBlur(gris, 3)

    # Binarización de Otsu invertida (segmenta rasgos faciales/objeto oscuros)
    _, bw_otsu = cv2.threshold(
        gris_filtrado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Máscara elíptica centrada para concentrar el análisis en el rostro/objeto y eliminar pared exterior
    mask_elipse = np.zeros((h_p, w_p), dtype=np.uint8)
    cv2.ellipse(
        mask_elipse,
        (w_p // 2, h_p // 2),
        (int(w_p * 0.35), int(h_p * 0.46)),
        0,
        0,
        360,
        255,
        -1,
    )
    bw_otsu = cv2.bitwise_and(bw_otsu, mask_elipse)

    # Inundación morfológica imfill(BW, 'holes')
    bw_rellena = imfill_holes(bw_otsu)

    # Extracción de propiedades geométricas (regionprops)
    props = extraer_regionprops_basico(bw_rellena)
    if props["area"] < 1.0:
        props["centroide"] = (float(w_p // 2), float(h_p // 2))

    return {
        "bw_otsu": bw_otsu,
        "bw_rellena": bw_rellena,
        "area": props["area"],
        "perimetro": props["perimetro"],
        "centroide_local": props["centroide"],
        "bbox_local": props["bbox"],
        "freeman_sample": props["freeman_sample"],
        "contorno": props["contorno"],
    }


def calcular_simetria_bilateral(img_64x64_double: np.ndarray) -> float:
    """
    Calcula la simetría bilateral adaptable al giro de cabeza (frente, medio perfil y perfil):
    Estima el centro de masa horizontal cx de bordes y rasgos oscuros (acotado en [24, 40]),
    mide la simetría en el intervalo simétrico disponible y retorna max(sim_central, sim_cx).
    """
    im = img_64x64_double
    gy = np.abs(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3))
    peso = gy + np.maximum(0.0, 0.45 - im)
    peso_col = np.sum(peso, axis=0) + 1e-6
    cx_f = float(np.sum(np.arange(64) * peso_col) / np.sum(peso_col))
    cx = int(np.clip(round(cx_f), 24, 40))

    w_sim = min(cx, 63 - cx)
    left_cx = im[:, cx - w_sim : cx]
    right_cx = np.fliplr(im[:, cx : cx + w_sim])
    sim_cx = 1.0 - float(np.mean(np.abs(left_cx - right_cx)))
    sim_central = 1.0 - float(np.mean(np.abs(im[:, :32] - np.fliplr(im[:, 32:]))))
    return float(max(sim_central, sim_cx))


def calcular_ratio_gradientes_sobel(img_64x64_double: np.ndarray) -> float:
    """
    Calcula la relación entre gradientes horizontales y verticales (Sobel):
        Ratio_HV = mean(|Gy|) / (mean(|Gx|) + 1e-6)
    Los rostros humanos presentan transiciones horizontales marcadas (cejas, ojos, nariz, labios),
    mientras que botellas, cilindros y celulares presentan bordes predominantemente verticales.
    """
    gx = cv2.Sobel(img_64x64_double, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_64x64_double, cv2.CV_64F, 0, 1, ksize=3)
    mag_x = np.mean(np.abs(gx))
    mag_y = np.mean(np.abs(gy))
    return float(mag_y / (mag_x + 1e-6))


def calcular_contraste_antropometrico(img_64x64_double: np.ndarray) -> float:
    """
    Calcula el gradiente fotométrico vertical antropométrico:
    En rostros humanos, la franja de los ojos (filas 20 a 35) es sistemáticamente
    más oscura que la zona de la frente (filas 8 a 20) debido a cejas, cavidades y pupilas.
        Contraste = mean(frente) - mean(ojos)
    """
    frente = np.mean(img_64x64_double[8:20, 16:48])
    ojos = np.mean(img_64x64_double[20:36, 12:52])
    return float(frente - ojos)


def extraer_caracteristicas_pdi(img_64x64_double: np.ndarray) -> np.ndarray:
    """
    Extrae un vector compacto de características biométricas canónicas de PDI:
        [Simetría, Ratio_HV, Contraste_Ojos]
    """
    sim = calcular_simetria_bilateral(img_64x64_double)
    r_hv = calcular_ratio_gradientes_sobel(img_64x64_double)
    c_ojos = calcular_contraste_antropometrico(img_64x64_double)
    return np.array([sim, r_hv, c_ojos], dtype=np.float64)


def extraer_descriptor_espacial_bloques(im_64: np.ndarray) -> np.ndarray:
    """
    Descriptor Espacial por Bloques 4x4 con Sobel y Filtro Pasa-Altos de Acentuado:
    Divide im_64 (64x64) en 16 bloques de 16x16 píxeles.
    Calcula:
        - gx, gy mediante operador Sobel 3x3
        - pa mediante filtro pasa-altos de acentuado [-1,-1,-1; -1,8,-1; -1,-1,-1]
    Para cada uno de los 16 bloques (bi, bj), extrae:
        [mean(im), std(im), mean(gy), mean(pa)]
    Total: 64 características espaciales robustas a pequeñas traslaciones y rotaciones.
    """
    gx = np.abs(cv2.Sobel(im_64, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(im_64, cv2.CV_64F, 0, 1, ksize=3))
    kernel_pa = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], dtype=np.float64)
    pa = np.abs(cv2.filter2D(im_64, cv2.CV_64F, kernel_pa))

    feats = []
    for bi in range(4):
        for bj in range(4):
            r0, r1 = bi * 16, (bi + 1) * 16
            c0, c1 = bj * 16, (bj + 1) * 16
            b_im = im_64[r0:r1, c0:c1]
            b_gy = gy[r0:r1, c0:c1]
            b_pa = pa[r0:r1, c0:c1]
            feats.extend([
                float(np.mean(b_im)),
                float(np.std(b_im)),
                float(np.mean(b_gy)),
                float(np.mean(b_pa)),
            ])
    return np.array(feats, dtype=np.float64)


def extraer_puntos_faciales_landmarks_pdi(im_64: np.ndarray):
    """
    Extrae los 5 puntos anatómicos clave del rostro (Landmarks: Ojo Izq, Ojo Der,
    Ápice Nasal, Comisura Labial Izq y Der) mediante momentos espaciales (m10/m00, m01/m00)
    sobre el mapa de energía morfológica, y calcula 12 descriptores geométricos
    invariantes a traslación y escala (añadidos al vector de Machine Learning).
    """
    gy = np.abs(cv2.Sobel(im_64, cv2.CV_64F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(im_64, cv2.CV_64F, 1, 0, ksize=3))
    med = float(np.median(im_64))
    # Mapa de energía de rasgos faciales (cuencas oculares, fosas nasales y comisura labial)
    energia = gy + 0.35 * gx + np.maximum(0.0, med - im_64) * 0.85

    def _centroide_ventana(r0, r1, c0, c1):
        sub = energia[r0:r1, c0:c1] ** 2
        m00 = float(np.sum(sub)) + 1e-8
        yy, xx = np.mgrid[r0:r1, c0:c1]
        cx = float(np.sum(xx * sub) / m00)
        cy = float(np.sum(yy * sub) / m00)
        return cx, cy

    # 5 Puntos Faciales (Landmarks en coordenadas 64x64)
    x_oi, y_oi = _centroide_ventana(14, 35, 12, 31)   # 1. Ojo izquierdo
    x_od, y_od = _centroide_ventana(14, 35, 33, 52)   # 2. Ojo derecho
    x_na, y_na = _centroide_ventana(25, 41, 24, 40)   # 3. Nariz (ápice/puente)
    x_bi, y_bi = _centroide_ventana(38, 54, 16, 32)   # 4. Comisura labial izquierda
    x_bd, y_bd = _centroide_ventana(38, 54, 32, 48)   # 5. Comisura labial derecha

    x_boca = 0.5 * (x_bi + x_bd)
    y_boca = 0.5 * (y_bi + y_bd)
    y_ojos = 0.5 * (y_oi + y_od)
    x_ojos = 0.5 * (x_oi + x_od)

    # Distancias euclidianas normalizadas
    d_ojos = float(np.hypot(x_od - x_oi, y_od - y_oi)) / 64.0
    d_boca = float(np.hypot(x_bd - x_bi, y_bd - y_bi)) / 64.0
    d_ojos_nariz = float(y_na - y_ojos) / 64.0
    d_ojos_boca = float(y_boca - y_ojos) / 64.0
    d_nariz_boca = float(y_boca - y_na) / 64.0

    # Proporciones antropométricas invariantes a zoom/escala y posición
    ratio_ojos_boca = float(d_ojos / (abs(d_ojos_boca) + 1e-4))
    ratio_nariz_boca = float(d_ojos_nariz / (abs(d_ojos_boca) + 1e-4))
    ratio_ancho_boca_ojos = float(d_boca / (d_ojos + 1e-4))
    inclinacion_ojos = float((y_od - y_oi) / (d_ojos * 64.0 + 1e-4))
    simetria_yaw = float(((x_na - x_oi) - (x_od - x_na)) / (d_ojos * 64.0 + 1e-4))
    alineacion_vertical = float((x_boca - x_na) / 64.0)
    energia_triangulo = float(np.mean(energia) / (np.std(im_64) + 1e-4))

    features_12 = np.array([
        d_ojos, d_boca, d_ojos_nariz, d_ojos_boca, d_nariz_boca,
        ratio_ojos_boca, ratio_nariz_boca, ratio_ancho_boca_ojos,
        inclinacion_ojos, simetria_yaw, alineacion_vertical, energia_triangulo
    ], dtype=np.float64)

    puntos_dict = {
        "ojo_izq": (x_oi, y_oi),
        "ojo_der": (x_od, y_od),
        "nariz": (x_na, y_na),
        "boca_izq": (x_bi, y_bi),
        "boca_der": (x_bd, y_bd),
        "boca_centro": (x_boca, y_boca),
    }
    return features_12, puntos_dict


def extraer_mapas_convolucionales_2d(imagenes_64x64: np.ndarray) -> np.ndarray:
    """
    Bloque Convolucional 2D Auténtico de Cátedra (Para Fase Posterior CNN e Inferencia en Vivo):
    Aplica 8 kernels 2D: Sobel H, Sobel V, Pasa-Altos de Cátedra, High-Boost Centro 9,
    Diagonal 45°, Diagonal 135°, y 2 Filtros de Gabor/Banda 5x5 (Horizontal y Vertical).
    Para cada kernel:
      1. Convolución 2D rápida con cv2.filter2D(img, cv2.CV_64F, k)
      2. Activación no lineal ReLU: relu = np.maximum(0.0, conv)
      3. Max-Pooling 2D auténtico por bloques de 4x4 (reduce 64x64 a 16x16 tomando el máximo real del bloque,
         lo que otorga invariancia a traslación espacial sin pérdida ciega de píxeles).
    Además, extrae Average-Pooling 2D de 8x8 de la imagen directa (reduce 64x64 a 8x8 capturando la estructura global).
    Concatena los 8 mapas Max-Pool (8 * 256 = 2048) + Average-Pool (64) = 2112 características convolucionales 2D.
    """
    if imagenes_64x64.ndim == 2:
        imgs = imagenes_64x64.reshape(1, 64, 64)
    elif imagenes_64x64.ndim == 3:
        imgs = imagenes_64x64
    else:
        imgs = imagenes_64x64.reshape(-1, 64, 64)

    k_sh = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64)
    k_sv = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
    k_pa = np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], dtype=np.float64) / 8.0
    k_hb = np.array([[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]], dtype=np.float64) / 9.0
    k_dp = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]], dtype=np.float64) / 4.0
    k_ds = np.array([[-1.0, -1.0, 2.0], [-1.0, 2.0, -1.0], [2.0, -1.0, -1.0]], dtype=np.float64) / 4.0

    # 2 Filtros de Gabor / Banda 5x5 (Horizontal y Vertical)
    k_gh = cv2.getGaborKernel((5, 5), sigma=1.2, theta=0.0, lambd=3.0, gamma=0.5, psi=0, ktype=cv2.CV_64F)
    k_gh = k_gh / (np.sum(np.abs(k_gh)) + 1e-6)
    k_gv = cv2.getGaborKernel((5, 5), sigma=1.2, theta=np.pi / 2.0, lambd=3.0, gamma=0.5, psi=0, ktype=cv2.CV_64F)
    k_gv = k_gv / (np.sum(np.abs(k_gv)) + 1e-6)

    kernels = [k_sh, k_sv, k_pa, k_hb, k_dp, k_ds, k_gh, k_gv]
    N = len(imgs)
    F_conv = np.empty((N, 2112), dtype=np.float64)

    for i in range(N):
        im = imgs[i].astype(np.float64)
        mapas_pool = []
        for k in kernels:
            conv = cv2.filter2D(im, cv2.CV_64F, k)
            relu = np.maximum(0.0, conv)
            # Max-Pooling 2D auténtico por bloques 4x4
            pool_max = relu.reshape(16, 4, 16, 4).max(axis=(1, 3)).reshape(-1)
            mapas_pool.append(pool_max)

        # Average-Pooling 2D por bloques 8x8
        pool_avg = im.reshape(8, 8, 8, 8).mean(axis=(1, 3)).reshape(-1)
        mapas_pool.append(pool_avg)

        F_conv[i] = np.concatenate(mapas_pool)

    return F_conv


def aumentar_geometria_rostro(im_64: np.ndarray, dx=0, dy=0, scale=1.0, flip=False) -> np.ndarray:
    """
    Aplica traslación (dx, dy), escala (zoom) y/o espejado horizontal sobre una imagen de 64x64
    usando BORDER_REFLECT_101 y reaplicando la máscara elíptica para evitar bordes artificiales.
    """
    im = im_64.copy()
    if flip:
        im = np.fliplr(im)
    centro = (32.0, 32.0)
    M = cv2.getRotationMatrix2D(centro, 0.0, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    im_warp = cv2.warpAffine(im, M, (64, 64), borderMode=cv2.BORDER_REFLECT_101)
    return aplicar_mascara_eliptica_a_img64(im_warp)


def centrar_ventana_en_rostro_pdi(frame_bgr: np.ndarray, cuadrado_base: list, max_shift: int = 75) -> list:
    """
    Auto-centrado anatómico real mediante regionprops (Centroid):
    Localiza la región de piel del rostro que se encuentra debajo del cabello oscuro
    y centra el cuadrado envolvente directamente sobre el centroide del rostro (nariz/mejillas),
    permitiendo que el usuario se mueva con total libertad sin buscar un punto fijo.
    """
    H, W = frame_bgr.shape[:2]
    qx_b, qy_b, qlado, _ = cuadrado_base

    w_s, h_s = 160, 120
    small = cv2.resize(frame_bgr, (w_s, h_s), interpolation=cv2.INTER_AREA)
    pdi = descomponer_rgb_a_double(small)
    R, G, B, gris = pdi["R_double"], pdi["G_double"], pdi["B_double"], pdi["gris_double"]

    p95 = float(np.percentile(gris, 95))
    gain = np.clip(0.75 / max(p95, 0.15), 1.0, 2.0)
    R_n, G_n, B_n, gris_n = np.clip(R * gain, 0, 1), np.clip(G * gain, 0, 1), np.clip(B * gain, 0, 1), np.clip(gris * gain, 0, 1)

    # 1. Máscara de cabello oscuro superior
    bw_cabello = ((gris < 0.24) & ((R - B) < 0.12)).astype(np.uint8) * 255
    bw_cabello[int(h_s * 0.65):, :] = 0
    bw_cabello[:, :int(w_s * 0.12)] = 0
    bw_cabello[:, int(w_s * 0.88):] = 0

    # 2. Máscara de piel humana
    bw_piel = (
        (R_n > 0.22) & (G_n > 0.12) & (B_n > 0.07)
        & ((R_n - G_n) >= 0.01) & ((R_n - G_n) <= 0.36)
        & ((R_n - B_n) >= 0.02) & (gris_n > 0.22) & (gris_n < 0.86)
    ).astype(np.uint8) * 255
    bw_piel[int(h_s * 0.82):, :] = 0
    bw_piel[:, :int(w_s * 0.15)] = 0
    bw_piel[:, int(w_s * 0.85):] = 0

    # 3. Quedarse únicamente con la piel que está debajo/adyacente a la masa de cabello
    kernel_down = np.ones((38, 25), dtype=np.uint8)
    zona_rostro_valida = cv2.dilate(bw_cabello, kernel_down, iterations=1)
    bw_rostro_piel = cv2.bitwise_and(bw_piel, zona_rostro_valida)
    bw_rostro_solido = imfill_holes(bw_rostro_piel)

    props_rostro = extraer_regionprops_basico(bw_rostro_solido)
    if props_rostro["area"] > 45.0:
        cx_s, cy_s = props_rostro["centroide"]
        # Centrar ligeramente arriba del centroide de piel (en el puente nasal, entre ojos y boca)
        cx_frame = int(cx_s * (W / float(w_s)))
        cy_frame = int((cy_s - 3.0) * (H / float(h_s)))

        cx_base = qx_b + qlado // 2
        cy_base = qy_b + qlado // 2

        dx = int(np.clip(cx_frame - cx_base, -max_shift, max_shift))
        dy = int(np.clip(cy_frame - cy_base, -45, 45))

        nuevo_qx = int(np.clip(qx_b + dx, 0, W - qlado))
        nuevo_qy = int(np.clip(qy_b + dy, 0, H - qlado))
        return [nuevo_qx, nuevo_qy, qlado, qlado]

    return cuadrado_base


def verificar_estructura_facial_pdi(parche_bgr: np.ndarray, img_64_double: np.ndarray):
    """
    Validación Antropométrica Universal en Zona-T (Etapa 1):
    - Evalúa exclusivamente el interior de la región facial (ignorando esquinas de ropa o muebles).
    - Invariante a iluminación cálida/fría o lateral y a desplazamientos de cabeza dentro del recuadro.
    - Acepta a CUALQUIER ser humano real (registrado o desconocido) para que pase a la Etapa 2.
    - Bloquea muñecos (plástico amarillo/rojo neón o lana blanca), botellas, perfumes, puños, manos y paredes.
    """
    if len(parche_bgr.shape) != 3:
        return True, {"motivo_rechazo": "", "sim_bilateral": 1.0}

    parche_64 = cv2.resize(parche_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    pdi_rgb = descomponer_rgb_a_double(parche_64)
    R = pdi_rgb["R_double"]
    G = pdi_rgb["G_double"]
    B = pdi_rgb["B_double"]
    gris = pdi_rgb["gris_double"]

    if np.allclose(R, G, atol=1e-3) and np.allclose(G, B, atol=1e-3):
        return True, {"motivo_rechazo": "", "sim_bilateral": 1.0}

    # Evaluar colores artificiales SOLO en el núcleo facial central (jamás en ropa/sillón de esquinas):
    R_nuc = R[12:52, 14:50]
    G_nuc = G[12:52, 14:50]
    B_nuc = B[12:52, 14:50]
    gris_nuc = gris[12:52, 14:50]

    # Compensación de desvío global verdoso del sensor OV2640 (ESP32-CAM) bajo luz de pantalla/LED nocturna:
    med_r = float(np.median(R_nuc))
    med_g = float(np.median(G_nuc))
    bias_ov2640_g = float(np.clip(med_g - med_r, 0.0, 0.16)) if (med_g < 0.72 and med_r > 0.10) else 0.0

    # Ganancia fotométrica local adaptativa: permite evaluar rostros reales en cuartos oscuros (p95~0.20)
    # y en aulas muy iluminadas (p95~0.99) con el mismo rigor geométrico.
    p95_nuc = float(np.percentile(gris_nuc, 95))
    p10_nuc = float(np.percentile(gris_nuc, 10))
    if p95_nuc < 0.075 or (p95_nuc - p10_nuc) < 0.042:
        return False, {"motivo_rechazo": "Objeto oscuro o superficie sin relieve", "sim_bilateral": 0.0}
    gain_loc = float(np.clip(0.75 / max(p95_nuc, 0.08), 0.75, 4.2))
    gris_rel = np.clip(gris * gain_loc, 0.0, 1.0)
    gris_nuc_rel = gris_rel[12:52, 14:50]

    # 1a. Etiquetas verdes/cian/azules intensas de botellas/objetos en el centro facial:
    bw_croma_artificial = (((G_nuc - R_nuc) - bias_ov2640_g) > 0.090) | ((B_nuc - R_nuc) > 0.100)
    # 1b. Cabello plástico amarillo/naranja/rojo chillón de muñecos de anime (ej. Rengoku) en corona:
    R_cor = R[6:36, 12:52]
    G_cor = G[6:36, 12:52]
    B_cor = B[6:36, 12:52]
    bw_muneco_neon = (R_cor > 0.68) & (B_cor < 0.24) & ((R_cor - B_cor) > 0.42) & (G_cor > 0.42)
    # 1c. Corona blanca brillante de muñeco de lana (ej. Gojo) MUCHO más clara que la cara inferior:
    # (No penaliza frentes humanas sobreexpuestas donde la cara inferior también está muy iluminada)
    corona_blanca = float(np.mean((gris[4:26, 12:52] > 0.80) & (np.abs(R[4:26, 12:52] - B[4:26, 12:52]) < 0.06)))
    cara_inf_oscura = float(np.mean(gris[34:54, 16:48])) < 0.52
    # 1d. Carteles o señalética rojo puro sobre fondo blanco (ej. señal de Prohibido Fumar en pared):
    bw_rojo_cartel = (R_nuc > 0.52) & ((R_nuc - G_nuc) > 0.28) & ((R_nuc - B_nuc) > 0.28)
    fondo_blanco_cartel = float(np.mean((gris_nuc > 0.80) & (np.abs(R_nuc - B_nuc) < 0.08)))
    # 1e. Gráficos/portadas impresas amarillo brillante o magenta/morado (ej. cuadernos, cajas de lápices):
    bw_grafico_impreso = (
        ((R_nuc > 0.50) & (G_nuc > 0.45) & (B_nuc < 0.35) & ((R_nuc - B_nuc) > 0.26) & ((G_nuc - B_nuc) > 0.20))
        | ((B_nuc > G_nuc + 0.03) & (R_nuc > G_nuc + 0.03))
    )

    if (
        float(np.mean(bw_croma_artificial)) > 0.15
        or float(np.mean(bw_muneco_neon)) > 0.14
        or (corona_blanca > 0.36 and cara_inf_oscura)
        or float(np.mean(bw_rojo_cartel)) > 0.06
        or (fondo_blanco_cartel > 0.38 and float(np.mean(bw_rojo_cartel)) > 0.02)
        or float(np.mean(bw_grafico_impreso)) > 0.08
    ):
        return False, {"motivo_rechazo": "Color artificial (Muneco/Cartel/Cuaderno)", "sim_bilateral": 0.0}

    # 2. Bloquear objetos negros sólidos o recortes desplazados a cuello/ropa oscura (invariante a luz ambiental):
    if float(np.mean(gris_nuc_rel < 0.16)) > 0.42:
        return False, {"motivo_rechazo": "Objeto oscuro en el centro", "sim_bilateral": 0.0}

    # 3. Bloquear líneas verticales continuas de puertas/paredes/pizarras:
    perfil_sup = np.mean(gris_rel[6:16, 12:52], axis=0)
    perfil_med = np.mean(gris_rel[20:30, 12:52], axis=0)
    if float(np.std(perfil_sup)) > 1e-4 and float(np.std(perfil_med)) > 1e-4:
        c_mat = np.corrcoef(perfil_sup, perfil_med)
        corr_v = float(c_mat[0, 1]) if not np.isnan(c_mat[0, 1]) else 0.0
        if corr_v > 0.92:
            return False, {"motivo_rechazo": "Estructura vertical continua (Puerta/Pared)", "sim_bilateral": 0.0}

    # 4. Simetría estructural adaptativa invariante a iluminación lateral (busca eje entre cols 20 y 44):
    mejor_sim_norm = 0.0
    for cx in range(20, 45, 2):
        w_s = min(cx - 8, 56 - cx)
        if w_s < 10:
            continue
        l_sub = gris_rel[14:48, cx - w_s : cx]
        r_sub = np.fliplr(gris_rel[14:48, cx : cx + w_s])
        l_zero = l_sub - np.mean(l_sub)
        r_zero = r_sub - np.mean(r_sub)
        sim_n = 1.0 - float(np.mean(np.abs(l_zero - r_zero)))
        if sim_n > mejor_sim_norm:
            mejor_sim_norm = sim_n

    std_izq = float(np.std(gris_rel[14:48, 12:32])) * 255.0
    std_der = float(np.std(gris_rel[14:48, 32:52])) * 255.0
    if min(std_izq, std_der) < 5.0 or mejor_sim_norm < 0.72:
        return False, {"motivo_rechazo": "Fondo liso / Objeto lateral asimetrico", "sim_bilateral": mejor_sim_norm}

    # 5. Textura ocular y cejas en tercio superior-medio (evaluado en gris_rel):
    franja_ocular = gris_rel[12:38, 10:54]
    gy_ojos = float(np.mean(np.abs(cv2.Sobel(franja_ocular, cv2.CV_64F, 0, 1, ksize=3))))
    gx_ojos = float(np.mean(np.abs(cv2.Sobel(franja_ocular, cv2.CV_64F, 1, 0, ksize=3))))
    if gy_ojos < 0.012 or gy_ojos < 0.55 * (gx_ojos + 1e-6):
        return False, {"motivo_rechazo": "Sin estructura horizontal de cejas/ojos", "sim_bilateral": mejor_sim_norm}

    # Presencia de rasgos oscuros (pupilas/pestañas/cejas) respecto a la piel:
    mediana_cara = float(np.median(gris_rel[14:50, 14:50]))
    ojo_izq = gris_rel[12:38, 10:32]
    ojo_der = gris_rel[12:38, 32:54]
    p10_izq = float(np.percentile(ojo_izq, 10))
    p10_der = float(np.percentile(ojo_der, 10))
    if max(p10_izq, p10_der) > max(0.60, mediana_cara * 0.96):
        return False, {"motivo_rechazo": "Superficie lisa sin ojos ni cejas (Mano/Puno)", "sim_bilateral": mejor_sim_norm}

    # 6. Presencia de nariz y labios en zona media-inferior (evaluado en gris_rel):
    zona_boca = gris_rel[32:56, 14:50]
    gy_boca = float(np.mean(np.abs(cv2.Sobel(zona_boca, cv2.CV_64F, 0, 1, ksize=3))))
    if gy_boca < 0.011 or (float(np.percentile(zona_boca, 90)) - float(np.percentile(zona_boca, 10))) < 0.055:
        return False, {"motivo_rechazo": "Sin nariz ni labios (Puno/Mano/Pared)", "sim_bilateral": mejor_sim_norm}

    # 6b. Bloquear estructuras de franjas estrictamente horizontales (dedos apilados / persianas):
    # En rostros reales el puente nasal, ojos y comisuras generan bordes verticales relevantes (gy/gx <= 2.15),
    # mientras dedos horizontales apilados presentan anisotropía horizontal extrema (gy/gx >= 2.35).
    nuc_grad = gris_rel[16:50, 14:50]
    gy_nuc = float(np.mean(np.abs(cv2.Sobel(nuc_grad, cv2.CV_64F, 0, 1, ksize=3))))
    gx_nuc = float(np.mean(np.abs(cv2.Sobel(nuc_grad, cv2.CV_64F, 1, 0, ksize=3))))
    if gy_nuc > 2.30 * (gx_nuc + 1e-6):
        return False, {"motivo_rechazo": "Patron de franjas horizontales (Dedos/Mano)", "sim_bilateral": mejor_sim_norm}

    # 7. Crominancia de piel humana en la región central (invariante a baja luz Y sobreexposición OV2640):
    R_n = np.clip(R_nuc * gain_loc, 0, 1)
    G_n = np.clip((G_nuc - bias_ov2640_g) * gain_loc, 0, 1)
    B_n = np.clip(B_nuc * gain_loc, 0, 1)
    gris_n = gris_nuc_rel
    bw_piel = (
        (R_n > 0.14) & (G_n > 0.08) & (B_n > 0.04)
        & ((R_n - G_n) >= -0.030) & ((R_n - G_n) <= 0.46)
        & ((R_n - B_n) >= -0.025)
        & (gris_n > 0.12) & (gris_n <= 1.0)
    )
    # Incluir píxeles de piel sobreexpuestos (casi blancos por auto-exposición fuerte) si conservan tono cálido/neutro
    bw_piel_brillante = (gris_nuc > 0.80) & (R_nuc >= B_nuc - 0.03) & (np.abs(R_nuc - G_nuc) <= 0.09)
    mask_piel_nuc = bw_piel | bw_piel_brillante
    piel_centro = float(np.mean(mask_piel_nuc))
    piel_superior = float(np.mean(mask_piel_nuc[4:24, 2:34]))
    # Continuidad facial en núcleo central (nariz/mejillas) y los 4 cuadrantes:
    # Bloquea manos/dedos sosteniendo objetos (tienen huecos oscuros o un cuadrante vacío),
    # mientras cualquier rostro humano real supera 0.78 en núcleo y 0.65 en los 4 cuadrantes.
    piel_core = float(np.mean(mask_piel_nuc[8:32, 8:28]))
    q_min_piel = min(
        float(np.mean(mask_piel_nuc[:20, :18])),
        float(np.mean(mask_piel_nuc[:20, 18:])),
        float(np.mean(mask_piel_nuc[20:, :18])),
        float(np.mean(mask_piel_nuc[20:, 18:])),
    )
    if piel_centro < 0.48 or piel_superior < 0.44 or piel_core < 0.62 or q_min_piel < 0.45:
        return False, {"motivo_rechazo": "Sin continuidad facial en 4 cuadrantes (Mano/Objeto/Cuaderno)", "sim_bilateral": mejor_sim_norm}

    # 7b. Bloquear estructuras acromáticas/grises puras (lámparas fluorescentes, ventanas, techos, marcos)
    # y focos quemados en blanco puro (>38% del núcleo en saturación blanca absoluta >0.92).
    pct_quemado_blanco = float(np.mean((gris_nuc > 0.92) & (np.abs(R_nuc - B_nuc) < 0.04)))
    if pct_quemado_blanco > 0.38:
        return False, {"motivo_rechazo": "Fuente de luz / Ventana / Lampara blanca", "sim_bilateral": mejor_sim_norm}
    p75_gris_nuc = float(np.percentile(gris_nuc, 75))
    if p75_gris_nuc < 0.90:
        calidez_rb = float(np.percentile(R_nuc - B_nuc, 75))
        calidez_rg = float(np.percentile(R_nuc - G_nuc, 75))
        if calidez_rb < 0.016 and calidez_rg < 0.014:
            return False, {"motivo_rechazo": "Superficie acromatica gris/techo/lampara", "sim_bilateral": mejor_sim_norm}

    # 7c. En recortes muy pequeños (< 86 px), exigir un mínimo de relieve/contraste y ausencia de
    # zonas negras sólidas de puerta/mueble en >30% de la caja para no confundir bisagras o bordes de madera.
    if parche_bgr.shape[0] < 86:
        if float(np.mean(gris_nuc_rel < 0.22)) > 0.28:
            return False, {"motivo_rechazo": "Borde oscuro de mueble/madera pequeno", "sim_bilateral": mejor_sim_norm}

    return True, {"sim_bilateral": mejor_sim_norm, "motivo_rechazo": ""}


def aplicar_nms_cajas(cajas: list, puntajes: list, umbral_iou: float = 0.32, umbral_contencion: float = 0.55) -> list:
    """
    Supresión de No-Máximos (NMS) y Supresión Anatómica de Torso/Cuello sobre ventanas candidatas [x, y, w, h]:
    1. Elimina sub-ventanas más pequeñas ubicadas directamente debajo de una cabeza superior
       (zona de cuello, capucha o solapa de chaqueta de la misma persona).
    2. Reutiliza calcular_iou y verifica también la razón de contención (intersección / área mínima)
       para eliminar cajas duplicadas o anidadas sobre el mismo rostro sin fusionar a dos personas
       distintas ubicadas lado a lado.

    Retorna los índices ordenados de las cajas seleccionadas.
    """
    if not cajas:
        return []

    n_c = len(cajas)
    es_cuello_inferior = [False] * n_c
    for i in range(n_c):
        xi, yi, li, _ = cajas[i]
        cxi = xi + li * 0.5
        cyi = yi + li * 0.5
        for k in range(n_c):
            if i == k:
                continue
            xk, yk, lk, _ = cajas[k]
            cxk = xk + lk * 0.5
            cyk = yk + lk * 0.5
            # Si la caja i es más pequeña y cae en la columna de cuello/pecho/chaqueta debajo de la cabeza k:
            if (
                li <= lk * 0.88
                and abs(cxi - cxk) <= lk * 0.78
                and cyi >= cyk + lk * 0.45
                and cyi <= cyk + lk * 1.65
            ):
                es_cuello_inferior[i] = True
                break

    orden = np.argsort(np.asarray(puntajes, dtype=np.float64))[::-1]
    seleccionados = []

    for idx in orden:
        idx = int(idx)
        if es_cuello_inferior[idx]:
            continue
        ca = cajas[idx]
        area_a = max(1.0, float(ca[2] * ca[3]))
        descartar = False

        for j in seleccionados:
            cb = cajas[j]
            if calcular_iou(ca, cb) >= umbral_iou:
                descartar = True
                break

            # Verificar contención (caja pequeña dentro de caja grande sobre la misma cara)
            xa = max(ca[0], cb[0])
            ya = max(ca[1], cb[1])
            xb = min(ca[0] + ca[2], cb[0] + cb[2])
            yb = min(ca[1] + ca[3], cb[1] + cb[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            area_b = max(1.0, float(cb[2] * cb[3]))
            contencion = float(inter) / min(area_a, area_b)
            if contencion >= umbral_contencion:
                descartar = True
                break

        if not descartar:
            seleccionados.append(idx)

    return seleccionados


def dividir_blob_compuesto_pdi(bbox: list, bw_piel: np.ndarray, profundidad: int = 0) -> list:
    """
    Divide un blob de piel anormalmente ancho (ej. dos o tres personas hombro con hombro)
    analizando los valles verticales de la proyección horizontal de píxeles de piel.
    Retorna una lista de sub-cajas [x, y, w, h] ajustadas a cada componente facial.
    """
    x, y, w, h = [int(v) for v in bbox]
    if w <= 12 or h <= 12 or profundidad >= 2:
        return [[x, y, w, h]]

    # Si la relación ancho/alto es normal para un solo rostro, no dividir
    if w <= int(h * 1.40):
        return [[x, y, w, h]]

    sub_mask = bw_piel[y : y + h, x : x + w]
    perfil_x = np.sum(sub_mask > 0, axis=0).astype(np.float64)
    if len(perfil_x) < 10:
        return [[x, y, w, h]]

    # Suavizado ligero del perfil horizontal
    k_s = np.ones(5, dtype=np.float64) / 5.0
    perfil_suave = np.convolve(perfil_x, k_s, mode="same")

    ini = max(2, int(w * 0.22))
    fin = min(w - 2, int(w * 0.78))
    if fin <= ini:
        return [[x, y, w, h]]

    rel_valle = int(np.argmin(perfil_suave[ini:fin])) + ini
    valle = float(perfil_suave[rel_valle])
    pico_izq = float(np.max(perfil_suave[:rel_valle])) if rel_valle > 0 else 0.0
    pico_der = float(np.max(perfil_suave[rel_valle:])) if rel_valle < w else 0.0
    pico_ref = min(pico_izq, pico_der)

    debe_dividir = (pico_ref > 1e-3 and valle < 0.68 * pico_ref) or (w > int(h * 1.95))
    if not debe_dividir:
        return [[x, y, w, h]]

    sub_cajas = []
    for (c0, c1) in [(0, rel_valle), (rel_valle, w)]:
        if (c1 - c0) < 10:
            continue
        trozo = sub_mask[:, c0:c1]
        ys, xs = np.nonzero(trozo > 0)
        if len(xs) < 40:
            continue
        nx = x + c0 + int(np.min(xs))
        ny = y + int(np.min(ys))
        nw = int(np.max(xs) - np.min(xs)) + 1
        nh = int(np.max(ys) - np.min(ys)) + 1
        if nw >= 10 and nh >= 10:
            sub_cajas.extend(dividir_blob_compuesto_pdi([nx, ny, nw, nh], bw_piel, profundidad + 1))

    return sub_cajas if sub_cajas else [[x, y, w, h]]


def proponer_ventanas_rostros_pdi(
    frame_bgr: np.ndarray,
    max_candidatos: int = 32,
    incluir_rejilla_respaldo: bool = True,
) -> list:
    """
    Generador Multi-Rostro 100% PDI Clásico (Sin clasificadores preentrenados):
    1. Segmentación cromática combinada de piel (YCrCb + RGB double con ganancia adaptativa)
       en resolución reducida (320x240) para máxima velocidad en CPU.
    2. Limpieza morfológica (Apertura + Cierre + imfill_holes) y extracción de TODAS las
       componentes conexas (connectedComponentsWithStats).
    3. Separación de blobs compuestos (personas juntas) mediante valles de proyección horizontal.
    4. Expansión anatómica al cuadrado envolvente de cabeza completa (coronilla/cabello a mentón)
       en 2-3 escalas locales para alinear exactamente con el dataset de entrenamiento 64x64.
    5. Rejilla ligera de respaldo multi-escala en caso de iluminación extrema o fondos cálidos.

    Retorna una lista de ventanas cuadradas candidatas [qx, qy, qlado, qlado].
    """
    H, W = frame_bgr.shape[:2]
    min_dim = min(H, W)
    lado_min = max(54, int(min_dim * 0.12))
    lado_max = int(min_dim * 0.72)

    propuestas = []

    def _agregar_si_nueva(caja_cand, umbral_dup=0.76):
        qx_c, qy_c, ql_c, _ = caja_cand
        if ql_c < lado_min or ql_c > min_dim:
            return
        for ex in propuestas:
            if calcular_iou(caja_cand, ex) >= umbral_dup:
                return
        propuestas.append([int(qx_c), int(qy_c), int(ql_c), int(ql_c)])

    if len(frame_bgr.shape) == 3:
        w_s, h_s = 320, 240
        sx = float(W) / float(w_s)
        sy = float(H) / float(h_s)
        small = cv2.resize(frame_bgr, (w_s, h_s), interpolation=cv2.INTER_AREA)

        # Canal A: Crominancia YCrCb en imagen original (preserva bordes limpios sin fusionar ropa cálida)
        pdi = descomponer_rgb_a_double(small)
        R, G, B, gris = pdi["R_double"], pdi["G_double"], pdi["B_double"], pdi["gris_double"]
        p95 = float(np.percentile(gris, 95))
        gain = float(np.clip(0.72 / max(p95, 0.10), 0.9, 3.2))

        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        Y_c = ycrcb[:, :, 0]
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]
        mask_ycrcb = (Y_c >= 20) & (Y_c <= 255) & (Cr >= 129) & (Cr <= 175) & (Cb >= 76) & (Cb <= 128)

        # Canal B: Crominancia RGB double adaptativa (invariante a baja iluminación y tinte OV2640)
        R_n = np.clip(R * gain, 0.0, 1.0)
        G_n = np.clip(G * gain, 0.0, 1.0)
        B_n = np.clip(B * gain, 0.0, 1.0)
        gris_n = np.clip(gris * gain, 0.0, 1.0)
        no_rojo_cartel = ~((R_n > 0.55) & ((R_n - G_n) > 0.30) & (G_n < 0.36))
        mask_rgb = (
            (R_n > 0.16) & (G_n > 0.10) & (B_n > 0.05)
            & ((R_n - G_n) >= 0.010) & ((R_n - G_n) <= 0.36)
            & ((R_n - B_n) >= 0.045)
            & (gris_n > 0.14) & (gris_n <= 1.0)
            & no_rojo_cartel
        )

        bw_piel = ((mask_ycrcb & no_rojo_cartel) | (mask_rgb & (Cr >= 128) & (Cr <= 176) & (Cb <= 129))).astype(np.uint8) * 255

        # Limpieza morfológica: apertura separa dedos/ruido, cierre une ojos/lentes dentro de la cara
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bw_piel = cv2.morphologyEx(bw_piel, cv2.MORPH_OPEN, k_open, iterations=1)
        bw_piel = cv2.morphologyEx(bw_piel, cv2.MORPH_CLOSE, k_close, iterations=1)
        bw_piel = imfill_holes(bw_piel)

        mascaras_candidatas = [bw_piel]

        # Canal A2 (Separación de pared beige/madera): si bw_piel se fundió con un fondo claro/cálido
        # (>22% del fotograma), agregar una máscara de crominancia facial más selectiva (Cr - Cb >= 10)
        # que separa los rostros humanos reales sin que el fondo los absorba en un solo blob gigante.
        if float(np.mean(bw_piel > 0)) > 0.22:
            diff_cr_cb = Cr.astype(np.int16) - Cb.astype(np.int16)
            m_sep = (
                (Y_c >= 24) & (Y_c <= 248)
                & (Cr >= 131) & (Cr <= 175)
                & (Cb >= 76) & (Cb <= 125)
                & (diff_cr_cb >= 10)
                & no_rojo_cartel
            ).astype(np.uint8) * 255
            m_sep = cv2.morphologyEx(m_sep, cv2.MORPH_OPEN, k_open, iterations=1)
            m_sep = cv2.morphologyEx(m_sep, cv2.MORPH_CLOSE, k_close, iterations=1)
            m_sep = imfill_holes(m_sep)
            mascaras_candidatas.append(m_sep)

        # Canal B2 (Baja iluminación extrema, p95 < 0.35): YCrCb sobre imagen amplificada como canal adicional
        if p95 < 0.35:
            small_low = np.clip(small.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            yc_low = cv2.cvtColor(small_low, cv2.COLOR_BGR2YCrCb)
            m_low = (
                (yc_low[:, :, 0] >= 25) & (yc_low[:, :, 0] <= 245)
                & (yc_low[:, :, 1] >= 134) & (yc_low[:, :, 1] <= 174)
                & (yc_low[:, :, 2] >= 78) & (yc_low[:, :, 2] <= 126)
                & no_rojo_cartel
            ).astype(np.uint8) * 255
            m_low = cv2.morphologyEx(m_low, cv2.MORPH_OPEN, k_open, iterations=1)
            m_low = cv2.morphologyEx(m_low, cv2.MORPH_CLOSE, k_close, iterations=1)
            m_low = imfill_holes(m_low)
            mascaras_candidatas.append(m_low)

        # Canal C: Compensación para sensor OV2640 (ESP32-CAM) bajo luz nocturna/pantalla con tinte verdoso (G >= R)
        if float(np.mean(G)) >= float(np.mean(R)) * 0.90:
            p75_izq = max(55.0, min(130.0, float(np.percentile(Y_c[:, : w_s // 2], 75))))
            p75_der = max(55.0, min(130.0, float(np.percentile(Y_c[:, w_s // 2 :], 75))))
            umbral_y_loc = np.zeros_like(Y_c, dtype=np.float64)
            umbral_y_loc[:, : w_s // 2] = p75_izq
            umbral_y_loc[:, w_s // 2 :] = p75_der
            mask_ov2640 = (
                (Y_c >= umbral_y_loc) & (Y_c < 205)
                & (Cr >= 108) & (Cr <= 138)
                & (Cb >= 96) & (Cb <= 134)
                & (G >= R * 0.93)
                & no_rojo_cartel
            )
            bw_ov = mask_ov2640.astype(np.uint8) * 255
            bw_ov = cv2.morphologyEx(bw_ov, cv2.MORPH_OPEN, k_open, iterations=1)
            bw_ov = cv2.morphologyEx(bw_ov, cv2.MORPH_CLOSE, k_close, iterations=1)
            bw_ov = imfill_holes(bw_ov)
            mascaras_candidatas.append(bw_ov)

        # Canal D: Rostros sobreexpuestos / alta iluminación (cuando el sensor OV2640 satura la piel a blanco Y>=195)
        ycrcb_raw = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        Y_raw = ycrcb_raw[:, :, 0]
        Cr_raw = ycrcb_raw[:, :, 1]
        Cb_raw = ycrcb_raw[:, :, 2]
        if float(np.mean(Y_raw >= 205)) >= 0.03:
            mask_brillante = (
                (Y_raw >= 195)
                & (Cr_raw >= 120) & (Cr_raw <= 156)
                & (Cb_raw >= 100) & (Cb_raw <= 136)
                & no_rojo_cartel
            )
            bw_bri = mask_brillante.astype(np.uint8) * 255
            bw_bri = cv2.morphologyEx(bw_bri, cv2.MORPH_OPEN, k_open, iterations=1)
            bw_bri = cv2.morphologyEx(bw_bri, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
            bw_bri = imfill_holes(bw_bri)
            mascaras_candidatas.append(bw_bri)

        area_min_s = 55  # ~16x16 en 320x240 -> detecta rostros incluso a 2.5 - 3 metros de distancia
        area_max_s = int(w_s * h_s * 0.34)
        sub_blobs = []

        for bw_cand in mascaras_candidatas:
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw_cand, connectivity=8)
            blobs_validos = []
            for lbl in range(1, num_labels):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area < area_min_s or area > area_max_s:
                    continue
                bx = int(stats[lbl, cv2.CC_STAT_LEFT])
                by = int(stats[lbl, cv2.CC_STAT_TOP])
                bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                if bw < 9 or bh < 11:
                    continue
                # Descartar puertas/paredes verticales de piso a techo o franjas horizontales de fondo
                if bh > int(h_s * 0.86) or bw > int(w_s * 0.72):
                    continue
                # Descartar solapas de chaqueta / manos pequeñas pegadas al borde inferior del fotograma
                cy_s = by + bh * 0.5
                if cy_s > (h_s * 0.84) and bh < (h_s * 0.24):
                    continue
                # Priorizar blobs con proporción facial humana (0.50 <= bw/bh <= 1.45) antes que paredes/puertas
                ar = float(bw) / float(max(1, bh))
                es_prop_facial = (0.48 <= ar <= 1.48) and (by > 2 or bh < int(h_s * 0.65))
                prioridad = (1 if es_prop_facial else 0, area)
                blobs_validos.append((prioridad, [bx, by, bw, bh]))

            blobs_validos.sort(key=lambda t: t[0], reverse=True)
            for _, bb in blobs_validos[:6]:
                sub_blobs.extend(dividir_blob_compuesto_pdi(bb, bw_cand))

        # Descartar paredes/fondos pegados al borde superior (by <= 2) antes de priorizar caras reales:
        # un rostro humano real tiene cabello/techo arriba y está flotando en la escena (by > 2).
        sub_blobs.sort(
            key=lambda b: (
                1 if (b[1] > 4 and b[2] >= 22 and b[3] >= 24 and 0.45 <= (float(b[2]) / max(1.0, float(b[3]))) <= 1.65) else 0,
                1 if (b[1] > 2) else 0,
                b[2] * b[3],
            ),
            reverse=True,
        )

        # Generar máximo 2 escalas por blob (en vez de 6) para garantizar que TODOS los blobs de
        # distintas personas (1, 2, 3 o 4 rostros en escena) alcancen a entrar dentro de limite_prop
        # en lugar de que 1 o 2 blobs gigantes de fondo consuman todo el cupo.
        variantes_blob = ((1.08, 0.48), (0.92, 0.49))
        limite_prop = (max_candidatos - 10) if incluir_rejilla_respaldo else max_candidatos

        for (sbx, sby, sbw, sbh) in sub_blobs[:16]:
            fx = sbx * sx
            fy = sby * sy
            fw = sbw * sx
            fh = sbh * sy

            # Acotar altura facial efectiva para abarcar rostro completo alineado con el dataset 64x64
            fh_eff = min(fh, fw * 1.38)
            lado_base = max(fw * 1.24, fh_eff * 1.14)
            lado_base = float(np.clip(lado_base, lado_min, lado_max))

            cx_f = fx + fw * 0.50

            # Anclaje anatómico superior (1 escala): cuando la piel continúa al cuello/camiseta (fh > fw)
            if fh > fw * 1.10:
                lado_c = int(np.clip(round(fw * 1.08), lado_min, lado_max))
                lado_c = (lado_c // 2) * 2
                cy_c = fy + lado_c * 0.47
                qx_c = int(np.clip(round(cx_f - lado_c / 2.0), 0, max(0, W - lado_c)))
                qy_c = int(np.clip(round(cy_c - lado_c / 2.0), 0, max(0, H - lado_c)))
                _agregar_si_nueva([qx_c, qy_c, lado_c, lado_c], umbral_dup=0.72)

            for sc, f_cy in variantes_blob:
                cy_f = fy + fh_eff * f_cy
                qlado = int(np.clip(round(lado_base * sc), lado_min, lado_max))
                qlado = (qlado // 2) * 2
                qx = int(np.clip(round(cx_f - qlado / 2.0), 0, max(0, W - qlado)))
                qy = int(np.clip(round(cy_f - qlado / 2.0), 0, max(0, H - qlado)))
                _agregar_si_nueva([qx, qy, qlado, qlado], umbral_dup=0.72)
                if len(propuestas) >= limite_prop:
                    break
            if len(propuestas) >= limite_prop:
                break

    # Rejilla ligera de respaldo multi-distancia (cerca, media y lejos) en zonas sin propuesta de piel
    if incluir_rejilla_respaldo:
        for factor_l, fracs_x, fracs_y in [
            (0.41, [0.16, 0.50, 0.78], [0.56, 0.42]),
            (0.30, [0.16, 0.42, 0.72], [0.50]),
            (0.20, [0.22, 0.50, 0.78], [0.38]),
        ]:
            qlado = (int(min_dim * factor_l) // 2) * 2
            if qlado < lado_min or qlado > min_dim:
                continue
            for fx_r in fracs_x:
                for fy_r in fracs_y:
                    if len(propuestas) >= max_candidatos:
                        break
                    qx = int(np.clip(round(W * fx_r - qlado / 2.0), 0, max(0, W - qlado)))
                    qy = int(np.clip(round(H * fy_r - qlado / 2.0), 0, max(0, H - qlado)))
                    # Usar umbral_dup=0.62 para que una caja mal encuadrada contra el techo/pared
                    # nunca impida evaluar una ventana facial centrada en ese sector.
                    _agregar_si_nueva([qx, qy, qlado, qlado], umbral_dup=0.62)

    return propuestas[:max_candidatos]


class ScalerPreservandoVarianza:
    """
    Escalador que preserva la relación de autovalores de Eigenfaces:
    Divide el bloque Z por la desviación estándar de la primera componente Z[:, 0],
    preservando lambda_i / lambda_1, y estandariza con z-score únicamente las
    métricas PDI adicionales (MSE, simetría, ratio_hv, contraste).
    """
    def __init__(self, n_pca: int):
        from sklearn.preprocessing import StandardScaler
        self.n_pca = n_pca
        self.scale_pca = 1.0
        self.scaler_extras = StandardScaler()

    def fit(self, F_mat):
        std_1 = float(np.std(F_mat[:, 0]))
        self.scale_pca = std_1 if std_1 > 1e-6 else 1.0
        if F_mat.shape[1] > self.n_pca:
            self.scaler_extras.fit(F_mat[:, self.n_pca:])
        return self

    def transform(self, F_mat):
        Z_scaled = (F_mat[:, :self.n_pca] / self.scale_pca).copy()
        # Atenuación espectral de las 2 primeras Eigenfaces de iluminación ambiental de sesión (Belhumeur et al., 1997)
        if Z_scaled.shape[1] >= 2:
            Z_scaled[:, 0] *= 0.18
            Z_scaled[:, 1] *= 0.35
        if F_mat.shape[1] > self.n_pca:
            # Multiplicar por 0.12 para que las Eigenfaces PCA dominen la métrica de distancia
            extras_scaled = self.scaler_extras.transform(F_mat[:, self.n_pca:]) * 0.12
            return np.hstack([Z_scaled, extras_scaled])
        return Z_scaled

    def fit_transform(self, F_mat):
        return self.fit(F_mat).transform(F_mat)


class BottleneckConvolucional2D:
    """
    Pirámide Jerárquica de Segunda Etapa (MaxPool 2x2 y 4x4) + Recorte de ROI Elíptica
    + Cuello de Botella Ortogonal (Convolutional Bottleneck) que preserva la relación
    espectral lambda_m / lambda_1 de los mapas de activación 2D:
      1. Elimina las celdas muertas en las esquinas exteriores de la máscara elíptica 16x16.
      2. Construye dos niveles adicionales de invarianza espacial (MaxPool 8x8 y MaxPool 4x4).
      3. Comprime las 1,888 activaciones jerárquicas a M=72 componentes convolucionales
         ortogonales, reduciendo los parámetros de la cabecera densa de ~149,500 a ~19,000.
    """
    def __init__(self, n_components: int = 72, factor_energia: float = 0.55):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        self.n_components = n_components
        self.factor_energia = factor_energia
        self.es_bottleneck = True
        self.scaler_pyr = StandardScaler()
        self.pca_conv = PCA(n_components=n_components, whiten=False, random_state=42)
        self.scale_c1 = 1.0

        yy, xx = np.mgrid[0:16, 0:16]
        self.mask_roi_16 = (((xx - 7.5) / 6.2) ** 2 + ((yy - 7.5) / 7.2) ** 2) <= 1.0

    def _construir_piramide_jerarquica(self, F_conv: np.ndarray) -> np.ndarray:
        N = F_conv.shape[0]
        maps_16 = F_conv[:, :2048].reshape(N, 8, 16, 16)
        avg_8 = F_conv[:, 2048:]

        # Nivel 1: Celdas activas dentro de la elipse facial (descarta esquinas muertas)
        roi_16 = maps_16[:, :, self.mask_roi_16].reshape(N, -1)

        # Nivel 2: Segundo Max-Pooling jerárquico 2x2 (16x16 -> 8x8)
        pool2_max = maps_16.reshape(N, 8, 8, 2, 8, 2).max(axis=(3, 5)).reshape(N, -1)

        # Nivel 3: Tercer Max-Pooling jerárquico 4x4 (16x16 -> 4x4, invarianza macro-espacial)
        pool3_max = maps_16.reshape(N, 8, 4, 4, 4, 4).max(axis=(3, 5)).reshape(N, -1)

        return np.hstack([roi_16, pool2_max, pool3_max, avg_8])

    def fit(self, F_conv: np.ndarray):
        pyr = self._construir_piramide_jerarquica(F_conv)
        pyr_s = self.scaler_pyr.fit_transform(pyr)
        C = self.pca_conv.fit_transform(pyr_s)
        std_1 = float(np.std(C[:, 0]))
        self.scale_c1 = std_1 if std_1 > 1e-6 else 1.0
        return self

    def transform(self, F_conv: np.ndarray) -> np.ndarray:
        pyr = self._construir_piramide_jerarquica(F_conv)
        pyr_s = self.scaler_pyr.transform(pyr)
        C = self.pca_conv.transform(pyr_s)
        C_scaled = (C / self.scale_c1) * self.factor_energia
        if C_scaled.shape[1] >= 2:
            # Atenuar modo de iluminación global en la primera componente convolucional
            C_scaled[:, 0] *= 0.35
        return C_scaled

    def fit_transform(self, F_conv: np.ndarray) -> np.ndarray:
        return self.fit(F_conv).transform(F_conv)




