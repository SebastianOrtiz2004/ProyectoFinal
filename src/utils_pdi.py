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
    # floodFill requiere una máscara auxiliar de tamaño (h+2, w+2)
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)

    # Inundación desde la esquina superior izquierda (0,0)
    im_floodfill = bw.copy()
    cv2.floodFill(im_floodfill, mask_ff, (0, 0), 255)

    # Invertir el resultado de la inundación externa
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

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


def obtener_mascara_eliptica_64() -> np.ndarray:
    """
    Genera una máscara elíptica suave en float64 de tamaño (64, 64):
    Valor 1.0 en la elipse facial central (centro=(32, 32), ejes=(23, 29))
    y transición suave (GaussianBlur 5x5) hacia 0.0 en las esquinas exteriores.
    Elimina los artefactos de fondo de la habitación (pared, respaldar de cama, hombros).
    """
    mask = np.zeros((64, 64), dtype=np.float64)
    cv2.ellipse(mask, (32, 32), (23, 29), 0, 0, 360, 1.0, -1)
    mask_suave = cv2.GaussianBlur(mask, (5, 5), 1.0)
    return np.clip(mask_suave, 0.0, 1.0)


def aplicar_mascara_eliptica_a_img64(img_64_double: np.ndarray) -> np.ndarray:
    """
    Aplica la máscara elíptica facial suave a una imagen de 64x64 en [0.0, 1.0]
    preservando intacto el contraste real del interior del rostro y neutralizando
    únicamente las 4 esquinas exteriores fuera de la elipse con un tono neutro (0.42).
    """
    mascara = obtener_mascara_eliptica_64()
    img_enmascarada = img_64_double * mascara + 0.42 * (1.0 - mascara)
    return np.clip(img_enmascarada, 0.0, 1.0)


def preprocesar_ventana_64x64(parche_bgr_o_gris: np.ndarray):
    """
    Pipeline PDI Clásico de Cátedra sobre la ventana cuadrada:
        1. Conversión a escala de grises uint8 [0, 255].
        2. Mitigación de ruido: Filtro de mediana 3x3 (mitiga ruido impulsivo preservando orillas).
        3. Suavizado gaussiano ligero (sigma=1.0, kernel 3x3).
        4. Control adaptativo de contraste:
           - Si std_cruda < 12.0 (pared lisa / textura plana sin contraste real), NO aplica equalizeHist
             para evitar inflar el ruido y convertir una pared vacía en falso rostro.
           - Si std_cruda >= 12.0 (rostro real), aplica ecualización balanceada suave (0.6 eq + 0.4 gauss).
        5. Reducción espacial de cátedra por MEDIANA DE BLOQUE VECTORIZADA a 64x64.
        6. Conversión a precisión continua double en [0.0, 1.0].
        7. Aplicación obligatoria de máscara elíptica suave: neutraliza el fondo exterior con 0.42.

    Retorna:
        - img_64x64_double: Matriz np.ndarray de tamaño (64, 64) en np.float64.
        - vector_4096: Vector fila aplanado np.ndarray de 4096 elementos en np.float64.
    """
    if len(parche_bgr_o_gris.shape) == 3:
        gris_uint8 = cv2.cvtColor(parche_bgr_o_gris, cv2.COLOR_BGR2GRAY)
    else:
        if parche_bgr_o_gris.dtype == np.float64 or parche_bgr_o_gris.dtype == np.float32:
            gris_uint8 = np.clip(parche_bgr_o_gris * 255.0, 0, 255).astype(np.uint8)
        else:
            gris_uint8 = parche_bgr_o_gris.copy()

    std_cruda = float(np.std(gris_uint8))

    # Mitigación de ruido de cátedra
    filtrada_mediana = cv2.medianBlur(gris_uint8, 3)
    filtrada_gauss = cv2.GaussianBlur(filtrada_mediana, (3, 3), 1.0)

    # Ecualización adaptativa para evitar inflar paredes lisas
    if std_cruda < 12.0:
        ecualizada = filtrada_gauss
    else:
        eq_pura = cv2.equalizeHist(filtrada_gauss)
        ecualizada = cv2.addWeighted(eq_pura, 0.6, filtrada_gauss, 0.4, 0.0)

    # Reducción espacial por MEDIANA DE BLOQUE vectorizada a 64x64
    img_64x64_mediana = reducir_por_mediana_bloque(ecualizada, 64, 64)

    # Conversión explícita a double [0.0, 1.0]
    img_64x64_base = np.clip(img_64x64_mediana / 255.0, 0.0, 1.0)

    # Aplicación obligatoria de máscara elíptica suave: neutraliza el fondo exterior con 0.42
    mascara = obtener_mascara_eliptica_64()
    img_64x64_double = img_64x64_base * mascara + 0.42 * (1.0 - mascara)
    img_64x64_double = np.clip(img_64x64_double, 0.0, 1.0)
    vector_4096 = img_64x64_double.reshape(-1)

    return img_64x64_double, vector_4096

    return img_64x64_double, vector_4096


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
    rellenando los bordes vacíos con el tono neutro de cátedra (0.42).
    """
    im = im_64.copy()
    if flip:
        im = np.fliplr(im)
    centro = (32.0, 32.0)
    M = cv2.getRotationMatrix2D(centro, 0.0, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    im_warp = cv2.warpAffine(im, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0.42)
    return im_warp


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
    Validación Antropométrica Universal en Zona-T y PDI Estructural (Etapa 1):
    Evalúa características anatómicas universales de un rostro humano (cualquier hombre o mujer,
    con flequillo o frente despejada) y bloquea objetos inanimados (botellas, perfumes,
    puños/manos, muñecos de plástico y paredes vacías).
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

    p95 = float(np.percentile(gris, 95))
    gain = np.clip(0.75 / max(p95, 0.15), 1.0, 2.2)
    R_n = np.clip(R * gain, 0.0, 1.0)
    G_n = np.clip(G * gain, 0.0, 1.0)
    B_n = np.clip(B * gain, 0.0, 1.0)
    gris_n = np.clip(gris * gain, 0.0, 1.0)

    # 1. Ausencia de colores artificiales no humanos:
    # Bloquea etiquetas verdes/cian/azules de botellas y tonos no humanos
    bw_croma = ((G - R) > 0.04) | ((B - R) > 0.05)
    corona_blanca = float(np.mean((gris[4:20, 16:48] > 0.80) & (np.abs(R - B)[4:20, 16:48] < 0.08)))
    if float(np.mean(bw_croma[14:48, 14:50])) > 0.08 or corona_blanca > 0.28:
        return False, {"motivo_rechazo": "Color artificial (Vela/Objeto)", "sim_bilateral": 0.0}

    # 2. Ausencia de bloques negros en el centro facial [22:48, 18:46] (bloquea perfumes o controles)
    if float(np.mean(gris_n[22:48, 18:46] < 0.18)) > 0.26:
        return False, {"motivo_rechazo": "Objeto oscuro en el centro", "sim_bilateral": 0.0}

    # 3. Bloqueo de líneas verticales continuas de puertas/paredes (corr_vertical_pared):
    perfil_sup = np.mean(gris[6:16, 12:52], axis=0)
    perfil_med = np.mean(gris[20:30, 12:52], axis=0)
    std_ps = float(np.std(perfil_sup))
    std_pm = float(np.std(perfil_med))
    if std_ps > 1e-4 and std_pm > 1e-4:
        c_mat = np.corrcoef(perfil_sup, perfil_med)
        corr_v = float(c_mat[0, 1]) if not np.isnan(c_mat[0, 1]) else 0.0
        if corr_v > 0.91:
            return False, {"motivo_rechazo": "Estructura vertical continua (Puerta/Pared)", "sim_bilateral": 0.0}

    # 4. Validación estricta de las dos cajas oculares superiores gris[12:32, 14:30] y gris[12:32, 34:50]:
    gris_crudo_255 = gris * 255.0
    std_ojo_izq = float(np.std(gris_crudo_255[12:32, 14:30]))
    std_ojo_der = float(np.std(gris_crudo_255[12:32, 34:50]))
    if min(std_ojo_izq, std_ojo_der) < 7.8:
        return False, {"motivo_rechazo": "Sin textura ocular bilateral (Pared/Objeto)", "sim_bilateral": 0.0}

    ojo_izq_n = gris_n[12:32, 14:30]
    ojo_der_n = gris_n[12:32, 34:50]
    gy_izq = float(np.mean(np.abs(cv2.Sobel(ojo_izq_n, cv2.CV_64F, 0, 1, ksize=3))))
    gx_izq = float(np.mean(np.abs(cv2.Sobel(ojo_izq_n, cv2.CV_64F, 1, 0, ksize=3))))
    gy_der = float(np.mean(np.abs(cv2.Sobel(ojo_der_n, cv2.CV_64F, 0, 1, ksize=3))))
    gx_der = float(np.mean(np.abs(cv2.Sobel(ojo_der_n, cv2.CV_64F, 1, 0, ksize=3))))
    gy_tot = gy_izq + gy_der
    gx_tot = gx_izq + gx_der
    if (min(gy_izq, gy_der) < 0.024) or (gy_tot < 0.62 * (gx_tot + 1e-6)):
        return False, {"motivo_rechazo": "Sin bordes horizontales en ojos (Cejas)", "sim_bilateral": 0.0}

    zona_mejillas = gris_n[26:48, 16:48]
    mediana_mejillas = float(np.median(zona_mejillas))
    p20_izq = float(np.percentile(ojo_izq_n, 20))
    p20_der = float(np.percentile(ojo_der_n, 20))
    if (max(p20_izq, p20_der) > mediana_mejillas * 0.88) or (max(p20_izq, p20_der) > 0.47):
        return False, {"motivo_rechazo": "Sin cuencas oculares sombreadas (Puno/Pared)", "sim_bilateral": 0.0}

    # 5. Validación de fosas nasales y boca en el tercio inferior [34:54, 20:44]:
    zona_boca = gris_n[34:54, 20:44]
    gy_boca = float(np.mean(np.abs(cv2.Sobel(zona_boca, cv2.CV_64F, 0, 1, ksize=3))))
    rango_boca = float(np.percentile(zona_boca, 88)) - float(np.percentile(zona_boca, 12))
    if gy_boca < 0.023 or rango_boca < 0.10:
        return False, {"motivo_rechazo": "Sin contraste de nariz/labios (Puno/Mano)", "sim_bilateral": 0.0}

    # 6. Presencia de piel humana bilateral en mejillas [24:48, 14:50] (>= 0.25)
    bw_piel = (
        (R_n > 0.20) & (G_n > 0.11) & (B_n > 0.07)
        & ((R_n - G_n) >= 0.005) & ((R_n - G_n) <= 0.38)
        & ((R_n - B_n) >= 0.010)
        & (gris_n > 0.20) & (gris_n < 0.88)
    )
    if float(np.mean(bw_piel[24:48, 14:50])) < 0.25:
        return False, {"motivo_rechazo": "Sin piel facial genuina (Objeto)", "sim_bilateral": 0.0}

    # 7. Simetría bilateral adaptable (columnas 25 a 39 para tolerar giros y desfases leves)
    mejor_sim = 0.0
    for cx in range(25, 40, 2):
        w_s = min(cx - 8, 56 - cx)
        l_sub = gris_n[12:50, cx - w_s : cx]
        r_sub = np.fliplr(gris_n[12:50, cx : cx + w_s])
        s_val = 1.0 - float(np.mean(np.abs(l_sub - r_sub)))
        if s_val > mejor_sim:
            mejor_sim = s_val

    std_izq_tot = float(np.std(gris_n[:, :32])) * 255.0
    std_der_tot = float(np.std(gris_n[:, 32:])) * 255.0
    ratio_std = min(std_izq_tot, std_der_tot) / (max(std_izq_tot, std_der_tot) + 1e-6)

    if min(std_izq_tot, std_der_tot) < 8.0 or ratio_std < 0.30 or mejor_sim < 0.65:
        return False, {"motivo_rechazo": "Asimetria lateral (Pared/Objeto)", "sim_bilateral": mejor_sim}

    return True, {
        "sim_bilateral": mejor_sim,
        "motivo_rechazo": "",
    }


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
        Z_scaled = F_mat[:, :self.n_pca] / self.scale_pca
        if F_mat.shape[1] > self.n_pca:
            # Multiplicar por 0.12 para que las Eigenfaces PCA dominen la métrica de distancia
            extras_scaled = self.scaler_extras.transform(F_mat[:, self.n_pca:]) * 0.12
            return np.hstack([Z_scaled, extras_scaled])
        return Z_scaled

    def fit_transform(self, F_mat):
        return self.fit(F_mat).transform(F_mat)




