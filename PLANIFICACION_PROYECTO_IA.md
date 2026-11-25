# Especificación Técnica y Arquitectura del Sistema: Control de Acceso Biométrico (Agente IA)

> Este documento constituye la especificación de ingeniería, matemática y algorítmica completa del sistema: modular en Python implementando adquisición híbrida con **Distribución Automática por Ventanas**, preprocesamiento PDI clásico en formato `double` $[0.0, 1.0]$ (mitigación de ruido, ecualización, binarización auxiliar, `imfill`, `regionprops` y cuadrado envolvente simétrico $\text{lado} = \max(w, h)$), extracción de características mediante **PCA (Eigenfaces)** y **MSE**, clasificación jerárquica en dos etapas evaluando algoritmos de Machine Learning (**k-NN, Árboles de Decisión, Random Forest, SVM, ANN** y un módulo posterior para **CNN**), control estricto de generalización sin *overfitting* ($E_{\text{in}} \approx E_{\text{test}}$), y demostración en tiempo real mediante cámara web simulando el actuador físico (apertura de puerta/cortina).

## 1. Arquitectura del Agente Inteligente

### 1.1 Características del Sistema

* **Soporte multiclase escalable:** Clasificación discriminativa de sujetos autorizados y rechazo de fondo o sujetos desconocidos.
* **Restricción de librerías:** Se permite el uso de `numpy`, `pandas`, `matplotlib`, `opencv-python` (`cv2`), `scikit-learn` (`sklearn`) y `joblib`. **No** se utilizan librerías de mallas o puntos faciales preentrenados (*MediaPipe / Dlib landmarks*), ya que toda la extracción de características proviene del pipeline propio de PDI y reducción dimensional matemática.

### 1.2 Conceptualización como Agente Inteligente

Siguiendo la teoría vista en clase, el prototipo se modela como un **Agente Basado en Metas / Utilidad**:

1. **Ambiente:** El entorno físico frente a la computadora (iluminación, fondo del aula o habitación, presencia de objetos o personas).
2. **Sensores:** La cámara web (y la cámara del teléfono móvil durante la ingesta offline) que capturan la luz reflejada como una matriz tridimensional RGB $I(x, y, c)$ con origen $(0,0)$ en la esquina superior izquierda, sin alterar el ambiente.
3. **Percepción y Clasificación en Dos Etapas (Doble Pipeline de Machine Learning):**
   * **Etapa 1 — Clasificación Binaria (¿Es Persona o No es Persona?):** Evalúa matemáticamente las ventanas extraídas de la escena para discriminar entre `Clase 0: No Persona (Fondo / Pared / Objeto)` y `Clase 1: Rostro Humano`.
   * **Etapa 2 — Reconocimiento Multiclase de Identidad (Percepción Facial):** Si y solo si la Etapa 1 confirma que la ventana contiene una persona (`Clase 1`), el vector de características pasa al clasificador multiclase para identificar a qué sujeto pertenece (`ID 0: Sebastián Ortiz`, `ID 1: Ángel Ayuquina`, `ID 2: Persona 3`, `ID 3: Persona 4`, o `ID 4: Desconocido / Otros`).
4. **Actuadores:** El mecanismo que modifica el ambiente tras la decisión perceptual. En este prototipo se representa mediante un **Actuador Gráfico en Tiempo Real** en la ventana de OpenCV (simulando la apertura y cierre de una **Puerta / Cortina / Ventana Automatizada** con indicadores visuales de estado).

## 2. Reglas de Oro para Obtener la Nota Máxima (Requisitos No Negociables)

1. **Generalización Excelente y Cero Overfitting:**
   * Es un requisito **no negociable** en todo el proyecto que el **Error de Entrada / Entrenamiento ($E_{\text{in}}$)** sea **muy semejante al Error de Testeo ($E_{\text{test}}$)**.
   * En cada tabla de resultados se debe calcular la **Brecha en puntos porcentuales (p. p.)**:
     $$\text{Brecha (p. p.)} = \text{Exactitud}_{\text{ent}}(\%) - \text{Exactitud}_{\text{pru}}(\%) = E_{\text{test}}(\%) - E_{\text{in}}(\%)$$
   * La brecha debe mantenerse acotada (idealmente $\le 5.0\text{ p. p.}$ y nunca superior a $8.0\text{ p. p.}$). Si un modelo obtiene $100\%$ en entrenamiento y $75\%$ en prueba, existe *overfitting* y debe ser regularizado (ajustando `max_depth`, `min_samples_leaf`, parámetro $C$ en SVM o `alpha` en ANN).

2. **Separación Rigurosa de Training, Validation y Testing:**
   * **Training ($X_{\text{ent}}, y_{\text{ent}}$) y Validation:** Se utilizan **exclusivamente** para ajustar parámetros del modelo y calibrar hiperparámetros mediante Validación Cruzada Estratificada de $5$ pliegues (`StratifiedKFold(n_splits=5)`).
   * **Testing ($X_{\text{pru}}, y_{\text{pru}}$):** Se mantiene estrictamente blindado e invisible durante todo el ajuste. Solo se evalúa una vez al final mediante `.predict(X_pru)` para medir la verdadera capacidad de **generalización**.
   * **Prohibición de Fuga de Datos (*Data Leakage*):** El modelo `PCA` y cualquier escalador deben ejecutarse con `.fit_transform(X_ent)` sobre el conjunto de entrenamiento y aplicarse al conjunto de prueba únicamente mediante `.transform(X_pru)`.

3. **Evaluación Exhaustiva de Todos los Algoritmos antes de Elegir un Ganador:**
   * Tanto en la **Etapa 1 (Persona vs. No Persona)** como en la **Etapa 2 (Identidad del Rostro)**, el vector de características extraído debe enviarse a **todos** los algoritmos del sílabo:
     1. **k-NN** (`KNeighborsClassifier`)
     2. **Árbol de Decisión / DT** (`DecisionTreeClassifier`)
     3. **Random Forest / RF** (`RandomForestClassifier`)
     4. **SVM** (`SVC`)
     5. **ANN / Red Neuronal Artificial** (`MLPClassifier`)
     6. **CNN / Red Neuronal Convolucional** (implementada en el script de fase posterior para comparativa).
   * Se debe imprimir por consola y exportar a CSV/PNG la tabla comparativa completa (verificando formalmente con `assert` que $\text{Exactitud (\%)} + \text{Error (\%)} = 100.00\%$) y fundamentar matemáticamente por qué se selecciona el modelo ganador.

## 3. Fundamentación Matemática y Pipeline de Cada Módulo

### 3.1 Módulo 1: Adquisición de Datos y Técnica de "Distribución por Ventanas" (Opción 1)

El sistema se alimenta de dos fuentes combinadas:
1. **Dataset Público Estándar:** **Olivetti Faces** (`sklearn.datasets.fetch_olivetti_faces`), compuesto por $400$ imágenes de rostros humanos ($40$ sujetos, $10$ fotos por sujeto) ya normalizadas en $[0.0, 1.0]$ con resolución de $64 \times 64$ píxeles.
2. **Dataset Propio Capturado por los Estudiantes (Celular + Webcam):**
   * Entre $50$ y $80$ fotografías por integrante (`ID 0: Sebastián Ortiz`, `ID 1: Ángel Ayuquina`, y opcionalmente `ID 2`, `ID 3`), combinando fotos tomadas a distancia media/lejana con el teléfono celular y fotos cercanas con diferentes gestos (serio, relajado, sonriendo, leve giro facial) con la cámara web de la laptop.

#### ¿Cómo funciona la Distribución Automática por Ventanas (Sin fotografiar objetos manualmente)?
Para entrenar la **Etapa 1 (Persona vs. No Persona)** sin que el usuario tenga que tomar fotos manuales de sillas o paredes, el script de adquisición aplica **Distribución por Ventanas** sobre cada fotografía o fotograma capturado de dimensión $(H \times W)$:
1. **Ventana Positiva (`Clase 1: Es Persona`):**
   * Se localiza la caja envolvente del rostro $[x, y, w, h]$ y se calcula el **cuadrado envolvente simétrico** $\text{lado} = \max(w, h)$ centrado en el rostro.
   * Este recorte cuadrado preprocesado a $64 \times 64$ se guarda simultáneamente como muestra positiva de `Clase 1 (Persona)` para la Etapa 1, y bajo la carpeta de identidad `id_X_nombre/` para la Etapa 2.
2. **Ventanas Negativas Automáticas (`Clase 0: No es Persona`):**
   * Del **mismo fotograma o fotografía original**, el algoritmo barre las esquinas y bordes periféricos (subventanas proporcionales en las esquinas superior izquierda, superior derecha, inferior izquierda e inferior derecha).
   * Verifica matemáticamente que la intersección sobre la unión ($\text{IoU}$) con el cuadrado del rostro sea exactamente cero ($\text{IoU} = 0$).
   * Guarda automáticamente de $2$ a $3$ ventanas de fondo por cada foto en `data/dataset_procesado_64x64/clase_0_no_persona/`.

### 3.2 Módulo 2: Preprocesamiento Digital de Imágenes (PDI Clásico de Cátedra)

1. **Reserva RGB, Descomposición Espectral y Paso a `double` (`float64`):**
   $$R_{\text{double}} = \frac{\text{double}(R)}{255.0}, \quad G_{\text{double}} = \frac{\text{double}(G)}{255.0}, \quad B_{\text{double}} = \frac{\text{double}(B)}{255.0}$$
   $$I_{\text{gris\_double}}(x, y) = 0.299 R_{\text{double}}(x, y) + 0.587 G_{\text{double}}(x, y) + 0.114 B_{\text{double}}(x, y)$$
2. **Algoritmo del Cuadrado Envolvente Simétrico ($\text{lado} = \max(\text{ancho}, \text{alto})$):**
   $$\text{lado} = \max(w, h), \qquad c_x = x + \frac{w}{2.0}, \qquad c_y = y + \frac{h}{2.0}$$
   $$x_{\text{esq}} = \max\left(0, \min\left(c_x - \frac{\text{lado}}{2.0}, \text{Ancho}_{\text{img}} - \text{lado}\right)\right)$$
   $$y_{\text{esq}} = \max\left(0, \min\left(c_y - \frac{\text{lado}}{2.0}, \text{Alto}_{\text{img}} - \text{lado}\right)\right)$$
3. **Mitigación de Ruido (Filtro de Mediana y Suavizado Gaussiano):**
   * Filtro Espacial de Mediana $3 \times 3$ (`cv2.medianBlur`) para ruido impulsivo preservando orillas.
   * Suavizado Gaussiano (`cv2.GaussianBlur`, $\sigma = 1.0$).
4. **Ecualización del Histograma:**
   $$s_k = T(r_k) = \frac{L - 1}{M \cdot N}\sum_{j=0}^{k} n_j$$
5. **Binarización de Otsu, Relleno Morfológico (`imfill`) y Propiedades Geométricas (`regionprops`):**
   * Umbral óptimo de Otsu (`cv2.THRESH_OTSU`).
   * Relleno morfológico de cavidades internas (`imfill_holes` con `cv2.floodFill`).
   * Descriptores: Área ($m_{00}$), Perímetro, Centroide continuo $(\bar{x}, \bar{y})$, BoundingBox y Código de Cadena de Freeman en 8-vecindad.
6. **Reducción Espacial Estandarizada a $64 \times 64$ en `double` $[0.0, 1.0]$.**

### 3.3 Módulo 3: Extracción de Características (PCA / Eigenfaces y MSE)

1. **PCA:**
   * $\boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i$, $\tilde{X} = X_{\text{ent}} - \boldsymbol{\mu}$.
   * Autovectores con $\ge 95\%$ de varianza explicada acumulada (`n_components=0.95, whiten=True`).
   * Proyección: $\mathbf{z}_{\text{PCA}} \in \mathbb{R}^K$.
2. **MSE de Reconstrucción y Descriptores Estadísticos:**
   * Reconstrucción: $\hat{\mathbf{x}} = \mathbf{z}_{\text{PCA}} V_K^T + \boldsymbol{\mu}$.
   * $\text{MSE}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{4096}\sum_{j=1}^{4096}(x_j - \hat{x}_j)^2$.
   * Vector final: $\mathbf{f} = [\mathbf{z}_{\text{PCA}}, \text{MSE}, \mu_{\text{intensidad}}, \sigma_{\text{contraste}}] \in \mathbb{R}^{K+3}$.

### 3.4 Módulo 4: Clasificación y Evaluación de Algoritmos

Modelos: k-NN, Árboles de Decisión (pre-podados), Random Forest, SVM (RBF/Lineal con regularización), ANN (MLPClassifier con regularización $L_2$), y CNN.
Partición 80/20 estratificada, GridSearchCV con StratifiedKFold(n_splits=5), cálculo de brechas, comprobaciones formales de exactitud + error = 100%, y función de selección:
$$J(\text{modelo}) = \text{Exactitud}_{\text{pru}}(\%) - 0.5 \cdot \max(0, \text{Brecha (p. p.)} - 3.0)$$
