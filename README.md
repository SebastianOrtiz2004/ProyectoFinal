# Sistema de Reconocimiento Facial y Control de Acceso (Agente IA)

**Universidad Técnica de Ambato (UTA)**  
**Facultad de Ingeniería en Sistemas, Electrónica e Industrial (FISEI)**  
**Carrera de Software — Asignatura: Inteligencia Artificial (7mo Semestre)**  
**Docente:** Ing. Rubén Nogales  
**Autores:**  
- **Sebastián Ortiz** (`ID 0: Sebastian_Ortiz`)
- **Ángel Ayuquina** (`ID 1: Angel_Ayuquina`)
- *Soporte escalable para `ID 2: Persona_3` e `ID 3: Persona_4`.*

---

## 1. Descripción del Proyecto y Arquitectura del Agente

Este proyecto implementa un **Agente Basado en Metas y Utilidad** para el control de acceso biométrico inteligente, fundamentado rigurosamente en la teoría matemática y algorítmica de Inteligencia Artificial y Procesamiento Digital de Imágenes (PDI) impartida en clase.

```
                                      +-----------------------------+
                                      |     Ambiente Físico         |
                                      +--------------+--------------+
                                                     |
                                                     v (Captura de Fotogramas)
                                      +-----------------------------+
                                      |   Sensor: Cámara Web / Cel  |
                                      +--------------+--------------+
                                                     |
                                                     v
                                      +-----------------------------+
                                      |  Pipeline PDI Clásico       |
                                      |  - Paso a double [0.0, 1.0] |
                                      |  - Cuadrado max(w,h)        |
                                      |  - Filtro Mediana 3x3       |
                                      |  - Ecualización Histograma  |
                                      |  - Reducción a 64x64        |
                                      +--------------+--------------+
                                                     |
                                                     v
                                      +-----------------------------+
                                      |  Extracción de Características|
                                      |  - PCA Eigenfaces (>=95% var)|
                                      |  - Error MSE Reconstrucción  |
                                      |  - Media y Varianza          |
                                      |  Vector f in R^(K+3)         |
                                      +--------------+--------------+
                                                     |
                      +------------------------------+------------------------------+
                      |                                                             |
                      v                                                             v
        +----------------------------+                               +----------------------------+
        |   ETAPA 1: Binaria         |                               |   ETAPA 2: Multiclase      |
        |   ¿Es Persona o Fondo?     |                               |   ¿Quién es la Persona?    |
        |   (Clase 0 vs Clase 1)     |                               |   (ID 0, ID 1, ..., Otros) |
        +--------------+-------------+                               +--------------+-------------+
                       |                                                            |
                       +-----------------------------+------------------------------+
                                                     |
                                                     v
                                      +-----------------------------+
                                      |  Actuador Gráfico en Vivo   |
                                      |  [PUERTA ABIERTA 🔓]        |
                                      |  [PUERTA CERRADA 🔒]        |
                                      +-----------------------------+
```

### 1.1 Modelo del Agente Inteligente
1. **Ambiente:** Escena frente a la cámara (iluminación variable, fondo del aula/habitación, personas autorizadas, extraños u objetos).
2. **Sensores:** Cámara web en tiempo real (RGB $I(x,y,c)$ con coordenadas discretas) y cámara del celular para dataset offline.
3. **Percepción en Dos Etapas (Doble Pipeline):**
   - **Etapa 1 (Clasificación Binaria):** Discrimina entre `Clase 0: No Persona (Fondo/Pared/Objeto)` y `Clase 1: Rostro Humano`.
   - **Etapa 2 (Reconocimiento de Identidad):** Si la Etapa 1 aprueba la ventana como humana, determina la identidad (`Sebastián Ortiz`, `Ángel Ayuquina` o `Desconocido / Otros`).
4. **Actuador:** Simulación gráfica en tiempo real de apertura y cierre de puerta/cortina con telemetría en vivo.

---

## 2. Reglas de Oro y Fundamentación Matemática

### 2.1 Control Estricto de Sobreajuste (Cero Overfitting)
- **Principio Fundamental:** $E_{\text{in}} \approx E_{\text{test}}$.
- **Brecha en Puntos Porcentuales:**
  $$\text{Brecha (p. p.)} = \text{Exactitud}_{\text{ent}}(\%) - \text{Exactitud}_{\text{pru}}(\%) = E_{\text{test}}(\%) - E_{\text{in}}(\%) \le 5.0\text{ p. p.}$$
- **Aislamiento Total:** El conjunto de prueba (`X_pru`, `y_pru`) se mantiene $100\%$ ciego durante el ajuste. El transformador `PCA` se calibra exclusivamente con `.fit_transform(X_ent)` y se aplica con `.transform(X_pru)` (sin fuga de datos / *data leakage*).
- **Validación Cruzada Estratificada:** 5 pliegues (`StratifiedKFold(n_splits=5)`) en `GridSearchCV` sobre el conjunto de entrenamiento.

### 2.2 PDI Clásico de Cátedra (`src/utils_pdi.py`)
1. **Paso a `double`:**
   $$R_{\text{double}} = \frac{R}{255.0}, \quad G_{\text{double}} = \frac{G}{255.0}, \quad B_{\text{double}} = \frac{B}{255.0}$$
   $$I_{\text{gris\_double}}(x, y) = 0.299 R_{\text{double}} + 0.587 G_{\text{double}} + 0.114 B_{\text{double}}$$
2. **Cuadrado Envolvente Simétrico:**
   $$\text{lado} = \max(w, h), \qquad c_x = x + \frac{w}{2.0}, \qquad c_y = y + \frac{h}{2.0}$$
   $$x_{\text{esq}} = \max\left(0, \min\left(c_x - \frac{\text{lado}}{2.0}, \text{Ancho}_{\text{img}} - \text{lado}\right)\right)$$
   $$y_{\text{esq}} = \max\left(0, \min\left(c_y - \frac{\text{lado}}{2.0}, \text{Alto}_{\text{img}} - \text{lado}\right)\right)$$
3. **Mitigación de Ruido:** Filtro de mediana $3 \times 3$ para ruido impulsivo preservando orillas + suavizado gaussiano $\sigma = 1.0$.
4. **Ecualización de Histograma:** Compensación de contraluz y sombras asimétricas.
5. **Binarización Otsu + `imfill` + `regionprops`:** Sella cavidades morfológicas y extrae momentos geométricos y códigos de cadena de Freeman en 8-vecindad.
6. **Entrada Estándar:** Matriz $64 \times 64$ en precisión `double` $[0.0, 1.0]$ ($\mathbf{x} \in \mathbb{R}^{4096}$).

### 2.3 Reducción Dimensional (PCA / Eigenfaces) y MSE
1. Centrado respecto a la media: $\tilde{X} = X_{\text{ent}} - \boldsymbol{\mu}$.
2. Proyección sobre las $K$ componentes que retienen el $\ge 95\%$ de la varianza explicada.
3. Reconstrucción y Error Cuadrático Medio:
   $$\text{MSE}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{4096}\sum_{j=1}^{4096}(x_j - \hat{x}_j)^2$$
4. Vector concatenado: $\mathbf{f} = [\mathbf{z}_{\text{PCA}}, \text{MSE}, \mu_{\text{intensidad}}, \sigma_{\text{contraste}}] \in \mathbb{R}^{K+3}$.

---

## 3. Estructura de Directorios

```
PROYECTO_IA_RECONOCIMIENTO/
│
├── data/
│   ├── fotos_crudas_celular/
│   │   ├── id_0_sebastian/
│   │   ├── id_1_angel/
│   │   ├── id_2_persona3/
│   │   └── id_3_persona4/
│   │
│   └── dataset_procesado_64x64/
│       ├── clase_0_no_persona/
│       ├── id_0_sebastian/
│       ├── id_1_angel/
│       ├── id_2_persona3/
│       └── id_3_persona4/
│
├── modelos_entrenados/
│   ├── etapa1_pca_humano.pkl
│   ├── etapa1_todos_modelos.pkl
│   ├── etapa1_mejor_modelo.pkl
│   ├── etapa2_pca_identidad.pkl
│   ├── etapa2_todos_modelos.pkl
│   ├── etapa2_mejor_modelo.pkl
│   └── metadata_clases.pkl
│
├── figuras_y_reportes/
│   ├── figura1_pipeline_pdi_completo.png
│   ├── figura2_eigenfaces_varianza_pca.png
│   ├── figura3_matrices_confusion_etapa1.png
│   ├── figura4_matrices_confusion_etapa2.png
│   ├── figura5_comparativa_brechas.png
│   ├── tabla_resultados_etapa1_humano.csv
│   └── tabla_resultados_etapa2_identidad.csv
│
├── src/
│   ├── utils_pdi.py
│   ├── 1_adquisicion_datos.py
│   ├── 2_entrenamiento_evaluacion.py
│   ├── 3_reconocimiento_en_vivo.py
│   └── 4_fase_posterior_cnn.py
│
├── requirements.txt
├── README.md
└── PLANIFICACION_PROYECTO_IA.md
```

---

## 4. Guía de Ejecución Paso a Paso

### Paso 1: Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Adquisición y Procesamiento de Imágenes
Ejecuta el script de adquisición:
```bash
python src/1_adquisicion_datos.py
```
- **Opción 1:** Captura interactiva con cámara web (aplica distribución automática por ventanas y guarda rostros y fondos de esquinas simultáneamente).
- **Opción 2:** Procesa fotos crudas del celular copiadas en `data/fotos_crudas_celular/`.
- **Opción 3:** Genera la Figura 1 diagnóstica de PDI (`figura1_pipeline_pdi_completo.png`).

### Paso 3: Entrenamiento Masivo y Evaluación de Modelos
```bash
python src/2_entrenamiento_evaluacion.py
```
- Evalúa los 5 algoritmos del sílabo: **Árbol de Decisión, Random Forest, k-NN, SVM y ANN (MLP)**.
- Imprime las tablas de cátedra con comprobación obligatoria ($\text{Exactitud} + \text{Error} = 100\%$).
- Genera las figuras comparativas, matrices de confusión y guarda los modelos entrenados en `modelos_entrenados/`.

### Paso 4: Demostración en Tiempo Real y Control de Acceso
```bash
python src/3_reconocimiento_en_vivo.py
```
**Controles del teclado durante la demostración:**
- `0`: Modo Automático Óptimo (usa el modelo ganador seleccionado matemáticamente).
- `1`: Fuerza **k-NN** en ambas etapas.
- `2`: Fuerza **Árbol de Decisión** en ambas etapas.
- `3`: Fuerza **Random Forest** en ambas etapas.
- `4`: Fuerza **SVM** en ambas etapas.
- `5`: Fuerza **Red Neuronal Artificial (ANN)** en ambas etapas.
- `q` o `ESC`: Salir limpiamente.

### Paso 5: Fase Posterior (Redes Neuronales Convolucionales - CNN)
```bash
python src/4_fase_posterior_cnn.py
```
Compara el enfoque de extracción automática de mapas de características con convolución espacial directa vs. el pipeline clásico PDI + PCA.
