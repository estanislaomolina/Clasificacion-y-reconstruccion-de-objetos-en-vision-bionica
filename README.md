 TP Final - Clasificación y reconstrucción de objetos en visión biónica

Autores: Joel Jablonski y Estanislao Molina Abeniacar  
Materia: Visión Artificial  
Universidad de San Andrés

## Resumen del proyecto

Este trabajo estudia cuánta información semántica y espacial se conserva en perceptos simulados de prótesis retinales.

La motivación original del proyecto fue evaluar qué filtros o transformaciones visuales podrían aplicarse a una imagen para que una persona usuaria de una prótesis visual pueda interpretarla mejor. Sin embargo, evaluar directamente filtros con usuarios reales o sujetos en experimentos psicofísicos requiere una validación perceptual más compleja. Por eso, el trabajo se reformuló como una etapa previa y medible: analizar si los perceptos simulados conservan información suficiente para recuperar propiedades del objeto original.

Para eso se definieron dos tareas:

1. **Clasificación de objetos**: predecir la clase del objeto a partir del percepto simulado.  
   Esta tarea mide preservación semántica.

2. **Reconstrucción de máscaras**: reconstruir la máscara binaria del objeto a partir del percepto.  
   Esta tarea mide preservación espacial y geométrica.

El pipeline utiliza máscaras de objetos extraídas de COCO, genera perceptos simulados mediante `pulse2percept` para distintos modelos de implante retinal, y evalúa modelos clásicos y neuronales sobre esos perceptos.

## Resultados principales

### Clasificación

El mejor modelo de clasificación fue un **KNN multi-implante**, que combina perceptos generados por Argus II, PRIMA y Alpha-AMS.

| Modelo | Accuracy test |
|---|---:|
| Baseline mayoría | 0.271 |
| KNN Argus II | 0.561 |
| KNN PRIMA | 0.571 |
| KNN Alpha-AMS | 0.567 |
| KNN multi-implante | **0.611** |
| CNN Argus II | 0.509 |
| CNN PRIMA | 0.563 |
| CNN Alpha-AMS | 0.561 |
| CNN+GRU | 0.575 |

El resultado más relevante es que el KNN multi-implante superó a la arquitectura neuronal CNN+GRU. Esto sugiere que, para el tamaño y características del conjunto de datos utilizado, una representación clásica basada en PCA y similitud directa fue más estable que una red neuronal entrenada desde cero.

### Reconstrucción

El mejor modelo de reconstrucción fue una **U-Net de fusión multi-implante**.

| Modelo | Dice | IoU | Pixel accuracy |
|---|---:|---:|---:|
| U-Net Argus II | 0.8829 | 0.8351 | 0.9856 |
| U-Net PRIMA | 0.8711 | 0.8246 | 0.9817 |
| U-Net Alpha-AMS | 0.9019 | 0.8623 | 0.9863 |
| U-Net fusión | **0.9048** | **0.8756** | **0.9891** |

La reconstrucción de máscaras fue más sólida que la clasificación. La U-Net recuperó correctamente la silueta global de los objetos, aunque perdió detalles finos, bordes irregulares, huecos internos y partes pequeñas.

La pixel accuracy debe interpretarse con cuidado porque una gran proporción de los píxeles corresponde al fondo negro. Por eso, Dice e IoU son métricas más informativas para esta tarea.

## Estructura del repositorio

La carpeta principal está organizada de la siguiente manera:

```text
tp_final_vision_artificial/
│
├── informe/
│   ├── informe.pdf
│   └── poster.pdf
│
├── data/
│
├── lib/
│
├── notebooks/
│   ├── 00_dataset_extraction.ipynb
│   ├── 01_generacion_perceptos.ipynb
│   ├── 02_clasificador_perceptos.ipynb
│   └── 03_reconstruccion_mascaras.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── src/
│   ├── data/
│   │   └── coco_extraction.py
│   │
│   ├── classification/
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── knn_baselines.py
│   │   └── visualization.py
│   │
│   ├── reconstruction/
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── losses.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── visualization.py
│   │
│   └── simulation/
│       └── percept_generation.py
│
├── requirements.txt
└── README.md
````

## Datos

El trabajo utiliza COCO como fuente de anotaciones y máscaras de objetos. Se seleccionaron cuatro clases:

* `person`
* `car`
* `chair`
* `dining table`

La distribución final fue:

| Clase        | Muestras |
| ------------ | -------: |
| person       |      900 |
| car          |      900 |
| chair        |      900 |
| dining table |      624 |

Para cada muestra se generaron tres perceptos, uno por cada implante:

* Argus II
* PRIMA
* Alpha-AMS

El split de entrenamiento, validación y test se realizó por `sample_id`, no por fila individual. Esto evita data leakage, ya que una misma máscara genera perceptos para varios implantes. Todos los perceptos asociados a una misma muestra deben pertenecer al mismo split.

## Instalación y setup

El proyecto fue desarrollado principalmente en Google Colab / Colab Enterprise, pero puede ejecutarse en un entorno local con Python.

### 1. Clonar o abrir la carpeta del proyecto

Si se trabaja en Google Drive, ubicarse en la raíz del proyecto:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Luego:

```bash
cd /content/drive/MyDrive/tp_final_vision_artificial
```

En Colab Enterprise o Google Cloud, la ruta puede ser distinta. Lo importante es que la carpeta actual contenga:

```text
src/
notebooks/
data/
outputs/
requirements.txt
README.md
```

### 2. Crear entorno virtual local

Si se ejecuta localmente:

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Nota: en Google Colab / Colab Enterprise, PyTorch suele venir preinstalado con soporte CUDA. Si aparece un conflicto con `torch`, se recomienda usar la versión ya instalada por Colab y no reinstalarla manualmente.

### 4. Verificar imports principales

```python
import numpy as np
import pandas as pd
import torch
import sklearn
import pulse2percept
```

## Orden recomendado de ejecución

### Paso 1: extracción de máscaras COCO

Notebook sugerido:

```text
notebooks/01_generacion_perceptos.ipynb
```

Scripts relacionados:

```text
src/data/coco_extraction.py
```

Este paso:

1. Carga anotaciones de COCO.
2. Selecciona las clases objetivo.
3. Decodifica segmentaciones.
4. Genera máscaras binarias.
5. Redimensiona las máscaras a `256 x 256` usando letterbox.
6. Guarda metadata asociada.

Salida esperada:

```text
data/metadata/metadata_masks.csv
```

o directamente:

```text
data/metadata/metadata_percepts.csv
```

dependiendo de la versión del pipeline usada.

### Paso 2: generación de perceptos simulados

Notebook sugerido:

```text
notebooks/01_generacion_perceptos.ipynb
```

Scripts relacionados:

```text
src/data/percept_generation.py
```

Este paso:

1. Carga máscaras binarias.
2. Inicializa modelos de implante:

   * Argus II
   * PRIMA
   * Alpha-AMS
3. Usa `pulse2percept` y `ScoreboardModel` para generar perceptos.
4. Guarda perceptos como imágenes PNG.
5. Guarda activaciones por electrodo en archivos `.npz`.
6. Genera o actualiza `metadata_percepts.csv`.

Flujo conceptual:

```text
máscara binaria
→ estímulo visual
→ activación por electrodo
→ percepto simulado
```

Salida esperada:

```text
data/metadata/metadata_percepts.csv
outputs/figures/perceptos_ejemplo.png
```

Los perceptos completos pueden no estar incluidos en la entrega final por tamaño.

### Paso 3: clasificación de perceptos

Notebook sugerido:

```text
notebooks/02_clasificador_perceptos.ipynb
```

Scripts relacionados:

```text
src/classification/
```

Este paso entrena y evalúa:

* KNN por implante individual.
* KNN multi-implante.
* CNN por implante individual.
* CNN+GRU multi-implante.

Para KNN, el flujo es:

```text
percepto
→ resize 64 x 64
→ normalización [0, 1]
→ flatten
→ StandardScaler
→ PCA
→ KNN
```

Para CNN, el percepto se procesa como imagen en escala de grises.

Para CNN+GRU, los perceptos de los tres implantes se procesan con un encoder convolucional compartido y luego se fusionan mediante una GRU.

Salida esperada:

```text
outputs/tables/classification_results.csv
outputs/figures/confusion_knn_multi.png
outputs/models/knn_multi_implant.joblib
```

### Paso 4: reconstrucción de máscaras

Notebook sugerido:

```text
notebooks/03_reconstruccion_mascaras.ipynb
```

Scripts relacionados:

```text
src/reconstruction/
```

Este paso entrena y evalúa:

* U-Net Argus II.
* U-Net PRIMA.
* U-Net Alpha-AMS.
* U-Net de fusión multi-implante.

La entrada puede ser:

```text
1 canal: percepto individual
```

o:

```text
3 canales: Argus II + PRIMA + Alpha-AMS
```

La salida es una máscara binaria reconstruida.

La función de pérdida utilizada combina:

```text
BCEWithLogitsLoss + Dice Loss
```

Las métricas principales son:

* Dice
* IoU
* Pixel accuracy

Salida esperada:

```text
outputs/tables/reconstruction_results.csv
outputs/figures/reconstrucciones_unet_fusion.png
outputs/figures/curvas_unet_fusion.png
```

## Reproducibilidad

Para evitar data leakage, el split se realiza por `sample_id`.

Esto es importante porque cada muestra original genera tres perceptos, uno por implante. Si la división se hiciera fila por fila, el mismo objeto podría aparecer en entrenamiento con un implante y en test con otro. Por eso, todos los perceptos de una misma muestra deben quedar en el mismo split.

Se recomienda fijar una semilla global:

```python
SEED = 42
```

y usarla en:

* `numpy`
* `torch`
* `train_test_split`
* cualquier partición aleatoria del dataset

## Archivos no incluidos por tamaño

Por limitaciones de almacenamiento y transferencia, la entrega puede no incluir:

* imágenes completas de COCO,
* anotaciones completas de COCO,
* todos los perceptos PNG generados,
* todos los archivos `.npz`,
* checkpoints completos de redes neuronales,
* logs de entrenamiento extensos,
* carpetas de caché.

La entrega sí incluye:

* informe final,
* póster final,
* código fuente,
* notebooks,
* metadata liviana,
* tablas de resultados,
* figuras finales,
* modelos livianos cuando corresponde.

## Interpretación de los resultados

Los resultados muestran que los perceptos simulados conservan información útil, aunque degradada.

En clasificación, los desempeños fueron moderados. El mejor modelo fue KNN multi-implante con accuracy de test de `0.611`, superando a CNN+GRU. Esto sugiere que la combinación de perceptos de distintos implantes aporta información complementaria.

En reconstrucción, los resultados fueron más altos. La U-Net de fusión alcanzó Dice `0.9048` e IoU `0.8756`, mostrando que la forma global del objeto puede recuperarse con buena calidad desde los perceptos simulados.

Estos resultados no constituyen una validación clínica ni garantizan interpretabilidad humana. Deben entenderse como una validación computacional preliminar para comparar representaciones, implantes y estrategias de preprocesamiento.

## Trabajo futuro

Este pipeline puede extenderse para evaluar filtros o transformaciones visuales antes de realizar estudios con usuarios.

Una posible evaluación futura sería:

```text
imagen original
→ filtro candidato
→ pulse2percept
→ clasificación / reconstrucción
→ métrica objetiva
```

Si un filtro mejora la accuracy de clasificación o el Dice de reconstrucción, entonces preserva más información útil bajo la simulación.

Líneas futuras:

* comparar filtros visuales simples,
* usar bordes, segmentación, saliency o simplificación de escenas,
* extender a imágenes RGB,
* trabajar con escenas completas,
* evaluar secuencias de video,
* incorporar modelos temporales de estimulación,
* entrenar un surrogate model diferenciable de `pulse2percept`,
* optimizar filtros de forma end-to-end.

## Referencias principales

* Beyeler et al., *pulse2percept: A Python-Based Simulation Framework for Bionic Vision*, 2017.
* Han et al., *Deep Learning-Based Scene Simplification for Bionic Vision*, 2021.
* Sanchez-Garcia et al., *Semantic and Structural Image Segmentation for Prosthetic Vision*, 2018.
* Lin et al., *Microsoft COCO: Common Objects in Context*, 2014.
