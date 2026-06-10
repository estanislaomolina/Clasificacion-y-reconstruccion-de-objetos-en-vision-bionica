# Encoder Semántico para Visión Prostética Simulada

Pipeline de visión por computadora que evalúa si el preprocesamiento semántico
mejora la utilidad de imágenes transmitidas a un simulador de implante retinal.

## Estructura del proyecto

```
tp/
├── notebooks/
│   └── 01_preprocessing.ipynb   # Pipeline completo sobre una imagen
├── src/
│   ├── coco_loader.py            # Carga de imágenes y máscaras GT de COCO
│   ├── encoder.py                # Encoder semántico (espacio de experimentación)
│   ├── simulator.py              # Wrapper de pulse2percept (3 implantes)
│   ├── metrics.py                # Contraste, IoU simulado, SSIM
│   └── visualization.py         # Figuras comparativas
├── data/
│   └── coco/
│       ├── annotations/
│       │   └── instances_val2017.json
│       └── val2017/              # Imágenes
├── outputs/
│   ├── figures/                  # Visualizaciones de máscaras GT
│   ├── percepts/                 # Figuras comparativas de percepción
│   ├── preprocessed/             # Imágenes tras el encoder semántico
│   └── metrics/                  # CSVs de métricas
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### Datos

Descargar COCO 2017 validation set:
- Imágenes: http://images.cocodataset.org/zips/val2017.zip → extraer en data/coco/val2017/
- Anotaciones: http://images.cocodataset.org/annotations/annotations_trainval2017.zip → extraer instances_val2017.json en data/coco/annotations/

## Orden de ejecución

1. `notebooks/01_preprocessing.ipynb` — exploración del pipeline sobre una imagen
2. *(próximos notebooks)* evaluación sobre múltiples imágenes, integración de YOLO11n-seg

## Espacio de experimentación

El encoder semántico se configura en `src/encoder.py` mediante `EncoderConfig`.
Las variables principales son:

- `enhance_techniques`: técnicas de resalte sobre objetos prioritarios
  - `"brightness"`: incremento de brillo
  - `"contrast"`: incremento de contraste
  - `"edges_canny"`: resalte de bordes con Canny
  - `"edges_sobel"`: resalte de bordes con Sobel
  - `"dilate"`: dilatación de la máscara
- `attenuate_technique`: cómo se trata el fondo
  - `"dim"`: atenuación por factor
  - `"blur"`: desenfoque gaussiano
  - `"black"`: fondo negro completo
- `min_priority`: prioridad mínima para considerar un objeto como relevante

## Implantes simulados

| Implante        | Electrodos | Descripción                          |
|-----------------|------------|--------------------------------------|
| `argus2`        | 60         | Argus II — referencia clínica        |
| `prima`         | ~378       | PRIMA — implante subretinal moderno  |
| `custom_grid`   | 900 (30×30)| Grilla de alta resolución — cota sup.|