# Clasificación de Enfermedades Respiratorias con CNN

**Proyecto Final — Corte 3 | Aprendizaje Supervisado**  
Maestría en Inteligencia Artificial · Segundo Cuatrimestre 2026

**Autores:** Bardales Calva José Andrés · Olvera Gonzalez David  
**Profesor:** Ernesto Garcia Amaro

---

## Descripción

Red Neuronal Convolucional (CNN) para clasificar radiografías de tórax en tres categorías:

| Clase | Descripción |
|-------|-------------|
| **COVID-19** | Neumonía viral por SARS-CoV-2 |
| **NEUMONIA** | Neumonía bacteriana/viral no-COVID |
| **NORMALL** | Pulmón sano |

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Test Accuracy | **96.12%** |
| Macro F1-Score | **90.39%** |
| Mejor Val Accuracy | **98.59%** (época 11/20) |
| COVID Precision | **1.000** (cero falsos positivos) |
| Parámetros | 25,784,835 |
| Tiempo entrenamiento | 33.3 min (CPU, AWS c5.2xlarge) |

### Métricas por clase

| Clase | Precisión | Recall | F1 |
|-------|-----------|--------|----|
| COVID | 1.000 | 0.862 | 0.926 |
| NEUMONIA | 0.984 | 0.974 | 0.979 |
| NORMALL | 0.719 | 0.920 | 0.807 |

---

## Arquitectura — RespiratoryCNN

```
Input (3×224×224)
    │
    ├─ Conv2D(3→32, 3×3) → BN → ReLU → MaxPool(2×2)   # 32×112×112
    ├─ Conv2D(32→64, 3×3) → BN → ReLU → MaxPool(2×2)  # 64×56×56
    ├─ Conv2D(64→128, 3×3) → BN → ReLU → MaxPool(2×2) # 128×28×28
    │
    ├─ Flatten → Linear(100352→256) → ReLU
    ├─ Dropout(p=0.3)
    └─ Linear(256→3) → Softmax
```

**Entrenamiento:** Adam (lr=1e-3) + CosineAnnealingLR + CrossEntropyLoss + Gradient Clipping

---

## Estructura del repositorio

```
├── scripts/
│   ├── experiment.py        # Script principal: entrenamiento + evaluación + figuras
│   └── app_inferencia.py    # App de inferencia web (Gradio)
│
├── ml/
│   ├── pipeline.py          # Pipeline MLOps (S3, Airflow, FastAPI)
│   ├── train.py             # Módulo de entrenamiento
│   └── requirements.txt     # Dependencias Python
│
├── docs/
│   ├── article/
│   │   └── report.md        # Artículo científico completo
│   └── outputs/
│       ├── fig1_class_distribution.png
│       ├── fig2_training_curves.png
│       ├── fig3_confusion_matrix.png
│       ├── fig4_roc_curves.png
│       ├── fig5_sample_predictions.png
│       ├── fig6_metrics_table.png
│       ├── metrics_summary.json
│       └── training_log.txt
│
└── .env.example             # Plantilla de variables de entorno (sin credenciales)
```

---

## Figuras de Resultados

### Distribución del Dataset
![Distribución de clases](docs/outputs/fig1_class_distribution.png)

### Curvas de Entrenamiento
![Curvas de entrenamiento](docs/outputs/fig2_training_curves.png)

### Matriz de Confusión
![Matriz de confusión](docs/outputs/fig3_confusion_matrix.png)

### Curvas ROC
![Curvas ROC](docs/outputs/fig4_roc_curves.png)

### Predicciones de Ejemplo
![Predicciones](docs/outputs/fig5_sample_predictions.png)

### Métricas por Clase
![Métricas](docs/outputs/fig6_metrics_table.png)

---

## Ejecutar localmente

### 1. Instalar dependencias

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib seaborn tqdm pillow numpy gradio
```

### 2. Entrenar el modelo

```bash
python scripts/experiment.py \
  --data-dir /ruta/al/dataset \
  --output-dir ./outputs \
  --epochs 20 \
  --batch-size 32
```

El dataset debe tener la estructura:
```
dataset/
├── train/
│   ├── COVID/
│   ├── NEUMONIA/
│   └── NORMALL/
├── val/
└── test/
```

### 3. Lanzar la app de inferencia

```bash
MODEL_PATH=./outputs/best_model.pth python scripts/app_inferencia.py
```

Abre `http://localhost:7860` en el browser, sube una radiografía y obtén el diagnóstico en tiempo real.

---

## Dataset

2,521 imágenes de rayos X de tórax, split 72/14/14:

| Partición | COVID | NEUMONIA | NORMALL | Total |
|-----------|------:|---------:|--------:|------:|
| Train | 135 | 1,544 | 126 | 1,805 |
| Val | 27 | 303 | 25 | 355 |
| Test | 29 | 307 | 25 | 361 |

---

## Stack tecnológico

- **Deep Learning:** PyTorch 2.11
- **Interfaz de inferencia:** Gradio
- **Cloud:** AWS EC2 (c5.2xlarge) + S3
- **Orquestación:** Apache Airflow
- **IaC:** Terraform
- **API:** FastAPI
