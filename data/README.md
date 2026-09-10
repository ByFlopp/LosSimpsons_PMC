# Datos

## Fuente

[Los Simpson Dataset](https://www.kaggle.com/datasets/alfaro96/los-simpson/data) (Kaggle, usuario `alfaro96`).

- ~20.000 imágenes RGB de resolución variable.
- 42 clases (un personaje por clase), distribución desbalanceada.
- Incluye un subconjunto de test (`kaggle_simpson_testset/`) sin etiquetar en el nombre de carpeta.

## Cómo obtenerlos

El dataset no se versiona en este repositorio (~1 GB de imágenes). Para reproducirlo:

1. Completar `KAGGLE_USERNAME` y `KAGGLE_KEY` en un archivo `.env` en la raíz del proyecto (ver `.env.example`).
2. Ejecutar:
   ```
   python scripts/descargar_dataset.py
   ```
3. Verificar que quede la carpeta `data/raw/simpsons_dataset/<personaje>/*.jpg`.

## Estructura esperada

```
data/
├── raw/                        # Descarga original de Kaggle (ignorado por git)
│   └── simpsons_dataset/
│       ├── homer_simpson/
│       ├── bart_simpson/
│       └── ...
└── processed/                  # Splits e imágenes preprocesadas generados por el notebook
    ├── train/
    ├── val/
    └── test/
```

El notebook [`notebooks/01_clasificacion_los_simpson_mlp.ipynb`](../notebooks/01_clasificacion_los_simpson_mlp.ipynb) genera el contenido de `data/processed/` a partir de `data/raw/`, usando una semilla fija para que la partición sea reproducible.
