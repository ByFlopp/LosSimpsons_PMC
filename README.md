# Clasificación de personajes de Los Simpson con un Perceptrón Multicapa

Informe técnico — Evaluación Parcial N°1, TLY1102 (Técnicas Avanzadas de Machine Learning I).

> El desarrollo completo y ejecutable de cada etapa está en [`notebooks/01_clasificacion_los_simpson_mlp.ipynb`](notebooks/01_clasificacion_los_simpson_mlp.ipynb). Este documento resume el problema, los objetivos, los KPIs, las fuentes de datos y la metodología.

## Descripción del problema de negocio

Un estudio de animación produce grandes volúmenes de material gráfico (capturas de episodios, material promocional, contenido de fans) que necesita etiquetarse por personaje para poder buscarlo, catalogarlo y reutilizarlo. Etiquetar manualmente miles de imágenes es lento y propenso a error humano.

Este proyecto aborda ese problema como una tarea de **clasificación de imágenes**: dado un fotograma con un personaje de Los Simpson, predecir automáticamente de qué personaje se trata, usando una red neuronal tipo **Perceptrón Multicapa (MLP)**.

### Variante del grupo

El dataset original tiene 42 clases con una distribución muy desbalanceada (personajes secundarios con menos de 50 imágenes frente a Homero o Bart con más de 2.000). Para obtener un problema tratable y mejor balanceado con un MLP —que no explota estructura espacial y escala mal en número de parámetros al aumentar clases y resolución—, la variante de este grupo restringe el problema a los **10 personajes con más imágenes** del dataset. La selección exacta y la semilla usada quedan documentadas en el notebook.

## Objetivos del proyecto

- Implementar un pipeline reproducible de clasificación de imágenes con un MLP, desde la carga de datos hasta el análisis de errores.
- Entrenar un modelo capaz de distinguir entre los 10 personajes seleccionados a partir de una imagen RGB.
- Evaluar el modelo con métricas estándar de clasificación e interpretar sus resultados, incluyendo sus limitaciones frente a arquitecturas más adecuadas para imágenes (CNN).

## Definición de KPIs

| KPI | Definición | Meta |
| --- | --- | --- |
| Accuracy de validación | Proporción de imágenes de validación correctamente clasificadas | ≥ 60% (10 clases, línea base aleatoria ≈ 10%) |
| F1-Score macro (test) | Promedio no ponderado del F1 por clase, penaliza el desbalance | ≥ 0.55 |
| Brecha train/val | Diferencia entre accuracy de entrenamiento y validación | < 15 puntos porcentuales (control de sobreajuste) |
| Clases con recall < 0.4 | Cantidad de personajes que el modelo confunde sistemáticamente | ≤ 2, con causa raíz identificada en el análisis de errores |

Estos KPIs resuelven el problema de negocio porque miden, respectivamente: si el modelo es útil en general (accuracy), si es útil de forma pareja entre personajes (F1 macro), si generaliza a imágenes nuevas (brecha train/val) y si hay personajes que requieren una solución distinta como más datos o una CNN (clases con bajo recall).

## Descripción de las fuentes de datos

- **Dataset:** [Los Simpson Dataset](https://www.kaggle.com/datasets/alfaro96/los-simpson/data) (Kaggle, `alfaro96`).
- **Contenido:** ~20.000 imágenes RGB, resolución variable, 42 clases (una por personaje), más un subconjunto de test.
- **Licencia/uso:** dataset público de Kaggle con fines educativos.
- **Detalle de la carga y del preprocesamiento:** [`data/README.md`](data/README.md).

## Preparación y análisis exploratorio de los datos (EDA)

Desarrollados en detalle en el notebook (sección b y c). En resumen:

- Conteo de imágenes por clase y visualización de la distribución para justificar la variante (top-10 clases).
- Inspección visual de ejemplos por clase para detectar variabilidad de pose, iluminación, oclusión y calidad.
- Revisión de dimensiones/aspect ratio de las imágenes para decidir el tamaño de reescalado.
- Partición reproducible en train/val/test (semilla fija), estratificada por clase.
- Preprocesamiento: reescalado a un tamaño fijo, normalización de píxeles a `[0, 1]` y aplanado del tensor de imagen a vector (requisito de entrada de un MLP).

## Metodología utilizada (CRISP-DM)

| Fase CRISP-DM | Aplicación en este proyecto |
| --- | --- |
| 1. Comprensión del negocio | Definición del problema de negocio, objetivos y KPIs (secciones anteriores). |
| 2. Comprensión de los datos | EDA: distribución de clases, calidad e inspección visual de las imágenes (notebook, secciones b). |
| 3. Preparación de los datos | Selección de la variante (top-10 clases), split train/val/test, reescalado, normalización y aplanado (notebook, sección c). |
| 4. Modelado | Diseño e implementación del MLP: capas, funciones de activación, función de pérdida y optimizador (notebook, sección d). |
| 5. Evaluación | Entrenamiento con seguimiento de curvas, métricas de clasificación, matriz de confusión y análisis de errores (notebook, secciones e-g). |
| 6. Despliegue | Fuera del alcance de esta evaluación; se documenta como trabajo futuro en las conclusiones del notebook. |

## Estructura del proyecto

```
LosSimpsons_PMC/
├── data/                # Datos crudos y procesados (ver data/README.md)
├── docs/                # Material de la asignatura usado como referencia
├── images/              # Figuras exportadas por el notebook (curvas, matriz de confusión, ejemplos)
├── models/              # Modelo entrenado (generado al ejecutar el notebook)
├── notebooks/           # Desarrollo completo y documentado del proyecto
├── scripts/             # Utilidades (descarga del dataset)
├── .env.example         # Plantilla de credenciales de Kaggle
├── requirements.txt     # Dependencias de Python
└── README.md            # Este informe técnico
```

## Cómo reproducir el proyecto

```bash
python -m venv .venv
.venv/Scripts/activate        # en Windows; en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # completar KAGGLE_USERNAME y KAGGLE_KEY
python scripts/descargar_dataset.py

jupyter notebook notebooks/01_clasificacion_los_simpson_mlp.ipynb
```
