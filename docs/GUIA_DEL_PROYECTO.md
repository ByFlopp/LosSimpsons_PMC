# Guía del proyecto: reconocer personajes de Los Simpson con un Perceptrón Multicapa

> Documento explicativo del trabajo desarrollado en
> [`notebooks/LosSimpsonsPMC.ipynb`](../notebooks/LosSimpsonsPMC.ipynb).
> Está escrito para que se entienda sin haber visto el código, pero sin esconder los detalles
> técnicos: cada idea intuitiva viene acompañada de la cifra, la fórmula o la decisión concreta
> que hay detrás.

---

## 1. En una frase

Entrenamos una red neuronal densa (un **Perceptrón Multicapa**, MLP) para que, dada la imagen de
un fotograma de Los Simpson, diga **cuál de los 25 personajes** aparece en ella. Lo logra con un
**53,3 % de acierto** en imágenes que nunca vio — contra un 4 % que sacaría adivinando al azar — y,
al analizar sus errores, el propio modelo termina demostrando **por qué para imágenes se usan redes
convolucionales y no densas**. Ese hallazgo es tan parte del resultado como la métrica.

---

## 2. El problema, en términos precisos

Es una **clasificación multiclase de etiqueta única**: una imagen entra, y sale exactamente una
etiqueta entre 25 posibles. No es detección (no dibujamos cajas alrededor del personaje) ni
multi-etiqueta (no admitimos "aquí salen Bart y Lisa").

| Elemento | Definición |
|---|---|
| Entrada `X` | Imagen RGB, redimensionada a 64×64 y aplanada → vector de **12 288** números en [0, 1] |
| Salida `y` | Vector de **25 probabilidades** (softmax); la predicción es el argmax |
| Pérdida | `categorical_crossentropy` |
| Datos | 16 137 imágenes de entrenamiento / 3 937 de prueba, 25 clases |

### El detalle que condiciona todo: el desbalance

Las clases no están repartidas de forma pareja, ni de lejos:

- `homer_simpson`: **1 796** imágenes de entrenamiento
- `selma_bouvier`: **82** imágenes

Es una razón de **≈ 22:1**. Esto tiene una consecuencia directa: un modelo que aprenda a responder
"Homero" ante la duda va a *parecer* bueno mirando solo el porcentaje de aciertos. Por eso fijamos
desde el principio dos **baselines** contra los cuales medirnos:

| Estrategia tonta | Accuracy |
|---|---|
| Responder al azar (1/25) | 4,0 % |
| Responder siempre `homer_simpson` | 11,4 % |

Cualquier modelo que no supere claramente ambos no aporta nada.

---

## 3. Pre-procesamiento: de una carpeta de PNG a una matriz de números

Una red densa no acepta "una imagen": acepta un **vector de longitud fija**. El pipeline hace esa
traducción en cinco pasos, y cada uno está verificado en el notebook con una tabla de comprobaciones
(forma, tipo de dato, rango, reversibilidad) que falla ruidosamente si algo no cuadra.

### 3.1 Redimensionado a 64×64

Las imágenes originales van desde 256×257 hasta 1912×1072 píxeles. Hay que unificarlas, y el tamaño
elegido **no es un detalle estético**: define el tamaño del modelo. Al aplanar, la capa de entrada
tiene `alto × ancho × 3` valores, y cada uno se conecta con todas las neuronas de la primera capa.
Los parámetros crecen de forma **cuadrática con el lado de la imagen**:

| Resolución | Vector de entrada | Parámetros de la 1ª capa (512 neuronas) |
|---|---|---|
| 32×32 | 3 072 | ≈ 1,6 M |
| **64×64** | **12 288** | **≈ 6,3 M** |
| 128×128 | 49 152 | ≈ 25,2 M |

Elegimos **64×64** porque los personajes de Los Simpson se distinguen por rasgos de **baja frecuencia
espacial** — silueta, color de piel, peinado, ropa — que sobreviven bien a esa escala. Duplicar el
lado cuadruplicaría los parámetros sin un beneficio equivalente: como el MLP aplana la imagen, no
puede aprovechar el detalle fino de todos modos.

### 3.2 Normalización: dividir por 255

Los píxeles vienen como enteros `uint8` en [0, 255]. Si se los damos crudos a la red, las
combinaciones lineales de la primera capa producen números enormes: las activaciones tipo tanh se
**saturan** (gradiente ≈ 0) y con ReLU los gradientes quedan tan dispares entre sí que hay que usar
learning rates diminutos para que el entrenamiento no se desestabilice.

Dividir por 255 deja todo en [0, 1] y en `float32` (el tipo con el que opera Keras, la mitad de
memoria que `float64`). Es una línea de código que decide si el entrenamiento converge o no.

### 3.3 Aplanado: el paso que se paga caro después

`(64, 64, 3)` → vector de `12 288`. Una capa `Dense` conecta cada neurona con **todas** las entradas;
internamente, para ella todas las entradas son intercambiables. No tiene forma de saber que dos
píxeles eran vecinos.

> **Este es el momento en que se pierde la estructura espacial**, y es la raíz de casi todas las
> limitaciones que aparecen en la sección 7. Si permutáramos las 12 288 columnas de `X` con una
> permutación fija, el modelo aprendería exactamente igual de bien — algo impensable para un ojo
> humano, y que revela cuánta información estamos tirando.

### 3.4 Codificación de etiquetas: one-hot

`homer_simpson` → índice `9` → vector `[0,…,1,…,0]` de 25 posiciones.

¿Por qué no dejarlo como el entero 9? Porque un entero **sugiere un orden y una distancia que no
existen**: insinúa que la clase 5 está "entre" la 4 y la 6, o que la 24 es "mayor" que la 1. Los
personajes son categorías nominales. El one-hot deja a las 25 clases equidistantes entre sí, y además
coincide exactamente con la forma de la salida softmax, que es lo que espera
`categorical_crossentropy`.

### 3.5 Partición: tres conjuntos, no dos

El dataset ya trae `train` y `test`, pero eso **no basta**. En cuanto usamos el test para decidir
cuántas capas poner o cuándo parar, deja de ser una estimación honesta: esas decisiones ya lo
incorporaron. Eso se llama **fuga de información**. Entonces:

- **Entrenamiento (80 % = 12 909 imgs)** → ajusta los pesos.
- **Validación (20 % = 3 228 imgs)** → nunca toca los pesos; sirve para las curvas, el `EarlyStopping`
  y la comparación entre experimentos.
- **Test (3 937 imgs)** → intacto, se abre **una sola vez**, al final.

La partición es **estratificada** (`stratify=y`): cada clase conserva su proporción en ambas partes.
Con un desbalance de 22:1, un reparto aleatorio simple podría dejar a `selma_bouvier` con dos o tres
imágenes de validación, y entonces su métrica sería puro azar. La semilla es fija (`random_state=42`)
para que las diferencias entre experimentos se deban al factor cambiado y no a la suerte del reparto.

### 3.6 Desbalance: `class_weight`

Consideramos cuatro estrategias:

| Estrategia | Veredicto |
|---|---|
| Submuestrear las mayoritarias | ✗ Tira información y el dataset ya es modesto |
| Sobremuestrear por duplicado | ✗ Los duplicados exactos favorecen la memorización |
| Aumento de datos | ~ Útil, pero encarece el entrenamiento → queda como propuesta |
| **Pesos por clase** | ✓ **Elegida:** no altera los datos y actúa justo donde está el sesgo |

Keras multiplica la pérdida de cada ejemplo por el peso de su clase, con la fórmula
`n_muestras / (n_clases × n_ejemplos_de_la_clase)`. Equivocarse con Selma (peso **7,82**) cuesta más
que equivocarse con Homero (peso **0,36**).

Dos precauciones que importan:

1. Los pesos se calculan **solo con `y_train`**. Usar validación para esto sería otra forma de fuga.
2. Se aplican **solo durante el entrenamiento**. Las métricas se miden sin pesos, porque deben
   reflejar el desempeño sobre la distribución real.

---

## 4. El modelo

### Arquitectura base

```
entrada (12 288)
  → Dense(512, relu) → BatchNorm → Dropout(0.3)
  → Dense(256, relu) → BatchNorm → Dropout(0.3)
  → Dense(25, softmax)
```

**6 431 257 parámetros entrenables**, de los cuales el **97,8 %** están en la primera capa: es la
única conectada a los 12 288 valores de entrada. Eso deja una razón de **498 parámetros por imagen de
entrenamiento**, que es exactamente lo que justifica regularizar con fuerza.

### Por qué cada pieza

| Decisión | Valor | Razón |
|---|---|---|
| Capas ocultas | 2 | Con una, la red es casi un clasificador lineal sobre píxeles; con muchas más, el sobreajuste crece más rápido que la ganancia |
| Neuronas | 512 → 256 | Embudo decreciente: obliga a **resumir** en vez de copiar la entrada |
| Activación oculta | ReLU | No se satura (derivada = 1 para z > 0) y es la más barata de calcular |
| Salida | Softmax (25) | Convierte puntuaciones en una distribución que suma 1 |
| Dropout | 0,3 | Apaga neuronas al azar en cada paso, así ninguna depende de otra en particular. 0,5 retrasaría demasiado la convergencia |
| BatchNormalization | Tras cada capa | Estabiliza la escala de las activaciones y permite un learning rate mayor |
| Optimizador | Adam (lr 1e-3) | Adapta el paso por parámetro; con entradas tan dispares converge bastante más rápido que SGD puro |
| Batch | 128 | Compromiso entre el ruido del gradiente y la estabilidad |

### Control del entrenamiento

En vez de fijar las épocas a mano, se pone un máximo alto (60) y se deja que tres *callbacks* decidan:

| Callback | Qué hace |
|---|---|
| `EarlyStopping` | Corta si `val_loss` no mejora en 8 épocas y **restaura los pesos de la mejor época** |
| `ReduceLROnPlateau` | Divide el learning rate a la mitad si la validación se estanca 4 épocas |
| `ModelCheckpoint` | Persiste en `models/` la mejor época |

La métrica que gobierna es `val_loss` y no `val_accuracy`: con clases desbalanceadas la accuracy puede
quedarse plana mientras la pérdida ya está empeorando.

**Resultado del base:** paró en la época 51, con la mejor en la 43. Accuracy de validación **0,480**,
F1 macro **0,438**, brecha train−val de **0,128** → sobreajuste moderado, contenido por el Dropout y
la parada temprana.

---

## 5. Experimentación: un factor a la vez

Cada experimento cambia **exactamente un** hiperparámetro respecto del base y deja todo lo demás
idéntico — misma semilla, misma partición, mismos pesos, mismos callbacks — de modo que la diferencia
observada sea atribuible a ese factor. Presupuesto común: 35 épocas, paciencia 6.

| Exp. | Qué cambia | val_accuracy | val_F1 macro | Params | Tiempo |
|---|---|---|---|---|---|
| **E2** | 3 capas (1024, 512, 256) | **0,510** | **0,486** | 13,25 M | 310 s |
| Base | 2 capas (512, 256) | 0,480 | 0,438 | 6,43 M | — |
| E5 | batch 512 | 0,441 | 0,408 | 6,43 M | 80 s |
| E4 | learning rate 1e-4 | 0,433 | 0,393 | 6,43 M | 137 s |
| E1 | 1 capa (512) | 0,280 | 0,253 | 6,31 M | 35 s |
| E3 | activación tanh | 0,034 | 0,007 | 6,43 M | 61 s |

### Qué enseña cada comparación

**Capacidad (E1 / Base / E2).** Pasar de 1 a 2 capas sí se nota mucho (+0,19 de F1): la segunda capa
aporta una etapa real de composición. Pasar de 2 a 3 **duplica los parámetros para ganar +0,05**.
Rendimientos decrecientes muy marcados, y la razón es estructural: añadir neuronas da más
combinaciones lineales de píxeles, pero **ninguna de ellas recupera la información espacial que se
perdió en el aplanado**. El cuello de botella no es la capacidad.

**Activación (E3).** El colapso de tanh es el ejemplo más claro del notebook: 0,034 de accuracy, o sea
*peor que responder siempre Homero*. La tanh está acotada en (−1, 1) y su derivada cae a cero en los
extremos; cuando una neurona se satura, el gradiente que la atraviesa se atenúa y la unidad deja de
aprender. Cortó en la época 8 restaurando los pesos de la 2ª. No es que tanh sea incapaz — necesita
muchas más épocas para llegar a un punto comparable — pero con el mismo presupuesto, ReLU gana sin
discusión.

**Optimización (E4 / E5).** Con lr 1e-4 cada paso es diez veces más corto: el descenso es más suave
pero, dentro del mismo presupuesto, el modelo se queda más lejos del mínimo. Un learning rate
demasiado pequeño no diverge — simplemente no llega a tiempo. Con batch 512 hay cuatro veces menos
actualizaciones por época; el gradiente es menos ruidoso, pero ese ruido tenía un efecto regularizador
que también se pierde.

### La selección

Se elige **E2** por **F1 macro de validación**, no por accuracy. Con 22:1 de desbalance, la accuracy
premia acertar en los personajes abundantes y apenas se resiente si el modelo ignora a los escasos; el
F1 macro promedia las 25 clases sin ponderar, así que sí lo penaliza. Como el objetivo declarado es
reconocer **a los 25**, esa es la métrica coherente con el problema.

---

## 6. Evaluación final sobre el test

Se abre el conjunto `test` — 3 937 imágenes nunca vistas — procesado con **exactamente las mismas
funciones** y, crítico, **el mismo diccionario `clase_a_indice`**. Si las clases se renumeraran a
partir del test, el índice 3 podría significar otro personaje y toda la evaluación quedaría
invalidada. El notebook lo verifica explícitamente antes de medir.

| Métrica | Valor | Qué mide |
|---|---|---|
| **Accuracy** | **0,533** | Proporción global de aciertos; la dominan las clases grandes |
| F1 ponderado | 0,532 | F1 promediado pesando por tamaño de clase |
| **F1 macro** | **0,507** | Promedio simple de las 25 clases: cada personaje pesa igual |
| Top-3 accuracy | 0,733 | La clase correcta está entre las 3 más probables |
| Pérdida | 1,703 | — |

**× 13,3 sobre el azar** y **× 4,7 sobre la clase mayoritaria.** El modelo aprendió algo real.

### Lo que dice el desglose por clase

| Mejores (F1) | | Peores (F1) | |
|---|---|---|---|
| `marge_simpson` | 0,725 | `maggie_simpson` | 0,242 |
| `kent_brockman` | 0,690 | `nelson_muntz` | 0,354 |
| `chief_wiggum` | 0,625 | `charles_montgomery_burns` | 0,372 |
| `sideshow_bob` | 0,613 | `lenny_leonard` | 0,411 |

Dos lecturas que conviene separar:

- **Precisión** de un personaje = "de todo lo que el modelo llamó Homero, ¿cuánto era Homero?"
- **Recall** = "de todos los Homeros que había, ¿cuántos encontró?"

Homero es el caso didáctico perfecto: **precisión 0,703 pero recall 0,369**. Cuando dice "Homero"
suele acertar, pero se le escapan casi dos tercios de los Homeros reales — porque `class_weight` hizo
justamente lo que debía, quitarle el privilegio de ser la respuesta por defecto.

La correlación entre imágenes de entrenamiento y F1 por clase es **0,289**: positiva, pero más débil
de lo que uno esperaría. Es decir, la cantidad de datos explica parte de la historia, no toda — la
similitud visual entre personajes explica el resto.

### Confusiones más frecuentes

| Real → Predicho | Casos | % de la clase real |
|---|---|---|
| `lisa_simpson` → `bart_simpson` | 64 | 23,6 % |
| `homer_simpson` → `abraham_grampa_simpson` | 37 | 8,2 % |
| `homer_simpson` → `sideshow_bob` | 28 | 6,2 % |
| `bart_simpson` → `lisa_simpson` | 27 | 10,0 % |

`bart_simpson` absorbe **127** imágenes de otras clases: es la "respuesta por defecto" del modelo.
Las causas identificables son cinco: paleta de colores compartida (todos amarillos), **fondos
compartidos** (el mismo salón, la misma escuela — y el fondo aporta una fracción enorme de los 12 288
valores de entrada), escenas con varios personajes pero una sola etiqueta, falta de invarianza al
encuadre, y escasez en las clases minoritarias.

### El experimento más revelador: mapas de saliencia

Se midió cuánto cambia la probabilidad de la clase predicha al perturbar cada píxel. El resultado
importa por lo que **no** muestra: la influencia no se concentra sobre el personaje, sino que queda
repartida como un moteado por todo el encuadre, fondo incluido. El 10 % de píxeles más influyentes
acumula apenas un **≈ 20 %** de la influencia total — muy cerca del 10 % que daría un reparto
perfectamente uniforme.

> El modelo **no decide dónde mirar, porque no tiene noción de "dónde"**. Lo que aprende son
> combinaciones de valores de color en posiciones fijas del encuadre.

---

## 7. Las limitaciones del MLP (y por qué existen las CNN)

Ninguna de estas se arregla con hiperparámetros. Son consecuencias de usar una red densa sobre
imágenes.

1. **Pérdida de información espacial.** Para la red, dos píxeles vecinos y dos en esquinas opuestas
   son igual de "cercanos": ninguno.
2. **Sin invarianza a traslación ni escala.** Cada peso de la primera capa está atado a una posición
   concreta. "Homero centrado", "Homero a la izquierda" y "Homero de lejos" son, para el modelo, tres
   conceptos sin relación entre sí, que debe aprender por separado.
3. **Explosión de parámetros.** Crecen cuadráticamente con el lado de la imagen. A 64×64 ya son
   millones; a 128×128 serían cuatro veces más. Eso obliga a trabajar con imágenes diminutas y
   dispara la razón parámetros/ejemplos.
4. **No reutiliza patrones locales.** Un filtro convolucional aprende una vez a detectar un borde y lo
   aplica en toda la imagen con los mismos pesos. El MLP debe aprender el mismo patrón dos veces si
   aparece en dos posiciones, con dos conjuntos de pesos y ejemplos distintos para cada uno.
5. **Sensibilidad al fondo.** Todos los píxeles entran con el mismo estatus. Si un personaje aparece
   a menudo en cierto escenario, el modelo aprenderá **el escenario**.

### Propuestas de mejora, ordenadas por impacto esperado

| Propuesta | Qué limitación ataca |
|---|---|
| **Redes convolucionales (CNN)** | Las cuatro primeras a la vez. Es *la* mejora |
| **Transfer learning** (MobileNet, ResNet, EfficientNet) | La vía más rápida a un salto grande con pocos datos propios |
| **Aumento de datos** (giros, recortes, brillo) | Sobreajuste, y sobre todo las clases minoritarias |
| **Focal loss / sobremuestreo con aumento** | La brecha accuracy − F1 macro |
| **Recortar al personaje antes de clasificar** | Elimina el fondo irrelevante (5) y normaliza el encuadre |
| **Subir a 96×96 o 128×128** | Solo viable ya con convoluciones; recupera detalle fino |
| **Validación cruzada** | Saber si las diferencias entre configuraciones son reales o ruido |

---

## 8. Tres conclusiones que se llevan a otros proyectos

1. **La métrica que eliges define el modelo que obtienes.** Optimizar accuracy en un problema
   desbalanceado lleva directamente a un modelo que ignora a las clases pequeñas y aun así parece
   bueno. Toda la selección de este trabajo cambia si se usa accuracy en vez de F1 macro.
2. **La representación de la entrada pesa más que la capacidad del modelo.** Duplicar los parámetros
   (E2) dio +0,05 de F1. El techo no estaba en cuántos pesos tenía la red, sino en que el aplanado
   destruye la estructura de la imagen antes de que la red vea nada.
3. **Chocar con los límites del MLP es la mejor forma de entender las CNN.** Falta de invarianza,
   explosión de parámetros, sensibilidad al fondo: la convolución y el pooling no son trucos, son la
   respuesta directa a cada uno de esos problemas concretos.

---

## 9. Cómo reproducirlo

```bash
# 1. Dependencias
pip install -r requirements.txt

# 2. Dataset - IMPORTANTE: descargar FUERA del repo
curl -L -o ~/Downloads/los-simpson.zip \
  https://www.kaggle.com/api/v1/datasets/download/alfaro96/los-simpson

# 3. Ajustar DATASET_BASE en la primera celda de codigo del notebook
#    y ejecutar de arriba a abajo
jupyter notebook notebooks/LosSimpsonsPMC.ipynb
```

La variable `conjunto` de la primera celda controla con qué mitad se entrena; el conjunto de
evaluación final es **siempre el opuesto**, derivado automáticamente. Los modelos entrenados quedan
en [`models/`](../models/) (`mlp_base.keras`, `mlp_e1`…`mlp_e5`, `mlp_final.keras`).

**Cifras de referencia** (semilla 42, entrenando con `train`): TensorFlow 2.21.0, 6,43 M de
parámetros en el base y 13,25 M en el final, ~5 s por época en el base.

---

## Glosario rápido

| Término | En una línea |
|---|---|
| **MLP** | Pila de capas densas; cada neurona se conecta con todas las de la capa siguiente |
| **Época** | Una pasada completa por todas las imágenes de entrenamiento |
| **Batch** | Grupo de imágenes que se promedian antes de actualizar los pesos (aquí 128) |
| **Learning rate** | Cuánto se mueven los pesos en cada actualización |
| **Softmax** | Convierte puntuaciones en probabilidades que suman 1 |
| **ReLU** | `max(0, z)`. No se satura y es baratísima de calcular |
| **Dropout** | Apagar neuronas al azar durante el entrenamiento para evitar dependencias |
| **BatchNorm** | Reescalar las activaciones para mantener media y varianza estables |
| **Sobreajuste** | La red memoriza el entrenamiento en vez de aprender a generalizar |
| **Precisión** | De lo que predije como X, ¿cuánto era X? |
| **Recall** | De todos los X que había, ¿cuántos encontré? |
| **F1** | Media armónica de precisión y recall |
| **F1 macro** | Promedio simple del F1 de las 25 clases: cada personaje pesa igual |
