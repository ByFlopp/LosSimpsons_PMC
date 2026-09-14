# Cómo ejecutar el notebook en GPU

Guía para correr [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb), la variante del proyecto que entrena sobre una GPU NVIDIA usando WSL.

Es una **copia exacta** del notebook base [LosSimpsonsPMC.ipynb](LosSimpsonsPMC.ipynb): mismo preprocesamiento, misma arquitectura, mismos experimentos y las mismas conclusiones. Solo cambian tres cosas: las rutas están en formato WSL, los modelos se guardan en `models_gpu/` en vez de `models/`, y se añade una sección inicial que verifica la GPU.

> Si no tienes GPU NVIDIA, no necesitas este archivo: usa el notebook base, que corre en CPU. Los resultados son los mismos.

## Dónde se ejecuta cada comando

Esto es lo que más confusión causa, así que conviene dejarlo claro antes de empezar. Hay **dos terminales distintas** y no son intercambiables:

| | Terminal de Ubuntu (WSL) | PowerShell (Windows) |
|---|---|---|
| **Cómo se abre** | Menú Inicio → escribe `Ubuntu`. O desde PowerShell, escribe `wsl` | Menú Inicio → escribe `PowerShell` |
| **Cómo se reconoce** | El prompt es `usuario@equipo:~$` | El prompt es `PS C:\Users\...>` |
| **Rutas** | `/home/usuario/...` y `/mnt/c/Users/...` | `C:\Users\...` |
| **Qué se hace aquí** | **Casi todo**: venv, pip, jupyter, kernel, entrenamiento | Solo instalar el driver NVIDIA y abrir el navegador |

**La regla corta: todo va en la terminal de Ubuntu (WSL), salvo dos excepciones que están marcadas explícitamente.**

En esta guía cada bloque de comandos lleva una etiqueta encima indicando dónde va. Si un comando de Ubuntu se pega en PowerShell no funcionará: dará errores de "comando no reconocido" o de rutas inexistentes.

> **Si trabajas en VS Code:** cuando el editor está conectado a WSL (abajo a la izquierda aparece `WSL: Ubuntu`), su terminal integrada **ya es la de Ubuntu**. Puedes usarla directamente para todo lo marcado como WSL.

Para salir de dudas en cualquier momento:

**➜ Terminal de Ubuntu (WSL)**

```bash
uname -r    # Si menciona "microsoft-standard-WSL2", estás en WSL
pwd         # Debe empezar con / y no con C:\
```

## Por qué hace falta una configuración especial

TensorFlow 2.21 instalado con `pip` trae las bibliotecas CUDA dentro del paquete `nvidia`, pero **no las encuentra solo**. Si lanzas Jupyter o un kernel sin preparar el entorno, verás esto:

```text
W ... Cannot dlopen some GPU libraries ... Skipping registering GPU devices...
GPUs: []
```

El notebook no falla en silencio ni entrena en CPU a escondidas: la celda de verificación lanza un `RuntimeError` explícito si no detecta GPU. Si llegas a ver ese error, es este problema.

La solución es que el proceso tenga `LD_LIBRARY_PATH` apuntando a `/usr/lib/wsl/lib` y a los directorios `lib/` dentro de `site-packages/nvidia/`. Hay dos formas de conseguirlo.

## Requisitos previos

### 1. Driver NVIDIA — se instala en Windows

**⚠️ Primera excepción: este paso va en Windows, NO dentro de WSL.**

Descarga el driver de tu tarjeta desde la web de NVIDIA e instálalo como cualquier programa de Windows. **No instales drivers NVIDIA dentro de Ubuntu**: WSL usa el del sistema anfitrión, y hacerlo por dentro rompe la configuración.

Para comprobar que WSL ve la GPU:

**➜ Terminal de Ubuntu (WSL)**

```bash
nvidia-smi
```

Debe listar tu tarjeta. Si responde `command not found`, falta el driver en Windows.

### 2. Entorno virtual — dentro de WSL

**➜ Terminal de Ubuntu (WSL)**

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install "tensorflow[and-cuda]" jupyter ipykernel
pip install -r "/mnt/c/Users/Laptop/Desktop/Trabajos/Duoc/3er Anio/2do Semestre/LosSimpsons_PMC/requirements.txt"
```

> Las rutas de esta guía son las del equipo de referencia, donde el usuario de WSL es `laptop` y el repositorio está en el escritorio de Windows. Si tu usuario o tu ubicación son distintos, ajusta `/home/laptop/...` y `/mnt/c/Users/Laptop/...` a los tuyos. Las comillas son necesarias porque la ruta contiene espacios.

> El entorno vive **dentro de WSL** (`~/.venv`), no en la carpeta del repositorio en Windows. Un `.venv` creado desde PowerShell no sirve: son instalaciones de Python distintas.

### 3. Dataset

Descárgalo fuera del repositorio. Puedes hacerlo con Windows normalmente; desde WSL se ve igual, solo cambia la forma de escribir la ruta:

| En Windows | En WSL |
|---|---|
| `C:\Users\Laptop\Desktop\Trabajos\LosSimpsonsDataset` | `/mnt/c/Users/Laptop/Desktop/Trabajos/LosSimpsonsDataset` |

En la celda de configuración del notebook, `DATASET_BASE` debe usar **siempre el formato WSL**:

```python
DATASET_BASE = Path("/mnt/c/Users/Laptop/Desktop/Trabajos/LosSimpsonsDataset")
```

La estructura esperada (25 carpetas en `train` y 25 en `test`) está descrita en el [README principal](../README.md).

## Opción A — Kernel dedicado, para trabajar en VS Code

Es la forma recomendada si editas el notebook dentro de VS Code, porque el kernel lleva la configuración CUDA en su propia definición y funciona sin depender de cómo abriste el editor.

**1. Crear el kernel** (una sola vez):

**➜ Terminal de Ubuntu (WSL)**

```bash
source ~/.venv/bin/activate
python3 -m ipykernel install --user --name simpsons-gpu --display-name "Python 3.12 (Simpsons GPU)"
```

**2. Añadirle las rutas CUDA.** Este es el paso que marca la diferencia; sin él el kernel existe pero no ve la GPU.

> **⚠️ Importante: `ipykernel install` borra este paso.** El comando del punto 1 **reescribe `kernel.json` entero**, así que elimina el bloque `env` que estás a punto de añadir. Si algún día el kernel deja de ver la GPU sin motivo aparente, lo más probable es que se haya vuelto a ejecutar el punto 1 (o que VS Code haya regenerado el kernel). La solución es repetir el punto 2. El script del final de esta sección es inmune a eso: **conserva** lo que ya hubiera en el archivo en vez de reemplazarlo, y puede ejecutarse las veces que haga falta.

Primero genera la lista de rutas:

**➜ Terminal de Ubuntu (WSL)**

```bash
find ~/.venv/lib/python3.12/site-packages/nvidia -path "*/lib/*.so*" -type f -printf "%h\n" | sort -u | paste -sd: -
```

Copia el resultado y edita `~/.local/share/jupyter/kernels/simpsons-gpu/kernel.json` para añadir el bloque `env`. Ese archivo está **dentro de WSL**, así que ábrelo desde VS Code conectado a WSL, o con `nano` en la terminal de Ubuntu:

Este es el archivo completo tal como quedó en el equipo de referencia. Si tu usuario de WSL también es `laptop` y el venv está en `~/.venv`, puedes copiarlo tal cual:

```json
{
  "argv": [
    "/home/laptop/.venv/bin/python3",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.12 (Simpsons GPU)",
  "language": "python",
  "env": {
    "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm/lib64:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/curand/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:/home/laptop/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib",
    "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/home/laptop/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc",
    "TF_CPP_MIN_LOG_LEVEL": "1"
  }
}
```

Son 12 directorios CUDA (`cublas`, `cudnn`, `cufft`, `nccl`…) más `/usr/lib/wsl/lib`, que es donde WSL expone `libcuda.so` del driver de Windows. El `find` de arriba los genera todos en el orden correcto; si tu usuario es distinto, ejecútalo y pega su salida después de `/usr/lib/wsl/lib:`.

### Script de reparación (recomendado)

En vez de editar el JSON a mano, este comando añade el bloque `env` **conservando** todo lo demás del archivo. Es la forma recomendada de hacer el paso 2, y también es lo que hay que ejecutar cada vez que el kernel deje de ver la GPU:

**➜ Terminal de Ubuntu (WSL)**

```bash
~/.venv/bin/python3 - <<'PY'
import json, os, glob
ruta = os.path.expanduser("~/.local/share/jupyter/kernels/simpsons-gpu/kernel.json")
venv = os.path.expanduser("~/.venv")
nvidia = f"{venv}/lib/python3.12/site-packages/nvidia"
libs = sorted({os.path.dirname(p) for p in glob.glob(f"{nvidia}/**/lib*/*.so*", recursive=True)})

spec = json.load(open(ruta))          # conserva argv, metadata, debugger, etc.
spec.setdefault("env", {}).update({
    "LD_LIBRARY_PATH": ":".join(["/usr/lib/wsl/lib", *libs]),
    "XLA_FLAGS": f"--xla_gpu_cuda_data_dir={nvidia}/cuda_nvcc",
    "TF_CPP_MIN_LOG_LEVEL": "1",
})
json.dump(spec, open(ruta, "w"), indent=2)
print(f"Reparado: {len(libs)} rutas CUDA añadidas")
PY
```

Debe imprimir `Reparado: 12 rutas CUDA añadidas`.

**Después de ejecutarlo hay que reiniciar el kernel en VS Code.** Un kernel ya en marcha conserva el entorno con el que arrancó; cambiar `kernel.json` no afecta a un proceso vivo. Usa el botón **Restart** del notebook (o *Command Palette* → `Jupyter: Restart Kernel`) y vuelve a ejecutar la sección 0.

**3. Abrir el notebook.** VS Code debe estar **conectado a WSL**, no abierto desde Windows. La forma más fiable de garantizarlo es lanzarlo desde Ubuntu:

**➜ Terminal de Ubuntu (WSL)**

```bash
cd "/mnt/c/Users/Laptop/Desktop/Trabajos/Duoc/3er Anio/2do Semestre/LosSimpsons_PMC"
code .
```

Comprueba que abajo a la izquierda diga `WSL: Ubuntu`. Luego, en el selector de kernel (arriba a la derecha del notebook), elige **"Python 3.12 (Simpsons GPU)"**. No el "Python 3.12.3" normal: ese entrena en CPU.

## Opción B — Script de arranque, para Jupyter en el navegador

Si prefieres no usar VS Code, [scripts/ejecutar_notebook_gpu.sh](../scripts/ejecutar_notebook_gpu.sh) prepara el entorno y levanta Jupyter en un solo paso:

**➜ Terminal de Ubuntu (WSL)**

```bash
cd "/mnt/c/Users/Laptop/Desktop/Trabajos/Duoc/3er Anio/2do Semestre/LosSimpsons_PMC"
./scripts/ejecutar_notebook_gpu.sh
```

El script imprime una URL `http://localhost:8888/...`.

**⚠️ Segunda excepción: esa URL se abre en el navegador de Windows** (Chrome, Edge, Brave). WSL comparte `localhost` con Windows, así que funciona sin configurar nada.

El script aborta con un mensaje claro si no encuentra el entorno virtual o las bibliotecas NVIDIA.

## Verificar que realmente está usando la GPU

Ejecuta la **sección 0** del notebook antes que nada. No se limita a listar dispositivos: fuerza una multiplicación de matrices y comprueba en qué dispositivo se ejecutó. Una salida correcta termina así:

```text
GPUs detectadas por TensorFlow: 1
  - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

Operacion de prueba ejecutada en: /job:localhost/replica:0/task:0/device:GPU:0
Entorno GPU verificado correctamente.
```

Si pasa esa celda, el resto del notebook se ejecuta en orden de principio a fin.

Para ver la GPU trabajando mientras entrena, abre **otra** terminal de Ubuntu:

**➜ Terminal de Ubuntu (WSL)**

```bash
watch -n 1 nvidia-smi
```

## Memoria RAM: el requisito que más sorprende

**El consumo de RAM depende por completo de `ALTO_OBJETIVO`/`ANCHO_OBJETIVO`**, y es la causa más común de que el kernel "se desconecte solo" a mitad de ejecución. No es un requisito de la GPU sino del preprocesamiento, que carga las 16.137 imágenes de entrenamiento en memoria.

El pico ocurre en la celda de normalización:

```python
imagenes_norm = imagenes.astype(np.float32) / VALOR_MAXIMO_PIXEL
```

Esa línea mantiene **tres arrays vivos a la vez**: el original en `uint8`, el resultado de `astype` en `float32`, y el resultado de la división, que es otro array nuevo (dividir no es una operación in-place).

| Resolución | Entrada del MLP | `uint8` | `float32` | **Pico** | ¿Cabe en 23 GB? |
|---|---:|---:|---:|---:|:--:|
| 64×64 | 12.288 | 0,20 GB | 0,79 GB | **1,78 GB** | ✅ |
| **128×128** (actual) | 49.152 | 0,79 GB | 3,17 GB | **7,14 GB** | ✅ |
| 256×256 | 196.608 | 3,17 GB | 12,69 GB | **28,55 GB** | ❌ |

Subir la resolución cuadruplica la memoria cada vez que se dobla el lado. A 256×256 el pico supera la RAM de la mayoría de los equipos: por eso el proyecto usa 128×128.

> Si necesitas exprimir memoria sin bajar la resolución, normaliza in-place y ahorras un array entero:
> ```python
> imagenes_norm = imagenes.astype(np.float32)
> imagenes_norm /= VALOR_MAXIMO_PIXEL
> ```
> El resultado es idéntico y el pico baja un 44 %.

**Al cambiar la resolución hay que actualizar los dos sitios donde aparece:** la celda de redimensionado y el valor de respaldo de la celda de predicción final (la que se usa al ejecutarla con un kernel nuevo, sin correr el notebook entero). Si quedan desincronizados, la predicción falla con un error de forma porque prepara la imagen a un tamaño distinto del que espera el modelo.

**El problema:** WSL toma por defecto el **50 % de la RAM de Windows**. En un equipo de 34 GB eso son ~15 GB. Si el pico lo supera, el kernel de Jupyter se queda sin memoria y Linux lo mata. En el notebook se ve como una desconexión repentina, **sin mensaje de error y sin traza de Python** — porque el proceso no falla, lo matan desde fuera. Ese silencio es justamente la firma del problema.

Para confirmar que fue esto:

**➜ Terminal de Ubuntu (WSL)**

```bash
dmesg | grep -i "killed process"
```

Si aparece una línea con `Out of memory: Killed process ... (python3)`, es este problema.

### Solución: subir el límite de RAM de WSL

Edita (o crea) el archivo `C:\Users\TU_USUARIO\.wslconfig` **en Windows** y añade:

```ini
[wsl2]
memory=24GB
swap=8GB
```

Ajusta el valor a tu equipo: deja al menos 8-10 GB para Windows. Con 128×128 bastan unos 12 GB para WSL; los 24 GB dan margen por si alguien sube la resolución.

Si tu equipo tiene poca RAM, la alternativa es bajar `ALTO_OBJETIVO`/`ANCHO_OBJETIVO` según la tabla de arriba. Ten en cuenta que **cambiar la resolución cambia los resultados**: hay que reentrenar y actualizar las métricas que reporta el [README principal](../README.md).

Para que el cambio surta efecto hay que reiniciar WSL por completo:

**➜ PowerShell (Windows)**

```powershell
wsl --shutdown
```

> Esto cierra todo lo que tengas abierto en WSL, VS Code incluido. Guarda antes. Al volver a abrir, comprueba con `free -h` en Ubuntu que el total refleja el nuevo límite.

## Ajustes de memoria de la GPU

El notebook activa `memory_growth` antes de entrenar, para que TensorFlow reserve VRAM a medida que la necesita en lugar de tomarla toda al iniciar.

### Por qué los datos se anclan a la RAM del host

Keras convierte los arrays de NumPy que recibe `fit()` en tensores sobre el **dispositivo por defecto**. Con una GPU visible, eso significa subir las matrices **completas** a la VRAM, no solo el lote de cada paso. A 64×64 pasa desapercibido (0,8 GB); a 128×128 son 2,4 GB de `X_train` más 0,6 GB de `X_val` que conviven con los pesos y los estados de Adam de cada configuración.

Medido en una RTX 4060 Laptop (8 GB) a 128×128, recorriendo las seis configuraciones de la experimentación:

| | VRAM al terminar cada configuración | Pico | Resultado |
|---|---:|---:|:--|
| Arrays de NumPy (por defecto) | 0,35 → 1,80 GB | 4,79 GB | ❌ **muere en E5 (batch 512)** |
| Anclados al host (lo que hace el notebook) | 0,35 → 2,17 GB | **2,93 GB** | ✅ termina |

Por eso la celda de entrenamiento hace esto antes del primer `fit()`:

```python
with tf.device("/CPU:0"):
    X_train = tf.convert_to_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)
    X_val = tf.convert_to_tensor(X_val)
    y_val = tf.convert_to_tensor(y_val)
```

Los datos se quedan en la RAM del host y cada paso copia únicamente su lote. No cambia el resultado del entrenamiento, `class_weight` sigue funcionando y el pico de VRAM baja a menos de 3 GB, así que **no hace falta tocar el batch 512 de E5**: la configuración del experimento se mantiene tal cual.

Si aun así ves `OOM`, `ResourceExhaustedError` o `Dst tensor is not initialized`, comprueba que esa conversión sigue en la celda de entrenamiento antes de bajar `TAMANO_BATCH`.

## Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `python3: command not found`, o `source` no se reconoce | Estás en PowerShell en vez de la terminal de Ubuntu | Abre Ubuntu desde el menú Inicio, o escribe `wsl` |
| `RuntimeError: No se detecto ninguna GPU` | El kernel no tiene `LD_LIBRARY_PATH` | Selecciona el kernel "Python 3.12 (Simpsons GPU)", no el Python normal |
| **Funcionaba y de pronto dejó de detectar la GPU** | El bloque `env` de `kernel.json` fue borrado (se reejecutó `ipykernel install`, o VS Code regeneró el kernel) | Ejecuta el **script de reparación** y reinicia el kernel. Comprueba con: `python3 -c "import json,os;print(list(json.load(open(os.path.expanduser('~/.local/share/jupyter/kernels/simpsons-gpu/kernel.json'))).get('env',{})))"` |
| Reparaste `kernel.json` pero sigue sin verla | El kernel en marcha conserva su entorno original | Reinicia el kernel (**Restart**), no basta con re-ejecutar la celda |
| `nvidia-smi: command not found` en WSL | Falta el driver NVIDIA para WSL | Instálalo **en Windows**, no dentro de Ubuntu |
| `Cannot dlopen some GPU libraries` | Faltan las rutas CUDA | Regenera el bloque `env` del kernel con el comando `find` de arriba |
| El kernel GPU no aparece en VS Code | VS Code no está conectado a WSL | Comprueba que abajo a la izquierda diga `WSL: Ubuntu`; si no, ábrelo con `code .` desde Ubuntu |
| VS Code: `WebSocket close with status code 1006` | El VS Code Server de WSL está incompleto | En Ubuntu: `ls ~/.vscode-server/bin/`; si está vacío, reconecta y deja que termine la descarga (~200 MB) |
| `FileNotFoundError` con el dataset | `DATASET_BASE` usa una ruta de Windows | Cambia `C:\...` por `/mnt/c/...` |
| **El kernel se desconecta solo, sin error** | El OOM killer lo mató: falta RAM en WSL | Sube `memory=24GB` en `.wslconfig` y reinicia con `wsl --shutdown`. Confirma con `dmesg \| grep -i "killed process"` |
| `InternalError: ... Dst tensor is not initialized` al entrenar (típicamente en E5) | Se acabó la **VRAM**: los datos se subieron enteros a la GPU | Comprueba que la celda de entrenamiento convierte `X_train`/`y_train`/`X_val`/`y_val` dentro de `tf.device("/CPU:0")` (ver [Ajustes de memoria de la GPU](#ajustes-de-memoria-de-la-gpu)) |
| El entrenamiento va lento y `nvidia-smi` marca 0 % | Está corriendo en CPU | Vuelve a ejecutar la sección 0 |

## Configuración de referencia

Esta es la combinación con la que se ejecutó el notebook, por si sirve de comparación:

| Componente | Versión |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| WSL | Ubuntu 24.04.5 LTS (WSL2) |
| Python | 3.12.3 |
| TensorFlow | 2.21.0 |
| Entorno | `~/.venv` dentro de WSL |
| Modelos generados | `models_gpu/` (ignorados por Git) |
