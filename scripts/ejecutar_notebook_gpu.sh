#!/usr/bin/env bash
# Inicia Jupyter en WSL con las bibliotecas CUDA de TensorFlow visibles desde el arranque.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VIRTUAL_ENV:-$HOME/.venv}"

if [[ ! -x "$VENV_DIR/bin/python3" || ! -x "$VENV_DIR/bin/jupyter" ]]; then
    echo "No se encontró un entorno virtual con Jupyter en: $VENV_DIR" >&2
    echo "Activa tu entorno o define VIRTUAL_ENV antes de ejecutar este script." >&2
    exit 1
fi

NVIDIA_DIR="$VENV_DIR/lib/python3.12/site-packages/nvidia"
if [[ ! -d "$NVIDIA_DIR" ]]; then
    echo "No se encontraron bibliotecas NVIDIA en: $NVIDIA_DIR" >&2
    echo "Instala TensorFlow con: python3 -m pip install 'tensorflow[and-cuda]'" >&2
    exit 1
fi

CUDA_LIBS="$({ find "$NVIDIA_DIR" -path '*/lib/*.so*' -type f -printf '%h\n'; } | sort -u | paste -sd: -)"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "GPU WSL: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "Iniciando Jupyter con CUDA configurado..."
echo "Copia la URL http://localhost:8888 que aparecerá abajo y ábrela en tu navegador Windows."
exec "$VENV_DIR/bin/jupyter" notebook --no-browser "$REPO_DIR/notebooks/LosSimpsons_GPU_4060M.ipynb"
