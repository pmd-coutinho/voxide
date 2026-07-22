#!/usr/bin/env bash
# Installs the user-local CUDA Python runtime used by Voxide's Nemotron 3.5
# engine. The model itself is intentionally downloaded from Voxide's Voice
# Engine screen so it can be removed independently of this runtime.
set -euo pipefail

runtime_root="${1:-${VOXIDE_NEMOTRON_RUNTIME_ROOT:-$HOME/.local/share/voxide-nemotron}}"
venv="$runtime_root/venv"
temporary="$runtime_root/tmp"

pick_python() {
  local candidate
  for candidate in "${VOXIDE_NEMOTRON_PYTHON:-}" python3.12 python3; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" - <<'PY'
import sys
if sys.version_info >= (3, 10):
    print(sys.executable)
    raise SystemExit(0)
raise SystemExit(1)
PY
    return 0
  done
  return 1
}

python_bin="$(pick_python)" || {
  echo "Nemotron requires Python 3.10 or newer (Python 3.12 is recommended)." >&2
  exit 1
}

mkdir -p "$runtime_root"
mkdir -p "$temporary"
# CUDA wheels are several gigabytes. Keep pip's unpacking work area in the
# user-selected runtime root instead of a frequently quota-limited /tmp.
export TMPDIR="$temporary"
export PIP_NO_CACHE_DIR=1
if [[ ! -x "$venv/bin/python" ]]; then
  "$python_bin" -m venv "$venv"
fi

"$venv/bin/python" -m pip install --upgrade pip
# PyTorch's CUDA 12.8 wheels carry the matching CUDA user-mode libraries, so
# this works with a current NVIDIA driver without sudo or a system toolkit.
"$venv/bin/python" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
"$venv/bin/python" -m pip install --upgrade 'transformers>=5.14,<5.15' 'librosa>=0.11' 'numpy>=2.0'
"$venv/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch installed, but CUDA is unavailable. Check the NVIDIA driver before using Nemotron.")
print(f"Nemotron CUDA runtime ready: PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}")
PY
printf '%s\n' 'Nemotron CUDA runtime v1' > "$runtime_root/.voxide-nemotron-runtime-v1"
