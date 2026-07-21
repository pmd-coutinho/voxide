#!/usr/bin/env bash
# Downloads the official shared sherpa-onnx CUDA runtime and NVIDIA's CUDA 12
# / cuDNN 9 wheel libraries without requiring sudo. It is intentionally kept
# outside the application: end users only download the ASR model from Voxide's
# Voice Engine screen.
set -euo pipefail

runtime_root="${VOXIDE_PARAKEET_RUNTIME_ROOT:-$HOME/.local/share/voxide-parakeet}"
runtime_version="1.13.4"
runtime_archive="sherpa-onnx-v${runtime_version}-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2"
runtime_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${runtime_version}/${runtime_archive}"
downloads="$runtime_root/downloads"
runtime="$runtime_root/runtime"
venv="$runtime_root/venv"
cuda_libs="$runtime_root/cuda-libs"

mkdir -p "$downloads"
if [[ -e "$runtime" || -e "$cuda_libs" ]]; then
  echo "Parakeet runtime already exists under $runtime_root; move it aside before running this setup again." >&2
  exit 1
fi
curl --fail --location --retry 3 --continue-at - \
  --output "$downloads/$runtime_archive" "$runtime_url"

staging="$(mktemp -d "$runtime_root/.runtime-install.XXXXXX")"
tar -xjf "$downloads/$runtime_archive" -C "$staging" --strip-components=1
mv "$staging" "$runtime"

python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade \
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-nvjitlink-cu12

site_packages="$("$venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$cuda_libs"
for directory in "$site_packages"/nvidia/*/lib; do
  [[ -d "$directory" ]] || continue
  for library in "$directory"/*.so*; do
    [[ -e "$library" ]] || continue
    ln -sfn "$library" "$cuda_libs/$(basename "$library")"
  done
done

cat <<EOF

Parakeet CUDA runtime installed in: $runtime

Use these values for a CUDA build:
  export SHERPA_ONNX_LIB_DIR="$runtime/lib"
  export PARAKEET_CUDA_LIB_DIRS="$cuda_libs"
EOF
