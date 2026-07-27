#!/usr/bin/env bash
# Runs the CUDA gates that CI cannot, because they need a GPU and there is no
# self-hosted runner attached (see the `cuda-linux-nvidia` job in
# .github/workflows/ci.yml, which is manual-dispatch only).
#
# Mirrors that job step for step, on the pinned toolchain, so passing here means
# the same thing it would have meant in CI:
#
#   fmt → dependency audit → tests → GPU inference fixtures → clippy → release build
#
# Usage:  scripts/check-cuda.sh [--quick]
#           --quick   skip the release build, which is the slow step
#
# Prerequisites, all user-local and set up once (see the README):
#   ~/.local/share/voxide-cuda/toolkit         CUDA toolkit (nvcc)
#   ~/.local/share/voxide-parakeet/runtime     sherpa-onnx GPU runtime
#   ~/.local/share/voxide-parakeet/venv        CUDA 12/cuDNN 9 libraries
set -euo pipefail

QUICK=0
[[ ${1:-} == "--quick" ]] && QUICK=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_TOOLKIT="$HOME/.local/share/voxide-cuda/toolkit"
PARAKEET_ROOT="$HOME/.local/share/voxide-parakeet"
# The pinned toolchain CI uses, so a lint that only fires on a newer clippy does
# not fail a check that CI would have passed.
TOOLCHAIN="+1.89"

export CUDA_HOME="$CUDA_TOOLKIT"
export CUDA_PATH="$CUDA_TOOLKIT"
export CUDAToolkit_ROOT="$CUDA_TOOLKIT"
export PATH="$CUDA_TOOLKIT/bin:$PATH"
export SHERPA_ONNX_LIB_DIR="$PARAKEET_ROOT/runtime/lib"
# build.rs reads this with env::split_paths, so PATH-style colon separation.
PARAKEET_CUDA_LIB_DIRS="$(printf '%s:' "$PARAKEET_ROOT"/venv/lib/python*/site-packages/nvidia/*/lib | sed 's/:$//')"
export PARAKEET_CUDA_LIB_DIRS
# nvcc refuses a host GCC newer than it knows about — nvcc 13.3 caps at GCC 15 —
# so CUDA compiler detection fails outright without this. GGML itself compiles
# fine under the newer GCC once allowed. Pinning the architecture also avoids
# compiling every other one; override for a different GPU.
export CMAKE_CUDA_FLAGS=--allow-unsupported-compiler
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"
# The GPU inference fixtures decode this model's own reference WAV. CI took the
# path from a repository variable; locally it defaults to the installed model.
export VOXIDE_PARAKEET_MODEL_DIR="${VOXIDE_PARAKEET_MODEL_DIR:-$HOME/.local/share/voxide/models/parakeet-tdt-0.6b-v3-int8}"

step() { printf '\n\033[1m── %s ──\033[0m\n' "$1"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$1" >&2; exit 1; }

step "Verifying the NVIDIA/CUDA contract"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || fail "No nvcc at $CUDA_HOME/bin/nvcc"
[[ -d "$SHERPA_ONNX_LIB_DIR" ]] || fail "No sherpa-onnx runtime at $SHERPA_ONNX_LIB_DIR"
[[ -n "$PARAKEET_CUDA_LIB_DIRS" ]] || fail "No CUDA library directories found under $PARAKEET_ROOT/venv"
[[ -f "$VOXIDE_PARAKEET_MODEL_DIR/test_wavs/en.wav" ]] \
    || fail "No Parakeet reference WAV at $VOXIDE_PARAKEET_MODEL_DIR/test_wavs/en.wav — download the model, or point VOXIDE_PARAKEET_MODEL_DIR at one"
"$CUDA_HOME/bin/nvcc" --version | tail -2 | head -1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

cd "$REPO_ROOT/src-tauri"

step "Checking formatting"
cargo $TOOLCHAIN fmt --check

step "Auditing the resolved CUDA dependency graph"
cargo $TOOLCHAIN tree --locked --features cuda --edges normal,build >/dev/null

step "Running CUDA lifecycle tests"
cargo $TOOLCHAIN test --lib --features cuda

step "Running CUDA model inference fixtures"
# Ignored by default because they load real models onto the GPU.
cargo $TOOLCHAIN test --lib --features cuda cuda_model -- --ignored

step "Linting the CUDA target"
cargo $TOOLCHAIN clippy --lib --features cuda -- -D warnings

if (( QUICK )); then
    printf '\n\033[1mSkipped the release build (--quick).\033[0m\n'
else
    step "Building the CUDA release target"
    # `custom-protocol` embeds the frontend, so dist/ has to exist first —
    # a bare cargo build would otherwise ship a stale UI or fail outright.
    (cd "$REPO_ROOT" && npm run build)
    cargo $TOOLCHAIN build --release --features cuda,custom-protocol
fi

printf '\n\033[32mAll CUDA gates passed.\033[0m\n'
