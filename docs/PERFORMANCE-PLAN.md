# Voxide dictation performance plan

Date: 2026-07-21
Workspace: `/home/pedrocoutinho/dev/dictation-app`
Goal: close the perceived stop→text latency gap between Voxide and the user's
faster-whisper script (`~/.local/share/dictation/dictate.sh`), and reach a
FluidVoice-class experience on Linux (Niri/Wayland, RTX 4080 Laptop + Intel iGPU).

This document is the handoff-capable plan: diagnosis, toolchain strategy,
phased changes with file-level pointers, and verification protocol.
Update the **Progress log** at the bottom as work lands.

---

## 1. Diagnosis (verified facts)

### The user's fast path (baseline to beat)

`~/.local/share/dictation/dictate.sh` + `server.py`:

- faster-whisper (CTranslate2) **large-v3 on CUDA fp16**, warm model server over
  a unix socket, Silero VAD via `vad_filter=True`, beam_size 5 (cheap on GPU),
  ffmpeg capture at 16 kHz mono, `wtype` insertion.
- Stop→text for a typical utterance: sub-second to ~1 s.

### Voxide today (slow path)

Measured/verified on this machine (2026-07-21):

1. **Current binary is Vulkan-enabled** (built 2026-07-21 11:20,
   `libggml-vulkan.a` present in `target/release/build/whisper-rs-sys-8db0eb7bd2a70466`,
   final binary linked 15:29). Runtime activation is **unverified** —
   whisper.cpp falls back to CPU silently, and nothing logs the active backend.
2. Selected model: **large-v3-turbo** (1.6 GB fp16 ggml), engine `whisper`,
   language `en`, streaming preview ON, AI enhancement OFF.
3. **Preview/final contention**: the live preview
   (`spawn_live_whisper_preview`, `src-tauri/src/lib.rs:5269-5337`) re-decodes
   up to 20 s of audio with **beam_size 5** every 250 ms–2.5 s. The final pass
   waits out any in-flight preview behind `INFERENCE_LOCK`
   (`src-tauri/src/speech.rs:25,264`). If stop lands just after a preview tick
   starts, the final pass waits for a full multi-second decode.
4. **Beam search 5** for every decode (`speech.rs:293-296`); whisper.cpp
   defaults keep `n_threads = min(4, cores)` — painful on the CPU fallback.
5. **VAD context rebuilt per transcription** from the ~885 KB silero model
   (`speech.rs:224`), plus a CPU VAD pass over the whole recording.
6. **Fresh `create_state()` per dictation** (`speech.rs:288`) — compute/KV
   buffers re-allocated every time.
7. `pactl --format=json list sources` **subprocess on every capture start**
   (`src-tauri/src/audio.rs:236-243`).
8. **100 ms blocking sleep** in the paste path before clipboard restore
   (`src-tauri/src/typing.rs:91`); typing runs synchronously on the async
   worker (`lib.rs:5654`).
9. Trigger path: process spawn → unix socket → Tauri event → webview JS →
   Tauri command (two IPC hops; small but perceptible).

### FluidVoice reference (macOS, `~/dev/FluidVoice`)

Headline speed comes from ANE-accelerated **Parakeet** models (NVIDIA NeMo
models — natively CUDA-friendly) plus portable architecture:

- keep-loaded models + startup auto-load + `ensureReady` at record start,
- capture prewarm + warm standby,
- timer-driven growing-buffer preview with per-model cadence and backpressure
  (skip tick while busy / when inference > interval),
- dual fast/quality manager instances (preview vs final),
- true delta-feeding streaming engines (Parakeet EOU) so the final pass only
  processes the tail,
- clipboard-free chunked text insertion, stage-by-stage benchmarks
  (`ASR_BENCH`).

### Linux equivalents (verified to exist)

- **sherpa-onnx**: Rust API + official Tauri support; model
  **`sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`** (25 European languages,
  incl. PT + EN), onnxruntime CUDA provider, bundled Silero VAD. No EOU
  token-streaming Parakeet in sherpa-onnx → VAD-segmented pseudo-streaming is
  the equivalent. Parakeet is **transcribe-only** (no PT→EN translate task —
  Whisper stays for translate).
- whisper-rs 0.16 exposes everything needed: `set_n_threads`, its raw abort
  callback API (cancel in-flight decode), `set_single_segment`,
  `set_temperature(_inc)`, VAD context reuse. The convenience
  `set_abort_callback_safe` helper must not be used here: its callback storage
  is incompatible with this version's FFI trampoline.

---

## 2. CUDA toolchain strategy (no sudo)

Facts established 2026-07-21:

- `dnf` has **no** cuda-toolkit (Fedora 44 + RPM Fusion searched).
- pip wheels `nvidia-cuda-nvcc-cu12` (12.6–12.9) ship **only `ptxas`** — no
  nvcc driver. ❌
- NVIDIA **redist CDN** ships granular tarballs with the standard toolkit
  layout — no sudo, CMake-friendly. ✅ Latest: **CUDA 13.3.1**.
- Already present: driver 595.80 + `xorg-x11-drv-nvidia-cuda-libs` (runtime),
  gcc/g++ 16.1.1, `/usr/bin/cmake`, ninja.

Plan: use the existing Python venv at `~/.local/share/voxide-cuda/venv` for
the NVIDIA Python packages, and stage the nvcc-capable toolkit beside it at
`~/.local/share/voxide-cuda/toolkit` from redist tarballs
(`redistrib_13.3.1.json`). The venv packages are useful for runtime/support
libraries; the redist toolkit is still required because pip alone does not
provide a complete nvcc layout:

| Component    | Tarball |
| ------------ | ------- |
| cuda_nvcc    | `cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.3.73-archive.tar.xz` |
| cuda_cudart  | `cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.3.29-archive.tar.xz` |
| cuda_nvrtc   | `cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-13.3.33-archive.tar.xz` |
| libcublas    | `libcublas/linux-x86_64/libcublas-linux-x86_64-13.6.0.2-archive.tar.xz` |
| cccl         | `cccl/linux-x86_64/cccl-linux-x86_64-<ver>-archive.tar.xz` (see `/tmp/redist.json`) |

Base URL: `https://developer.download.nvidia.com/compute/cuda/redist/`.
Extract each `--strip-components=1` into the toolkit dir (archives contain a
top-level `cuda_nvcc-linux-x86_64-…-archive/` wrapper; some components nest
`targets/x86_64-linux/{include,lib}` which must be merged up).

Build env (needed for every CUDA build):

```sh
export CUDA_HOME="$HOME/.local/share/voxide-cuda/toolkit"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAToolkit_ROOT="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:${LD_LIBRARY_PATH:-}"
export CMAKE_CUDA_ARCHITECTURES=89        # RTX 4080 Laptop = Ada sm_89
export CMAKE_CUDA_FLAGS=--allow-unsupported-compiler  # Fedora gcc 16
npm exec tauri build -- --no-bundle --features cuda
```

The staged CUDA 13.3 redist uses `lib` rather than the `lib64` layout that
nvcc and whisper-rs expect. Create the compatibility link once:

```sh
ln -s lib "$CUDA_HOME/lib64"
ln -s "$HOME/.local/share/voxide-cuda/venv/lib/python3.14/site-packages/nvidia/cuda_runtime/lib/libculibos.a" "$CUDA_HOME/lib/libculibos.a"
```

`src-tauri/build.rs` discovers `CUDA_HOME`/`CUDAToolkit_ROOT`, passes that
directory to rust-lld, and embeds a Linux RUNPATH. The release binary can
therefore find the user-local cuBLAS/cuDART libraries when launched directly
from the Niri keybinding.

Runtime env for binaries built before the CUDA RUNPATH support (current CUDA
builds embed the toolkit path automatically):

```sh
export LD_LIBRARY_PATH="$HOME/.local/share/voxide-cuda/toolkit/lib64:${LD_LIBRARY_PATH:-}"
```

Risks / fallbacks:

- nvcc vs gcc 16: if rejected, add `--allow-unsupported-compiler` via
  `CMAKE_CUDA_FLAGS` (CMake initializes it from env). If headers genuinely
  break, try a compat gcc (`g++` from an older toolset) via
  `CMAKE_CUDA_HOST_COMPILER`.
- whisper-rs 0.16 `cuda` helper module may not compile (same issue as its
  `vulkan` module — see comment in `Cargo.toml:24-28`); fallback: retarget the
  cargo feature at `whisper-rs-sys/cuda`, mirroring the vulkan workaround.
- Keep the `vulkan` feature intact for portability; CUDA just wins on this
  machine. Do not enable both in one binary (backend selection ambiguity).

---

## 3. Phases

### Phase 0 — Instrumentation (do first)

Add stage timing so every later change is measured, not guessed.

- `speech.rs::transcribe_samples`: time INFERENCE_LOCK wait, VAD pass,
  `create_state`, `state.full` decode; return/forward timings.
- `lib.rs::stop_native_dictation` (~5455-5785): time post-processing and
  insertion; emit one summary line via `debug_log`, e.g.
  `Dictation timing (audio_s: 4.2, lock_wait_ms: 0, vad_ms: 12, decode_ms: 850, post_ms: 1, insert_ms: 40)`.
- `speech.rs::load_context`: log the chosen GPU device
  (`preferred_gpu_device()` result) and whether GPU was requested, so
  Vulkan/CUDA activation is visible in `~/.local/share/voxide/logs/voxide.log`.
- Add an ignored bench test (pattern of `speech.rs` `vad_debug`) that times a
  full transcribe of a real WAV so Vulkan vs CUDA can be compared headlessly:
  `VOXIDE_TEST_WAV=… cargo test --release --features cuda decode_bench -- --ignored --nocapture`.
  (Generate the WAV with espeak-ng, as done for earlier runtime tests.)

### Phase 1 — CUDA backend

1. Stage the redist toolkit (section 2) and verify `nvcc --version` runs.
2. `cargo build --release --features cuda` first (fast fail on nvcc issues),
   then the full `npm exec tauri build -- --no-bundle --features cuda`.
3. Verify via Phase 0 logs: active device = RTX 4080; decode timings on the
   bench WAV: CPU/Vulkan build vs CUDA build vs `dictate.sh` on the same audio.
4. Expected: large-v3-turbo decode of a few seconds of speech → ~100–300 ms.

### Phase 2 — Preview contention (biggest perceived-latency fix)

- Preview ticks take `INFERENCE_LOCK` with `try_lock` and skip the tick when
  busy (final pass never queues behind a preview start).
- Preview decode params: Beam 2 on a GPU (greedy on CPU fallback) + shorter
  window (~8 s instead of 20 s), VAD gating, and a consecutive-hypothesis
  common-prefix display so provisional words do not rewrite stable overlay
  text.
- On stop, cancel any in-flight preview: wire the existing
  `preview_generation` bump to a scoped raw whisper-rs abort callback, so the
  final pass waits milliseconds, not a full preview decode.

### Phase 3 — Decode-path efficiency

- Keep the Silero VAD context warm next to `CONTEXT_CACHE` (reload only when
  the model path changes) — `speech.rs:224`.
- Reuse one `WhisperState` per lane instead of `create_state()` per dictation
  (`speech.rs:288`); guard with the same inference lock.
- `parameters.set_n_threads(physical cores)` when the CPU fallback is active
  (GPU builds ignore it).
- Remove the 100 ms sleep (`typing.rs:91`): restore the clipboard from a
  detached background task after a short async delay; typing completion no
  longer waits for it.

### Phase 4 — Settings & models

- New **beam size** setting: Auto (beam 5 on GPU / greedy on CPU) / Greedy /
  Beam 2 / Beam 5 — settings UI + `FullParams` wiring.
- Add quantized model downloads (q5_0/q8_0 ggml, e.g. large-v3-turbo-q5_0):
  smaller, faster to load, faster CPU fallback (this is what FluidVoice ships).

### Phase 5 — Start latency / perceived speed

- Cache mic routing instead of spawning `pactl` per capture start
  (`audio.rs:236-243`); refresh on TTL or device-change.
- Pre-enumerate capture devices at startup so `AudioCapture::start` only
  builds + plays the stream (FluidVoice prewarm pattern, pragmatic cpal
  version).
- Fire the stop-cue sound at capture-stop (before decode), not after text.

### Phase 6 — Parakeet engine via sherpa-onnx (endgame)

- New `parakeet` voice engine alongside whisper/cloud; sherpa-onnx Rust API,
  onnxruntime CUDA provider (no nvcc needed; runtime libs already installed).
- Model: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (25 European languages,
  PT+EN) managed in the existing model-download UI.
- Port FluidVoice's architecture: VAD-segmented pseudo-streaming with a warm
  recognizer; decode completed VAD segments during recording; the final pass at
  stop only decodes the tail → near-zero stop→text latency; segment results
  drive the overlay preview.
- Caveats: transcribe-only (Whisper stays for PT→EN translate); no EOU
  token-streaming variant in sherpa-onnx.
- Re-assess priority after Phase 2 measurements — with CUDA Whisper the
  remaining win is streaming UX, not raw speed.

---

## 4. Verification protocol

Per phase:

1. `cargo fmt && cargo test` (in `src-tauri/`; CUDA env for CUDA builds) and
   `npm run build` must stay green.
2. Bench WAV (espeak-ng) through the ignored `decode_bench` test; record
   decode_ms for CPU/Vulkan vs CUDA builds in the progress log below.
3. Runtime (user at the desk): dictate via `Mod+Ctrl+D`, read the Phase 0
   `Dictation timing` lines in `~/.local/share/voxide/logs/voxide.log`, compare
   against `Mod+Shift+D` (the faster-whisper script) on the same utterances.

Environment notes specific to this machine:

- Niri bindings: `Mod+Ctrl+D` → `target/release/voxide --trigger dictate`,
  `Mod+Ctrl+Escape` → `--trigger cancel`.
- Models: `~/.local/share/voxide/models` (ggml-medium, ggml-large-v3-turbo,
  ggml-small, silero). Selected: large-v3-turbo.
- Log: `~/.local/share/voxide/logs/voxide.log`.
- `VOXIDE_GPU_DEVICE=<n>` overrides ggml GPU pick if the discrete GPU is not
  chosen.

---

## 5. Progress log

- 2026-07-21: Plan written. dnf cuda-toolkit confirmed absent; pip nvcc wheels
  confirmed to lack the nvcc driver; redist 13.3.1 tarballs chosen. pip venv at
  `~/.local/share/voxide-cuda/venv` has cu12 runtime wheels (useful for
  cuBLAS/cuDNN paths only — superseded by redist toolkit).
- 2026-07-21: Phase 0 landed. `WhisperTranscription` records inference-lock,
  VAD, warm-state, and decode timings; completed dictation writes a bounded
  timing summary to `voxide.log`; the selected GPU is logged as a requested
  backend; and `decode_bench_on_real_wav` is available as an ignored test.
- 2026-07-21: Phases 2–5 landed for the current Whisper path. Previews use an
  8-second VAD-gated window, Beam 2 on GPU (greedy on CPU fallback), skip
  while the inference lane is occupied, and abort when stop invalidates their
  generation. The overlay only advances words confirmed by consecutive
  snapshots. VAD and Whisper state are cached;
  CPU fallback uses physical-core decoding; clipboard restoration is detached;
  pactl sources have a short TTL cache and capture devices are prewarmed. The
  Voice Engine screen now exposes Auto/Greedy/Beam 2/Beam 5 and q5/q8
  large-v3-turbo downloads.
- 2026-07-21: Phase 1 verified with the venv-backed CUDA 13.3.1 toolkit.
  `cargo build --release --features cuda` and `npm exec tauri build --
  --no-bundle --features cuda` pass. The binary has an sm_89 `libggml-cuda.a`,
  RUNPATH to the user-local toolkit, and resolves cuBLAS/cuDART. On the RTX
  4080 Laptop, `decode_bench` on an 8.0 s espeak WAV using large-v3-turbo
  measured `vad_ms: 48`, `state_ms: 4`, and `decode_ms: 421`; whisper.cpp
  confirmed `using CUDA0 backend`.
- 2026-07-21: Phase 6 assessment: the venv now has ONNX Runtime GPU 1.27 and
  cuDNN 9, and reports `CUDAExecutionProvider`. The official Linux sherpa-onnx
  shared artifact ships a CPU-only `libonnxruntime.so`, so adding it directly
  would not satisfy the planned CUDA Parakeet engine. It needs its own GPU
  runtime packaging and VAD-segment state model, and is intentionally deferred
  rather than shipping a misleading CPU-only engine.
- 2026-07-21: Live preview regression fixed. The `whisper-rs` 0.16
  `set_abort_callback_safe` helper made every CUDA snapshot abort during
  encoder setup, while its errors were silently discarded. Voxide now uses a
  scoped raw callback backed by the preview generation, isolated warm state
  for preview/final passes, and logs emitted, empty, and skipped preview
  outcomes. Previews are VAD-gated and their display only advances words that
  recur in consecutive snapshots. The CUDA bench now verifies 1–8 s growing
  snapshots (228–296 ms) and cancellation of a stale generation.
- Status: Phases 0–5 complete for the local Whisper engine. Phase 6 remains a
  separate product integration: CUDA Whisper already reaches the raw decode
  target; Parakeet would now be pursued for VAD-segment streaming UX rather
  than speed alone.
