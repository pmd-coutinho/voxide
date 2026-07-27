# Roadmap from the voxtype comparison

Tracks the ranked list produced by comparing Voxide against
[voxtype](https://github.com/peteonrails/voxtype). Items 1–3 shipped; 4 is
partly built; 5–6 are planned with their blockers established rather than
guessed, because in every item so far the expensive surprise was found by
checking before coding, not after.

| # | Item | State |
| - | ---- | ----- |
| 1 | libei output leg + ordered insertion chain | **shipped** |
| 2 | Held-modifier guard | **shipped** |
| 3 | Compositor keybinding writer | **shipped** |
| 4 | Cohere Transcribe on CPU | **engine works & verified** — see `COHERE_ENGINE.md` |
| 5 | Overlay on layer-shell | planned; blocker identified below |
| 6a | Eager chunked transcription | planned |
| 6b | GTCRN speech enhancement | planned; model located |
| 6c | GPU isolation option | planned |
| 6d | Packaging: tag-triggered release | **shipped** |

Shipped items 1–3 also fixed three things the comparison did not predict: enigo
fans every keystroke out to *all* connected Linux backends (so text was
synthesized twice on any Wayland session running Xwayland), niri rejects its whole
config on a duplicate keybind (so a fixed chord would have broken the desktop),
and the Windows build could not save its database at all.

## 5. Overlay on layer-shell

The overlay is currently a Tauri webview window. That is why it needed the
logical-pixel sizing fix, and why `renderOverlay` cannot use `innerHTML` per frame
without WebKitGTK refusing to repaint. A `zwlr_layer_shell_v1` surface is the
right primitive: it cannot steal focus, it positions itself against the output
rather than guessing, and it has no DPI semantics to get wrong.

**Blocker, established rather than assumed: it has to be a separate process.**
Tauri here links `webkit2gtk-4.1`, which is **GTK 3**. GTK 3 and GTK 4 cannot
coexist in one process, so `gtk4-layer-shell` cannot be loaded into the app.
Neither `gtk4` nor `gtk4-layer-shell` is even installed on the development host
(`pkg-config` finds neither), so this also adds build and packaging dependencies.
This is why voxtype ships its OSD as separate binaries rather than in-process.

Two viable shapes, both a second process:

- **`gtk4` 0.11 + `gtk4-layer-shell` 0.8** — least code, but pulls the whole GTK 4
  stack in as a build and runtime dependency alongside GTK 3.
- **`smithay-client-toolkit` 0.21 + `wgpu`** — no GTK at all and no new system
  packages, at the cost of drawing the waveform by hand.

SCTK is the better fit: the overlay is a level meter and a line of text, the
drawing is not the hard part, and avoiding a second toolkit keeps the packaging
story from doubling. Either way the work splits into a small `voxide-overlay`
binary, an IPC channel (the existing `$XDG_RUNTIME_DIR` trigger socket already
proves the pattern), and deleting the webview overlay path.

Do this before item 6b: it removes a whole class of bug rather than adding a
capability, and the overlay is on the latency path.

## 6a. Eager chunked transcription

Whisper currently waits for the recording to stop before decoding. Chunk during
capture with a small overlap, decode chunks concurrently, and stitch with
deduplication at the boundaries — voxtype's `eager.rs` is the reference.

Two constraints specific to this codebase. `INFERENCE_LOCK` in `speech.rs`
serialises inference because concurrent decodes on one `WhisperContext` corrupt
results, so chunks queue rather than truly parallelise unless a second context is
built — which costs the memory the warm-context cache exists to avoid. And VAD is
deliberately a gate on the whole utterance, not a segmenter: mid-utterance
segmentation was tried and measurably lost words. Chunk boundaries must therefore
not be VAD-derived.

Because of the inference lock the honest win here is *perceived* latency on slow
CPUs, not throughput. Worth measuring before building: instrument the existing
`decode_ms` against wall-clock capture and see how much is actually recoverable.

## 6b. GTCRN speech enhancement

A 48k-parameter, ~520 KB ONNX denoiser in front of the ASR, cleaning noise and
speaker bleed-through. Useful for plain dictation on a bad microphone, not just
for meeting capture.

ONNX exports are on the Hub — `bitsydarel/gtcrn-onnx` and the sherpa-onnx
variants. The STFT front end already exists in `cohere_fbank.rs` and can be
generalised: GTCRN wants 512-point FFT with a 256 hop and a sqrt-Hann window,
against Cohere's 512/160 and symmetric Hann.

Gate it behind a setting that defaults **off**, and verify on real recordings
before recommending it — a denoiser that removes phonemes along with noise makes
transcription worse, and the failure is invisible without listening.

## 6c. GPU isolation option

whisper.cpp does not return GPU memory after an in-process transcription, so VRAM
grows across a long session. voxtype's answer is an opt-in subprocess that
transcribes and exits, trading model-load latency for bounded memory.

Voxide already has the harder half of this: `nemotron.rs` runs a child process
and streams PCM to it over stdin, and `pronunciation.rs` does the same for a
sidecar. The pattern to copy is local. What needs deciding is the default —
voxtype's guidance is explicitly not to assume users want isolation, because
keeping the model warm is why the second dictation is fast.

Measure first: log VRAM across a long session and confirm the growth is real on
this whisper-rs version before adding a mode.

## 6d. Packaging

`tauri build` already produces deb, rpm and AppImage. The gap versus voxtype is
distribution rather than formats: AUR, Nix, Homebrew, and signed prebuilt binaries
from a reproducible pipeline.

`.github/workflows/release.yml` now covers the first half of that: pushing a
`v*` tag builds the portable Linux, macOS and Windows binaries on the toolchain CI
gates against, renames them per platform, and attaches them to a GitHub release
with a `SHA256SUMS.txt`. `workflow_dispatch` runs the builds but stops before
publishing, so it doubles as a dry run against a branch.

Deliberately absent: the CUDA build. It needs a self-hosted GPU runner and there
is none, so a release must not claim GPU support it was never built with — build
it locally with `scripts/check-cuda.sh` and attach it by hand.

Still open: AUR, Nix and Homebrew, and the deb/rpm/AppImage bundles (which need
`libappindicator-gtk3-devel` and `librsvg2-devel` for pkg-config, plus `NO_STRIP=1`
on distributions with `.relr.dyn` sections).

Two host-specific facts worth keeping: AppImage bundling needs `NO_STRIP=1` on
distributions with `.relr.dyn` sections, and the bundler wants
`libappindicator-gtk3-devel` and `librsvg2-devel` for pkg-config.
