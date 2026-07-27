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
| 4 | Cohere Transcribe on CPU | **5 of 7** — see `COHERE_ENGINE.md` |
| 5 | Overlay on layer-shell | planned; blocker identified below |
| 6a | Eager chunked transcription | **declined** — already covered, and its mechanism regresses here |
| 6b | GTCRN speech enhancement | **built & verified**, not yet wired |
| 6c | Bounded GPU memory on idle | **shipped** (different mechanism) |
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

## 6a. Eager chunked transcription — declined, with reasons

Voxtype's `eager.rs` chunks audio during recording, decodes chunks in parallel and
stitches them at the boundaries, so decode work overlaps capture instead of
starting at the stop. Recommending it for Voxide was a mistake in the original
comparison: it was drawn from voxtype's module list without checking what Voxide
already does.

**The latency win already exists here.** `spawn_live_whisper_preview` decodes
during capture — a rolling 8-second window every 600 ms, paced by how long
transcription actually takes on the machine. Voxtype needs `eager.rs` because it
has no live preview; Voxide has had one all along. What overlapping decode with
capture buys, this codebase already banks.

**And the mechanism eager mode needs is known to regress here.** Chunk-and-stitch
requires cutting mid-utterance and reassembling. That was tried in this codebase
and *measurably lost words*, which is why VAD is a gate on the whole utterance
rather than a segmenter and why the final pass decodes the complete buffer. Adding
eager chunking would reintroduce a regression that was deliberately removed.

`INFERENCE_LOCK` independently caps the gain: concurrent decodes on one
`WhisperContext` corrupt results, so chunks would queue rather than parallelise
unless a second context were built — costing exactly the memory the warm-context
cache and the new idle timeout exist to bound.

The one genuine remaining gap is narrower than the item as written: the final pass
re-decodes audio the preview has already seen, so preview work is thrown away. If
that is worth reclaiming it should be framed as *reusing the preview's output*,
not as chunking, and it needs the measurement below first — instrument `decode_ms`
against wall-clock capture and see how much is actually recoverable.

## 6b. GTCRN speech enhancement

A 48k-parameter, ~520 KB ONNX denoiser in front of the ASR, cleaning noise and
speaker bleed-through. Useful for plain dictation on a bad microphone, not just
for meeting capture.

`denoise.rs` implements it behind the `denoise` feature. The model is streaming
with fully static shapes: one STFT frame in (`[1, 257, 1, 2]`, interleaved real
and imaginary), one enhanced frame out, plus three recurrent caches threaded from
call to call. Caches start zeroed per utterance so one recording cannot leak into
the next.

STFT is 512-point with a 256 hop and a **sqrt-Hann** window used for both analysis
and synthesis: squared it is a periodic Hann, which sums to exactly 1 at 50%
overlap, so overlap-add is unity gain with no correction pass.

Verified objectively rather than by ear: a 300 Hz tone under a 4 Hz envelope plus
deterministic broadband noise goes in at **6.01 dB SNR and comes out at
10.47 dB — a 4.46 dB gain**. That one number validates the whole chain, because
the window, hop, scaling, conjugate mirror and cache threading all have to be
correct for noise to fall instead of the signal garbling.

`ort` uses `load-dynamic`, so onnxruntime is dlopened at runtime instead of linked
at build time — the build host needs no ONNX Runtime, and `ORT_DYLIB_PATH` can
point at the copy the Parakeet runtime already ships.

**Still to do:** wire it into the capture path behind a setting that defaults
off, and validate on real recordings. An SNR gain on synthetic noise is not proof
it preserves phonemes, and that failure is invisible without listening.

## 6c. Bounded GPU memory on idle — shipped, not as a subprocess

The problem voxtype solves with GPU isolation is that whisper.cpp holds GPU memory
for as long as a context is alive, so an app left open after one dictation keeps a
multi-gigabyte model resident. Its answer is an opt-in child process that
transcribes and exits.

Voxide solves the same problem by releasing the warm caches after
`MODEL_IDLE_TIMEOUT` (5 minutes) of no use, swept several times per window from a
background task. Chosen over the subprocess because it costs nothing on the case
that actually matters — dictations that follow one another keep the model warm and
reload nothing — whereas isolation pays a model load every time. voxtype's own
guidance is not to assume users want isolation for exactly that reason, which makes
an idle timeout the better default rather than a second mode to explain.

`release_idle_models` takes `INFERENCE_LOCK` before dropping anything, since
freeing a context underneath a running decode would be a use-after-free; if a
decode holds the lock it declines and the next sweep retries. States are dropped
before the context they borrow from.

Not covered: a single very long session that never idles, if whisper-rs leaks
across decodes on one context. Subprocess isolation is the fallback, and the
child-process pattern already exists in `nemotron.rs` and `pronunciation.rs`.

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
