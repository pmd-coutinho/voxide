# Voxide production-readiness plan

Date: 2026-07-22  
Comparison baseline: Voxide at `/home/pedrocoutinho/dev/dictation-app` and
FluidVoice at `/home/pedrocoutinho/dev/FluidVoice`

## 1. Purpose

This document turns the FluidVoice comparison into an implementation plan for
making Voxide reliable enough for broad day-to-day use and eventual public
distribution.

Voxide already has substantial product breadth: local and cloud speech engines,
global shortcuts, live preview, AI post-processing, prompt profiles, rewrite and
command modes, history, an audio archive, file transcription, a loopback API,
dictionary support, update checks, analytics controls, and cross-platform
packaging work. The largest remaining gap is not the number of features. It is
the reliability of the foundations beneath them:

- lossless and correctly timed microphone capture;
- explicit ownership of recording and inference sessions;
- consistent preview/final semantics;
- model and native-runtime lifecycle management;
- fault isolation and automatic recovery;
- integration testing with real audio;
- reproducible, signed distribution;
- enough diagnostics to explain a bad transcription without logging private
  content.

The guiding decision is therefore:

> Stabilize capture, lifecycle, testing, and distribution before adding more
> production voice engines. Keep experimental engines available, but do not let
> them define the default experience until they pass objective release gates.

## 2. Executive priorities

| Priority | Workstream | Primary outcome |
| --- | --- | --- |
| P0 | Canonical audio capture pipeline | No silent sample loss, clock drift, or ambiguous recording boundaries |
| P0 | ASR engine contract and session coordinator | One owner for prepare/start/preview/final/cancel/failure transitions |
| P0 | Audio and dictation regression suite | Preview, pause, final, cancellation, and device failures become reproducible |
| P1 | CUDA/model component manager | Reproducible installation without build-machine paths or fragile setup scripts |
| P1 | Structured diagnostics and recovery | Every session can explain capture, preview, inference, and delivery health |
| P1 | True-streaming Parakeet investigation | A low-latency preview path that does not repeatedly decode growing audio |
| P1 | Signed CI release pipeline | Tested CPU/GPU artifacts with known dependencies and provenance |
| P2 | Automatic updater, rollback, and data recovery | Safe upgrades and recovery from corrupt settings or application releases |
| P2 | Pronunciation and edit-driven learning | Higher personal accuracy once the core pipeline is dependable |

## 3. What the comparison shows

### 3.1 Areas where Voxide is already strong

Voxide should preserve these strengths rather than attempting a line-for-line
FluidVoice port:

- Cross-platform Rust/Tauri architecture instead of a macOS-only product.
- Local Whisper on CPU, Vulkan, or CUDA.
- CUDA Parakeet TDT on Linux/NVIDIA.
- A functional Nemotron CUDA experiment retained behind an explicit maturity
  label.
- OpenAI-compatible cloud transcription.
- Prompt routing, rewrite, and reviewed command execution.
- Searchable history, optional WAV archive, file transcription, and local API.
- Keyring-backed secrets and intentionally content-free diagnostic logs.
- Atomic temporary-file replacement for the main JSON database.
- Download-time size/content checks and staging for several model paths.
- Wayland portal and compositor-trigger support.
- Direct insertion with clipboard fallback and clipboard restoration.

### 3.2 FluidVoice practices worth adopting

FluidVoice's strongest lessons are operational rather than visual:

- A provider contract that separates readiness, streaming, final inference,
  file inference, cancellation, and cache management.
- Cached providers with single-flight preparation and orderly retirement.
- Audio prewarming and bounded warm standby.
- A direct audio backend with a compatibility fallback.
- Session timestamps, exact start/stop trimming, stateful resampling, and
  capture-duration validation.
- Route-change and device-disconnection recovery.
- Per-model preview cadence and backpressure.
- Separate preview and final managers where the engine benefits from them.
- True delta-fed streaming for Parakeet Flash and Nemotron.
- Stage-level `ASR_BENCH` diagnostics.
- Model artifact completeness and corrupt-proxy-response validation.
- Checked-in audio fixtures and CI build/test enforcement.
- Schema-versioned backups.
- Signed update validation and rollback backups.
- Automatic correction tracking and pronunciation customization.

### 3.3 FluidVoice patterns not to copy blindly

FluidVoice is a useful behavioral reference, not an architectural ideal.
Voxide should not copy:

- Very large coordinator, settings, and UI files. FluidVoice itself contains
  several files over 100 KB; Voxide's `src-tauri/src/lib.rs` and `src/main.ts`
  already need decomposition.
- macOS-only assumptions such as CoreML/ANE availability, Accessibility APIs,
  the notch UI, or application bundle replacement mechanics.
- Branch-based production dependencies without immutable revision pinning.
- Subjective hard-coded model quality ratings as substitutes for measured
  evaluation.
- More string-diff heuristics as the main answer to unstable ASR preview.
- Every experimental engine simply because an implementation exists.

## 4. Critical finding: the current audio callback can lose samples

The current CPAL callback appends samples with `try_lock`:

```rust
if let Ok(mut samples) = destination.try_lock() {
    samples.extend(data.iter().copied().map(f32::from_sample));
}
```

If another thread holds the buffer lock, the callback discards that complete
audio block without reporting an error or incrementing a counter. Preview
snapshots clone the growing buffer while holding the same mutex. The clone cost
increases with recording length, so longer recordings increase the opportunity
for callback contention and sample loss.

Current preview paths also repeatedly:

1. clone device-native interleaved samples;
2. downmix the clone;
3. resample the clone from its beginning;
4. pass the result to an engine;
5. repeat the work on the next tick.

Consequences can include:

- gaps or discontinuities in the recognizer input;
- preview hypotheses that become less stable over time;
- a preview that stops refreshing under load;
- misleading wall-clock duration versus captured-audio duration;
- duplicated CPU and memory bandwidth work;
- no diagnostic evidence that samples were lost.

This is the first defect the production-hardening work should address.

## 5. Target architecture

```text
                       device events
                            │
                            ▼
┌───────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ CPAL callback │───►│ bounded raw ring │───►│ capture worker      │
│ copy only     │    │ seq + timestamps │    │ downmix + resample  │
└───────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                        │ 16 kHz mono
                                                        ▼
                                             ┌─────────────────────┐
                                             │ canonical timeline  │
                                             │ cursor + health     │
                                             └───────┬───────┬─────┘
                                                     │       │
                                      incremental    │       │ snapshot/final
                                                     ▼       ▼
                                             ┌─────────────────────┐
                                             │ ASR session         │
                                             │ coordinator         │
                                             └──────────┬──────────┘
                                                        │
                          ┌─────────────────────────────┼────────────────────┐
                          ▼                             ▼                    ▼
                    partial event                final result          health trace
                          │                             │                    │
                          ▼                             ▼                    ▼
                       overlay              formatting/insertion      diagnostics
```

### 5.1 Ownership rules

- The audio callback owns no growing `Vec`, UI object, model, or async runtime
  handle.
- The capture worker is the only component that converts device audio into the
  canonical 16 kHz mono timeline.
- One dictation coordinator owns the active session and its cancellation token.
- One engine session owns engine-specific preview/final state.
- The overlay displays events but never decides transcript correctness.
- Only the final result enters deterministic cleanup, AI processing, history,
  clipboard, or insertion.
- Downloads and engine selection are separate operations.
- A model is selectable only when its engine reports a passing readiness check.

### 5.2 Dictation state machine

```text
Idle
 ├─ prepare request ─► Preparing ─► Idle
 └─ record request ──► Starting ──► Recording ──► Finalizing
                           │             │              │
                           │             ├─ cancel ─────┤
                           │             └─ failure ────┤
                           ▼                            ▼
                         Failed ◄──────────────────── Cancelling
                           │                            │
                           └──────── recovery ──────────┘

Finalizing ─► PostProcessing ─► Delivering ─► Idle
     │               │              │
     └────────────── failure ────────┴──────► Failed ─► Idle
```

Required invariants:

- At most one active microphone capture.
- At most one active ASR session.
- A new recording cannot reuse state from an older generation.
- Cancel waits for, or safely detaches, every session-owned task.
- Finalization takes priority over preview.
- Engine switching cannot occur halfway through a session.
- A failed engine preparation cannot change the selected engine.
- A stale task cannot emit overlay text after its session is complete.
- History is written only after a final result exists.
- Audio archive files are either referenced by history or cleaned up.

### 5.3 Proposed Rust engine contract

The exact signatures can evolve, but the behavioral contract should resemble:

```rust
#[derive(Clone, Debug)]
pub struct EngineCapabilities {
    pub availability: EngineAvailability,
    pub maturity: EngineMaturity,
    pub preview_mode: PreviewMode,
    pub final_mode: FinalMode,
    pub supports_files: bool,
    pub supports_translation: bool,
    pub supports_vocabulary: bool,
    pub supported_languages: LanguageSupport,
    pub minimum_preview_samples: usize,
    pub preferred_preview_interval: Duration,
}

pub enum PreviewMode {
    None,
    FullSnapshot,
    RollingSnapshot,
    Incremental,
}

pub enum FinalMode {
    IndependentFullDecode,
    FlushActiveStream,
}

#[async_trait]
pub trait SpeechEngine: Send {
    fn id(&self) -> EngineId;
    fn capabilities(&self) -> &EngineCapabilities;
    async fn prepare(&mut self, progress: ProgressSink) -> Result<(), EngineError>;
    async fn health(&mut self) -> EngineHealth;
    async fn begin_session(&mut self, config: SessionConfig) -> Result<(), EngineError>;
    async fn push_audio(&mut self, chunk: AudioChunk<'_>) -> Result<(), EngineError>;
    async fn partial(&mut self) -> Result<Option<PartialTranscript>, EngineError>;
    async fn finish(&mut self) -> Result<FinalTranscript, EngineError>;
    async fn cancel(&mut self);
    async fn unload(&mut self);
}
```

Offline engines can implement `push_audio` by retaining a canonical snapshot
cursor and only decoding when the coordinator requests `partial`. Streaming
engines consume each new chunk exactly once. The coordinator does not need to
know which engine uses which mechanism.

Errors should be typed rather than reduced immediately to strings:

```rust
pub enum EngineErrorKind {
    Unavailable,
    ModelMissing,
    RuntimeMissing,
    IncompatibleRuntime,
    DownloadCorrupt,
    DeviceLost,
    CaptureOverflow,
    Timeout,
    Cancelled,
    OutOfMemory,
    SidecarExited,
    ProtocolViolation,
    InferenceFailed,
}
```

User-facing text can then be derived centrally while diagnostics retain a safe,
structured category.

## 6. Phase 0: establish a measurable baseline

### 6.1 Work

- Add a session identifier to every capture and inference log entry.
- Emit structured, content-free stage timing for all engines, not only Whisper.
- Capture the selected engine, model/runtime version, device label hash or safe
  stable identifier, sample rate, channel count, and backend.
- Record wall duration and captured-sample duration.
- Count callback blocks, raw frames, accepted frames, dropped frames, ring
  high-water mark, discontinuities, and route changes.
- Record time to first audio, first preview, latest preview, stop request, final
  result, post-processing completion, and insertion completion.
- Record preview tick count, skip reason, queue delay, inference duration, audio
  duration, and real-time factor.
- Add a user-triggered diagnostic export that excludes transcript text, prompts,
  command contents, file paths, clipboard contents, API keys, and raw audio.
- Create a fixed local evaluation corpus with consented or synthetic WAV files.
- Save current CPU/CUDA performance and result snapshots as a baseline artifact.

### 6.2 Suggested session trace

```json
{
  "schema": 1,
  "session_id": "random-id",
  "engine": "parakeet-tdt-v3-int8",
  "engine_maturity": "stable",
  "runtime_version": "sherpa-onnx-1.13.4-cuda12-cudnn9",
  "capture": {
    "sample_rate": 48000,
    "channels": 2,
    "callback_count": 721,
    "input_frames": 346080,
    "canonical_frames": 115360,
    "dropped_frames": 0,
    "discontinuities": 0,
    "ring_high_water_frames": 4096,
    "wall_duration_ms": 7217,
    "canonical_duration_ms": 7210
  },
  "preview": {
    "ticks": 11,
    "completed": 8,
    "skipped_busy": 3,
    "first_partial_ms": 1184,
    "last_partial_ms": 6642
  },
  "final": {
    "queue_ms": 0,
    "inference_ms": 284,
    "rtf": 0.039
  },
  "delivery": {
    "postprocess_ms": 2,
    "insert_ms": 11
  },
  "outcome": "success"
}
```

### 6.3 Acceptance criteria

- A failed or slow session can be assigned to capture, scheduling, inference,
  post-processing, or insertion without transcript logging.
- Captured duration differs from wall duration by no more than a documented
  tolerance during healthy capture.
- Every skipped preview has a reason.
- Every engine reports an implementation/runtime version.
- Diagnostic export is reviewed for private-content leakage.

## 7. Phase 1: canonical audio capture pipeline

### 7.1 Real-time callback rules

The callback may:

- copy a bounded packet into preallocated/ring storage;
- attach a monotonically increasing sequence and device timestamp;
- update lock-free counters;
- signal a worker.

The callback must not:

- block on a mutex;
- grow a vector;
- allocate for every sample or packet;
- clone the recording;
- resample or perform expensive downmixing;
- emit Tauri events;
- log to disk;
- call a model;
- silently discard data.

If the ring is full, the callback must increment an overflow counter. The
session should be marked degraded and the user should receive an actionable
error if the loss exceeds a strict threshold.

### 7.2 Capture worker

The worker should:

- drain device packets in sequence order;
- convert every supported CPAL format to `f32`;
- downmix using a documented channel policy;
- resample once into stateful 16 kHz mono;
- preserve fractional resampling phase across packet boundaries;
- reset phase only on an identified discontinuity or format change;
- append canonical samples to chunked or segmented storage;
- publish bounded audio-level updates from canonical samples;
- notify the coordinator when a new canonical range is ready;
- preserve enough metadata to validate clock and continuity.

Avoid a single indefinitely growing contiguous allocation. Reasonable designs
include fixed-size sample pages, a segmented vector, or a canonical ring plus a
session archive. Snapshot access should copy only the range an engine needs and
should never block the device callback.

### 7.3 Session boundaries

- Record the requested start timestamp before activating capture.
- Reject preroll that predates the session unless an explicit preroll feature
  is later introduced.
- On stop, freeze an exact boundary before stopping the device.
- Drain already-acquired packets only up to that boundary.
- Ensure the last phoneme is not lost by stopping the worker prematurely.
- Prevent late packets from entering the next session.
- Use session generation plus timestamps, not only elapsed `Instant`, to decide
  ownership.

### 7.4 Device lifecycle

- Detect stream errors and forward them into coordinator state.
- Listen for device removal/default-route changes where the platform permits.
- If using a manual device and it disappears, either recover it within a
  bounded window or stop with a clear error.
- If using system default, attempt a controlled reopen on the new default.
- Never continue displaying `Listening` while the capture buffer is no longer
  growing.
- Keep microphone selection behavior consistent for Whisper, Parakeet,
  Nemotron, cloud, and system engines.
- Record the actual resolved source, not only the configured label.

### 7.5 Tests

- All supported integer and float input formats.
- Mono, stereo, and multichannel downmix.
- 8/16/32/44.1/48/96 kHz inputs.
- Stateful resampling across randomly sized packets.
- Exact expected output length within one-sample tolerance.
- No discontinuity at packet boundaries.
- Deliberate ring overflow is counted and surfaced.
- Concurrent preview snapshots cannot lose callback data.
- Start/stop in the middle of packets.
- Rapid stop/start does not mix generations.
- Device error while recording transitions to failure once.
- Ten-minute recording memory growth stays within the chosen design bound.

### 7.6 Acceptance criteria

- Zero silently dropped packets in the code path.
- Normal preview activity cannot block capture.
- Healthy 44.1 and 48 kHz recordings remain synchronized with wall time over at
  least 30 minutes.
- The canonical final buffer is byte-for-byte deterministic for the same packet
  fixture.
- Capture error and overflow are visible in diagnostics and UI.

## 8. Phase 2: ASR provider contract and coordinator

### 8.1 Decompose backend responsibilities

Suggested modules:

```text
src-tauri/src/asr/
  mod.rs
  capabilities.rs
  coordinator.rs
  error.rs
  registry.rs
  session.rs
  transcript.rs
  engines/
    whisper.rs
    parakeet_tdt.rs
    parakeet_flash.rs
    nemotron.rs
    apple_speech.rs
    cloud.rs

src-tauri/src/capture/
  mod.rs
  device.rs
  packet.rs
  pipeline.rs
  resampler.rs
  ring.rs
  timeline.rs

src-tauri/src/components/
  manifest.rs
  downloader.rs
  installer.rs
  runtime.rs
  verification.rs
```

The exact layout is flexible; the important goal is to move engine and session
logic out of `lib.rs` without a behavior-changing big-bang rewrite.

### 8.2 Coordinator behavior

- Resolve a complete immutable session configuration at record start.
- Validate the selected engine, model, runtime, language, device, and feature
  compatibility before opening the microphone when practical.
- Preserve the focused application/window context for the session.
- Start capture independently of optional media pause so audio is not delayed.
- Schedule preview based on capabilities.
- Apply backpressure: at most one preview inference per engine session.
- Give finalization priority and cancel/abort stale preview work.
- Wait for session-owned streaming work before resetting engine state.
- Guarantee exactly one terminal result: success, cancelled, or failed.
- Make cleanup idempotent.
- Keep the last known-good engine selection if preparation or download fails.

Startup failures now use generation-guarded rollback: an engine reservation,
microphone-start, or preview-setup error clears the admitted coordinator
session and preserved application context only if it still owns that generation.
The regression tests also prove an old failure cannot roll back a newer session.

### 8.3 Model readiness

- Cache loaded engines deliberately, with a documented memory policy.
- Single-flight concurrent `prepare` requests for the same component.
- Cancel and drain an old provider before replacing it.
- Perform startup preload only for installed, selected, stable engines and only
  when resource policy permits.
- Expose `unloaded`, `loading`, `ready`, `degraded`, and `failed` states.
- Include a manual engine self-test in Settings.
- On memory pressure, evict experimental or inactive engines before the selected
  stable engine.

### 8.4 Capability-driven UI

The UI should derive controls from engine capabilities instead of duplicating
engine conditions:

- availability and reason;
- Stable/Beta/Experimental maturity;
- installed/runtime-ready/model-ready/loaded health;
- input support;
- language selection or auto-detection;
- transcription versus translation;
- preview mode and expected latency;
- custom-vocabulary support;
- file transcription support;
- hardware/runtime requirements;
- approximate download, disk, RAM, and VRAM needs.

Engine installation must not implicitly select the engine. Selection should be
an explicit user action that succeeds only after validation, with rollback to
the previous selection on failure.

### 8.5 Acceptance criteria

- Adding an engine does not require another top-level match branch in the
  dictation lifecycle.
- Engine selection, input selection, and readiness cannot become inconsistent.
- Cancellation tests prove that stale tasks cannot emit preview or final text.
- Repeated rapid recordings do not overlap finalization or corrupt engine state.
- All engines pass the same fake-audio lifecycle contract suite.

## 9. Phase 3: define correct preview semantics

### 9.1 Core contract

- Preview is display-only and always provisional.
- Preview never enters history, insertion, AI processing, or clipboard output.
- Final transcription is authoritative.
- Stale-session preview events are rejected by session ID.
- Silence does not create unrelated appended text.
- Preview failure never prevents final transcription.
- Finalization cannot wait behind a newly admitted preview.
- UI labels and styling should make provisional state understandable without
  being visually distracting.

### 9.2 Offline snapshot engines

Parakeet TDT and similar engines decode growing audio snapshots rather than a
native stream. For these engines:

- Start only after the model-specific minimum audio duration.
- Never queue multiple snapshots.
- Pace the next snapshot from actual inference cost.
- Stop snapshot admission immediately when finalization begins.
- Decode canonical 16 kHz samples; do not resample the full native recording on
  every tick.
- Prefer the complete growing capture while it remains within an explicit
  performance budget.
- For very long dictation, either reduce cadence or use a deliberately designed
  rolling/committed-prefix algorithm.
- Use token timestamps when complete and trustworthy to hide only a known
  unstable acoustic tail.
- If timing metadata is incomplete, fall back conservatively rather than
  pretending a word is stable.
- On silence, keep the last trustworthy display and do not append an unaligned
  hypothesis.

String reconciliation cannot prove acoustic correctness. A longest-common-
prefix or overlap heuristic may reduce visual jumping, but it must not append
unrelated hypotheses simply to keep the overlay moving.

### 9.3 Incremental engines

For Parakeet Flash, Nemotron, or future true-streaming engines:

- Feed each canonical audio sample once.
- Maintain a per-session absolute sample cursor.
- Reset the engine if the cursor moves backwards or a new session begins.
- Emit partial text from the active stream.
- Flush only the remaining tail on stop.
- If the stream fails, do not silently construct a final from a damaged state.
  Either retry through a documented independent-final path or return an
  actionable failure.
- Track committed versus revisable tokens if the model exposes that distinction.

### 9.4 Preview regression scenarios

The fixed corpus must include:

- `one two three`, pause, `one two three`, pause, `test again`;
- short words and number sequences;
- long internal pauses;
- trailing room silence;
- background fan/keyboard noise;
- speech ending near a chunk boundary;
- quick stop immediately after the last word;
- accented English and Portuguese/English switching where supported;
- utterances containing likely hallucination phrases;
- ten-, thirty-, and sixty-second recordings;
- repeat runs to expose nondeterministic preview behavior.

Assertions should focus on behavior, not an unrealistically exact partial:

- preview refreshes after each new spoken section;
- no new lexical content appears during a silence-only suffix;
- stale preview cannot appear after stop;
- final transcript contains the expected utterance within the engine's WER
  threshold;
- final result does not contain preview-only hallucinations;
- preview cadence and final latency remain within engine-specific budgets.

## 10. Phase 4: testing and continuous integration

### 10.1 Test layers

#### Unit tests

- Resampler phase and output length.
- Ring behavior and overflow accounting.
- Session state transitions.
- Capability serialization.
- Preview stability helpers.
- Typed error mapping.
- Manifest and checksum verification.
- Settings and backup migrations.
- Sidecar framing and protocol parsing.

#### Contract tests

Run every engine adapter against a fake engine/timeline harness:

- prepare once and concurrently;
- begin/push/partial/finish;
- cancel during prepare, preview, and final;
- engine switch while idle;
- forbidden switch while active;
- stale session emissions;
- engine failure and recovery;
- download does not alter selection;
- finalization priority over preview.

#### Audio integration tests

- Checked-in WAV fixtures with expected duration/sample count.
- A checked-in synthetic spoken-vowel fixture verifies the built-in WAV decoder
  without requiring FFmpeg or a text-to-speech tool at test time. It proves
  media decoding only; it is not a substitute for a labeled ASR/WER corpus.
- Packetized versions of those fixtures at several device rates.
- Capture simulation with route changes and discontinuities.
- Preview sequence recording, not only final text.
- File and microphone paths use equivalent canonical audio.

#### Real-engine tests

- CPU Whisper smoke test in ordinary CI where practical.
- CUDA Whisper, Parakeet TDT, and experimental engines on a self-hosted NVIDIA
  runner.
- Track WER, first-partial latency, final latency, RTF, and peak memory.
- Store benchmark artifacts per commit without treating ordinary small timing
  noise as a failure.
- Fail on material regression against an agreed tolerance.

#### Desktop smoke tests

- App launches from the actual produced bundle.
- Web assets load through Tauri's production protocol.
- Settings can be created, persisted, reopened, and migrated.
- Tray action reaches the running instance.
- Loopback API binds only to loopback.
- A test insertion target receives text where platform automation permits.

### 10.2 Initial CI matrix

| Job | Platform | Required checks |
| --- | --- | --- |
| Frontend | Linux | install, formatting/lint, TypeScript/Vite build |
| Rust portable | Linux | fmt, clippy, tests, release compile |
| Rust portable | macOS | tests, release compile, bundle smoke |
| Rust portable | Windows | tests, release compile, installer smoke |
| Vulkan compile | Linux | feature compile and link check |
| CUDA compile | Self-hosted Linux/NVIDIA | feature build, dependency audit, launch |
| CUDA runtime | Self-hosted Linux/NVIDIA | golden audio, engine health, performance |
| Packaging | all supported release OSes | signed artifact, clean-machine smoke test |

Pin action versions and toolchains, cache only safe build inputs, and retain test
reports, benchmark traces, artifact manifests, and checksums.

Current workflow status: the portable Linux/macOS/Windows jobs and a guarded
`self-hosted, linux, x64, nvidia-cuda` CUDA compile/lint/test gate are checked
in. The CUDA runner must provide a compatible NVIDIA driver, `nvcc`, ALSA, and
GTK development packages. That gate proves feature compilation and lifecycle
tests only; model-backed golden-audio, health, performance, signed-bundle, and
clean-machine checks remain required before a GPU release is approved.

### 10.3 Quality gates

No merge to the release branch if:

- formatting, lint, unit, or contract tests fail;
- portable builds fail on any supported platform;
- an engine-selection migration fails;
- privacy-log tests detect prohibited content;
- model manifests are internally inconsistent;
- a stable engine exceeds agreed WER or latency regression limits;
- bundle smoke tests fail.

Experimental GPU engines may be allowed to report a non-blocking quality result,
but their lifecycle, crash, protocol, and privacy tests must still pass.

## 11. Phase 5: componentized model and CUDA runtime management

### 11.1 Problems to remove

- Absolute build-machine RPATHs.
- End users running repository setup scripts manually.
- Runtime readiness inferred only from marker-file existence.
- Mutable `main` model revisions.
- Size-only validation for large model weights.
- Downloads that select an engine as a side effect.
- Partial replacement that deletes the last working component before the new
  component is proven healthy.
- No standardized disk-space, driver, VRAM, or protocol compatibility check.

### 11.2 Component manifest

Each downloadable component should have an application-owned manifest:

```json
{
  "schema": 1,
  "id": "parakeet-tdt-v3-int8-cuda",
  "version": "1.0.0",
  "engine_api": 1,
  "platform": "linux-x86_64",
  "requirements": {
    "nvidia_driver_min": "...",
    "cuda_major": 12,
    "cudnn_major": 9,
    "disk_bytes": 900000000,
    "vram_bytes_recommended": 3000000000
  },
  "files": [
    {
      "path": "runtime/lib/libsherpa-onnx-c-api.so",
      "bytes": 123,
      "sha256": "..."
    }
  ],
  "sources": [
    {
      "url": "https://.../immutable-release-asset",
      "sha256": "..."
    }
  ]
}
```

The manifest itself should ship with the signed application or be verified by a
signature rooted in the application release key.

### 11.3 Installation transaction

1. Resolve platform and hardware requirements.
2. Confirm free disk space with a safety margin.
3. Download to a unique staging directory.
4. Support resume only when the server and checksum model make it safe.
5. Validate status, content type, byte count, archive structure, and SHA-256.
6. Extract without path traversal or unsafe symlinks.
7. Validate every required file.
8. Launch a bounded runtime self-test.
9. Write a component receipt containing versions and checksums.
10. Atomically switch an `active` pointer/directory.
11. Retain the prior known-good component until the new version has completed a
    real engine health check.
12. Clean staging on failure/cancellation/startup recovery.

Current implementation: the Nemotron runtime installer builds its Python
environment in an application-owned staging directory, runs a CUDA health
probe, records the resolved package set in a receipt-backed manifest, and only
then replaces the active runtime. The previous runtime remains intact when
installation or activation fails. Immutable wheel manifests and signatures are
still required before treating that Python runtime as a release-grade,
reproducible artifact.

At application startup, Voxide also removes only its own abandoned component
transaction directories after a 24-hour grace period; it never scans or
recursively removes arbitrary application-data directories.

### 11.4 Runtime loading

- Prefer bundle-relative or component-root-relative library lookup.
- Avoid `LD_LIBRARY_PATH` requirements for desktop shortcut launches.
- Validate ABI and sidecar protocol versions before model load.
- Clearly distinguish application CUDA support, installed runtime support,
  driver availability, and model availability.
- Provide `Verify installation`, `Repair`, `Remove`, and `Open storage location`
  actions.
- Report actual GPU/provider selection in diagnostics.

Current implementation: CUDA engine settings expose an explicit `Verify
installation` action. It rehashes the runtime/model artifacts against their
component receipts only when requested, while ordinary recording readiness uses
the verified install receipt and required-file inventory so it does not block a
hotkey by hashing multi-gigabyte model weights.

Nemotron settings also provide a guarded runtime-removal action. It refuses to
remove the app-owned Python/CUDA directory while capture, finalization, or the
cache-aware sidecar is active, and intentionally leaves the separately managed
model in place for a later runtime repair.

CUDA engine settings can open only their application-owned component storage;
custom user-supplied Whisper paths are never followed by that action.

### 11.5 Sidecar supervision

Nemotron already uses a child process, but production supervision should add:

- a versioned handshake;
- startup deadline;
- per-request identifiers;
- maximum message size;
- strict JSON schema validation;
- heartbeat or bounded health request;
- stderr capture with redaction and rotation;
- exit-code and signal reporting;
- inference timeout and cooperative cancellation;
- forced termination after a cancellation grace period;
- bounded restart policy;
- crash-loop detection;
- GPU OOM categorization;
- cleanup of orphan processes at application startup;
- no transcript contents in ordinary diagnostic logs.

Longer term, moving native GPU inference behind a sidecar boundary would prevent
an ONNX Runtime or CUDA native crash from terminating the Tauri UI. This is a
larger change and should follow the common engine contract.

### 11.6 Acceptance criteria

- A release artifact runs on a clean compatible machine without paths from the
  build host.
- Every installed component is traceable to a version and verified digest.
- Interrupted installation preserves the previous working engine.
- Failed installation does not change engine selection.
- Repair detects and replaces corrupt files.
- Sidecar failure returns the app to a usable state without restart whenever
  safe.

## 12. Phase 6: engine roadmap and maturity policy

### 12.1 Stable default: Parakeet TDT v3 CUDA

Keep Parakeet TDT as the preferred production CUDA transcription engine while
its measured final accuracy remains best for the target use case.

Required hardening:

- canonical audio input;
- no capture contention from full-buffer snapshots;
- timestamp-tail logic only when timing metadata is complete;
- preview backpressure and final priority;
- vocabulary boost restricted to final decoding unless measured otherwise;
- clear language coverage;
- long-recording performance policy;
- clean runtime packaging;
- golden-corpus thresholds.

### 12.2 Compatibility engine: Whisper

Whisper remains important for:

- broad language coverage;
- translation;
- CPU and non-NVIDIA operation;
- fallback when specialized engines are unavailable;
- well-understood offline behavior.

Required hardening:

- keep final and preview state isolated;
- abort stale preview promptly;
- give final inference admission priority;
- validate active CPU/Vulkan/CUDA backend;
- keep VAD behavior limited to Whisper and documented;
- prevent silence annotations/hallucination text from entering output;
- benchmark model/thread/backend choices;
- provide an automatic fallback model only with explicit user policy.

### 12.3 Investigate: Parakeet Flash / realtime EOU

FluidVoice's Parakeet Flash is genuinely incremental: it sends only new audio to
a streaming EOU manager in approximately 160 ms chunks and finalizes the same
stream. That can provide faster, more coherent preview without repeated
full-buffer inference.

The FluidVoice implementation is CoreML/FluidAudio-specific and cannot be
directly ported to Linux. A Voxide investigation must first establish:

- an official or trustworthy CUDA/ONNX-compatible model/runtime path;
- model license and redistribution terms;
- exact language coverage;
- decoder and EOU behavior;
- CPU/GPU memory requirements;
- runtime packaging feasibility;
- final accuracy compared with TDT v3;
- preview behavior across pauses and noise.

It should remain Experimental until it passes the release gates below.

### 12.4 Experimental: Nemotron

Retain Nemotron as an experiment, as agreed. Do not delete the implementation,
runtime, or model management work, but:

- label it Experimental;
- do not select it automatically after installation;
- keep it out of onboarding defaults;
- show its substantial runtime/model/VRAM requirements;
- do not describe it as better than Parakeet without measured corpus results;
- preserve trace collection so the experiment remains useful;
- isolate its sidecar failures;
- require explicit opt-in to quality telemetry if aggregate results are ever
  collected.

### 12.5 Engine promotion gates

An engine may move from Experimental to Beta only when:

- installation and removal are transactional;
- it passes lifecycle/cancellation tests;
- it survives 100 rapid session cycles;
- it has no known privacy leak;
- it completes the multilingual/target-language corpus;
- preview updates reliably after separated speech segments;
- silence does not append unrelated text above the agreed tolerance;
- final WER is documented against the stable default;
- first-partial and final latency are documented;
- memory/VRAM use is bounded and shown to the user;
- application crashes are isolated or absent in stress testing.

Promotion from Beta to Stable additionally requires:

- a period of field use without critical lifecycle defects;
- signed/reproducible runtime distribution;
- automated clean-machine installation tests;
- a rollback-compatible component version;
- no unresolved high-severity accessibility, data-loss, or crash issues;
- user-facing documentation for limitations and recovery.

## 13. Phase 7: application updates, settings, and data recovery

### 13.1 Application updates

The current update path discovers GitHub releases and opens a trusted release
page. Production distribution should add:

- signed update metadata;
- signed platform artifacts;
- stable and beta channels;
- download progress and cancellation;
- artifact hash/signature verification;
- safe application replacement using the platform-supported mechanism;
- health check after restart;
- retention of at least one known-good rollback package;
- rollback UI after a failed launch or explicit user request;
- release notes and compatibility warnings;
- no silent downgrade.

Use Tauri's supported updater architecture where it fits rather than reproducing
FluidVoice's macOS-specific bundle replacement.

### 13.2 Settings schema

Add an explicit persisted schema version and stepwise migrations:

```rust
struct PersistedDatabase {
    schema_version: u32,
    data: AppDatabase,
}
```

Requirements:

- migrations are deterministic and tested from every supported historical
  version;
- unknown future major versions are rejected without overwriting them;
- a backup is created before migration;
- failed migration preserves the original file;
- normalization occurs within a named migration or post-migration invariant
  pass;
- credentials remain outside backups;
- engine IDs and model IDs are not overloaded in one setting;
- unavailable engines remain visible with a reason but cannot become dangling
  selections.

### 13.3 Persistence durability

- Write temporary data in the same filesystem as the destination.
- Flush the file and, where useful, its parent directory before reporting
  success.
- Rotate a small number of last-known-good database backups.
- On parse failure, offer recovery from the newest valid backup.
- Never normalize and overwrite a corrupt file before preserving it.
- Make audio-history/database updates recoverable if the process stops between
  file creation and database commit.
- Periodically remove orphaned owned WAV/staging files after a grace period.

### 13.4 Backup format

- Explicit major/minor schema.
- Application version and export timestamp.
- Settings, profiles, bindings, dictionary, history, and optional metadata.
- Explicitly exclude API keys, analytics identity, model files, runtimes, and
  raw audio unless the user selects a separate audio export.
- Validate before mutating current state.
- Restore transactionally, with a pre-restore backup.
- Provide a dry-run summary of items to be replaced where practical.

## 14. Phase 8: user-facing robustness

### 14.1 Engine settings

- Stable/Beta/Experimental badges.
- Availability reason and repair action.
- Installed versus loaded distinction.
- Actual active backend and GPU where safe to show.
- Input selection for every microphone-based engine.
- Language configuration derived from capabilities.
- Disk/RAM/VRAM estimates.
- Model/runtime version and storage size.
- Self-test and copy-diagnostics actions.
- Installation never changes selection implicitly.
- Removing the active engine requires confirmation and a fallback choice.

### 14.2 Recording overlay

- Distinguish `Listening`, `Reconnecting microphone`, `Transcribing`,
  `Refining`, `Complete`, and `Failed`.
- Never leave `Listening` visible if capture has stopped growing.
- Clearly provisional preview styling.
- Do not show internal engine errors as transcript text.
- On recoverable preview failure, continue recording and show a subtle state;
  final transcription should still run.
- On capture failure, stop rather than presenting a misleading final based on
  incomplete audio.

### 14.3 Fallback policy

Fallback should be explicit and configurable. Suggested defaults:

- Preview failure: keep recording; no engine switch.
- AI post-processing failure: deterministic cleaned raw transcript, matching
  current behavior.
- Insertion failure: preserve the final result in history and offer copy/retry.
- Stable local engine unavailable before recording: block with repair guidance;
  do not silently send audio to a cloud engine.
- GPU OOM: unload inactive GPU components and retry once only when safe; then
  fail with guidance. Do not silently choose a lower-quality engine unless the
  user has enabled fallback.
- Device loss: bounded reconnect for default-device mode; otherwise stop with a
  recoverable error.

### 14.4 Support diagnostics

Provide a redacted bundle containing:

- application/build version;
- OS, architecture, desktop/session type;
- engine/runtime/component versions;
- non-secret settings relevant to capture and inference;
- latest bounded structured session traces;
- capture health counters;
- sanitized sidecar exit/protocol information;
- update/component health;
- no transcripts, prompts, selected text, commands, paths, clipboard contents,
  raw audio, API keys, or tokens.

## 15. Accuracy features to adopt later

Once the pipeline is trustworthy, FluidVoice has two quality features worth
adapting:

### 15.1 Post-transcription edit tracking

Observe explicit user edits to a recently inserted dictation, with strict
privacy boundaries and opt-in behavior. Repeated corrections can propose a
dictionary rule.

Guardrails:

- never record arbitrary typed content;
- scope observation to text Voxide just inserted;
- keep matching local;
- require repeated evidence;
- show the exact proposed spoken/replacement pair;
- require acceptance;
- support dismiss/cooldown/delete;
- do not send examples remotely without separate explicit consent.

### 15.2 Pronunciation customization

Where an engine exposes suitable encoder features or native vocabulary APIs,
support a short pronunciation enrollment flow. Keep this engine-specific behind
capabilities rather than presenting it universally.

This is higher-value than adding another general-purpose LLM provider, but it
is lower priority than correct capture and predictable preview.

## 16. Detailed verification matrix

### 16.1 Capture matrix

| Dimension | Cases |
| --- | --- |
| Sample format | signed/unsigned 8/16/24/32/64-bit where CPAL exposes it, f32, f64 |
| Sample rate | 8k, 16k, 32k, 44.1k, 48k, 96k |
| Channels | mono, stereo, multichannel |
| Packet size | tiny, typical, large, irregular/random |
| Session | short tap, normal, long, rapid repeat, cancel |
| Device | default, manually selected, disappears, default changes |
| Load | idle, concurrent preview, slow inference, UI busy |
| Failure | overflow, callback error, worker panic/error, discontinuity |

### 16.2 Engine matrix

| Behavior | Whisper | Parakeet TDT | Parakeet Flash candidate | Nemotron | Cloud | Apple Speech |
| --- | --- | --- | --- | --- | --- | --- |
| Prepare/load | Required | Required | Required | Runtime + model | Credential/network | Permission/system |
| Preview type | Rolling/snapshot | Full snapshot | Incremental | Incremental | Periodic request | System-dependent |
| Final type | Full decode | Full decode | Stream flush | Stream flush | Full request | System-dependent |
| VAD | Whisper-specific | No app segmentation | Model/EOU | No app segmentation | Provider-specific | System-specific |
| Vocabulary | Prompt/context | Final boost | Investigate | Language prompt | Provider-specific | Contextual words |
| Translation | Yes where supported | No | No | No | Model-specific | No |
| Maturity target | Stable | Stable default | Experimental | Experimental | Stable if configured | Platform-specific |

### 16.3 Lifecycle matrix

Test each engine for:

- prepare success/failure/cancel;
- missing/corrupt component;
- start before ready;
- normal preview and final;
- stop before first preview;
- stop during preview;
- cancel during preview;
- cancel during final;
- repeated start/stop;
- engine switch while idle;
- attempted switch while active;
- app shutdown while loaded;
- component removal while selected;
- model/runtime upgrade and rollback;
- timeout/OOM/process exit;
- stale event rejection.

### 16.4 Production performance budgets

Budgets should be measured and finalized on representative hardware. Initial
targets for the verified RTX 4080 Laptop configuration can be:

- capture start to first canonical audio: under 150 ms;
- no sample loss during healthy operation;
- first stable/provisional preview:
  - true-streaming engine: under 800 ms;
  - Parakeet TDT snapshot: under 1.5 s;
  - Whisper: model/backend-specific;
- Parakeet TDT stop-to-final for a typical short utterance: under 750 ms;
- preview work must not add more than 100 ms to final queue time;
- direct insertion after final formatting: under 100 ms in the normal path;
- idle warm engine must not leave microphone capture active;
- sustained memory growth across 100 sessions: effectively zero after caches
  stabilize;
- long-session canonical audio storage growth: linear and predicted;
- no UI hang longer than one animation frame due to inference or file I/O.

These are starting targets, not claims about the current implementation.

## 17. Rollout strategy

Use feature flags and narrow changes rather than rewriting all engines at once.

### Milestone A: observe

- Land session IDs, capture counters, and redacted trace export.
- Add fixtures and baseline results.
- Do not change default transcription behavior.

### Milestone B: replace capture beneath one engine

- Add the new capture pipeline behind `canonical_audio_pipeline`.
- Exercise it first with Whisper or a fake engine.
- Compare canonical WAV output against the old pipeline.
- Enable it for Parakeet only after capture tests pass.
- Retain a temporary fallback for diagnosis, not indefinitely.

### Milestone C: introduce coordinator and adapters

- Wrap existing Whisper without changing inference.
- Wrap Parakeet TDT.
- Move Nemotron sidecar ownership into its adapter.
- Convert cloud and Apple Speech last.
- Delete old top-level branches only after parity tests.

### Milestone D: component manager

- Manage Parakeet model first.
- Manage Parakeet CUDA runtime.
- Adopt Nemotron runtime/model without selecting it automatically.
- Migrate existing installations into receipts after verification.

### Milestone E: engine experiments

- Prototype Parakeet Flash behind Experimental.
- Collect local corpus results.
- Promote only through objective gates.

### Milestone F: distribution

- Enable signed updater after CI produces signed clean-machine-tested bundles.
- Add rollback and data recovery before broad automatic rollout.

## 18. Suggested implementation order and dependencies

```text
0. Baseline diagnostics + fixtures
   │
   ├──► 1. Canonical audio pipeline
   │       │
   │       ├──► 3. Correct preview scheduling/semantics
   │       └──► 4. Audio integration and stress tests
   │
   └──► 2. ASR contract + coordinator
           │
           ├──► 5. Component/runtime manager
           │       └──► 6. Signed release packaging
           │
           └──► 7. Parakeet Flash experiment

6. Signed release packaging ─► 8. Updater + rollback
1 + 2 + 4 stable ───────────► 9. Pronunciation/edit learning
```

Practical sequence:

1. Add capture/session diagnostics.
2. Add synthetic packet and WAV fixtures.
3. Replace the callback mutex with a bounded ring.
4. Add the stateful canonical resampler and exact session boundaries.
5. Verify that the pause-heavy preview reproduction has zero dropped frames.
6. Introduce typed engine capabilities and errors.
7. Introduce the coordinator and wrap Whisper.
8. Wrap Parakeet TDT and remove its direct capture-state ownership.
9. Wrap Nemotron and add sidecar supervision.
10. Make the UI capability-driven and separate install from select.
11. Add CI and self-hosted CUDA gates.
12. Replace absolute CUDA runtime paths with component-relative distribution.
13. Prototype Parakeet Flash.
14. Add signed updater/rollback and versioned database recovery.
15. Revisit personalized pronunciation/edit learning.

## 19. Definition of production-ready

Voxide should not be called production-ready merely because it builds and
transcribes on the developer machine. For the initial supported release scope,
all of the following should be true:

### Reliability

- No known silent audio-loss path.
- Capture health is measured.
- Device loss has a deterministic outcome.
- Cancel and rapid restart are race-free.
- Preview cannot corrupt or delay final transcription beyond its budget.
- Stable engines survive stress cycles without leaks or crashes.

### Accuracy

- Stable engines pass a documented fixed corpus.
- Preview silence hallucination behavior is bounded and tested.
- Final WER and major known limitations are documented.
- Experimental quality is never marketed as stable quality.

### Distribution

- Clean-machine installation is automated and tested.
- CUDA/native dependencies are reproducible and relocatable.
- Application and component artifacts are signed/verified.
- Licenses and redistribution obligations are documented.
- Updates and rollback do not risk user data.

### Data and privacy

- Settings have versioned migration and recovery.
- Backups are versioned and exclude secrets.
- Diagnostics exclude content by automated test.
- Audio retention is explicit, bounded, and removable.
- Cloud use is never an implicit fallback.

### Supportability

- A user can distinguish device, runtime, model, and credential failures.
- A redacted diagnostic bundle is sufficient to investigate most failures.
- Engine/runtime versions are visible.
- Repair and rollback actions exist.
- Release notes and known limitations are accessible.

### Engineering process

- CI covers supported portable platforms.
- A CUDA runner covers the production GPU path.
- Integration fixtures exercise real audio behavior.
- Release artifacts, manifests, checksums, and test reports are retained.
- Stable/Beta/Experimental promotion is evidence-driven.

## 20. Immediate next deliverable

The first implementation deliverable should be narrowly scoped:

> Build the canonical capture pipeline with a bounded callback ring, stateful
> 16 kHz conversion, exact session boundaries, dropped-frame diagnostics, and a
> deterministic packet-fixture test harness. Route one existing engine through
> it without changing that engine's decoding behavior.

Completion checklist:

- [x] No callback path uses `Mutex<Vec<f32>>` or silently ignores failure.
- [x] Raw packets have sequence/timestamp metadata.
- [x] Resampling preserves fractional phase across packets.
- [x] Canonical audio is the only input used by preview and final inference.
- [x] Preview snapshots cannot block capture.
- [x] Stop drains through an exact session boundary.
- [x] Device/callback failure reaches coordinator/UI state.
- [x] Capture health appears in redacted diagnostics.
- [x] 44.1 and 48 kHz fixture tests pass.
- [x] Pause-heavy dictation fixture produces multiple preview updates.
- [x] Silence-only suffix does not append preview text in the behavioral test.
- [x] Existing final transcription tests remain green.
- [x] Frontend build and portable Rust tests remain green.
- [ ] CUDA feature build and runtime tests pass on the guarded NVIDIA runner.

Only after this deliverable is verified should the engine lifecycle refactor or
Parakeet Flash experiment become the active implementation focus.
