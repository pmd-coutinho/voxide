# Capture & transcription improvement roadmap

Verified gap analysis of the Voxide Rust/Tauri port against the FluidVoice
macOS Swift original (`~/dev/FluidVoice`), scoped to the **capture** and
**transcription** pipelines. Each item was independently verified for (a)
whether it is a *real* gap in the port and (b) whether the idea is *portable*
to Linux/Rust (not blocked by a macOS-only framework such as AVAudioEngine,
CoreAudio HAL, CoreML, or Apple Speech).

Evidence is cited as `path:line`. FluidVoice paths are relative to
`~/dev/FluidVoice/Sources/Fluid`; Voxide paths are relative to
`src-tauri/src`.

## What the port already does well

The port's transcription path is mature and in several respects *ahead* of
FluidVoice's plain `WhisperProvider`. The following were investigated and are
**not** gaps:

- **Silero VAD gating** as a lenient gate (not a scalpel) — `speech.rs:443-539`.
- **Dual hallucination filter** (no-speech-probability + mean-logprob) —
  `speech.rs:748-785`.
- **Monotonic preview stabilization** — `lib.rs:7256+`.
- **Vocabulary biasing already ported** — whisper `initial_prompt`
  (`speech.rs:711-719`) and sherpa-onnx BPE context-graph hotwords
  (`parakeet.rs:217-345`).
- **Model-download markup byte-sniffing** — `looks_like_markup` (`lib.rs:5639`)
  at both download and load time, plus pinned SHA256 receipts for
  Parakeet/Nemotron (strictly stronger than FluidVoice's markup sniff).

The real portable gaps cluster in three areas: **audio-capture latency &
robustness**, **vocabulary-biasing depth**, and **AI post-processing
correctness/privacy**.

## Design constraints (do not regress)

These are hard-won lessons already encoded in the port; recommendations below
respect them:

- Silero VAD runs as a **gate**, not mid-utterance segment extraction —
  reintroducing stitching measurably lost words.
- `INFERENCE_LOCK` serializes live preview and final decode on the shared
  context; concurrent GPU inference corrupts results.
- The overlay must mutate existing DOM nodes, never replace subtrees via
  `innerHTML` per frame (WebKitGTK repaint trap).

---

## Tier 1 — highest value

### 1. Prewarm capture *resolution* off the hotkey path
- **Category:** capture · **Impact:** high · **Effort:** medium · **Verdict:** CONFIRMED (cheap subset)
- **Status:** ✅ Implemented. `AudioCapture::start` split into `prepare`
  (device/config/`pactl` routing) + `start_prepared` (build+play only); a
  `PreparedInput` is cached in `NativeCaptureState.prepared_input` at startup,
  reused on the hotkey path via `start_dictation_capture`, and re-prewarmed
  after each dictation frees the mic (`RefreshCapturePrewarmWhenDropped`).
  Routing env-var writes serialized by `pulse::routing_lock`. Idle prewarm
  opens no stream (no mic indicator). Reviewed clean for concurrency/parity.
- **Gap:** `AudioCapture::start` (`audio.rs:132-261`) does everything on the
  hotkey press — `default_host()`, a synchronous `pactl` subprocess for
  routing, device enumeration, `default_input_config()`, ring alloc, worker
  spawn, then `stream.play()`. `prewarm_input_devices` (`audio.rs:385`) only
  counts device names.
- **FluidVoice:** `prepareDirectAudioInputIfPossible` (`ASRService.swift:738`)
  builds the device/ring without starting IO; the hotkey path only calls
  `start()`.
- **Do:** cache the resolved `cpal::Device` + `SupportedStreamConfig` + pulse
  routing decision behind a `Mutex<Option<PreparedInput>>`, refreshed at
  startup / on permission-grant / on device-preference change. The hotkey path
  then only builds+plays. Removes the synchronous `pactl` fork+exec from the
  press.
- **Caveat:** do **not** port FluidVoice's *warm running stream* naively — on
  Linux, building a cpal stream opens the PCM / registers a PipeWire node,
  which trips the OS recording indicator, so "prepared-not-running" does not
  map cleanly. Ship the resolution-caching subset (privacy-safe: device still
  opens only on record). Gate any always-hot-mic variant behind an explicit
  opt-in setting.

### 2. Bound the Parakeet live-preview to a recent window
- **Category:** engines · **Impact:** high · **Effort:** small (~1 line) · **Verdict:** CONFIRMED (primary)
- **Status:** ✅ Implemented. `spawn_live_parakeet_preview` now snapshots a 20 s
  trailing window (`PARAKEET_PREVIEW_WINDOW`) instead of `snapshot_all()`. The
  final decode still uses the complete capture; the nemotron stream keeps
  `snapshot_all()` (needs the full buffer for deltas).
- **Gap:** `spawn_live_parakeet_preview` calls `snapshot_all()` every 600 ms
  (`lib.rs:7510`) and re-decodes the *entire growing buffer* — O(total) work
  that holds `INFERENCE_LOCK` progressively longer, delaying the final decode.
  It is the outlier: whisper uses `snapshot_recent(8s)` (`lib.rs:7387`), cloud
  uses `snapshot_recent(20s)` (`lib.rs:7792`), nemotron feeds deltas.
- **Do:** `capture.snapshot_recent(Duration::from_secs(20))`. Mirrors the other
  engines and directly reduces lock contention. Verify the
  `fluidvoice_preview_reconcile` prefix-alignment (`lib.rs:7753`) degrades
  gracefully past the window (it does — same as cloud today). Add a test beside
  `parakeet_preview_matches_fluidvoice_full_snapshot_reconciliation`
  (`lib.rs:11547`).

### 3. Recover from mid-recording device loss instead of hard-aborting
- **Category:** capture · **Impact:** medium-high · **Effort:** medium · **Verdict:** CONFIRMED
- **Gap:** `spawn_capture_error_monitor` (`lib.rs:1842`) cancels the whole
  session and shows "Microphone connection lost" on *any* `stream_errors > 0`
  (`audio.rs:215-218` only bumps a counter). A USB/Bluetooth blip mid-dictation
  loses the entire recording.
- **FluidVoice:** debounces ~1 s and rebuilds the backend, keeping the session
  alive (`ASRService.swift:2357-2399`); only falls back to abort if rebuild
  fails.
- **Do (layer 1):** on error, debounce (~1 s, coalesce bursts), then rebuild an
  `AudioCapture` that **shares the existing `canonical_samples` Arc** (preserve
  the timeline + do not reset the engine cursor); fall back to today's terminal
  error only if rebuild fails. Watch per-stream `CaptureHealth` counters so the
  monitor doesn't re-trigger on stale counts.
- **Do (layer 2, optional):** proactive `pactl subscribe` listener (the module
  already shells to `pactl`) to catch "system default source switched while old
  device still alive."
- **UX:** audio during the gap is lost — show "reconnecting microphone…", not
  zero-loss continuity.

---

## Tier 2 — vocabulary-biasing cluster (primary accuracy lever)

Separately shippable but touch the same path; best done together.

### 4. Cross-link the correction dictionary into the boost vocab
- **Category:** engines · **Impact:** medium · **Effort:** small · **Verdict:** PARTIAL
- **Status:** ✅ Implemented. `recognition_vocabulary` now also feeds each
  `DictionaryEntry.replacement` (target only, never `spoken`) into the boost
  vocab, deduped, gated on `vocabulary_boosting_enabled`, sharing the 200-term
  cap. Flows to both whisper `initial_prompt` and the sherpa hotword graph.
- **Gap:** `recognition_vocabulary` (`lib.rs:1051`) reads only `custom_words`;
  `DictionaryEntry{spoken,replacement}` (`lib.rs:980`) is used only for post-hoc
  replacement, so correction targets are never acoustically boosted.
- **FluidVoice:** folds each replacement into the boost vocab at weight 8.0
  (`ParakeetVocabularyStore.swift:258`).
- **Do:** also push each `entry.replacement` (deduped, trimmed) into the
  returned vocab. **Only the replacement target** — do *not* boost
  `entry.spoken` (the misheard form), which would bias the transducer toward
  the wrong token. Flows through both whisper `initial_prompt` and sherpa
  hotwords automatically.

### 5. Honor per-term weights via sherpa `phrase :score`
- **Category:** engines · **Impact:** medium · **Effort:** medium · **Verdict:** PARTIAL
- **Status:** ⏸️ Deferred. Needs a weighted vocab threaded through the shared
  engine path, new Settings, a weight scale-mapping, and (to help most users) a
  Settings UI control — the add-word UI doesn't set weights, so backend-only
  helps only local-API/import users, and the accuracy payoff is unproven.
  Revisit if boosting proves too weak/strong in practice.
- **Gap:** `CustomWordEntry.weight` is persisted (`lib.rs:1046`) but discarded —
  `recognition_vocabulary` returns `Vec<String>` and decode uses a single
  global `hotwords_score=1.5` (`parakeet.rs:224`).
- **Do:** sort by weight at the **recognition-build path only** (not
  `normalize_custom_words` — that breaks the reference-normalization test at
  `lib.rs:11217`), and emit `PHRASE :score` per `/`-separated phrase via
  `create_stream_with_hotwords` (already called at `parakeet.rs:340`).
  **Scale-mapping is mandatory:** FluidVoice's 5–13 scale ≠ sherpa's ~1.5, so
  `score = 1.5 * (weight / 10.0)` clamped to ~0.5–4.0; omit `:score` when weight
  is `None` (falls back to global). Add a UI boost-strength control
  (mild/balanced/strong) since the add-word UI (`main.ts:1629`) never sets
  weight today.

### 6. Min-term-length filter + configurable global boost
- **Category:** engines · **Impact:** medium · **Effort:** small · **Verdict:** PARTIAL
- **Status:** ⏸️ Deferred (with #5). A min-length filter can't be hardcoded
  safely — it would drop legitimate short terms (the languages "C"/"R", "Qt",
  "Go") — so it needs a configurable knob, i.e. new Settings/UI. Bundled with
  #5's tuning work.
- **Gap:** `vocabulary_hotwords` (`parakeet.rs:317`) has no length guard.
  FluidVoice defaults `minTermLength=3` (`ParakeetVocabularyStore.swift:83`).
- **Do:** add a Unicode-aware `word.chars().count() >= min_term_length` filter
  **in the parakeet path only** (the whisper `initial_prompt` shares the vocab
  and short terms are harmless there). Make it **configurable** (users need
  "Qt", "Go", "AI", "C#", "3D"), not hardcoded. Also expose the global boost as
  a setting instead of the hardcoded `1.5`.

### 7. Surface which boosted terms actually landed
- **Category:** engines · **Impact:** low-medium · **Effort:** small (~30 lines) · **Verdict:** CONFIRMED
- **Status:** ✅ Implemented (log). `detected_vocabulary_terms` does whole-word,
  case-insensitive matching of the active vocab against the final Parakeet
  transcript and logs `BOOST_HIT: …`. The Tauri event / UI status badge is
  deferred (needs UI); logging delivers the diagnostic value now.
- **Gap:** no post-decode inspection of which boosted terms appear in the
  output. Standalone — does **not** depend on weights.
- **FluidVoice:** `detectBoostedTerms` (`FluidAudioProvider.swift:518`) →
  "Word boost: ON (N) • last hit: X".
- **Do:** word-boundary substring scan (`format!(" {term} ")` against a
  space-padded normalized transcript) of the final Parakeet transcript against
  the active vocab; log `BOOST_HIT` and emit a Tauri event for UI status. Scope
  to the parakeet/nemotron hotword path (whisper `initial_prompt` "landed"
  semantics are weaker). Caveat in-code: a hit means the term *appears*, not
  that boosting *caused* it.

---

## Tier 3 — AI post-processing correctness & privacy

### 8. Pre-gate AI post-processing on a verified provider fingerprint
- **Category:** architecture · **Impact:** medium · **Effort:** medium · **Verdict:** CONFIRMED
- **Status:** ✅ Implemented. `Settings.verified_provider_fingerprints` +
  `provider_fingerprint()` + `ensure_provider_verified()`; the automatic
  dictation path (`post_process_dictation_outcome`) now falls back to
  deterministic text (surfacing the reason) unless the live `SHA256(base|key)`
  matches a stored one. Fingerprint is captured on success in both
  `fetch_ai_provider_models` (the "Fetch models" control) and `enhance_text`
  (playground/rewrite — verifies over the chat endpoint, so providers without a
  `/models` endpoint verify too). Reviewed clean for gate correctness.
  **Scope:** dictation-only. Rewrite mode (`enhance_text`) and command mode
  (`request_command_plan`) are explicit user-initiated actions and remain
  ungated by design; their success doubles as verification for dictation. If a
  future guarantee should cover rewrite/command payloads too, that's a
  follow-up.
- **Gap:** `post_process_dictation_outcome` (`lib.rs:4928`) fires whenever
  enhancement is enabled — a rotated key / edited base URL / removed model is
  not caught, and **the transcript is transmitted to an unverified endpoint**.
- **FluidVoice:** `DictationAIPostProcessingGate.isProviderConfigured`
  (`DictationAIPostProcessingGate.swift:53`) requires a stored
  `SHA256("base|key")` fingerprint recorded on a successful connection test.
- **Do:** both primitives already exist — `is_local_endpoint`
  (`provider.rs:2249`) and `sha2` (imported at `lib.rs:18`). Add
  `verified_provider_fingerprints: HashMap<String,String>` to Settings; store
  hex over current base+key on `fetch_ai_provider_models` success
  (`lib.rs:3731`); before the AI branch require non-empty id+model, local
  endpoint OR present key, AND `fingerprint(current) == stored` — else fall back
  to `dictionary_corrected`. Self-correcting: an edited URL/rotated key fails
  the match automatically.
- **Framing:** primary win is privacy (never send to an unverified endpoint) +
  surfacing config drift instead of a silent downgrade; the latency win is
  real but specific to unreachable/edited base URLs (3× connect-retry ×
  streaming/complete double path).

### 9. `${transcript}` placeholder / single-user-turn fold
- **Category:** formatting · **Impact:** low-medium · **Effort:** small · **Verdict:** CONFIRMED
- **Status:** ✅ Implemented. `fold_dictation_prompt` substitutes the
  transcript for `${transcript}` and sends a single user turn (empty system)
  when present, else keeps the classic split; `openai_messages` now omits a
  blank system message. Opt-in, backward-compatible.
- **Gap:** every provider builder hard-codes `[system, user]`
  (`provider.rs:1745-1800`, `1881-1904`).
- **FluidVoice:** supports a `${transcript}` placeholder and folds
  instruction+transcript into one user turn
  (`DictationPostProcessingService.swift:201-228`), improving instruction-
  following on local/instruct models.
- **Do:** if the system prompt contains `${transcript}`, substitute and send a
  single user turn with empty system; else keep today's layout
  (backward-compatible, opt-in via placeholder — safer than FluidVoice, which
  always folds). Optionally skip the empty `{role:"system"}` entry. **Not** a
  KV-cache win — the current layout already has a stable prefix.

---

## Tier 4 — robustness & formatting polish

### 10. Reset resampler on packet discontinuity + drop audio in whole blocks
- **Category:** capture · **Impact:** medium · **Effort:** medium · **Verdict:** CONFIRMED
- **Status:** ✅ Implemented. `append_samples` now writes each block all-or-
  nothing via `write_chunk_uninit`/`fill_from_iter` (whole-block drop leaves a
  detectable sequence hole); `StatefulMonoResampler::reset()` added; the worker
  flushes the pre-gap batch and resets at a discontinuity so no interpolation
  bridges the hole. Adversarially reviewed (CORRECT, 92) incl. frame-alignment
  and no-gap-regression.
- **Gap:** on ring overflow, `append_samples` (`audio.rs:526-565`) drops *tail
  samples* while keeping one `packet_sequence` per block, so the consumer's gap
  check (`audio.rs:589`) never fires and `StatefulMonoResampler` interpolates
  across the hole → a click + time compression. There is no `reset()`.
- **FluidVoice:** drops *whole packets* (`CoreAudioCaptureSupport.c:259`) and
  calls `resetResamplerLocked` on any discontinuity
  (`ASRService.swift:3855-3865`).
- **Do:** atomic block writes via `rtrb`'s `write_chunk_uninit` (drop the whole
  block on overflow so the sequence gap becomes visible); add
  `StatefulMonoResampler::reset()` and split/flush the coalesce batch at the
  discontinuity boundary. Rare path (needs ~2 s consumer starvation given the
  2 s ring), so favor correctness/clarity. Leave the file-decode paths
  untouched (contiguous by construction).

### 11. Capture-clock divergence watchdog (warn-only)
- **Category:** capture · **Impact:** low-medium · **Effort:** small (~10 lines) · **Verdict:** PARTIAL
- **Status:** ✅ Implemented. `capture_clock_diverges` (loose asymmetric band
  `!(0.5..=1.5)` above a 500 ms floor) logs a warning at finalize; warn-only,
  never returns `Err`. Reviewed CORRECT (95).
- **Gap:** wall vs canonical duration is computed and **logged** at finalize
  (`lib.rs:8017-8033`) but never acted on. A stalled/mis-clocked source yields a
  silently short/garbled transcript.
- **Do:** add a tripwire after the health log. **Use a loose asymmetric band**,
  not FluidVoice's tight 0.7–1.3 — the port's wall clock brackets the audio
  (starts before first callback, ends after stop latency), so canonical is
  legitimately below wall. Flag e.g. `ratio < 0.5 || ratio > 1.5`, only when
  `wall_duration_ms >= 500`. **Warn-only** — do *not* `Err` (a false positive
  would discard a real dictation). Skip FluidVoice's single-backend failover /
  disable state machine (the port is cpal-only).

### 12. Persist the preferred mic by stable node `name`
- **Category:** capture · **Impact:** medium · **Effort:** small · **Verdict:** PARTIAL (half 1 only)
- **Status:** ⏸️ Deferred. Changes the persisted `selected_input_device` format
  and touches the frontend picker (`main.ts`) plus a one-time migration —
  crosses into frontend/persistence work. Real but latent (duplicate identical
  USB mics); revisit with a frontend pass.
- **Gap:** `input_device_names()` (`audio.rs:363`) persists the *description*,
  then `route_to_requested_source` (`audio.rs:477`) matches it back to
  `source.name` — two identical USB mics share a description, so the `find`
  silently binds whichever `pactl` lists first (latent correctness bug).
  Descriptions are also localized/firmware-dependent.
- **Do:** persist the stable `name` (`bluez_input.AA:BB:…`,
  `alsa_input.usb-…`); keep description as display label only; match
  `source.name == requested`. Add a one-time migration (match by description
  once, rewrite to the resolved `name`).
- **Dropped (half 2):** FluidVoice's 250/750/1500/2500 ms re-assertion backoff
  — the port never mutates an OS default (routing is per-process env vars
  recomputed each `AudioCapture::start`), so there is nothing to re-assert.
  A bounded retry *at capture start* if the preferred `name` isn't present yet
  (Bluetooth mid-connect) is the only residual, and it's minor.

### 13. CPU fallback when Parakeet CUDA load fails
- **Category:** engines · **Impact:** low-medium · **Effort:** small-medium · **Verdict:** PARTIAL
- **Status:** ✅ Implemented. On a failed CUDA `create()`, retry once with
  `provider="cpu"` and a real `num_threads` (`speech::cpu_decode_threads`, now
  `pub(crate)`) for both greedy and boosted configs; scoped to offline Parakeet.
  Reviewed CORRECT (93). (Preview rate-limiting in degraded mode not done — #2's
  windowing already bounds it.)
- **Gap:** `parakeet.rs:215` hardcodes `provider="cuda"` and hard-errors on
  `create()` failure; sherpa has no built-in CPU fallback (unlike whisper.cpp,
  which already degrades — `speech.rs:388`).
- **Do:** retry `create()` once with `provider="cpu"` for **both** boosted and
  unboosted paths — **but also** fix `num_threads` (hardcoded 1 at
  `parakeet.rs:216`; reuse `cpu_decode_threads()`/`physical_cpu_count()` from
  `speech.rs:337/368`) or CPU decode is unusably slow for a 0.6B TDT model.
  Rate-limit/suppress live preview in degraded mode. Scope to **offline
  Parakeet only** — `nemotron.rs:48` deliberately refuses CPU for streaming.
  Log a one-time degraded-mode warning.

### 14. Unicode-aware filler trimming
- **Category:** formatting · **Impact:** low · **Effort:** small · **Verdict:** PARTIAL (part 1 only)
- **Status:** ✅ Implemented. `remove_filler_words` trims `!is_alphanumeric()
  && !is_whitespace() && !is_control()` (`char::is_punctuation` is not in std),
  a true superset of the old ASCII trim that also handles Unicode punctuation
  while preserving newlines. Review caught (and fixed) an initial version that
  dropped attached newlines.
- **Gap:** `remove_filler_words` (`formatting.rs:441`) trims
  `is_ascii_punctuation()`, so a filler wrapped in smart quotes/em-dash/ellipsis
  survives filtering.
- **FluidVoice:** trims `.punctuationCharacters` (Unicode Pc/Pd/Ps/Pe/Pi/Pf/Po)
  — `ASRService.swift:3411`.
- **Do:** swap the predicate to `char::is_punctuation()` (the faithful port).
  Extend the test at `formatting.rs:1223` with smart-quoted / en-dash-wrapped
  fillers.
- **Dropped:** the bundled "strip `!`/`?` in GAAV trailing-period removal" —
  FluidVoice does the same ASCII-`.`-only thing (`ASRService.swift:3520`), so
  it's not a gap; stripping `?` would mangle "Ready?".

### 15. Context-gated smart punctuation (dot + symbol only)
- **Category:** formatting · **Impact:** medium · **Effort:** medium · **Verdict:** PARTIAL
- **Status:** ⏸️ Deferred. FluidVoice's version is unshipped dead code, so this
  is a *new* heuristic needing fresh design + a validated test corpus (TLD/host
  tables, symbol-operand rules) and likely a new opt-in setting — not a
  port-of-proven-behavior. Highest design risk of the tier; revisit
  deliberately.
- **Gap:** the port's `apply_spoken_punctuation` (`formatting.rs:507`) has no
  dot-context (TLD/host tables → `api.json`) or symbol-operand heuristics.
- **Caution:** FluidVoice's version is **dead code** — the flag-setting
  `makeRules()` (`SpokenPunctuationFormatting.swift:208`) has zero callers, so
  it never shipped and is untested by FluidVoice's own E2E suite. Treat this as
  a **new heuristic to validate with fresh tests**, not a port of proven
  behavior.
- **Do:** port only the genuinely-absent parts — `has_dot_context` (using
  `phf`/`&[&str]` TLD/host/rejected-previous sets) and `has_symbol_context`
  (short/numeric operand check), gated behind new `PhraseRule` flags. **Skip**
  the slash-path and at-sign-app gates — they overlap the port's existing
  verb/app heuristics in `apply_literal_dictation_formatting`
  (`formatting.rs:908`, `has_spoken_slash_command_context`,
  `is_relaxed_mention_app`). A prefix-free smart-punctuation mode, if desired,
  should be an explicit opt-in setting.

---

## Considered and dropped

| Idea | Verdict | Why |
| --- | --- | --- |
| Byte-sniff downloaded model content | REJECTED (false gap) | Already ported: `looks_like_markup` (`lib.rs:5639`) at download + load, plus SHA256 receipts (stronger than FluidVoice). |
| Per-user acoustic pronunciation enrollment | REJECTED (not portable) | Needs encoder hidden-state tensors; sherpa/whisper-rs expose only text/tokens/timestamps. The text-only fallback duplicates existing dictionary/alias correction. |
| Timestamp boundary-trim of first/last callback | Deferred | cpal has no `StreamInstant::now()`, and on the cold-start architecture the leading trim *worsens* first-word clipping. Only meaningful as step 2 *after* item 1's warm stream. |
| Silence-aware chunk boundaries (Parakeet) | Deferred | The port's chunk is 20 min (`media.rs:12`), so it fixes ≤1 clipped word per 20-min seam; and it brushes against the "no mid-utterance stitching" lesson. Cheap overlap-and-dedup at the seam is the safer version if ever needed. |
| Mic re-assert backoff loop | Dropped | See item 12 half 2 — no OS default to clobber on Linux. |
| dB level meter with smoothing | Out of scope | Cosmetic UI; the port's adaptive floor/peak meter (`audio.rs:711`) is adequate. |
| Pre-model-load memory preflight | Out of scope | UX gate, not a pipeline quality/latency win. |

---

## Progress

- **Tier 1 — capture latency/robustness:** #1 ✅ #2 ✅ #3 ⏳ (mid-recording
  recovery, still open)
- **Tier 2 — vocabulary biasing:** #4 ✅ #7 ✅ · #5 ⏸️ #6 ⏸️ (need Settings/UI)
- **Tier 3 — AI post-processing:** #8 ✅ #9 ✅ (complete)
- **Tier 4 — robustness & formatting:** #10 ✅ #11 ✅ #13 ✅ #14 ✅ · #12 ⏸️
  (frontend/persistence) #15 ⏸️ (needs fresh design)

Still open and worth doing: **#3** (recover from mid-recording device loss
instead of aborting — Tier 1, `CONFIRMED`), then **#12** and **#5/#6** when a
frontend pass is in scope.

_Generated from a verified cross-repo comparison (5 CONFIRMED, 12 PARTIAL, 2
REJECTED findings) against FluidVoice `~/dev/FluidVoice`._
