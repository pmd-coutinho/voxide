# Cohere Transcribe — implementation notes

Adding [Cohere Transcribe][model] as a **CPU-capable** engine. Voxide's two
strongest engines, Parakeet TDT and Nemotron, are both Linux/NVIDIA-CUDA only, so
everyone without an NVIDIA GPU is on Whisper. Cohere Transcribe sits at the top of
the Open ASR Leaderboard, is Apache 2.0, and quantises to a size that runs on a
CPU with punctuation, capitalisation and inverse text normalisation built in.

Not yet implemented. This file records what was established up front so none of it
has to be re-derived — in particular the feature extractor, which is the part that
fails silently if it is even slightly wrong.

[model]: https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX

## The model

`onnx-community/cohere-transcribe-03-2026-ONNX`, Apache 2.0, 15 languages
(ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh + en).

Five precisions ship. External `.onnx_data` files hold the weights, so both the
graph and every shard have to be downloaded together:

| Precision | Encoder | Decoder | Total |
| --------- | ------- | ------- | ----- |
| fp32 | 7.6 GB (4 shards) | 676 MB | ~8.3 GB |
| fp16 | 3.8 GB (2 shards) | 338 MB | ~4.1 GB |
| quantized (int8) | 2.1 GB | 196 MB | ~2.3 GB |
| q4 | 2.0 GB | 109 MB | ~2.1 GB |
| **q4f16** | **1.4 GB** | **98 MB** | **~1.5 GB** |

q4f16 is the one to ship: smallest, and the precision voxtype reports running at
9–11× realtime on a Zen 4 CPU. **Note the encoder dominates** — 1.4 GB of the
1.5 GB — and it is a 48-layer encoder at `d_model` 1280, so the CPU cost is
concentrated there. Treat the 9–11× figure as unverified on our side until
measured; it should be benchmarked on this project's own audio before any claim
goes in the README.

### Architecture (from `config.json`)

| | |
| --- | --- |
| Encoder | `parakeet_encoder`, 48 layers, `d_model` 1280, 8 heads, 128 mel bins |
| Decoder | 8 layers, 8 heads, `head_dim` 128, `hidden_size` 1024 |
| Vocab | 16384 |

The encoder being NVIDIA's Parakeet encoder is why the front end below is
NeMo-shaped rather than Whisper-shaped.

## Feature extraction

This is the high-risk part: a subtly wrong front end produces fluent,
plausible, *wrong* transcripts rather than an error. The algorithm below is
transcribed from `transformers.models.cohere_asr.feature_extraction_cohere_asr`
(v5.14.1) — the reference implementation, not inferred from the config file.

Constants come from `preprocessor_config.json`: `sampling_rate` 16000,
`feature_size` 128, `n_fft` 512, `win_length`/`n_window_size` 400,
`hop_length`/`n_window_stride` 160, `preemphasis` 0.97, `dither` 1e-5,
`normalize` `"per_feature"`, `max_audio_clip_s` 35.0,
`overlap_chunk_second` 5.0, `min_energy_window_samples` 1600.

In order:

1. **Mono** — mean across channels.

2. **Chunk if longer than 35 s.** Not a fixed split: from
   `idx + chunk_size - 5 s` to `idx + chunk_size`, step through
   1600-sample windows, compute `sqrt(mean(w²))`, and cut at the **quietest**
   one. Segments shorter than 1600 samples cut at the midpoint. So long audio is
   split at pauses rather than mid-word.

3. **Dither** — `waveform[:valid] += 1e-5 * randn(valid)`, where the generator is
   seeded with `manual_seed(valid_samples)`, i.e. the sample count.
   **This is not reproducible in Rust**: it depends on PyTorch's RNG. It is a
   training-time regulariser at 1e-5 amplitude; skip it and compare against a
   reference generated with `dither=0`. Do not try to match it bit for bit.

4. **Preemphasis** — `[x[0], x[1:] - 0.97 * x[:-1]]`, on the **waveform**, after
   dither and before the STFT.

5. **STFT** — `n_fft` 512, `hop` 160, `win_length` 400,
   `hann(400, periodic=False)` (i.e. **symmetric**, unlike Whisper's periodic
   window), zero-padded to 512 and centred. `center=True` with
   `pad_mode="constant"` means **256 zeros are padded on each side**.

6. **Power** — magnitude, then squared: `(re² + im²)`.

7. **Mel** — `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0,
   fmax=8000, norm="slaney")`, matrix-multiplied onto the power spectrum. Slaney
   mel scale (librosa's default, *not* HTK) and Slaney normalisation.

8. **Log** — `ln(mel + 2⁻²⁴)`. The guard is `2**-24`, not the usual `1e-10`.

9. **Per-feature CMVN**, over the time axis, per mel bin, **ignoring padding**:
   mean over valid frames; variance with an **`N - 1`** denominator (sample, not
   population); then `(x - mean) / (std + 1e-5)`.

Result: `[batch, frames, 128]`.

Easy things to get wrong, each of which silently degrades output: a periodic
window, HTK mel, magnitude instead of power, `log10` instead of `ln`, an `N`
denominator in the variance, or normalising across bins instead of per bin.

## Decoder prefix

Cohere uses a Whisper-style multi-token prompt. Verified against
`tokenizer.json`'s `added_tokens` rather than copied:

| ID | Token |
| -- | ----- |
| 4 | `<|startoftranscript|>` |
| 62 | `<|en|>` |
| 5 | `<|pnc|>` (punctuation; `<|nopnc|>` = 6) |
| 8 | `<|itn|>` (inverse text normalisation; `<|noitn|>` = 9) |
| 11 | `<|notimestamp|>` |
| 13 | `<|nodiarize|>` (`<|diarize|>` = 12) |

So English transcription with punctuation and ITN, no timestamps, no
diarisation is `[4, 62, 5, 8, 11, 13]`.

`generation_config.json` sets `decoder_start_token_id: 13764`, which is **not** a
special token — it is an ordinary vocab entry. Transformers only uses it when no
decoder input is supplied, and we supply the full prefix, so it is irrelevant
here. `eos_token_id` is 3 (`<|endoftext|>`) and `pad_token_id` is 2.

## Verifying the front end

Numeric ground truth needs the reference extractor, which pulls in torch and
librosa:

```sh
python3 -m venv /tmp/cohere-ref
/tmp/cohere-ref/bin/pip install 'transformers>=5.3' numpy librosa torch --index-url https://download.pytorch.org/whl/cpu
/tmp/cohere-ref/bin/python -c "
from transformers import AutoFeatureExtractor
fe = AutoFeatureExtractor.from_pretrained('onnx-community/cohere-transcribe-03-2026-ONNX')
fe.dither = 0.0   # not reproducible in Rust; compare without it
import numpy as np, soundfile as sf
audio, sr = sf.read('<a 16 kHz mono wav>')
np.save('/tmp/reference-features.npy', fe(audio, sampling_rate=sr)['input_features'])
"
```

Compare the Rust output against that array per element. Tolerance rather than
equality: the STFT and mel matmul are float-order sensitive, so ~1e-3 absolute in
log-mel space is the right bar, and anything larger is a real discrepancy rather
than rounding.

The end-to-end oracle is stronger and worth doing once the sessions are wired: run
a known WAV all the way through and check the transcript. A front end that is
wrong in a way tolerance-testing misses will still produce visibly wrong words.

### One detail that only reading the reference reveals

`features_lengths` is computed as
`(audio_len + 2 * (n_fft / 2) - n_fft) / hop`, which for 8000 samples gives
**50** — one fewer than the 51 frames the centred STFT actually produces. The
reference therefore takes its per-bin mean and variance over the first 50 frames
(with a `49` denominator), scales all 51 by them, and then **zeroes the last one**
through the attention mask.

Missing this is not a boundary curiosity. Using 51 frames for the statistics
shifts every bin of every frame by roughly 0.1 in normalised space — around 10% —
while looking entirely plausible. It was caught only by comparing element for
element against the reference output.

## ONNX graph contract

Read from the q4f16 graphs themselves (opset 21, IR 10). The weights live in
separate `.onnx_data` shards, so the graph protobufs can be fetched for ~1.6 MB
without the 1.5 GB of parameters — worth doing before writing any session code.

### Encoder — `encoder_model_q4f16.onnx`

| Direction | Name | Type | Shape |
| --------- | ---- | ---- | ----- |
| in | `input_features` | f32 | `[batch, sequence_length, 128]` |
| out | `last_hidden_state` | f32 | `[batch, encoder_sequence_length, 1024]` |

The encoder output is **1024-wide, not 1280**: `d_model` 1280 is internal and a
head projects to 1024 before it leaves the graph. Feed it the frames from
`cohere_fbank`, unnormalised frames included — the mask has already zeroed them.

### Decoder — `decoder_model_merged_q4f16.onnx`

One graph serves both passes. Inputs:

| Name | Type | Shape |
| ---- | ---- | ----- |
| `input_ids` | i64 | `[batch, sequence_length]` |
| `attention_mask` | i64 | `[batch, total_sequence_length]` |
| `position_ids` | i64 | `[batch, sequence_length]` |
| `num_logits_to_keep` | i64 | `[]` (scalar) |
| `encoder_hidden_states` | **f32** | `[batch, encoder_sequence_length, 1024]` |
| `past_key_values.{0..7}.decoder.{key,value}` | **f16** | `[batch, 8, past_decoder_sequence_length, 128]` |
| `past_key_values.{0..7}.encoder.{key,value}` | **f16** | `[batch, 8, past_encoder_sequence_length, 128]` |

Outputs: `logits` **f16** `[batch, num_logits_to_keep, 16384]`, plus 32
`present.{0..7}.{decoder,encoder}.{key,value}` tensors — 33 in total.

Two things to get right:

- **The cache is f16 while `encoder_hidden_states` is f32.** This variant is
  mixed precision, so the `half` crate is needed for the cache and the logits;
  do not assume one element type across the graph.
- **Encoder K/V must be threaded through, not recomputed.** On the first call the
  `past_key_values.N.encoder.*` inputs are empty and the graph projects the
  encoder output itself; the resulting `present.N.encoder.*` must be fed back on
  every later step. Recomputing them per token is correct but throws away most of
  the speed, and on a 48-layer encoder output that is the whole budget.

The prefix-fill pass passes the full 6-token prefix with empty decoder past and
`num_logits_to_keep = 1`; each incremental step passes one token with the full
past. `position_ids` continue from the prefix length.

## Remaining work

1. ~~`cohere` cargo feature.~~ Done: `cohere = ["dep:rustfft"]`.
2. ~~`cohere_fbank.rs` — the algorithm above, plus its reference-comparison
   test.~~ Done and verified against `CohereAsrFeatureExtractor` element for
   element: worst deviation 4.2e-4 on a synthetic chirp and 2.5e-5 on real
   speech, both within float32 STFT ordering noise. Two fixtures live in
   `src-tauri/fixtures/`; regenerate them with the reference env described above.
   The module is `#![allow(dead_code)]` until step 3 consumes it.
3. ~~Read the ONNX graph I/O contract.~~ Done, above — no assumptions left to
   make about names, shapes or element types.
4. `ort` + `tokenizers` + `half` behind the same feature, and the two sessions.
3. `cohere.rs` — two ONNX sessions. The decoder is *merged*: one graph serves the
   prefix-fill pass (empty past, multi-token input) and incremental generation
   (full past, single token). Encoder K/V is projected on the first decoder call
   and reused via `past_key_values.N.encoder.{key,value}`, so it must not be
   recomputed per step. Confirm the real input/output names by inspecting the
   graph rather than assuming.
5. Model download: five precisions, external data shards, sha256 per file. The
   existing resumable-download path already handles large archives.
6. Engine registration, settings UI, and a benchmark before advertising a speed.

Voxtype's `src/transcribe/cohere.rs` is a working reference implementation for
step 4. Its doc comment claims the K/V cache is F32; in the q4f16 export it is
**F16**, so that shape is per-variant and should be read from the graph rather
than copied. Its 1024-wide encoder output is correct.
