#!/usr/bin/env python3
"""Parakeet acoustic pronunciation-enrollment feasibility spike (route B).

THROWAWAY EXPERIMENT — not wired into the build. It answers one question:
can we reproduce FluidVoice's speaker-adaptive pronunciation enrollment for
Parakeet on Voxide's stack, WITHOUT loading a full PyTorch/NeMo model, by
reusing the int8 encoder ONNX we already ship?

It proves the chain end to end:
  1. run the SHIPPED encoder.int8.onnx directly via onnxruntime (CPU is fine),
  2. feed it a NeMo-style 128-bin log-mel frontend built here in Python
     (this is the fidelity risk the spike exists to probe),
  3. pool a per-word encoder embedding over a frame span (FluidVoice's
     `PronunciationEmbeddingMatcher.embedding(from:frameRange:)`),
  4. match that embedding across clips by cosine, and localize it inside a
     longer utterance with a sliding window (FluidVoice's `bestMatches`).

Interpretation: if the enrolled word scores clearly higher on a clip that
CONTAINS it than on one that does not — and the best window lands at the right
time — the acoustic-enrollment idea is viable on this stack and the remaining
work is plumbing (store, UI, decode-time integration, text substitution). If
separation is weak, the Python mel frontend needs to match NeMo's preprocessor
more exactly (use nemo_toolkit's AudioToMelSpectrogramPreprocessor) before
committing.

Usage:
  # self-contained: synthesizes clips with espeak-ng
  python parakeet_pronunciation_spike.py --word kubernetes \
      --carrier "let us deploy the service on {word} tomorrow" \
      --distractor docker

  # or bring your own recordings (16 kHz mono wav recommended):
  python parakeet_pronunciation_spike.py --word kubernetes \
      --enroll enroll.wav --probe sentence_with_word.wav --negative other.wav

Deps (throwaway venv): numpy librosa soundfile onnxruntime
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.signal.windows import hann

# --- NeMo FastConformer / Parakeet TDT 0.6b v2 frontend parameters ----------
# Read from the model's baked metadata_props (feat_dim=128, subsampling=8) plus
# the standard NeMo AudioToMelSpectrogramPreprocessor config for this family.
# These are the values under test for fidelity.
SR = 16_000
N_FFT = 512
WIN_LENGTH = 400          # 25 ms
HOP_LENGTH = 160          # 10 ms  -> mel frame stride 10 ms
N_MELS = 128
PREEMPH = 0.97
SUBSAMPLING = 8           # encoder downsamples 8x -> 80 ms per encoder frame
LOG_ZERO_GUARD = 2.0 ** -24
NORM_EPS = 1e-5
FRAME_DURATION = HOP_LENGTH * SUBSAMPLING / SR   # 0.08 s / encoder frame


# --- Slaney mel filterbank (librosa's default; NeMo uses the same) ----------
# Hand-rolled in numpy so the spike needs no librosa/numba, whose old wheels
# don't build on recent Pythons. This is a faithful port of librosa.filters.mel
# with htk=False, norm="slaney".
def _hz_to_mel(freq) -> np.ndarray:
    freq = np.array(freq, dtype=float, ndmin=1)  # copy -> writable, always 1-D
    f_sp = 200.0 / 3
    mel = freq / f_sp
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    region = freq >= min_log_hz
    mel[region] = min_log_mel + np.log(freq[region] / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel) -> np.ndarray:
    mel = np.array(mel, dtype=float, ndmin=1)
    f_sp = 200.0 / 3
    freq = f_sp * mel
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    region = mel >= min_log_mel
    freq[region] = min_log_hz * np.exp(logstep * (mel[region] - min_log_mel))
    return freq


def _mel_filterbank() -> np.ndarray:
    n_freqs = N_FFT // 2 + 1
    fft_freqs = np.linspace(0.0, SR / 2, n_freqs)
    hz_pts = _mel_to_hz(
        np.linspace(_hz_to_mel(0.0)[0], _hz_to_mel(SR / 2)[0], N_MELS + 2)
    )
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    fb = np.zeros((N_MELS, n_freqs))
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    fb *= (2.0 / (hz_pts[2 : N_MELS + 2] - hz_pts[:N_MELS]))[:, None]  # Slaney norm
    return fb


MEL_FB = _mel_filterbank()


def _load(wav_path: Path) -> np.ndarray:
    rate, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if rate != SR:
        data = resample_poly(data, SR, rate).astype(np.float32)
    return data


def _stft_power(y: np.ndarray) -> np.ndarray:
    """center=True STFT power spectrum, matching librosa/NeMo windowing."""
    win = hann(WIN_LENGTH, sym=False)                 # periodic Hann
    pad = (N_FFT - WIN_LENGTH) // 2
    window = np.zeros(N_FFT, dtype=np.float32)
    window[pad : pad + WIN_LENGTH] = win               # 400-tap window centered in 512
    y = np.pad(y, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(y) - N_FFT) // HOP_LENGTH
    frames = np.stack(
        [y[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] for i in range(n_frames)], axis=1
    )
    spec = np.fft.rfft(frames * window[:, None], axis=0)
    return np.abs(spec) ** 2.0                          # [n_freqs, T]


def log_mel(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """NeMo-style 128-bin per-feature-normalized log-mel.

    Returns (features[128, T] float32, per_frame_energy[T]). Energy is the
    pre-normalization mean log-mel per frame, used only to trim silence when
    picking the enrolled word's frame span.
    """
    y = _load(wav_path)
    if y.size == 0:
        raise ValueError(f"{wav_path} is empty")
    y = np.append(y[0], y[1:] - PREEMPH * y[:-1]).astype(np.float32)  # preemphasis
    mel = MEL_FB @ _stft_power(y)             # [128, T]
    logmel = np.log(mel + LOG_ZERO_GUARD)
    energy = logmel.mean(axis=0)              # [T], pre-norm
    mean = logmel.mean(axis=1, keepdims=True)
    std = logmel.std(axis=1, keepdims=True)
    feats = (logmel - mean) / (std + NORM_EPS)  # per_feature normalization
    return feats.astype(np.float32), energy


def encode(session: ort.InferenceSession, feats: np.ndarray) -> np.ndarray:
    """Run the encoder ONNX. Returns encoder features [frames, d_model]."""
    inputs = {i.name: i for i in session.get_inputs()}
    # audio_signal: [B, feat_dim, T]; length: [B] number of mel frames.
    signal_name = next(n for n in inputs if "signal" in n or inputs[n].shape[-1] != 1)
    length_name = next(n for n in inputs if n != signal_name)
    feed = {
        signal_name: feats[np.newaxis, :, :],                 # [1, 128, T]
        length_name: np.array([feats.shape[1]], dtype=np.int64),
    }
    outputs = session.run(None, feed)
    enc = next(o for o in outputs if o.ndim == 3)             # [1, d_model, frames]
    lengths = next((o for o in outputs if o.ndim == 1), None)
    enc = enc[0].T                                            # [frames, d_model]
    if lengths is not None:
        enc = enc[: int(lengths[0])]
    return enc.astype(np.float32)


def voiced_encoder_span(energy: np.ndarray) -> tuple[int, int]:
    """Voiced mel-frame range -> encoder-frame range (÷ subsampling)."""
    lo, hi = energy.min(), energy.max()
    thresh = lo + 0.35 * (hi - lo)
    voiced = np.where(energy > thresh)[0]
    if voiced.size == 0:
        start, end = 0, energy.size
    else:
        start, end = int(voiced[0]), int(voiced[-1]) + 1
    return start // SUBSAMPLING, max(start // SUBSAMPLING + 1, end // SUBSAMPLING)


def pooled(enc: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean-pool a frame span into one L2-normalized embedding."""
    start = max(0, min(start, enc.shape[0] - 1))
    end = max(start + 1, min(end, enc.shape[0]))
    v = enc[start:end].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def best_window(enroll_emb: np.ndarray, probe: np.ndarray, width: int):
    """Slide the enrolled embedding over probe frames; best cosine + location."""
    width = max(1, min(width, probe.shape[0]))
    best_cos, best_start = -1.0, 0
    for s in range(0, probe.shape[0] - width + 1):
        cos = float(enroll_emb @ pooled(probe, s, s + width))
        if cos > best_cos:
            best_cos, best_start = cos, s
    return best_cos, best_start


def espeak(text: str, out: Path) -> None:
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(out), text],
        check=True,
        capture_output=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    default_model = (
        Path.home()
        / ".local/share/voxide/models/parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx"
    )
    ap.add_argument("--encoder", type=Path, default=default_model)
    ap.add_argument("--word", default="kubernetes")
    ap.add_argument("--carrier", default="let us deploy the service on {word} tomorrow")
    ap.add_argument("--distractor", default="docker")
    ap.add_argument("--enroll", type=Path)
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--negative", type=Path)
    args = ap.parse_args()

    if not args.encoder.is_file():
        print(f"encoder ONNX not found: {args.encoder}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="parakeet-spike-"))
    if args.enroll and args.probe and args.negative:
        enroll_wav, probe_wav, neg_wav = args.enroll, args.probe, args.negative
    else:
        try:
            enroll_wav, probe_wav, neg_wav = tmp / "e.wav", tmp / "p.wav", tmp / "n.wav"
            espeak(args.word, enroll_wav)
            espeak(args.carrier.format(word=args.word), probe_wav)
            espeak(args.carrier.format(word=args.distractor), neg_wav)
            print(f"synthesized clips with espeak-ng in {tmp}\n")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(
                f"espeak-ng unavailable ({exc}); pass --enroll/--probe/--negative wavs.",
                file=sys.stderr,
            )
            return 2

    print(f"encoder: {args.encoder}")
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        str(args.encoder), providers=["CPUExecutionProvider"]
    )
    print(f"session load: {time.perf_counter() - t0:.2f}s  "
          f"inputs={[i.name for i in session.get_inputs()]}  "
          f"outputs={[o.name for o in session.get_outputs()]}\n")

    def features(wav: Path):
        feats, energy = log_mel(wav)
        t = time.perf_counter()
        enc = encode(session, feats)
        dt = time.perf_counter() - t
        return feats, energy, enc, dt

    e_feats, e_energy, e_enc, e_dt = features(enroll_wav)
    p_feats, _, p_enc, p_dt = features(probe_wav)
    n_feats, _, n_enc, n_dt = features(neg_wav)

    d_model = e_enc.shape[1]
    print(f"encoder output d_model={d_model}  frame_stride={FRAME_DURATION*1000:.0f}ms")
    print(f"encoder fwd latency (CPU): enroll {e_dt*1000:.0f}ms  "
          f"probe {p_dt*1000:.0f}ms  negative {n_dt*1000:.0f}ms\n")

    v_start, v_end = voiced_encoder_span(e_energy)
    enroll_emb = pooled(e_enc, v_start, v_end)
    width = v_end - v_start
    print(f"enrolled '{args.word}': voiced encoder frames [{v_start}:{v_end}] "
          f"(~{width*FRAME_DURATION:.2f}s), embedding dim {enroll_emb.size}\n")

    p_cos, p_at = best_window(enroll_emb, p_enc, width)
    n_cos, n_at = best_window(enroll_emb, n_enc, width)
    margin = p_cos - n_cos

    print("=== RESULT ===")
    print(f"probe    (contains '{args.word}'): best cosine {p_cos:.3f} "
          f"@ {p_at*FRAME_DURATION:.2f}s")
    print(f"negative ('{args.distractor}' instead): best cosine {n_cos:.3f} "
          f"@ {n_at*FRAME_DURATION:.2f}s")
    print(f"separation margin: {margin:+.3f}")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nprocess peak RSS: {peak_mb:.0f} MB (encoder session + onnxruntime, CPU)")

    # The SEPARATION MARGIN is the discriminator that matters, not the absolute
    # cosine (which rises with the exact NeMo frontend, >=3 averaged
    # enrollments, and real speech). A clear positive margin means the acoustic
    # embedding tells the enrolled word apart from a distractor.
    verdict = (
        f"PROMISING — margin {margin:+.3f}; the enrolled word's acoustic "
        "embedding clearly separates the clip that contains it from the one "
        "that does not. Acoustic enrollment is viable on this stack."
        if margin >= 0.15 and p_cos >= 0.45
        else "INCONCLUSIVE — weak separation; try nemo_toolkit's exact "
        "preprocessor for the mel frontend, better pooling, or real speech."
    )
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
