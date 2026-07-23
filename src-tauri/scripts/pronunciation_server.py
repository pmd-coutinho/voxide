#!/usr/bin/env python3
"""JSON-lines service for acoustic pronunciation enrollment (Parakeet).

The desktop process owns microphone capture and sends 16 kHz mono float32 PCM
to this process. This service runs the SHIPPED Parakeet `encoder.int8.onnx`
directly via onnxruntime + a NeMo-style 128-bin log-mel frontend, and computes
acoustic word embeddings the high-level sherpa recognizer never exposes. It is
deliberately lightweight (onnxruntime + numpy + scipy — no torch/NeMo) and
reuses the encoder already installed for the Parakeet engine.

Protocol (one JSON object per stdin line):

  ping   { requestId, protocolVersion }
  enroll { requestId, audio }                        # base64 float32 LE PCM
  match  { requestId, audio, prototypes:[{label, values:[f32], frames}] }
  shutdown { requestId }

Responses (one per request, echoing requestId):

  ready    { type, modelId, protocolVersion, hiddenSize?, frameDurationMs, provider }
  enrolled { type, embedding:[f32], hiddenSize, frames }
  matched  { type, frameDurationMs, matches:[{label, startTime, endTime, score}] }
  stopped  { type }
  error    { type, code, message }

Every request produces exactly one JSON response on stdout; diagnostics go only
to stderr so stdout stays machine-readable.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- protocol -------------------------------------------------------------
PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
SAMPLE_RATE = 16_000

# --- NeMo FastConformer / Parakeet TDT 0.6b v2 frontend (from model metadata:
# feat_dim=128, subsampling=8). Validated by the pronunciation spike. ------
N_FFT = 512
WIN_LENGTH = 400          # 25 ms
HOP_LENGTH = 160          # 10 ms mel-frame stride
N_MELS = 128
PREEMPH = 0.97
SUBSAMPLING = 8           # encoder downsamples 8x -> 80 ms per encoder frame
LOG_ZERO_GUARD = 2.0 ** -24
NORM_EPS = 1e-5
FRAME_DURATION = HOP_LENGTH * SUBSAMPLING / SAMPLE_RATE  # 0.08 s / encoder frame


def respond(payload: dict[str, Any], request_id: int | None = None) -> None:
    if request_id is not None:
        payload = {**payload, "requestId": request_id}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def request_id_from(request: dict[str, Any]) -> int:
    request_id = request.get("requestId")
    if type(request_id) is not int or request_id < 1:
        raise ValueError("Pronunciation requestId must be a positive integer")
    return request_id


def validate_request(request: dict[str, Any]) -> tuple[str, int]:
    action = request.get("action")
    if not isinstance(action, str):
        raise ValueError("Pronunciation action must be a string")
    request_id = request_id_from(request)
    allowed = {
        "ping": {"action", "requestId", "protocolVersion"},
        "enroll": {"action", "requestId", "audio"},
        "match": {"action", "requestId", "audio", "prototypes"},
        "shutdown": {"action", "requestId"},
    }
    if action not in allowed:
        raise ValueError("Unknown pronunciation runtime action")
    if set(request) - allowed[action]:
        raise ValueError("Pronunciation request contains unsupported fields")
    if action == "ping" and request.get("protocolVersion") != PROTOCOL_VERSION:
        raise ValueError("Incompatible pronunciation protocol version")
    if action in {"enroll", "match"} and not isinstance(request.get("audio"), str):
        raise ValueError("Pronunciation audio must be a base64 string")
    if action == "match" and not isinstance(request.get("prototypes"), list):
        raise ValueError("Pronunciation prototypes must be a list")
    return action, request_id


def safe_error_payload(error: Exception) -> dict[str, str]:
    message = str(error).lower()
    if "out of memory" in message:
        return {"type": "error", "code": "gpu_out_of_memory",
                "message": "The GPU ran out of memory during pronunciation matching."}
    if isinstance(error, ValueError):
        return {"type": "error", "code": "invalid_request",
                "message": "The pronunciation service received an invalid request."}
    if "cuda" in message or "nvidia" in message or "onnxruntime" in message:
        return {"type": "error", "code": "runtime_failure",
                "message": "The pronunciation runtime failed. Reinstall pronunciation support."}
    return {"type": "error", "code": "matching_failed",
            "message": "The pronunciation service could not complete the request."}


def pcm_from_base64(encoded: str | None) -> np.ndarray:
    if not encoded:
        return np.empty(0, dtype=np.float32)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as error:  # noqa: BLE001 -- return a protocol error to Rust.
        raise ValueError(f"Audio payload is not valid base64: {error}") from error
    if len(raw) % 4:
        raise ValueError("Audio payload is not 32-bit float PCM")
    return np.frombuffer(raw, dtype="<f4").copy()


# --- Slaney mel filterbank (librosa default; NeMo uses the same). Hand-rolled
# in numpy so the runtime needs no librosa/numba. ------------------------
def _hz_to_mel(freq) -> np.ndarray:
    freq = np.array(freq, dtype=float, ndmin=1)
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
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2, n_freqs)
    hz_pts = _mel_to_hz(
        np.linspace(_hz_to_mel(0.0)[0], _hz_to_mel(SAMPLE_RATE / 2)[0], N_MELS + 2)
    )
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    fb = np.zeros((N_MELS, n_freqs))
    for i in range(N_MELS):
        fb[i] = np.maximum(0.0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    fb *= (2.0 / (hz_pts[2 : N_MELS + 2] - hz_pts[:N_MELS]))[:, None]  # Slaney norm
    return fb


MEL_FB = _mel_filterbank()


def _stft_power(y: np.ndarray) -> np.ndarray:
    from scipy.signal.windows import hann

    win = hann(WIN_LENGTH, sym=False)
    pad = (N_FFT - WIN_LENGTH) // 2
    window = np.zeros(N_FFT, dtype=np.float32)
    window[pad : pad + WIN_LENGTH] = win
    y = np.pad(y, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(y) - N_FFT) // HOP_LENGTH
    frames = np.stack(
        [y[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] for i in range(n_frames)], axis=1
    )
    return np.abs(np.fft.rfft(frames * window[:, None], axis=0)) ** 2.0


def log_mel(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NeMo-style 128-bin per-feature-normalized log-mel from 16 kHz mono PCM.
    Returns (features[128, T] float32, per_frame_energy[T])."""
    if samples.size == 0:
        raise ValueError("Audio is empty")
    y = np.append(samples[0], samples[1:] - PREEMPH * samples[:-1]).astype(np.float32)
    mel = MEL_FB @ _stft_power(y)
    logmel = np.log(mel + LOG_ZERO_GUARD)
    energy = logmel.mean(axis=0)
    feats = (logmel - logmel.mean(axis=1, keepdims=True)) / (logmel.std(axis=1, keepdims=True) + NORM_EPS)
    return feats.astype(np.float32), energy


def voiced_encoder_span(energy: np.ndarray) -> tuple[int, int]:
    lo, hi = energy.min(), energy.max()
    voiced = np.where(energy > lo + 0.35 * (hi - lo))[0]
    if voiced.size == 0:
        start, end = 0, energy.size
    else:
        start, end = int(voiced[0]), int(voiced[-1]) + 1
    return start // SUBSAMPLING, max(start // SUBSAMPLING + 1, end // SUBSAMPLING)


def pooled(enc: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, min(start, enc.shape[0] - 1))
    end = max(start + 1, min(end, enc.shape[0]))
    v = enc[start:end].mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


class EncoderRuntime:
    """Lazily loads encoder.int8.onnx; CUDA EP if available, else CPU."""

    def __init__(self, model_dir: Path) -> None:
        self._encoder = model_dir / "encoder.int8.onnx"
        self._session = None
        self._signal = None
        self._length = None
        self.provider = "unloaded"

    def ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if not self._encoder.is_file():
            raise ValueError(f"Parakeet encoder is missing at {self._encoder}")
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(str(self._encoder), providers=providers)
        self.provider = self._session.get_providers()[0]
        names = {i.name: i for i in self._session.get_inputs()}
        self._signal = next(n for n in names if "signal" in n or names[n].shape[-1] != 1)
        self._length = next(n for n in names if n != self._signal)

    def _run(self, feats: np.ndarray) -> np.ndarray:
        """Run the encoder on precomputed mel features -> [frames, hidden]."""
        self.ensure_loaded()
        outputs = self._session.run(
            None,
            {self._signal: feats[np.newaxis, :, :],
             self._length: np.array([feats.shape[1]], dtype=np.int64)},
        )
        enc = next(o for o in outputs if o.ndim == 3)[0].T          # [frames, hidden]
        lengths = next((o for o in outputs if o.ndim == 1), None)
        if lengths is not None:
            enc = enc[: int(lengths[0])]
        return enc.astype(np.float32)

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """Encoder features [frames, hidden] for a whole clip (match path)."""
        feats, _energy = log_mel(samples)
        return self._run(feats)

    def embedding(self, samples: np.ndarray) -> tuple[np.ndarray, int]:
        """Pool the voiced span into one embedding; returns (vector, frames)."""
        feats, energy = log_mel(samples)
        enc = self._run(feats)
        start, end = voiced_encoder_span(energy)
        return pooled(enc, start, end), end - start


def best_window(prototype: np.ndarray, enc: np.ndarray, width: int) -> tuple[float, int]:
    width = max(1, min(width, enc.shape[0]))
    best_cos, best_start = -1.0, 0
    for s in range(0, enc.shape[0] - width + 1):
        cos = float(prototype @ pooled(enc, s, s + width))
        if cos > best_cos:
            best_cos, best_start = cos, s
    return best_cos, best_start


def handle_match(runtime: EncoderRuntime, request: dict[str, Any]) -> dict[str, Any]:
    enc = runtime.encode(pcm_from_base64(request["audio"]))
    matches = []
    for proto in request["prototypes"]:
        if not isinstance(proto, dict):
            raise ValueError("Each prototype must be an object")
        label = proto.get("label")
        values = proto.get("values")
        frames = proto.get("frames")
        if not isinstance(label, str) or not isinstance(values, list) or type(frames) is not int:
            raise ValueError("Prototype requires label:str, values:[f32], frames:int")
        vec = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        score, start = best_window(vec, enc, frames)
        matches.append({
            "label": label,
            "startTime": round(start * FRAME_DURATION, 4),
            "endTime": round((start + max(1, min(frames, enc.shape[0]))) * FRAME_DURATION, 4),
            "score": round(score, 4),
        })
    return {"type": "matched", "frameDurationMs": round(FRAME_DURATION * 1000), "matches": matches}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = EncoderRuntime(args.model_dir)
    stdin = sys.stdin.buffer
    while raw_line := stdin.readline(MAX_MESSAGE_BYTES + 1):
        request_id: int | None = None
        try:
            if len(raw_line) > MAX_MESSAGE_BYTES:
                raise ValueError("Pronunciation request exceeds the service safety limit")
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("Pronunciation request must be a JSON object")
            action, request_id = validate_request(request)
            if action == "ping":
                respond({"type": "ready", "modelId": args.model_dir.name,
                         "protocolVersion": PROTOCOL_VERSION,
                         "frameDurationMs": round(FRAME_DURATION * 1000)}, request_id)
            elif action == "enroll":
                embedding, frames = runtime.embedding(pcm_from_base64(request["audio"]))
                respond({"type": "enrolled", "embedding": embedding.tolist(),
                         "hiddenSize": int(embedding.size), "frames": int(frames)}, request_id)
            elif action == "match":
                respond(handle_match(runtime, request), request_id)
            elif action == "shutdown":
                respond({"type": "stopped"}, request_id)
                return 0
        except Exception as error:  # noqa: BLE001 -- never crash on a bad request.
            payload = safe_error_payload(error)
            sys.stderr.write(f"pronunciation service error: {payload['code']}\n")
            sys.stderr.flush()
            respond(payload, request_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
