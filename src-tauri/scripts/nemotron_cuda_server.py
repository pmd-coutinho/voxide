#!/usr/bin/env python3
"""JSON-lines CUDA service for NVIDIA Nemotron 3.5 ASR streaming.

The desktop process owns microphone capture and sends 16 kHz mono float32 PCM
to this process.  Keeping PyTorch and the model here isolates the large CUDA
runtime from the Rust application while preserving Nemotron's native,
cache-aware streaming path.

Protocol (one JSON object per stdin line):

  ping { requestId, protocolVersion }
  start { requestId, language, lookaheadTokens }
  append { requestId, audio }            # base64 float32 little-endian PCM
  finish { requestId, audio? }
  transcribe { requestId, audio, language }
  shutdown { requestId }

Every request produces exactly one JSON response with the same request ID on
stdout. Diagnostics are written only to stderr so stdout remains machine-readable.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Iterator


MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16_000
PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
# The model accepts 0, 3, 6, or 13 lookahead tokens. NVIDIA's streaming
# example uses six (560 ms), a practical accuracy/latency default for desktop
# dictation; Voxide can request a different supported profile per session.
DEFAULT_LOOKAHEAD_TOKENS = 6


def respond(payload: dict[str, Any], request_id: int | None = None) -> None:
    """Write a protocol message without ever contaminating stdout."""
    if request_id is not None:
        payload = {**payload, "requestId": request_id}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def request_id_from(request: dict[str, Any]) -> int:
    request_id = request.get("requestId")
    if type(request_id) is not int or request_id < 1:
        raise ValueError("Nemotron requestId must be a positive integer")
    return request_id


def required_string(request: dict[str, Any], field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Nemotron {field} must be a string")
    return value


def optional_audio(request: dict[str, Any]) -> str | None:
    audio = request.get("audio")
    if audio is not None and not isinstance(audio, str):
        raise ValueError("Nemotron audio must be a base64 string")
    return audio


def validate_request(request: dict[str, Any]) -> tuple[str, int]:
    """Enforce the versioned JSON-line request schema before model access."""
    action = required_string(request, "action")
    request_id = request_id_from(request)
    allowed_fields = {
        "ping": {"action", "requestId", "protocolVersion"},
        "start": {"action", "requestId", "language", "lookaheadTokens"},
        "append": {"action", "requestId", "audio"},
        "finish": {"action", "requestId", "audio"},
        "transcribe": {"action", "requestId", "audio", "language"},
        "abort": {"action", "requestId"},
        "shutdown": {"action", "requestId"},
    }
    if action not in allowed_fields:
        raise ValueError("Unknown Nemotron runtime action")
    if set(request) - allowed_fields[action]:
        raise ValueError("Nemotron request contains unsupported fields")
    if action == "ping" and request.get("protocolVersion") != PROTOCOL_VERSION:
        raise ValueError("Incompatible Nemotron protocol version")
    if action in {"start", "transcribe"}:
        required_string(request, "language")
    if action in {"append", "finish", "transcribe"}:
        optional_audio(request)
    if action == "start":
        lookahead = request.get("lookaheadTokens")
        if type(lookahead) is not int or lookahead not in {0, 3, 6, 13}:
            raise ValueError("Nemotron lookaheadTokens must be one of 0, 3, 6, or 13")
    return action, request_id


def safe_error_payload(error: Exception) -> dict[str, str]:
    """Map failures to supportable categories without logging request content."""
    message = str(error).lower()
    if "out of memory" in message:
        return {
            "type": "error",
            "code": "gpu_out_of_memory",
            "message": "The GPU ran out of memory while transcribing.",
        }
    if isinstance(error, TimeoutError):
        return {
            "type": "error",
            "code": "inference_timeout",
            "message": "Nemotron did not finish transcription before its deadline.",
        }
    if isinstance(error, ValueError):
        return {
            "type": "error",
            "code": "invalid_request",
            "message": "The Nemotron service received an invalid request.",
        }
    if "cuda" in message or "nvidia" in message:
        return {
            "type": "error",
            "code": "cuda_runtime_failure",
            "message": "The Nemotron CUDA runtime failed. Verify the NVIDIA driver and CUDA runtime.",
        }
    return {
        "type": "error",
        "code": "inference_failed",
        "message": "The Nemotron CUDA service could not complete transcription.",
    }


def pcm_from_base64(encoded: str | None):
    import numpy as np

    if not encoded:
        return np.empty(0, dtype=np.float32)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as error:  # noqa: BLE001 -- return a protocol error to Rust.
        raise ValueError(f"Audio payload is not valid base64: {error}") from error
    if len(raw) % 4:
        raise ValueError("Audio payload is not 32-bit float PCM")
    return np.frombuffer(raw, dtype="<f4").copy()


class StreamingSession:
    """Feeds exact cache-aware chunks to one running RNN-T generation call."""

    def __init__(self, runtime: "NemotronRuntime", language: str, lookahead_tokens: int) -> None:
        import numpy as np

        self.runtime = runtime
        self.language = language
        self.lookahead_tokens = lookahead_tokens
        self.first_chunk = True
        self.audio = np.empty(0, dtype=np.float32)
        self.features: queue.Queue[Any | None] = queue.Queue()
        self.text = ""
        self.error: Exception | None = None
        self.finished = False
        self.generation_done = threading.Event()
        self.text_done = threading.Event()
        self._start_generation()

    def _required_samples(self) -> int:
        if self.first_chunk:
            return self.runtime.processor.num_samples_first_audio_chunk
        return self.runtime.processor.num_samples_per_audio_chunk

    def _advance_after_chunk(self, was_first_chunk: bool) -> None:
        """Keep the raw-STFT overlap needed by the next streaming feature block.

        The model caches encoder state, but the frontend still needs overlapping
        waveform around the FFT window. Treating `num_samples_*_audio_chunk`
        as disjoint PCM blocks drops samples at every boundary and makes the
        features drift from a normal full-audio extraction. The processor's
        frame counts specify the actual advance; retain the leading overlap for
        the next `center=False` block.
        """
        processor = self.runtime.processor
        extractor = processor.feature_extractor
        if was_first_chunk:
            advance = (
                processor.num_mel_frames_first_audio_chunk * extractor.hop_length
                - extractor.n_fft // 2
            )
        else:
            advance = processor.num_mel_frames_per_audio_chunk * extractor.hop_length
        self.audio = self.audio[max(0, advance):]

    def _to_features(self, audio, first_chunk: bool):
        features = self.runtime.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=first_chunk,
            language=self.language,
            return_tensors="pt",
        )
        # The first centered STFT call includes one trailing all-zero frame in
        # its tensor shape. The processor's attention mask marks the exact
        # valid count; streaming `generate` requires that exact shape rather
        # than the padded tensor extent.
        valid_frames = int(features.attention_mask[0].sum().item())
        return features.input_features[:, :valid_frames]

    def _input_features(self) -> Iterator[Any]:
        while True:
            chunk = self.features.get()
            if chunk is None:
                return
            yield chunk

    def _start_generation(self) -> None:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.runtime.processor.tokenizer,
            skip_special_tokens=True,
        )

        def generate() -> None:
            try:
                self.runtime.model.generate(
                    input_features=self._input_features(),
                    prompt_ids=self.runtime.prompt_ids(self.language),
                    num_lookahead_tokens=self.lookahead_tokens,
                    streamer=streamer,
                )
            except Exception as error:  # noqa: BLE001 -- surface across the protocol boundary.
                self.error = error
            finally:
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001 -- the reader will see the generation exception.
                    pass
                self.generation_done.set()

        def collect_text() -> None:
            try:
                for fragment in streamer:
                    self.text += fragment
            except Exception as error:  # noqa: BLE001 -- timeout/stream errors are terminal.
                if self.error is None:
                    self.error = error
            finally:
                self.text_done.set()

        threading.Thread(target=generate, name="nemotron-generate", daemon=True).start()
        threading.Thread(target=collect_text, name="nemotron-text", daemon=True).start()

    def _enqueue_available(self, final: bool) -> int:
        import numpy as np

        added = 0
        while len(self.audio) >= self._required_samples():
            required = self._required_samples()
            audio_chunk = self.audio[:required]
            was_first_chunk = self.first_chunk
            self.features.put(self._to_features(audio_chunk, was_first_chunk))
            self._advance_after_chunk(was_first_chunk)
            self.first_chunk = False
            added += 1

        if final and (self.first_chunk or len(self.audio)):
            required = self._required_samples()
            audio_chunk = np.pad(self.audio, (0, max(0, required - len(self.audio))))[:required]
            self.audio = np.empty(0, dtype=np.float32)
            self.features.put(self._to_features(audio_chunk, self.first_chunk))
            self.first_chunk = False
            added += 1
        return added

    def append(self, audio) -> str:
        if self.finished:
            raise RuntimeError("The Nemotron streaming session has already finished")
        if len(audio):
            self.audio = __import__("numpy").concatenate((self.audio, audio))
        self._enqueue_available(final=False)
        # `generate()` intentionally blocks in the feature iterator until the
        # next audio chunk arrives.  Returning the transcript accumulated so
        # far keeps the protocol non-blocking; the next 80 ms poll carries any
        # newly emitted stable words to the overlay.
        if self.error is not None:
            raise RuntimeError(f"Nemotron streaming failed: {self.error}") from self.error
        return self.text.strip()

    def finish(self, audio) -> str:
        if self.finished:
            return self.text.strip()
        if len(audio):
            self.audio = __import__("numpy").concatenate((self.audio, audio))
        self._enqueue_available(final=True)
        self.features.put(None)
        if not self.generation_done.wait(90.0):
            raise TimeoutError("Nemotron did not finalize the streaming transcription within 90 seconds")
        self.text_done.wait(5.0)
        if self.error is not None:
            raise RuntimeError(f"Nemotron streaming failed: {self.error}") from self.error
        self.finished = True
        return self.text.strip()

    def abort(self) -> None:
        if self.finished:
            return
        # Drop queued-but-not-decoded chunks so cancellation does not turn
        # into an invisible final transcription. The chunk currently on the
        # GPU, if any, is allowed to complete before the sentinel ends it.
        while True:
            try:
                self.features.get_nowait()
            except queue.Empty:
                break
        self.features.put(None)
        self.generation_done.wait(15.0)
        self.finished = True


class NemotronRuntime:
    def __init__(self, model_directory: Path) -> None:
        self.model_directory = model_directory
        self.model = None
        self.processor = None
        self.torch = None

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        required = ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json")
        missing = [name for name in required if not (self.model_directory / name).is_file()]
        if missing:
            raise RuntimeError(f"Nemotron model is incomplete; missing: {', '.join(missing)}")

        import torch
        from transformers import AutoModelForRNNT, AutoProcessor, logging

        logging.set_verbosity_error()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the Nemotron runtime. Check the NVIDIA driver and CUDA PyTorch installation.")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_directory, local_files_only=True)
        self.model = AutoModelForRNNT.from_pretrained(
            self.model_directory,
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()

    def prompt_ids(self, language: str):
        assert self.processor is not None
        return self.processor._resolve_prompt_ids(language, 1).to("cuda")

    def begin(self, language: str, lookahead_tokens: int) -> StreamingSession:
        self.ensure_loaded()
        assert self.processor is not None
        self.processor.set_num_lookahead_tokens(lookahead_tokens)
        # Validate language before opening a generation thread, and keep the
        # public error concise rather than exposing the full supported map.
        try:
            self.prompt_ids(language)
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"Unsupported Nemotron language '{language}'. Use auto or a supported BCP-47 locale.") from error
        return StreamingSession(self, language, lookahead_tokens)

    def transcribe(self, audio, language: str) -> str:
        self.ensure_loaded()
        assert self.processor is not None and self.model is not None and self.torch is not None
        if not len(audio):
            return ""
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, language=language, return_tensors="pt")
        inputs = {
            name: (
                value.to("cuda", dtype=self.model.dtype)
                if name == "input_features"
                else value.to("cuda")
                if hasattr(value, "to")
                else value
            )
            for name, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs)
        sequences = getattr(generated, "sequences", generated)
        return self.processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voxide Nemotron CUDA JSON-lines server")
    parser.add_argument("--model-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = NemotronRuntime(args.model_dir)
    session: StreamingSession | None = None
    stdin = sys.stdin.buffer
    while raw_line := stdin.readline(MAX_MESSAGE_BYTES + 1):
        request_id: int | None = None
        try:
            if len(raw_line) > MAX_MESSAGE_BYTES:
                raise ValueError("Nemotron request exceeds the service safety limit")
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("Nemotron request must be a JSON object")
            action, request_id = validate_request(request)
            if action == "ping":
                respond(
                    {"type": "ready", "modelId": MODEL_ID, "protocolVersion": PROTOCOL_VERSION},
                    request_id,
                )
            elif action == "start":
                language = required_string(request, "language")
                lookahead_tokens = request["lookaheadTokens"]
                session = runtime.begin(language, lookahead_tokens)
                respond(
                    {
                        "type": "started",
                        "latencyMs": runtime.processor.streaming_latency_ms,
                        "lookaheadTokens": lookahead_tokens,
                    },
                    request_id,
                )
            elif action == "append":
                if session is None:
                    raise RuntimeError("Start a Nemotron streaming session before appending audio")
                text = session.append(pcm_from_base64(optional_audio(request)))
                respond({"type": "partial", "text": text}, request_id)
            elif action == "finish":
                if session is None:
                    raise RuntimeError("Start a Nemotron streaming session before finishing")
                text = session.finish(pcm_from_base64(optional_audio(request)))
                session = None
                respond({"type": "final", "text": text}, request_id)
            elif action == "transcribe":
                language = required_string(request, "language")
                text = runtime.transcribe(pcm_from_base64(optional_audio(request)), language)
                respond({"type": "final", "text": text}, request_id)
            elif action == "abort":
                if session is not None:
                    session.abort()
                    session = None
                respond({"type": "aborted"}, request_id)
            elif action == "shutdown":
                respond({"type": "stopped"}, request_id)
                return 0
        except Exception as error:  # noqa: BLE001 -- never crash the persistent service for a bad request.
            payload = safe_error_payload(error)
            sys.stderr.write(f"Nemotron service error: {payload['code']}\n")
            sys.stderr.flush()
            respond(payload, request_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
