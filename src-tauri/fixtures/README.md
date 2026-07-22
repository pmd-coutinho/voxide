# Audio test fixtures

spoken-a-espeak-ng-8khz.wav.b64 is a short synthetic spoken-vowel WAV,
generated locally with espeak-ng and downsampled to mono 8 kHz PCM. It is
stored as base64 text so it can be reviewed in a source-only patch.

The corresponding media test verifies the built-in WAV path without requiring
FFmpeg or a text-to-speech installation at test time. It is a decoding fixture,
not an ASR-accuracy or WER corpus.
