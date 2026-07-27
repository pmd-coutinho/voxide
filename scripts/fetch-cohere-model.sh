#!/usr/bin/env bash
# Fetches the q4f16 Cohere Transcribe variant into the app's models directory,
# then records a sha256 manifest so the in-app downloader can verify against it.
set -euo pipefail
REPO=onnx-community/cohere-transcribe-03-2026-ONNX
BASE="https://huggingface.co/$REPO/resolve/main"
DEST="$HOME/.local/share/voxide/models/cohere-transcribe-03-2026-q4f16"
mkdir -p "$DEST/onnx"
FILES=(
  onnx/encoder_model_q4f16.onnx
  onnx/encoder_model_q4f16.onnx_data
  onnx/decoder_model_merged_q4f16.onnx
  onnx/decoder_model_merged_q4f16.onnx_data
  tokenizer.json
  tokenizer_config.json
  config.json
  generation_config.json
  preprocessor_config.json
)
for f in "${FILES[@]}"; do
  out="$DEST/$f"
  # --continue-at resumes a partial file, which matters for the 1.4 GB shard.
  curl -sL --fail --retry 5 --retry-delay 3 --continue-at - "$BASE/$f" -o "$out" \
    || { echo "FAILED: $f" >&2; exit 1; }
  printf '%-46s %10s bytes\n' "$f" "$(stat -c%s "$out")"
done
( cd "$DEST" && sha256sum "${FILES[@]}" > sha256-manifest.txt )
echo "manifest written to $DEST/sha256-manifest.txt"
du -sh "$DEST"
