//! Cohere Transcribe: encoder and merged decoder.
//!
//! Front end is [`crate::cohere_fbank`]; the graph contract and the reasoning
//! behind every constant here is in `docs/COHERE_ENGINE.md`.
//!
//! ## The merged decoder
//!
//! One graph serves two different calls. The **prefix fill** passes all six
//! prompt tokens with an empty past and gets back the encoder's projected K/V.
//! Each **incremental step** passes a single token with the full past. Both use
//! `num_logits_to_keep = 1`, since only the last position's distribution is ever
//! needed for greedy decoding.
//!
//! The encoder K/V arrives as `present.N.encoder.*` on the first call and must be
//! threaded back unchanged on every later one. Recomputing it per token is
//! numerically fine and ruinously slow — it is a projection of a 48-layer
//! encoder's output over the whole utterance.
//!
//! Mixed precision is not optional: `encoder_hidden_states` is f32 while the K/V
//! cache and `logits` are f16 in the q4f16 export.

#![allow(dead_code)]

use std::{path::Path, sync::Mutex};

use half::f16;
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

use crate::cohere_fbank::{CohereFbank, MEL_BINS};

/// Decoder layers, and the K/V geometry per layer.
const LAYERS: usize = 8;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
/// Width the encoder graph emits, after its internal 1280-wide model is projected.
const ENCODER_WIDTH: usize = 1024;

/// English transcription, punctuation and inverse text normalisation on, no
/// timestamps, no diarisation. Verified against the tokenizer's `added_tokens`
/// rather than copied from another implementation.
const PREFIX: [i64; 6] = [4, 62, 5, 8, 11, 13];
/// `<|endoftext|>`.
const EOS: i64 = 3;
/// A hard stop so a degenerate loop cannot spin forever on bad audio.
const MAX_NEW_TOKENS: usize = 440;

pub struct CohereTranscriber {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    fbank: CohereFbank,
}

/// One layer's four cache tensors, kept as flat f16 with their sequence lengths.
#[derive(Clone)]
struct LayerCache {
    decoder_key: Vec<f16>,
    decoder_value: Vec<f16>,
    encoder_key: Vec<f16>,
    encoder_value: Vec<f16>,
    decoder_len: usize,
    encoder_len: usize,
}

impl LayerCache {
    /// The first decoder call gets zero-length past on both sides; the graph then
    /// projects the encoder K/V itself.
    fn empty() -> Self {
        Self {
            decoder_key: Vec::new(),
            decoder_value: Vec::new(),
            encoder_key: Vec::new(),
            encoder_value: Vec::new(),
            decoder_len: 0,
            encoder_len: 0,
        }
    }
}

impl CohereTranscriber {
    /// `model_directory` is the layout `scripts/fetch-cohere-model.sh` produces.
    pub fn load(model_directory: &Path) -> Result<Self, String> {
        let encoder_path = model_directory.join("onnx/encoder_model_q4f16.onnx");
        let decoder_path = model_directory.join("onnx/decoder_model_merged_q4f16.onnx");
        let tokenizer_path = model_directory.join("tokenizer.json");
        for path in [&encoder_path, &decoder_path, &tokenizer_path] {
            if !path.is_file() {
                return Err(format!("Cohere model file is missing: {}", path.display()));
            }
        }
        Ok(Self {
            encoder: Mutex::new(session(&encoder_path)?),
            decoder: Mutex::new(session(&decoder_path)?),
            tokenizer: Tokenizer::from_file(&tokenizer_path)
                .map_err(|error| format!("Could not load the Cohere tokenizer: {error}"))?,
            fbank: CohereFbank::new(),
        })
    }

    /// Transcribes 16 kHz mono audio. Chunks over 35 s are decoded separately and
    /// joined, matching how the front end splits them.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String, String> {
        let mut pieces = Vec::new();
        for frames in self.fbank.features(samples) {
            if frames.is_empty() {
                continue;
            }
            let hidden = self.encode(&frames)?;
            pieces.push(self.decode(&hidden)?);
        }
        Ok(pieces
            .iter()
            .map(|piece| piece.trim())
            .filter(|piece| !piece.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }

    /// Runs the encoder and returns `(frames, [1, T, 1024])` flattened.
    fn encode(&self, frames: &[Vec<f32>]) -> Result<(usize, Vec<f32>), String> {
        let flat: Vec<f32> = frames.iter().flatten().copied().collect();
        let mut encoder = self
            .encoder
            .lock()
            .map_err(|_| "The Cohere encoder lock was poisoned".to_string())?;
        let outputs = encoder
            .run(ort::inputs![
                "input_features" => Tensor::from_array(([1usize, frames.len(), MEL_BINS], flat))
                    .map_err(tensor_error)?,
            ])
            .map_err(|error| format!("Cohere encoder inference failed: {error}"))?;
        let (shape, values) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(tensor_error)?;
        let width = *shape.last().unwrap_or(&0) as usize;
        if width != ENCODER_WIDTH {
            return Err(format!(
                "The Cohere encoder returned width {width}, expected {ENCODER_WIDTH}"
            ));
        }
        Ok((values.len() / ENCODER_WIDTH, values.to_vec()))
    }

    /// Greedy decode against the encoder output.
    fn decode(&self, hidden: &(usize, Vec<f32>)) -> Result<String, String> {
        let (encoder_frames, encoder_values) = hidden;
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| "The Cohere decoder lock was poisoned".to_string())?;

        let mut caches = vec![LayerCache::empty(); LAYERS];
        let mut tokens: Vec<i64> = PREFIX.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        // First call submits the whole prefix; later calls submit one token.
        let mut pending: Vec<i64> = PREFIX.to_vec();
        let mut consumed = 0usize;

        for _ in 0..MAX_NEW_TOKENS {
            let step = pending.len();
            let total = consumed + step;
            let position_ids: Vec<i64> = (consumed..total).map(|p| p as i64).collect();
            let attention_mask = vec![1i64; total];

            let mut inputs = ort::inputs![
                "input_ids" => Tensor::from_array(([1usize, step], pending.clone())).map_err(tensor_error)?,
                "attention_mask" => Tensor::from_array(([1usize, total], attention_mask)).map_err(tensor_error)?,
                "position_ids" => Tensor::from_array(([1usize, step], position_ids)).map_err(tensor_error)?,
                "num_logits_to_keep" => Tensor::from_array(([0usize; 0], vec![1i64])).map_err(tensor_error)?,
                "encoder_hidden_states" => Tensor::from_array((
                    [1usize, *encoder_frames, ENCODER_WIDTH], encoder_values.clone()
                )).map_err(tensor_error)?,
            ];
            for (layer, cache) in caches.iter().enumerate() {
                for (suffix, values, length) in [
                    ("decoder.key", &cache.decoder_key, cache.decoder_len),
                    ("decoder.value", &cache.decoder_value, cache.decoder_len),
                    ("encoder.key", &cache.encoder_key, cache.encoder_len),
                    ("encoder.value", &cache.encoder_value, cache.encoder_len),
                ] {
                    inputs.push((
                        format!("past_key_values.{layer}.{suffix}").into(),
                        cache_tensor(values, length)?.into(),
                    ));
                }
            }

            let outputs = decoder
                .run(inputs)
                .map_err(|error| format!("Cohere decoder inference failed: {error}"))?;

            let (_, logits) = outputs["logits"]
                .try_extract_tensor::<f16>()
                .map_err(tensor_error)?;
            // `num_logits_to_keep = 1`, so this is the last position's row.
            let next = logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
                .map(|(index, _)| index as i64)
                .ok_or("The Cohere decoder returned no logits")?;

            for (layer, cache) in caches.iter_mut().enumerate() {
                for (suffix, slot, length) in [
                    ("decoder.key", &mut cache.decoder_key, true),
                    ("decoder.value", &mut cache.decoder_value, true),
                    ("encoder.key", &mut cache.encoder_key, false),
                    ("encoder.value", &mut cache.encoder_value, false),
                ] {
                    let (shape, values) = outputs[format!("present.{layer}.{suffix}").as_str()]
                        .try_extract_tensor::<f16>()
                        .map_err(tensor_error)?;
                    *slot = values.to_vec();
                    let sequence = shape.get(2).copied().unwrap_or(0) as usize;
                    if length {
                        cache.decoder_len = sequence;
                    } else {
                        cache.encoder_len = sequence;
                    }
                }
            }
            drop(outputs);

            consumed = total;
            if next == EOS {
                break;
            }
            generated.push(next as u32);
            tokens.push(next);
            pending = vec![next];
        }

        self.tokenizer
            .decode(&generated, true)
            .map_err(|error| format!("Could not decode Cohere tokens: {error}"))
    }
}

fn session(path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|error| format!("Could not create an ONNX session builder: {error}"))?
        .commit_from_file(path)
        .map_err(|error| format!("Could not load {}: {error}", path.display()))
}

/// Builds a K/V cache tensor, including the zero-length case the first decoder
/// call needs.
///
/// Goes through `ndarray` rather than `(shape, vec)`: ort's raw-data path rejects
/// any dimension below 1, but the empty past is exactly `[1, 8, 0, 128]`, and an
/// `ndarray` with a zero-length axis is perfectly legal.
fn cache_tensor(values: &[f16], length: usize) -> Result<Tensor<f16>, String> {
    if length == 0 {
        // `from_array`'s raw-data path rejects any dimension below 1, and ort's
        // ndarray path has no f16 impl, so the empty past is allocated instead.
        return Tensor::<f16>::new(
            &ort::memory::Allocator::default(),
            [1, HEADS as i64, 0, HEAD_DIM as i64],
        )
        .map_err(tensor_error);
    }
    Tensor::from_array(([1usize, HEADS, length, HEAD_DIM], values.to_vec())).map_err(tensor_error)
}

fn tensor_error<E: std::fmt::Display>(error: E) -> String {
    format!("Cohere tensor error: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_directory() -> std::path::PathBuf {
        std::env::var("VOXIDE_COHERE_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                directories::ProjectDirs::from("dev", "pmdcoutinho", "Voxide")
                    .map(|d| {
                        d.data_local_dir()
                            .join("models")
                            .join("cohere-transcribe-03-2026-q4f16")
                    })
                    .unwrap_or_default()
            })
    }

    #[test]
    fn the_prefix_matches_the_tokenizer_special_tokens() {
        // Guards against copying another implementation's prefix: each ID has to
        // be the token this build's tokenizer actually assigns.
        let directory = model_directory();
        let path = directory.join("tokenizer.json");
        if !path.is_file() {
            return;
        }
        let tokenizer = Tokenizer::from_file(&path).expect("tokenizer loads");
        for (id, expected) in [
            (4u32, "<|startoftranscript|>"),
            (62, "<|en|>"),
            (5, "<|pnc|>"),
            (8, "<|itn|>"),
            (11, "<|notimestamp|>"),
            (13, "<|nodiarize|>"),
            (3, "<|endoftext|>"),
        ] {
            assert_eq!(
                tokenizer.id_to_token(id).as_deref(),
                Some(expected),
                "token {id}"
            );
        }
        assert_eq!(PREFIX.to_vec(), vec![4i64, 62, 5, 8, 11, 13]);
        assert_eq!(EOS, 3);
    }

    /// The end-to-end oracle: real speech in, readable words out. This is what a
    /// tolerance test on the front end cannot substitute for — a front end or a
    /// cache-threading error that survives numeric checks still produces visibly
    /// wrong words here.
    #[test]
    #[ignore = "needs the 1.5 GB q4f16 model and ORT_DYLIB_PATH"]
    fn transcribes_real_speech() {
        let directory = model_directory();
        if !directory.join("onnx/encoder_model_q4f16.onnx").is_file() {
            eprintln!("skipping: no model at {}", directory.display());
            return;
        }
        let transcriber = CohereTranscriber::load(&directory).expect("model loads");

        // The same real-speech samples the front end fixture carries.
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let raw = include_str!("../fixtures/cohere-fbank-speech-reference.json");
        let parsed: serde_json::Value = serde_json::from_str(raw).expect("valid JSON");
        let samples: Vec<f32> = STANDARD
            .decode(parsed["samples_f32_le_base64"].as_str().expect("samples"))
            .expect("decodes")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let started = std::time::Instant::now();
        let text = transcriber
            .transcribe(&samples)
            .expect("transcription runs");
        let elapsed = started.elapsed();
        let audio_seconds = samples.len() as f32 / 16_000.0;
        println!(
            "transcript: {text:?}\n{:.2} s audio in {:.2} s  ({:.2}x realtime)",
            audio_seconds,
            elapsed.as_secs_f32(),
            audio_seconds / elapsed.as_secs_f32()
        );
        assert!(!text.trim().is_empty(), "produced no text");
        // Loose: 0.5 s of speech, but a broken cache or front end yields either
        // nothing or obvious garbage rather than a few plausible words.
        assert!(
            text.chars().any(|c| c.is_alphabetic()),
            "no alphabetic output: {text:?}"
        );
    }
}
