//! Opt-in true-streaming preview for the Parakeet engine.
//!
//! Parakeet's own model is an offline TDT recognizer, so its default live
//! preview re-decodes a rolling window every tick (see `spawn_live_parakeet_preview`
//! in `lib.rs`). This module adds FluidVoice's "Flash" analog instead: a small
//! streaming zipformer driven by sherpa-onnx's `OnlineRecognizer`, fed only the
//! newly captured audio each tick, with real end-of-utterance endpointing.
//!
//! It is deliberately a SEPARATE, opt-in model and always runs on the CPU
//! execution provider so it never touches the CUDA context the offline TDT
//! final decode owns (no shared inference lock, no concurrent-CUDA risk). The
//! preview text is weaker than — and diverges from — the authoritative final
//! CUDA decode, exactly as FluidVoice's Flash-vs-offline split does.

use std::path::{Path, PathBuf};

pub const MODEL_ID: &str = "streaming-zipformer-en-2023-06-26";
pub const MODEL_ARCHIVE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2";
/// GitHub release asset digest verified on 2026-07-24. The release tag is a
/// mutable URL, so installation must authenticate the downloaded archive.
pub const MODEL_ARCHIVE_SHA256: &str =
    "639e25b578e9e997131402199419c13a941f8e4e198e2da1ce57dbf5cf401282";
pub const MODEL_ARCHIVE_BYTES: u64 = 310_414_022;
const MODEL_ARCHIVE_ROOT: &str = "sherpa-onnx-streaming-zipformer-en-2023-06-26";

// The int8 encoder/decoder/joiner keep the CPU cost of a per-tick streaming
// decode low; the model also ships fp32 copies and a bpe.model we do not need.
const ENCODER: &str = "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const DECODER: &str = "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const JOINER: &str = "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
const TOKENS: &str = "tokens.txt";
const REQUIRED_FILES: [&str; 4] = [ENCODER, DECODER, JOINER, TOKENS];

pub fn is_compiled() -> bool {
    cfg!(feature = "parakeet")
}

pub fn model_directory(models_directory: &Path) -> PathBuf {
    models_directory.join(MODEL_ID)
}

pub fn archive_root() -> &'static str {
    MODEL_ARCHIVE_ROOT
}

pub fn required_files() -> &'static [&'static str] {
    &REQUIRED_FILES
}

pub fn model_is_installed(directory: &Path) -> bool {
    directory.is_dir()
        && REQUIRED_FILES
            .iter()
            .all(|file| directory.join(file).is_file())
}

pub fn installation_error(directory: &Path) -> String {
    let missing = REQUIRED_FILES
        .iter()
        .filter(|file| !directory.join(file).is_file())
        .copied()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        "The streaming preview model is ready".into()
    } else {
        format!(
            "The streaming preview model is missing {}",
            missing.join(", ")
        )
    }
}

#[cfg(feature = "parakeet")]
mod implementation {
    use std::{
        path::{Path, PathBuf},
        sync::{Arc, Mutex, OnceLock},
    };

    use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};

    use super::{model_is_installed, DECODER, ENCODER, JOINER, TOKENS};

    // The streaming recognizer is stateless across sessions, so cache one warm
    // CPU instance (keyed by model directory) and hand out clones. Loading it is
    // a few hundred milliseconds.
    type CachedRecognizer = (PathBuf, Arc<OnlineRecognizer>);
    static RECOGNIZER: OnceLock<Mutex<Option<CachedRecognizer>>> = OnceLock::new();

    fn recognizer(model_directory: &Path) -> Result<Arc<OnlineRecognizer>, String> {
        if !model_is_installed(model_directory) {
            return Err(format!(
                "The streaming Parakeet preview model is missing from {}",
                model_directory.display()
            ));
        }
        let cache = RECOGNIZER.get_or_init(|| Mutex::new(None));
        let mut cache = cache
            .lock()
            .map_err(|_| "Streaming preview model cache lock was poisoned".to_string())?;
        if let Some((directory, recognizer)) = cache.as_ref() {
            if directory == model_directory {
                return Ok(Arc::clone(recognizer));
            }
        }
        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder =
            Some(model_directory.join(ENCODER).display().to_string());
        config.model_config.transducer.decoder =
            Some(model_directory.join(DECODER).display().to_string());
        config.model_config.transducer.joiner =
            Some(model_directory.join(JOINER).display().to_string());
        config.model_config.tokens = Some(model_directory.join(TOKENS).display().to_string());
        // MANDATORY cpu: this preview must never touch the CUDA context the
        // offline TDT final owns, so it needs no inference lock and cannot
        // corrupt a concurrent CUDA decode.
        config.model_config.provider = Some("cpu".into());
        config.model_config.num_threads = 2;
        config.decoding_method = Some("greedy_search".into());
        // End-of-utterance detection (sherpa defaults): commit a segment after
        // ~2.4 s of trailing silence, or ~1.2 s once a token has been decoded.
        config.enable_endpoint = true;
        config.rule1_min_trailing_silence = 2.4;
        config.rule2_min_trailing_silence = 1.2;
        config.rule3_min_utterance_length = 20.0;
        let recognizer = OnlineRecognizer::create(&config).ok_or_else(|| {
            "Could not load the streaming Parakeet preview model on CPU. Check that the model files are intact.".to_string()
        })?;
        let recognizer = Arc::new(recognizer);
        *cache = Some((model_directory.to_path_buf(), Arc::clone(&recognizer)));
        Ok(recognizer)
    }

    /// A live streaming-preview session: a persistent CPU `OnlineStream` fed the
    /// capture delta each tick, with endpoint-committed segments accumulated
    /// across pauses so the visible transcript is monotonic.
    pub struct Session {
        recognizer: Arc<OnlineRecognizer>,
        stream: OnlineStream,
        fed_samples: usize,
        committed: String,
    }

    impl Session {
        pub fn start(model_directory: &Path) -> Result<Self, String> {
            let recognizer = recognizer(model_directory)?;
            let stream = recognizer.create_stream();
            Ok(Self {
                recognizer,
                stream,
                fed_samples: 0,
                committed: String::new(),
            })
        }

        /// Feeds the not-yet-seen tail of the resampled capture and returns the
        /// current display text: endpoint-committed segments plus the live
        /// partial. `samples` is the whole capture-so-far (16 kHz mono); only
        /// the delta past `fed_samples` is appended, keeping the stream's
        /// cache-aware context intact (never re-fed or sliced mid-utterance).
        pub fn observe(&mut self, samples: &[f32]) -> String {
            if samples.len() > self.fed_samples {
                self.stream
                    .accept_waveform(16_000, &samples[self.fed_samples..]);
                self.fed_samples = samples.len();
            }
            while self.recognizer.is_ready(&self.stream) {
                self.recognizer.decode(&self.stream);
            }
            let partial = self
                .recognizer
                .get_result(&self.stream)
                .map(|result| result.text)
                .unwrap_or_default();
            let partial = partial.trim();
            if self.recognizer.is_endpoint(&self.stream) {
                if !partial.is_empty() {
                    if !self.committed.is_empty() {
                        self.committed.push(' ');
                    }
                    self.committed.push_str(partial);
                }
                self.recognizer.reset(&self.stream);
                self.committed.clone()
            } else if partial.is_empty() {
                self.committed.clone()
            } else if self.committed.is_empty() {
                partial.to_owned()
            } else {
                format!("{} {}", self.committed, partial)
            }
        }
    }
}

#[cfg(feature = "parakeet")]
pub use implementation::Session;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_layout_requires_all_streaming_files() {
        let directory = std::env::temp_dir().join(format!(
            "voxide-parakeet-stream-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        assert!(!model_is_installed(&directory));
        for file in REQUIRED_FILES {
            std::fs::write(directory.join(file), []).unwrap();
        }
        assert!(model_is_installed(&directory));
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn archive_is_pinned_to_a_digest_and_size() {
        assert_eq!(MODEL_ARCHIVE_SHA256.len(), 64);
        assert!(MODEL_ARCHIVE_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
        assert!(MODEL_ARCHIVE_BYTES > 50 * 1024 * 1024);
        assert!(MODEL_ARCHIVE_URL.starts_with("https://"));
    }
}
