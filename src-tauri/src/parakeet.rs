//! NVIDIA Parakeet TDT integration.
//!
//! Parakeet TDT is an offline (rather than token-streaming) recognizer. The
//! live UI mirrors FluidVoice's v3 route: periodically decode the complete
//! capture for a full preview hypothesis, then run a separate full-audio
//! decode when recording stops.

use std::path::{Path, PathBuf};

use crate::{audio, media};

pub const MODEL_ID: &str = "parakeet-tdt-0.6b-v2-int8";
pub const MODEL_ARCHIVE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2";
/// GitHub release asset digest verified on 2026-07-22. The release tag is a
/// mutable URL, so installation must authenticate the downloaded archive.
pub const MODEL_ARCHIVE_SHA256: &str =
    "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad";
pub const MODEL_ARCHIVE_BYTES: u64 = 482_468_385;
const MODEL_ARCHIVE_ROOT: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
const REQUIRED_FILES: [&str; 4] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

/// TDT is an offline recognizer, so the newest tokens in a live full-buffer
/// decode are tentative: the model has not yet seen the following audio that
/// can disambiguate them. Keep that short tail out of the display only. The
/// final transcription always decodes and returns the complete recording.
#[cfg(feature = "parakeet")]
pub const PREVIEW_TRAILING_GUARD_SECONDS: f32 = 0.75;

/// Builds a display-safe prefix from Sherpa's per-token timing result.
///
/// Returning `None` means the runtime did not provide a complete timing map;
/// callers can then choose their compatibility fallback explicitly. Tokens
/// are SentencePiece fragments, so concatenating them is the faithful text
/// reconstruction instead of joining them with spaces.
#[cfg(any(feature = "parakeet", test))]
fn stable_preview_text(
    tokens: &[String],
    timestamps: Option<&[f32]>,
    durations: Option<&[f32]>,
    cutoff_seconds: f32,
) -> Option<String> {
    let timestamps = timestamps?;
    if tokens.len() != timestamps.len()
        || durations.is_some_and(|durations| durations.len() != tokens.len())
    {
        return None;
    }
    let durations = durations.unwrap_or(&[]);
    let stable_token_count = timestamps
        .iter()
        .enumerate()
        .take_while(|(index, timestamp)| {
            let duration = durations.get(*index).copied().unwrap_or_default();
            **timestamp + duration <= cutoff_seconds
        })
        .count();
    Some(tokens[..stable_token_count].concat().trim().to_owned())
}

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
        "Parakeet is ready".into()
    } else {
        format!("Parakeet model is missing {}", missing.join(", "))
    }
}

/// Decode a media file in bounded chunks, matching the application's
/// Whisper/file-transcription contract. The ASR output remains the model's
/// direct text before caller-owned deterministic formatting.
pub fn transcribe_media_file(
    path: &Path,
    model_directory: &Path,
    progress: Option<crate::speech::ProgressCallback>,
) -> Result<(String, u64), String> {
    let duration_ms = media::file_duration_ms(path)?;
    let total_chunks = ((duration_ms as f64 / 1_000.0) / media::TRANSCRIPTION_CHUNK_SECONDS)
        .ceil()
        .max(1.0) as usize;
    let mut chunks = Vec::with_capacity(total_chunks);
    for chunk in 0..total_chunks {
        let start_seconds = chunk as f64 * media::TRANSCRIPTION_CHUNK_SECONDS;
        let remaining_seconds = (duration_ms as f64 / 1_000.0 - start_seconds).max(0.0);
        let audio = media::decode_audio_segment(
            path,
            start_seconds,
            remaining_seconds.min(media::TRANSCRIPTION_CHUNK_SECONDS),
        )?;
        let samples = audio::mono_resample_for_whisper(audio)?;
        if audio::has_minimum_transcription_samples(&samples) {
            let text = transcribe_samples(&samples, model_directory)?;
            if !text.trim().is_empty() {
                chunks.push(text);
            }
        }
        if let Some(progress) = &progress {
            progress(chunk + 1, total_chunks);
        }
    }
    Ok((chunks.join(" "), duration_ms))
}

#[cfg(feature = "parakeet")]
mod implementation {
    use std::{
        path::{Path, PathBuf},
        sync::{Arc, Mutex, OnceLock},
    };

    use sherpa_onnx::{
        OfflineRecognizer, OfflineRecognizerConfig, OfflineRecognizerResult,
        OfflineTransducerModelConfig,
    };

    use crate::debug_log;

    use super::{
        model_is_installed, stable_preview_text, PREVIEW_TRAILING_GUARD_SECONDS, REQUIRED_FILES,
    };

    #[derive(Clone)]
    struct CachedRecognizer {
        model_directory: PathBuf,
        vocabulary_boosting: bool,
        recognizer: Arc<OfflineRecognizer>,
    }

    static RECOGNIZER_CACHE: OnceLock<Mutex<Vec<CachedRecognizer>>> = OnceLock::new();
    // The CUDA execution provider is efficient at one interactive decode at a
    // time. A single gate also prevents an older preview from delaying the
    // final tail when a dictation stops.
    static INFERENCE_LOCK: Mutex<()> = Mutex::new(());

    fn recognizer(
        model_directory: &Path,
        vocabulary_boosting: bool,
    ) -> Result<Arc<OfflineRecognizer>, String> {
        if !model_is_installed(model_directory) {
            return Err(format!(
                "The Parakeet model is missing from {}",
                model_directory.display()
            ));
        }
        let cache = RECOGNIZER_CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let mut cache = cache
            .lock()
            .map_err(|_| "Parakeet model cache lock was poisoned".to_string())?;
        if let Some(cached) = cache.iter().find(|cached| {
            cached.model_directory == model_directory
                && cached.vocabulary_boosting == vocabulary_boosting
        }) {
            return Ok(Arc::clone(&cached.recognizer));
        }

        let tokens = model_directory
            .join(REQUIRED_FILES[3])
            .display()
            .to_string();
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(
                model_directory
                    .join(REQUIRED_FILES[0])
                    .display()
                    .to_string(),
            ),
            decoder: Some(
                model_directory
                    .join(REQUIRED_FILES[1])
                    .display()
                    .to_string(),
            ),
            joiner: Some(
                model_directory
                    .join(REQUIRED_FILES[2])
                    .display()
                    .to_string(),
            ),
        };
        config.model_config.tokens = Some(tokens.clone());
        config.model_config.model_type = Some("nemo_transducer".into());
        config.model_config.provider = Some("cuda".into());
        config.model_config.num_threads = 1;
        if vocabulary_boosting {
            // The FluidVoice final manager applies its optional vocabulary
            // rescoring only after recording stops. Sherpa's equivalent for
            // this TDT export is modified-beam search plus a BPE context graph.
            // `tokens.txt` is the model's SentencePiece vocabulary.
            config.decoding_method = Some("modified_beam_search".into());
            config.max_active_paths = 4;
            config.hotwords_score = 1.5;
            config.model_config.modeling_unit = Some("bpe".into());
            config.model_config.bpe_vocab = Some(tokens);
        } else {
            config.decoding_method = Some("greedy_search".into());
        }
        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            if vocabulary_boosting {
                "Could not load Parakeet vocabulary boosting with CUDA. Check that the bundled sherpa-onnx runtime supports Parakeet TDT modified-beam decoding.".to_string()
            } else {
                "Could not load Parakeet with CUDA. Check that the CUDA 12 and cuDNN 9 runtime libraries are installed beside the Voxide CUDA build.".to_string()
            }
        })?;
        let recognizer = Arc::new(recognizer);
        cache.push(CachedRecognizer {
            model_directory: model_directory.to_path_buf(),
            vocabulary_boosting,
            recognizer: Arc::clone(&recognizer),
        });
        Ok(recognizer)
    }

    pub fn preload(model_directory: &Path) -> Result<(), String> {
        recognizer(model_directory, false).map(|_| ())
    }

    /// Preloads FluidVoice's equivalent of its optional final-only boosted
    /// manager. If no usable terms are configured, the normal preview/final
    /// recognizer is the only one needed.
    pub fn preload_with_vocabulary(
        model_directory: &Path,
        vocabulary: &[String],
    ) -> Result<(), String> {
        if vocabulary_hotwords(vocabulary).is_some() {
            recognizer(model_directory, true).map(|_| ())
        } else {
            preload(model_directory)
        }
    }

    pub fn transcribe(samples: &[f32], model_directory: &Path) -> Result<String, String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        decode(samples, model_directory, None).map(|result| result.text.trim().to_owned())
    }

    /// Decodes a full live snapshot, but returns only tokens that end before
    /// the unstable tail. This is deliberately display-only: the final pass
    /// below still receives every captured sample and every decoded token.
    pub fn transcribe_preview(samples: &[f32], model_directory: &Path) -> Result<String, String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let result = decode(samples, model_directory, None)?;
        let cutoff_seconds = samples.len() as f32 / 16_000.0 - PREVIEW_TRAILING_GUARD_SECONDS;
        Ok(stable_preview_text(
            &result.tokens,
            result.timestamps.as_deref(),
            result.durations.as_deref(),
            cutoff_seconds,
        )
        .unwrap_or_else(|| result.text.trim().to_owned()))
    }

    /// Uses Sherpa's TDT context graph for FluidVoice-style final vocabulary
    /// boosting. Preview intentionally calls `transcribe` instead, because
    /// FluidVoice applies vocabulary rescoring only to the final manager.
    pub fn transcribe_with_vocabulary(
        samples: &[f32],
        model_directory: &Path,
        vocabulary: &[String],
    ) -> Result<String, String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let Some(hotwords) = vocabulary_hotwords(vocabulary) else {
            return transcribe(samples, model_directory);
        };
        match decode(samples, model_directory, Some(&hotwords)) {
            Ok(result) => Ok(result.text.trim().to_owned()),
            Err(error) => {
                // FluidVoice similarly falls back to its ordinary final
                // manager if the boosted manager cannot transcribe. A custom
                // term must never make a completed dictation fail outright.
                debug_log::append(&format!(
                    "Parakeet vocabulary boosting failed; retrying the unboosted final decode: {error}"
                ));
                transcribe(samples, model_directory)
            }
        }
    }

    fn vocabulary_hotwords(vocabulary: &[String]) -> Option<String> {
        let hotwords = vocabulary
            .iter()
            .map(|word| word.trim())
            // Sherpa uses `/` as its per-stream phrase separator and its Rust
            // binding passes the string through CString. Ignore malformed
            // user terms rather than letting one term abort dictation.
            .filter(|word| !word.is_empty() && !word.contains(['\0', '/', '\n', '\r']))
            .take(200)
            .collect::<Vec<_>>()
            .join("/");
        (!hotwords.is_empty()).then_some(hotwords)
    }

    fn decode(
        samples: &[f32],
        model_directory: &Path,
        hotwords: Option<&str>,
    ) -> Result<OfflineRecognizerResult, String> {
        let recognizer = recognizer(model_directory, hotwords.is_some())?;
        let _inference = INFERENCE_LOCK
            .lock()
            .map_err(|_| "Parakeet inference lock was poisoned".to_string())?;
        let stream = hotwords
            .map(|hotwords| recognizer.create_stream_with_hotwords(hotwords))
            .unwrap_or_else(|| recognizer.create_stream());
        stream.accept_waveform(16_000, samples);
        recognizer.decode(&stream);
        stream
            .get_result()
            .ok_or_else(|| "Parakeet did not return a transcription result".to_string())
    }
}

#[cfg(feature = "parakeet")]
pub use implementation::{
    preload, preload_with_vocabulary, transcribe as transcribe_samples,
    transcribe_preview as transcribe_preview_samples,
    transcribe_with_vocabulary as transcribe_samples_with_vocabulary,
};

#[cfg(not(feature = "parakeet"))]
pub fn preload(_: &Path) -> Result<(), String> {
    Err("Parakeet is included only in the CUDA build".into())
}

#[cfg(not(feature = "parakeet"))]
pub fn preload_with_vocabulary(_: &Path, _: &[String]) -> Result<(), String> {
    Err("Parakeet is included only in the CUDA build".into())
}

#[cfg(not(feature = "parakeet"))]
pub fn transcribe_samples(_: &[f32], _: &Path) -> Result<String, String> {
    Err("Parakeet is included only in the CUDA build".into())
}

#[cfg(not(feature = "parakeet"))]
pub fn transcribe_preview_samples(_: &[f32], _: &Path) -> Result<String, String> {
    Err("Parakeet is included only in the CUDA build".into())
}

#[cfg(not(feature = "parakeet"))]
pub fn transcribe_samples_with_vocabulary(
    _: &[f32],
    _: &Path,
    _: &[String],
) -> Result<String, String> {
    Err("Parakeet is included only in the CUDA build".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_layout_requires_all_parakeet_components() {
        let directory =
            std::env::temp_dir().join(format!("voxide-parakeet-test-{}", std::process::id()));
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
    fn live_preview_omits_only_the_unstable_timed_tail() {
        let tokens = [" One", " two", " three", " four"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let timestamps = [0.0, 0.25, 0.5, 0.75];
        let durations = [0.2, 0.2, 0.2, 0.2];

        assert_eq!(
            stable_preview_text(&tokens, Some(&timestamps), Some(&durations), 0.72),
            Some("One two three".into())
        );
    }

    #[test]
    fn live_preview_falls_back_when_token_timings_are_incomplete() {
        let tokens = [" One", " two"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(stable_preview_text(&tokens, Some(&[0.0]), None, 1.0), None);
    }

    #[test]
    #[ignore = "requires a CUDA build plus VOXIDE_PARAKEET_MODEL_DIR"]
    fn cuda_model_transcribes_its_reference_wav() {
        if !is_compiled() {
            return;
        }
        let directory = std::env::var_os("VOXIDE_PARAKEET_MODEL_DIR")
            .map(PathBuf::from)
            .expect("set VOXIDE_PARAKEET_MODEL_DIR to an installed Parakeet model");
        assert!(model_is_installed(&directory));
        let wav = directory.join("test_wavs/en.wav");
        let duration_ms = media::file_duration_ms(&wav).expect("read reference WAV duration");
        let captured = media::decode_audio_segment(&wav, 0.0, duration_ms as f64 / 1_000.0)
            .expect("decode reference WAV");
        let samples = audio::mono_resample_for_whisper(captured).expect("resample reference WAV");
        let text =
            transcribe_samples(&samples, &directory).expect("decode reference WAV with CUDA");
        println!("Parakeet reference transcript: {text}");
        assert!(!text.trim().is_empty());

        let preview = transcribe_preview_samples(&samples, &directory)
            .expect("decode a timestamp-guarded Parakeet preview with CUDA");
        println!("Parakeet timestamp-guarded preview: {preview}");
        assert!(!preview.trim().is_empty());
    }

    #[test]
    #[ignore = "requires a CUDA build plus VOXIDE_PARAKEET_MODEL_DIR"]
    fn cuda_model_accepts_final_vocabulary_boosting() {
        if !is_compiled() {
            return;
        }
        let directory = std::env::var_os("VOXIDE_PARAKEET_MODEL_DIR")
            .map(PathBuf::from)
            .expect("set VOXIDE_PARAKEET_MODEL_DIR to an installed Parakeet model");
        assert!(model_is_installed(&directory));
        let wav = directory.join("test_wavs/en.wav");
        let duration_ms = media::file_duration_ms(&wav).expect("read reference WAV duration");
        let captured = media::decode_audio_segment(&wav, 0.0, duration_ms as f64 / 1_000.0)
            .expect("decode reference WAV");
        let samples = audio::mono_resample_for_whisper(captured).expect("resample reference WAV");
        let vocabulary = ["country".to_owned()];
        preload_with_vocabulary(&directory, &vocabulary)
            .expect("preload CUDA vocabulary-boosted final model");
        let text = transcribe_samples_with_vocabulary(&samples, &directory, &vocabulary)
            .expect("decode reference WAV with CUDA vocabulary boosting");
        println!("Parakeet vocabulary transcript: {text}");
        assert!(text.to_ascii_lowercase().contains("country"));
    }
}
