//! NVIDIA Parakeet TDT integration.
//!
//! Parakeet TDT is an offline (rather than token-streaming) recognizer.  The
//! live UI therefore feeds complete VAD utterances to the same final-quality
//! recognizer and leaves the newest utterance provisional.  This preserves the
//! fast, stable streaming feel without inventing words during room silence.

use std::path::{Path, PathBuf};

use crate::{audio, media};

pub const MODEL_ID: &str = "parakeet-tdt-0.6b-v3-int8";
pub const MODEL_ARCHIVE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2";
const MODEL_ARCHIVE_ROOT: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
const REQUIRED_FILES: [&str; 4] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

pub fn is_compiled() -> bool {
    cfg!(feature = "parakeet")
}

pub fn model_directory(models_directory: &Path) -> PathBuf {
    models_directory.join(MODEL_ID)
}

pub fn archive_root() -> &'static str {
    MODEL_ARCHIVE_ROOT
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
/// Whisper/file-transcription contract. Parakeet does not use vocabulary
/// prompts, so the ASR output intentionally remains the model's direct text.
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

    use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

    use super::{model_is_installed, REQUIRED_FILES};

    static RECOGNIZER_CACHE: OnceLock<Mutex<Option<(PathBuf, Arc<OfflineRecognizer>)>>> =
        OnceLock::new();
    // The CUDA execution provider is efficient at one interactive decode at a
    // time. A single gate also prevents an older preview from delaying the
    // final tail when a dictation stops.
    static INFERENCE_LOCK: Mutex<()> = Mutex::new(());

    fn recognizer(model_directory: &Path) -> Result<Arc<OfflineRecognizer>, String> {
        if !model_is_installed(model_directory) {
            return Err(format!(
                "The Parakeet model is missing from {}",
                model_directory.display()
            ));
        }
        let cache = RECOGNIZER_CACHE.get_or_init(|| Mutex::new(None));
        let mut cache = cache
            .lock()
            .map_err(|_| "Parakeet model cache lock was poisoned".to_string())?;
        if let Some((_, recognizer)) = cache.as_ref().filter(|(path, _)| path == model_directory) {
            return Ok(Arc::clone(recognizer));
        }

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
        config.model_config.tokens = Some(
            model_directory
                .join(REQUIRED_FILES[3])
                .display()
                .to_string(),
        );
        config.model_config.model_type = Some("nemo_transducer".into());
        config.model_config.provider = Some("cuda".into());
        config.model_config.num_threads = 1;
        config.decoding_method = Some("greedy_search".into());
        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            "Could not load Parakeet with CUDA. Check that the CUDA 12 and cuDNN 9 runtime libraries are installed beside the Voxide CUDA build.".to_string()
        })?;
        let recognizer = Arc::new(recognizer);
        *cache = Some((model_directory.to_path_buf(), Arc::clone(&recognizer)));
        Ok(recognizer)
    }

    pub fn preload(model_directory: &Path) -> Result<(), String> {
        recognizer(model_directory).map(|_| ())
    }

    pub fn transcribe(samples: &[f32], model_directory: &Path) -> Result<String, String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let recognizer = recognizer(model_directory)?;
        let _inference = INFERENCE_LOCK
            .lock()
            .map_err(|_| "Parakeet inference lock was poisoned".to_string())?;
        let stream = recognizer.create_stream();
        stream.accept_waveform(16_000, samples);
        recognizer.decode(&stream);
        stream
            .get_result()
            .map(|result| result.text.trim().to_owned())
            .ok_or_else(|| "Parakeet did not return a transcription result".to_string())
    }
}

#[cfg(feature = "parakeet")]
pub use implementation::{preload, transcribe as transcribe_samples};

#[cfg(not(feature = "parakeet"))]
pub fn preload(_: &Path) -> Result<(), String> {
    Err("Parakeet is included only in the CUDA build".into())
}

#[cfg(not(feature = "parakeet"))]
pub fn transcribe_samples(_: &[f32], _: &Path) -> Result<String, String> {
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
    }
}
