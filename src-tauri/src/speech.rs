use std::{
    ffi::CStr,
    os::raw::c_int,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{audio, media};

pub type ProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync + 'static>;

/// The most recently used Whisper model, kept warm across transcriptions.
/// Loading a model reads hundreds of megabytes from disk, which would
/// otherwise be paid at the start of every dictation.
static CONTEXT_CACHE: OnceLock<Mutex<Option<(PathBuf, Arc<WhisperContext>)>>> = OnceLock::new();

/// Loads the model into the warm cache ahead of the first dictation.
pub fn preload_whisper(model_path: &Path) -> Result<(), String> {
    load_context(model_path).map(|_| ())
}

pub fn transcribe_whisper(
    samples: Vec<f32>,
    model_path: &Path,
    language: &str,
    custom_words: &[String],
) -> Result<String, String> {
    if !model_path.is_file() {
        return Err(format!(
            "Whisper model is not installed: {}",
            model_path.display()
        ));
    }
    if samples.is_empty() {
        return Err("No audio was captured".into());
    }
    if !audio::has_minimum_transcription_samples(&samples) {
        return Err("Audio too short for Whisper transcription".into());
    }

    let context = load_context(model_path)?;
    let text = transcribe_samples(&context, &samples, language, custom_words)?;
    if text.is_empty() {
        return Err("The voice engine did not recognize any speech".into());
    }
    Ok(text)
}

pub fn transcribe_media_file(
    path: &Path,
    model_path: &Path,
    language: &str,
    custom_words: &[String],
    progress: Option<ProgressCallback>,
) -> Result<(String, u64), String> {
    let duration_ms = media::file_duration_ms(path)?;
    let total_chunks = ((duration_ms as f64 / 1000.0) / media::TRANSCRIPTION_CHUNK_SECONDS)
        .ceil()
        .max(1.0) as usize;
    let context = load_context(model_path)?;
    let mut text = Vec::new();
    for chunk in 0..total_chunks {
        let start = chunk as f64 * media::TRANSCRIPTION_CHUNK_SECONDS;
        let remaining = (duration_ms as f64 / 1000.0 - start).max(0.0);
        let audio = media::decode_audio_segment(
            path,
            start,
            remaining.min(media::TRANSCRIPTION_CHUNK_SECONDS),
        )?;
        let samples = audio::mono_resample_for_whisper(audio)?;
        // Keep file transcription aligned with Voxide's buffered path:
        // sub-second chunks (normally the final tail of a file) are skipped.
        if !audio::has_minimum_transcription_samples(&samples) {
            if let Some(progress) = &progress {
                progress(chunk + 1, total_chunks);
            }
            continue;
        }
        let result = transcribe_samples(&context, &samples, language, custom_words)?;
        if !result.is_empty() {
            text.push(result);
        }
        if let Some(progress) = &progress {
            progress(chunk + 1, total_chunks);
        }
    }
    let text = text.join(" ");
    Ok((text, duration_ms))
}

/// whisper.cpp's `gpu_device` parameter selects the N-th GPU-type device in
/// ggml's registry order, and hybrid laptops enumerate the integrated GPU
/// before the discrete one. Prefer a discrete GPU when both are present,
/// avoid software rasterizers, and let VOXIDE_GPU_DEVICE override the pick.
/// Returns None when no GPU backend is compiled in or no device exists.
fn preferred_gpu_device() -> Option<c_int> {
    if let Some(index) = std::env::var("VOXIDE_GPU_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<c_int>().ok())
    {
        return Some(index);
    }
    let mut best: Option<(i32, c_int)> = None;
    let mut gpu_index: c_int = 0;
    unsafe {
        for device_index in 0..whisper_rs_sys::ggml_backend_dev_count() {
            let device = whisper_rs_sys::ggml_backend_dev_get(device_index);
            let device_type = whisper_rs_sys::ggml_backend_dev_type(device);
            let discrete = device_type
                == whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU;
            let integrated = device_type
                == whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU;
            if !discrete && !integrated {
                continue;
            }
            let description = CStr::from_ptr(whisper_rs_sys::ggml_backend_dev_description(device))
                .to_string_lossy()
                .to_lowercase();
            let software =
                description.contains("llvmpipe") || description.contains("swiftshader");
            let score = if software {
                -1
            } else if discrete {
                2
            } else {
                1
            };
            if best.map_or(true, |(best_score, _)| score > best_score) {
                best = Some((score, gpu_index));
            }
            gpu_index += 1;
        }
    }
    best.map(|(_, index)| index)
}

fn load_context(model_path: &Path) -> Result<Arc<WhisperContext>, String> {
    if !model_path.is_file() {
        return Err(format!(
            "Whisper model is not installed: {}",
            model_path.display()
        ));
    }
    let cache = CONTEXT_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| "Whisper model cache lock was poisoned".to_string())?;
    if let Some((path, context)) = cached.as_ref() {
        if path == model_path {
            return Ok(Arc::clone(context));
        }
    }
    // Ask for GPU inference unconditionally: builds without a GPU backend
    // (or machines without a usable device) fall back to CPU inside
    // whisper.cpp. whisper-rs only defaults this on for its own GPU
    // features, which the vulkan build bypasses (see Cargo.toml).
    let mut parameters = WhisperContextParameters::default();
    parameters.use_gpu(true);
    if let Some(device) = preferred_gpu_device() {
        parameters.gpu_device(device);
    }
    let context = Arc::new(
        WhisperContext::new_with_params(model_path, parameters)
            .map_err(|error| format!("Could not load Whisper model: {error}"))?,
    );
    *cached = Some((model_path.to_path_buf(), Arc::clone(&context)));
    Ok(context)
}

fn transcribe_samples(
    context: &WhisperContext,
    samples: &[f32],
    language: &str,
    custom_words: &[String],
) -> Result<String, String> {
    let mut state = context
        .create_state()
        .map_err(|error| format!("Could not prepare Whisper: {error}"))?;
    let mut parameters = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let language = language.trim();
    parameters.set_language((!language.is_empty()).then_some(language));
    parameters.set_print_special(false);
    parameters.set_print_progress(false);
    parameters.set_print_realtime(false);
    parameters.set_print_timestamps(false);
    // Dictation wants speech only — no "[sound of plastic crinkling]" tags.
    parameters.set_suppress_nst(true);
    let vocabulary = custom_words
        .iter()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty())
        .take(200)
        .collect::<Vec<_>>()
        .join(", ");
    if !vocabulary.is_empty() {
        parameters.set_initial_prompt(&format!("Recognition vocabulary: {vocabulary}"));
    }
    state
        .full(parameters, samples)
        .map_err(|error| format!("Whisper could not transcribe the recording: {error}"))?;

    let text = state
        .as_iter()
        .filter_map(|segment| clean_transcription_segment(&segment.to_string()))
        .collect::<Vec<_>>()
        .join(" ");
    Ok(text)
}

/// Strips Whisper's non-speech artifacts from a segment: noise annotations
/// such as "[sound of plastic crinkling]" or "(water splashing)", music
/// markers, and the dialogue dash it prepends to some utterances. Token-level
/// suppression removes most of these; this catches the ones that slip
/// through. Returns None when nothing speakable remains.
fn clean_transcription_segment(segment: &str) -> Option<String> {
    static ANNOTATION_PATTERN: OnceLock<regex::Regex> = OnceLock::new();
    let pattern = ANNOTATION_PATTERN.get_or_init(|| {
        regex::Regex::new(r"\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪|^\*[^*]+\*$")
            .expect("annotation pattern is valid")
    });
    let without_annotations = pattern.replace_all(segment.trim(), " ");
    let without_dash = without_annotations
        .trim()
        .trim_start_matches("- ")
        .trim_start_matches('♪');
    let cleaned = without_dash.split_whitespace().collect::<Vec<_>>().join(" ");
    (!cleaned.is_empty()).then_some(cleaned)
}

#[cfg(test)]
mod tests {
    use super::clean_transcription_segment;

    #[test]
    fn drops_pure_noise_annotations() {
        assert_eq!(
            clean_transcription_segment(" [sound of plastic crinkling]"),
            None
        );
        assert_eq!(clean_transcription_segment("(water splashing)"), None);
        assert_eq!(clean_transcription_segment(" ♪ upbeat music ♪"), None);
        assert_eq!(clean_transcription_segment("*coughs*"), None);
    }

    #[test]
    fn keeps_speech_and_removes_inline_artifacts() {
        assert_eq!(
            clean_transcription_segment("(water splashing) - All right."),
            Some("All right.".into())
        );
        assert_eq!(
            clean_transcription_segment(" Alright [door slams] let's continue."),
            Some("Alright let's continue.".into())
        );
        assert_eq!(
            clean_transcription_segment(" The quick brown fox."),
            Some("The quick brown fox.".into())
        );
    }
}
