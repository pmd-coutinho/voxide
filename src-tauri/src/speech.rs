use std::{
    ffi::CStr,
    os::raw::c_int,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadContext,
    WhisperVadContextParams, WhisperVadParams,
};

use crate::{audio, media};

pub type ProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync + 'static>;

/// The most recently used Whisper model, kept warm across transcriptions.
/// Loading a model reads hundreds of megabytes from disk, which would
/// otherwise be paid at the start of every dictation.
static CONTEXT_CACHE: OnceLock<Mutex<Option<(PathBuf, Arc<WhisperContext>)>>> = OnceLock::new();

/// Serializes inference: the live preview and the final transcription share
/// the warm context, and concurrent runs on one context corrupt each other
/// on GPU backends. The final pass simply waits out an in-flight preview.
static INFERENCE_LOCK: Mutex<()> = Mutex::new(());

/// Loads the model into the warm cache ahead of the first dictation.
pub fn preload_whisper(model_path: &Path) -> Result<(), String> {
    load_context(model_path).map(|_| ())
}

pub fn transcribe_whisper(
    samples: Vec<f32>,
    model_path: &Path,
    vad_model_path: Option<&Path>,
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
    let text = transcribe_samples(&context, &samples, vad_model_path, language, custom_words)?;
    if text.is_empty() {
        return Err("The voice engine did not recognize any speech".into());
    }
    Ok(text)
}

pub fn transcribe_media_file(
    path: &Path,
    model_path: &Path,
    vad_model_path: Option<&Path>,
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
        let result =
            transcribe_samples(&context, &samples, vad_model_path, language, custom_words)?;
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

/// Runs Silero voice-activity detection and keeps only the speech audio,
/// mirroring `whisper_full`'s own VAD preprocessing — which whisper-rs's
/// state-based API bypasses entirely. Returns None when VAD is unavailable
/// (the caller transcribes unfiltered) and an empty buffer when the audio
/// contains no speech at all.
fn speech_only_samples(samples: &[f32], vad_model_path: &Path) -> Option<Vec<f32>> {
    const SAMPLES_PER_CENTISECOND: f32 = audio::WHISPER_SAMPLE_RATE as f32 / 100.0;
    let model = vad_model_path.to_str()?;
    let mut context_params = WhisperVadContextParams::new();
    context_params.set_use_gpu(false);
    let mut vad = WhisperVadContext::new(model, context_params).ok()?;
    let segments = vad
        .segments_from_samples(WhisperVadParams::new(), samples)
        .ok()?;
    // Match whisper_full's reconstruction: extend every non-final segment by
    // the overlap window and join segments with 0.1 s of silence.
    let overlap_samples = (0.1 * audio::WHISPER_SAMPLE_RATE as f32) as usize;
    let joiner = vec![0.0f32; overlap_samples];
    let mut speech: Vec<f32> = Vec::new();
    let segment_count = segments.num_segments();
    for index in 0..segment_count {
        let Some(segment) = segments.get_segment(index) else {
            continue;
        };
        let limit = samples.len().saturating_sub(1);
        let start = ((segment.start * SAMPLES_PER_CENTISECOND) as usize).min(limit);
        let mut end = (segment.end * SAMPLES_PER_CENTISECOND) as usize;
        if index < segment_count - 1 {
            end += overlap_samples;
        }
        let end = end.min(limit);
        if end <= start {
            continue;
        }
        if !speech.is_empty() {
            speech.extend_from_slice(&joiner);
        }
        speech.extend_from_slice(&samples[start..end]);
    }
    Some(speech)
}

fn transcribe_samples(
    context: &WhisperContext,
    samples: &[f32],
    vad_model_path: Option<&Path>,
    language: &str,
    custom_words: &[String],
) -> Result<String, String> {
    let _inference = INFERENCE_LOCK
        .lock()
        .map_err(|_| "Whisper inference lock was poisoned".to_string())?;
    // Strip non-speech audio before decoding: silence and noise are what
    // Whisper hallucinates subtitle boilerplate on.
    let mut filtered_speech;
    let mut samples = samples;
    if let Some(vad_model_path) = vad_model_path {
        if let Some(speech) = speech_only_samples(samples, vad_model_path) {
            if speech.is_empty() {
                return Ok(String::new());
            }
            filtered_speech = speech;
            audio::pad_short_transcription_samples(&mut filtered_speech);
            samples = &filtered_speech;
        }
    }
    let mut state = context
        .create_state()
        .map_err(|error| format!("Could not prepare Whisper: {error}"))?;
    // Beam search decodes noisy audio markedly better than greedy sampling
    // and is what comparable dictation setups use.
    let mut parameters = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    let language = language.trim();
    parameters.set_language((!language.is_empty()).then_some(language));
    parameters.set_print_special(false);
    parameters.set_print_progress(false);
    parameters.set_print_realtime(false);
    parameters.set_print_timestamps(false);
    // Dictation wants speech only — no "[sound of plastic crinkling]" tags.
    parameters.set_suppress_nst(true);
    parameters.set_no_speech_thold(NO_SPEECH_PROBABILITY_LIMIT);
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
        .filter(|segment| !segment_is_silence(segment))
        .filter_map(|segment| clean_transcription_segment(&segment.to_string()))
        .collect::<Vec<_>>()
        .join(" ");
    Ok(text)
}

/// Segments whose no-speech probability reaches this limit are silence
/// candidates. Matches Whisper's conventional default threshold.
const NO_SPEECH_PROBABILITY_LIMIT: f32 = 0.6;
/// Silence candidates are only discarded when the decoded text is also
/// low-confidence (mean token log-probability below this), mirroring the
/// reference Whisper implementation's dual test. Confident speech survives
/// even when its no-speech score is jumpy.
const LOW_CONFIDENCE_MEAN_LOGPROB: f32 = -1.0;

/// Whisper hallucinates subtitle boilerplate ("Thank you for watching!") on
/// silence and noise. Those segments combine a high no-speech probability
/// with low-confidence text; real speech decodes confidently.
fn segment_is_silence(segment: &whisper_rs::WhisperSegment) -> bool {
    if segment.no_speech_probability() < NO_SPEECH_PROBABILITY_LIMIT {
        return false;
    }
    let token_count = segment.n_tokens();
    if token_count == 0 {
        return true;
    }
    let mean_logprob = (0..token_count)
        .filter_map(|index| segment.get_token(index))
        .map(|token| token.token_probability().max(f32::EPSILON).ln())
        .sum::<f32>()
        / token_count as f32;
    mean_logprob < LOW_CONFIDENCE_MEAN_LOGPROB
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

    /// Reproduces the silence-hallucination report ("Thanks for watching!")
    /// against the real local models. Run manually:
    ///   cargo test hallucination -- --ignored --nocapture
    #[test]
    #[ignore]
    fn noise_only_audio_produces_no_text() {
        let home = std::env::var("HOME").expect("HOME is set");
        let models = std::path::PathBuf::from(home).join(".local/share/voxide/models");
        let model = models.join("ggml-small.bin");
        let vad = models.join("ggml-silero-v5.1.2.bin");
        assert!(model.is_file(), "whisper model missing: {model:?}");
        assert!(vad.is_file(), "vad model missing: {vad:?}");
        // 4 seconds of quiet noise with a couple of louder "breath" bursts.
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f64 / u64::MAX as f64) as f32 - 0.5
        };
        let samples: Vec<f32> = (0..64_000)
            .map(|index| {
                let burst = matches!(index, 16_000..=19_200 | 40_000..=44_800);
                random() * if burst { 0.06 } else { 0.008 }
            })
            .collect();
        let with_vad =
            super::transcribe_whisper(samples.clone(), &model, Some(&vad), "en", &[]);
        let without_vad = super::transcribe_whisper(samples, &model, None, "en", &[]);
        println!("with vad: {with_vad:?}");
        println!("without vad: {without_vad:?}");
        assert!(
            with_vad.is_err() || with_vad.as_deref().unwrap_or("").trim().is_empty(),
            "noise-only audio transcribed as {with_vad:?}"
        );
    }

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
