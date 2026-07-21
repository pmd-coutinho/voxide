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

/// Amplifies quiet recordings to a healthy level for voice-activity
/// detection. Gain is derived from a high percentile of the magnitudes, not
/// the absolute peak, so a single click or plosive pop cannot veto the boost
/// that quiet speech needs; capped so near-silence is not blown up into
/// full-scale noise.
fn normalize_peak(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mut magnitudes: Vec<f32> = samples.iter().map(|sample| sample.abs()).collect();
    let index = (magnitudes.len() - 1).saturating_mul(995) / 1000;
    magnitudes.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
    let reference = magnitudes[index];
    if reference > 1e-4 && reference < 0.9 {
        let gain = (0.9 / reference).clamp(1.0, 40.0);
        for sample in samples.iter_mut() {
            *sample = (*sample * gain).clamp(-1.0, 1.0);
        }
    }
}

/// Voice-activity verdict for a recording: where speech begins and ends, or
/// None when the audio contains no speech at all.
enum SpeechBounds {
    None,
    Range(usize, usize),
}

/// Runs Silero voice-activity detection as a gate, not a scalpel: it decides
/// whether speech exists and where its outer edges are, but the audio Whisper
/// decodes is never cut apart or re-stitched — mid-utterance surgery costs
/// transcription accuracy. Detection runs on the (normalized) probe signal;
/// returns None when VAD itself is unavailable.
fn detect_speech_bounds(probe: &[f32], vad_model_path: &Path) -> Option<SpeechBounds> {
    const SAMPLES_PER_CENTISECOND: f32 = audio::WHISPER_SAMPLE_RATE as f32 / 100.0;
    // Generous margin around the detected speech so edge trimming can never
    // clip a first or last word.
    const EDGE_MARGIN_SAMPLES: usize = (audio::WHISPER_SAMPLE_RATE / 2) as usize;
    let model = vad_model_path.to_str()?;
    let mut context_params = WhisperVadContextParams::new();
    context_params.set_use_gpu(false);
    let mut vad = WhisperVadContext::new(model, context_params).ok()?;
    // whisper.cpp's VAD defaults split on 100 ms of silence; dictation wants
    // faster-whisper's tuning, where an utterance with natural pauses reads
    // as one padded stretch of speech. As a pure gate the params are biased
    // toward acceptance: a false accept just decodes audio that downstream
    // filters still guard, while a false reject eats the user's dictation.
    let mut vad_params = WhisperVadParams::new();
    vad_params.set_threshold(0.35);
    vad_params.set_min_speech_duration(150);
    vad_params.set_min_silence_duration(2_000);
    vad_params.set_speech_pad(400);
    let segments = vad.segments_from_samples(vad_params, probe).ok()?;
    let mut bounds: Option<(usize, usize)> = None;
    for index in 0..segments.num_segments() {
        let Some(segment) = segments.get_segment(index) else {
            continue;
        };
        let start = (segment.start * SAMPLES_PER_CENTISECOND) as usize;
        let end = (segment.end * SAMPLES_PER_CENTISECOND) as usize;
        bounds = Some(match bounds {
            Some((first, last)) => (first.min(start), last.max(end)),
            None => (start, end),
        });
    }
    Some(match bounds {
        Some((first, last)) => SpeechBounds::Range(
            first.saturating_sub(EDGE_MARGIN_SAMPLES),
            last.saturating_add(EDGE_MARGIN_SAMPLES).min(probe.len()),
        ),
        None => SpeechBounds::None,
    })
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
    // Whisper hallucinates subtitle boilerplate on silence and noise, so a
    // voice-activity gate decides whether speech exists and where its outer
    // edges are. Detection runs on a peak-normalized probe copy (quiet
    // built-in microphones starve the VAD model), but Whisper decodes the
    // ORIGINAL audio — unamplified and contiguous — trimmed only at the
    // edges, because that is what transcribes most accurately.
    let mut edge_trimmed;
    let mut samples = samples;
    if let Some(vad_model_path) = vad_model_path {
        let mut probe = samples.to_vec();
        normalize_peak(&mut probe);
        match detect_speech_bounds(&probe, vad_model_path) {
            Some(SpeechBounds::None) => return Ok(String::new()),
            Some(SpeechBounds::Range(start, end)) if start > 0 || end < samples.len() => {
                edge_trimmed = samples[start.min(samples.len())..end].to_vec();
                audio::pad_short_transcription_samples(&mut edge_trimmed);
                samples = &edge_trimmed;
            }
            _ => {}
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

    /// Debugs VAD behavior on a real recording. Point VOXIDE_TEST_WAV at a
    /// 16 kHz mono WAV and run:
    ///   VOXIDE_TEST_WAV=... cargo test vad_debug -- --ignored --nocapture
    #[test]
    #[ignore]
    fn vad_debug_on_real_recording() {
        use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};
        let wav = std::env::var("VOXIDE_TEST_WAV").expect("set VOXIDE_TEST_WAV");
        let home = std::env::var("HOME").expect("HOME is set");
        let models = std::path::PathBuf::from(home).join(".local/share/voxide/models");
        let mut reader = hound::WavReader::open(&wav).expect("wav opens");
        let spec = reader.spec();
        println!("wav: {spec:?}");
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|sample| sample.unwrap() as f32 / i16::MAX as f32)
                .collect(),
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|sample| sample.unwrap()).collect()
            }
        };
        println!("samples: {} ({:.2}s)", samples.len(), samples.len() as f32 / 16_000.0);
        let vad_path = models.join("ggml-silero-v5.1.2.bin");
        let mut context_params = WhisperVadContextParams::new();
        context_params.set_use_gpu(false);
        let mut vad = WhisperVadContext::new(vad_path.to_str().unwrap(), context_params)
            .expect("vad context");
        let segments = vad
            .segments_from_samples(WhisperVadParams::new(), &samples)
            .expect("vad segments");
        println!("vad segments: {}", segments.num_segments());
        for index in 0..segments.num_segments() {
            let segment = segments.get_segment(index).unwrap();
            println!("  segment {index}: start={} end={}", segment.start, segment.end);
        }
        let mut probe = samples.clone();
        super::normalize_peak(&mut probe);
        match super::detect_speech_bounds(&probe, &vad_path) {
            Some(super::SpeechBounds::Range(start, end)) => {
                println!("speech bounds: {start}..{end} of {}", samples.len());
            }
            Some(super::SpeechBounds::None) => println!("speech bounds: none"),
            None => println!("speech bounds: vad unavailable"),
        }
        let model = models.join("ggml-small.bin");
        let with_vad =
            super::transcribe_whisper(samples.clone(), &model, Some(&vad_path), "en", &[]);
        let without_vad = super::transcribe_whisper(samples, &model, None, "en", &[]);
        println!("with vad: {with_vad:?}");
        println!("without vad: {without_vad:?}");
    }

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
