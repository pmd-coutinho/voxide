use std::{
    collections::HashSet,
    ffi::{c_void, CStr},
    os::raw::c_int,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};

use serde::{Deserialize, Serialize};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams,
};

use crate::{audio, debug_log, media};

pub type ProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync + 'static>;

/// The most recently used Whisper model, kept warm across transcriptions.
/// Loading a model reads hundreds of megabytes from disk, which would
/// otherwise be paid at the start of every dictation.
static CONTEXT_CACHE: OnceLock<Mutex<Option<(PathBuf, Arc<WhisperContext>)>>> = OnceLock::new();

/// A state owns Whisper's reusable compute buffers. Previews deliberately
/// keep their own state: an aborted preview must never leave the final decode
/// reusing partial decoder state. Inference remains serialized below because
/// the two states still share one Whisper context/GPU backend.
static FINAL_STATE_CACHE: OnceLock<Mutex<Option<(PathBuf, WhisperState)>>> = OnceLock::new();
static PREVIEW_STATE_CACHE: OnceLock<Mutex<Option<(PathBuf, WhisperState)>>> = OnceLock::new();

/// Silero's model is small but loading it for every utterance is still
/// needless startup work on the stop-to-text path.
static VAD_CONTEXT_CACHE: OnceLock<Mutex<Option<(PathBuf, WhisperVadContext)>>> = OnceLock::new();

/// Serializes inference: the live preview and the final transcription share
/// the warm context, and concurrent runs on one context corrupt each other
/// on GPU backends. The final pass simply waits out an in-flight preview.
static INFERENCE_LOCK: Mutex<()> = Mutex::new(());

/// A final transcription gets priority over previews. A preview only ever
/// takes the inference lock when no final pass is waiting, preventing a new
/// preview tick from extending stop-to-text latency.
static FINAL_INFERENCE_PENDING: AtomicU64 = AtomicU64::new(0);

/// Makes the final/preview priority decision atomic with acquiring the
/// inference mutex. It is held only long enough to admit work, never during
/// VAD or decoding.
static INFERENCE_ADMISSION_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BeamSize {
    /// Beam 5 on an available hardware GPU; greedy on a CPU fallback.
    #[default]
    Auto,
    Greedy,
    Beam2,
    Beam5,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TranscriptionTimings {
    pub lock_wait_ms: u64,
    pub vad_ms: u64,
    pub state_ms: u64,
    pub decode_ms: u64,
}

#[derive(Debug)]
pub struct WhisperTranscription {
    pub text: String,
    pub timings: TranscriptionTimings,
}

/// Selects final-quality or cheap, cancellable preview decoding. The preview
/// generation is owned by the capture state; incrementing it makes any
/// in-flight preview return from whisper.cpp at its next abort check.
#[derive(Clone)]
pub struct TranscriptionOptions {
    beam_size: BeamSize,
    preview_generation: Option<(Arc<AtomicU64>, u64)>,
}

impl TranscriptionOptions {
    pub fn final_decode(beam_size: BeamSize) -> Self {
        Self {
            beam_size,
            preview_generation: None,
        }
    }

    pub fn preview(generation: Arc<AtomicU64>, expected_generation: u64) -> Self {
        Self {
            beam_size: BeamSize::Greedy,
            preview_generation: Some((generation, expected_generation)),
        }
    }

    fn is_preview(&self) -> bool {
        self.preview_generation.is_some()
    }
}

/// Storage passed to whisper.cpp while a preview decode is running. It lives
/// on the Rust stack for the whole synchronous `state.full` call below, so the
/// C callback never observes dangling state.
struct PreviewAbortCallback {
    generation: Arc<AtomicU64>,
    expected_generation: u64,
}

unsafe extern "C" fn should_abort_preview(user_data: *mut c_void) -> bool {
    // The callback is installed only with a pointer to PreviewAbortCallback
    // whose owner remains alive until `state.full` has returned.
    let callback = unsafe { &*(user_data as *const PreviewAbortCallback) };
    callback.generation.load(Ordering::SeqCst) != callback.expected_generation
}

/// Loads the model into the warm cache ahead of the first dictation.
pub fn preload_whisper(model_path: &Path) -> Result<(), String> {
    load_context(model_path).map(|_| ())
}

/// Preloads the VAD model after it has been downloaded at application startup.
pub fn preload_vad(vad_model_path: &Path) -> Result<(), String> {
    let model = vad_model_path
        .to_str()
        .ok_or_else(|| "The VAD model path is not valid Unicode".to_string())?;
    let cache = VAD_CONTEXT_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| "Whisper VAD cache lock was poisoned".to_string())?;
    if cached
        .as_ref()
        .is_some_and(|(path, _)| path == vad_model_path)
    {
        return Ok(());
    }
    let mut params = WhisperVadContextParams::new();
    params.set_use_gpu(false);
    let vad = WhisperVadContext::new(model, params)
        .map_err(|error| format!("Could not load Whisper VAD: {error}"))?;
    *cached = Some((vad_model_path.to_path_buf(), vad));
    Ok(())
}

#[cfg(test)]
pub fn transcribe_whisper(
    samples: Vec<f32>,
    model_path: &Path,
    vad_model_path: Option<&Path>,
    language: &str,
    custom_words: &[String],
) -> Result<String, String> {
    transcribe_whisper_with_options(
        samples,
        model_path,
        vad_model_path,
        language,
        custom_words,
        TranscriptionOptions::final_decode(BeamSize::Auto),
    )
    .map(|result| result.text)
}

pub fn transcribe_whisper_with_options(
    samples: Vec<f32>,
    model_path: &Path,
    vad_model_path: Option<&Path>,
    language: &str,
    custom_words: &[String],
    options: TranscriptionOptions,
) -> Result<WhisperTranscription, String> {
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
    let result = transcribe_samples(
        model_path,
        &context,
        &samples,
        vad_model_path,
        language,
        custom_words,
        &options,
    )?;
    if result.text.is_empty() && !options.is_preview() {
        return Err("The voice engine did not recognize any speech".into());
    }
    Ok(result)
}

pub fn transcribe_media_file(
    path: &Path,
    model_path: &Path,
    vad_model_path: Option<&Path>,
    language: &str,
    custom_words: &[String],
    beam_size: BeamSize,
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
        let result = transcribe_samples(
            model_path,
            &context,
            &samples,
            vad_model_path,
            language,
            custom_words,
            &TranscriptionOptions::final_decode(beam_size),
        )?;
        if !result.text.is_empty() {
            text.push(result.text);
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
            let discrete =
                device_type == whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU;
            let integrated =
                device_type == whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU;
            if !discrete && !integrated {
                continue;
            }
            let description = CStr::from_ptr(whisper_rs_sys::ggml_backend_dev_description(device))
                .to_string_lossy()
                .to_lowercase();
            let software = description.contains("llvmpipe") || description.contains("swiftshader");
            // A software rasterizer is slower than the CPU fallback and must
            // not accidentally win just because it is the only "GPU" entry.
            if software {
                gpu_index += 1;
                continue;
            }
            let score = if discrete { 2 } else { 1 };
            if best.map_or(true, |(best_score, _)| score > best_score) {
                best = Some((score, gpu_index));
            }
            gpu_index += 1;
        }
    }
    best.map(|(_, index)| index)
}

fn selected_gpu_description(selected: Option<c_int>) -> String {
    let Some(selected) = selected else {
        return "none (CPU fallback if no backend becomes available)".into();
    };
    let mut gpu_index: c_int = 0;
    unsafe {
        for device_index in 0..whisper_rs_sys::ggml_backend_dev_count() {
            let device = whisper_rs_sys::ggml_backend_dev_get(device_index);
            let device_type = whisper_rs_sys::ggml_backend_dev_type(device);
            let is_gpu = matches!(
                device_type,
                whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                    | whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            );
            if !is_gpu {
                continue;
            }
            if gpu_index == selected {
                let description =
                    CStr::from_ptr(whisper_rs_sys::ggml_backend_dev_description(device))
                        .to_string_lossy();
                return format!("{description} (index {selected})");
            }
            gpu_index += 1;
        }
    }
    format!("index {selected}")
}

fn has_hardware_gpu_backend() -> bool {
    preferred_gpu_device().is_some()
}

fn physical_cpu_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        let mut cores = HashSet::new();
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !name.starts_with("cpu")
                    || name.len() <= 3
                    || !name[3..].bytes().all(|byte| byte.is_ascii_digit())
                {
                    continue;
                }
                let topology = entry.path().join("topology");
                let package = std::fs::read_to_string(topology.join("physical_package_id"));
                let core = std::fs::read_to_string(topology.join("core_id"));
                if let (Ok(package), Ok(core)) = (package, core) {
                    cores.insert((package.trim().to_owned(), core.trim().to_owned()));
                }
            }
        }
        if !cores.is_empty() {
            return cores.len();
        }
    }
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

pub(crate) fn cpu_decode_threads() -> c_int {
    physical_cpu_count().clamp(1, c_int::MAX as usize) as c_int
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
    let selected_gpu = preferred_gpu_device();
    if let Some(device) = selected_gpu {
        parameters.gpu_device(device);
    }
    debug_log::append(&format!(
        "Whisper context loading (gpu_requested: true, selected_gpu: {})",
        selected_gpu_description(selected_gpu)
    ));
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
    let ranges = detect_speech_ranges(probe, vad_model_path, 2_000)?;
    const EDGE_MARGIN_SAMPLES: usize = (audio::WHISPER_SAMPLE_RATE / 2) as usize;
    let bounds = ranges
        .into_iter()
        .fold(None::<(usize, usize)>, |bounds, (start, end)| {
            Some(match bounds {
                Some((first, last)) => (first.min(start), last.max(end)),
                None => (start, end),
            })
        });
    Some(match bounds {
        Some((first, last)) => SpeechBounds::Range(
            first.saturating_sub(EDGE_MARGIN_SAMPLES),
            last.saturating_add(EDGE_MARGIN_SAMPLES).min(probe.len()),
        ),
        None => SpeechBounds::None,
    })
}

/// Returns the individual VAD speech ranges. Final transcription keeps the
/// original audio intact, but preview decoding removes long internal pauses:
/// Whisper otherwise has an opportunity to turn those pauses into subtitle
/// boilerplate such as "Thank you".
fn detect_speech_ranges(
    probe: &[f32],
    vad_model_path: &Path,
    minimum_silence_duration_ms: i32,
) -> Option<Vec<(usize, usize)>> {
    const SAMPLES_PER_CENTISECOND: f32 = audio::WHISPER_SAMPLE_RATE as f32 / 100.0;
    preload_vad(vad_model_path).ok()?;
    let cache = VAD_CONTEXT_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache.lock().ok()?;
    let (_, vad) = cached.as_mut().filter(|(path, _)| path == vad_model_path)?;
    // whisper.cpp's VAD defaults split on 100 ms of silence; dictation wants
    // faster-whisper's tuning, where an utterance with natural pauses reads
    // as one padded stretch of speech. As a pure gate the params are biased
    // toward acceptance: a false accept just decodes audio that downstream
    // filters still guard, while a false reject eats the user's dictation.
    let mut vad_params = WhisperVadParams::new();
    vad_params.set_threshold(0.35);
    vad_params.set_min_speech_duration(150);
    vad_params.set_min_silence_duration(minimum_silence_duration_ms);
    vad_params.set_speech_pad(400);
    let segments = vad.segments_from_samples(vad_params, probe).ok()?;
    let mut ranges = Vec::new();
    for index in 0..segments.num_segments() {
        let Some(segment) = segments.get_segment(index) else {
            continue;
        };
        let start = (segment.start * SAMPLES_PER_CENTISECOND) as usize;
        let end = (segment.end * SAMPLES_PER_CENTISECOND) as usize;
        if end > start {
            ranges.push((start.min(probe.len()), end.min(probe.len())));
        }
    }
    // VAD speech padding can make neighboring ranges overlap. Merge them so
    // reconstructing preview audio never duplicates a word at a boundary.
    let mut merged = Vec::<(usize, usize)>::new();
    for (start, end) in ranges {
        if let Some((_, previous_end)) = merged
            .last_mut()
            .filter(|(_, previous_end)| start <= *previous_end)
        {
            *previous_end = (*previous_end).max(end);
        } else {
            merged.push((start, end));
        }
    }
    Some(merged)
}

fn preview_speech_only_samples(
    samples: &[f32],
    probe: &[f32],
    vad_model_path: &Path,
) -> Option<Vec<f32>> {
    // A 650 ms gap marks a completed phrase for the preview. Keep a short
    // separator after splicing segments so Whisper retains a natural word
    // boundary without decoding multi-second room silence.
    const PREVIEW_MIN_SILENCE_MS: i32 = 650;
    const PREVIEW_GAP_SAMPLES: usize = (audio::WHISPER_SAMPLE_RATE / 8) as usize;
    let ranges = detect_speech_ranges(probe, vad_model_path, PREVIEW_MIN_SILENCE_MS)?;
    if ranges.is_empty() {
        return Some(Vec::new());
    }
    let capacity = ranges.iter().map(|(start, end)| end - start).sum::<usize>()
        + PREVIEW_GAP_SAMPLES.saturating_mul(ranges.len().saturating_sub(1));
    let mut speech = Vec::with_capacity(capacity);
    for (index, (start, end)) in ranges.into_iter().enumerate() {
        if index != 0 {
            speech.extend(std::iter::repeat_n(0.0, PREVIEW_GAP_SAMPLES));
        }
        speech.extend_from_slice(&samples[start..end]);
    }
    Some(speech)
}

fn transcribe_samples(
    model_path: &Path,
    context: &WhisperContext,
    samples: &[f32],
    vad_model_path: Option<&Path>,
    language: &str,
    custom_words: &[String],
    options: &TranscriptionOptions,
) -> Result<WhisperTranscription, String> {
    let lock_started = Instant::now();
    let _inference = if options.is_preview() {
        let _admission = INFERENCE_ADMISSION_LOCK
            .lock()
            .map_err(|_| "Whisper inference admission lock was poisoned".to_string())?;
        if FINAL_INFERENCE_PENDING.load(Ordering::SeqCst) != 0 {
            return Err("Whisper preview skipped: final transcription is pending".into());
        }
        INFERENCE_LOCK.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => {
                "Whisper preview skipped: inference is busy".to_string()
            }
            std::sync::TryLockError::Poisoned(_) => {
                "Whisper inference lock was poisoned".to_string()
            }
        })?
    } else {
        let _admission = INFERENCE_ADMISSION_LOCK
            .lock()
            .map_err(|_| "Whisper inference admission lock was poisoned".to_string())?;
        FINAL_INFERENCE_PENDING.fetch_add(1, Ordering::SeqCst);
        drop(_admission);
        let guard = INFERENCE_LOCK
            .lock()
            .map_err(|_| "Whisper inference lock was poisoned".to_string());
        FINAL_INFERENCE_PENDING.fetch_sub(1, Ordering::SeqCst);
        guard?
    };
    let mut timings = TranscriptionTimings {
        lock_wait_ms: lock_started.elapsed().as_millis() as u64,
        ..Default::default()
    };
    // VAD gates both live snapshots and final audio. Final decoding keeps its
    // contiguous audio; preview decoding additionally removes long internal
    // pauses, which are a common source of Whisper's live hallucinations.
    // Detection runs on a peak-normalized probe copy.
    let mut vad_prepared_samples: Vec<f32>;
    let mut samples = samples;
    if let Some(vad_model_path) = vad_model_path {
        let vad_started = Instant::now();
        let mut probe = samples.to_vec();
        normalize_peak(&mut probe);
        if options.is_preview() {
            match preview_speech_only_samples(samples, &probe, vad_model_path) {
                Some(speech) if speech.is_empty() => {
                    timings.vad_ms = vad_started.elapsed().as_millis() as u64;
                    return Ok(WhisperTranscription {
                        text: String::new(),
                        timings,
                    });
                }
                Some(speech) => {
                    vad_prepared_samples = speech;
                    audio::pad_short_transcription_samples(&mut vad_prepared_samples);
                    samples = &vad_prepared_samples;
                }
                None => {}
            }
        } else {
            match detect_speech_bounds(&probe, vad_model_path) {
                Some(SpeechBounds::None) => {
                    timings.vad_ms = vad_started.elapsed().as_millis() as u64;
                    return Ok(WhisperTranscription {
                        text: String::new(),
                        timings,
                    });
                }
                Some(SpeechBounds::Range(start, end)) if start > 0 || end < samples.len() => {
                    vad_prepared_samples = samples[start.min(samples.len())..end].to_vec();
                    audio::pad_short_transcription_samples(&mut vad_prepared_samples);
                    samples = &vad_prepared_samples;
                }
                _ => {}
            }
        }
        timings.vad_ms = vad_started.elapsed().as_millis() as u64;
    }
    let state_started = Instant::now();
    let state_cache = if options.is_preview() {
        PREVIEW_STATE_CACHE.get_or_init(|| Mutex::new(None))
    } else {
        FINAL_STATE_CACHE.get_or_init(|| Mutex::new(None))
    };
    let mut cached_state = state_cache
        .lock()
        .map_err(|_| "Whisper state cache lock was poisoned".to_string())?;
    if !cached_state
        .as_ref()
        .is_some_and(|(path, _)| path == model_path)
    {
        let state = context
            .create_state()
            .map_err(|error| format!("Could not prepare Whisper: {error}"))?;
        *cached_state = Some((model_path.to_path_buf(), state));
    }
    timings.state_ms = state_started.elapsed().as_millis() as u64;
    let (_, state) = cached_state
        .as_mut()
        .expect("Whisper state cache is populated above");
    let sampling_strategy = if options.is_preview() {
        // CUDA reaches the same sub-second budget with Beam 5 on the short,
        // VAD-spliced preview input. Matching the final decoder avoids the
        // greedy/Beam-2 pause hallucinations seen in live dictation. CPU
        // fallbacks keep greedy to avoid contending with the recording UI.
        if has_hardware_gpu_backend() {
            SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        }
    } else {
        match options.beam_size {
            BeamSize::Auto if has_hardware_gpu_backend() => SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            },
            BeamSize::Auto | BeamSize::Greedy => SamplingStrategy::Greedy { best_of: 1 },
            BeamSize::Beam2 => SamplingStrategy::BeamSearch {
                beam_size: 2,
                patience: -1.0,
            },
            BeamSize::Beam5 => SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            },
        }
    };
    let mut parameters = FullParams::new(sampling_strategy);
    let language = language.trim();
    parameters.set_language((!language.is_empty()).then_some(language));
    parameters.set_print_special(false);
    parameters.set_print_progress(false);
    parameters.set_print_realtime(false);
    parameters.set_print_timestamps(false);
    // Dictation wants speech only — no "[sound of plastic crinkling]" tags.
    parameters.set_suppress_nst(true);
    parameters.set_no_speech_thold(NO_SPEECH_PROBABILITY_LIMIT);
    if !has_hardware_gpu_backend() {
        parameters.set_n_threads(cpu_decode_threads());
    }
    let preview_abort =
        options
            .preview_generation
            .as_ref()
            .map(|(generation, expected_generation)| PreviewAbortCallback {
                generation: Arc::clone(generation),
                expected_generation: *expected_generation,
            });
    if let Some(callback) = preview_abort.as_ref() {
        // whisper-rs 0.16's convenience abort-callback wrapper does not keep
        // a correctly typed allocation for this callback. Use the raw API
        // with explicitly owned, scoped storage instead.
        unsafe {
            parameters.set_abort_callback(Some(should_abort_preview));
            parameters.set_abort_callback_user_data(
                (callback as *const PreviewAbortCallback).cast_mut().cast(),
            );
        }
    }
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
    let decode_started = Instant::now();
    state
        .full(parameters, samples)
        .map_err(|error| format!("Whisper could not transcribe the recording: {error}"))?;
    timings.decode_ms = decode_started.elapsed().as_millis() as u64;

    let text = state
        .as_iter()
        .filter(|segment| !segment_is_silence(segment))
        .filter_map(|segment| clean_transcription_segment(&segment.to_string()))
        .collect::<Vec<_>>()
        .join(" ");
    Ok(WhisperTranscription { text, timings })
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
    let cleaned = without_dash
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
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
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|sample| sample.unwrap())
                .collect(),
        };
        println!(
            "samples: {} ({:.2}s)",
            samples.len(),
            samples.len() as f32 / 16_000.0
        );
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
            println!(
                "  segment {index}: start={} end={}",
                segment.start, segment.end
            );
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

    /// Measures the actual local decode path without the desktop capture or
    /// insertion stages. Run a 16 kHz mono WAV through CPU/Vulkan/CUDA builds:
    ///   VOXIDE_TEST_WAV=... cargo test --release decode_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn decode_bench_on_real_wav() {
        let wav = std::env::var("VOXIDE_TEST_WAV").expect("set VOXIDE_TEST_WAV");
        let home = std::env::var("HOME").expect("HOME is set");
        let models = std::path::PathBuf::from(home).join(".local/share/voxide/models");
        let model = std::env::var_os("VOXIDE_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| models.join("ggml-large-v3-turbo.bin"));
        let vad = models.join("ggml-silero-v5.1.2.bin");
        assert!(model.is_file(), "whisper model missing: {model:?}");
        assert!(vad.is_file(), "VAD model missing: {vad:?}");

        let mut reader = hound::WavReader::open(&wav).expect("wav opens");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000, "decode bench expects 16 kHz WAV");
        assert_eq!(spec.channels, 1, "decode bench expects mono WAV");
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|sample| sample.expect("valid WAV sample") as f32 / i16::MAX as f32)
                .collect(),
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|sample| sample.expect("valid WAV sample"))
                .collect(),
        };
        let result = super::transcribe_whisper_with_options(
            samples.clone(),
            &model,
            Some(&vad),
            "en",
            &[],
            super::TranscriptionOptions::final_decode(super::BeamSize::Auto),
        )
        .expect("decode succeeds");
        println!(
            "decode_bench (lock_wait_ms: {}, vad_ms: {}, state_ms: {}, decode_ms: {}, text_chars: {})",
            result.timings.lock_wait_ms,
            result.timings.vad_ms,
            result.timings.state_ms,
            result.timings.decode_ms,
            result.text.chars().count(),
        );

        // Exercise the exact live-preview options over a growing recording.
        // This catches regressions where a warm preview state, cancellation,
        // or partial-audio handling silently leaves the overlay at
        // "Listening…" while final transcription still succeeds.
        let generation = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1));
        let mut preview_emissions = 0;
        for seconds in 1..=(samples.len() / crate::audio::WHISPER_SAMPLE_RATE as usize) {
            let end = (seconds * crate::audio::WHISPER_SAMPLE_RATE as usize).min(samples.len());
            let preview = super::transcribe_whisper_with_options(
                samples[..end].to_vec(),
                &model,
                Some(&vad),
                "en",
                &[],
                super::TranscriptionOptions::preview(std::sync::Arc::clone(&generation), 1),
            );
            match preview {
                Ok(preview) => {
                    if !preview.text.trim().is_empty() {
                        preview_emissions += 1;
                    }
                    println!(
                        "preview_bench (audio_s: {seconds}, decode_ms: {}, text_chars: {})",
                        preview.timings.decode_ms,
                        preview.text.chars().count(),
                    );
                }
                Err(error) => println!("preview_bench (audio_s: {seconds}, error: {error})"),
            }
        }
        assert!(
            preview_emissions > 0,
            "no live-preview snapshot produced text"
        );

        generation.store(2, std::sync::atomic::Ordering::SeqCst);
        let cancelled = super::transcribe_whisper_with_options(
            samples[..crate::audio::WHISPER_SAMPLE_RATE as usize].to_vec(),
            &model,
            Some(&vad),
            "en",
            &[],
            super::TranscriptionOptions::preview(std::sync::Arc::clone(&generation), 1),
        );
        assert!(
            cancelled.is_err(),
            "a stale preview generation must abort before decoding"
        );
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
        let with_vad = super::transcribe_whisper(samples.clone(), &model, Some(&vad), "en", &[]);
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
