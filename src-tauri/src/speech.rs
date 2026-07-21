use std::{path::Path, sync::Arc};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{audio, media};

pub type ProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync + 'static>;

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

fn load_context(model_path: &Path) -> Result<WhisperContext, String> {
    if !model_path.is_file() {
        return Err(format!(
            "Whisper model is not installed: {}",
            model_path.display()
        ));
    }
    WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .map_err(|error| format!("Could not load Whisper model: {error}"))
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
        .map(|segment| segment.to_string())
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_owned();
    Ok(text)
}
