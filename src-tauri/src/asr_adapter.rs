//! Engine-specific live and file transcription adapters.
//!
//! This module owns the existing engine dispatch while the command layer owns
//! session admission, capture, and post-processing. Keeping it separate makes
//! lifecycle behavior reviewable without a disruptive engine rewrite.

use crate::*;

impl VoiceEngine {
    /// Validate the immutable engine portion of a live session before a
    /// microphone is opened. This makes a bad selection fail without creating
    /// an invisible capture, and keeps selection separate from installation.
    pub(crate) fn prepare_live_capture(
        self,
        settings: &Settings,
        state: &AppState,
    ) -> Result<(), String> {
        match self {
            Self::Whisper => {
                let model = whisper_model_path(settings, state)?;
                if valid_whisper_model_file(&model) {
                    Ok(())
                } else {
                    Err("The selected Whisper model is missing, empty, or invalid. Download it again before recording.".into())
                }
            }
            Self::Parakeet => {
                if !parakeet::is_compiled() {
                    return Err("Parakeet is available in Voxide's CUDA build".into());
                }
                let model = parakeet_model_path(state)?;
                if parakeet_model_is_verified(&model) {
                    Ok(())
                } else {
                    Err("The Parakeet model is missing. Download it before recording.".into())
                }
            }
            Self::Nemotron => {
                if !nemotron::is_compiled() {
                    return Err("Nemotron is available in Voxide's CUDA build for Linux/NVIDIA".into());
                }
                let runtime = nemotron_runtime_path(state)?;
                if !nemotron_runtime_is_verified(&runtime) {
                    return Err("Install the Nemotron CUDA runtime from Voice Engine before dictating.".into());
                }
                let model = nemotron_model_path(state)?;
                if !nemotron_model_is_verified(&model) {
                    return Err("Download the Nemotron model from Voice Engine before dictating.".into());
                }
                ensure_nemotron_server_script(&runtime).map(|_| ())
            }
            Self::Cloud => cloud_transcription_readiness(settings, state).map(|_| ()),
            Self::AppleSpeech if apple_speech::is_supported() => Ok(()),
            Self::AppleSpeech => Err("Apple Speech is available only on macOS. Select Whisper or a compatible cloud provider on this platform.".into()),
        }
    }

    /// Reserve or reset state owned by an engine for one coordinator
    /// generation. Called only after the coordinator admitted the session.
    pub(crate) fn begin_live_session(
        self,
        capture_state: &NativeCaptureState,
        session_id: u64,
    ) -> Result<(), String> {
        if self.is_nemotron() {
            let mut live = match capture_state.nemotron_live.try_lock() {
                Ok(live) => live,
                Err(_) => return Err("Nemotron is still finishing the previous dictation".into()),
            };
            live.generation = session_id;
            live.fed_samples = 0;
            live.session_started = false;
            live.start_error = None;
        }
        if self.is_parakeet() {
            *capture_state
                .parakeet_live
                .lock()
                .map_err(|_| "Parakeet live state lock was poisoned".to_string())? =
                ParakeetLiveState {
                    generation: session_id,
                    ..Default::default()
                };
        }
        Ok(())
    }

    /// Starts the selected engine's preview adapter. Each adapter receives the
    /// same canonical capture timeline and coordinator generation; only its
    /// declared preview mode determines how it consumes that timeline.
    pub(crate) fn spawn_live_preview(
        self,
        app: AppHandle,
        capture_state: &NativeCaptureState,
        session_id: u64,
        state: &AppState,
        settings: &Settings,
        custom_words: Vec<String>,
        cloud_profile: Option<provider::AiProviderProfile>,
        dictionary: Vec<DictionaryEntry>,
    ) -> Result<(), String> {
        match self {
            Self::Whisper if settings.enable_streaming_preview => {
                let model = whisper_model_path(settings, state)?;
                spawn_live_whisper_preview(
                    app,
                    session_id,
                    Arc::clone(&capture_state.preview_generation),
                    model,
                    vad_model_path(state),
                    settings.language.clone(),
                    custom_words,
                    settings.transcription_preview_char_limit,
                );
            }
            Self::Parakeet if settings.enable_streaming_preview => {
                let model = parakeet_model_path(state)?;
                spawn_live_parakeet_preview(
                    app,
                    session_id,
                    model,
                    settings.clone(),
                    dictionary,
                    settings.transcription_preview_char_limit,
                );
            }
            Self::Nemotron => {
                let runtime = nemotron_runtime_path(state)?;
                let model = nemotron_model_path(state)?;
                let script = ensure_nemotron_server_script(&runtime)?;
                spawn_live_nemotron_stream(
                    app,
                    session_id,
                    runtime,
                    script,
                    model,
                    settings.language.clone(),
                    settings.nemotron_streaming_mode.lookahead_tokens(),
                    settings.enable_streaming_preview,
                    settings.transcription_preview_char_limit,
                );
            }
            Self::Cloud if settings.enable_streaming_preview => {
                if let Some(profile) = cloud_profile {
                    if let Ok(api_key) = provider_api_key(&profile.id) {
                        spawn_live_cloud_preview(
                            app,
                            session_id,
                            profile,
                            api_key,
                            settings.cloud_transcription_model.clone(),
                            settings.language.clone(),
                            settings.transcription_preview_char_limit,
                        );
                    }
                }
            }
            Self::AppleSpeech if settings.enable_streaming_preview => {
                spawn_live_apple_speech_preview(
                    app,
                    session_id,
                    settings.apple_speech_locale.clone(),
                    custom_words,
                    settings.transcription_preview_char_limit,
                );
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) async fn transcribe_live_final(
        self,
        state: &AppState,
        capture_state: &NativeCaptureState,
        settings: &Settings,
        recording_generation: u64,
        samples: Vec<f32>,
        custom_words: Vec<String>,
    ) -> Result<EngineFinalTranscript, String> {
        self.prepare_live_capture(settings, state)?;
        match self {
            Self::Whisper => {
                let model = whisper_model_path(settings, state)?;
                let language = settings.language.clone();
                let vad_model = vad_model_path(state);
                let beam_size = settings.whisper_beam_size;
                let result = tauri::async_runtime::spawn_blocking(move || {
                    speech::transcribe_whisper_with_options(
                        samples,
                        &model,
                        vad_model.as_deref(),
                        &language,
                        &custom_words,
                        speech::TranscriptionOptions::final_decode(beam_size),
                    )
                })
                .await
                .map_err(|error| format!("Voice engine task failed: {error}"))??;
                Ok(EngineFinalTranscript {
                    text: result.text,
                    whisper_timings: Some(result.timings),
                })
            }
            Self::Parakeet => {
                let model = parakeet_model_path(state)?;
                let text = tauri::async_runtime::spawn_blocking(move || {
                    transcribe_parakeet_final(&samples, &model, &custom_words)
                })
                .await
                .map_err(|error| format!("Parakeet voice engine task failed: {error}"))??;
                Ok(EngineFinalTranscript {
                    text,
                    whisper_timings: None,
                })
            }
            Self::Nemotron => {
                let runtime = nemotron_runtime_path(state)?;
                let model = nemotron_model_path(state)?;
                let script = ensure_nemotron_server_script(&runtime)?;
                let text = finish_nemotron_live(
                    capture_state,
                    recording_generation,
                    &samples,
                    &runtime,
                    &script,
                    &model,
                    &settings.language,
                    settings.nemotron_streaming_mode.lookahead_tokens(),
                )
                .await?;
                Ok(EngineFinalTranscript {
                    text,
                    whisper_timings: None,
                })
            }
            Self::Cloud => {
                let profile = {
                    let database = state
                        .database
                        .lock()
                        .map_err(|_| "Voxide data lock was poisoned".to_string())?;
                    selected_provider(&database, None)?
                };
                let api_key = provider_api_key(&profile.id)?;
                let wav = audio::wav_bytes_from_16khz_mono(&samples)?;
                let text = provider::transcribe_openai_compatible_audio(
                    &profile,
                    api_key.as_deref(),
                    &settings.cloud_transcription_model,
                    &settings.language,
                    wav,
                )
                .await?;
                Ok(EngineFinalTranscript {
                    text,
                    whisper_timings: None,
                })
            }
            Self::AppleSpeech => {
                let language = settings.apple_speech_locale.clone();
                let text = tauri::async_runtime::spawn_blocking(move || {
                    apple_speech::transcribe_samples(&samples, &language, &custom_words)
                })
                .await
                .map_err(|error| format!("Apple Speech task failed: {error}"))??;
                Ok(EngineFinalTranscript {
                    text,
                    whisper_timings: None,
                })
            }
        }
    }

    /// File transcription shares the same selected-engine adapter as live
    /// dictation, but intentionally keeps its independent media decoding and
    /// progress semantics. The command layer receives only a final result.
    pub(crate) async fn transcribe_file(
        self,
        state: &AppState,
        settings: &Settings,
        path: PathBuf,
        custom_words: Vec<String>,
        progress: speech::ProgressCallback,
    ) -> Result<(String, u64), String> {
        match self {
            Self::Whisper => {
                let model = whisper_model_path(settings, state)?;
                if !valid_whisper_model_file(&model) {
                    return Err("The selected Whisper model is missing, empty, or invalid. Download it again before transcribing a file.".into());
                }
                let language = settings.language.clone();
                let vad_model = vad_model_path(state);
                let beam_size = settings.whisper_beam_size;
                tauri::async_runtime::spawn_blocking(move || {
                    speech::transcribe_media_file(
                        &path,
                        &model,
                        vad_model.as_deref(),
                        &language,
                        &custom_words,
                        beam_size,
                        Some(progress),
                    )
                })
                .await
                .map_err(|error| format!("File transcription task failed: {error}"))?
            }
            Self::Parakeet => {
                if !parakeet::is_compiled() {
                    return Err("Parakeet is available in Voxide's CUDA build".into());
                }
                let model = parakeet_model_path(state)?;
                if !parakeet_model_is_verified(&model) {
                    return Err(
                        "The Parakeet model is missing. Download it before transcribing a file."
                            .into(),
                    );
                }
                tauri::async_runtime::spawn_blocking(move || {
                    parakeet::transcribe_media_file(&path, &model, Some(progress))
                })
                .await
                .map_err(|error| format!("Parakeet file transcription task failed: {error}"))?
            }
            Self::Nemotron => {
                transcribe_nemotron_media_file(
                    state,
                    &path,
                    &settings.language,
                    settings.nemotron_streaming_mode.lookahead_tokens(),
                    Some(progress),
                )
                .await
            }
            Self::Cloud => {
                let profile = {
                    let database = state
                        .database
                        .lock()
                        .map_err(|_| "Voxide data lock was poisoned".to_string())?;
                    selected_provider(&database, None)?
                };
                let api_key = provider_api_key(&profile.id)?;
                provider::transcribe_openai_compatible_media(
                    &profile,
                    api_key.as_deref(),
                    &settings.cloud_transcription_model,
                    &settings.language,
                    &path,
                    Some(progress),
                )
                .await
            }
            Self::AppleSpeech => {
                let language = settings.apple_speech_locale.clone();
                tauri::async_runtime::spawn_blocking(move || {
                    transcribe_apple_media_file(&path, &language, &custom_words, Some(progress))
                })
                .await
                .map_err(|error| format!("Apple Speech task failed: {error}"))?
            }
        }
    }

    /// Best-effort warm-up for the selected local engine. It never changes
    /// readiness or selection: a failed preload is merely recorded and the
    /// regular adapter still performs its normal validation on use.
    pub(crate) async fn preload(
        self,
        state: &AppState,
        settings: &Settings,
        vocabulary: Vec<String>,
    ) {
        match self {
            Self::Whisper => {
                ensure_vad_model(state).await;
                let vad = vad_model_path(state);
                let model = match whisper_model_path(settings, state) {
                    Ok(path) if path.is_file() => path,
                    _ => return,
                };
                let _ = tauri::async_runtime::spawn_blocking(move || {
                    if let Some(vad) = vad {
                        match speech::preload_vad(&vad) {
                            Ok(()) => debug_log::append("Whisper VAD model preloaded"),
                            Err(error) => {
                                debug_log::append(&format!("Whisper VAD preload failed: {error}"))
                            }
                        }
                    }
                    match speech::preload_whisper(&model) {
                        Ok(()) => debug_log::append("Whisper model preloaded"),
                        Err(error) => {
                            debug_log::append(&format!("Whisper preload failed: {error}"))
                        }
                    }
                })
                .await;
            }
            Self::Parakeet if parakeet::is_compiled() => {
                let model = match parakeet_model_path(state) {
                    Ok(path) if parakeet_model_is_verified(&path) => path,
                    _ => return,
                };
                let _ = tauri::async_runtime::spawn_blocking(move || {
                    match parakeet::preload(&model) {
                        Ok(()) => debug_log::append("Parakeet CUDA preview model preloaded"),
                        Err(error) => debug_log::append(&format!(
                            "Parakeet preview preload failed: {error}"
                        )),
                    }
                    if vocabulary.is_empty() {
                        return;
                    }
                    match parakeet::preload_with_vocabulary(&model, &vocabulary) {
                        Ok(()) => {
                            debug_log::append("Parakeet CUDA vocabulary final model preloaded")
                        }
                        Err(error) => debug_log::append(&format!(
                            "Parakeet vocabulary preload failed; final dictation will fall back if needed: {error}"
                        )),
                    }
                })
                .await;
            }
            Self::Parakeet | Self::Nemotron | Self::Cloud | Self::AppleSpeech => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_openai_compatible_cloud_engine_prepares_without_a_key_or_network() {
        let mut database = AppDatabase::default();
        database.settings.selected_voice_engine = VoiceEngine::Cloud;
        database.settings.selected_ai_provider = "ollama".into();
        database.settings.cloud_transcription_model = "whisper-1".into();
        let state = AppState {
            database: Mutex::new(database.clone()),
            path: std::env::temp_dir().join(format!(
                "voxide-asr-adapter-test-{}.json",
                uuid::Uuid::new_v4()
            )),
            startup_recovery_notice: Mutex::new(None),
        };

        VoiceEngine::Cloud
            .prepare_live_capture(&database.settings, &state)
            .expect("local OpenAI-compatible endpoints do not require a key at capture start");
    }

    #[test]
    fn parakeet_reservation_resets_adapter_state_for_the_admitted_generation() {
        let capture_state = NativeCaptureState::default();
        {
            let mut live = capture_state
                .parakeet_live
                .lock()
                .expect("Parakeet live lock");
            live.generation = 4;
            live.previous_full_text = "stale preview".into();
        }

        VoiceEngine::Parakeet
            .begin_live_session(&capture_state, 9)
            .expect("reservation should not require a CUDA model");

        let live = capture_state
            .parakeet_live
            .lock()
            .expect("Parakeet live lock");
        assert_eq!(live.generation, 9);
        assert!(live.previous_full_text.is_empty());
    }

    #[test]
    fn nemotron_reservation_resets_adapter_state_for_the_admitted_generation() {
        let capture_state = NativeCaptureState::default();
        {
            let mut live = capture_state
                .nemotron_live
                .try_lock()
                .expect("Nemotron live lock");
            live.generation = 4;
            live.fed_samples = 1_024;
            live.session_started = true;
            live.start_error = Some("stale stream failure".into());
        }

        VoiceEngine::Nemotron
            .begin_live_session(&capture_state, 9)
            .expect("reservation should reset state before the CUDA stream starts");

        let live = capture_state
            .nemotron_live
            .try_lock()
            .expect("Nemotron live lock");
        assert_eq!(live.generation, 9);
        assert_eq!(live.fed_samples, 0);
        assert!(!live.session_started);
        assert!(live.start_error.is_none());
    }

    #[test]
    fn nemotron_reservation_refuses_a_stream_that_is_still_finalizing() {
        let capture_state = NativeCaptureState::default();
        let _finalizing_stream = capture_state
            .nemotron_live
            .try_lock()
            .expect("Nemotron live lock");

        let error = VoiceEngine::Nemotron
            .begin_live_session(&capture_state, 9)
            .expect_err("an active finalization must retain the single Nemotron stream");

        assert_eq!(error, "Nemotron is still finishing the previous dictation");
    }
}
