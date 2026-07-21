use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, RawQuery, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tokio::sync::oneshot;

use crate::{
    active_recognition_vocabulary, audio, media, normalize_custom_words,
    post_process_dictation_outcome, provider, provider_api_key, selected_provider, speech,
    transcribe_apple_media_file, valid_whisper_model_file, whisper_model_path, AppState,
    CustomWordEntry, DictionaryEntry, VoiceEngine,
};

const MAX_REQUEST_BYTES: usize = 25 * 1024 * 1024;

#[derive(Default)]
pub struct LocalApiControl {
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    port: Mutex<Option<u16>>,
}

#[derive(Clone)]
struct ApiState {
    app: AppHandle,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

fn json<T: Serialize>(status: StatusCode, value: T) -> Response {
    (status, Json(value)).into_response()
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    json(
        status,
        ErrorBody {
            error: message.into(),
        },
    )
}

pub async fn start(control: &LocalApiControl, app: AppHandle, port: u16) -> Result<u16, String> {
    if control
        .shutdown
        .lock()
        .map_err(|_| "Local API lock was poisoned".to_string())?
        .is_some()
    {
        return control
            .port
            .lock()
            .map_err(|_| "Local API lock was poisoned".to_string())?
            .ok_or("The local API is in an invalid state".into());
    }

    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
        .await
        .map_err(|error| format!("Could not listen on 127.0.0.1:{port}: {error}"))?;
    let actual_port = listener
        .local_addr()
        .map_err(|error| format!("Could not determine local API port: {error}"))?
        .port();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    *control
        .shutdown
        .lock()
        .map_err(|_| "Local API lock was poisoned".to_string())? = Some(shutdown_tx);
    *control
        .port
        .lock()
        .map_err(|_| "Local API lock was poisoned".to_string())? = Some(actual_port);

    let router = router(app);
    tauri::async_runtime::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
    });
    Ok(actual_port)
}

pub fn stop(control: &LocalApiControl) -> Result<(), String> {
    if let Some(shutdown) = control
        .shutdown
        .lock()
        .map_err(|_| "Local API lock was poisoned".to_string())?
        .take()
    {
        let _ = shutdown.send(());
    }
    *control
        .port
        .lock()
        .map_err(|_| "Local API lock was poisoned".to_string())? = None;
    Ok(())
}

pub fn running_port(control: &LocalApiControl) -> Result<Option<u16>, String> {
    control
        .port
        .lock()
        .map(|port| *port)
        .map_err(|_| "Local API lock was poisoned".to_string())
}

fn router(app: AppHandle) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/history", get(history))
        .route(
            "/v1/dictionary/replacements",
            get(get_replacements).post(write_replacements),
        )
        .route(
            "/v1/dictionary/custom-words",
            get(get_custom_words).post(write_custom_words),
        )
        .route("/v1/transcribe", post(transcribe))
        .route("/v1/postprocess", post(postprocess))
        .fallback(route_not_found)
        .method_not_allowed_fallback(method_not_allowed)
        // Axum's extractor default is much smaller than Voxide's public
        // local-API contract. Allow one byte beyond 25 MiB so the endpoints
        // can enforce the exact source limit, and reshape axum's plain-text
        // over-limit rejection into the source JSON error for larger bodies.
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES + 1))
        .layer(axum::middleware::from_fn(source_shaped_payload_limit))
        .with_state(ApiState { app })
}

/// The reference server answers every oversized request with the JSON body
/// `{"error":"Request too large."}`. Axum's body-limit rejection is plain
/// text, so replace any non-JSON 413 with the source shape.
async fn source_shaped_payload_limit(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let response = next.run(request).await;
    let is_json = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|content_type| content_type.starts_with("application/json"));
    if response.status() == StatusCode::PAYLOAD_TOO_LARGE && !is_json {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "Request too large.");
    }
    response
}

async fn route_not_found() -> Response {
    error(StatusCode::NOT_FOUND, "Route not found.")
}

async fn method_not_allowed() -> Response {
    error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed.")
}

async fn health() -> Response {
    json(
        StatusCode::OK,
        serde_json::json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }),
    )
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HistoryResponse {
    count: usize,
    items: Vec<HistoryItem>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HistoryItem {
    id: String,
    #[serde(serialize_with = "serialize_iso8601_seconds")]
    timestamp: chrono::DateTime<Utc>,
    original_text: String,
    final_text: String,
    raw_text: String,
    processed_text: String,
    app_name: String,
    window_title: String,
    character_count: usize,
    was_ai_processed: bool,
    ai_processing_error: Option<String>,
}

fn serialize_iso8601_seconds<S>(value: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&value.to_rfc3339_opts(SecondsFormat::Secs, true))
}

async fn history(State(api): State<ApiState>, RawQuery(query): RawQuery) -> Response {
    let limit = bounded_history_limit(query.as_deref());
    let state = api.app.state::<AppState>();
    let database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Voxide data lock was poisoned",
            )
        }
    };
    let items = database
        .dictation_history
        .iter()
        .take(limit)
        .map(|entry| HistoryItem {
            id: entry.id.clone(),
            timestamp: entry.created_at,
            original_text: entry.raw_text.clone().unwrap_or_else(|| entry.text.clone()),
            final_text: entry.text.clone(),
            raw_text: entry.raw_text.clone().unwrap_or_else(|| entry.text.clone()),
            processed_text: entry.text.clone(),
            app_name: entry.source_application.clone().unwrap_or_default(),
            window_title: entry.source_window_title.clone().unwrap_or_default(),
            character_count: entry.text.chars().count(),
            was_ai_processed: entry.was_ai_processed,
            ai_processing_error: entry.ai_processing_error.clone(),
        })
        .collect::<Vec<_>>();
    json(
        StatusCode::OK,
        HistoryResponse {
            count: items.len(),
            items,
        },
    )
}

/// Mirrors the source bounded-limit helper: an omitted or non-integer limit
/// uses the default instead of making the whole route fail deserialization.
fn bounded_history_limit(query: Option<&str>) -> usize {
    query
        .into_iter()
        .flat_map(|query| query.split('&'))
        .filter_map(|part| part.split_once('='))
        .find_map(|(key, value)| (key == "limit").then_some(value))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100)
        .clamp(1, 1_000)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplacementEntry {
    id: Option<String>,
    triggers: Vec<String>,
    replacement: String,
}

fn api_replacement_entries(dictionary: &[DictionaryEntry]) -> Vec<ReplacementEntry> {
    let mut groups = Vec::<ReplacementEntry>::new();
    for entry in dictionary {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| case_insensitively_equal(&group.replacement, &entry.replacement))
        {
            group.triggers.push(entry.spoken.clone());
        } else {
            groups.push(ReplacementEntry {
                id: Some(entry.id.clone()),
                triggers: vec![entry.spoken.clone()],
                replacement: entry.replacement.clone(),
            });
        }
    }
    groups
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReplacementWriteRequest {
    mode: Option<WriteMode>,
    entries: Option<Vec<ReplacementEntry>>,
    triggers: Option<Vec<String>>,
    replacement: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum WriteMode {
    Append,
    Replace,
}

async fn get_replacements(State(api): State<ApiState>) -> Response {
    let state = api.app.state::<AppState>();
    let database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Voxide data lock was poisoned",
            )
        }
    };
    let items = api_replacement_entries(&database.dictionary);
    json(
        StatusCode::OK,
        serde_json::json!({ "count": items.len(), "items": items }),
    )
}

async fn write_replacements(State(api): State<ApiState>, body: Bytes) -> Response {
    if body.len() > MAX_REQUEST_BYTES {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "Request too large.");
    }
    let (payload, has_entries) = match decode_write_request::<ReplacementWriteRequest>(&body) {
        Ok(payload) => payload,
        Err(decode_error) => {
            return error(
                StatusCode::BAD_REQUEST,
                format!("Invalid replacement payload: {decode_error}"),
            )
        }
    };
    let entries = if has_entries {
        Some(payload.entries.unwrap_or_default())
    } else {
        payload
            .replacement
            .map(|replacement| ReplacementEntry {
                id: None,
                triggers: payload.triggers.unwrap_or_default(),
                replacement,
            })
            .map(|entry| vec![entry])
    };
    let Some(entries) = entries else {
        return error(
            StatusCode::BAD_REQUEST,
            "Expected entries or triggers/replacement",
        );
    };
    let state = api.app.state::<AppState>();
    let updated = match state.update(|database| {
        let mut stored = if payload.mode == Some(WriteMode::Replace) {
            Vec::new()
        } else {
            database.dictionary.clone()
        };
        let mut incoming = Vec::new();
        for entry in entries {
            let replacement = entry.replacement;
            let triggers = entry
                .triggers
                .iter()
                .map(|trigger| trim_swift_whitespace(trigger).to_lowercase())
                .filter(|trigger| !trigger.is_empty())
                .collect::<Vec<_>>();
            if triggers.is_empty() || replacement.trim().is_empty() {
                continue;
            }
            // The reference stores a replacement group rather than one record
            // per trigger. Its overwrite rule is group-based: a new entry with
            // the same replacement replaces every existing trigger in that
            // group, even when the caller supplies a different trigger list.
            stored.retain(|existing| {
                entry.id.as_deref() != Some(existing.id.as_str())
                    && !case_insensitively_equal(&existing.replacement, &replacement)
            });
            incoming.retain(|existing: &DictionaryEntry| {
                entry.id.as_deref() != Some(existing.id.as_str())
                    && !case_insensitively_equal(&existing.replacement, &replacement)
            });
            let group_id = entry.id.clone().unwrap_or_else(|| {
                format!(
                    "dictionary-api-{}",
                    Utc::now().timestamp_nanos_opt().unwrap_or_default()
                )
            });
            for (index, spoken) in triggers.into_iter().enumerate() {
                incoming.push(DictionaryEntry {
                    id: if index == 0 {
                        group_id.clone()
                    } else {
                        format!("{group_id}-{index}")
                    },
                    spoken,
                    replacement: replacement.clone(),
                    created_at: Utc::now(),
                });
            }
        }
        incoming.extend(stored);
        database.dictionary = incoming;
        database.dictionary.clone()
    }) {
        Ok(updated) => updated,
        Err(message) => return error(StatusCode::INTERNAL_SERVER_ERROR, message),
    };
    let items = api_replacement_entries(&updated);
    json(
        StatusCode::OK,
        serde_json::json!({ "count": items.len(), "items": items }),
    )
}

#[derive(Debug, Clone, Deserialize)]
struct CustomWordWriteRequest {
    mode: Option<WriteMode>,
    entries: Option<Vec<CustomWordEntry>>,
    text: Option<String>,
    weight: Option<f32>,
    aliases: Option<Vec<String>>,
}

async fn get_custom_words(State(api): State<ApiState>) -> Response {
    let state = api.app.state::<AppState>();
    let database = match state.database.lock() {
        Ok(database) => database,
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Voxide data lock was poisoned",
            )
        }
    };
    json(
        StatusCode::OK,
        serde_json::json!({ "count": database.custom_words.len(), "items": database.custom_words }),
    )
}

async fn write_custom_words(State(api): State<ApiState>, body: Bytes) -> Response {
    if body.len() > MAX_REQUEST_BYTES {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "Request too large.");
    }
    let (payload, has_entries) = match decode_write_request::<CustomWordWriteRequest>(&body) {
        Ok(payload) => payload,
        Err(decode_error) => {
            return error(
                StatusCode::BAD_REQUEST,
                format!("Invalid custom words payload: {decode_error}"),
            )
        }
    };
    let entries = if has_entries {
        Some(payload.entries.unwrap_or_default())
    } else {
        payload.text.map(|text| {
            vec![CustomWordEntry {
                text,
                weight: payload.weight,
                aliases: payload.aliases.unwrap_or_default(),
            }]
        })
    };
    let Some(entries) = entries else {
        return error(StatusCode::BAD_REQUEST, "Expected entries or text");
    };
    let state = api.app.state::<AppState>();
    let updated = match state.update(|database| {
        let mut words = if payload.mode == Some(WriteMode::Replace) {
            Vec::new()
        } else {
            database.custom_words.clone()
        };
        for word in entries {
            let text = word.text.trim().to_owned();
            if text.is_empty() {
                continue;
            }
            words.retain(|existing| !case_insensitively_equal(&existing.text, &text));
            words.push(CustomWordEntry {
                text,
                weight: word.weight,
                aliases: word.aliases,
            });
        }
        database.custom_words = normalize_custom_words(words);
        database.custom_words.clone()
    }) {
        Ok(updated) => updated,
        Err(message) => return error(StatusCode::INTERNAL_SERVER_ERROR, message),
    };
    json(
        StatusCode::OK,
        serde_json::json!({ "count": updated.len(), "items": updated }),
    )
}

/// Codable's keyed containers distinguish an omitted `entries` key from an
/// explicit JSON `null`. The reference treats the latter as an explicitly
/// supplied empty list, so preserve that distinction before deserializing the
/// request's typed fields.
fn decode_write_request<T: serde::de::DeserializeOwned>(
    body: &[u8],
) -> Result<(T, bool), serde_json::Error> {
    let value = serde_json::from_slice::<serde_json::Value>(body)?;
    let has_entries = value
        .as_object()
        .is_some_and(|object| object.contains_key("entries"));
    let payload = serde_json::from_value(value)?;
    Ok((payload, has_entries))
}

/// `CharacterSet.whitespaces`, used by the source entry initializer, trims
/// horizontal whitespace but deliberately leaves line breaks intact.
fn trim_swift_whitespace(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character != '\r' && character != '\n' && character.is_whitespace()
    })
}

fn case_insensitively_equal(left: &str, right: &str) -> bool {
    left.to_lowercase() == right.to_lowercase()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TranscribeRequest {
    path: Option<String>,
    audio_base64: Option<String>,
    filename: Option<String>,
}

struct TemporaryRequestAudio {
    path: PathBuf,
}

impl TemporaryRequestAudio {
    fn create(bytes: &[u8], filename: Option<&str>) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("Missing audio body".into());
        }
        let extension = filename
            .and_then(|filename| Path::new(filename).extension())
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .filter(|extension| {
                !extension.is_empty()
                    && extension.len() <= 10
                    && extension
                        .chars()
                        .all(|character| character.is_ascii_alphanumeric())
            })
            .unwrap_or_else(|| "wav".into());
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        for attempt in 0..10_u8 {
            let path = std::env::temp_dir().join(format!(
                "voxide-local-api-{}-{nonce}-{attempt}.{extension}",
                std::process::id()
            ));
            let file = match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "Could not prepare temporary request audio: {error}"
                    ))
                }
            };
            let mut file = file;
            if let Err(error) = file.write_all(bytes).and_then(|()| file.flush()) {
                let _ = fs::remove_file(&path);
                return Err(format!("Could not store temporary request audio: {error}"));
            }
            return Ok(Self { path });
        }
        Err("Could not allocate temporary request audio after several attempts".into())
    }
}

impl Drop for TemporaryRequestAudio {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscribeResponse {
    text: String,
    confidence: f32,
    sample_count: usize,
    provider: String,
}

/// The Local API distinguishes complete media files from raw request audio.
/// Raw data may still be an FFmpeg-readable container, but is ultimately sent
/// to the ASR engine as one padded sample buffer, matching `transcribeSamplesForAPI`.
fn decode_request_media_samples(path: &Path) -> Result<(Vec<f32>, usize), String> {
    let duration_ms = media::file_duration_ms(path)?;
    let audio = media::decode_audio_segment(path, 0.0, duration_ms as f64 / 1_000.0)?;
    let mut samples = audio::mono_resample_for_whisper(audio)?;
    let sample_count = samples.len();
    if samples.is_empty() {
        return Err("No audio was captured".into());
    }
    audio::pad_short_transcription_samples(&mut samples);
    Ok((samples, sample_count))
}

async fn transcribe(State(api): State<ApiState>, headers: HeaderMap, body: Bytes) -> Response {
    if body.len() > MAX_REQUEST_BYTES {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "Request too large.");
    }
    let is_json = is_json_content_type(&headers);
    let request = if is_json {
        match serde_json::from_slice::<TranscribeRequest>(&body) {
            Ok(request) => request,
            Err(decode_error) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("Invalid JSON audio payload: {decode_error}"),
                )
            }
        }
    } else {
        TranscribeRequest {
            path: None,
            audio_base64: Some(STANDARD.encode(&body)),
            filename: headers
                .get("x-filename")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
        }
    };
    let has_explicit_path = request
        .path
        .as_deref()
        .is_some_and(|path| !path.trim().is_empty());
    let (input_path, _temporary_audio) = match request.path.filter(|path| !path.trim().is_empty()) {
        Some(path) => (PathBuf::from(path), None),
        None => match request
            .audio_base64
            .as_deref()
            .ok_or_else(|| "Missing audio path or audioBase64".to_owned())
            .and_then(|encoded| {
                STANDARD
                    .decode(encoded)
                    .map_err(|error| format!("Invalid audioBase64: {error}"))
            })
            .and_then(|bytes| TemporaryRequestAudio::create(&bytes, request.filename.as_deref()))
        {
            Ok(temporary) => (temporary.path.clone(), Some(temporary)),
            Err(message) => return error(StatusCode::BAD_REQUEST, message),
        },
    };
    let raw_audio_path = (!has_explicit_path).then(|| input_path.clone());
    let path = has_explicit_path.then_some(input_path);
    let state = api.app.state::<AppState>();
    let (settings, custom_words) = match state.database.lock() {
        Ok(database) => {
            let settings = database.settings.clone();
            let custom_words = active_recognition_vocabulary(&settings, &database.custom_words);
            (settings, custom_words)
        }
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Voxide data lock was poisoned",
            )
        }
    };
    let language = settings.language.clone();
    let transcribed = match settings.selected_voice_engine {
        VoiceEngine::Whisper => {
            let model_path = match whisper_model_path(&settings, &state) {
                Ok(path) => path,
                Err(message) => return error(StatusCode::BAD_REQUEST, message),
            };
            if !valid_whisper_model_file(&model_path) {
                return error(
                    StatusCode::BAD_REQUEST,
                    "The selected Whisper model is missing, empty, or invalid. Download it again before using local API inference.",
                );
            }
            match path {
                Some(path) => {
                    let path = std::path::PathBuf::from(path);
                    match tauri::async_runtime::spawn_blocking(move || {
                        speech::transcribe_media_file(
                            &path,
                            &model_path,
                            &language,
                            &custom_words,
                            None,
                        )
                        .map(|(text, duration_ms)| {
                            let sample_count = duration_ms
                                .saturating_mul(16)
                                .try_into()
                                .unwrap_or(usize::MAX);
                            (text, sample_count)
                        })
                    })
                    .await
                    {
                        Ok(result) => result,
                        Err(task_error) => {
                            return error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Voice engine task failed: {task_error}"),
                            )
                        }
                    }
                }
                None => {
                    let (samples, sample_count) = match raw_audio_path
                        .as_deref()
                        .ok_or_else(|| "Missing audio path or audioBase64".to_owned())
                        .and_then(decode_request_media_samples)
                    {
                        Ok(samples) => samples,
                        Err(message) => return error(StatusCode::BAD_REQUEST, message),
                    };
                    match tauri::async_runtime::spawn_blocking(move || {
                        speech::transcribe_whisper(samples, &model_path, &language, &custom_words)
                            .map(|text| (text, sample_count))
                    })
                    .await
                    {
                        Ok(result) => result,
                        Err(task_error) => {
                            return error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Voice engine task failed: {task_error}"),
                            )
                        }
                    }
                }
            }
            .map(|(text, sample_count)| (text, sample_count, "Whisper".to_owned()))
        }
        VoiceEngine::Cloud => {
            let profile = match state.database.lock() {
                Ok(database) => match selected_provider(&database, None) {
                    Ok(profile) => profile,
                    Err(message) => return error(StatusCode::BAD_REQUEST, message),
                },
                Err(_) => {
                    return error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Voxide data lock was poisoned",
                    )
                }
            };
            let api_key = match provider_api_key(&profile.id) {
                Ok(api_key) => api_key,
                Err(message) => return error(StatusCode::BAD_REQUEST, message),
            };
            let result = match path {
                Some(path) => {
                    let path = std::path::PathBuf::from(path);
                    provider::transcribe_openai_compatible_media(
                        &profile,
                        api_key.as_deref(),
                        &settings.cloud_transcription_model,
                        &language,
                        &path,
                        None,
                    )
                    .await
                    .map(|(text, duration_ms)| {
                        let sample_count = duration_ms
                            .saturating_mul(16)
                            .try_into()
                            .unwrap_or(usize::MAX);
                        (text, sample_count)
                    })
                }
                None => {
                    let (samples, sample_count) = match raw_audio_path
                        .as_deref()
                        .ok_or_else(|| "Missing audio path or audioBase64".to_owned())
                        .and_then(decode_request_media_samples)
                    {
                        Ok(samples) => samples,
                        Err(message) => return error(StatusCode::BAD_REQUEST, message),
                    };
                    let wav = match audio::wav_bytes_from_16khz_mono(&samples) {
                        Ok(wav) => wav,
                        Err(message) => return error(StatusCode::BAD_REQUEST, message),
                    };
                    provider::transcribe_openai_compatible_audio(
                        &profile,
                        api_key.as_deref(),
                        &settings.cloud_transcription_model,
                        &language,
                        wav,
                    )
                    .await
                    .map(|text| (text, sample_count))
                }
            };
            result.map(|(text, sample_count)| (text, sample_count, profile.name))
        }
        VoiceEngine::AppleSpeech => match path {
            Some(path) => {
                let path = std::path::PathBuf::from(path);
                match tauri::async_runtime::spawn_blocking(move || {
                    transcribe_apple_media_file(&path, &language, &custom_words, None).map(
                        |(text, duration_ms)| {
                            let sample_count = duration_ms
                                .saturating_mul(16)
                                .try_into()
                                .unwrap_or(usize::MAX);
                            (text, sample_count)
                        },
                    )
                })
                .await
                {
                    Ok(result) => result,
                    Err(task_error) => {
                        return error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Apple Speech task failed: {task_error}"),
                        )
                    }
                }
            }
            None => {
                let (samples, sample_count) = match raw_audio_path
                    .as_deref()
                    .ok_or_else(|| "Missing audio path or audioBase64".to_owned())
                    .and_then(decode_request_media_samples)
                {
                    Ok(samples) => samples,
                    Err(message) => return error(StatusCode::BAD_REQUEST, message),
                };
                match tauri::async_runtime::spawn_blocking(move || {
                    crate::apple_speech::transcribe_samples(&samples, &language, &custom_words)
                        .map(|text| (text, sample_count))
                })
                .await
                {
                    Ok(result) => result,
                    Err(task_error) => {
                        return error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Apple Speech task failed: {task_error}"),
                        )
                    }
                }
            }
        }
        .map(|(text, sample_count)| (text, sample_count, "Apple Speech".to_owned())),
        _ => {
            return error(
                StatusCode::BAD_REQUEST,
                "The selected portable voice engine does not expose local API inference yet",
            )
        }
    };
    let (text, sample_count, provider) = match transcribed {
        Ok(result) => result,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    json(
        StatusCode::OK,
        TranscribeResponse {
            text,
            confidence: 1.0,
            sample_count,
            provider,
        },
    )
}

#[derive(Debug, Deserialize)]
struct TextRequest {
    text: String,
}

async fn postprocess(State(api): State<ApiState>, headers: HeaderMap, body: Bytes) -> Response {
    if body.len() > MAX_REQUEST_BYTES {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "Request too large.");
    }
    let is_json = is_json_content_type(&headers);
    let text = if is_json {
        match serde_json::from_slice::<TextRequest>(&body) {
            Ok(request) => request.text,
            Err(decode_error) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("Invalid JSON text payload: {decode_error}"),
                )
            }
        }
    } else {
        match String::from_utf8(body.to_vec()) {
            Ok(text) => text,
            Err(_) => return error(StatusCode::BAD_REQUEST, "Text body must be UTF-8"),
        }
    };
    let state = api.app.state::<AppState>();
    let provider = match state.database.lock() {
        Ok(database) => match selected_provider(&database, None) {
            Ok(profile) => profile,
            Err(message) => return error(StatusCode::BAD_REQUEST, message),
        },
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Voxide data lock was poisoned",
            )
        }
    };
    let mut ignore_delta = |_delta: &str| {};
    match post_process_dictation_outcome(
        &state,
        text,
        true,
        None,
        None,
        None,
        None,
        &mut ignore_delta,
    )
    .await
    {
        Ok(outcome) if outcome.ai_fallback_error.is_none() => json(
            StatusCode::OK,
            serde_json::json!({
                "text": outcome.text,
                "provider": provider.id,
                "model": outcome.processing_model.unwrap_or(provider.model),
            }),
        ),
        Ok(outcome) => error(
            StatusCode::BAD_REQUEST,
            outcome
                .ai_fallback_error
                .unwrap_or_else(|| "AI post-processing failed".into()),
        ),
        Err(message) => error(StatusCode::BAD_REQUEST, message),
    }
}

fn is_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("application/json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_and_pads_short_wav_request_audio() {
        let path = std::env::temp_dir().join(format!(
            "voxide-local-api-test-{}.wav",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("WAV writer initializes");
        for index in 0..8_000 {
            writer
                .write_sample(if index == 1 { i16::MAX / 2 } else { 0_i16 })
                .expect("sample writes");
        }
        writer.finalize().expect("WAV finalizes");
        let (samples, sample_count) =
            decode_request_media_samples(&path).expect("WAV request media decodes");
        std::fs::remove_file(&path).expect("temporary WAV is removed");
        assert_eq!(sample_count, 8_000);
        assert_eq!(samples.len(), audio::MINIMUM_TRANSCRIPTION_SAMPLES);
        assert!(samples[1] > 0.49);
    }

    #[test]
    fn temporary_request_audio_preserves_the_filename_extension_and_cleans_up() {
        let temporary = TemporaryRequestAudio::create(&[1, 2, 3], Some("recording.webm"))
            .expect("temporary request audio is created");
        let path = temporary.path.clone();
        assert_eq!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("webm")
        );
        assert_eq!(
            std::fs::read(&path).expect("temporary bytes read"),
            vec![1, 2, 3]
        );

        drop(temporary);
        assert!(!path.exists());
    }

    #[test]
    fn recognizes_json_content_types_case_insensitively() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            "Application/JSON; charset=utf-8".parse().unwrap(),
        );

        assert!(is_json_content_type(&headers));
    }

    #[test]
    fn history_limit_matches_the_source_fallback_and_bounds() {
        assert_eq!(bounded_history_limit(None), 100);
        assert_eq!(bounded_history_limit(Some("limit=invalid")), 100);
        assert_eq!(bounded_history_limit(Some("limit=0")), 1);
        assert_eq!(bounded_history_limit(Some("limit=1200")), 1_000);
        assert_eq!(bounded_history_limit(Some("other=true&limit=25")), 25);
    }

    #[test]
    fn history_dates_use_the_source_iso8601_seconds_format() {
        let item = HistoryItem {
            id: "dictation-test".into(),
            timestamp: DateTime::parse_from_rfc3339("2026-07-20T12:34:56.789Z")
                .expect("fixed timestamp parses")
                .with_timezone(&Utc),
            original_text: "raw".into(),
            final_text: "final".into(),
            raw_text: "raw".into(),
            processed_text: "final".into(),
            app_name: String::new(),
            window_title: String::new(),
            character_count: 5,
            was_ai_processed: false,
            ai_processing_error: None,
        };
        let value = serde_json::to_value(item).expect("history item serializes");
        assert_eq!(value["timestamp"], "2026-07-20T12:34:56Z");
    }

    #[test]
    fn dictionary_write_payloads_distinguish_null_entries_from_omission() {
        let (replacement, has_replacement_entries) =
            decode_write_request::<ReplacementWriteRequest>(
                br#"{"entries":null,"triggers":["ignored"],"replacement":"Ignored"}"#,
            )
            .expect("replacement payload decodes");
        assert!(has_replacement_entries);
        assert!(replacement.entries.is_none());

        let (custom_words, has_custom_word_entries) =
            decode_write_request::<CustomWordWriteRequest>(br#"{"text":"Voxide"}"#)
                .expect("custom vocabulary payload decodes");
        assert!(!has_custom_word_entries);
        assert!(custom_words.entries.is_none());
        assert_eq!(custom_words.text.as_deref(), Some("Voxide"));
    }

    #[test]
    fn replacement_triggers_preserve_newlines_like_the_reference_initializer() {
        assert_eq!(trim_swift_whitespace(" \tVoxide\n "), "Voxide\n");
    }

    #[test]
    fn replacement_api_groups_multiple_triggers_by_replacement() {
        let created_at = Utc::now();
        let entries = api_replacement_entries(&[
            DictionaryEntry {
                id: "dictionary-1".into(),
                spoken: "vox side".into(),
                replacement: "Voxide".into(),
                created_at,
            },
            DictionaryEntry {
                id: "dictionary-2".into(),
                spoken: "voxide app".into(),
                replacement: "voxide".into(),
                created_at,
            },
            DictionaryEntry {
                id: "dictionary-3".into(),
                spoken: "codex".into(),
                replacement: "Codex".into(),
                created_at,
            },
        ]);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id.as_deref(), Some("dictionary-1"));
        assert_eq!(
            entries[0].triggers,
            vec!["vox side".to_string(), "voxide app".to_string()]
        );
        assert_eq!(entries[1].replacement, "Codex");
    }

    #[test]
    fn replacement_groups_use_unicode_case_insensitive_comparison() {
        let created_at = Utc::now();
        let entries = api_replacement_entries(&[
            DictionaryEntry {
                id: "dictionary-1".into(),
                spoken: "first".into(),
                replacement: "Äpp".into(),
                created_at,
            },
            DictionaryEntry {
                id: "dictionary-2".into(),
                spoken: "second".into(),
                replacement: "äpp".into(),
                created_at,
            },
        ]);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].triggers, ["first", "second"]);
    }
}
