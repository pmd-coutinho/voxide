use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use chrono::{DateTime, Datelike, Local, NaiveDate, Timelike, Utc, Weekday};
use directories::{BaseDirs, ProjectDirs};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
};
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize, State};
use tauri_plugin_autostart::ManagerExt as AutoStartManagerExt;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
#[cfg(not(target_os = "linux"))]
use tauri_plugin_notification::NotificationExt;
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

mod analytics;
mod apple_speech;
mod asr;
mod asr_adapter;
mod audio;
mod debug_log;
mod formatting;
mod local_api;
mod media;
mod nemotron;
mod parakeet;
mod permissions;
#[cfg(target_os = "linux")]
mod portal_hotkeys;
mod provider;
mod session;
mod speech;
#[cfg(unix)]
pub mod trigger;
mod typing;
mod update;

const DATABASE_FILE: &str = "voxide.json";
const DATABASE_SCHEMA_VERSION: u32 = 1;
const DATABASE_BACKUP_COUNT: usize = 3;
const BACKUP_VERSION: u32 = 1;
const TRANSCRIPTION_PREVIEW_MIN_CHARACTERS: usize = 50;
const TRANSCRIPTION_PREVIEW_MAX_CHARACTERS: usize = 800;
const TRANSCRIPTION_PREVIEW_CHARACTER_STEP: usize = 50;
const DEFAULT_TRANSCRIPTION_PREVIEW_CHARACTERS: usize = 150;
const NEMOTRON_RUNTIME_ID: &str = "nemotron-cuda-runtime";
const NEMOTRON_MODEL_DOWNLOAD_ID: &str = "nemotron-3.5-asr-streaming-0.6b";
const NEMOTRON_SERVER_SOURCE: &str = include_str!("../scripts/nemotron_cuda_server.py");
const NEMOTRON_RUNTIME_VERSION: &str = "1";
const NEMOTRON_RUNTIME_HEALTH_FILE: &str = "runtime-health.json";
const NEMOTRON_RUNTIME_MARKER_FILE: &str = ".voxide-nemotron-runtime-v1";
const COMPONENT_RECEIPT_FILE: &str = ".voxide-component-receipt.json";
const COMPONENT_RECEIPT_SCHEMA: u32 = 1;
const COMPONENT_STAGING_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
struct AppDatabase {
    settings: Settings,
    dictation_history: Vec<DictationEntry>,
    file_transcription_history: Vec<FileTranscriptionEntry>,
    dictionary: Vec<DictionaryEntry>,
    #[serde(default)]
    dictionary_learning_records: Vec<DictionaryLearningRecord>,
    #[serde(default)]
    dictionary_learning_last_shown_at: Option<DateTime<Utc>>,
    prompt_profiles: Vec<PromptProfile>,
    #[serde(default)]
    dictation_prompt_configurations: Vec<DictationPromptConfiguration>,
    #[serde(default)]
    app_prompt_bindings: Vec<AppPromptBinding>,
    ai_providers: Vec<provider::AiProviderProfile>,
    custom_words: Vec<CustomWordEntry>,
    command_chats: Vec<CommandChat>,
    active_command_chat_id: Option<String>,
}

/// The on-disk envelope is deliberately separate from `AppDatabase` so schema
/// migrations are explicit and a newer application never normalizes over data
/// it does not understand. Older Voxide installs wrote `AppDatabase` directly;
/// `OnDiskDatabase` retains that read path only for the one-time migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersistedDatabase {
    schema_version: u32,
    data: AppDatabase,
}

/// A component-owned record written only after every declared artifact has
/// verified and before the staging directory becomes active. It is an audit
/// record, not a readiness marker, so older valid installations remain usable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ComponentReceipt {
    schema: u32,
    id: String,
    version: String,
    source: String,
    files: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OnDiskDatabase {
    Versioned(PersistedDatabase),
    Legacy(AppDatabase),
}

fn decode_persisted_database(contents: &str) -> Result<(AppDatabase, bool), String> {
    match serde_json::from_str(contents).map_err(|error| error.to_string())? {
        OnDiskDatabase::Versioned(envelope) => {
            if envelope.schema_version > DATABASE_SCHEMA_VERSION {
                return Err(format!(
                    "this data uses schema version {}, but this Voxide version supports up to {}",
                    envelope.schema_version, DATABASE_SCHEMA_VERSION
                ));
            }
            Ok((
                envelope.data,
                envelope.schema_version != DATABASE_SCHEMA_VERSION,
            ))
        }
        OnDiskDatabase::Legacy(database) => Ok((database, true)),
    }
}

impl Default for AppDatabase {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            dictation_history: Vec::new(),
            file_transcription_history: Vec::new(),
            dictionary: Vec::new(),
            dictionary_learning_records: Vec::new(),
            dictionary_learning_last_shown_at: None,
            prompt_profiles: editable_prompt_modes()
                .into_iter()
                .map(default_prompt_profile)
                .collect(),
            dictation_prompt_configurations: Vec::new(),
            app_prompt_bindings: Vec::new(),
            ai_providers: provider::AiProviderProfile::built_in(),
            custom_words: Vec::new(),
            command_chats: vec![CommandChat::new()],
            active_command_chat_id: None,
        }
    }
}

fn normalize_command_chats(database: &mut AppDatabase) {
    database
        .command_chats
        .sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    database.command_chats.truncate(30);
    for chat in &mut database.command_chats {
        if chat.messages.len() > 100 {
            chat.messages.drain(..chat.messages.len() - 100);
        }
    }
    let active_is_valid = database
        .active_command_chat_id
        .as_deref()
        .is_some_and(|id| database.command_chats.iter().any(|chat| chat.id == id));
    if !active_is_valid {
        if database.command_chats.is_empty() {
            database.command_chats.push(CommandChat::new());
        }
        database.active_command_chat_id =
            database.command_chats.first().map(|chat| chat.id.clone());
    }
}

/// Keep the persisted provider list compatible with the fixed built-in provider
/// catalog. The Swift app keeps built-ins outside its custom-provider list; the
/// Tauri port persists a unified list so the settings UI can edit their endpoint
/// and model. Restoring a missing built-in here prevents an old backup or a
/// partial write from turning a built-in into an unselectable dangling setting.
fn restore_missing_builtin_provider_profiles(providers: &mut Vec<provider::AiProviderProfile>) {
    for builtin in provider::AiProviderProfile::built_in() {
        if let Some(saved) = providers
            .iter_mut()
            .find(|profile| profile.id.eq_ignore_ascii_case(&builtin.id))
        {
            // IDs identify both settings references and secure-keychain entries.
            // Built-in IDs are canonical and must therefore retain their stable
            // lower-case spelling even when an older store used a display name.
            saved.id = builtin.id;
            if saved.name.trim().is_empty() {
                saved.name = builtin.name;
            }
            if saved.base_url.trim().is_empty() {
                saved.base_url = builtin.base_url;
            } else if saved.id == "anthropic"
                && saved.base_url.trim_end_matches('/') == "https://api.anthropic.com"
            {
                // This was the Tauri port's original built-in default. It is
                // not a user-defined endpoint and the reference catalog uses
                // the explicit `/v1` base URL.
                saved.base_url = builtin.base_url;
            }
        } else {
            providers.push(builtin);
        }
    }
}

fn validate_and_normalize_ai_provider_profiles(
    providers: &mut [provider::AiProviderProfile],
) -> Result<(), String> {
    if providers.is_empty() {
        return Err("Keep at least one AI provider configured".into());
    }

    let builtin_ids = provider::AiProviderProfile::built_in()
        .into_iter()
        .map(|profile| profile.id)
        .collect::<HashSet<_>>();
    let mut seen_ids = HashSet::new();
    let mut has_enabled_provider = false;

    for profile in &mut *providers {
        profile.id = profile.id.trim().to_owned();
        profile.name = profile.name.trim().to_owned();
        profile.base_url = profile.base_url.trim().trim_end_matches('/').to_owned();
        profile.model = profile.model.trim().to_owned();
        // Request parameters are derived from the selected model at call time;
        // never accept a stale or untrusted value through the command boundary.
        profile.request_parameters.clear();

        if profile.id.is_empty() || profile.name.is_empty() {
            return Err("Every AI provider needs an ID and a display name".into());
        }
        let normalized_id = profile.id.to_ascii_lowercase();
        if !seen_ids.insert(normalized_id) {
            return Err(format!("AI provider IDs must be unique: {}", profile.id));
        }

        let endpoint = reqwest::Url::parse(&profile.base_url)
            .map_err(|_| format!("{} needs a complete HTTP or HTTPS base URL", profile.name))?;
        if !matches!(endpoint.scheme(), "http" | "https") || endpoint.host_str().is_none() {
            return Err(format!(
                "{} needs a complete HTTP or HTTPS base URL",
                profile.name
            ));
        }
        has_enabled_provider |= profile.enabled;
    }

    if !has_enabled_provider {
        return Err("Keep at least one AI provider enabled".into());
    }

    let configured_builtin_ids = providers
        .iter()
        .map(|profile| profile.id.as_str())
        .collect::<HashSet<_>>();
    if let Some(missing) = builtin_ids
        .iter()
        .find(|id| !configured_builtin_ids.contains(id.as_str()))
    {
        return Err(format!(
            "Built-in AI provider '{missing}' cannot be removed"
        ));
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BackupArchive {
    version: u32,
    exported_at: DateTime<Utc>,
    database: AppDatabase,
}

fn normalize_database(database: &mut AppDatabase) {
    // A settings backup can be restored onto a platform/build where its
    // selected engine cannot run (for example a CUDA-only engine in a
    // portable build). Keep the unavailable engine visible in Settings, but
    // never leave it as the active selection: every persisted selection must
    // be immediately usable or fall back to the portable default.
    if !database.settings.selected_voice_engine.runtime_available() {
        database.settings.selected_voice_engine = VoiceEngine::Whisper;
    }
    // A previous Parakeet download path overwrote selected_model with
    // Parakeet's model ID. If that persisted while Whisper was selected,
    // resolving Whisper's model status failed before the user could choose a
    // valid model. Preserve custom local model paths, otherwise repair it.
    if matches!(
        database.settings.selected_voice_engine,
        VoiceEngine::Whisper
    ) && database
        .settings
        .local_model_path
        .as_deref()
        .is_none_or(|path| path.trim().is_empty())
        && whisper_model_filename(&database.settings.selected_model).is_err()
    {
        database.settings.selected_model = "base".into();
    }
    database.settings.user_typing_wpm = database.settings.user_typing_wpm.clamp(1, 200);
    database.settings.transcription_sound_volume =
        database.settings.transcription_sound_volume.clamp(0.0, 1.0);
    database.settings.audio_history_budget_gb =
        normalize_audio_history_budget_gb(database.settings.audio_history_budget_gb);
    database.settings.overlay_bottom_offset =
        normalize_overlay_bottom_offset(database.settings.overlay_bottom_offset);
    database.settings.transcription_preview_char_limit = normalize_transcription_preview_char_limit(
        database.settings.transcription_preview_char_limit,
    );
    database.settings.apple_speech_locale =
        normalize_apple_speech_locale(&database.settings.apple_speech_locale);
    database.dictionary = normalize_dictionary_entries(std::mem::take(&mut database.dictionary));
    database.custom_words = normalize_custom_words(std::mem::take(&mut database.custom_words));
    database.settings.punctuation_dictionary_prefix =
        formatting::normalize_punctuation_prefix(&database.settings.punctuation_dictionary_prefix)
            .unwrap_or_else(|| "literal".into());
    database.settings.punctuation_dictionary_rules =
        formatting::migrate_legacy_port_punctuation_rules(std::mem::take(
            &mut database.settings.punctuation_dictionary_rules,
        ));
    if !matches!(
        database.settings.transcription_start_sound.as_str(),
        "none" | "cue_1" | "cue_2" | "cue_3" | "cue_4" | "cue_5"
    ) {
        database.settings.transcription_start_sound = "cue_1".into();
    }
    database
        .file_transcription_history
        .sort_by(|left, right| right.created_at.cmp(&left.created_at));
    database.file_transcription_history.truncate(50);
    restore_missing_builtin_provider_profiles(&mut database.ai_providers);
    if database.ai_providers.iter().all(|profile| !profile.enabled) {
        if let Some(profile) = database.ai_providers.first_mut() {
            // Versions before profile enablement allowed disabling every row.
            // Preserve a usable configuration when loading one of those stores.
            profile.enabled = true;
        }
    }
    if database.settings.selected_ai_provider == "OpenAI" {
        database.settings.selected_ai_provider = "openai".into();
    }
    let provider_ids = database
        .ai_providers
        .iter()
        .filter(|profile| profile.enabled)
        .map(|profile| profile.id.as_str())
        .collect::<HashSet<_>>();
    if !provider_ids.contains(database.settings.selected_ai_provider.as_str()) {
        database.settings.selected_ai_provider = database
            .ai_providers
            .first()
            .map(|profile| profile.id.clone())
            .unwrap_or_default();
    }
    for provider_id in [
        &mut database.settings.selected_rewrite_ai_provider,
        &mut database.settings.selected_command_ai_provider,
    ] {
        if provider_id
            .as_deref()
            .is_some_and(|id| !provider_ids.contains(id))
        {
            *provider_id = None;
        }
    }
    for mode in editable_prompt_modes() {
        if !database
            .prompt_profiles
            .iter()
            .any(|profile| profile.mode == mode)
        {
            database.prompt_profiles.push(default_prompt_profile(mode));
        }
        let active_id = database
            .settings
            .active_prompt_profile_id(mode)
            .map(str::to_owned);
        let is_active_id_valid = active_id.as_deref().is_some_and(|id| {
            database
                .prompt_profiles
                .iter()
                .any(|profile| profile.id == id && profile.mode == mode)
        });
        if !is_active_id_valid {
            let fallback_id = database
                .prompt_profiles
                .iter()
                .find(|profile| profile.mode == mode)
                .map(|profile| profile.id.clone());
            database
                .settings
                .set_active_prompt_profile_id(mode, fallback_id);
        }
    }
    let mut seen_app_bindings = HashSet::new();
    database.app_prompt_bindings.retain(|binding| {
        let application = binding.application.trim();
        !application.is_empty()
            && editable_prompt_modes().contains(&binding.mode)
            && database.prompt_profiles.iter().any(|profile| {
                profile.mode == binding.mode && profile.id == binding.prompt_profile_id
            })
            && seen_app_bindings.insert((binding.mode.clone(), application.to_lowercase()))
    });
    for assignment in &mut database.settings.prompt_shortcut_assignments {
        assignment.prompt_profile_id = assignment.prompt_profile_id.trim().to_owned();
        assignment.hotkey = assignment.hotkey.trim().to_owned();
    }
    let mut seen_prompt_profiles = HashSet::new();
    let mut seen_prompt_hotkeys = HashSet::new();
    database
        .settings
        .prompt_shortcut_assignments
        .retain(|assignment| {
            !assignment.hotkey.is_empty()
                && database.prompt_profiles.iter().any(|profile| {
                    profile.mode == DictationMode::Dictate
                        && profile.id == assignment.prompt_profile_id
                })
                && seen_prompt_profiles.insert(assignment.prompt_profile_id.clone())
                && seen_prompt_hotkeys.insert(assignment.hotkey.to_lowercase())
        });
    let prompt_mode_profile_is_valid = database
        .settings
        .prompt_mode_selected_prompt_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .is_some_and(|id| {
            database
                .prompt_profiles
                .iter()
                .any(|profile| profile.mode == DictationMode::Dictate && profile.id == id)
        });
    if !prompt_mode_profile_is_valid {
        database.settings.prompt_mode_selected_prompt_id = None;
    }
    let mut configured_profiles = HashSet::new();
    database
        .dictation_prompt_configurations
        .retain_mut(|configuration| {
            configuration.prompt_profile_id = configuration.prompt_profile_id.trim().to_owned();
            configuration.provider_id = configuration
                .provider_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_owned);
            configuration.model = configuration
                .model
                .as_deref()
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(str::to_owned);
            let profile_exists = database.prompt_profiles.iter().any(|profile| {
                profile.mode == DictationMode::Dictate
                    && profile.id == configuration.prompt_profile_id
            });
            let provider_exists = configuration
                .provider_id
                .as_deref()
                .is_none_or(|provider_id| {
                    database
                        .ai_providers
                        .iter()
                        .any(|provider| provider.id == provider_id && provider.enabled)
                });
            let has_override = configuration.provider_id.is_some() || configuration.model.is_some();
            profile_exists
                && provider_exists
                && has_override
                && configured_profiles.insert(configuration.prompt_profile_id.clone())
        });
    normalize_command_chats(database);
}

fn normalize_overlay_bottom_offset(offset: f64) -> f64 {
    if offset.is_finite() {
        offset.clamp(10.0, 1_000.0)
    } else {
        50.0
    }
}

fn normalize_audio_history_budget_gb(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.max(0.1)
    } else {
        4.0
    }
}

fn normalize_transcription_preview_char_limit(value: usize) -> usize {
    let clamped = value.clamp(
        TRANSCRIPTION_PREVIEW_MIN_CHARACTERS,
        TRANSCRIPTION_PREVIEW_MAX_CHARACTERS,
    );
    let offset = clamped - TRANSCRIPTION_PREVIEW_MIN_CHARACTERS;
    let snapped_offset = ((offset as f64 / TRANSCRIPTION_PREVIEW_CHARACTER_STEP as f64).round()
        as usize)
        * TRANSCRIPTION_PREVIEW_CHARACTER_STEP;
    (TRANSCRIPTION_PREVIEW_MIN_CHARACTERS + snapped_offset).clamp(
        TRANSCRIPTION_PREVIEW_MIN_CHARACTERS,
        TRANSCRIPTION_PREVIEW_MAX_CHARACTERS,
    )
}

fn tail_characters(text: &str, max_characters: usize) -> String {
    if max_characters == 0 || text.is_empty() {
        return String::new();
    }
    let start = text
        .char_indices()
        .rev()
        .nth(max_characters.saturating_sub(1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    text[start..].to_owned()
}

fn default_apple_speech_locale() -> String {
    ["LC_ALL", "LC_MESSAGES", "LANG"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok())
        .map(|locale| normalize_apple_speech_locale(&locale))
        .unwrap_or_else(|| "en-US".into())
}

fn normalize_apple_speech_locale(locale: &str) -> String {
    let locale = locale.trim().split('.').next().unwrap_or_default().trim();
    if locale.is_empty() || matches!(locale, "C" | "POSIX") {
        "en-US".into()
    } else {
        locale.replace('_', "-")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
struct Settings {
    onboarding_completed: bool,
    onboarding_step: u8,
    onboarding_ai_skipped: bool,
    onboarding_playground_validated: bool,
    theme: Theme,
    accent_color: String,
    language: String,
    apple_speech_locale: String,
    primary_dictation_hotkey: String,
    secondary_dictation_hotkey: Option<String>,
    prompt_mode_hotkey: Option<String>,
    prompt_mode_selected_prompt_id: Option<String>,
    prompt_shortcut_assignments: Vec<PromptShortcutAssignment>,
    command_mode_hotkey: Option<String>,
    rewrite_mode_hotkey: Option<String>,
    cancel_recording_hotkey: Option<String>,
    paste_last_transcription_hotkey: Option<String>,
    hotkey_activation_mode: HotkeyActivationMode,
    enable_streaming_preview: bool,
    enable_ai_streaming: bool,
    show_thinking_tokens: bool,
    transcription_preview_char_limit: usize,
    transcription_start_sound: String,
    transcription_sound_volume: f32,
    copy_to_clipboard: bool,
    type_into_active_application: bool,
    #[serde(default, deserialize_with = "deserialize_text_insertion_mode")]
    text_insertion_mode: typing::TextInsertionMode,
    remove_filler_words_enabled: bool,
    filler_words: Vec<String>,
    auto_convert_punctuation_enabled: bool,
    punctuation_dictionary_prefix: String,
    punctuation_dictionary_rules: Vec<formatting::PunctuationRule>,
    literal_dictation_formatting_enabled: bool,
    gaav_lowercase_first_letter_enabled: bool,
    gaav_remove_trailing_period_enabled: bool,
    continuous_dictation_spacing_enabled: bool,
    context_aware_capitalization_enabled: bool,
    notify_ai_processing_failures: bool,
    save_transcription_history: bool,
    automatic_dictionary_learning_enabled: bool,
    vocabulary_boosting_enabled: bool,
    user_typing_wpm: u16,
    weekends_dont_break_streak: bool,
    audio_history_enabled: bool,
    audio_history_budget_gb: f64,
    selected_voice_engine: VoiceEngine,
    selected_model: String,
    #[serde(default)]
    whisper_beam_size: speech::BeamSize,
    #[serde(default)]
    nemotron_streaming_mode: NemotronStreamingMode,
    local_model_path: Option<String>,
    selected_input_device: Option<String>,
    cloud_transcription_model: String,
    selected_dictation_prompt_profile: Option<String>,
    selected_rewrite_prompt_profile: Option<String>,
    selected_command_prompt_profile: Option<String>,
    dictation_prompt_routing_scope: PromptRoutingScope,
    edit_prompt_routing_scope: PromptRoutingScope,
    ai_enhancement_enabled: bool,
    selected_ai_provider: String,
    selected_rewrite_ai_provider: Option<String>,
    selected_command_ai_provider: Option<String>,
    model_reasoning_configs: HashMap<String, ModelReasoningConfig>,
    command_mode_confirm_before_execute: bool,
    overlay_position: OverlayPosition,
    overlay_bottom_offset: f64,
    overlay_size: OverlaySize,
    launch_at_startup: bool,
    show_main_window_at_login_launch: bool,
    show_in_dock: bool,
    share_anonymous_analytics: bool,
    auto_update_check_enabled: bool,
    beta_releases_enabled: bool,
    last_update_check_at: Option<DateTime<Utc>>,
    update_prompt_snoozed_until: Option<DateTime<Utc>>,
    snoozed_update_version: Option<String>,
    local_api_enabled: bool,
    local_api_port: u16,
}

/// Optional provider/model request parameter for reasoning-capable models.
/// Keys are stored as `providerID:modelID`, matching Voxide's portable
/// backup contract. Disabled entries intentionally override a smart default.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct ModelReasoningConfig {
    parameter_name: String,
    parameter_value: String,
    is_enabled: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            onboarding_completed: false,
            onboarding_step: 0,
            onboarding_ai_skipped: false,
            onboarding_playground_validated: false,
            theme: Theme::System,
            accent_color: "#7c5cff".into(),
            language: "en".into(),
            apple_speech_locale: default_apple_speech_locale(),
            primary_dictation_hotkey: "Alt+Space".into(),
            secondary_dictation_hotkey: None,
            prompt_mode_hotkey: None,
            prompt_mode_selected_prompt_id: None,
            prompt_shortcut_assignments: Vec::new(),
            command_mode_hotkey: None,
            rewrite_mode_hotkey: None,
            cancel_recording_hotkey: Some("Escape".into()),
            paste_last_transcription_hotkey: None,
            hotkey_activation_mode: HotkeyActivationMode::Toggle,
            enable_streaming_preview: true,
            enable_ai_streaming: true,
            show_thinking_tokens: true,
            transcription_preview_char_limit: DEFAULT_TRANSCRIPTION_PREVIEW_CHARACTERS,
            transcription_start_sound: "cue_1".into(),
            transcription_sound_volume: 1.0,
            copy_to_clipboard: false,
            type_into_active_application: true,
            text_insertion_mode: typing::TextInsertionMode::Standard,
            remove_filler_words_enabled: true,
            filler_words: formatting::default_filler_words(),
            auto_convert_punctuation_enabled: true,
            punctuation_dictionary_prefix: "literal".into(),
            punctuation_dictionary_rules: formatting::default_punctuation_rules(),
            literal_dictation_formatting_enabled: false,
            gaav_lowercase_first_letter_enabled: false,
            gaav_remove_trailing_period_enabled: false,
            continuous_dictation_spacing_enabled: false,
            context_aware_capitalization_enabled: false,
            notify_ai_processing_failures: true,
            save_transcription_history: true,
            automatic_dictionary_learning_enabled: true,
            vocabulary_boosting_enabled: false,
            user_typing_wpm: 40,
            weekends_dont_break_streak: true,
            audio_history_enabled: false,
            audio_history_budget_gb: 4.0,
            selected_voice_engine: VoiceEngine::Whisper,
            selected_model: "base".into(),
            whisper_beam_size: speech::BeamSize::Auto,
            nemotron_streaming_mode: NemotronStreamingMode::Balanced,
            local_model_path: None,
            selected_input_device: None,
            cloud_transcription_model: "gpt-4o-mini-transcribe".into(),
            selected_dictation_prompt_profile: Some("default-dictate".into()),
            selected_rewrite_prompt_profile: Some("default-rewrite".into()),
            selected_command_prompt_profile: Some("default-command".into()),
            dictation_prompt_routing_scope: PromptRoutingScope::AllApps,
            edit_prompt_routing_scope: PromptRoutingScope::AllApps,
            ai_enhancement_enabled: false,
            selected_ai_provider: "openai".into(),
            selected_rewrite_ai_provider: None,
            selected_command_ai_provider: None,
            model_reasoning_configs: HashMap::new(),
            command_mode_confirm_before_execute: true,
            overlay_position: OverlayPosition::Bottom,
            overlay_bottom_offset: 50.0,
            overlay_size: OverlaySize::Medium,
            launch_at_startup: false,
            show_main_window_at_login_launch: true,
            show_in_dock: true,
            share_anonymous_analytics: false,
            auto_update_check_enabled: true,
            beta_releases_enabled: false,
            last_update_check_at: None,
            update_prompt_snoozed_until: None,
            snoozed_update_version: None,
            local_api_enabled: false,
            local_api_port: 47_733,
        }
    }
}

fn deserialize_text_insertion_mode<'de, D>(
    deserializer: D,
) -> Result<typing::TextInsertionMode, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(typing::TextInsertionMode::from_persisted(value.as_deref()))
}

impl Settings {
    fn reasoning_config_for(
        &self,
        provider: &provider::AiProviderProfile,
    ) -> Option<ModelReasoningConfig> {
        let key = format!("{}:{}", provider.id, provider.model);
        if let Some(config) = self.model_reasoning_configs.get(&key) {
            return config.is_enabled.then(|| config.clone());
        }
        let model = provider.model.to_ascii_lowercase();
        let (parameter_name, parameter_value) = if model.starts_with("gpt-5") {
            ("reasoning_effort", "low")
        } else if model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4") {
            ("reasoning_effort", "medium")
        } else if model.contains("gpt-oss") || model.starts_with("openai/") {
            ("reasoning_effort", "low")
        } else if model.contains("deepseek") && model.contains("reasoner") {
            ("enable_thinking", "true")
        } else {
            return None;
        };
        Some(ModelReasoningConfig {
            parameter_name: parameter_name.into(),
            parameter_value: parameter_value.into(),
            is_enabled: true,
        })
    }

    fn active_prompt_profile_id(&self, mode: DictationMode) -> Option<&str> {
        match mode {
            DictationMode::Dictate => self.selected_dictation_prompt_profile.as_deref(),
            DictationMode::Rewrite => self.selected_rewrite_prompt_profile.as_deref(),
            DictationMode::Command => self.selected_command_prompt_profile.as_deref(),
            DictationMode::Prompt | DictationMode::File => None,
        }
    }

    fn prompt_routing_scope(&self, mode: DictationMode) -> PromptRoutingScope {
        match mode {
            DictationMode::Dictate => self.dictation_prompt_routing_scope,
            DictationMode::Rewrite | DictationMode::Command => self.edit_prompt_routing_scope,
            DictationMode::Prompt | DictationMode::File => PromptRoutingScope::AllApps,
        }
    }

    fn set_active_prompt_profile_id(&mut self, mode: DictationMode, id: Option<String>) {
        match mode {
            DictationMode::Dictate => self.selected_dictation_prompt_profile = id,
            DictationMode::Rewrite => self.selected_rewrite_prompt_profile = id,
            DictationMode::Command => self.selected_command_prompt_profile = id,
            DictationMode::Prompt | DictationMode::File => {}
        }
    }
}

fn analytics_common_properties(settings: &Settings) -> Map<String, Value> {
    let mut properties = Map::new();
    properties.insert(
        "app_version".into(),
        Value::String(env!("CARGO_PKG_VERSION").into()),
    );
    properties.insert(
        "app_build".into(),
        Value::String(env!("CARGO_PKG_VERSION").into()),
    );
    properties.insert(
        "os_version".into(),
        Value::String(std::env::consts::OS.into()),
    );
    properties.insert("arch".into(), Value::String(std::env::consts::ARCH.into()));
    properties.insert(
        "environment".into(),
        Value::String(
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .into(),
        ),
    );
    properties.insert(
        "ai_processing_enabled".into(),
        Value::Bool(settings.ai_enhancement_enabled),
    );
    properties.insert(
        "streaming_preview_enabled".into(),
        Value::Bool(settings.enable_streaming_preview),
    );
    properties.insert(
        "ai_streaming_enabled".into(),
        Value::Bool(settings.enable_ai_streaming),
    );
    properties.insert(
        "press_and_hold_mode".into(),
        Value::Bool(matches!(
            &settings.hotkey_activation_mode,
            HotkeyActivationMode::Hold
        )),
    );
    properties.insert(
        "hotkey_activation_mode".into(),
        Value::String(
            match &settings.hotkey_activation_mode {
                HotkeyActivationMode::Toggle => "toggle",
                HotkeyActivationMode::Hold => "hold",
                HotkeyActivationMode::Automatic => "automatic",
            }
            .into(),
        ),
    );
    properties.insert(
        "copy_to_clipboard_enabled".into(),
        Value::Bool(settings.copy_to_clipboard),
    );
    properties
}

fn analytics_properties_with(
    settings: &Settings,
    additions: impl IntoIterator<Item = (&'static str, Value)>,
) -> Map<String, Value> {
    let mut properties = analytics_common_properties(settings);
    properties.extend(
        additions
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    );
    properties
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DictationEntry {
    id: String,
    text: String,
    raw_text: Option<String>,
    created_at: DateTime<Utc>,
    duration_ms: Option<u64>,
    mode: DictationMode,
    source_application: Option<String>,
    #[serde(default)]
    source_window_title: Option<String>,
    audio_file: Option<String>,
    #[serde(default)]
    audio_model: Option<String>,
    #[serde(default)]
    was_ai_processed: bool,
    #[serde(default)]
    processing_model: Option<String>,
    #[serde(default)]
    ai_processing_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileTranscriptionEntry {
    id: String,
    file_name: String,
    text: String,
    created_at: DateTime<Utc>,
    duration_ms: Option<u64>,
    #[serde(default)]
    processing_time_ms: Option<u64>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum FileTranscriptionExportFormat {
    Text,
    Json,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileTranscriptionExport<'a> {
    text: &'a str,
    confidence: f32,
    duration: f64,
    processing_time: f64,
    file_name: &'a str,
    timestamp: DateTime<Utc>,
}

/// The reference file-history store only retains completed transcriptions that
/// contain visible text.  In particular, a deliberately skipped sub-second
/// file must not create an empty history entry.
fn should_save_file_transcription(text: &str) -> bool {
    !text.trim().is_empty()
}

fn file_transcription_export(
    entry: &FileTranscriptionEntry,
    format: FileTranscriptionExportFormat,
) -> Result<Vec<u8>, String> {
    let duration = entry.duration_ms.unwrap_or_default() as f64 / 1_000.0;
    let processing_time = entry.processing_time_ms.unwrap_or_default() as f64 / 1_000.0;
    let confidence = entry.confidence.unwrap_or(if entry.text.trim().is_empty() {
        0.0
    } else {
        1.0
    });
    match format {
        FileTranscriptionExportFormat::Text => Ok(format!(
            "Transcription: {}\nDate: {}\nDuration: {duration:.1}s\nProcessing Time: {processing_time:.1}s\nConfidence: {:.1}%\n\n---\n\n{}",
            entry.file_name,
            entry.created_at.to_rfc3339(),
            confidence * 100.0,
            entry.text,
        )
        .into_bytes()),
        FileTranscriptionExportFormat::Json => serde_json::to_vec_pretty(&FileTranscriptionExport {
            text: &entry.text,
            confidence,
            duration,
            processing_time,
            file_name: &entry.file_name,
            timestamp: entry.created_at,
        })
        .map_err(|error| format!("Could not encode the transcription JSON: {error}")),
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FileTranscriptionProgress {
    completed_chunks: usize,
    total_chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DictionaryEntry {
    id: String,
    spoken: String,
    replacement: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DictionaryLearningRecord {
    heard_text: String,
    corrected_text: String,
    #[serde(default)]
    occurrences: Vec<DateTime<Utc>>,
    #[serde(default)]
    last_shown_at: Option<DateTime<Utc>>,
    #[serde(default)]
    dismissed_until: Option<DateTime<Utc>>,
    #[serde(default)]
    dismissal_count: u8,
    #[serde(default)]
    is_accepted: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DictionaryLearningSuggestion {
    heard_text: String,
    corrected_text: String,
    occurrences: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DictionaryTransferEntry {
    spoken: String,
    replacement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DictionaryTransferDocument {
    version: u32,
    replacements: Vec<DictionaryTransferEntry>,
    #[serde(default)]
    custom_words: Vec<CustomWordEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DictionaryImportDocument {
    Current(DictionaryTransferDocument),
    Legacy(Vec<DictionaryTransferEntry>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DictionaryImportResult {
    dictionary: Vec<DictionaryEntry>,
    custom_words: Vec<CustomWordEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CustomWordEntry {
    text: String,
    weight: Option<f32>,
    #[serde(default)]
    aliases: Vec<String>,
}

fn recognition_vocabulary(words: &[CustomWordEntry]) -> Vec<String> {
    let mut seen = HashSet::new();
    words
        .iter()
        .flat_map(|word| std::iter::once(&word.text).chain(word.aliases.iter()))
        .map(|word| word.trim())
        .filter(|word| !word.is_empty())
        .filter(|word| seen.insert(word.to_lowercase()))
        .take(200)
        .map(str::to_owned)
        .collect()
}

fn active_recognition_vocabulary(settings: &Settings, words: &[CustomWordEntry]) -> Vec<String> {
    settings
        .vocabulary_boosting_enabled
        .then(|| recognition_vocabulary(words))
        .unwrap_or_default()
}

fn normalize_custom_words(words: Vec<CustomWordEntry>) -> Vec<CustomWordEntry> {
    const MAX_CUSTOM_VOCABULARY_TERMS: usize = 256;

    let mut seen = HashSet::new();
    words
        .into_iter()
        .filter_map(|word| {
            let text = word.text.trim().to_owned();
            if text.is_empty() || !seen.insert(text.to_lowercase()) {
                return None;
            }
            let mut aliases = word
                .aliases
                .into_iter()
                .map(|alias| alias.trim().to_owned())
                .filter(|alias| !alias.is_empty())
                .filter(|alias| alias.to_lowercase() != text.to_lowercase())
                .map(|alias| alias.to_lowercase())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            aliases.sort();
            Some(CustomWordEntry {
                text,
                weight: word.weight,
                aliases,
            })
        })
        .take(MAX_CUSTOM_VOCABULARY_TERMS)
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptProfile {
    id: String,
    name: String,
    prompt: String,
    mode: DictationMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppPromptBinding {
    id: String,
    application: String,
    mode: DictationMode,
    prompt_profile_id: String,
}

/// Per-Dictate-profile provider/model override. Empty fields defer to the
/// global dictation provider, matching Voxide's prompt configuration
/// routing without duplicating secure credentials.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct DictationPromptConfiguration {
    prompt_profile_id: String,
    provider_id: Option<String>,
    model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptShortcutAssignment {
    prompt_profile_id: String,
    hotkey: String,
}

impl Default for PromptProfile {
    fn default() -> Self {
        default_prompt_profile(DictationMode::Dictate)
    }
}

fn default_prompt_profile(mode: DictationMode) -> PromptProfile {
    let (id, name, prompt) = match mode {
        DictationMode::Dictate => (
            "default-dictate",
            "Default dictation",
            "Clean up spoken text while preserving the speaker's meaning.",
        ),
        DictationMode::Rewrite => (
            "default-rewrite",
            "Default rewrite",
            "Rewrite the requested text faithfully and output only the final rewritten text.",
        ),
        DictationMode::Command => (
            "default-command",
            "Default command",
            "Plan a single safe, explicit desktop action. Never execute it yourself.",
        ),
        DictationMode::Prompt | DictationMode::File => (
            "default-general",
            "Default",
            "Respond helpfully and preserve the user's intent.",
        ),
    };
    PromptProfile {
        id: id.into(),
        name: name.into(),
        prompt: prompt.into(),
        mode,
    }
}

fn editable_prompt_modes() -> [DictationMode; 3] {
    [
        DictationMode::Dictate,
        DictationMode::Rewrite,
        DictationMode::Command,
    ]
}

fn prompt_for_mode(database: &AppDatabase, mode: DictationMode) -> PromptProfile {
    database
        .prompt_profiles
        .iter()
        .find(|profile| {
            profile.mode == mode
                && database.settings.active_prompt_profile_id(mode) == Some(profile.id.as_str())
        })
        .or_else(|| {
            database
                .prompt_profiles
                .iter()
                .find(|profile| profile.mode == mode)
        })
        .cloned()
        .unwrap_or_else(|| default_prompt_profile(mode))
}

fn prompt_for_mode_and_application(
    database: &AppDatabase,
    mode: DictationMode,
    application: Option<&str>,
) -> PromptProfile {
    let application = application
        .map(str::trim)
        .filter(|application| !application.is_empty());
    let bound_profile = application
        .and_then(|application| {
            database.app_prompt_bindings.iter().find(|binding| {
                binding.mode == mode && binding.application.eq_ignore_ascii_case(application)
            })
        })
        .and_then(|binding| {
            database
                .prompt_profiles
                .iter()
                .find(|profile| profile.mode == mode && profile.id == binding.prompt_profile_id)
        })
        .cloned();
    bound_profile.unwrap_or_else(|| {
        if database.settings.prompt_routing_scope(mode) == PromptRoutingScope::SelectedAppsOnly {
            default_prompt_profile(mode)
        } else {
            prompt_for_mode(database, mode)
        }
    })
}

fn prompt_for_mode_with_override(
    database: &AppDatabase,
    mode: DictationMode,
    prompt_profile_id: Option<&str>,
    application: Option<&str>,
) -> PromptProfile {
    prompt_profile_id
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .and_then(|id| {
            database
                .prompt_profiles
                .iter()
                .find(|profile| profile.mode == mode && profile.id == id)
        })
        .cloned()
        .unwrap_or_else(|| prompt_for_mode_and_application(database, mode, application))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum Theme {
    System,
    Light,
    Dark,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum PromptRoutingScope {
    AllApps,
    SelectedAppsOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum HotkeyActivationMode {
    Toggle,
    Hold,
    Automatic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum VoiceEngine {
    Whisper,
    Parakeet,
    Nemotron,
    AppleSpeech,
    Cloud,
}

impl VoiceEngine {
    const ALL: [Self; 5] = [
        Self::Whisper,
        Self::Parakeet,
        Self::Nemotron,
        Self::AppleSpeech,
        Self::Cloud,
    ];

    fn requires_provider_profile(self) -> bool {
        matches!(self, Self::Cloud)
    }

    fn is_nemotron(self) -> bool {
        matches!(self, Self::Nemotron)
    }

    fn is_parakeet(self) -> bool {
        matches!(self, Self::Parakeet)
    }

    fn capabilities(self) -> &'static asr::EngineCapabilities {
        match self {
            Self::Whisper => &asr::WHISPER,
            Self::Parakeet => &asr::PARAKEET,
            Self::Nemotron => &asr::NEMOTRON,
            Self::AppleSpeech => &asr::APPLE_SPEECH,
            Self::Cloud => &asr::CLOUD,
        }
    }

    fn audio_model_id(self, settings: &Settings) -> String {
        match self {
            Self::Whisper => settings.selected_model.clone(),
            Self::Cloud => settings.cloud_transcription_model.clone(),
            Self::AppleSpeech => "apple-speech".into(),
            Self::Parakeet => "parakeet".into(),
            Self::Nemotron => "nemotron".into(),
        }
    }

    /// Content-free implementation/runtime identifier for diagnostics and
    /// support exports. It must never contain a local path, transcript, or
    /// provider credential.
    fn diagnostic_runtime_version(self, state: &AppState) -> String {
        match self {
            Self::Whisper => "whisper-rs 0.16.0".into(),
            Self::Parakeet => "sherpa-onnx 1.13.4 CUDA 12/cuDNN 9".into(),
            Self::Nemotron => nemotron_runtime_path(state)
                .ok()
                .and_then(|runtime| nemotron_runtime_version(&runtime))
                .unwrap_or_else(|| {
                    format!(
                        "Nemotron runtime receipt v{NEMOTRON_RUNTIME_VERSION}, model {}",
                        &nemotron::MODEL_REVISION[..12]
                    )
                }),
            Self::AppleSpeech => "Apple Speech.framework".into(),
            Self::Cloud => "OpenAI-compatible HTTP API".into(),
        }
    }

    fn diagnostic_model_id(self, settings: &Settings) -> String {
        let model = self.audio_model_id(settings);
        diagnostic_version_value(&Value::String(model.clone()))
            .map(str::to_owned)
            .unwrap_or_else(|| "<redacted-model-id>".into())
    }

    fn runtime_available(self) -> bool {
        match self {
            Self::Parakeet => parakeet::is_compiled(),
            Self::Nemotron => nemotron::is_compiled(),
            Self::AppleSpeech => apple_speech::is_supported(),
            Self::Whisper | Self::Cloud => true,
        }
    }

    fn unavailable_reason(self) -> Option<&'static str> {
        if self.runtime_available() {
            return None;
        }
        match self {
            Self::Parakeet | Self::Nemotron => Some("Requires Voxide's CUDA build"),
            Self::AppleSpeech => Some("Available only on macOS"),
            Self::Whisper | Self::Cloud => None,
        }
    }

    fn descriptor(self) -> VoiceEngineDescriptor {
        let capabilities = asr::SpeechEngine::capabilities(&self);
        VoiceEngineDescriptor {
            id: capabilities.id,
            label: capabilities.label,
            description: capabilities.description,
            maturity: capabilities.maturity,
            preview_mode: capabilities.preview_mode,
            final_mode: capabilities.final_mode,
            supports_files: capabilities.supports_files,
            supports_translation: capabilities.supports_translation,
            supports_vocabulary: capabilities.supports_vocabulary,
            requires_cuda: capabilities.requires_cuda,
            available: self.runtime_available(),
            unavailable_reason: self.unavailable_reason(),
        }
    }
}

struct EngineFinalTranscript {
    text: String,
    whisper_timings: Option<speech::TranscriptionTimings>,
}

impl asr::SpeechEngine for VoiceEngine {
    fn engine_id(&self) -> &'static str {
        (*self).capabilities().id
    }

    fn capabilities(&self) -> &'static asr::EngineCapabilities {
        (*self).capabilities()
    }
}

/// Nemotron's cache-aware encoder supports fixed right-context choices.
/// Balanced matches NVIDIA's streaming example: it remains responsive while
/// giving the RNN-T decoder substantially more acoustic context than the
/// lowest-latency profile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
enum NemotronStreamingMode {
    Fast,
    #[default]
    Balanced,
    Quality,
}

impl NemotronStreamingMode {
    fn lookahead_tokens(self) -> u8 {
        match self {
            Self::Fast => 3,
            Self::Balanced => nemotron::DEFAULT_LOOKAHEAD_TOKENS,
            Self::Quality => 13,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum OverlaySize {
    #[serde(alias = "minimal")]
    Pill,
    Small,
    #[serde(alias = "standard")]
    Medium,
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum OverlayPosition {
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum DictationMode {
    Dictate,
    Prompt,
    Rewrite,
    Command,
    File,
}

struct AppState {
    database: Mutex<AppDatabase>,
    path: PathBuf,
    startup_recovery_notice: Mutex<Option<String>>,
}

impl AppState {
    fn load() -> Result<Self, String> {
        let root = ProjectDirs::from("dev", "pmdcoutinho", "Voxide")
            .map(|dirs| dirs.data_local_dir().to_path_buf())
            .ok_or("Could not determine the local application-data directory")?;
        fs::create_dir_all(&root)
            .map_err(|error| format!("Could not create application-data directory: {error}"))?;
        let path = root.join(DATABASE_FILE);
        let (mut database, needs_schema_migration, recovered_from_backup) =
            match fs::read_to_string(&path) {
                Ok(contents) => match decode_persisted_database(&contents) {
                    Ok((database, needs_schema_migration)) => {
                        (database, needs_schema_migration, None)
                    }
                    Err(error) if database_declares_future_schema(&contents) => {
                        return Err(format!("Could not read Voxide data: {error}"));
                    }
                    Err(primary_error) => {
                        let (database, needs_schema_migration, backup) =
                            newest_valid_database_backup(&path).map_err(|backup_error| {
                                format!(
                                    "Could not read Voxide data: {primary_error}. {backup_error}"
                                )
                            })?;
                        (database, needs_schema_migration, Some(backup))
                    }
                },
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    (AppDatabase::default(), false, None)
                }
                Err(error) => return Err(format!("Could not open Voxide data: {error}")),
            };
        normalize_database(&mut database);

        let state = Self {
            database: Mutex::new(database),
            path,
            startup_recovery_notice: Mutex::new(None),
        };
        if let Some(backup) = recovered_from_backup {
            let preserved = quarantine_corrupt_database(&state.path)?;
            eprintln!(
                "Recovered Voxide data from {}; preserved unreadable data at {}",
                backup.display(),
                preserved.display()
            );
            let database = state
                .database
                .lock()
                .map_err(|_| "Voxide data lock was poisoned".to_string())?;
            state.persist(&database)?;
            let mut notice = state
                .startup_recovery_notice
                .lock()
                .map_err(|_| "Voxide recovery notice lock was poisoned".to_string())?;
            *notice = Some(
                "Voxide restored your data from its newest valid backup. The unreadable database was preserved in Voxide storage for recovery.".into(),
            );
        } else if needs_schema_migration {
            state.back_up_pre_migration_database()?;
            let database = state
                .database
                .lock()
                .map_err(|_| "Voxide data lock was poisoned".to_string())?;
            state.persist(&database)?;
        }
        Ok(state)
    }

    fn take_startup_recovery_notice(&self) -> Result<Option<String>, String> {
        self.startup_recovery_notice
            .lock()
            .map(|mut notice| notice.take())
            .map_err(|_| "Voxide recovery notice lock was poisoned".to_string())
    }

    fn persist(&self, database: &AppDatabase) -> Result<(), String> {
        let contents = serde_json::to_vec_pretty(&PersistedDatabase {
            schema_version: DATABASE_SCHEMA_VERSION,
            data: database.clone(),
        })
        .map_err(|error| format!("Could not encode Voxide data: {error}"))?;
        let temporary_path = self.path.with_extension("json.tmp");
        let mut temporary = fs::File::create(&temporary_path)
            .map_err(|error| format!("Could not save Voxide data: {error}"))?;
        temporary
            .write_all(&contents)
            .and_then(|()| temporary.sync_all())
            .map_err(|error| format!("Could not save Voxide data: {error}"))?;
        drop(temporary);
        rotate_database_backups(&self.path)?;
        fs::rename(&temporary_path, &self.path)
            .map_err(|error| format!("Could not finalize Voxide data: {error}"))?;
        sync_parent_directory(&self.path)?;
        Ok(())
    }

    fn back_up_pre_migration_database(&self) -> Result<(), String> {
        if !self.path.exists() {
            return Ok(());
        }
        let backup = self.path.with_extension("json.pre-schema-v1-backup");
        if backup.exists() {
            return Ok(());
        }
        fs::copy(&self.path, &backup).map_err(|error| {
            format!("Could not back up Voxide data before schema migration: {error}")
        })?;
        Ok(())
    }

    fn models_directory(&self) -> Result<PathBuf, String> {
        let directory = self
            .path
            .parent()
            .ok_or("Could not determine the models directory")?
            .join("models");
        fs::create_dir_all(&directory)
            .map_err(|error| format!("Could not create models directory: {error}"))?;
        Ok(directory)
    }

    fn data_directory(&self) -> Result<PathBuf, String> {
        self.path
            .parent()
            .map(Path::to_path_buf)
            .ok_or("Could not determine the local application-data directory".into())
    }

    fn audio_history_directory(&self) -> Result<PathBuf, String> {
        let directory = self
            .path
            .parent()
            .ok_or("Could not determine the audio history directory")?
            .join("audio-history");
        fs::create_dir_all(&directory)
            .map_err(|error| format!("Could not create audio history directory: {error}"))?;
        Ok(directory)
    }

    fn analytics_identity_path(&self) -> PathBuf {
        self.path.with_file_name("analytics-identity.json")
    }

    fn update<T>(&self, change: impl FnOnce(&mut AppDatabase) -> T) -> Result<T, String> {
        let mut database = self
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        let value = change(&mut database);
        self.persist(&database)?;
        Ok(value)
    }
}

/// Retain a short, ordered history of complete database snapshots before an
/// atomic replacement. All files stay beside the live database so copies and
/// renames remain on the same filesystem.
fn rotate_database_backups(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Ok(());
    }
    for index in (1..DATABASE_BACKUP_COUNT).rev() {
        let source = database_backup_path(path, index - 1);
        if source.is_file() {
            let destination = database_backup_path(path, index);
            fs::copy(&source, &destination).map_err(|error| {
                format!(
                    "Could not rotate Voxide database backup {} to {}: {error}",
                    source.display(),
                    destination.display()
                )
            })?;
        }
    }
    let newest = database_backup_path(path, 0);
    fs::copy(path, &newest).map_err(|error| {
        format!(
            "Could not create a Voxide database backup at {}: {error}",
            newest.display()
        )
    })?;
    fs::File::open(&newest)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("Could not flush Voxide database backup: {error}"))?;
    Ok(())
}

fn database_backup_path(path: &Path, index: usize) -> PathBuf {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(DATABASE_FILE);
    path.with_file_name(format!("{filename}.backup-{index}"))
}

/// Renames are only crash-durable once the containing directory metadata has
/// reached storage. Windows does not expose an equivalent portable directory
/// handle, so the file flush remains the cross-platform baseline there.
#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("Could not determine the Voxide data directory")?;
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("Could not flush the Voxide data directory: {error}"))
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn newest_valid_database_backup(path: &Path) -> Result<(AppDatabase, bool, PathBuf), String> {
    for index in 0..DATABASE_BACKUP_COUNT {
        let backup = database_backup_path(path, index);
        let Ok(contents) = fs::read_to_string(&backup) else {
            continue;
        };
        if let Ok((database, needs_schema_migration)) = decode_persisted_database(&contents) {
            return Ok((database, needs_schema_migration, backup));
        }
    }
    Err(format!(
        "No valid Voxide database backup was found beside {}",
        path.display()
    ))
}

fn database_declares_future_schema(contents: &str) -> bool {
    serde_json::from_str::<Value>(contents)
        .ok()
        .and_then(|value| value.get("schemaVersion").and_then(Value::as_u64))
        .is_some_and(|version| version > DATABASE_SCHEMA_VERSION as u64)
}

/// Keep unreadable data intact for manual inspection before restoring a known
/// good snapshot. The name is app-owned and collision-resistant, so recovery
/// never overwrites a user-selected file.
fn quarantine_corrupt_database(path: &Path) -> Result<PathBuf, String> {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(DATABASE_FILE);
    let preserved = path.with_file_name(format!("{filename}.corrupt-{}", uuid::Uuid::new_v4()));
    fs::rename(path, &preserved).map_err(|error| {
        format!(
            "Could not preserve unreadable Voxide data at {}: {error}",
            preserved.display()
        )
    })?;
    Ok(preserved)
}

/// A prewarmed capture target cached between recordings, tagged with the input
/// device it was resolved for so a stale entry (e.g. after a mic-preference
/// change) is re-resolved instead of reused.
struct PreparedCapture {
    device_key: Option<String>,
    prepared: audio::PreparedInput,
}

struct NativeCaptureState {
    capture: Mutex<Option<audio::AudioCapture>>,
    capture_started_at: Mutex<Option<Instant>>,
    // Device + config + routing resolved off the record hotkey path so a press
    // only opens hardware. Populated at startup and refreshed on a cold miss.
    prepared_input: Mutex<Option<PreparedCapture>>,
    session: Arc<Mutex<session::Coordinator>>,
    context: Mutex<DictationContext>,
    continuous_context: Mutex<ContinuousDictationContext>,
    preview_generation: Arc<AtomicU64>,
    parakeet_live: Mutex<ParakeetLiveState>,
    nemotron_live: Arc<tokio::sync::Mutex<NemotronLiveState>>,
}

impl Default for NativeCaptureState {
    fn default() -> Self {
        Self {
            capture: Mutex::new(None),
            capture_started_at: Mutex::new(None),
            prepared_input: Mutex::new(None),
            session: Arc::new(Mutex::new(session::Coordinator::default())),
            context: Mutex::new(DictationContext::default()),
            continuous_context: Mutex::new(ContinuousDictationContext::default()),
            preview_generation: Arc::new(AtomicU64::new(0)),
            parakeet_live: Mutex::new(ParakeetLiveState::default()),
            nemotron_live: Arc::new(tokio::sync::Mutex::new(NemotronLiveState::default())),
        }
    }
}

fn begin_preview(capture_state: &NativeCaptureState, session_id: u64) -> bool {
    let admission = capture_state
        .session
        .lock()
        .map(|mut coordinator| coordinator.begin_preview(session_id));
    match admission {
        Ok(session::PreviewAdmission::Admitted) => true,
        Ok(session::PreviewAdmission::Busy) => {
            debug_log::append(&format!(
                "Preview skipped (session: {session_id}, reason: busy)"
            ));
            false
        }
        Ok(session::PreviewAdmission::Inactive) => {
            debug_log::append(&format!(
                "Preview skipped (session: {session_id}, reason: finalizing_or_stale)"
            ));
            false
        }
        Err(_) => {
            debug_log::append(&format!(
                "Preview skipped (session: {session_id}, reason: coordinator_lock)"
            ));
            false
        }
    }
}

fn finish_preview(capture_state: &NativeCaptureState, session_id: u64) {
    if let Ok(mut coordinator) = capture_state.session.lock() {
        coordinator.finish_preview(session_id);
    }
}

fn validate_voice_engine_switch(
    current: VoiceEngine,
    requested: VoiceEngine,
    coordinator: &session::Coordinator,
) -> Result<(), String> {
    if current != requested && !coordinator.is_idle() {
        return Err(
            "Stop or cancel the current dictation before changing the voice engine.".into(),
        );
    }
    Ok(())
}

/// Invalidates only the specified generation. A stale startup failure must not
/// suppress previews from a newer recording that has already replaced it.
fn invalidate_preview_generation(capture_state: &NativeCaptureState, session_id: u64) -> bool {
    let next_generation = session_id.wrapping_add(1).max(1);
    capture_state
        .preview_generation
        .compare_exchange(
            session_id,
            next_generation,
            Ordering::SeqCst,
            Ordering::SeqCst,
        )
        .is_ok()
}

fn reset_dictation_context(capture_state: &NativeCaptureState) {
    let _ = capture_state
        .context
        .lock()
        .map(|mut context| *context = DictationContext::default());
}

/// Makes every failure between coordinator admission and a usable capture
/// indistinguishable from a cancelled recording. This path is deliberately
/// idempotent: an engine reservation, microphone start, or preview setup may
/// fail at different points, but none may leave stale session or application
/// context state behind.
fn rollback_native_dictation_start(capture_state: &NativeCaptureState, session_id: u64) {
    if !invalidate_preview_generation(capture_state, session_id) {
        return;
    }
    let _ = capture_state
        .capture
        .lock()
        .map(|mut capture| capture.take());
    let _ = capture_state
        .capture_started_at
        .lock()
        .map(|mut started_at| started_at.take());
    let _ = capture_state
        .session
        .lock()
        .map(|mut session| session.cancel(session_id));
    reset_dictation_context(capture_state);
}

/// CPAL reports device and route failures on a callback that must stay
/// lock-free and UI-free. Poll its atomic health counter from the async side
/// instead, then end the affected generation exactly once with a visible,
/// actionable state.
fn spawn_capture_error_monitor(app: AppHandle, session_id: u64) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let capture_state = app.state::<NativeCaptureState>();
            if capture_state.preview_generation.load(Ordering::SeqCst) != session_id {
                return;
            }
            let stream_errors = capture_state
                .capture
                .lock()
                .ok()
                .and_then(|capture| {
                    capture
                        .as_ref()
                        .map(|capture| capture.health().stream_errors)
                })
                .unwrap_or_default();
            if stream_errors == 0 {
                continue;
            }
            if !invalidate_preview_generation(&capture_state, session_id) {
                return;
            }

            let capture = capture_state
                .capture
                .lock()
                .ok()
                .and_then(|mut capture| capture.take());
            if capture.is_none() {
                return;
            }
            let _ = capture_state
                .capture_started_at
                .lock()
                .map(|mut started_at| started_at.take());
            let _ = capture_state
                .session
                .lock()
                .map(|mut coordinator| coordinator.cancel(session_id));
            reset_dictation_context(&capture_state);
            drop(capture);
            let nemotron_live = Arc::clone(&capture_state.nemotron_live);
            tauri::async_runtime::spawn(async move {
                let mut live = nemotron_live.lock().await;
                if live.generation == session_id && live.session_started {
                    if let Some(server) = live.server.as_mut() {
                        if server.abort().await.is_err() {
                            server.terminate();
                        }
                    }
                }
                if live.generation == session_id {
                    live.fed_samples = 0;
                    live.session_started = false;
                    live.generation = 0;
                    live.start_error = None;
                }
            });
            debug_log::append(&format!(
                "Capture failed (session: {session_id}, stream_errors: {stream_errors})"
            ));
            emit_overlay(
                &app,
                "error",
                "Microphone connection lost. Reconnect or reselect the input device.",
            );
            let _ = app.emit("dictation-capture-failed", CaptureFailure { session_id });
            update_tray_status(&app, TrayVisualState::Ready);
            return;
        }
    });
}

/// Ensures finalization cannot leave the coordinator stuck when any later
/// transcription, post-processing, or insertion step returns an error.
struct FinishDictationSession {
    coordinator: Arc<Mutex<session::Coordinator>>,
    id: u64,
}

impl Drop for FinishDictationSession {
    fn drop(&mut self) {
        if let Ok(mut coordinator) = self.coordinator.lock() {
            coordinator.finish(self.id);
        }
    }
}

/// The focused application/window and preceding text are session-scoped input
/// to formatting. Do not retain that transient context after a final result or
/// any error path has completed.
struct ResetDictationContextWhenDropped<'a> {
    context: &'a Mutex<DictationContext>,
}

impl Drop for ResetDictationContextWhenDropped<'_> {
    fn drop(&mut self) {
        if let Ok(mut context) = self.context.lock() {
            *context = DictationContext::default();
        }
    }
}

/// Display-only state for Parakeet's full-buffer live preview.
///
/// This mirrors FluidVoice's TDT v3 implementation: each preview is a fresh
/// decode of the complete capture, then reconciled with the previous snapshot.
/// The final transcription is always another independent full-audio decode.
#[derive(Debug, Clone, Default)]
struct ParakeetLiveState {
    generation: u64,
    previous_full_text: String,
}

/// State held across Nemotron's native cache-aware stream. The child process
/// survives successful recordings so the CUDA model stays warm; each stream
/// itself is reset by the Python service after `finish`.
#[derive(Default)]
struct NemotronLiveState {
    generation: u64,
    fed_samples: usize,
    session_started: bool,
    server: Option<nemotron::Server>,
    start_error: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct DictationContext {
    source_application: Option<String>,
    source_window_title: Option<String>,
    prompt_profile_id: Option<String>,
    preceding_text: String,
}

#[derive(Debug, Default)]
struct ContinuousDictationContext {
    source_application: Option<String>,
    source_window_title: Option<String>,
    inserted_text: String,
}

impl ContinuousDictationContext {
    fn preceding_text_for(&self, application: Option<&str>, window_title: Option<&str>) -> String {
        (self.source_application.as_deref() == application
            && self.source_window_title.as_deref() == window_title)
            .then(|| self.inserted_text.clone())
            .unwrap_or_default()
    }

    fn record(&mut self, application: Option<String>, window_title: Option<String>, text: String) {
        self.source_application = application;
        self.source_window_title = window_title;
        self.inserted_text = text;
    }
}

#[derive(Default)]
struct TrayState {
    status: Mutex<Option<MenuItem<tauri::Wry>>>,
    dictate: Mutex<Option<MenuItem<tauri::Wry>>>,
    paste_last: Mutex<Option<MenuItem<tauri::Wry>>>,
}

#[derive(Clone, Copy)]
enum TrayVisualState {
    Ready,
    Recording,
    Processing,
}

struct ResetTrayWhenDropped {
    app: AppHandle,
}

impl Drop for ResetTrayWhenDropped {
    fn drop(&mut self) {
        update_tray_status(&self.app, TrayVisualState::Ready);
    }
}

/// Re-resolves and re-caches the prewarmed capture target once a dictation
/// finishes and frees the microphone. This keeps the cached device/config/
/// routing fresh — so a source whose sound-server node name changed under an
/// unchanged description is picked up before the next recording — while still
/// forking `pactl` off the record hotkey path (at the same once-per-recording
/// cadence the pre-prewarm code used, just after the fact instead of before).
struct RefreshCapturePrewarmWhenDropped {
    app: AppHandle,
}

impl Drop for RefreshCapturePrewarmWhenDropped {
    fn drop(&mut self) {
        spawn_capture_prewarm(self.app.clone());
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
enum HotkeyAction {
    Dictate,
    Prompt,
    PromptProfile(String),
    Command,
    Rewrite,
    Cancel,
    PasteLast,
}

#[derive(Default)]
struct HotkeyRegistry {
    bindings: Mutex<HashMap<String, HotkeyAction>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HotkeyEvent {
    action: HotkeyAction,
    phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_profile_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HotkeyBackendStatus {
    backend: String,
    state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TrayAction {
    action: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CommandStreamUpdate {
    conversation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommandPlan {
    kind: String,
    #[serde(default)]
    conversation_id: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    purpose: Option<String>,
    #[serde(default)]
    working_directory: Option<String>,
    /// Present only for provider-native function calls. It is returned to the
    /// frontend so the reviewed result can be attached to that exact call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// Internal history metadata; the user-facing plan only exposes its
    /// reviewed command fields above.
    #[serde(skip, default)]
    tool_call: Option<provider::CommandToolCall>,
    #[serde(skip_deserializing, default)]
    destructive: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CommandExecutionResult {
    success: bool,
    command: String,
    output: String,
    error: Option<String>,
    exit_code: i32,
    execution_time_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommandChat {
    id: String,
    title: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    messages: Vec<CommandChatMessage>,
    #[serde(default)]
    source_application: Option<String>,
}

impl CommandChat {
    fn new() -> Self {
        let now = Utc::now();
        Self {
            id: make_id("command-chat"),
            title: "New chat".into(),
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            source_application: None,
        }
    }

    fn append(&mut self, role: CommandChatRole, content: impl Into<String>) {
        self.append_with_tool_metadata(role, content, None, None, None);
    }

    fn append_with_tool_metadata(
        &mut self,
        role: CommandChatRole,
        content: impl Into<String>,
        tool_call: Option<provider::CommandToolCall>,
        tool_call_id: Option<String>,
        thinking: Option<String>,
    ) {
        let content = content.into();
        let now = Utc::now();
        if self.title == "New chat" && role == CommandChatRole::User {
            self.title = compact_command_chat_title(&content);
        }
        self.messages.push(CommandChatMessage {
            id: make_id("command-message"),
            role,
            content,
            tool_call,
            tool_call_id,
            thinking,
            created_at: now,
        });
        if self.messages.len() > 100 {
            self.messages.drain(..self.messages.len() - 100);
        }
        self.updated_at = now;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum CommandChatRole {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommandChatMessage {
    id: String,
    role: CommandChatRole,
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call: Option<provider::CommandToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// Provider-supplied reasoning is retained for local, opt-in display only.
    /// It is never included in the subsequent provider conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    created_at: DateTime<Utc>,
}

fn compact_command_chat_title(content: &str) -> String {
    let title = content.trim().chars().take(50).collect::<String>();
    if title.is_empty() {
        "New chat".into()
    } else if content.trim().chars().count() > title.chars().count() {
        format!("{}…", title.chars().take(47).collect::<String>())
    } else {
        title
    }
}

fn make_id(prefix: &str) -> String {
    format!("{prefix}-{}", Utc::now().format("%Y%m%d%H%M%S%3f"))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BootstrapData {
    database: AppDatabase,
    recovery_notice: Option<String>,
}

#[tauri::command]
fn bootstrap(state: State<'_, AppState>) -> Result<BootstrapData, String> {
    let database = state
        .database
        .lock()
        .map(|database| database.clone())
        .map_err(|_| "Voxide data lock was poisoned".to_string())?;
    Ok(BootstrapData {
        database,
        recovery_notice: state.take_startup_recovery_notice()?,
    })
}

fn apply_launch_at_startup(app: &AppHandle, enabled: bool) -> Result<(), String> {
    let autolaunch = app.autolaunch();
    if enabled {
        autolaunch
            .enable()
            .map_err(|error| format!("Could not enable launch at startup: {error}"))
    } else {
        autolaunch
            .disable()
            .map_err(|error| format!("Could not disable launch at startup: {error}"))
    }
}

fn apply_taskbar_visibility(app: &AppHandle, show_in_dock: bool) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        window
            .set_skip_taskbar(!show_in_dock)
            .map_err(|error| format!("Could not update taskbar visibility: {error}"))?;
    }
    Ok(())
}

fn launched_from_autostart(arguments: impl IntoIterator<Item = String>) -> bool {
    arguments
        .into_iter()
        .any(|argument| argument == "--voxide-autostart")
}

#[tauri::command]
fn save_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    capture_state: State<'_, NativeCaptureState>,
    settings: Settings,
) -> Result<Settings, String> {
    let (previous_analytics_enabled, previous_audio_history_budget_gb, previous_voice_engine) =
        state
            .database
            .lock()
            .map(|database| {
                (
                    database.settings.share_anonymous_analytics,
                    database.settings.audio_history_budget_gb,
                    database.settings.selected_voice_engine,
                )
            })
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
    if previous_voice_engine != settings.selected_voice_engine {
        let coordinator = capture_state
            .session
            .lock()
            .map_err(|_| "Dictation session lock was poisoned".to_string())?;
        validate_voice_engine_switch(
            previous_voice_engine,
            settings.selected_voice_engine,
            &coordinator,
        )?;
    }
    apply_launch_at_startup(&app, settings.launch_at_startup)?;
    apply_taskbar_visibility(&app, settings.show_in_dock)?;
    let save = || {
        state.update(|database| {
            database.settings = settings.clone();
            normalize_database(database);
            database.settings.clone()
        })
    };
    let saved = if previous_voice_engine != settings.selected_voice_engine {
        let coordinator = capture_state
            .session
            .lock()
            .map_err(|_| "Dictation session lock was poisoned".to_string())?;
        validate_voice_engine_switch(
            previous_voice_engine,
            settings.selected_voice_engine,
            &coordinator,
        )?;
        save()?
    } else {
        save()?
    };
    if let Err(error) = apply_overlay_window_layout(&app, &saved) {
        debug_log::append(&format!("Could not apply updated overlay layout: {error}"));
    }
    if saved.audio_history_budget_gb != previous_audio_history_budget_gb {
        enforce_audio_history_budget(&state, saved.audio_history_budget_gb)?;
    }
    let analytics = app.state::<analytics::AnalyticsService>();
    analytics.set_enabled(saved.share_anonymous_analytics);
    if previous_analytics_enabled != saved.share_anonymous_analytics {
        analytics.capture(
            "analytics_consent_changed",
            saved.share_anonymous_analytics,
            analytics_properties_with(
                &saved,
                [("enabled", Value::Bool(saved.share_anonymous_analytics))],
            ),
        );
    }
    Ok(saved)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UpdateAvailableEvent {
    latest_version: String,
    release_url: String,
}

fn should_check_for_updates(settings: &Settings) -> bool {
    settings.auto_update_check_enabled
        && settings
            .last_update_check_at
            .is_none_or(|last_check| Utc::now() >= last_check + chrono::Duration::hours(1))
}

fn should_show_update_prompt(settings: &Settings, version: &str) -> bool {
    if settings
        .snoozed_update_version
        .as_deref()
        .is_some_and(|snoozed| snoozed != version)
    {
        return true;
    }
    settings
        .update_prompt_snoozed_until
        .is_none_or(|until| Utc::now() >= until)
}

async fn check_for_update_with_settings(
    state: &AppState,
    include_prerelease: bool,
) -> Result<update::UpdateCheckResult, String> {
    let result = update::check_for_update(env!("CARGO_PKG_VERSION"), include_prerelease).await;
    // A failed request counts as a check as well. This matches the reference behavior and avoids
    // repeatedly hammering GitHub while a network or service outage is in progress.
    let _ = state.update(|database| database.settings.last_update_check_at = Some(Utc::now()));
    result
}

#[tauri::command]
async fn check_for_updates(
    state: State<'_, AppState>,
) -> Result<update::UpdateCheckResult, String> {
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    check_for_update_with_settings(&state, settings.beta_releases_enabled).await
}

#[tauri::command]
async fn recent_release_notes(
    state: State<'_, AppState>,
) -> Result<Vec<update::ReleaseNote>, String> {
    let include_prerelease = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .beta_releases_enabled;
    update::recent_release_notes(include_prerelease).await
}

#[tauri::command]
fn snooze_update_prompt(state: State<'_, AppState>, version: String) -> Result<(), String> {
    let version = version.trim();
    if version.is_empty() || version.len() > 128 {
        return Err("The update version is not valid".into());
    }
    state.update(|database| {
        database.settings.update_prompt_snoozed_until =
            Some(Utc::now() + chrono::Duration::hours(24));
        database.settings.snoozed_update_version = Some(version.to_owned());
    })
}

#[tauri::command]
fn open_update_release(release_url: String) -> Result<(), String> {
    if !update::is_release_url(&release_url) {
        return Err("The update release URL is not trusted".into());
    }
    tauri_plugin_opener::open_url(&release_url, None::<&str>)
        .map_err(|error| format!("Could not open the update release page: {error}"))
}

#[tauri::command]
fn open_provider_website(provider_id: String) -> Result<(), String> {
    let provider_id = provider_id.trim();
    let (url, _) = provider::provider_website(provider_id)
        .ok_or("This provider does not have a built-in setup page")?;
    tauri_plugin_opener::open_url(url, None::<&str>)
        .map_err(|error| format!("Could not open the provider setup page: {error}"))
}

async fn check_for_updates_automatically(app: AppHandle) {
    let state = app.state::<AppState>();
    let settings = match state.database.lock() {
        Ok(database) => database.settings.clone(),
        Err(_) => return,
    };
    if !should_check_for_updates(&settings) {
        return;
    }
    let result = match check_for_update_with_settings(&state, settings.beta_releases_enabled).await
    {
        Ok(result) => result,
        Err(error) => {
            eprintln!("Voxide automatic update check failed: {error}");
            return;
        }
    };
    let (Some(version), Some(release_url)) = (result.latest_version, result.release_url) else {
        return;
    };
    let should_prompt = state
        .database
        .lock()
        .map(|database| should_show_update_prompt(&database.settings, &version))
        .unwrap_or(false);
    if !should_prompt {
        return;
    }
    notify_update_available(&app, &version, &release_url);
    let _ = app.emit(
        "voxide-update-available",
        UpdateAvailableEvent {
            latest_version: version,
            release_url,
        },
    );
}

#[tauri::command]
fn set_onboarding_step(state: State<'_, AppState>, step: u8) -> Result<Settings, String> {
    if step > 5 {
        return Err("The onboarding step is not valid".into());
    }
    state.update(|database| {
        database.settings.onboarding_step = step;
        database.settings.clone()
    })
}

#[tauri::command]
fn complete_onboarding(state: State<'_, AppState>) -> Result<Settings, String> {
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    match settings.selected_voice_engine {
        VoiceEngine::Whisper => {
            let model_path = whisper_model_path(&settings, &state)?;
            if !valid_whisper_model_file(&model_path) {
                return Err("Download the selected Whisper model before completing setup".into());
            }
        }
        VoiceEngine::Parakeet => {
            if !parakeet::is_compiled() {
                return Err("Parakeet is available in Voxide's CUDA build".into());
            }
            let model_path = parakeet_model_path(&state)?;
            if !parakeet_model_is_verified(&model_path) {
                return Err("Download the Parakeet model before completing setup".into());
            }
        }
        VoiceEngine::Nemotron => {
            if !nemotron::is_compiled() {
                return Err("Nemotron is available in Voxide's CUDA build for Linux/NVIDIA".into());
            }
            let runtime = nemotron_runtime_path(&state)?;
            if !nemotron_runtime_is_verified(&runtime) {
                return Err("Install the Nemotron CUDA runtime before completing setup".into());
            }
            let model = nemotron_model_path(&state)?;
            if !nemotron_model_is_verified(&model) {
                return Err("Download the Nemotron model before completing setup".into());
            }
        }
        VoiceEngine::Cloud => {
            if settings.cloud_transcription_model.trim().is_empty() {
                return Err(
                    "Choose a compatible cloud transcription model before completing setup".into(),
                );
            }
            let database = state
                .database
                .lock()
                .map_err(|_| "Voxide data lock was poisoned".to_string())?;
            let profile = selected_provider(&database, None)?;
            if !profile.enabled
                || !matches!(
                    profile.api_style,
                    provider::ProviderApiStyle::OpenAiCompatible
                )
            {
                return Err(
                    "Choose an enabled OpenAI-compatible provider for cloud transcription".into(),
                );
            }
        }
        VoiceEngine::AppleSpeech => {
            if !apple_speech::is_supported() {
                return Err(
                    "Apple Speech is available only on macOS. Select Whisper or a compatible cloud provider on this platform."
                        .into(),
                );
            }
        }
    }
    state.update(|database| {
        database.settings.onboarding_completed = true;
        database.settings.onboarding_step = 5;
        database.settings.clone()
    })
}

#[tauri::command]
fn reset_onboarding(state: State<'_, AppState>) -> Result<Settings, String> {
    state.update(|database| {
        database.settings.onboarding_completed = false;
        database.settings.onboarding_step = 0;
        database.settings.onboarding_ai_skipped = false;
        database.settings.onboarding_playground_validated = false;
        database.settings.clone()
    })
}

#[tauri::command]
fn export_backup(state: State<'_, AppState>, destination: String) -> Result<(), String> {
    let database = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .clone();
    let destination = PathBuf::from(destination.trim());
    if destination.as_os_str().is_empty() {
        return Err("Choose a destination for the backup".into());
    }
    let archive = BackupArchive {
        version: BACKUP_VERSION,
        exported_at: Utc::now(),
        database,
    };
    let contents = serde_json::to_vec_pretty(&archive)
        .map_err(|error| format!("Could not encode the backup: {error}"))?;
    fs::write(destination, contents).map_err(|error| format!("Could not save the backup: {error}"))
}

#[tauri::command]
async fn import_backup(
    app: AppHandle,
    state: State<'_, AppState>,
    registry: State<'_, HotkeyRegistry>,
    control: State<'_, local_api::LocalApiControl>,
    source: String,
) -> Result<AppDatabase, String> {
    let source = PathBuf::from(source.trim());
    let metadata = fs::metadata(&source)
        .map_err(|error| format!("Could not inspect the selected backup: {error}"))?;
    if metadata.len() > 50 * 1024 * 1024 {
        return Err("The selected backup is larger than 50 MB".into());
    }
    let contents =
        fs::read(&source).map_err(|error| format!("Could not read the backup: {error}"))?;
    let mut archive: BackupArchive = serde_json::from_slice(&contents)
        .map_err(|error| format!("The selected file is not a valid Voxide backup: {error}"))?;
    if archive.version > BACKUP_VERSION {
        return Err("This backup was created by a newer version of Voxide".into());
    }
    normalize_database(&mut archive.database);
    apply_hotkeys(&app, &registry, &archive.database.settings)?;
    apply_launch_at_startup(&app, archive.database.settings.launch_at_startup)?;
    apply_taskbar_visibility(&app, archive.database.settings.show_in_dock)?;
    let restored = state.update(|database| {
        *database = archive.database.clone();
        database.clone()
    })?;
    app.state::<analytics::AnalyticsService>()
        .set_enabled(restored.settings.share_anonymous_analytics);
    local_api::stop(&control)?;
    if restored.settings.local_api_enabled {
        local_api::start(&control, app, restored.settings.local_api_port).await?;
    }
    Ok(restored)
}

#[tauri::command]
fn save_dictation(
    state: State<'_, AppState>,
    text: String,
    raw_text: Option<String>,
    duration_ms: Option<u64>,
    mode: DictationMode,
    source_application: Option<String>,
    source_window_title: Option<String>,
    audio_file: Option<String>,
    audio_model: Option<String>,
    was_ai_processed: Option<bool>,
    processing_model: Option<String>,
    ai_processing_error: Option<String>,
) -> Result<DictationEntry, String> {
    if text.trim().is_empty() {
        return Err("Cannot save an empty transcription".into());
    }

    let (entry, audio_history_budget_gb) = state.update(|database| {
        let entry = DictationEntry {
            id: make_id("dictation"),
            text,
            raw_text,
            created_at: Utc::now(),
            duration_ms,
            mode,
            source_application,
            source_window_title,
            audio_file,
            audio_model,
            was_ai_processed: was_ai_processed.unwrap_or(false),
            processing_model,
            ai_processing_error,
        };
        database.dictation_history.insert(0, entry.clone());
        (entry, database.settings.audio_history_budget_gb)
    })?;
    if entry.audio_file.is_some() {
        enforce_audio_history_budget(&state, audio_history_budget_gb)?;
    }
    state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .dictation_history
        .iter()
        .find(|saved| saved.id == entry.id)
        .cloned()
        .ok_or("The saved dictation could not be found".into())
}

#[tauri::command]
fn copy_completed_dictation(text: String) -> Result<(), String> {
    typing::copy_text_to_clipboard(&text)
}

#[tauri::command]
fn copy_text_to_clipboard(text: String) -> Result<(), String> {
    typing::copy_text_to_clipboard(&text)
}

fn dictionary_learning_boundary(character: char) -> bool {
    character.is_whitespace()
        || matches!(
            character,
            ',' | '.'
                | '!'
                | '?'
                | ';'
                | ':'
                | '"'
                | '“'
                | '”'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
        )
}

fn dictionary_learning_clean(value: String) -> String {
    value
        .trim_matches(|character: char| dictionary_learning_boundary(character))
        .to_owned()
}

fn automatic_dictionary_candidate(before: &str, after: &str) -> Option<(String, String)> {
    if before == after || before.chars().count() > 100_000 || after.chars().count() > 100_000 {
        return None;
    }
    let before = before.chars().collect::<Vec<_>>();
    let after = after.chars().collect::<Vec<_>>();
    let mut prefix = 0;
    while prefix < before.len() && prefix < after.len() && before[prefix] == after[prefix] {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < before.len().saturating_sub(prefix)
        && suffix < after.len().saturating_sub(prefix)
        && before[before.len() - suffix - 1] == after[after.len() - suffix - 1]
    {
        suffix += 1;
    }
    let expand = |value: &[char], mut start: usize, mut end: usize| {
        while start > 0 && !dictionary_learning_boundary(value[start - 1]) {
            start -= 1;
        }
        while end < value.len() && !dictionary_learning_boundary(value[end]) {
            end += 1;
        }
        (start, end)
    };
    let (before_start, before_end) = expand(&before, prefix, before.len() - suffix);
    let (after_start, after_end) = expand(&after, prefix, after.len() - suffix);
    let heard = dictionary_learning_clean(before[before_start..before_end].iter().collect());
    let corrected = dictionary_learning_clean(after[after_start..after_end].iter().collect());
    let valid = |value: &str| {
        !value.is_empty()
            && value.chars().count() <= 40
            && value.split_whitespace().count() <= 3
            && value.chars().any(char::is_alphanumeric)
            && value
                .chars()
                .filter(|character| character.is_alphanumeric())
                .count()
                >= 2
            && value.chars().any(char::is_alphabetic)
    };
    (valid(&heard) && valid(&corrected) && heard.to_lowercase() != corrected.to_lowercase())
        .then_some((heard, corrected))
}

fn dictionary_learning_pair_matches(
    record: &DictionaryLearningRecord,
    heard: &str,
    corrected: &str,
) -> bool {
    record.heard_text.eq_ignore_ascii_case(heard)
        && record.corrected_text.eq_ignore_ascii_case(corrected)
}

fn prune_dictionary_learning_records(database: &mut AppDatabase, now: DateTime<Utc>) {
    let retention = chrono::Duration::days(30);
    for record in &mut database.dictionary_learning_records {
        record
            .occurrences
            .retain(|occurrence| *occurrence >= now - retention);
    }
    database.dictionary_learning_records.retain(|record| {
        record.is_accepted
            || record
                .last_shown_at
                .is_some_and(|shown_at| shown_at >= now - retention)
            || !record.occurrences.is_empty()
    });
    database.dictionary_learning_records.sort_by(|left, right| {
        right
            .occurrences
            .last()
            .cmp(&left.occurrences.last())
            .then_with(|| left.heard_text.cmp(&right.heard_text))
    });
    database.dictionary_learning_records.truncate(200);
}

fn record_automatic_dictionary_candidate(database: &mut AppDatabase, before: &str, after: &str) {
    if !database.settings.automatic_dictionary_learning_enabled {
        return;
    }
    let Some((heard, corrected)) = automatic_dictionary_candidate(before, after) else {
        return;
    };
    let now = Utc::now();
    prune_dictionary_learning_records(database, now);
    if let Some(record) = database
        .dictionary_learning_records
        .iter_mut()
        .find(|record| dictionary_learning_pair_matches(record, &heard, &corrected))
    {
        record
            .occurrences
            .retain(|occurrence| *occurrence >= now - chrono::Duration::days(7));
        record.occurrences.push(now);
        return;
    }
    database
        .dictionary_learning_records
        .push(DictionaryLearningRecord {
            heard_text: heard,
            corrected_text: corrected,
            occurrences: vec![now],
            last_shown_at: None,
            dismissed_until: None,
            dismissal_count: 0,
            is_accepted: false,
        });
    prune_dictionary_learning_records(database, now);
}

#[tauri::command]
fn dictionary_learning_suggestions(
    state: State<'_, AppState>,
) -> Result<Vec<DictionaryLearningSuggestion>, String> {
    state.update(|database| {
        if !database.settings.automatic_dictionary_learning_enabled {
            return Vec::new();
        }
        let now = Utc::now();
        prune_dictionary_learning_records(database, now);
        if database
            .dictionary_learning_last_shown_at
            .is_some_and(|shown_at| shown_at > now - chrono::Duration::minutes(10))
        {
            return Vec::new();
        }
        let suggestion = database
            .dictionary_learning_records
            .iter_mut()
            .filter(|record| {
                !record.is_accepted
                    && record.dismissal_count < 3
                    && record.dismissed_until.is_none_or(|until| until <= now)
                    && record
                        .occurrences
                        .iter()
                        .filter(|occurrence| **occurrence >= now - chrono::Duration::days(7))
                        .count()
                        >= 2
            })
            .max_by(|left, right| left.occurrences.len().cmp(&right.occurrences.len()));
        let Some(record) = suggestion else {
            return Vec::new();
        };
        record.last_shown_at = Some(now);
        database.dictionary_learning_last_shown_at = Some(now);
        vec![DictionaryLearningSuggestion {
            heard_text: record.heard_text.clone(),
            corrected_text: record.corrected_text.clone(),
            occurrences: record.occurrences.len(),
        }]
    })
}

#[tauri::command]
fn accept_dictionary_learning_suggestion(
    state: State<'_, AppState>,
    heard_text: String,
    corrected_text: String,
) -> Result<Vec<DictionaryEntry>, String> {
    let heard_text = heard_text.trim().to_owned();
    let corrected_text = corrected_text.trim().to_owned();
    if automatic_dictionary_candidate(&heard_text, &corrected_text).is_none() {
        return Err("That dictionary suggestion is not valid".into());
    }
    state.update(|database| -> Result<_, String> {
        let record = database
            .dictionary_learning_records
            .iter_mut()
            .find(|record| dictionary_learning_pair_matches(record, &heard_text, &corrected_text))
            .ok_or("That dictionary suggestion is no longer available")?;
        record.is_accepted = true;
        record.dismissed_until = None;
        if let Some(existing) = database
            .dictionary
            .iter_mut()
            .find(|entry| entry.spoken.eq_ignore_ascii_case(&heard_text))
        {
            existing.replacement = corrected_text;
        } else {
            database.dictionary.push(DictionaryEntry {
                id: make_id("dictionary"),
                spoken: heard_text.to_lowercase(),
                replacement: corrected_text,
                created_at: Utc::now(),
            });
        }
        Ok(database.dictionary.clone())
    })?
}

#[tauri::command]
fn dismiss_dictionary_learning_suggestion(
    state: State<'_, AppState>,
    heard_text: String,
    corrected_text: String,
) -> Result<(), String> {
    state.update(|database| -> Result<_, String> {
        let record = database
            .dictionary_learning_records
            .iter_mut()
            .find(|record| dictionary_learning_pair_matches(record, &heard_text, &corrected_text))
            .ok_or("That dictionary suggestion is no longer available")?;
        record.dismissal_count = record.dismissal_count.saturating_add(1);
        record.dismissed_until = Some(Utc::now() + chrono::Duration::days(7));
        Ok(())
    })?
}

#[tauri::command]
fn update_dictation(
    state: State<'_, AppState>,
    id: String,
    text: String,
    raw_text: Option<String>,
) -> Result<DictationEntry, String> {
    let text = text.trim().to_owned();
    if text.is_empty() {
        return Err("A transcription cannot be empty".into());
    }
    state.update(|database| {
        let entry = database
            .dictation_history
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or("That dictation no longer exists")?;
        let previous_text = entry.text.clone();
        entry.text = text;
        entry.raw_text = raw_text.map(|raw_text| raw_text.trim().to_owned());
        let entry = entry.clone();
        record_automatic_dictionary_candidate(database, &previous_text, &entry.text);
        Ok(entry)
    })?
}

const FEEDBACK_ISSUES_URL: &str = "https://github.com/pmd-coutinho/voxide/issues/new";

/// Feedback goes through the public GitHub issue tracker; nothing is
/// submitted anywhere until the user files the issue themselves.
#[tauri::command]
fn open_feedback_issue() -> Result<(), String> {
    tauri_plugin_opener::open_url(FEEDBACK_ISSUES_URL, None::<&str>)
        .map_err(|error| format!("Could not open the issue tracker: {error}"))
}

/// Environment summary plus recent diagnostic log lines, for the user to
/// paste into an issue when relevant.
#[tauri::command]
fn feedback_debug_information(state: State<'_, AppState>) -> String {
    let settings = state
        .database
        .lock()
        .map(|database| database.settings.clone())
        .unwrap_or_default();
    let engine = settings.selected_voice_engine;
    let mut information = format!(
        "App Version: {}\nOS: {}\nArchitecture: {}\nDate: {}\nSpeech Engine: {}\nEngine Runtime: {}\nEngine Model: {}\n",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        Utc::now().to_rfc3339(),
        asr::SpeechEngine::engine_id(&engine),
        engine.diagnostic_runtime_version(&state),
        engine.diagnostic_model_id(&settings),
    );
    let recent_log_entries = debug_log::recent_lines(30);
    if !recent_log_entries.is_empty() {
        information.push_str("\nRecent Log Entries:\n");
        information.push_str(&recent_log_entries);
        information.push('\n');
    }
    information
}

#[tauri::command]
fn delete_dictation(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let audio_file = state.update(|database| {
        let audio_file = database
            .dictation_history
            .iter()
            .find(|entry| entry.id == id)
            .and_then(|entry| entry.audio_file.clone());
        database.dictation_history.retain(|entry| entry.id != id);
        audio_file
    })?;
    if let Some(audio_file) = audio_file {
        delete_saved_audio_file(&state, &audio_file)?;
    }
    Ok(())
}

#[tauri::command]
fn clear_dictation_history(state: State<'_, AppState>) -> Result<(), String> {
    let audio_files = state.update(|database| {
        let audio_files = database
            .dictation_history
            .iter()
            .filter_map(|entry| entry.audio_file.clone())
            .collect::<Vec<_>>();
        database.dictation_history.clear();
        audio_files
    })?;
    for audio_file in audio_files {
        delete_saved_audio_file(&state, &audio_file)?;
    }
    Ok(())
}

#[tauri::command]
fn export_dictation(
    state: State<'_, AppState>,
    id: String,
    destination: String,
) -> Result<(), String> {
    let text = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .dictation_history
        .iter()
        .find(|entry| entry.id == id)
        .map(|entry| entry.text.clone())
        .ok_or("That dictation no longer exists")?;
    let destination = PathBuf::from(destination.trim());
    if destination.as_os_str().is_empty() {
        return Err("Choose a destination for the export".into());
    }
    fs::write(&destination, text)
        .map_err(|error| format!("Could not export the dictation: {error}"))
}

#[tauri::command]
fn delete_file_transcription(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.update(|database| {
        database
            .file_transcription_history
            .retain(|entry| entry.id != id)
    })
}

#[tauri::command]
fn clear_file_transcription_history(state: State<'_, AppState>) -> Result<(), String> {
    state.update(|database| database.file_transcription_history.clear())
}

#[tauri::command]
fn export_file_transcription(
    state: State<'_, AppState>,
    id: String,
    destination: String,
    format: Option<FileTranscriptionExportFormat>,
) -> Result<(), String> {
    let entry = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .file_transcription_history
        .iter()
        .find(|entry| entry.id == id)
        .cloned()
        .ok_or("That file transcription no longer exists")?;
    let destination = PathBuf::from(destination.trim());
    if destination.as_os_str().is_empty() {
        return Err("Choose a destination for the export".into());
    }
    let format = format.unwrap_or(FileTranscriptionExportFormat::Text);
    let contents = file_transcription_export(&entry, format)?;
    fs::write(&destination, contents)
        .map_err(|error| format!("Could not export the transcription: {error}"))
}

#[tauri::command]
fn save_file_transcription(
    state: State<'_, AppState>,
    file_name: String,
    text: String,
    duration_ms: Option<u64>,
) -> Result<FileTranscriptionEntry, String> {
    if file_name.trim().is_empty() || text.trim().is_empty() {
        return Err("A file name and transcription are required".into());
    }

    state.update(|database| {
        let entry = FileTranscriptionEntry {
            id: make_id("file"),
            file_name,
            text,
            created_at: Utc::now(),
            duration_ms,
            processing_time_ms: None,
            confidence: None,
        };
        database.file_transcription_history.insert(0, entry.clone());
        database.file_transcription_history.truncate(50);
        entry
    })
}

/// Feeds a media file through the same native cache-aware Nemotron stream as
/// microphone dictation. This avoids a second, non-streaming file-only model
/// path and keeps memory bounded for long recordings.
async fn transcribe_nemotron_media_file(
    state: &AppState,
    path: &Path,
    language: &str,
    lookahead_tokens: u8,
    progress: Option<speech::ProgressCallback>,
) -> Result<(String, u64), String> {
    const FILE_CHUNK_SECONDS: f64 = 10.0;
    if !nemotron::is_compiled() {
        return Err("Nemotron is available in Voxide's CUDA build for Linux/NVIDIA".into());
    }
    let model = nemotron_model_path(state)?;
    let runtime = nemotron_runtime_path(state)?;
    if !nemotron_runtime_is_verified(&runtime) {
        return Err("Install the Nemotron CUDA runtime before transcribing a file".into());
    }
    if !nemotron_model_is_verified(&model) {
        return Err("Download the Nemotron model before transcribing a file".into());
    }
    let script = ensure_nemotron_server_script(&runtime)?;
    let mut server =
        nemotron::Server::launch(&nemotron::python_path(&runtime), &script, &model).await?;
    server.start(language, lookahead_tokens).await?;
    let duration_ms = media::file_duration_ms(path)?;
    let total_chunks = ((duration_ms as f64 / 1_000.0) / FILE_CHUNK_SECONDS)
        .ceil()
        .max(1.0) as usize;
    for chunk in 0..total_chunks {
        let start_seconds = chunk as f64 * FILE_CHUNK_SECONDS;
        let remaining_seconds = (duration_ms as f64 / 1_000.0 - start_seconds).max(0.0);
        let file = path.to_path_buf();
        let audio = tauri::async_runtime::spawn_blocking(move || {
            media::decode_audio_segment(
                &file,
                start_seconds,
                remaining_seconds.min(FILE_CHUNK_SECONDS),
            )
        })
        .await
        .map_err(|error| format!("Nemotron file decode task failed: {error}"))??;
        let samples = audio::mono_resample_for_whisper(audio)?;
        if !samples.is_empty() {
            let _ = server.append(&samples).await?;
        }
        if let Some(progress) = &progress {
            progress(chunk + 1, total_chunks);
        }
    }
    let text = server.finish(&[]).await?;
    Ok((text, duration_ms))
}

#[tauri::command]
async fn transcribe_file(
    app: AppHandle,
    state: State<'_, AppState>,
    file_path: String,
) -> Result<Option<FileTranscriptionEntry>, String> {
    let path = PathBuf::from(file_path.trim());
    if !path.is_file() {
        return Err("Choose a readable audio or video file".into());
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("The selected file name is not valid Unicode")?
        .to_owned();
    let started = Instant::now();
    let (settings, custom_words) = {
        let database = state
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        let settings = database.settings.clone();
        let custom_words = active_recognition_vocabulary(&settings, &database.custom_words);
        (settings, custom_words)
    };
    let progress_app = app.clone();
    let progress = Arc::new(move |completed_chunks, total_chunks| {
        let _ = progress_app.emit(
            "file-transcription-progress",
            FileTranscriptionProgress {
                completed_chunks,
                total_chunks,
            },
        );
    });
    let (raw_text, duration_ms) = settings
        .selected_voice_engine
        .transcribe_file(&state, &settings, path.clone(), custom_words, progress)
        .await?;
    // Meeting/file transcription is intentionally separate from live
    // dictation. Voxide returns the ASR provider's file result directly,
    // without applying dictation formatting, dictionary replacement, or an
    // AI-enhancement prompt.
    let text = raw_text;
    let confidence = if text.trim().is_empty() { 0.0 } else { 1.0 };
    let entry = if should_save_file_transcription(&text) {
        state
            .update(|database| {
                let entry = FileTranscriptionEntry {
                    id: make_id("file"),
                    file_name,
                    text: text.clone(),
                    created_at: Utc::now(),
                    duration_ms: Some(duration_ms),
                    processing_time_ms: Some(started.elapsed().as_millis() as u64),
                    confidence: Some(confidence),
                };
                database.file_transcription_history.insert(0, entry.clone());
                database.file_transcription_history.truncate(50);
                entry
            })?
            .into()
    } else {
        None
    };
    let file_type = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .unwrap_or_else(|| "unknown".into());
    app.state::<analytics::AnalyticsService>().capture(
        "meeting_transcription_completed",
        settings.share_anonymous_analytics,
        analytics_properties_with(
            &settings,
            [
                ("success", Value::Bool(true)),
                ("file_type", Value::String(file_type)),
                (
                    "words_bucket",
                    Value::String(analytics::word_count_bucket(&text).into()),
                ),
                (
                    "duration_bucket",
                    Value::String(analytics::milliseconds_bucket(duration_ms).into()),
                ),
            ],
        ),
    );
    Ok(entry)
}

#[tauri::command]
fn save_dictionary(
    state: State<'_, AppState>,
    dictionary: Vec<DictionaryEntry>,
) -> Result<Vec<DictionaryEntry>, String> {
    if dictionary.len() > 10_000 {
        return Err("Keep at most 10,000 dictionary correction entries".into());
    }
    let dictionary = normalize_dictionary_entries(dictionary);
    state.update(|database| {
        database.dictionary = dictionary.clone();
        dictionary
    })
}

#[tauri::command]
fn custom_words(state: State<'_, AppState>) -> Result<Vec<CustomWordEntry>, String> {
    state
        .database
        .lock()
        .map(|database| database.custom_words.clone())
        .map_err(|_| "Voxide data lock was poisoned".to_string())
}

#[tauri::command]
fn save_custom_words(
    state: State<'_, AppState>,
    words: Vec<CustomWordEntry>,
) -> Result<Vec<CustomWordEntry>, String> {
    let words = normalize_custom_words(words);
    state.update(|database| {
        database.custom_words = words.clone();
        words
    })
}

#[tauri::command]
fn export_dictionary(state: State<'_, AppState>, destination: String) -> Result<(), String> {
    let document = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .clone();
    let destination = PathBuf::from(destination.trim());
    if destination.as_os_str().is_empty() {
        return Err("Choose a destination for the dictionary export".into());
    }
    let contents = serde_json::to_vec_pretty(&DictionaryTransferDocument {
        version: 1,
        replacements: document
            .dictionary
            .iter()
            .map(|entry| DictionaryTransferEntry {
                spoken: entry.spoken.clone(),
                replacement: entry.replacement.clone(),
            })
            .collect(),
        custom_words: document.custom_words,
    })
    .map_err(|error| format!("Could not encode the dictionary export: {error}"))?;
    fs::write(destination, contents)
        .map_err(|error| format!("Could not save the dictionary export: {error}"))
}

#[tauri::command]
fn import_dictionary(
    state: State<'_, AppState>,
    source: String,
) -> Result<DictionaryImportResult, String> {
    let source = PathBuf::from(source.trim());
    let contents = fs::read(&source)
        .map_err(|error| format!("Could not read the selected dictionary: {error}"))?;
    let document: DictionaryImportDocument =
        serde_json::from_slice(&contents).map_err(|error| {
            format!("The selected file is not a valid Voxide dictionary export: {error}")
        })?;
    let (entries, custom_words) = match document {
        DictionaryImportDocument::Current(document) => {
            (document.replacements, document.custom_words)
        }
        DictionaryImportDocument::Legacy(entries) => (entries, Vec::new()),
    };
    if entries.len() > 10_000 {
        return Err("A dictionary can contain at most 10,000 entries".into());
    }
    let mut seen = HashSet::new();
    let dictionary = entries
        .into_iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let spoken = entry.spoken.trim().to_lowercase();
            let replacement = entry.replacement.trim().to_owned();
            (!spoken.is_empty() && !replacement.is_empty() && seen.insert(spoken.clone())).then(
                || DictionaryEntry {
                    id: format!("{}-{index}", make_id("dictionary")),
                    spoken,
                    replacement,
                    created_at: Utc::now(),
                },
            )
        })
        .collect::<Vec<_>>();
    if dictionary.is_empty() {
        return Err("The selected dictionary has no valid correction entries".into());
    }
    state.update(|database| {
        database.dictionary = dictionary.clone();
        database.custom_words = normalize_custom_words(custom_words);
        DictionaryImportResult {
            dictionary,
            custom_words: database.custom_words.clone(),
        }
    })
}

#[tauri::command]
fn save_prompt_profiles(
    app: AppHandle,
    state: State<'_, AppState>,
    registry: State<'_, HotkeyRegistry>,
    profiles: Vec<PromptProfile>,
) -> Result<Vec<PromptProfile>, String> {
    if profiles.is_empty() || profiles.len() > 100 {
        return Err("Keep between one and 100 prompt profiles".into());
    }
    let mut ids = HashSet::new();
    if profiles.iter().any(|profile| {
        profile.id.trim().is_empty()
            || profile.name.trim().is_empty()
            || profile.prompt.trim().is_empty()
            || !ids.insert(profile.id.trim().to_owned())
    }) {
        return Err("Each prompt profile needs a unique ID, name, and prompt".into());
    }
    if profiles.iter().any(|profile| {
        !matches!(
            profile.mode,
            DictationMode::Dictate | DictationMode::Rewrite | DictationMode::Command
        )
    }) {
        return Err("Prompt profiles must target dictate, rewrite, or command mode".into());
    }
    let saved_profiles = state.update(|database| {
        database.prompt_profiles = profiles;
        normalize_database(database);
        database.prompt_profiles.clone()
    })?;
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    apply_hotkeys(&app, &registry, &settings)?;
    Ok(saved_profiles)
}

#[tauri::command]
fn save_dictation_prompt_configurations(
    state: State<'_, AppState>,
    configurations: Vec<DictationPromptConfiguration>,
) -> Result<Vec<DictationPromptConfiguration>, String> {
    if configurations.len() > 100 {
        return Err("Keep at most 100 Dictate prompt provider configurations".into());
    }
    state.update(|database| {
        database.dictation_prompt_configurations = configurations;
        normalize_database(database);
        database.dictation_prompt_configurations.clone()
    })
}

#[tauri::command]
fn set_active_prompt_profile(
    state: State<'_, AppState>,
    profile_id: String,
) -> Result<Settings, String> {
    let profile_id = profile_id.trim();
    if profile_id.is_empty() {
        return Err("Choose a prompt profile".into());
    }
    state.update(|database| {
        let profile = database
            .prompt_profiles
            .iter()
            .find(|profile| profile.id == profile_id)
            .cloned()
            .ok_or("That prompt profile no longer exists")?;
        database
            .settings
            .set_active_prompt_profile_id(profile.mode, Some(profile.id));
        Ok(database.settings.clone())
    })?
}

#[tauri::command]
fn app_prompt_bindings(state: State<'_, AppState>) -> Result<Vec<AppPromptBinding>, String> {
    state
        .database
        .lock()
        .map(|database| database.app_prompt_bindings.clone())
        .map_err(|_| "Voxide data lock was poisoned".to_string())
}

#[tauri::command]
fn save_app_prompt_bindings(
    state: State<'_, AppState>,
    bindings: Vec<AppPromptBinding>,
) -> Result<Vec<AppPromptBinding>, String> {
    if bindings.len() > 100 {
        return Err("Keep at most 100 app prompt bindings".into());
    }
    state.update(|database| -> Result<_, String> {
        let mut seen = HashSet::new();
        let bindings = bindings
            .into_iter()
            .map(|binding| AppPromptBinding {
                id: if binding.id.trim().is_empty() {
                    make_id("app-prompt")
                } else {
                    binding.id.trim().to_owned()
                },
                application: binding.application.trim().to_owned(),
                mode: binding.mode,
                prompt_profile_id: binding.prompt_profile_id.trim().to_owned(),
            })
            .map(|binding| {
                let valid_mode = editable_prompt_modes().contains(&binding.mode);
                let valid_profile = database.prompt_profiles.iter().any(|profile| {
                    profile.mode == binding.mode && profile.id == binding.prompt_profile_id
                });
                let unique = seen.insert((binding.mode, binding.application.to_lowercase()));
                (valid_mode && valid_profile && unique && !binding.application.is_empty())
                    .then_some(binding)
                    .ok_or("Each app prompt binding needs a unique app, mode, and matching prompt profile")
            })
            .collect::<Result<Vec<_>, _>>()?;
        database.app_prompt_bindings = bindings;
        Ok(database.app_prompt_bindings.clone())
    })?
}

const KEYRING_SERVICE: &str = "dev.pmdcoutinho.voxide";

fn provider_keyring_entry(provider_id: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYRING_SERVICE, &format!("provider:{provider_id}"))
        .map_err(|error| format!("Could not access secure credential storage: {error}"))
}

fn provider_api_key(provider_id: &str) -> Result<Option<String>, String> {
    match provider_keyring_entry(provider_id)?.get_password() {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value)),
        Ok(_) | Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(format!(
            "Could not read the API key from secure storage: {error}"
        )),
    }
}

fn cloud_transcription_profile(
    settings: &Settings,
    database: &AppDatabase,
) -> Result<provider::AiProviderProfile, String> {
    if settings.cloud_transcription_model.trim().is_empty() {
        return Err("Choose a cloud transcription model before recording.".into());
    }
    let profile = selected_provider(database, None)?;
    if !matches!(
        profile.api_style,
        provider::ProviderApiStyle::OpenAiCompatible
    ) {
        return Err(format!(
            "{} does not expose an OpenAI-compatible audio transcription endpoint. Choose a compatible provider in AI Enhancement.",
            profile.name
        ));
    }
    Ok(profile)
}

/// Validates only the local configuration required to begin cloud capture. A
/// real request is deliberately not made here: microphone start must not
/// perform an unprompted network call, while missing credentials should still
/// fail before audio is captured.
fn cloud_transcription_readiness(
    settings: &Settings,
    state: &AppState,
) -> Result<provider::AiProviderProfile, String> {
    let profile = {
        let database = state
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        cloud_transcription_profile(settings, &database)?
    };
    if !provider::is_local_endpoint(&profile.base_url) && provider_api_key(&profile.id)?.is_none() {
        return Err(format!(
            "Set an API key for {} in AI Enhancement before recording.",
            profile.name
        ));
    }
    Ok(profile)
}

fn configured_provider(
    database: &AppDatabase,
    provider_id: Option<&str>,
) -> Result<provider::AiProviderProfile, String> {
    let provider_id = provider_id.unwrap_or(&database.settings.selected_ai_provider);
    let profile = database
        .ai_providers
        .iter()
        .find(|profile| profile.id == provider_id)
        .cloned()
        .ok_or_else(|| format!("AI provider '{provider_id}' is not configured"))?;
    if !profile.enabled {
        return Err(format!("AI provider '{}' is disabled", profile.name));
    }
    Ok(profile)
}

fn apply_reasoning_configuration(
    database: &AppDatabase,
    profile: &mut provider::AiProviderProfile,
) {
    if let Some(config) = database.settings.reasoning_config_for(&profile) {
        let parameter_name = config.parameter_name.trim();
        let parameter_value = config.parameter_value.trim();
        if !parameter_name.is_empty() {
            let value = if parameter_name == "enable_thinking" {
                Value::Bool(parameter_value.eq_ignore_ascii_case("true"))
            } else {
                Value::String(parameter_value.into())
            };
            profile
                .request_parameters
                .insert(parameter_name.into(), value);
        }
    }
}

fn selected_provider(
    database: &AppDatabase,
    provider_id: Option<&str>,
) -> Result<provider::AiProviderProfile, String> {
    let mut profile = configured_provider(database, provider_id)?;
    apply_reasoning_configuration(database, &mut profile);
    Ok(profile)
}

fn dictation_provider_for_prompt_profile(
    database: &AppDatabase,
    prompt_profile_id: &str,
) -> Result<provider::AiProviderProfile, String> {
    let configuration = database
        .dictation_prompt_configurations
        .iter()
        .find(|configuration| configuration.prompt_profile_id == prompt_profile_id);
    let provider_id = configuration.and_then(|configuration| configuration.provider_id.as_deref());
    let mut provider = configured_provider(database, provider_id)?;
    if let Some(model) = configuration.and_then(|configuration| configuration.model.as_deref()) {
        provider.model = model.to_owned();
    }
    apply_reasoning_configuration(database, &mut provider);
    Ok(provider)
}

fn provider_for_mode(
    database: &AppDatabase,
    mode: DictationMode,
) -> Result<provider::AiProviderProfile, String> {
    let selected = match mode {
        DictationMode::Rewrite => database.settings.selected_rewrite_ai_provider.as_deref(),
        DictationMode::Command => database.settings.selected_command_ai_provider.as_deref(),
        DictationMode::Dictate | DictationMode::Prompt | DictationMode::File => None,
    };
    selected_provider(database, selected)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AiProviderView {
    profile: provider::AiProviderProfile,
    has_api_key: bool,
}

#[tauri::command]
fn ai_providers(state: State<'_, AppState>) -> Result<Vec<AiProviderView>, String> {
    let profiles = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .ai_providers
        .clone();
    profiles
        .into_iter()
        .map(|profile| {
            let has_api_key = provider_api_key(&profile.id)?.is_some();
            Ok(AiProviderView {
                profile,
                has_api_key,
            })
        })
        .collect()
}

#[tauri::command]
fn save_ai_providers(
    state: State<'_, AppState>,
    mut providers: Vec<provider::AiProviderProfile>,
) -> Result<Vec<provider::AiProviderProfile>, String> {
    validate_and_normalize_ai_provider_profiles(&mut providers)?;
    state.update(|database| {
        database.ai_providers = providers.clone();
        normalize_database(database);
        database.ai_providers.clone()
    })
}

#[tauri::command]
fn set_provider_api_key(provider_id: String, api_key: String) -> Result<bool, String> {
    let entry = provider_keyring_entry(&provider_id)?;
    if api_key.trim().is_empty() {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(false),
            Err(error) => Err(format!(
                "Could not remove the API key from secure storage: {error}"
            )),
        }
    } else {
        entry
            .set_password(&api_key)
            .map_err(|error| format!("Could not save the API key in secure storage: {error}"))?;
        Ok(true)
    }
}

#[tauri::command]
fn move_provider_api_key(old_provider_id: String, new_provider_id: String) -> Result<bool, String> {
    let old_provider_id = old_provider_id.trim();
    let new_provider_id = new_provider_id.trim();
    if old_provider_id.is_empty() || new_provider_id.is_empty() {
        return Err("Both provider IDs are required to move an API key".into());
    }
    if old_provider_id == new_provider_id {
        return Ok(provider_api_key(old_provider_id)?.is_some());
    }
    let old_entry = provider_keyring_entry(old_provider_id)?;
    let api_key = match old_entry.get_password() {
        Ok(api_key) if !api_key.trim().is_empty() => api_key,
        Ok(_) | Err(keyring::Error::NoEntry) => return Ok(false),
        Err(error) => {
            return Err(format!(
                "Could not read the existing API key from secure storage: {error}"
            ))
        }
    };
    let new_entry = provider_keyring_entry(new_provider_id)?;
    new_entry
        .set_password(&api_key)
        .map_err(|error| format!("Could not store the moved API key securely: {error}"))?;
    match old_entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(true),
        Err(error) => Err(format!(
            "The API key was copied but the old secure credential could not be removed: {error}"
        )),
    }
}

#[tauri::command]
async fn fetch_ai_provider_models(
    state: State<'_, AppState>,
    provider_id: String,
) -> Result<Vec<String>, String> {
    let profile = {
        let database = state
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        selected_provider(&database, Some(&provider_id))?
    };
    provider::fetch_models(&profile, provider_api_key(&profile.id)?.as_deref()).await
}

#[tauri::command]
async fn enhance_text(
    app: AppHandle,
    state: State<'_, AppState>,
    text: String,
    system_prompt: String,
    provider_id: Option<String>,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("Text is required for AI enhancement".into());
    }
    let (profile, settings) = {
        let database = state
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        (
            selected_provider(&database, provider_id.as_deref())?,
            database.settings.clone(),
        )
    };
    let api_key = provider_api_key(&profile.id)?;
    let started = Instant::now();
    let result = provider::process_with_options(
        &profile,
        api_key.as_deref(),
        &system_prompt,
        &text,
        0.7,
        provider::is_reasoning_model(&profile.model).then_some(32_000),
    )
    .await;
    app.state::<analytics::AnalyticsService>().capture(
        "rewrite_run_completed",
        settings.share_anonymous_analytics,
        analytics_properties_with(
            &settings,
            [
                ("success", Value::Bool(result.is_ok())),
                (
                    "latency_bucket",
                    Value::String(
                        analytics::milliseconds_bucket(started.elapsed().as_millis() as u64).into(),
                    ),
                ),
            ],
        ),
    );
    result
}

#[tauri::command]
fn command_chats(state: State<'_, AppState>) -> Result<Vec<CommandChat>, String> {
    let database = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?;
    let mut chats = database.command_chats.clone();
    chats.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(chats)
}

#[tauri::command]
fn create_command_chat(state: State<'_, AppState>) -> Result<CommandChat, String> {
    state.update(|database| {
        let chat = CommandChat::new();
        database.active_command_chat_id = Some(chat.id.clone());
        database.command_chats.insert(0, chat.clone());
        normalize_command_chats(database);
        chat
    })
}

#[tauri::command]
fn select_command_chat(state: State<'_, AppState>, chat_id: String) -> Result<CommandChat, String> {
    let chat_id = chat_id.trim();
    state.update(|database| {
        let chat = database
            .command_chats
            .iter()
            .find(|chat| chat.id == chat_id)
            .cloned()
            .ok_or("That command chat no longer exists")?;
        database.active_command_chat_id = Some(chat.id.clone());
        Ok(chat)
    })?
}

#[tauri::command]
fn clear_command_chat(state: State<'_, AppState>, chat_id: String) -> Result<CommandChat, String> {
    let chat_id = chat_id.trim();
    state.update(|database| {
        let chat = database
            .command_chats
            .iter_mut()
            .find(|chat| chat.id == chat_id)
            .ok_or("That command chat no longer exists")?;
        chat.messages.clear();
        chat.title = "New chat".into();
        chat.updated_at = Utc::now();
        Ok(chat.clone())
    })?
}

#[tauri::command]
fn delete_command_chat(state: State<'_, AppState>, chat_id: String) -> Result<CommandChat, String> {
    let chat_id = chat_id.trim();
    state.update(|database| -> Result<CommandChat, String> {
        let previous_count = database.command_chats.len();
        database.command_chats.retain(|chat| chat.id != chat_id);
        if database.command_chats.len() == previous_count {
            return Err("That command chat no longer exists".into());
        }
        normalize_command_chats(database);
        let active_id = database
            .active_command_chat_id
            .as_deref()
            .ok_or("Voxide could not select a command chat")?;
        database
            .command_chats
            .iter()
            .find(|chat| chat.id == active_id)
            .cloned()
            .ok_or_else(|| "Voxide could not select a command chat".to_string())
    })?
}

fn command_chat_context(messages: &[CommandChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .take(20)
        .rev()
        .map(|message| {
            let role = match message.role {
                CommandChatRole::User => "User",
                CommandChatRole::Assistant => "Assistant",
                CommandChatRole::Tool => "Tool result",
            };
            let content = message.content.chars().take(2_000).collect::<String>();
            format!("{role}: {content}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn command_provider_messages(
    messages: &[CommandChatMessage],
) -> Result<Vec<provider::CommandProviderMessage>, String> {
    let mut provider_messages = Vec::new();
    let mut last_tool_call_id = None;
    for message in messages.iter().rev().take(20).rev() {
        match message.role {
            CommandChatRole::User => {
                provider_messages.push(provider::CommandProviderMessage::User(
                    message.content.chars().take(2_000).collect(),
                ))
            }
            CommandChatRole::Assistant => {
                if let Some(tool_call) = &message.tool_call {
                    last_tool_call_id = Some(tool_call.id.clone());
                }
                provider_messages.push(provider::CommandProviderMessage::Assistant {
                    content: message.content.chars().take(2_000).collect(),
                    tool_call: message.tool_call.clone(),
                });
            }
            CommandChatRole::Tool => {
                provider_messages.push(provider::CommandProviderMessage::Tool {
                    content: message.content.chars().take(2_000).collect(),
                    // Older saved conversations predate exact provider IDs. The source
                    // continues those conversations with the immediately preceding tool
                    // call (or a harmless sentinel when history was truncated).
                    tool_call_id: message
                        .tool_call_id
                        .clone()
                        .or_else(|| last_tool_call_id.clone())
                        .unwrap_or_else(|| "call_unknown".into()),
                })
            }
        }
    }
    Ok(provider_messages)
}

fn command_steps_since_latest_request(messages: &[CommandChatMessage]) -> usize {
    messages
        .iter()
        .rposition(|message| message.role == CommandChatRole::User)
        .map(|request_index| {
            messages[request_index + 1..]
                .iter()
                .filter(|message| message.role == CommandChatRole::Tool)
                .count()
        })
        .unwrap_or(0)
}

fn command_plan_summary(plan: &CommandPlan) -> String {
    match plan.kind.as_str() {
        "command" => format!(
            "Proposed command: {}\nPurpose: {}",
            plan.command.as_deref().unwrap_or_default(),
            plan.purpose.as_deref().unwrap_or("No purpose supplied.")
        ),
        "answer" => plan.answer.clone().unwrap_or_default(),
        _ => String::new(),
    }
}

fn command_system_prompt(command_prompt: &str) -> String {
    format!("{command_prompt}\n\nYou are Voxide Command Mode. {platform} Plan exactly one safe, user-requested desktop shell action, but never execute it. When the terminal tool is available, use it for an action and use a normal answer for a request that needs no shell action. For providers without tools, return exactly one JSON object and no Markdown. For an action, use {{\"kind\":\"command\",\"command\":\"shell command\",\"purpose\":\"short explanation\",\"workingDirectory\":null}}. For an answer, use {{\"kind\":\"answer\",\"answer\":\"helpful answer\"}}. Do not use sudo, do not request credentials, and do not hide, chain, or encode commands. The person will review the exact command before it runs.", platform = platform_command_guidance())
}

#[cfg(target_os = "macos")]
fn platform_command_guidance() -> &'static str {
    "The user is on macOS and reviewed commands run in zsh. Use `osascript -e 'tell application \"App Name\" to …'` for native application automation when appropriate."
}

#[cfg(target_os = "windows")]
fn platform_command_guidance() -> &'static str {
    "The user is on Windows and reviewed commands run through cmd.exe. Use native cmd commands, or invoke `powershell -NoProfile -Command` when PowerShell is needed; state the target application or path plainly."
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
fn platform_command_guidance() -> &'static str {
    "The user is on Linux and reviewed commands run in POSIX sh. Use standard shell commands and only use desktop-automation utilities after checking that they are available."
}

async fn request_command_plan<F, G>(
    profile: &provider::AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    provider_messages: Result<Vec<provider::CommandProviderMessage>, String>,
    fallback_input: &str,
    enable_streaming: bool,
    mut on_delta: F,
    mut on_thinking: G,
) -> Result<CommandPlan, String>
where
    F: FnMut(&str) + Send,
    G: FnMut(&str) + Send,
{
    if let Ok(messages) = provider_messages {
        let response = if enable_streaming {
            provider::process_command_with_tools_streaming(
                profile,
                api_key,
                system_prompt,
                &messages,
                &mut on_delta,
                &mut on_thinking,
            )
            .await
        } else {
            provider::process_command_with_tools(profile, api_key, system_prompt, &messages).await
        };
        match response {
            Ok(response) => return command_plan_from_provider_response(response),
            // Some local OpenAI-compatible servers deliberately do not expose
            // function calling. Preserve Command Mode for those endpoints with
            // the same JSON protocol used by previous app versions.
            Err(_) => {}
        }
    }
    let response = provider::process_with_options(
        profile,
        api_key,
        system_prompt,
        fallback_input,
        0.1,
        provider::is_reasoning_model(&profile.model).then_some(32_000),
    )
    .await?;
    parse_command_plan(&response)
}

fn command_plan_from_provider_response(
    response: provider::CommandProviderResponse,
) -> Result<CommandPlan, String> {
    match response {
        provider::CommandProviderResponse::Text { content, thinking } => {
            // Tool-capable providers return an ordinary assistant message when
            // no shell action is needed. Retain JSON support too, because some
            // OpenAI-compatible endpoints still follow the fallback prompt.
            match parse_command_plan(&content) {
                Ok(mut plan) => {
                    plan.thinking = thinking;
                    Ok(plan)
                }
                Err(_) if !content.trim().is_empty() => Ok(CommandPlan {
                    kind: "answer".into(),
                    conversation_id: None,
                    answer: Some(content),
                    thinking,
                    command: None,
                    purpose: None,
                    working_directory: None,
                    tool_call_id: None,
                    tool_call: None,
                    destructive: false,
                }),
                Err(error) => Err(error),
            }
        }
        provider::CommandProviderResponse::ToolCall {
            content,
            thinking,
            tool_call,
        } => {
            if tool_call.name != "execute_terminal_command" {
                return Err(format!(
                    "The AI provider requested unsupported Command Mode tool `{}`",
                    tool_call.name
                ));
            }
            let command = tool_call
                .arguments
                .get("command")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|command| !command.is_empty())
                .ok_or("The AI provider returned an empty Command Mode tool command")?
                .to_owned();
            let purpose = tool_call
                .arguments
                .get("purpose")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|purpose| !purpose.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    let content = content.trim();
                    (!content.is_empty()).then(|| content.to_owned())
                });
            let working_directory = tool_call
                .arguments
                .get("workingDirectory")
                .or_else(|| tool_call.arguments.get("working_directory"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|directory| !directory.is_empty())
                .map(str::to_owned);
            let mut plan = CommandPlan {
                kind: "command".into(),
                conversation_id: None,
                answer: None,
                thinking,
                command: Some(command),
                purpose,
                working_directory,
                tool_call_id: Some(tool_call.id.clone()),
                tool_call: Some(tool_call),
                destructive: false,
            };
            let command = plan.command.as_deref().expect("checked above");
            if command.len() > 16_384 || command.contains('\0') {
                return Err("The AI provider returned an invalid Command Mode tool command".into());
            }
            plan.destructive = is_destructive_command(command);
            Ok(plan)
        }
    }
}

fn append_command_plan(chat: &mut CommandChat, plan: &CommandPlan) {
    chat.append_with_tool_metadata(
        CommandChatRole::Assistant,
        command_plan_summary(plan),
        plan.tool_call.clone(),
        None,
        plan.thinking.clone(),
    );
}

#[tauri::command]
async fn plan_command(
    app: AppHandle,
    state: State<'_, AppState>,
    request: String,
    chat_id: Option<String>,
    source_application: Option<String>,
) -> Result<CommandPlan, String> {
    let request = request.trim().to_owned();
    if request.is_empty() {
        return Err("Describe the command or question first".into());
    }
    let source_application = source_application
        .as_deref()
        .map(str::trim)
        .filter(|application| !application.is_empty())
        .map(str::to_owned);
    let (profile, command_prompt, conversation_id, context, provider_messages, enable_ai_streaming) =
        state.update(|database| -> Result<_, String> {
            normalize_command_chats(database);
            let requested_id = chat_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty());
            let conversation_id = requested_id
                .map(str::to_owned)
                .or_else(|| database.active_command_chat_id.clone())
                .ok_or("Voxide could not create a command chat")?;
            let (context, chat_source_application, provider_messages) = {
                let chat = database
                    .command_chats
                    .iter_mut()
                    .find(|chat| chat.id == conversation_id)
                    .ok_or("That command chat no longer exists")?;
                if source_application.is_some() {
                    chat.source_application = source_application.clone();
                }
                let context = command_chat_context(&chat.messages);
                chat.append(CommandChatRole::User, request.clone());
                (
                    context,
                    chat.source_application.clone(),
                    command_provider_messages(&chat.messages),
                )
            };
            database.active_command_chat_id = Some(conversation_id.clone());
            Ok((
                provider_for_mode(database, DictationMode::Command)?,
                prompt_for_mode_and_application(
                    database,
                    DictationMode::Command,
                    chat_source_application.as_deref(),
                )
                .prompt,
                conversation_id,
                context,
                provider_messages,
                database.settings.enable_ai_streaming,
            ))
        })??;
    let system_prompt = command_system_prompt(&command_prompt);
    let input = if context.is_empty() {
        request
    } else {
        format!("Earlier conversation (use only as context):\n{context}\n\nNew user request:\n{request}")
    };
    let api_key = provider_api_key(&profile.id)?;
    let app_for_stream = app.clone();
    let conversation_for_stream = conversation_id.clone();
    let app_for_thinking = app.clone();
    let conversation_for_thinking = conversation_id.clone();
    let mut plan = request_command_plan(
        &profile,
        api_key.as_deref(),
        &system_prompt,
        provider_messages,
        &input,
        enable_ai_streaming,
        move |text| {
            let _ = app_for_stream.emit(
                "command-stream",
                CommandStreamUpdate {
                    conversation_id: conversation_for_stream.clone(),
                    text: Some(text.to_owned()),
                    thinking: None,
                },
            );
        },
        move |thinking| {
            let _ = app_for_thinking.emit(
                "command-stream",
                CommandStreamUpdate {
                    conversation_id: conversation_for_thinking.clone(),
                    text: None,
                    thinking: Some(thinking.to_owned()),
                },
            );
        },
    )
    .await?;
    plan.conversation_id = Some(conversation_id.clone());
    state.update(|database| {
        if let Some(chat) = database
            .command_chats
            .iter_mut()
            .find(|chat| chat.id == conversation_id)
        {
            append_command_plan(chat, &plan);
        }
    })?;
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    app.state::<analytics::AnalyticsService>().capture(
        "command_mode_run_completed",
        settings.share_anonymous_analytics,
        analytics_properties_with(
            &settings,
            [
                ("success", Value::Bool(true)),
                ("plan_kind", Value::String(plan.kind.clone())),
                ("destructive", Value::Bool(plan.destructive)),
            ],
        ),
    );
    Ok(plan)
}

#[tauri::command]
async fn continue_command(
    app: AppHandle,
    state: State<'_, AppState>,
    conversation_id: String,
) -> Result<CommandPlan, String> {
    let conversation_id = conversation_id.trim().to_owned();
    if conversation_id.is_empty() {
        return Err("Choose a command conversation to continue".into());
    }
    let (profile, command_prompt, context, provider_messages, enable_ai_streaming) = state
        .update(|database| -> Result<_, String> {
            let (context, source_application, provider_messages) = {
                let chat = database
                    .command_chats
                    .iter()
                    .find(|chat| chat.id == conversation_id)
                    .ok_or("That command chat no longer exists")?;
                let completed_steps = command_steps_since_latest_request(&chat.messages);
                if completed_steps >= 20 {
                    return Err("This command request reached the 20-step limit".into());
                }
                (
                    command_chat_context(&chat.messages),
                    chat.source_application.clone(),
                    command_provider_messages(&chat.messages),
                )
            };
            Ok((
                provider_for_mode(database, DictationMode::Command)?,
                prompt_for_mode_and_application(
                    database,
                    DictationMode::Command,
                    source_application.as_deref(),
                )
                .prompt,
                context,
                provider_messages,
                database.settings.enable_ai_streaming,
            ))
        })??;
    let system_prompt = command_system_prompt(&command_prompt);
    let input = format!("Conversation so far:\n{context}\n\nThe most recent reviewed command has completed. Use its tool result to either return a final answer or propose the next single safe command needed to satisfy the original request.");
    let api_key = provider_api_key(&profile.id)?;
    let app_for_stream = app.clone();
    let conversation_for_stream = conversation_id.clone();
    let app_for_thinking = app.clone();
    let conversation_for_thinking = conversation_id.clone();
    let mut plan = request_command_plan(
        &profile,
        api_key.as_deref(),
        &system_prompt,
        provider_messages,
        &input,
        enable_ai_streaming,
        move |text| {
            let _ = app_for_stream.emit(
                "command-stream",
                CommandStreamUpdate {
                    conversation_id: conversation_for_stream.clone(),
                    text: Some(text.to_owned()),
                    thinking: None,
                },
            );
        },
        move |thinking| {
            let _ = app_for_thinking.emit(
                "command-stream",
                CommandStreamUpdate {
                    conversation_id: conversation_for_thinking.clone(),
                    text: None,
                    thinking: Some(thinking.to_owned()),
                },
            );
        },
    )
    .await?;
    plan.conversation_id = Some(conversation_id.clone());
    state.update(|database| {
        if let Some(chat) = database
            .command_chats
            .iter_mut()
            .find(|chat| chat.id == conversation_id)
        {
            append_command_plan(chat, &plan);
        }
    })?;
    Ok(plan)
}

#[tauri::command]
fn cancel_command_plan(
    state: State<'_, AppState>,
    conversation_id: Option<String>,
    tool_call_id: Option<String>,
) -> Result<(), String> {
    let conversation_id = conversation_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned);
    let tool_call_id = tool_call_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned);
    state.update(|database| {
        let conversation_id = conversation_id
            .as_deref()
            .or(database.active_command_chat_id.as_deref());
        if let Some(conversation_id) = conversation_id {
            if let Some(chat) = database
                .command_chats
                .iter_mut()
                .find(|chat| chat.id == conversation_id)
            {
                let tool_call_id = tool_call_id.or_else(|| {
                    chat.messages.iter().rev().find_map(|message| {
                        (message.role == CommandChatRole::Assistant)
                            .then_some(message.tool_call.as_ref())
                            .flatten()
                            .map(|tool_call| tool_call.id.clone())
                    })
                });
                if let Some(tool_call_id) = tool_call_id {
                    chat.append_with_tool_metadata(
                        CommandChatRole::Tool,
                        r#"{"success":false,"command":"","output":"","error":"Command cancelled by user.","exitCode":-1,"executionTimeMs":0}"#,
                        None,
                        Some(tool_call_id),
                        None,
                    );
                }
                chat.append(CommandChatRole::Assistant, "Command cancelled.");
            }
        }
    })
}

#[tauri::command]
async fn execute_approved_command(
    app: AppHandle,
    state: State<'_, AppState>,
    plan: CommandPlan,
) -> Result<CommandExecutionResult, String> {
    if plan.kind != "command" {
        return Err("Only a reviewed command plan can be executed".into());
    }
    let command = plan
        .command
        .as_deref()
        .map(str::trim)
        .filter(|command| !command.is_empty())
        .ok_or("The reviewed command is empty")?
        .to_owned();
    if command.len() > 16_384 || command.contains('\0') {
        return Err("The reviewed command is not valid".into());
    }
    let conversation_id = plan.conversation_id.clone();
    let tool_call_id = plan.tool_call_id.clone();
    let result = execute_shell_command(command, plan.working_directory).await?;
    if let Some(conversation_id) = conversation_id {
        let summary = if result.success {
            format!(
                "Command completed (exit {}):\n{}",
                result.exit_code, result.output
            )
        } else {
            format!(
                "Command failed (exit {}):\n{}",
                result.exit_code,
                result.error.as_deref().unwrap_or(&result.output)
            )
        };
        state.update(|database| {
            if let Some(chat) = database
                .command_chats
                .iter_mut()
                .find(|chat| chat.id == conversation_id)
            {
                chat.append_with_tool_metadata(
                    CommandChatRole::Tool,
                    summary,
                    None,
                    tool_call_id,
                    None,
                );
            }
        })?;
    }
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    app.state::<analytics::AnalyticsService>().capture(
        "command_mode_run_completed",
        settings.share_anonymous_analytics,
        analytics_properties_with(
            &settings,
            [
                ("success", Value::Bool(result.success)),
                ("execution", Value::Bool(true)),
                (
                    "execution_time_bucket",
                    Value::String(
                        analytics::milliseconds_bucket(result.execution_time_ms as u64).into(),
                    ),
                ),
            ],
        ),
    );
    Ok(result)
}

fn parse_command_plan(response: &str) -> Result<CommandPlan, String> {
    let mut json = response.trim();
    if let Some(stripped) = json
        .strip_prefix("```json")
        .or_else(|| json.strip_prefix("```"))
    {
        json = stripped.trim();
    }
    if let Some(stripped) = json.strip_suffix("```") {
        json = stripped.trim();
    }
    let mut plan: CommandPlan = serde_json::from_str(json).map_err(|_| {
        "The AI provider did not return a command plan. Try a model that follows JSON instructions.".to_string()
    })?;
    match plan.kind.as_str() {
        "answer" => {
            if plan
                .answer
                .as_deref()
                .map_or(true, |answer| answer.trim().is_empty())
            {
                return Err("The AI provider returned an empty answer".into());
            }
        }
        "command" => {
            let command = plan
                .command
                .as_deref()
                .map(str::trim)
                .filter(|command| !command.is_empty())
                .ok_or("The AI provider returned an empty command")?;
            if command.len() > 16_384 || command.contains('\0') {
                return Err("The AI provider returned an invalid command".into());
            }
            plan.destructive = is_destructive_command(command);
        }
        _ => return Err("The AI provider returned an unsupported command plan".into()),
    }
    Ok(plan)
}

fn is_destructive_command(command: &str) -> bool {
    let command = command.to_ascii_lowercase();
    // Keep the reference's prefix and shell-composition checks. These are
    // deliberately conservative: a move, privilege escalation, process kill,
    // permission mutation, or filesystem overwrite needs the same review as a
    // deletion before Command Mode executes it.
    if [
        "rm ",
        "rm\t",
        "rmdir ",
        "rm -",
        "mv ",
        "mv\t",
        "sudo ",
        "kill ",
        "pkill ",
        "killall ",
        "chmod ",
        "chown ",
        "chgrp ",
        "dd ",
        "mkfs",
        "format",
        "> ",
        "truncate ",
        "shred ",
    ]
    .iter()
    .any(|prefix| command.starts_with(prefix))
        || [
            "| rm ", "| sudo ", "| dd ", "; rm ", "; sudo ", "&& rm ", "&& sudo ", "xargs rm",
            "xargs -i",
        ]
        .iter()
        .any(|pattern| command.contains(pattern))
        || command.contains("rm -")
    {
        return true;
    }

    // Preserve the Tauri port's extra cross-platform safeguards for Windows,
    // PowerShell, database, repository, and system-management commands.
    [
        "rm ",
        "rmdir ",
        "del ",
        "erase ",
        "remove-item",
        "format ",
        "mkfs",
        "dd ",
        "drop table",
        "drop database",
        "truncate table",
        "git reset --hard",
        "git clean -f",
        "shutdown",
        "reboot",
        "kill -9",
        "sudo ",
        "chmod -r",
        "chown -r",
    ]
    .iter()
    .any(|pattern| command.contains(pattern))
}

async fn execute_shell_command(
    command: String,
    working_directory: Option<String>,
) -> Result<CommandExecutionResult, String> {
    let started = Instant::now();
    let mut process = command_shell_process(&command);
    if let Some(directory) = working_directory
        .as_deref()
        .map(str::trim)
        .filter(|directory| !directory.is_empty())
    {
        let path = PathBuf::from(directory);
        if !path.is_dir() {
            return Err(format!(
                "The requested working directory does not exist: {directory}"
            ));
        }
        process.current_dir(path);
    } else if let Some(home) = BaseDirs::new().map(|directories| directories.home_dir().to_owned())
    {
        process.current_dir(home);
    }
    process.kill_on_drop(true);
    let output = match tokio::time::timeout(Duration::from_secs(30), process.output()).await {
        Ok(result) => {
            result.map_err(|error| format!("Could not start the reviewed command: {error}"))?
        }
        Err(_) => {
            return Ok(CommandExecutionResult {
                success: false,
                command,
                output: String::new(),
                error: Some("Command timed out after 30 seconds and was stopped.".into()),
                exit_code: -1,
                execution_time_ms: started.elapsed().as_millis(),
            })
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    Ok(CommandExecutionResult {
        success: output.status.success(),
        command,
        output: stdout,
        error: (!stderr.is_empty()).then_some(stderr),
        exit_code: output.status.code().unwrap_or(-1),
        execution_time_ms: started.elapsed().as_millis(),
    })
}

#[cfg(target_os = "windows")]
fn command_shell_process(command: &str) -> tokio::process::Command {
    let mut process = tokio::process::Command::new("cmd");
    process.args(["/C", command]);
    process
}

#[cfg(target_os = "macos")]
fn command_shell_process(command: &str) -> tokio::process::Command {
    let mut process = tokio::process::Command::new("/bin/zsh");
    process.args(["-c", command]);
    if let Some(path) = std::env::var_os("PATH") {
        let mut expanded_path = std::ffi::OsString::from("/opt/homebrew/bin:/usr/local/bin:");
        expanded_path.push(path);
        process.env("PATH", expanded_path);
    }
    process
}

#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
fn command_shell_process(command: &str) -> tokio::process::Command {
    let mut process = tokio::process::Command::new("/bin/sh");
    process.args(["-lc", command]);
    process
}

#[tauri::command]
fn capture_selected_text(app: AppHandle) -> Result<CapturedSelection, String> {
    let window = app
        .get_webview_window("main")
        .ok_or("Voxide main window is unavailable")?;
    window
        .hide()
        .map_err(|error| format!("Could not return focus to the source application: {error}"))?;
    std::thread::sleep(Duration::from_millis(120));
    let (source_application, _) = active_application_context();
    let text = typing::capture_selected_text()?;
    Ok(CapturedSelection {
        text,
        source_application,
    })
}

#[tauri::command]
fn replace_selected_text(
    app: AppHandle,
    state: State<'_, AppState>,
    text: String,
) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("There is no rewritten text to insert".into());
    }
    let window = app
        .get_webview_window("main")
        .ok_or("Voxide main window is unavailable")?;
    window
        .hide()
        .map_err(|error| format!("Could not return focus to the source application: {error}"))?;
    std::thread::sleep(Duration::from_millis(120));
    let insertion_mode = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .text_insertion_mode;
    typing::type_into_active_application(&text, insertion_mode)
}

fn apply_dictionary(text: &str, dictionary: &[DictionaryEntry]) -> String {
    let mut output = text.to_owned();
    let mut patterns = dictionary
        .iter()
        .filter_map(|entry| {
            let spoken = entry.spoken.trim();
            (!spoken.is_empty()).then(|| {
                let pattern = dictionary_regex_pattern(spoken);
                let regex = regex::RegexBuilder::new(&pattern)
                    .case_insensitive(true)
                    .build()
                    .ok()?;
                Some((
                    pattern.encode_utf16().count(),
                    regex,
                    entry.replacement.as_str(),
                ))
            })?
        })
        .collect::<Vec<_>>();
    // The macOS implementation applies the longest escaped regex patterns first, so a
    // phrase such as "vox side" wins over a shorter overlapping "vox" trigger.
    patterns.sort_by(|left, right| right.0.cmp(&left.0));

    for (_, regex, replacement) in patterns {
        output = replace_dictionary_matches(&output, &regex, replacement);
    }
    output
}

pub(crate) fn normalize_dictionary_entries(entries: Vec<DictionaryEntry>) -> Vec<DictionaryEntry> {
    entries
        .into_iter()
        .filter_map(|mut entry| {
            entry.spoken = entry.spoken.trim().to_lowercase();
            entry.replacement = entry.replacement.trim().to_owned();
            (!entry.spoken.is_empty() && !entry.replacement.is_empty()).then_some(entry)
        })
        .collect()
}

fn dictionary_regex_pattern(trigger: &str) -> String {
    let mut escaped = String::with_capacity(trigger.len());
    for character in trigger.chars() {
        if matches!(
            character,
            '\\' | '^' | '$' | '.' | '|' | '?' | '*' | '+' | '(' | ')' | '[' | ']' | '{' | '}'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn dictionary_is_word_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

fn dictionary_match_has_word_boundaries(text: &str, start: usize, end: usize) -> bool {
    let starts_at_word_boundary = start == 0
        || !text[..start]
            .chars()
            .next_back()
            .is_some_and(dictionary_is_word_character);
    let ends_at_word_boundary = end == text.len()
        || !text[end..]
            .chars()
            .next()
            .is_some_and(dictionary_is_word_character);
    starts_at_word_boundary && ends_at_word_boundary
}

fn replace_dictionary_matches(text: &str, regex: &regex::Regex, replacement: &str) -> String {
    let mut accepted_ranges = Vec::new();
    let mut search_start = 0;
    while let Some(found) = regex.find_at(text, search_start) {
        if dictionary_match_has_word_boundaries(text, found.start(), found.end()) {
            accepted_ranges.push((found.start(), found.end()));
            search_start = found.end();
        } else {
            // Regex iterators deliberately skip overlapping candidates. Advance by one
            // scalar after a rejected candidate so a later overlapping word-boundary
            // match remains eligible, matching NSRegularExpression's behavior.
            search_start = found.start()
                + text[found.start()..]
                    .chars()
                    .next()
                    .expect("dictionary regex matches are never empty")
                    .len_utf8();
        }
    }
    if accepted_ranges.is_empty() {
        return text.to_owned();
    }

    let mut output = String::with_capacity(text.len() + replacement.len() * accepted_ranges.len());
    let mut cursor = 0;
    for (start, end) in accepted_ranges {
        output.push_str(&text[cursor..start]);
        output.push_str(replacement);
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    output
}

fn output_formatting(settings: &Settings) -> formatting::OutputFormatting<'_> {
    formatting::OutputFormatting {
        remove_filler_words: settings.remove_filler_words_enabled,
        filler_words: &settings.filler_words,
        auto_convert_punctuation: settings.auto_convert_punctuation_enabled,
        punctuation_prefix: &settings.punctuation_dictionary_prefix,
        punctuation_rules: &settings.punctuation_dictionary_rules,
        literal_dictation_formatting: settings.literal_dictation_formatting_enabled,
        lowercase_first_letter: settings.gaav_lowercase_first_letter_enabled,
        remove_trailing_period: settings.gaav_remove_trailing_period_enabled,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DictationCleanupStyle {
    Standard,
    FluidVoiceParakeet,
}

fn prepare_dictation_text(
    raw_text: &str,
    formatting: &formatting::OutputFormatting<'_>,
    dictionary: &[DictionaryEntry],
    cleanup_style: DictationCleanupStyle,
) -> String {
    match cleanup_style {
        DictationCleanupStyle::Standard => {
            let preformatted = formatting::apply_before_ai(raw_text, formatting);
            apply_dictionary(&preformatted, dictionary)
        }
        // FluidVoice runs its preview and final path as:
        // filler removal -> custom dictionary -> spoken punctuation. Keep that
        // order for Parakeet so a dictionary replacement can intentionally
        // contain a spoken punctuation command.
        DictationCleanupStyle::FluidVoiceParakeet => {
            let without_fillers = if formatting.remove_filler_words {
                formatting::remove_filler_words(raw_text, formatting.filler_words)
            } else {
                raw_text.to_owned()
            };
            let dictionary_corrected = apply_dictionary(&without_fillers, dictionary);
            if formatting.auto_convert_punctuation {
                formatting::apply_spoken_punctuation(
                    &dictionary_corrected,
                    formatting.punctuation_prefix,
                    formatting.punctuation_rules,
                )
            } else {
                dictionary_corrected
            }
        }
    }
}

fn deterministic_dictation_cleanup(
    state: &AppState,
    raw_text: &str,
    cleanup_style: DictationCleanupStyle,
) -> Result<String, String> {
    let database = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?;
    let formatting = output_formatting(&database.settings);
    let dictionary_corrected =
        prepare_dictation_text(raw_text, &formatting, &database.dictionary, cleanup_style);
    Ok(formatting::apply_final_output(
        &dictionary_corrected,
        &formatting,
    ))
}

#[derive(Debug)]
struct PostProcessOutcome {
    text: String,
    ai_fallback_error: Option<String>,
    was_ai_processed: bool,
    processing_model: Option<String>,
}

fn effective_dictation_system_prompt(
    profile: &PromptProfile,
    prompt_override: Option<&str>,
) -> String {
    prompt_override
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| profile.prompt.clone())
}

async fn post_process_dictation_outcome(
    state: &AppState,
    raw_text: String,
    force_prompt_processing: bool,
    source_application: Option<&str>,
    source_window_title: Option<&str>,
    prompt_profile_id: Option<&str>,
    prompt_override: Option<&str>,
    cleanup_style: DictationCleanupStyle,
    on_delta: &mut (dyn FnMut(&str) + Send),
) -> Result<PostProcessOutcome, String> {
    let (dictionary, settings, profile, provider) = {
        let database = state
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        let profile = prompt_for_mode_with_override(
            &database,
            DictationMode::Dictate,
            prompt_profile_id,
            source_application,
        );
        let provider = dictation_provider_for_prompt_profile(&database, &profile.id)?;
        (
            database.dictionary.clone(),
            database.settings.clone(),
            profile,
            provider,
        )
    };
    // Prompt Test Mode intentionally operates on the editor's unsaved draft.
    // Keep the persisted profile for routing/provider selection, but use the
    // non-empty draft as the system prompt for this one isolated run.
    let system_prompt = effective_dictation_system_prompt(&profile, prompt_override);
    let formatting = output_formatting(&settings);
    let dictionary_corrected =
        prepare_dictation_text(&raw_text, &formatting, &dictionary, cleanup_style);
    let (processed, ai_fallback_error, was_ai_processed, processing_model) = if !settings
        .ai_enhancement_enabled
        && !force_prompt_processing
    {
        (dictionary_corrected, None, false, None)
    } else {
        let enhanced = match provider_api_key(&provider.id) {
            Ok(api_key) => {
                let streaming = provider::process_streaming_with_options(
                    &provider,
                    api_key.as_deref(),
                    &system_prompt,
                    &dictionary_corrected,
                    Duration::from_secs(120),
                    0.2,
                    None,
                    &mut *on_delta,
                )
                .await;
                match streaming {
                    Ok(text) => Ok(text),
                    Err(streaming_error) => {
                        // Voxide shows live refinement when the provider
                        // supports SSE, then retries the same request as a
                        // complete reply for providers/proxies that do not.
                        provider::process_with_options_timeout(
                                &provider,
                                api_key.as_deref(),
                                &system_prompt,
                                &dictionary_corrected,
                                Duration::from_secs(120),
                                0.2,
                                None,
                            )
                            .await
                            .map_err(|fallback_error| {
                                format!(
                                    "Streaming refinement failed ({streaming_error}); complete-reply retry failed: {fallback_error}"
                                )
                            })
                    }
                }
            }
            Err(error) => Err(error),
        };
        match enhanced {
            Ok(text) => (text, None, true, Some(provider.model.clone())),
            Err(error) => (dictionary_corrected, Some(error), false, None),
        }
    };
    Ok(PostProcessOutcome {
        text: formatting::apply_final_output_with_context(
            &processed,
            &formatting,
            source_application,
            source_window_title,
        ),
        ai_fallback_error,
        was_ai_processed,
        processing_model,
    })
}

/// Shows a desktop notification without panicking the async runtime.
///
/// On Linux the notification plugin spawns its `show` future onto the Tauri
/// Tokio runtime, where `notify-rust` (with the workspace's zbus/tokio
/// feature unification) constructs a nested blocking runtime and aborts with
/// "Cannot start a runtime from within a runtime". Calling `notify-rust`
/// directly from a dedicated plain thread keeps it off every runtime worker.
fn show_desktop_notification(app: &AppHandle, title: &str, body: &str) {
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        let title = title.to_owned();
        let body = body.to_owned();
        std::thread::spawn(move || {
            let _ = notify_rust::Notification::new()
                .appname("Voxide")
                .summary(&title)
                .body(&body)
                .icon("voxide")
                .show();
        });
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = app.notification().builder().title(title).body(body).show();
    }
}

/// Announces an available update. On Linux the notification carries an
/// interactive "Open release page" action (the reference app's update alert
/// is actionable); other platforms use the plain notification path until
/// their native action APIs are wired up.
fn notify_update_available(app: &AppHandle, version: &str, release_url: &str) {
    let body = format!("Version {version} is ready. Open Voxide to review it.");
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        let release_url = release_url.to_owned();
        std::thread::spawn(move || {
            let shown = notify_rust::Notification::new()
                .appname("Voxide")
                .summary("Voxide update available")
                .body(&body)
                .icon("voxide")
                .action("open-release", "Open release page")
                .show();
            if let Ok(handle) = shown {
                handle.wait_for_action(|action| {
                    if action == "open-release" && update::is_release_url(&release_url) {
                        let _ = tauri_plugin_opener::open_url(&release_url, None::<&str>);
                    }
                });
            }
        });
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = release_url;
        show_desktop_notification(app, "Voxide update available", &body);
    }
}

fn notify_ai_fallback(app: &AppHandle, error: &str) {
    show_desktop_notification(
        app,
        "AI Enhancement failed",
        "Typed raw transcription instead.",
    );
    eprintln!("Voxide AI enhancement fallback: {error}");
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DailyUsage {
    date: String,
    words: usize,
    transcriptions: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UsageTopApp {
    app: String,
    count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UsageMilestone {
    target: usize,
    achieved: bool,
    label: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UsageStats {
    today_dictations: usize,
    today_words: usize,
    today_time_saved_minutes: f64,
    total_dictations: usize,
    total_words: usize,
    total_characters: usize,
    total_time_saved_minutes: f64,
    average_words_per_dictation: usize,
    ai_processed_count: usize,
    ai_enhancement_rate: usize,
    current_streak: usize,
    best_streak: usize,
    daily_activity_7: Vec<DailyUsage>,
    daily_activity_30: Vec<DailyUsage>,
    top_apps: Vec<UsageTopApp>,
    peak_hour: Option<u32>,
    longest_transcription_words: usize,
    most_words_in_day: usize,
    most_transcriptions_in_day: usize,
    word_milestones: Vec<UsageMilestone>,
    transcription_milestones: Vec<UsageMilestone>,
    streak_milestones: Vec<UsageMilestone>,
}

fn dictation_word_count(entry: &DictationEntry) -> usize {
    entry.text.split_whitespace().count()
}

fn local_usage_day(timestamp: DateTime<Utc>) -> NaiveDate {
    timestamp.with_timezone(&Local).date_naive()
}

fn is_weekend(day: NaiveDate) -> bool {
    matches!(day.weekday(), Weekday::Sat | Weekday::Sun)
}

fn previous_usage_day(day: NaiveDate, skip_weekends: bool) -> NaiveDate {
    let mut previous = day
        .checked_sub_signed(chrono::Duration::days(1))
        .unwrap_or(day);
    while skip_weekends && is_weekend(previous) {
        previous = previous
            .checked_sub_signed(chrono::Duration::days(1))
            .unwrap_or(previous);
    }
    previous
}

fn current_usage_day(mut today: NaiveDate, skip_weekends: bool) -> NaiveDate {
    while skip_weekends && is_weekend(today) {
        today = previous_usage_day(today, skip_weekends);
    }
    today
}

fn current_usage_streak(
    active_days: &HashSet<NaiveDate>,
    today: NaiveDate,
    skip_weekends: bool,
) -> usize {
    let today = current_usage_day(today, skip_weekends);
    let start = if active_days.contains(&today) {
        today
    } else {
        let previous = previous_usage_day(today, skip_weekends);
        if active_days.contains(&previous) {
            previous
        } else {
            return 0;
        }
    };
    let mut streak = 0;
    let mut day = start;
    while active_days.contains(&day) {
        streak += 1;
        day = previous_usage_day(day, skip_weekends);
    }
    streak
}

fn best_usage_streak(active_days: &HashSet<NaiveDate>, skip_weekends: bool) -> usize {
    let mut days = active_days.iter().copied().collect::<Vec<_>>();
    days.sort_unstable();
    let mut best = 0;
    let mut streak = 0;
    let mut previous = None;
    for day in days {
        if previous.is_some_and(|previous| previous == previous_usage_day(day, skip_weekends)) {
            streak += 1;
        } else {
            streak = 1;
        }
        best = best.max(streak);
        previous = Some(day);
    }
    best
}

fn time_saved_minutes(words: usize, typing_wpm: u16) -> f64 {
    let typing_wpm = typing_wpm.clamp(1, 200) as f64;
    let words = words as f64;
    (words / typing_wpm - words / 150.0).max(0.0)
}

fn usage_milestones(total: usize, milestones: &[(usize, &str)]) -> Vec<UsageMilestone> {
    milestones
        .iter()
        .map(|(target, label)| UsageMilestone {
            target: *target,
            achieved: total >= *target,
            label: (*label).into(),
        })
        .collect()
}

fn daily_usage(
    day_totals: &HashMap<NaiveDate, (usize, usize)>,
    today: NaiveDate,
    days: usize,
) -> Vec<DailyUsage> {
    (0..days)
        .rev()
        .filter_map(|offset| today.checked_sub_signed(chrono::Duration::days(offset as i64)))
        .map(|day| {
            let (words, transcriptions) = day_totals.get(&day).copied().unwrap_or_default();
            DailyUsage {
                date: day.to_string(),
                words,
                transcriptions,
            }
        })
        .collect()
}

fn calculate_usage_stats(
    entries: &[DictationEntry],
    settings: &Settings,
    now: DateTime<Local>,
) -> UsageStats {
    let today = now.date_naive();
    let mut today_dictations = 0;
    let mut today_words = 0;
    let mut total_words = 0;
    let mut total_characters = 0;
    let mut ai_processed_count = 0;
    let mut day_totals = HashMap::<NaiveDate, (usize, usize)>::new();
    let mut app_counts = HashMap::<String, usize>::new();
    let mut hour_counts = HashMap::<u32, usize>::new();

    for entry in entries {
        let words = dictation_word_count(entry);
        let day = local_usage_day(entry.created_at);
        total_words += words;
        total_characters += entry.text.chars().count();
        ai_processed_count += usize::from(entry.was_ai_processed);
        let day_total = day_totals.entry(day).or_default();
        day_total.0 += words;
        day_total.1 += 1;
        if day == today {
            today_dictations += 1;
            today_words += words;
        }
        let app = entry
            .source_application
            .as_deref()
            .filter(|app| !app.trim().is_empty())
            .unwrap_or("Unknown")
            .to_owned();
        *app_counts.entry(app).or_default() += 1;
        *hour_counts
            .entry(entry.created_at.with_timezone(&Local).hour())
            .or_default() += 1;
    }

    let active_days = day_totals.keys().copied().collect::<HashSet<_>>();
    let mut top_apps = app_counts
        .into_iter()
        .map(|(app, count)| UsageTopApp { app, count })
        .collect::<Vec<_>>();
    top_apps.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.app.cmp(&right.app))
    });
    top_apps.truncate(5);
    let peak_hour = hour_counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(hour, _)| hour);
    let current_streak =
        current_usage_streak(&active_days, today, settings.weekends_dont_break_streak);
    let best_streak = best_usage_streak(&active_days, settings.weekends_dont_break_streak);
    let total_dictations = entries.len();

    UsageStats {
        today_dictations,
        today_words,
        today_time_saved_minutes: time_saved_minutes(today_words, settings.user_typing_wpm),
        total_dictations,
        total_words,
        total_characters,
        total_time_saved_minutes: time_saved_minutes(total_words, settings.user_typing_wpm),
        average_words_per_dictation: if total_dictations == 0 {
            0
        } else {
            total_words / total_dictations
        },
        ai_processed_count,
        ai_enhancement_rate: if total_dictations == 0 {
            0
        } else {
            ai_processed_count * 100 / total_dictations
        },
        current_streak,
        best_streak,
        daily_activity_7: daily_usage(&day_totals, today, 7),
        daily_activity_30: daily_usage(&day_totals, today, 30),
        top_apps,
        peak_hour,
        longest_transcription_words: entries
            .iter()
            .map(dictation_word_count)
            .max()
            .unwrap_or_default(),
        most_words_in_day: day_totals
            .values()
            .map(|(words, _)| *words)
            .max()
            .unwrap_or_default(),
        most_transcriptions_in_day: day_totals
            .values()
            .map(|(_, transcriptions)| *transcriptions)
            .max()
            .unwrap_or_default(),
        word_milestones: usage_milestones(
            total_words,
            &[
                (1_000, "1K"),
                (10_000, "10K"),
                (50_000, "50K"),
                (100_000, "100K"),
                (500_000, "500K"),
                (1_000_000, "1M"),
            ],
        ),
        transcription_milestones: usage_milestones(
            total_dictations,
            &[
                (50, "50"),
                (100, "100"),
                (500, "500"),
                (1_000, "1K"),
                (5_000, "5K"),
                (10_000, "10K"),
            ],
        ),
        streak_milestones: usage_milestones(
            best_streak,
            &[
                (7, "7 days"),
                (14, "14 days"),
                (30, "30 days"),
                (60, "60 days"),
                (100, "100 days"),
                (365, "1 year"),
            ],
        ),
    }
}

#[tauri::command]
fn usage_stats(state: State<'_, AppState>) -> Result<UsageStats, String> {
    let database = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?;
    Ok(calculate_usage_stats(
        &database.dictation_history,
        &database.settings,
        Local::now(),
    ))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NativeCaptureStarted {
    sample_rate: u32,
    channels: u16,
    source_application: Option<String>,
    source_window_title: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CapturedSelection {
    text: String,
    source_application: Option<String>,
}

fn active_application_context() -> (Option<String>, Option<String>) {
    let Ok(window) = active_win_pos_rs::get_active_window() else {
        return (None, None);
    };
    let application = if window.app_name.trim().is_empty() {
        window
            .process_path
            .file_stem()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
    } else {
        Some(window.app_name)
    };
    let title = (!window.title.trim().is_empty()).then_some(window.title);
    (application, title)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NativeTranscriptionResult {
    text: String,
    raw_text: String,
    duration_ms: u64,
    audio_file: Option<String>,
    audio_model: Option<String>,
    was_ai_processed: bool,
    processing_model: Option<String>,
    ai_processing_error: Option<String>,
    source_application: Option<String>,
    source_window_title: Option<String>,
    inserted_into_active_application: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AudioHistoryManifestRow {
    audio: String,
    text: String,
    raw_transcript: String,
    final_transcript: String,
    timestamp: String,
    duration_milliseconds: u64,
    sample_rate: u32,
    channels: u16,
    app: String,
    model: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct OverlayUpdate {
    state: String,
    mode: String,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureFailure {
    session_id: u64,
}

/// Hides the dictation overlay when dropped. Held across the fallible span
/// of stop_native_dictation so no error path can leave the overlay stuck on
/// "Transcribing…"; the extra hide after a successful run is a no-op.
struct OverlayHideOnDrop(AppHandle);

impl Drop for OverlayHideOnDrop {
    fn drop(&mut self) {
        emit_overlay(&self.0, "hidden", "");
    }
}

fn emit_overlay(app: &AppHandle, state: &str, text: impl Into<String>) {
    let _ = app.emit(
        "overlay-update",
        OverlayUpdate {
            state: state.into(),
            mode: "dictate".into(),
            text: text.into(),
        },
    );
}

fn overlay_window_dimensions(size: OverlaySize) -> (u32, u32) {
    match size {
        OverlaySize::Pill => (100, 46),
        OverlaySize::Small => (300, 124),
        OverlaySize::Medium => (380, 156),
        OverlaySize::Large => (600, 288),
    }
}

/// The tallest the overlay may grow when its transcript content needs more
/// room. The pill layout intentionally never grows.
fn overlay_maximum_height(size: OverlaySize) -> u32 {
    let (_, base_height) = overlay_window_dimensions(size);
    match size {
        OverlaySize::Pill => base_height,
        _ => base_height * 2,
    }
}

/// Mirrors the reference overlay's centered, work-area-aware placement while
/// using Tauri's physical desktop coordinates on every supported platform.
fn apply_overlay_window_layout(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    layout_overlay_window(app, settings, None)
}

/// Sizes and anchors the overlay. `content_height` (physical pixels) grows
/// the window to fit the live transcript, bounded between the configured
/// overlay size and its growth cap, while keeping the configured top/bottom
/// anchor fixed.
fn layout_overlay_window(
    app: &AppHandle,
    settings: &Settings,
    content_height: Option<u32>,
) -> Result<(), String> {
    let Some(window) = app.get_webview_window("overlay") else {
        return Ok(());
    };
    let (width, base_height) = overlay_window_dimensions(settings.overlay_size);
    let height = content_height
        .map(|requested| {
            requested.clamp(base_height, overlay_maximum_height(settings.overlay_size))
        })
        .unwrap_or(base_height);
    window
        .set_size(PhysicalSize::new(width, height))
        .map_err(|error| format!("Could not resize the dictation overlay: {error}"))?;
    let monitor = window
        .current_monitor()
        .or_else(|_| window.primary_monitor())
        .map_err(|error| format!("Could not determine the dictation-overlay monitor: {error}"))?
        .or_else(|| window.primary_monitor().ok().flatten());
    let Some(monitor) = monitor else {
        return Ok(());
    };
    let work_area = monitor.work_area();
    let horizontal_space = work_area.size.width.saturating_sub(width);
    let x = work_area.position.x + (horizontal_space / 2) as i32;
    let offset = (settings.overlay_bottom_offset * monitor.scale_factor()).round() as i32;
    let minimum_y = work_area.position.y.saturating_add(10);
    let maximum_y = work_area
        .position
        .y
        .saturating_add(work_area.size.height.saturating_sub(height) as i32)
        .saturating_sub(40);
    let raw_y = match settings.overlay_position {
        OverlayPosition::Top => work_area.position.y.saturating_add(offset),
        OverlayPosition::Bottom => work_area
            .position
            .y
            .saturating_add(work_area.size.height.saturating_sub(height) as i32)
            .saturating_sub(offset),
    };
    let y = raw_y.clamp(minimum_y, maximum_y.max(minimum_y));
    window
        .set_position(PhysicalPosition::new(x, y))
        .map_err(|error| format!("Could not position the dictation overlay: {error}"))
}

/// Grows or shrinks the overlay window so the live transcript fits, within
/// the configured overlay size's bounds. Called by the overlay webview after
/// it measures its rendered content.
#[tauri::command]
fn resize_overlay_to_content(
    app: AppHandle,
    state: State<'_, AppState>,
    content_height: u32,
) -> Result<(), String> {
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    layout_overlay_window(&app, &settings, Some(content_height))
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VoiceModelStatus {
    id: String,
    installed: bool,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_installed: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VoiceEngineAvailability {
    engines: Vec<VoiceEngineDescriptor>,
}

/// Runtime state paired with the static ASR contract. `available` only means
/// this build/platform can run the engine; model and credential readiness are
/// reported separately by `voice_model_status`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VoiceEngineDescriptor {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    maturity: asr::EngineMaturity,
    preview_mode: asr::PreviewMode,
    final_mode: asr::FinalMode,
    supports_files: bool,
    supports_translation: bool,
    supports_vocabulary: bool,
    requires_cuda: bool,
    available: bool,
    unavailable_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AccessibilityPermissionStatus {
    /// Only macOS exposes a portable process-level accessibility trust check.
    supported: bool,
    trusted: Option<bool>,
    guidance: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelDownloadProgress {
    id: String,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
}

fn whisper_model_filename(model_id: &str) -> Result<&'static str, String> {
    match model_id {
        "tiny" => Ok("ggml-tiny.bin"),
        "base" => Ok("ggml-base.bin"),
        "small" => Ok("ggml-small.bin"),
        "medium" => Ok("ggml-medium.bin"),
        "tiny.en" => Ok("ggml-tiny.en.bin"),
        "base.en" => Ok("ggml-base.en.bin"),
        "small.en" => Ok("ggml-small.en.bin"),
        "medium.en" => Ok("ggml-medium.en.bin"),
        "large-v3-turbo" => Ok("ggml-large-v3-turbo.bin"),
        "large-v3-turbo-q5_0" => Ok("ggml-large-v3-turbo-q5_0.bin"),
        "large-v3-turbo-q8_0" => Ok("ggml-large-v3-turbo-q8_0.bin"),
        "large-v3" => Ok("ggml-large-v3.bin"),
        _ => Err(format!("Unsupported built-in Whisper model: {model_id}")),
    }
}

fn looks_like_markup(bytes: &[u8]) -> bool {
    let mut bytes = bytes;
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        bytes = &bytes[3..];
    }
    while let Some(byte) = bytes.first() {
        if !matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
            break;
        }
        bytes = &bytes[1..];
    }
    if bytes.len() < 2 || bytes[0] != b'<' {
        return false;
    }
    let second = bytes[1];
    second.is_ascii_alphabetic() || matches!(second, b'!' | b'?' | b'/')
}

fn model_content_type_is_markup(content_type: Option<&reqwest::header::HeaderValue>) -> bool {
    content_type
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
        })
        .is_some_and(|value| {
            matches!(
                value.as_str(),
                "text/html" | "application/xhtml+xml" | "application/xml" | "text/xml"
            )
        })
}

fn validate_whisper_model_download(
    path: &std::path::Path,
    expected_bytes: Option<u64>,
    downloaded_bytes: u64,
) -> Result<(), String> {
    if downloaded_bytes == 0 {
        return Err("The model download was empty".into());
    }
    if expected_bytes.is_some_and(|expected| expected != downloaded_bytes) {
        return Err("The model download was incomplete".into());
    }
    let metadata = fs::metadata(path)
        .map_err(|error| format!("Could not verify the model download: {error}"))?;
    if !metadata.is_file() || metadata.len() != downloaded_bytes {
        return Err("The model download was not saved as a complete regular file".into());
    }
    let mut prefix = [0_u8; 512];
    let prefix_length = fs::File::open(path)
        .and_then(|mut file| file.read(&mut prefix))
        .map_err(|error| format!("Could not inspect the model download: {error}"))?;
    if looks_like_markup(&prefix[..prefix_length]) {
        return Err("The model download was an HTML or XML page, not Whisper model data. A network proxy or firewall may be blocking model downloads.".into());
    }
    Ok(())
}

fn valid_whisper_model_file(path: &std::path::Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() || metadata.len() == 0 {
        return false;
    }
    let mut prefix = [0_u8; 512];
    fs::File::open(path)
        .and_then(|mut file| file.read(&mut prefix))
        .map(|length| !looks_like_markup(&prefix[..length]))
        .unwrap_or(false)
}

/// Silero voice-activity-detection model used to strip non-speech audio
/// before Whisper decodes it. Small (~2 MB); fetched once at startup.
const VAD_MODEL_FILENAME: &str = "ggml-silero-v5.1.2.bin";
const VAD_MODEL_REVISION: &str = "e5614ed76a5dd4b03fad5068c89efcd2617a9d1e";
const VAD_MODEL_SHA256: &str = "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf";
const VAD_MODEL_BYTES: usize = 885_098;

fn vad_model_url() -> String {
    format!(
        "https://huggingface.co/ggml-org/whisper-vad/resolve/{VAD_MODEL_REVISION}/{VAD_MODEL_FILENAME}"
    )
}

fn vad_model_path(state: &AppState) -> Option<PathBuf> {
    let path = state.models_directory().ok()?.join(VAD_MODEL_FILENAME);
    path.is_file().then_some(path)
}

async fn ensure_vad_model(state: &AppState) {
    let Ok(directory) = state.models_directory() else {
        return;
    };
    let path = directory.join(VAD_MODEL_FILENAME);
    if vad_model_is_verified(&path) {
        return;
    }
    let downloaded = async {
        let response = reqwest::get(vad_model_url()).await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        let bytes = response.bytes().await.ok()?;
        if bytes.len() != VAD_MODEL_BYTES || looks_like_markup(&bytes) {
            return None;
        }
        if format!("{:x}", Sha256::digest(&bytes)) != VAD_MODEL_SHA256 {
            return None;
        }
        let temporary = path.with_extension("bin.tmp");
        fs::write(&temporary, &bytes).ok()?;
        fs::rename(&temporary, &path).ok()
    }
    .await;
    debug_log::append(&format!(
        "VAD model download {}",
        if downloaded.is_some() {
            "completed"
        } else {
            "failed; dictation continues without voice-activity detection"
        }
    ));
}

fn vad_model_is_verified(path: &Path) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.len() == VAD_MODEL_BYTES as u64)
        .unwrap_or(false)
        && sha256_file(path)
            .map(|digest| digest == VAD_MODEL_SHA256)
            .unwrap_or(false)
}

fn whisper_model_path(settings: &Settings, state: &AppState) -> Result<PathBuf, String> {
    if let Some(path) = settings
        .local_model_path
        .as_deref()
        .filter(|path| !path.trim().is_empty())
    {
        return Ok(PathBuf::from(path));
    }
    Ok(state
        .models_directory()?
        .join(whisper_model_filename(&settings.selected_model)?))
}

fn parakeet_model_path(state: &AppState) -> Result<PathBuf, String> {
    Ok(parakeet::model_directory(&state.models_directory()?))
}

fn nemotron_model_path(state: &AppState) -> Result<PathBuf, String> {
    Ok(nemotron::model_directory(&state.models_directory()?))
}

fn nemotron_runtime_path(state: &AppState) -> Result<PathBuf, String> {
    Ok(nemotron::runtime_directory(&state.data_directory()?))
}

fn nemotron_runtime_version(runtime_directory: &Path) -> Option<String> {
    let metadata = fs::read(runtime_directory.join(NEMOTRON_RUNTIME_HEALTH_FILE)).ok()?;
    let metadata: serde_json::Value = serde_json::from_slice(&metadata).ok()?;
    let torch = diagnostic_version_value(metadata.get("torch")?)?;
    let cuda = diagnostic_version_value(metadata.get("cuda")?)?;
    Some(format!(
        "PyTorch {torch}, CUDA {cuda}, model {}",
        &nemotron::MODEL_REVISION[..12]
    ))
}

fn diagnostic_version_value(value: &serde_json::Value) -> Option<&str> {
    let value = value.as_str()?;
    (value.len() <= 64
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '+' | '_')
        }))
    .then_some(value)
}

/// Materialize the embedded sidecar beside the user-local Python environment.
/// This lets an installed app run the exact server it was built with, while a
/// source checkout needs no fragile path assumptions in development.
fn ensure_nemotron_server_script(runtime_directory: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(runtime_directory)
        .map_err(|error| format!("Could not create the Nemotron runtime directory: {error}"))?;
    let script = runtime_directory.join("nemotron_cuda_server.py");
    let should_write = fs::read_to_string(&script)
        .map(|current| current != NEMOTRON_SERVER_SOURCE)
        .unwrap_or(true);
    if should_write {
        fs::write(&script, NEMOTRON_SERVER_SOURCE)
            .map_err(|error| format!("Could not install the Nemotron CUDA service: {error}"))?;
    }
    Ok(script)
}

fn nemotron_status(state: &AppState) -> Result<VoiceModelStatus, String> {
    let model = nemotron_model_path(state)?;
    let runtime = nemotron_runtime_path(state)?;
    let runtime_installed = nemotron_runtime_is_verified(&runtime);
    let model_installed = nemotron_model_is_verified(&model);
    let ready = nemotron::is_compiled() && runtime_installed && model_installed;
    let path = if !nemotron::is_compiled() {
        "Nemotron is included in Voxide's CUDA build for Linux/NVIDIA".into()
    } else if !runtime_installed {
        format!(
            "Install the user-local CUDA runtime at {} before downloading or using the model",
            runtime.display()
        )
    } else if !model_installed {
        format!("Download the Nemotron model to {}", model.display())
    } else {
        model.display().to_string()
    };
    Ok(VoiceModelStatus {
        id: nemotron::MODEL_ID.into(),
        installed: ready,
        path,
        runtime_installed: Some(runtime_installed),
    })
}

/// Decodes common desktop media formats into short WAV-sized Speech.framework
/// requests. Apple documents an approximately one-minute recognition limit,
/// so long files are chunked before they reach the native API.
pub(crate) fn transcribe_apple_media_file(
    path: &std::path::Path,
    language: &str,
    custom_words: &[String],
    progress: Option<speech::ProgressCallback>,
) -> Result<(String, u64), String> {
    const APPLE_SPEECH_CHUNK_SECONDS: f64 = 55.0;
    let duration_ms = media::file_duration_ms(path)?;
    let total_chunks = ((duration_ms as f64 / 1000.0) / APPLE_SPEECH_CHUNK_SECONDS)
        .ceil()
        .max(1.0) as usize;
    let mut transcriptions = Vec::with_capacity(total_chunks);
    for chunk in 0..total_chunks {
        let start_seconds = chunk as f64 * APPLE_SPEECH_CHUNK_SECONDS;
        let remaining_seconds = (duration_ms as f64 / 1000.0 - start_seconds).max(0.0);
        let audio = media::decode_audio_segment(
            path,
            start_seconds,
            remaining_seconds.min(APPLE_SPEECH_CHUNK_SECONDS),
        )?;
        let samples = audio::mono_resample_for_whisper(audio)?;
        if !audio::has_minimum_transcription_samples(&samples) {
            if let Some(progress) = &progress {
                progress(chunk + 1, total_chunks);
            }
            continue;
        }
        let text = apple_speech::transcribe_samples(&samples, language, custom_words)?;
        if !text.trim().is_empty() {
            transcriptions.push(text);
        }
        if let Some(progress) = &progress {
            progress(chunk + 1, total_chunks);
        }
    }
    let text = transcriptions.join(" ");
    Ok((text, duration_ms))
}

#[tauri::command]
fn voice_model_status(state: State<'_, AppState>) -> Result<VoiceModelStatus, String> {
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    match settings.selected_voice_engine {
        VoiceEngine::Cloud => match cloud_transcription_readiness(&settings, &state) {
            Ok(profile) => Ok(VoiceModelStatus {
                id: settings.cloud_transcription_model,
                installed: true,
                path: format!("Ready: {} cloud transcription", profile.name),
                runtime_installed: None,
            }),
            Err(error) => Ok(VoiceModelStatus {
                id: settings.cloud_transcription_model,
                installed: false,
                path: error,
                runtime_installed: None,
            }),
        },
        VoiceEngine::AppleSpeech => Ok(VoiceModelStatus {
            id: "apple-speech".into(),
            installed: apple_speech::is_supported(),
            path: if apple_speech::is_supported() {
                "macOS Speech.framework (permission is requested on first use)".into()
            } else {
                "Apple Speech is available only on macOS".into()
            },
            runtime_installed: None,
        }),
        VoiceEngine::Whisper => {
            let path = whisper_model_path(&settings, &state)?;
            Ok(VoiceModelStatus {
                id: settings.selected_model,
                installed: valid_whisper_model_file(&path),
                path: path.display().to_string(),
                runtime_installed: None,
            })
        }
        VoiceEngine::Parakeet => {
            let path = parakeet_model_path(&state)?;
            Ok(VoiceModelStatus {
                id: parakeet::MODEL_ID.into(),
                installed: parakeet::is_compiled() && parakeet_model_is_verified(&path),
                path: if parakeet::is_compiled() {
                    path.display().to_string()
                } else {
                    "Parakeet is included in the CUDA build".into()
                },
                runtime_installed: None,
            })
        }
        VoiceEngine::Nemotron => nemotron_status(&state),
    }
}

/// Rehash CUDA components only on explicit user request. Normal recording
/// readiness relies on the receipt written after install so a 2.6 GB model
/// never delays a hotkey; this command provides the deliberate repair check.
#[tauri::command]
async fn verify_voice_engine_installation(
    state: State<'_, AppState>,
) -> Result<VoiceModelStatus, String> {
    let settings = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .clone();
    match settings.selected_voice_engine {
        VoiceEngine::Parakeet => {
            if !parakeet::is_compiled() {
                return Err("Parakeet is available in Voxide's CUDA build".into());
            }
            let model = parakeet_model_path(&state)?;
            let verified = tauri::async_runtime::spawn_blocking(move || {
                parakeet::model_is_installed(&model)
                    && component_receipt_is_verified(
                        &model,
                        parakeet::MODEL_ID,
                        parakeet::MODEL_ARCHIVE_SHA256,
                    )
            })
            .await
            .map_err(|error| format!("Parakeet verification task failed: {error}"))?;
            if !verified {
                return Err(
                    "Parakeet verification failed. Download the model again to repair it.".into(),
                );
            }
        }
        VoiceEngine::Nemotron => {
            if !nemotron::is_compiled() {
                return Err("Nemotron is available in Voxide's CUDA build for Linux/NVIDIA".into());
            }
            let runtime = nemotron_runtime_path(&state)?;
            let model = nemotron_model_path(&state)?;
            let verified = tauri::async_runtime::spawn_blocking(move || {
                nemotron::runtime_is_installed(&runtime)
                    && component_receipt_is_verified(
                        &runtime,
                        NEMOTRON_RUNTIME_ID,
                        NEMOTRON_RUNTIME_VERSION,
                    )
                    && nemotron::model_is_installed(&model)
                    && component_receipt_is_verified(
                        &model,
                        nemotron::MODEL_ID,
                        nemotron::MODEL_REVISION,
                    )
            })
            .await
            .map_err(|error| format!("Nemotron verification task failed: {error}"))?;
            if !verified {
                return Err("Nemotron verification failed. Reinstall the CUDA runtime or download the model again to repair it.".into());
            }
        }
        VoiceEngine::Whisper | VoiceEngine::Cloud | VoiceEngine::AppleSpeech => {
            return voice_model_status(state);
        }
    }
    voice_model_status(state)
}

#[tauri::command]
fn voice_engine_availability() -> VoiceEngineAvailability {
    VoiceEngineAvailability {
        engines: VoiceEngine::ALL
            .into_iter()
            .map(VoiceEngine::descriptor)
            .collect(),
    }
}

#[tauri::command]
fn delete_whisper_model(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<VoiceModelStatus, String> {
    let filename = whisper_model_filename(model_id.trim())?;
    let path = state.models_directory()?.join(filename);
    if !path.exists() {
        return Err(format!(
            "The downloaded Whisper model is not installed: {model_id}"
        ));
    }
    if !path.is_file() {
        return Err("The downloaded Whisper model path is not a regular file".into());
    }
    fs::remove_file(&path)
        .map_err(|error| format!("Could not remove the Whisper model: {error}"))?;
    Ok(VoiceModelStatus {
        id: model_id,
        installed: false,
        path: path.display().to_string(),
        runtime_installed: None,
    })
}

#[tauri::command]
fn delete_parakeet_model(state: State<'_, AppState>) -> Result<VoiceModelStatus, String> {
    let path = parakeet_model_path(&state)?;
    if !path.exists() {
        return Err("The downloaded Parakeet model is not installed".into());
    }
    if !path.is_dir() {
        return Err("The Parakeet model path is not a directory".into());
    }
    fs::remove_dir_all(&path)
        .map_err(|error| format!("Could not remove the Parakeet model: {error}"))?;
    Ok(VoiceModelStatus {
        id: parakeet::MODEL_ID.into(),
        installed: false,
        path: path.display().to_string(),
        runtime_installed: None,
    })
}

#[tauri::command]
fn delete_nemotron_model(state: State<'_, AppState>) -> Result<VoiceModelStatus, String> {
    let path = nemotron_model_path(&state)?;
    if !path.exists() {
        return Err("The downloaded Nemotron model is not installed".into());
    }
    if !path.is_dir() {
        return Err("The Nemotron model path is not a directory".into());
    }
    fs::remove_dir_all(&path)
        .map_err(|error| format!("Could not remove the Nemotron model: {error}"))?;
    nemotron_status(&state)
}

/// Remove the user-local Python/CUDA runtime only when no dictation or
/// cache-aware sidecar can still reference it. The separately downloaded model
/// is deliberately retained so reinstalling a repaired runtime does not force
/// another multi-gigabyte model download.
#[tauri::command]
async fn remove_nemotron_cuda_runtime(
    state: State<'_, AppState>,
    capture_state: State<'_, NativeCaptureState>,
) -> Result<VoiceModelStatus, String> {
    let session_is_idle = capture_state
        .session
        .lock()
        .map(|coordinator| coordinator.is_idle())
        .map_err(|_| "Dictation session lock was poisoned".to_string())?;
    let capture_is_inactive = capture_state
        .capture
        .lock()
        .map(|capture| capture.is_none())
        .map_err(|_| "Audio capture lock was poisoned".to_string())?;
    if !session_is_idle || !capture_is_inactive {
        return Err("Stop the active dictation before removing the Nemotron CUDA runtime.".into());
    }
    let mut live = capture_state.nemotron_live.try_lock().map_err(|_| {
        "Nemotron is still finishing a previous dictation. Try again shortly.".to_string()
    })?;
    if live.session_started {
        return Err("Nemotron is still finishing a previous dictation. Try again shortly.".into());
    }
    if let Some(mut server) = live.server.take() {
        server.terminate();
    }
    live.fed_samples = 0;
    live.generation = 0;
    live.start_error = None;
    drop(live);

    let runtime = nemotron_runtime_path(&state)?;
    if !runtime.exists() {
        return Err("The Nemotron CUDA runtime is not installed.".into());
    }
    if !runtime.is_dir() {
        return Err("The Nemotron CUDA runtime path is not a directory.".into());
    }
    let runtime_to_remove = runtime.clone();
    tauri::async_runtime::spawn_blocking(move || fs::remove_dir_all(&runtime_to_remove))
        .await
        .map_err(|error| format!("Nemotron runtime removal task failed: {error}"))?
        .map_err(|error| format!("Could not remove the Nemotron CUDA runtime: {error}"))?;
    nemotron_status(&state)
}

/// Open only an application-owned component directory; this command never
/// follows a custom Whisper model path supplied by the user.
#[tauri::command]
fn open_voice_engine_storage(state: State<'_, AppState>) -> Result<(), String> {
    let engine = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .selected_voice_engine;
    let directory = match engine {
        VoiceEngine::Parakeet => {
            let model = parakeet_model_path(&state)?;
            if model.is_dir() {
                model
            } else {
                state.models_directory()?
            }
        }
        VoiceEngine::Nemotron => {
            let runtime = nemotron_runtime_path(&state)?;
            if runtime.is_dir() {
                runtime
            } else {
                state.data_directory()?
            }
        }
        VoiceEngine::Whisper | VoiceEngine::Cloud | VoiceEngine::AppleSpeech => {
            state.models_directory()?
        }
    };
    tauri_plugin_opener::open_path(&directory, None::<&str>)
        .map_err(|error| format!("Could not open the component storage location: {error}"))
}

#[tauri::command]
async fn install_nemotron_cuda_runtime(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<VoiceModelStatus, String> {
    if !nemotron::is_compiled() {
        return Err("Nemotron is included in Voxide's CUDA build for Linux/NVIDIA. Install a CUDA build first.".into());
    }
    let runtime = nemotron_runtime_path(&state)?;
    let runtime_parent = runtime
        .parent()
        .ok_or("Could not determine the Nemotron runtime directory")?;
    fs::create_dir_all(runtime_parent)
        .map_err(|error| format!("Could not create the Nemotron runtime directory: {error}"))?;
    let staging = runtime_parent.join(format!(
        ".nemotron-runtime-install-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir(&staging)
        .map_err(|error| format!("Could not stage the Nemotron CUDA runtime: {error}"))?;
    let _ = app.emit(
        "model-download-progress",
        ModelDownloadProgress {
            id: NEMOTRON_RUNTIME_ID.into(),
            downloaded_bytes: 0,
            total_bytes: None,
        },
    );
    if let Err(error) = install_nemotron_runtime_staging(&staging).await {
        let _ = tokio::fs::remove_dir_all(&staging).await;
        return Err(error);
    }
    let staged_component = staging.clone();
    let destination = runtime.clone();
    let activation = tauri::async_runtime::spawn_blocking(move || {
        replace_component_directory(&staged_component, &destination)
    })
    .await
    .map_err(|error| format!("Nemotron runtime activation task failed: {error}"))?;
    if let Err(error) = activation {
        let _ = tokio::fs::remove_dir_all(&staging).await;
        return Err(error);
    }
    ensure_nemotron_server_script(&runtime)?;
    if !nemotron_runtime_is_verified(&runtime) {
        return Err("The Nemotron CUDA runtime did not pass its verified health check".into());
    }
    nemotron_status(&state)
}

/// Builds and checks a runtime inside a unique staging directory. A failed pip
/// command or GPU health check therefore cannot corrupt the active runtime.
async fn install_nemotron_runtime_staging(staging: &Path) -> Result<(), String> {
    fs::create_dir(staging.join("tmp")).map_err(|error| {
        format!("Could not prepare the Nemotron runtime work directory: {error}")
    })?;
    let python = find_nemotron_python().await?;
    let venv = staging.join("venv");
    run_nemotron_runtime_command(
        &python,
        vec!["-m".into(), "venv".into(), venv.as_os_str().to_owned()],
        staging,
        "create the Python environment",
    )
    .await?;
    let venv_python = nemotron::python_path(staging);
    run_nemotron_runtime_command(
        &venv_python,
        vec![
            "-m".into(),
            "pip".into(),
            "install".into(),
            "--disable-pip-version-check".into(),
            "--no-input".into(),
            "--upgrade".into(),
            "pip".into(),
        ],
        staging,
        "prepare pip",
    )
    .await?;
    run_nemotron_runtime_command(
        &venv_python,
        vec![
            "-m".into(),
            "pip".into(),
            "install".into(),
            "--disable-pip-version-check".into(),
            "--no-input".into(),
            "--upgrade".into(),
            "--index-url".into(),
            "https://download.pytorch.org/whl/cu128".into(),
            "torch".into(),
        ],
        staging,
        "install PyTorch CUDA",
    )
    .await?;
    run_nemotron_runtime_command(
        &venv_python,
        vec![
            "-m".into(),
            "pip".into(),
            "install".into(),
            "--disable-pip-version-check".into(),
            "--no-input".into(),
            "--upgrade".into(),
            "transformers>=5.14,<5.15".into(),
            "librosa>=0.11".into(),
            "numpy>=2.0".into(),
        ],
        staging,
        "install Nemotron dependencies",
    )
    .await?;
    let health = run_nemotron_runtime_command(
        &venv_python,
        vec![
            "-c".into(),
            "import json, sys, torch, transformers, librosa, numpy; from importlib.metadata import distributions; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'; assert torch.cuda.is_available(), 'PyTorch cannot access an NVIDIA CUDA device'; packages = sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in distributions() if d.metadata.get(\"Name\")); print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__, 'librosa': librosa.__version__, 'numpy': numpy.__version__, 'packages': packages}, sort_keys=True))".into(),
        ],
        staging,
        "run the CUDA health check",
    )
    .await?;
    let health: serde_json::Value = serde_json::from_slice(&health.stdout)
        .map_err(|_| "The Nemotron CUDA health check returned invalid metadata".to_string())?;
    if health["torch"].as_str().is_none() || health["cuda"].as_str().is_none() {
        return Err(
            "The Nemotron CUDA health check did not report PyTorch and CUDA versions".into(),
        );
    }
    fs::write(
        staging.join(NEMOTRON_RUNTIME_HEALTH_FILE),
        serde_json::to_vec_pretty(&health)
            .map_err(|error| format!("Could not encode Nemotron runtime metadata: {error}"))?,
    )
    .map_err(|error| format!("Could not save Nemotron runtime metadata: {error}"))?;
    fs::write(
        staging.join(NEMOTRON_RUNTIME_MARKER_FILE),
        format!("Nemotron CUDA runtime v{NEMOTRON_RUNTIME_VERSION}\n"),
    )
    .map_err(|error| format!("Could not finalize the Nemotron runtime marker: {error}"))?;
    write_component_receipt(
        staging,
        &ComponentReceipt {
            schema: COMPONENT_RECEIPT_SCHEMA,
            id: NEMOTRON_RUNTIME_ID.into(),
            version: NEMOTRON_RUNTIME_VERSION.into(),
            source: "https://download.pytorch.org/whl/cu128; PyPI".into(),
            files: component_file_hashes(
                staging,
                [NEMOTRON_RUNTIME_HEALTH_FILE, NEMOTRON_RUNTIME_MARKER_FILE],
            )?,
        },
    )
}

async fn find_nemotron_python() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(configured) = std::env::var_os("VOXIDE_NEMOTRON_PYTHON") {
        candidates.push(PathBuf::from(configured));
    }
    candidates.extend([PathBuf::from("python3.12"), PathBuf::from("python3")]);
    for candidate in candidates {
        let output = tokio::process::Command::new(&candidate)
            .args([
                "-c",
                "import sys; raise SystemExit(sys.version_info < (3, 10))",
            ])
            .output()
            .await;
        if output.is_ok_and(|output| output.status.success()) {
            return Ok(candidate);
        }
    }
    Err("Nemotron requires Python 3.10 or newer (Python 3.12 is recommended).".into())
}

async fn run_nemotron_runtime_command(
    program: &Path,
    arguments: Vec<std::ffi::OsString>,
    working_directory: &Path,
    operation: &str,
) -> Result<std::process::Output, String> {
    let output = tokio::process::Command::new(program)
        .args(arguments)
        .current_dir(working_directory)
        .env("PIP_NO_CACHE_DIR", "1")
        .env("TMPDIR", working_directory.join("tmp"))
        .output()
        .await
        .map_err(|error| format!("Could not {operation}: {error}"))?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(format!(
            "Could not {operation} (exit status {})",
            output.status
        ))
    }
}

const NEMOTRON_MODEL_FILES: [&str; 6] = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// SHA-256 values for every downloaded artifact at `nemotron::MODEL_REVISION`.
/// They are deliberately application-owned rather than trusting an HTTP
/// response's mutable metadata or download-time size alone.
fn nemotron_model_sha256(file: &str) -> Option<&'static str> {
    match file {
        "config.json" => Some("b6574a9110c3473053acebd5b6945ade927896e0121075886712a63dbeca8056"),
        "generation_config.json" => {
            Some("37dbba85d5e2c4c48319202e167ac2107684621aa658f65e422e072d6d58f52e")
        }
        "model.safetensors" => {
            Some("9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174")
        }
        "processor_config.json" => {
            Some("c3e6cbac505049ac27d5d6cde69be5a74d519a523fa9cd9ba6807f197f3a5153")
        }
        "tokenizer.json" => {
            Some("f99d803848330edcb551b81ae77f5baad4ec01199a11b2c2dd5212298213cd77")
        }
        "tokenizer_config.json" => {
            Some("9aac075ebd401089d4ddce37952580c88e01eec24483d55f712f66e13f3d8ea5")
        }
        _ => None,
    }
}

fn nemotron_model_url(file: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{file}?download=true",
        nemotron::MODEL_REPOSITORY,
        nemotron::MODEL_REVISION
    )
}

async fn nemotron_download_size(client: &reqwest::Client, file: &str) -> Option<u64> {
    let response = client.head(nemotron_model_url(file)).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.content_length().or_else(|| {
        response
            .headers()
            .get("x-linked-size")
            .and_then(|size| size.to_str().ok())
            .and_then(|size| size.parse().ok())
    })
}

/// Swap a fully verified staged component into place without deleting the last
/// known-good installation first. Both paths are siblings in the app-owned
/// component directory, so rename is an atomic filesystem operation. If the
/// second rename fails, restore the old component before reporting failure.
fn replace_component_directory(staging: &Path, destination: &Path) -> Result<(), String> {
    let backup = destination.with_file_name(format!(
        ".{}-previous-{}",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("component"),
        uuid::Uuid::new_v4()
    ));
    let had_previous = destination.exists();
    if had_previous {
        fs::rename(destination, &backup).map_err(|error| {
            format!("Could not stage the previous component for replacement: {error}")
        })?;
    }
    if let Err(error) = fs::rename(staging, destination) {
        if had_previous {
            let _ = fs::rename(&backup, destination);
        }
        return Err(format!("Could not activate the new component: {error}"));
    }
    if had_previous {
        if let Err(error) = fs::remove_dir_all(&backup) {
            debug_log::append(&format!(
                "Could not remove superseded component backup: {error}"
            ));
        }
    }
    Ok(())
}

/// Remove only abandoned directories whose names are created by our component
/// transactions. A grace period avoids racing a currently active installer;
/// unrelated user data is never considered for removal.
fn cleanup_abandoned_component_directories(
    directories: &[PathBuf],
    minimum_age: Duration,
) -> usize {
    const OWNED_PREFIXES: [&str; 5] = [
        ".nemotron-runtime-install-",
        ".nemotron-runtime-previous-",
        ".nemotron-3.5-asr-streaming-0.6b-download-",
        ".nemotron-3.5-asr-streaming-0.6b-previous-",
        ".parakeet-install-",
    ];
    let mut removed = 0;
    for directory in directories {
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };
            if !OWNED_PREFIXES
                .iter()
                .any(|prefix| file_name.starts_with(prefix))
            {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let old_enough = metadata
                .modified()
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age >= minimum_age);
            if metadata.is_dir() && old_enough && fs::remove_dir_all(&path).is_ok() {
                removed += 1;
            }
        }
    }
    removed
}

#[tauri::command]
async fn download_nemotron_model(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<VoiceModelStatus, String> {
    if !nemotron::is_compiled() {
        return Err("Nemotron is included in Voxide's CUDA build for Linux/NVIDIA. Install a CUDA build first.".into());
    }
    let runtime = nemotron_runtime_path(&state)?;
    if !nemotron_runtime_is_verified(&runtime) {
        return Err("Install the Nemotron CUDA runtime before downloading the model.".into());
    }
    let models = state.models_directory()?;
    let destination = nemotron::model_directory(&models);
    if nemotron_model_is_verified(&destination) {
        return nemotron_status(&state);
    }
    let staging = models.join(format!(
        ".{}-download-{}",
        nemotron::MODEL_ID,
        uuid::Uuid::new_v4()
    ));
    tokio::fs::create_dir_all(&staging)
        .await
        .map_err(|error| format!("Could not prepare the Nemotron model download: {error}"))?;
    let client = reqwest::Client::new();
    let sizes = futures_util::future::join_all(
        NEMOTRON_MODEL_FILES
            .iter()
            .map(|file| nemotron_download_size(&client, file)),
    )
    .await;
    let total_bytes = sizes
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .map(|sizes| sizes.into_iter().sum());
    let download_result = async {
        let mut downloaded_bytes = 0_u64;
        for file in NEMOTRON_MODEL_FILES {
            let expected_digest = nemotron_model_sha256(file)
                .ok_or_else(|| format!("No checksum is registered for Nemotron {file}"))?;
            let response = client
                .get(nemotron_model_url(file))
                .send()
                .await
                .map_err(|error| format!("Could not download Nemotron {file}: {error}"))?;
            if !response.status().is_success() {
                return Err(format!("Could not download Nemotron {file}: HTTP {}", response.status()));
            }
            if model_content_type_is_markup(response.headers().get(reqwest::header::CONTENT_TYPE)) {
                return Err(format!("Could not download Nemotron {file}: the server returned HTML or XML instead of model data."));
            }
            let output_path = staging.join(file);
            let mut output = tokio::fs::File::create(&output_path)
                .await
                .map_err(|error| format!("Could not create the Nemotron {file} download: {error}"))?;
            use tokio::io::AsyncWriteExt;
            let mut stream = response.bytes_stream();
            let mut file_bytes = 0_u64;
            let mut digest = Sha256::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|error| format!("The Nemotron {file} download was interrupted: {error}"))?;
                output
                    .write_all(&chunk)
                    .await
                    .map_err(|error| format!("Could not save the Nemotron {file} download: {error}"))?;
                let count = chunk.len() as u64;
                digest.update(&chunk);
                downloaded_bytes += count;
                file_bytes += count;
                let _ = app.emit(
                    "model-download-progress",
                    ModelDownloadProgress {
                        id: NEMOTRON_MODEL_DOWNLOAD_ID.into(),
                        downloaded_bytes,
                        total_bytes,
                    },
                );
            }
            output
                .flush()
                .await
                .map_err(|error| format!("Could not finalize the Nemotron {file} download: {error}"))?;
            if file_bytes == 0 {
                return Err(format!("The Nemotron {file} download was empty"));
            }
            let actual_digest = format!("{:x}", digest.finalize());
            if actual_digest != expected_digest {
                return Err(format!(
                    "The Nemotron {file} checksum did not match the pinned model revision"
                ));
            }
            if file == "model.safetensors" && file_bytes < 1_000_000_000 {
                return Err("The Nemotron model download is unexpectedly small; refusing to install it.".into());
            }
        }
        Ok::<(), String>(())
    }
    .await;
    if let Err(error) = download_result {
        let _ = tokio::fs::remove_dir_all(&staging).await;
        return Err(error);
    }
    if !nemotron::model_is_installed(&staging) {
        let _ = tokio::fs::remove_dir_all(&staging).await;
        return Err("The Nemotron model download is incomplete".into());
    }
    let receipt_directory = staging.clone();
    let receipt_result = tauri::async_runtime::spawn_blocking(move || {
        write_component_receipt(
            &receipt_directory,
            &ComponentReceipt {
                schema: COMPONENT_RECEIPT_SCHEMA,
                id: nemotron::MODEL_ID.into(),
                version: nemotron::MODEL_REVISION.into(),
                source: format!(
                    "https://huggingface.co/{}/tree/{}",
                    nemotron::MODEL_REPOSITORY,
                    nemotron::MODEL_REVISION
                ),
                files: component_file_hashes(&receipt_directory, NEMOTRON_MODEL_FILES)?,
            },
        )
    })
    .await;
    match receipt_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let _ = tokio::fs::remove_dir_all(&staging).await;
            return Err(error);
        }
        Err(error) => {
            let _ = tokio::fs::remove_dir_all(&staging).await;
            return Err(format!("Nemotron receipt task failed: {error}"));
        }
    }
    let staged_component = staging.clone();
    let destination_component = destination.clone();
    tauri::async_runtime::spawn_blocking(move || {
        replace_component_directory(&staged_component, &destination_component)
    })
    .await
    .map_err(|error| format!("Nemotron installation task failed: {error}"))??;
    nemotron_status(&state)
}

#[tauri::command]
fn audio_input_devices() -> Result<Vec<String>, String> {
    audio::input_device_names()
}

#[tauri::command]
fn accessibility_permission_status() -> AccessibilityPermissionStatus {
    AccessibilityPermissionStatus {
        supported: cfg!(target_os = "macos"),
        trusted: permissions::accessibility_trusted(),
        guidance: permissions::accessibility_guidance().into(),
    }
}

#[tauri::command]
fn open_accessibility_settings() -> Result<(), String> {
    permissions::open_accessibility_settings()
}

#[tauri::command]
async fn download_whisper_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
) -> Result<VoiceModelStatus, String> {
    let filename = whisper_model_filename(&model_id)?;
    let directory = state.models_directory()?;
    let destination = directory.join(filename);
    let temporary = directory.join(format!("{filename}.download"));
    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}?download=true"
    );
    let response = reqwest::get(&url)
        .await
        .map_err(|error| format!("Could not download the Whisper model: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Could not download the Whisper model: HTTP {}",
            response.status()
        ));
    }
    if model_content_type_is_markup(response.headers().get(reqwest::header::CONTENT_TYPE)) {
        return Err("Could not download the Whisper model: the server returned an HTML or XML page instead of model data. A network proxy or firewall may be blocking model downloads.".into());
    }

    let total_bytes = response.content_length();
    let download_result = async {
        let mut downloaded_bytes = 0_u64;
        let mut stream = response.bytes_stream();
        let mut output = tokio::fs::File::create(&temporary)
            .await
            .map_err(|error| format!("Could not create the model download: {error}"))?;
        use tokio::io::AsyncWriteExt;
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|error| format!("The model download was interrupted: {error}"))?;
            output
                .write_all(&chunk)
                .await
                .map_err(|error| format!("Could not save the model download: {error}"))?;
            downloaded_bytes += chunk.len() as u64;
            let _ = app.emit(
                "model-download-progress",
                ModelDownloadProgress {
                    id: model_id.clone(),
                    downloaded_bytes,
                    total_bytes,
                },
            );
        }
        output
            .flush()
            .await
            .map_err(|error| format!("Could not finalize the model download: {error}"))?;
        drop(output);
        Ok::<u64, String>(downloaded_bytes)
    }
    .await;
    let downloaded_bytes = match download_result {
        Ok(downloaded_bytes) => downloaded_bytes,
        Err(error) => {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(error);
        }
    };
    if let Err(error) = validate_whisper_model_download(&temporary, total_bytes, downloaded_bytes) {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error);
    }
    if destination.is_file() {
        tokio::fs::remove_file(&destination)
            .await
            .map_err(|error| format!("Could not replace the existing Whisper model: {error}"))?;
    } else if destination.exists() {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(
            "Could not install the Whisper model because its destination is not a regular file"
                .into(),
        );
    }
    tokio::fs::rename(&temporary, &destination)
        .await
        .map_err(|error| format!("Could not install the Whisper model: {error}"))?;

    Ok(VoiceModelStatus {
        id: model_id,
        installed: true,
        path: destination.display().to_string(),
        runtime_installed: None,
    })
}

fn validate_parakeet_archive(
    path: &Path,
    expected_bytes: Option<u64>,
    downloaded_bytes: u64,
) -> Result<(), String> {
    if downloaded_bytes != parakeet::MODEL_ARCHIVE_BYTES {
        return Err("The Parakeet download size did not match the pinned release asset".into());
    }
    if expected_bytes.is_some_and(|expected| expected != parakeet::MODEL_ARCHIVE_BYTES) {
        return Err("The Parakeet server reported an unexpected release asset size".into());
    }
    let metadata = fs::metadata(path)
        .map_err(|error| format!("Could not verify the Parakeet download: {error}"))?;
    if !metadata.is_file() || metadata.len() != downloaded_bytes {
        return Err("The Parakeet download was not saved as a complete regular file".into());
    }
    let mut prefix = [0_u8; 3];
    fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut prefix))
        .map_err(|error| format!("Could not inspect the Parakeet download: {error}"))?;
    if &prefix != b"BZh" {
        return Err("The Parakeet download is not a bzip2 model archive".into());
    }
    if sha256_file(path)? != parakeet::MODEL_ARCHIVE_SHA256 {
        return Err("The Parakeet download checksum did not match the pinned release asset".into());
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("Could not hash a downloaded component: {error}"))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("Could not hash a downloaded component: {error}"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn component_file_hashes(
    directory: &Path,
    files: impl IntoIterator<Item = &'static str>,
) -> Result<BTreeMap<String, String>, String> {
    files
        .into_iter()
        .map(|file| sha256_file(&directory.join(file)).map(|digest| (file.to_owned(), digest)))
        .collect()
}

fn write_component_receipt(directory: &Path, receipt: &ComponentReceipt) -> Result<(), String> {
    let contents = serde_json::to_vec_pretty(receipt)
        .map_err(|error| format!("Could not encode the component receipt: {error}"))?;
    let temporary = directory.join(format!("{COMPONENT_RECEIPT_FILE}.tmp"));
    fs::write(&temporary, contents)
        .map_err(|error| format!("Could not write the component receipt: {error}"))?;
    fs::rename(&temporary, directory.join(COMPONENT_RECEIPT_FILE))
        .map_err(|error| format!("Could not finalize the component receipt: {error}"))
}

/// A receipt is meaningful only while every named artifact still matches its
/// recorded digest. Reject non-normal relative paths as receipts live in a
/// user-writable directory and must never direct verification outside the
/// component root.
fn component_receipt_is_verified(
    directory: &Path,
    expected_id: &str,
    expected_version: &str,
) -> bool {
    let receipt = fs::read(directory.join(COMPONENT_RECEIPT_FILE))
        .ok()
        .and_then(|contents| serde_json::from_slice::<ComponentReceipt>(&contents).ok());
    let Some(receipt) = receipt else {
        return false;
    };
    if receipt.schema != COMPONENT_RECEIPT_SCHEMA
        || receipt.id != expected_id
        || receipt.version != expected_version
        || receipt.files.is_empty()
    {
        return false;
    }
    receipt
        .files
        .into_iter()
        .all(|(relative, expected_digest)| {
            let path = Path::new(&relative);
            path.components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
                && sha256_file(&directory.join(path))
                    .map(|actual_digest| actual_digest == expected_digest)
                    .unwrap_or(false)
        })
}

/// Fast readiness check for a component that was fully hashed before atomic
/// activation. Rehashing a multi-gigabyte speech model at every hotkey would
/// turn a safety check into a reliability regression; ordinary readiness
/// instead checks the immutable receipt identity and complete file inventory.
/// `component_receipt_is_verified` remains the full-content verifier for
/// installation and explicit integrity checks.
fn component_receipt_is_recorded(
    directory: &Path,
    expected_id: &str,
    expected_version: &str,
    expected_files: &[&str],
) -> bool {
    let receipt = fs::read(directory.join(COMPONENT_RECEIPT_FILE))
        .ok()
        .and_then(|contents| serde_json::from_slice::<ComponentReceipt>(&contents).ok());
    let Some(receipt) = receipt else {
        return false;
    };
    receipt.schema == COMPONENT_RECEIPT_SCHEMA
        && receipt.id == expected_id
        && receipt.version == expected_version
        && receipt.files.len() == expected_files.len()
        && expected_files
            .iter()
            .all(|file| receipt.files.contains_key(*file))
}

pub(crate) fn parakeet_model_is_verified(model_directory: &Path) -> bool {
    parakeet::model_is_installed(model_directory)
        && component_receipt_is_recorded(
            model_directory,
            parakeet::MODEL_ID,
            parakeet::MODEL_ARCHIVE_SHA256,
            parakeet::required_files(),
        )
}

fn nemotron_model_is_verified(model_directory: &Path) -> bool {
    nemotron::model_is_installed(model_directory)
        && component_receipt_is_recorded(
            model_directory,
            nemotron::MODEL_ID,
            nemotron::MODEL_REVISION,
            &NEMOTRON_MODEL_FILES,
        )
}

fn nemotron_runtime_is_verified(runtime_directory: &Path) -> bool {
    nemotron::runtime_is_installed(runtime_directory)
        && component_receipt_is_verified(
            runtime_directory,
            NEMOTRON_RUNTIME_ID,
            NEMOTRON_RUNTIME_VERSION,
        )
}

fn install_parakeet_archive(
    archive_path: &Path,
    models_directory: &Path,
) -> Result<PathBuf, String> {
    let staging = models_directory.join(format!(".parakeet-install-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&staging)
        .map_err(|error| format!("Could not prepare the Parakeet installation: {error}"))?;
    let install_result = (|| {
        let archive = fs::File::open(archive_path)
            .map_err(|error| format!("Could not open the Parakeet archive: {error}"))?;
        let decoder = bzip2::read::BzDecoder::new(archive);
        let mut archive = tar::Archive::new(decoder);
        for entry in archive
            .entries()
            .map_err(|error| format!("Could not read the Parakeet archive: {error}"))?
        {
            entry
                .map_err(|error| format!("Could not read a Parakeet archive entry: {error}"))?
                .unpack_in(&staging)
                .map_err(|error| {
                    format!("Could not extract the Parakeet archive safely: {error}")
                })?;
        }
        let extracted = staging.join(parakeet::archive_root());
        if !parakeet::model_is_installed(&extracted) {
            return Err(parakeet::installation_error(&extracted));
        }
        write_component_receipt(
            &extracted,
            &ComponentReceipt {
                schema: COMPONENT_RECEIPT_SCHEMA,
                id: parakeet::MODEL_ID.into(),
                version: parakeet::MODEL_ARCHIVE_SHA256.into(),
                source: parakeet::MODEL_ARCHIVE_URL.into(),
                files: component_file_hashes(
                    &extracted,
                    parakeet::required_files().iter().copied(),
                )?,
            },
        )?;
        let destination = parakeet::model_directory(models_directory);
        replace_component_directory(&extracted, &destination)?;
        Ok(destination)
    })();
    let _ = fs::remove_dir_all(&staging);
    install_result
}

#[tauri::command]
async fn download_parakeet_model(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<VoiceModelStatus, String> {
    if !parakeet::is_compiled() {
        return Err(
            "Parakeet is included in the CUDA build. Install a CUDA build of Voxide first.".into(),
        );
    }
    let directory = state.models_directory()?;
    let archive_name = format!("{}.tar.bz2", parakeet::MODEL_ID);
    let temporary = directory.join(format!("{archive_name}.download"));
    let response = reqwest::get(parakeet::MODEL_ARCHIVE_URL)
        .await
        .map_err(|error| format!("Could not download the Parakeet model: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Could not download the Parakeet model: HTTP {}",
            response.status()
        ));
    }
    if model_content_type_is_markup(response.headers().get(reqwest::header::CONTENT_TYPE)) {
        return Err("Could not download the Parakeet model: the server returned HTML or XML instead of model data.".into());
    }
    let total_bytes = response.content_length();
    let download_result = async {
        let mut downloaded_bytes = 0_u64;
        let mut stream = response.bytes_stream();
        let mut output = tokio::fs::File::create(&temporary)
            .await
            .map_err(|error| format!("Could not create the Parakeet download: {error}"))?;
        use tokio::io::AsyncWriteExt;
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|error| format!("The Parakeet download was interrupted: {error}"))?;
            output
                .write_all(&chunk)
                .await
                .map_err(|error| format!("Could not save the Parakeet download: {error}"))?;
            downloaded_bytes += chunk.len() as u64;
            let _ = app.emit(
                "model-download-progress",
                ModelDownloadProgress {
                    id: parakeet::MODEL_ID.into(),
                    downloaded_bytes,
                    total_bytes,
                },
            );
        }
        output
            .flush()
            .await
            .map_err(|error| format!("Could not finalize the Parakeet download: {error}"))?;
        Ok::<u64, String>(downloaded_bytes)
    }
    .await;
    let downloaded_bytes = match download_result {
        Ok(downloaded_bytes) => downloaded_bytes,
        Err(error) => {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(error);
        }
    };
    let archive_to_validate = temporary.clone();
    let validation = tauri::async_runtime::spawn_blocking(move || {
        validate_parakeet_archive(&archive_to_validate, total_bytes, downloaded_bytes)
    })
    .await
    .map_err(|error| format!("Parakeet validation task failed: {error}"))?;
    if let Err(error) = validation {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error);
    }
    let install_directory = directory.clone();
    let install_archive = temporary.clone();
    let destination = tauri::async_runtime::spawn_blocking(move || {
        install_parakeet_archive(&install_archive, &install_directory)
    })
    .await
    .map_err(|error| format!("Parakeet installation task failed: {error}"))?;
    let _ = tokio::fs::remove_file(&temporary).await;
    let destination = destination?;
    Ok(VoiceModelStatus {
        id: parakeet::MODEL_ID.into(),
        installed: true,
        path: destination.display().to_string(),
        runtime_installed: None,
    })
}

/// Resolves the current input device (device handle, negotiated config, and
/// sound-server routing) off the record hotkey path and caches it so the next
/// press only opens the stream. Safe to call whenever no capture is active —
/// at startup and after a dictation frees the microphone — because it only
/// queries device metadata and sets routing env vars; it never opens a stream
/// (so idle prewarm shows no microphone indicator).
fn spawn_capture_prewarm(handle: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let selected_device = handle
            .state::<AppState>()
            .database
            .lock()
            .ok()
            .and_then(|database| database.settings.selected_input_device.clone());
        let prewarm_device = selected_device.clone();
        match tauri::async_runtime::spawn_blocking(move || {
            audio::AudioCapture::prepare(prewarm_device.as_deref())
        })
        .await
        {
            Ok(Ok(prepared)) => {
                let capture_state = handle.state::<NativeCaptureState>();
                if let Ok(mut guard) = capture_state.prepared_input.lock() {
                    *guard = Some(PreparedCapture {
                        device_key: selected_device,
                        prepared,
                    });
                }
                debug_log::append(
                    "Audio capture input prewarmed (device, config, and routing resolved)",
                );
            }
            Ok(Err(error)) => debug_log::append(&format!("Audio capture prewarm failed: {error}")),
            Err(error) => debug_log::append(&format!("Audio capture prewarm task failed: {error}")),
        }
    });
}

/// Opens the microphone for dictation, preferring the prewarmed target that
/// was resolved off the hotkey path. On a cache hit the press only builds and
/// plays the stream; a miss — or a stale entry after a mic-preference change —
/// resolves fresh, caches it for next time, and starts from it. A prewarmed
/// target that fails to open (e.g. the device was unplugged since prewarm)
/// falls back to a fresh resolution rather than failing the recording.
fn start_dictation_capture(
    capture_state: &NativeCaptureState,
    selected_device: Option<&str>,
    on_level: audio::LevelCallback,
) -> Result<audio::AudioCapture, String> {
    let cached = capture_state.prepared_input.lock().ok().and_then(|guard| {
        guard
            .as_ref()
            .filter(|prepared| prepared.device_key.as_deref() == selected_device)
            .map(|prepared| prepared.prepared.clone())
    });
    if let Some(prepared) = cached {
        match audio::AudioCapture::start_prepared(&prepared, Some(on_level.clone())) {
            Ok(capture) => return Ok(capture),
            Err(error) => {
                debug_log::append(&format!(
                    "Prewarmed microphone failed to open ({error}); re-resolving the input device"
                ));
                if let Ok(mut guard) = capture_state.prepared_input.lock() {
                    *guard = None;
                }
            }
        }
    }
    let prepared = audio::AudioCapture::prepare(selected_device)?;
    if let Ok(mut guard) = capture_state.prepared_input.lock() {
        *guard = Some(PreparedCapture {
            device_key: selected_device.map(str::to_owned),
            prepared: prepared.clone(),
        });
    }
    audio::AudioCapture::start_prepared(&prepared, Some(on_level))
}

#[tauri::command]
fn start_native_dictation(
    app: AppHandle,
    state: State<'_, AppState>,
    capture_state: State<'_, NativeCaptureState>,
    prompt_profile_id: Option<String>,
) -> Result<NativeCaptureStarted, String> {
    let mut capture = capture_state
        .capture
        .lock()
        .map_err(|_| "Audio capture lock was poisoned".to_string())?;
    if capture.is_some() {
        return Err("Dictation is already recording".into());
    }
    let prompt_profile_id = prompt_profile_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned);
    let (source_application, source_window_title) = active_application_context();
    let preceding_text = capture_state
        .continuous_context
        .lock()
        .map_err(|_| "Continuous dictation context lock was poisoned".to_string())?
        .preceding_text_for(
            source_application.as_deref(),
            source_window_title.as_deref(),
        );
    let dictation_context = DictationContext {
        source_application: source_application.clone(),
        source_window_title: source_window_title.clone(),
        prompt_profile_id: prompt_profile_id.clone(),
        preceding_text,
    };
    let last_emit = Arc::new(Mutex::new(Instant::now() - Duration::from_secs(1)));
    let app_for_levels = app.clone();
    let (settings, custom_words, cloud_profile, dictionary) = {
        let database = state
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        if let Some(profile_id) = prompt_profile_id.as_deref() {
            let profile_exists = database
                .prompt_profiles
                .iter()
                .any(|profile| profile.mode == DictationMode::Dictate && profile.id == profile_id);
            if !profile_exists {
                return Err("The selected dictation prompt profile no longer exists".into());
            }
        }
        let settings = database.settings.clone();
        let cloud_profile = settings
            .selected_voice_engine
            .requires_provider_profile()
            .then(|| selected_provider(&database, None))
            .transpose()?;
        let custom_words = active_recognition_vocabulary(&settings, &database.custom_words);
        (
            settings,
            custom_words,
            cloud_profile,
            database.dictionary.clone(),
        )
    };
    settings
        .selected_voice_engine
        .prepare_live_capture(&settings, &state)?;
    if let Err(error) = apply_overlay_window_layout(&app, &settings) {
        debug_log::append(&format!(
            "Could not apply dictation overlay layout: {error}"
        ));
    }
    let preview_generation = capture_state
        .session
        .lock()
        .map_err(|_| "Dictation session lock was poisoned".to_string())?
        .start()
        .map_err(str::to_string)?;
    capture_state
        .preview_generation
        .store(preview_generation, Ordering::SeqCst);
    if let Err(error) = capture_state
        .context
        .lock()
        .map(|mut context| *context = dictation_context)
        .map_err(|_| "Dictation context lock was poisoned".to_string())
    {
        rollback_native_dictation_start(&capture_state, preview_generation);
        return Err(error);
    }
    // Reserve the selected engine before opening the microphone. This is
    // especially important for cache-aware engines whose prior finalization
    // may still own their single stream.
    if let Err(error) = settings
        .selected_voice_engine
        .begin_live_session(&capture_state, preview_generation)
    {
        rollback_native_dictation_start(&capture_state, preview_generation);
        return Err(error);
    }
    let level_callback: audio::LevelCallback = Arc::new(move |level| {
        let Ok(mut last_emit) = last_emit.lock() else {
            return;
        };
        if last_emit.elapsed() >= Duration::from_millis(33) {
            *last_emit = Instant::now();
            let _ = app_for_levels.emit("dictation-audio-level", level);
        }
    });
    let started = match start_dictation_capture(
        &capture_state,
        settings.selected_input_device.as_deref(),
        level_callback,
    ) {
        Ok(capture) => capture,
        Err(error) => {
            rollback_native_dictation_start(&capture_state, preview_generation);
            return Err(error);
        }
    };
    let sample_rate = started.sample_rate();
    let channels = started.channels();
    let mut capture_started_at = capture_state
        .capture_started_at
        .lock()
        .map_err(|_| "Dictation capture timing lock was poisoned".to_string())?;
    *capture = Some(started);
    *capture_started_at = Some(Instant::now());
    drop(capture_started_at);
    drop(capture);
    debug_log::append(&format!(
        "Dictation capture started (session: {preview_generation}, engine: {}, runtime: {}, streaming_preview: {}, sample_rate: {sample_rate}, channels: {channels})",
        asr::SpeechEngine::engine_id(&settings.selected_voice_engine),
        settings.selected_voice_engine.diagnostic_runtime_version(&state),
        settings.enable_streaming_preview
    ));
    spawn_capture_error_monitor(app.clone(), preview_generation);
    update_tray_status(&app, TrayVisualState::Recording);
    if settings.enable_streaming_preview {
        emit_overlay(&app, "recording", "Listening…");
    }
    if let Err(error) = settings.selected_voice_engine.spawn_live_preview(
        app.clone(),
        &capture_state,
        preview_generation,
        &state,
        &settings,
        custom_words,
        cloud_profile,
        dictionary,
    ) {
        // No preview/setup error may leave an active but invisible microphone
        // stream behind. The coordinator owns the generation, so invalidating
        // it also prevents any already-spawned adapter from publishing text.
        rollback_native_dictation_start(&capture_state, preview_generation);
        update_tray_status(&app, TrayVisualState::Ready);
        return Err(error);
    }
    Ok(NativeCaptureStarted {
        sample_rate,
        channels,
        source_application,
        source_window_title,
    })
}

/// Produces a deliberately conservative live transcript. Whisper previews are
/// independent decodes of a growing, then rolling, audio window. The tail is
/// therefore provisional. A word that has survived an overlap between two
/// windows is old enough to make the overlay monotonic: it is never rewritten
/// by a later, noisier snapshot. The current provisional tail remains visible
/// after that protected prefix, so the overlay keeps progressing while the
/// user speaks. The actual final decode still replaces the overlay when
/// recording stops.
#[derive(Default)]
struct WhisperPreviewStability {
    /// The best chronological sequence assembled from preview windows. Its
    /// suffix remains editable until a subsequent window overlaps it.
    timeline_words: Vec<String>,
    /// Prefix of `timeline_words` which has appeared in at least two aligned
    /// windows and is consequently safe to show to the user.
    confirmed_words: usize,
}

impl WhisperPreviewStability {
    fn observe(&mut self, candidate: &str) -> Option<String> {
        let current_words = candidate
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if current_words.is_empty() {
            return None;
        }
        if self.timeline_words.is_empty() {
            self.timeline_words = current_words;
            return self.visible_text();
        }

        // The window initially grows from the same start, then rolls forward
        // once it reaches eight seconds. Align the new snapshot's prefix to
        // any point in the assembled timeline, rather than requiring an
        // absolute common prefix. That lets "three four five" extend an
        // earlier "one two three four" without retracting "one two".
        let (start, overlap) = self
            .timeline_words
            .iter()
            .enumerate()
            .map(|(start, _)| {
                let overlap = self.timeline_words[start..]
                    .iter()
                    .zip(&current_words)
                    .take_while(|(previous, current)| preview_words_match(previous, current))
                    .count();
                (start, overlap)
            })
            .max_by_key(|(_, overlap)| *overlap)?;

        // One generic word is not enough evidence that a rolling window has
        // aligned ("the" and "I" are common). A one-word match at the start
        // is still useful before anything has been confirmed: it lets an early
        // correction replace only the provisional tail.
        if overlap == 0 || (overlap == 1 && start != 0) {
            return self.adopt_unaligned_hypothesis(current_words);
        }
        let matched_end = start + overlap;

        // A later snapshot is allowed to correct only the volatile tail. If
        // it disagrees before a confirmed word, keep the overlay unchanged
        // and wait for a better-aligned snapshot rather than visibly rewriting
        // text that was already shown.
        if matched_end < self.confirmed_words {
            return if start == 0 {
                self.visible_text()
            } else {
                self.adopt_unaligned_hypothesis(current_words)
            };
        }

        self.timeline_words.truncate(matched_end);
        self.timeline_words
            .extend(current_words.into_iter().skip(overlap));
        self.confirmed_words = self.confirmed_words.max(matched_end);
        self.visible_text()
    }

    /// If Whisper's first few snapshots have no usable word overlap, they
    /// cannot be safely committed. They are still useful live feedback. After
    /// text has been committed, treat an unrelated snapshot as a new
    /// provisional tail rather than freezing the visible overlay at the old
    /// prefix.
    fn adopt_unaligned_hypothesis(&mut self, current_words: Vec<String>) -> Option<String> {
        if self.confirmed_words == 0 {
            self.timeline_words = current_words;
        } else {
            self.timeline_words.extend(current_words);
        }
        self.visible_text()
    }

    fn visible_text(&self) -> Option<String> {
        (!self.timeline_words.is_empty()).then(|| self.timeline_words.join(" "))
    }

    /// A VAD-empty preview means the rolling window has reached room silence.
    /// Retain only words that survived an earlier overlap; a volatile tail at
    /// this point is the most likely source of silence hallucinations.
    fn observe_silence(&mut self) -> Option<String> {
        if self.confirmed_words != 0 {
            self.timeline_words.truncate(self.confirmed_words);
        }
        self.visible_text()
    }
}

fn preview_words_match(previous: &str, current: &str) -> bool {
    let previous = previous.trim_matches(|character: char| character.is_ascii_punctuation());
    let current = current.trim_matches(|character: char| character.is_ascii_punctuation());
    previous.eq_ignore_ascii_case(current)
}

fn spawn_live_whisper_preview(
    app: AppHandle,
    preview_generation: u64,
    preview_cancellation: Arc<AtomicU64>,
    model_path: PathBuf,
    vad_model: Option<PathBuf>,
    language: String,
    custom_words: Vec<String>,
    preview_char_limit: usize,
) {
    tauri::async_runtime::spawn(async move {
        // Pace the preview by how long transcription actually takes on this
        // machine: GPU inference refreshes several times per second while
        // CPU inference backs off to its own cost instead of piling up.
        let mut interval = Duration::from_millis(600);
        let mut stability = WhisperPreviewStability::default();
        loop {
            tokio::time::sleep(interval).await;
            let capture_state = app.state::<NativeCaptureState>();
            if capture_state.preview_generation.load(Ordering::SeqCst) != preview_generation {
                return;
            }
            let captured = capture_state.capture.lock().ok().and_then(|capture| {
                capture
                    .as_ref()
                    .and_then(|capture| capture.snapshot_recent(Duration::from_secs(8)).ok())
            });
            let Some(captured) = captured else {
                return;
            };
            if captured.duration_ms < 1_000 {
                continue;
            }
            let captured_duration_ms = captured.duration_ms;
            let samples = match audio::mono_resample_for_whisper(captured) {
                Ok(samples) => samples,
                Err(_) => continue,
            };
            if !begin_preview(&capture_state, preview_generation) {
                continue;
            }
            let model_path = model_path.clone();
            let vad_model = vad_model.clone();
            let language = language.clone();
            let custom_words = custom_words.clone();
            let preview_cancellation = Arc::clone(&preview_cancellation);
            let transcription_started = Instant::now();
            let preview_result = tauri::async_runtime::spawn_blocking(move || {
                speech::transcribe_whisper_with_options(
                    samples,
                    &model_path,
                    vad_model.as_deref(),
                    &language,
                    &custom_words,
                    speech::TranscriptionOptions::preview(preview_cancellation, preview_generation),
                )
            })
            .await
            .map_err(|error| format!("preview task failed: {error}"))
            .and_then(|result| result);
            finish_preview(&capture_state, preview_generation);
            interval = transcription_started
                .elapsed()
                .mul_f32(1.5)
                .clamp(Duration::from_millis(250), Duration::from_millis(2_500));
            match preview_result {
                Ok(result) if !result.text.trim().is_empty() => {
                    let Some(stable_text) = stability.observe(&result.text) else {
                        debug_log::append(&format!(
                            "Live Whisper preview is waiting for confirmation (audio_ms: {})",
                            captured_duration_ms
                        ));
                        continue;
                    };
                    let capture_state = app.state::<NativeCaptureState>();
                    if capture_state.preview_generation.load(Ordering::SeqCst) == preview_generation
                    {
                        debug_log::append(&format!(
                            "Live Whisper preview emitted (audio_ms: {}, decode_ms: {})",
                            captured_duration_ms, result.timings.decode_ms
                        ));
                        emit_overlay(
                            &app,
                            "recording",
                            tail_characters(&stable_text, preview_char_limit),
                        );
                    }
                }
                Ok(_) => {
                    debug_log::append(&format!(
                        "Live Whisper preview was VAD-empty (audio_ms: {})",
                        captured_duration_ms
                    ));
                    if let Some(stable_text) = stability.observe_silence() {
                        let capture_state = app.state::<NativeCaptureState>();
                        if capture_state.preview_generation.load(Ordering::SeqCst)
                            == preview_generation
                        {
                            emit_overlay(
                                &app,
                                "recording",
                                tail_characters(&stable_text, preview_char_limit),
                            );
                        }
                    }
                }
                Err(error) => debug_log::append(&format!(
                    "Live Whisper preview skipped (audio_ms: {}): {error}",
                    captured_duration_ms
                )),
            }
        }
    });
}

/// Mirrors FluidVoice's Parakeet TDT v2 preview cadence. TDT is an offline
/// model, so each snapshot is a fresh decode—not VAD-sliced audio and not a
/// stateful streaming decoder. The decode is bounded to a trailing window
/// (like the cloud preview) rather than the whole capture: re-decoding a
/// growing recording every tick is O(total) work that holds `INFERENCE_LOCK`
/// progressively longer and starves the final decode. The overlay only shows
/// the recent tail anyway, and `fluidvoice_preview_reconcile` gracefully falls
/// back to current-only text once the window slides past the recording start,
/// so bounding the window costs no visible context. The authoritative final
/// pass still decodes the complete capture. Voxide additionally hides the
/// timestamped, still-tentative tail of each CUDA hypothesis before showing
/// it, because Sherpa's INT8 ONNX decoder can otherwise invent words at the
/// live endpoint that disappear in the final full decode.
fn spawn_live_parakeet_preview(
    app: AppHandle,
    preview_generation: u64,
    model_path: PathBuf,
    settings: Settings,
    dictionary: Vec<DictionaryEntry>,
    preview_char_limit: usize,
) {
    tauri::async_runtime::spawn(async move {
        const PREVIEW_INTERVAL: Duration = Duration::from_millis(600);
        const MINIMUM_PREVIEW_SAMPLES: usize = audio::WHISPER_SAMPLE_RATE as usize;
        // Trailing audio window each preview decodes. Bounds per-tick cost so a
        // long recording cannot grow the decode (and its `INFERENCE_LOCK` hold)
        // without limit; matches the cloud preview's window.
        const PARAKEET_PREVIEW_WINDOW: Duration = Duration::from_secs(20);

        let mut skip_next_snapshot = false;
        loop {
            tokio::time::sleep(PREVIEW_INTERVAL).await;
            let capture_state = app.state::<NativeCaptureState>();
            if capture_state.preview_generation.load(Ordering::SeqCst) != preview_generation {
                return;
            }
            if skip_next_snapshot {
                skip_next_snapshot = false;
                debug_log::append("Live Parakeet preview skipped after a slow snapshot");
                continue;
            }
            let captured = capture_state.capture.lock().ok().and_then(|capture| {
                capture
                    .as_ref()
                    // Bound preview decode cost to a recent window instead of
                    // the whole (growing) capture, matching the cloud preview.
                    .and_then(|capture| capture.snapshot_recent(PARAKEET_PREVIEW_WINDOW).ok())
            });
            let Some(captured) = captured else {
                return;
            };
            let samples = match audio::mono_resample_for_whisper(captured) {
                Ok(samples) => samples,
                Err(error) => {
                    debug_log::append(&format!(
                        "Live Parakeet preview could not resample audio: {error}"
                    ));
                    continue;
                }
            };
            if samples.len() < MINIMUM_PREVIEW_SAMPLES {
                continue;
            }
            if !begin_preview(&capture_state, preview_generation) {
                continue;
            }
            let sample_count = samples.len();
            let started = Instant::now();
            let model_path = model_path.clone();
            let result = tauri::async_runtime::spawn_blocking(move || {
                parakeet::transcribe_preview_samples(&samples, &model_path)
            })
            .await
            .map_err(|error| format!("preview task failed: {error}"))
            .and_then(|result| result);
            finish_preview(&capture_state, preview_generation);
            let elapsed = started.elapsed();
            skip_next_snapshot = elapsed > PREVIEW_INTERVAL;
            let capture_state = app.state::<NativeCaptureState>();
            if capture_state.preview_generation.load(Ordering::SeqCst) != preview_generation {
                return;
            }
            let mut stored = match capture_state.parakeet_live.lock() {
                Ok(stored) if stored.generation == preview_generation => stored,
                _ => return,
            };
            match result {
                Ok(text) if !text.trim().is_empty() => {
                    let text = parakeet_preview_cleanup(&text, &settings, &dictionary);
                    if text.is_empty() {
                        continue;
                    }
                    let text = fluidvoice_preview_reconcile(&stored.previous_full_text, &text);
                    stored.previous_full_text = text.clone();
                    debug_log::append(&format!(
                        "Live Parakeet preview emitted (audio_ms: {}, decode_ms: {})",
                        (sample_count as u64).saturating_mul(1_000)
                            / audio::WHISPER_SAMPLE_RATE as u64,
                        elapsed.as_millis()
                    ));
                    emit_overlay(
                        &app,
                        "recording",
                        tail_characters(&text, preview_char_limit),
                    );
                }
                Ok(_) => debug_log::append(&format!(
                    "Live Parakeet preview was empty (audio_ms: {})",
                    (sample_count as u64).saturating_mul(1_000) / audio::WHISPER_SAMPLE_RATE as u64
                )),
                Err(error) => debug_log::append(&format!(
                    "Live Parakeet preview skipped (audio_ms: {}): {error}",
                    (sample_count as u64).saturating_mul(1_000) / audio::WHISPER_SAMPLE_RATE as u64
                )),
            }
        }
    });
}

/// Drives NVIDIA's cache-aware FastConformer stream. Unlike the Parakeet TDT
/// preview above, this never re-decodes a growing recording: every successful
/// poll sends only the samples captured since the previous poll to the Python
/// sidecar, whose encoder/decoder caches retain the prior context.
fn spawn_live_nemotron_stream(
    app: AppHandle,
    preview_generation: u64,
    runtime_directory: PathBuf,
    script: PathBuf,
    model_directory: PathBuf,
    language: String,
    lookahead_tokens: u8,
    show_preview: bool,
    preview_char_limit: usize,
) {
    tauri::async_runtime::spawn(async move {
        const POLL_INTERVAL: Duration = Duration::from_millis(80);
        loop {
            tokio::time::sleep(POLL_INTERVAL).await;
            let capture_state = app.state::<NativeCaptureState>();
            if capture_state.preview_generation.load(Ordering::SeqCst) != preview_generation {
                return;
            }
            let captured = capture_state.capture.lock().ok().and_then(|capture| {
                capture
                    .as_ref()
                    .and_then(|capture| capture.snapshot_all().ok())
            });
            let Some(captured) = captured else {
                return;
            };
            let samples = match audio::mono_resample_for_whisper(captured) {
                Ok(samples) => samples,
                Err(error) => {
                    debug_log::append(&format!(
                        "Live Nemotron preview could not resample audio: {error}"
                    ));
                    continue;
                }
            };
            let result = {
                let mut live = capture_state.nemotron_live.lock().await;
                if live.generation != preview_generation {
                    return;
                }
                if let Some(error) = &live.start_error {
                    Err(error.clone())
                } else {
                    if live.server.is_none() {
                        let server = nemotron::Server::launch(
                            &nemotron::python_path(&runtime_directory),
                            &script,
                            &model_directory,
                        )
                        .await;
                        match server {
                            Ok(server) => live.server = Some(server),
                            Err(error) => {
                                live.start_error = Some(error.clone());
                                return debug_log::append(&format!(
                                    "Live Nemotron stream could not start: {error}"
                                ));
                            }
                        }
                    }
                    if !live.session_started {
                        let started = live
                            .server
                            .as_mut()
                            .expect("Nemotron server was just initialized")
                            .start(&language, lookahead_tokens)
                            .await;
                        if let Err(error) = started {
                            live.start_error = Some(error.clone());
                            Err(error)
                        } else {
                            live.session_started = true;
                            Ok(())
                        }
                    } else {
                        Ok(())
                    }
                    .and_then(|()| {
                        let start = live.fed_samples.min(samples.len());
                        if start == samples.len() {
                            return Ok(None);
                        }
                        Ok(Some(samples[start..].to_vec()))
                    })
                }
            };
            let delta = match result {
                Ok(Some(delta)) => delta,
                Ok(None) => continue,
                Err(error) => {
                    debug_log::append(&format!("Live Nemotron stream stopped: {error}"));
                    let mut live = capture_state.nemotron_live.lock().await;
                    if live.generation == preview_generation {
                        live.start_error = Some(error);
                    }
                    return;
                }
            };
            if !begin_preview(&capture_state, preview_generation) {
                continue;
            }
            let partial = {
                let mut live = capture_state.nemotron_live.lock().await;
                if live.generation != preview_generation {
                    drop(live);
                    finish_preview(&capture_state, preview_generation);
                    return;
                }
                let result = live
                    .server
                    .as_mut()
                    .expect("Nemotron server was initialized before appending")
                    .append(&delta)
                    .await;
                if result.is_ok() {
                    live.fed_samples = samples.len();
                }
                result
            };
            finish_preview(&capture_state, preview_generation);
            match partial {
                Ok(text) if show_preview && !text.trim().is_empty() => {
                    if capture_state.preview_generation.load(Ordering::SeqCst) == preview_generation
                    {
                        emit_overlay(
                            &app,
                            "recording",
                            tail_characters(&text, preview_char_limit),
                        );
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    debug_log::append(&format!("Live Nemotron audio append failed: {error}"));
                    let mut live = capture_state.nemotron_live.lock().await;
                    if live.generation == preview_generation {
                        live.start_error = Some(error);
                    }
                    return;
                }
            }
        }
    });
}

/// FluidVoice applies deterministic cleanup to each v3 snapshot, but does not
/// run its optional AI enhancement until recording has stopped.
fn parakeet_preview_cleanup(
    text: &str,
    settings: &Settings,
    dictionary: &[DictionaryEntry],
) -> String {
    let formatting = output_formatting(settings);
    prepare_dictation_text(
        text,
        &formatting,
        dictionary,
        DictationCleanupStyle::FluidVoiceParakeet,
    )
}

/// Direct Rust equivalent of FluidVoice's `smartDiffUpdate`. Its source
/// calculates a common prefix to avoid a visibly disruptive replacement; the
/// returned hypothesis remains the newest complete snapshot, which is why the
/// overlay can correct itself as the capture grows.
fn fluidvoice_preview_reconcile(previous: &str, current: &str) -> String {
    let previous = previous.trim();
    let current = current.trim();
    if previous.is_empty() || current.is_empty() {
        return current.to_owned();
    }
    let previous_words = previous.split_whitespace().collect::<Vec<_>>();
    let current_words = current.split_whitespace().collect::<Vec<_>>();
    let shared_prefix = previous_words
        .iter()
        .zip(&current_words)
        .take_while(|(previous, current)| preview_words_match(previous, current))
        .count();
    if shared_prefix > previous_words.len() / 2 {
        current_words.join(" ")
    } else {
        current.to_owned()
    }
}

fn spawn_live_cloud_preview(
    app: AppHandle,
    preview_generation: u64,
    profile: provider::AiProviderProfile,
    api_key: Option<String>,
    model: String,
    language: String,
    preview_char_limit: usize,
) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(2_500)).await;
            let capture_state = app.state::<NativeCaptureState>();
            if capture_state.preview_generation.load(Ordering::SeqCst) != preview_generation {
                return;
            }
            let captured = capture_state.capture.lock().ok().and_then(|capture| {
                capture
                    .as_ref()
                    .and_then(|capture| capture.snapshot_recent(Duration::from_secs(20)).ok())
            });
            let Some(captured) = captured else {
                return;
            };
            if captured.duration_ms < 1_000 {
                continue;
            }
            let wav = match audio::mono_resample_for_whisper(captured)
                .and_then(|samples| audio::wav_bytes_from_16khz_mono(&samples))
            {
                Ok(wav) => wav,
                Err(_) => continue,
            };
            if !begin_preview(&capture_state, preview_generation) {
                continue;
            }
            let partial = tokio::time::timeout(
                Duration::from_secs(20),
                provider::transcribe_openai_compatible_audio(
                    &profile,
                    api_key.as_deref(),
                    &model,
                    &language,
                    wav,
                ),
            )
            .await
            .ok()
            .and_then(Result::ok);
            finish_preview(&capture_state, preview_generation);
            if let Some(partial) = partial.filter(|partial| !partial.trim().is_empty()) {
                let capture_state = app.state::<NativeCaptureState>();
                if capture_state.preview_generation.load(Ordering::SeqCst) == preview_generation {
                    emit_overlay(
                        &app,
                        "recording",
                        tail_characters(&partial, preview_char_limit),
                    );
                }
            }
        }
    });
}

fn spawn_live_apple_speech_preview(
    app: AppHandle,
    preview_generation: u64,
    language: String,
    custom_words: Vec<String>,
    preview_char_limit: usize,
) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let capture_state = app.state::<NativeCaptureState>();
            if capture_state.preview_generation.load(Ordering::SeqCst) != preview_generation {
                return;
            }
            let captured = capture_state.capture.lock().ok().and_then(|capture| {
                capture
                    .as_ref()
                    .and_then(|capture| capture.snapshot_recent(Duration::from_secs(20)).ok())
            });
            let Some(captured) = captured else {
                return;
            };
            if captured.duration_ms < 1_000 {
                continue;
            }
            let samples = match audio::mono_resample_for_whisper(captured) {
                Ok(samples) => samples,
                Err(_) => continue,
            };
            if !begin_preview(&capture_state, preview_generation) {
                continue;
            }
            let language = language.clone();
            let custom_words = custom_words.clone();
            let partial = tokio::time::timeout(
                Duration::from_secs(30),
                tauri::async_runtime::spawn_blocking(move || {
                    apple_speech::transcribe_samples(&samples, &language, &custom_words)
                }),
            )
            .await
            .ok()
            .and_then(Result::ok)
            .and_then(Result::ok);
            finish_preview(&capture_state, preview_generation);
            if let Some(partial) = partial.filter(|partial| !partial.trim().is_empty()) {
                let capture_state = app.state::<NativeCaptureState>();
                if capture_state.preview_generation.load(Ordering::SeqCst) == preview_generation {
                    emit_overlay(
                        &app,
                        "recording",
                        tail_characters(&partial, preview_char_limit),
                    );
                }
            }
        }
    });
}

/// FluidVoice's final v3 path always decodes the entire recording through a
/// separate final manager. Keep the same full-audio contract here: preview
/// snapshots never decide the final segmentation or contribute text. When the
/// user enables vocabulary boosting, use it only for this final pass.
fn transcribe_parakeet_final(
    samples: &[f32],
    model_path: &Path,
    vocabulary: &[String],
) -> Result<String, String> {
    parakeet::transcribe_samples_with_vocabulary(samples, model_path, vocabulary)
}

/// Flushes the active Nemotron stream with the capture tail that the 80 ms
/// polling task has not observed yet. When a user stops immediately after
/// recording starts, this lazily creates the same stream and feeds the whole
/// recording, so final transcription never depends on preview timing.
async fn finish_nemotron_live(
    capture_state: &NativeCaptureState,
    recording_generation: u64,
    samples: &[f32],
    runtime_directory: &Path,
    script: &Path,
    model_directory: &Path,
    language: &str,
    lookahead_tokens: u8,
) -> Result<String, String> {
    let mut live = capture_state.nemotron_live.lock().await;
    if let Some(error) = live.start_error.take() {
        return Err(format!("Nemotron CUDA stream could not start: {error}"));
    }
    if live.generation != recording_generation {
        live.generation = recording_generation;
        live.fed_samples = 0;
        live.session_started = false;
    }
    if live.server.is_none() {
        live.server = Some(
            nemotron::Server::launch(
                &nemotron::python_path(runtime_directory),
                script,
                model_directory,
            )
            .await?,
        );
    }
    if !live.session_started {
        live.server
            .as_mut()
            .expect("Nemotron server was initialized")
            .start(language, lookahead_tokens)
            .await?;
        live.session_started = true;
    }
    let start = live.fed_samples.min(samples.len());
    let result = live
        .server
        .as_mut()
        .expect("Nemotron streaming session was initialized")
        .finish(&samples[start..])
        .await;
    live.fed_samples = 0;
    live.session_started = false;
    live.generation = 0;
    result
}

/// Prompt and engine tests must not deliver or retain ordinary dictation.
/// Keeping this policy in the backend prevents a frontend mode regression from
/// typing, saving, or counting a test recording.
fn is_isolated_dictation_test(prompt_test: bool, engine_test: bool) -> bool {
    prompt_test || engine_test
}

#[tauri::command]
async fn stop_native_dictation(
    app: AppHandle,
    state: State<'_, AppState>,
    capture_state: State<'_, NativeCaptureState>,
    prompt_mode: bool,
    instruction_mode: Option<DictationMode>,
    prompt_test_mode: Option<bool>,
    prompt_test_prompt: Option<String>,
    engine_test_mode: Option<bool>,
) -> Result<NativeTranscriptionResult, String> {
    let recording_generation = capture_state
        .preview_generation
        .fetch_add(1, Ordering::SeqCst);
    let capture = capture_state
        .capture
        .lock()
        .map_err(|_| "Audio capture lock was poisoned".to_string())?
        .take()
        .ok_or("Dictation is not recording")?;
    // Once this recording releases the microphone, re-prewarm the next one off
    // the hotkey path (fires on every exit path below, including errors).
    let _refresh_prewarm = RefreshCapturePrewarmWhenDropped { app: app.clone() };
    let capture_started_at = capture_state
        .capture_started_at
        .lock()
        .map_err(|_| "Dictation capture timing lock was poisoned".to_string())?
        .take();
    let finalizing = capture_state
        .session
        .lock()
        .map_err(|_| "Dictation session lock was poisoned".to_string())?
        .begin_finalizing(recording_generation);
    if !finalizing {
        return Err("Dictation session was superseded before finalization".into());
    }
    let _finish_session = FinishDictationSession {
        coordinator: Arc::clone(&capture_state.session),
        id: recording_generation,
    };
    let context = capture_state
        .context
        .lock()
        .map_err(|_| "Dictation context lock was poisoned".to_string())?
        .clone();
    let _reset_context = ResetDictationContextWhenDropped {
        context: &capture_state.context,
    };
    update_tray_status(&app, TrayVisualState::Processing);
    let _reset_tray = ResetTrayWhenDropped { app: app.clone() };
    let (captured, capture_health) = capture.finish_with_health()?;
    let wall_duration_ms = capture_started_at
        .map(|started_at| started_at.elapsed().as_millis() as u64)
        .unwrap_or(captured.duration_ms);
    debug_log::append(&format!(
        "Capture health (session: {recording_generation}, wall_duration_ms: {wall_duration_ms}, canonical_duration_ms: {}, callbacks: {}, input_samples: {}, accepted_samples: {}, dropped_samples: {}, overflow_blocks: {}, ring_high_water_samples: {}, canonical_samples: {}, discontinuities: {}, capture_delay_ns: {}, stream_errors: {})",
        captured.duration_ms,
        capture_health.callback_blocks,
        capture_health.input_samples,
        capture_health.accepted_samples,
        capture_health.dropped_samples,
        capture_health.overflow_blocks,
        capture_health.ring_high_water_samples,
        capture_health.canonical_samples,
        capture_health.discontinuities,
        capture_health.latest_capture_delay_ns,
        capture_health.stream_errors,
    ));
    if capture_health.dropped_samples != 0 {
        return Err(format!(
            "Microphone capture overflowed and lost {} samples. Try closing CPU-intensive applications or selecting another input device, then record again.",
            capture_health.dropped_samples
        ));
    }
    if capture_health.stream_errors != 0 {
        return Err("The microphone reported a device error while recording. Reconnect or reselect the input device, then record again.".into());
    }
    let duration_ms = captured.duration_ms;
    let (settings, custom_words) = {
        let database = state
            .database
            .lock()
            .map_err(|_| "Voxide data lock was poisoned".to_string())?;
        let settings = database.settings.clone();
        let custom_words = active_recognition_vocabulary(&settings, &database.custom_words);
        (settings, custom_words)
    };
    let prompt_test_prompt = prompt_test_prompt
        .as_deref()
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .map(str::to_owned);
    let is_prompt_test = prompt_test_mode.unwrap_or(false);
    let is_engine_test = engine_test_mode.unwrap_or(false);
    let is_isolated_test = is_isolated_dictation_test(is_prompt_test, is_engine_test);
    let _overlay_cleanup = OverlayHideOnDrop(app.clone());
    if settings.enable_streaming_preview {
        emit_overlay(&app, "processing", "Transcribing…");
    }
    let should_save_audio_history = !is_isolated_test
        && settings.save_transcription_history
        && settings.audio_history_enabled
        && settings.audio_history_budget_gb > 0.0;
    let mut samples = audio::mono_resample_for_whisper(captured)?;
    let audio_history_wav = should_save_audio_history
        .then(|| audio::wav_bytes_from_16khz_mono(&samples))
        .transpose()?;
    if samples.is_empty() {
        return Ok(NativeTranscriptionResult {
            text: String::new(),
            raw_text: String::new(),
            duration_ms,
            audio_file: None,
            audio_model: None,
            was_ai_processed: false,
            processing_model: None,
            ai_processing_error: None,
            source_application: context.source_application,
            source_window_title: context.source_window_title,
            inserted_into_active_application: false,
        });
    }
    audio::pad_short_transcription_samples(&mut samples);

    let transcription_started = Instant::now();
    let EngineFinalTranscript {
        text: raw_text,
        whisper_timings,
    } = settings
        .selected_voice_engine
        .transcribe_live_final(
            &state,
            &capture_state,
            &settings,
            recording_generation,
            samples,
            custom_words,
        )
        .await?;
    let transcription_latency_ms = transcription_started.elapsed().as_millis() as u64;
    let cleanup_style = if settings.selected_voice_engine.is_parakeet() {
        DictationCleanupStyle::FluidVoiceParakeet
    } else {
        DictationCleanupStyle::Standard
    };
    let post_processing_started = Instant::now();
    let post_processing = if is_engine_test
        || matches!(
            instruction_mode,
            Some(DictationMode::Command | DictationMode::Rewrite)
        ) {
        PostProcessOutcome {
            text: deterministic_dictation_cleanup(&state, &raw_text, cleanup_style)?,
            ai_fallback_error: None,
            was_ai_processed: false,
            processing_model: None,
        }
    } else {
        let show_refinement_preview =
            settings.enable_streaming_preview && (settings.ai_enhancement_enabled || prompt_mode);
        let preview_char_limit = settings.transcription_preview_char_limit;
        if show_refinement_preview {
            emit_overlay(&app, "processing", "Refining…");
        }
        let refinement_app = app.clone();
        let mut refinement_text = String::new();
        let mut last_refinement_emit = Instant::now() - Duration::from_secs(1);
        let mut on_delta = move |delta: &str| {
            if delta.is_empty() {
                return;
            }
            refinement_text.push_str(delta);
            if show_refinement_preview
                && last_refinement_emit.elapsed() >= Duration::from_millis(33)
            {
                last_refinement_emit = Instant::now();
                emit_overlay(
                    &refinement_app,
                    "processing",
                    tail_characters(&refinement_text, preview_char_limit),
                );
            }
        };
        post_process_dictation_outcome(
            &state,
            raw_text.clone(),
            prompt_mode || is_prompt_test,
            context.source_application.as_deref(),
            context.source_window_title.as_deref(),
            context.prompt_profile_id.as_deref(),
            prompt_test_prompt.as_deref(),
            cleanup_style,
            &mut on_delta,
        )
        .await?
    };
    if !is_isolated_test && settings.notify_ai_processing_failures {
        if let Some(error) = post_processing.ai_fallback_error.as_deref() {
            notify_ai_fallback(&app, error);
        }
    }
    let mut text = post_processing.text;
    if instruction_mode.is_none() && !is_isolated_test {
        text = formatting::apply_continuous_dictation_formatting(
            &text,
            &context.preceding_text,
            settings.continuous_dictation_spacing_enabled,
            settings.context_aware_capitalization_enabled,
        );
    }
    text = formatting::apply_terminal_literal_autocomplete_spacing(
        &text,
        settings.literal_dictation_formatting_enabled,
        context.source_application.as_deref(),
        context.source_window_title.as_deref(),
    );
    let post_processing_latency_ms = post_processing_started.elapsed().as_millis() as u64;
    if settings.enable_streaming_preview {
        emit_overlay(
            &app,
            "complete",
            tail_characters(&text, settings.transcription_preview_char_limit),
        );
    }
    let insertion_started = Instant::now();
    let inserted_into_active_application =
        if !is_isolated_test && settings.type_into_active_application {
            typing::type_into_active_application(&text, settings.text_insertion_mode)?;
            true
        } else {
            false
        };
    let insertion_latency_ms = insertion_started.elapsed().as_millis() as u64;
    if inserted_into_active_application
        && (settings.continuous_dictation_spacing_enabled
            || settings.context_aware_capitalization_enabled)
    {
        capture_state
            .continuous_context
            .lock()
            .map_err(|_| "Continuous dictation context lock was poisoned".to_string())?
            .record(
                context.source_application.clone(),
                context.source_window_title.clone(),
                text.clone(),
            );
    }

    // The reference attaches the recording only after the dictation has made
    // it through output delivery. Waiting until that point avoids leaving an
    // unreferenced WAV behind if desktop text insertion fails.
    let audio_file = if should_save_audio_history {
        let path = state
            .audio_history_directory()?
            .join(format!("{}.wav", make_id("dictation")));
        fs::write(
            &path,
            audio_history_wav.expect("audio history bytes are prepared when enabled"),
        )
        .map_err(|error| format!("Could not save dictation audio history: {error}"))?;
        Some(path.display().to_string())
    } else {
        None
    };
    let audio_model = audio_file
        .as_ref()
        .map(|_| settings.selected_voice_engine.audio_model_id(&settings));

    if settings.enable_streaming_preview {
        emit_overlay(&app, "hidden", "");
    }
    let mode = match instruction_mode {
        Some(DictationMode::Command) => "command",
        Some(DictationMode::Rewrite) => "rewrite",
        _ => "dictation",
    };
    let analytics = app.state::<analytics::AnalyticsService>();
    if !is_isolated_test {
        analytics.capture(
            "dictation_post_processing_completed",
            settings.share_anonymous_analytics,
            analytics_properties_with(
                &settings,
                [
                    ("latency_ms", Value::from(post_processing_latency_ms)),
                    ("input_chars", Value::from(raw_text.chars().count() as u64)),
                    (
                        "ai_used",
                        Value::Bool(settings.ai_enhancement_enabled || prompt_mode),
                    ),
                    (
                        "fallback",
                        Value::Bool(post_processing.ai_fallback_error.is_some()),
                    ),
                ],
            ),
        );
        analytics.capture(
            "transcription_completed",
            settings.share_anonymous_analytics,
            analytics_properties_with(
                &settings,
                [
                    ("mode", Value::String(mode.into())),
                    (
                        "words_bucket",
                        Value::String(analytics::word_count_bucket(&text).into()),
                    ),
                    (
                        "duration_bucket",
                        Value::String(analytics::milliseconds_bucket(duration_ms).into()),
                    ),
                ],
            ),
        );
    }
    let output_method = if settings.copy_to_clipboard {
        "clipboard"
    } else if inserted_into_active_application {
        "typed"
    } else {
        "history_only"
    };
    if !is_isolated_test {
        let timings = whisper_timings.unwrap_or_default();
        let decode_ms = if matches!(settings.selected_voice_engine, VoiceEngine::Whisper) {
            timings.decode_ms
        } else {
            transcription_latency_ms
        };
        let real_time_factor = if duration_ms == 0 {
            0.0
        } else {
            transcription_latency_ms as f64 / duration_ms as f64
        };
        debug_log::append(&format!(
            "Dictation timing (session: {recording_generation}, capture_wall_ms: {wall_duration_ms}, audio_s: {:.1}, final_ms: {transcription_latency_ms}, rtf: {real_time_factor:.3}, lock_wait_ms: {}, vad_ms: {}, state_ms: {}, decode_ms: {}, post_ms: {}, insert_ms: {})",
            duration_ms as f64 / 1_000.0,
            timings.lock_wait_ms,
            timings.vad_ms,
            timings.state_ms,
            decode_ms,
            post_processing_latency_ms,
            insertion_latency_ms,
        ));
        debug_log::append(&format!(
            "Dictation completed (engine: {:?}, duration_ms: {duration_ms}, ai_processed: {}, output: {output_method})",
            settings.selected_voice_engine, post_processing.was_ai_processed
        ));
        analytics.capture(
            "output_delivered",
            settings.share_anonymous_analytics,
            analytics_properties_with(
                &settings,
                [
                    ("mode", Value::String(mode.into())),
                    ("method", Value::String(output_method.into())),
                ],
            ),
        );
    }
    Ok(NativeTranscriptionResult {
        text,
        raw_text,
        duration_ms,
        audio_file,
        audio_model,
        was_ai_processed: post_processing.was_ai_processed,
        processing_model: post_processing.processing_model,
        ai_processing_error: post_processing.ai_fallback_error,
        source_application: context.source_application,
        source_window_title: context.source_window_title,
        inserted_into_active_application,
    })
}

fn enforce_audio_history_budget(state: &AppState, budget_gb: f64) -> Result<(), String> {
    let directory = state.audio_history_directory()?;
    let budget_bytes = (budget_gb * 1_073_741_824.0).round() as u64;
    let oldest_first_file_names = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .dictation_history
        .iter()
        .rev()
        .filter_map(|entry| entry.audio_file.as_deref())
        .filter_map(audio_history_file_name)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let referenced_file_names = oldest_first_file_names
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    remove_unreferenced_audio_files(&directory, &referenced_file_names)?;
    let removed_files = prune_audio_directory(&directory, budget_bytes, &oldest_first_file_names)?;
    if removed_files.is_empty() {
        return Ok(());
    }
    let removed_file_names = removed_files
        .iter()
        .filter_map(|path| path.file_name())
        .filter_map(|name| name.to_str())
        .map(str::to_owned)
        .collect::<HashSet<_>>();
    state.update(|database| {
        for entry in &mut database.dictation_history {
            if entry
                .audio_file
                .as_deref()
                .and_then(audio_history_file_name)
                .is_some_and(|file_name| removed_file_names.contains(file_name))
            {
                entry.audio_file = None;
                entry.audio_model = None;
            }
        }
    })?;
    Ok(())
}

fn audio_history_file_name(stored_path: &str) -> Option<&str> {
    Path::new(stored_path.trim())
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .filter(|file_name| file_name.to_ascii_lowercase().ends_with(".wav"))
}

fn remove_unreferenced_audio_files(
    directory: &std::path::Path,
    referenced_file_names: &HashSet<String>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("Could not inspect audio history: {error}"))?
        .filter_map(Result::ok)
    {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            && !referenced_file_names.contains(file_name)
        {
            fs::remove_file(&path)
                .map_err(|error| format!("Could not remove unreferenced audio history: {error}"))?;
        }
    }
    Ok(())
}

fn prune_audio_directory(
    directory: &std::path::Path,
    budget_bytes: u64,
    oldest_first_file_names: &[String],
) -> Result<Vec<PathBuf>, String> {
    let mut files = fs::read_dir(directory)
        .map_err(|error| format!("Could not inspect audio history: {error}"))?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let is_wav = path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"));
            if !is_wav {
                return None;
            }
            let metadata = entry.metadata().ok()?;
            let file_name = path.file_name()?.to_str()?.to_owned();
            Some((file_name, (path, metadata.len())))
        })
        .collect::<HashMap<_, _>>();
    let mut total_bytes = files.values().map(|(_, size)| *size).sum::<u64>();
    let mut removed = Vec::new();
    // The reference store keeps history newest-first and removes entries in
    // reverse order. Use that persisted history order rather than filesystem
    // timestamps, which can change when users migrate or restore recordings.
    for file_name in oldest_first_file_names {
        if total_bytes <= budget_bytes {
            break;
        }
        let Some((path, size)) = files.remove(file_name) else {
            continue;
        };
        fs::remove_file(&path)
            .map_err(|error| format!("Could not prune old audio history: {error}"))?;
        total_bytes = total_bytes.saturating_sub(size);
        removed.push(path);
    }
    Ok(removed)
}

fn saved_audio_path(state: &AppState, stored_path: &str) -> Result<PathBuf, String> {
    let stored_path = PathBuf::from(stored_path.trim());
    let file_name = stored_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .filter(|file_name| !file_name.is_empty())
        .ok_or("The saved audio file name is invalid")?;
    if !file_name.to_ascii_lowercase().ends_with(".wav") {
        return Err("The saved audio file is not a WAV recording".into());
    }
    Ok(state.audio_history_directory()?.join(file_name))
}

fn delete_saved_audio_file(state: &AppState, stored_path: &str) -> Result<(), String> {
    let path = saved_audio_path(state, stored_path)?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("Could not delete saved dictation audio: {error}")),
    }
}

fn wav_recording_format(path: &std::path::Path) -> Result<(u32, u16), String> {
    let mut header = [0_u8; 44];
    fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|error| format!("Could not read saved WAV audio: {error}"))?;
    if &header[..4] != b"RIFF" || &header[8..12] != b"WAVE" || &header[12..16] != b"fmt " {
        return Err("The saved audio is not a valid WAV recording".into());
    }
    let channels = u16::from_le_bytes([header[22], header[23]]);
    let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
    if channels == 0 || sample_rate == 0 {
        return Err("The saved WAV audio has an invalid format".into());
    }
    Ok((sample_rate, channels))
}

fn audio_history_archive_entries(
    state: &AppState,
    entry_id: Option<&str>,
) -> Result<Vec<(String, PathBuf, AudioHistoryManifestRow)>, String> {
    let entries = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .dictation_history
        .clone();
    let mut archive_entries = Vec::new();
    for entry in entries {
        if entry_id.is_some_and(|id| entry.id != id) {
            continue;
        }
        let Some(stored_audio) = entry.audio_file.as_deref() else {
            continue;
        };
        let source = saved_audio_path(state, stored_audio)?;
        if !source.is_file() {
            continue;
        }
        let (sample_rate, channels) = wav_recording_format(&source)?;
        let id_suffix = entry
            .id
            .chars()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>();
        let file_name = format!(
            "{}_{}.wav",
            entry.created_at.format("%Y-%m-%dT%H-%M-%SZ"),
            id_suffix
        );
        let relative_path = format!("audio/{file_name}");
        let raw_transcript = entry.raw_text.clone().unwrap_or_else(|| entry.text.clone());
        archive_entries.push((
            relative_path.clone(),
            source,
            AudioHistoryManifestRow {
                audio: relative_path,
                text: raw_transcript.clone(),
                raw_transcript,
                final_transcript: entry.text,
                timestamp: entry.created_at.to_rfc3339(),
                duration_milliseconds: entry.duration_ms.unwrap_or_default(),
                sample_rate,
                channels,
                app: entry.source_application.unwrap_or_default(),
                model: entry.audio_model.unwrap_or_default(),
            },
        ));
    }
    archive_entries.sort_by(|left, right| left.2.timestamp.cmp(&right.2.timestamp));
    Ok(archive_entries)
}

fn export_audio_history_archive(
    state: &AppState,
    entry_id: Option<&str>,
    destination: &str,
) -> Result<usize, String> {
    let destination = PathBuf::from(destination.trim());
    if destination.as_os_str().is_empty()
        || !destination
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("zip"))
    {
        return Err("Choose a ZIP destination for the audio export".into());
    }
    let entries = audio_history_archive_entries(state, entry_id)?;
    if entries.is_empty() {
        return Err("No saved dictation audio is available to export".into());
    }
    let temporary = PathBuf::from(format!("{}.tmp", destination.display()));
    let result = (|| -> Result<(), String> {
        let file = fs::File::create(&temporary)
            .map_err(|error| format!("Could not create the audio export: {error}"))?;
        let mut archive = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        let manifest = entries
            .iter()
            .map(|(_, _, row)| serde_json::to_string(row))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("Could not encode the audio export manifest: {error}"))?
            .join("\n");
        archive
            .start_file("manifest.jsonl", options)
            .map_err(|error| format!("Could not create the audio export manifest: {error}"))?;
        archive
            .write_all(format!("{manifest}\n").as_bytes())
            .map_err(|error| format!("Could not write the audio export manifest: {error}"))?;
        for (relative_path, source, _) in &entries {
            archive
                .start_file(relative_path, options)
                .map_err(|error| format!("Could not create an audio archive entry: {error}"))?;
            let mut input = fs::File::open(source)
                .map_err(|error| format!("Could not read saved audio for export: {error}"))?;
            std::io::copy(&mut input, &mut archive)
                .map_err(|error| format!("Could not add saved audio to the export: {error}"))?;
        }
        archive
            .finish()
            .map_err(|error| format!("Could not finalize the audio export: {error}"))?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if destination.exists() {
        fs::remove_file(&destination)
            .map_err(|error| format!("Could not replace the selected audio export: {error}"))?;
    }
    fs::rename(&temporary, &destination)
        .map_err(|error| format!("Could not finalize the audio export: {error}"))?;
    Ok(entries.len())
}

#[tauri::command]
fn export_audio_history(state: State<'_, AppState>, destination: String) -> Result<usize, String> {
    export_audio_history_archive(&state, None, &destination)
}

#[tauri::command]
fn export_dictation_audio_pair(
    state: State<'_, AppState>,
    id: String,
    destination: String,
) -> Result<usize, String> {
    let id = id.trim();
    if id.is_empty() {
        return Err("Choose a dictation with saved audio to export".into());
    }
    export_audio_history_archive(&state, Some(id), &destination)
}

#[tauri::command]
fn delete_saved_audio(state: State<'_, AppState>) -> Result<usize, String> {
    let audio_files = state.update(|database| {
        let audio_files = database
            .dictation_history
            .iter()
            .filter_map(|entry| entry.audio_file.clone())
            .collect::<HashSet<_>>();
        for entry in &mut database.dictation_history {
            entry.audio_file = None;
            entry.audio_model = None;
        }
        audio_files
    })?;
    let count = audio_files.len();
    for audio_file in audio_files {
        delete_saved_audio_file(&state, &audio_file)?;
    }
    Ok(count)
}

#[tauri::command]
fn cancel_native_dictation(
    app: AppHandle,
    capture_state: State<'_, NativeCaptureState>,
) -> Result<bool, String> {
    let recording_generation = capture_state
        .preview_generation
        .fetch_add(1, Ordering::SeqCst);
    let capture = capture_state
        .capture
        .lock()
        .map_err(|_| "Audio capture lock was poisoned".to_string())?
        .take();
    let cancelled = capture.is_some();
    if cancelled {
        let _ = capture_state
            .capture_started_at
            .lock()
            .map(|mut started_at| started_at.take());
        let _ = capture_state
            .session
            .lock()
            .map(|mut session| session.cancel(recording_generation));
        let nemotron_live = Arc::clone(&capture_state.nemotron_live);
        tauri::async_runtime::spawn(async move {
            let mut live = nemotron_live.lock().await;
            if live.session_started {
                if let Some(server) = live.server.as_mut() {
                    if let Err(error) = server.abort().await {
                        debug_log::append(&format!(
                            "Could not cancel the Nemotron stream cleanly: {error}"
                        ));
                        server.terminate();
                    }
                }
            }
            live.fed_samples = 0;
            live.session_started = false;
            live.generation = 0;
            live.start_error = None;
        });
        reset_dictation_context(&capture_state);
        emit_overlay(&app, "hidden", "");
        update_tray_status(&app, TrayVisualState::Ready);
    }
    Ok(cancelled)
}

#[tauri::command]
fn paste_last_transcription(state: State<'_, AppState>) -> Result<String, String> {
    let database = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?;
    let text = database
        .dictation_history
        .first()
        .map(|entry| entry.text.clone())
        .ok_or("There is no saved transcription to paste")?;
    let insertion_mode = database.settings.text_insertion_mode;
    drop(database);
    typing::type_into_active_application(&text, insertion_mode)?;
    Ok(text)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LocalApiConfiguration {
    enabled: bool,
    port: u16,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalApiStatus {
    enabled: bool,
    port: Option<u16>,
    url: Option<String>,
}

#[tauri::command]
async fn configure_local_api(
    app: AppHandle,
    state: State<'_, AppState>,
    control: State<'_, local_api::LocalApiControl>,
    configuration: LocalApiConfiguration,
) -> Result<LocalApiStatus, String> {
    if configuration.enabled && configuration.port == 0 {
        return Err("A local API port between 1 and 65535 is required".into());
    }
    state.update(|database| {
        database.settings.local_api_enabled = configuration.enabled;
        database.settings.local_api_port = configuration.port;
    })?;
    local_api::stop(&control)?;
    let port = if configuration.enabled {
        Some(local_api::start(&control, app, configuration.port).await?)
    } else {
        None
    };
    Ok(LocalApiStatus {
        enabled: configuration.enabled,
        port,
        url: port.map(|port| format!("http://127.0.0.1:{port}")),
    })
}

#[tauri::command]
fn local_api_status(
    state: State<'_, AppState>,
    control: State<'_, local_api::LocalApiControl>,
) -> Result<LocalApiStatus, String> {
    let enabled = state
        .database
        .lock()
        .map_err(|_| "Voxide data lock was poisoned".to_string())?
        .settings
        .local_api_enabled;
    let port = local_api::running_port(&control)?;
    Ok(LocalApiStatus {
        enabled,
        port,
        url: port.map(|port| format!("http://127.0.0.1:{port}")),
    })
}

#[tauri::command]
fn set_primary_hotkey(
    app: AppHandle,
    state: State<'_, AppState>,
    registry: State<'_, HotkeyRegistry>,
    hotkey: String,
) -> Result<String, String> {
    let settings = state.update(|database| {
        database.settings.primary_dictation_hotkey = hotkey.clone();
        database.settings.clone()
    })?;
    apply_hotkeys(&app, &registry, &settings)?;
    Ok(hotkey)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HotkeyConfiguration {
    primary_dictation_hotkey: String,
    secondary_dictation_hotkey: Option<String>,
    prompt_mode_hotkey: Option<String>,
    prompt_mode_selected_prompt_id: Option<String>,
    #[serde(default)]
    prompt_shortcut_assignments: Vec<PromptShortcutAssignment>,
    command_mode_hotkey: Option<String>,
    rewrite_mode_hotkey: Option<String>,
    cancel_recording_hotkey: Option<String>,
    paste_last_transcription_hotkey: Option<String>,
    hotkey_activation_mode: HotkeyActivationMode,
}

#[tauri::command]
fn configure_hotkeys(
    app: AppHandle,
    state: State<'_, AppState>,
    registry: State<'_, HotkeyRegistry>,
    configuration: HotkeyConfiguration,
) -> Result<Settings, String> {
    let settings = state.update(|database| {
        database.settings.primary_dictation_hotkey = configuration.primary_dictation_hotkey;
        database.settings.secondary_dictation_hotkey =
            sanitize_optional_hotkey(configuration.secondary_dictation_hotkey);
        database.settings.prompt_mode_hotkey =
            sanitize_optional_hotkey(configuration.prompt_mode_hotkey);
        database.settings.prompt_mode_selected_prompt_id =
            configuration.prompt_mode_selected_prompt_id.and_then(|id| {
                let id = id.trim().to_owned();
                (!id.is_empty()).then_some(id)
            });
        database.settings.prompt_shortcut_assignments = configuration.prompt_shortcut_assignments;
        database.settings.command_mode_hotkey =
            sanitize_optional_hotkey(configuration.command_mode_hotkey);
        database.settings.rewrite_mode_hotkey =
            sanitize_optional_hotkey(configuration.rewrite_mode_hotkey);
        database.settings.cancel_recording_hotkey =
            sanitize_optional_hotkey(configuration.cancel_recording_hotkey);
        database.settings.paste_last_transcription_hotkey =
            sanitize_optional_hotkey(configuration.paste_last_transcription_hotkey);
        database.settings.hotkey_activation_mode = configuration.hotkey_activation_mode;
        normalize_database(database);
        database.settings.clone()
    })?;
    apply_hotkeys(&app, &registry, &settings)?;
    Ok(settings)
}

#[tauri::command]
fn hotkey_backend_status(app: AppHandle) -> HotkeyBackendStatus {
    #[cfg(target_os = "linux")]
    if portal_hotkeys::is_wayland_session() {
        return portal_hotkeys::current_status(&app);
    }
    let _ = &app;
    HotkeyBackendStatus {
        backend: "native".into(),
        state: "active".into(),
        detail: None,
    }
}

fn sanitize_optional_hotkey(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim().to_owned();
        (!value.is_empty()).then_some(value)
    })
}

fn configured_hotkeys(settings: &Settings) -> Vec<(String, HotkeyAction)> {
    let optional = |hotkey: &Option<String>, action| {
        hotkey
            .as_ref()
            .filter(|hotkey| !hotkey.trim().is_empty())
            .map(|hotkey| (hotkey.clone(), action))
    };
    std::iter::once((
        settings.primary_dictation_hotkey.clone(),
        HotkeyAction::Dictate,
    ))
    .chain(optional(
        &settings.secondary_dictation_hotkey,
        HotkeyAction::Dictate,
    ))
    .chain(optional(&settings.prompt_mode_hotkey, HotkeyAction::Prompt))
    .chain(
        settings
            .prompt_shortcut_assignments
            .iter()
            .map(|assignment| {
                (
                    assignment.hotkey.clone(),
                    HotkeyAction::PromptProfile(assignment.prompt_profile_id.clone()),
                )
            }),
    )
    .chain(optional(
        &settings.command_mode_hotkey,
        HotkeyAction::Command,
    ))
    .chain(optional(
        &settings.rewrite_mode_hotkey,
        HotkeyAction::Rewrite,
    ))
    .chain(optional(
        &settings.cancel_recording_hotkey,
        HotkeyAction::Cancel,
    ))
    .chain(optional(
        &settings.paste_last_transcription_hotkey,
        HotkeyAction::PasteLast,
    ))
    .collect()
}

fn apply_hotkeys(
    app: &AppHandle,
    registry: &HotkeyRegistry,
    settings: &Settings,
) -> Result<(), String> {
    let parsed = configured_hotkeys(settings)
        .into_iter()
        .map(|(hotkey, action)| {
            let shortcut = hotkey
                .parse::<Shortcut>()
                .map_err(|error| format!("Invalid shortcut '{hotkey}': {error}"))?;
            Ok((shortcut, action))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let unique_shortcuts = parsed
        .iter()
        .map(|(shortcut, _)| shortcut.to_string())
        .collect::<HashSet<_>>();
    if unique_shortcuts.len() != parsed.len() {
        return Err("Each Voxide shortcut must use a unique key combination".into());
    }

    #[cfg(target_os = "linux")]
    let use_portal = portal_hotkeys::is_wayland_session();
    #[cfg(not(target_os = "linux"))]
    let use_portal = false;

    if use_portal {
        // X11-based global shortcut grabs cannot observe keys pressed in
        // native Wayland windows, so Wayland sessions bind through the XDG
        // GlobalShortcuts portal instead. Binding needs user approval and
        // completes asynchronously; progress is reported through the
        // voxide-hotkey-backend event and hotkey_backend_status.
        #[cfg(target_os = "linux")]
        portal_hotkeys::apply(app, configured_hotkeys(settings));
    } else {
        let manager = app.global_shortcut();
        manager
            .unregister_all()
            .map_err(|error| format!("Could not clear existing shortcuts: {error}"))?;
        for (shortcut, _) in &parsed {
            manager
                .register(*shortcut)
                .map_err(|error| format!("Could not register shortcut '{shortcut}': {error}"))?;
        }
    }
    let mut bindings = registry
        .bindings
        .lock()
        .map_err(|_| "Hotkey registry lock was poisoned".to_string())?;
    bindings.clear();
    bindings.extend(
        parsed
            .into_iter()
            .map(|(shortcut, action)| (shortcut.to_string(), action)),
    );
    Ok(())
}

fn register_initial_hotkeys(app: &AppHandle, state: &AppState, registry: &HotkeyRegistry) {
    let settings = match state.database.lock() {
        Ok(database) => database.settings.clone(),
        Err(_) => return,
    };
    let _ = apply_hotkeys(app, registry, &settings);
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Whether the main window is waiting for the frontend's first render before
/// it is shown for the first time.
struct PendingFirstShow(AtomicBool);

fn reveal_main_window_if_pending(app: &AppHandle) {
    let pending = app.state::<PendingFirstShow>();
    if pending.0.swap(false, Ordering::SeqCst) {
        show_main_window(app);
    }
}

#[tauri::command]
fn frontend_ready(app: AppHandle) {
    reveal_main_window_if_pending(&app);
}

fn emit_tray_action(app: &AppHandle, action: &str) {
    let _ = app.emit(
        "voxide-tray-action",
        TrayAction {
            action: action.into(),
        },
    );
}

fn update_tray_status(app: &AppHandle, visual_state: TrayVisualState) {
    let (status, dictate, paste_last, tooltip) = match visual_state {
        TrayVisualState::Ready => (
            "Ready to Record",
            "Start Dictation",
            true,
            "Voxide — ready to record",
        ),
        TrayVisualState::Recording => ("Recording…", "Stop Dictation", false, "Voxide — recording"),
        TrayVisualState::Processing => (
            "Transcribing…",
            "Transcribing…",
            false,
            "Voxide — transcribing",
        ),
    };
    let tray_state = app.state::<TrayState>();
    if let Ok(item) = tray_state.status.lock() {
        if let Some(item) = item.as_ref() {
            let _ = item.set_text(status);
        }
    }
    if let Ok(item) = tray_state.dictate.lock() {
        if let Some(item) = item.as_ref() {
            let _ = item.set_text(dictate);
            let _ = item.set_enabled(!matches!(visual_state, TrayVisualState::Processing));
        }
    }
    if let Ok(item) = tray_state.paste_last.lock() {
        if let Some(item) = item.as_ref() {
            let _ = item.set_enabled(paste_last);
        }
    }
    if let Some(tray) = app.tray_by_id("voxide-tray") {
        let _ = tray.set_tooltip(Some(tooltip));
    }
}

fn setup_tray(app: &AppHandle) -> tauri::Result<()> {
    let status = MenuItem::with_id(app, "tray-status", "Ready to Record", false, None::<&str>)?;
    let dictate = MenuItem::with_id(
        app,
        "tray-dictate",
        "Start / Stop Dictation",
        true,
        None::<&str>,
    )?;
    let paste_last = MenuItem::with_id(
        app,
        "tray-paste-last",
        "Paste Last Transcription",
        true,
        None::<&str>,
    )?;
    let show = MenuItem::with_id(app, "tray-show", "Open Voxide", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "tray-settings", "Settings", true, None::<&str>)?;
    let dictionary = MenuItem::with_id(
        app,
        "tray-dictionary",
        "Custom Dictionary",
        true,
        None::<&str>,
    )?;
    let quit = MenuItem::with_id(app, "tray-quit", "Quit Voxide", true, None::<&str>)?;
    let menu = Menu::with_items(
        app,
        &[
            &status,
            &dictate,
            &paste_last,
            &PredefinedMenuItem::separator(app)?,
            &show,
            &settings,
            &dictionary,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;
    let mut tray = TrayIconBuilder::with_id("voxide-tray")
        .menu(&menu)
        .tooltip("Voxide")
        .on_menu_event(|app, event| match event.id().as_ref() {
            "tray-show" => show_main_window(app),
            "tray-dictate" => emit_tray_action(app, "toggleDictation"),
            "tray-paste-last" => emit_tray_action(app, "pasteLast"),
            "tray-settings" => {
                show_main_window(app);
                emit_tray_action(app, "settings");
            }
            "tray-dictionary" => {
                show_main_window(app);
                emit_tray_action(app, "dictionary");
            }
            "tray-quit" => app.exit(0),
            _ => {}
        });
    if let Some(icon) = app.default_window_icon().cloned() {
        tray = tray.icon(icon);
    }
    tray.build(app)?;
    let tray_state = app.state::<TrayState>();
    if let Ok(mut item) = tray_state.status.lock() {
        *item = Some(status);
    }
    if let Ok(mut item) = tray_state.dictate.lock() {
        *item = Some(dictate);
    }
    if let Ok(mut item) = tray_state.paste_last.lock() {
        *item = Some(paste_last);
    }
    update_tray_status(app, TrayVisualState::Ready);
    Ok(())
}

pub fn run() {
    let state = AppState::load().unwrap_or_else(|error| panic!("Voxide could not start: {error}"));
    let analytics = analytics::AnalyticsService::load(state.analytics_identity_path());

    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // A second launch focuses the running instance instead of
            // starting another app.
            show_main_window(app);
        }))
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    let action = app.try_state::<HotkeyRegistry>().and_then(|registry| {
                        registry
                            .bindings
                            .lock()
                            .ok()
                            .and_then(|bindings| bindings.get(&shortcut.to_string()).cloned())
                    });
                    if let Some(action) = action {
                        let (action, prompt_profile_id) = match action {
                            HotkeyAction::PromptProfile(profile_id) => {
                                (HotkeyAction::Prompt, Some(profile_id))
                            }
                            action => (action, None),
                        };
                        let phase = match event.state {
                            ShortcutState::Pressed => "pressed",
                            ShortcutState::Released => "released",
                        };
                        let _ = app.emit(
                            "voxide-hotkey",
                            HotkeyEvent {
                                action,
                                phase: phase.into(),
                                prompt_profile_id,
                            },
                        );
                    }
                })
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_autostart::Builder::new()
                .app_name("Voxide")
                .arg("--voxide-autostart")
                .build(),
        )
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .manage(state)
        .manage(analytics)
        .manage(NativeCaptureState::default())
        .manage(PendingFirstShow(AtomicBool::new(false)))
        .manage(TrayState::default())
        .manage(HotkeyRegistry::default())
        .manage(local_api::LocalApiControl::default());
    #[cfg(target_os = "linux")]
    let builder = builder.manage(portal_hotkeys::PortalHotkeyState::default());
    builder
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .setup(|app| {
            debug_log::append(&format!(
                "Application started (version: {}, os: {}, arch: {})",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH
            ));
            let state = app.state::<AppState>();
            let cleanup_directories = [state.models_directory(), state.data_directory()]
                .into_iter()
                .filter_map(Result::ok)
                .collect::<Vec<_>>();
            tauri::async_runtime::spawn(async move {
                let removed = tauri::async_runtime::spawn_blocking(move || {
                    cleanup_abandoned_component_directories(
                        &cleanup_directories,
                        COMPONENT_STAGING_MAX_AGE,
                    )
                })
                .await
                .unwrap_or_default();
                if removed != 0 {
                    debug_log::append(&format!(
                        "Removed abandoned component transaction directories (count: {removed})"
                    ));
                }
            });
            let registry = app.state::<HotkeyRegistry>();
            let startup_settings = state
                .database
                .lock()
                .map(|database| database.settings.clone())
                .unwrap_or_default();
            let analytics = app.state::<analytics::AnalyticsService>();
            let first_open = analytics.bootstrap(startup_settings.share_anonymous_analytics);
            analytics.start_flush_loop();
            if first_open {
                analytics.capture(
                    "app_first_open",
                    startup_settings.share_anonymous_analytics,
                    analytics_common_properties(&startup_settings),
                );
            }
            analytics.capture(
                "app_open",
                startup_settings.share_anonymous_analytics,
                analytics_properties_with(
                    &startup_settings,
                    [(
                        "accessibility_trusted",
                        Value::Bool(permissions::accessibility_trusted().unwrap_or(false)),
                    )],
                ),
            );
            register_initial_hotkeys(app.handle(), &state, &registry);
            #[cfg(unix)]
            trigger::start_listener(app.handle().clone());
            setup_tray(app.handle())?;
            let show_in_dock = state
                .database
                .lock()
                .map(|database| database.settings.show_in_dock)
                .unwrap_or(true);
            apply_taskbar_visibility(app.handle(), show_in_dock)?;
            let show_main_window = !launched_from_autostart(std::env::args())
                || state
                    .database
                    .lock()
                    .map(|database| database.settings.show_main_window_at_login_launch)
                    .unwrap_or(true);
            if show_main_window {
                // The window stays hidden until the frontend reports its
                // first render (frontend_ready), so launch never flashes an
                // empty webview. If the frontend dies before reporting,
                // reveal the window anyway so the failure is visible.
                app.state::<PendingFirstShow>()
                    .0
                    .store(true, Ordering::SeqCst);
                let fallback = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(8)).await;
                    reveal_main_window_if_pending(&fallback);
                });
            }
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<AppState>();
                let settings = state
                    .database
                    .lock()
                    .ok()
                    .map(|database| database.settings.clone());
                if let Some(settings) = settings.filter(|settings| settings.local_api_enabled) {
                    let control = handle.state::<local_api::LocalApiControl>();
                    let _ =
                        local_api::start(&control, handle.clone(), settings.local_api_port).await;
                }
            });
            let preload_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Warm the selected local engine so the first dictation does
                // not pay model-load latency. Whisper additionally fetches
                // its VAD model; FluidVoice-style Parakeet does not use it.
                let state = preload_handle.state::<AppState>();
                let Ok((settings, vocabulary)) = state.database.lock().map(|database| {
                    let settings = database.settings.clone();
                    let vocabulary =
                        active_recognition_vocabulary(&settings, &database.custom_words);
                    (settings, vocabulary)
                }) else {
                    return;
                };
                settings
                    .selected_voice_engine
                    .preload(&state, &settings, vocabulary)
                    .await;
            });
            // Resolve the current input device (device handle, config, and
            // sound-server routing) so the first record action only opens the
            // stream instead of enumerating, negotiating, and forking `pactl`
            // on the hotkey path. Refreshed after each dictation and on a cold
            // miss inside start_dictation_capture.
            spawn_capture_prewarm(app.handle().clone());
            let update_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Defer the first check so startup remains responsive, then mirror the reference
                // app's hourly cadence for as long as this process is alive.
                tokio::time::sleep(Duration::from_secs(3)).await;
                check_for_updates_automatically(update_handle.clone()).await;
                loop {
                    tokio::time::sleep(Duration::from_secs(60 * 60)).await;
                    check_for_updates_automatically(update_handle.clone()).await;
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            bootstrap,
            frontend_ready,
            save_settings,
            check_for_updates,
            recent_release_notes,
            snooze_update_prompt,
            open_update_release,
            open_provider_website,
            set_onboarding_step,
            complete_onboarding,
            reset_onboarding,
            export_backup,
            import_backup,
            save_dictation,
            copy_completed_dictation,
            copy_text_to_clipboard,
            update_dictation,
            dictionary_learning_suggestions,
            accept_dictionary_learning_suggestion,
            dismiss_dictionary_learning_suggestion,
            open_feedback_issue,
            feedback_debug_information,
            delete_dictation,
            clear_dictation_history,
            export_dictation,
            export_audio_history,
            export_dictation_audio_pair,
            delete_saved_audio,
            delete_file_transcription,
            clear_file_transcription_history,
            export_file_transcription,
            save_file_transcription,
            transcribe_file,
            save_dictionary,
            custom_words,
            save_custom_words,
            export_dictionary,
            import_dictionary,
            save_prompt_profiles,
            save_dictation_prompt_configurations,
            set_active_prompt_profile,
            app_prompt_bindings,
            save_app_prompt_bindings,
            ai_providers,
            save_ai_providers,
            set_provider_api_key,
            move_provider_api_key,
            fetch_ai_provider_models,
            enhance_text,
            command_chats,
            create_command_chat,
            select_command_chat,
            clear_command_chat,
            delete_command_chat,
            plan_command,
            continue_command,
            cancel_command_plan,
            execute_approved_command,
            capture_selected_text,
            replace_selected_text,
            usage_stats,
            voice_model_status,
            verify_voice_engine_installation,
            voice_engine_availability,
            delete_whisper_model,
            delete_parakeet_model,
            delete_nemotron_model,
            audio_input_devices,
            accessibility_permission_status,
            open_accessibility_settings,
            download_whisper_model,
            download_parakeet_model,
            install_nemotron_cuda_runtime,
            remove_nemotron_cuda_runtime,
            open_voice_engine_storage,
            download_nemotron_model,
            start_native_dictation,
            stop_native_dictation,
            cancel_native_dictation,
            paste_last_transcription,
            configure_local_api,
            local_api_status,
            set_primary_hotkey,
            configure_hotkeys,
            hotkey_backend_status,
            resize_overlay_to_content
        ])
        .run(tauri::generate_context!())
        .expect("error while running Voxide");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_ready_for_local_dictation() {
        let database = AppDatabase::default();
        assert!(database.dictionary.is_empty());
        assert_eq!(database.settings.primary_dictation_hotkey, "Alt+Space");
        assert_eq!(database.settings.selected_model, "base");
        assert_eq!(database.settings.transcription_start_sound, "cue_1");
        assert_eq!(database.settings.transcription_sound_volume, 1.0);
        assert!(database.settings.command_mode_confirm_before_execute);
        assert_eq!(database.settings.overlay_position, OverlayPosition::Bottom);
        assert_eq!(database.settings.overlay_bottom_offset, 50.0);
        assert_eq!(database.settings.overlay_size, OverlaySize::Medium);
        assert!(matches!(
            database.settings.selected_voice_engine,
            VoiceEngine::Whisper
        ));
    }

    #[test]
    fn engine_self_test_suppresses_ordinary_dictation_side_effects() {
        assert!(is_isolated_dictation_test(false, true));
        assert!(is_isolated_dictation_test(true, false));
        assert!(is_isolated_dictation_test(true, true));
        assert!(!is_isolated_dictation_test(false, false));
    }

    #[test]
    fn startup_recovery_notice_is_delivered_only_once() {
        let state = AppState {
            database: Mutex::new(AppDatabase::default()),
            path: std::env::temp_dir().join("voxide-recovery-notice-test.json"),
            startup_recovery_notice: Mutex::new(Some("Recovered from backup".into())),
        };

        assert_eq!(
            state
                .take_startup_recovery_notice()
                .expect("recovery notice lock"),
            Some("Recovered from backup".into())
        );
        assert_eq!(
            state
                .take_startup_recovery_notice()
                .expect("recovery notice lock"),
            None
        );
    }

    #[test]
    fn failed_dictation_start_rolls_back_session_context_and_generation() {
        let capture_state = NativeCaptureState::default();
        let session_id = capture_state
            .session
            .lock()
            .expect("session lock")
            .start()
            .expect("session starts");
        capture_state
            .preview_generation
            .store(session_id, Ordering::SeqCst);
        *capture_state.context.lock().expect("context lock") = DictationContext {
            source_application: Some("Example".into()),
            source_window_title: Some("Draft".into()),
            prompt_profile_id: Some("profile".into()),
            preceding_text: "previous text".into(),
        };

        rollback_native_dictation_start(&capture_state, session_id);

        assert!(
            capture_state
                .session
                .lock()
                .expect("session lock")
                .is_idle(),
            "the coordinator must not remain active after startup failure"
        );
        assert_eq!(
            capture_state.preview_generation.load(Ordering::SeqCst),
            session_id.wrapping_add(1).max(1)
        );
        let context = capture_state.context.lock().expect("context lock");
        assert!(context.source_application.is_none());
        assert!(context.source_window_title.is_none());
        assert!(context.prompt_profile_id.is_none());
        assert!(context.preceding_text.is_empty());
    }

    #[test]
    fn stale_startup_failure_does_not_roll_back_a_newer_generation() {
        let capture_state = NativeCaptureState::default();
        let stale_session = capture_state
            .session
            .lock()
            .expect("session lock")
            .start()
            .expect("first session starts");
        capture_state
            .session
            .lock()
            .expect("session lock")
            .cancel(stale_session);
        let active_session = capture_state
            .session
            .lock()
            .expect("session lock")
            .start()
            .expect("second session starts");
        capture_state
            .preview_generation
            .store(active_session, Ordering::SeqCst);
        *capture_state.context.lock().expect("context lock") = DictationContext {
            source_application: Some("Example".into()),
            ..Default::default()
        };

        rollback_native_dictation_start(&capture_state, stale_session);

        assert_eq!(
            capture_state.session.lock().expect("session lock").state(),
            session::SessionState::Recording { id: active_session }
        );
        assert_eq!(
            capture_state.preview_generation.load(Ordering::SeqCst),
            active_session
        );
        assert_eq!(
            capture_state
                .context
                .lock()
                .expect("context lock")
                .source_application
                .as_deref(),
            Some("Example")
        );
    }

    #[test]
    fn finalization_context_guard_clears_transient_application_data() {
        let capture_state = NativeCaptureState::default();
        *capture_state.context.lock().expect("context lock") = DictationContext {
            source_application: Some("Example".into()),
            source_window_title: Some("Draft".into()),
            prompt_profile_id: Some("profile".into()),
            preceding_text: "sensitive preceding text".into(),
        };

        {
            let _reset_context = ResetDictationContextWhenDropped {
                context: &capture_state.context,
            };
        }

        let context = capture_state.context.lock().expect("context lock");
        assert!(context.source_application.is_none());
        assert!(context.source_window_title.is_none());
        assert!(context.prompt_profile_id.is_none());
        assert!(context.preceding_text.is_empty());
    }

    #[test]
    fn voice_engine_switch_is_rejected_while_a_session_is_active() {
        let mut coordinator = session::Coordinator::default();
        assert!(validate_voice_engine_switch(
            VoiceEngine::Whisper,
            VoiceEngine::Cloud,
            &coordinator
        )
        .is_ok());
        let session_id = coordinator.start().expect("session starts");
        assert_eq!(
            validate_voice_engine_switch(VoiceEngine::Whisper, VoiceEngine::Cloud, &coordinator)
                .expect_err("engine switch must be rejected"),
            "Stop or cancel the current dictation before changing the voice engine."
        );
        assert!(
            validate_voice_engine_switch(VoiceEngine::Whisper, VoiceEngine::Whisper, &coordinator)
                .is_ok(),
            "saving unrelated settings remains allowed"
        );
        assert!(coordinator.cancel(session_id));
        assert!(validate_voice_engine_switch(
            VoiceEngine::Whisper,
            VoiceEngine::Cloud,
            &coordinator
        )
        .is_ok());
    }

    #[test]
    fn cloud_profile_readiness_requires_a_model_and_compatible_provider() {
        let database = AppDatabase::default();
        let mut settings = database.settings.clone();
        let profile =
            cloud_transcription_profile(&settings, &database).expect("default profile is valid");
        assert!(matches!(
            profile.api_style,
            provider::ProviderApiStyle::OpenAiCompatible
        ));

        settings.cloud_transcription_model.clear();
        assert_eq!(
            cloud_transcription_profile(&settings, &database)
                .expect_err("empty model must be rejected"),
            "Choose a cloud transcription model before recording."
        );

        let mut anthropic_database = AppDatabase::default();
        anthropic_database.settings.selected_ai_provider = "anthropic".into();
        let error = cloud_transcription_profile(&anthropic_database.settings, &anthropic_database)
            .expect_err("Anthropic does not offer the OpenAI audio endpoint");
        assert!(error.contains("OpenAI-compatible audio transcription endpoint"));
    }

    #[test]
    fn engine_catalog_is_a_complete_single_source_of_capabilities() {
        let descriptors = VoiceEngine::ALL
            .into_iter()
            .map(VoiceEngine::descriptor)
            .collect::<Vec<_>>();
        let mut ids = descriptors
            .iter()
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), VoiceEngine::ALL.len());

        let nemotron = VoiceEngine::Nemotron.descriptor();
        assert_eq!(nemotron.preview_mode, asr::PreviewMode::Incremental);
        assert_eq!(nemotron.final_mode, asr::FinalMode::FlushActiveStream);
        assert!(nemotron.requires_cuda);
        assert!(!nemotron.supports_vocabulary);

        let whisper = VoiceEngine::Whisper.descriptor();
        assert_eq!(whisper.preview_mode, asr::PreviewMode::FullSnapshot);
        assert_eq!(whisper.final_mode, asr::FinalMode::IndependentFullDecode);
        assert!(whisper.supports_files);
        assert!(whisper.supports_translation);
    }

    #[test]
    fn engine_catalog_serializes_runtime_and_capability_fields_for_the_ui() {
        let catalog = voice_engine_availability();
        let value = serde_json::to_value(catalog).expect("engine catalog serializes");
        let whisper = value["engines"]
            .as_array()
            .expect("engine list")
            .iter()
            .find(|engine| engine["id"] == "whisper")
            .expect("whisper descriptor");
        assert_eq!(whisper["previewMode"], "fullSnapshot");
        assert_eq!(whisper["finalMode"], "independentFullDecode");
        assert_eq!(whisper["maturity"], "stable");
        assert_eq!(whisper["available"], true);
    }

    #[test]
    fn overlay_preferences_follow_source_values_and_migrate_port_names() {
        assert_eq!(
            serde_json::from_str::<OverlaySize>(r#""minimal""#).expect("old pill name"),
            OverlaySize::Pill
        );
        assert_eq!(
            serde_json::from_str::<OverlaySize>(r#""standard""#).expect("old medium name"),
            OverlaySize::Medium
        );
        assert_eq!(overlay_window_dimensions(OverlaySize::Pill), (100, 46));
        assert_eq!(overlay_window_dimensions(OverlaySize::Large), (600, 288));
        assert_eq!(normalize_overlay_bottom_offset(-4.0), 10.0);
        assert_eq!(normalize_overlay_bottom_offset(5_000.0), 1_000.0);
        assert_eq!(normalize_overlay_bottom_offset(f64::NAN), 50.0);
    }

    #[test]
    fn model_reasoning_configs_follow_defaults_and_explicit_disable() {
        let mut database = AppDatabase::default();
        let groq = selected_provider(&database, Some("groq")).expect("Groq exists");
        assert_eq!(
            groq.request_parameters.get("reasoning_effort"),
            Some(&Value::String("low".into()))
        );

        let key = format!("{}:{}", groq.id, groq.model);
        database.settings.model_reasoning_configs.insert(
            key,
            ModelReasoningConfig {
                parameter_name: "reasoning_effort".into(),
                parameter_value: "high".into(),
                is_enabled: false,
            },
        );
        let disabled = selected_provider(&database, Some("groq")).expect("Groq exists");
        assert!(disabled.request_parameters.is_empty());

        database.settings.model_reasoning_configs.insert(
            format!("{}:{}", disabled.id, disabled.model),
            ModelReasoningConfig {
                parameter_name: "enable_thinking".into(),
                parameter_value: "false".into(),
                is_enabled: true,
            },
        );
        let overridden = selected_provider(&database, Some("groq")).expect("Groq exists");
        assert_eq!(
            overridden.request_parameters.get("enable_thinking"),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn recording_cue_preferences_are_normalized() {
        let mut database = AppDatabase::default();
        database.settings.transcription_start_sound = "unknown".into();
        database.settings.transcription_sound_volume = 4.0;

        normalize_database(&mut database);

        assert_eq!(database.settings.transcription_start_sound, "cue_1");
        assert_eq!(database.settings.transcription_sound_volume, 1.0);
    }

    #[test]
    fn invalid_whisper_model_from_legacy_parakeet_selection_is_repaired() {
        let mut database = AppDatabase::default();
        database.settings.selected_voice_engine = VoiceEngine::Whisper;
        database.settings.selected_model = parakeet::MODEL_ID.into();

        normalize_database(&mut database);

        assert_eq!(database.settings.selected_model, "base");
    }

    #[test]
    fn unavailable_engine_selection_falls_back_to_a_portable_engine() {
        let mut database = AppDatabase::default();
        database.settings.selected_voice_engine = VoiceEngine::Parakeet;
        database.settings.selected_model = parakeet::MODEL_ID.into();

        normalize_database(&mut database);

        if parakeet::is_compiled() {
            assert_eq!(
                database.settings.selected_voice_engine,
                VoiceEngine::Parakeet
            );
        } else {
            assert_eq!(
                database.settings.selected_voice_engine,
                VoiceEngine::Whisper
            );
            assert_eq!(database.settings.selected_model, "base");
        }
    }

    #[test]
    fn audio_history_budget_matches_the_reference_binary_gigabyte_contract() {
        assert_eq!(Settings::default().audio_history_budget_gb, 4.0);

        let mut database = AppDatabase::default();
        database.settings.audio_history_budget_gb = 0.001;
        normalize_database(&mut database);
        assert_eq!(database.settings.audio_history_budget_gb, 0.1);

        database.settings.audio_history_budget_gb = 0.0;
        normalize_database(&mut database);
        assert_eq!(database.settings.audio_history_budget_gb, 4.0);
    }

    #[test]
    fn restored_punctuation_preferences_follow_the_reference_normalization_contract() {
        let mut database = AppDatabase::default();
        database.settings.punctuation_dictionary_prefix = "  LITERAL ".into();
        database.settings.punctuation_dictionary_rules = vec![
            formatting::PunctuationRule {
                aliases: vec!["  COMMA".into(), "comma".into(), " ".into()],
                symbol: " , ".into(),
            },
            formatting::PunctuationRule {
                aliases: vec!["invalid".into()],
                symbol: " ".into(),
            },
        ];

        normalize_database(&mut database);

        assert_eq!(database.settings.punctuation_dictionary_prefix, "literal");
        assert_eq!(database.settings.punctuation_dictionary_rules.len(), 1);
        assert_eq!(
            database.settings.punctuation_dictionary_rules[0].aliases,
            ["comma"]
        );
        assert_eq!(
            database.settings.punctuation_dictionary_rules[0].symbol,
            ","
        );
    }

    #[test]
    fn transcription_preview_length_matches_the_reference_range_and_step() {
        let mut database = AppDatabase::default();
        database.settings.transcription_preview_char_limit = 1;
        normalize_database(&mut database);
        assert_eq!(
            database.settings.transcription_preview_char_limit,
            TRANSCRIPTION_PREVIEW_MIN_CHARACTERS
        );

        database.settings.transcription_preview_char_limit = 126;
        normalize_database(&mut database);
        assert_eq!(database.settings.transcription_preview_char_limit, 150);

        database.settings.transcription_preview_char_limit = 900;
        normalize_database(&mut database);
        assert_eq!(
            database.settings.transcription_preview_char_limit,
            TRANSCRIPTION_PREVIEW_MAX_CHARACTERS
        );
        assert_eq!(tail_characters("αβγδε", 3), "γδε");
    }

    #[test]
    fn apple_speech_locales_are_normalized_without_changing_other_engines() {
        assert_eq!(normalize_apple_speech_locale("pt_BR.UTF-8"), "pt-BR");
        assert_eq!(normalize_apple_speech_locale(" POSIX "), "en-US");

        let mut database = AppDatabase::default();
        database.settings.apple_speech_locale = "es_MX.UTF-8".into();
        normalize_database(&mut database);
        assert_eq!(database.settings.apple_speech_locale, "es-MX");
    }

    #[test]
    fn apple_speech_availability_matches_the_current_platform() {
        assert_eq!(apple_speech::is_supported(), cfg!(target_os = "macos"));
    }

    #[test]
    fn ids_include_the_requested_resource_prefix() {
        assert!(make_id("dictation").starts_with("dictation-"));
    }

    #[test]
    fn dictionary_only_replaces_whole_spoken_terms() {
        let dictionary = vec![DictionaryEntry {
            id: "dictionary-1".into(),
            spoken: "vox side".into(),
            replacement: "Voxide".into(),
            created_at: Utc::now(),
        }];
        assert_eq!(
            apply_dictionary("vox side works", &dictionary),
            "Voxide works"
        );
        assert_eq!(
            apply_dictionary("vox sides work", &dictionary),
            "vox sides work"
        );
    }

    #[test]
    fn dictionary_entries_follow_the_reference_storage_normalization() {
        let entries = normalize_dictionary_entries(vec![
            DictionaryEntry {
                id: "dictionary-valid".into(),
                spoken: "  Vox Side  ".into(),
                replacement: "  Voxide  ".into(),
                created_at: Utc::now(),
            },
            DictionaryEntry {
                id: "dictionary-empty".into(),
                spoken: "  ".into(),
                replacement: "ignored".into(),
                created_at: Utc::now(),
            },
        ]);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "dictionary-valid");
        assert_eq!(entries[0].spoken, "vox side");
        assert_eq!(entries[0].replacement, "Voxide");
    }

    #[test]
    fn dictionary_matches_reference_pattern_order_boundaries_and_literal_replacements() {
        let now = Utc::now();
        let dictionary = vec![
            DictionaryEntry {
                id: "dictionary-short".into(),
                spoken: "vox".into(),
                replacement: "short".into(),
                created_at: now,
            },
            DictionaryEntry {
                id: "dictionary-long".into(),
                spoken: "vox side".into(),
                replacement: "Voxide".into(),
                created_at: now,
            },
            DictionaryEntry {
                id: "dictionary-chain".into(),
                spoken: "voxide".into(),
                replacement: "Final".into(),
                created_at: now,
            },
            DictionaryEntry {
                id: "dictionary-literal".into(),
                spoken: "c++".into(),
                replacement: "$1".into(),
                created_at: now,
            },
            DictionaryEntry {
                id: "dictionary-underscore".into(),
                spoken: "bar".into(),
                replacement: "baz".into(),
                created_at: now,
            },
        ];

        assert_eq!(
            apply_dictionary("VOX SIDE uses C++ beside foo_bar and bar.", &dictionary),
            "Final uses $1 beside foo_bar and baz."
        );
    }

    #[test]
    fn automatic_dictionary_learning_extracts_small_meaningful_corrections() {
        assert_eq!(
            automatic_dictionary_candidate(
                "Send this to Flude Voice now.",
                "Send this to Voxide now."
            ),
            Some(("Flude Voice".into(), "Voxide".into()))
        );
        assert_eq!(
            automatic_dictionary_candidate("Nothing changed.", "Nothing changed."),
            None
        );
        assert_eq!(
            automatic_dictionary_candidate("Call a cat now.", "Call a cut now."),
            Some(("cat".into(), "cut".into()))
        );
    }

    #[test]
    fn recognition_vocabulary_includes_unique_terms_and_aliases() {
        let words = [
            CustomWordEntry {
                text: "Voxide".into(),
                weight: None,
                aliases: vec!["vox side".into()],
            },
            CustomWordEntry {
                text: "voxide".into(),
                weight: Some(2.0),
                aliases: vec!["Acme".into()],
            },
        ];
        let vocabulary = recognition_vocabulary(&words);
        assert_eq!(vocabulary, vec!["Voxide", "vox side", "Acme"]);

        let mut settings = Settings::default();
        assert!(active_recognition_vocabulary(&settings, &words).is_empty());
        settings.vocabulary_boosting_enabled = true;
        assert_eq!(active_recognition_vocabulary(&settings, &words), vocabulary);
    }

    #[test]
    fn missing_provider_data_migrates_to_builtin_profiles() {
        let mut old_data =
            serde_json::to_value(AppDatabase::default()).expect("database serializes");
        old_data
            .as_object_mut()
            .expect("database is an object")
            .remove("aiProviders");
        let restored: AppDatabase =
            serde_json::from_value(old_data).expect("old database data migrates");
        assert!(!restored.ai_providers.is_empty());
    }

    #[test]
    fn provider_profiles_are_normalized_and_require_a_usable_catalog() {
        let mut profiles = provider::AiProviderProfile::built_in();
        profiles[0].id = " openai ".into();
        profiles[0].name = " OpenAI ".into();
        profiles[0].base_url = " https://api.openai.com/v1/ ".into();
        profiles[0].model = " gpt-4.1 ".into();
        profiles[0]
            .request_parameters
            .insert("untrusted".into(), Value::Bool(true));
        validate_and_normalize_ai_provider_profiles(&mut profiles)
            .expect("valid profiles are accepted");
        assert_eq!(profiles[0].id, "openai");
        assert_eq!(profiles[0].name, "OpenAI");
        assert_eq!(profiles[0].base_url, "https://api.openai.com/v1");
        assert_eq!(profiles[0].model, "gpt-4.1");
        assert!(profiles[0].request_parameters.is_empty());

        let mut duplicate = profiles.clone();
        duplicate.push(provider::AiProviderProfile {
            id: "OPENAI".into(),
            name: "Duplicate".into(),
            api_style: provider::ProviderApiStyle::OpenAiCompatible,
            base_url: "https://example.invalid/v1".into(),
            model: "model".into(),
            enabled: true,
            request_parameters: Default::default(),
        });
        assert!(validate_and_normalize_ai_provider_profiles(&mut duplicate)
            .expect_err("duplicate IDs are rejected")
            .contains("unique"));

        let mut invalid_endpoint = profiles.clone();
        invalid_endpoint[0].base_url = "file:///tmp/provider".into();
        assert!(
            validate_and_normalize_ai_provider_profiles(&mut invalid_endpoint)
                .expect_err("non-HTTP endpoints are rejected")
                .contains("HTTP or HTTPS")
        );

        let mut disabled = profiles.clone();
        for profile in &mut disabled {
            profile.enabled = false;
        }
        assert!(validate_and_normalize_ai_provider_profiles(&mut disabled)
            .expect_err("at least one provider must remain enabled")
            .contains("enabled"));
    }

    #[test]
    fn database_provider_repair_restores_missing_builtin_profiles() {
        let mut database = AppDatabase::default();
        database.ai_providers.truncate(1);
        normalize_database(&mut database);

        assert_eq!(
            database.ai_providers.len(),
            provider::AiProviderProfile::built_in().len()
        );
        assert_eq!(
            database
                .ai_providers
                .iter()
                .find(|profile| profile.id == "anthropic")
                .map(|profile| profile.base_url.as_str()),
            Some("https://api.anthropic.com/v1")
        );

        let mut legacy = AppDatabase::default();
        legacy
            .ai_providers
            .iter_mut()
            .find(|profile| profile.id == "anthropic")
            .expect("Anthropic is built in")
            .base_url = "https://api.anthropic.com".into();
        normalize_database(&mut legacy);
        assert_eq!(
            legacy
                .ai_providers
                .iter()
                .find(|profile| profile.id == "anthropic")
                .map(|profile| profile.base_url.as_str()),
            Some("https://api.anthropic.com/v1")
        );
    }

    #[test]
    fn database_provider_repair_restores_an_enabled_fallback() {
        let mut database = AppDatabase::default();
        for profile in &mut database.ai_providers {
            profile.enabled = false;
        }
        database.settings.selected_ai_provider = "anthropic".into();
        normalize_database(&mut database);

        assert!(database.ai_providers[0].enabled);
        assert_eq!(database.settings.selected_ai_provider, "openai");
    }

    #[test]
    fn settings_migrate_missing_hotkeys_and_local_api() {
        let mut data = serde_json::to_value(AppDatabase::default()).expect("database serializes");
        let settings = data
            .get_mut("settings")
            .and_then(serde_json::Value::as_object_mut)
            .expect("settings is an object");
        for key in [
            "onboardingStep",
            "onboardingAiSkipped",
            "onboardingPlaygroundValidated",
            "appleSpeechLocale",
            "secondaryDictationHotkey",
            "showMainWindowAtLoginLaunch",
            "promptModeHotkey",
            "commandModeHotkey",
            "rewriteModeHotkey",
            "cancelRecordingHotkey",
            "pasteLastTranscriptionHotkey",
            "transcriptionPreviewCharLimit",
            "transcriptionStartSound",
            "transcriptionSoundVolume",
            "textInsertionMode",
            "removeFillerWordsEnabled",
            "fillerWords",
            "autoConvertPunctuationEnabled",
            "punctuationDictionaryPrefix",
            "punctuationDictionaryRules",
            "literalDictationFormattingEnabled",
            "gaavLowercaseFirstLetterEnabled",
            "gaavRemoveTrailingPeriodEnabled",
            "notifyAiProcessingFailures",
            "saveTranscriptionHistory",
            "vocabularyBoostingEnabled",
            "cloudTranscriptionModel",
            "selectedDictationPromptProfile",
            "selectedRewritePromptProfile",
            "selectedCommandPromptProfile",
            "dictationPromptRoutingScope",
            "editPromptRoutingScope",
            "selectedRewriteAiProvider",
            "selectedCommandAiProvider",
            "shareAnonymousAnalytics",
            "autoUpdateCheckEnabled",
            "betaReleasesEnabled",
            "lastUpdateCheckAt",
            "updatePromptSnoozedUntil",
            "snoozedUpdateVersion",
            "localApiEnabled",
            "localApiPort",
        ] {
            settings.remove(key);
        }
        data.as_object_mut()
            .expect("database is an object")
            .remove("customWords");
        let restored: AppDatabase =
            serde_json::from_value(data).expect("old settings data migrates");
        assert_eq!(
            restored.settings.cancel_recording_hotkey.as_deref(),
            Some("Escape")
        );
        assert_eq!(restored.settings.local_api_port, 47_733);
        assert!(!restored.settings.share_anonymous_analytics);
        assert!(restored.settings.auto_update_check_enabled);
        assert_eq!(restored.settings.transcription_start_sound, "cue_1");
        assert_eq!(restored.settings.transcription_sound_volume, 1.0);
        assert_eq!(
            restored.settings.transcription_preview_char_limit,
            DEFAULT_TRANSCRIPTION_PREVIEW_CHARACTERS
        );
        assert_eq!(
            restored.settings.text_insertion_mode,
            typing::TextInsertionMode::Standard
        );
        assert_eq!(restored.settings.onboarding_step, 0);
        assert!(!restored.settings.apple_speech_locale.trim().is_empty());
        assert!(!restored.settings.onboarding_playground_validated);
        assert!(restored.settings.show_main_window_at_login_launch);
        assert!(restored.settings.remove_filler_words_enabled);
        assert!(restored.settings.auto_convert_punctuation_enabled);
        assert_eq!(restored.settings.punctuation_dictionary_prefix, "literal");
        assert!(!restored.settings.punctuation_dictionary_rules.is_empty());
        assert!(restored.settings.notify_ai_processing_failures);
        assert!(restored.settings.save_transcription_history);
        assert_eq!(
            restored.settings.cloud_transcription_model,
            "gpt-4o-mini-transcribe"
        );
        assert_eq!(
            restored.settings.selected_rewrite_prompt_profile.as_deref(),
            Some("default-rewrite")
        );
        assert!(restored.settings.selected_rewrite_ai_provider.is_none());
        assert!(restored.settings.selected_command_ai_provider.is_none());
        assert!(restored.custom_words.is_empty());
    }

    #[test]
    fn text_insertion_mode_accepts_reference_values_and_recovers_from_unknown_ones() {
        let mut data = serde_json::to_value(AppDatabase::default()).expect("database serializes");
        data.get_mut("settings")
            .and_then(serde_json::Value::as_object_mut)
            .expect("settings is an object")
            .insert(
                "textInsertionMode".into(),
                Value::String("reliablePaste".into()),
            );
        let reliable: AppDatabase = serde_json::from_value(data.clone())
            .expect("reference reliable-paste setting migrates");
        assert_eq!(
            reliable.settings.text_insertion_mode,
            typing::TextInsertionMode::ReliablePaste
        );

        data.get_mut("settings")
            .and_then(serde_json::Value::as_object_mut)
            .expect("settings is an object")
            .insert("textInsertionMode".into(), Value::String("unknown".into()));
        let restored: AppDatabase =
            serde_json::from_value(data).expect("unknown insertion modes safely recover");
        assert_eq!(
            restored.settings.text_insertion_mode,
            typing::TextInsertionMode::Standard
        );
    }

    #[test]
    fn prompt_profiles_resolve_the_explicit_active_profile_for_each_mode() {
        let mut database = AppDatabase::default();
        database.prompt_profiles.push(PromptProfile {
            id: "concise-dictation".into(),
            name: "Concise dictation".into(),
            prompt: "Return concise spoken notes.".into(),
            mode: DictationMode::Dictate,
        });
        database.settings.selected_dictation_prompt_profile = Some("concise-dictation".into());

        assert_eq!(
            prompt_for_mode(&database, DictationMode::Dictate).id,
            "concise-dictation"
        );

        database.settings.prompt_mode_selected_prompt_id = Some("concise-dictation".into());
        normalize_database(&mut database);
        assert_eq!(
            database.settings.prompt_mode_selected_prompt_id.as_deref(),
            Some("concise-dictation")
        );

        database.settings.selected_rewrite_prompt_profile = Some("missing".into());
        database.settings.prompt_mode_selected_prompt_id = Some("missing".into());
        normalize_database(&mut database);
        assert_eq!(
            database.settings.selected_rewrite_prompt_profile.as_deref(),
            Some("default-rewrite")
        );
        assert!(database.settings.prompt_mode_selected_prompt_id.is_none());
    }

    #[test]
    fn prompt_test_uses_a_nonempty_draft_without_mutating_the_saved_profile() {
        let profile = PromptProfile {
            id: "dictate-profile".into(),
            name: "Dictate".into(),
            prompt: "Saved prompt".into(),
            mode: DictationMode::Dictate,
        };

        assert_eq!(
            effective_dictation_system_prompt(&profile, Some(" Draft prompt ")),
            "Draft prompt"
        );
        assert_eq!(profile.prompt, "Saved prompt");
        assert_eq!(
            effective_dictation_system_prompt(&profile, Some("   ")),
            "Saved prompt"
        );
    }

    #[test]
    fn application_prompt_bindings_override_the_global_profile_for_that_mode() {
        let mut database = AppDatabase::default();
        database.prompt_profiles.push(PromptProfile {
            id: "email-dictation".into(),
            name: "Email dictation".into(),
            prompt: "Write a polished email.".into(),
            mode: DictationMode::Dictate,
        });
        database.app_prompt_bindings.push(AppPromptBinding {
            id: "email-binding".into(),
            application: "Mail".into(),
            mode: DictationMode::Dictate,
            prompt_profile_id: "email-dictation".into(),
        });

        assert_eq!(
            prompt_for_mode_and_application(&database, DictationMode::Dictate, Some("mail")).id,
            "email-dictation"
        );
        assert_eq!(
            prompt_for_mode_and_application(&database, DictationMode::Dictate, Some("Notes")).id,
            "default-dictate"
        );
        assert_eq!(
            prompt_for_mode_with_override(
                &database,
                DictationMode::Dictate,
                Some("email-dictation"),
                Some("Notes"),
            )
            .id,
            "email-dictation"
        );

        database.settings.selected_dictation_prompt_profile = Some("email-dictation".into());
        database.settings.dictation_prompt_routing_scope = PromptRoutingScope::SelectedAppsOnly;
        assert_eq!(
            prompt_for_mode_and_application(&database, DictationMode::Dictate, Some("Notes")).id,
            "default-dictate"
        );
        assert_eq!(
            prompt_for_mode_and_application(&database, DictationMode::Dictate, Some("Mail")).id,
            "email-dictation"
        );
    }

    #[test]
    fn mode_specific_ai_provider_overrides_the_global_provider() {
        let mut database = AppDatabase::default();
        database.settings.selected_ai_provider = "openai".into();
        database.settings.selected_command_ai_provider = Some("anthropic".into());

        assert_eq!(
            provider_for_mode(&database, DictationMode::Rewrite)
                .expect("global provider should resolve")
                .id,
            "openai"
        );
        assert_eq!(
            provider_for_mode(&database, DictationMode::Command)
                .expect("command provider should resolve")
                .id,
            "anthropic"
        );
    }

    #[test]
    fn dictate_prompt_profiles_can_override_the_provider_and_model() {
        let mut database = AppDatabase::default();
        database
            .dictation_prompt_configurations
            .push(DictationPromptConfiguration {
                prompt_profile_id: "default-dictate".into(),
                provider_id: Some("groq".into()),
                model: Some("profile-specific-model".into()),
            });

        let provider = dictation_provider_for_prompt_profile(&database, "default-dictate")
            .expect("profile routing should select an enabled provider");
        assert_eq!(provider.id, "groq");
        assert_eq!(provider.model, "profile-specific-model");

        database
            .dictation_prompt_configurations
            .push(DictationPromptConfiguration {
                prompt_profile_id: "missing-profile".into(),
                provider_id: Some("groq".into()),
                model: None,
            });
        normalize_database(&mut database);
        assert_eq!(database.dictation_prompt_configurations.len(), 1);
    }

    #[test]
    fn configured_hotkeys_include_enabled_secondary_actions() {
        let mut settings = Settings::default();
        settings.command_mode_hotkey = Some("Alt+C".into());
        settings.secondary_dictation_hotkey = Some("Alt+D".into());
        settings
            .prompt_shortcut_assignments
            .push(PromptShortcutAssignment {
                prompt_profile_id: "default-dictate".into(),
                hotkey: "Alt+P".into(),
            });
        let bindings = configured_hotkeys(&settings);
        assert_eq!(
            bindings
                .iter()
                .filter(|(_, action)| action == &HotkeyAction::Dictate)
                .count(),
            2
        );
        assert!(bindings
            .iter()
            .any(|(_, action)| action == &HotkeyAction::Command));
        assert!(bindings
            .iter()
            .any(|(_, action)| action == &HotkeyAction::Cancel));
        assert!(bindings.iter().any(|(_, action)| {
            action == &HotkeyAction::PromptProfile("default-dictate".into())
        }));
    }

    #[test]
    fn prompt_shortcuts_drop_deleted_or_invalid_dictation_profiles() {
        let mut database = AppDatabase::default();
        database.prompt_profiles.push(PromptProfile {
            id: "temporary-dictate".into(),
            name: "Temporary dictation".into(),
            prompt: "Format a temporary dictation.".into(),
            mode: DictationMode::Dictate,
        });
        database.settings.prompt_shortcut_assignments.extend([
            PromptShortcutAssignment {
                prompt_profile_id: "temporary-dictate".into(),
                hotkey: " Alt+P ".into(),
            },
            PromptShortcutAssignment {
                prompt_profile_id: "missing-profile".into(),
                hotkey: "Alt+M".into(),
            },
        ]);
        normalize_database(&mut database);

        assert_eq!(database.settings.prompt_shortcut_assignments.len(), 1);
        assert_eq!(
            database.settings.prompt_shortcut_assignments[0].hotkey,
            "Alt+P"
        );

        database
            .prompt_profiles
            .retain(|profile| profile.id != "temporary-dictate");
        normalize_database(&mut database);
        assert!(database.settings.prompt_shortcut_assignments.is_empty());
    }

    #[test]
    fn command_plan_marks_destructive_shell_actions() {
        let plan = parse_command_plan(r#"{"kind":"command","command":"rm -rf ./build","purpose":"Remove the build directory"}"#)
            .expect("plan should parse");
        assert!(plan.destructive);
        assert_eq!(
            serde_json::to_value(&plan)
                .expect("plan should serialize")
                .get("destructive")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert!(is_destructive_command("git reset --hard HEAD"));
        for command in [
            "mv old-name new-name",
            "kill 1234",
            "chmod 600 secrets.txt",
            "printf data | sudo tee config",
            "find . -print0 | xargs rm",
        ] {
            assert!(
                is_destructive_command(command),
                "the reference requires confirmation for {command:?}"
            );
        }
        assert!(!is_destructive_command("git status --short"));
        assert!(!is_destructive_command("ls -la"));
    }

    #[test]
    fn native_command_tool_calls_are_reviewable_and_round_trip_into_history() {
        let tool_call = provider::CommandToolCall {
            id: "call-list".into(),
            name: "execute_terminal_command".into(),
            arguments: serde_json::json!({
                "command": "ls -la",
                "purpose": "Inspect the current directory",
                "workingDirectory": "/tmp",
            }),
        };
        let mut plan =
            command_plan_from_provider_response(provider::CommandProviderResponse::ToolCall {
                content: "Checking the directory first.".into(),
                thinking: None,
                tool_call,
            })
            .expect("native tool call should produce a reviewed plan");
        plan.conversation_id = Some("chat-1".into());

        assert_eq!(plan.command.as_deref(), Some("ls -la"));
        assert_eq!(plan.tool_call_id.as_deref(), Some("call-list"));
        assert_eq!(plan.working_directory.as_deref(), Some("/tmp"));
        assert!(!plan.destructive);

        let mut chat = CommandChat::new();
        chat.append(CommandChatRole::User, "List this directory");
        append_command_plan(&mut chat, &plan);
        chat.append_with_tool_metadata(
            CommandChatRole::Tool,
            r#"{"success":true,"output":"Cargo.toml"}"#,
            None,
            plan.tool_call_id.clone(),
            None,
        );
        let messages = command_provider_messages(&chat.messages)
            .expect("native tool history should stay provider-compatible");
        assert!(matches!(
            &messages[1],
            provider::CommandProviderMessage::Assistant {
                tool_call: Some(call),
                ..
            } if call.id == "call-list"
        ));
        assert!(matches!(
            &messages[2],
            provider::CommandProviderMessage::Tool { tool_call_id, .. }
                if tool_call_id == "call-list"
        ));
    }

    #[test]
    fn command_history_preserves_legacy_tool_results_and_resets_step_limit_per_request() {
        let mut chat = CommandChat::new();
        chat.append(CommandChatRole::User, "Inspect the directory");
        chat.append_with_tool_metadata(
            CommandChatRole::Assistant,
            "Checking first.",
            Some(provider::CommandToolCall {
                id: "call-inspect".into(),
                name: "execute_terminal_command".into(),
                arguments: serde_json::json!({ "command": "ls" }),
            }),
            None,
            None,
        );
        // Saved chats from before exact call IDs are still valid provider context.
        chat.append(CommandChatRole::Tool, "directory contents");
        let messages = command_provider_messages(&chat.messages)
            .expect("legacy tool history should stay provider-compatible");
        assert!(matches!(
            &messages[2],
            provider::CommandProviderMessage::Tool { tool_call_id, .. }
                if tool_call_id == "call-inspect"
        ));

        for _ in 0..20 {
            chat.append(CommandChatRole::Tool, "completed step");
        }
        assert_eq!(command_steps_since_latest_request(&chat.messages), 21);
        chat.append(CommandChatRole::User, "Now summarize the result");
        assert_eq!(command_steps_since_latest_request(&chat.messages), 0);
    }

    #[test]
    fn native_command_provider_text_becomes_an_answer_plan() {
        let plan = command_plan_from_provider_response(provider::CommandProviderResponse::Text {
            content: "Your repository is already clean.".into(),
            thinking: None,
        })
        .expect("plain native provider text should be a valid answer");
        assert_eq!(plan.kind, "answer");
        assert_eq!(
            plan.answer.as_deref(),
            Some("Your repository is already clean.")
        );
        assert!(plan.command.is_none());
    }

    #[test]
    fn native_command_provider_reasoning_stays_display_only_on_structured_plans() {
        let plan = command_plan_from_provider_response(provider::CommandProviderResponse::Text {
            content: r#"{"kind":"answer","answer":"The working tree is clean."}"#.into(),
            thinking: Some("I should inspect the repository status first.".into()),
        })
        .expect("structured native provider response should be a valid plan");

        assert_eq!(plan.answer.as_deref(), Some("The working tree is clean."));
        assert_eq!(
            plan.thinking.as_deref(),
            Some("I should inspect the repository status first.")
        );
        let mut chat = CommandChat::new();
        append_command_plan(&mut chat, &plan);
        assert_eq!(
            chat.messages[0].thinking.as_deref(),
            Some("I should inspect the repository status first.")
        );
        assert!(command_provider_messages(&chat.messages)
            .expect("display-only thinking must not affect provider history")
            .iter()
            .all(|message| !format!("{message:?}").contains("inspect the repository")));
    }

    #[test]
    fn command_chats_keep_an_active_local_conversation() {
        let mut database = AppDatabase::default();
        normalize_database(&mut database);
        let active_id = database
            .active_command_chat_id
            .clone()
            .expect("a default chat should be active");
        let chat = database
            .command_chats
            .iter_mut()
            .find(|chat| chat.id == active_id)
            .expect("active chat should exist");
        chat.append(CommandChatRole::User, "Show the working tree");
        chat.append(CommandChatRole::Assistant, "Proposed command: git status");

        assert!(command_chat_context(&chat.messages).contains("Show the working tree"));

        database.command_chats.clear();
        database.active_command_chat_id = None;
        normalize_database(&mut database);
        assert_eq!(database.command_chats.len(), 1);
        assert!(database.active_command_chat_id.is_some());
    }

    #[test]
    fn command_plan_accepts_non_execution_answers() {
        let plan = parse_command_plan(
            r#"{"kind":"answer","answer":"Your calendar opens from the task bar."}"#,
        )
        .expect("answer should parse");
        assert_eq!(plan.kind, "answer");
        assert!(!plan.destructive);
    }

    #[test]
    fn whisper_model_catalog_rejects_unknown_downloads() {
        assert_eq!(whisper_model_filename("base"), Ok("ggml-base.bin"));
        assert_eq!(whisper_model_filename("base.en"), Ok("ggml-base.en.bin"));
        assert!(whisper_model_filename("not-a-whisper-model").is_err());
    }

    #[test]
    fn whisper_model_validation_rejects_proxy_markup_and_incomplete_files() {
        assert!(looks_like_markup(b"\xef\xbb\xbf \n<html>blocked</html>"));
        assert!(looks_like_markup(b"<?xml version=\"1.0\"?>"));
        assert!(!looks_like_markup(b"ggml\0\x01\x02"));
        assert!(model_content_type_is_markup(Some(
            &reqwest::header::HeaderValue::from_static("text/html; charset=utf-8")
        )));
        assert!(!model_content_type_is_markup(Some(
            &reqwest::header::HeaderValue::from_static("application/octet-stream")
        )));

        let path = std::env::temp_dir().join(format!(
            "voxide-whisper-model-validation-{}-{}.bin",
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::write(&path, b"ggml\0\x01\x02").expect("temporary model should be written");
        assert!(valid_whisper_model_file(&path));
        assert!(validate_whisper_model_download(&path, Some(7), 7).is_ok());
        assert!(validate_whisper_model_download(&path, Some(8), 7).is_err());

        fs::write(&path, b"<!doctype html><html>blocked</html>")
            .expect("temporary proxy response should be written");
        assert!(!valid_whisper_model_file(&path));
        assert!(validate_whisper_model_download(&path, None, 35).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn autostart_launch_marker_is_detected_without_hiding_manual_launches() {
        assert!(launched_from_autostart([
            "voxide".into(),
            "--voxide-autostart".into(),
        ]));
        assert!(!launched_from_autostart(["voxide".into()]));
    }

    #[test]
    fn command_plan_summaries_preserve_a_follow_up_plans_context() {
        let command = parse_command_plan(
            r#"{"kind":"command","command":"git status --short","purpose":"Inspect changes"}"#,
        )
        .expect("command should parse");
        let answer = parse_command_plan(r#"{"kind":"answer","answer":"Everything is clean."}"#)
            .expect("answer should parse");

        assert!(command_plan_summary(&command).contains("git status --short"));
        assert_eq!(command_plan_summary(&answer), "Everything is clean.");
        assert!(command_system_prompt("Use the requested style.").contains("exactly one JSON"));
    }

    #[test]
    fn audio_history_pruning_follows_history_order_not_file_mtime() {
        let directory =
            std::env::temp_dir().join(format!("voxide-audio-pruning-test-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("temporary directory should be created");
        let newest_history_entry = directory.join("newest-history-entry.wav");
        let oldest_history_entry = directory.join("oldest-history-entry.wav");
        fs::write(&newest_history_entry, [0_u8; 8]).expect("newer file should be written");
        std::thread::sleep(Duration::from_millis(5));
        fs::write(&oldest_history_entry, [0_u8; 8]).expect("older file should be written");
        prune_audio_directory(
            &directory,
            8,
            &[
                "oldest-history-entry.wav".into(),
                "newest-history-entry.wav".into(),
            ],
        )
        .expect("history should prune");
        assert!(!oldest_history_entry.exists());
        assert!(newest_history_entry.exists());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn audio_budget_pruning_clears_history_attachments_and_orphans() {
        let root = std::env::temp_dir().join(format!(
            "voxide-audio-attachment-pruning-test-{}-{}",
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(&root).expect("temporary directory should be created");
        let state = AppState {
            database: Mutex::new(AppDatabase::default()),
            path: root.join("voxide.json"),
            startup_recovery_notice: Mutex::new(None),
        };
        let directory = state
            .audio_history_directory()
            .expect("audio history directory should be created");
        let newer = directory.join("newer.wav");
        let older = directory.join("older.wav");
        let orphan = directory.join("orphan.wav");
        fs::write(&newer, [0_u8; 8]).expect("newer file should be written");
        std::thread::sleep(Duration::from_millis(5));
        fs::write(&older, [0_u8; 8]).expect("older file should be written");
        fs::write(&orphan, [0_u8; 8]).expect("orphan file should be written");
        let now = Utc::now();
        state
            .update(|database| {
                database.dictation_history = vec![
                    DictationEntry {
                        id: "newer".into(),
                        text: "Newer".into(),
                        raw_text: None,
                        created_at: now,
                        duration_ms: None,
                        mode: DictationMode::Dictate,
                        source_application: None,
                        source_window_title: None,
                        audio_file: Some(newer.display().to_string()),
                        audio_model: Some("base".into()),
                        was_ai_processed: false,
                        processing_model: None,
                        ai_processing_error: None,
                    },
                    DictationEntry {
                        id: "older".into(),
                        text: "Older".into(),
                        raw_text: None,
                        created_at: now - chrono::Duration::seconds(1),
                        duration_ms: None,
                        mode: DictationMode::Dictate,
                        source_application: None,
                        source_window_title: None,
                        audio_file: Some(older.display().to_string()),
                        audio_model: Some("base".into()),
                        was_ai_processed: false,
                        processing_model: None,
                        ai_processing_error: None,
                    },
                ];
            })
            .expect("history should persist");

        enforce_audio_history_budget(&state, 8.0 / 1_073_741_824.0)
            .expect("history audio should prune");

        let database = state.database.lock().expect("database should be readable");
        assert!(database.dictation_history[0].audio_file.is_some());
        assert!(database.dictation_history[1].audio_file.is_none());
        assert!(database.dictation_history[1].audio_model.is_none());
        assert!(!older.exists());
        assert!(newer.exists());
        assert!(!orphan.exists());
        drop(database);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn backup_archive_round_trips_persisted_data_without_api_keys() {
        let database = AppDatabase::default();
        let archive = BackupArchive {
            version: BACKUP_VERSION,
            exported_at: Utc::now(),
            database,
        };
        let encoded = serde_json::to_vec(&archive).expect("backup should encode");
        let decoded: BackupArchive =
            serde_json::from_slice(&encoded).expect("backup should decode");
        assert_eq!(decoded.version, BACKUP_VERSION);
        assert_eq!(decoded.database.settings.selected_ai_provider, "openai");
    }

    #[test]
    fn persisted_database_wraps_current_schema_and_reads_legacy_data() {
        let database = AppDatabase::default();
        let wrapped = serde_json::to_string(&PersistedDatabase {
            schema_version: DATABASE_SCHEMA_VERSION,
            data: database.clone(),
        })
        .expect("versioned database serializes");
        let (decoded, needs_migration) =
            decode_persisted_database(&wrapped).expect("versioned database decodes");
        assert!(!needs_migration);
        assert_eq!(decoded.settings.selected_ai_provider, "openai");

        let legacy = serde_json::to_string(&database).expect("legacy database serializes");
        let (decoded, needs_migration) =
            decode_persisted_database(&legacy).expect("legacy database decodes");
        assert!(needs_migration);
        assert_eq!(decoded.settings.selected_ai_provider, "openai");
    }

    #[test]
    fn persisted_database_rejects_a_newer_schema_without_normalizing_it() {
        let contents = format!(
            r#"{{"schemaVersion":{},"data":{}}}"#,
            DATABASE_SCHEMA_VERSION + 1,
            serde_json::to_string(&AppDatabase::default()).expect("database serializes")
        );
        assert!(decode_persisted_database(&contents)
            .expect_err("future schema must be rejected")
            .contains("supports up to"));
    }

    #[test]
    fn database_persistence_writes_a_versioned_envelope() {
        let root =
            std::env::temp_dir().join(format!("voxide-persistence-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).expect("temporary data directory creates");
        let state = AppState {
            database: Mutex::new(AppDatabase::default()),
            path: root.join(DATABASE_FILE),
            startup_recovery_notice: Mutex::new(None),
        };
        let database = state.database.lock().expect("database lock").clone();
        state.persist(&database).expect("database persists");
        let contents = fs::read_to_string(&state.path).expect("database file reads");
        let json: Value = serde_json::from_str(&contents).expect("database JSON parses");
        assert_eq!(json["schemaVersion"], DATABASE_SCHEMA_VERSION);
        assert!(json["data"].is_object());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn database_persistence_rotates_last_known_good_snapshots() {
        let directory = std::env::temp_dir().join(format!(
            "voxide-database-backup-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&directory).expect("test directory");
        let path = directory.join(DATABASE_FILE);
        fs::write(&path, "initial snapshot").expect("initial database");
        let state = AppState {
            database: Mutex::new(AppDatabase::default()),
            path: path.clone(),
            startup_recovery_notice: Mutex::new(None),
        };

        let first = AppDatabase::default();
        state.persist(&first).expect("first persistence");
        let first_snapshot = fs::read_to_string(&path).expect("first live database");
        assert_eq!(
            fs::read_to_string(database_backup_path(&path, 0)).expect("newest backup"),
            "initial snapshot"
        );

        let mut second = AppDatabase::default();
        second.settings.language = "pt".into();
        state.persist(&second).expect("second persistence");
        assert_eq!(
            fs::read_to_string(database_backup_path(&path, 0)).expect("newest backup"),
            first_snapshot
        );
        assert_eq!(
            fs::read_to_string(database_backup_path(&path, 1)).expect("previous backup"),
            "initial snapshot"
        );
        assert!(!database_backup_path(&path, DATABASE_BACKUP_COUNT).exists());

        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn newest_valid_database_backup_skips_corrupt_newer_snapshots() {
        let directory = std::env::temp_dir().join(format!(
            "voxide-database-recovery-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&directory).expect("test directory");
        let path = directory.join(DATABASE_FILE);
        fs::write(database_backup_path(&path, 0), "{not valid JSON")
            .expect("corrupt newest backup");
        let expected = AppDatabase::default();
        fs::write(
            database_backup_path(&path, 1),
            serde_json::to_string(&PersistedDatabase {
                schema_version: DATABASE_SCHEMA_VERSION,
                data: expected.clone(),
            })
            .expect("backup serializes"),
        )
        .expect("valid older backup");

        let (recovered, needs_migration, source) =
            newest_valid_database_backup(&path).expect("older backup recovers");
        assert!(!needs_migration);
        assert_eq!(source, database_backup_path(&path, 1));
        assert_eq!(
            recovered.settings.selected_voice_engine,
            expected.settings.selected_voice_engine
        );

        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn corrupt_database_is_quarantined_without_overwriting_it() {
        let directory = std::env::temp_dir().join(format!(
            "voxide-database-quarantine-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&directory).expect("test directory");
        let path = directory.join(DATABASE_FILE);
        fs::write(&path, "unreadable data").expect("corrupt database");

        let preserved = quarantine_corrupt_database(&path).expect("database is quarantined");

        assert!(!path.exists());
        assert!(preserved.is_file());
        assert_eq!(
            fs::read_to_string(preserved).expect("preserved data"),
            "unreadable data"
        );

        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn component_replacement_activates_a_verified_staging_directory() {
        let root = std::env::temp_dir().join(format!("voxide-component-{}", uuid::Uuid::new_v4()));
        let destination = root.join("component");
        let staging = root.join("staging");
        fs::create_dir_all(&destination).expect("old component directory creates");
        fs::create_dir_all(&staging).expect("staged component directory creates");
        fs::write(destination.join("version"), "old").expect("old component writes");
        fs::write(staging.join("version"), "new").expect("staged component writes");

        replace_component_directory(&staging, &destination).expect("component replacement works");
        assert_eq!(
            fs::read_to_string(destination.join("version")).unwrap(),
            "new"
        );
        assert!(!staging.exists());
        assert_eq!(
            fs::read_dir(&root)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().contains("previous"))
                .count(),
            0
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn nemotron_downloads_use_the_pinned_model_revision() {
        let url = nemotron_model_url("config.json");
        assert!(url.contains(nemotron::MODEL_REVISION));
        assert!(!url.contains("/resolve/main/"));
    }

    #[test]
    fn nemotron_pinned_model_has_a_checksum_for_every_downloaded_file() {
        for file in NEMOTRON_MODEL_FILES {
            let checksum = nemotron_model_sha256(file)
                .unwrap_or_else(|| panic!("missing checksum for {file}"));
            assert_eq!(checksum.len(), 64);
            assert!(checksum.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn parakeet_archive_is_pinned_to_a_digest_and_exact_size() {
        assert_eq!(parakeet::MODEL_ARCHIVE_SHA256.len(), 64);
        assert!(parakeet::MODEL_ARCHIVE_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
        assert!(parakeet::MODEL_ARCHIVE_BYTES > 50 * 1024 * 1024);
    }

    #[test]
    fn vad_download_uses_a_pinned_revision_and_digest() {
        let url = vad_model_url();
        assert!(url.contains(VAD_MODEL_REVISION));
        assert!(!url.contains("/resolve/main/"));
        assert_eq!(VAD_MODEL_SHA256.len(), 64);
        assert!(VAD_MODEL_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(VAD_MODEL_BYTES, 885_098);
    }

    #[test]
    fn component_hashing_is_streamed_and_deterministic() {
        let path =
            std::env::temp_dir().join(format!("voxide-component-hash-{}", uuid::Uuid::new_v4()));
        fs::write(&path, b"voxide").expect("component fixture writes");
        assert_eq!(
            sha256_file(&path).expect("component fixture hashes"),
            "7bcd53ec8f339c1a407912b52875daaf8f04e7d69c19ca831172fb1f3f83343d"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn component_receipt_is_atomic_and_records_verified_files() {
        let directory =
            std::env::temp_dir().join(format!("voxide-component-receipt-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&directory).expect("receipt directory creates");
        fs::write(directory.join("model.bin"), b"voxide").expect("component fixture writes");
        let files =
            component_file_hashes(&directory, ["model.bin"]).expect("component fixture hashes");
        write_component_receipt(
            &directory,
            &ComponentReceipt {
                schema: COMPONENT_RECEIPT_SCHEMA,
                id: "test-component".into(),
                version: "test-version".into(),
                source: "https://example.invalid/component".into(),
                files,
            },
        )
        .expect("receipt writes");
        let receipt: ComponentReceipt = serde_json::from_slice(
            &fs::read(directory.join(COMPONENT_RECEIPT_FILE)).expect("receipt reads"),
        )
        .expect("receipt parses");
        assert_eq!(receipt.id, "test-component");
        assert_eq!(
            receipt.files["model.bin"],
            "7bcd53ec8f339c1a407912b52875daaf8f04e7d69c19ca831172fb1f3f83343d"
        );
        assert!(!directory
            .join(format!("{COMPONENT_RECEIPT_FILE}.tmp"))
            .exists());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn component_receipts_detect_tampering_and_reject_unsafe_paths() {
        let directory =
            std::env::temp_dir().join(format!("voxide-component-verify-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&directory).expect("receipt directory creates");
        fs::write(directory.join("runtime-health.json"), b"verified")
            .expect("component fixture writes");
        write_component_receipt(
            &directory,
            &ComponentReceipt {
                schema: COMPONENT_RECEIPT_SCHEMA,
                id: NEMOTRON_RUNTIME_ID.into(),
                version: NEMOTRON_RUNTIME_VERSION.into(),
                source: "https://example.invalid/runtime".into(),
                files: component_file_hashes(&directory, ["runtime-health.json"])
                    .expect("component fixture hashes"),
            },
        )
        .expect("receipt writes");
        assert!(component_receipt_is_verified(
            &directory,
            NEMOTRON_RUNTIME_ID,
            NEMOTRON_RUNTIME_VERSION,
        ));

        fs::write(directory.join("runtime-health.json"), b"tampered")
            .expect("component fixture tampers");
        assert!(!component_receipt_is_verified(
            &directory,
            NEMOTRON_RUNTIME_ID,
            NEMOTRON_RUNTIME_VERSION,
        ));

        let mut receipt: ComponentReceipt = serde_json::from_slice(
            &fs::read(directory.join(COMPONENT_RECEIPT_FILE)).expect("receipt reads"),
        )
        .expect("receipt parses");
        receipt.files.clear();
        receipt.files.insert("../outside".into(), "anything".into());
        write_component_receipt(&directory, &receipt).expect("unsafe receipt writes");
        assert!(!component_receipt_is_verified(
            &directory,
            NEMOTRON_RUNTIME_ID,
            NEMOTRON_RUNTIME_VERSION,
        ));
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn component_startup_cleanup_removes_only_owned_abandoned_directories() {
        let directory =
            std::env::temp_dir().join(format!("voxide-component-cleanup-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&directory).expect("cleanup directory creates");
        let owned = directory.join(".nemotron-runtime-install-interrupted");
        let unrelated = directory.join("important-user-directory");
        fs::create_dir_all(&owned).expect("owned staging creates");
        fs::create_dir_all(&unrelated).expect("unrelated directory creates");
        assert_eq!(
            cleanup_abandoned_component_directories(&[directory.clone()], Duration::ZERO),
            1
        );
        assert!(!owned.exists());
        assert!(unrelated.exists());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn diagnostic_runtime_versions_accept_only_safe_version_tokens() {
        assert_eq!(
            diagnostic_version_value(&Value::String("2.9.1+cu128".into())),
            Some("2.9.1+cu128")
        );
        assert!(diagnostic_version_value(&Value::String("/home/alice/secret".into())).is_none());
        assert!(diagnostic_version_value(&Value::String("version\nsecret".into())).is_none());
        assert!(diagnostic_version_value(&Value::String("x".repeat(65))).is_none());
    }

    #[test]
    fn diagnostic_model_id_redacts_untrusted_cloud_configuration() {
        let mut settings = Settings::default();
        settings.cloud_transcription_model = "gpt-4o-mini-transcribe".into();
        assert_eq!(
            VoiceEngine::Cloud.diagnostic_model_id(&settings),
            "gpt-4o-mini-transcribe"
        );
        settings.cloud_transcription_model = "https://example.test?api_key=secret".into();
        assert_eq!(
            VoiceEngine::Cloud.diagnostic_model_id(&settings),
            "<redacted-model-id>"
        );
    }

    #[test]
    fn capture_failure_event_uses_the_frontend_session_id_field() {
        let value = serde_json::to_value(CaptureFailure { session_id: 42 })
            .expect("capture failure event serializes");
        assert_eq!(value["sessionId"], 42);
        assert!(value.get("session_id").is_none());
    }

    #[test]
    fn dictionary_transfer_entries_only_include_portable_fields() {
        let transfer = DictionaryTransferEntry {
            spoken: "vox side".into(),
            replacement: "Voxide".into(),
        };
        let json = serde_json::to_string(&transfer).expect("entry should serialize");
        assert!(json.contains("spoken"));
        assert!(!json.contains("createdAt"));
    }

    #[test]
    fn dictionary_transfer_document_keeps_custom_vocabulary_and_reads_legacy_entries() {
        let document = DictionaryTransferDocument {
            version: 1,
            replacements: vec![DictionaryTransferEntry {
                spoken: "vox side".into(),
                replacement: "Voxide".into(),
            }],
            custom_words: vec![CustomWordEntry {
                text: "Acme".into(),
                weight: None,
                aliases: vec!["acme dev".into()],
            }],
        };
        let encoded = serde_json::to_vec(&document).expect("document should serialize");
        let decoded: DictionaryImportDocument =
            serde_json::from_slice(&encoded).expect("document should decode");
        assert!(matches!(decoded, DictionaryImportDocument::Current(_)));

        let legacy: DictionaryImportDocument =
            serde_json::from_str(r#"[{"spoken":"vox side","replacement":"Voxide"}]"#)
                .expect("legacy array should still decode");
        assert!(matches!(legacy, DictionaryImportDocument::Legacy(_)));
    }

    #[test]
    fn custom_vocabulary_uses_the_reference_persisted_normalization_contract() {
        let mut words = vec![
            CustomWordEntry {
                text: "  Voxide  ".into(),
                weight: Some(-3.0),
                aliases: vec![
                    " vox side ".into(),
                    "VOX SIDE".into(),
                    "Voxide".into(),
                    "acme".into(),
                ],
            },
            CustomWordEntry {
                text: "voxide".into(),
                weight: Some(9.0),
                aliases: vec![],
            },
        ];
        words.extend((0..300).map(|index| CustomWordEntry {
            text: format!("term-{index}"),
            weight: None,
            aliases: vec![],
        }));

        let normalized = normalize_custom_words(words);

        assert_eq!(normalized.len(), 256);
        assert_eq!(normalized[0].text, "Voxide");
        assert_eq!(normalized[0].weight, Some(-3.0));
        assert_eq!(normalized[0].aliases, ["acme", "vox side"]);
    }

    #[test]
    fn file_transcription_exports_match_the_text_and_json_contract() {
        let entry = FileTranscriptionEntry {
            id: "file-1".into(),
            file_name: "meeting.m4a".into(),
            text: "Hello from the meeting.".into(),
            created_at: Utc::now(),
            duration_ms: Some(12_500),
            processing_time_ms: Some(2_500),
            confidence: Some(1.0),
        };
        let text = String::from_utf8(
            file_transcription_export(&entry, FileTranscriptionExportFormat::Text)
                .expect("text export encodes"),
        )
        .expect("text export is UTF-8");
        assert!(text.contains("Transcription: meeting.m4a"));
        assert!(text.contains("Duration: 12.5s"));
        assert!(text.contains("Confidence: 100.0%"));
        let json: serde_json::Value = serde_json::from_slice(
            &file_transcription_export(&entry, FileTranscriptionExportFormat::Json)
                .expect("JSON export encodes"),
        )
        .expect("JSON export parses");
        assert_eq!(json["fileName"], "meeting.m4a");
        assert_eq!(json["text"], "Hello from the meeting.");
        assert_eq!(json["duration"], 12.5);
        assert_eq!(json["processingTime"], 2.5);
    }

    #[test]
    fn file_transcription_history_ignores_blank_engine_results() {
        assert!(!should_save_file_transcription(" \n\t "));
        assert!(should_save_file_transcription("Recognized speech"));
    }

    #[test]
    fn usage_stats_calculate_local_activity_streaks_and_records() {
        let now = Local::now();
        let make_entry = |id: &str,
                          text: &str,
                          created_at: DateTime<Local>,
                          app: &str,
                          was_ai_processed: bool| DictationEntry {
            id: id.into(),
            text: text.into(),
            raw_text: Some(text.into()),
            created_at: created_at.with_timezone(&Utc),
            duration_ms: None,
            mode: DictationMode::Dictate,
            source_application: Some(app.into()),
            source_window_title: None,
            audio_file: None,
            audio_model: None,
            was_ai_processed,
            processing_model: was_ai_processed.then_some("model".into()),
            ai_processing_error: None,
        };
        let entries = vec![
            make_entry("today", "one two three", now, "Editor", true),
            make_entry(
                "yesterday",
                "four five",
                now - chrono::Duration::days(1),
                "Editor",
                false,
            ),
            make_entry(
                "older",
                "six",
                now - chrono::Duration::days(5),
                "Browser",
                false,
            ),
        ];
        let mut settings = Settings::default();
        settings.user_typing_wpm = 60;
        settings.weekends_dont_break_streak = false;
        let stats = calculate_usage_stats(&entries, &settings, now);

        assert_eq!(stats.today_dictations, 1);
        assert_eq!(stats.today_words, 3);
        assert_eq!(stats.total_dictations, 3);
        assert_eq!(stats.total_words, 6);
        assert_eq!(stats.total_characters, 25);
        assert_eq!(stats.average_words_per_dictation, 2);
        assert_eq!(stats.ai_processed_count, 1);
        assert_eq!(stats.ai_enhancement_rate, 33);
        assert_eq!(stats.current_streak, 2);
        assert_eq!(stats.best_streak, 2);
        assert_eq!(stats.daily_activity_7.len(), 7);
        assert_eq!(stats.daily_activity_30.len(), 30);
        assert_eq!(stats.top_apps[0].app, "Editor");
        assert_eq!(stats.top_apps[0].count, 2);
        assert_eq!(stats.longest_transcription_words, 3);
        assert_eq!(stats.most_words_in_day, 3);
        assert_eq!(stats.most_transcriptions_in_day, 1);
        assert!((stats.total_time_saved_minutes - 0.06).abs() < f64::EPSILON);
    }

    #[test]
    fn audio_history_export_writes_a_manifest_and_only_saved_wav_files() {
        let root = std::env::temp_dir().join(format!(
            "voxide-audio-export-test-{}-{}",
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(&root).expect("temporary root should be created");
        let state = AppState {
            database: Mutex::new(AppDatabase::default()),
            path: root.join("voxide.json"),
            startup_recovery_notice: Mutex::new(None),
        };
        let audio_path = state
            .audio_history_directory()
            .expect("audio history path should be created")
            .join("saved.wav");
        fs::write(
            &audio_path,
            audio::wav_bytes_from_16khz_mono(&[0.0, 0.2, -0.2]).expect("test WAV should encode"),
        )
        .expect("test audio should be written");
        state
            .update(|database| {
                database.dictation_history.push(DictationEntry {
                    id: "dictation-20260720000000123".into(),
                    text: "Final transcription".into(),
                    raw_text: Some("Raw transcription".into()),
                    created_at: DateTime::parse_from_rfc3339("2026-07-20T12:34:56Z")
                        .expect("fixed timestamp should parse")
                        .with_timezone(&Utc),
                    duration_ms: Some(120),
                    mode: DictationMode::Dictate,
                    source_application: Some("Editor".into()),
                    source_window_title: Some("Document".into()),
                    audio_file: Some(audio_path.display().to_string()),
                    audio_model: Some("base".into()),
                    was_ai_processed: false,
                    processing_model: None,
                    ai_processing_error: None,
                });
            })
            .expect("history should persist");
        let archive_path = root.join("audio-export.zip");
        assert_eq!(
            export_audio_history_archive(&state, None, &archive_path.display().to_string())
                .expect("audio archive should export"),
            1
        );
        let archive_file = fs::File::open(&archive_path).expect("archive should be written");
        let mut archive = zip::ZipArchive::new(archive_file).expect("archive should be valid");
        let mut manifest = String::new();
        archive
            .by_name("manifest.jsonl")
            .expect("archive should contain a manifest")
            .read_to_string(&mut manifest)
            .expect("manifest should be UTF-8");
        let row: serde_json::Value =
            serde_json::from_str(manifest.trim()).expect("manifest row parses");
        assert_eq!(row["rawTranscript"], "Raw transcription");
        assert_eq!(row["finalTranscript"], "Final transcription");
        assert_eq!(row["model"], "base");
        let audio_name = row["audio"].as_str().expect("manifest has audio path");
        assert_eq!(audio_name, "audio/2026-07-20T12-34-56Z_00000123.wav");
        assert!(archive.by_name(audio_name).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn live_preview_keeps_a_provisional_tail_while_committing_repeated_words() {
        let mut preview = WhisperPreviewStability::default();
        assert_eq!(preview.observe("hello world"), Some("hello world".into()));
        assert_eq!(preview.observe("hello word"), Some("hello word".into()));
        assert_eq!(
            preview.observe("hello word again"),
            Some("hello word again".into())
        );
        assert_eq!(
            preview.observe("hello word again today"),
            Some("hello word again today".into())
        );
    }

    #[test]
    fn live_preview_keeps_confirmed_text_when_the_audio_window_moves() {
        let mut preview = WhisperPreviewStability::default();
        assert_eq!(
            preview.observe("one two three"),
            Some("one two three".into())
        );
        assert_eq!(
            preview.observe("one two three four"),
            Some("one two three four".into())
        );
        assert_eq!(
            preview.observe("three four five"),
            Some("one two three four five".into())
        );
        assert_eq!(
            preview.observe("three four five six"),
            Some("one two three four five six".into())
        );
    }

    #[test]
    fn live_preview_never_rewrites_a_confirmed_word() {
        let mut preview = WhisperPreviewStability::default();
        assert_eq!(
            preview.observe("hello world again"),
            Some("hello world again".into())
        );
        assert_eq!(
            preview.observe("hello world again today"),
            Some("hello world again today".into())
        );
        // "world" is confirmed, so a later weak hypothesis cannot turn it
        // into "word" in the visible overlay.
        assert_eq!(
            preview.observe("hello word again today"),
            Some("hello world again today".into())
        );
        assert_eq!(
            preview.observe("hello world again today tomorrow"),
            Some("hello world again today tomorrow".into())
        );
    }

    #[test]
    fn live_preview_stays_visible_when_early_snapshots_do_not_align() {
        let mut preview = WhisperPreviewStability::default();
        assert_eq!(preview.observe("first guess"), Some("first guess".into()));
        assert_eq!(preview.observe("second guess"), Some("second guess".into()));
        assert_eq!(preview.observe("third guess"), Some("third guess".into()));
    }

    #[test]
    fn live_preview_ignores_punctuation_when_matching_snapshots() {
        let mut preview = WhisperPreviewStability::default();
        assert_eq!(
            preview.observe("one, two, three"),
            Some("one, two, three".into())
        );
        assert_eq!(
            preview.observe("one two three test"),
            Some("one, two, three test".into())
        );
    }

    #[test]
    fn live_preview_discards_an_unconfirmed_tail_after_silence() {
        let mut preview = WhisperPreviewStability::default();
        assert_eq!(
            preview.observe("one two three"),
            Some("one two three".into())
        );
        assert_eq!(
            preview.observe("one two three thank you"),
            Some("one two three thank you".into())
        );
        assert_eq!(preview.observe_silence(), Some("one two three".into()));
    }

    #[test]
    fn pause_heavy_preview_fixture_updates_after_each_spoken_section() {
        let mut preview = WhisperPreviewStability::default();
        let mut updates = Vec::new();

        updates.push(
            preview
                .observe("one two three")
                .expect("first spoken section"),
        );
        updates.push(preview.observe_silence().expect("first pause retains text"));
        updates.push(
            preview
                .observe("one two three test again")
                .expect("second spoken section"),
        );
        updates.push(
            preview
                .observe_silence()
                .expect("second pause retains text"),
        );
        updates.push(
            preview
                .observe("one two three test again final words")
                .expect("third spoken section"),
        );

        assert_eq!(updates[0], "one two three");
        assert_eq!(updates[1], "one two three");
        assert_eq!(updates[2], "one two three test again");
        assert_eq!(updates[3], "one two three");
        assert_eq!(updates[4], "one two three test again final words");
    }

    #[test]
    fn parakeet_preview_matches_fluidvoice_full_snapshot_reconciliation() {
        assert_eq!(
            fluidvoice_preview_reconcile("", "one two three"),
            "one two three"
        );
        assert_eq!(
            fluidvoice_preview_reconcile("one two three", "one two three test"),
            "one two three test"
        );
        // A significant correction replaces the preview with FluidVoice's
        // newest full hypothesis instead of retaining a VAD-committed prefix.
        assert_eq!(
            fluidvoice_preview_reconcile("one two three test", "one two tree test again"),
            "one two tree test again"
        );
    }

    #[test]
    fn parakeet_preview_uses_the_same_deterministic_cleanup_as_fluidvoice() {
        let mut settings = Settings::default();
        settings.remove_filler_words_enabled = true;
        settings.auto_convert_punctuation_enabled = true;
        let dictionary = vec![DictionaryEntry {
            id: "dictionary-1".into(),
            spoken: "vox side".into(),
            replacement: "Voxide".into(),
            created_at: Utc::now(),
        }];
        assert_eq!(
            parakeet_preview_cleanup("um vox side literal comma hello", &settings, &dictionary,),
            "Voxide, hello"
        );
        let punctuation_dictionary = vec![DictionaryEntry {
            id: "dictionary-2".into(),
            spoken: "dot word".into(),
            replacement: "literal comma".into(),
            created_at: Utc::now(),
        }];
        assert_eq!(
            parakeet_preview_cleanup("dot word hello", &settings, &punctuation_dictionary),
            ", hello"
        );
    }
}
