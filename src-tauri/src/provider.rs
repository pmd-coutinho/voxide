use std::{collections::BTreeMap, path::Path, sync::Arc, time::Duration};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{audio, media};

/// Five minutes of 16 kHz mono PCM stays well below the common 25 MB
/// transcription upload limit once it is wrapped in WAV.
const CLOUD_TRANSCRIPTION_CHUNK_SECONDS: f64 = 5.0 * 60.0;
const LLM_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const LLM_MAX_NETWORK_ATTEMPTS: usize = 3;

pub type AudioTranscriptionProgress = Arc<dyn Fn(usize, usize) + Send + Sync + 'static>;

/// A single OpenAI-compatible function call retained with Command Mode history.
/// Keeping the call ID is important: the next provider request must associate a
/// terminal result with the command that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub enum CommandProviderMessage {
    User(String),
    Assistant {
        content: String,
        tool_call: Option<CommandToolCall>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[derive(Debug, Clone)]
pub enum CommandProviderResponse {
    Text {
        content: String,
        thinking: Option<String>,
    },
    ToolCall {
        content: String,
        thinking: Option<String>,
        tool_call: CommandToolCall,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiProviderProfile {
    pub id: String,
    pub name: String,
    pub api_style: ProviderApiStyle,
    pub base_url: String,
    pub model: String,
    pub enabled: bool,
    /// Request options resolved from the persisted provider/model configuration.
    /// This is intentionally runtime-only: the settings store owns the mapping
    /// so changing a provider model does not accidentally carry its options to
    /// another model.
    #[serde(skip)]
    pub request_parameters: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProviderApiStyle {
    OpenAiCompatible,
    Anthropic,
}

impl AiProviderProfile {
    pub fn built_in() -> Vec<Self> {
        vec![
            Self::openai("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4.1"),
            Self {
                id: "anthropic".into(),
                name: "Anthropic".into(),
                api_style: ProviderApiStyle::Anthropic,
                base_url: "https://api.anthropic.com/v1".into(),
                model: "claude-sonnet-4-20250514".into(),
                enabled: true,
                request_parameters: BTreeMap::new(),
            },
            Self::openai("xai", "xAI", "https://api.x.ai/v1", "grok-3-fast"),
            Self::openai(
                "groq",
                "Groq",
                "https://api.groq.com/openai/v1",
                "openai/gpt-oss-120b",
            ),
            Self::openai(
                "cerebras",
                "Cerebras",
                "https://api.cerebras.ai/v1",
                "gpt-oss-120b",
            ),
            Self::openai(
                "google",
                "Google Gemini",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "gemini-2.5-flash",
            ),
            Self::openai(
                "openrouter",
                "OpenRouter",
                "https://openrouter.ai/api/v1",
                "openai/gpt-oss-20b",
            ),
            Self::openai("ollama", "Ollama", "http://localhost:11434/v1", ""),
            Self::openai("lmstudio", "LM Studio", "http://localhost:1234/v1", ""),
        ]
    }

    fn openai(id: &str, name: &str, base_url: &str, model: &str) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api_style: ProviderApiStyle::OpenAiCompatible,
            base_url: base_url.into(),
            model: model.into(),
            enabled: true,
            request_parameters: BTreeMap::new(),
        }
    }
}

/// The source app exposes these only for its built-in catalog. Keeping the
/// mapping in the provider layer means custom profiles never receive an
/// arbitrary URL-opening capability through the frontend.
pub fn provider_website(provider_id: &str) -> Option<(&'static str, &'static str)> {
    match provider_id {
        "openai" => Some(("https://platform.openai.com/api-keys", "Get API key")),
        "anthropic" => Some(("https://platform.claude.com/settings/keys", "Get API key")),
        "xai" => Some(("https://console.x.ai/", "Get API key")),
        "groq" => Some(("https://console.groq.com/keys", "Get API key")),
        "cerebras" => Some(("https://cloud.cerebras.ai/platform", "Get API key")),
        "google" => Some(("https://aistudio.google.com/apikey", "Get API key")),
        "openrouter" => Some(("https://openrouter.ai/settings/keys", "Get API key")),
        "ollama" => Some((
            "https://docs.ollama.com/api/openai-compatibility",
            "Setup guide",
        )),
        "lmstudio" => Some(("https://lmstudio.ai/docs/local-server", "Setup guide")),
        _ => None,
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    stream: bool,
    #[serde(flatten)]
    request_parameters: &'a BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
struct OpenAiResponsesRequest<'a> {
    model: &'a str,
    input: Vec<OpenAiMessage<'a>>,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    stream: bool,
    #[serde(flatten)]
    request_parameters: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessageResponse,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessageResponse {
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(flatten)]
    request_parameters: &'a BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AudioTranscriptionResponse {
    text: String,
}

pub async fn transcribe_openai_compatible_audio(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    model: &str,
    language: &str,
    wav_bytes: Vec<u8>,
) -> Result<String, String> {
    if !profile.enabled {
        return Err(format!("{} is disabled", profile.name));
    }
    if !matches!(profile.api_style, ProviderApiStyle::OpenAiCompatible) {
        return Err(format!(
            "{} does not expose an OpenAI-compatible audio transcription endpoint",
            profile.name
        ));
    }
    if model.trim().is_empty() {
        return Err("Choose a cloud transcription model before recording".into());
    }
    let endpoint = if profile.base_url.contains("audio/transcriptions") {
        profile.base_url.clone()
    } else {
        join_endpoint(&profile.base_url, "audio/transcriptions")
    };
    let file = reqwest::multipart::Part::bytes(wav_bytes)
        .file_name("voxide-dictation.wav")
        .mime_str("audio/wav")
        .map_err(|error| format!("Could not prepare audio for {}: {error}", profile.name))?;
    let mut form = reqwest::multipart::Form::new()
        .text("model", model.trim().to_owned())
        .part("file", file);
    if !language.trim().is_empty() {
        form = form.text("language", language.trim().to_owned());
    }
    let mut request = reqwest::Client::new().post(endpoint).multipart(form);
    if !is_local_endpoint(&profile.base_url) {
        let key = api_key
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| format!("An API key is required for {}", profile.name))?;
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("{} transcription request failed: {error}", profile.name))?;
    let status = response.status();
    let body = response.text().await.map_err(|error| {
        format!(
            "Could not read {} transcription response: {error}",
            profile.name
        )
    })?;
    if !status.is_success() {
        return Err(format!(
            "{} transcription returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }
    let response: AudioTranscriptionResponse = serde_json::from_str(&body).map_err(|error| {
        format!(
            "{} returned an unsupported transcription response: {error}",
            profile.name
        )
    })?;
    (!response.text.trim().is_empty())
        .then_some(response.text)
        .ok_or_else(|| format!("{} returned an empty transcription", profile.name))
}

pub async fn transcribe_openai_compatible_media(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    model: &str,
    language: &str,
    path: &Path,
    progress: Option<AudioTranscriptionProgress>,
) -> Result<(String, u64), String> {
    let duration_ms = media::file_duration_ms(path)?;
    let total_chunks = ((duration_ms as f64 / 1000.0) / CLOUD_TRANSCRIPTION_CHUNK_SECONDS)
        .ceil()
        .max(1.0) as usize;
    let mut chunks = Vec::new();

    for chunk in 0..total_chunks {
        let start_seconds = chunk as f64 * CLOUD_TRANSCRIPTION_CHUNK_SECONDS;
        let remaining_seconds = (duration_ms as f64 / 1000.0 - start_seconds).max(0.0);
        let audio = media::decode_audio_segment(
            path,
            start_seconds,
            remaining_seconds.min(CLOUD_TRANSCRIPTION_CHUNK_SECONDS),
        )?;
        let samples = audio::mono_resample_for_whisper(audio)?;
        if !audio::has_minimum_transcription_samples(&samples) {
            if let Some(progress) = &progress {
                progress(chunk + 1, total_chunks);
            }
            continue;
        }
        let wav = audio::wav_bytes_from_16khz_mono(&samples)?;
        let text =
            transcribe_openai_compatible_audio(profile, api_key, model, language, wav).await?;
        if !text.trim().is_empty() {
            chunks.push(text);
        }
        if let Some(progress) = &progress {
            progress(chunk + 1, total_chunks);
        }
    }

    let text = chunks.join(" ");
    Ok((text, duration_ms))
}

pub async fn process_with_options(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    input: &str,
    temperature: f32,
    max_tokens: Option<u32>,
) -> Result<String, String> {
    process_with_options_timeout(
        profile,
        api_key,
        system_prompt,
        input,
        LLM_REQUEST_TIMEOUT,
        temperature,
        max_tokens,
    )
    .await
}

/// Runs a complete non-streaming request with the caller's end-to-end timeout.
/// Voxide uses this for regular Rewrite and its longer dictation cleanup
/// path, where partial output is intentionally not inserted into the overlay.
pub async fn process_with_options_timeout(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    input: &str,
    timeout: Duration,
    temperature: f32,
    max_tokens: Option<u32>,
) -> Result<String, String> {
    if !profile.enabled {
        return Err(format!("{} is disabled", profile.name));
    }
    if profile.model.trim().is_empty() {
        return Err(format!(
            "Choose a model for {} before using AI enhancement",
            profile.name
        ));
    }

    match profile.api_style {
        ProviderApiStyle::OpenAiCompatible => {
            process_openai_compatible(
                profile,
                api_key,
                system_prompt,
                input,
                timeout,
                temperature,
                max_tokens,
            )
            .await
        }
        ProviderApiStyle::Anthropic => {
            process_anthropic(
                profile,
                api_key,
                system_prompt,
                input,
                timeout,
                temperature,
                max_tokens,
            )
            .await
        }
    }
}

pub async fn process_streaming_with_options<F>(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    input: &str,
    timeout: Duration,
    temperature: f32,
    max_tokens: Option<u32>,
    mut on_delta: F,
) -> Result<String, String>
where
    F: FnMut(&str) + Send,
{
    if !profile.enabled {
        return Err(format!("{} is disabled", profile.name));
    }
    if profile.model.trim().is_empty() {
        return Err(format!(
            "Choose a model for {} before using AI enhancement",
            profile.name
        ));
    }
    match profile.api_style {
        ProviderApiStyle::OpenAiCompatible => {
            process_openai_compatible_streaming(
                profile,
                api_key,
                system_prompt,
                input,
                timeout,
                temperature,
                max_tokens,
                &mut on_delta,
            )
            .await
        }
        ProviderApiStyle::Anthropic => {
            process_anthropic_streaming(
                profile,
                api_key,
                system_prompt,
                input,
                timeout,
                temperature,
                max_tokens,
                &mut on_delta,
            )
            .await
        }
    }
}

/// Ask a provider to either answer Command Mode or call the terminal tool.
/// Tool arguments are fully accumulated before a response is returned, so the
/// caller still has one complete, reviewable command to present to the user.
pub async fn process_command_with_tools(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
) -> Result<CommandProviderResponse, String> {
    if !profile.enabled {
        return Err(format!("{} is disabled", profile.name));
    }
    if profile.model.trim().is_empty() {
        return Err(format!(
            "Choose a model for {} before using Command Mode",
            profile.name
        ));
    }
    match profile.api_style {
        ProviderApiStyle::OpenAiCompatible => {
            process_openai_command_with_tools(profile, api_key, system_prompt, messages).await
        }
        ProviderApiStyle::Anthropic => {
            process_anthropic_command_with_tools(profile, api_key, system_prompt, messages).await
        }
    }
}

/// Streaming Command Mode equivalent of [`process_command_with_tools`]. Text
/// and display-only reasoning deltas are delivered independently while
/// function-call arguments are kept private until their JSON object is
/// complete.
pub async fn process_command_with_tools_streaming<F, G>(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
    mut on_content: F,
    mut on_thinking: G,
) -> Result<CommandProviderResponse, String>
where
    F: FnMut(&str) + Send,
    G: FnMut(&str) + Send,
{
    if !profile.enabled {
        return Err(format!("{} is disabled", profile.name));
    }
    if profile.model.trim().is_empty() {
        return Err(format!(
            "Choose a model for {} before using Command Mode",
            profile.name
        ));
    }
    match profile.api_style {
        ProviderApiStyle::OpenAiCompatible => {
            process_openai_command_with_tools_streaming(
                profile,
                api_key,
                system_prompt,
                messages,
                &mut on_content,
                &mut on_thinking,
            )
            .await
        }
        ProviderApiStyle::Anthropic => {
            process_anthropic_command_with_tools_streaming(
                profile,
                api_key,
                system_prompt,
                messages,
                &mut on_content,
                &mut on_thinking,
            )
            .await
        }
    }
}

async fn process_openai_command_with_tools(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
) -> Result<CommandProviderResponse, String> {
    let use_responses_api = uses_openai_responses_api(profile);
    let endpoint = openai_endpoint(profile, use_responses_api);
    let client = llm_client()?;
    let request = openai_request_builder(profile, api_key, client.post(endpoint))?;
    let payload = if use_responses_api {
        openai_responses_command_request(profile, system_prompt, messages)?
    } else {
        openai_chat_command_request(profile, system_prompt, messages)?
    };
    let response = send_llm_request(profile, request.json(&payload)).await?;
    let status = response.status();
    let body = response.text().await.map_err(|error| {
        format!(
            "Could not read {} Command Mode response: {error}",
            profile.name
        )
    })?;
    if !status.is_success() {
        return Err(format!(
            "{} Command Mode tools returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }
    let response: Value = serde_json::from_str(&body).map_err(|error| {
        format!(
            "{} returned an unsupported Command Mode tools response: {error}",
            profile.name
        )
    })?;
    parse_openai_command_response(&response, use_responses_api)
}

async fn process_anthropic_command_with_tools(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
) -> Result<CommandProviderResponse, String> {
    let key = api_key
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| format!("An API key is required for {}", profile.name))?;
    let request = llm_client()?
        .post(anthropic_messages_endpoint(&profile.base_url))
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&anthropic_command_request(
            profile,
            system_prompt,
            messages,
        )?);
    let response = send_llm_request(profile, request).await?;
    let status = response.status();
    let body = response.text().await.map_err(|error| {
        format!(
            "Could not read {} Command Mode response: {error}",
            profile.name
        )
    })?;
    if !status.is_success() {
        return Err(format!(
            "{} Command Mode tools returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }
    let response: Value = serde_json::from_str(&body).map_err(|error| {
        format!(
            "{} returned an unsupported Command Mode tools response: {error}",
            profile.name
        )
    })?;
    parse_anthropic_command_response(&response)
}

async fn process_openai_command_with_tools_streaming(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) -> Result<CommandProviderResponse, String> {
    let use_responses_api = uses_openai_responses_api(profile);
    let endpoint = openai_endpoint(profile, use_responses_api);
    let client = llm_client()?;
    let request = openai_request_builder(profile, api_key, client.post(endpoint))?;
    let mut payload = if use_responses_api {
        openai_responses_command_request(profile, system_prompt, messages)?
    } else {
        openai_chat_command_request(profile, system_prompt, messages)?
    };
    payload["stream"] = Value::Bool(true);
    consume_command_stream(
        profile,
        send_llm_request(profile, request.timeout(LLM_REQUEST_TIMEOUT).json(&payload)).await,
        if use_responses_api {
            CommandStreamProtocol::OpenAiResponses
        } else {
            CommandStreamProtocol::OpenAiChat
        },
        on_content,
        on_thinking,
    )
    .await
}

async fn process_anthropic_command_with_tools_streaming(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) -> Result<CommandProviderResponse, String> {
    let key = api_key
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| format!("An API key is required for {}", profile.name))?;
    let mut payload = anthropic_command_request(profile, system_prompt, messages)?;
    payload["stream"] = Value::Bool(true);
    let request = llm_client()?
        .post(anthropic_messages_endpoint(&profile.base_url))
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .timeout(LLM_REQUEST_TIMEOUT)
        .json(&payload);
    consume_command_stream(
        profile,
        send_llm_request(profile, request).await,
        CommandStreamProtocol::Anthropic,
        on_content,
        on_thinking,
    )
    .await
}

fn terminal_command_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "execute_terminal_command",
            "description": "Propose one terminal or shell command for the user's requested desktop task. The user reviews the exact command before Voxide executes it. Use this for file operations, git, package managers, or other CLI actions. Check prerequisites, carry out one action, or verify a completed action as appropriate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run."
                    },
                    "workingDirectory": {
                        "type": "string",
                        "description": "Optional working directory."
                    },
                    "purpose": {
                        "type": "string",
                        "description": "A brief explanation of why this command is needed."
                    }
                },
                "required": ["command", "purpose"]
            }
        }
    })
}

fn openai_chat_command_request(
    profile: &AiProviderProfile,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
) -> Result<Value, String> {
    let mut request_messages = vec![json!({
        "role": "system",
        "content": system_prompt,
    })];
    request_messages.extend(command_messages_for_chat(messages)?);
    let mut body = json!({
        "model": profile.model,
        "messages": request_messages,
        "stream": false,
        "tools": [terminal_command_tool()],
        "tool_choice": "auto",
    });
    if !is_temperature_unsupported(&profile.model) {
        body["temperature"] = json!(0.1);
    }
    if is_reasoning_model(&profile.model) {
        body["max_completion_tokens"] = json!(32_000);
    }
    apply_request_parameters(&mut body, profile, false);
    Ok(body)
}

fn command_messages_for_chat(messages: &[CommandProviderMessage]) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| match message {
            CommandProviderMessage::User(content) => Ok(json!({
                "role": "user",
                "content": content,
            })),
            CommandProviderMessage::Assistant { content, tool_call } => {
                let mut message = json!({
                    "role": "assistant",
                    "content": content,
                });
                if let Some(tool_call) = tool_call {
                    let arguments =
                        serde_json::to_string(&tool_call.arguments).map_err(|error| {
                            format!("Could not encode stored Command Mode tool arguments: {error}")
                        })?;
                    message["tool_calls"] = json!([{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": arguments,
                        }
                    }]);
                }
                Ok(message)
            }
            CommandProviderMessage::Tool {
                content,
                tool_call_id,
            } => Ok(json!({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            })),
        })
        .collect()
}

fn openai_responses_command_request(
    profile: &AiProviderProfile,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
) -> Result<Value, String> {
    let mut input = vec![json!({
        "role": "system",
        "content": system_prompt,
    })];
    for message in messages {
        match message {
            CommandProviderMessage::User(content) => input.push(json!({
                "role": "user",
                "content": content,
            })),
            CommandProviderMessage::Assistant { content, tool_call } => {
                if !content.trim().is_empty() {
                    input.push(json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
                if let Some(tool_call) = tool_call {
                    let arguments =
                        serde_json::to_string(&tool_call.arguments).map_err(|error| {
                            format!("Could not encode stored Command Mode tool arguments: {error}")
                        })?;
                    input.push(json!({
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": arguments,
                    }));
                }
            }
            CommandProviderMessage::Tool {
                content,
                tool_call_id,
            } => input.push(json!({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": content,
            })),
        }
    }
    let tool = terminal_command_tool();
    let function = tool
        .get("function")
        .cloned()
        .ok_or("Could not construct the Command Mode tool definition")?;
    let mut response_tool = json!({
        "type": "function",
        "name": function["name"],
        "description": function["description"],
        "parameters": function["parameters"],
        "strict": false,
    });
    // Avoid sending a null description when a provider validates the schema
    // more strictly than OpenAI's endpoint.
    if response_tool["description"].is_null() {
        response_tool
            .as_object_mut()
            .expect("response tool is an object")
            .remove("description");
    }
    let mut body = json!({
        "model": profile.model,
        "input": input,
        "store": false,
        "stream": false,
        "tools": [response_tool],
        "tool_choice": "auto",
    });
    if !is_temperature_unsupported(&profile.model) {
        body["temperature"] = json!(0.1);
    }
    if is_reasoning_model(&profile.model) {
        body["max_output_tokens"] = json!(32_000);
    }
    apply_request_parameters(&mut body, profile, true);
    Ok(body)
}

fn anthropic_command_request(
    profile: &AiProviderProfile,
    system_prompt: &str,
    messages: &[CommandProviderMessage],
) -> Result<Value, String> {
    let mut request_messages = Vec::new();
    for message in messages {
        match message {
            CommandProviderMessage::User(content) => request_messages.push(json!({
                "role": "user",
                "content": content,
            })),
            CommandProviderMessage::Assistant { content, tool_call } => {
                let mut content_blocks = Vec::new();
                if !content.trim().is_empty() {
                    content_blocks.push(json!({ "type": "text", "text": content }));
                }
                if let Some(tool_call) = tool_call {
                    content_blocks.push(json!({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments,
                    }));
                }
                if content_blocks.is_empty() {
                    content_blocks.push(json!({ "type": "text", "text": "" }));
                }
                request_messages.push(json!({
                    "role": "assistant",
                    "content": content_blocks,
                }));
            }
            CommandProviderMessage::Tool {
                content,
                tool_call_id,
            } => request_messages.push(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }],
            })),
        }
    }
    let tool = terminal_command_tool();
    let function = tool
        .get("function")
        .ok_or("Could not construct the Command Mode tool definition")?;
    let mut body = json!({
        "model": profile.model,
        "max_tokens": if is_reasoning_model(&profile.model) { 32_000 } else { 4_096 },
        "system": system_prompt,
        "messages": request_messages,
        "tools": [{
            "name": function["name"],
            "description": function["description"],
            "input_schema": function["parameters"],
        }],
        "tool_choice": { "type": "auto" },
    });
    if !is_temperature_unsupported(&profile.model) {
        body["temperature"] = json!(0.1);
    }
    apply_request_parameters(&mut body, profile, false);
    Ok(body)
}

fn apply_request_parameters(body: &mut Value, profile: &AiProviderProfile, responses_api: bool) {
    let Some(body) = body.as_object_mut() else {
        return;
    };
    for (name, value) in &profile.request_parameters {
        if responses_api && name == "reasoning_effort" {
            body.insert("reasoning".into(), json!({ "effort": value }));
        } else {
            body.insert(name.clone(), value.clone());
        }
    }
}

fn responses_request_parameters(profile: &AiProviderProfile) -> BTreeMap<String, Value> {
    profile
        .request_parameters
        .iter()
        .map(|(name, value)| {
            if name == "reasoning_effort" {
                ("reasoning".into(), json!({ "effort": value }))
            } else {
                (name.clone(), value.clone())
            }
        })
        .collect()
}

fn parse_openai_command_response(
    response: &Value,
    responses_api: bool,
) -> Result<CommandProviderResponse, String> {
    if responses_api {
        let output = response
            .get("output")
            .and_then(Value::as_array)
            .ok_or("The AI provider returned an unsupported Responses API Command Mode response")?;
        let mut content = String::new();
        let mut thinking = String::new();
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    for part in item
                        .get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        if part.get("type").and_then(Value::as_str) == Some("output_text") {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                content.push_str(text);
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let tool_call = command_tool_call_from_values(
                        item.get("call_id").or_else(|| item.get("id")),
                        item.get("name"),
                        item.get("arguments"),
                    )?;
                    let (content, tagged_thinking) = split_nonstream_thinking_tags(&content);
                    return Ok(CommandProviderResponse::ToolCall {
                        content,
                        thinking: combined_thinking(&thinking, tagged_thinking.as_deref()),
                        tool_call,
                    });
                }
                Some("reasoning") => append_reasoning_value(item, &mut thinking),
                _ => {}
            }
        }
        return nonempty_command_text_with_thinking(content, Some(thinking));
    }

    let message = response
        .pointer("/choices/0/message")
        .ok_or("The AI provider returned an unsupported Command Mode response")?;
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let mut thinking = String::new();
    append_reasoning_value(message, &mut thinking);
    if let Some(tool) = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .and_then(|calls| calls.first())
    {
        let tool_call = command_tool_call_from_values(
            tool.get("id"),
            tool.pointer("/function/name"),
            tool.pointer("/function/arguments"),
        )?;
        let (content, tagged_thinking) = split_nonstream_thinking_tags(&content);
        return Ok(CommandProviderResponse::ToolCall {
            content,
            thinking: combined_thinking(&thinking, tagged_thinking.as_deref()),
            tool_call,
        });
    }
    nonempty_command_text_with_thinking(content, Some(thinking))
}

fn parse_anthropic_command_response(response: &Value) -> Result<CommandProviderResponse, String> {
    let content_blocks = response
        .get("content")
        .and_then(Value::as_array)
        .ok_or("The AI provider returned an unsupported Anthropic Command Mode response")?;
    let mut content = String::new();
    let mut thinking = String::new();
    for block in content_blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    content.push_str(text);
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.trim().is_empty())
                    .ok_or("The AI provider returned an Anthropic Command Mode tool call without an ID")?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.trim().is_empty())
                    .ok_or("The AI provider returned an Anthropic Command Mode tool call without a name")?;
                let arguments = block
                    .get("input")
                    .filter(|input| input.is_object())
                    .cloned()
                    .ok_or("The AI provider returned Anthropic Command Mode tool arguments that are not an object")?;
                let (content, tagged_thinking) = split_nonstream_thinking_tags(&content);
                return Ok(CommandProviderResponse::ToolCall {
                    content,
                    thinking: combined_thinking(&thinking, tagged_thinking.as_deref()),
                    tool_call: CommandToolCall {
                        id: id.to_owned(),
                        name: name.to_owned(),
                        arguments,
                    },
                });
            }
            Some("thinking") => append_reasoning_value(block, &mut thinking),
            _ => {}
        }
    }
    nonempty_command_text_with_thinking(content, Some(thinking))
}

fn command_tool_call_from_values(
    id: Option<&Value>,
    name: Option<&Value>,
    arguments: Option<&Value>,
) -> Result<CommandToolCall, String> {
    let id = id
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or("The AI provider returned a Command Mode tool call without an ID")?;
    let name = name
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or("The AI provider returned a Command Mode tool call without a name")?;
    let arguments = arguments
        .and_then(Value::as_str)
        .ok_or("The AI provider returned Command Mode tool arguments in an unsupported format")?;
    let arguments: Value = serde_json::from_str(arguments)
        .map_err(|_| "The AI provider returned invalid Command Mode tool arguments")?;
    if !arguments.is_object() {
        return Err(
            "The AI provider returned Command Mode tool arguments that are not an object".into(),
        );
    }
    Ok(CommandToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments,
    })
}

#[derive(Clone, Copy)]
enum CommandStreamProtocol {
    OpenAiChat,
    OpenAiResponses,
    Anthropic,
}

#[derive(Default)]
struct CommandStreamToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Default)]
struct CommandStreamState {
    content: String,
    thinking_filter: StreamingThinkingFilter,
    thinking: String,
    tool_calls: BTreeMap<usize, CommandStreamToolCall>,
}

async fn consume_command_stream(
    profile: &AiProviderProfile,
    response: Result<reqwest::Response, String>,
    protocol: CommandStreamProtocol,
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) -> Result<CommandProviderResponse, String> {
    let response = response?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "{} Command Mode tools returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }

    let mut state = CommandStreamState::default();
    let mut buffer = String::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("{} stream failed: {error}", profile.name))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(index) = buffer.find('\n') {
            let line = buffer.drain(..=index).collect::<String>();
            append_command_stream_event(&line, protocol, &mut state, on_content, on_thinking);
        }
    }
    if !buffer.trim().is_empty() {
        append_command_stream_event(&buffer, protocol, &mut state, on_content, on_thinking);
    }
    state
        .thinking_filter
        .finish_with_thinking(&mut state.content, on_content, on_thinking);
    let thinking = combined_thinking(&state.thinking, state.thinking_filter.thinking().as_deref());

    if let Some((_, tool_call)) = state.tool_calls.into_iter().next() {
        let id = tool_call
            .id
            .filter(|id| !id.trim().is_empty())
            .ok_or("The AI provider returned a streamed Command Mode tool call without an ID")?;
        let name = tool_call
            .name
            .filter(|name| !name.trim().is_empty())
            .ok_or("The AI provider returned a streamed Command Mode tool call without a name")?;
        let arguments: Value = serde_json::from_str(&tool_call.arguments).map_err(|_| {
            "The AI provider returned invalid streamed Command Mode tool arguments".to_string()
        })?;
        if !arguments.is_object() {
            return Err(
                "The AI provider returned streamed Command Mode tool arguments that are not an object"
                    .into(),
            );
        }
        return Ok(CommandProviderResponse::ToolCall {
            content: state.content,
            thinking,
            tool_call: CommandToolCall {
                id,
                name,
                arguments,
            },
        });
    }
    nonempty_command_text_with_thinking(state.content, thinking)
}

fn append_command_stream_event(
    line: &str,
    protocol: CommandStreamProtocol,
    state: &mut CommandStreamState,
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) {
    let line = line.trim();
    let payload = line.strip_prefix("data:").unwrap_or(line).trim();
    if payload.is_empty() || payload == "[DONE]" {
        return;
    }
    let Ok(value) = serde_json::from_str::<Value>(payload) else {
        return;
    };
    match protocol {
        CommandStreamProtocol::OpenAiChat => {
            append_openai_chat_command_event(&value, state, on_content, on_thinking)
        }
        CommandStreamProtocol::OpenAiResponses => {
            append_openai_responses_command_event(&value, state, on_content, on_thinking)
        }
        CommandStreamProtocol::Anthropic => {
            append_anthropic_command_event(&value, state, on_content, on_thinking)
        }
    }
}

fn append_command_content(
    content: Option<&str>,
    state: &mut CommandStreamState,
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) {
    if let Some(content) = content.filter(|content| !content.is_empty()) {
        state.thinking_filter.push_with_thinking(
            content,
            &mut state.content,
            on_content,
            on_thinking,
        );
    }
}

fn append_command_thinking(
    thinking: Option<&str>,
    state: &mut CommandStreamState,
    on_thinking: &mut (dyn FnMut(&str) + Send),
) {
    if let Some(thinking) = thinking.filter(|thinking| !thinking.is_empty()) {
        state.thinking.push_str(thinking);
        on_thinking(thinking);
    }
}

fn append_openai_chat_command_event(
    value: &Value,
    state: &mut CommandStreamState,
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) {
    let Some(delta) = value.pointer("/choices/0/delta") else {
        return;
    };
    append_command_content(
        delta.get("content").and_then(Value::as_str),
        state,
        on_content,
        on_thinking,
    );
    append_command_thinking(
        delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .or_else(|| delta.get("thinking"))
            .and_then(Value::as_str),
        state,
        on_thinking,
    );
    for (position, call) in delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let index = call
            .get("index")
            .and_then(Value::as_u64)
            .map(|index| index as usize)
            .unwrap_or(position);
        let accumulated = state.tool_calls.entry(index).or_default();
        if let Some(id) = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            accumulated.id = Some(id.to_owned());
        }
        if let Some(function) = call.get("function") {
            if let Some(name) = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            {
                accumulated.name = Some(name.to_owned());
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                accumulated.arguments.push_str(arguments);
            }
        }
    }
}

fn append_openai_responses_command_event(
    value: &Value,
    state: &mut CommandStreamState,
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) {
    let event_type = value.get("type").and_then(Value::as_str);
    match event_type {
        Some("response.output_text.delta") => append_command_content(
            value.get("delta").and_then(Value::as_str),
            state,
            on_content,
            on_thinking,
        ),
        Some("response.reasoning_summary_text.delta")
        | Some("response.reasoning_text.delta")
        | Some("response.reasoning.delta") => append_command_thinking(
            value.get("delta").and_then(Value::as_str),
            state,
            on_thinking,
        ),
        Some("response.output_item.added") => {
            let Some(item) = value.get("item") else {
                return;
            };
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                return;
            }
            let index = value
                .get("output_index")
                .and_then(Value::as_u64)
                .map(|index| index as usize)
                .unwrap_or(0);
            let accumulated = state.tool_calls.entry(index).or_default();
            if let Some(id) = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                accumulated.id = Some(id.to_owned());
            }
            if let Some(name) = item
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            {
                accumulated.name = Some(name.to_owned());
            }
            if let Some(arguments) = item
                .get("arguments")
                .and_then(Value::as_str)
                .filter(|arguments| !arguments.is_empty())
            {
                accumulated.arguments = arguments.to_owned();
            }
        }
        Some("response.function_call_arguments.delta") => {
            let index = value
                .get("output_index")
                .and_then(Value::as_u64)
                .map(|index| index as usize)
                .unwrap_or(0);
            let accumulated = state.tool_calls.entry(index).or_default();
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                accumulated.arguments.push_str(delta);
            }
        }
        Some("response.function_call_arguments.done") => {
            let index = value
                .get("output_index")
                .and_then(Value::as_u64)
                .map(|index| index as usize)
                .unwrap_or(0);
            let accumulated = state.tool_calls.entry(index).or_default();
            if let Some(id) = value
                .get("call_id")
                .or_else(|| value.get("item_id"))
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                accumulated.id = Some(id.to_owned());
            }
            if let Some(name) = value
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            {
                accumulated.name = Some(name.to_owned());
            }
            if let Some(arguments) = value.get("arguments").and_then(Value::as_str) {
                accumulated.arguments = arguments.to_owned();
            }
        }
        _ => {}
    }
}

fn append_anthropic_command_event(
    value: &Value,
    state: &mut CommandStreamState,
    on_content: &mut (dyn FnMut(&str) + Send),
    on_thinking: &mut (dyn FnMut(&str) + Send),
) {
    let index = value
        .get("index")
        .and_then(Value::as_u64)
        .map(|index| index as usize)
        .unwrap_or(0);
    match value.get("type").and_then(Value::as_str) {
        Some("content_block_start") => {
            let Some(block) = value.get("content_block") else {
                return;
            };
            if block.get("type").and_then(Value::as_str) == Some("thinking") {
                append_command_thinking(
                    block.get("thinking").and_then(Value::as_str),
                    state,
                    on_thinking,
                );
                return;
            }
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                return;
            }
            let accumulated = state.tool_calls.entry(index).or_default();
            if let Some(id) = block
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                accumulated.id = Some(id.to_owned());
            }
            if let Some(name) = block
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            {
                accumulated.name = Some(name.to_owned());
            }
            if let Some(input) = block
                .get("input")
                .filter(|input| input.as_object().is_some_and(|input| !input.is_empty()))
            {
                accumulated.arguments = input.to_string();
            }
        }
        Some("content_block_delta") => {
            let Some(delta) = value.get("delta") else {
                return;
            };
            match delta.get("type").and_then(Value::as_str) {
                Some("thinking_delta") => append_command_thinking(
                    delta.get("thinking").and_then(Value::as_str),
                    state,
                    on_thinking,
                ),
                Some("text_delta") => append_command_content(
                    delta.get("text").and_then(Value::as_str),
                    state,
                    on_content,
                    on_thinking,
                ),
                Some("input_json_delta") => {
                    if let Some(partial_json) = delta.get("partial_json").and_then(Value::as_str) {
                        state
                            .tool_calls
                            .entry(index)
                            .or_default()
                            .arguments
                            .push_str(partial_json);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn append_reasoning_value(value: &Value, output: &mut String) {
    for key in [
        "thinking",
        "reasoning",
        "reasoning_content",
        "reasoningContent",
    ] {
        if let Some(text) = value.get(key).and_then(Value::as_str) {
            output.push_str(text);
        }
    }
    for key in ["summary", "content"] {
        for item in value
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(text) = item
                .get("text")
                .or_else(|| item.get("thinking"))
                .and_then(Value::as_str)
            {
                output.push_str(text);
            }
        }
    }
}

fn combined_thinking(primary: &str, secondary: Option<&str>) -> Option<String> {
    let mut values = Vec::new();
    if !primary.trim().is_empty() {
        values.push(primary.trim());
    }
    if let Some(secondary) = secondary.filter(|value| !value.trim().is_empty()) {
        values.push(secondary.trim());
    }
    (!values.is_empty()).then(|| values.join("\n"))
}

fn nonempty_command_text_with_thinking(
    content: String,
    explicit_thinking: Option<String>,
) -> Result<CommandProviderResponse, String> {
    let (content, tagged_thinking) = split_nonstream_thinking_tags(&content);
    if content.trim().is_empty() {
        Err("The AI provider returned neither a Command Mode tool call nor an answer".into())
    } else {
        Ok(CommandProviderResponse::Text {
            content,
            thinking: combined_thinking(
                explicit_thinking.as_deref().unwrap_or_default(),
                tagged_thinking.as_deref(),
            ),
        })
    }
}

pub async fn fetch_models(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
) -> Result<Vec<String>, String> {
    let endpoint = match profile.api_style {
        ProviderApiStyle::OpenAiCompatible => join_endpoint(&profile.base_url, "models"),
        ProviderApiStyle::Anthropic => anthropic_models_endpoint(&profile.base_url),
    };
    let mut request = reqwest::Client::new().get(endpoint);
    if let Some(api_key) = api_key.filter(|key| !key.trim().is_empty()) {
        request = match profile.api_style {
            ProviderApiStyle::OpenAiCompatible => request.bearer_auth(api_key),
            ProviderApiStyle::Anthropic => request
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01"),
        };
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("Could not fetch models from {}: {error}", profile.name))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("Could not read model list: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "{} returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }
    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|error| format!("{} returned an invalid model list: {error}", profile.name))?;
    let models = json
        .get("data")
        .and_then(serde_json::Value::as_array)
        .or_else(|| json.get("models").and_then(serde_json::Value::as_array))
        .ok_or_else(|| format!("{} returned an unsupported model-list format", profile.name))?
        .iter()
        .filter_map(|model| model.get("id").or_else(|| model.get("name")))
        .filter_map(serde_json::Value::as_str)
        .map(|name| name.trim_start_matches("models/").to_owned())
        .collect::<Vec<_>>();
    Ok(models)
}

fn anthropic_models_endpoint(base_url: &str) -> String {
    if base_url.contains("/v1/models") {
        base_url.to_owned()
    } else if base_url.contains("/v1/messages") {
        base_url.replacen("/v1/messages", "/v1/models", 1)
    } else if base_url.trim_end_matches('/').ends_with("/v1") {
        join_endpoint(base_url, "models")
    } else {
        join_endpoint(base_url, "v1/models")
    }
}

fn anthropic_messages_endpoint(base_url: &str) -> String {
    if base_url.contains("/v1/messages") {
        base_url.to_owned()
    } else if base_url.trim_end_matches('/').ends_with("/v1") {
        join_endpoint(base_url, "messages")
    } else {
        join_endpoint(base_url, "v1/messages")
    }
}

async fn process_openai_compatible(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    input: &str,
    timeout: Duration,
    temperature: f32,
    max_tokens: Option<u32>,
) -> Result<String, String> {
    let use_responses_api = uses_openai_responses_api(profile);
    let client = llm_client()?;
    let request = openai_request_builder(
        profile,
        api_key,
        client.post(openai_endpoint(profile, use_responses_api)),
    )?;
    let request = if use_responses_api {
        request.json(&openai_responses_request(
            profile,
            system_prompt,
            input,
            false,
            temperature,
            max_tokens,
        ))
    } else {
        request.json(&openai_chat_request(
            profile,
            system_prompt,
            input,
            false,
            temperature,
            max_tokens,
        ))
    };
    let response = send_llm_request(profile, request.timeout(timeout)).await?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("Could not read {} response: {error}", profile.name))?;
    if !status.is_success() {
        return Err(format!(
            "{} returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }
    if use_responses_api {
        let response: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
            format!("{} returned an unsupported response: {error}", profile.name)
        })?;
        responses_output_text(&response)
            .and_then(|text| nonempty_nonstream_text(text))
            .ok_or_else(|| format!("{} returned an empty response", profile.name))
    } else {
        let response: OpenAiResponse = serde_json::from_str(&body).map_err(|error| {
            format!("{} returned an unsupported response: {error}", profile.name)
        })?;
        response
            .choices
            .into_iter()
            .find_map(|choice| choice.message.content)
            .and_then(nonempty_nonstream_text)
            .ok_or_else(|| format!("{} returned an empty response", profile.name))
    }
}

#[allow(dead_code)]
async fn process_openai_compatible_streaming(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    input: &str,
    timeout: Duration,
    temperature: f32,
    max_tokens: Option<u32>,
    on_delta: &mut (dyn FnMut(&str) + Send),
) -> Result<String, String> {
    let use_responses_api = uses_openai_responses_api(profile);
    let client = llm_client()?;
    let request = openai_request_builder(
        profile,
        api_key,
        client.post(openai_endpoint(profile, use_responses_api)),
    )?;
    let request = if use_responses_api {
        request.json(&openai_responses_request(
            profile,
            system_prompt,
            input,
            true,
            temperature,
            max_tokens,
        ))
    } else {
        request.json(&openai_chat_request(
            profile,
            system_prompt,
            input,
            true,
            temperature,
            max_tokens,
        ))
    };
    consume_streaming_response(
        profile,
        send_llm_request(profile, request.timeout(timeout)).await,
        on_delta,
    )
    .await
}

fn openai_chat_request<'a>(
    profile: &'a AiProviderProfile,
    system_prompt: &'a str,
    input: &'a str,
    stream: bool,
    temperature: f32,
    max_tokens: Option<u32>,
) -> OpenAiRequest<'a> {
    OpenAiRequest {
        model: &profile.model,
        messages: vec![
            OpenAiMessage {
                role: "system",
                content: system_prompt,
            },
            OpenAiMessage {
                role: "user",
                content: input,
            },
        ],
        temperature: (!is_temperature_unsupported(&profile.model)).then_some(temperature),
        max_completion_tokens: is_reasoning_model(&profile.model)
            .then_some(max_tokens)
            .flatten(),
        stream,
        request_parameters: &profile.request_parameters,
    }
}

fn openai_responses_request<'a>(
    profile: &'a AiProviderProfile,
    system_prompt: &'a str,
    input: &'a str,
    stream: bool,
    temperature: f32,
    max_tokens: Option<u32>,
) -> OpenAiResponsesRequest<'a> {
    OpenAiResponsesRequest {
        model: &profile.model,
        input: vec![
            OpenAiMessage {
                role: "system",
                content: system_prompt,
            },
            OpenAiMessage {
                role: "user",
                content: input,
            },
        ],
        store: false,
        temperature: (!is_temperature_unsupported(&profile.model)).then_some(temperature),
        max_output_tokens: max_tokens,
        stream,
        request_parameters: responses_request_parameters(profile),
    }
}

fn openai_request_builder(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    mut request: reqwest::RequestBuilder,
) -> Result<reqwest::RequestBuilder, String> {
    if !is_local_endpoint(&profile.base_url) {
        let key = api_key
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| format!("An API key is required for {}", profile.name))?;
        request = request.bearer_auth(key);
    }
    Ok(request)
}

fn uses_openai_responses_api(profile: &AiProviderProfile) -> bool {
    if profile.base_url.contains("/responses") {
        return true;
    }
    let Ok(url) = reqwest::Url::parse(&profile.base_url) else {
        return false;
    };
    url.host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("api.openai.com"))
        && is_reasoning_model(&profile.model)
}

fn openai_endpoint(profile: &AiProviderProfile, use_responses_api: bool) -> String {
    if use_responses_api {
        if profile.base_url.contains("/responses") {
            profile.base_url.clone()
        } else if profile.base_url.contains("/chat/completions") {
            profile.base_url.replace("/chat/completions", "/responses")
        } else {
            join_endpoint(&profile.base_url, "responses")
        }
    } else if profile.base_url.contains("/chat/completions")
        || profile.base_url.contains("/api/chat")
        || profile.base_url.contains("/api/generate")
    {
        profile.base_url.clone()
    } else {
        join_endpoint(&profile.base_url, "chat/completions")
    }
}

fn responses_output_text(response: &serde_json::Value) -> Option<String> {
    response
        .get("output_text")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            response
                .get("output")
                .and_then(serde_json::Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter(|item| {
                            item.get("type").and_then(serde_json::Value::as_str) == Some("message")
                        })
                        .flat_map(|item| {
                            item.get("content")
                                .and_then(serde_json::Value::as_array)
                                .into_iter()
                                .flatten()
                        })
                        .filter(|content| {
                            content.get("type").and_then(serde_json::Value::as_str)
                                == Some("output_text")
                        })
                        .filter_map(|content| {
                            content.get("text").and_then(serde_json::Value::as_str)
                        })
                        .collect::<String>()
                })
                .filter(|text| !text.is_empty())
        })
}

async fn process_anthropic(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    input: &str,
    timeout: Duration,
    temperature: f32,
    max_tokens: Option<u32>,
) -> Result<String, String> {
    let key = api_key
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| format!("An API key is required for {}", profile.name))?;
    let body = AnthropicRequest {
        model: &profile.model,
        max_tokens: max_tokens.unwrap_or(4_096),
        system: system_prompt,
        messages: vec![OpenAiMessage {
            role: "user",
            content: input,
        }],
        stream: false,
        temperature: (!is_temperature_unsupported(&profile.model)).then_some(temperature),
        request_parameters: &profile.request_parameters,
    };
    let request = llm_client()?
        .post(anthropic_messages_endpoint(&profile.base_url))
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&body);
    let response = send_llm_request(profile, request.timeout(timeout)).await?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("Could not read {} response: {error}", profile.name))?;
    if !status.is_success() {
        return Err(format!(
            "{} returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }
    let response: AnthropicResponse = serde_json::from_str(&body)
        .map_err(|error| format!("{} returned an unsupported response: {error}", profile.name))?;
    response
        .content
        .into_iter()
        .find(|content| content.content_type == "text")
        .and_then(|content| content.text)
        .and_then(nonempty_nonstream_text)
        .ok_or_else(|| format!("{} returned an empty response", profile.name))
}

fn nonempty_nonstream_text(text: String) -> Option<String> {
    let text = strip_nonstream_thinking_tags(&text);
    (!text.trim().is_empty()).then_some(text)
}

fn strip_nonstream_thinking_tags(text: &str) -> String {
    split_nonstream_thinking_tags(text).0
}

fn split_nonstream_thinking_tags(text: &str) -> (String, Option<String>) {
    let mut result = String::new();
    let mut filter = StreamingThinkingFilter::default();
    let mut ignore_delta = |_delta: &str| {};
    filter.push(text, &mut result, &mut ignore_delta);
    filter.finish(&mut result, &mut ignore_delta);
    (result, filter.thinking())
}

#[allow(dead_code)]
async fn process_anthropic_streaming(
    profile: &AiProviderProfile,
    api_key: Option<&str>,
    system_prompt: &str,
    input: &str,
    timeout: Duration,
    temperature: f32,
    max_tokens: Option<u32>,
    on_delta: &mut (dyn FnMut(&str) + Send),
) -> Result<String, String> {
    let key = api_key
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| format!("An API key is required for {}", profile.name))?;
    let body = AnthropicRequest {
        model: &profile.model,
        max_tokens: max_tokens.unwrap_or(4_096),
        system: system_prompt,
        messages: vec![OpenAiMessage {
            role: "user",
            content: input,
        }],
        stream: true,
        temperature: (!is_temperature_unsupported(&profile.model)).then_some(temperature),
        request_parameters: &profile.request_parameters,
    };
    let request = llm_client()?
        .post(anthropic_messages_endpoint(&profile.base_url))
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&body);
    consume_streaming_response(
        profile,
        send_llm_request(profile, request.timeout(timeout)).await,
        on_delta,
    )
    .await
}

fn llm_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(LLM_REQUEST_TIMEOUT)
        .build()
        .map_err(|error| format!("Could not initialize the AI HTTP client: {error}"))
}

async fn send_llm_request(
    profile: &AiProviderProfile,
    request: reqwest::RequestBuilder,
) -> Result<reqwest::Response, String> {
    let mut last_error = None;
    for attempt in 0..LLM_MAX_NETWORK_ATTEMPTS {
        let Some(request) = request.try_clone() else {
            return Err(format!(
                "Could not prepare an AI request for {}",
                profile.name
            ));
        };
        match request.send().await {
            Ok(response) => return Ok(response),
            Err(error) => {
                let retryable = error.is_timeout() || error.is_connect();
                last_error = Some(error);
                if !retryable || attempt + 1 == LLM_MAX_NETWORK_ATTEMPTS {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(200 * (attempt as u64 + 1))).await;
            }
        }
    }
    Err(format!(
        "{} request failed: {}",
        profile.name,
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "unknown network error".into())
    ))
}

#[allow(dead_code)]
async fn consume_streaming_response(
    profile: &AiProviderProfile,
    response: Result<reqwest::Response, String>,
    on_delta: &mut (dyn FnMut(&str) + Send),
) -> Result<String, String> {
    let response = response?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "{} returned HTTP {status}: {}",
            profile.name,
            truncate(&body)
        ));
    }
    let mut result = String::new();
    let mut buffer = String::new();
    let mut thinking_filter = StreamingThinkingFilter::default();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("{} stream failed: {error}", profile.name))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(index) = buffer.find('\n') {
            let line = buffer.drain(..=index).collect::<String>();
            append_stream_delta(&line, &mut result, on_delta, &mut thinking_filter);
        }
    }
    if !buffer.trim().is_empty() {
        append_stream_delta(&buffer, &mut result, on_delta, &mut thinking_filter);
    }
    thinking_filter.finish(&mut result, on_delta);
    if result.trim().is_empty() {
        return Err(format!(
            "{} returned an empty streaming response",
            profile.name
        ));
    }
    Ok(result)
}

#[allow(dead_code)]
fn append_stream_delta(
    line: &str,
    result: &mut String,
    on_delta: &mut (dyn FnMut(&str) + Send),
    thinking_filter: &mut StreamingThinkingFilter,
) {
    let line = line.trim();
    let payload = line.strip_prefix("data:").unwrap_or(line).trim();
    if payload.is_empty() || payload == "[DONE]" {
        return;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return;
    };
    let delta = value
        .get("delta")
        .filter(|_| {
            value.get("type").and_then(serde_json::Value::as_str)
                == Some("response.output_text.delta")
        })
        .or_else(|| value.pointer("/choices/0/delta/content"))
        .or_else(|| value.pointer("/choices/0/message/content"))
        .or_else(|| value.pointer("/delta/text"))
        .or_else(|| value.pointer("/message/content"))
        .or_else(|| value.pointer("/content/0/text"))
        .or_else(|| value.get("output_text"))
        .or_else(|| value.pointer("/output/0/content/0/text"))
        .or_else(|| value.get("response"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !delta.is_empty() {
        thinking_filter.push(delta, result, on_delta);
    }
}

#[derive(Default)]
struct StreamingThinkingFilter {
    pending: String,
    in_thinking: bool,
    thinking: String,
}

impl StreamingThinkingFilter {
    const TAG_FRAGMENT_LENGTH: usize = 11;

    fn push(&mut self, delta: &str, result: &mut String, on_delta: &mut (dyn FnMut(&str) + Send)) {
        self.push_with_thinking(delta, result, on_delta, &mut |_| {});
    }

    fn push_with_thinking(
        &mut self,
        delta: &str,
        result: &mut String,
        on_delta: &mut (dyn FnMut(&str) + Send),
        on_thinking: &mut (dyn FnMut(&str) + Send),
    ) {
        self.pending.push_str(delta);
        loop {
            if self.in_thinking {
                if let Some((index, length)) = thinking_closing_tag(&self.pending) {
                    let thinking = self.pending[..index].to_owned();
                    self.thinking.push_str(&thinking);
                    on_thinking(&thinking);
                    self.pending.drain(..index + length);
                    self.in_thinking = false;
                    continue;
                }
                self.retain_only_possible_tag_fragment(on_thinking);
                return;
            }
            if let Some((index, length)) = thinking_opening_tag(&self.pending) {
                self.emit_prefix(index, result, on_delta);
                self.pending.drain(..length);
                self.in_thinking = true;
                continue;
            }
            self.emit_all_but_possible_tag_fragment(result, on_delta);
            return;
        }
    }

    fn finish(&mut self, result: &mut String, on_delta: &mut (dyn FnMut(&str) + Send)) {
        self.finish_with_thinking(result, on_delta, &mut |_| {});
    }

    fn finish_with_thinking(
        &mut self,
        result: &mut String,
        on_delta: &mut (dyn FnMut(&str) + Send),
        on_thinking: &mut (dyn FnMut(&str) + Send),
    ) {
        if !self.in_thinking && !self.pending.is_empty() {
            let text = std::mem::take(&mut self.pending);
            result.push_str(&text);
            on_delta(&text);
        } else if self.in_thinking && !self.pending.is_empty() {
            let thinking = std::mem::take(&mut self.pending);
            self.thinking.push_str(&thinking);
            on_thinking(&thinking);
        }
        self.pending.clear();
    }

    fn emit_prefix(
        &mut self,
        length: usize,
        result: &mut String,
        on_delta: &mut (dyn FnMut(&str) + Send),
    ) {
        if length == 0 {
            return;
        }
        let text = self.pending.drain(..length).collect::<String>();
        result.push_str(&text);
        on_delta(&text);
    }

    fn emit_all_but_possible_tag_fragment(
        &mut self,
        result: &mut String,
        on_delta: &mut (dyn FnMut(&str) + Send),
    ) {
        let length = prefix_before_trailing_characters(&self.pending, Self::TAG_FRAGMENT_LENGTH);
        self.emit_prefix(length, result, on_delta);
    }

    fn retain_only_possible_tag_fragment(&mut self, on_thinking: &mut (dyn FnMut(&str) + Send)) {
        let length = prefix_before_trailing_characters(&self.pending, Self::TAG_FRAGMENT_LENGTH);
        if length > 0 {
            let thinking = self.pending.drain(..length).collect::<String>();
            self.thinking.push_str(&thinking);
            on_thinking(&thinking);
        }
    }

    fn thinking(&self) -> Option<String> {
        (!self.thinking.trim().is_empty()).then(|| self.thinking.clone())
    }
}

fn thinking_opening_tag(text: &str) -> Option<(usize, usize)> {
    ["<thinking>", "<think>"]
        .into_iter()
        .filter_map(|tag| text.find(tag).map(|index| (index, tag.len())))
        .min_by_key(|(index, _)| *index)
}

fn thinking_closing_tag(text: &str) -> Option<(usize, usize)> {
    ["</thinking>", "</think>"]
        .into_iter()
        .filter_map(|tag| text.find(tag).map(|index| (index, tag.len())))
        .min_by_key(|(index, _)| *index)
}

fn prefix_before_trailing_characters(text: &str, trailing_characters: usize) -> usize {
    let character_count = text.chars().count();
    if character_count <= trailing_characters {
        return 0;
    }
    text.char_indices()
        .nth(character_count - trailing_characters)
        .map(|(index, _)| index)
        .unwrap_or_default()
}

fn join_endpoint(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn is_local_endpoint(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let host = host.to_ascii_lowercase();
    host == "localhost"
        || host == "::1"
        || host.starts_with("127.")
        || host.starts_with("10.")
        || host.starts_with("192.168.")
        || host
            .strip_prefix("172.")
            .and_then(|remainder| remainder.split('.').next())
            .and_then(|octet| octet.parse::<u8>().ok())
            .is_some_and(|octet| (16..=31).contains(&octet))
}

pub fn is_reasoning_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("gpt-5")
        || model.contains("gpt-oss")
        || model.starts_with("openai/")
        || (model.contains("deepseek") && model.contains("reasoner"))
}

pub fn is_temperature_unsupported(model: &str) -> bool {
    if is_reasoning_model(model) {
        return true;
    }
    let model = model.to_ascii_lowercase();
    let model = model.replace('.', "-");
    model.contains("claude-opus-4-7")
        || model.contains("claude-opus-4-8")
        || model.contains("claude-sonnet-5")
        || model.contains("claude-fable")
        || model.contains("claude-mythos")
}

fn truncate(text: &str) -> String {
    text.chars().take(800).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_openai_deltas_are_accumulated_and_emitted() {
        let mut result = String::new();
        let mut emitted = String::new();
        let mut on_delta = |delta: &str| emitted.push_str(delta);
        let mut thinking_filter = StreamingThinkingFilter::default();

        append_stream_delta(
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            &mut result,
            &mut on_delta,
            &mut thinking_filter,
        );
        append_stream_delta(
            r#"data: {"choices":[{"delta":{"content":" world"}}]}"#,
            &mut result,
            &mut on_delta,
            &mut thinking_filter,
        );
        append_stream_delta(
            "data: [DONE]",
            &mut result,
            &mut on_delta,
            &mut thinking_filter,
        );
        thinking_filter.finish(&mut result, &mut on_delta);

        assert_eq!(result, "Hello world");
        assert_eq!(emitted, "Hello world");
    }

    #[test]
    fn streaming_anthropic_and_ollama_deltas_are_supported() {
        let mut result = String::new();
        let mut emitted = String::new();
        let mut on_delta = |delta: &str| emitted.push_str(delta);
        let mut thinking_filter = StreamingThinkingFilter::default();

        append_stream_delta(
            r#"data: {"type":"content_block_delta","delta":{"text":"First"}}"#,
            &mut result,
            &mut on_delta,
            &mut thinking_filter,
        );
        append_stream_delta(
            r#"{"response":" second"}"#,
            &mut result,
            &mut on_delta,
            &mut thinking_filter,
        );
        thinking_filter.finish(&mut result, &mut on_delta);

        assert_eq!(result, "First second");
        assert_eq!(emitted, "First second");
    }

    #[test]
    fn streaming_openai_responses_deltas_are_accumulated_and_emitted() {
        let mut result = String::new();
        let mut emitted = String::new();
        let mut on_delta = |delta: &str| emitted.push_str(delta);
        let mut thinking_filter = StreamingThinkingFilter::default();

        append_stream_delta(
            r#"data: {"type":"response.output_text.delta","delta":"Responses"}"#,
            &mut result,
            &mut on_delta,
            &mut thinking_filter,
        );
        append_stream_delta(
            r#"data: {"type":"response.output_text.delta","delta":" API"}"#,
            &mut result,
            &mut on_delta,
            &mut thinking_filter,
        );
        thinking_filter.finish(&mut result, &mut on_delta);

        assert_eq!(result, "Responses API");
        assert_eq!(emitted, "Responses API");
    }

    #[test]
    fn streaming_thinking_tags_are_filtered_when_split_between_chunks() {
        let mut result = String::new();
        let mut emitted = String::new();
        let mut on_delta = |delta: &str| emitted.push_str(delta);
        let mut thinking_filter = StreamingThinkingFilter::default();

        thinking_filter.push("Draft <thi", &mut result, &mut on_delta);
        thinking_filter.push("nk>private reasoning</th", &mut result, &mut on_delta);
        thinking_filter.push("ink> final", &mut result, &mut on_delta);
        thinking_filter.finish(&mut result, &mut on_delta);

        assert_eq!(result, "Draft  final");
        assert_eq!(emitted, "Draft  final");
    }

    #[test]
    fn parses_openai_command_tool_calls_and_preserves_the_call_id() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "I will inspect the repository first.",
                    "tool_calls": [{
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "execute_terminal_command",
                            "arguments": "{\"command\":\"git status --short\",\"purpose\":\"Check the repository state\"}"
                        }
                    }]
                }
            }]
        });

        let CommandProviderResponse::ToolCall {
            content, tool_call, ..
        } = parse_openai_command_response(&response, false).expect("tool call should parse")
        else {
            panic!("expected a tool call");
        };
        assert_eq!(content, "I will inspect the repository first.");
        assert_eq!(tool_call.id, "call_status");
        assert_eq!(tool_call.name, "execute_terminal_command");
        assert_eq!(tool_call.arguments["command"], "git status --short");
    }

    #[test]
    fn streamed_openai_command_tool_arguments_are_accumulated_before_review() {
        let mut state = CommandStreamState::default();
        let mut emitted = String::new();
        let mut emitted_thinking = String::new();
        let mut on_delta = |delta: &str| emitted.push_str(delta);
        let mut on_thinking = |delta: &str| emitted_thinking.push_str(delta);

        append_command_stream_event(
            r#"data: {"choices":[{"delta":{"content":"I will check first. ","tool_calls":[{"index":0,"id":"call_pwd","function":{"name":"execute_terminal_command","arguments":"{\"command\":\"pwd\","}}]}}]}"#,
            CommandStreamProtocol::OpenAiChat,
            &mut state,
            &mut on_delta,
            &mut on_thinking,
        );
        append_command_stream_event(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"purpose\":\"Check the current directory\"}"}}]}}]}"#,
            CommandStreamProtocol::OpenAiChat,
            &mut state,
            &mut on_delta,
            &mut on_thinking,
        );
        state
            .thinking_filter
            .finish(&mut state.content, &mut on_delta);

        let tool = state.tool_calls.remove(&0).expect("streamed tool call");
        assert_eq!(emitted, "I will check first. ");
        assert!(emitted_thinking.is_empty());
        assert_eq!(state.content, "I will check first. ");
        assert_eq!(tool.id.as_deref(), Some("call_pwd"));
        assert_eq!(tool.name.as_deref(), Some("execute_terminal_command"));
        assert_eq!(
            serde_json::from_str::<Value>(&tool.arguments).expect("complete streamed arguments")
                ["purpose"],
            "Check the current directory"
        );
    }

    #[test]
    fn streamed_responses_and_anthropic_tool_calls_keep_their_provider_call_ids() {
        let mut responses = CommandStreamState::default();
        let mut anthropic = CommandStreamState::default();
        let mut ignored = |_delta: &str| {};
        let mut ignored_thinking = |_delta: &str| {};

        append_command_stream_event(
            r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_internal","call_id":"call_response","name":"execute_terminal_command","arguments":""}}"#,
            CommandStreamProtocol::OpenAiResponses,
            &mut responses,
            &mut ignored,
            &mut ignored_thinking,
        );
        append_command_stream_event(
            r#"data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"command\":\"git status\"}"}"#,
            CommandStreamProtocol::OpenAiResponses,
            &mut responses,
            &mut ignored,
            &mut ignored_thinking,
        );
        append_command_stream_event(
            r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_anthropic","name":"execute_terminal_command","input":{}}}"#,
            CommandStreamProtocol::Anthropic,
            &mut anthropic,
            &mut ignored,
            &mut ignored_thinking,
        );
        append_command_stream_event(
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"ls\"}"}}"#,
            CommandStreamProtocol::Anthropic,
            &mut anthropic,
            &mut ignored,
            &mut ignored_thinking,
        );

        let response_tool = responses
            .tool_calls
            .remove(&1)
            .expect("Responses tool call");
        let anthropic_tool = anthropic
            .tool_calls
            .remove(&2)
            .expect("Anthropic tool call");
        assert_eq!(response_tool.id.as_deref(), Some("call_response"));
        assert_eq!(
            serde_json::from_str::<Value>(&response_tool.arguments).expect("Responses arguments")
                ["command"],
            "git status"
        );
        assert_eq!(anthropic_tool.id.as_deref(), Some("toolu_anthropic"));
        assert_eq!(
            serde_json::from_str::<Value>(&anthropic_tool.arguments).expect("Anthropic arguments")
                ["command"],
            "ls"
        );
    }

    #[test]
    fn streamed_command_reasoning_is_emitted_separately_from_visible_content() {
        let mut state = CommandStreamState::default();
        let mut content = String::new();
        let mut thinking = String::new();
        let mut on_content = |delta: &str| content.push_str(delta);
        let mut on_thinking = |delta: &str| thinking.push_str(delta);

        append_command_stream_event(
            r#"data: {"choices":[{"delta":{"reasoning_content":"Inspecting safely. ","content":"I will check first. "}}]}"#,
            CommandStreamProtocol::OpenAiChat,
            &mut state,
            &mut on_content,
            &mut on_thinking,
        );
        append_command_stream_event(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Then I will report back."}}"#,
            CommandStreamProtocol::Anthropic,
            &mut state,
            &mut on_content,
            &mut on_thinking,
        );
        state.thinking_filter.finish_with_thinking(
            &mut state.content,
            &mut on_content,
            &mut on_thinking,
        );

        assert_eq!(content, "I will check first. ");
        assert_eq!(state.content, "I will check first. ");
        assert_eq!(thinking, "Inspecting safely. Then I will report back.");
        assert_eq!(state.thinking, thinking);
    }

    #[test]
    fn builds_responses_input_with_tool_call_and_result_associations() {
        let profile = AiProviderProfile::openai(
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            "gpt-5-mini",
        );
        let call = CommandToolCall {
            id: "call_check".into(),
            name: "execute_terminal_command".into(),
            arguments: json!({
                "command": "pwd",
                "purpose": "Check the working directory",
            }),
        };
        let body = openai_responses_command_request(
            &profile,
            "system",
            &[
                CommandProviderMessage::User("Where am I?".into()),
                CommandProviderMessage::Assistant {
                    content: "Checking now.".into(),
                    tool_call: Some(call),
                },
                CommandProviderMessage::Tool {
                    content: "{\"success\":true,\"output\":\"/tmp\"}".into(),
                    tool_call_id: "call_check".into(),
                },
            ],
        )
        .expect("request should build");

        assert_eq!(body["input"][3]["type"], "function_call");
        assert_eq!(body["input"][3]["call_id"], "call_check");
        assert_eq!(body["input"][4]["type"], "function_call_output");
        assert_eq!(body["input"][4]["call_id"], "call_check");
        assert_eq!(body["tools"][0]["name"], "execute_terminal_command");
    }

    #[test]
    fn anthropic_tool_use_is_parsed_and_tool_results_keep_the_call_association() {
        let response = json!({
            "content": [
                {"type": "text", "text": "I will check first."},
                {
                    "type": "tool_use",
                    "id": "toolu_status",
                    "name": "execute_terminal_command",
                    "input": {
                        "command": "git status --short",
                        "purpose": "Inspect the repository state"
                    }
                }
            ]
        });
        let CommandProviderResponse::ToolCall {
            content, tool_call, ..
        } = parse_anthropic_command_response(&response).expect("tool use should parse")
        else {
            panic!("expected a tool call");
        };
        assert_eq!(content, "I will check first.");
        assert_eq!(tool_call.id, "toolu_status");
        assert_eq!(tool_call.arguments["command"], "git status --short");

        let profile = AiProviderProfile {
            id: "anthropic".into(),
            name: "Anthropic".into(),
            api_style: ProviderApiStyle::Anthropic,
            base_url: "https://api.anthropic.com".into(),
            model: "claude-sonnet-4-20250514".into(),
            enabled: true,
            request_parameters: BTreeMap::new(),
        };
        let body = anthropic_command_request(
            &profile,
            "system",
            &[
                CommandProviderMessage::Assistant {
                    content: "Checking now.".into(),
                    tool_call: Some(tool_call),
                },
                CommandProviderMessage::Tool {
                    content: "{\"success\":true}".into(),
                    tool_call_id: "toolu_status".into(),
                },
            ],
        )
        .expect("request should build");
        assert_eq!(body["messages"][0]["content"][1]["type"], "tool_use");
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_result");
        assert_eq!(
            body["messages"][1]["content"][0]["tool_use_id"],
            "toolu_status"
        );
    }

    #[test]
    fn nonstream_responses_strip_thinking_blocks_before_returning_content() {
        assert_eq!(
            nonempty_nonstream_text("<think>private reasoning</think>Finished answer".into())
                .as_deref(),
            Some("Finished answer")
        );
        assert_eq!(
            nonempty_nonstream_text("<thinking>private reasoning</thinking>".into()),
            None
        );
        assert_eq!(
            strip_nonstream_thinking_tags("Answer <think>hidden</think> after"),
            "Answer  after"
        );
    }

    #[test]
    fn command_responses_keep_display_only_thinking_separate_from_content() {
        let response = json!({
            "choices": [{"message": {"content": "<think>inspect first</think>Everything is ready."}}]
        });
        let CommandProviderResponse::Text { content, thinking } =
            parse_openai_command_response(&response, false).expect("answer should parse")
        else {
            panic!("expected a text response");
        };
        assert_eq!(content, "Everything is ready.");
        assert_eq!(thinking.as_deref(), Some("inspect first"));
    }

    #[test]
    fn selects_responses_api_for_openai_reasoning_models() {
        let reasoning = AiProviderProfile::openai(
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            "gpt-5-mini",
        );
        let standard =
            AiProviderProfile::openai("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4.1");
        let explicit = AiProviderProfile::openai(
            "custom",
            "Custom",
            "https://example.invalid/v1/responses",
            "any-model",
        );

        assert!(uses_openai_responses_api(&reasoning));
        assert_eq!(
            openai_endpoint(&reasoning, true),
            "https://api.openai.com/v1/responses"
        );
        assert!(!uses_openai_responses_api(&standard));
        assert!(uses_openai_responses_api(&explicit));
    }

    #[test]
    fn reads_text_from_openai_responses_payloads_and_applies_request_parameters() {
        let response: serde_json::Value = serde_json::json!({
            "output": [{
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello "},
                    {"type": "output_text", "text": "world"}
                ]
            }]
        });
        assert_eq!(
            responses_output_text(&response).as_deref(),
            Some("Hello world")
        );

        let mut profile = AiProviderProfile::openai(
            "groq",
            "Groq",
            "https://api.groq.com/openai/v1",
            "openai/gpt-oss-120b",
        );
        profile
            .request_parameters
            .insert("reasoning_effort".into(), json!("low"));
        let body = serde_json::to_value(openai_chat_request(
            &profile,
            "system",
            "input",
            true,
            0.2,
            Some(32_000),
        ))
        .expect("request serializes");
        assert_eq!(body["reasoning_effort"], "low");
        let response_body = serde_json::to_value(openai_responses_request(
            &profile,
            "system",
            "input",
            true,
            0.2,
            Some(32_000),
        ))
        .expect("response request serializes");
        assert_eq!(response_body["reasoning"]["effort"], "low");
        assert!(response_body.get("reasoning_effort").is_none());
        assert_eq!(body["max_completion_tokens"], 32_000);
        assert_eq!(response_body["max_output_tokens"], 32_000);
        assert!(is_reasoning_model(&profile.model));
    }

    #[test]
    fn recognizes_every_private_172_network_as_local() {
        assert!(is_local_endpoint("http://172.16.0.2/v1"));
        assert!(is_local_endpoint("http://172.31.255.254/v1"));
        assert!(!is_local_endpoint("http://172.15.0.2/v1"));
        assert!(!is_local_endpoint("http://172.32.0.2/v1"));
    }

    #[test]
    fn builds_anthropic_models_endpoints_from_root_or_v1_base_urls() {
        assert_eq!(
            anthropic_models_endpoint("https://api.anthropic.com"),
            "https://api.anthropic.com/v1/models"
        );
        assert_eq!(
            anthropic_models_endpoint("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/models"
        );
        assert_eq!(
            anthropic_models_endpoint("https://api.anthropic.com/v1/models"),
            "https://api.anthropic.com/v1/models"
        );
        assert_eq!(
            anthropic_models_endpoint("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com/v1/models"
        );
        assert_eq!(
            anthropic_messages_endpoint("https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            anthropic_messages_endpoint("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            anthropic_messages_endpoint("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn built_in_provider_setup_pages_match_the_source_catalog() {
        assert_eq!(
            provider_website("openai"),
            Some(("https://platform.openai.com/api-keys", "Get API key"))
        );
        assert_eq!(
            provider_website("ollama"),
            Some((
                "https://docs.ollama.com/api/openai-compatibility",
                "Setup guide"
            ))
        );
        assert_eq!(provider_website("custom-provider"), None);
    }
}
