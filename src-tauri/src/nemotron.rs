//! NVIDIA Nemotron 3.5 ASR CUDA sidecar integration.
//!
//! The official model is implemented in current Hugging Face Transformers and
//! uses PyTorch CUDA.  Rather than reproduce its cache-aware FastConformer and
//! RNN-T decoder in Rust, Voxide owns a small local JSON-lines process that
//! keeps the CUDA model warm across dictations.  Audio never leaves the
//! machine: only 16 kHz float PCM is written to the child process's stdin.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    time::timeout,
};

use crate::debug_log;

pub const MODEL_ID: &str = "nemotron-3.5-asr-streaming-0.6b";
pub const MODEL_REPOSITORY: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";
/// Immutable Hugging Face commit verified on 2026-07-22. Never download model
/// artifacts from a mutable branch: component receipts and reproduction depend
/// on every file resolving to this exact revision.
pub const MODEL_REVISION: &str = "f3d333391852ba876df169dcc9ba902d25b6ab0b";
/// NVIDIA's documented balanced streaming configuration (560 ms chunks).
pub const DEFAULT_LOOKAHEAD_TOKENS: u8 = 6;
pub const PROTOCOL_VERSION: u32 = 1;
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const RUNTIME_MARKER: &str = ".voxide-nemotron-runtime-v1";
const REQUIRED_MODEL_FILES: [&str; 5] = [
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// Nemotron is deliberately a CUDA/Linux engine.  Python could technically
/// fall back to CPU, but a 0.6B real-time recognizer would provide a poor and
/// surprising desktop experience there.
pub fn is_compiled() -> bool {
    cfg!(all(feature = "cuda", target_os = "linux"))
}

pub fn model_directory(models_directory: &Path) -> PathBuf {
    models_directory.join(MODEL_ID)
}

pub fn runtime_directory(data_directory: &Path) -> PathBuf {
    data_directory.join("nemotron-runtime")
}

pub fn python_path(runtime_directory: &Path) -> PathBuf {
    runtime_directory.join("venv").join("bin").join("python")
}

pub fn model_is_installed(directory: &Path) -> bool {
    directory.is_dir()
        && REQUIRED_MODEL_FILES
            .iter()
            .all(|name| directory.join(name).is_file())
}

pub fn runtime_is_installed(runtime_directory: &Path) -> bool {
    python_path(runtime_directory).is_file() && runtime_directory.join(RUNTIME_MARKER).is_file()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Response {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    protocol_version: Option<u32>,
}

/// One persistent Python process. Requests are intentionally serialized: one
/// RNN-T stream owns the model at a time, and dictation itself is also a
/// single-capture workflow.
pub struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

impl Server {
    pub async fn launch(
        python: &Path,
        script: &Path,
        model_directory: &Path,
    ) -> Result<Self, String> {
        if !python.is_file() {
            return Err("Nemotron CUDA runtime is not installed. Use Voice Engine → Install CUDA runtime first.".into());
        }
        if !script.is_file() {
            return Err(
                "The bundled Nemotron CUDA service is missing. Reinstall or update Voxide.".into(),
            );
        }
        if !model_is_installed(model_directory) {
            return Err(
                "The Nemotron model is not installed. Download it from Voice Engine first.".into(),
            );
        }
        let mut child = Command::new(python)
            .arg(script)
            .arg("--model-dir")
            .arg(model_directory)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("Could not start the Nemotron CUDA service: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or("Could not open the Nemotron service input")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("Could not open the Nemotron service output")?;
        if let Some(stderr) = child.stderr.take() {
            tauri::async_runtime::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    debug_log::append(&format!("Nemotron CUDA: {line}"));
                }
            });
        }
        let mut server = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        };
        let response = server
            .request(json!({ "action": "ping", "protocolVersion": PROTOCOL_VERSION }))
            .await?;
        validate_handshake(&response)?;
        Ok(server)
    }

    pub async fn start(&mut self, language: &str, lookahead_tokens: u8) -> Result<(), String> {
        let response = self
            .request(json!({
                "action": "start",
                "language": language,
                "lookaheadTokens": lookahead_tokens,
            }))
            .await?;
        if response.kind == "started" {
            return Ok(());
        }
        Err(unexpected_response("start", response))
    }

    pub async fn append(&mut self, samples: &[f32]) -> Result<String, String> {
        let response = self
            .request(json!({
                "action": "append",
                "audio": encode_pcm(samples),
            }))
            .await?;
        if response.kind == "partial" {
            return Ok(response.text);
        }
        Err(unexpected_response("append", response))
    }

    pub async fn finish(&mut self, samples: &[f32]) -> Result<String, String> {
        let response = self
            .request(json!({
                "action": "finish",
                "audio": encode_pcm(samples),
            }))
            .await?;
        if response.kind == "final" {
            return Ok(response.text);
        }
        Err(unexpected_response("finish", response))
    }

    pub async fn abort(&mut self) -> Result<(), String> {
        let response = self.request(json!({ "action": "abort" })).await?;
        if response.kind == "aborted" {
            Ok(())
        } else {
            Err(unexpected_response("abort", response))
        }
    }

    async fn request(&mut self, request: Value) -> Result<Response, String> {
        let encoded = serde_json::to_string(&request)
            .map_err(|error| format!("Could not encode a Nemotron request: {error}"))?;
        if encoded.len() > MAX_MESSAGE_BYTES {
            return Err("Nemotron audio request exceeds the service safety limit".into());
        }
        self.stdin
            .write_all(encoded.as_bytes())
            .await
            .map_err(|error| {
                format!("Could not send audio to the Nemotron CUDA service: {error}")
            })?;
        self.stdin.write_all(b"\n").await.map_err(|error| {
            format!("Could not send audio to the Nemotron CUDA service: {error}")
        })?;
        self.stdin.flush().await.map_err(|error| {
            format!("Could not send audio to the Nemotron CUDA service: {error}")
        })?;
        let line = timeout(Duration::from_secs(180), self.stdout.next_line())
            .await
            .map_err(|_| "Nemotron CUDA service timed out".to_string())?
            .map_err(|error| format!("Could not read the Nemotron CUDA service response: {error}"))?
            .ok_or("Nemotron CUDA service stopped unexpectedly")?;
        if line.len() > MAX_MESSAGE_BYTES {
            return Err("Nemotron CUDA service returned an oversized response".into());
        }
        let response: Response = serde_json::from_str(&line)
            .map_err(|error| format!("Nemotron CUDA service returned invalid JSON: {error}"))?;
        if response.kind == "error" {
            return Err(if response.message.trim().is_empty() {
                "Nemotron CUDA service reported an unknown error".into()
            } else {
                response.message
            });
        }
        Ok(response)
    }

    /// Ends the Python child if a recording is cancelled while a generation
    /// request is still blocked waiting for the next audio chunk.
    pub fn terminate(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn validate_handshake(response: &Response) -> Result<(), String> {
    if response.kind != "ready" {
        return Err("The Nemotron CUDA service did not complete startup.".into());
    }
    if response.protocol_version != Some(PROTOCOL_VERSION) {
        return Err("The Nemotron CUDA service is incompatible with this Voxide version. Reinstall the CUDA runtime.".into());
    }
    Ok(())
}

impl Drop for Server {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn unexpected_response(action: &str, response: Response) -> String {
    if response.message.trim().is_empty() {
        format!(
            "Nemotron CUDA service returned '{}' while handling {action}",
            response.kind
        )
    } else {
        response.message
    }
}

fn encode_pcm(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len().saturating_mul(std::mem::size_of::<f32>()));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    BASE64.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_payload_is_little_endian_float32_base64() {
        let decoded = BASE64.decode(encode_pcm(&[0.5, -1.0])).expect("base64 PCM");
        assert_eq!(decoded, [0, 0, 0, 63, 0, 0, 128, 191]);
    }

    #[test]
    fn installation_requires_the_real_transformers_files() {
        let temporary =
            std::env::temp_dir().join(format!("voxide-nemotron-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temporary).expect("create temporary model directory");
        assert!(!model_is_installed(&temporary));
        for file in REQUIRED_MODEL_FILES {
            std::fs::write(temporary.join(file), b"test").expect("write model marker");
        }
        assert!(model_is_installed(&temporary));
        std::fs::remove_dir_all(temporary).expect("remove temporary model directory");
    }

    #[test]
    fn handshake_requires_the_exact_sidecar_protocol_version() {
        let ready = Response {
            kind: "ready".into(),
            text: String::new(),
            message: String::new(),
            protocol_version: Some(PROTOCOL_VERSION),
        };
        assert!(validate_handshake(&ready).is_ok());
        assert!(validate_handshake(&Response {
            protocol_version: Some(PROTOCOL_VERSION + 1),
            ..ready
        })
        .is_err());
    }
}
