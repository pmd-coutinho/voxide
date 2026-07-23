//! Acoustic pronunciation-enrollment sidecar integration.
//!
//! FluidVoice matches enrolled words on the ASR encoder's hidden states, which
//! sherpa-onnx's high-level recognizer never exposes. Rather than run a second
//! onnxruntime in-process alongside sherpa, Voxide owns a small local
//! JSON-lines process that runs the SHIPPED Parakeet `encoder.int8.onnx`
//! directly (onnxruntime + numpy/scipy — no torch/NeMo) to produce per-word
//! acoustic embeddings and match them. Audio never leaves the machine: only
//! 16 kHz float PCM is written to the child's stdin. The sidecar is opt-in and
//! spawned only while pronunciation matching is in use.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    time::timeout,
};

use crate::debug_log;

pub const PROTOCOL_VERSION: u32 = 1;
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const RUNTIME_MARKER: &str = ".voxide-pronunciation-runtime-v1";
/// The sidecar reads the encoder from the installed Parakeet model directory.
const ENCODER_FILE: &str = "encoder.int8.onnx";

/// Pronunciation enrollment reuses the Parakeet encoder, so it shares
/// Parakeet's CUDA/Linux gating.
pub fn is_compiled() -> bool {
    cfg!(all(feature = "cuda", target_os = "linux"))
}

pub fn runtime_directory(data_directory: &Path) -> PathBuf {
    data_directory.join("pronunciation-runtime")
}

pub fn python_path(runtime_directory: &Path) -> PathBuf {
    runtime_directory.join("venv").join("bin").join("python")
}

pub fn runtime_is_installed(runtime_directory: &Path) -> bool {
    python_path(runtime_directory).is_file() && runtime_directory.join(RUNTIME_MARKER).is_file()
}

/// A stored enrollment (or their averaged prototype) sent to the matcher.
#[derive(Debug, Clone, Serialize)]
pub struct Prototype {
    pub label: String,
    pub values: Vec<f32>,
    pub frames: u32,
}

/// One embedding for an enrolled word plus the encoder-frame span it covered.
#[derive(Debug, Clone)]
pub struct Enrollment {
    pub embedding: Vec<f32>,
    pub hidden_size: usize,
    pub frames: u32,
}

/// The best acoustic match for one prototype against a decoded utterance.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PronunciationMatch {
    pub label: String,
    pub start_time: f32,
    pub end_time: f32,
    pub score: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Response {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    protocol_version: Option<u32>,
    #[serde(default)]
    request_id: Option<u64>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    frames: Option<u32>,
    #[serde(default)]
    matches: Option<Vec<PronunciationMatch>>,
}

/// One persistent Python process. Requests are serialized: enrollment and
/// finalize matching are both single-capture workflows.
pub struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    next_request_id: u64,
}

impl Server {
    pub async fn launch(
        python: &Path,
        script: &Path,
        model_directory: &Path,
    ) -> Result<Self, String> {
        if !python.is_file() {
            return Err("Pronunciation support is not installed. Install it from the Custom Dictionary screen first.".into());
        }
        if !script.is_file() {
            return Err(
                "The bundled pronunciation service is missing. Reinstall or update Voxide.".into(),
            );
        }
        if !model_directory.join(ENCODER_FILE).is_file() {
            return Err(
                "The Parakeet model is not installed. Download it from Voice Engine first.".into(),
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
            .map_err(|error| format!("Could not start the pronunciation service: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or("Could not open the pronunciation service input")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("Could not open the pronunciation service output")?;
        if let Some(stderr) = child.stderr.take() {
            tauri::async_runtime::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    debug_log::append(&format!("Pronunciation service: {line}"));
                }
            });
        }
        let mut server = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
            next_request_id: 1,
        };
        let response = server
            .request_with_timeout(
                json!({ "action": "ping", "protocolVersion": PROTOCOL_VERSION }),
                STARTUP_TIMEOUT,
            )
            .await?;
        validate_handshake(&response)?;
        Ok(server)
    }

    /// Extract one acoustic embedding for a spoken word.
    pub async fn enroll(&mut self, samples: &[f32]) -> Result<Enrollment, String> {
        let response = self
            .request(json!({ "action": "enroll", "audio": encode_pcm(samples) }))
            .await?;
        if response.kind != "enrolled" {
            return Err(unexpected_response("enroll", response));
        }
        let embedding = response
            .embedding
            .filter(|values| !values.is_empty())
            .ok_or("The pronunciation service returned an empty embedding")?;
        let hidden_size = response.hidden_size.unwrap_or(embedding.len());
        Ok(Enrollment {
            hidden_size,
            frames: response.frames.unwrap_or(0),
            embedding,
        })
    }

    /// Match every prototype against one decoded utterance's audio.
    pub async fn match_prototypes(
        &mut self,
        samples: &[f32],
        prototypes: &[Prototype],
    ) -> Result<Vec<PronunciationMatch>, String> {
        let response = self
            .request(json!({
                "action": "match",
                "audio": encode_pcm(samples),
                "prototypes": prototypes,
            }))
            .await?;
        if response.kind != "matched" {
            return Err(unexpected_response("match", response));
        }
        Ok(response.matches.unwrap_or_default())
    }

    async fn request(&mut self, request: Value) -> Result<Response, String> {
        self.request_with_timeout(request, REQUEST_TIMEOUT).await
    }

    async fn request_with_timeout(
        &mut self,
        mut request: Value,
        request_timeout: Duration,
    ) -> Result<Response, String> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        request
            .as_object_mut()
            .ok_or("Pronunciation request must be a JSON object")?
            .insert("requestId".into(), json!(request_id));
        let encoded = serde_json::to_string(&request)
            .map_err(|error| format!("Could not encode a pronunciation request: {error}"))?;
        if encoded.len() > MAX_MESSAGE_BYTES {
            return Err("Pronunciation audio request exceeds the service safety limit".into());
        }
        self.stdin
            .write_all(encoded.as_bytes())
            .await
            .map_err(|error| {
                format!("Could not send audio to the pronunciation service: {error}")
            })?;
        self.stdin.write_all(b"\n").await.map_err(|error| {
            format!("Could not send audio to the pronunciation service: {error}")
        })?;
        self.stdin.flush().await.map_err(|error| {
            format!("Could not send audio to the pronunciation service: {error}")
        })?;
        let line = match timeout(request_timeout, self.stdout.next_line()).await {
            Ok(result) => result,
            Err(_) => {
                self.terminate();
                return Err("Pronunciation service timed out and was stopped".into());
            }
        }
        .map_err(|error| format!("Could not read the pronunciation service response: {error}"))?
        .ok_or("Pronunciation service stopped unexpectedly")?;
        if line.len() > MAX_MESSAGE_BYTES {
            return Err("Pronunciation service returned an oversized response".into());
        }
        let response: Response = serde_json::from_str(&line)
            .map_err(|error| format!("Pronunciation service returned invalid JSON: {error}"))?;
        validate_response_request_id(&response, request_id)?;
        if response.kind == "error" {
            return Err(if response.message.trim().is_empty() {
                "Pronunciation service reported an unknown error".into()
            } else {
                response.message
            });
        }
        Ok(response)
    }

    pub fn terminate(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn validate_handshake(response: &Response) -> Result<(), String> {
    if response.kind != "ready" {
        return Err("The pronunciation service did not complete startup.".into());
    }
    if response.protocol_version != Some(PROTOCOL_VERSION) {
        return Err("The pronunciation service is incompatible with this Voxide version. Reinstall pronunciation support.".into());
    }
    Ok(())
}

fn validate_response_request_id(response: &Response, request_id: u64) -> Result<(), String> {
    if response.request_id != Some(request_id) {
        return Err("Pronunciation service returned a response for a different request".into());
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
            "Pronunciation service returned '{}' while handling {action}",
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
    fn handshake_requires_the_exact_sidecar_protocol_version() {
        let mut ready = Response {
            kind: "ready".into(),
            message: String::new(),
            protocol_version: Some(PROTOCOL_VERSION),
            request_id: Some(1),
            embedding: None,
            hidden_size: None,
            frames: None,
            matches: None,
        };
        assert!(validate_handshake(&ready).is_ok());
        ready.protocol_version = Some(PROTOCOL_VERSION + 1);
        assert!(validate_handshake(&ready).is_err());
    }

    #[test]
    fn response_must_belong_to_the_active_request() {
        let response = Response {
            kind: "matched".into(),
            message: String::new(),
            protocol_version: None,
            request_id: Some(7),
            embedding: None,
            hidden_size: None,
            frames: None,
            matches: Some(Vec::new()),
        };
        assert!(validate_response_request_id(&response, 7).is_ok());
        assert!(validate_response_request_id(&response, 8).is_err());
    }

    #[test]
    fn prototype_serializes_to_the_sidecar_schema() {
        let value = serde_json::to_value(Prototype {
            label: "Kubernetes".into(),
            values: vec![0.1, 0.2],
            frames: 10,
        })
        .expect("serialize prototype");
        assert_eq!(value["label"], "Kubernetes");
        assert_eq!(value["frames"], 10);
        assert_eq!(value["values"].as_array().map(Vec::len), Some(2));
        assert!((value["values"][1].as_f64().unwrap() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn match_response_parses_camel_case_spans() {
        let response: Response = serde_json::from_str(
            r#"{"type":"matched","requestId":3,"matches":[{"label":"Kubernetes","startTime":1.76,"endTime":2.56,"score":0.51}]}"#,
        )
        .expect("parse matched response");
        let matches = response.matches.expect("matches present");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].label, "Kubernetes");
        assert!((matches[0].score - 0.51).abs() < 1e-6);
    }
}
