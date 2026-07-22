//! Shared ASR engine contract.
//!
//! The desktop command layer deliberately knows only about a selected engine
//! and its capabilities. Engine-specific setup, preview, and final-decode
//! adapters live behind that selection in `lib.rs` while they are migrated out
//! of the legacy command module incrementally. Keeping the capability contract
//! here makes the UI and lifecycle use the same source of truth today.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum EngineMaturity {
    Stable,
    Experimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum PreviewMode {
    FullSnapshot,
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum FinalMode {
    IndependentFullDecode,
    FlushActiveStream,
}

/// Static, user-relevant behavior of an ASR provider. Runtime readiness is
/// intentionally separate: a CUDA build can expose Parakeet while its model is
/// still absent, and installing a model must never change the selected engine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EngineCapabilities {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub maturity: EngineMaturity,
    pub preview_mode: PreviewMode,
    pub final_mode: FinalMode,
    pub supports_files: bool,
    pub supports_translation: bool,
    pub supports_vocabulary: bool,
    pub requires_cuda: bool,
}

/// Minimal common surface implemented by every selectable engine. Stateful
/// lifecycle operations are added through the adapter methods on `VoiceEngine`
/// during the non-big-bang migration, but capability consumers never need a
/// second engine switch.
pub(crate) trait SpeechEngine {
    fn engine_id(&self) -> &'static str;
    fn capabilities(&self) -> &'static EngineCapabilities;
}

pub(crate) const WHISPER: EngineCapabilities = EngineCapabilities {
    id: "whisper",
    label: "Whisper",
    description: "Local models with broad language support",
    maturity: EngineMaturity::Stable,
    preview_mode: PreviewMode::FullSnapshot,
    final_mode: FinalMode::IndependentFullDecode,
    supports_files: true,
    supports_translation: true,
    supports_vocabulary: true,
    requires_cuda: false,
};

pub(crate) const PARAKEET: EngineCapabilities = EngineCapabilities {
    id: "parakeet",
    label: "Parakeet",
    description: "Local NVIDIA CUDA transcription with full-buffer preview",
    maturity: EngineMaturity::Stable,
    preview_mode: PreviewMode::FullSnapshot,
    final_mode: FinalMode::IndependentFullDecode,
    supports_files: true,
    supports_translation: false,
    supports_vocabulary: true,
    requires_cuda: true,
};

pub(crate) const NEMOTRON: EngineCapabilities = EngineCapabilities {
    id: "nemotron",
    label: "Nemotron Speech",
    description: "Local NVIDIA CUDA true-streaming transcription",
    maturity: EngineMaturity::Experimental,
    preview_mode: PreviewMode::Incremental,
    final_mode: FinalMode::FlushActiveStream,
    supports_files: true,
    supports_translation: false,
    supports_vocabulary: false,
    requires_cuda: true,
};

pub(crate) const APPLE_SPEECH: EngineCapabilities = EngineCapabilities {
    id: "appleSpeech",
    label: "System speech",
    description: "Use the operating-system speech service",
    maturity: EngineMaturity::Stable,
    preview_mode: PreviewMode::Incremental,
    final_mode: FinalMode::FlushActiveStream,
    supports_files: true,
    supports_translation: false,
    supports_vocabulary: true,
    requires_cuda: false,
};

pub(crate) const CLOUD: EngineCapabilities = EngineCapabilities {
    id: "cloud",
    label: "Compatible cloud API",
    description: "OpenAI-compatible transcription endpoint",
    maturity: EngineMaturity::Stable,
    preview_mode: PreviewMode::FullSnapshot,
    final_mode: FinalMode::IndependentFullDecode,
    supports_files: true,
    supports_translation: false,
    supports_vocabulary: false,
    requires_cuda: false,
};
