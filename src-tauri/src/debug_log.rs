//! Small, privacy-preserving diagnostic log used only for explicit feedback.
//!
//! Entries must be operational metadata. Callers must not include dictation,
//! command, file-path, credential, or clipboard content.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Duration, SystemTime},
};

use chrono::Utc;
use directories::ProjectDirs;

const MAX_BYTES: u64 = 1024 * 1024;
const MAX_AGE: Duration = Duration::from_secs(72 * 60 * 60);

static WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn directory() -> Option<PathBuf> {
    ProjectDirs::from("dev", "pmdcoutinho", "Voxide")
        .map(|directories| directories.data_local_dir().join("logs"))
}

fn current_path() -> Option<PathBuf> {
    directory().map(|directory| directory.join("voxide.log"))
}

fn backup_path() -> Option<PathBuf> {
    directory().map(|directory| directory.join("voxide.log.1"))
}

fn rotate_if_needed(path: &PathBuf) {
    let Ok(metadata) = fs::metadata(path) else {
        return;
    };
    let older_than_limit = metadata
        .modified()
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age >= MAX_AGE);
    if metadata.len() < MAX_BYTES && !older_than_limit {
        return;
    }
    if let Some(backup) = backup_path() {
        let _ = fs::remove_file(&backup);
        let _ = fs::rename(path, backup);
    }
}

/// Adds a single bounded metadata entry. Failures are deliberately ignored so
/// diagnostics cannot interfere with dictation or other user actions.
pub fn append(event: &str) {
    let Some(path) = current_path() else {
        return;
    };
    let lock = WRITE_LOCK.get_or_init(|| Mutex::new(()));
    let Ok(_lock) = lock.lock() else {
        return;
    };
    let Some(directory) = path.parent() else {
        return;
    };
    if fs::create_dir_all(directory).is_err() {
        return;
    }
    rotate_if_needed(&path);
    let sanitized = event
        .chars()
        .filter(|character| !character.is_control() || *character == ' ')
        .take(500)
        .collect::<String>();
    let sanitized = redact_sensitive_tokens(&sanitized);
    let line = format!("{} {}\n", Utc::now().to_rfc3339(), sanitized);
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}

/// The diagnostic API is deliberately metadata-only, but error messages from
/// dependencies can still contain a path, URL, or credential-like token.
/// Redact those shapes centrally before a user can export the log.
fn redact_sensitive_tokens(value: &str) -> String {
    value
        .split_whitespace()
        .map(|token| {
            let lower = token.to_ascii_lowercase();
            let looks_like_path = token.starts_with('/')
                || token.starts_with("~/")
                || token.starts_with("file:")
                || token.contains("\\\\")
                || (token.len() > 2
                    && token.as_bytes()[1] == b':'
                    && matches!(token.as_bytes()[0], b'a'..=b'z' | b'A'..=b'Z'));
            let looks_like_url = token.contains("://");
            let looks_like_secret = lower.contains("api_key")
                || lower.contains("authorization")
                || lower.contains("bearer")
                || lower.contains("token=")
                || lower.contains("sk-");
            if looks_like_secret {
                "<redacted-secret>"
            } else if looks_like_path {
                "<redacted-path>"
            } else if looks_like_url {
                "<redacted-url>"
            } else {
                token
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns only the last `limit` lines of the current log for an explicitly
/// user-initiated feedback report.
pub fn recent_lines(limit: usize) -> String {
    let Some(path) = current_path() else {
        return String::new();
    };
    let Ok(contents) = fs::read_to_string(path) else {
        return String::new();
    };
    let mut lines = contents.lines().rev().take(limit).collect::<Vec<_>>();
    lines.reverse();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_entries_are_bounded_and_strip_newlines() {
        let value = "line one\nline two";
        let sanitized = value
            .chars()
            .filter(|character| !character.is_control() || *character == ' ')
            .take(500)
            .collect::<String>();
        assert_eq!(sanitized, "line oneline two");
    }

    #[test]
    fn diagnostics_redact_paths_urls_and_credential_like_tokens() {
        let value = redact_sensitive_tokens(
            "failed /home/alice/private.wav https://example.test?token=abc api_key=secret C:\\Users\\alice",
        );
        assert!(!value.contains("alice"));
        assert!(!value.contains("api_key=secret"));
        assert_eq!(
            value,
            "failed <redacted-path> <redacted-secret> <redacted-secret> <redacted-path>"
        );
    }
}
