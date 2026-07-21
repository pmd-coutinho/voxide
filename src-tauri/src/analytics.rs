use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

// Analytics are inert unless a PostHog ingestion key is supplied at build
// time (VOXIDE_POSTHOG_KEY); source builds and forks send nothing anywhere.
const POSTHOG_API_KEY: &str = match option_env!("VOXIDE_POSTHOG_KEY") {
    Some(key) => key,
    None => "",
};
const POSTHOG_HOST: &str = "https://eu.i.posthog.com";
const MAX_QUEUED_EVENTS: usize = 200;
const FLUSH_AT: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct AnalyticsIdentity {
    anonymous_install_id: Option<String>,
    first_open_at: Option<DateTime<Utc>>,
}

struct AnalyticsCore {
    identity_path: PathBuf,
    identity: AnalyticsIdentity,
    enabled: bool,
    queue: Vec<Value>,
}

#[derive(Clone)]
pub struct AnalyticsService {
    core: Arc<Mutex<AnalyticsCore>>,
    client: reqwest::Client,
}

impl AnalyticsService {
    pub fn load(identity_path: PathBuf) -> Self {
        let identity = fs::read(&identity_path)
            .ok()
            .and_then(|contents| serde_json::from_slice(&contents).ok())
            .unwrap_or_default();
        Self {
            core: Arc::new(Mutex::new(AnalyticsCore {
                identity_path,
                identity,
                enabled: false,
                queue: Vec::new(),
            })),
            client: reqwest::Client::new(),
        }
    }

    /// Enables or disables collection and returns whether this is the install's first launch.
    /// Identity is written independently from the main backup, just as the reference keeps it in
    /// app defaults rather than its exported settings archive.
    pub fn bootstrap(&self, enabled: bool) -> bool {
        let mut core = match self.core.lock() {
            Ok(core) => core,
            Err(_) => return false,
        };
        core.enabled = enabled;
        let first_open = core.identity.first_open_at.is_none();
        if core.identity.anonymous_install_id.is_none() {
            core.identity.anonymous_install_id = Some(Uuid::new_v4().to_string());
        }
        if first_open {
            core.identity.first_open_at = Some(Utc::now());
        }
        persist_identity(&core.identity_path, &core.identity);
        first_open
    }

    pub fn set_enabled(&self, enabled: bool) {
        if let Ok(mut core) = self.core.lock() {
            core.enabled = enabled;
            if !enabled {
                // Honor opt-out immediately; no queued event should survive it.
                core.queue.clear();
            }
        }
    }

    pub fn capture(&self, event_name: &str, enabled: bool, mut properties: Map<String, Value>) {
        if POSTHOG_API_KEY.is_empty() {
            return;
        }
        let should_flush = {
            let mut core = match self.core.lock() {
                Ok(core) => core,
                Err(_) => return,
            };
            if core.enabled != enabled {
                core.enabled = enabled;
                if !enabled {
                    core.queue.clear();
                }
            }
            if !core.enabled {
                return;
            }
            let Some(distinct_id) = core.identity.anonymous_install_id.clone() else {
                return;
            };
            properties.insert("distinct_id".into(), Value::String(distinct_id));
            properties.insert("$lib".into(), Value::String("Voxide".into()));
            properties.insert("$lib_version".into(), Value::String("1".into()));
            core.queue.push(json!({
                "event": event_name,
                "timestamp": Utc::now().to_rfc3339(),
                "properties": properties,
            }));
            if core.queue.len() > MAX_QUEUED_EVENTS {
                let excess = core.queue.len() - MAX_QUEUED_EVENTS;
                core.queue.drain(..excess);
            }
            core.queue.len() >= FLUSH_AT
        };
        if should_flush {
            self.spawn_flush();
        }
    }

    pub fn start_flush_loop(&self) {
        let analytics = self.clone();
        tauri::async_runtime::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                analytics.flush().await;
            }
        });
    }

    fn spawn_flush(&self) {
        let analytics = self.clone();
        tauri::async_runtime::spawn(async move {
            analytics.flush().await;
        });
    }

    async fn flush(&self) {
        let batch = {
            let mut core = match self.core.lock() {
                Ok(core) => core,
                Err(_) => return,
            };
            if !core.enabled || core.queue.is_empty() || POSTHOG_API_KEY.is_empty() {
                return;
            }
            std::mem::take(&mut core.queue)
        };
        // Deliberately fire-and-forget. The reference does not retry failed batches and analytics
        // must never delay dictation or application shutdown.
        let _ = self
            .client
            .post(format!("{POSTHOG_HOST}/batch"))
            .json(&json!({ "api_key": POSTHOG_API_KEY, "batch": batch }))
            .timeout(Duration::from_secs(8))
            .send()
            .await;
    }
}

fn persist_identity(path: &PathBuf, identity: &AnalyticsIdentity) {
    let Ok(contents) = serde_json::to_vec(identity) else {
        return;
    };
    let temporary = path.with_extension("json.tmp");
    if fs::write(&temporary, contents).is_ok() {
        let _ = fs::rename(temporary, path);
    }
}

pub fn word_count_bucket(text: &str) -> &'static str {
    let count = text.split_whitespace().count();
    match count {
        0 => "0",
        1..=5 => "1-5",
        6..=20 => "6-20",
        21..=50 => "21-50",
        51..=100 => "51-100",
        101..=300 => "101-300",
        _ => "301+",
    }
}

pub fn milliseconds_bucket(milliseconds: u64) -> &'static str {
    match milliseconds {
        0..=99 => "<100ms",
        100..=299 => "100-300ms",
        300..=999 => "300ms-1s",
        1_000..=2_999 => "1-3s",
        3_000..=9_999 => "3-10s",
        _ => "10s+",
    }
}

#[cfg(test)]
mod tests {
    use super::{milliseconds_bucket, word_count_bucket};

    #[test]
    fn buckets_telemetry_without_retaining_transcription_content() {
        assert_eq!(word_count_bucket(""), "0");
        assert_eq!(word_count_bucket("one two three"), "1-5");
        assert_eq!(word_count_bucket(&"word ".repeat(21)), "21-50");
        assert_eq!(milliseconds_bucket(99), "<100ms");
        assert_eq!(milliseconds_bucket(100), "100-300ms");
        assert_eq!(milliseconds_bucket(10_000), "10s+");
    }
}
