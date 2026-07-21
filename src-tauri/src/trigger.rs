//! External hotkey triggers over a Unix socket.
//!
//! Wayland compositors without a working XDG GlobalShortcuts backend (for
//! example Niri, whose distributions route the portal to the GNOME backend
//! that requires GNOME Shell) can still drive Voxide by binding keys in
//! the compositor configuration to `voxide --trigger <action>`. The
//! spawned process forwards the action to the running instance and exits.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as TokioBufReader};
use tokio::net::UnixListener;

use crate::{debug_log, HotkeyAction, HotkeyEvent};

pub const ACTIONS: [&str; 6] = [
    "dictate",
    "prompt",
    "command",
    "rewrite",
    "cancel",
    "paste-last",
];

fn action_from_name(name: &str) -> Option<HotkeyAction> {
    match name {
        "dictate" => Some(HotkeyAction::Dictate),
        "prompt" => Some(HotkeyAction::Prompt),
        "command" => Some(HotkeyAction::Command),
        "rewrite" => Some(HotkeyAction::Rewrite),
        "cancel" => Some(HotkeyAction::Cancel),
        "paste-last" => Some(HotkeyAction::PasteLast),
        _ => None,
    }
}

fn socket_path() -> PathBuf {
    let base = std::env::var("XDG_RUNTIME_DIR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("voxide-trigger.sock")
}

/// Sends one action to the running Voxide instance. Used by the
/// `--trigger` command-line mode; returns the instance's reply.
pub fn send(action: &str) -> Result<(), String> {
    if action_from_name(action).is_none() {
        return Err(format!(
            "Unknown trigger action '{action}'. Supported actions: {}",
            ACTIONS.join(", ")
        ));
    }
    let path = socket_path();
    let mut stream = UnixStream::connect(&path).map_err(|error| {
        format!(
            "Could not reach a running Voxide instance at {}: {error}",
            path.display()
        )
    })?;
    stream
        .write_all(format!("{action}\n").as_bytes())
        .map_err(|error| format!("Could not send the trigger action: {error}"))?;
    let mut reply = String::new();
    BufReader::new(&stream)
        .read_line(&mut reply)
        .map_err(|error| format!("Could not read the trigger reply: {error}"))?;
    let reply = reply.trim();
    if reply == "ok" {
        Ok(())
    } else {
        Err(format!("Voxide rejected the trigger: {reply}"))
    }
}

/// Starts the trigger listener for this instance. An existing stale socket
/// from a previous run is replaced; the newest instance owns the socket.
pub fn start_listener(app: AppHandle) {
    let path = socket_path();
    tauri::async_runtime::spawn(async move {
        let _ = std::fs::remove_file(&path);
        let listener = match UnixListener::bind(&path) {
            Ok(listener) => listener,
            Err(error) => {
                debug_log::append(&format!(
                    "Trigger socket unavailable ({}): {error}",
                    path.display()
                ));
                return;
            }
        };
        debug_log::append(&format!("Trigger socket listening at {}", path.display()));
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                let mut reader = TokioBufReader::new(stream);
                let mut line = String::new();
                if reader.read_line(&mut line).await.is_err() {
                    return;
                }
                let name = line.trim();
                let reply = match action_from_name(name) {
                    Some(action) => {
                        // Compositor keybindings fire once per press, so a
                        // trigger behaves like a tap: press and release.
                        for phase in ["pressed", "released"] {
                            let _ = app.emit(
                                "voxide-hotkey",
                                HotkeyEvent {
                                    action: action.clone(),
                                    phase: phase.into(),
                                    prompt_profile_id: None,
                                },
                            );
                        }
                        "ok".to_owned()
                    }
                    None => format!(
                        "unknown action '{name}' (supported: {})",
                        ACTIONS.join(", ")
                    ),
                };
                let mut stream = reader.into_inner();
                let _ = stream.write_all(format!("{reply}\n").as_bytes()).await;
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_documented_action_maps_to_a_hotkey_action() {
        for name in ACTIONS {
            assert!(action_from_name(name).is_some(), "unmapped action {name}");
        }
        assert!(action_from_name("unknown").is_none());
        assert!(action_from_name("").is_none());
    }

    #[test]
    fn unknown_actions_are_rejected_before_touching_the_socket() {
        let error = send("not-an-action").unwrap_err();
        assert!(error.contains("Unknown trigger action"));
    }
}
