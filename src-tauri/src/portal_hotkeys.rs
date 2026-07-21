//! Wayland global shortcuts through the XDG Desktop Portal.
//!
//! The `global-hotkey` crate behind `tauri-plugin-global-shortcut` only
//! supports X11 on Linux, so native Wayland sessions register shortcuts
//! through `org.freedesktop.portal.GlobalShortcuts` instead. The portal
//! requires user approval for the binding and emits `Activated`/`Deactivated`
//! signals, which preserve hold-mode press/release semantics.

use std::collections::HashMap;
use std::sync::Mutex;

use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
use futures_util::StreamExt;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::watch;

use crate::{debug_log, HotkeyAction, HotkeyBackendStatus, HotkeyEvent};

pub const STATUS_EVENT: &str = "voxide-hotkey-backend";

/// True when the app runs inside a Wayland session, where X11-based global
/// shortcut grabs cannot see keys pressed in native Wayland windows.
pub fn is_wayland_session() -> bool {
    if std::env::var("WAYLAND_DISPLAY").is_ok_and(|value| !value.trim().is_empty()) {
        return true;
    }
    std::env::var("XDG_SESSION_TYPE")
        .is_ok_and(|value| value.trim().eq_ignore_ascii_case("wayland"))
}

#[derive(Debug, Clone)]
struct ShortcutSpec {
    id: String,
    description: String,
    preferred_trigger: Option<String>,
    action: HotkeyAction,
}

pub struct PortalHotkeyState {
    shutdown: Mutex<Option<watch::Sender<bool>>>,
    status: Mutex<HotkeyBackendStatus>,
}

impl Default for PortalHotkeyState {
    fn default() -> Self {
        Self {
            shutdown: Mutex::new(None),
            status: Mutex::new(HotkeyBackendStatus {
                backend: "portal".into(),
                state: "inactive".into(),
                detail: None,
            }),
        }
    }
}

pub fn current_status(app: &AppHandle) -> HotkeyBackendStatus {
    let state = app.state::<PortalHotkeyState>();
    let status = state
        .status
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    status.clone()
}

fn set_status(app: &AppHandle, state_name: &str, detail: Option<String>) {
    let status = HotkeyBackendStatus {
        backend: "portal".into(),
        state: state_name.into(),
        detail,
    };
    debug_log::append(&format!(
        "Portal hotkeys: {}{}",
        status.state,
        status
            .detail
            .as_deref()
            .map(|detail| format!(" ({detail})"))
            .unwrap_or_default()
    ));
    let managed = app.state::<PortalHotkeyState>();
    if let Ok(mut current) = managed.status.lock() {
        *current = status.clone();
    }
    let _ = app.emit(STATUS_EVENT, status);
}

/// Replaces the active portal binding with the given shortcut set. The portal
/// has no unbind call at interface version 1, so each change closes the old
/// session and binds a fresh one, which the compositor may confirm with the
/// user.
pub fn apply(app: &AppHandle, hotkeys: Vec<(String, HotkeyAction)>) {
    let specs = shortcut_specs(&hotkeys);
    let (sender, receiver) = watch::channel(false);
    {
        let state = app.state::<PortalHotkeyState>();
        let mut shutdown = state
            .shutdown
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(previous) = shutdown.replace(sender) {
            let _ = previous.send(true);
        }
    }
    set_status(app, "initializing", None);
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        run_portal_session(app, specs, receiver).await;
    });
}

async fn run_portal_session(
    app: AppHandle,
    specs: Vec<ShortcutSpec>,
    mut shutdown: watch::Receiver<bool>,
) {
    let actions: HashMap<String, HotkeyAction> = specs
        .iter()
        .map(|spec| (spec.id.clone(), spec.action.clone()))
        .collect();
    let portal = match GlobalShortcuts::new().await {
        Ok(portal) => portal,
        Err(error) => {
            set_status(
                &app,
                "unavailable",
                Some(format!(
                    "The desktop global shortcuts portal is not available: {error}"
                )),
            );
            return;
        }
    };
    if let Err(error) = register_host_app(&portal, &app).await {
        debug_log::append(&format!(
            "Portal hotkeys: host-app registration unavailable ({error})"
        ));
    }
    let session = match portal.create_session().await {
        Ok(session) => session,
        Err(error) => {
            let message = error.to_string();
            let detail = if message.contains("app id is required") {
                format!(
                    "The desktop portal could not identify Voxide. Install the application (or add a {}.desktop entry) and re-apply the shortcuts",
                    app.config().identifier
                )
            } else {
                format!("Could not start a portal shortcut session: {message}")
            };
            set_status(&app, "error", Some(detail));
            return;
        }
    };
    // The session serializes as its D-Bus object path; signals arrive for
    // every session on the connection and must be filtered to this one.
    let session_path = serde_json::to_value(&session)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned));
    let streams = futures_util::try_join!(portal.receive_activated(), portal.receive_deactivated());
    let (mut activated, mut deactivated) = match streams {
        Ok(streams) => streams,
        Err(error) => {
            set_status(
                &app,
                "error",
                Some(format!("Could not listen for portal shortcuts: {error}")),
            );
            let _ = session.close().await;
            return;
        }
    };
    let new_shortcuts: Vec<NewShortcut> = specs
        .iter()
        .map(|spec| {
            let shortcut = NewShortcut::new(spec.id.clone(), spec.description.clone());
            match spec.preferred_trigger.as_deref() {
                Some(trigger) => shortcut.preferred_trigger(trigger),
                None => shortcut,
            }
        })
        .collect();
    let bound = match portal.bind_shortcuts(&session, &new_shortcuts, None).await {
        Ok(request) => request.response().map_err(|error| error.to_string()),
        Err(error) => Err(error.to_string()),
    };
    match bound {
        Ok(response) => {
            let bound_count = response.shortcuts().len();
            set_status(
                &app,
                "active",
                Some(format!(
                    "{bound_count} of {} shortcuts bound through the desktop portal",
                    specs.len()
                )),
            );
        }
        Err(error) => {
            set_status(
                &app,
                "denied",
                Some(format!(
                    "Global shortcut binding was not granted: {error}. If your compositor has no working shortcuts portal, bind keys in its configuration to `voxide --trigger dictate` (and the other actions) instead"
                )),
            );
            let _ = session.close().await;
            return;
        }
    }
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            event = activated.next() => {
                let Some(event) = event else { break };
                if session_matches(session_path.as_deref(), event.session_handle().as_str()) {
                    emit_hotkey(&app, &actions, event.shortcut_id(), "pressed");
                }
            }
            event = deactivated.next() => {
                let Some(event) = event else { break };
                if session_matches(session_path.as_deref(), event.session_handle().as_str()) {
                    emit_hotkey(&app, &actions, event.shortcut_id(), "released");
                }
            }
        }
    }
    let _ = session.close().await;
}

/// Returns whether `<app_id>.desktop` exists in the XDG application
/// directories. The portal registry only accepts app ids that resolve to a
/// desktop entry.
fn desktop_entry_exists(app_id: &str) -> bool {
    let file_name = format!("{app_id}.desktop");
    let mut directories: Vec<std::path::PathBuf> = Vec::new();
    match std::env::var("XDG_DATA_HOME") {
        Ok(home) if !home.trim().is_empty() => directories.push(home.into()),
        _ => {
            if let Ok(home) = std::env::var("HOME") {
                directories.push(std::path::Path::new(&home).join(".local/share"));
            }
        }
    }
    let data_dirs = std::env::var("XDG_DATA_DIRS")
        .ok()
        .filter(|dirs| !dirs.trim().is_empty())
        .unwrap_or_else(|| "/usr/local/share:/usr/share".into());
    directories.extend(
        data_dirs
            .split(':')
            .filter(|directory| !directory.is_empty())
            .map(std::path::PathBuf::from),
    );
    directories
        .iter()
        .any(|directory| directory.join("applications").join(&file_name).exists())
}

/// The app ids worth trying for portal registration: the Tauri identifier
/// plus the product/binary names the Linux bundles may use for the desktop
/// entry.
fn host_app_id_candidates(app: &AppHandle) -> Vec<String> {
    let mut candidates = vec![app.config().identifier.clone()];
    let product = app.package_info().name.clone();
    if !product.is_empty() {
        candidates.push(product.clone());
        candidates.push(product.to_lowercase());
    }
    candidates.dedup();
    candidates
}

/// Associates this process's portal D-Bus connection with an app id through
/// `org.freedesktop.host.portal.Registry`. Without it the portal rejects
/// `CreateSession` for unsandboxed apps launched outside a `.desktop`
/// activation ("An app id is required"). The registry requires the id to
/// match an installed desktop entry, so the first candidate with one wins.
/// Registration is once per connection, so repeat calls on rebinding fail
/// harmlessly.
async fn register_host_app(portal: &GlobalShortcuts<'_>, app: &AppHandle) -> Result<(), String> {
    use std::sync::atomic::{AtomicBool, Ordering};
    static HOST_APP_REGISTERED: AtomicBool = AtomicBool::new(false);
    if HOST_APP_REGISTERED.load(Ordering::Acquire) || ashpd::is_sandboxed().await {
        return Ok(());
    }
    let candidates = host_app_id_candidates(app);
    let app_id = candidates
        .iter()
        .find(|candidate| desktop_entry_exists(candidate))
        .unwrap_or(&candidates[0])
        .clone();
    let registry: ashpd::zbus::Proxy = ashpd::zbus::proxy::Builder::new(portal.connection())
        .interface("org.freedesktop.host.portal.Registry")
        .and_then(|builder| builder.path("/org/freedesktop/portal/desktop"))
        .and_then(|builder| builder.destination("org.freedesktop.portal.Desktop"))
        .map_err(|error| error.to_string())?
        .cache_properties(ashpd::zbus::proxy::CacheProperties::No)
        .build()
        .await
        .map_err(|error| error.to_string())?;
    let options = HashMap::<String, ashpd::zvariant::Value>::new();
    registry
        .call_method("Register", &(app_id.as_str(), options))
        .await
        .map_err(|error| error.to_string())?;
    debug_log::append(&format!("Portal hotkeys: registered host app id {app_id}"));
    HOST_APP_REGISTERED.store(true, Ordering::Release);
    Ok(())
}

fn session_matches(session_path: Option<&str>, event_session: &str) -> bool {
    session_path.map_or(true, |path| path == event_session)
}

fn emit_hotkey(
    app: &AppHandle,
    actions: &HashMap<String, HotkeyAction>,
    shortcut_id: &str,
    phase: &str,
) {
    let Some(action) = actions.get(shortcut_id).cloned() else {
        return;
    };
    let (action, prompt_profile_id) = match action {
        HotkeyAction::PromptProfile(profile_id) => (HotkeyAction::Prompt, Some(profile_id)),
        action => (action, None),
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

/// Portal shortcut ids must stay stable across launches so the compositor can
/// remember the user's approved bindings.
fn shortcut_specs(hotkeys: &[(String, HotkeyAction)]) -> Vec<ShortcutSpec> {
    let mut dictate_count = 0u32;
    hotkeys
        .iter()
        .map(|(accelerator, action)| {
            let (id, description) = match action {
                HotkeyAction::Dictate => {
                    dictate_count += 1;
                    if dictate_count == 1 {
                        ("dictate".to_owned(), "Start or stop dictation".to_owned())
                    } else {
                        (
                            format!("dictate-{dictate_count}"),
                            "Start or stop dictation (secondary shortcut)".to_owned(),
                        )
                    }
                }
                HotkeyAction::Prompt => (
                    "prompt-mode".to_owned(),
                    "Start prompt mode dictation".to_owned(),
                ),
                HotkeyAction::PromptProfile(profile_id) => (
                    format!("prompt-profile-{profile_id}"),
                    "Start dictation with a prompt profile".to_owned(),
                ),
                HotkeyAction::Command => (
                    "command-mode".to_owned(),
                    "Start command mode dictation".to_owned(),
                ),
                HotkeyAction::Rewrite => (
                    "rewrite-mode".to_owned(),
                    "Rewrite the selected text".to_owned(),
                ),
                HotkeyAction::Cancel => (
                    "cancel-recording".to_owned(),
                    "Cancel the active recording".to_owned(),
                ),
                HotkeyAction::PasteLast => (
                    "paste-last-transcription".to_owned(),
                    "Paste the last transcription".to_owned(),
                ),
            };
            ShortcutSpec {
                id,
                description,
                preferred_trigger: trigger_from_accelerator(accelerator),
                action: action.clone(),
            }
        })
        .collect()
}

/// Converts the app's accelerator strings (W3C `KeyboardEvent.code` names from
/// the settings recorder, plus common Tauri aliases) into the XDG shortcuts
/// specification trigger format, e.g. `Ctrl+Shift+KeyD` → `CTRL+SHIFT+d`.
/// Returns `None` for keys without a known XKB keysym name; the shortcut is
/// then bound without a preferred trigger and the user picks one in the
/// compositor's dialog.
fn trigger_from_accelerator(accelerator: &str) -> Option<String> {
    let mut has_ctrl = false;
    let mut has_alt = false;
    let mut has_shift = false;
    let mut has_logo = false;
    let mut key: Option<String> = None;
    for part in accelerator.split('+').map(str::trim) {
        if part.is_empty() {
            return None;
        }
        match part.to_ascii_lowercase().as_str() {
            "ctrl" | "control" | "commandorcontrol" | "commandorctrl" | "cmdorctrl"
            | "cmdorcontrol" => has_ctrl = true,
            "alt" | "option" => has_alt = true,
            "shift" => has_shift = true,
            "super" | "meta" | "cmd" | "command" => has_logo = true,
            _ => {
                if key.is_some() {
                    return None;
                }
                key = Some(keysym_from_code(part)?);
            }
        }
    }
    let key = key?;
    let mut trigger = String::new();
    for (enabled, name) in [
        (has_ctrl, "CTRL"),
        (has_alt, "ALT"),
        (has_shift, "SHIFT"),
        (has_logo, "LOGO"),
    ] {
        if enabled {
            trigger.push_str(name);
            trigger.push('+');
        }
    }
    trigger.push_str(&key);
    Some(trigger)
}

fn keysym_from_code(code: &str) -> Option<String> {
    if let Some(letter) = code.strip_prefix("Key") {
        if letter.len() == 1 && letter.chars().all(|c| c.is_ascii_alphabetic()) {
            return Some(letter.to_ascii_lowercase());
        }
    }
    if let Some(digit) = code.strip_prefix("Digit") {
        if digit.len() == 1 && digit.chars().all(|c| c.is_ascii_digit()) {
            return Some(digit.to_owned());
        }
    }
    if code.len() == 1 && code.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Some(code.to_ascii_lowercase());
    }
    if let Some(number) = code.strip_prefix('F') {
        if !number.is_empty()
            && number.chars().all(|c| c.is_ascii_digit())
            && (1..=24).contains(&number.parse::<u8>().ok()?)
        {
            return Some(code.to_owned());
        }
    }
    if let Some(numpad) = code.strip_prefix("Numpad") {
        let keysym = match numpad {
            "Add" => "KP_Add",
            "Subtract" => "KP_Subtract",
            "Multiply" => "KP_Multiply",
            "Divide" => "KP_Divide",
            "Decimal" => "KP_Decimal",
            "Enter" => "KP_Enter",
            digit if digit.len() == 1 && digit.chars().all(|c| c.is_ascii_digit()) => {
                return Some(format!("KP_{digit}"));
            }
            _ => return None,
        };
        return Some(keysym.to_owned());
    }
    let keysym = match code {
        "Space" => "space",
        "Enter" | "Return" => "Return",
        "Escape" | "Esc" => "Escape",
        "Tab" => "Tab",
        "Backspace" => "BackSpace",
        "Delete" => "Delete",
        "Insert" => "Insert",
        "Home" => "Home",
        "End" => "End",
        "PageUp" => "Page_Up",
        "PageDown" => "Page_Down",
        "ArrowUp" | "Up" => "Up",
        "ArrowDown" | "Down" => "Down",
        "ArrowLeft" | "Left" => "Left",
        "ArrowRight" | "Right" => "Right",
        "Comma" => "comma",
        "Period" => "period",
        "Slash" => "slash",
        "Backslash" => "backslash",
        "Semicolon" => "semicolon",
        "Quote" => "apostrophe",
        "BracketLeft" => "bracketleft",
        "BracketRight" => "bracketright",
        "Minus" => "minus",
        "Equal" => "equal",
        "Backquote" => "grave",
        "PrintScreen" => "Print",
        "ScrollLock" => "Scroll_Lock",
        "Pause" => "Pause",
        _ => return None,
    };
    Some(keysym.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_default_dictation_accelerator() {
        assert_eq!(
            trigger_from_accelerator("Alt+Space").as_deref(),
            Some("ALT+space")
        );
    }

    #[test]
    fn converts_recorder_style_accelerators() {
        assert_eq!(
            trigger_from_accelerator("Ctrl+Shift+KeyD").as_deref(),
            Some("CTRL+SHIFT+d")
        );
        assert_eq!(
            trigger_from_accelerator("Super+Digit1").as_deref(),
            Some("LOGO+1")
        );
        assert_eq!(
            trigger_from_accelerator("Ctrl+Alt+Comma").as_deref(),
            Some("CTRL+ALT+comma")
        );
        assert_eq!(
            trigger_from_accelerator("Ctrl+NumpadAdd").as_deref(),
            Some("CTRL+KP_Add")
        );
    }

    #[test]
    fn converts_tauri_style_accelerators() {
        assert_eq!(
            trigger_from_accelerator("CommandOrControl+Shift+D").as_deref(),
            Some("CTRL+SHIFT+d")
        );
        assert_eq!(trigger_from_accelerator("F5").as_deref(), Some("F5"));
        assert_eq!(
            trigger_from_accelerator("Escape").as_deref(),
            Some("Escape")
        );
    }

    #[test]
    fn rejects_unknown_or_malformed_accelerators() {
        assert_eq!(trigger_from_accelerator("Ctrl+MediaPlayPause"), None);
        assert_eq!(trigger_from_accelerator("Ctrl+"), None);
        assert_eq!(trigger_from_accelerator("Ctrl+KeyA+KeyB"), None);
        assert_eq!(trigger_from_accelerator(""), None);
        assert_eq!(trigger_from_accelerator("F25"), None);
    }

    #[test]
    fn shortcut_ids_are_stable_and_distinct() {
        let hotkeys = vec![
            ("Alt+Space".to_owned(), HotkeyAction::Dictate),
            ("Ctrl+Shift+KeyD".to_owned(), HotkeyAction::Dictate),
            (
                "Ctrl+Shift+KeyP".to_owned(),
                HotkeyAction::PromptProfile("profile-a".into()),
            ),
            ("Escape".to_owned(), HotkeyAction::Cancel),
        ];
        let specs = shortcut_specs(&hotkeys);
        let ids: Vec<&str> = specs.iter().map(|spec| spec.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "dictate",
                "dictate-2",
                "prompt-profile-profile-a",
                "cancel-recording"
            ]
        );
        assert_eq!(
            specs[2].action,
            HotkeyAction::PromptProfile("profile-a".into())
        );
        assert_eq!(specs[0].preferred_trigger.as_deref(), Some("ALT+space"));
    }
}
