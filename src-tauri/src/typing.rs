use std::{
    fmt, thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arboard::{Clipboard, ImageData};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

use crate::debug_log;

#[cfg(target_os = "linux")]
use crate::{libei_input, modifier_guard};

/// Chooses whether insertion should prefer clipboard-free desktop input or a
/// temporary clipboard paste. Values intentionally match Voxide backups.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TextInsertionMode {
    #[default]
    Standard,
    ReliablePaste,
}

impl TextInsertionMode {
    pub fn from_persisted(value: Option<&str>) -> Self {
        match value {
            Some("reliablePaste") => Self::ReliablePaste,
            _ => Self::Standard,
        }
    }
}

/// One way of getting synthetic key events to the focused application.
///
/// These are tried in order rather than all at once, which matters more than it
/// looks: `enigo` connects to *every* Linux protocol it can and forwards each
/// event to all of them. A bare `Enigo::new(&Settings::default())` on a Wayland
/// session that also runs Xwayland therefore holds both a virtual-keyboard and
/// an XTEST connection and types the text once through each. Pinning the
/// display name of an unwanted protocol to an empty string makes that leg fail
/// to connect, which is how each variant below gets exclusive use of exactly
/// one protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeySynthesis {
    /// The operating system's own input API — `SendInput` on Windows,
    /// `CGEvent` on macOS. The only backend outside Linux.
    #[cfg(not(target_os = "linux"))]
    Native,
    /// Wayland's `zwp_virtual_keyboard_v1`. Uploads its own keymap, so any
    /// Unicode text can be typed regardless of the user's layout. Implemented
    /// by wlroots-based compositors (Sway, Niri, Hyprland, river).
    #[cfg(target_os = "linux")]
    WaylandVirtualKeyboard,
    /// XTEST, either against a real X server or through Xwayland. Mutter and
    /// KWin forward Xwayland's XTEST requests to native Wayland clients;
    /// wlroots compositors do not, so on those this reaches X11 clients only.
    #[cfg(target_os = "linux")]
    X11Xtest,
    /// libei through the `org.freedesktop.portal.RemoteDesktop` portal. This is
    /// the only synthesis path on a pure-Wayland GNOME or KDE session that
    /// implements neither `zwp_virtual_keyboard_v1` nor Xwayland. Restricted to
    /// shortcuts: libei hands out the compositor's keymap instead of accepting
    /// one, so it can only reach keysyms the user's layout already has.
    #[cfg(target_os = "linux")]
    Libei,
}

impl fmt::Display for KeySynthesis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            #[cfg(not(target_os = "linux"))]
            Self::Native => "native",
            #[cfg(target_os = "linux")]
            Self::WaylandVirtualKeyboard => "wayland-virtual-keyboard",
            #[cfg(target_os = "linux")]
            Self::X11Xtest => "x11-xtest",
            #[cfg(target_os = "linux")]
            Self::Libei => "libei-portal",
        };
        formatter.write_str(name)
    }
}

impl KeySynthesis {
    /// Settings that leave this backend as the only one `enigo` can connect to.
    /// `None` for backends that do not go through `enigo` at all.
    ///
    /// An empty display name cannot be connected to, so naming it for a protocol
    /// drops that leg of enigo's fan-out and leaves only the wanted one.
    #[cfg(target_os = "linux")]
    fn enigo_settings(self) -> Option<Settings> {
        const UNREACHABLE_DISPLAY: &str = "";
        let defaults = Settings::default();
        match self {
            Self::WaylandVirtualKeyboard => Some(Settings {
                x11_display: Some(UNREACHABLE_DISPLAY.to_string()),
                ..defaults
            }),
            Self::X11Xtest => Some(Settings {
                wayland_display: Some(UNREACHABLE_DISPLAY.to_string()),
                ..defaults
            }),
            Self::Libei => None,
        }
    }

    /// Off Linux there is a single backend and nothing to disambiguate.
    #[cfg(not(target_os = "linux"))]
    fn enigo_settings(self) -> Option<Settings> {
        match self {
            Self::Native => Some(Settings::default()),
        }
    }

    /// Whether this backend can type arbitrary text, as opposed to only being
    /// able to reach keys that exist in the active keyboard layout.
    fn can_type_arbitrary_text(self) -> bool {
        #[cfg(target_os = "linux")]
        return !matches!(self, Self::Libei);
        #[cfg(not(target_os = "linux"))]
        return true;
    }

    fn connect(self) -> Result<Enigo, String> {
        let settings = self
            .enigo_settings()
            .ok_or_else(|| format!("{self} does not provide a general-purpose input connection"))?;
        Enigo::new(&settings).map_err(|error| error.to_string())
    }
}

/// The synthesis backends to try, most preferred first.
///
/// Ordering rationale on Linux: the virtual-keyboard protocol is the only path
/// that can type arbitrary Unicode without borrowing the user's layout, so it
/// leads. XTEST follows because it is nearly always present and needs no user
/// approval. libei is last because reaching it means asking the portal for
/// remote-control permission, which shows a system dialog the first time.
fn synthesis_chain() -> Vec<KeySynthesis> {
    #[cfg(target_os = "linux")]
    {
        let mut chain = Vec::new();
        if crate::portal_hotkeys::is_wayland_session() {
            chain.push(KeySynthesis::WaylandVirtualKeyboard);
        }
        chain.push(KeySynthesis::X11Xtest);
        if crate::portal_hotkeys::is_wayland_session() && libei_input::is_possible() {
            chain.push(KeySynthesis::Libei);
        }
        chain
    }
    #[cfg(not(target_os = "linux"))]
    {
        vec![KeySynthesis::Native]
    }
}

/// What the Settings screen reports about text insertion on this desktop.
///
/// Worth surfacing because which rung wins is invisible otherwise, and the
/// answer differs per compositor in ways users cannot be expected to guess.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InsertionDiagnostics {
    /// Backend names in the order they will be tried, most preferred first.
    pub chain: Vec<String>,
    /// The subset of `chain` that can type arbitrary text rather than only
    /// driving the clipboard shortcuts.
    pub direct_capable: Vec<String>,
    /// State of the libei rung, which is the only one that needs permission.
    #[cfg(target_os = "linux")]
    pub libei: crate::libei_input::Status,
    /// Whether held chord modifiers can be detected before typing.
    #[cfg(target_os = "linux")]
    pub modifier_guard: crate::modifier_guard::Availability,
}

/// Reports the insertion chain without synthesizing anything.
pub fn insertion_diagnostics() -> InsertionDiagnostics {
    let chain = synthesis_chain();
    InsertionDiagnostics {
        chain: chain.iter().map(ToString::to_string).collect(),
        direct_capable: chain
            .iter()
            .filter(|backend| backend.can_type_arbitrary_text())
            .map(ToString::to_string)
            .collect(),
        #[cfg(target_os = "linux")]
        libei: libei_input::status(),
        #[cfg(target_os = "linux")]
        modifier_guard: modifier_guard::availability(),
    }
}

/// How a backend is asked to deliver the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    /// Synthesize the text itself, leaving the clipboard untouched.
    DirectText,
    /// Put the text on the clipboard, synthesize the platform paste shortcut,
    /// then restore the previous clipboard contents.
    ClipboardPaste,
}

impl fmt::Display for Strategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::DirectText => "direct",
            Self::ClipboardPaste => "clipboard-paste",
        };
        formatter.write_str(name)
    }
}

/// Records why each rung of the chain declined, so a total failure can explain
/// itself and a partial one can be diagnosed from the debug log. Deliberately
/// holds backend names and error strings only — never dictation text.
struct AttemptLog(Vec<String>);

impl AttemptLog {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn record(&mut self, backend: KeySynthesis, strategy: Strategy, error: &str) {
        debug_log::append(&format!(
            "text insertion via {backend} ({strategy}) failed: {error}"
        ));
        self.0.push(format!("{backend} {strategy}: {error}"));
    }

    fn into_message(self) -> String {
        if self.0.is_empty() {
            return with_compositor_hint(
                "No text insertion backend is available on this desktop session".to_string(),
            );
        }
        with_compositor_hint(format!(
            "Could not insert dictation. Tried {} — {}",
            plural_attempts(self.0.len()),
            self.0.join("; ")
        ))
    }
}

fn plural_attempts(count: usize) -> String {
    if count == 1 {
        "1 input path".to_string()
    } else {
        format!("{count} input paths")
    }
}

/// Inserts a nonempty transcription using the selected reliability policy.
/// Standard mode leaves the clipboard untouched whenever direct insertion
/// works; reliable-paste mode deliberately takes the clipboard path first.
///
/// Every combination of strategy and backend is tried in order until one
/// reports success, so a desktop that supports only one of them still works
/// without the user having to discover which setting to change.
pub fn type_into_active_application(text: &str, mode: TextInsertionMode) -> Result<(), String> {
    if text.trim().is_empty() {
        return Ok(());
    }
    // Once per insertion, not per rung: the modifiers are whatever they are,
    // and a fallback attempt should not pay the wait a second time.
    #[cfg(target_os = "linux")]
    let _ = modifier_guard::wait_for_release();
    let mut attempts = AttemptLog::new();
    for (backend, strategy) in insertion_plan(mode) {
        let outcome = match strategy {
            Strategy::DirectText => insert_direct_text(backend, text),
            Strategy::ClipboardPaste => insert_via_clipboard_paste(backend, text),
        };
        match outcome {
            Ok(()) => {
                debug_log::append(&format!(
                    "inserted {} characters via {backend} ({strategy})",
                    text.chars().count()
                ));
                return Ok(());
            }
            Err(error) => attempts.record(backend, strategy, &error),
        }
    }
    Err(attempts.into_message())
}

/// The ordered (backend, strategy) pairs to attempt for a given mode.
///
/// Both modes end up covering the same set of pairs; they differ only in which
/// strategy is offered first, so a failure of the preferred path still lands
/// the text rather than being reported as an error the user has to act on.
fn insertion_plan(mode: TextInsertionMode) -> Vec<(KeySynthesis, Strategy)> {
    let chain = synthesis_chain();
    let direct: Vec<_> = chain
        .iter()
        .copied()
        .filter(|backend| backend.can_type_arbitrary_text())
        .map(|backend| (backend, Strategy::DirectText))
        .collect();
    let paste: Vec<_> = chain
        .iter()
        .copied()
        .map(|backend| (backend, Strategy::ClipboardPaste))
        .collect();
    match mode {
        TextInsertionMode::Standard => [direct, paste].concat(),
        TextInsertionMode::ReliablePaste => [paste, direct].concat(),
    }
}

fn insert_direct_text(backend: KeySynthesis, text: &str) -> Result<(), String> {
    let mut input = backend.connect()?;
    input.text(text).map_err(|error| error.to_string())
}

/// Appends actionable guidance to input-synthesis failures on Wayland, where
/// simulated keyboard input depends on compositor support for the
/// virtual-keyboard protocol and there is no OS permission prompt to point
/// users toward.
fn with_compositor_hint(message: String) -> String {
    #[cfg(target_os = "linux")]
    if crate::portal_hotkeys::is_wayland_session() {
        return format!(
            "{message}. This Wayland compositor may restrict simulated keyboard input; if insertion keeps failing, switch the text insertion mode in Settings or check the compositor's virtual-keyboard support"
        );
    }
    message
}

/// Copies final dictation text without synthesizing an input shortcut. This is
/// intentionally separate from the temporary clipboard used by reliable paste
/// so an enabled “copy completed dictations” preference is dependable in all
/// supported desktop webviews.
pub fn copy_text_to_clipboard(text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Ok(());
    }
    Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(text))
        .map_err(|error| format!("Could not copy the completed dictation: {error}"))
}

fn insert_via_clipboard_paste(backend: KeySynthesis, text: &str) -> Result<(), String> {
    let mut clipboard = Clipboard::new()
        .map_err(|error| format!("Could not access the system clipboard: {error}"))?;
    let previous_contents = clipboard_snapshot(&mut clipboard)?;
    clipboard
        .set_text(text)
        .map_err(|error| format!("Could not prepare clipboard paste text: {error}"))?;
    let pasted = send_shortcut(backend, Shortcut::Paste);
    // The target application needs a brief window to read the temporary
    // clipboard contents, but restoring them must not add 100 ms to the
    // user's perceived stop-to-text latency. Keeping this Clipboard alive in
    // the detached task is important on Wayland, where it owns the clipboard
    // data source until the paste completes.
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        let _ = restore_clipboard_snapshot(&mut clipboard, previous_contents);
    });
    pasted?;
    Ok(())
}

/// Copies the current selection through the platform's normal copy shortcut.
///
/// The clipboard is used as a portable fallback for accessibility APIs. Text
/// and image clipboard contents are restored after the temporary sentinel; an
/// unknown rich payload is left untouched by declining the capture.
pub fn capture_selected_text() -> Result<String, String> {
    let mut clipboard = Clipboard::new()
        .map_err(|error| format!("Could not access the system clipboard: {error}"))?;
    let previous_contents = clipboard_snapshot(&mut clipboard)?;
    let sentinel = format!(
        "__voxide_selection_{}__",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    clipboard
        .set_text(&sentinel)
        .map_err(|error| format!("Could not prepare the system clipboard: {error}"))?;

    let copied = copy_selection_through_any_backend(&mut clipboard, &sentinel);
    let restore = restore_clipboard_snapshot(&mut clipboard, previous_contents);

    restore?;
    let text = copied?;
    if text.trim().is_empty() {
        return Err(
            "No selected text was copied. Select text in another application and try again.".into(),
        );
    }
    Ok(text)
}

/// Sends the copy shortcut through each backend in turn, stopping at the first
/// one that actually puts something other than the sentinel on the clipboard.
///
/// Success has to be judged by the clipboard rather than by the synthesis call,
/// because XTEST in particular reports success for events the compositor then
/// discards.
fn copy_selection_through_any_backend(
    clipboard: &mut Clipboard,
    sentinel: &str,
) -> Result<String, String> {
    let mut attempts = AttemptLog::new();
    for backend in synthesis_chain() {
        let outcome = send_shortcut(backend, Shortcut::Copy)
            .and_then(|()| wait_for_copied_text(clipboard, sentinel));
        match outcome {
            Ok(text) => {
                debug_log::append(&format!("captured selection via {backend}"));
                return Ok(text);
            }
            Err(error) => attempts.record(backend, Strategy::ClipboardPaste, &error),
        }
    }
    Err(attempts.into_message())
}

enum ClipboardSnapshot {
    Text(String),
    Image(ImageData<'static>),
}

fn clipboard_snapshot(clipboard: &mut Clipboard) -> Result<ClipboardSnapshot, String> {
    clipboard
        .get_text()
        .map(ClipboardSnapshot::Text)
        .or_else(|_| clipboard.get_image().map(ClipboardSnapshot::Image))
        .map_err(|_| {
            "Voxide cannot safely capture a selection while the clipboard contains data that cannot be restored. Copy plain text or an image once, or clear the clipboard, then try again.".to_string()
        })
}

fn restore_clipboard_snapshot(
    clipboard: &mut Clipboard,
    snapshot: ClipboardSnapshot,
) -> Result<(), String> {
    let restored = match snapshot {
        ClipboardSnapshot::Text(text) => clipboard.set_text(text),
        ClipboardSnapshot::Image(image) => clipboard.set_image(image),
    };
    restored.map_err(|error| {
        format!("Voxide captured text but could not restore the clipboard: {error}")
    })
}

/// The two clipboard shortcuts Voxide ever needs to synthesize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shortcut {
    Copy,
    Paste,
}

impl Shortcut {
    fn character(self) -> char {
        match self {
            Self::Copy => 'c',
            Self::Paste => 'v',
        }
    }
}

/// The modifier that pairs with C and V for clipboard actions on this platform.
#[cfg(target_os = "macos")]
const CLIPBOARD_MODIFIER: Key = Key::Meta;
#[cfg(not(target_os = "macos"))]
const CLIPBOARD_MODIFIER: Key = Key::Control;

fn send_shortcut(backend: KeySynthesis, shortcut: Shortcut) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    if backend == KeySynthesis::Libei {
        return libei_input::send_clipboard_shortcut(shortcut.character());
    }
    let mut input = backend.connect()?;
    input
        .key(CLIPBOARD_MODIFIER, Direction::Press)
        .and_then(|()| input.key(Key::Unicode(shortcut.character()), Direction::Click))
        .and_then(|()| input.key(CLIPBOARD_MODIFIER, Direction::Release))
        .map_err(|error| error.to_string())
}

fn wait_for_copied_text(clipboard: &mut Clipboard, sentinel: &str) -> Result<String, String> {
    for _ in 0..20 {
        thread::sleep(Duration::from_millis(15));
        match clipboard.get_text() {
            Ok(text) if text != sentinel => return Ok(text),
            Ok(_) => continue,
            Err(error) => return Err(format!("Could not read the copied selection: {error}")),
        }
    }
    Err("The selected application did not place text on the clipboard. Check its copy permission and try again.".into())
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use super::KeySynthesis;
    use super::{insertion_plan, Strategy, TextInsertionMode};

    #[test]
    fn persisted_insertion_modes_follow_the_reference_contract() {
        assert_eq!(
            TextInsertionMode::from_persisted(Some("reliablePaste")),
            TextInsertionMode::ReliablePaste
        );
        assert_eq!(
            TextInsertionMode::from_persisted(Some("standard")),
            TextInsertionMode::Standard
        );
        assert_eq!(
            TextInsertionMode::from_persisted(Some("future-mode")),
            TextInsertionMode::Standard
        );
        assert_eq!(
            TextInsertionMode::from_persisted(None),
            TextInsertionMode::Standard
        );
    }

    #[test]
    fn empty_completed_dictations_do_not_require_clipboard_access() {
        assert!(super::copy_text_to_clipboard(" \n\t ").is_ok());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn each_enigo_backend_disables_the_protocols_it_does_not_own() {
        // The whole ordered chain depends on this: enigo forwards every event
        // to all connected backends, so a rung is only exclusive if the other
        // protocols are pinned to a display name that cannot be reached.
        let wayland = KeySynthesis::WaylandVirtualKeyboard
            .enigo_settings()
            .expect("the virtual-keyboard backend goes through enigo");
        assert_eq!(wayland.x11_display.as_deref(), Some(""));
        assert_eq!(wayland.wayland_display, None);

        let xtest = KeySynthesis::X11Xtest
            .enigo_settings()
            .expect("the XTEST backend goes through enigo");
        assert_eq!(xtest.wayland_display.as_deref(), Some(""));
        assert_eq!(xtest.x11_display, None);

        assert!(
            KeySynthesis::Libei.enigo_settings().is_none(),
            "libei is driven directly, not through enigo"
        );
    }

    #[test]
    fn standard_mode_tries_direct_insertion_before_touching_the_clipboard() {
        let plan = insertion_plan(TextInsertionMode::Standard);
        let first_paste = plan
            .iter()
            .position(|(_, strategy)| *strategy == Strategy::ClipboardPaste);
        let last_direct = plan
            .iter()
            .rposition(|(_, strategy)| *strategy == Strategy::DirectText);
        if let (Some(first_paste), Some(last_direct)) = (first_paste, last_direct) {
            assert!(
                last_direct < first_paste,
                "standard mode must exhaust direct insertion first: {plan:?}"
            );
        }
    }

    #[test]
    fn reliable_paste_mode_tries_the_clipboard_before_direct_insertion() {
        let plan = insertion_plan(TextInsertionMode::ReliablePaste);
        let first_direct = plan
            .iter()
            .position(|(_, strategy)| *strategy == Strategy::DirectText);
        let last_paste = plan
            .iter()
            .rposition(|(_, strategy)| *strategy == Strategy::ClipboardPaste);
        if let (Some(first_direct), Some(last_paste)) = (first_direct, last_paste) {
            assert!(
                last_paste < first_direct,
                "reliable-paste mode must exhaust the clipboard first: {plan:?}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn libei_is_only_offered_for_clipboard_shortcuts() {
        // libei receives the compositor's keymap rather than supplying one, so
        // it cannot be trusted to type text outside the user's layout.
        for mode in [
            TextInsertionMode::Standard,
            TextInsertionMode::ReliablePaste,
        ] {
            for (backend, strategy) in insertion_plan(mode) {
                if backend == KeySynthesis::Libei {
                    assert_eq!(strategy, Strategy::ClipboardPaste);
                }
            }
        }
    }

    /// Prints the chain this desktop resolves to, and which backends actually
    /// connect. Run inside the session under test:
    ///   cargo test insertion_chain_probe -- --ignored --nocapture
    #[test]
    #[ignore = "reports on the live desktop session"]
    fn insertion_chain_probe() {
        let diagnostics = super::insertion_diagnostics();
        println!("chain: {:?}", diagnostics.chain);
        println!("can type text directly: {:?}", diagnostics.direct_capable);
        #[cfg(target_os = "linux")]
        println!("libei: {:?}", diagnostics.libei);
        #[cfg(target_os = "linux")]
        println!("modifier guard: {:?}", diagnostics.modifier_guard);
        for backend in super::synthesis_chain() {
            if backend.enigo_settings().is_none() {
                println!("  {backend}: driven directly, not through enigo");
                continue;
            }
            match backend.connect() {
                Ok(_) => println!("  {backend}: connected"),
                Err(error) => println!("  {backend}: unavailable ({error})"),
            }
        }
    }

    #[test]
    fn diagnostics_serialize_under_the_names_the_settings_screen_reads() {
        // The Settings notice reads these keys off the command result, and a
        // casing drift would silently render an empty panel rather than fail.
        let json = serde_json::to_value(super::insertion_diagnostics())
            .expect("the diagnostics must serialize");
        let object = json
            .as_object()
            .expect("diagnostics serialize as an object");
        assert!(object.contains_key("chain"), "missing chain: {json}");
        assert!(
            object.contains_key("directCapable"),
            "missing directCapable: {json}"
        );
        assert!(json["chain"].is_array());
        assert!(json["directCapable"].is_array());

        #[cfg(target_os = "linux")]
        {
            let libei = object.get("libei").expect("Linux reports a libei status");
            for key in ["connected", "attempted", "detail"] {
                assert!(libei.get(key).is_some(), "missing libei.{key}: {libei}");
            }
            let guard = object
                .get("modifierGuard")
                .expect("Linux reports modifier-guard availability");
            // The notice branches on `state` and reads `detail`, so both have to
            // survive serialization under those exact names.
            let state = guard.get("state").and_then(|state| state.as_str());
            assert!(
                matches!(state, Some("active") | Some("unavailable")),
                "unexpected modifierGuard state: {guard}"
            );
            assert!(guard.get("detail").is_some(), "missing detail: {guard}");
        }
    }

    #[test]
    fn diagnostics_never_claim_libei_can_type_text() {
        // The notice tells users libei drives a clipboard paste rather than
        // typing; that promise has to hold in the data behind it.
        let diagnostics = super::insertion_diagnostics();
        assert!(!diagnostics
            .direct_capable
            .contains(&"libei-portal".to_string()));
        for backend in &diagnostics.direct_capable {
            assert!(
                diagnostics.chain.contains(backend),
                "{backend} can type but is not in the chain"
            );
        }
    }

    #[test]
    fn every_planned_pair_is_unique() {
        for mode in [
            TextInsertionMode::Standard,
            TextInsertionMode::ReliablePaste,
        ] {
            let plan = insertion_plan(mode);
            let mut seen = plan.clone();
            seen.sort_by_key(|(backend, strategy)| (format!("{backend}"), format!("{strategy}")));
            seen.dedup();
            assert_eq!(seen.len(), plan.len(), "duplicate rung in plan: {plan:?}");
        }
    }
}
