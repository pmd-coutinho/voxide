use std::{
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arboard::{Clipboard, ImageData};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

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

/// Inserts a nonempty transcription using the selected reliability policy.
/// Standard mode leaves the clipboard untouched whenever direct insertion
/// works; reliable-paste mode deliberately takes the clipboard path first.
pub fn type_into_active_application(text: &str, mode: TextInsertionMode) -> Result<(), String> {
    if text.trim().is_empty() {
        return Ok(());
    }
    if mode == TextInsertionMode::ReliablePaste {
        return paste_text_into_active_application(text).map_err(|paste_error| {
            with_compositor_hint(format!(
                "Could not insert dictation through the reliable clipboard-paste path ({paste_error})"
            ))
        });
    }
    let direct_error = match Enigo::new(&Settings::default()) {
        Ok(mut input) => match input.text(text) {
            Ok(()) => return Ok(()),
            Err(error) => error.to_string(),
        },
        Err(error) => error.to_string(),
    };
    paste_text_into_active_application(text).map_err(|paste_error| {
        with_compositor_hint(format!(
            "Could not insert dictation directly ({direct_error}) or paste it through the clipboard ({paste_error})"
        ))
    })
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

fn paste_text_into_active_application(text: &str) -> Result<(), String> {
    let mut clipboard = Clipboard::new()
        .map_err(|error| format!("Could not access the system clipboard: {error}"))?;
    let previous_contents = clipboard_snapshot(&mut clipboard)?;
    clipboard
        .set_text(text)
        .map_err(|error| format!("Could not prepare clipboard paste text: {error}"))?;
    let pasted = send_paste_shortcut();
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

    let copied = send_copy_shortcut()
        .and_then(|()| wait_for_copied_text(&mut clipboard, &sentinel))
        .map_err(with_compositor_hint);
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

fn send_copy_shortcut() -> Result<(), String> {
    let mut input = Enigo::new(&Settings::default())
        .map_err(|error| format!("Could not connect to the desktop input service: {error}"))?;
    #[cfg(target_os = "macos")]
    let modifier = Key::Meta;
    #[cfg(not(target_os = "macos"))]
    let modifier = Key::Control;

    input
        .key(modifier, Direction::Press)
        .and_then(|()| input.key(Key::Unicode('c'), Direction::Click))
        .and_then(|()| input.key(modifier, Direction::Release))
        .map_err(|error| format!("Could not copy the selected text: {error}"))
}

fn send_paste_shortcut() -> Result<(), String> {
    let mut input = Enigo::new(&Settings::default())
        .map_err(|error| format!("Could not connect to the desktop input service: {error}"))?;
    #[cfg(target_os = "macos")]
    let modifier = Key::Meta;
    #[cfg(not(target_os = "macos"))]
    let modifier = Key::Control;

    input
        .key(modifier, Direction::Press)
        .and_then(|()| input.key(Key::Unicode('v'), Direction::Click))
        .and_then(|()| input.key(modifier, Direction::Release))
        .map_err(|error| format!("Could not paste through the active application: {error}"))
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
    use super::TextInsertionMode;

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
}
