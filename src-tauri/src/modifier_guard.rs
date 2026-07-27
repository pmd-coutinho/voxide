//! Holds synthesized text back while a chord modifier is still pressed.
//!
//! Dictation is normally triggered by a chord — `Mod+Space`, `Mod+Ctrl+D` — and
//! the text is synthesized the moment transcription finishes. If the user is
//! still resting on `Mod` or `Ctrl` at that point, the compositor reads each
//! synthesized keystroke as *modifier + key* and dispatches its own keybindings
//! instead of inserting characters. The visible result ranges from missing text
//! to windows being closed or workspaces switched by the dictation itself.
//!
//! The fix is to look at what is actually held before typing, and wait briefly.
//!
//! ## Why evdev, and what happens without it
//!
//! There is no portable way to ask a Wayland compositor which keys are down —
//! a client only learns about keys delivered to its own focused surface, and the
//! window being typed into belongs to somebody else. The kernel will answer
//! directly though: `EVIOCGKEY` returns a *snapshot* of the pressed-key bitmap
//! for an input device without consuming events, which works identically under
//! any compositor and under X11.
//!
//! That requires read access to `/dev/input/event*`, which on most
//! distributions means membership in the `input` group. When the devices are not
//! readable the guard reports [`Availability::Unavailable`] and becomes a no-op:
//! insertion proceeds exactly as it did before, with no added latency. It does
//! not substitute a blind delay, because paying a fixed cost on every dictation
//! to hedge against a race that may not affect this desktop is a worse trade
//! than leaving the behaviour unchanged and saying so in the diagnostics.

use std::time::{Duration, Instant};

use evdev::{Device, KeyCode};

use crate::debug_log;

/// Modifiers whose presence means "do not type yet". Latching keys are excluded
/// deliberately: Caps Lock and Num Lock can be *on* indefinitely without a
/// finger on them, so waiting for them would hang every dictation.
const CHORD_MODIFIERS: &[KeyCode] = &[
    KeyCode::KEY_LEFTCTRL,
    KeyCode::KEY_RIGHTCTRL,
    KeyCode::KEY_LEFTALT,
    KeyCode::KEY_RIGHTALT,
    KeyCode::KEY_LEFTSHIFT,
    KeyCode::KEY_RIGHTSHIFT,
    KeyCode::KEY_LEFTMETA,
    KeyCode::KEY_RIGHTMETA,
];

/// Longest the guard will hold text back. A physically stuck key, or a modifier
/// the compositor is holding for its own reasons, must not strand a dictation:
/// past this point the text is delivered anyway, which is the pre-guard
/// behaviour and strictly better than losing it.
const MAX_WAIT: Duration = Duration::from_millis(600);

/// Snapshots are cheap ioctls, so polling can be fine-grained enough that the
/// wait ends within a frame of the user lifting the key.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Whether the kernel will tell us what is held.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase", tag = "state", content = "detail")]
pub enum Availability {
    /// At least one keyboard device answered an `EVIOCGKEY` probe.
    Active { keyboards: usize },
    /// No readable keyboard device. Carries the reason so Settings can tell the
    /// user what to do about it.
    Unavailable { reason: String },
}

/// What a single wait actually did, for the diagnostic log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The guard could not run; insertion proceeded immediately.
    NotAvailable,
    /// Nothing was held, so nothing was delayed.
    NothingHeld,
    /// Modifiers were held and then released.
    Released { waited: Duration },
    /// Modifiers were still held at [`MAX_WAIT`]; text was delivered anyway.
    TimedOut { held: Vec<&'static str> },
}

impl Outcome {
    /// How the outcome should read in the diagnostic log, or `None` when there
    /// is nothing worth recording. The common case — nothing held — is silent so
    /// the log is not one line longer per dictation.
    fn log_line(&self) -> Option<String> {
        match self {
            Self::NotAvailable | Self::NothingHeld => None,
            Self::Released { waited } => Some(format!(
                "held modifiers cleared after {} ms; typing now",
                waited.as_millis()
            )),
            Self::TimedOut { held } => Some(format!(
                "modifiers still held after {} ms ({}); typing anyway",
                MAX_WAIT.as_millis(),
                held.join("+")
            )),
        }
    }
}

/// Reports whether the guard can run, without waiting for anything.
pub fn availability() -> Availability {
    match open_keyboards() {
        Ok(keyboards) if !keyboards.is_empty() => Availability::Active {
            keyboards: keyboards.len(),
        },
        Ok(_) => Availability::Unavailable {
            reason: "No readable keyboard found under /dev/input".into(),
        },
        Err(reason) => Availability::Unavailable { reason },
    }
}

/// Blocks until no chord modifier is held, [`MAX_WAIT`] elapses, or the kernel
/// cannot be asked.
///
/// Devices are re-enumerated per call rather than cached, so a keyboard plugged
/// in mid-session is picked up without watching for hotplug. The cost is a
/// directory scan and a handful of `open` calls, which is negligible next to the
/// transcription that just finished.
pub fn wait_for_release() -> Outcome {
    let outcome = match open_keyboards() {
        Ok(keyboards) if keyboards.is_empty() => Outcome::NotAvailable,
        Err(_) => Outcome::NotAvailable,
        Ok(keyboards) => {
            let started = Instant::now();
            loop {
                let held = held_modifiers(&keyboards);
                if held.is_empty() {
                    break if started.elapsed() < POLL_INTERVAL {
                        Outcome::NothingHeld
                    } else {
                        Outcome::Released {
                            waited: started.elapsed(),
                        }
                    };
                }
                if started.elapsed() >= MAX_WAIT {
                    break Outcome::TimedOut { held };
                }
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    };
    if let Some(line) = outcome.log_line() {
        debug_log::append(&line);
    }
    outcome
}

/// Names of the chord modifiers currently pressed on any keyboard.
///
/// A key is reported held if *any* device says so: with several keyboards
/// attached, only one of them has the finger on it.
fn held_modifiers(keyboards: &[Device]) -> Vec<&'static str> {
    let mut held = Vec::new();
    for device in keyboards {
        // A device can disappear between enumeration and probing (unplugged
        // mid-wait); treat that as "nothing held on it" rather than an error.
        let Ok(pressed) = device.get_key_state() else {
            continue;
        };
        for key in CHORD_MODIFIERS {
            if pressed.contains(*key) {
                let name = modifier_name(*key);
                if !held.contains(&name) {
                    held.push(name);
                }
            }
        }
    }
    held
}

fn modifier_name(key: KeyCode) -> &'static str {
    match key {
        KeyCode::KEY_LEFTCTRL | KeyCode::KEY_RIGHTCTRL => "Ctrl",
        KeyCode::KEY_LEFTALT | KeyCode::KEY_RIGHTALT => "Alt",
        KeyCode::KEY_LEFTSHIFT | KeyCode::KEY_RIGHTSHIFT => "Shift",
        KeyCode::KEY_LEFTMETA | KeyCode::KEY_RIGHTMETA => "Super",
        _ => "modifier",
    }
}

/// Opens every readable input device that reports the modifier keys we care
/// about. `Err` means `/dev/input` itself could not be listed; an empty `Ok`
/// means it could, but nothing in it was both readable and a keyboard.
fn open_keyboards() -> Result<Vec<Device>, String> {
    let entries = std::fs::read_dir("/dev/input")
        .map_err(|error| format!("/dev/input could not be listed: {error}"))?;
    let mut keyboards = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("event"))
        {
            continue;
        }
        // Unreadable devices are the normal case for a user outside the `input`
        // group, so this is a skip and not a failure.
        let Ok(device) = Device::open(&path) else {
            continue;
        };
        let reports_modifiers = device
            .supported_keys()
            .is_some_and(|keys| CHORD_MODIFIERS.iter().any(|key| keys.contains(*key)));
        if reports_modifiers {
            keyboards.push(device);
        }
    }
    Ok(keyboards)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latching_keys_are_not_treated_as_chord_modifiers() {
        // Caps Lock can be on with no finger on it; waiting for it would hang
        // every dictation instead of preventing a misdirected keystroke.
        assert!(!CHORD_MODIFIERS.contains(&KeyCode::KEY_CAPSLOCK));
        assert!(!CHORD_MODIFIERS.contains(&KeyCode::KEY_NUMLOCK));
        assert!(!CHORD_MODIFIERS.contains(&KeyCode::KEY_SCROLLLOCK));
    }

    #[test]
    fn both_sides_of_every_modifier_are_covered() {
        // A chord held with the right-hand Ctrl must count exactly as much as
        // the left-hand one.
        for (left, right) in [
            (KeyCode::KEY_LEFTCTRL, KeyCode::KEY_RIGHTCTRL),
            (KeyCode::KEY_LEFTALT, KeyCode::KEY_RIGHTALT),
            (KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_RIGHTSHIFT),
            (KeyCode::KEY_LEFTMETA, KeyCode::KEY_RIGHTMETA),
        ] {
            assert!(CHORD_MODIFIERS.contains(&left));
            assert!(CHORD_MODIFIERS.contains(&right));
            assert_eq!(modifier_name(left), modifier_name(right));
        }
    }

    #[test]
    fn the_wait_is_bounded_well_below_a_users_patience() {
        // The guard sits between "stop speaking" and "text appears", so its
        // worst case is felt directly.
        assert!(MAX_WAIT <= Duration::from_millis(600));
        assert!(POLL_INTERVAL < MAX_WAIT);
    }

    #[test]
    fn quiet_outcomes_do_not_write_a_line_per_dictation() {
        assert!(Outcome::NotAvailable.log_line().is_none());
        assert!(Outcome::NothingHeld.log_line().is_none());
        assert!(Outcome::Released {
            waited: Duration::from_millis(40)
        }
        .log_line()
        .is_some());
        assert!(Outcome::TimedOut {
            held: vec!["Super"]
        }
        .log_line()
        .is_some());
    }

    #[test]
    fn a_timeout_names_the_keys_that_blocked_it() {
        let line = Outcome::TimedOut {
            held: vec!["Super", "Ctrl"],
        }
        .log_line()
        .expect("a timeout is worth logging");
        assert!(line.contains("Super+Ctrl"), "{line}");
        assert!(line.contains("typing anyway"), "{line}");
    }

    #[test]
    fn waiting_never_blocks_when_the_kernel_cannot_be_asked() {
        // On a machine outside the `input` group this is the real code path, and
        // it must add no latency at all.
        let started = Instant::now();
        let outcome = wait_for_release();
        if outcome == Outcome::NotAvailable {
            assert!(
                started.elapsed() < MAX_WAIT,
                "an unavailable guard must not wait"
            );
        }
    }

    #[test]
    fn availability_explains_itself_either_way() {
        match availability() {
            Availability::Active { keyboards } => assert!(keyboards > 0),
            Availability::Unavailable { reason } => assert!(!reason.trim().is_empty()),
        }
    }
}
