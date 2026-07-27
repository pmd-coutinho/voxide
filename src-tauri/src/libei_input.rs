//! Synthetic key events through libei and the RemoteDesktop portal.
//!
//! This is the last rung of the text-insertion chain in [`crate::typing`], and
//! on some desktops it is the only one that exists. GNOME and KDE implement
//! neither `zwp_virtual_keyboard_v1` (which `wtype` and enigo's Wayland backend
//! need) nor, in a pure-Wayland session with Xwayland disabled, XTEST. libei is
//! the protocol they do offer, reached by asking
//! `org.freedesktop.portal.RemoteDesktop` for a keyboard device and receiving a
//! socket to speak EI over.
//!
//! ## Deliberate limitation: shortcuts only
//!
//! libei hands the client the *compositor's* keymap rather than accepting one,
//! the opposite of the virtual-keyboard protocol. Arbitrary Unicode therefore
//! cannot be typed — only keysyms the user's active layout already contains.
//! Rather than silently mangle dictation on a Dvorak or Cyrillic layout, this
//! module exposes only the clipboard shortcuts (Ctrl+C, Ctrl+V), which every
//! layout has. The text itself travels through the clipboard, which carries any
//! Unicode faithfully.
//!
//! ## Threading
//!
//! Everything runs on one dedicated thread with its own current-thread Tokio
//! runtime. Two reasons: the portal handshake is async while the EI handshake is
//! blocking, and the EI objects are neither `Send` nor cheap to rebuild. Callers
//! talk to the thread over a channel, so a stuck compositor cannot wedge the
//! dictation pipeline beyond the reply timeout.
//!
//! ## Permission
//!
//! `RemoteDesktop` shows a system dialog the first time it is started. The
//! session is requested with [`PersistMode::ExplicitlyRevoked`] and the returned
//! restore token is cached on disk, so approval is asked for once rather than
//! once per launch. Nothing here is attempted until the cheaper rungs of the
//! chain have already failed.

use std::{
    fs::{self, File},
    os::unix::{fs::FileExt, net::UnixStream},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use ashpd::desktop::{
    remote_desktop::{DeviceType, RemoteDesktop},
    PersistMode, Session,
};
use enumflags2::BitFlags;
use reis::{
    ei,
    event::{DeviceCapability, EiEvent, EiEventConverter},
    handshake, PendingRequestResult,
};
use xkbcommon::xkb;

use crate::debug_log;

/// How long a caller waits for the worker thread to answer.
///
/// Deliberately short even for the first call, which may be sitting behind a
/// portal permission dialog. Blocking a dictation until the user notices and
/// dismisses that dialog would be worse than failing it: the establishment
/// continues on the worker thread regardless, so the *next* dictation finds a
/// ready session. See [`Command::ClipboardShortcut::deadline`] for why the
/// abandoned request cannot then paste into whatever is focused later.
const REPLY_TIMEOUT: Duration = Duration::from_secs(5);
/// How long to wait for the compositor to advertise a seat, keyboard device and
/// keymap after the EI handshake completes.
const DEVICE_TIMEOUT: Duration = Duration::from_secs(5);

/// Name Voxide identifies itself by in the compositor's input-emulation UI.
const CLIENT_NAME: &str = "Voxide";

/// Set once a failure proves libei will not work in this session, so the chain
/// stops offering the rung instead of re-prompting on every dictation.
static UNAVAILABLE: AtomicBool = AtomicBool::new(false);

static WORKER: OnceLock<mpsc::Sender<Command>> = OnceLock::new();

enum Command {
    /// Press the platform clipboard modifier, click `character`, release.
    ClipboardShortcut {
        character: char,
        /// When the caller stops caring. Establishing a session can block the
        /// worker for as long as a permission dialog stays on screen, and by
        /// then the dictation that queued this request has already reported a
        /// failure and moved on — its text is no longer on the clipboard.
        /// Synthesizing the paste at that point would insert whatever the
        /// clipboard now holds into whatever now has focus, so a request past
        /// its deadline is answered but never sent.
        deadline: Instant,
        reply: mpsc::Sender<Result<(), String>>,
    },
    /// Report what the module knows without creating a session.
    Status { reply: mpsc::Sender<Status> },
}

/// What the Settings screen shows about this backend.
#[derive(Debug, Clone, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    /// Whether a portal session has been established and a keyboard bound.
    pub connected: bool,
    /// Whether a session has ever been attempted in this run.
    pub attempted: bool,
    /// Human-readable detail: the active seat, or why the last attempt failed.
    pub detail: String,
}

/// Whether it is worth offering libei as a rung of the insertion chain.
///
/// Cheap and side-effect free by design — it must not touch D-Bus, because the
/// chain is rebuilt for every dictation and consulting the portal here would
/// cost a round trip even when the earlier rungs are about to succeed.
pub fn is_possible() -> bool {
    !UNAVAILABLE.load(Ordering::Relaxed) && crate::portal_hotkeys::is_wayland_session()
}

/// Sends the platform clipboard shortcut for `character` (`'c'` or `'v'`).
pub fn send_clipboard_shortcut(character: char) -> Result<(), String> {
    let (reply, replies) = mpsc::channel();
    worker()
        .send(Command::ClipboardShortcut {
            character,
            deadline: Instant::now() + REPLY_TIMEOUT,
            reply,
        })
        .map_err(|_| "The libei input thread is no longer running".to_string())?;
    match replies.recv_timeout(REPLY_TIMEOUT) {
        Ok(outcome) => outcome,
        Err(_) => Err(format!(
            "libei did not respond within {} seconds; if a permission dialog is waiting, approving it will make the next dictation work",
            REPLY_TIMEOUT.as_secs()
        )),
    }
}

/// Reports the current state for diagnostics. Never establishes a session.
pub fn status() -> Status {
    if !crate::portal_hotkeys::is_wayland_session() {
        return Status {
            detail: "Not a Wayland session; libei is not used".into(),
            ..Status::default()
        };
    }
    let Some(worker) = WORKER.get() else {
        return Status {
            detail: "Not needed yet — a faster input path is working".into(),
            ..Status::default()
        };
    };
    let (reply, replies) = mpsc::channel();
    if worker.send(Command::Status { reply }).is_err() {
        return Status {
            attempted: true,
            detail: "The libei input thread is no longer running".into(),
            ..Status::default()
        };
    }
    replies.recv_timeout(REPLY_TIMEOUT).unwrap_or(Status {
        attempted: true,
        detail: "The libei input thread did not respond".into(),
        ..Status::default()
    })
}

fn worker() -> &'static mpsc::Sender<Command> {
    WORKER.get_or_init(|| {
        let (commands, inbox) = mpsc::channel();
        thread::Builder::new()
            .name("voxide-libei".into())
            .spawn(move || run_worker(&inbox))
            .expect("the libei input thread must be spawnable");
        commands
    })
}

fn run_worker(inbox: &mpsc::Receiver<Command>) {
    let mut keyboard: Option<Emulator> = None;
    let mut attempted = false;
    let mut detail = String::new();

    while let Ok(command) = inbox.recv() {
        match command {
            Command::Status { reply } => {
                let _ = reply.send(Status {
                    connected: keyboard.is_some(),
                    attempted,
                    detail: if detail.is_empty() {
                        "Idle".to_string()
                    } else {
                        detail.clone()
                    },
                });
            }
            Command::ClipboardShortcut {
                character,
                deadline,
                reply,
            } => {
                attempted = true;
                // A session that has gone away (compositor restart, permission
                // revoked) must not poison every later attempt, so a failed
                // send drops the session and the next call rebuilds it.
                if keyboard.is_none() {
                    match Emulator::connect() {
                        Ok(emulator) => {
                            detail = emulator.describe();
                            debug_log::append(&format!("libei session established ({detail})"));
                            keyboard = Some(emulator);
                        }
                        Err(error) => {
                            detail = error.clone();
                            // A refused or unimplemented portal will not start
                            // working later in this session; stop offering the
                            // rung so dictation does not pay for it every time.
                            UNAVAILABLE.store(true, Ordering::Relaxed);
                            debug_log::append(&format!("libei unavailable: {error}"));
                            let _ = reply.send(Err(error));
                            continue;
                        }
                    }
                }
                // Establishing the session above can outlast the caller. Its
                // dictation has already failed over and the clipboard has moved
                // on, so pasting now would insert unrelated content into
                // whatever gained focus meanwhile.
                if Instant::now() > deadline {
                    debug_log::append(
                        "libei became ready after the request expired; no keys were sent",
                    );
                    let _ = reply.send(Err(
                        "libei is ready now but this dictation had already finished".into(),
                    ));
                    continue;
                }
                let outcome = keyboard
                    .as_mut()
                    .expect("a session was just established")
                    .send_clipboard_shortcut(character);
                if let Err(error) = &outcome {
                    detail = error.clone();
                    keyboard = None;
                }
                let _ = reply.send(outcome);
            }
        }
    }
}

/// A live EI keyboard device plus the portal session keeping it alive.
struct Emulator {
    /// Held only so the portal session is not closed. Dropping it revokes the
    /// EI connection, so it must outlive the device.
    _session: Session<'static, RemoteDesktop<'static>>,
    context: ei::Context,
    converter: EiEventConverter,
    device: ei::Device,
    keyboard: ei::Keyboard,
    keymap: xkb::Keymap,
    seat_name: String,
    sequence: u32,
    started_at: Instant,
}

impl Emulator {
    fn connect() -> Result<Self, String> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| format!("Could not start the libei runtime: {error}"))?;
        let (session, socket) = runtime.block_on(open_portal_socket())?;
        Self::from_socket(session, socket)
    }

    fn from_socket(
        session: Session<'static, RemoteDesktop<'static>>,
        socket: UnixStream,
    ) -> Result<Self, String> {
        let context = ei::Context::new(socket)
            .map_err(|error| format!("Could not wrap the libei socket: {error}"))?;
        let response = handshake::ei_handshake_blocking(
            &context,
            CLIENT_NAME,
            ei::handshake::ContextType::Sender,
        )
        .map_err(|error| format!("The libei handshake failed: {error}"))?;
        context
            .flush()
            .map_err(|error| format!("Could not flush the libei handshake: {error}"))?;

        let mut converter = EiEventConverter::new(&context, response);
        let bound = bind_keyboard_seat(&context, &mut converter)?;
        let (device, keyboard, keymap, seat_name) = bound;

        Ok(Self {
            _session: session,
            context,
            converter,
            device,
            keyboard,
            keymap,
            seat_name,
            sequence: 0,
            started_at: Instant::now(),
        })
    }

    fn describe(&self) -> String {
        format!("keyboard bound on seat “{}”", self.seat_name)
    }

    /// Presses the clipboard modifier, clicks `character`, releases.
    fn send_clipboard_shortcut(&mut self, character: char) -> Result<(), String> {
        // Control on every platform this module compiles for; macOS uses Meta
        // but never reaches libei.
        let modifier = self.evdev_keycode("Control_L")?;
        let letter = self.evdev_keycode(&character.to_string())?;

        let serial = self.converter.connection().serial();
        self.sequence = self.sequence.wrapping_add(1);
        self.device.start_emulating(serial, self.sequence);

        for (keycode, state) in [
            (modifier, ei::keyboard::KeyState::Press),
            (letter, ei::keyboard::KeyState::Press),
            (letter, ei::keyboard::KeyState::Released),
            (modifier, ei::keyboard::KeyState::Released),
        ] {
            self.keyboard.key(keycode, state);
            self.device
                .frame(serial, self.started_at.elapsed().as_micros() as u64);
        }

        self.device.stop_emulating(serial);
        self.context
            .flush()
            .map_err(|error| format!("Could not flush the libei key events: {error}"))?;
        // Surface a compositor-side disconnect now rather than on the next
        // dictation, so the caller can fall through to another backend.
        self.drain_events()
    }

    /// Resolves a keysym name to the evdev keycode libei expects.
    ///
    /// libei speaks evdev codes while xkb keycodes are offset by 8, hence the
    /// subtraction. Only the unshifted level is considered: a shortcut is only
    /// correct if the letter needs no extra modifier, and reaching for a shifted
    /// level would send Ctrl+Shift+something instead.
    fn evdev_keycode(&self, keysym_name: &str) -> Result<u32, String> {
        let target = xkb::keysym_from_name(keysym_name, xkb::KEYSYM_NO_FLAGS);
        if target == xkb::Keysym::from(0u32) {
            return Err(format!("“{keysym_name}” is not a known keysym"));
        }
        let min = self.keymap.min_keycode().raw();
        let max = self.keymap.max_keycode().raw();
        for raw in min..=max {
            let keycode = xkb::Keycode::new(raw);
            let layouts = self.keymap.num_layouts_for_key(keycode);
            for layout in 0..layouts {
                if self.keymap.num_levels_for_key(keycode, layout) == 0 {
                    continue;
                }
                if self
                    .keymap
                    .key_get_syms_by_level(keycode, layout, 0)
                    .contains(&target)
                {
                    return raw.checked_sub(8).ok_or_else(|| {
                        format!("keysym “{keysym_name}” maps to a reserved keycode")
                    });
                }
            }
        }
        Err(format!(
            "The active keyboard layout has no unshifted key for “{keysym_name}”"
        ))
    }

    /// Reads whatever the compositor has sent and fails if it disconnected us.
    fn drain_events(&mut self) -> Result<(), String> {
        pump(&self.context, &mut self.converter)?;
        while let Some(event) = self.converter.next_event() {
            match event {
                EiEvent::Disconnected(_) => {
                    return Err("The compositor closed the libei connection".into())
                }
                EiEvent::DeviceRemoved(_) => {
                    return Err("The compositor removed the libei keyboard".into())
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Creates a RemoteDesktop session limited to keyboard access and returns the
/// EI socket it hands out.
async fn open_portal_socket(
) -> Result<(Session<'static, RemoteDesktop<'static>>, UnixStream), String> {
    // Annotating the proxy `'static` makes `create_session` hand back a
    // `'static` session, which the emulator can then own outright.
    let proxy: RemoteDesktop<'static> = RemoteDesktop::new()
        .await
        .map_err(|error| format!("The RemoteDesktop portal is unavailable: {error}"))?;
    let session = proxy
        .create_session()
        .await
        .map_err(|error| format!("Could not create a RemoteDesktop session: {error}"))?;
    proxy
        .select_devices(
            &session,
            BitFlags::from(DeviceType::Keyboard),
            restore_token().as_deref(),
            PersistMode::ExplicitlyRevoked,
        )
        .await
        .map_err(|error| format!("Could not request keyboard access: {error}"))?;
    let started = proxy
        .start(&session, None)
        .await
        .map_err(|error| format!("Keyboard access was not granted: {error}"))?
        .response()
        .map_err(|error| format!("Keyboard access was declined: {error}"))?;
    if !started.devices().contains(DeviceType::Keyboard) {
        return Err("The portal granted a session without keyboard access".into());
    }
    if let Some(token) = started.restore_token() {
        store_restore_token(token);
    }
    let descriptor = proxy
        .connect_to_eis(&session)
        .await
        .map_err(|error| format!("The portal did not provide a libei socket: {error}"))?;
    // The session carries its own D-Bus proxy, so the RemoteDesktop handle is
    // free to drop here; only the session has to outlive the EI connection.
    Ok((session, UnixStream::from(descriptor)))
}

/// Binds the first seat that offers a keyboard, then waits for the device and
/// its keymap to arrive.
fn bind_keyboard_seat(
    context: &ei::Context,
    converter: &mut EiEventConverter,
) -> Result<(ei::Device, ei::Keyboard, xkb::Keymap, String), String> {
    let deadline = Instant::now() + DEVICE_TIMEOUT;
    let mut seat_name = String::from("default");
    let mut bound_any_seat = false;

    while Instant::now() < deadline {
        pump(context, converter)?;
        while let Some(event) = converter.next_event() {
            match event {
                EiEvent::SeatAdded(added) => {
                    // reis does not expose a seat's advertised capabilities, so
                    // every seat is asked for a keyboard. Requesting one that a
                    // seat does not offer is a no-op, and the device events
                    // below are what actually confirm we got one.
                    let seat = added.seat;
                    seat_name = seat.name().unwrap_or("default").to_string();
                    seat.bind_capabilities(BitFlags::from(DeviceCapability::Keyboard));
                    bound_any_seat = true;
                    context.flush().map_err(|error| {
                        format!("Could not flush the libei seat binding: {error}")
                    })?;
                }
                EiEvent::DeviceAdded(added) => {
                    if let Some(bound) = keyboard_from_device(&added.device) {
                        return Ok((
                            added.device.device().clone(),
                            bound.0,
                            bound.1,
                            seat_name.clone(),
                        ));
                    }
                }
                EiEvent::DeviceResumed(resumed) => {
                    if let Some(bound) = keyboard_from_device(&resumed.device) {
                        return Ok((
                            resumed.device.device().clone(),
                            bound.0,
                            bound.1,
                            seat_name.clone(),
                        ));
                    }
                }
                EiEvent::Disconnected(_) => {
                    return Err("The compositor closed the libei connection".into())
                }
                _ => {}
            }
        }
    }
    Err(if bound_any_seat {
        "The compositor never provided a libei keyboard device".to_string()
    } else {
        "No libei seat offers keyboard emulation".to_string()
    })
}

/// Extracts the keyboard interface and compiled keymap from a device, if it has
/// both. A keyboard without a keymap cannot be used: keysyms could not be
/// resolved to keycodes.
fn keyboard_from_device(device: &reis::event::Device) -> Option<(ei::Keyboard, xkb::Keymap)> {
    if !device.has_capability(DeviceCapability::Keyboard) {
        return None;
    }
    let keyboard = device.interface::<ei::Keyboard>()?;
    let keymap = compile_keymap(device.keymap()?)?;
    Some((keyboard, keymap))
}

fn compile_keymap(keymap: &reis::event::Keymap) -> Option<xkb::Keymap> {
    if keymap.type_ != ei::keyboard::KeymapType::Xkb {
        debug_log::append("libei offered a non-XKB keymap");
        return None;
    }
    // The keymap arrives as a (usually sealed) memfd. `read_at` keeps the
    // borrowed descriptor's offset untouched, which matters because reis hands
    // out a shared reference to it.
    let mut buffer = vec![0u8; keymap.size as usize];
    let file = File::from(keymap.fd.try_clone().ok()?);
    file.read_exact_at(&mut buffer, 0).ok()?;
    // Trailing NUL bytes are conventional in the XKB memfd and would otherwise
    // make xkbcommon reject the text.
    let text = String::from_utf8(buffer)
        .ok()?
        .trim_end_matches('\0')
        .to_string();
    let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
    xkb::Keymap::new_from_string(
        &context,
        text,
        xkb::KEYMAP_FORMAT_TEXT_V1,
        xkb::KEYMAP_COMPILE_NO_FLAGS,
    )
}

/// Reads whatever is available on the non-blocking EI socket into `converter`.
///
/// A short sleep rather than a poll: the only waits here are the seat/device
/// handshake and the post-send disconnect check, both bounded by their callers.
fn pump(context: &ei::Context, converter: &mut EiEventConverter) -> Result<(), String> {
    match context.read() {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            thread::sleep(Duration::from_millis(2));
        }
        Err(error) => return Err(format!("The libei socket failed: {error}")),
    }
    while let Some(pending) = context.pending_event() {
        match pending {
            PendingRequestResult::Request(event) => converter
                .handle_event(event)
                .map_err(|error| format!("A libei event could not be handled: {error}"))?,
            PendingRequestResult::ParseError(_) => {
                return Err("A libei event could not be parsed".into())
            }
            PendingRequestResult::InvalidObject(_) => {}
        }
    }
    Ok(())
}

fn restore_token_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("dev", "pmdcoutinho", "Voxide")
        .map(|directories| directories.data_local_dir().join("libei-restore-token"))
}

/// The portal token that lets a later session skip the permission dialog. Not a
/// credential for any external service, so it lives beside the other local state
/// rather than in the OS keyring.
fn restore_token() -> Option<String> {
    let token = fs::read_to_string(restore_token_path()?).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

fn store_restore_token(token: &str) {
    let Some(path) = restore_token_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    // Best effort: a token that cannot be cached only costs another dialog.
    let _ = fs::write(&path, token);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_reports_plainly_outside_wayland() {
        // The X11 and macOS paths must never claim a libei session exists.
        if !crate::portal_hotkeys::is_wayland_session() {
            let status = status();
            assert!(!status.connected);
            assert!(!status.attempted);
            assert!(status.detail.contains("Wayland"));
        }
    }

    #[test]
    fn libei_is_not_offered_outside_wayland() {
        if !crate::portal_hotkeys::is_wayland_session() {
            assert!(!is_possible());
        }
    }

    #[test]
    fn an_unavailable_verdict_retires_the_rung() {
        // Recorded once so a desktop without the portal does not pay a failed
        // D-Bus round trip on every dictation.
        let previous = UNAVAILABLE.swap(true, Ordering::Relaxed);
        assert!(!is_possible());
        UNAVAILABLE.store(previous, Ordering::Relaxed);
    }

    /// Exercises the real portal + EI handshake on the current desktop and
    /// prints what happened, without asserting: the correct outcome differs per
    /// compositor. Run it inside the session under test:
    ///   cargo test libei_session -- --ignored --nocapture
    ///
    /// Expect a permission dialog on GNOME and KDE. On wlroots compositors
    /// (Niri, Sway, Hyprland) the RemoteDesktop portal is usually absent, and a
    /// clear failure here is the correct result — the virtual-keyboard rung
    /// handles those desktops and libei is never reached in practice.
    #[test]
    #[ignore = "opens a real RemoteDesktop portal session and may prompt"]
    fn libei_session_probe() {
        println!(
            "wayland session: {}",
            crate::portal_hotkeys::is_wayland_session()
        );
        println!("offered as a rung: {}", is_possible());
        match Emulator::connect() {
            Ok(mut emulator) => {
                println!("connected: {}", emulator.describe());
                for name in ["Control_L", "v", "c"] {
                    match emulator.evdev_keycode(name) {
                        Ok(code) => println!("  keysym {name} -> evdev {code}"),
                        Err(error) => println!("  keysym {name} -> unavailable: {error}"),
                    }
                }
                // Not sent: this would paste the current clipboard into whatever
                // window happens to be focused in the live session.
                println!("(skipping the actual Ctrl+V so the probe cannot type into your desktop)");
            }
            Err(error) => println!("could not connect: {error}"),
        }
    }

    #[test]
    fn a_missing_restore_token_is_absent_rather_than_empty() {
        // An empty file must not be offered to the portal as a token, which it
        // would reject and then re-prompt for.
        assert!(restore_token().is_none_or(|token| !token.is_empty()));
    }
}
