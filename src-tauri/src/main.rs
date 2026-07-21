fn main() {
    // `voxide --trigger <action>` forwards a hotkey action to the running
    // instance and exits. It lets Wayland compositors without a working XDG
    // GlobalShortcuts backend (for example Niri or Sway) bind Voxide
    // shortcuts in their own configuration.
    let mut arguments = std::env::args().skip(1);
    if arguments.next().as_deref() == Some("--trigger") {
        let action = arguments.next().unwrap_or_default();
        #[cfg(unix)]
        match voxide_lib::trigger::send(&action) {
            Ok(()) => std::process::exit(0),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        #[cfg(not(unix))]
        {
            eprintln!("--trigger is only supported on Linux and macOS; use the configured global shortcuts instead.");
            std::process::exit(1);
        }
    }
    voxide_lib::run();
}
