//! Small, platform-scoped permission probes used by the portable UI.
//!
//! Accessibility trust is the one permission that macOS exposes as a stable
//! process-level status. Windows and Linux do not have an equivalent desktop
//! API that applies to every input backend, so they intentionally report an
//! unavailable status instead of guessing.

#[cfg(target_os = "macos")]
#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> u8;
}

pub fn accessibility_trusted() -> Option<bool> {
    #[cfg(target_os = "macos")]
    {
        // AXIsProcessTrusted returns the C `Boolean` type (an unsigned byte).
        return Some(unsafe { AXIsProcessTrusted() != 0 });
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

pub fn accessibility_guidance() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "Enable Voxide in System Settings → Privacy & Security → Accessibility to improve direct text insertion and active-app targeting."
    }
    #[cfg(target_os = "windows")]
    {
        "Windows has no universal accessibility-trust status. If direct insertion is blocked, allow desktop input automation for Voxide or use Clipboard Paste mode."
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        "Linux input permissions depend on the desktop session. If direct insertion is blocked, grant the compositor/input utility access or use Clipboard Paste mode."
    }
}

pub fn open_accessibility_settings() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
            .spawn()
            .map_err(|error| format!("Could not open macOS Accessibility settings: {error}"))?;
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", "ms-settings:easeofaccess-keyboard"])
            .spawn()
            .map_err(|error| {
                format!("Could not open Windows keyboard accessibility settings: {error}")
            })?;
        return Ok(());
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        Err("Open your desktop environment's accessibility or input-permissions settings, then retry direct insertion.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guidance_is_never_empty() {
        assert!(!accessibility_guidance().is_empty());
    }
}
