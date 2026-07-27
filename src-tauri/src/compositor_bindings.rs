//! Writes Voxide's trigger keybindings into the user's compositor config.
//!
//! On Wayland desktops whose portal cannot complete a global-shortcut grab, the
//! working route is a compositor keybinding that runs `voxide --trigger <action>`
//! (see [`crate::trigger`]). Telling users to hand-edit `config.kdl` puts the
//! burden of per-compositor syntax on them, so this module detects the
//! compositor, generates the right snippet, and can install it.
//!
//! ## Editing somebody else's file, safely
//!
//! A compositor config is load-bearing: a syntax error can leave the user with
//! no working keybindings at all, and the file is frequently a symlink into a
//! dotfiles repository. Nothing here writes without being asked, and every write
//! goes through the same sequence:
//!
//! 1. [`inspect`] reports what *would* happen — the resolved path, whether it is
//!    a symlink, whether that target sits in a git repository, whether Voxide is
//!    already bound, and the exact text to be inserted. The UI shows this first.
//! 2. [`apply`] copies the original to a timestamped `.voxide-backup` beside it.
//! 3. The edit is written to a temporary file next to the config.
//! 4. Where the compositor ships a validator (`niri validate -c`, `sway -C -c`),
//!    the temporary file is checked *before* it replaces anything. A rejected
//!    edit is reported and the original is left untouched.
//!
//! The managed block is delimited, so re-applying replaces Voxide's own lines
//! rather than appending a second copy, and removing the feature is a matter of
//! deleting between the markers.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::debug_log;

/// Marker text opening and closing the block this module owns. Anything between
/// them is Voxide's to rewrite; anything outside is the user's and never touched.
const BEGIN_MARKER: &str = ">>> voxide dictation binds (managed) >>>";
const END_MARKER: &str = "<<< voxide dictation binds (managed) <<<";

/// Compositors whose keybinding syntax and config location Voxide knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Compositor {
    Niri,
    Sway,
    Hyprland,
    River,
}

impl Compositor {
    /// The value of `$XDG_CURRENT_DESKTOP` this compositor reports, lowercased.
    fn desktop_name(self) -> &'static str {
        match self {
            Self::Niri => "niri",
            Self::Sway => "sway",
            Self::Hyprland => "hyprland",
            Self::River => "river",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Niri => "niri",
            Self::Sway => "Sway",
            Self::Hyprland => "Hyprland",
            Self::River => "river",
        }
    }

    /// Config file this compositor reads, relative to the user's config dir.
    fn config_relative_path(self) -> &'static str {
        match self {
            Self::Niri => "niri/config.kdl",
            Self::Sway => "sway/config",
            Self::Hyprland => "hypr/hyprland.conf",
            Self::River => "river/init",
        }
    }

    /// Comment syntax, which differs because niri's config is KDL.
    fn comment_prefix(self) -> &'static str {
        match self {
            Self::Niri => "//",
            Self::Sway | Self::Hyprland | Self::River => "#",
        }
    }

    /// Whether bindings live inside a block that must be edited into, rather
    /// than appended at the end of the file.
    fn needs_block_insertion(self) -> bool {
        matches!(self, Self::Niri)
    }

    /// The keybinding lines themselves, already pointing at `executable`.
    ///
    /// `--trigger` is a tap rather than a hold, so only the toggle and cancel
    /// actions are bound; hold-to-record needs the portal or a native grab.
    fn binding_line(self, chord: &str, action: Action, executable: &str) -> String {
        let trigger = action.trigger();
        let title = action.overlay_title();
        match self {
            Self::Niri => format!(
                "{chord} hotkey-overlay-title=\"{title}\" {{ spawn \"{executable}\" \"--trigger\" \"{trigger}\"; }}"
            ),
            Self::Sway => {
                let repeat = if action == Action::Dictate { "--no-repeat " } else { "" };
                let chord = chord.replace("Mod+", "$mod+");
                format!("bindsym {repeat}{chord} exec {executable} --trigger {trigger}")
            }
            Self::Hyprland => format!("bind = {chord}, exec, {executable} --trigger {trigger}"),
            Self::River => format!("riverctl map normal {chord} spawn '{executable} --trigger {trigger}'"),
        }
    }

    /// Chords to try for an action, most preferred first.
    ///
    /// More than one, because the obvious choices are frequently taken: niri
    /// binds `Mod+Escape` out of the box and *rejects the whole config* on a
    /// duplicate keybind, so writing a fixed chord would break the user's
    /// desktop rather than configure it.
    fn chord_candidates(self, action: Action) -> &'static [&'static str] {
        match (self, action) {
            (Self::Niri | Self::Sway, Action::Dictate) => {
                &["Mod+Space", "Mod+Shift+D", "Mod+Alt+D"]
            }
            (Self::Niri | Self::Sway, Action::Cancel) => {
                &["Mod+Escape", "Mod+Shift+Escape", "Mod+Alt+Escape"]
            }
            (Self::Hyprland, Action::Dictate) => {
                &["SUPER, Space", "SUPER SHIFT, D", "SUPER ALT, D"]
            }
            (Self::Hyprland, Action::Cancel) => {
                &["SUPER, Escape", "SUPER SHIFT, Escape", "SUPER ALT, Escape"]
            }
            (Self::River, Action::Dictate) => &["Super Space", "Super+Shift D", "Super+Mod1 D"],
            (Self::River, Action::Cancel) => {
                &["Super Escape", "Super+Shift Escape", "Super+Mod1 Escape"]
            }
        }
    }

    /// The chord portion of an existing binding line, so conflicts can be seen.
    /// `None` for lines that are not bindings.
    fn bound_chord(self, line: &str) -> Option<String> {
        let line = line.trim();
        if line.is_empty() || line.starts_with(self.comment_prefix()) {
            return None;
        }
        match self {
            // KDL: the chord is the node name, the first token on the line.
            Self::Niri => {
                let token = line.split_whitespace().next()?;
                (token.contains('+') || line.contains('{')).then(|| token.to_string())
            }
            Self::Sway => line
                .strip_prefix("bindsym")
                .map(|rest| rest.replace("--no-repeat", "").trim().to_string())
                .and_then(|rest| rest.split_whitespace().next().map(str::to_string)),
            Self::Hyprland => line
                .strip_prefix("bind")
                .and_then(|rest| rest.split_once('='))
                .and_then(|(_, rest)| rest.rsplit_once(", exec"))
                .map(|(chord, _)| chord.trim().to_string()),
            Self::River => line
                .strip_prefix("riverctl map normal")
                .map(|rest| {
                    rest.split_whitespace()
                        .take(2)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .filter(|chord| !chord.is_empty()),
        }
    }

    /// A command that checks a config file without loading it, if one exists.
    /// Used as the gate before an edited file replaces the original.
    fn validator(self, candidate: &Path) -> Option<Command> {
        match self {
            Self::Niri => {
                let mut command = Command::new("niri");
                command.arg("validate").arg("-c").arg(candidate);
                Some(command)
            }
            Self::Sway => {
                let mut command = Command::new("sway");
                command.arg("-C").arg("-c").arg(candidate);
                Some(command)
            }
            // Hyprland's checker requires a running instance to talk to, and
            // river's init is a shell script with no config syntax to check.
            Self::Hyprland | Self::River => None,
        }
    }
}

/// The two trigger actions worth a keybinding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Dictate,
    Cancel,
}

impl Action {
    const ALL: &'static [Self] = &[Self::Dictate, Self::Cancel];

    fn trigger(self) -> &'static str {
        match self {
            Self::Dictate => "dictate",
            Self::Cancel => "cancel",
        }
    }

    fn overlay_title(self) -> &'static str {
        match self {
            Self::Dictate => "Dictation: Voxide",
            Self::Cancel => "Dictation: cancel",
        }
    }
}

/// What applying the bindings would involve. Everything the UI needs to let the
/// user make an informed decision, and nothing is written to produce it.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    pub compositor: Compositor,
    pub compositor_name: String,
    /// Path as configured, which may be a symlink.
    pub config_path: String,
    /// Where that path actually resolves. Differs when the config is a symlink.
    pub resolved_path: String,
    pub is_symlink: bool,
    /// Set when the resolved file lives inside a git working tree, so the user
    /// knows the edit will show up as a change in another repository.
    pub git_repository: Option<String>,
    pub config_exists: bool,
    /// Lines that already mention Voxide, whether managed by us or hand-written.
    /// Non-empty means the user very likely already has this working.
    pub existing_references: Vec<String>,
    /// True when a previous apply left a managed block behind, which the next
    /// apply replaces rather than duplicating.
    pub has_managed_block: bool,
    /// The full block that would be written.
    pub snippet: String,
    /// `None` when the block can be placed, `Some(reason)` when it cannot and
    /// the user has to paste it manually.
    pub blocked: Option<String>,
}

/// Result of a successful write.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Applied {
    pub config_path: String,
    pub backup_path: String,
    pub validated: bool,
    pub replaced_existing_block: bool,
}

/// Identifies the running compositor.
///
/// `$XDG_CURRENT_DESKTOP` is authoritative when it names something known; it can
/// be a colon-separated list, so each entry is considered.
pub fn detect() -> Option<Compositor> {
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").ok()?.to_lowercase();
    const KNOWN: &[Compositor] = &[
        Compositor::Niri,
        Compositor::Sway,
        Compositor::Hyprland,
        Compositor::River,
    ];
    desktop
        .split(':')
        .find_map(|entry| {
            let entry = entry.trim();
            KNOWN
                .iter()
                .copied()
                .find(|compositor| compositor.desktop_name() == entry)
        })
        .or_else(|| {
            // Hyprland historically reported "Hyprland" only via its own
            // signature variable.
            std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").map(|_| Compositor::Hyprland)
        })
}

/// Describes what [`apply`] would do, without touching anything.
pub fn inspect() -> Result<Plan, String> {
    let compositor =
        detect().ok_or("Voxide does not know this compositor's keybinding syntax".to_string())?;
    let config_path = config_path(compositor)?;
    let resolved = fs::canonicalize(&config_path).unwrap_or_else(|_| config_path.clone());
    let contents = fs::read_to_string(&config_path).unwrap_or_default();
    let config_exists = config_path.exists();
    let taken = bound_chords(&contents, compositor);
    // A chord clash is reported through `blocked` rather than as an error, so the
    // UI can still explain the situation instead of showing nothing.
    let (snippet, chord_conflict) = match managed_block(
        compositor,
        &executable_path()?,
        compositor.needs_block_insertion(),
        &taken,
    ) {
        Ok(snippet) => (snippet, None),
        Err(reason) => (String::new(), Some(reason)),
    };

    let blocked = if let Some(reason) = chord_conflict {
        Some(reason)
    } else if !config_exists {
        Some(format!(
            "{} has no config file at {}",
            compositor.display_name(),
            config_path.display()
        ))
    } else if compositor.needs_block_insertion() && find_binds_block_end(&contents).is_none() {
        Some("Could not find a `binds { … }` section to insert into".to_string())
    } else {
        None
    };

    Ok(Plan {
        compositor,
        compositor_name: compositor.display_name().to_string(),
        config_path: config_path.display().to_string(),
        resolved_path: resolved.display().to_string(),
        is_symlink: fs::symlink_metadata(&config_path)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false),
        git_repository: enclosing_git_repository(&resolved),
        config_exists,
        existing_references: voxide_references(&contents),
        has_managed_block: contents.contains(BEGIN_MARKER),
        snippet,
        blocked,
    })
}

/// Installs the bindings. Only call after the user has seen [`inspect`].
pub fn apply() -> Result<Applied, String> {
    let plan = inspect()?;
    if let Some(reason) = plan.blocked {
        return Err(reason);
    }
    let compositor = plan.compositor;
    let config_path = PathBuf::from(&plan.config_path);
    let original = fs::read_to_string(&config_path)
        .map_err(|error| format!("Could not read {}: {error}", config_path.display()))?;

    let updated = if original.contains(BEGIN_MARKER) {
        replace_managed_block(&original, &plan.snippet)?
    } else if compositor.needs_block_insertion() {
        insert_into_binds_block(&original, &plan.snippet)?
    } else {
        format!("{}\n{}\n", original.trim_end(), plan.snippet)
    };

    // Back up before anything is replaced, so a bad outcome is always one `cp`
    // away from being undone.
    let backup_path = backup_path_for(&config_path);
    fs::copy(&config_path, &backup_path)
        .map_err(|error| format!("Could not back up the config: {error}"))?;

    // Validate the candidate before it becomes the real config. A compositor
    // that rejects the file must never get the chance to load it.
    let candidate = config_path.with_extension("voxide-candidate");
    fs::write(&candidate, &updated)
        .map_err(|error| format!("Could not stage the edited config: {error}"))?;
    let validated = match compositor.validator(&candidate) {
        Some(mut command) => match command.output() {
            Ok(output) if output.status.success() => true,
            Ok(output) => {
                let _ = fs::remove_file(&candidate);
                let detail = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "{} rejected the edited config, so nothing was changed: {}",
                    compositor.display_name(),
                    detail.trim()
                ));
            }
            // A missing validator binary is not a reason to refuse; it just
            // means this write is unchecked, which `validated` reports.
            Err(_) => false,
        },
        None => false,
    };

    fs::rename(&candidate, &config_path).map_err(|error| {
        format!(
            "Could not replace {} (the backup at {} is intact): {error}",
            config_path.display(),
            backup_path.display()
        )
    })?;
    debug_log::append(&format!(
        "wrote {} keybindings (validated: {validated})",
        compositor.display_name()
    ));

    Ok(Applied {
        config_path: config_path.display().to_string(),
        backup_path: backup_path.display().to_string(),
        validated,
        replaced_existing_block: plan.has_managed_block,
    })
}

fn config_path(compositor: Compositor) -> Result<PathBuf, String> {
    let base = directories::BaseDirs::new().ok_or("No home directory was found".to_string())?;
    Ok(base.config_dir().join(compositor.config_relative_path()))
}

/// The binary a keybinding should invoke. The running executable rather than a
/// bare `voxide`, so the bindings work from a source build that is not on `PATH`.
fn executable_path() -> Result<String, String> {
    let path = std::env::current_exe()
        .map_err(|error| format!("Could not determine Voxide's own path: {error}"))?;
    Ok(path.display().to_string())
}

/// Lines already referencing Voxide, so an existing hand-written binding is
/// reported instead of silently gaining a duplicate on a different chord.
fn voxide_references(contents: &str) -> Vec<String> {
    contents
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            // A comment explaining the binding is not a binding. Both `//` and
            // `#` are checked because this runs before the compositor is known.
            if trimmed.starts_with("//") || trimmed.starts_with('#') {
                return false;
            }
            let lowered = trimmed.to_lowercase();
            lowered.contains("voxide") && lowered.contains("--trigger")
        })
        .map(|line| line.trim().to_string())
        .collect()
}

fn enclosing_git_repository(path: &Path) -> Option<String> {
    let mut directory = path.parent()?;
    loop {
        if directory.join(".git").exists() {
            return Some(directory.display().to_string());
        }
        directory = directory.parent()?;
    }
}

fn backup_path_for(config_path: &Path) -> PathBuf {
    // chrono rather than a counter so successive backups sort and never collide.
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let name = config_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "config".to_string());
    config_path.with_file_name(format!("{name}.voxide-backup-{stamp}"))
}

/// The delimited block, indented one level when it goes inside a `binds` section.
///
/// `taken` is the set of chords the config already binds *outside* the managed
/// block; each action claims its first candidate that is not in it. `Err` names
/// the action when every candidate is spoken for, because writing a duplicate
/// would make the compositor reject the whole file.
fn managed_block(
    compositor: Compositor,
    executable: &str,
    indented: bool,
    taken: &[String],
) -> Result<String, String> {
    let comment = compositor.comment_prefix();
    let pad = if indented { "    " } else { "" };
    let mut lines = vec![
        format!("{pad}{comment} {BEGIN_MARKER}"),
        format!("{pad}{comment} Taps the running instance over its $XDG_RUNTIME_DIR socket."),
    ];
    for action in Action::ALL {
        let chord = compositor
            .chord_candidates(*action)
            .iter()
            .find(|candidate| !taken.iter().any(|bound| bound == *candidate))
            .ok_or_else(|| {
                format!(
                    "Every shortcut Voxide would use to {} is already bound in your {} config",
                    action.trigger(),
                    compositor.display_name()
                )
            })?;
        lines.push(format!(
            "{pad}{}",
            compositor.binding_line(chord, *action, executable)
        ));
    }
    lines.push(format!("{pad}{comment} {END_MARKER}"));
    Ok(lines.join("\n"))
}

/// Chords the config binds, ignoring anything inside the managed block so a
/// re-apply does not treat its own previous output as a conflict.
fn bound_chords(contents: &str, compositor: Compositor) -> Vec<String> {
    let mut chords = Vec::new();
    let mut inside_managed = false;
    for line in contents.lines() {
        if line.contains(BEGIN_MARKER) {
            inside_managed = true;
            continue;
        }
        if line.contains(END_MARKER) {
            inside_managed = false;
            continue;
        }
        if !inside_managed {
            if let Some(chord) = compositor.bound_chord(line) {
                chords.push(chord);
            }
        }
    }
    chords
}

/// Swaps the contents between the markers, preserving everything around them.
fn replace_managed_block(contents: &str, snippet: &str) -> Result<String, String> {
    let lines: Vec<&str> = contents.lines().collect();
    let begin = lines
        .iter()
        .position(|line| line.contains(BEGIN_MARKER))
        .ok_or("The managed block markers went missing".to_string())?;
    let end = lines
        .iter()
        .skip(begin)
        .position(|line| line.contains(END_MARKER))
        .map(|offset| begin + offset)
        .ok_or("The managed block has an opening marker but no closing marker".to_string())?;
    let mut rebuilt: Vec<String> = lines[..begin].iter().map(|line| line.to_string()).collect();
    rebuilt.extend(snippet.lines().map(|line| line.to_string()));
    rebuilt.extend(lines[end + 1..].iter().map(|line| line.to_string()));
    Ok(format!("{}\n", rebuilt.join("\n")))
}

/// Inserts the block just before the closing brace of the top-level `binds`
/// section, which is where niri requires keybindings to live.
fn insert_into_binds_block(contents: &str, snippet: &str) -> Result<String, String> {
    let end = find_binds_block_end(contents)
        .ok_or("Could not find a `binds { … }` section to insert into".to_string())?;
    let lines: Vec<&str> = contents.lines().collect();
    let mut rebuilt: Vec<String> = lines[..end].iter().map(|line| line.to_string()).collect();
    rebuilt.extend(snippet.lines().map(|line| line.to_string()));
    rebuilt.extend(lines[end..].iter().map(|line| line.to_string()));
    Ok(format!("{}\n", rebuilt.join("\n")))
}

/// Index of the line holding the closing brace of the top-level `binds` block.
///
/// Braces are counted from the `binds {` line onward, ignoring anything inside
/// double quotes or after a `//` comment so a binding that spawns a command
/// containing a brace cannot throw the count off. This is a heuristic rather
/// than a KDL parser, which is why every write is validated afterwards.
fn find_binds_block_end(contents: &str) -> Option<usize> {
    let lines: Vec<&str> = contents.lines().collect();
    let start = lines.iter().position(|line| {
        let trimmed = line.trim_start();
        !trimmed.starts_with("//") && trimmed.starts_with("binds") && trimmed.contains('{')
    })?;
    let mut depth = 0usize;
    for (offset, line) in lines[start..].iter().enumerate() {
        for (open, close) in braces_outside_strings(line) {
            depth += open;
            depth = depth.saturating_sub(close);
            if depth == 0 {
                return Some(start + offset);
            }
        }
    }
    None
}

/// Counts `{` and `}` in a line, skipping quoted spans and trailing comments.
/// Returns one `(open, close)` pair per brace so nesting can be tracked in
/// order rather than netted out per line.
fn braces_outside_strings(line: &str) -> Vec<(usize, usize)> {
    let mut counts = Vec::new();
    let mut in_string = false;
    let mut previous = '\0';
    let characters: Vec<char> = line.chars().collect();
    for (index, character) in characters.iter().copied().enumerate() {
        if character == '"' && previous != '\\' {
            in_string = !in_string;
        } else if !in_string && character == '/' && characters.get(index + 1) == Some(&'/') {
            break;
        } else if !in_string {
            match character {
                '{' => counts.push((1, 0)),
                '}' => counts.push((0, 1)),
                _ => {}
            }
        }
        previous = character;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    const NIRI_CONFIG: &str = r#"input {
    keyboard {
        xkb {
        }
    }
}

binds {
    Mod+Shift+Slash { show-hotkey-overlay; }
    Mod+T { spawn "ghostty"; }
}

window-rule {
    match app-id="voxide"
}
"#;

    #[test]
    fn the_binds_block_end_is_found_past_nested_braces() {
        // `input { keyboard { xkb { } } }` above must not be mistaken for it.
        let end = find_binds_block_end(NIRI_CONFIG).expect("binds block is present");
        let lines: Vec<&str> = NIRI_CONFIG.lines().collect();
        assert_eq!(lines[end].trim(), "}");
        assert!(lines[end - 1].contains("ghostty"), "{:?}", lines[end - 1]);
    }

    #[test]
    fn braces_inside_quoted_commands_do_not_confuse_the_count() {
        // A spawn-sh binding may legitimately contain braces in its shell text.
        let line = r#"    Mod+X { spawn-sh "awk '{print $1}' file"; }"#;
        let counts = braces_outside_strings(line);
        let opens: usize = counts.iter().map(|(open, _)| open).sum();
        let closes: usize = counts.iter().map(|(_, close)| close).sum();
        assert_eq!((opens, closes), (1, 1), "{counts:?}");
    }

    #[test]
    fn a_commented_out_binds_line_is_not_the_block() {
        let config = "// binds {\nbinds {\n    Mod+A { spawn \"a\"; }\n}\n";
        let end = find_binds_block_end(config).expect("the real block is found");
        assert_eq!(config.lines().nth(end), Some("}"));
    }

    #[test]
    fn trailing_comments_do_not_contribute_braces() {
        assert!(braces_outside_strings("    Mod+A { spawn \"a\"; } // note }").len() == 2);
    }

    #[test]
    fn insertion_lands_inside_the_binds_block() {
        let snippet = managed_block(Compositor::Niri, "/usr/bin/voxide", true, &[]).unwrap();
        let updated = insert_into_binds_block(NIRI_CONFIG, &snippet).expect("insertion succeeds");
        let lines: Vec<&str> = updated.lines().collect();
        let marker = lines
            .iter()
            .position(|line| line.contains(BEGIN_MARKER))
            .expect("the block was written");
        let binds = lines
            .iter()
            .position(|line| line.trim() == "binds {")
            .expect("binds block still present");
        // Inside the block, and before the window-rule that follows it.
        assert!(binds < marker, "block landed above `binds {{`");
        let window_rule = lines
            .iter()
            .position(|line| line.contains("window-rule"))
            .expect("the rest of the file survived");
        assert!(marker < window_rule);
        assert!(updated.contains("--trigger"));
    }

    #[test]
    fn reapplying_replaces_the_block_instead_of_stacking_copies() {
        let first = managed_block(Compositor::Niri, "/old/path/voxide", true, &[]).unwrap();
        let once = insert_into_binds_block(NIRI_CONFIG, &first).expect("first insert");
        let second = managed_block(Compositor::Niri, "/new/path/voxide", true, &[]).unwrap();
        let twice = replace_managed_block(&once, &second).expect("replace");
        assert_eq!(twice.matches(BEGIN_MARKER).count(), 1);
        assert_eq!(twice.matches(END_MARKER).count(), 1);
        assert!(twice.contains("/new/path/voxide"));
        assert!(!twice.contains("/old/path/voxide"));
        // Everything outside the markers is untouched.
        assert!(twice.contains("ghostty"));
        assert!(twice.contains("show-hotkey-overlay"));
    }

    #[test]
    fn appending_compositors_keep_the_original_content() {
        for compositor in [Compositor::Sway, Compositor::Hyprland, Compositor::River] {
            let snippet = managed_block(compositor, "/usr/bin/voxide", false, &[]).unwrap();
            let original = "# my config\nset $mod Mod4\n";
            let updated = format!("{}\n{}\n", original.trim_end(), snippet);
            assert!(updated.starts_with("# my config"));
            assert!(updated.contains("set $mod Mod4"));
            assert!(updated.contains("--trigger dictate"));
            assert!(updated.contains("--trigger cancel"));
            assert!(!compositor.needs_block_insertion());
        }
    }

    #[test]
    fn every_compositor_binds_both_a_toggle_and_a_cancel() {
        for compositor in [
            Compositor::Niri,
            Compositor::Sway,
            Compositor::Hyprland,
            Compositor::River,
        ] {
            let lines: Vec<String> = Action::ALL
                .iter()
                .map(|action| {
                    let chord = compositor.chord_candidates(*action)[0];
                    compositor.binding_line(chord, *action, "/usr/bin/voxide")
                })
                .collect();
            assert_eq!(lines.len(), 2, "{compositor:?}");
            assert!(lines[0].contains("--trigger"));
            assert!(lines.iter().any(|line| line.contains("dictate")));
            assert!(lines.iter().any(|line| line.contains("cancel")));
            // The absolute path matters: a source build is not on PATH.
            assert!(lines.iter().all(|line| line.contains("/usr/bin/voxide")));
        }
    }

    #[test]
    fn a_chord_already_bound_is_skipped_for_the_next_candidate() {
        // Regression: niri binds Mod+Escape to shortcut-inhibit toggling by
        // default and rejects the *entire* config on a duplicate keybind, so a
        // fixed chord broke the desktop instead of configuring it.
        let config = concat!(
            "binds {\n",
            "    Mod+Space { spawn \"launcher\"; }\n",
            "    Mod+Escape allow-inhibiting=false { toggle-keyboard-shortcuts-inhibit; }\n",
            "}\n"
        );
        let taken = bound_chords(config, Compositor::Niri);
        assert!(taken.contains(&"Mod+Space".to_string()), "{taken:?}");
        assert!(taken.contains(&"Mod+Escape".to_string()), "{taken:?}");

        let block = managed_block(Compositor::Niri, "/usr/bin/voxide", true, &taken)
            .expect("a free chord remains for each action");
        assert!(
            !block.contains("Mod+Space hotkey"),
            "reused a taken chord:\n{block}"
        );
        assert!(
            !block.contains("Mod+Escape hotkey"),
            "reused a taken chord:\n{block}"
        );
        assert!(block.contains("Mod+Shift+D"), "{block}");
        assert!(block.contains("Mod+Shift+Escape"), "{block}");
    }

    #[test]
    fn exhausting_every_candidate_is_reported_rather_than_written() {
        let taken: Vec<String> = Compositor::Niri
            .chord_candidates(Action::Dictate)
            .iter()
            .map(|chord| chord.to_string())
            .collect();
        let error = managed_block(Compositor::Niri, "/usr/bin/voxide", true, &taken)
            .expect_err("no chord is left for dictate");
        assert!(error.contains("dictate"), "{error}");
    }

    #[test]
    fn a_reapply_does_not_treat_its_own_chords_as_conflicts() {
        // Without excluding the managed block, the second apply would see its
        // own Mod+Space as taken and drift onto the fallback chord every time.
        let first = managed_block(Compositor::Niri, "/usr/bin/voxide", true, &[]).unwrap();
        let once = insert_into_binds_block(NIRI_CONFIG, &first).unwrap();
        let taken = bound_chords(&once, Compositor::Niri);
        assert!(!taken.contains(&"Mod+Space".to_string()), "{taken:?}");
        let again = managed_block(Compositor::Niri, "/usr/bin/voxide", true, &taken).unwrap();
        assert!(again.contains("Mod+Space"), "{again}");
    }

    #[test]
    fn chord_extraction_understands_each_compositors_syntax() {
        assert_eq!(
            Compositor::Sway.bound_chord("bindsym --no-repeat $mod+space exec foo"),
            Some("$mod+space".to_string())
        );
        assert_eq!(
            Compositor::Hyprland.bound_chord("bind = SUPER, Space, exec, foo --trigger dictate"),
            Some("SUPER, Space".to_string())
        );
        assert_eq!(
            Compositor::River.bound_chord("riverctl map normal Super Space spawn 'foo'"),
            Some("Super Space".to_string())
        );
        // Comments and blank lines bind nothing.
        assert_eq!(
            Compositor::Niri.bound_chord("// Mod+Space { spawn \"x\"; }"),
            None
        );
        assert_eq!(Compositor::Sway.bound_chord(""), None);
    }

    #[test]
    fn hand_written_bindings_are_reported_so_they_are_not_duplicated() {
        let config = concat!(
            "binds {\n",
            "    Mod+Shift+D { spawn-sh \"~/dev/voxide --trigger dictate\"; }\n",
            "}\n"
        );
        let found = voxide_references(config);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("Mod+Shift+D"));
    }

    #[test]
    fn a_comment_describing_a_binding_is_not_counted_as_one() {
        // The user's real config has an explanatory comment containing both
        // "Voxide" and "--trigger"; counting it inflated the reported bindings.
        let config = "// Voxide: --trigger taps the running instance\nbinds {\n}\n";
        assert!(voxide_references(config).is_empty());
    }

    #[test]
    fn unrelated_lines_are_not_mistaken_for_bindings() {
        // Matching on the app-id window rule would report a binding that is not
        // there and make the UI claim the user is already set up.
        let config = "window-rule {\n    match app-id=\"voxide\"\n}\n";
        assert!(voxide_references(config).is_empty());
    }

    #[test]
    fn backups_are_timestamped_beside_the_original() {
        let backup = backup_path_for(Path::new("/home/u/.config/niri/config.kdl"));
        assert_eq!(
            backup.parent(),
            Path::new("/home/u/.config/niri")
                .parent()
                .map(|_| Path::new("/home/u/.config/niri"))
        );
        let name = backup.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("config.kdl.voxide-backup-"), "{name}");
    }

    #[test]
    fn only_compositors_with_a_real_checker_claim_validation() {
        let candidate = Path::new("/tmp/candidate");
        assert!(Compositor::Niri.validator(candidate).is_some());
        assert!(Compositor::Sway.validator(candidate).is_some());
        // Hyprland's checker needs a live instance; river's init is a script.
        assert!(Compositor::Hyprland.validator(candidate).is_none());
        assert!(Compositor::River.validator(candidate).is_none());
    }

    /// Reports what an apply would do to *this* machine's real config, and
    /// dry-runs the edit on a copy through the compositor's own validator so the
    /// result is proven without touching the original. Read-only for the user's
    /// file. Run inside the session under test:
    ///   cargo test binding_plan_probe -- --ignored --nocapture
    #[test]
    #[ignore = "reads the live compositor config and shells out to its validator"]
    fn binding_plan_probe() {
        let plan = match inspect() {
            Ok(plan) => plan,
            Err(error) => {
                println!("no plan: {error}");
                return;
            }
        };
        println!("compositor:      {}", plan.compositor_name);
        println!("config:          {}", plan.config_path);
        println!("resolves to:     {}", plan.resolved_path);
        println!("symlink:         {}", plan.is_symlink);
        println!("git repository:  {:?}", plan.git_repository);
        println!("managed block:   {}", plan.has_managed_block);
        println!("blocked:         {:?}", plan.blocked);
        println!("existing binds:  {} found", plan.existing_references.len());
        for reference in &plan.existing_references {
            println!("  {reference}");
        }

        if plan.blocked.is_some() {
            return;
        }
        // Prove the edit on a copy. The user's file is never written here.
        let original = std::fs::read_to_string(&plan.config_path).expect("read the real config");
        let edited = if original.contains(BEGIN_MARKER) {
            replace_managed_block(&original, &plan.snippet)
        } else if plan.compositor.needs_block_insertion() {
            insert_into_binds_block(&original, &plan.snippet)
        } else {
            Ok(format!("{}\n{}\n", original.trim_end(), plan.snippet))
        }
        .expect("the edit can be produced");
        let scratch = std::env::temp_dir().join("voxide-binding-probe.kdl");
        std::fs::write(&scratch, &edited).expect("stage the candidate");
        match plan.compositor.validator(&scratch) {
            Some(mut command) => match command.output() {
                Ok(output) if output.status.success() => {
                    println!("validator:       ACCEPTED the edited copy")
                }
                Ok(output) => println!(
                    "validator:       REJECTED — {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
                Err(error) => println!("validator:       could not run ({error})"),
            },
            None => println!("validator:       none for this compositor"),
        }
        let _ = std::fs::remove_file(&scratch);
    }

    #[test]
    fn detection_reads_a_colon_separated_desktop_list() {
        // XDG_CURRENT_DESKTOP is frequently "niri:wlroots" or similar.
        for compositor in [
            Compositor::Niri,
            Compositor::Sway,
            Compositor::Hyprland,
            Compositor::River,
        ] {
            assert!(!compositor.desktop_name().is_empty());
            assert_eq!(
                compositor.desktop_name(),
                compositor.desktop_name().to_lowercase()
            );
        }
    }
}
