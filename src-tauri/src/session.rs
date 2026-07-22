/// Small, engine-independent ownership guard for a dictation lifecycle.
///
/// It intentionally knows nothing about audio or models: the caller owns those
/// resources, while this type makes illegal overlapping and stale terminal
/// transitions impossible to miss in tests and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Idle,
    Recording { id: u64 },
    Finalizing { id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewAdmission {
    Admitted,
    Busy,
    Inactive,
}

#[derive(Debug, Default)]
pub struct Coordinator {
    next_id: u64,
    state: SessionState,
    preview_in_flight: Option<u64>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::Idle
    }
}

impl Coordinator {
    pub fn is_idle(&self) -> bool {
        self.state == SessionState::Idle
    }

    #[cfg(test)]
    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn start(&mut self) -> Result<u64, &'static str> {
        if self.state != SessionState::Idle {
            return Err("A dictation session is already active");
        }
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.state = SessionState::Recording { id: self.next_id };
        Ok(self.next_id)
    }

    pub fn begin_finalizing(&mut self, id: u64) -> bool {
        if self.state == (SessionState::Recording { id }) {
            self.state = SessionState::Finalizing { id };
            true
        } else {
            false
        }
    }

    /// Admits at most one display-only preview task for the active recording.
    /// Finalization changes the state before it begins final inference, so no
    /// newly admitted preview can delay or overwrite a terminal result.
    pub fn begin_preview(&mut self, id: u64) -> PreviewAdmission {
        if self.state != (SessionState::Recording { id }) {
            return PreviewAdmission::Inactive;
        }
        if self.preview_in_flight.is_some() {
            return PreviewAdmission::Busy;
        }
        self.preview_in_flight = Some(id);
        PreviewAdmission::Admitted
    }

    pub fn finish_preview(&mut self, id: u64) -> bool {
        if self.preview_in_flight == Some(id) {
            self.preview_in_flight = None;
            true
        } else {
            false
        }
    }

    pub fn cancel(&mut self, id: u64) -> bool {
        match self.state {
            SessionState::Recording { id: active } | SessionState::Finalizing { id: active }
                if active == id =>
            {
                self.state = SessionState::Idle;
                self.preview_in_flight = None;
                true
            }
            _ => false,
        }
    }

    pub fn finish(&mut self, id: u64) -> bool {
        if self.state == (SessionState::Finalizing { id }) {
            self.state = SessionState::Idle;
            self.preview_in_flight = None;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_active_session_can_transition_to_one_terminal_outcome() {
        let mut coordinator = Coordinator::default();
        let first = coordinator.start().expect("first session starts");
        assert!(coordinator.start().is_err());
        assert!(!coordinator.begin_finalizing(first + 1));
        assert!(coordinator.begin_finalizing(first));
        assert!(!coordinator.cancel(first + 1));
        assert!(coordinator.finish(first));
        assert_eq!(coordinator.state(), SessionState::Idle);
        assert!(coordinator.start().expect("next session starts") > first);
    }

    #[test]
    fn cancellation_rejects_stale_terminal_transitions() {
        let mut coordinator = Coordinator::default();
        assert!(coordinator.is_idle());
        let id = coordinator.start().expect("session starts");
        assert!(!coordinator.is_idle());
        assert!(coordinator.cancel(id));
        assert!(coordinator.is_idle());
        assert!(!coordinator.begin_finalizing(id));
        assert!(!coordinator.finish(id));
    }

    #[test]
    fn finalization_stops_new_previews_without_waiting_for_the_old_one() {
        let mut coordinator = Coordinator::default();
        let id = coordinator.start().expect("session starts");
        assert_eq!(coordinator.begin_preview(id), PreviewAdmission::Admitted);
        assert_eq!(coordinator.begin_preview(id), PreviewAdmission::Busy);
        assert!(coordinator.begin_finalizing(id));
        assert_eq!(coordinator.begin_preview(id), PreviewAdmission::Inactive);
        assert!(coordinator.finish_preview(id));
        assert!(coordinator.finish(id));
    }
}
