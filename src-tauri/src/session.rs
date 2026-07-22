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

#[derive(Debug, Default)]
pub struct Coordinator {
    next_id: u64,
    state: SessionState,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::Idle
    }
}

impl Coordinator {
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

    pub fn cancel(&mut self, id: u64) -> bool {
        match self.state {
            SessionState::Recording { id: active } | SessionState::Finalizing { id: active }
                if active == id =>
            {
                self.state = SessionState::Idle;
                true
            }
            _ => false,
        }
    }

    pub fn finish(&mut self, id: u64) -> bool {
        if self.state == (SessionState::Finalizing { id }) {
            self.state = SessionState::Idle;
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
        let id = coordinator.start().expect("session starts");
        assert!(coordinator.cancel(id));
        assert!(!coordinator.begin_finalizing(id));
        assert!(!coordinator.finish(id));
    }
}
