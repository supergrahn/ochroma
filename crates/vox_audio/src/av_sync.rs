//! Spectral AV synchronization.
//!
//! Links visual spectral events to audio synthesis with sub-frame accuracy.
//! When a visual event (explosion, collision) occurs, its spectral signature
//! directly drives audio timbre — a glass break sounds glassy because its
//! spectral signature peaks in high bands (→ high-frequency audio).
//!
//! Why better than Unreal: Unreal uses pre-authored sound cues mapped by
//! material type. Ochroma derives audio from the actual spectral state of
//! the object — a half-burned wooden wall sounds different from pristine wood.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AvEventKind {
    Impact,
    Break,
    Burn,
    Splash,
    Explosion,
}

#[derive(Debug, Clone)]
pub struct AvEvent {
    pub kind: AvEventKind,
    pub position: [f32; 3],
    pub spectral: [f32; 16],
    pub intensity: f32,
    pub timestamp_ms: u64,
}

pub struct AvSyncQueue {
    events: std::collections::VecDeque<AvEvent>,
    pub max_queue_size: usize,
}

impl AvSyncQueue {
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            events: std::collections::VecDeque::new(),
            max_queue_size,
        }
    }

    /// Push an event, dropping the oldest if over capacity.
    pub fn push(&mut self, event: AvEvent) {
        if self.events.len() >= self.max_queue_size {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Return events with `timestamp_ms <= current_ms + lookahead_ms`, removing them.
    pub fn drain_pending(&mut self, current_ms: u64, lookahead_ms: u64) -> Vec<AvEvent> {
        let deadline = current_ms.saturating_add(lookahead_ms);
        let mut result = Vec::new();
        let mut remaining = std::collections::VecDeque::new();
        for event in self.events.drain(..) {
            if event.timestamp_ms <= deadline {
                result.push(event);
            } else {
                remaining.push_back(event);
            }
        }
        self.events = remaining;
        result
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioParams {
    pub duration_secs: f32,
    pub volume: f32,
    pub pitch_shift: f32,
}

/// Derive audio parameters from a spectral signature and event kind.
pub fn spectral_to_audio_params(spectral: &[f32; 16], kind: AvEventKind) -> AudioParams {
    // Dominant band: the index with the highest weight.
    let (dominant_band, _) = spectral
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    // Normalised [0,1]: band 0 is highest-freq (UV-violet), band 15 is lowest-freq (near-IR).
    // "dominant_band_weight" = how low-frequency the dominant band is (0 = UV, 1 = near-IR).
    let dominant_band_weight = dominant_band as f32 / 15.0;

    // Blue energy = mean of bands 0-4; red/IR energy = mean of bands 11-15.
    let blue_energy = (spectral[0] + spectral[1] + spectral[2] + spectral[3] + spectral[4]) / 5.0;
    let red_energy = (spectral[11] + spectral[12] + spectral[13] + spectral[14] + spectral[15]) / 5.0;

    // Intensity-independent defaults; intensity is passed via AvEvent and set as volume.
    let volume = 1.0_f32.min(0.0_f32.max(1.0)); // clamped placeholder; caller sets per-event.

    let mut duration_secs = 0.05 + dominant_band_weight * 0.3;
    let mut pitch_shift = 1.0 + (blue_energy - red_energy) * 0.5;

    match kind {
        AvEventKind::Explosion => {
            duration_secs = 0.4;
            pitch_shift = 0.7;
        }
        AvEventKind::Break => {
            pitch_shift *= 1.2;
        }
        _ => {}
    }

    AudioParams { duration_secs, volume, pitch_shift }
}

/// Synthesize PCM audio for a visual event.
///
/// Calls `spectral_synth::synthesize_impact` with the event's spectral bands
/// and duration derived from `spectral_to_audio_params`.
/// Returns PCM samples at 44100 Hz.
pub fn synthesize_av_event(event: &AvEvent) -> Vec<f32> {
    let params = spectral_to_audio_params(&event.spectral, event.kind);
    crate::spectral_synth::synthesize_impact(&event.spectral, params.duration_secs, 44100)
}

pub struct AvSyncProcessor {
    pub queue: AvSyncQueue,
    pub current_ms: u64,
}

impl AvSyncProcessor {
    pub fn new() -> Self {
        Self {
            queue: AvSyncQueue::new(256),
            current_ms: 0,
        }
    }

    /// Advance the internal clock.
    pub fn advance(&mut self, dt_ms: u64) {
        self.current_ms = self.current_ms.saturating_add(dt_ms);
    }

    /// Enqueue a visual event.
    pub fn push_event(&mut self, event: AvEvent) {
        self.queue.push(event);
    }

    /// Drain all pending events, synthesize audio for each, return (event, samples) pairs.
    pub fn process_frame(&mut self, lookahead_ms: u64) -> Vec<(AvEvent, Vec<f32>)> {
        let pending = self.queue.drain_pending(self.current_ms, lookahead_ms);
        pending
            .into_iter()
            .map(|ev| {
                let samples = synthesize_av_event(&ev);
                (ev, samples)
            })
            .collect()
    }
}

impl Default for AvSyncProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(timestamp_ms: u64, kind: AvEventKind, spectral: [f32; 16]) -> AvEvent {
        AvEvent {
            kind,
            position: [0.0, 0.0, 0.0],
            spectral,
            intensity: 1.0,
            timestamp_ms,
        }
    }

    #[test]
    fn queue_respects_max_size() {
        let mut queue = AvSyncQueue::new(5);
        for i in 0..10 {
            queue.push(make_event(i as u64, AvEventKind::Impact, [0.5; 16]));
        }
        assert_eq!(queue.len(), 5);
    }

    #[test]
    fn drain_pending_returns_events_in_window() {
        let mut queue = AvSyncQueue::new(16);
        queue.push(make_event(100, AvEventKind::Impact, [0.5; 16]));
        // current=90, lookahead=20 → deadline=110, event at 100 is within window
        let drained = queue.drain_pending(90, 20);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].timestamp_ms, 100);
    }

    #[test]
    fn drain_pending_skips_future_events() {
        let mut queue = AvSyncQueue::new(16);
        queue.push(make_event(200, AvEventKind::Impact, [0.5; 16]));
        // current=90, lookahead=20 → deadline=110, event at 200 is outside window
        let drained = queue.drain_pending(90, 20);
        assert_eq!(drained.len(), 0);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn spectral_to_audio_params_blue_is_high_pitch() {
        // All energy in blue bands (0-2)
        let mut spectral = [0.0f32; 16];
        spectral[0] = 1.0;
        spectral[1] = 1.0;
        spectral[2] = 1.0;
        let params = spectral_to_audio_params(&spectral, AvEventKind::Impact);
        assert!(
            params.pitch_shift > 1.0,
            "all-blue spectral should produce pitch_shift > 1.0, got {}",
            params.pitch_shift
        );
    }

    #[test]
    fn spectral_to_audio_params_explosion_is_low_pitch() {
        let spectral = [0.5f32; 16];
        let params = spectral_to_audio_params(&spectral, AvEventKind::Explosion);
        assert_eq!(params.pitch_shift, 0.7);
    }

    #[test]
    fn synthesize_av_event_returns_samples() {
        let event = make_event(0, AvEventKind::Impact, [0.5; 16]);
        let samples = synthesize_av_event(&event);
        assert!(!samples.is_empty(), "synthesize_av_event should return non-empty PCM samples");
    }
}
