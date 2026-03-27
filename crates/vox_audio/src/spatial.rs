//! 3D spatial audio manager with distance attenuation and stereo panning.
//!
//! Wraps rodio for real audio output when the `audio-backend` feature is enabled.
//! Falls back to silent mode (all methods succeed, no sound) when rodio is unavailable
//! or fails to initialise.

use glam::Vec3;
use std::path::Path;

/// Default distance attenuation factor.
const DEFAULT_ATTENUATION: f32 = 0.1;

/// Maximum distance before a source is considered silent.
const MAX_DISTANCE: f32 = 500.0;

// ── Backend imports (only when rodio is available) ────────────────────────
#[cfg(feature = "audio-backend")]
use rodio::{OutputStream, OutputStreamHandle, Sink, Source};

/// Listener state: position and orientation in world space.
#[derive(Debug, Clone, Copy)]
pub struct Listener {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: -Vec3::Z,
            up: Vec3::Y,
        }
    }
}

impl Listener {
    /// Right vector derived from forward x up.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize_or_zero()
    }
}

/// Whether a source is 3D (spatial) or 2D (global, no attenuation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceKind {
    /// Spatial 3D source with a world position.
    Spatial { position: Vec3 },
    /// Non-spatial (music, UI). Plays at constant volume.
    Global,
}

/// Internal bookkeeping for a playing source.
struct ActiveSource {
    handle: u32,
    kind: SourceKind,
    base_volume: f32,
    #[allow(dead_code)]
    looping: bool,
    /// True when backed by a real rodio Sink (only possible with audio-backend).
    #[cfg(feature = "audio-backend")]
    sink: Option<Sink>,
    /// Tracks whether the source has been marked as finished (silent mode or elapsed).
    finished: bool,
    /// Optional duration for auto-finishing (seconds). None = until stopped.
    duration: Option<f32>,
    /// Elapsed time since the source started playing.
    elapsed: f32,
}

/// 3D spatial audio manager.
///
/// Provides `play_3d`, `play_2d`, and `play_tone` with automatic distance
/// attenuation and stereo panning computed each `tick`.
pub struct SpatialAudioManager {
    listener: Listener,
    sources: Vec<ActiveSource>,
    next_handle: u32,
    attenuation_factor: f32,
    available: bool,
    // rodio backend (kept alive as long as the manager exists).
    #[cfg(feature = "audio-backend")]
    _stream: Option<OutputStream>,
    #[cfg(feature = "audio-backend")]
    stream_handle: Option<OutputStreamHandle>,
}

impl SpatialAudioManager {
    /// Create a new spatial audio manager.
    ///
    /// Attempts to initialise rodio. If that fails (missing drivers, CI, etc.)
    /// the manager enters silent mode — every method still succeeds.
    pub fn new() -> Self {
        #[cfg(feature = "audio-backend")]
        {
            match OutputStream::try_default() {
                Ok((stream, handle)) => Self {
                    listener: Listener::default(),
                    sources: Vec::new(),
                    next_handle: 1,
                    attenuation_factor: DEFAULT_ATTENUATION,
                    available: true,
                    _stream: Some(stream),
                    stream_handle: Some(handle),
                },
                Err(e) => {
                    eprintln!("[ochroma-audio] rodio init failed, running silent: {e}");
                    Self {
                        listener: Listener::default(),
                        sources: Vec::new(),
                        next_handle: 1,
                        attenuation_factor: DEFAULT_ATTENUATION,
                        available: false,
                        _stream: None,
                        stream_handle: None,
                    }
                }
            }
        }
        #[cfg(not(feature = "audio-backend"))]
        {
            Self {
                listener: Listener::default(),
                sources: Vec::new(),
                next_handle: 1,
                attenuation_factor: DEFAULT_ATTENUATION,
                available: false,
            }
        }
    }

    /// Create a manager that is explicitly in silent mode (useful for tests).
    pub fn new_silent() -> Self {
        Self {
            listener: Listener::default(),
            sources: Vec::new(),
            next_handle: 1,
            attenuation_factor: DEFAULT_ATTENUATION,
            available: false,
            #[cfg(feature = "audio-backend")]
            _stream: None,
            #[cfg(feature = "audio-backend")]
            stream_handle: None,
        }
    }

    // ── Listener ──────────────────────────────────────────────────────────

    /// Update the listener position and orientation.
    pub fn set_listener(&mut self, position: Vec3, forward: Vec3, up: Vec3) {
        self.listener = Listener {
            position,
            forward: forward.normalize_or_zero(),
            up: up.normalize_or_zero(),
        };
    }

    /// Get a copy of the current listener state.
    pub fn listener(&self) -> Listener {
        self.listener
    }

    // ── Playback ──────────────────────────────────────────────────────────

    /// Play a .wav file at a 3D position. Returns a handle for later control.
    pub fn play_3d(&mut self, path: &Path, position: Vec3, volume: f32, looping: bool) -> u32 {
        let handle = self.alloc_handle();

        #[cfg(feature = "audio-backend")]
        let sink = self.try_play_file(path, volume, looping);

        self.sources.push(ActiveSource {
            handle,
            kind: SourceKind::Spatial { position },
            base_volume: volume,
            looping,
            #[cfg(feature = "audio-backend")]
            sink,
            finished: false,
            duration: None,
            elapsed: 0.0,
        });

        handle
    }

    /// Play a .wav file as a 2D global sound (music, UI). Returns a handle.
    pub fn play_2d(&mut self, path: &Path, volume: f32) -> u32 {
        let handle = self.alloc_handle();

        #[cfg(feature = "audio-backend")]
        let sink = self.try_play_file(path, volume, false);

        self.sources.push(ActiveSource {
            handle,
            kind: SourceKind::Global,
            base_volume: volume,
            looping: false,
            #[cfg(feature = "audio-backend")]
            sink,
            finished: false,
            duration: None,
            elapsed: 0.0,
        });

        handle
    }

    /// Play a procedurally generated sine tone. Returns a handle.
    pub fn play_tone(&mut self, frequency: f32, duration: f32, volume: f32) -> u32 {
        let handle = self.alloc_handle();

        #[cfg(feature = "audio-backend")]
        let sink = self.try_play_tone(frequency, duration, volume);

        self.sources.push(ActiveSource {
            handle,
            kind: SourceKind::Global,
            base_volume: volume,
            looping: false,
            #[cfg(feature = "audio-backend")]
            sink,
            finished: false,
            duration: Some(duration),
            elapsed: 0.0,
        });

        handle
    }

    /// Stop and remove a source by handle.
    pub fn stop(&mut self, handle: u32) {
        if let Some(idx) = self.sources.iter().position(|s| s.handle == handle) {
            let source = self.sources.remove(idx);
            #[cfg(feature = "audio-backend")]
            if let Some(sink) = source.sink {
                sink.stop();
            }
            let _ = source; // silence unused warning in no-backend mode
        }
    }

    /// Returns true if the source with the given handle is still active.
    pub fn is_playing(&self, handle: u32) -> bool {
        self.sources.iter().any(|s| s.handle == handle && !s.finished)
    }

    /// Move an existing 3D source to a new position.
    pub fn set_source_position(&mut self, handle: u32, position: Vec3) {
        if let Some(src) = self.sources.iter_mut().find(|s| s.handle == handle) {
            src.kind = SourceKind::Spatial { position };
        }
    }

    /// Number of active (non-finished) sources.
    pub fn active_count(&self) -> usize {
        self.sources.iter().filter(|s| !s.finished).count()
    }

    /// Whether the audio backend is available (rodio initialised successfully).
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Set the distance attenuation factor (default 0.1).
    pub fn set_attenuation_factor(&mut self, factor: f32) {
        self.attenuation_factor = factor;
    }

    // ── Tick ──────────────────────────────────────────────────────────────

    /// Update all spatial sources: recalculate volumes and panning, remove
    /// finished sources.
    pub fn tick(&mut self, dt: f32) {
        let listener = self.listener;
        let atten = self.attenuation_factor;

        for source in &mut self.sources {
            source.elapsed += dt;

            // Auto-finish if duration exceeded.
            if let Some(dur) = source.duration {
                if source.elapsed >= dur {
                    source.finished = true;
                }
            }

            // Check rodio sink emptiness.
            #[cfg(feature = "audio-backend")]
            if let Some(ref sink) = source.sink {
                if sink.empty() {
                    source.finished = true;
                }
            }

            // Update volume & panning for spatial sources.
            if let SourceKind::Spatial { position } = source.kind {
                let (vol, _pan) = compute_spatial(
                    position,
                    &listener,
                    source.base_volume,
                    atten,
                );

                #[cfg(feature = "audio-backend")]
                if let Some(ref sink) = source.sink {
                    sink.set_volume(vol);
                    // rodio doesn't expose per-sink panning directly, but
                    // volume-based spatial gives a reasonable approximation.
                }

                let _ = vol; // silence unused in no-backend
            }
        }

        // Remove finished sources.
        #[cfg(feature = "audio-backend")]
        self.sources.retain(|s| !s.finished);
        #[cfg(not(feature = "audio-backend"))]
        self.sources.retain(|s| !s.finished);
    }

    // ── Spatial math (public for testing) ─────────────────────────────────

    /// Compute the attenuated volume and stereo pan for a spatial source.
    ///
    /// Returns `(volume, pan)` where pan is in `[-1, 1]` (negative = left).
    pub fn compute_spatial_for(
        &self,
        source_position: Vec3,
        base_volume: f32,
    ) -> (f32, f32) {
        compute_spatial(
            source_position,
            &self.listener,
            base_volume,
            self.attenuation_factor,
        )
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn alloc_handle(&mut self) -> u32 {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }

    #[cfg(feature = "audio-backend")]
    fn try_play_file(&self, path: &Path, volume: f32, looping: bool) -> Option<Sink> {
        let handle = self.stream_handle.as_ref()?;
        let sink = Sink::try_new(handle).ok()?;
        let file = std::fs::File::open(path).ok()?;
        let reader = std::io::BufReader::new(file);
        if let Ok(decoder) = rodio::Decoder::new(reader) {
            if looping {
                sink.append(decoder.repeat_infinite());
            } else {
                sink.append(decoder);
            }
        }
        sink.set_volume(volume);
        Some(sink)
    }

    #[cfg(feature = "audio-backend")]
    fn try_play_tone(&self, frequency: f32, duration: f32, volume: f32) -> Option<Sink> {
        let handle = self.stream_handle.as_ref()?;
        let sink = Sink::try_new(handle).ok()?;
        let source = rodio::source::SineWave::new(frequency)
            .take_duration(std::time::Duration::from_secs_f32(duration))
            .amplify(volume);
        sink.append(source);
        Some(sink)
    }
}

impl Default for SpatialAudioManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Free functions ────────────────────────────────────────────────────────

/// Pure-math spatial computation: distance attenuation + stereo pan.
///
/// `volume = base_vol / (1 + distance * attenuation_factor)`
/// `pan = dot(normalised_direction, listener_right)` clamped to `[-1, 1]`
pub fn compute_spatial(
    source_pos: Vec3,
    listener: &Listener,
    base_volume: f32,
    attenuation_factor: f32,
) -> (f32, f32) {
    let diff = source_pos - listener.position;
    let distance = diff.length();

    // Volume: inverse-distance model.
    let volume = if distance > MAX_DISTANCE {
        0.0
    } else {
        base_volume / (1.0 + distance * attenuation_factor)
    };

    // Pan: project direction onto listener right.
    let pan = if distance < 1e-6 {
        0.0
    } else {
        let dir = diff / distance;
        let right = listener.right();
        dir.dot(right).clamp(-1.0, 1.0)
    };

    (volume, pan)
}
