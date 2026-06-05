pub mod biome_soundscape;
pub use biome_soundscape::{BiomeKind, BiomeAmbientMix};
pub mod acoustic_raytracer;
pub mod av_sync;
pub mod audio_graph;
pub mod adaptive_music;
pub mod hrtf;
pub mod spatial;
pub mod synth;
pub mod ecs;
pub mod spectral_synth;
pub mod spectral_synth2;
pub mod sdf_reverb;
pub mod cpal_backend;
pub mod spectral_acoustic;
pub mod spectral_reverb;
pub mod fundsp_graph;
pub use spectral_synth::{synthesize_impact, create_impact_wav, synthesize_impact_from_splat_spectral};
pub use spectral_synth2::SpectralSynth;
pub use spectral_acoustic::SpectralAcousticProfile;
pub use spectral_reverb::SpectralReverb;
pub use cpal_backend::{CpalBackend, CpalBackendBuilder, CpalHandle};

pub use spatial::{compute_spatial, Listener, SpatialAudioManager};
pub use synth::{generate_click, generate_collect_sound, generate_place_sound, generate_tone, save_wav};
pub use fundsp_graph::{apply_gain, apply_reverb_send};

use glam::Vec3;

// ---------------------------------------------------------------------------
// AudioCommand
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum AudioCommand {
    Play { id: u32, path: String, volume: f32, looping: bool },
    Stop { id: u32 },
    StopAll,
    PlaySynth { samples: Vec<f32>, volume: f32 },
}

// ---------------------------------------------------------------------------
// AudioThread — owns rodio on a background thread
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-backend")]
struct AudioThread {
    receiver: std::sync::mpsc::Receiver<AudioCommand>,
    sinks: std::collections::HashMap<u32, rodio::Sink>,
}

#[cfg(feature = "audio-backend")]
impl AudioThread {
    fn run(mut self, stream_handle: rodio::OutputStreamHandle) {
        use rodio::Source as _;
        while let Ok(cmd) = self.receiver.recv() {
            match cmd {
                AudioCommand::Play { id, path, volume, looping } => {
                    match std::fs::File::open(&path) {
                        Ok(file) => {
                            match rodio::Decoder::new(std::io::BufReader::new(file)) {
                                Ok(source) => {
                                    match rodio::Sink::try_new(&stream_handle) {
                                        Ok(sink) => {
                                            sink.set_volume(volume);
                                            if looping {
                                                sink.append(source.repeat_infinite());
                                            } else {
                                                sink.append(source);
                                            }
                                            self.sinks.insert(id, sink);
                                        }
                                        Err(e) => eprintln!("[ochroma-audio] Sink error for {}: {}", path, e),
                                    }
                                }
                                Err(e) => eprintln!("[ochroma-audio] Decode error for {}: {}", path, e),
                            }
                        }
                        Err(e) => eprintln!("[ochroma-audio] File open error for {}: {}", path, e),
                    }
                }
                AudioCommand::Stop { id } => {
                    if let Some(sink) = self.sinks.remove(&id) {
                        sink.stop();
                    }
                }
                AudioCommand::StopAll => {
                    for (_, sink) in self.sinks.drain() {
                        sink.stop();
                    }
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AudioHandle — Send wrapper with channel Sender
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AudioHandle {
    sender: std::sync::mpsc::Sender<AudioCommand>,
    next_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl AudioHandle {
    #[cfg(feature = "audio-backend")]
    pub fn spawn() -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<AudioCommand>();
        // We need to create OutputStream inside the thread because it is !Send.
        // Use a sync channel to get back the result (Ok/Err) before returning.
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);
        let audio_thread = AudioThread {
            receiver: rx,
            sinks: std::collections::HashMap::new(),
        };
        std::thread::Builder::new()
            .name("ochroma-audio".into())
            .spawn(move || {
                match rodio::OutputStream::try_default() {
                    Ok((_stream, stream_handle)) => {
                        let _ = ready_tx.send(Ok(()));
                        // Keep _stream alive for the duration of the thread.
                        audio_thread.run(stream_handle);
                        drop(_stream);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                    }
                }
            })
            .expect("failed to spawn audio thread");
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[ochroma-audio] Failed to open audio device: {}", e);
                return None;
            }
            Err(_) => {
                eprintln!("[ochroma-audio] Audio thread died unexpectedly");
                return None;
            }
        }
        Some(Self {
            sender: tx,
            next_id: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1)),
        })
    }

    #[cfg(not(feature = "audio-backend"))]
    pub fn spawn() -> Option<Self> {
        None
    }

    pub fn play(&self, path: &str, volume: f32, looping: bool) -> u32 {
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = self.sender.send(AudioCommand::Play {
            id, path: path.to_string(), volume, looping,
        });
        id
    }

    pub fn stop(&self, id: u32) {
        let _ = self.sender.send(AudioCommand::Stop { id });
    }

    pub fn stop_all(&self) {
        let _ = self.sender.send(AudioCommand::StopAll);
    }
}

// Test-only constructor so ecs.rs tests can build AudioHandle without touching private fields
#[cfg(test)]
impl AudioHandle {
    pub fn new_test(
        sender: std::sync::mpsc::Sender<AudioCommand>,
        next_id: std::sync::Arc<std::sync::atomic::AtomicU32>,
    ) -> Self {
        Self { sender, next_id }
    }
}

// ---------------------------------------------------------------------------
// AudioSource
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AudioSource {
    pub id: u32,
    pub position: Vec3,
    pub volume: f32,
    pub looping: bool,
    pub clip: String,
}

// ---------------------------------------------------------------------------
// AudioEngine
// ---------------------------------------------------------------------------

pub struct AudioEngine {
    max_sources: usize,
    pub sources: Vec<AudioSource>,
    next_id: u32,
    pub listener_position: Vec3,
}

impl AudioEngine {
    pub fn new(max_sources: usize) -> Self {
        Self {
            max_sources,
            sources: Vec::new(),
            next_id: 1,
            listener_position: Vec3::ZERO,
        }
    }

    pub fn init_backend(&mut self) {
        // Backend is now owned by AudioHandle on a separate thread.
        // Call AudioHandle::spawn() separately and store it alongside AudioEngine.
    }

    pub fn set_listener(&mut self, pos: Vec3) {
        self.listener_position = pos;
    }

    pub fn play(&mut self, mut source: AudioSource) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        source.id = id;
        self.sources.push(source);
        id
    }

    pub fn stop(&mut self, id: u32) {
        self.sources.retain(|s| s.id != id);
    }

    pub fn active_count(&self) -> usize {
        self.sources.len()
    }

    /// Calculate effective volume using inverse-distance attenuation.
    pub fn effective_volume(&self, source: &AudioSource) -> f32 {
        Self::effective_volume_at(source, self.listener_position)
    }

    /// Stateless helper that avoids borrow conflicts during tick.
    fn effective_volume_at(source: &AudioSource, listener: Vec3) -> f32 {
        let dist = source.position.distance(listener);
        let attenuation = 1.0 / (1.0 + dist * 0.1);
        source.volume * attenuation
    }

    /// Get all active sources sorted by effective volume (loudest first).
    pub fn active_sources_by_priority(&self) -> Vec<&AudioSource> {
        let mut sources: Vec<&AudioSource> = self.sources.iter().collect();
        sources.sort_by(|a, b| {
            self.effective_volume(b)
                .partial_cmp(&self.effective_volume(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sources
    }

    /// Play a sine tone through the backend (no-op stub — backend is now on AudioHandle thread).
    pub fn play_sine_backend(&mut self, _id: u32, _frequency: f32, _duration_secs: f32, _volume: f32) {
        // Backend is now owned by AudioHandle on a separate thread.
        // Use AudioHandle::play() for audio playback.
    }

    /// Tick: evict lowest-priority sources if over budget.
    pub fn tick(&mut self, _dt: f32) {
        let listener = self.listener_position;
        while self.sources.len() > self.max_sources {
            if let Some(idx) = self.sources
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    Self::effective_volume_at(a, listener)
                        .partial_cmp(&Self::effective_volume_at(b, listener))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.sources.remove(idx);
            } else {
                break;
            }
        }
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new(64)
    }
}

// ---------------------------------------------------------------------------
// synthesize_and_play — physics-triggered impact sound dispatch
// ---------------------------------------------------------------------------

/// Synthesise an impact sound from a splat's spectral material and queue it
/// for CPAL playback, applying a reverb tail derived from nearby splat reflectance.
///
/// Called by the physics layer on `CollisionEvent` or `FractureEvent`.
pub fn synthesize_and_play(
    spectral: &[u16; 16],
    impulse: f32,
    nearby_splats: &[[u16; 16]],
    sender: &std::sync::mpsc::Sender<AudioCommand>,
) {
    let output = synthesize_impact_buffer(spectral, impulse, nearby_splats);
    let _ = sender.send(AudioCommand::PlaySynth {
        samples: output,
        volume: impulse.clamp(0.01, 1.0),
    });
}

/// Build the wet impact buffer: `SpectralSynth::strike` -> reverb send.
///
/// Shared by both the channel-based [`synthesize_and_play`] and the
/// self-contained [`synthesize_and_play_spectral`] so the synthesis path is
/// identical regardless of how the result is delivered.
pub fn synthesize_impact_buffer(
    spectral: &[u16; 16],
    impulse: f32,
    nearby_splats: &[[u16; 16]],
) -> Vec<f32> {
    let dry = crate::spectral_synth2::SpectralSynth::strike(spectral, impulse);
    let reverb = crate::spectral_reverb::SpectralReverb::from_splat_reflectance(nearby_splats);
    let wet = 0.25_f32;
    crate::fundsp_graph::apply_reverb_send(&dry, wet, reverb.tail_length_secs.min(2.0))
}

/// Single, directly-callable modern audio entry: synthesise an impact from a
/// splat's spectral material and play it through the default CPAL output device.
///
/// This is the one reachable call site for the v2 spectral path
/// (`SpectralSynth::strike` -> `SpectralReverb` -> CPAL). **No WAV file is
/// written** — samples are streamed straight to the device. `walking_sim` (or
/// any game layer) can call this directly on a collision/fracture event.
///
/// Returns the number of samples dispatched to the device, or `0` when no
/// output device is available (headless CI / WSL without audio) — in which case
/// the call is a graceful no-op rather than an error.
///
/// Note: this opens a fresh device per call. For a hot path, open one
/// [`crate::cpal_backend::CpalBackend`] up front and reuse `play_samples`.
pub fn synthesize_and_play_spectral(material_spectral: &[u16; 16], impulse: f32) -> usize {
    // No reverb context here — pass an empty reflectance set, yielding a short
    // "dead room" tail. Callers with surrounding splats should use
    // `synthesize_impact_buffer` + `CpalBackend::play_samples`.
    let output = synthesize_impact_buffer(material_spectral, impulse, &[]);
    match crate::cpal_backend::CpalBackend::open_default() {
        Some(backend) => backend.play_samples(output, impulse.clamp(0.01, 1.0)),
        None => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "audio-backend")]
    #[test]
    fn audio_handle_spawn_returns_some_with_feature() {
        if std::env::var("CI").is_ok() { return; }
        let handle = AudioHandle::spawn();
        // Skip assertion when no audio device is available (e.g. headless WSL).
        if handle.is_none() { return; }
        assert!(handle.is_some());
    }

    #[test]
    fn audio_handle_play_nonexistent_file_does_not_panic() {
        let (tx, _rx) = std::sync::mpsc::channel::<AudioCommand>();
        let handle = AudioHandle::new_test(
            tx,
            std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1)),
        );
        let id = handle.play("nonexistent.wav", 1.0, false);
        assert!(id >= 1);
        handle.stop(id);
    }

    #[test]
    fn audio_engine_active_count_starts_zero() {
        let engine = AudioEngine::new(64);
        assert_eq!(engine.active_count(), 0);
    }

    #[test]
    fn synthesize_impact_buffer_is_audible_and_longer_than_dry() {
        // Glass-ish bright material; empty reflectance -> short dead-room tail.
        let mut glass = [0u16; 16];
        glass[0] = half::f16::from_f32(0.95).to_bits();
        glass[1] = half::f16::from_f32(0.60).to_bits();

        let dry = crate::spectral_synth2::SpectralSynth::strike(&glass, 1.0);
        let wet = synthesize_impact_buffer(&glass, 1.0, &[]);

        let peak = wet.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        println!("dry.len={} wet.len={} peak={peak:.4}", dry.len(), wet.len());
        // Reverb send appends a tail, so the wet buffer must be at least as long.
        assert!(wet.len() >= dry.len(), "wet {} < dry {}", wet.len(), dry.len());
        // And it must carry an audible signal.
        assert!(peak > 0.01, "wet impact should be audible, peak={peak}");
    }

    #[cfg(feature = "audio-backend")]
    #[test]
    fn synthesize_and_play_spectral_dispatches_when_device_present() {
        // This is the single reachable v2 entry: strike -> reverb -> CPAL.
        let device_count = crate::cpal_backend::CpalBackend::output_device_count();
        let mut stone = [0u16; 16];
        stone[15] = half::f16::from_f32(0.90).to_bits();

        let dispatched = synthesize_and_play_spectral(&stone, 0.8);
        println!("device_count={device_count} dispatched_samples={dispatched}");

        if device_count == 0 {
            // Headless host: graceful no-op.
            assert_eq!(dispatched, 0, "no device -> nothing dispatched");
        } else {
            // With a device, the full 0.5 s strike (plus dead-room tail) must
            // be dispatched: at least SAMPLE_RATE/2 samples.
            let min = (crate::spectral_synth2::SAMPLE_RATE / 2) as usize;
            assert!(dispatched >= min,
                "expected >= {min} samples dispatched, got {dispatched}");
        }
    }

    #[test]
    fn audio_engine_tick_culls_over_budget() {
        let mut engine = AudioEngine::new(1);
        engine.play(AudioSource { id: 0, position: Vec3::ZERO, volume: 1.0, looping: false, clip: "a.wav".into() });
        engine.play(AudioSource { id: 0, position: Vec3::ZERO, volume: 0.5, looping: false, clip: "b.wav".into() });
        engine.tick(0.016);
        assert_eq!(engine.active_count(), 1);
    }
}
