pub mod acoustic_raytracer;
pub mod audio_graph;
pub mod spatial;
pub mod synth;

pub use spatial::{compute_spatial, Listener, SpatialAudioManager};
pub use synth::{generate_click, generate_collect_sound, generate_place_sound, generate_tone, save_wav};

use glam::Vec3;

#[cfg(feature = "audio-backend")]
use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
#[cfg(feature = "audio-backend")]
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AudioSource {
    pub id: u32,
    pub position: Vec3,
    pub volume: f32,
    pub looping: bool,
    pub clip: String,
}

#[cfg(feature = "audio-backend")]
pub struct AudioBackend {
    _stream: OutputStream, // must keep alive
    stream_handle: OutputStreamHandle,
    sinks: HashMap<u32, Sink>,
}

#[cfg(feature = "audio-backend")]
impl AudioBackend {
    pub fn new() -> Option<Self> {
        match OutputStream::try_default() {
            Ok((stream, handle)) => Some(Self {
                _stream: stream,
                stream_handle: handle,
                sinks: HashMap::new(),
            }),
            Err(e) => {
                eprintln!("[ochroma-audio] Failed to initialize audio: {}", e);
                None
            }
        }
    }

    pub fn play_sine(&mut self, id: u32, frequency: f32, duration_secs: f32, volume: f32) {
        if let Ok(sink) = Sink::try_new(&self.stream_handle) {
            let source = rodio::source::SineWave::new(frequency)
                .take_duration(std::time::Duration::from_secs_f32(duration_secs))
                .amplify(volume);
            sink.append(source);
            self.sinks.insert(id, sink);
        }
    }

    pub fn stop(&mut self, id: u32) {
        if let Some(sink) = self.sinks.remove(&id) {
            sink.stop();
        }
    }

    pub fn is_playing(&self, id: u32) -> bool {
        self.sinks.get(&id).map(|s| !s.empty()).unwrap_or(false)
    }

    /// Clean up finished sinks.
    pub fn tick(&mut self) {
        self.sinks.retain(|_, sink| !sink.empty());
    }
}

pub struct AudioEngine {
    max_sources: usize,
    pub sources: Vec<AudioSource>,
    next_id: u32,
    pub listener_position: Vec3,
    #[cfg(feature = "audio-backend")]
    pub backend: Option<AudioBackend>,
}

impl AudioEngine {
    pub fn new(max_sources: usize) -> Self {
        Self {
            max_sources,
            sources: Vec::new(),
            next_id: 1,
            listener_position: Vec3::ZERO,
            #[cfg(feature = "audio-backend")]
            backend: None,
        }
    }

    #[cfg(feature = "audio-backend")]
    pub fn init_backend(&mut self) {
        self.backend = AudioBackend::new();
    }

    #[cfg(not(feature = "audio-backend"))]
    pub fn init_backend(&mut self) {
        // No audio backend available — rodio requires libasound2-dev on Linux.
        // Install with: sudo apt-get install libasound2-dev
        // Then build with: cargo build --features audio-backend
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

    /// Play a sine tone through the backend (if available).
    #[cfg(feature = "audio-backend")]
    pub fn play_sine_backend(&mut self, id: u32, frequency: f32, duration_secs: f32, volume: f32) {
        if let Some(ref mut backend) = self.backend {
            backend.play_sine(id, frequency, duration_secs, volume);
        }
    }

    #[cfg(not(feature = "audio-backend"))]
    pub fn play_sine_backend(&mut self, _id: u32, _frequency: f32, _duration_secs: f32, _volume: f32) {
        // No backend — silent
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
        #[cfg(feature = "audio-backend")]
        if let Some(backend) = &mut self.backend {
            backend.tick();
        }
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new(64)
    }
}
