//! CPAL device backend — cross-platform (WASAPI / CoreAudio / ALSA).

#[cfg(feature = "audio-backend")]
use std::sync::{Arc, Mutex};

pub use crate::AudioCommand;

/// Builder for the CPAL audio backend.
pub struct CpalBackendBuilder {
    preferred_sample_rate: Option<u32>,
}

impl CpalBackendBuilder {
    pub fn new() -> Self {
        Self { preferred_sample_rate: None }
    }

    pub fn sample_rate(mut self, hz: u32) -> Self {
        self.preferred_sample_rate = Some(hz);
        self
    }

    /// Build the CPAL stream on the calling thread, spawn a dispatch thread for commands.
    /// Returns `None` when no audio device is available (headless CI, WSL without audio).
    #[cfg(feature = "audio-backend")]
    pub fn build(
        self,
        receiver: std::sync::mpsc::Receiver<crate::AudioCommand>,
    ) -> Option<CpalHandle> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host   = cpal::default_host();
        let device = host.default_output_device()?;
        let config = device.default_output_config().ok()?;
        let sr     = self.preferred_sample_rate.unwrap_or(config.sample_rate().0);

        let queue: Arc<Mutex<std::collections::VecDeque<(Vec<f32>, f32, usize)>>> =
            Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let queue_write = Arc::clone(&queue);

        let channels = config.channels() as usize;
        let stream_config = cpal::StreamConfig {
            channels: config.channels(),
            sample_rate: cpal::SampleRate(sr),
            buffer_size: cpal::BufferSize::Default,
        };

        let err_fn = |e| eprintln!("[ochroma-audio/cpal] stream error: {e}");

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &stream_config,
                move |data: &mut [f32], _| {
                    let mut q = queue.lock().unwrap();
                    for frame in data.chunks_mut(channels) {
                        let sample = if let Some((buf, vol, pos)) = q.front_mut() {
                            let s = buf.get(*pos).copied().unwrap_or(0.0) * *vol;
                            *pos += 1;
                            if *pos >= buf.len() { q.pop_front(); }
                            s
                        } else { 0.0 };
                        for ch in frame.iter_mut() { *ch = sample; }
                    }
                },
                err_fn,
                None,
            ).ok()?,
            _ => return None,
        };

        stream.play().ok()?;

        // Dispatch thread: receives AudioCommand, fills queue.
        // Stream is kept alive in CpalHandle (not moved into thread).
        std::thread::Builder::new()
            .name("ochroma-cpal-dispatch".into())
            .spawn(move || {
                while let Ok(cmd) = receiver.recv() {
                    match cmd {
                        crate::AudioCommand::PlaySynth { samples, volume } => {
                            if let Ok(mut q) = queue_write.lock() {
                                q.push_back((samples, volume, 0));
                            }
                        }
                        crate::AudioCommand::StopAll => {
                            if let Ok(mut q) = queue_write.lock() { q.clear(); }
                        }
                        _ => {}
                    }
                }
            })
            .ok()?;

        Some(CpalHandle { _stream: SendStream(stream) })
    }

    #[cfg(not(feature = "audio-backend"))]
    pub fn build(
        self,
        _receiver: std::sync::mpsc::Receiver<crate::AudioCommand>,
    ) -> Option<CpalHandle> {
        None
    }
}

impl Default for CpalBackendBuilder {
    fn default() -> Self { Self::new() }
}

/// Opaque handle — keeps the CPAL stream alive.
pub struct CpalHandle {
    #[cfg(feature = "audio-backend")]
    _stream: SendStream,
    #[cfg(not(feature = "audio-backend"))]
    _marker: std::marker::PhantomData<()>,
}

#[cfg(feature = "audio-backend")]
struct SendStream(cpal::Stream);
// SAFETY: cpal::Stream is not Send because of raw pointers on some platforms,
// but we only hold it in CpalHandle to keep it alive — we never send it to another thread.
#[cfg(feature = "audio-backend")]
unsafe impl Send for SendStream {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpal_backend_builder_exists() {
        let _b = CpalBackendBuilder::new();
    }

    #[test]
    fn audio_command_play_synth_roundtrip() {
        let samples = vec![0.0f32; 512];
        let cmd = crate::AudioCommand::PlaySynth { samples: samples.clone(), volume: 1.0 };
        match cmd {
            crate::AudioCommand::PlaySynth { samples: s, volume: v } => {
                println!("samples.len={} volume={v}", s.len());
                assert_eq!(s.len(), 512);
                assert!((v - 1.0).abs() < 1e-6);
            }
            _ => panic!("wrong variant"),
        }
    }
}
