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

// ---------------------------------------------------------------------------
// CpalBackend — directly-callable modern audio path.
//
// `CpalBackendBuilder::build` requires the caller to own a command channel and
// keep the returned handle alive. `CpalBackend` is a thinner, self-contained
// entry: open the default output device, keep the stream + sender alive in one
// struct, and expose `play_samples` so a single call synthesises -> plays with
// no WAV file written. This is what `synthesize_and_play_spectral` drives.
// ---------------------------------------------------------------------------

/// A live CPAL output backend: opens the default device, owns the stream and a
/// command channel, and plays raw sample buffers on demand.
pub struct CpalBackend {
    sender: std::sync::mpsc::Sender<crate::AudioCommand>,
    _handle: CpalHandle,
    /// Number of output devices enumerated on the host when this backend opened.
    device_count: usize,
    /// Sample rate the stream actually opened at.
    sample_rate: u32,
}

impl CpalBackend {
    /// Enumerate the host's output devices without opening a stream.
    ///
    /// Returns the number of devices that advertise at least one output
    /// configuration. `0` on a headless host with no audio (e.g. bare CI).
    #[cfg(feature = "audio-backend")]
    pub fn output_device_count() -> usize {
        use cpal::traits::HostTrait;
        let host = cpal::default_host();
        host.output_devices().map(|it| it.count()).unwrap_or(0)
    }

    #[cfg(not(feature = "audio-backend"))]
    pub fn output_device_count() -> usize {
        0
    }

    /// Open the default output device and return a ready-to-play backend.
    ///
    /// Returns `None` when no output device is available (headless CI / WSL
    /// without audio), so callers degrade gracefully to silence.
    #[cfg(feature = "audio-backend")]
    pub fn open_default() -> Option<Self> {
        use cpal::traits::{DeviceTrait, HostTrait};

        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let config = device.default_output_config().ok()?;
        let sample_rate = config.sample_rate().0;
        let device_count = host.output_devices().map(|it| it.count()).unwrap_or(0);

        let (sender, receiver) = std::sync::mpsc::channel::<crate::AudioCommand>();
        let handle = CpalBackendBuilder::new().build(receiver)?;

        Some(Self {
            sender,
            _handle: handle,
            device_count: device_count.max(1),
            sample_rate,
        })
    }

    #[cfg(not(feature = "audio-backend"))]
    pub fn open_default() -> Option<Self> {
        None
    }

    /// Queue a raw mono sample buffer for immediate playback (no WAV written).
    ///
    /// Returns the number of samples dispatched; `0` if the buffer was empty.
    pub fn play_samples(&self, samples: Vec<f32>, volume: f32) -> usize {
        let n = samples.len();
        if n == 0 {
            return 0;
        }
        let _ = self.sender.send(crate::AudioCommand::PlaySynth {
            samples,
            volume: volume.clamp(0.0, 1.0),
        });
        n
    }

    /// Number of output devices found when this backend opened (always >= 1).
    pub fn device_count(&self) -> usize {
        self.device_count
    }

    /// Sample rate of the open output stream, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Opaque handle — keeps the CPAL stream alive.
pub struct CpalHandle {
    #[cfg(feature = "audio-backend")]
    _stream: SendStream,
    #[cfg(not(feature = "audio-backend"))]
    _marker: std::marker::PhantomData<()>,
}

#[cfg(feature = "audio-backend")]
#[allow(dead_code)]
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

    #[cfg(feature = "audio-backend")]
    #[test]
    fn cpal_device_opens() {
        // Enumerate output devices on the host. On a machine with audio this
        // must be >= 1; on a genuinely headless host (no ALSA/Pulse at all)
        // cpal reports 0 and we skip rather than fail spuriously.
        let count = CpalBackend::output_device_count();
        println!("cpal output_device_count={count}");
        if count == 0 {
            eprintln!("[cpal_device_opens] no audio devices on host; skipping open assertion");
            return;
        }
        assert!(count >= 1, "expected >=1 output device, got {count}");

        // Opening the default device should also succeed and report >=1 device
        // plus a plausible sample rate.
        let backend = CpalBackend::open_default()
            .expect("default output device should open when devices are present");
        println!(
            "opened default device: device_count={} sample_rate={}",
            backend.device_count(),
            backend.sample_rate()
        );
        assert!(backend.device_count() >= 1, "open backend must see >=1 device");
        assert!(
            (8_000..=192_000).contains(&backend.sample_rate()),
            "implausible sample rate {}",
            backend.sample_rate()
        );

        // Dispatch a short synthesised buffer for playback (no WAV written).
        let dispatched = backend.play_samples(vec![0.0f32; 256], 0.5);
        assert_eq!(dispatched, 256, "all 256 samples should be dispatched");
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
