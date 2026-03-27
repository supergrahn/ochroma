//! Procedural sound synthesis and WAV export.
//!
//! Generates audio samples entirely in software -- no system audio libraries
//! required for compilation or testing. Useful for UI sounds, notifications,
//! and placeholder audio during development.

use std::f32::consts::PI;

/// Generate a sine-wave tone as raw PCM samples in the range `[-1.0, 1.0]`.
pub fn generate_tone(frequency: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (t * frequency * 2.0 * PI).sin()
        })
        .collect()
}

/// Generate a short UI click sound (brief high-frequency burst).
pub fn generate_click() -> Vec<f32> {
    let sample_rate = 44100u32;
    let duration = 0.05f32;
    let num_samples = (sample_rate as f32 * duration) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let decay = 1.0 - (i as f32 / num_samples as f32);
            (t * 800.0 * 2.0 * PI).sin() * decay
        })
        .collect()
}

/// Generate a placement / build sound (mid-frequency tone with linear decay).
pub fn generate_place_sound() -> Vec<f32> {
    let sample_rate = 44100u32;
    let duration = 0.15f32;
    let num_samples = (sample_rate as f32 * duration) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let decay = 1.0 - (i as f32 / num_samples as f32);
            (t * 440.0 * 2.0 * PI).sin() * decay
        })
        .collect()
}

/// Generate a collect / pickup sound (rising pitch with decay).
pub fn generate_collect_sound() -> Vec<f32> {
    let sample_rate = 44100u32;
    let duration = 0.2f32;
    let num_samples = (sample_rate as f32 * duration) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let freq = 440.0 + t * 2000.0; // rising pitch
            let decay = 1.0 - t / duration;
            (t * freq * 2.0 * PI).sin() * decay * 0.5
        })
        .collect()
}

/// Generate a damage / hit sound (noise-like burst with fast decay).
pub fn generate_hit_sound() -> Vec<f32> {
    let sample_rate = 44100u32;
    let duration = 0.12f32;
    let num_samples = (sample_rate as f32 * duration) as usize;
    // Simple pseudo-noise via multiple inharmonic frequencies.
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let decay = (-t * 30.0).exp(); // fast exponential decay
            let signal = (t * 150.0 * 2.0 * PI).sin()
                + 0.5 * (t * 370.0 * 2.0 * PI).sin()
                + 0.3 * (t * 830.0 * 2.0 * PI).sin();
            signal * decay * 0.4
        })
        .collect()
}

/// Save PCM samples as a 16-bit mono WAV file.
pub fn save_wav(samples: &[f32], sample_rate: u32, path: &std::path::Path) -> Result<(), String> {
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut data = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&file_size.to_le_bytes());
    data.extend_from_slice(b"WAVE");

    // fmt sub-chunk
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&16u32.to_le_bytes()); // sub-chunk size
    data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    data.extend_from_slice(&1u16.to_le_bytes()); // mono
    data.extend_from_slice(&sample_rate.to_le_bytes());
    data.extend_from_slice(&byte_rate.to_le_bytes());
    data.extend_from_slice(&2u16.to_le_bytes()); // block align
    data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data sub-chunk
    data.extend_from_slice(b"data");
    data.extend_from_slice(&data_size.to_le_bytes());

    for &sample in samples {
        let s16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        data.extend_from_slice(&s16.to_le_bytes());
    }

    std::fs::write(path, data).map_err(|e| e.to_string())
}
