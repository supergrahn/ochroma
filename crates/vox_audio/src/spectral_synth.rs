//! Spectral audio synthesis.
//!
//! Maps Ochroma's 16 spectral bands (380nm–755nm at 25nm steps) to audio resonance
//! frequencies via a physically-motivated psychoacoustic mapping:
//!
//!   Band 0  (380nm, UV-violet)    → 16 kHz  (very bright, crystalline)
//!   Band 1  (405nm, violet)       →  8 kHz  (bright, glassy)
//!   Band 2  (430nm, violet-blue)  →  6 kHz
//!   Band 3  (455nm, blue)         →  4 kHz
//!   Band 4  (480nm, cyan-blue)    →  3 kHz
//!   Band 5  (505nm, cyan)         →  2 kHz
//!   Band 6  (530nm, green)        →  1.5 kHz
//!   Band 7  (555nm, yellow-green) →  1 kHz  (mid)
//!   Band 8  (580nm, yellow)       →  700 Hz
//!   Band 9  (605nm, orange-yellow)→  500 Hz
//!   Band 10 (630nm, orange)       →  350 Hz
//!   Band 11 (655nm, red-orange)   →  250 Hz
//!   Band 12 (680nm, red)          →  180 Hz
//!   Band 13 (705nm, deep red)     →  125 Hz
//!   Band 14 (730nm, near-IR)      →   80 Hz
//!   Band 15 (755nm, near-IR)      →   50 Hz  (deep, rumbling)

pub const FREQ_MAP: [f32; 16] = [
    16000.0, 8000.0, 6000.0, 4000.0, 3000.0, 2000.0, 1500.0, 1000.0,
    700.0,   500.0,  350.0,  250.0,  180.0,  125.0,   80.0,   50.0,
];

pub fn synthesize_impact(spectral_weights: &[f32; 16], duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut output = vec![0.0f32; n_samples];

    for (band, &freq) in FREQ_MAP.iter().enumerate() {
        let weight = spectral_weights[band];
        if weight < 0.01 { continue; }
        let decay_rate = -8.0 - (band as f32 * 1.0); // -8 to -23
        for (i, sample) in output.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            let envelope = (decay_rate * t).exp();
            *sample += weight * envelope * (2.0 * std::f32::consts::PI * freq * t).sin();
        }
    }

    let peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.001 {
        for s in &mut output { *s /= peak; }
    }
    output
}

pub fn create_impact_wav(spectral_weights: &[f32; 16], duration_secs: f32) -> std::path::PathBuf {
    let samples = synthesize_impact(spectral_weights, duration_secs, 44100);
    let path = std::env::temp_dir().join("ochroma_impact.wav");
    crate::synth::save_wav(&samples, 44100, &path).expect("failed to write impact WAV");
    path
}

pub fn synthesize_impact_from_splat_spectral(
    splat_spectral: &[u16; 16],
    duration_secs: f32,
    sample_rate: u32,
) -> Vec<f32> {
    let weights: [f32; 16] = std::array::from_fn(|i| {
        half::f16::from_bits(splat_spectral[i]).to_f32().clamp(0.0, 1.0)
    });
    synthesize_impact(&weights, duration_secs, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_impact_returns_correct_length() {
        let weights = [0.5f32; 16];
        let samples = synthesize_impact(&weights, 0.1, 44100);
        assert_eq!(samples.len(), 4410, "0.1s × 44100Hz = 4410 samples");
    }

    #[test]
    fn synthesize_impact_is_normalized() {
        let weights = [1.0f32; 16];
        let samples = synthesize_impact(&weights, 0.1, 44100);
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak <= 1.0 + 1e-5, "peak should be ≤ 1.0, got {}", peak);
    }

    #[test]
    fn synthesize_impact_all_zero_returns_silence() {
        let weights = [0.0f32; 16];
        let samples = synthesize_impact(&weights, 0.1, 44100);
        assert!(samples.iter().all(|&s| s == 0.0), "all-zero weights should produce silence");
    }

    #[test]
    fn high_blue_weight_sounds_different_from_high_red_weight() {
        let mut blue_weights = [0.0f32; 16];
        blue_weights[0] = 1.0;
        let mut red_weights = [0.0f32; 16];
        red_weights[15] = 1.0;
        let blue = synthesize_impact(&blue_weights, 0.1, 44100);
        let red = synthesize_impact(&red_weights, 0.1, 44100);
        let diff: f32 = blue.iter().zip(red.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "different spectral weights should produce different audio, diff={}", diff);
    }

    #[test]
    fn create_impact_wav_creates_file() {
        let weights = [0.3f32; 16];
        let path = create_impact_wav(&weights, 0.05);
        assert!(path.exists(), "WAV file should exist at {:?}", path);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn synthesize_impact_from_splat_spectral_round_trips() {
        let f16_half = half::f16::from_f32(0.5).to_bits();
        let splat_spectral = [f16_half; 16];
        let samples = synthesize_impact_from_splat_spectral(&splat_spectral, 0.1, 44100);
        assert_eq!(samples.len(), 4410);
    }
}
