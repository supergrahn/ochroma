//! SpectralSynth — synthesises impact sounds from GaussianSplat spectral profiles.

const FREQ_MAP: [f32; 16] = [12000.0, 10000.0, 8000.0, 6500.0, 5000.0, 4000.0, 3000.0, 2200.0, 1500.0, 1100.0, 800.0, 580.0, 400.0, 300.0, 250.0, 200.0];

pub const SAMPLE_RATE: u32 = 44_100;
pub const HARMONICS:   u32 = 4;

pub struct SpectralSynth;

impl SpectralSynth {
    /// Weighted-average resonance frequency from spectral material profile.
    /// Short-wavelength (band 0) → high Hz. Long-wavelength (band 15) → low Hz.
    pub fn resonance_freq(spectral: &[u16; 16]) -> f32 {
        let mut weight_sum = 0.0f32;
        let mut freq_sum   = 0.0f32;
        for (band, &freq) in FREQ_MAP.iter().enumerate() {
            let w = half::f16::from_bits(spectral[band]).to_f32().max(0.0);
            weight_sum += w;
            freq_sum   += w * freq;
        }
        if weight_sum < 1e-6 { return 440.0; }
        freq_sum / weight_sum
    }

    fn damping(spectral: &[u16; 16]) -> f32 {
        let weights: Vec<f32> = spectral.iter()
            .map(|&b| half::f16::from_bits(b).to_f32().max(0.0))
            .collect();
        let mean = weights.iter().sum::<f32>() / 16.0;
        let var  = weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / 16.0;
        3.0 + (var / 0.25).min(1.0) * 17.0
    }

    pub fn strike(spectral: &[u16; 16], impulse: f32) -> Vec<f32> {
        let n_samples = (SAMPLE_RATE as f32 * 0.5) as usize;
        let mut buf   = vec![0.0f32; n_samples];

        let fundamental = Self::resonance_freq(spectral);
        let decay       = -Self::damping(spectral);

        for harmonic in 0..HARMONICS {
            let freq   = fundamental * (harmonic + 1) as f32;
            let amp    = impulse / (harmonic + 1) as f32;
            let weight = Self::band_weight_at_freq(spectral, freq);
            if weight < 1e-4 { continue; }
            for (i, sample) in buf.iter_mut().enumerate() {
                let t        = i as f32 / SAMPLE_RATE as f32;
                let envelope = (decay * t).exp();
                *sample     += amp * weight * envelope
                    * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
        }

        let peak = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1e-6 {
            for s in &mut buf { *s /= peak; }
        }
        buf
    }

    fn band_weight_at_freq(spectral: &[u16; 16], freq: f32) -> f32 {
        let freq = freq.clamp(FREQ_MAP[15], FREQ_MAP[0]);
        for i in 0..15 {
            let hi = FREQ_MAP[i];
            let lo = FREQ_MAP[i + 1];
            if freq <= hi && freq >= lo {
                let t  = (hi - freq) / (hi - lo);
                let w0 = half::f16::from_bits(spectral[i]).to_f32().max(0.0);
                let w1 = half::f16::from_bits(spectral[i + 1]).to_f32().max(0.0);
                return w0 * (1.0 - t) + w1 * t;
            }
        }
        half::f16::from_bits(spectral[15]).to_f32().max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resonance_freq_blue_material_is_high() {
        let mut spectral = [0u16; 16];
        spectral[0] = half::f16::from_f32(1.0).to_bits();
        let hz = SpectralSynth::resonance_freq(&spectral);
        println!("blue resonance_hz={hz}");
        assert!(hz > 4000.0, "blue-dominant material resonance={hz}");
    }

    #[test]
    fn resonance_freq_red_material_is_low() {
        let mut spectral = [0u16; 16];
        spectral[15] = half::f16::from_f32(1.0).to_bits();
        let hz = SpectralSynth::resonance_freq(&spectral);
        println!("red resonance_hz={hz}");
        assert!(hz < 500.0, "red-dominant material resonance={hz}");
    }

    #[test]
    fn strike_returns_nonempty_audio() {
        let spectral = [half::f16::from_f32(0.5).to_bits(); 16];
        let samples = SpectralSynth::strike(&spectral, 1.0);
        assert!(!samples.is_empty());
    }

    #[test]
    fn strike_is_normalised() {
        let spectral = [half::f16::from_f32(1.0).to_bits(); 16];
        let samples = SpectralSynth::strike(&spectral, 1.0);
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak <= 1.0 + 1e-5, "peak={peak}");
    }

    #[test]
    fn strike_all_zero_spectral_is_silence() {
        let spectral = [0u16; 16];
        let samples = SpectralSynth::strike(&spectral, 1.0);
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn blue_strike_sounds_different_from_red_strike() {
        let mut blue = [0u16; 16]; blue[0]  = half::f16::from_f32(1.0).to_bits();
        let mut red  = [0u16; 16]; red[15]  = half::f16::from_f32(1.0).to_bits();
        let b = SpectralSynth::strike(&blue, 1.0);
        let r = SpectralSynth::strike(&red,  1.0);
        let diff: f32 = b.iter().zip(r.iter()).map(|(a,x)| (a-x).abs()).sum();
        assert!(diff > 0.1, "blue vs red should differ, diff={diff}");
    }
}
