//! SpectralAcousticProfile — derives acoustic synthesis parameters from the
//! forge-spectral material database. One material profile drives both rendering
//! and audio. No separate "audio material" needed.

/// Acoustic synthesis parameters derived from spectral reflectance.
#[derive(Debug, Clone, Copy)]
pub struct SpectralAcousticProfile {
    /// Fundamental resonance frequency in Hz.
    pub resonance_hz: f32,
    /// Quality factor: higher = longer ring / slower energy loss.
    /// Q = 0.2 (sand, soil) → Q = 15.0 (metal).
    pub q_factor: f32,
    /// Reverberation time (RT60) in seconds when surface lines a room.
    /// 0.1s (dead outdoors: sand, asphalt) → 6.0s (snow field).
    pub rt60_secs: f32,
}

impl SpectralAcousticProfile {
    /// Derive acoustic profile from arbitrary 16-band spectral data.
    pub fn from_spectral(bands_f16: &[u16; 16]) -> Self {
        const FREQ_MAP: [f32; 16] = [12000.0, 10000.0, 8000.0, 6500.0, 5000.0, 4000.0, 3000.0, 2200.0, 1500.0, 1100.0, 800.0, 580.0, 400.0, 300.0, 250.0, 200.0];
        let bands: [f32; 16] = std::array::from_fn(|i| {
            half::f16::from_bits(bands_f16[i]).to_f32().max(0.0)
        });
        let sum: f32 = bands.iter().sum();
        let mean = sum / 16.0;

        let log_freq_sum: f32 = bands.iter().zip(FREQ_MAP.iter())
            .map(|(&w, &f)| w * f.ln())
            .sum();
        let resonance_hz = if sum > 1e-6 {
            (log_freq_sum / sum).exp()
        } else {
            440.0
        };

        let variance: f32 = bands.iter().map(|&b| (b - mean).powi(2)).sum::<f32>() / 16.0;
        let q_factor = (5.0 / (variance + 0.1)).clamp(0.2, 15.0);

        let rt60_secs = (mean * 6.0).clamp(0.05, 8.0);

        Self { resonance_hz, q_factor, rt60_secs }
    }

    pub fn metal()    -> Self { Self { resonance_hz: 8000.0, q_factor: 15.0, rt60_secs: 3.5 } }
    pub fn glass()    -> Self { Self { resonance_hz: 6000.0, q_factor: 10.0, rt60_secs: 0.3 } }
    pub fn concrete() -> Self { Self { resonance_hz:  300.0, q_factor:  2.0, rt60_secs: 2.0 } }
    pub fn rock()     -> Self { Self { resonance_hz:  250.0, q_factor:  1.5, rt60_secs: 1.5 } }
    pub fn brick()    -> Self { Self { resonance_hz:  400.0, q_factor:  1.5, rt60_secs: 1.8 } }
    pub fn bark()     -> Self { Self { resonance_hz:  200.0, q_factor:  1.8, rt60_secs: 0.8 } }
    pub fn gravel()   -> Self { Self { resonance_hz:  350.0, q_factor:  0.8, rt60_secs: 0.4 } }
    pub fn soil()     -> Self { Self { resonance_hz:  100.0, q_factor:  0.4, rt60_secs: 0.1 } }
    pub fn sand()     -> Self { Self { resonance_hz:   80.0, q_factor:  0.3, rt60_secs: 0.1 } }
    pub fn asphalt()  -> Self { Self { resonance_hz:  120.0, q_factor:  0.5, rt60_secs: 0.1 } }
    pub fn snow()     -> Self { Self { resonance_hz:   60.0, q_factor:  0.2, rt60_secs: 6.0 } }
    pub fn foliage()  -> Self { Self { resonance_hz:  800.0, q_factor:  0.7, rt60_secs: 0.3 } }
    pub fn water()    -> Self { Self { resonance_hz:  800.0, q_factor:  2.5, rt60_secs: 0.4 } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_has_higher_q_than_soil() {
        let metal_q = SpectralAcousticProfile::metal().q_factor;
        println!("metal q_factor={}", metal_q as u32);
        assert!(metal_q > SpectralAcousticProfile::soil().q_factor,
            "metal sustains longer than soil");
    }

    #[test]
    fn snow_has_longest_rt60() {
        let rt60s = [
            SpectralAcousticProfile::metal().rt60_secs,
            SpectralAcousticProfile::concrete().rt60_secs,
            SpectralAcousticProfile::snow().rt60_secs,
            SpectralAcousticProfile::asphalt().rt60_secs,
        ];
        let max = rt60s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("snow rt60={}", SpectralAcousticProfile::snow().rt60_secs as u32);
        assert_eq!(max, SpectralAcousticProfile::snow().rt60_secs,
            "snow field should have longest RT60");
    }

    #[test]
    fn glass_higher_resonance_than_soil() {
        assert!(SpectralAcousticProfile::glass().resonance_hz >
                SpectralAcousticProfile::soil().resonance_hz,
            "glass rings higher than soil");
    }

    #[test]
    fn from_spectral_metal_like_profile() {
        let bands = [half::f16::from_f32(0.65).to_bits(); 16];
        let profile = SpectralAcousticProfile::from_spectral(&bands);
        assert!(profile.q_factor > 3.0,
            "flat high-reflectance should give Q > 3.0, got {}", profile.q_factor);
        assert!(profile.rt60_secs > 1.0,
            "flat high-reflectance should give RT60 > 1.0s, got {}", profile.rt60_secs);
    }

    #[test]
    fn from_spectral_dead_material() {
        let bands = [half::f16::from_f32(0.06).to_bits(); 16];
        let profile = SpectralAcousticProfile::from_spectral(&bands);
        assert!(profile.rt60_secs < 0.5,
            "dark material should give RT60 < 0.5s, got {}", profile.rt60_secs);
    }
}
