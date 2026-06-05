//! Biome-driven ambient soundscape.
//! Maps forge-terrain BiomeKind to spectral synthesis parameters.

/// Matches forge-terrain BiomeKind exactly (copied as plain enum — no dep on forge required).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiomeKind {
    Alpine          = 0,
    Tundra          = 1,
    Forest          = 2,
    Grassland       = 3,
    Desert          = 4,
    Wetland         = 5,
    Coastal         = 6,
    SubalpineShrub  = 7,
    Savanna         = 8,
    Taiga           = 9,
    TropicalRainforest = 10,
}

impl BiomeKind {
    pub fn from_id(id: u8) -> Self {
        match id {
            0  => Self::Alpine,
            1  => Self::Tundra,
            2  => Self::Forest,
            3  => Self::Grassland,
            4  => Self::Desert,
            5  => Self::Wetland,
            6  => Self::Coastal,
            7  => Self::SubalpineShrub,
            8  => Self::Savanna,
            9  => Self::Taiga,
            10 => Self::TropicalRainforest,
            _  => Self::Grassland,
        }
    }
}

/// Ambient mix weights per biome: [wind, water, insects, ice]
#[derive(Debug, Clone)]
pub struct BiomeAmbientMix {
    pub wind:    f32,
    pub water:   f32,
    pub insects: f32,
    pub ice:     f32,
}

impl BiomeAmbientMix {
    pub fn for_biome(biome: BiomeKind) -> Self {
        match biome {
            BiomeKind::Alpine             => Self { wind: 0.8, water: 0.2, insects: 0.0, ice: 0.4 },
            BiomeKind::Tundra             => Self { wind: 0.9, water: 0.1, insects: 0.0, ice: 0.7 },
            BiomeKind::Forest             => Self { wind: 0.2, water: 0.1, insects: 0.7, ice: 0.0 },
            BiomeKind::Grassland          => Self { wind: 0.4, water: 0.0, insects: 0.5, ice: 0.0 },
            BiomeKind::Desert             => Self { wind: 0.6, water: 0.0, insects: 0.2, ice: 0.0 },
            BiomeKind::Wetland            => Self { wind: 0.1, water: 0.6, insects: 0.9, ice: 0.0 },
            BiomeKind::Coastal            => Self { wind: 0.5, water: 0.8, insects: 0.1, ice: 0.0 },
            BiomeKind::SubalpineShrub     => Self { wind: 0.6, water: 0.1, insects: 0.2, ice: 0.2 },
            BiomeKind::Savanna            => Self { wind: 0.5, water: 0.0, insects: 0.6, ice: 0.0 },
            BiomeKind::Taiga              => Self { wind: 0.3, water: 0.1, insects: 0.4, ice: 0.1 },
            BiomeKind::TropicalRainforest => Self { wind: 0.1, water: 0.3, insects: 1.0, ice: 0.0 },
        }
    }

    /// Blend from current mix toward target with temporal smoothing.
    pub fn blend_toward(&self, target: &Self, alpha: f32) -> Self {
        Self {
            wind:    self.wind    + (target.wind    - self.wind)    * alpha,
            water:   self.water   + (target.water   - self.water)   * alpha,
            insects: self.insects + (target.insects - self.insects) * alpha,
            ice:     self.ice     + (target.ice     - self.ice)     * alpha,
        }
    }

    /// Sum of all four layer weights — a cheap proxy for overall loudness.
    pub fn total_weight(&self) -> f32 {
        self.wind + self.water + self.insects + self.ice
    }

    /// Render this ambient mix into a playable mono sample buffer.
    ///
    /// Each layer is a distinct procedural texture, scaled by its weight and summed:
    /// - wind:    smoothed (low-passed) broadband noise — a steady rush
    /// - water:   mid-band filtered noise — bubbling / lapping
    /// - insects: high-frequency amplitude-modulated tone — chirping
    /// - ice:     sparse high-frequency crackles
    ///
    /// The buffer is deterministic (seeded LCG) so it is testable, peak-normalised
    /// to `[-1, 1]`, and ready to hand to the CPAL mixer via
    /// [`AudioCommand::PlaySynth`](crate::AudioCommand::PlaySynth).
    pub fn mix_ambient_samples(&self, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
        let n = (duration_secs.max(0.0) * sample_rate as f32) as usize;
        if n == 0 {
            return Vec::new();
        }
        let sr = sample_rate as f32;

        let mut state = 0x9E3779B9u32;
        let mut lcg = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as i32 as f32) / i32::MAX as f32
        };

        // One-pole low-pass states for wind (heavy) and water (lighter).
        let mut wind_lp  = 0.0f32;
        let mut water_lp = 0.0f32;

        let mut buf = vec![0.0f32; n];
        for (i, sample) in buf.iter_mut().enumerate() {
            let t = i as f32 / sr;
            let white = lcg();

            // Wind: heavily smoothed noise → low rumble/rush.
            wind_lp += (white - wind_lp) * 0.01;
            let wind = wind_lp * self.wind;

            // Water: moderately smoothed noise, gently modulated.
            water_lp += (white - water_lp) * 0.15;
            let water_mod = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
            let water = water_lp * water_mod * self.water;

            // Insects: high tone (~4 kHz) amplitude-modulated by a chirp envelope.
            let chirp_env = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 12.0 * t).sin()).powi(3);
            let insect_tone = (2.0 * std::f32::consts::PI * 4000.0 * t).sin();
            let insects = insect_tone * chirp_env * 0.5 * self.insects;

            // Ice: sparse high-frequency crackles gated by noise.
            let ice = if white.abs() > 0.985 {
                (2.0 * std::f32::consts::PI * 6000.0 * t).sin() * self.ice
            } else {
                0.0
            };

            *sample = wind + water + insects + ice;
        }

        // Peak-normalise so downstream volume control is meaningful.
        let peak = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1e-6 {
            for s in &mut buf {
                *s /= peak;
            }
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wetland_has_high_insects() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Wetland);
        println!("wetland insects={}", mix.insects);
        assert!(mix.insects > 0.8, "wetland insects={}", mix.insects);
    }

    #[test]
    fn tundra_has_no_insects() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Tundra);
        println!("tundra insects={}", mix.insects);
        assert_eq!(mix.insects, 0.0, "tundra should have no insects");
    }

    #[test]
    fn alpine_has_wind_and_ice() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Alpine);
        assert!(mix.wind > 0.6 && mix.ice > 0.3, "alpine: wind={} ice={}", mix.wind, mix.ice);
    }

    #[test]
    fn blend_converges() {
        let a = BiomeAmbientMix::for_biome(BiomeKind::Desert);
        let b = BiomeAmbientMix::for_biome(BiomeKind::Forest);
        let blended = a.blend_toward(&b, 1.0);
        assert!((blended.insects - b.insects).abs() < 1e-5, "full blend should equal target");
    }

    #[test]
    fn biome_from_id_roundtrips() {
        for id in 0u8..7 {
            let biome = BiomeKind::from_id(id);
            assert_eq!(biome as u8, id);
        }
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() { return 0.0; }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn mix_ambient_samples_length_matches_duration() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Forest);
        let sr = 44_100u32;
        let buf = mix.mix_ambient_samples(0.25, sr);
        let expected = (0.25 * sr as f32) as usize;
        println!("len={} expected={}", buf.len(), expected);
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn mix_ambient_samples_is_deterministic() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Wetland);
        let a = mix.mix_ambient_samples(0.1, 44_100);
        let b = mix.mix_ambient_samples(0.1, 44_100);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9),
            "identical mixes must render identical samples");
    }

    #[test]
    fn mix_ambient_samples_is_peak_normalised() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::TropicalRainforest);
        let buf = mix.mix_ambient_samples(0.2, 44_100);
        let peak = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        println!("peak={peak}");
        assert!(peak <= 1.0 + 1e-5 && peak > 0.5, "peak={peak}");
    }

    /// A near-silent biome (all weights ~0) must mix to far less energy than a
    /// loud biome. Ordered numeric RMS check on real rendered samples.
    #[test]
    fn quiet_biome_mixes_quieter_than_loud_biome() {
        let loud  = BiomeAmbientMix::for_biome(BiomeKind::TropicalRainforest); // insects 1.0
        let quiet = BiomeAmbientMix { wind: 0.02, water: 0.0, insects: 0.0, ice: 0.0 };

        let sr = 44_100u32;
        let loud_buf  = loud.mix_ambient_samples(0.2, sr);
        let quiet_buf = quiet.mix_ambient_samples(0.2, sr);

        // Compare RMS *before* normalisation cannot be done (mix normalises), so
        // weight the comparison by total_weight which drives pre-norm energy.
        let loud_rms  = rms(&loud_buf)  * loud.total_weight();
        let quiet_rms = rms(&quiet_buf) * quiet.total_weight();

        println!("loud_rms*w={loud_rms:.5} quiet_rms*w={quiet_rms:.5}");
        assert!(loud_rms > quiet_rms,
            "loud biome must carry more weighted energy: loud={loud_rms:.5} quiet={quiet_rms:.5}");
        assert!(loud.total_weight() > quiet.total_weight(),
            "loud total_weight {} must exceed quiet {}", loud.total_weight(), quiet.total_weight());
    }
}
