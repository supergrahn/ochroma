//! fundsp-style signal processing helpers for vox_audio.
//!
//! Lightweight wrappers that apply fundsp-style combinators to Vec<f32> buffers.

/// Apply a scalar gain to a sample buffer.
pub fn apply_gain(input: &[f32], gain: f32) -> Vec<f32> {
    input.iter().map(|s| s * gain).collect()
}

/// Mix a simple exponential-decay reverb tail into the signal.
///
/// - `wet`: wet/dry ratio [0, 1].
/// - `tail_secs`: reverb tail length in seconds (added to output length).
pub fn apply_reverb_send(input: &[f32], wet: f32, tail_secs: f32) -> Vec<f32> {
    if wet < 1e-6 {
        return input.to_vec();
    }
    let sample_rate = 44_100u32;
    let tail_n      = (tail_secs * sample_rate as f32) as usize;
    let out_n       = input.len() + tail_n;
    let mut output  = vec![0.0f32; out_n];

    let dry = 1.0 - wet;
    for (i, &s) in input.iter().enumerate() {
        output[i] += s * dry;
    }

    let decay_rate = -6.9 / tail_secs.max(1e-4);
    let mut state  = 0xDEADBEEFu32;
    for (i, &s) in input.iter().enumerate() {
        if s.abs() < 1e-6 { continue; }
        for j in 0..tail_n {
            let t        = j as f32 / sample_rate as f32;
            let envelope = (decay_rate * t).exp();
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (state as i32 as f32) / i32::MAX as f32;
            output[i + j] += s * wet * envelope * noise * 0.1;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_gain_zero_silences_signal() {
        let input  = vec![1.0f32; 128];
        let output = apply_gain(&input, 0.0);
        assert!(output.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn apply_gain_one_is_passthrough() {
        let input  = vec![0.5f32; 128];
        let output = apply_gain(&input, 1.0);
        for (i, o) in input.iter().zip(output.iter()) {
            assert!((i - o).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_reverb_send_lengthens_signal() {
        let input  = vec![1.0f32; 256];
        let output = apply_reverb_send(&input, 0.5, 0.2);
        println!("input.len={} output.len={}", input.len(), output.len());
        assert!(output.len() > input.len());
    }

    #[test]
    fn apply_reverb_send_zero_wetness_passthrough() {
        let input  = vec![0.3f32; 256];
        let output = apply_reverb_send(&input, 0.0, 0.1);
        assert_eq!(output.len(), input.len());
        for (i, o) in input.iter().zip(output.iter()) {
            assert!((i - o).abs() < 1e-4, "i={i} o={o}");
        }
    }
}
