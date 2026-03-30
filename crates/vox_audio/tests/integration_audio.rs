use vox_audio::{SpectralSynth, SpectralReverb, AudioCommand, synthesize_and_play};

#[test]
fn glass_impact_produces_play_synth_command() {
    let mut glass_spectral = [0u16; 16];
    glass_spectral[0] = half::f16::from_f32(0.95).to_bits();
    glass_spectral[1] = half::f16::from_f32(0.70).to_bits();
    glass_spectral[2] = half::f16::from_f32(0.40).to_bits();

    let stone_reflectance_val = half::f16::from_f32(0.85).to_bits();
    let nearby_splats: Vec<[u16; 16]> = (0..32)
        .map(|_| [stone_reflectance_val; 16])
        .collect();

    let (tx, rx) = std::sync::mpsc::channel::<AudioCommand>();

    synthesize_and_play(&glass_spectral, 1.0, &nearby_splats, &tx);

    let cmd = rx.try_recv().expect("expected AudioCommand::PlaySynth");
    match cmd {
        AudioCommand::PlaySynth { samples, volume } => {
            let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            println!("glass impact: samples.len={} peak={peak:.4} volume={volume}", samples.len());
            assert!(!samples.is_empty(), "synthesised buffer must not be empty");
            assert!(volume > 0.0 && volume <= 1.0, "volume out of range: {volume}");
            assert!(peak > 0.01, "glass impact should produce audible signal, peak={peak}");
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn resonance_freq_of_glass_exceeds_stone() {
    let mut glass = [0u16; 16];
    glass[0] = half::f16::from_f32(1.0).to_bits();

    let mut stone = [0u16; 16];
    stone[15] = half::f16::from_f32(1.0).to_bits();

    let glass_hz = SpectralSynth::resonance_freq(&glass);
    let stone_hz = SpectralSynth::resonance_freq(&stone);

    println!("glass={glass_hz} Hz, stone={stone_hz} Hz");
    assert!(glass_hz > stone_hz, "glass={glass_hz} Hz, stone={stone_hz} Hz");
}

#[test]
fn stone_room_reverb_longer_than_carpet_room() {
    let stone_v  = half::f16::from_f32(0.85).to_bits();
    let carpet_v = half::f16::from_f32(0.08).to_bits();

    let stone_room:  Vec<[u16; 16]> = (0..16).map(|_| [stone_v;  16]).collect();
    let carpet_room: Vec<[u16; 16]> = (0..16).map(|_| [carpet_v; 16]).collect();

    let stone_reverb  = SpectralReverb::from_splat_reflectance(&stone_room);
    let carpet_reverb = SpectralReverb::from_splat_reflectance(&carpet_room);

    println!("stone={:.2}s carpet={:.2}s", stone_reverb.tail_length_secs, carpet_reverb.tail_length_secs);
    assert!(stone_reverb.tail_length_secs > carpet_reverb.tail_length_secs,
        "stone={:.2}s carpet={:.2}s", stone_reverb.tail_length_secs, carpet_reverb.tail_length_secs);
}
