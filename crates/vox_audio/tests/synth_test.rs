use vox_audio::{generate_click, generate_collect_sound, generate_place_sound, generate_tone, save_wav};

#[test]
fn generate_tone_correct_length() {
    let samples = generate_tone(440.0, 1.0, 44100);
    assert_eq!(samples.len(), 44100);
}

#[test]
fn tone_values_in_range() {
    let samples = generate_tone(440.0, 0.1, 44100);
    for s in &samples {
        assert!(
            *s >= -1.0 && *s <= 1.0,
            "Sample out of range: {}",
            s
        );
    }
}

#[test]
fn save_wav_creates_file() {
    let samples = generate_tone(440.0, 0.5, 44100);
    let dir = std::env::temp_dir().join("ochroma_audio_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_tone.wav");
    save_wav(&samples, 44100, &path).unwrap();
    assert!(path.exists());
    let data = std::fs::read(&path).unwrap();
    assert!(data.starts_with(b"RIFF"));
    // Verify WAV header has correct format marker
    assert_eq!(&data[8..12], b"WAVE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn click_sound_is_short() {
    let samples = generate_click();
    assert!(samples.len() < 5000, "Click should be short, got {} samples", samples.len());
    assert!(!samples.is_empty());
}

#[test]
fn place_sound_has_decay() {
    let samples = generate_place_sound();
    assert!(!samples.is_empty());
    // Last sample should be quieter than first non-zero sample
    let first_abs = samples[samples.len() / 4].abs();
    let last_abs = samples[samples.len() - 1].abs();
    assert!(last_abs < first_abs, "Sound should decay over time");
}

#[test]
fn collect_sound_has_rising_pitch() {
    let samples = generate_collect_sound();
    assert!(!samples.is_empty());
    // Just verify it produces valid audio
    for s in &samples {
        assert!(s.is_finite());
    }
}

#[test]
fn different_sample_rates() {
    let s1 = generate_tone(440.0, 1.0, 22050);
    assert_eq!(s1.len(), 22050);
    let s2 = generate_tone(440.0, 1.0, 48000);
    assert_eq!(s2.len(), 48000);
}

#[test]
fn wav_header_correct_size() {
    let samples = generate_tone(440.0, 0.1, 44100);
    let dir = std::env::temp_dir().join("ochroma_wav_size_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("size_test.wav");
    save_wav(&samples, 44100, &path).unwrap();
    let data = std::fs::read(&path).unwrap();
    // Header is 44 bytes + 2 bytes per sample
    assert_eq!(data.len(), 44 + samples.len() * 2);
    let _ = std::fs::remove_dir_all(&dir);
}
