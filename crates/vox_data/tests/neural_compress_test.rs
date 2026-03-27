use vox_core::types::GaussianSplat;
use vox_data::neural_compress::*;

fn make_splats(count: usize) -> Vec<GaussianSplat> {
    (0..count)
        .map(|i| GaussianSplat {
            position: [i as f32, 0.0, 0.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [15360; 8],
        })
        .collect()
}

#[test]
fn fast_compression_high_ratio() {
    let compressor = NeuralCompressor::new(CompressionQuality::Fast);
    let splats = make_splats(1000);
    let compressed = compressor.compress(&splats);
    assert!(
        compressed.compression_ratio() > 3.0,
        "Fast should compress well, ratio={:.1}",
        compressed.compression_ratio()
    );
}

#[test]
fn quality_compression_preserves_more() {
    let fast = NeuralCompressor::new(CompressionQuality::Fast);
    let quality = NeuralCompressor::new(CompressionQuality::Quality);
    let splats = make_splats(100);
    let fast_size = fast.compress(&splats).size_bytes();
    let quality_size = quality.compress(&splats).size_bytes();
    assert!(quality_size > fast_size, "Quality should use more bytes");
}

#[test]
fn estimate_matches_actual() {
    let compressor = NeuralCompressor::new(CompressionQuality::Balanced);
    let splats = make_splats(1000);
    let estimated = compressor.estimate_compressed_size(1000);
    let actual = compressor.compress(&splats).size_bytes();
    // Estimate should be in the right ballpark (within 5x)
    assert!(
        (estimated as f32 / actual as f32) > 0.2 && (estimated as f32 / actual as f32) < 5.0
    );
}
