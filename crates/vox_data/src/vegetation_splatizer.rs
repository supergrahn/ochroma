//! VegetationSplatizer — converts vegetation meshes with PROSPECT-PRO spectral
//! embeddings (6 PCA components) to GaussianSplats with 16-band spectral values.

use half::f16;
use vox_core::types::GaussianSplat;

/// PCA basis matrix: 6 components × 16 wavelength bands.
/// Rows are principal components, columns are wavelength bands (380–755nm, 25nm steps).
const PCA_BASIS: [[f32; 16]; 6] = [
    [ 0.31, 0.32, 0.33, 0.34, 0.33, 0.32, 0.31, 0.30,  0.28, 0.25, 0.22, 0.19, 0.17, 0.25, 0.40, 0.45],
    [ 0.12, 0.11, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01, -0.02,-0.05,-0.08,-0.11,-0.13, 0.15, 0.38, 0.42],
    [-0.05,-0.04,-0.03,-0.01, 0.02, 0.05, 0.08, 0.10,  0.08, 0.06, 0.04, 0.02, 0.01,-0.08,-0.20,-0.22],
    [ 0.02, 0.02, 0.01, 0.01,-0.01,-0.02,-0.03,-0.04, -0.03,-0.02,-0.01, 0.01, 0.02, 0.03, 0.05, 0.06],
    [-0.01,-0.01, 0.00, 0.01, 0.02, 0.01, 0.00,-0.01, -0.02,-0.01, 0.00, 0.01, 0.02,-0.01,-0.03,-0.04],
    [ 0.00, 0.00, 0.01, 0.01, 0.00,-0.01,-0.01, 0.00,  0.01, 0.01, 0.00,-0.01,-0.01, 0.00, 0.01, 0.01],
];

/// Back-project a 6-component PCA embedding to 16-band spectral reflectance.
/// Result is clamped to [0, 1].
pub fn backproject_pca(embedding: &[f32; 6]) -> [f32; 16] {
    let mut spectrum = [0.0f32; 16];
    for (comp, &weight) in embedding.iter().enumerate() {
        for band in 0..16 {
            spectrum[band] += weight * PCA_BASIS[comp][band];
        }
    }
    for v in &mut spectrum {
        *v = v.clamp(0.0, 1.0);
    }
    spectrum
}

/// Minimal mesh type for testing. Production uses the actual EditorMesh from vox_core.
#[derive(Default)]
pub struct EditorMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
    pub spectral_embedding: Option<Vec<[f32; 6]>>,
}


/// Convert a vegetation mesh (with spectral_embedding) to GaussianSplats.
/// Each triangle becomes one splat. Spectral value = average of vertex embeddings
/// back-projected to 16 bands. Splat scale derived from triangle area.
pub fn splatize_vegetation_mesh(mesh: &EditorMesh, splat_scale: f32) -> Vec<GaussianSplat> {
    let embeddings = match &mesh.spectral_embedding {
        Some(e) => e,
        None => return splatize_mesh_flat_foliage(mesh, splat_scale),
    };

    mesh.indices.iter().map(|tri| {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p: [f32; 3] = std::array::from_fn(|d| {
            (mesh.positions[i0][d] + mesh.positions[i1][d] + mesh.positions[i2][d]) / 3.0
        });
        let n: [f32; 3] = std::array::from_fn(|d| {
            (mesh.normals[i0][d] + mesh.normals[i1][d] + mesh.normals[i2][d]) / 3.0
        });
        let avg_emb: [f32; 6] = std::array::from_fn(|c| {
            (embeddings[i0][c] + embeddings[i1][c] + embeddings[i2][c]) / 3.0
        });
        let spectral_f32 = backproject_pca(&avg_emb);
        let edge1 = [
            p[0] - mesh.positions[i0][0],
            p[1] - mesh.positions[i0][1],
            p[2] - mesh.positions[i0][2],
        ];
        let area = (edge1[0]*edge1[0] + edge1[1]*edge1[1] + edge1[2]*edge1[2]).sqrt() * 0.5;
        let scale = (area * splat_scale).max(0.01);

        let spectral: [u16; GaussianSplat::BANDS] =
            std::array::from_fn(|i| f16::from_f32(spectral_f32[i]).to_bits());
        GaussianSplat::surface(
            p, n, [0.0, 0.0, -1.0],
            scale, scale,
            (0.85 * 255.0) as u8,
            spectral,
        )
    }).collect()
}

/// Fallback: splatize without spectral_embedding using flat Foliage USGS curve.
fn splatize_mesh_flat_foliage(mesh: &EditorMesh, splat_scale: f32) -> Vec<GaussianSplat> {
    const FOLIAGE_F32: [f32; 16] = [
        0.04, 0.04, 0.05, 0.07, 0.08, 0.10, 0.12, 0.12,
        0.08, 0.05, 0.04, 0.04, 0.05, 0.20, 0.45, 0.55,
    ];
    let foliage: [u16; GaussianSplat::BANDS] =
        std::array::from_fn(|i| f16::from_f32(FOLIAGE_F32[i]).to_bits());

    mesh.indices.iter().map(|tri| {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p: [f32; 3] = std::array::from_fn(|d| {
            (mesh.positions[i0][d] + mesh.positions[i1][d] + mesh.positions[i2][d]) / 3.0
        });
        let n: [f32; 3] = std::array::from_fn(|d| {
            (mesh.normals[i0][d] + mesh.normals[i1][d] + mesh.normals[i2][d]) / 3.0
        });
        GaussianSplat::surface(
            p, n, [0.0, 0.0, -1.0],
            splat_scale * 0.1, splat_scale * 0.1,
            (0.85 * 255.0) as u8,
            foliage,
        )
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_backprojection_produces_valid_spectrum() {
        let embedding = [0.8f32, 0.4, -0.1, 0.0, 0.0, 0.0];
        let spectrum = backproject_pca(&embedding);
        for (i, &v) in spectrum.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "band {i} out of range: {v}");
        }
        let red_edge_avg = (spectrum[12] + spectrum[13] + spectrum[14]) / 3.0;
        let green_avg = (spectrum[6] + spectrum[7] + spectrum[8]) / 3.0;
        println!("red_edge_avg = {:.3} > green_avg = {:.3}", red_edge_avg, green_avg);
        assert!(
            red_edge_avg > green_avg,
            "red-edge should exceed green for leaf: red_edge_avg = {:.3} vs green_avg = {:.3}",
            red_edge_avg, green_avg
        );
    }

    #[test]
    fn test_splatize_vegetation_mesh() {
        let mesh = EditorMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0]; 3],
            indices: vec![[0u32, 1, 2]],
            spectral_embedding: Some(vec![
                [0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.4, 0.0, 0.0, 0.0, 0.0],
            ]),
        };
        let splats = splatize_vegetation_mesh(&mesh, 1.0);
        assert_eq!(splats.len(), 1, "one triangle → one splat");
        assert!(splats[0].spectral()[0] != 0, "spectral[0] = {}, should be nonzero", splats[0].spectral()[0]);
    }
}
