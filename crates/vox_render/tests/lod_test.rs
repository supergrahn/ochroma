use vox_render::lod::{select_lod, LodLevel, reduce_splat_indices};

#[test]
fn close_distance_selects_full() {
    assert_eq!(select_lod(50.0), LodLevel::Full);
    assert_eq!(select_lod(199.0), LodLevel::Full);
}

#[test]
fn far_distance_selects_reduced() {
    assert_eq!(select_lod(201.0), LodLevel::Reduced);
    assert_eq!(select_lod(1000.0), LodLevel::Reduced);
}

#[test]
fn boundary_is_200m() {
    assert_eq!(select_lod(200.0), LodLevel::Full);
    assert_eq!(select_lod(200.1), LodLevel::Reduced);
}

#[test]
fn reduce_splats_produces_correct_count() {
    let indices: Vec<usize> = (0..1000).collect();
    let reduced = reduce_splat_indices(&indices, 0.4);
    assert_eq!(reduced.len(), 400);
}
