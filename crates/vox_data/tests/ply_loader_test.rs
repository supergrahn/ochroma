use vox_data::ply_loader::*;
use std::io::Cursor;

#[test]
fn load_test_ply() {
    let positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let ply_data = create_test_ply(&positions);
    let mut reader = Cursor::new(&ply_data);
    let splats = load_ply_from_reader(&mut reader).unwrap();

    assert_eq!(splats.len(), 3);
    assert!((splats[0].position[0] - 1.0).abs() < 0.001);
    assert!((splats[1].position[1] - 5.0).abs() < 0.001);
    assert!((splats[2].position[2] - 9.0).abs() < 0.001);
}

#[test]
fn opacity_is_sigmoid_decoded() {
    let positions = vec![[0.0, 0.0, 0.0]];
    let ply_data = create_test_ply(&positions);
    let mut reader = Cursor::new(&ply_data);
    let splats = load_ply_from_reader(&mut reader).unwrap();

    // logit 2.0 -> sigmoid ~ 0.88 -> 0.88 * 255 ~ 224
    assert!(splats[0].opacity > 200 && splats[0].opacity < 240,
        "Expected opacity ~224, got {}", splats[0].opacity);
}

#[test]
fn scales_are_exp_decoded() {
    let positions = vec![[0.0, 0.0, 0.0]];
    let ply_data = create_test_ply(&positions);
    let mut reader = Cursor::new(&ply_data);
    let splats = load_ply_from_reader(&mut reader).unwrap();

    // log-scale -4.6 -> exp ~ 0.01
    assert!((splats[0].scale[0] - 0.01).abs() < 0.005,
        "Expected scale ~0.01, got {}", splats[0].scale[0]);
}

#[test]
fn rotation_is_identity() {
    let positions = vec![[0.0, 0.0, 0.0]];
    let ply_data = create_test_ply(&positions);
    let mut reader = Cursor::new(&ply_data);
    let splats = load_ply_from_reader(&mut reader).unwrap();

    // w=1,x=0,y=0,z=0 -> stored as [x,y,z,w] in i16 -> [0,0,0,32767]
    assert_eq!(splats[0].rotation[3], 32767, "W should be 32767 for identity quat");
    assert!(splats[0].rotation[0].abs() < 100, "X should be ~0");
}

#[test]
fn spectral_from_rgb_produces_nonzero() {
    let positions = vec![[0.0, 0.0, 0.0]];
    let ply_data = create_test_ply(&positions);
    let mut reader = Cursor::new(&ply_data);
    let splats = load_ply_from_reader(&mut reader).unwrap();

    // With f_dc = 0.0, colour = 0.5 + 0 = 0.5 (neutral grey)
    let has_nonzero = splats[0].spectral.iter().any(|&s| s != 0);
    assert!(has_nonzero, "Spectral bands should be non-zero for grey");
}

#[test]
fn handles_many_vertices() {
    let positions: Vec<[f32; 3]> = (0..10000).map(|i| [i as f32, 0.0, 0.0]).collect();
    let ply_data = create_test_ply(&positions);
    let mut reader = Cursor::new(&ply_data);
    let splats = load_ply_from_reader(&mut reader).unwrap();
    assert_eq!(splats.len(), 10000);
}
