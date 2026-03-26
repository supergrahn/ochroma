use glam::Vec3;
use vox_core::svo::SpatialHash;

#[test]
fn insert_and_query_by_voxel() {
    let mut sh = SpatialHash::new(8.0);
    sh.insert(1, Vec3::new(1.0, 2.0, 3.0));
    sh.insert(2, Vec3::new(1.5, 2.5, 3.5));
    sh.insert(3, Vec3::new(100.0, 0.0, 0.0));
    let nearby = sh.query_voxel(Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(nearby.len(), 2);
    assert!(nearby.contains(&1));
    assert!(nearby.contains(&2));
}

#[test]
fn query_radius() {
    let mut sh = SpatialHash::new(8.0);
    sh.insert(1, Vec3::new(0.0, 0.0, 0.0));
    sh.insert(2, Vec3::new(5.0, 0.0, 0.0));
    sh.insert(3, Vec3::new(50.0, 0.0, 0.0));
    let nearby = sh.query_radius(Vec3::ZERO, 10.0);
    assert!(nearby.contains(&1));
    assert!(nearby.contains(&2));
    assert!(!nearby.contains(&3));
}

#[test]
fn remove_instance() {
    let mut sh = SpatialHash::new(8.0);
    sh.insert(1, Vec3::new(0.0, 0.0, 0.0));
    sh.remove(1, Vec3::new(0.0, 0.0, 0.0));
    assert!(sh.query_voxel(Vec3::ZERO).is_empty());
}
