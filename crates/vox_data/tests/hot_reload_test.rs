use std::path::PathBuf;
use vox_data::hot_reload::AssetWatcher;

#[test]
fn unchanged_file_returns_empty() {
    let dir = std::env::temp_dir().join("vox_hot_reload_test_unchanged");
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join("asset.bin");
    std::fs::write(&file, b"hello").unwrap();

    let mut watcher = AssetWatcher::new(0.0);
    watcher.watch(file.clone());

    // First poll should return empty since we recorded mtime on watch.
    let changed = watcher.poll(1.0);
    assert!(changed.is_empty(), "expected empty, got {:?}", changed);

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn modified_file_triggers_detection() {
    let dir = std::env::temp_dir().join("vox_hot_reload_test_modified");
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join("asset.bin");
    std::fs::write(&file, b"hello").unwrap();

    let mut watcher = AssetWatcher::new(0.0);
    watcher.watch(file.clone());

    // Wait a tiny bit so the filesystem mtime granularity catches the change.
    std::thread::sleep(std::time::Duration::from_millis(50));
    std::fs::write(&file, b"world").unwrap();

    let changed = watcher.poll(1.0);
    assert!(
        changed.contains(&file),
        "expected {:?} in {:?}",
        file,
        changed
    );

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn poll_respects_interval() {
    let dir = std::env::temp_dir().join("vox_hot_reload_test_interval");
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join("asset.bin");
    std::fs::write(&file, b"hello").unwrap();

    let mut watcher = AssetWatcher::new(1.0); // 1 second interval
    watcher.watch(file.clone());

    // dt=0.5 should not trigger a poll.
    std::thread::sleep(std::time::Duration::from_millis(50));
    std::fs::write(&file, b"world").unwrap();

    let changed = watcher.poll(0.5);
    assert!(changed.is_empty(), "should not poll yet");

    // dt=0.6 pushes accumulator past 1.0.
    let changed = watcher.poll(0.6);
    assert!(
        changed.contains(&file),
        "should detect change now: {:?}",
        changed
    );

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn watch_and_unwatch() {
    let mut watcher = AssetWatcher::new(0.0);
    let path = PathBuf::from("/tmp/nonexistent_vox_test_file");
    watcher.watch(path.clone());
    assert_eq!(watcher.watched_count(), 1);
    watcher.unwatch(&path);
    assert_eq!(watcher.watched_count(), 0);
}
