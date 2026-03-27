use vox_data::save::{save_game, load_game, GameState, SaveHeader};

#[test]
fn save_and_load_round_trip() {
    let dir = std::env::temp_dir().join("ochroma_save_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let state = GameState {
        header: SaveHeader {
            version: 1,
            city_name: "TestCity".into(),
            game_time_hours: 123.456,
            citizen_count: 5000,
            funds: 99999.99,
        },
        data: vec![1, 2, 3, 4, 5],
    };

    let path = dir.join("test.ochroma_save");
    save_game(&state, &path).unwrap();
    assert!(path.exists());

    let loaded = load_game(&path).unwrap();
    assert_eq!(loaded.header.city_name, "TestCity");
    assert_eq!(loaded.header.citizen_count, 5000);
    assert_eq!(loaded.data, vec![1, 2, 3, 4, 5]);

    let _ = std::fs::remove_dir_all(&dir);
}
