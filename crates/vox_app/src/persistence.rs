use std::path::PathBuf;
use vox_data::save::{GameState, SaveHeader, save_game, load_game, SaveError};

/// Get the default save directory.
pub fn save_dir() -> PathBuf {
    let mut dir = dirs_next::data_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push("ochroma");
    dir.push("saves");
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Save the current game state.
pub fn save_current(
    city_name: &str,
    game_time_hours: f64,
    citizen_count: u32,
    funds: f64,
    slot: &str,
) -> Result<PathBuf, SaveError> {
    let path = save_dir().join(format!("{}.ochroma_save", slot));
    let state = GameState {
        header: SaveHeader {
            version: 1,
            city_name: city_name.to_string(),
            game_time_hours,
            citizen_count,
            funds,
        },
        data: Vec::new(), // TODO: serialize full ECS state
    };
    save_game(&state, &path)?;
    println!("[ochroma] Game saved to {}", path.display());
    Ok(path)
}

/// Load a game state from a slot.
pub fn load_from_slot(slot: &str) -> Result<GameState, SaveError> {
    let path = save_dir().join(format!("{}.ochroma_save", slot));
    let state = load_game(&path)?;
    println!("[ochroma] Game loaded from {}", path.display());
    Ok(state)
}

/// List available save files.
pub fn list_saves() -> Vec<String> {
    let dir = save_dir();
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "ochroma_save"))
                .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
                .collect()
        })
        .unwrap_or_default()
}
