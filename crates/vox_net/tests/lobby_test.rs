use vox_net::lobby::*;

#[test]
fn create_lobby_with_host() {
    let lobby = LobbyState::new("Player1", "TestCity", 4);
    assert_eq!(lobby.player_count(), 1);
    assert_eq!(lobby.players[0].role, PlayerRole::Mayor);
}

#[test]
fn add_players_up_to_max() {
    let mut lobby = LobbyState::new("Host", "City", 3);
    assert!(lobby.add_player("P2", PlayerRole::Councillor).is_some());
    assert!(lobby.add_player("P3", PlayerRole::Spectator).is_some());
    assert!(lobby.add_player("P4", PlayerRole::Councillor).is_none()); // full
    assert!(lobby.is_full());
}

#[test]
fn role_permissions() {
    let lobby = LobbyState::new("Mayor", "City", 4);
    assert!(lobby.can_player_act(0, "place_road"));
    assert!(lobby.can_player_act(0, "budget")); // mayor can do budget
}

#[test]
fn councillor_cannot_change_budget() {
    let mut lobby = LobbyState::new("Mayor", "City", 4);
    lobby.add_player("Council", PlayerRole::Councillor);
    assert!(lobby.can_player_act(1, "place_road")); // can place
    assert!(!lobby.can_player_act(1, "budget")); // cannot budget
}

#[test]
fn spectator_cannot_act() {
    let mut lobby = LobbyState::new("Mayor", "City", 4);
    lobby.add_player("Viewer", PlayerRole::Spectator);
    assert!(!lobby.can_player_act(1, "place_road"));
}

#[test]
fn chat_history() {
    let mut chat = ChatHistory::new(100);
    chat.add(0, "Player1", "Hello!", 0.0);
    chat.add(1, "Player2", "Hi there!", 1.0);
    assert_eq!(chat.recent(10).len(), 2);
    assert_eq!(chat.recent(1).len(), 1);
    assert_eq!(chat.recent(1)[0].text, "Hi there!");
}
