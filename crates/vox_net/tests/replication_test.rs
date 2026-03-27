use vox_net::replication::{NetMessage, PlayerAction, ReplicationServer};

#[test]
fn message_round_trip() {
    let msg = NetMessage::PlayerInput {
        player_id: 1,
        action: PlayerAction::PlaceRoad { start: [0.0, 0.0, 0.0], end: [100.0, 0.0, 0.0] },
    };
    let bytes = msg.serialize();
    let decoded = NetMessage::deserialize(&bytes).unwrap();
    match decoded {
        NetMessage::PlayerInput { player_id, .. } => assert_eq!(player_id, 1),
        _ => panic!("Wrong message type"),
    }
}

#[test]
fn server_processes_input() {
    let mut server = ReplicationServer::new();
    let input = NetMessage::PlayerInput {
        player_id: 1,
        action: PlayerAction::Zone { position: [50.0, 50.0], zone_type: "residential".into() },
    };
    let responses = server.process_message(&input);
    assert!(!responses.is_empty());
    match &responses[0] {
        NetMessage::StateDelta { tick, deltas } => {
            assert!(*tick > 0);
            assert!(!deltas.is_empty());
        }
        _ => panic!("Expected StateDelta"),
    }
}

#[test]
fn ping_pong() {
    let mut server = ReplicationServer::new();
    let responses = server.process_message(&NetMessage::Ping { timestamp: 12345 });
    match &responses[0] {
        NetMessage::Pong { timestamp } => assert_eq!(*timestamp, 12345),
        _ => panic!("Expected Pong"),
    }
}
