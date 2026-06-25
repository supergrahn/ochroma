use vox_net::replication::{CommandPayload, NetMessage, ReplicationServer};

#[test]
fn message_round_trip() {
    let raw = b"place_road:{\"start\":[0,0,0],\"end\":[100,0,0]}";
    let msg = NetMessage::PlayerInput {
        player_id: 1,
        command: CommandPayload::encode(raw),
    };
    let bytes = msg.serialize();
    let decoded = NetMessage::deserialize(&bytes).unwrap();
    match decoded {
        NetMessage::PlayerInput { player_id, command } => {
            assert_eq!(player_id, 1);
            assert_eq!(command.as_bytes(), raw);
        }
        _ => panic!("Wrong message type"),
    }
}

#[test]
fn server_processes_input() {
    let mut server = ReplicationServer::new();
    let raw = b"zone:{\"position\":[50,50],\"zone_type\":\"residential\"}";
    let input = NetMessage::PlayerInput {
        player_id: 1,
        command: CommandPayload::encode(raw),
    };
    let responses = server.process_message(&input);
    assert!(!responses.is_empty());
    match &responses[0] {
        NetMessage::StateDelta { tick, deltas } => {
            assert!(*tick > 0);
            assert!(!deltas.is_empty());
            // engine stores the raw command bytes verbatim in the delta
            assert_eq!(deltas[0].data, raw);
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
