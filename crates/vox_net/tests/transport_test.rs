use vox_net::replication::{NetMessage, PlayerAction};
use vox_net::transport::{GameClient, GameServer};

#[test]
fn message_serialization_round_trip() {
    let msg = NetMessage::PlayerInput {
        player_id: 42,
        action: PlayerAction::PlaceRoad {
            start: [0.0, 0.0, 0.0],
            end: [100.0, 0.0, 0.0],
        },
    };
    let bytes = msg.serialize();
    assert!(!bytes.is_empty());
    let decoded = NetMessage::deserialize(&bytes).unwrap();
    match decoded {
        NetMessage::PlayerInput { player_id, .. } => assert_eq!(player_id, 42),
        _ => panic!("Wrong message type after deserialization"),
    }
}

#[test]
fn ping_pong_serialization() {
    let ping = NetMessage::Ping { timestamp: 123456 };
    let bytes = ping.serialize();
    let decoded = NetMessage::deserialize(&bytes).unwrap();
    match decoded {
        NetMessage::Ping { timestamp } => assert_eq!(timestamp, 123456),
        _ => panic!("Wrong message type"),
    }
}

#[test]
fn state_delta_serialization() {
    let msg = NetMessage::StateDelta {
        tick: 99,
        deltas: vec![vox_net::replication::EntityDelta {
            entity_id: 1,
            component: "position".to_string(),
            data: vec![1, 2, 3],
            timestamp: 99,
        }],
    };
    let bytes = msg.serialize();
    let decoded = NetMessage::deserialize(&bytes).unwrap();
    match decoded {
        NetMessage::StateDelta { tick, deltas } => {
            assert_eq!(tick, 99);
            assert_eq!(deltas.len(), 1);
            assert_eq!(deltas[0].entity_id, 1);
        }
        _ => panic!("Wrong message type"),
    }
}

#[tokio::test]
async fn client_server_round_trip() {
    // Start server on port 0 (OS picks a free port)
    let mut server = GameServer::new(0);
    server.start().await.unwrap();
    let port = server.bound_port().expect("Server should have a bound port");

    // Connect client
    let mut client = GameClient::new();
    client
        .connect(&format!("127.0.0.1:{}", port))
        .await
        .unwrap();
    assert!(client.is_connected());

    // Accept the connection on server side
    let mut server_stream = server.accept().await.unwrap();

    // Client sends a message
    let msg = NetMessage::Ping { timestamp: 42 };
    client.send(&msg).await.unwrap();

    // Server receives it
    let received = GameServer::recv(&mut server_stream).await.unwrap();
    match &received {
        NetMessage::Ping { timestamp } => assert_eq!(*timestamp, 42),
        _ => panic!("Expected Ping"),
    }

    // Server sends a response
    let response = NetMessage::Pong { timestamp: 42 };
    GameServer::send(&mut server_stream, &response).await.unwrap();

    // Client receives it
    let received = client.recv().await.unwrap();
    match received {
        NetMessage::Pong { timestamp } => assert_eq!(timestamp, 42),
        _ => panic!("Expected Pong"),
    }
}

#[tokio::test]
async fn client_not_connected_error() {
    let mut client = GameClient::new();
    assert!(!client.is_connected());
    let result = client.send(&NetMessage::Ping { timestamp: 0 }).await;
    assert!(result.is_err());
}
