//! End-to-end QUIC replication test over a real loopback socket.
//!
//! Exercises the public 2-client API: a `QuicServer` listens, a `QuicClient`
//! completes a real TLS 1.3 handshake, then a `ReplicationPacket` (the per-band
//! delta codec) is sent across the wire and the OTHER side must receive and decode
//! the byte-identical packet. Assertions check real field equality, never `is_ok()`.

use vox_net::{QuicClient, QuicServer, ReplicationPacket};

#[tokio::test]
async fn two_client_replication_packet_crosses_real_socket() {
    // 1. Server binds a real UDP/QUIC socket on loopback (port 0 = OS picks).
    let server = QuicServer::listen("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();
    assert!(server_addr.port() > 0, "server must bind a real port");

    // 2. Server accepts the established connection and reads one packet (on a task,
    //    so the client connect and server accept handshakes run concurrently).
    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("server failed to accept connection");
        // remote_address proves a real peer socket is attached, not a stub.
        let peer = conn.remote_address();
        assert_eq!(peer.ip().to_string(), "127.0.0.1", "peer must be loopback");
        conn.recv_packet().await.expect("server failed to receive packet")
    });

    // 3. Client performs the real handshake (NOT a discarded future).
    let client = QuicClient::connect(&server_addr.to_string(), "localhost")
        .await
        .expect("client handshake must complete over loopback");

    // 4. Build a non-trivial per-band delta packet: bands 1, 4, 11 changed.
    let before = [500u16; 16];
    let mut after = before;
    after[1] = 1500;
    after[4] = 600;
    after[11] = 9999;
    let sent = ReplicationPacket::from_delta(0xCAFE, &before, &after, 0);
    // Sanity: the codec captured exactly the three changed bands, in order.
    assert_eq!(sent.changed_bands, (1 << 1) | (1 << 4) | (1 << 11));
    assert_eq!(sent.values, vec![1500, 600, 9999]);

    // 5. Send it across the socket.
    client
        .connection()
        .send_packet(&sent)
        .await
        .expect("client failed to send packet");

    // 6. The other side decodes a byte-identical packet.
    let received = server_task.await.expect("server task panicked");

    assert_eq!(received.entity_id, 0xCAFE, "entity_id must round-trip exactly");
    assert_eq!(
        received.changed_bands,
        (1 << 1) | (1 << 4) | (1 << 11),
        "changed_bands bitmask must round-trip exactly"
    );
    assert_eq!(received.values, vec![1500, 600, 9999], "band values must round-trip exactly");
    assert_eq!(received, sent, "received packet must equal the sent packet field-for-field");

    // 7. The delta, applied to the original state, must reproduce the sender's after-state.
    let mut applied = before;
    received.apply_to(&mut applied).expect("apply_to must succeed");
    assert_eq!(applied, after, "applying the delta must reconstruct the sender's post-state");
}

#[tokio::test]
async fn server_to_client_direction_also_works() {
    // Replication is bidirectional: the server can push a packet the client decodes.
    let server = QuicServer::listen("127.0.0.1:0").await.unwrap();
    let server_addr = server.local_addr().unwrap();

    let full = ReplicationPacket::full(77, &[4321u16; 16]);
    let expected = full.clone();
    let server_task = tokio::spawn(async move {
        let conn = server.accept().await.expect("accept failed");
        conn.send_packet(&full).await.expect("server send failed");
        // Keep the connection (and endpoint) alive until the client confirms receipt,
        // otherwise dropping it here would close the stream before delivery completes.
        // The client signals readiness by sending an ack packet back.
        let _ack = conn.recv_packet().await.expect("server failed to read ack");
    });

    let client = QuicClient::connect(&server_addr.to_string(), "localhost")
        .await
        .expect("handshake failed");

    let received = client
        .connection()
        .recv_packet()
        .await
        .expect("client recv failed");

    // Ack so the server knows it can safely tear down.
    client
        .connection()
        .send_packet(&ReplicationPacket::full(0, &[0u16; 16]))
        .await
        .expect("client failed to send ack");
    server_task.await.expect("server task panicked");
    let sent = expected;

    assert_eq!(received.entity_id, 77);
    assert_eq!(received.changed_bands, 0xFFFF, "full packet marks all 16 bands");
    assert_eq!(received.values.len(), 16);
    assert_eq!(received.values, vec![4321u16; 16]);
    assert_eq!(received, sent, "client must decode the exact packet the server sent");
}
