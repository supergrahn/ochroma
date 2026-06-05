//! QUIC transport using Quinn with self-signed TLS 1.3 certificates.
//!
//! Replaces the TCP transport entirely. All game traffic goes over QUIC:
//! - Reliable ordered streams: lobby, authentication, asset manifests, replication packets.
//! - Unreliable datagrams: per-frame spectral replication updates (future work).
//!
//! Uses rcgen to generate a self-signed certificate at startup.
//! For production, certificate pinning or a CA-signed cert replaces rcgen output.
//!
//! # End-to-end 2-client flow
//!
//! ```no_run
//! # use vox_net::quic_transport::{QuicServer, QuicClient};
//! # use vox_net::replication_packet::ReplicationPacket;
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let server = QuicServer::listen("127.0.0.1:0").await?;
//! let addr = server.local_addr()?;
//!
//! // Client side: handshake actually completes here.
//! let client = QuicClient::connect(&addr.to_string(), "localhost").await?;
//!
//! // Server side: accept the established connection.
//! let server_conn = server.accept().await?;
//!
//! // Client sends a per-band delta packet over a reliable stream.
//! let packet = ReplicationPacket::full(7, &[1234u16; 16]);
//! client.connection().send_packet(&packet).await?;
//!
//! // Server receives the IDENTICAL packet.
//! let received = server_conn.recv_packet().await?;
//! assert_eq!(received, packet);
//! # Ok(())
//! # }
//! ```

use std::net::SocketAddr;
use std::sync::Arc;
use thiserror::Error;

use crate::replication_packet::{PacketError, ReplicationPacket};

/// ALPN protocol identifier negotiated by both client and server. The handshake
/// fails unless BOTH sides advertise this exact byte string, so the server config
/// must set it too (a prior bug only set it on the client, breaking every handshake).
pub const ALPN_PROTOCOL: &[u8] = b"ochroma-quic-v1";

/// Upper bound on a single replication stream payload (defensive cap for `read_to_end`).
const MAX_PACKET_BYTES: usize = 16 * 1024 * 1024;

#[derive(Error, Debug)]
pub enum TransportError {
    #[error("TLS certificate generation failed: {0}")]
    CertGen(String),
    #[error("Quinn endpoint creation failed: {0}")]
    Endpoint(String),
    #[error("Connection failed: {0}")]
    Connection(String),
    #[error("stream error: {0}")]
    Stream(String),
    #[error("packet decode error: {0}")]
    Packet(#[from] PacketError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Self-signed TLS certificate + private key pair.
pub struct SelfSignedCert {
    pub cert_der: Vec<u8>,
    pub key_der: Vec<u8>,
}

impl SelfSignedCert {
    /// Generate a self-signed certificate for the given hostname.
    /// Uses rcgen with ECDSA P-256 keys.
    pub fn generate(hostname: &str) -> Result<Self, TransportError> {
        let certified = rcgen::generate_simple_self_signed(vec![hostname.to_string()])
            .map_err(|e| TransportError::CertGen(e.to_string()))?;
        Ok(Self {
            cert_der: certified.cert.der().to_vec(),
            key_der: certified.key_pair.serialize_der(), // rcgen 0.13 field name (was `signing_key` in older versions)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportRole {
    Server,
    Client,
}

/// Install the ring crypto provider once (idempotent — later calls are no-ops).
fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Build the server-side `quinn::ServerConfig` from a freshly generated self-signed cert.
/// Crucially advertises [`ALPN_PROTOCOL`] so the TLS handshake can negotiate with the client.
fn server_config() -> Result<quinn::ServerConfig, TransportError> {
    let cert = SelfSignedCert::generate("localhost")?;

    let server_cert = rustls::pki_types::CertificateDer::from(cert.cert_der.clone());
    let server_key = rustls::pki_types::PrivateKeyDer::try_from(cert.key_der.clone())
        .map_err(|e| TransportError::CertGen(e.to_string()))?;

    let mut tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![server_cert], server_key)
        .map_err(|e| TransportError::CertGen(e.to_string()))?;
    // Without this the server offers no ALPN, the client requires `ochroma-quic-v1`,
    // and the handshake aborts with NO_APPLICATION_PROTOCOL.
    tls_config.alpn_protocols = vec![ALPN_PROTOCOL.to_vec()];

    let server_config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(tls_config)
            .map_err(|e| TransportError::CertGen(e.to_string()))?,
    ));
    Ok(server_config)
}

/// Build the client-side `quinn::ClientConfig`. Skips certificate verification
/// (dev only) and advertises the same ALPN protocol as the server.
fn client_config() -> Result<quinn::ClientConfig, TransportError> {
    let mut tls_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    tls_config.alpn_protocols = vec![ALPN_PROTOCOL.to_vec()];

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(tls_config)
            .map_err(|e| TransportError::Connection(e.to_string()))?,
    ));
    Ok(client_config)
}

/// Legacy endpoint-only transport handle.
///
/// Kept for backward compatibility with existing callers that store an
/// `Option<QuicTransport>` and read its public `role` field. New code should use
/// [`QuicServer`] / [`QuicClient`], which actually complete the handshake and can
/// send/receive [`ReplicationPacket`]s. `QuicTransport::connect` only initiates the
/// connection (the handshake future is driven by the caller, if at all), matching
/// the original behavior.
pub struct QuicTransport {
    pub endpoint: quinn::Endpoint,
    pub role: TransportRole,
}

impl QuicTransport {
    /// Start a QUIC server endpoint listening on `addr` with a self-signed certificate.
    pub async fn listen(addr: &str) -> Result<Self, TransportError> {
        ensure_crypto_provider();
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e: std::net::AddrParseError| TransportError::Endpoint(e.to_string()))?;
        let endpoint = quinn::Endpoint::server(server_config()?, addr)
            .map_err(|e| TransportError::Endpoint(e.to_string()))?;
        Ok(Self { endpoint, role: TransportRole::Server })
    }

    /// Create a client endpoint configured to reach `addr`. The returned `Connecting`
    /// future is intentionally not awaited here (legacy behavior); prefer [`QuicClient::connect`].
    pub async fn connect(addr: &str, server_name: &str) -> Result<Self, TransportError> {
        ensure_crypto_provider();
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e: std::net::AddrParseError| TransportError::Connection(e.to_string()))?;
        let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| TransportError::Endpoint(e.to_string()))?;
        endpoint.set_default_client_config(client_config()?);
        // Validate reachability of the target address; the handshake is deferred.
        let _connecting = endpoint
            .connect(addr, server_name)
            .map_err(|e| TransportError::Connection(e.to_string()))?;
        Ok(Self { endpoint, role: TransportRole::Client })
    }

    /// Local address the endpoint is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        self.endpoint.local_addr().map_err(TransportError::Io)
    }
}

/// A fully-established QUIC connection. Wraps a `quinn::Connection` and exposes
/// reliable, ordered, length-prefixed [`ReplicationPacket`] exchange. Each
/// `send_packet`/`recv_packet` pair uses its own bidirectional stream, so packets
/// never interleave on the wire and a `recv_packet` reads exactly one packet.
#[derive(Clone)]
pub struct QuicConnection {
    inner: quinn::Connection,
}

impl QuicConnection {
    /// The peer's socket address (loopback in tests).
    pub fn remote_address(&self) -> SocketAddr {
        self.inner.remote_address()
    }

    /// Send one [`ReplicationPacket`] over a fresh bidirectional stream.
    /// Opens the stream, writes the encoded per-band delta, and `finish`es so the
    /// peer's `read_to_end` terminates cleanly.
    pub async fn send_packet(&self, packet: &ReplicationPacket) -> Result<(), TransportError> {
        let (mut send, _recv) = self
            .inner
            .open_bi()
            .await
            .map_err(|e| TransportError::Stream(e.to_string()))?;
        let bytes = packet.encode();
        send.write_all(&bytes)
            .await
            .map_err(|e| TransportError::Stream(e.to_string()))?;
        send.finish()
            .map_err(|e| TransportError::Stream(e.to_string()))?;
        Ok(())
    }

    /// Accept the next inbound bidirectional stream and decode exactly one
    /// [`ReplicationPacket`] from it. Returns the decoded packet — byte-identical
    /// to what the peer encoded — or a decode error.
    pub async fn recv_packet(&self) -> Result<ReplicationPacket, TransportError> {
        let (_send, mut recv) = self
            .inner
            .accept_bi()
            .await
            .map_err(|e| TransportError::Stream(e.to_string()))?;
        let bytes = recv
            .read_to_end(MAX_PACKET_BYTES)
            .await
            .map_err(|e| TransportError::Stream(e.to_string()))?;
        let packet = ReplicationPacket::decode(&bytes)?;
        Ok(packet)
    }

    /// Borrow the underlying quinn connection for advanced use (datagrams, extra streams).
    pub fn raw(&self) -> &quinn::Connection {
        &self.inner
    }
}

/// QUIC server: owns the listening endpoint and accepts established connections.
pub struct QuicServer {
    endpoint: quinn::Endpoint,
}

impl QuicServer {
    /// Start a QUIC server listening on `addr` with a self-signed certificate.
    /// Pass `127.0.0.1:0` to let the OS pick a free port (read it back via [`local_addr`]).
    ///
    /// [`local_addr`]: QuicServer::local_addr
    pub async fn listen(addr: &str) -> Result<Self, TransportError> {
        ensure_crypto_provider();

        let addr: SocketAddr = addr
            .parse()
            .map_err(|e: std::net::AddrParseError| TransportError::Endpoint(e.to_string()))?;

        let endpoint = quinn::Endpoint::server(server_config()?, addr)
            .map_err(|e| TransportError::Endpoint(e.to_string()))?;

        Ok(Self { endpoint })
    }

    /// Wait for the next client, drive its handshake to completion, and return the
    /// established [`QuicConnection`]. This is where the server side of the TLS
    /// handshake actually runs — it does NOT return until the connection is usable.
    pub async fn accept(&self) -> Result<QuicConnection, TransportError> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| TransportError::Connection("endpoint closed".into()))?;
        let connecting = incoming
            .accept()
            .map_err(|e| TransportError::Connection(e.to_string()))?;
        let inner = connecting
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;
        Ok(QuicConnection { inner })
    }

    /// Local address the listening endpoint is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        self.endpoint.local_addr().map_err(TransportError::Io)
    }

    pub fn role(&self) -> TransportRole {
        TransportRole::Server
    }
}

/// QUIC client: a fully-established connection to a [`QuicServer`].
pub struct QuicClient {
    endpoint: quinn::Endpoint,
    connection: QuicConnection,
}

impl QuicClient {
    /// Connect to a QUIC server at `addr` and DRIVE THE HANDSHAKE TO COMPLETION.
    /// Unlike a bare `endpoint.connect()` (which only returns a pending future), this
    /// awaits the returned `Connecting`, so on success the connection is immediately
    /// usable for [`QuicConnection::send_packet`].
    pub async fn connect(addr: &str, server_name: &str) -> Result<Self, TransportError> {
        ensure_crypto_provider();

        let addr: SocketAddr = addr
            .parse()
            .map_err(|e: std::net::AddrParseError| TransportError::Connection(e.to_string()))?;

        let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| TransportError::Endpoint(e.to_string()))?;
        endpoint.set_default_client_config(client_config()?);

        let connecting = endpoint
            .connect(addr, server_name)
            .map_err(|e| TransportError::Connection(e.to_string()))?;
        // Await the handshake — THIS is the byte exchange the old code skipped.
        let inner = connecting
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;

        Ok(Self {
            endpoint,
            connection: QuicConnection { inner },
        })
    }

    /// The established connection — use it to send/receive replication packets.
    pub fn connection(&self) -> &QuicConnection {
        &self.connection
    }

    /// Local address the client endpoint is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        self.endpoint.local_addr().map_err(TransportError::Io)
    }

    pub fn role(&self) -> TransportRole {
        TransportRole::Client
    }
}

/// Development-only: skip TLS certificate verification for self-signed certs.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_signed_cert_generates_non_empty_der() {
        let cert = SelfSignedCert::generate("localhost").unwrap();
        assert!(!cert.cert_der.is_empty(), "cert DER should not be empty");
        assert!(!cert.key_der.is_empty(), "key DER should not be empty");
    }

    #[test]
    fn self_signed_cert_generates_different_certs_each_call() {
        let a = SelfSignedCert::generate("localhost").unwrap();
        let b = SelfSignedCert::generate("localhost").unwrap();
        assert_ne!(
            a.key_der, b.key_der,
            "each cert generation should produce a unique key"
        );
    }

    #[tokio::test]
    async fn server_listen_binds_successfully() {
        let server = QuicServer::listen("127.0.0.1:0").await.unwrap();
        let addr = server.local_addr().unwrap();
        println!("server bound to port {}", addr.port());
        assert!(addr.port() > 0, "server should be bound to a non-zero port");
        assert_eq!(server.role(), TransportRole::Server);
    }

    #[tokio::test]
    async fn legacy_quic_transport_client_creates_endpoint() {
        let server = QuicTransport::listen("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();
        assert_eq!(server.role, TransportRole::Server);

        let client = QuicTransport::connect(&server_addr.to_string(), "localhost")
            .await
            .unwrap();
        assert_eq!(client.role, TransportRole::Client);
        assert!(client.local_addr().unwrap().port() > 0);
    }

    /// End-to-end: real loopback handshake completes, then a per-band delta
    /// ReplicationPacket round-trips over a QUIC stream and decodes byte-identically.
    #[tokio::test]
    async fn quic_replication_packet_round_trip_over_loopback() {
        let server = QuicServer::listen("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();

        // Accept on the server in a task so client connect + server accept run concurrently.
        let accept_task = tokio::spawn(async move {
            let conn = server.accept().await.expect("server accept failed");
            conn.recv_packet().await.expect("server recv failed")
        });

        let client = QuicClient::connect(&server_addr.to_string(), "localhost")
            .await
            .expect("client connect (handshake) failed");
        assert_eq!(client.role(), TransportRole::Client);

        // A delta packet: bands 2 and 5 changed.
        let before = [1000u16; 16];
        let mut after = before;
        after[2] = 2000;
        after[5] = 3000;
        let sent = ReplicationPacket::from_delta(42, &before, &after, 0);
        assert_eq!(sent.values, vec![2000, 3000], "sanity: delta captured both bands");

        client
            .connection()
            .send_packet(&sent)
            .await
            .expect("client send failed");

        let received = accept_task.await.expect("accept task panicked");

        // Real field equality across the whole packet (not is_ok()).
        assert_eq!(received.entity_id, 42, "entity_id must survive the round trip");
        assert_eq!(
            received.changed_bands,
            (1 << 2) | (1 << 5),
            "changed_bands bitmask must match exactly"
        );
        assert_eq!(received.values, vec![2000, 3000], "band values must match exactly");
        assert_eq!(received, sent, "decoded packet must equal the sent packet");

        // And the delta truly reconstructs the post-state when applied.
        let mut reconstructed = before;
        received.apply_to(&mut reconstructed).unwrap();
        assert_eq!(reconstructed, after, "applying the received delta must yield the sender's after-state");
    }

    /// Two distinct packets sent back-to-back arrive in order, each on its own stream.
    #[tokio::test]
    async fn quic_two_packets_preserve_identity() {
        let server = QuicServer::listen("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();

        let accept_task = tokio::spawn(async move {
            let conn = server.accept().await.expect("server accept failed");
            let a = conn.recv_packet().await.expect("recv 1 failed");
            let b = conn.recv_packet().await.expect("recv 2 failed");
            (a, b)
        });

        let client = QuicClient::connect(&server_addr.to_string(), "localhost")
            .await
            .expect("client connect failed");

        let p1 = ReplicationPacket::full(1, &[111u16; 16]);
        let mut after = [0u16; 16];
        after[7] = 4242;
        let p2 = ReplicationPacket::from_delta(2, &[0u16; 16], &after, 0);

        client.connection().send_packet(&p1).await.unwrap();
        client.connection().send_packet(&p2).await.unwrap();

        let (got1, got2) = accept_task.await.expect("accept task panicked");
        assert_eq!(got1, p1, "first packet identity preserved");
        assert_eq!(got2, p2, "second packet identity preserved");
        assert_eq!(got1.entity_id, 1);
        assert_eq!(got2.entity_id, 2);
        assert_eq!(got2.changed_bands, 1 << 7);
        assert_eq!(got2.values, vec![4242]);
    }
}
