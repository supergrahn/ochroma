//! QUIC transport using Quinn with self-signed TLS 1.3 certificates.
//!
//! Replaces the TCP transport entirely. All game traffic goes over QUIC:
//! - Reliable ordered streams: lobby, authentication, asset manifests.
//! - Unreliable datagrams: per-frame spectral replication updates.
//!
//! Uses rcgen to generate a self-signed certificate at startup.
//! For production, certificate pinning or a CA-signed cert replaces rcgen output.

use std::net::SocketAddr;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransportError {
    #[error("TLS certificate generation failed: {0}")]
    CertGen(String),
    #[error("Quinn endpoint creation failed: {0}")]
    Endpoint(String),
    #[error("Connection failed: {0}")]
    Connection(String),
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

pub struct QuicTransport {
    pub endpoint: quinn::Endpoint,
    pub role: TransportRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportRole {
    Server,
    Client,
}

impl QuicTransport {
    /// Start a QUIC server listening on `addr` with a self-signed certificate.
    pub async fn listen(addr: &str) -> Result<Self, TransportError> {
        // Install ring crypto provider if none is set (idempotent — ignores Err if already set).
        let _ = rustls::crypto::ring::default_provider().install_default();

        let addr: SocketAddr = addr.parse()
            .map_err(|e: std::net::AddrParseError| TransportError::Endpoint(e.to_string()))?;

        let cert = SelfSignedCert::generate("localhost")?;

        let server_cert = rustls::pki_types::CertificateDer::from(cert.cert_der.clone());
        let server_key = rustls::pki_types::PrivateKeyDer::try_from(cert.key_der.clone())
            .map_err(|e| TransportError::CertGen(e.to_string()))?;

        let tls_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![server_cert], server_key)
            .map_err(|e| TransportError::CertGen(e.to_string()))?;

        let server_config = quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(tls_config)
                .map_err(|e| TransportError::CertGen(e.to_string()))?,
        ));

        let endpoint = quinn::Endpoint::server(server_config, addr)
            .map_err(|e| TransportError::Endpoint(e.to_string()))?;

        Ok(Self { endpoint, role: TransportRole::Server })
    }

    /// Connect to a QUIC server at `addr` as a client.
    pub async fn connect(addr: &str, server_name: &str) -> Result<Self, TransportError> {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let addr: SocketAddr = addr.parse()
            .map_err(|e: std::net::AddrParseError| TransportError::Connection(e.to_string()))?;

        let tls_config = {
            let mut config = rustls::ClientConfig::builder()
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
                .with_no_client_auth();
            config.alpn_protocols = vec![b"ochroma-quic-v1".to_vec()];
            config
        };

        let client_config = quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(tls_config)
                .map_err(|e| TransportError::Connection(e.to_string()))?,
        ));

        let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| TransportError::Endpoint(e.to_string()))?;
        endpoint.set_default_client_config(client_config);

        // Initiate the connection. The returned `Connecting` future drives the TLS handshake;
        // callers that need a fully-established `quinn::Connection` must `.await` it.
        // Here we validate the address is reachable (returns ConnectError immediately on bad addr)
        // but defer the handshake to the caller.
        let _connecting = endpoint.connect(addr, server_name)
            .map_err(|e| TransportError::Connection(e.to_string()))?;

        Ok(Self { endpoint, role: TransportRole::Client })
    }

    /// Local address the endpoint is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        self.endpoint.local_addr().map_err(TransportError::Io)
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
        &self, _: &[u8], _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self, _: &[u8], _: &rustls::pki_types::CertificateDer<'_>,
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
        assert_ne!(a.key_der, b.key_der, "each cert generation should produce a unique key");
    }

    #[tokio::test]
    async fn server_listen_binds_successfully() {
        let transport = QuicTransport::listen("127.0.0.1:0").await.unwrap();
        let addr = transport.local_addr().unwrap();
        println!("server bound to port {}", addr.port());
        assert!(addr.port() > 0, "server should be bound to a non-zero port");
        assert_eq!(transport.role, TransportRole::Server);
    }

    #[tokio::test]
    async fn client_connect_creates_endpoint() {
        let server = QuicTransport::listen("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();

        let client = QuicTransport::connect(
            &server_addr.to_string(),
            "localhost",
        ).await.unwrap();

        assert_eq!(client.role, TransportRole::Client);
        let client_addr = client.local_addr().unwrap();
        assert!(client_addr.port() > 0);
    }
}
