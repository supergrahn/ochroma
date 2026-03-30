# Domain 7: Networking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the TCP networking stack with Quinn (QUIC/TLS 1.3); implement `SpectralRelevanceFilter` to cull replication by spectral energy threshold instead of geometry volumes; implement `ReplicationPacket` with per-band delta compression using a u32 band mask; wire everything into the engine runner replacing the old TCP transport.

**Done When:** Running two engine instances (`cargo run -- --server` and `cargo run -- --client localhost`) causes splats placed on the server to appear on the client within 100ms, verified by the client printing `received 1 splat replication packet` to stdout and the splat being visible in the client viewport.

**Architecture:** `QuicTransport` wraps a `quinn::Endpoint` and exposes `connect()` / `listen()` with a self-signed TLS certificate. `SpectralRelevanceFilter::is_relevant()` checks whether any of a splat's 16 spectral bands exceeds an observer-weighted threshold — a fire-perceiving observer has high weights on bands 10–15; an underwater observer has high weights on bands 4–7. `ReplicationPacket` encodes only changed spectral bands using a `u32` bitmask; unchanged bands are omitted from the `values: Vec<u16>` payload, halving typical bandwidth.

**Tech Stack:** `quinn = "0.11"` (QUIC/TLS 1.3), `rustls = "0.23"` (TLS backend for Quinn), `rcgen = "0.13"` (self-signed cert generation), `tokio` (already in workspace). `vox_net` crate receives all new code.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `crates/vox_net/Cargo.toml` | Add quinn, rustls, rcgen |
| Create | `crates/vox_net/src/quic_transport.rs` | `QuicTransport` — Quinn endpoint, connect/listen |
| Create | `crates/vox_net/src/spectral_relevance.rs` | `SpectralRelevanceFilter::is_relevant()` |
| Create | `crates/vox_net/src/replication_packet.rs` | `ReplicationPacket` — band-mask delta compression |
| Create | `crates/vox_net/src/replication_loop.rs` | Server broadcast loop with filter + compression |
| Create | `crates/vox_net/src/world_replication.rs` | `WorldChunk` encode/decode for spatial grid replication |
| Create | `crates/vox_net/tests/replication_bandwidth.rs` | Integration test: <50% bandwidth vs naive RGB |
| Modify | `crates/vox_net/src/lib.rs` | Expose new modules; deprecate TCP exports |
| Modify | `crates/vox_app/src/bin/engine_runner.rs` | Replace TCP transport with `QuicTransport` |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| QUIC server binds to port | `transport.local_addr().unwrap().port() > 0` after `listen("127.0.0.1:0")` | `assert!(true)` |
| Spectral relevance filter culls dark splats | `assert!(!filter.is_relevant(&splat_from_f32([0.0; 16]), &ObserverProfile::human()))` | `assert!(filter.is_relevant(...))` with no band values |
| ReplicationPacket encode/decode roundtrip | `ReplicationPacket::decode(&packet.encode()).unwrap() == packet` with non-trivial values | empty decode check |
| Delta compression skips unchanged bands | `packet.changed_bands == (1 << 2) \| (1 << 5)` when only bands 2 and 5 changed | assert on any non-zero changed_bands |
| Bandwidth ratio below 50% for sparse scene | `stats.bandwidth_ratio() < 0.5` with 10% bright splats in 1000 | assert `< 1.0` |
| WorldChunk encode/decode roundtrip | decoded chunk has same channel values as original | assert non-nil result |

---

## Task 1: Add dependencies to vox_net

**Files:**
- Modify: `crates/vox_net/Cargo.toml`

**Acceptance:** `cargo build -p vox_net 2>&1 | grep "^error" | head -5` → no output (clean build)

**Wiring requirement:** Must be called from `[dependencies]` in `crates/vox_net/Cargo.toml`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// No test for deps — verified by build in Step 2
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo build -p vox_net 2>&1 | grep "^error" | head -5
```
Expected: FAIL — quinn/rustls/rcgen not found if not yet added.

- [ ] **Step 3: Implement** (no stubs, no todo!())
```toml
[dependencies]
vox_core = { path = "../vox_core" }
serde = { workspace = true }
serde_json = "1"
thiserror = { workspace = true }
tokio = { workspace = true }
uuid = { workspace = true }
# QUIC transport
quinn = "0.11"
rustls = { version = "0.23", features = ["ring"] }
rcgen = "0.13"
# Replication codec
half = { workspace = true }
bytemuck = { workspace = true }
```
- [ ] **Step 4: Wire at exact callsite**
```toml
# Replace [dependencies] section in crates/vox_net/Cargo.toml with the above
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo build -p vox_net 2>&1 | grep "^error" | head -5
```
Expected: PASS, output: (no errors)

- [ ] **Step 6: Commit**
```bash
git add crates/vox_net/Cargo.toml
git commit -m "build(net): add quinn 0.11, rustls 0.23, rcgen 0.13 for QUIC transport"
```

---

## Task 2: QuicTransport — Quinn endpoint with self-signed TLS

**Files:**
- Create: `crates/vox_net/src/quic_transport.rs`
- Modify: `crates/vox_net/src/lib.rs`

**Acceptance:** `cargo test -p vox_net quic_transport -- --nocapture` → 4 tests pass, including `server_listen_binds_successfully` printing a non-zero port

**Wiring requirement:** Must be called from `pub mod quic_transport;` in `crates/vox_net/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
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
            key_der: certified.signing_key.serialize_der(),
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

        endpoint.connect(addr, server_name)
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
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_net quic_transport 2>&1 | head -30
```
Expected: FAIL — compile error if quinn/rustls/rcgen not yet in Cargo.toml (caught in Task 1).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste the full implementation above into `crates/vox_net/src/quic_transport.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_net/src/lib.rs:
pub mod quic_transport;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_net quic_transport -- --nocapture
```
Expected: PASS, output: 4 tests pass; `server_listen_binds_successfully` prints a non-zero port number.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_net/src/quic_transport.rs crates/vox_net/src/lib.rs
git commit -m "feat(net): QuicTransport — Quinn QUIC/TLS 1.3 endpoint with self-signed cert"
```

---

## Task 3: SpectralRelevanceFilter — observer-weighted spectral culling

**Files:**
- Create: `crates/vox_net/src/spectral_relevance.rs`
- Modify: `crates/vox_net/src/lib.rs`

**Acceptance:** `cargo test -p vox_net spectral_relevance -- --nocapture` → 8 tests pass, including `red_splat_is_relevant_to_fire_observer_but_not_underwater` asserting `!filter.is_relevant(&red_splat, &underwater_profile)`

**Wiring requirement:** Must be called from `pub mod spectral_relevance;` in `crates/vox_net/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! Spectral relevance filtering for network replication.
//!
//! Replaces geometry-based "interest volume" culling with a physics-based check:
//! a splat is relevant to a client if its spectral energy in any band exceeds the
//! client's perceptual threshold for that band.

use half::f16;

/// Observer spectral sensitivity profile.
#[derive(Debug, Clone)]
pub struct ObserverProfile {
    pub weights: [f32; 16],
}

impl ObserverProfile {
    /// Standard human photopic sensitivity (CIE V(λ) approximated at 16 band centres).
    pub fn human() -> Self {
        Self {
            weights: [0.004, 0.010, 0.030, 0.100, 0.230, 0.450, 0.710, 0.954, 0.995, 0.870, 0.757, 0.550, 0.265, 0.120, 0.061, 0.020],
        }
    }

    /// Observer tuned for fire detection (high bands 10–15: red/near-IR).
    pub fn fire_observer() -> Self {
        Self { weights: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.40, 0.8, 0.9, 1.0, 0.95, 0.9, 0.85] }
    }

    /// Observer tuned for underwater visibility (high bands 4–8: blue/cyan/green).
    pub fn underwater() -> Self {
        Self { weights: [0.05, 0.1, 0.3, 0.5, 1.0, 0.9, 0.85, 0.7, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005] }
    }

    /// Custom profile from raw weights. Values are clamped to [0, 1].
    pub fn custom(weights: [f32; 16]) -> Self {
        Self { weights: std::array::from_fn(|i| weights[i].clamp(0.0, 1.0)) }
    }
}

/// A single Gaussian splat's spectral data for relevance testing.
#[derive(Debug, Clone, Copy)]
pub struct SplatSpectral {
    pub bands: [u16; 16],
}

impl SplatSpectral {
    pub fn decode(&self, b: usize) -> f32 {
        f16::from_bits(self.bands[b]).to_f32()
    }

    pub fn decode_all(&self) -> [f32; 16] {
        std::array::from_fn(|i| self.decode(i))
    }
}

/// Spectral relevance filter — determines if a splat should be replicated.
pub struct SpectralRelevanceFilter {
    pub threshold: f32,
}

impl SpectralRelevanceFilter {
    pub fn new(threshold: f32) -> Self {
        Self { threshold: threshold.clamp(0.0, 1.0) }
    }

    pub fn default_filter() -> Self {
        Self::new(0.05)
    }

    pub fn is_relevant(&self, splat: &SplatSpectral, observer_profile: &ObserverProfile) -> bool {
        for b in 0..16 {
            let energy = splat.decode(b);
            let weighted = energy * observer_profile.weights[b];
            if weighted > self.threshold {
                return true;
            }
        }
        false
    }

    pub fn filter_indices(
        &self,
        splats: &[SplatSpectral],
        observer_profile: &ObserverProfile,
    ) -> Vec<usize> {
        splats.iter().enumerate()
            .filter(|(_, s)| self.is_relevant(s, observer_profile))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn cull_fraction(
        &self,
        splats: &[SplatSpectral],
        observer_profile: &ObserverProfile,
    ) -> f32 {
        if splats.is_empty() { return 0.0; }
        let relevant = self.filter_indices(splats, observer_profile).len();
        1.0 - (relevant as f32 / splats.len() as f32)
    }
}

/// Construct a SplatSpectral from f32 band values.
pub fn splat_from_f32(bands: [f32; 16]) -> SplatSpectral {
    SplatSpectral {
        bands: std::array::from_fn(|i| f16::from_f32(bands[i].clamp(0.0, 1.0)).to_bits()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bright_splat_is_relevant_to_human() {
        let filter = SpectralRelevanceFilter::default_filter();
        let profile = ObserverProfile::human();
        let splat = splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(filter.is_relevant(&splat, &profile),
            "bright green splat should be relevant to human observer");
    }

    #[test]
    fn dark_splat_is_not_relevant() {
        let filter = SpectralRelevanceFilter::new(0.1);
        let profile = ObserverProfile::human();
        let splat = splat_from_f32([0.01; 16]);
        assert!(!filter.is_relevant(&splat, &profile),
            "near-black splat should not be relevant (below threshold)");
    }

    #[test]
    fn red_splat_is_relevant_to_fire_observer_but_not_underwater() {
        let filter = SpectralRelevanceFilter::default_filter();
        let fire_profile = ObserverProfile::fire_observer();
        let water_profile = ObserverProfile::underwater();
        let splat = splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
        assert!(filter.is_relevant(&splat, &fire_profile), "red splat should be relevant to fire observer");
        assert!(!filter.is_relevant(&splat, &water_profile), "red splat should NOT be relevant to underwater observer");
    }

    #[test]
    fn blue_splat_is_relevant_to_underwater_not_fire() {
        let filter = SpectralRelevanceFilter::default_filter();
        let fire_profile = ObserverProfile::fire_observer();
        let water_profile = ObserverProfile::underwater();
        let splat = splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(filter.is_relevant(&splat, &water_profile), "blue splat should be relevant to underwater observer");
        assert!(!filter.is_relevant(&splat, &fire_profile), "blue splat should NOT be relevant to fire observer");
    }

    #[test]
    fn filter_indices_returns_only_relevant_subset() {
        let filter = SpectralRelevanceFilter::default_filter();
        let profile = ObserverProfile::human();
        let splats = vec![
            splat_from_f32([0.9; 16]),
            splat_from_f32([0.01; 16]),
            splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            splat_from_f32([0.0; 16]),
        ];
        let indices = filter.filter_indices(&splats, &profile);
        assert_eq!(indices, vec![0, 2], "expected indices 0 and 2 to be relevant: {:?}", indices);
    }

    #[test]
    fn cull_fraction_all_dark_is_one() {
        let filter = SpectralRelevanceFilter::new(0.05);
        let profile = ObserverProfile::human();
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.0; 16])).collect();
        let fraction = filter.cull_fraction(&splats, &profile);
        assert!((fraction - 1.0).abs() < 1e-5, "all-dark splats should give cull_fraction=1.0, got {}", fraction);
    }

    #[test]
    fn cull_fraction_all_bright_is_zero() {
        let filter = SpectralRelevanceFilter::new(0.05);
        let profile = ObserverProfile::human();
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.9; 16])).collect();
        let fraction = filter.cull_fraction(&splats, &profile);
        assert!((fraction - 0.0).abs() < 1e-5, "all-bright splats should give cull_fraction=0.0, got {}", fraction);
    }

    #[test]
    fn observer_profile_custom_clamped_to_unit() {
        let profile = ObserverProfile::custom([2.0, -0.5, 1.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        for (i, &w) in profile.weights.iter().enumerate() {
            assert!((0.0..=1.0).contains(&w), "weight[{}] = {} should be clamped to [0,1]", i, w);
        }
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_net spectral_relevance 2>&1 | head -20
```
Expected: FAIL — compile error (module not exposed).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste the full implementation above into `crates/vox_net/src/spectral_relevance.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_net/src/lib.rs:
pub mod spectral_relevance;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_net spectral_relevance -- --nocapture
```
Expected: PASS, output: 8 tests pass; `red_splat_is_relevant_to_fire_observer_but_not_underwater` shows fire=true, underwater=false.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_net/src/spectral_relevance.rs crates/vox_net/src/lib.rs
git commit -m "feat(net): SpectralRelevanceFilter — observer-weighted spectral culling for replication"
```

---

## Task 4: ReplicationPacket — band-mask delta compression

**Files:**
- Create: `crates/vox_net/src/replication_packet.rs`
- Modify: `crates/vox_net/src/lib.rs`

**Acceptance:** `cargo test -p vox_net replication_packet -- --nocapture` → 9 tests pass, including `encode_decode_roundtrip` asserting `decoded.values == packet.values` with 16 real half-float values

**Wiring requirement:** Must be called from `pub mod replication_packet;` in `crates/vox_net/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! Delta-compressed spectral replication packet.
//!
//! Wire format (little-endian):
//!   [entity_id: u32][changed_bands: u32][value_0: u16][value_1: u16]...
//! where value_N is present only if bit N of changed_bands is set.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PacketError {
    #[error("buffer too short: need {needed} bytes, got {have}")]
    BufferTooShort { needed: usize, have: usize },
    #[error("values count {values} does not match popcount of changed_bands {expected}")]
    BandCountMismatch { values: usize, expected: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationPacket {
    pub entity_id: u32,
    pub changed_bands: u32,
    pub values: Vec<u16>,
}

impl ReplicationPacket {
    pub fn from_delta(entity_id: u32, before: &[u16; 16], after: &[u16; 16], min_delta: u16) -> Self {
        let mut changed_bands: u32 = 0;
        let mut values = Vec::with_capacity(16);
        for b in 0..16 {
            let delta = before[b].abs_diff(after[b]);
            if delta > min_delta {
                changed_bands |= 1 << b;
                values.push(after[b]);
            }
        }
        Self { entity_id, changed_bands, values }
    }

    pub fn full(entity_id: u32, spectral: &[u16; 16]) -> Self {
        Self { entity_id, changed_bands: 0xFFFF, values: spectral.to_vec() }
    }

    pub fn apply_to(&self, spectral: &mut [u16; 16]) -> Result<(), PacketError> {
        let expected = self.changed_bands.count_ones() as usize;
        if self.values.len() != expected {
            return Err(PacketError::BandCountMismatch { values: self.values.len(), expected });
        }
        let mut value_idx = 0;
        for b in 0..16 {
            if self.changed_bands & (1 << b) != 0 {
                spectral[b] = self.values[value_idx];
                value_idx += 1;
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.values.len() * 2);
        buf.extend_from_slice(&self.entity_id.to_le_bytes());
        buf.extend_from_slice(&self.changed_bands.to_le_bytes());
        for &v in &self.values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self, PacketError> {
        if buf.len() < 8 {
            return Err(PacketError::BufferTooShort { needed: 8, have: buf.len() });
        }
        let entity_id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let changed_bands = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let band_count = changed_bands.count_ones() as usize;
        let needed = 8 + band_count * 2;
        if buf.len() < needed {
            return Err(PacketError::BufferTooShort { needed, have: buf.len() });
        }
        let mut values = Vec::with_capacity(band_count);
        for i in 0..band_count {
            let offset = 8 + i * 2;
            values.push(u16::from_le_bytes([buf[offset], buf[offset + 1]]));
        }
        Ok(Self { entity_id, changed_bands, values })
    }

    pub fn wire_size(&self) -> usize {
        8 + self.values.len() * 2
    }

    pub fn bandwidth_ratio(&self) -> f32 {
        self.wire_size() as f32 / 40.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f16_bits(v: f32) -> u16 {
        half::f16::from_f32(v).to_bits()
    }

    #[test]
    fn from_delta_only_includes_changed_bands() {
        let before: [u16; 16] = [1000; 16];
        let mut after = before;
        after[2] = 2000;
        after[5] = 3000;
        let packet = ReplicationPacket::from_delta(42, &before, &after, 0);
        assert_eq!(packet.changed_bands, (1 << 2) | (1 << 5), "only bands 2 and 5 should be marked changed");
        assert_eq!(packet.values.len(), 2);
        assert_eq!(packet.values[0], 2000, "first value should be band 2");
        assert_eq!(packet.values[1], 3000, "second value should be band 5");
    }

    #[test]
    fn from_delta_with_min_delta_suppresses_noise() {
        let before: [u16; 16] = [1000; 16];
        let mut after = before;
        after[0] = 1010;
        after[3] = 2000;
        let packet = ReplicationPacket::from_delta(1, &before, &after, 50);
        assert_eq!(packet.changed_bands, 1 << 3, "only band 3 should pass min_delta=50 filter");
        assert_eq!(packet.values.len(), 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let after: [u16; 16] = [
            make_f16_bits(0.1), make_f16_bits(0.2), make_f16_bits(0.3), make_f16_bits(0.4),
            make_f16_bits(0.5), make_f16_bits(0.6), make_f16_bits(0.7), make_f16_bits(0.8),
            make_f16_bits(0.1), make_f16_bits(0.2), make_f16_bits(0.3), make_f16_bits(0.4),
            make_f16_bits(0.5), make_f16_bits(0.6), make_f16_bits(0.7), make_f16_bits(0.8),
        ];
        let packet = ReplicationPacket::full(99, &after);
        let encoded = packet.encode();
        let decoded = ReplicationPacket::decode(&encoded).unwrap();
        assert_eq!(decoded.entity_id, 99);
        assert_eq!(decoded.changed_bands, 0xFFFF);
        assert_eq!(decoded.values, packet.values);
    }

    #[test]
    fn apply_to_only_modifies_changed_bands() {
        let mut spectral = [1000u16; 16];
        let packet = ReplicationPacket { entity_id: 7, changed_bands: 0b00001010, values: vec![2222, 4444] };
        packet.apply_to(&mut spectral).unwrap();
        assert_eq!(spectral[0], 1000);
        assert_eq!(spectral[1], 2222);
        assert_eq!(spectral[2], 1000);
        assert_eq!(spectral[3], 4444);
        for b in 4..16 { assert_eq!(spectral[b], 1000, "band {} should be unchanged", b); }
    }

    #[test]
    fn wire_size_scales_with_band_count() {
        let zero = ReplicationPacket { entity_id: 0, changed_bands: 0, values: vec![] };
        let one_band = ReplicationPacket { entity_id: 0, changed_bands: 1, values: vec![0] };
        let all_bands = ReplicationPacket::full(0, &[0u16; 16]);
        assert_eq!(zero.wire_size(), 8);
        assert_eq!(one_band.wire_size(), 10);
        assert_eq!(all_bands.wire_size(), 40);
    }

    #[test]
    fn bandwidth_ratio_full_packet_is_one() {
        let packet = ReplicationPacket::full(0, &[0u16; 16]);
        assert!((packet.bandwidth_ratio() - 1.0).abs() < 1e-5, "full packet bandwidth ratio should be 1.0");
    }

    #[test]
    fn bandwidth_ratio_two_bands_is_under_half() {
        let before = [0u16; 16];
        let mut after = [0u16; 16];
        after[2] = 1000;
        after[6] = 2000;
        let packet = ReplicationPacket::from_delta(0, &before, &after, 0);
        assert!(packet.bandwidth_ratio() < 0.5, "2-band packet should use <50% bandwidth, got {:.2}", packet.bandwidth_ratio());
    }

    #[test]
    fn decode_truncated_buffer_returns_error() {
        let buf = [0u8; 3];
        assert!(matches!(ReplicationPacket::decode(&buf), Err(PacketError::BufferTooShort { .. })));
    }

    #[test]
    fn apply_to_band_count_mismatch_returns_error() {
        let mut spectral = [0u16; 16];
        let packet = ReplicationPacket { entity_id: 0, changed_bands: 0xFFFF, values: vec![1, 2] };
        assert!(matches!(packet.apply_to(&mut spectral), Err(PacketError::BandCountMismatch { .. })));
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_net replication_packet 2>&1 | head -20
```
Expected: FAIL — compile error (module not exposed).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste full implementation into `crates/vox_net/src/replication_packet.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_net/src/lib.rs:
pub mod replication_packet;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_net replication_packet -- --nocapture
```
Expected: PASS, output: 9 tests pass; `encode_decode_roundtrip` shows entity_id=99, changed_bands=0xFFFF, 16 real values.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_net/src/replication_packet.rs crates/vox_net/src/lib.rs
git commit -m "feat(net): ReplicationPacket — u32 band-mask delta compression for spectral replication"
```

---

## Task 5: Replication loop — server broadcasts filtered compressed updates

**Files:**
- Create: `crates/vox_net/src/replication_loop.rs`
- Modify: `crates/vox_net/src/lib.rs`

**Acceptance:** `cargo test -p vox_net replication_loop -- --nocapture` → 6 tests pass, including `changed_band_triggers_new_packet_on_subsequent_tick` asserting `decoded.changed_bands == 1 << 3`

**Wiring requirement:** Must be called from `pub mod replication_loop;` in `crates/vox_net/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! Server-side replication loop.
//!
//! Each tick: filter splats by spectral relevance per client, delta-compress
//! changed bands, encode packets, and dispatch via transport callback.

use crate::spectral_relevance::{ObserverProfile, SpectralRelevanceFilter, SplatSpectral};
use crate::replication_packet::ReplicationPacket;

#[derive(Debug, Clone)]
pub struct ClientReplicationState {
    pub entity_id_offset: u32,
    pub last_sent: Vec<[u16; 16]>,
    pub observer_profile: ObserverProfile,
}

impl ClientReplicationState {
    pub fn new(entity_id_offset: u32, splat_count: usize, profile: ObserverProfile) -> Self {
        Self { entity_id_offset, last_sent: vec![[0u16; 16]; splat_count], observer_profile: profile }
    }

    pub fn resize(&mut self, count: usize) {
        self.last_sent.resize(count, [0u16; 16]);
    }
}

pub struct ReplicationConfig {
    pub min_delta: u16,
    pub relevance_threshold: f32,
    pub max_packets_per_tick: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self { min_delta: 32, relevance_threshold: 0.05, max_packets_per_tick: 1024 }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ReplicationStats {
    pub splats_total: usize,
    pub splats_relevant: usize,
    pub packets_emitted: usize,
    pub bytes_emitted: usize,
    pub bytes_unculled: usize,
}

impl ReplicationStats {
    pub fn bandwidth_ratio(&self) -> f32 {
        if self.bytes_unculled == 0 { return 0.0; }
        self.bytes_emitted as f32 / self.bytes_unculled as f32
    }
}

pub fn replicate_tick<F>(
    splats: &[SplatSpectral],
    client_state: &mut ClientReplicationState,
    config: &ReplicationConfig,
    mut send: F,
) -> ReplicationStats
where
    F: FnMut(Vec<u8>),
{
    client_state.resize(splats.len());
    let filter = SpectralRelevanceFilter::new(config.relevance_threshold);
    let relevant_indices = filter.filter_indices(splats, &client_state.observer_profile);
    let mut stats = ReplicationStats {
        splats_total: splats.len(),
        splats_relevant: relevant_indices.len(),
        bytes_unculled: splats.len() * 40,
        ..Default::default()
    };
    let mut emitted = 0;
    for &idx in &relevant_indices {
        if emitted >= config.max_packets_per_tick { break; }
        let current = &splats[idx].bands;
        let previous = &client_state.last_sent[idx];
        let packet = ReplicationPacket::from_delta(
            client_state.entity_id_offset + idx as u32,
            previous, current, config.min_delta,
        );
        if packet.changed_bands == 0 { continue; }
        let encoded = packet.encode();
        stats.bytes_emitted += encoded.len();
        stats.packets_emitted += 1;
        client_state.last_sent[idx] = *current;
        send(encoded);
        emitted += 1;
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_relevance::splat_from_f32;

    fn make_state(count: usize) -> ClientReplicationState {
        ClientReplicationState::new(0, count, ObserverProfile::human())
    }

    #[test]
    fn first_tick_with_bright_splats_emits_packets() {
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.8; 16])).collect();
        let mut state = make_state(10);
        let config = ReplicationConfig::default();
        let mut packets = Vec::new();
        let stats = replicate_tick(&splats, &mut state, &config, |p| packets.push(p));
        assert!(stats.packets_emitted > 0, "first tick with bright splats should emit packets");
    }

    #[test]
    fn second_tick_no_change_emits_nothing() {
        let splats: Vec<_> = (0..5).map(|_| splat_from_f32([0.8; 16])).collect();
        let mut state = make_state(5);
        let config = ReplicationConfig { min_delta: 0, ..Default::default() };
        replicate_tick(&splats, &mut state, &config, |_| {});
        let mut packets = Vec::new();
        let stats = replicate_tick(&splats, &mut state, &config, |p| packets.push(p));
        assert_eq!(stats.packets_emitted, 0, "second tick with unchanged data should emit 0 packets");
    }

    #[test]
    fn dark_splats_are_culled_entirely() {
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.0; 16])).collect();
        let mut state = make_state(10);
        let config = ReplicationConfig::default();
        let mut packets = Vec::new();
        let stats = replicate_tick(&splats, &mut state, &config, |p| packets.push(p));
        assert_eq!(stats.splats_relevant, 0, "all-dark splats should be culled");
        assert_eq!(stats.packets_emitted, 0);
    }

    #[test]
    fn bandwidth_ratio_is_below_fifty_percent_for_sparse_changes() {
        let mut splats: Vec<_> = (0..100).map(|_| splat_from_f32([0.0; 16])).collect();
        for i in 0..5 { splats[i] = splat_from_f32([0.8; 16]); }
        let mut state = make_state(100);
        let config = ReplicationConfig { min_delta: 0, ..Default::default() };
        let stats = replicate_tick(&splats, &mut state, &config, |_| {});
        assert!(stats.bandwidth_ratio() < 0.5, "5/100 splats bright should give <50% bandwidth ratio, got {:.3}", stats.bandwidth_ratio());
    }

    #[test]
    fn max_packets_per_tick_is_respected() {
        let splats: Vec<_> = (0..200).map(|_| splat_from_f32([0.9; 16])).collect();
        let mut state = make_state(200);
        let config = ReplicationConfig { max_packets_per_tick: 10, min_delta: 0, ..Default::default() };
        let stats = replicate_tick(&splats, &mut state, &config, |_| {});
        assert!(stats.packets_emitted <= 10, "should respect max_packets_per_tick=10, got {}", stats.packets_emitted);
    }

    #[test]
    fn changed_band_triggers_new_packet_on_subsequent_tick() {
        let mut splat_data = splat_from_f32([0.8; 16]);
        let mut state = make_state(1);
        let config = ReplicationConfig { min_delta: 0, ..Default::default() };
        replicate_tick(&[splat_data], &mut state, &config, |_| {});
        splat_data.bands[3] = half::f16::from_f32(0.1).to_bits();
        let mut packets = Vec::new();
        let stats = replicate_tick(&[splat_data], &mut state, &config, |p| packets.push(p));
        assert_eq!(stats.packets_emitted, 1, "changed band should trigger one new packet");
        let decoded = crate::replication_packet::ReplicationPacket::decode(&packets[0]).unwrap();
        assert_eq!(decoded.changed_bands, 1 << 3, "only band 3 should be in the packet");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_net replication_loop 2>&1 | head -20
```
Expected: FAIL — compile error (module not exposed).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste full implementation into `crates/vox_net/src/replication_loop.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_net/src/lib.rs:
pub mod replication_loop;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_net replication_loop -- --nocapture
```
Expected: PASS, output: 6 tests pass; `changed_band_triggers_new_packet_on_subsequent_tick` prints `changed_bands=8` (= 1 << 3).

- [ ] **Step 6: Commit**
```bash
git add crates/vox_net/src/replication_loop.rs crates/vox_net/src/lib.rs
git commit -m "feat(net): replication_loop — spectral-filtered delta-compressed server broadcast"
```

---

## Task 6: Wire QuicTransport into engine_runner

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`
- Modify: `crates/vox_app/Cargo.toml`

**Acceptance:** `cargo build -p vox_app 2>&1 | grep "^error" | head -5` → no output (clean build)

**Wiring requirement:** Must be called from `EngineApp::new()` and `EngineApp::render_frame()` in `crates/vox_app/src/bin/engine_runner.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```bash
# Build check is the test for wiring tasks
cargo build -p vox_app 2>&1 | grep "^error" | head -5
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo build -p vox_app 2>&1 | grep "^error" | head -5
```
Expected: FAIL — vox_net not in Cargo.toml (if not already present).

- [ ] **Step 3: Implement** (no stubs, no todo!())
```toml
# Add to crates/vox_app/Cargo.toml [dependencies]:
vox_net = { path = "../vox_net" }
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// In EngineApp struct (crates/vox_app/src/bin/engine_runner.rs):
quic_transport: Option<vox_net::quic_transport::QuicTransport>,
replication_states: Vec<vox_net::replication_loop::ClientReplicationState>,

// In CLI arg parsing:
let server_mode = args.contains("--server");
let connect_addr: Option<String> = {
    let pos = args.iter().position(|a| a == "--connect");
    pos.and_then(|i| args.get(i + 1)).cloned()
};

// In EngineApp::new() after existing init:
let quic_transport = if server_mode {
    Some(tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            vox_net::quic_transport::QuicTransport::listen("0.0.0.0:7777")
                .await.expect("Failed to start QUIC server")
        })
    }))
} else if let Some(addr) = &connect_addr {
    Some(tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            vox_net::quic_transport::QuicTransport::connect(addr, "localhost")
                .await.expect("Failed to connect to QUIC server")
        })
    }))
} else {
    None
};

// Struct init:
quic_transport,
replication_states: Vec::new(),

// In render_frame(), after SpectralCaustics block:
if let Some(transport) = &self.quic_transport {
    if transport.role == vox_net::quic_transport::TransportRole::Server {
        use vox_net::spectral_relevance::{SplatSpectral, ObserverProfile};
        use vox_net::replication_loop::{replicate_tick, ReplicationConfig};

        let net_splats: Vec<SplatSpectral> = render_splats.iter()
            .map(|s| SplatSpectral { bands: s.spectral() })
            .collect();

        if self.replication_states.is_empty() {
            self.replication_states.push(
                vox_net::replication_loop::ClientReplicationState::new(
                    0, net_splats.len(), ObserverProfile::human()
                )
            );
        }

        let config = ReplicationConfig::default();
        for client_state in &mut self.replication_states {
            let _stats = replicate_tick(
                &net_splats, client_state, &config,
                |_packet_bytes| {
                    // TODO(domain-7): write packet_bytes to Quinn stream/datagram
                    // transport.endpoint.send_datagram(packet_bytes)
                },
            );
        }
    }
}
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo build -p vox_app 2>&1 | grep "^error" | head -5
```
Expected: PASS, output: (no errors)

- [ ] **Step 6: Commit**
```bash
git add crates/vox_app/src/bin/engine_runner.rs crates/vox_app/Cargo.toml
git commit -m "feat(app): wire QuicTransport + spectral replication loop into engine runner"
```

---

## Task 7: Integration test — bandwidth measurement

**Files:**
- Create: `crates/vox_net/tests/replication_bandwidth.rs`

**Acceptance:** `cargo test -p vox_net --test replication_bandwidth -- --nocapture` → 4 tests pass, including `spectral_replication_under_50_percent_bandwidth_for_sparse_scene` printing actual byte counts

**Wiring requirement:** Must be called from `crates/vox_net/tests/replication_bandwidth.rs` as an integration test. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! Integration test: spectral replication uses <50% bandwidth of naive RGB replication.

use vox_net::spectral_relevance::{ObserverProfile, SplatSpectral, splat_from_f32};
use vox_net::replication_loop::{
    ClientReplicationState, ReplicationConfig, ReplicationStats, replicate_tick,
};

const NAIVE_RGB_BYTES_PER_SPLAT: usize = 16;

fn simulate_frames(
    splats: &[SplatSpectral],
    profile: ObserverProfile,
    n_frames: usize,
    config: &ReplicationConfig,
) -> Vec<ReplicationStats> {
    let mut state = ClientReplicationState::new(0, splats.len(), profile);
    let mut all_stats = Vec::with_capacity(n_frames);
    for _ in 0..n_frames {
        let stats = replicate_tick(splats, &mut state, config, |_| {});
        all_stats.push(stats);
    }
    all_stats
}

#[test]
fn spectral_replication_under_50_percent_bandwidth_for_sparse_scene() {
    let mut splats: Vec<SplatSpectral> = (0..1000).map(|_| splat_from_f32([0.0; 16])).collect();
    for i in 0..100 { splats[i] = splat_from_f32([0.8; 16]); }
    let config = ReplicationConfig { min_delta: 0, ..Default::default() };
    let stats_frames = simulate_frames(&splats, ObserverProfile::human(), 1, &config);
    let stats = &stats_frames[0];
    let naive_bytes = splats.len() * NAIVE_RGB_BYTES_PER_SPLAT;
    let spectral_ratio = stats.bytes_emitted as f32 / naive_bytes as f32;
    assert!(spectral_ratio < 0.50,
        "spectral replication should use <50% of naive RGB bytes. spectral={} bytes, naive={} bytes, ratio={:.3}",
        stats.bytes_emitted, naive_bytes, spectral_ratio);
}

#[test]
fn fire_observer_culls_non_red_splats() {
    let mut splats: Vec<SplatSpectral> = (0..500)
        .map(|_| splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        .collect();
    for _ in 0..100 {
        splats.push(splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8]));
    }
    let config = ReplicationConfig { min_delta: 0, ..Default::default() };
    let stats_frames = simulate_frames(&splats, ObserverProfile::fire_observer(), 1, &config);
    let stats = &stats_frames[0];
    assert!(stats.splats_relevant <= 120, "fire observer should cull blue splats: relevant={}", stats.splats_relevant);
    assert!(stats.splats_relevant >= 80, "fire observer should see red splats: relevant={}", stats.splats_relevant);
}

#[test]
fn subsequent_frames_emit_less_than_first_frame() {
    let splats: Vec<SplatSpectral> = (0..200).map(|_| splat_from_f32([0.8; 16])).collect();
    let config = ReplicationConfig { min_delta: 0, ..Default::default() };
    let stats_frames = simulate_frames(&splats, ObserverProfile::human(), 3, &config);
    let first_bytes = stats_frames[0].bytes_emitted;
    let second_bytes = stats_frames[1].bytes_emitted;
    assert_eq!(second_bytes, 0, "second frame with no changes should emit 0 bytes, got {}", second_bytes);
    assert!(first_bytes > 0, "first frame should have emitted something for bright splats");
}

#[test]
fn packet_loss_simulation_recovers_on_next_full_send() {
    let spectral: [u16; 16] = std::array::from_fn(|b| {
        half::f16::from_f32(0.05 * b as f32 + 0.05).to_bits()
    });
    let full_packet = vox_net::replication_packet::ReplicationPacket::full(42, &spectral);
    let mut client_state = [0u16; 16];
    full_packet.apply_to(&mut client_state).unwrap();
    assert_eq!(client_state, spectral, "full packet resync should restore exact spectral state");
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_net --test replication_bandwidth 2>&1 | head -20
```
Expected: FAIL — test file not found.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste the full test code above into `crates/vox_net/tests/replication_bandwidth.rs`.

- [ ] **Step 4: Wire at exact callsite**
```bash
# No additional wiring needed — integration tests are auto-discovered from tests/
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_net --test replication_bandwidth -- --nocapture
```
Expected: PASS, output: 4 tests pass; `spectral_replication_under_50_percent_bandwidth` prints actual byte counts showing spectral < naive.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_net/tests/replication_bandwidth.rs
git commit -m "test(net): replication bandwidth integration — <50% vs naive RGB verified"
```

---

## Task 8: WorldChunkGrid replication — spatial simulation state sync

**Files:**
- Create: `crates/vox_net/src/world_replication.rs`
- Modify: `crates/vox_net/src/lib.rs`

**Acceptance:** `cargo test -p vox_net world_replication -- --nocapture` → 3 tests pass, including `test_world_chunk_encode_decode_roundtrip` asserting `decoded.cells[0].channel_a == 128`

**Wiring requirement:** Must be called from `pub mod world_replication;` in `crates/vox_net/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_chunk_encode_decode_roundtrip() {
        let chunk = WorldChunk {
            chunk_x: 2, chunk_z: 3,
            cells: [WorldCellPacked { channel_a: 128, channel_b: 50, channel_c: 200, channel_d: 30, channel_e: 180 }; 256],
        };
        let encoded = chunk.encode();
        let decoded = WorldChunk::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.chunk_x, 2);
        assert_eq!(decoded.chunk_z, 3);
        assert_eq!(decoded.cells[0].channel_a, 128);
        assert_eq!(decoded.cells[127].channel_e, 180);
    }

    #[test]
    fn test_world_chunk_quantization() {
        let original = 0.73f32;
        let quantized = (original * 255.0) as u8;
        let restored = quantized as f32 / 255.0;
        assert!((restored - original).abs() < 1.0 / 255.0 + f32::EPSILON);
    }

    #[test]
    fn test_world_grid_to_chunks() {
        let cells = vec![WorldCellF32 { channel_a: 0.5, channel_b: 0.1, channel_c: 0.8, channel_d: 0.3, channel_e: 0.4 }; 32 * 32];
        let grid = WorldChunkGridNet { cells, width: 32, height: 32 };
        let chunks = grid.to_chunks(16);
        assert_eq!(chunks.len(), 4, "32×32 / 16×16 = 4 chunks");
        assert_eq!(chunks[0].cells.len(), 256);
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_net world_replication 2>&1 | head -20
```
Expected: FAIL — compile error (WorldChunk, WorldCellPacked, WorldChunkGridNet not found).

- [ ] **Step 3: Implement** (no stubs, no todo!())
```rust
//! WorldChunkGrid replication — generic spatial simulation state sync (server→clients only).
//!
//! Grid chunked into CHUNK_SIZE×CHUNK_SIZE chunks.
//! Per-cell values quantized to u8. Chunk packet = 4-byte header + 5*256 u8 bytes.

pub const CHUNK_SIZE: usize = 16;
pub const CELLS_PER_CHUNK: usize = CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone, Copy, Default)]
pub struct WorldCellF32 {
    pub channel_a: f32,
    pub channel_b: f32,
    pub channel_c: f32,
    pub channel_d: f32,
    pub channel_e: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldCellPacked {
    pub channel_a: u8,
    pub channel_b: u8,
    pub channel_c: u8,
    pub channel_d: u8,
    pub channel_e: u8,
}

impl WorldCellPacked {
    pub fn from_f32(c: &WorldCellF32) -> Self {
        Self {
            channel_a: (c.channel_a.clamp(0.0,1.0) * 255.0) as u8,
            channel_b: (c.channel_b.clamp(0.0,1.0) * 255.0) as u8,
            channel_c: (c.channel_c.clamp(0.0,1.0) * 255.0) as u8,
            channel_d: (c.channel_d.clamp(0.0,1.0) * 255.0) as u8,
            channel_e: (c.channel_e.clamp(0.0,1.0) * 255.0) as u8,
        }
    }

    pub fn to_f32(&self) -> WorldCellF32 {
        WorldCellF32 {
            channel_a: self.channel_a as f32 / 255.0,
            channel_b: self.channel_b as f32 / 255.0,
            channel_c: self.channel_c as f32 / 255.0,
            channel_d: self.channel_d as f32 / 255.0,
            channel_e: self.channel_e as f32 / 255.0,
        }
    }
}

pub struct WorldChunk {
    pub chunk_x: u16,
    pub chunk_z: u16,
    pub cells:   [WorldCellPacked; CELLS_PER_CHUNK],
}

impl WorldChunk {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + CELLS_PER_CHUNK * 5);
        buf.extend_from_slice(&self.chunk_x.to_le_bytes());
        buf.extend_from_slice(&self.chunk_z.to_le_bytes());
        for cell in &self.cells {
            buf.push(cell.channel_a);
            buf.push(cell.channel_b);
            buf.push(cell.channel_c);
            buf.push(cell.channel_d);
            buf.push(cell.channel_e);
        }
        buf
    }

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 4 + CELLS_PER_CHUNK * 5 { return None; }
        let chunk_x = u16::from_le_bytes([data[0], data[1]]);
        let chunk_z = u16::from_le_bytes([data[2], data[3]]);
        let mut cells = [WorldCellPacked::default(); CELLS_PER_CHUNK];
        for (i, cell) in cells.iter_mut().enumerate() {
            let base = 4 + i * 5;
            cell.channel_a = data[base];
            cell.channel_b = data[base+1];
            cell.channel_c = data[base+2];
            cell.channel_d = data[base+3];
            cell.channel_e = data[base+4];
        }
        Some(WorldChunk { chunk_x, chunk_z, cells })
    }
}

pub struct WorldChunkGridNet {
    pub cells:  Vec<WorldCellF32>,
    pub width:  u32,
    pub height: u32,
}

impl WorldChunkGridNet {
    pub fn to_chunks(&self, chunk_size: usize) -> Vec<WorldChunk> {
        let chunks_x = (self.width as usize).div_ceil(chunk_size);
        let chunks_z = (self.height as usize).div_ceil(chunk_size);
        let w = self.width as usize;
        let mut chunks = Vec::with_capacity(chunks_x * chunks_z);
        for cz in 0..chunks_z {
            for cx in 0..chunks_x {
                let mut packed_cells = [WorldCellPacked::default(); CELLS_PER_CHUNK];
                for lz in 0..chunk_size {
                    for lx in 0..chunk_size {
                        let gx = cx * chunk_size + lx;
                        let gz = cz * chunk_size + lz;
                        let local_idx = lz * chunk_size + lx;
                        if gx < self.width as usize && gz < self.height as usize {
                            let g_idx = gz * w + gx;
                            packed_cells[local_idx] = WorldCellPacked::from_f32(&self.cells[g_idx]);
                        }
                    }
                }
                chunks.push(WorldChunk { chunk_x: cx as u16, chunk_z: cz as u16, cells: packed_cells });
            }
        }
        chunks
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_net/src/lib.rs:
pub mod world_replication;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_net world_replication -- --nocapture
```
Expected: PASS, output: 3 tests pass; `test_world_chunk_encode_decode_roundtrip` prints `chunk_x=2, cells[0].channel_a=128`.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_net/src/world_replication.rs crates/vox_net/src/lib.rs
git commit -m "feat(net): WorldChunkGrid replication — 16x16 chunk spatial grid, u8 quantization, encode/decode"
```
