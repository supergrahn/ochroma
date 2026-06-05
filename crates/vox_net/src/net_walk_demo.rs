//! Headless 2-player networked walk demo over QUIC loopback.
//!
//! This drives the *real* [`QuicServer`]/[`QuicClient`] transport: a host and a
//! client each own one walking player (position + 16-band spectral signature) and
//! replicate their own state to the peer every tick via [`PlayerStatePacket`].
//!
//! The whole thing is a blocking function ([`run_loopback_walk_demo`]) that owns a
//! tokio runtime internally, so callers (e.g. the `net_walk` binary) need no async
//! runtime of their own. Each side independently tracks the *other* player's latest
//! replicated position; after the run the report exposes both views plus per-side
//! message counts so callers can assert honest replication occurred.

use crate::quic_transport::TransportError;
use crate::replication_packet::PlayerStatePacket;
use crate::{QuicClient, QuicServer};

/// Configuration for one demo run. Movement is fully deterministic so the final
/// positions can be predicted exactly from these numbers.
#[derive(Debug, Clone, Copy)]
pub struct WalkDemoConfig {
    /// Number of simulation ticks each side runs.
    pub ticks: u32,
    /// Seconds of simulated movement per tick.
    pub dt: f32,
    /// Host walks along +X at this speed (m/s).
    pub host_speed_x: f32,
    /// Client walks along +Z at this speed (m/s).
    pub client_speed_z: f32,
}

impl Default for WalkDemoConfig {
    fn default() -> Self {
        Self { ticks: 60, dt: 0.016, host_speed_x: 2.0, client_speed_z: 3.0 }
    }
}

impl WalkDemoConfig {
    /// Predicted host final position after all ticks: walked +X only.
    pub fn expected_host_final(&self) -> [f32; 3] {
        [self.host_speed_x * self.dt * self.ticks as f32, 0.0, 0.0]
    }

    /// Predicted client final position after all ticks: walked +Z only.
    pub fn expected_client_final(&self) -> [f32; 3] {
        [0.0, 0.0, self.client_speed_z * self.dt * self.ticks as f32]
    }

    /// One tick of host movement along X (used as the within-one-tick tolerance).
    pub fn host_tick_step(&self) -> f32 {
        self.host_speed_x * self.dt
    }

    /// One tick of client movement along Z (used as the within-one-tick tolerance).
    pub fn client_tick_step(&self) -> f32 {
        self.client_speed_z * self.dt
    }
}

/// Result of a completed demo run. `*_saw_*` fields are each side's last-received
/// view of the peer's position.
#[derive(Debug, Clone, Copy)]
pub struct WalkDemoReport {
    /// Host's own final position (authoritative, locally simulated).
    pub host_final: [f32; 3],
    /// Client's own final position (authoritative, locally simulated).
    pub client_final: [f32; 3],
    /// Position the host received for the client (its latest replicated state).
    pub host_saw_client: [f32; 3],
    /// Position the client received for the host (its latest replicated state).
    pub client_saw_host: [f32; 3],
    /// Number of player-state packets the host received from the client.
    pub msgs_host: u32,
    /// Number of player-state packets the client received from the host.
    pub msgs_client: u32,
    /// Last spectral signature the host received from the client.
    pub host_saw_client_spectral: [u16; 16],
    /// Last spectral signature the client received from the host.
    pub client_saw_host_spectral: [u16; 16],
}

#[derive(Debug, thiserror::Error)]
pub enum WalkDemoError {
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),
    #[error("runtime build failed: {0}")]
    Runtime(String),
    #[error("task join failed: {0}")]
    Join(String),
}

const HOST_ENTITY_ID: u32 = 1;
const CLIENT_ENTITY_ID: u32 = 2;

/// Per-tick spectral signature for a player: a simple deterministic function of
/// the entity id and tick so both sides can be cross-checked.
fn spectral_for(entity_id: u32, tick: u32) -> [u16; 16] {
    let mut s = [0u16; 16];
    for (b, v) in s.iter_mut().enumerate() {
        *v = (entity_id as u16)
            .wrapping_mul(1000)
            .wrapping_add(tick as u16)
            .wrapping_add(b as u16);
    }
    s
}

/// Run the headless 2-player loopback walk demo to completion (blocking).
///
/// Spins up a multi-thread tokio runtime, hosts a QUIC server, connects a client
/// over loopback, then runs both players for `cfg.ticks` ticks. Each tick a side
/// moves its own player and sends a [`PlayerStatePacket`]; concurrently it receives
/// the peer's packet and records it. Returns a [`WalkDemoReport`] with both sides'
/// views and message counts.
pub fn run_loopback_walk_demo(cfg: WalkDemoConfig) -> Result<WalkDemoReport, WalkDemoError> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| WalkDemoError::Runtime(e.to_string()))?;

    rt.block_on(async move { run_async(cfg).await })
}

async fn run_async(cfg: WalkDemoConfig) -> Result<WalkDemoReport, WalkDemoError> {
    let server = QuicServer::listen("127.0.0.1:0").await?;
    let server_addr = server.local_addr()?;

    // Host side: accept the established connection in a task so the client's
    // connect handshake and the server's accept run concurrently.
    let host_task = tokio::spawn(async move {
        let conn = server.accept().await?;
        run_side(
            conn,
            HOST_ENTITY_ID,
            move |tick| {
                // Host walks +X only.
                [cfg.host_speed_x * cfg.dt * tick as f32, 0.0, 0.0]
            },
            cfg.ticks,
        )
        .await
    });

    let client = QuicClient::connect(&server_addr.to_string(), "localhost").await?;
    let client_conn = client.connection().clone();

    let client_result = run_side(
        client_conn,
        CLIENT_ENTITY_ID,
        move |tick| {
            // Client walks +Z only.
            [0.0, 0.0, cfg.client_speed_z * cfg.dt * tick as f32]
        },
        cfg.ticks,
    )
    .await?;

    let host_result = host_task
        .await
        .map_err(|e| WalkDemoError::Join(e.to_string()))??;

    Ok(WalkDemoReport {
        host_final: host_result.own_final,
        client_final: client_result.own_final,
        host_saw_client: host_result.peer_final,
        client_saw_host: client_result.peer_final,
        msgs_host: host_result.msgs_received,
        msgs_client: client_result.msgs_received,
        host_saw_client_spectral: host_result.peer_spectral,
        client_saw_host_spectral: client_result.peer_spectral,
    })
}

struct SideResult {
    own_final: [f32; 3],
    peer_final: [f32; 3],
    peer_spectral: [u16; 16],
    msgs_received: u32,
}

/// Drive one player for `ticks` ticks. Each tick: compute own position via
/// `position_at(tick)`, send it, then receive exactly one packet from the peer.
/// Because each side sends-then-receives in lockstep over its own bidirectional
/// streams, the loop stays balanced and never deadlocks.
async fn run_side<F>(
    conn: crate::QuicConnection,
    entity_id: u32,
    position_at: F,
    ticks: u32,
) -> Result<SideResult, WalkDemoError>
where
    F: Fn(u32) -> [f32; 3],
{
    let mut own_final = [0.0f32; 3];
    let mut peer_final = [0.0f32; 3];
    let mut peer_spectral = [0u16; 16];
    let mut msgs_received = 0u32;

    for tick in 1..=ticks {
        let pos = position_at(tick);
        own_final = pos;
        let spectral = spectral_for(entity_id, tick);
        let packet = PlayerStatePacket::new(entity_id, pos, spectral);

        // Send our state and receive the peer's concurrently for this tick.
        let send_fut = conn.send_player_state(&packet);
        let recv_fut = conn.recv_player_state();
        let (send_res, recv_res) = tokio::join!(send_fut, recv_fut);
        send_res?;
        let received = recv_res?;

        peer_final = received.position;
        peer_spectral = received.spectral;
        msgs_received += 1;
    }

    Ok(SideResult { own_final, peer_final, peer_spectral, msgs_received })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_finals_match_movement_math() {
        let cfg = WalkDemoConfig::default();
        // 2.0 m/s * 0.016 s * 60 ticks = 1.92 m along X.
        assert!((cfg.expected_host_final()[0] - 1.92).abs() < 1e-4);
        assert_eq!(cfg.expected_host_final()[2], 0.0);
        // 3.0 m/s * 0.016 s * 60 ticks = 2.88 m along Z.
        assert!((cfg.expected_client_final()[2] - 2.88).abs() < 1e-4);
        assert_eq!(cfg.expected_client_final()[0], 0.0);
    }

    #[test]
    fn spectral_is_deterministic_and_distinguishes_entities() {
        assert_eq!(spectral_for(1, 5), spectral_for(1, 5), "same inputs -> same signature");
        assert_ne!(
            spectral_for(1, 5),
            spectral_for(2, 5),
            "different entities must produce different signatures"
        );
        // Band 0 of host entity at tick 5: 1*1000 + 5 + 0 = 1005.
        assert_eq!(spectral_for(1, 5)[0], 1005);
        // Band 3 of client entity at tick 7: 2*1000 + 7 + 3 = 2010.
        assert_eq!(spectral_for(2, 7)[3], 2010);
    }

    /// Full real-QUIC-loopback run: both sides replicate to within one tick and
    /// exchange more than 30 packets each.
    #[test]
    fn loopback_walk_replicates_peer_position_within_one_tick() {
        let cfg = WalkDemoConfig::default();
        let report = run_loopback_walk_demo(cfg).expect("demo run failed");

        // Each side ran every tick and received one packet per tick.
        assert_eq!(report.msgs_host, cfg.ticks, "host should receive one packet per tick");
        assert_eq!(report.msgs_client, cfg.ticks, "client should receive one packet per tick");
        assert!(report.msgs_host > 30, "host msgs must exceed 30");
        assert!(report.msgs_client > 30, "client msgs must exceed 30");

        // Own positions match the deterministic math exactly.
        let host_exp = cfg.expected_host_final();
        let client_exp = cfg.expected_client_final();
        assert!((report.host_final[0] - host_exp[0]).abs() < 1e-4, "host x = 1.92");
        assert!((report.client_final[2] - client_exp[2]).abs() < 1e-4, "client z = 2.88");

        // Each side tracked the OTHER's final position to within one tick of movement.
        let host_tol = cfg.host_tick_step();
        let client_tol = cfg.client_tick_step();
        // Client's view of host along X.
        assert!(
            (report.client_saw_host[0] - host_exp[0]).abs() <= host_tol + 1e-4,
            "client_saw_host.x {} should track host.x {} within one tick {}",
            report.client_saw_host[0], host_exp[0], host_tol
        );
        // Host's view of client along Z.
        assert!(
            (report.host_saw_client[2] - client_exp[2]).abs() <= client_tol + 1e-4,
            "host_saw_client.z {} should track client.z {} within one tick {}",
            report.host_saw_client[2], client_exp[2], client_tol
        );

        // The replicated spectral signatures are the peer's, not our own.
        assert_eq!(
            report.host_saw_client_spectral[0],
            (CLIENT_ENTITY_ID as u16) * 1000 + (cfg.ticks as u16),
            "host must see client's band-0 signature at the final tick"
        );
        assert_eq!(
            report.client_saw_host_spectral[0],
            (HOST_ENTITY_ID as u16) * 1000 + (cfg.ticks as u16),
            "client must see host's band-0 signature at the final tick"
        );
    }
}
