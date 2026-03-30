# Domain 7 — Networking & Multiplayer

**Status:** Draft — 2026-03-29
**Scope:** Transport layer, splat state replication, client-side prediction, rollback netcode, server architecture, spectral networking
**Engine:** Ochroma spectral Gaussian Splatting — Rust workspace, wgpu 24, WGSL shaders, tokio async runtime, quinn QUIC

---

## Goals

Ochroma's multiplayer layer must support 2–32 concurrent players in a shared spectral Gaussian Splatting world with authoritative server physics, client-side prediction and reconciliation, and a spectral-aware replication strategy that minimises bandwidth without sacrificing the visual fidelity of animated splat fields. The networking stack must be fully engine-agnostic — it lives in the `vox_net` crate, knows nothing about city-building or any game-layer concept, and exports only types that operate on `GaussianSplat`, `NetEntityId`, and abstract `InputState`. The transport target is QUIC via the `quinn` crate. Latency target for prediction-reconciliation round-trip is invisible on connections up to 150 ms RTT; the rollback window covers up to 166 ms of mis-prediction.

---

## 7.1 Network Architecture & Transport

### Authoritative Server Model

The server owns all physics simulation (Rapier 3D) and canonical game state. Clients receive snapshots and events; they run local prediction to hide latency but never push authoritative state. The server validates every client input and may reject or correct it. This model prevents the most common cheat vectors (position teleportation, impossible force application) and simplifies state reconciliation because there is exactly one source of truth.

### QUIC via quinn

All transport uses QUIC (RFC 9000) through the `quinn` crate. QUIC provides:
- Multiplexed streams with independent head-of-line blocking removal per stream.
- Unreliable datagrams (RFC 9221) for per-frame state that is rendered obsolete by the next frame anyway.
- Built-in TLS 1.3 for encryption without a separate handshake round-trip.
- Connection migration, useful for mobile clients switching networks mid-session.

Stream layout:

| Channel | Type | Content |
|---|---|---|
| Stream 0 | Reliable ordered (QUIC stream) | Connection setup, config exchange, session metadata |
| Per-entity stream | Reliable ordered (QUIC stream) | High-priority events: damage, death, door state, spectral threshold events |
| Per-frame datagrams | Unreliable QUIC datagram | SplatDelta bundles, InputState, interpolation hints |

### Packet Structure

```rust
// crates/vox_net/src/packet.rs

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetPacket {
    /// Monotonically increasing sequence number for this sender.
    pub sequence: u32,
    /// Highest sequence number this sender has received from the remote.
    pub ack: u32,
    /// Bitmask: bit N set means sequence (ack - N - 1) was received.
    pub ack_bits: u32,
    pub payload: NetPayload,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum NetPayload {
    SplatDelta(SplatDeltaPacket),
    InputState(InputStatePacket),
    EntityEvent(EntityEventPacket),
    WorldEvent(WorldEventPacket),
    ChatMessage(ChatMessagePacket),
}
```

Serialisation uses `bincode` (fixed-size little-endian) rather than JSON or `postcard` — the binary format is compact and zero-copy friendly with `bytemuck`. Maximum unreliable datagram payload is 1200 bytes, staying below typical PMTU to avoid IP fragmentation.

### NetworkManager

`NetworkManager` is the top-level runtime type, created once per process (server or client). It owns a `quinn::Endpoint`, drives send/recv within the engine tick loop via `tokio::runtime::Runtime::block_on` (the engine runs tokio in a dedicated thread so as not to block the render thread), and exposes a pair of `flume` channels (`net_tx`, `net_rx`) for the game layer to push outgoing packets and drain incoming ones.

```rust
// crates/vox_net/src/manager.rs

pub struct NetworkManager {
    endpoint: quinn::Endpoint,
    runtime:  tokio::runtime::Handle,
    net_tx:   flume::Sender<NetPacket>,
    net_rx:   flume::Receiver<NetPacket>,
}

impl NetworkManager {
    pub fn tick(&mut self) { /* drain recv, dispatch callbacks, flush send queue */ }
}
```

### Server-Side Types

```rust
pub struct ServerNetworkManager {
    endpoint:     quinn::Endpoint,
    clients:      HashMap<ClientId, ClientConnection>,
    world_state:  ServerWorldState,
}

pub struct ClientConnection {
    connection:      quinn::Connection,
    last_acked_seq:  u32,
    input_buffer:    VecDeque<InputStatePacket>,  // last 128 frames
    interest_zone:   ReplicationInterestZone,
}

pub struct ServerWorldState {
    tick:            u64,
    splat_sets:      HashMap<SplatSetId, ReplicatedSplatSet>,
    entities:        HashMap<NetEntityId, NetworkEntity>,
}
```

### Client-Side Types

```rust
pub struct ClientNetworkManager {
    connection:       quinn::Connection,
    local_prediction: LocalPredictionState,
    server_ack_seq:   u32,
}
```

### Tick Rates

- Server simulation: 60 Hz. Physics, input application, and entity updates all run at 60 Hz.
- Server→client state broadcast: 20 Hz. A `SplatDeltaPacket` is assembled and sent as an unreliable datagram at 20 Hz; clients interpolate smoothly between the two most recently received snapshots.
- Client→server input: 60 Hz, unreliable datagrams. The server buffers and applies inputs in order; missing inputs are extrapolated for one frame and flagged.

---

## 7.2 Splat State Replication

### Replication Categories

GaussianSplat has 52 bytes of fields (position 12, scale 12, rotation 8, opacity 1, spectral 16, padding 3). Most fields are static for most splats in a scene. The replication strategy partitions splats into categories based on their expected change frequency:

| Category | Position | Rotation | Spectral | Strategy |
|---|---|---|---|---|
| Static (terrain, architecture) | Never | Never | Rarely | Send once on cell load; spectral-only delta on material event |
| Animated (characters, creatures) | 20 Hz | 20 Hz | 1 Hz | Full transform delta at 20 Hz |
| Dynamic (physics objects) | 20 Hz | 20 Hz | On event | Full transform delta at 20 Hz |
| Particles | Client-local | Client-local | Client-local | Server sends emitter spawn/despawn only |

This is encoded in `ReplicatedSplatSet`:

```rust
// crates/vox_net/src/replication.rs

pub enum SplatCategory {
    Static,
    Animated { owner_entity: NetEntityId },
    Dynamic  { physics_body: u64 },
    Particle { emitter:      NetEntityId },
}

pub struct ReplicatedSplatSet {
    pub set_id:     SplatSetId,
    pub category:   SplatCategory,
    pub base_positions: Vec<[f32; 3]>,  // world-space origin for delta encoding
    pub splats:     Vec<GaussianSplat>,
    pub last_sent:  HashMap<ClientId, u64>,  // tick when last snapshot was acked per client
}
```

### SplatDelta Wire Format

```rust
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SplatDeltaPacket {
    pub set_id:    u32,
    pub tick:      u64,
    /// Indices into the set's splat array. u16 caps sets at 65535 splats.
    pub indices:   Vec<u16>,
    /// Position deltas in i16 fixed-point: 1 unit = 1mm; range ±32.767m from base_positions[i].
    pub positions: Vec<[i16; 3]>,
    /// Rotation deltas: 8-bit delta quaternion components, scaled by 1/127.
    pub rotations: Vec<[i8; 4]>,
    /// Spectral deltas: 3-bit per band, Huffman-coded. Included only when spectral changed.
    pub spectral:  Option<SpectralDeltaBlock>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SpectralDeltaBlock {
    pub set_id:  u32,
    pub indices: Vec<u16>,
    /// Huffman-coded stream. Decoded to i8 per band per splat; add to last-acked spectral.
    pub data:    Vec<u8>,
}
```

Position encoding: the fixed-point scheme uses `((world_pos - base_pos) * 1000.0).round() as i16`, giving 1 mm precision over a ±32.767 m window around each splat's base position. If a splat moves further than that window (rare for non-teleporting objects), the server sends a base-position reset on the reliable stream and the client re-anchors.

Rotation delta: the server computes `delta_quat = server_quat * client_last_quat.conjugate()`, takes the `xyz` imaginary components (the `w` component is reconstructed as `sqrt(1 - x²-y²-z²)` on the client), scales each to `i8` range. For near-identity rotations (static splats within animated sets) the delta is zero and RLE-encodes away entirely.

### DeltaCompressor

```rust
pub struct DeltaCompressor {
    /// Per-client, per-set: the last state we know the client has acked.
    client_acked_state: HashMap<(ClientId, SplatSetId), Vec<GaussianSplat>>,
}

impl DeltaCompressor {
    /// Produce a SplatDeltaPacket containing only splats that changed since last ack.
    pub fn compress(&self, set: &ReplicatedSplatSet, client: ClientId) -> Option<SplatDeltaPacket>;
    /// Called when client acks a tick; update the acked state snapshot.
    pub fn record_ack(&mut self, client: ClientId, set_id: SplatSetId, tick: u64);
}
```

The compressor iterates the set's splats, compares each against the client's last-acked snapshot, and emits indices only for splats that differ by more than a configurable threshold (1 mm for position, 0.5° for rotation, 0.5% for spectral bands). Unchanged runs are skipped entirely. A final pass run-length encodes the index list (sorted ascending, so consecutive changed splats emit a range rather than individual indices).

### Bandwidth Budget

For a 10-player session with 10 animated characters × 500 splats each:
- Raw delta: 500 splats × 20 Hz × (6 bytes position + 4 bytes rotation + 2 bytes index) = 120 KB/s per client.
- After RLE compression on mostly-static frames (character standing still): ~10–30 KB/s.
- Peak during heavy animation (combat): ~100–300 KB/s per client.

Static terrain (tens of millions of splats) contributes zero ongoing bandwidth; it is transmitted once during cell load via reliable stream and then replicated only when a `SpectralEvent` fires (fire, damage, flood).

---

## 7.3 Input Replication & Client Prediction

### InputState

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InputStatePacket {
    pub frame:    u64,
    pub move_xy:  [f16_bits; 2],  // glam::Vec2 encoded as f16 bits, range -1..1
    pub look_xy:  [f16_bits; 2],
    pub buttons:  u8,             // bit 0 = jump, bit 1 = fire, bit 2 = interact, bits 3-7 reserved
}
```

The client sends an `InputStatePacket` every simulation frame (60 Hz) as an unreliable datagram. Redundant sending (re-transmitting the last 3 frames in each datagram) ensures that a single lost packet does not cause a full input gap.

### Server Input Buffer

```rust
pub struct ClientInputBuffer {
    ring:     [Option<InputStatePacket>; 128],
    head:     u64,
}

impl ClientInputBuffer {
    pub fn insert(&mut self, pkt: InputStatePacket) { /* discard duplicates, fill ring */ }
    pub fn get(&self, frame: u64) -> Option<&InputStatePacket>;
    pub fn extrapolate(&self, frame: u64) -> InputStatePacket { /* repeat last known */ }
}
```

The ring buffer holds 128 frames (~2.1 seconds at 60 Hz). If no input arrives for a frame (packet loss), the server calls `extrapolate()` and applies the previous frame's input, flagging the frame as extrapolated. Two consecutive extrapolated frames trigger a `ClientLagEvent` logged to the anomaly detector.

### Client-Side Prediction

```rust
pub struct ClientPredictor {
    pending_inputs:   VecDeque<InputStatePacket>,   // sent but not yet acked
    predicted_states: VecDeque<PredictedState>,     // one per pending input
}

pub struct PredictedState {
    pub frame:          u64,
    pub local_position: glam::Vec3,
    pub local_velocity: glam::Vec3,
    pub local_splats:   Vec<GaussianSplat>,  // local character splats only
}
```

Each frame the client:
1. Reads the current `InputState` from the `InputSystem`.
2. Applies it to the local predicted world (runs the same physics equations the server will run, minus Rapier — the client uses a simplified kinematic integrator for prediction to avoid duplicating full Rapier state).
3. Pushes `InputStatePacket` to `pending_inputs` and records `PredictedState`.
4. Sends the packet to the server.

When a server acknowledgment arrives for frame F:
1. Find the matching `PredictedState` in `pending_inputs` at frame F.
2. Compare server's authoritative position to `predicted_states[F].local_position`.
3. If the positional error exceeds `RECONCILE_THRESHOLD` (default 5 cm), trigger rollback: restore server state for frame F, then re-simulate frames F+1..current by replaying `pending_inputs`.
4. Discard all `pending_inputs` and `predicted_states` up to and including frame F.

Re-simulation cost scales with `N = current_frame - F` (typically < 10 on connections under 100 ms RTT). Each re-simulation step is a single kinematic integration step, not a full Rapier world tick.

### Spectral Prediction

The client speculatively applies spectral changes triggered by local events (player fires a weapon → emitter fires → nearby splats gain emission in band 7). These predicted spectral values are rendered immediately. On server ack, if the server's spectral values differ (e.g. a hit did not register), the client blends the server values in over 200 ms using `InertialBlender`, preventing visible flicker from sudden spectral corrections.

---

## 7.4 Entity Replication System

### Core Types

```rust
pub type NetEntityId = u64;   // globally unique, assigned by server
pub type ClientId    = u32;

pub struct NetworkEntity {
    pub net_id:               NetEntityId,
    pub owner:                ClientId,
    pub replicated_components: Vec<ReplicatedComponent>,
    pub replication_priority: f32,   // 0.0–1.0; updated per-client based on distance
}

pub enum ReplicatedComponent {
    Transform(TransformState),
    Health(HealthState),
    AnimationState(AnimState),
    PhysicsBody(PhysicsBodyState),
    SpectralProfile(SpectralProfileState),
}
```

`NetEntityId` is assigned by the server at entity spawn and is stable for the entity's lifetime. Clients map `NetEntityId` to their local ECS entity handle via a `HashMap<NetEntityId, LocalEntityHandle>` maintained by `EntityReplicationSystem`.

### Owner Authority

The owning client sends authoritative inputs for its entity; the server simulates those inputs and sends corrections. Other clients receive the server's reconciled state only — they never receive raw inputs from the owning client. This means non-owning clients always see the server's authoritative view, which lags by up to one network round-trip but is always correct.

### Interest Management

```rust
pub struct ReplicationInterestZone {
    pub center: glam::Vec3,
    pub radius: f32,          // spatial interest radius, typically 200m
    pub spectral_emitters: Vec<NetEntityId>,  // emissive entities visible despite distance
}
```

Entities whose world position falls outside a client's `ReplicationInterestZone` are not replicated to that client. The interest zone is updated each tick from the client's reported camera position. `spectral_emitters` extends the zone: any entity with spectral band 5–7 (NIR/UV-range emissive) magnitude > 0.5 is added to the extended set and replicated regardless of distance, because high-band emitters are visually salient at long range in the spectral renderer.

### Replication Priority

Each `NetworkEntity` has a per-client `replication_priority` float recomputed each broadcast tick:

```
priority = (1.0 / distance_to_client) * category_weight * relevance_bias
```

`category_weight`: animated characters = 2.0; dynamic physics = 1.5; static = 0.1. The replication system sorts entities by priority and, when bandwidth is constrained (datagram size limit), sends the highest-priority entities first in each packet.

---

## 7.5 Rollback Netcode

Rollback is an optional subsystem targeted at action/competitive game modes where 150 ms RTT latency would otherwise cause visible misprediction snaps. It stores full world snapshots and re-simulates forward from any prior frame.

### State Snapshot

```rust
pub struct RollbackState {
    pub frame:            u64,
    pub splat_snapshot:   Vec<GaussianSplat>,          // full copy of dynamic splat sets
    pub physics_snapshot: Vec<u8>,                      // bincode of Rapier world via rapier3d::prelude::serialize
    pub input_history:    HashMap<ClientId, InputStatePacket>,
}
```

`RollbackBuffer` is a fixed-size ring buffer of `ROLLBACK_DEPTH = 10` frames (166 ms at 60 Hz):

```rust
pub struct RollbackBuffer {
    states: [Option<RollbackState>; 10],
    head:   usize,
}

impl RollbackBuffer {
    pub fn push(&mut self, state: RollbackState);
    pub fn get(&self, frame: u64) -> Option<&RollbackState>;
}
```

### Rollback Procedure

When the server's authoritative state for frame F differs from the local prediction:

1. `rollback_buffer.get(F)` retrieves the snapshot for frame F.
2. Restore `rapier3d::prelude::deserialize(&physics_snapshot)` — Rapier's built-in serialization handles full world state including island solver state.
3. Restore `splat_snapshot` to the local dynamic splat sets.
4. Re-simulate frames F+1..current: for each frame, apply `input_history[frame]` (using client's own corrected inputs; other clients' inputs sourced from the rollback history), step the Rapier world with a 1/60s dt, update splat positions from physics body transforms.
5. `InertialBlender` smooths the final visual output: for each splat that moved more than 1 cm in the rollback correction, apply a 100 ms exponential blend toward the corrected position so the correction is imperceptible at normal animation speeds.

### Performance Target

Full rollback + re-simulation of 10 frames with 100 physics rigid bodies must complete in < 2 ms on a modern CPU (a single physics step at this scale takes ~0.1 ms in Rapier; 10 steps = ~1 ms, leaving headroom for splat position updates). Splat position update is O(N_dynamic_splats) per frame; for 5000 dynamic splats across 10 frames this is 50000 simple vector additions — well under 1 ms on a single core.

---

## 7.6 Server Architecture

### Dedicated Server

A dedicated server runs Ochroma headlessly: no window, no EWA renderer, no audio thread, no wgpu device. The binary is compiled with `--no-default-features` and feature flag `server-only`, which gates out all `vox_render` and `vox_audio` dependencies.

```
ochroma-server
  └─ ServerRuntime
       ├─ PhysicsThread   (Rapier 3D, 60 Hz)
       ├─ GameLogicThread (entity updates, AI, scripting)
       ├─ NetworkThread   (quinn endpoint, tokio runtime)
       └─ ReplicationThread (delta compression, broadcast at 20 Hz)
```

The `ServerRuntime` coordinates these via `flume` channels and a shared `Arc<Mutex<ServerWorldState>>`. Physics and game logic run synchronously on a single thread pair (physics writes state, game logic reads it) to avoid the complexity of parallel physics mutation. Network and replication are async on tokio.

### Listen Server

For hosted multiplayer (one player hosts, others connect), `EngineApp` spawns a `ServerRuntime` on a background thread and creates a `ClientNetworkManager` that connects to `127.0.0.1` on an ephemeral port. The local client gets zero-latency loopback — `quinn` supports local UDP loopback with ~0.1 ms RTT. The host client's inputs are inserted directly into `ClientInputBuffer` without going through the network stack, saving one serialise/deserialise cycle.

### Lobby System

```rust
pub struct LobbyManager {
    lobbies: HashMap<LobbyId, Lobby>,
    db:      rusqlite::Connection,  // SQLite for persistence
}

pub struct Lobby {
    pub id:          LobbyId,
    pub players:     Vec<ClientId>,
    pub state:       LobbyState,
    pub game_config: GameConfig,
    pub created_at:  chrono::DateTime<chrono::Utc>,
}

pub enum LobbyState {
    WaitingForPlayers { min: u8, max: u8 },
    Countdown { seconds_remaining: u8 },
    InGame { started_at: chrono::DateTime<chrono::Utc> },
    PostGame { results: GameResults },
}
```

Lobby metadata (player lists, config, session start/end times, results) is persisted to SQLite via `rusqlite`. The schema is simple: `lobbies`, `players`, `sessions`, `results`. This enables session replay for debugging and allows resuming disconnected sessions.

### Anti-Cheat

Every client input is validated server-side before application:

- Position delta sanity: `|new_pos - old_pos| / dt <= max_speed + epsilon`. Reject inputs exceeding 2× the character's maximum configured speed.
- Force application: validate that any physics impulse magnitude is within the character's ability bounds.
- Event rate limiting: no client may fire more than `max_fire_rate` `EntityEvent` packets per second.
- Anomaly logging: all rejected inputs are appended to an anomaly log (`anomaly.log`) with client ID, input details, and frame number. Repeated anomalies trigger a `ClientSuspendEvent` that the game layer can act on.

---

## 7.7 Spectral Networking Features

### SpectralEvent

```rust
pub enum SpectralEvent {
    BandThresholdCrossed { entity: NetEntityId, band: u8, value: f32, direction: ThresholdDir },
    MaterialChanged      { set_id: SplatSetId, spectral_profile: SpectralProfileId },
    EmitterFired         { emitter: NetEntityId, band: u8, intensity: f32 },
}
```

`SpectralEvent` is sent on an entity's per-entity reliable QUIC stream (so it is never lost). The server fires these when monitoring a `SpectralThresholdMonitor` component on an entity: each tick, the monitor samples the aggregate spectral value of the entity's splat set across each band and emits the event if a configured threshold is crossed.

### SpectralSync

For cooperative game modes requiring shared spectral field state (e.g. "this door opens when the cumulative band-3 emission in this 10m³ volume exceeds 5.0"):

```rust
pub struct SpectralSyncVolume {
    pub volume_id:   u32,
    pub aabb:        (glam::Vec3, glam::Vec3),
    pub band:        u8,
    pub threshold:   f32,
    pub current_sum: f32,   // updated each tick by SpectralSumPass compute shader
}
```

The server tracks `SpectralSyncVolume` entries. The `SpectralSumPass` (a 1D reduce compute shader operating on the GPU splat buffer) computes `current_sum` each tick for volumes with at least one connected client nearby. `current_sum` is broadcast to clients in the `WorldEvent` payload whenever it changes by more than 1%.

### Spectral Delta Compression

Spectral band values change slowly: a burning splat's band 6 (red-orange) increases by < 2% per frame. The compression pipeline exploits this:

1. Compute the difference between the server's current spectral value and the last-acked value per band per splat.
2. Quantise to 3-bit signed integers (range -4..+3 in units of 0.5% of full scale). If the delta is 0, omit the splat from the packet entirely.
3. Concatenate the 3-bit deltas into a bitstream (3 bits × 8 bands = 24 bits = 3 bytes per changed splat).
4. Apply Huffman coding seeded from a static probability table derived from common spectral profiles (smooth sky, emissive fire, metallic surface). The coder is in `vox_net/src/spectral_compress.rs` and produces a `Vec<u8>` plus a 16-byte prefix encoding the Huffman table override (if the current-frame distribution diverges significantly from the seeded table, the server sends the actual table in the prefix; otherwise the prefix is zeroed and the static table is used).

For a character with 500 splats, spectral data changes on ~10% of splats per frame at peak activity → 50 splats × 3 bytes = 150 bytes pre-Huffman → ~80 bytes post-Huffman. At 20 Hz broadcast = 1.6 KB/s for spectral alone. For static scene splats, spectral delta is 0 bytes/s between material events.

---

## File Map

```
crates/vox_net/
  Cargo.toml                          # quinn, serde, bincode, flume, tokio, rusqlite
  src/
    lib.rs                            # pub mod exports
    packet.rs                         # NetPacket, NetPayload, all packet structs
    manager.rs                        # NetworkManager, tick()
    server/
      manager.rs                      # ServerNetworkManager
      world_state.rs                  # ServerWorldState, ReplicatedSplatSet
      lobby.rs                        # LobbyManager, Lobby, LobbyState
      anti_cheat.rs                   # input validation, anomaly logging
    client/
      manager.rs                      # ClientNetworkManager
      predictor.rs                    # ClientPredictor, PredictedState
    replication/
      compressor.rs                   # DeltaCompressor
      interest.rs                     # ReplicationInterestZone
      entity.rs                       # NetworkEntity, ReplicatedComponent
    rollback/
      buffer.rs                       # RollbackBuffer, RollbackState
      inertial_blend.rs               # InertialBlender for visual smoothing
    spectral/
      event.rs                        # SpectralEvent, SpectralSyncVolume
      compress.rs                     # SpectralDeltaCompressor, Huffman coder
    transport/
      quic.rs                         # quinn endpoint init, stream management
      input_buffer.rs                 # ClientInputBuffer ring buffer

crates/vox_app/
  src/
    net_integration.rs                # wires NetworkManager into EngineApp tick

bin/
  ochroma-server/
    main.rs                           # headless ServerRuntime entry point
```

---

## Milestones

### M1 — Transport Foundation
Implement `vox_net` crate with `quinn` endpoint, `NetPacket` serialisation/deserialisation, and a loopback integration test (server echo, client send → server → client roundtrip verified). Unreliable datagrams and reliable streams both exercised. No game logic yet.

**Duration:** 3 days
**Done when:** `cargo test -p vox_net` passes; Wireshark confirms QUIC datagrams on loopback.

### M2 — Basic Splat Replication
`DeltaCompressor` implemented for `Animated` and `Dynamic` splat categories. `ServerNetworkManager` broadcasts `SplatDeltaPacket` at 20 Hz. Client receives and applies deltas to a `ReplicatedSplatSet`. Two-player scene with one animated character visible to both players without corruption.

**Duration:** 5 days
**Done when:** Two ochroma-server + two ochroma-client instances run concurrently; animated character position matches server state within 1 cm on both clients after 60 seconds.

### M3 — Input Prediction & Reconciliation
`ClientPredictor` and `ClientInputBuffer` implemented. Client applies inputs locally; server validates and sends authoritative corrections; client reconciles without visible snap on connections up to 100 ms RTT (simulated with `tc netem delay 50ms`).

**Duration:** 4 days
**Done when:** Integration test with 50 ms simulated RTT shows < 5 cm positional error at reconciliation; no visible pop in the EWA renderer output.

### M4 — Rollback Netcode
`RollbackBuffer` and full re-simulation implemented. Triggered by injected mis-prediction in integration tests. `InertialBlender` smooths corrections. Performance benchmark passes (10-frame rollback < 2 ms).

**Duration:** 4 days
**Done when:** `cargo bench -p vox_net rollback` reports < 2 ms; visual test shows no perceptible snap during a forced rollback event.

### M5 — Full Server Architecture
`ochroma-server` binary compiles without `vox_render` or `vox_audio`. `LobbyManager` with SQLite persistence. `SpectralEvent` and `SpectralSyncVolume` implemented and tested in a 4-player cooperative scenario. Anti-cheat validation rejects injected impossible-speed inputs. Listen server mode tested.

**Duration:** 5 days
**Done when:** 4-player session runs for 10 minutes without desync; anomaly log correctly captures injected cheat inputs; lobby state persists across server restart.

---

## Acceptance Criteria

1. A 10-player session runs at 20 Hz replication for 30 minutes without observable state divergence between server and any client (positions agree within 5 cm, spectral within 2%).
2. Client-side prediction hides all latency on connections up to 150 ms RTT — no visible movement stuttering in the EWA renderer.
3. A forced 10-frame rollback completes in < 2 ms (measured by `cargo bench`).
4. Bandwidth per client does not exceed 400 KB/s in a worst-case 10-animated-character scenario.
5. Static terrain splats consume zero ongoing replication bandwidth after initial cell load.
6. `SpectralEvent` is never lost (reliable stream delivery confirmed by integration test with 5% simulated packet loss).
7. Anti-cheat correctly rejects 100% of inputs that exceed 2× max speed in the anomaly-injection test.
8. `ochroma-server` binary has no link-time dependency on `vox_render` or `vox_audio` (verified by `cargo tree`).
9. The `vox_net` crate contains no game-layer concepts (buildings, zoning, traffic) — verified by `grep` in CI.

---

## Effort Estimate

| Component | Engineer-Days |
|---|---|
| Transport (quinn, NetPacket, streams) | 3 |
| Splat replication (DeltaCompressor, categories) | 5 |
| Input prediction & reconciliation | 4 |
| Rollback netcode + InertialBlender | 4 |
| Server architecture (lobby, SQLite, anti-cheat, headless binary) | 5 |
| Spectral networking (SpectralEvent, SpectralSync, Huffman compress) | 3 |
| Integration tests + CI harness | 3 |
| **Total** | **27** |
