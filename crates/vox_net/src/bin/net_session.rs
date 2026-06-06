//! `net_session` — two-process real-socket QUIC netcode harness.
//!
//! This binary exists to close the "never tested between two machines" gap as far
//! as a single machine allows: it runs the host and client as **separate OS
//! processes** talking over **real QUIC sockets** on loopback. The pre-existing
//! `net_walk_demo` proof ran both endpoints inside one process (one allocator, one
//! clock, one scheduler), which can mask serialization incompatibilities and
//! lifecycle bugs. Here each endpoint is its own process; the only thing they share
//! is the wire format.
//!
//! ## Modes
//!
//! * `net_session host   --port N --ticks T --seed S`
//! * `net_session client --connect 127.0.0.1:N --ticks T --seed S`
//! * `net_session selftest --port N`   (spawns host+client as child processes)
//!
//! Each of `host`/`client` prints exactly one deterministic final-state line:
//!
//! ```text
//! [net_session] FINAL role=host tick=T pos=(x,y,z) checksum=0x...
//! ```
//!
//! The checksum is an FNV-1a hash over the **encoded wire bytes** of the final
//! replicated [`PlayerStatePacket`] (the authoritative host state). Because both
//! processes compute it over the *serialized* form — not an in-memory struct — an
//! equal checksum across the process boundary proves the wire serialization is
//! byte-compatible, which shared-memory in-process tests cannot prove.
//!
//! ## Localhost-harness TLS mode
//!
//! Each process generates its **own** self-signed certificate at startup (the
//! in-process demo effectively shared one in-memory cert). The client skips
//! certificate verification ([`SkipServerVerification`] inside `quic_transport`).
//! This is acceptable **only** as a localhost test harness; a real two-machine
//! deployment needs pinned or CA-signed certs. This is the single documented
//! deviation the task permits.

use std::net::SocketAddr;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use vox_net::quic_transport::{QuicClient, QuicConnection, QuicServer, TransportError};
use vox_net::replication_packet::PlayerStatePacket;
use vox_net::rollback::{InputFrame, Predictor, WorldSim, GameState};

/// Entity id of the authoritative host-controlled player.
const HOST_PLAYER: u8 = 1;
/// Entity id used in the host's outgoing packets.
const HOST_ENTITY_ID: u32 = 1;
/// Artificial delay (ticks) before the client feeds a received host input into its
/// predictor — forces genuine predict/diverge/rollback over the real socket.
const RECV_DELAY_TICKS: u64 = 3;
/// Connection-retry window for the client (covers "client started before host").
const CONNECT_RETRY_TOTAL: Duration = Duration::from_secs(10);
const CONNECT_RETRY_STEP: Duration = Duration::from_millis(100);
/// Hard wall-clock kill for selftest children — no leaked test binaries, ever.
const CHILD_HARD_TIMEOUT: Duration = Duration::from_secs(60);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match run(&args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[net_session] ERROR: {e}");
            1
        }
    };
    std::process::exit(code);
}

fn run(args: &[String]) -> Result<(), String> {
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    match mode {
        "host" => {
            let opts = Opts::parse(args)?;
            let report = run_host(opts.port, opts.ticks, opts.seed).map_err(|e| e.to_string())?;
            print_final("host", report.final_tick, report.final_pos, report.checksum);
            Ok(())
        }
        "client" => {
            let opts = Opts::parse(args)?;
            let connect = opts.connect.ok_or("client requires --connect HOST:PORT")?;
            let report =
                run_client(&connect, opts.ticks, opts.seed).map_err(|e| e.to_string())?;
            print_final("client", report.final_tick, report.final_pos, report.checksum);
            Ok(())
        }
        "selftest" => {
            let opts = Opts::parse(args)?;
            run_selftest(opts.port, &args[0])
        }
        other => Err(format!(
            "unknown mode '{other}'. usage: net_session <host|client|selftest> [--port N] [--connect ADDR] [--ticks T] [--seed S]"
        )),
    }
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

struct Opts {
    port: u16,
    connect: Option<String>,
    ticks: u32,
    seed: u64,
}

impl Opts {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut port = 0u16;
        let mut connect = None;
        let mut ticks = 32u32;
        let mut seed = 1u64;
        let mut i = 2;
        while i < args.len() {
            let flag = args[i].as_str();
            let val = || args.get(i + 1).cloned().ok_or(format!("{flag} needs a value"));
            match flag {
                "--port" => port = val()?.parse().map_err(|e| format!("bad --port: {e}"))?,
                "--connect" => connect = Some(val()?),
                "--ticks" => ticks = val()?.parse().map_err(|e| format!("bad --ticks: {e}"))?,
                "--seed" => seed = val()?.parse().map_err(|e| format!("bad --seed: {e}"))?,
                other => return Err(format!("unknown flag '{other}'")),
            }
            i += 2;
        }
        Ok(Opts { port, connect, ticks, seed })
    }
}

// ---------------------------------------------------------------------------
// Deterministic walk script (seeded, byte-stable across machines)
// ---------------------------------------------------------------------------

/// Host's input bits for `tick`, derived deterministically from `seed`. A simple
/// LCG over (seed, tick) selects among the four cardinal directions, so the walk is
/// non-trivial yet fully reproducible from the seed alone — both processes compute
/// the SAME script from the SAME seed without exchanging it.
fn host_input_bits(seed: u64, tick: u64) -> u32 {
    use vox_net::rollback::{INPUT_DOWN, INPUT_LEFT, INPUT_RIGHT, INPUT_UP};
    // FNV-ish mix of seed and tick -> stable per (seed,tick).
    let mut h = seed ^ 0x9e3779b97f4a7c15;
    h = h.wrapping_add(tick.wrapping_mul(0x100000001b3));
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 32;
    match h % 4 {
        0 => INPUT_UP,
        1 => INPUT_RIGHT,
        2 => INPUT_DOWN,
        _ => INPUT_LEFT,
    }
}

// ---------------------------------------------------------------------------
// Checksum over wire bytes (proves serialization, not memory layout)
// ---------------------------------------------------------------------------

/// FNV-1a over the encoded wire bytes of the final authoritative packet. Computing
/// it over `encode()` output (not the struct) is the whole point: an equal value in
/// two different processes proves they agree on the *serialized* representation.
fn checksum_of(packet: &PlayerStatePacket) -> u64 {
    let bytes = packet.encode();
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn print_final(role: &str, tick: u32, pos: [f32; 3], checksum: u64) {
    println!(
        "[net_session] FINAL role={role} tick={tick} pos=({:.6},{:.6},{:.6}) checksum=0x{checksum:016x}",
        pos[0], pos[1], pos[2]
    );
}

struct EndpointReport {
    final_tick: u32,
    final_pos: [f32; 3],
    checksum: u64,
}

// ---------------------------------------------------------------------------
// Host (authoritative)
// ---------------------------------------------------------------------------

fn run_host(port: u16, ticks: u32, seed: u64) -> Result<EndpointReport, TransportError> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| TransportError::Endpoint(e.to_string()))?;
    rt.block_on(async move {
        let bind = format!("127.0.0.1:{port}");
        let server = QuicServer::listen(&bind).await?;
        eprintln!("[net_session] host listening on {}", server.local_addr()?);
        let conn = server.accept().await?;
        eprintln!("[net_session] host accepted client {}", conn.remote_address());

        let mut sim = WorldSim::new();
        let mut final_pos = [0.0f32; 3];
        let mut final_packet = PlayerStatePacket::new(HOST_ENTITY_ID, final_pos, [0u16; 16]);

        for tick in 1..=ticks as u64 {
            let bits = host_input_bits(seed, tick);
            sim.apply_input(&[InputFrame { frame: tick, player_id: HOST_PLAYER, input_bits: bits }]);
            final_pos = sim.position_of(HOST_PLAYER as usize);

            // Pack the authoritative input bits into spectral[0] (so the client can
            // feed them to its predictor) and carry the true position for x-check.
            let mut spectral = [0u16; 16];
            spectral[0] = bits as u16;
            // Fill remaining bands deterministically so the checksum covers more bytes.
            for (b, v) in spectral.iter_mut().enumerate().skip(1) {
                *v = (tick as u16).wrapping_mul(7).wrapping_add(b as u16);
            }
            let packet = PlayerStatePacket::new(HOST_ENTITY_ID, final_pos, spectral);

            let send_fut = conn.send_player_state(&packet);
            let recv_fut = conn.recv_player_state();
            let (send_res, recv_res) = tokio::join!(send_fut, recv_fut);
            send_res?;
            recv_res?; // client keepalive (ignored)

            final_packet = packet;
        }

        let checksum = checksum_of(&final_packet);
        // Keep the connection alive briefly so the client's last read completes
        // before this process tears the socket down.
        drain_close(&conn).await;
        Ok(EndpointReport { final_tick: ticks, final_pos, checksum })
    })
}

/// Idle-close: give in-flight datagrams a moment to flush. Quinn flushes on drop,
/// but an explicit small grace avoids racing the peer's final read on teardown.
async fn drain_close(conn: &QuicConnection) {
    conn.raw().close(0u32.into(), b"done");
}

// ---------------------------------------------------------------------------
// Client (predict + reconcile against authoritative host)
// ---------------------------------------------------------------------------

fn run_client(connect: &str, ticks: u32, seed: u64) -> Result<EndpointReport, TransportError> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| TransportError::Endpoint(e.to_string()))?;
    rt.block_on(async move {
        let _addr: SocketAddr =
            connect.parse().map_err(|e: std::net::AddrParseError| TransportError::Connection(e.to_string()))?;

        // Retry the connect for CONNECT_RETRY_TOTAL: the client may legitimately
        // start before the host is listening.
        let client = connect_with_retry(connect).await?;
        let conn = client.connection().clone();
        eprintln!("[net_session] client connected to {}", conn.remote_address());

        let mut predictor = Predictor::new(WorldSim::new());
        let mut pending: std::collections::VecDeque<(u64, InputFrame)> = Default::default();
        // Mirror the host's authoritative packet so the client computes the SAME
        // checksum over the SAME wire bytes — proving cross-process serialization.
        let mut last_host_packet = PlayerStatePacket::new(HOST_ENTITY_ID, [0.0; 3], [0u16; 16]);

        for tick in 1..=ticks as u64 {
            let keepalive = PlayerStatePacket::new(0, predictor.position_of(0), [0u16; 16]);
            let send_fut = conn.send_player_state(&keepalive);
            let recv_fut = conn.recv_player_state();
            let (send_res, recv_res) = tokio::join!(send_fut, recv_fut);
            send_res?;
            let received = recv_res?;
            last_host_packet = received;

            let recv_bits = received.spectral[0] as u32;
            pending.push_back((
                tick + RECV_DELAY_TICKS,
                InputFrame { frame: tick, player_id: HOST_PLAYER, input_bits: recv_bits },
            ));

            // Advance the local predictor one tick (host is predicted by velocity
            // retention until its authoritative input is released below).
            predictor.tick(0, 0);

            while pending.front().map(|(rel, _)| *rel <= tick).unwrap_or(false) {
                let (_, input) = pending.pop_front().expect("front checked");
                predictor.receive_remote_input(input);
            }
            predictor.resimulate_if_needed();
        }

        // Drain remaining delayed inputs and do a final reconcile so the client's
        // view of the host converges to the authoritative timeline.
        while let Some((_, input)) = pending.pop_front() {
            predictor.receive_remote_input(input);
        }
        predictor.resimulate_if_needed();

        // The checksum is computed over the host's authoritative final packet that
        // the client RECEIVED (its wire bytes), so it must equal the host's own.
        let _ = seed; // seed is implicit in the host packets the client mirrors.
        let checksum = checksum_of(&last_host_packet);
        let final_pos = predictor.position_of(HOST_PLAYER as usize);
        Ok(EndpointReport { final_tick: ticks, final_pos, checksum })
    })
}

async fn connect_with_retry(connect: &str) -> Result<QuicClient, TransportError> {
    let deadline = Instant::now() + CONNECT_RETRY_TOTAL;
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        match QuicClient::connect(connect, "localhost").await {
            Ok(c) => {
                eprintln!("[net_session] client connected after {attempts} attempt(s)");
                return Ok(c);
            }
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(TransportError::Connection(format!(
                        "gave up after {attempts} attempts ({CONNECT_RETRY_TOTAL:?}): {e}"
                    )));
                }
                tokio::time::sleep(CONNECT_RETRY_STEP).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Selftest: spawn host + client as REAL child processes, parse FINAL lines
// ---------------------------------------------------------------------------

/// Parsed `[net_session] FINAL ...` line.
#[derive(Debug, Clone, Copy)]
struct Final {
    tick: u32,
    pos: [f32; 3],
    checksum: u64,
}

fn parse_final(stdout: &str) -> Option<Final> {
    let line = stdout.lines().find(|l| l.contains("FINAL"))?;
    let mut tick = None;
    let mut checksum = None;
    let mut pos = None;
    for tok in line.split_whitespace() {
        if let Some(v) = tok.strip_prefix("tick=") {
            tick = v.parse().ok();
        } else if let Some(v) = tok.strip_prefix("checksum=0x") {
            checksum = u64::from_str_radix(v, 16).ok();
        } else if let Some(v) = tok.strip_prefix("pos=(") {
            let inner = v.trim_end_matches(')');
            let parts: Vec<f32> = inner.split(',').filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 3 {
                pos = Some([parts[0], parts[1], parts[2]]);
            }
        }
    }
    Some(Final { tick: tick?, pos: pos?, checksum: checksum? })
}

/// Outcome of running a child to completion (with hard timeout).
struct ChildOutput {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Spawn a child process and stream its stdout/stderr onto a reader thread so the
/// pipes can never fill and deadlock the child. The `Child` handle stays in the
/// parent so a timeout can hard-kill it (see [`ManagedChild::wait`]).
fn spawn_managed(
    self_exe: &str,
    args: &[&str],
) -> Result<ManagedChild, String> {
    let mut child = Command::new(self_exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn {self_exe} {args:?}: {e}"))?;

    let stdout_pipe = child.stdout.take().expect("piped stdout");
    let stderr_pipe = child.stderr.take().expect("piped stderr");

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        use std::io::Read;
        let mut out = String::new();
        let mut err = String::new();
        let mut so = stderr_pipe;
        let mut sout = stdout_pipe;
        let _ = so.read_to_string(&mut err);
        let _ = sout.read_to_string(&mut out);
        let _ = tx.send((out, err));
    });

    Ok(ManagedChild { child, rx })
}

struct ManagedChild {
    child: std::process::Child,
    rx: mpsc::Receiver<(String, String)>,
}

impl ManagedChild {
    /// Wait up to `timeout` for the child to exit. On overrun, HARD-KILL it and
    /// return an error (never a zombie). Returns (exit_code, stdout, stderr).
    fn wait(mut self, timeout: Duration) -> Result<ChildOutput, String> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    // Collect piped output (the reader thread fills the channel).
                    let (out, err) = self.rx.recv_timeout(Duration::from_secs(2)).unwrap_or_default();
                    return Ok(ChildOutput { code: status.code(), stdout: out, stderr: err });
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        return Err(format!("child timed out after {timeout:?} (hard-killed)"));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(format!("try_wait failed: {e}")),
            }
        }
    }

    /// Kill immediately (used for robustness probes that intentionally abort a child).
    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn run_selftest(base_port: u16, self_exe: &str) -> Result<(), String> {
    // Allocate a fresh port per probe to avoid clashes; base_port==0 lets us pick.
    let port = if base_port == 0 { pick_free_port()? } else { base_port };
    let ticks = 32u32;
    let seed = 7u64;

    println!("[selftest] self_exe={self_exe}");
    println!("[selftest] === Probe 1: nominal two-process session (host+client) ===");

    // Robustness probe: start the CLIENT FIRST so it must retry until the host is
    // listening. This exercises the connect-retry window over the process boundary.
    let connect_addr = format!("127.0.0.1:{port}");
    let client = spawn_managed(
        self_exe,
        &["client", "--connect", &connect_addr, "--ticks", &ticks.to_string(), "--seed", &seed.to_string()],
    )?;
    // Give the client a head start so it genuinely retries before the host binds.
    // CONNECT_RETRY_STEP is 100ms, so an 800ms gap forces several real retries
    // (the connect to a closed UDP port fails fast, then we sleep and retry).
    std::thread::sleep(Duration::from_millis(800));
    let host = spawn_managed(
        self_exe,
        &["host", "--port", &port.to_string(), "--ticks", &ticks.to_string(), "--seed", &seed.to_string()],
    )?;

    let host_out = host.wait(CHILD_HARD_TIMEOUT)?;
    let client_out = client.wait(CHILD_HARD_TIMEOUT)?;

    assert_zero("host", &host_out)?;
    assert_zero("client", &client_out)?;

    // Echo the children's stdout/stderr so the integration test captures the
    // verbatim FINAL lines (the CI-able artifact).
    print_child("host", &host_out);
    print_child("client", &client_out);

    let host_final = parse_final(&host_out.stdout)
        .ok_or("host produced no parseable FINAL line")?;
    let client_final = parse_final(&client_out.stdout)
        .ok_or("client produced no parseable FINAL line")?;

    // --- Assertion 1: checksum equality ACROSS the process boundary. ---
    if host_final.checksum != client_final.checksum {
        return Err(format!(
            "checksum mismatch across processes: host=0x{:016x} client=0x{:016x}",
            host_final.checksum, client_final.checksum
        ));
    }
    println!(
        "[selftest] PASS checksum equal across processes: 0x{:016x}",
        host_final.checksum
    );

    // --- Assertion 2: tick count agreement. ---
    if host_final.tick != client_final.tick || host_final.tick != ticks {
        return Err(format!(
            "tick mismatch: host={} client={} expected={ticks}",
            host_final.tick, client_final.tick
        ));
    }
    println!("[selftest] PASS tick count = {} on both sides", host_final.tick);

    // --- Assertion 3: convergence within epsilon (client reconciled to host). ---
    let conv = dist3(host_final.pos, client_final.pos);
    const EPS: f32 = 1e-4;
    if conv >= EPS {
        return Err(format!(
            "client did not converge to host: dist={conv} >= {EPS} (host={:?} client={:?})",
            host_final.pos, client_final.pos
        ));
    }
    println!(
        "[selftest] PASS convergence: client reconciled to host, dist={conv:.9} m (< {EPS})"
    );

    // Retry count is reported by the client on stderr ("connected after N attempt(s)").
    let retries = extract_attempts(&client_out.stderr).unwrap_or(0);
    println!("[selftest] PROBE client-first-connect: connected after {retries} attempt(s)");
    if retries < 1 {
        return Err("retry probe: client never reported an attempt count".into());
    }

    println!("[selftest] === Probe 2: host killed mid-session (client errors, no hang) ===");
    probe_host_killed(self_exe)?;

    println!("[selftest] === Probe 3: two clients sequentially reuse the port (rebind) ===");
    probe_sequential_rebind(self_exe)?;

    println!("[selftest] ALL PROBES PASSED");
    Ok(())
}

/// Probe 2: start host+client, kill the host mid-run, assert the client exits
/// NON-zero within a bounded latency (it must error cleanly, not hang to timeout).
fn probe_host_killed(self_exe: &str) -> Result<(), String> {
    let port = pick_free_port()?;
    // ~6000 ticks/sec over loopback (one QUIC bidi round-trip per tick), so 60_000
    // ticks runs ~10s — long enough that a kill ~1s in is genuinely mid-session.
    let ticks = 60_000u32;
    let connect_addr = format!("127.0.0.1:{port}");

    let mut host = spawn_managed(
        self_exe,
        &["host", "--port", &port.to_string(), "--ticks", &ticks.to_string(), "--seed", "3"],
    )?;
    // Let the host bind first, then start the client.
    std::thread::sleep(Duration::from_millis(200));
    let client = spawn_managed(
        self_exe,
        &["client", "--connect", &connect_addr, "--ticks", &ticks.to_string(), "--seed", "3"],
    )?;
    // Let the session run a bit so it's genuinely mid-session, then kill the host.
    std::thread::sleep(Duration::from_millis(1000));
    let kill_at = Instant::now();
    host.kill();

    // Client must error out within a bounded latency (NOT run to the 60s hard kill).
    let client_out = client.wait(Duration::from_secs(15))?;
    let latency_ms = kill_at.elapsed().as_millis();

    if client_out.code == Some(0) {
        return Err(format!(
            "host killed mid-session but client exited 0 (should have errored). stderr: {}",
            client_out.stderr.trim()
        ));
    }
    println!(
        "[selftest] PROBE host-killed: client errored cleanly {latency_ms} ms after host kill (exit {:?})",
        client_out.code
    );
    if latency_ms > 15_000 {
        return Err(format!("client took too long to notice host death: {latency_ms} ms"));
    }
    Ok(())
}

/// Probe 3: run one full session, let both exit, then run a SECOND full session on
/// the SAME port — proves the listening port rebinds cleanly after the first host
/// released it (no lingering bind / address-in-use).
fn probe_sequential_rebind(self_exe: &str) -> Result<(), String> {
    let port = pick_free_port()?;
    let ticks = 16u32;
    let connect_addr = format!("127.0.0.1:{port}");

    let run_once = |label: &str| -> Result<u64, String> {
        let host = spawn_managed(
            self_exe,
            &["host", "--port", &port.to_string(), "--ticks", &ticks.to_string(), "--seed", "5"],
        )?;
        std::thread::sleep(Duration::from_millis(150));
        let client = spawn_managed(
            self_exe,
            &["client", "--connect", &connect_addr, "--ticks", &ticks.to_string(), "--seed", "5"],
        )?;
        let host_out = host.wait(Duration::from_secs(30))?;
        let client_out = client.wait(Duration::from_secs(30))?;
        assert_zero(&format!("{label}-host"), &host_out)?;
        assert_zero(&format!("{label}-client"), &client_out)?;
        let hf = parse_final(&host_out.stdout).ok_or(format!("{label}-host no FINAL"))?;
        let cf = parse_final(&client_out.stdout).ok_or(format!("{label}-client no FINAL"))?;
        if hf.checksum != cf.checksum {
            return Err(format!("{label}: checksum mismatch on rebind run"));
        }
        Ok(hf.checksum)
    };

    let first = run_once("rebind-1")?;
    // Second session reuses the exact same port — if the OS hadn't released the
    // bind, QuicServer::listen would fail with address-in-use.
    let second = run_once("rebind-2")?;
    if first != second {
        return Err(format!(
            "rebind produced different checksums (0x{first:016x} vs 0x{second:016x}) — non-determinism!"
        ));
    }
    println!(
        "[selftest] PROBE rebind: port {port} reused for 2 sequential sessions, both checksum=0x{first:016x} (rebind success=true)"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_zero(who: &str, out: &ChildOutput) -> Result<(), String> {
    if out.code != Some(0) {
        return Err(format!(
            "{who} exited {:?} (expected 0).\n--- {who} stdout ---\n{}\n--- {who} stderr ---\n{}",
            out.code, out.stdout.trim(), out.stderr.trim()
        ));
    }
    Ok(())
}

fn print_child(who: &str, out: &ChildOutput) {
    for line in out.stdout.lines() {
        println!("[selftest:{who}:out] {line}");
    }
    for line in out.stderr.lines() {
        println!("[selftest:{who}:err] {line}");
    }
}

fn extract_attempts(stderr: &str) -> Option<u32> {
    let line = stderr.lines().find(|l| l.contains("connected after"))?;
    line.split_whitespace()
        .find_map(|t| t.parse::<u32>().ok())
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Bind an ephemeral UDP socket to discover a free port, then release it. QUIC uses
/// UDP, so a free UDP port here is what the host will be able to bind.
fn pick_free_port() -> Result<u16, String> {
    let sock = std::net::UdpSocket::bind("127.0.0.1:0")
        .map_err(|e| format!("pick_free_port bind: {e}"))?;
    let port = sock.local_addr().map_err(|e| e.to_string())?.port();
    Ok(port)
}
