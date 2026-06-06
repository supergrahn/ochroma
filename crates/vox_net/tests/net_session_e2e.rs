//! CI-able end-to-end test: runs the `net_session` binary in `selftest` mode, which
//! itself spawns the host and client as **separate OS processes** talking over real
//! QUIC sockets on loopback. This is the artifact that closes the "never tested
//! between two machines" gap as far as one machine permits: unlike the in-process
//! `net_walk_demo` proof (one allocator, one clock, one scheduler), every assertion
//! here is verified across a real process + socket boundary.
//!
//! `selftest` exits 0 only if ALL of the following hold, so this test asserts on a
//! real computed outcome (the child's exit code) AND echoes the verbatim FINAL lines
//! + checksums into the captured output:
//!   * the host and client compute an EQUAL checksum over the final replicated
//!     packet's WIRE BYTES (proves cross-process serialization, not shared memory),
//!   * both report the same tick count,
//!   * the client's reconciled view converges to the host within epsilon,
//!   * robustness probes pass: client-before-host retry, host-killed-mid-session
//!     clean error within bound, and sequential port rebind.

use std::process::Command;

#[test]
fn net_session_selftest_two_processes_over_real_quic() {
    let exe = env!("CARGO_BIN_EXE_net_session");

    // `selftest --port 0` lets the harness pick free ports per probe. The binary
    // itself hard-kills any child that overruns 60s, so this test can never leak a
    // process; we still cap the whole run defensively.
    let output = Command::new(exe)
        .args(["selftest", "--port", "0"])
        .output()
        .expect("failed to launch net_session selftest binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Echo everything so `cargo test -- --nocapture` shows the verbatim FINAL lines,
    // checksums, and probe values — the human-visible CI artifact.
    println!("===== net_session selftest stdout =====\n{stdout}");
    if !stderr.trim().is_empty() {
        println!("===== net_session selftest stderr =====\n{stderr}");
    }

    assert!(
        output.status.success(),
        "selftest exited {:?} (expected success). stdout above, stderr:\n{stderr}",
        output.status.code()
    );

    // Independently re-verify the key invariants from the captured output so this
    // test fails loudly (not just on exit code) if the binary's own checks regress.
    assert!(
        stdout.contains("PASS checksum equal across processes"),
        "missing cross-process checksum-equality proof in output"
    );
    assert!(
        stdout.contains("PASS convergence"),
        "missing convergence proof in output"
    );
    assert!(
        stdout.contains("PROBE host-killed: client errored cleanly"),
        "missing host-killed robustness probe in output"
    );
    assert!(
        stdout.contains("PROBE rebind:") && stdout.contains("rebind success=true"),
        "missing sequential port-rebind probe in output"
    );
    assert!(
        stdout.contains("ALL PROBES PASSED"),
        "selftest did not report all probes passing"
    );

    // The two FINAL lines (host + client) must both be present and carry the SAME
    // checksum token — the across-process serialization proof, re-derived here.
    let host_final = stdout
        .lines()
        .find(|l| l.contains("FINAL role=host"))
        .expect("no host FINAL line in output");
    let client_final = stdout
        .lines()
        .find(|l| l.contains("FINAL role=client"))
        .expect("no client FINAL line in output");
    let host_sum = checksum_token(host_final).expect("host FINAL missing checksum");
    let client_sum = checksum_token(client_final).expect("client FINAL missing checksum");
    assert_eq!(
        host_sum, client_sum,
        "host/client final checksums must match across the process boundary:\n  {host_final}\n  {client_final}"
    );
}

/// Extract the `checksum=0x...` token from a FINAL line.
fn checksum_token(line: &str) -> Option<&str> {
    line.split_whitespace().find(|t| t.starts_with("checksum=0x"))
}
