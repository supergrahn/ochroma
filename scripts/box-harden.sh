#!/usr/bin/env bash
# box-harden.sh — one-shot resilience hardening for the Ochroma dev box.
#
# Review this file, then run it yourself. It is split into:
#   PART A (no sudo)  — enable the box-guard watchdog timer + a soft memory
#                       ceiling on your user session (graceful, reversible).
#   PART B (sudo)     — kernel-side freeze protection. Printed, NOT auto-run.
#
# Nothing here is destructive to your work. The watchdog only prunes
# regenerable target/*/incremental caches under disk pressure (see box-guard.sh).
set -euo pipefail

echo "=== PART A: user-level (no sudo) ============================================"

# 1) box-guard watchdog every 3 min (monitor mem/disk, prune incremental on low disk)
systemctl --user daemon-reload
systemctl --user enable --now box-guard.timer
echo "  [ok] box-guard.timer enabled"

# 2) Soft memory ceiling on the interactive app slice. MemoryHigh is a THROTTLE,
#    not a kill: when the slice's RSS crosses it the kernel reclaims aggressively
#    (paging to the 8G swap) BEFORE the whole box thrashes into a hard freeze.
#    Sized to leave headroom for the desktop/compositor on a 46G box.
#    Reverse with: systemctl --user set-property app.slice MemoryHigh=infinity
systemctl --user set-property app.slice MemoryHigh=38G
echo "  [ok] app.slice MemoryHigh=38G (reclaim-before-freeze)"

echo
echo "=== PART B: kernel-side (REQUIRES sudo — review, then run manually) ========="
cat <<'SUDO'
# B1) Make systemd-oomd act on MEMORY PRESSURE, not just swap exhaustion.
#     Default oomd only kills when swap is ~nearly full — by then the desktop is
#     already frozen. This makes it kill the worst cgroup when sustained memory
#     pressure (PSI) on the user slice exceeds 60% for 20s — i.e. before freeze.
sudo mkdir -p /etc/systemd/system/user@.service.d
sudo tee /etc/systemd/system/user@.service.d/50-oomd.conf >/dev/null <<'EOF'
[Service]
ManagedOOMMemoryPressure=kill
ManagedOOMMemoryPressureLimit=60%
EOF

sudo mkdir -p /etc/systemd/oomd.conf.d
sudo tee /etc/systemd/oomd.conf.d/50-pressure.conf >/dev/null <<'EOF'
[OOM]
DefaultMemoryPressureDurationSec=20s
EOF

# B2) A hard MemoryMax backstop on the user manager so a runaway can never take
#     100% of RAM (leaves ~6G for kernel + system services). High enough never to
#     bother normal work; low enough that global OOM/freeze can't happen.
sudo systemctl set-property user@1000.service MemoryMax=40G

# B3) Lower swappiness so the box prefers reclaiming page cache over swapping out
#     hot anonymous pages (less thrash under spikes). Persist it too.
sudo sysctl -w vm.swappiness=20
echo 'vm.swappiness=20' | sudo tee /etc/sysctl.d/90-box-harden.conf

# Apply oomd changes:
sudo systemctl daemon-reload
sudo systemctl restart systemd-oomd
SUDO

echo
echo "PART A applied. PART B above is yours to run after review."
