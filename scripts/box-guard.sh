#!/usr/bin/env bash
# box-guard.sh — workstation resilience watchdog for the Ochroma dev box.
#
# WHY: this box runs heavy parallel agent/build workloads. The two historical
# ways it "goes down" are (a) a memory spike from concurrent cargo/rustc
# invocations dragging the desktop into a hard freeze, and (b) target-dir bloat
# silently filling the disk (observed regrowth to 93G+). This guard watches for
# both and takes only SAFE, REVERSIBLE action: it never kills a user process —
# it logs the top consumers for postmortem, and it prunes only regenerable
# build artifacts (incremental/ caches) when disk pressure is real.
#
# The actual freeze-prevention is done by the kernel (systemd-oomd) and the
# soft MemoryHigh ceiling on the user slice; this script is the early-warning +
# disk-janitor layer that complements them.
#
# Designed to be run on a timer (see box-guard.timer). Idempotent, no sudo.
set -uo pipefail

# ---- thresholds (override via environment) ----------------------------------
MEM_WARN_AVAIL_MB="${BOXGUARD_MEM_WARN_AVAIL_MB:-4096}"   # warn when avail RAM < 4 GiB
DISK_WARN_AVAIL_GB="${BOXGUARD_DISK_WARN_AVAIL_GB:-60}"   # warn when free disk < 60 GiB
DISK_ACT_AVAIL_GB="${BOXGUARD_DISK_ACT_AVAIL_GB:-40}"     # prune incremental when < 40 GiB
DISK_CRIT_AVAIL_GB="${BOXGUARD_DISK_CRIT_AVAIL_GB:-20}"   # loud critical when < 20 GiB
WATCH_ROOT="${BOXGUARD_WATCH_ROOT:-/home/tom-espen/src}"  # where target/ dirs live
CHECK_PATH="${BOXGUARD_CHECK_PATH:-/}"                    # filesystem to measure

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/box-guard"
LOG="$STATE_DIR/box-guard.log"
mkdir -p "$STATE_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s %s\n' "$(ts)" "$*" >> "$LOG"; }

# ---- memory -----------------------------------------------------------------
avail_mb=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
swap_total_kb=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
swap_free_kb=$(awk '/SwapFree/ {print $2}' /proc/meminfo)
swap_used_mb=$(( (swap_total_kb - swap_free_kb) / 1024 ))

if (( avail_mb < MEM_WARN_AVAIL_MB )); then
  log "MEM WARN: ${avail_mb}MiB available (< ${MEM_WARN_AVAIL_MB}MiB), swap used ${swap_used_mb}MiB. Top RSS:"
  # informational only — never kill. Capture the offenders for postmortem.
  ps -eo pid,rss,pcpu,comm --sort=-rss 2>/dev/null | head -8 \
    | awk '{printf "    pid=%s rss=%dMiB cpu=%s%% %s\n", $1, $2/1024, $3, $4}' >> "$LOG"
fi

# ---- disk -------------------------------------------------------------------
avail_gb=$(df -BG --output=avail "$CHECK_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')
avail_gb="${avail_gb:-9999}"

prune_incremental() {
  local freed_before freed_after
  freed_before=$(df -BM --output=avail "$CHECK_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')
  # Only touch regenerable incremental caches — never deps/ or the final binaries
  # of a possibly-in-progress build. cargo rebuilds these transparently.
  while IFS= read -r -d '' inc; do
    rm -rf "$inc" 2>/dev/null && log "  pruned $inc"
  done < <(find "$WATCH_ROOT" -type d -path '*/target/*/incremental' -prune -print0 2>/dev/null)
  freed_after=$(df -BM --output=avail "$CHECK_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')
  log "  disk reclaimed: $(( ${freed_after:-0} - ${freed_before:-0} ))MiB"
}

if (( avail_gb < DISK_CRIT_AVAIL_GB )); then
  log "DISK CRITICAL: ${avail_gb}GiB free on $CHECK_PATH (< ${DISK_CRIT_AVAIL_GB}GiB) — pruning incremental caches"
  prune_incremental
  log "  NOTE: still low after prune — consider 'cargo clean' on idle repos or clearing old datasets under $WATCH_ROOT"
elif (( avail_gb < DISK_ACT_AVAIL_GB )); then
  log "DISK LOW: ${avail_gb}GiB free (< ${DISK_ACT_AVAIL_GB}GiB) — pruning incremental caches"
  prune_incremental
elif (( avail_gb < DISK_WARN_AVAIL_GB )); then
  log "DISK WARN: ${avail_gb}GiB free (< ${DISK_WARN_AVAIL_GB}GiB) — watching"
fi

# ---- heartbeat (always, so silence == guard-not-running) --------------------
log "ok: mem_avail=${avail_mb}MiB swap_used=${swap_used_mb}MiB disk_avail=${avail_gb}GiB"

# keep the log bounded (last ~2000 lines)
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 2000 ]; then
  tail -1500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
