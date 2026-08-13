#!/bin/sh
# run_clean.sh — ION orphan cleanup + global timeout wrapper (zero-code fallback).
#
# Companion to the in-process watchdog in smollm2_pool_demo.c
# (DESIGN_ION_CLEANUP.md).  A hung-but-alive process leaks the 24MB ION
# carveout pool and poisons later runs; a dead process always releases ION.
# This script:
#   1) If the ION debugfs shows ION is non-trivially used, kills any stale
#      holders of this model binary (exact name match on `ps w`, never pkill -f).
#   2) Runs the target command under a global timeout (600s SIGTERM, +15s
#      SIGKILL) so even a silent hang cannot leave a live orphan behind.
#
# Usage:
#   sh run_clean.sh <binary> [args...]      # clean stale, then run
#   sh run_clean.sh --clean <binary>        # only clean stale, do not run
#
# ION debugfs (when CONFIG_DEBUG_FS + ION_DUMP enabled):
#   /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
#   Format varies: `usage rate:N%` (CV1800B) or `used: N` (older kernels).

ION_SUM=/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
SCRIPT=$(basename "$0")
CLEAN_ONLY=0
[ "$1" = "--clean" ] && { CLEAN_ONLY=1; shift; }

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--clean] <binary> [args...]" >&2
    exit 2
fi
BIN="$1"; shift
PROC=$(basename "$BIN")

# ---- ION usage parser: echo "0" if clean, non-zero rate% if a live holder.
ion_used() {
    [ -r "$ION_SUM" ] || { echo 0; return; }
    # format A (CV1800B): "usage rate:N%, memory usage peak ..."
    rate=$(grep -m1 'usage rate:' "$ION_SUM" 2>/dev/null \
           | sed 's/.*usage rate:[[:space:]]*//; s/%.*//' | tr -cd 0-9)
    if [ -n "$rate" ]; then
        [ "$rate" -gt 5 ] 2>/dev/null && echo "$rate" || echo 0
        return
    fi
    # format B: "used: N"
    used=$(grep -m1 'used:' "$ION_SUM" 2>/dev/null | tr -cd 0-9)
    if [ -n "$used" ]; then
        [ "$used" -gt 2000000 ] 2>/dev/null && echo "$used" || echo 0
    else
        echo 0
    fi
}

# ---- 1) stale holder cleanup ----
clean_stale() {
    used=$(ion_used)
    if [ "$used" -gt 0 ] 2>/dev/null; then
        echo "[clean] ION usage=${used} — killing stale '$PROC' holders..." >&2
        killed=0
        # Exact name match on `ps w`.  Never pkill -f: exclude this script's
        # own cmdline (which contains $PROC as an argument) and grep itself.
        for p in $(ps w 2>/dev/null | grep -w "$PROC" | grep -v "$SCRIPT" \
                   | grep -v 'grep' | awk '{print $1}'); do
            [ "$p" = "$$" ] && continue
            echo "  kill -9 $p" >&2
            kill -9 "$p" 2>/dev/null && killed=1
        done
        [ "$killed" = 1 ] && sleep 1
        echo "[clean] ION after kill: $(ion_used)" >&2
    else
        echo "[clean] ION clean (usage=${used}), no stale holders" >&2
    fi
}
clean_stale

[ "$CLEAN_ONLY" = 1 ] && exit 0

# ---- 2) run under global timeout ----
# BusyBox may lack `timeout`; fall back to background sleep+kill.
if command -v timeout >/dev/null 2>&1; then
    exec timeout -k 15 600 "$BIN" "$@"
else
    "$BIN" "$@" &
    pid=$!
    ( sleep 600; echo "[wd] 600s timeout, killing $pid" >&2; kill -9 "$pid" 2>/dev/null ) &
    wait $pid
fi
