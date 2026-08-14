#!/bin/sh
# phase7b2_readfix_run.sh — On-Duo Phase 7b round-2: dual-core prefetch thread A/B.
#
# 关键对照 (decode M=1, DECODE_STEPS 步, PROFILE=1 decode-only 剖析):
#   1) mmap    VERIFY=0 : 基线 compute floor (纯净 decode-only profile)
#   2) mmap_th VERIFY=1 : 双核预取线程 (正确性红线 + 性能)
#   3) mmap_th VERIFY=0 : 双核预取线程 floor
# 用法(板上): sh phase7b2_readfix_run.sh [STEPS]
set -x
cd /data/qwen || exit 1
WD=/data/qwen
STEPS="${1:-6}"

# ---- 前置: 开 swap (standing instruction) ----
if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    if [ ! -f /swap ]; then dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null; fi
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap status ====="; swapon --show 2>/dev/null; free -m | head -2

run_engine() {  # $1=LW_READ  $2=VERIFY  $3=logname
    if command -v timeout >/dev/null 2>&1; then
        LW_READ=$1 VERIFY=$2 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            timeout -k 15 900 $WD/qwen_engine_lmhead2_phase7b 2>&1 | tee $WD/$3.log
    else
        ( LW_READ=$1 VERIFY=$2 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            $WD/qwen_engine_lmhead2_phase7b 2>&1 | tee $WD/$3.log ) &
        local pid=$!
        ( sleep 900; echo "[wd] 900s timeout $3" >&2; kill -9 $pid 2>/dev/null ) &
        wait $pid
    fi
}

echo "===== [1/3] decode mmap VERIFY=0 (baseline floor, clean decode profile) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7b
run_engine mmap 0 decode_mmap_v0

echo "===== [2/3] decode mmap_th VERIFY=1 (dual-core prefetch, correctness) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7b
run_engine mmap_th 1 decode_mmap_th

echo "===== [3/3] decode mmap_th VERIFY=0 (dual-core prefetch floor) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7b
run_engine mmap_th 0 decode_mmap_th_v0

echo "===== DONE ====="
