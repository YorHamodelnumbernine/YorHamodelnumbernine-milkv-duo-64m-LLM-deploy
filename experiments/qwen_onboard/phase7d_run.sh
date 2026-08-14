#!/bin/sh
# phase7d_run.sh — On-Duo Phase 7d A/B: merged up/gate N-tile cmdbuf.
#
# 对比 decode (M=1, KV cache, DECODE_STEPS 步, PROFILE=1):
#   1) MERGE=0 VERIFY=0 : Phase 7c 基线 (per-tile up/gate, 6 sub/pass)
#   2) MERGE=1 VERIFY=0 : Phase 7d merged (1 sub/pass, 8x608)
#   3) MERGE=1 VERIFY=1 : 正确性红线 (NEXT 3/3 + decode bad1=bad2=rbad=0)
# 每档前 drop_caches 保证冷页缓存 (公平对比).
# 用法(板上): sh phase7d_run.sh [STEPS]
set -x
cd /data/qwen || exit 1
WD=/data/qwen
STEPS="${1:-6}"

# ---- 前置: 开 swap (standing instruction), 不依赖它但提供 OOM 兜底 ----
if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    if [ ! -f /swap ]; then dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null; fi
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap status ====="; swapon --show 2>/dev/null; free -m | head -2

drop_caches() {
    sync 2>/dev/null
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "[warn] drop_caches not permitted"
}

run_engine() {  # $1=MERGE  $2=VERIFY  $3=logname
    drop_caches
    if command -v timeout >/dev/null 2>&1; then
        MERGE=$1 VERIFY=$2 RSH=1 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            timeout -k 15 900 $WD/qwen_engine_lmhead2_phase7d 2>&1 | tee $WD/$3.log
    else
        ( MERGE=$1 VERIFY=$2 RSH=1 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            $WD/qwen_engine_lmhead2_phase7d 2>&1 | tee $WD/$3.log ) &
        local pid=$!
        ( sleep 900; echo "[wd] 900s timeout $3" >&2; kill -9 $pid 2>/dev/null ) &
        wait $pid
    fi
}

echo "===== [1/3] MERGE=0 VERIFY=0 (Phase 7c baseline, per-tile up/gate) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7d
run_engine 0 0 decode_m0_v0

echo "===== [2/3] MERGE=1 VERIFY=0 (Phase 7d merged up/gate, perf) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7d
run_engine 1 0 decode_m1_v0

echo "===== [3/3] MERGE=1 VERIFY=1 (Phase 7d correctness: NEXT 3/3 + decode bit-exact) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7d
run_engine 1 1 decode_m1_v1

echo "===== DONE ====="
