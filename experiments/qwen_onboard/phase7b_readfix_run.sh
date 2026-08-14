#!/bin/sh
# phase7b_readfix_run.sh — On-Duo Phase 7b A/B: cross-layer double-buffer + TIU profiling.
#
# 对比 decode (M=1, KV cache, DECODE_STEPS 步) 的 4 档:
#   1) mmap   VERIFY=1  : Phase 6 基线 + host int32 参考 (同 session 基准)
#   2) mmap   VERIFY=0  : 基线 compute floor (隔离 host 参考开销)
#   3) mmap_db VERIFY=1 : Phase 7b 跨层双缓冲 (正确性 + 性能)
#   4) mmap_db VERIFY=0 : 双缓冲 compute floor (理论下限)
# 每档 PROFILE=1: eng_matmul 分段计时 (dequant/runcmd/verify/other...).
# 用法(板上): sh phase7b_readfix_run.sh [STEPS]
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

# ---- [1/4] decode mmap VERIFY=1 (Phase 6 基线) ----
echo "===== [1/4] decode mmap VERIFY=1 (Phase 6 baseline) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7b
run_engine mmap 1 decode_mmap

# ---- [2/4] decode mmap VERIFY=0 (基线 compute floor) ----
echo "===== [2/4] decode mmap VERIFY=0 (baseline compute floor) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7b
run_engine mmap 0 decode_mmap_v0

# ---- [3/4] decode mmap_db VERIFY=1 (双缓冲, 正确性) ----
echo "===== [3/4] decode mmap_db VERIFY=1 (double-buffer, correctness) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7b
run_engine mmap_db 1 decode_mmap_db

# ---- [4/4] decode mmap_db VERIFY=0 (双缓冲 floor) ----
echo "===== [4/4] decode mmap_db VERIFY=0 (double-buffer floor) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_phase7b
run_engine mmap_db 0 decode_mmap_db_v0

echo "===== DONE ====="
