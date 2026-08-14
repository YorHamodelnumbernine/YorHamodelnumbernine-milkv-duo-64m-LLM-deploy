#!/bin/sh
# phase7_readfix_run.sh — On-Duo Phase 7 layer-weight read-path A/B.
#
# 1) 读路径吞吐归因 (layer_read_bench, 8 种机制, cold-cache 重读 24 层)
# 2) decode 基线   : LW_READ=mmap    (Phase 6 当前路径, ~8MB/s)
# 3) decode 修复   : LW_READ=mmap_ra (mmap + readahead 整层预取)
# 4) decode 对照   : LW_READ=pread   (顺序 pread, 有匿名内存风险, 仅对照)
#
# 每个引擎调用内嵌 3-prompt 回归 (expected_next 3/3 + bad1/bad2/r_opt bit-exact),
# 因此一次运行同时给出回归 + decode per-token 对比.
#
# 用法(板上): sh phase7_readfix_run.sh [STEPS]
#   STEPS 默认 10 (decode 步数). 前置: swap 已开 (见下), run_clean.sh 可用.
set -x
cd /data/qwen || exit 1
WD=/data/qwen
STEPS="${1:-10}"

# ---- 前置: 开 swap (standing instruction), 不依赖它但提供 OOM 兜底 ----
if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    if [ ! -f /swap ]; then dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null; fi
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap status ====="; swapon --show 2>/dev/null; free -m | head -2

# ---- [1/3] 读路径归因 ----
echo "===== [1/3] layer_read_bench (attribution, cold=1, reps=2) ====="
$WD/layer_read_bench --dir $WD 1,2,3,4,5,6,7,8,9 --reps 2 --cold 2>&1 | tee $WD/layer_read_bench.log

# ---- ION 清理一次 (防前次崩溃残留占用 carveout) ----
sh $WD/run_clean.sh --clean

# ---- 900s 每档 timeout (BusyBox 兜底): 防 pread 对照 swap 抖动超 600s 被误杀 ----
run_engine() {  # $1=LW_READ  $2=logname
    if command -v timeout >/dev/null 2>&1; then
        LW_READ=$1 DECODE=1 DECODE_STEPS=$STEPS timeout -k 15 900 \
            $WD/qwen_engine_lmhead2_phase7 2>&1 | tee $WD/$2.log
    else
        ( LW_READ=$1 DECODE=1 DECODE_STEPS=$STEPS $WD/qwen_engine_lmhead2_phase7 \
            2>&1 | tee $WD/$2.log ) &
        local pid=$!
        ( sleep 900; echo "[wd] 900s timeout $2" >&2; kill -9 $pid 2>/dev/null ) &
        wait $pid
    fi
}

# ---- [2/3] decode 基线 (mmap) ----
echo "===== [2/3] decode baseline LW_READ=mmap STEPS=$STEPS ====="
run_engine mmap decode_mmap

# ---- [3/3] decode 修复 (mmap_ra) ----
echo "===== [3/3] decode fix LW_READ=mmap_ra STEPS=$STEPS ====="
run_engine mmap_ra decode_mmap_ra

# ---- [4] decode 对照 (pread) ----
echo "===== [4] decode comparison LW_READ=pread STEPS=$STEPS ====="
run_engine pread decode_pread

echo "===== DONE ====="
