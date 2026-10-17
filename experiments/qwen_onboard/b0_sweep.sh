#!/bin/sh
# b0_sweep.sh — B-P0 扫测: 基线 + Δ{0,6,12} × DONTNEED{0,1} (同 B2 口径).
# 口径: GSC_ION=1 CENT_ION=0 LW_READ=mmap RSH=1 VERIFY=0 PROFILE=1 CHAT=1
#       CHAT_INPUT=/data/qwen/input_tokens_35.bin DECODE_STEPS=6
# 每轮 drop_caches 冷启动, tee 独立 log.
cd /data/qwen || exit 1
BIN=qwen_engine_lmhead2_b0
WD=/data/qwen
drop_caches() { sync 2>/dev/null; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true; }
run() {
    name="$1"; shift
    drop_caches
    echo "===== [$name] $* ====="
    # env: 显式变量赋值 (busybox ash 不识别 "$@" 展开的 NAME=value 为赋值)
    env GSC_ION=1 CENT_ION=0 LW_READ=mmap RSH=1 VERIFY=0 PROFILE=1 CHAT=1 \
        CHAT_INPUT=/data/qwen/input_tokens_35.bin DECODE_STEPS=6 "$@" \
        sh $WD/run_clean.sh $WD/$BIN 2>&1 | tee $WD/b0_$name.log
}
echo "===== B-P0 sweep start $(date) ====="
run base
run d0_nn0   B_PREFETCH=1 B_PREFETCH_DELTA=0  B_PREFETCH_DONTNEED=0
run d0_nn1   B_PREFETCH=1 B_PREFETCH_DELTA=0  B_PREFETCH_DONTNEED=1
run d6_nn0   B_PREFETCH=1 B_PREFETCH_DELTA=6  B_PREFETCH_DONTNEED=0
run d6_nn1   B_PREFETCH=1 B_PREFETCH_DELTA=6  B_PREFETCH_DONTNEED=1
run d12_nn0  B_PREFETCH=1 B_PREFETCH_DELTA=12 B_PREFETCH_DONTNEED=0
run d12_nn1  B_PREFETCH=1 B_PREFETCH_DELTA=12 B_PREFETCH_DONTNEED=1
echo "===== B-P0 sweep done $(date) ====="
