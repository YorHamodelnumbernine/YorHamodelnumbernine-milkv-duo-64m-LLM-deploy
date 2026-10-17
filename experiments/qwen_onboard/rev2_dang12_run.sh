#!/bin/sh
# rev2_dang12_run.sh — INT8 KV 复核补充旁证: 档1 (LW_READ=mmap GSC_ION=1) / 档2 (CENT_ION=0).
# 每档: 3-prompt 回归 + DECODE_STEPS=2 (M=1, INT8 KV), VERIFY=1 (bit-exact 红线), RSH=1 (离线 rsafe).
# 用法(板上): sh rev2_dang12_run.sh
set -x
cd /data/qwen || exit 1
WD=/data/qwen

# ---- swap 兜底 (standing instruction) ----
if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    [ -f /swap ] || dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap ====="; swapon --show 2>/dev/null; free -m | head -2

drop_caches() { sync 2>/dev/null; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "[warn] drop_caches denied"; }

echo "===== 档1: LW_READ=mmap GSC_ION=1 (mmap 基线锚点, DECODE_STEPS=2) ====="
drop_caches
sh $WD/run_clean.sh --clean $WD/qwen_engine_int8kv
LW_READ=mmap GSC_ION=1 VERIFY=1 RSH=1 DECODE=1 DECODE_STEPS=2 PROFILE=1 \
    sh $WD/run_clean.sh $WD/qwen_engine_int8kv 2>&1 | tee $WD/rev2_dang1_int8.log

echo "===== 档2: CENT_ION=0 (centroid DDR 回退, 释放 1.75MB ION; DECODE_STEPS=2) ====="
drop_caches
sh $WD/run_clean.sh --clean $WD/qwen_engine_int8kv
CENT_ION=0 GSC_ION=1 VERIFY=1 RSH=1 DECODE=1 DECODE_STEPS=2 PROFILE=1 \
    sh $WD/run_clean.sh $WD/qwen_engine_int8kv 2>&1 | tee $WD/rev2_dang2_int8.log

echo "===== DONE ====="
