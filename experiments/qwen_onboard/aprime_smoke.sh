#!/bin/sh
# aprime_smoke.sh — quick A' validation: A config only (LW_READ=ion_db), 5 steps.
# 检查: 1) ION 分配成功 (19-ION gsc + 4 SD 槽)  2) decode 无 crash 3) t_layers/total 对比旧 11.0s.
# 用法(板上): sh aprime_smoke.sh [STEPS]
set -x
cd /data/qwen || exit 1
WD=/data/qwen
STEPS="${1:-5}"

if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    if [ ! -f /swap ]; then dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null; fi
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap status ====="; swapon --show 2>/dev/null; free -m | head -2

sync 2>/dev/null; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_aprime

echo "===== [SMOKE] LW_READ=ion_db VERIFY=0 STEPS=$STEPS ====="
LW_READ=ion_db GSC_ION=0 VERIFY=0 RSH=1 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
    timeout -k 15 900 $WD/qwen_engine_lmhead2_aprime 2>&1 | tee $WD/aprime_smoke.log

echo "===== ION summary ====="
grep -E '\[0\]|used:|peak' /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary 2>/dev/null | head -4
echo "===== SMOKE DONE ====="
