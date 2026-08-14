#!/bin/sh
# ion_trace.sh — trace ION used trajectory: GSC_ION=1 vs ion_db init/steady-state.
# 目标: 溯源 carveout "memory usage peak 25300992" 来源 (boot 累计高水位)。
#   GSC_ION=1 (mmap B 基线): 预期 init 后 used ~23.3MB (gsc 21.33 + mem + pools)
#   ion_db (A' 整层):         预期 init 后 used ~18.76MB (SD_BUF 16.8 + mem + pools)
# 若 GSC_ION=1 顶到 ~25.3MB -> 峰值来自 B 基线 gsc 缓存, 与 ion_db/B-2 稳态无关。
cd /data/qwen || exit 1
ION_SUM=/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
pkill -9 -x qwen_engine_lmhead2_aprime 2>/dev/null; sleep 1

used() { grep -m1 '^\[0\]' "$ION_SUM" 2>/dev/null | sed -n 's/.*used:\([0-9]*\).*/\1/p'; }

trace() {
  label="$1"; shift
  sh run_clean.sh --clean qwen_engine_lmhead2_aprime
  echo "===== $label ====="
  env "$@" ./qwen_engine_lmhead2_aprime > /data/qwen/ion_trace_$label.log 2>&1 &
  P=$!
  for i in $(seq 1 22); do
    echo "t=$((i*2))s used=$(used)"
    sleep 2
  done
  pkill -9 -x qwen_engine_lmhead2_aprime 2>/dev/null; sleep 1
  echo "  after-kill used=$(used)"
}

trace gsc_ion1 LW_READ=mmap GSC_ION=1 VERIFY=0 RSH=1
trace ion_db LW_READ=ion_db VERIFY=0 RSH=1

echo "===== final ====="
grep -E 'used:|peak' "$ION_SUM" | head -4
echo "ion_trace done"
