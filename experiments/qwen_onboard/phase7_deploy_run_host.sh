#!/bin/bash
# phase7_deploy_run_host.sh — Host-side Phase 7 / A' deploy + A/B orchestration.
#
# 前置: Duo 已 USB 连接 (192.168.42.1), duo_push/ssh/pull.py 可用.
#
# 用法:
#   sh phase7_deploy_run_host.sh [DECODE_STEPS]          # legacy readfix (默认)
#   sh phase7_deploy_run_host.sh aprime [DECODE_STEPS]   # A': ion_db vs mmap 基线 A/B + 回归
#
# A' 模式流程:
#   构建 qwen_engine_lmhead2_aprime (LW_READ=ion_db 含, 默认 mmap 不变)
#   -> push 二进制 + phase7f_aprime_run.sh + run_clean.sh
#   -> 运行 phase7f_aprime_run.sh STEPS (3 档: B 基线 mmap 11.29s / A perf ion_db 验收9.5s
#      / A corr VERIFY=1 回归)
#   -> pull decode_b_mmap / decode_a_iondb / decode_a_iondb_v1 / ion_after
#   -> 提示: python3 phase7_analyze_logs.py (对比表 + 验收线 + 回归清单)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=~/Documents/MilkV_duo_project
export PATH=$ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
TPU=$ROOT/cvitek-tdl-sdk-cv180x/sample/3rd/tpu

MODE="readfix"
if [ $# -gt 0 ] && { [ "$1" = "aprime" ] || [ "$1" = "ion_db" ]; }; then
  MODE="aprime"; shift
fi
# aprime 档沿用 A/B 惯例 6 步; legacy readfix 默认 10 步
if [ "$MODE" = "aprime" ]; then STEPS="${1:-6}"; else STEPS="${1:-10}"; fi

build_engine() {  # $1=out binary name
  echo "[build] $1 (LW_READ=ion_db 含, 默认 mmap 不变)"
  riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
    -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
    -I $TPU/include -o "$1" qwen_engine_lmhead2.c -lm -s \
    -L $TPU/lib -lcviruntime -lcvikernel
}

if [ "$MODE" = "aprime" ]; then
  # ---- A' A/B ----
  BIN=qwen_engine_lmhead2_aprime
  build_engine "$BIN"

  echo "[push] $BIN + run script -> /data/qwen"
  python3 $ROOT/duo_push.py "$BIN" /data/qwen/"$BIN"
  python3 $ROOT/duo_push.py phase7f_aprime_run.sh /data/qwen/phase7f_aprime_run.sh
  python3 $ROOT/duo_push.py ../../run_clean.sh /data/qwen/run_clean.sh

  echo "[run] phase7f_aprime_run.sh STEPS=$STEPS (B 基线 + A perf + A corr, 预计 ~30-45 min)"
  python3 $ROOT/duo_ssh.py "sh /data/qwen/phase7f_aprime_run.sh $STEPS" 3600

  echo "[pull] logs"
  for f in decode_b_mmap decode_a_iondb decode_a_iondb_v1 ion_after; do
    python3 $ROOT/duo_pull.py /data/qwen/$f.log ./$f.log 2>/dev/null || echo "pull $f.log failed"
  done
  echo "[done] logs in $(pwd): decode_b_mmap.log decode_a_iondb.log decode_a_iondb_v1.log ion_after.log"
  echo "下一步: python3 phase7_analyze_logs.py (对比表 + 验收线 9.5s + 回归清单)"
else
  # ---- legacy readfix (Phase 7 读路径归因, 默认) ----
  echo "[build] engine (LW_READ switch) + layer_read_bench"
  riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
    -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
    -I $TPU/include -o qwen_engine_lmhead2_phase7 qwen_engine_lmhead2.c -lm -s \
    -L $TPU/lib -lcviruntime -lcvikernel
  riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
    -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
    -o layer_read_bench layer_read_bench.c -lm -s

  echo "[push] binaries + run scripts -> /data/qwen"
  python3 $ROOT/duo_push.py qwen_engine_lmhead2_phase7 /data/qwen/qwen_engine_lmhead2_phase7
  python3 $ROOT/duo_push.py layer_read_bench /data/qwen/layer_read_bench
  python3 $ROOT/duo_push.py phase7_readfix_run.sh /data/qwen/phase7_readfix_run.sh
  python3 $ROOT/duo_push.py ../../run_clean.sh /data/qwen/run_clean.sh

  echo "[run] phase7_readfix_run.sh STEPS=$STEPS (预计 ~20-25 min)"
  python3 $ROOT/duo_ssh.py "sh /data/qwen/phase7_readfix_run.sh $STEPS" 2400

  echo "[pull] logs"
  for f in layer_read_bench decode_mmap decode_mmap_ra decode_pread; do
    python3 $ROOT/duo_pull.py /data/qwen/$f.log ./$f.log 2>/dev/null || echo "pull $f.log failed"
  done
  echo "[done] logs in $(pwd): layer_read_bench.log decode_mmap.log decode_mmap_ra.log decode_pread.log"
fi
