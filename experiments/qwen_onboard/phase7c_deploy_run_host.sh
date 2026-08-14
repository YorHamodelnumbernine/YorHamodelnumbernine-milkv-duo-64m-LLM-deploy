#!/bin/bash
# phase7c_deploy_run_host.sh — Host-side Phase 7c deploy + A/B orchestration.
#
# 前置: Duo 已 USB 连接 (192.168.42.1), duo_push/ssh/pull.py 可用.
# 流程: 构建 riscv64 二进制 -> push 二进制 + rsafe.bin + 运行脚本 ->
#       -> 运行 phase7c_run.sh (3 档 decode A/B) -> pull 回 3 份 log.
#
# 用法: sh phase7c_deploy_run_host.sh [DECODE_STEPS]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=~/Documents/MilkV_duo_project
export PATH=$ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
TPU=$ROOT/cvitek-tdl-sdk-cv180x/sample/3rd/tpu
STEPS="${1:-6}"

# ---- 1. 构建 ----
echo "[build] qwen_engine_lmhead2_phase7c"
riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
  -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
  -I $TPU/include -o qwen_engine_lmhead2_phase7c qwen_engine_lmhead2.c -lm -s \
  -L $TPU/lib -lcviruntime -lcvikernel

# ---- 2. push ----
echo "[push] binary + rsafe.bin + run script -> /data/qwen"
python3 $ROOT/duo_push.py qwen_engine_lmhead2_phase7c /data/qwen/qwen_engine_lmhead2_phase7c
python3 $ROOT/duo_push.py rsafe.bin /data/qwen/rsafe.bin
python3 $ROOT/duo_push.py phase7c_run.sh /data/qwen/phase7c_run.sh
python3 $ROOT/duo_push.py ../../run_clean.sh /data/qwen/run_clean.sh

# ---- 3. run (3 档 decode A/B) ----
echo "[run] phase7c_run.sh STEPS=$STEPS (预计 ~10-15 min)"
python3 $ROOT/duo_ssh.py "sh /data/qwen/phase7c_run.sh $STEPS" 2400

# ---- 4. pull logs ----
echo "[pull] logs"
for f in decode_rsh0_v0 decode_rsh1_v0 decode_rsh1_v1; do
  python3 $ROOT/duo_pull.py /data/qwen/$f.log ./$f.log 2>/dev/null || echo "pull $f.log failed"
done
echo "[done] logs: decode_rsh0_v0.log decode_rsh1_v0.log decode_rsh1_v1.log"
