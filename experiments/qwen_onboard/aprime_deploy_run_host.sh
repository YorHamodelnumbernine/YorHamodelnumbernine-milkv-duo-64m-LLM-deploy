#!/bin/bash
# aprime_deploy_run_host.sh — A' (LW_READ=ion_db) 部署 + A/B 验证编排.
#
# 前置: Duo 已 USB 连接 (192.168.42.1), duo_push/ssh/pull.py 可用.
# 流程: 构建 riscv64 二进制 -> push 二进制 + run 脚本 ->
#       -> 运行 aprime_run.sh (4 档 decode A/B) -> pull 回 4 份 log.
#
# 用法: sh aprime_deploy_run_host.sh [DECODE_STEPS]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=~/Documents/MilkV_duo_project
export PATH=$ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
TPU=$ROOT/cvitek-tdl-sdk-cv180x/sample/3rd/tpu
STEPS="${1:-6}"

# ---- 1. 构建 ----
echo "[build] qwen_engine_lmhead2_aprime"
riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
  -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
  -I $TPU/include -o qwen_engine_lmhead2_aprime qwen_engine_lmhead2.c -lm -s \
  -L $TPU/lib -lcviruntime -lcvikernel

# ---- 2. push ----
echo "[push] binary + run script -> /data/qwen"
python3 $ROOT/duo_push.py qwen_engine_lmhead2_aprime /data/qwen/qwen_engine_lmhead2_aprime
python3 $ROOT/duo_push.py aprime_run.sh /data/qwen/aprime_run.sh
python3 $ROOT/duo_push.py ../../run_clean.sh /data/qwen/run_clean.sh

# ---- 3. run (4 档 decode A/B) ----
echo "[run] aprime_run.sh STEPS=$STEPS (预计 ~25-35 min)"
python3 $ROOT/duo_ssh.py "sh /data/qwen/aprime_run.sh $STEPS" 3600

# ---- 4. pull logs ----
echo "[pull] logs"
for f in aprime_b_mmap aprime_a_perf aprime_a_ver aprime_a_perf2; do
  python3 $ROOT/duo_pull.py /data/qwen/$f.log ./$f.log 2>/dev/null || echo "pull $f.log failed"
done
echo "[done] logs: aprime_b_mmap.log aprime_a_perf.log aprime_a_ver.log aprime_a_perf2.log"
