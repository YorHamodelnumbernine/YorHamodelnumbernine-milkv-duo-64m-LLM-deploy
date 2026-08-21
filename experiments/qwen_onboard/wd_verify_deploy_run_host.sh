#!/bin/bash
# wd_verify_deploy_run_host.sh — ION 看门狗验证 drop-in 部署 + 执行 (host 侧).
#
# 设计要点:
#   - 从 `git archive HEAD` (干净提交态) 构建两二进制, 排除工作区 in-progress
#     INT4 dequant 改动 (dequant_kal_rvv 签名变更) 干扰, 验证对象 = 已提交看门狗.
#   - 板子需已释放 (CEO 确认后执行); 本脚本不检查并发占用, 由协调门禁保证.
#
# 流程: 导出干净 HEAD 树 -> 构建 qwen_engine_lmhead2_aprime + smollm2_pool_demo
#       -> push 两二进制 + wd_verify_run.sh + run_clean.sh
#       -> 板上跑 wd_verify_run.sh (三项 <1min + 3-prompt 回归 ~60s)
#       -> pull 回 3 份 log (wd_qh.log wd_sm.log wd_reg.log)
#
# 用法: sh wd_verify_deploy_run_host.sh
# 前置: Duo 已 USB 连接 (192.168.42.1), duo_push/duo_ssh/duo_pull 可用.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=~/Documents/MilkV_duo_project
export PATH=$ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
TPU=$ROOT/cvitek-tdl-sdk-cv180x/sample/3rd/tpu
BENCH=$ROOT/tpu_bench
TMP=$(mktemp -d /tmp/wdverify.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

echo "[export] clean committed tree (HEAD=$(git -C "$BENCH" rev-parse --short HEAD))"
git -C "$BENCH" archive HEAD | tar -x -C "$TMP"

echo "[build] qwen_engine_lmhead2_aprime (committed watchdog)"
riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -march=rv64imafdcv0p7xthead \
  -mcmodel=medany -mabi=lp64d -O3 -std=gnu11 -fsigned-char \
  -I $TPU/include -o "$TMP/qwen_engine_lmhead2_aprime" \
  "$TMP/experiments/qwen_onboard/qwen_engine_lmhead2.c" -lm -s \
  -L $TPU/lib -lcviruntime -lcvikernel

echo "[build] smollm2_pool_demo (committed watchdog)"
make -C "$TMP" smollm2_pool_demo

echo "[push] binaries + scripts -> board"
python3 $ROOT/duo_push.py "$TMP/qwen_engine_lmhead2_aprime" /data/qwen/qwen_engine_lmhead2_aprime
python3 $ROOT/duo_push.py "$TMP/smollm2_pool_demo" /root/smollm2_pool_demo
python3 $ROOT/duo_push.py wd_verify_run.sh /data/qwen/wd_verify_run.sh
python3 $ROOT/duo_push.py "$BENCH/run_clean.sh" /data/qwen/run_clean.sh

echo "[run] wd_verify_run.sh (三项 hang + 3-prompt 回归, 预计 ~2min)"
python3 $ROOT/duo_ssh.py "sh /data/qwen/wd_verify_run.sh" 600

echo "[pull] logs"
for f in wd_qh wd_sm wd_reg; do
  python3 $ROOT/duo_pull.py /data/qwen/$f.log ./$f.log 2>/dev/null || echo "pull $f.log failed"
done
echo "[done] logs: wd_qh.log wd_sm.log wd_reg.log"
