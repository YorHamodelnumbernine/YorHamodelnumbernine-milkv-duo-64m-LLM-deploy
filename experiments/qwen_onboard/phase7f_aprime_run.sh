#!/bin/sh
# phase7f_aprime_run.sh — On-Duo A' (LW_READ=ion_db) A/B vs mmap 基线.
#
# [Phase 8 回归工具链] 本脚本与 phase7_analyze_logs.py 一起构成 A'/B-2 回归工具链,
#   用于复测「ion_db 实验路径 vs mmap 出货基线」的 A/B 对比与验收线判定.
#   出货配置已锁定为 LW_READ=mmap GSC_ION=1 (11.29s/token 锚点), ion_db 为
#   已收口(不达 9.5s)的实验路径; 任何后续改动回归时按本脚本三档跑批即可.
#
# 对比 decode (M=1, KV cache, DECODE_STEPS 步, PROFILE=1, RSH=1 离线 rsafe):
#   1) B 基线 : LW_READ=mmap  GSC_ION=1 VERIFY=0  (复现 11.29s 锚点, cold mmap + gsc ION)
#   2) A perf : LW_READ=ion_db GSC_ION=0 VERIFY=0  (验收线 9.5s, 预期 6.9-9.1s)
#   3) A corr : LW_READ=ion_db GSC_ION=0 VERIFY=1  (回归: NEXT 3/3 + bit-exact + A' 指标)
# 每档前 drop_caches 保证冷页缓存 (公平对比); 每档前 run_clean.sh --clean 清 ION.
# 跑完后把 ION debugfs 存到 ion_after.log (host 侧 A/B 平台做无泄漏判定).
#
# 用法(板上): sh phase7f_aprime_run.sh [STEPS]   (默认 6)
set -x
cd /data/qwen || exit 1
WD=/data/qwen
BIN=qwen_engine_lmhead2_aprime
STEPS="${1:-6}"

# ---- 前置: 开 swap (standing instruction) ----
if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    if [ ! -f /swap ]; then dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null; fi
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap status ====="; swapon --show 2>/dev/null; free -m | head -2

drop_caches() {
    sync 2>/dev/null
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "[warn] drop_caches not permitted"
}

run_engine() {  # $1=LW_READ  $2=GSC_ION  $3=VERIFY  $4=logname
    drop_caches
    if command -v timeout >/dev/null 2>&1; then
        LW_READ=$1 GSC_ION=$2 VERIFY=$3 RSH=1 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            timeout -k 15 900 $WD/$BIN 2>&1 | tee $WD/$4.log
    else
        ( LW_READ=$1 GSC_ION=$2 VERIFY=$3 RSH=1 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            $WD/$BIN 2>&1 | tee $WD/$4.log ) &
        local pid=$!
        ( sleep 900; echo "[wd] 900s timeout $4" >&2; kill -9 $pid 2>/dev/null ) &
        wait $pid
    fi
}

echo "===== [1/3] B 基线: LW_READ=mmap GSC_ION=1 VERIFY=0 (锚点 11.29s) ====="
sh $WD/run_clean.sh --clean $BIN
run_engine mmap 1 0 decode_b_mmap

echo "===== [2/3] A perf: LW_READ=ion_db VERIFY=0 (验收线 9.5s) ====="
sh $WD/run_clean.sh --clean $BIN
run_engine ion_db 0 0 decode_a_iondb

echo "===== [3/3] A corr: LW_READ=ion_db VERIFY=1 (回归: NEXT 3/3 + bit-exact) ====="
sh $WD/run_clean.sh --clean $BIN
run_engine ion_db 0 1 decode_a_iondb_v1

echo "===== ION 无泄漏检查 (全部 run 退出后) ====="
cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary 2>/dev/null | head -8 > $WD/ion_after.log
pgrep -x $BIN 2>/dev/null | wc -l >> $WD/ion_after.log

echo "===== DONE ====="
