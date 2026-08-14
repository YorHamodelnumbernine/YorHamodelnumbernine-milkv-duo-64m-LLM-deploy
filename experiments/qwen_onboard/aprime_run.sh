#!/bin/sh
# aprime_run.sh — A' (LW_READ=ion_db) A/B 验证 (on-Duo).
#
# 对比 decode (M=1, KV cache, DECODE_STEPS 步, PROFILE=1):
#   1) B 基线   : LW_READ=mmap GSC_ION=1 VERIFY=0   -> 复现 11.29s/token 锚点
#   2) A 性能   : LW_READ=ion_db VERIFY=0           -> 期望 6.9-9.1s, 验收线 9.5s, 全档 <10s
#   3) A 正确性 : LW_READ=ion_db VERIFY=1           -> NEXT 3/3 + bad1=bad2=r_opt=rsh=0 + 回归项
#   4) A 稳定复跑: LW_READ=ion_db VERIFY=0          -> 与 2) 差 ±0.3s 内
# 每档前 run_clean.sh --clean (清 stale ION) + drop_caches (冷页缓存公平对比).
# 用法(板上): sh aprime_run.sh [STEPS]
set -x
cd /data/qwen || exit 1
WD=/data/qwen
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
    sh $WD/run_clean.sh --clean qwen_engine_lmhead2_aprime
    if command -v timeout >/dev/null 2>&1; then
        LW_READ=$1 GSC_ION=$2 VERIFY=$3 RSH=1 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            timeout -k 15 1200 $WD/qwen_engine_lmhead2_aprime 2>&1 | tee $WD/$4.log
    else
        ( LW_READ=$1 GSC_ION=$2 VERIFY=$3 RSH=1 DECODE=1 DECODE_STEPS=$STEPS PROFILE=1 \
            $WD/qwen_engine_lmhead2_aprime 2>&1 | tee $WD/$4.log ) &
        pid=$!
        ( sleep 1200; echo "[wd] 1200s timeout $4" >&2; kill -9 $pid 2>/dev/null ) &
        wait $pid
    fi
}

echo "===== [1/4] B 基线: LW_READ=mmap GSC_ION=1 VERIFY=0 (期望 ~11.29s/token) ====="
run_engine mmap 1 0 aprime_b_mmap

echo "===== [2/4] A 性能: LW_READ=ion_db VERIFY=0 (期望 6.9-9.1s, 验收 <10s) ====="
run_engine ion_db 0 0 aprime_a_perf

echo "===== [3/4] A 正确性: LW_READ=ion_db VERIFY=1 (NEXT 3/3 + bit-exact + 回归) ====="
run_engine ion_db 0 1 aprime_a_ver

echo "===== [4/4] A 稳定性复跑: LW_READ=ion_db VERIFY=0 (与 [2/4] 差 ±0.3s 内) ====="
run_engine ion_db 0 0 aprime_a_perf2

echo "===== DONE ====="
