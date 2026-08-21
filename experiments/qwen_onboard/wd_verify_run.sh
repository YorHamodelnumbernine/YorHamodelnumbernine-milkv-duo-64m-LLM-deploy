#!/bin/sh
# wd_verify_run.sh — ION 看门狗 drop-in 验证 (on-Duo, 板子释放后立即执行).
#
# 三项 (drop-in, 总耗时 ~2min):
#   [1] qwen init-window hang : QH_INIT_HANG=1 WD_TIMEOUT_SEC=10
#        期望看门狗 ~10s 内 _exit(1) (退出码 1), ION 归零
#   [2] smollm2 hang          : SM_HANG_TEST=1 SM_WD_TIMEOUT=5
#        期望看门狗 ~5s 内  _exit(1) (退出码 1), ION 归零
#   [3] 3-prompt 正常回归     : 无 hang env
#        期望 24L regression expected_next 3/3 OK + TIU BIT-EXACT,
#        日志无 [WATCHDOG] (零误杀确认)
#
# 判定: rc=1 表示 watchdog _exit 触发 (与 timeout SIGKILL 的 137 区分);
#       耗时 ~超时值 (排除 30s timeout 兜底); ION pct 归零.
#
# 用法(板上): sh /data/qwen/wd_verify_run.sh
# 前置: 两二进制已就位 (/data/qwen/qwen_engine_lmhead2_aprime + /root/smollm2_pool_demo),
#       run_clean.sh 已 push 到 /data/qwen/.
set -x
WD=/data/qwen
cd "$WD" || exit 1

# ---- 前置: swap (standing instruction) ----
if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    if [ ! -f /swap ]; then dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null; fi
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap ====="; swapon --show 2>/dev/null; free -m | head -2

ION_SUM=/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
ion_pct() { grep -m1 'usage rate:' "$ION_SUM" 2>/dev/null | sed 's/.*usage rate:[[:space:]]*//; s/%.*//' | tr -cd 0-9; }
ion_pct2() { p=$(ion_pct); [ -z "$p" ] && echo 0 || echo "$p"; }

PASS=0; FAIL=0
note() { # $1=name $2=test_result(0=pass) $3=detail
    if [ "$2" -eq 0 ]; then PASS=$((PASS+1)); echo "[PASS] $1 | $3";
    else FAIL=$((FAIL+1)); echo "[FAIL] $1 | $3"; fi
}

echo "===== [1/3] qwen init-window hang (QH_INIT_HANG=1 WD_TIMEOUT_SEC=10) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_aprime
t0=$(date +%s)
QH_INIT_HANG=1 WD_TIMEOUT_SEC=10 timeout -k 5 30 $WD/qwen_engine_lmhead2_aprime > $WD/wd_qh.log 2>&1
rc=$?
t1=$(date +%s); el=$((t1-t0)); ion=$(ion_pct2)
echo "  rc=$rc elapsed=${el}s ion=${ion}% (expect rc=1 el~10s ion~0%)"
grep -E "QH_INIT_HANG|WATCHDOG" $WD/wd_qh.log
[ "$rc" -eq 1 ] && [ "$el" -ge 9 ] && [ "$ion" -le 5 ]
note "qwen init-window hang 触发" $? "rc=$rc el=${el}s ion=${ion}%"

echo "===== [2/3] smollm2 hang (SM_HANG_TEST=1 SM_WD_TIMEOUT=5) ====="
sh $WD/run_clean.sh --clean smollm2_pool_demo
t0=$(date +%s)
SM_HANG_TEST=1 SM_WD_TIMEOUT=5 timeout -k 5 30 /root/smollm2_pool_demo \
    /root/smollm2_instruct/ /root/input_tokens.bin 1 > $WD/wd_sm.log 2>&1
rc=$?
t1=$(date +%s); el=$((t1-t0)); ion=$(ion_pct2)
echo "  rc=$rc elapsed=${el}s ion=${ion}% (expect rc=1 el~5s ion~0%)"
grep -E "SM_HANG_TEST|NO HEARTBEAT|wd\]" $WD/wd_sm.log
[ "$rc" -eq 1 ] && [ "$el" -ge 4 ] && [ "$ion" -le 5 ]
note "smollm2 hang 触发" $? "rc=$rc el=${el}s ion=${ion}%"

echo "===== [3/3] 3-prompt 正常回归 (零误杀) ====="
sh $WD/run_clean.sh --clean qwen_engine_lmhead2_aprime
$WD/qwen_engine_lmhead2_aprime > $WD/wd_reg.log 2>&1
rc=$?
nwd=$(grep -c "WATCHDOG" $WD/wd_reg.log)
echo "  rc=$rc [WATCHDOG] occurrences=$nwd (expect 0)"
grep -E "expected_next|24L regression|total wall" $WD/wd_reg.log
[ "$rc" -eq 0 ] && [ "$nwd" -eq 0 ]
note "3-prompt 回归零误杀" $? "rc=$rc watchdog=$nwd"

echo "===== RESULT: PASS=$PASS FAIL=$FAIL ====="
echo "===== WD-VERIFY DONE ====="
exit $FAIL
