#!/bin/sh
# a1_followup_verify.sh — A1 follow-up 验证 (ms3/5/7 post-prefill 释放 fix + prefill dequant%).
# 二进制 qwen_a1_followup = 16f32ab + ms3/5/7 释放 fix + prefill PROFILE 测量点 (PROFILE=1 门控).
# 验收口径 (CEO msg 848/852):
#   ① 3-prompt bit-exact 回归 (2130/12095/99366) + TIU BIT-EXACT — A1_MERGE=1
#   ② prefill dequant% A/B (A1_MERGE=1 vs 0) — 3-prompt 回归 PROFILE("prefill 3-prompt")
#   ③ CHAT decode 无退化 (A1_MERGE=1 ms3/5/7 释放后 vs B2 11.21s) + CHAT prefill dequant%
# 用法(板上): sh a1_followup_verify.sh
set -x
cd /data/qwen || exit 1
WD=/data/qwen
BIN=qwen_a1_followup

# ---- swap 兜底 ----
if ! swapon --show 2>/dev/null | grep -q '/swap'; then
    [ -f /swap ] || dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null
    mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
echo "===== swap ====="; swapon --show 2>/dev/null; free -m | head -2
drop_caches() { sync 2>/dev/null; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true; }

echo "===== [1/5] 3-prompt regression A1_MERGE=1 (bit-exact + prefill dequant%) ====="
drop_caches; sh $WD/run_clean.sh --clean $WD/$BIN
GSC_ION=1 CENT_ION=0 LW_READ=mmap A1_MERGE=1 VERIFY=1 PROFILE=1 RSH=1 \
    sh $WD/run_clean.sh $WD/$BIN 2>&1 | tee $WD/a1_fl_reg_a1.log
grep -E 'expected_next|P1/P2 bit-exact|TIU runs|24L regression|PROFILE|dequant_rvv|dqcopy|eng_matmul wall' $WD/a1_fl_reg_a1.log | head -20

echo "===== [2/5] 3-prompt regression A1_MERGE=0 (B2 prefill dequant% 基线) ====="
drop_caches; sh $WD/run_clean.sh --clean $WD/$BIN
GSC_ION=1 CENT_ION=0 LW_READ=mmap A1_MERGE=0 VERIFY=1 PROFILE=1 RSH=1 \
    sh $WD/run_clean.sh $WD/$BIN 2>&1 | tee $WD/a1_fl_reg_b2.log
grep -E 'expected_next|P1/P2 bit-exact|TIU runs|24L regression|PROFILE|dequant_rvv|dqcopy|eng_matmul wall' $WD/a1_fl_reg_b2.log | head -20

echo "===== [3/5] CHAT decode A1_MERGE=1 (ms3/5/7 释放后 decode 恢复) ====="
drop_caches; sh $WD/run_clean.sh --clean $WD/$BIN
GSC_ION=1 CENT_ION=0 LW_READ=mmap CHAT=1 DECODE_STEPS=6 A1_MERGE=1 PROFILE=1 VERIFY=1 RSH=1 \
    sh $WD/run_clean.sh $WD/$BIN 2>&1 | tee $WD/a1_fl_chat_a1.log
grep -E 'CHAT prefill done|CHAT decode avg|CHAT bit-exact|PROFILE|dequant_rvv|dqcopy|eng_matmul wall' $WD/a1_fl_chat_a1.log | head -20

echo "===== [4/5] CHAT decode A1_MERGE=0 (B2 基线) ====="
drop_caches; sh $WD/run_clean.sh --clean $WD/$BIN
GSC_ION=1 CENT_ION=0 LW_READ=mmap CHAT=1 DECODE_STEPS=6 A1_MERGE=0 PROFILE=1 VERIFY=1 RSH=1 \
    sh $WD/run_clean.sh $WD/$BIN 2>&1 | tee $WD/a1_fl_chat_b2.log
grep -E 'CHAT prefill done|CHAT decode avg|CHAT bit-exact|PROFILE|dequant_rvv|dqcopy|eng_matmul wall' $WD/a1_fl_chat_b2.log | head -20

echo "===== [5/5] ION after ====="
cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary 2>/dev/null | head -4
echo "===== DONE ====="
