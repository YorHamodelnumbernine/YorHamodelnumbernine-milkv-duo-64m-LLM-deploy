#!/bin/bash
# aprime_conn_check.sh — A' (ION 双缓冲) 重启后连通性回归 + decode A/B 基线
#
# 用途: Duo 物理断电重启后一键确认环境就绪, 然后才允许 A' 下板实现.
#   1) RNDIS 连通    : ping 192.168.42.1
#   2) SSH 可登录    : duo_ssh.py "uname -a; free -m; uptime"
#   3) SD 顺序读速率 : drop_caches 后 cat 24 层 (192.1MiB), 期望 >= 20MiB/s
#   4) ION 状态      : carveout debugfs summary (无 stale 占用, 余量正常)
#   5) decode 基线(B): 若板上已有 qwen_engine_lmhead2_phase7e, 跑 6 步 GSC_ION=1
#                      基线, 解析 per-token 与 11.29s 锚点比对 (B 基线 A/B 对照用)
#
# 用法:
#   sh aprime_conn_check.sh              # 全部检查 (含 decode 基线, ~15-20min)
#   sh aprime_conn_check.sh --skip-decode # 只做连通性 + SD 速率 (快速, <2min)
#   sh aprime_conn_check.sh --steps 6     # 自定义 decode 步数 (默认 6)
#
# 依赖: ~/Documents/MilkV_duo_project/duo_ssh.py (paramiko), Duo 已 USB 连接
set -u
ROOT=~/Documents/MilkV_duo_project
SSH="python3 $ROOT/duo_ssh.py"
IP=192.168.42.1
BASE_AVG=11.29          # decode_e1_v0.log 锚点 (Phase 7e GSC_ION=1 生产口径)
BASE_TOL=0.5            # 允许漂移 (s/token)
STEPS=6
SKIP_DECODE=0
for a in "$@"; do
  case "$a" in
    --skip-decode) SKIP_DECODE=1 ;;
    --steps) ;; # 值由下一参数处理
    --steps=*) STEPS="${a#*=}" ;;
    [0-9]*) STEPS="$a" ;;
  esac
done

PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "==== A' 连通性回归 $(date '+%F %T') ===="
echo "  Duo=$IP  基线锚点=${BASE_AVG}s/token (±${BASE_TOL})  steps=$STEPS"

# ---- 1. RNDIS ----
echo "---- [1/5] RNDIS ping $IP ----"
if ping -c 3 -W 2 $IP >/dev/null 2>&1; then
  ok "ping $IP (RNDIS 枚举)"
else
  bad "ping $IP — Duo 不可达; 请确认 USB 重插 (CDCEther 枚举 enxe206575f3db8) 或物理断电重启"
  echo "==== 连通性回归: FAIL=$FAIL PASS=$PASS ===="
  exit 1
fi

# ---- 2. SSH ----
echo "---- [2/5] SSH 登录 ----"
SSH_OUT=$($SSH "uname -a; free -m | head -2; uptime" 15 2>/dev/null)
if [ $? -eq 0 ] && echo "$SSH_OUT" | grep -qiE 'Linux|root'; then
  ok "SSH 可登录: $(echo "$SSH_OUT" | head -1)"
  echo "$SSH_OUT" | sed 's/^/    /'
else
  bad "SSH 失败 (root/milkv@$IP)"
  echo "==== 连通性回归: FAIL=$FAIL PASS=$PASS ===="
  exit 1
fi

# ---- 3. SD 顺序读速率 ----
echo "---- [3/5] SD 顺序读速率 (192.1MiB cold, 期望 >=20MiB/s) ----"
# 板上用 BusyBox time 内建计时; cat 24 层触发顺序读
SD_CMD='sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; { time cat /data/qwen/layer*_kal.bin > /dev/null; } 2>&1; ls -la /data/qwen/layer0_kal.bin'
SD_OUT=$($SSH "$SD_CMD" 120 2>/dev/null)
SD_REAL=$(echo "$SD_OUT" | grep -oE 'real[[:space:]]+[0-9]+m[[:space:]]*[0-9.]+s|real[[:space:]]+[0-9.]+s' | head -1)
if [ -n "$SD_REAL" ]; then
  # 兼容 'real 0m9.43s' / 'real 0m 9.43s' / 'real 9.43s'
  if echo "$SD_REAL" | grep -q 'm'; then
    MIN=$(echo "$SD_REAL" | sed -E 's/.*[^0-9]([0-9]+)m.*/\1/')
    SEC=$(echo "$SD_REAL" | sed -E 's/.*[^0-9]([0-9]+)m[[:space:]]*([0-9.]+)s.*/\2/')
    SECS=$(echo "$MIN * 60 + $SEC" | bc 2>/dev/null)
  else
    SECS=$(echo "$SD_REAL" | sed -E 's/.*real[[:space:]]+([0-9.]+)s.*/\1/')
  fi
  RATE=$(echo "scale=2; 192.1 / $SECS" | bc 2>/dev/null)
  RATE_OK=$(echo "$RATE >= 20" | bc 2>/dev/null)
  echo "    cold read 24 层: $SD_REAL = ${RATE:-?}MiB/s (层文件: $(echo "$SD_OUT" | grep 'layer0_kal.bin' | awk '{print $5}')B)"
  if [ "$RATE_OK" = "1" ]; then ok "SD 顺序读 ${RATE}MiB/s (>=20)"; else bad "SD 读 ${RATE}MiB/s 低于 20MiB/s — 检查卡/接口"; fi
else
  echo "    SD 计时输出: $SD_OUT" | sed 's/^/    /'
  bad "SD 速率测量失败 (time/cat 输出异常)"
fi

# ---- 4. ION 状态 ----
echo "---- [4/5] ION carveout 状态 ----"
ION_SUM=/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
ION_OUT=$($SSH "cat $ION_SUM 2>/dev/null | head -8; echo '---'; pgrep -x qwen_engine_lmhead2_phase7e 2>/dev/null | wc -l" 15 2>/dev/null)
echo "$ION_OUT" | sed 's/^/    /'
if echo "$ION_OUT" | grep -q "used:"; then
  ok "ION debugfs 可读 (无 stale 进程: $(echo "$ION_OUT" | tail -1) 个引擎进程)"
else
  bad "ION debugfs 不可读或为空"
fi

# ---- 5. decode 基线 (B) ----
if [ "$SKIP_DECODE" = "1" ]; then
  echo "---- [5/5] decode 基线: --skip-decode 跳过 ----"
else
  echo "---- [5/5] decode 基线 (B): GSC_ION=1 VERIFY=1, ${STEPS} 步, 期望 ~${BASE_AVG}s/token ----"
  # 板上二进制存在性检查
  # duo_ssh.py 输出尾部带 [RC=N], 精确比较前先剥离 (取 ^YES$ 匹配行)
  HAS=$($SSH "test -x /data/qwen/qwen_engine_lmhead2_phase7e && echo YES || echo NO" 15 2>/dev/null)
  HAS=$(echo "$HAS" | grep -c '^YES$')
  if [ "$HAS" != "1" ]; then
    bad "板上无 qwen_engine_lmhead2_phase7e — 需先 push (phase7e_deploy_run_host.sh) 再跑基线"
  else
    # 前置 swap (standing instruction) + drop_caches + 单档 GSC_ION=1 基线
    # (BusyBox 无 timeout 时回退后台 sleep+kill, 同 phase7e_run.sh 模式)
    RUN='cd /data/qwen || exit 1
if ! swapon --show 2>/dev/null | grep -q /swap; then
  [ ! -f /swap ] && dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null
  mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true
fi
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
export GSC_ION=1 VERIFY=1 RSH=1 DECODE=1 DECODE_STEPS='"$STEPS"' PROFILE=1
if command -v timeout >/dev/null 2>&1; then
  timeout -k 15 900 ./qwen_engine_lmhead2_phase7e 2>&1 | tee /data/qwen/aprime_baseline.log
else
  (./qwen_engine_lmhead2_phase7e 2>&1 | tee /data/qwen/aprime_baseline.log) & P=$!
  ( sleep 900; kill -9 $P 2>/dev/null ) &
  wait $P
fi'
    BASE_LOG=$($SSH "$RUN" 1800 2>/dev/null)
    AVG=$(echo "$BASE_LOG" | grep -oE 'decode avg per-token = [0-9.]+' | grep -oE '[0-9.]+' | head -1)
    NEXT=$(echo "$BASE_LOG" | grep -c 'expected_next=.*OK')
    BEX=$(echo "$BASE_LOG" | grep -oE 'decode bit-exact: bad1=[0-9]+ bad2=[0-9]+ r_opt=[0-9]+ rsh=[0-9]+' | head -1)
    echo "    decode avg = ${AVG:-?}s/token (锚点 ${BASE_AVG})"
    echo "    NEXT 3/3 OK 行数 = ${NEXT}"
    echo "    $BEX"
    echo "$BASE_LOG" | grep '^DECODE pos' | sed 's/^/    /'
    if [ -n "$AVG" ]; then
      D=$(echo "scale=2; $AVG - $BASE_AVG" | bc 2>/dev/null)
      DABS=$(echo "$D" | sed 's/^-//')
      IN_TOL=$(echo "$DABS <= $BASE_TOL" | bc 2>/dev/null)
      if [ "$IN_TOL" = "1" ]; then ok "decode 基线 ${AVG}s/token (与 ${BASE_AVG} 锚点吻合)"; else bad "decode 基线 ${AVG}s/token 漂移 ${D}s (>±${BASE_TOL}) — 环境异常"; fi
    else
      bad "decode 基线解析失败 (log 缺 avg 行)"
    fi
  fi
fi

echo "==== A' 连通性回归: FAIL=$FAIL PASS=$PASS ===="
[ "$FAIL" = "0" ] && echo "==> 环境就绪, 允许 A' 下板实现" || echo "==> 存在 FAIL, 修复后重跑; 未通过前不下板 A'"
exit $([ "$FAIL" = "0" ] && echo 0 || echo 1)
