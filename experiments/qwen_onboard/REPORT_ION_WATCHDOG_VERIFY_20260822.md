# ION 看门狗验证通过报告（2026-08-22）

验证对象：已提交的 ION 看门狗增强（`7c8b44d` qwen init 窗口覆盖 + `bd5be99` smollm2 原子心跳/write(2)），
从 `git archive HEAD`（f0e5631）干净构建，**不包含**工作区未提交的 INT4 A2 dequant 改动。
板卡状态：验证前 ION 0%、无 qwen/smollm 进程；验证过程中无并发进程（TPU 侧已暂停）。

## 结果汇总

| 项 | 命令/环境 | 判定 | 证据 |
|---|---|---|---|
| Test 1: qwen init 窗口挂死 | `QH_INIT_HANG=1 WD_TIMEOUT_SEC=10` | **PASS** | rc=1, elapsed=10s, ION 0% |
| Test 2: 正常路径零误杀 (decode) | `DECODE=1 DECODE_STEPS=3 VERIFY=0` (detached) | **PASS** | rc=0, 0×WATCHDOG, decode bit-exact, ION 0% |
| Test 3: smollm2 hang | `SM_HANG_TEST=1 SM_WD_TIMEOUT=5` | **PASS** | rc=1, elapsed=6s, ION 0% |
| 3-prompt 正常回归 | 无 hang env | **PASS** | expected_next 3/3 OK, TIU bit-exact, 0×WATCHDOG, rc=0, ION 0% |

## 详细证据

### Test 1 — qwen init 窗口挂死（QH_INIT_HANG=1 WD_TIMEOUT_SEC=10）
- 看门狗在 init 窗口内（wd_start 之后、CVI_RT_Init 之前）强制 `_exit(1)`。
- 日志：`[WATCHDOG] QH_INIT_HANG set - spinning main in init window` → `[WATCHDOG] heartbeat timeout, force _exit(1)`。
- rc=1（区别于 timeout SIGKILL 的 137），elapsed=10s，结束后 ION usage rate 0%。

### Test 2 — 正常路径零误杀（DECODE=1 DECODE_STEPS=3 VERIFY=0，detached PID 535）
- 正常 decode 路径跑完 3 步，0 次 `[WATCHDOG]` 触发（零误杀）。
- 内置 3-prompt 回归段：expected_next 3/3 OK；decode 段 bit-exact：bad1=0 bad2=0 r_opt=0 rsh=0。
- 结束后 ION usage rate 0%。

### Test 3 — smollm2 hang（SM_HANG_TEST=1 SM_WD_TIMEOUT=5）
- 看门狗 5s 超时判定无心跳，强制 `_exit(1)`。
- 日志：`[wd] SM_HANG_TEST set` → `[wd] NO HEARTBEAT >5s - force _exit`。
- rc=1，elapsed=6s，结束后 ION usage rate 0%。

### 3-prompt 正常回归（零误杀）
- expected_next 3/3 OK（2130/12095/99366），TIU internal BIT-EXACT。
- `[WATCHDOG]` 出现次数 0，rc=0，total wall=159.52s，结束后 ION 0%。

## 结论

看门狗行为符合设计：
1. **挂死兜底**：qwen init 窗口 / smollm2 主循环挂死均 ≤ 超时强制退出，ION 归零（内核关 fd 释放），无孤儿泄漏。
2. **正常路径零误杀**：decode / 3-prompt 回归全程 0 次误触发。
3. **实现一致性**：smollm2 心跳原子化（long long __atomic）+ write(2) 已与 qwen 引擎对齐。

配套脚本：`wd_verify_run.sh`（板上 drop-in 三项）/ `wd_verify_deploy_run_host.sh`（host 干净构建+部署+回拉）。
设计文档：`DESIGN_ION_WATCHDOG_ENHANCE_20260822.md`。
