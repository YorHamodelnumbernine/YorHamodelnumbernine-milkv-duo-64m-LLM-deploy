# ION 看门狗增强/收口设计（既有 ec68ed4 的补充，非重造）

日期：2026-08-22 | 作者：推理引擎工程师 | 状态：**离线设计交付物（未上板、未改生产代码）**
定位：对既有 `ec68ed4`（smollm2_pool_demo.c 看门狗 + run_clean.sh）的**收口**——补齐 qwen 引擎看门狗初始化窗口空档 + 两引擎看门狗实现一致性。
关联：`DESIGN_ION_CLEANUP.md`（原设计）/ `run_clean.sh` / `smollm2_pool_demo.c` / `qwen_engine_lmhead2.c`
硬约束沿用：ION 余量 1.85MiB=6.9%；任何新增 ION 消费者必须先回收 centroid（`CENT_ION=0`）或收缩 gsc。

---

## 0. 结论摘要

1. **qwen 引擎看门狗存在初始化窗口空档（本设计核心）**：`wd_start()` 在 `qwen_engine_lmhead2.c` L2403，即**全部 ION 分配/权重加载之后**才启动。而 `CVI_RT_Init`/`MemAlloc` 库内 "reopen ion dev 重试→assert" 死循环、gsc/权重 SD 读 stall、pool_build 挂死——**恰是 DESIGN_ION_CLEANUP 识别的孤儿泄漏根因**——全部落在看门狗覆盖之前。该窗口期挂死 → 进程活、ION 泄漏、毒化后续 run。
2. **smollm2 看门狗实现有两处正确性/安全性短板**：心跳用 `volatile double`（C906 32-bit 下 64-bit 读写可撕裂）；看门狗线程用 `fprintf`（主线程卡在 SDK 持锁时二次死锁）。qwen 引擎版已用 `__atomic` + `write(2)`，正确。
3. **修复方向**：qwen 引擎 `wd_start()` 提前到 `ion_abort_install()` 之后（CVI_RT_Init 之前）+ 初始化里程碑打点；smollm2 心跳改原子 long long + `write(2)`。`run_clean.sh` 已泛化（按 `$BIN` 名匹配），无需改。
4. **ION 回收策略（CEO 硬约束落地）**：看门狗是**防泄漏兜底**，不是回收手段；主动回收顺序 = 回收 centroid（`CENT_ION=0` 回 DDR）→ 收缩 gsc 层数（`GSC_ION_LAYERS`）→ 关 merged pool。设计文档化，代码不新增 ION 消费者。

---

## 1. 现状盘点（两引擎看门狗对照）

| 项 | smollm2_pool_demo.c (ec68ed4) | qwen_engine_lmhead2.c |
|---|---|---|
| 启动位置 | `wd_start()` L2110，**CVI_RT_Init 之前** | `wd_start()` L2403，**ION 分配全部完成后** |
| 初始化覆盖 | tpu_init/pool_init/embed/init 里程碑均打点 | **无任何初始化打点** |
| 主循环覆盖 | forward 入口 / 每权重批 / 每层 fallback / LM_Head chunk | 每层（run_prompt L1860 / run_decode_step L2004） |
| 超时 | `SM_WD_TIMEOUT` env 可调，默认 30s | 固定 `WD_TIMEOUT_NS=90s` |
| 心跳存储 | `volatile double`（**32-bit 撕裂风险**） | `__atomic` long long ns（正确） |
| 看门狗日志 | `fprintf(stderr,...)`（**死锁风险**） | `write(2, ...)` async-signal-safe（正确） |
| 配套脚本 | `run_clean.sh`（按 `$BIN` 泛化匹配） | 无（可复用 run_clean.sh） |

**结论**：qwen 引擎看门狗功能完整（主循环覆盖好、实现正确），唯一实质缺口 = **初始化窗口空档**。smollm2 看门狗启动覆盖好，但实现有撕裂/死锁两处隐患。两引擎各自的最优形态合并即收口。

---

## 2. qwen 引擎看门狗增强（核心改动）

### 2.1 提前启动

`wd_start()` 从 L2403 移到 `ion_abort_install()` 之后、任何 ION 分配之前：

```
main():
  ion_abort_install();   /* L2170, SIGABRT 捕获 */
  wd_start();            /* ← 新增: 看门狗覆盖 CVI_RT_Init/MemAlloc 窗口 */
  ...
  CVI_RT_Init(&rt);      /* L2209 */
  ...
```

理由：
- 看门狗线程只依赖 libc（clock_gettime/pthread），不依赖 CVI，可在 `CVI_RT_Init` 前安全启动。
- 一旦 `CVI_RT_Init`/`rt_alloc_safe` 进入库内重试死循环，90s 后 `_exit(1)` → 内核关 fd → ION 释放。这正是 DESIGN_ION_CLEANUP §2.1 的原始动机。

### 2.2 初始化里程碑打点

新增 `wd_kick()`（防误杀 + 覆盖长 SD 读段）：

| 位置 | 事件 | 预计耗时 |
|---|---|---|
| L2195 后 | `lmhead2_load()`（cluster 文件读） | ~1s |
| L2209 后 | `CVI_RT_Init` | <1s |
| L2211 后 | neuron `rt_alloc_safe` | <1s |
| L2271 后 | `gsc_ion_load()`（24 层 gsc SD 读） | ~9-10s |
| L2279 后 | `lmhead2_cent_place()`（centroid 放置） | <1s |
| L2309 后 | `pool_build()` ×12 | ~1-2s |
| L2339 后 | KV cache malloc + memset | <1s |

> 打点粒度：以上每个里程碑之间均为单一大块 ION 分配/SD 读，最坏间隔 ~10s（gsc load）远小于 90s 超时 → 无误杀风险。主循环（regression/prefill/decode）沿用既有每层打点。

### 2.3 超时参数化（可选）

`WD_TIMEOUT_NS` 固定 90s → 改为 `WD_TIMEOUT_SEC` env 可调，默认 90：

```c
static int wd_timeout_sec(void) {
    const char *e = getenv("WD_TIMEOUT_SEC");
    if (e && atoi(e) >= 10) return atoi(e);
    return 90;
}
```

- 90s 默认对 decode（~11-13s/token，每层打点）与 init（~10s 最长间隔）均有余量。
- 调低（如 30s）可更快回收卡死进程；调高供慢 SD 环境。

### 2.4 改动量预估

`qwen_engine_lmhead2.c`：+3 行（wd_start 提前 + 7 处 wd_kick + 参数化），净增 ~10 行。零风险（wd_kick 是 no-op 当 `g_wd_enabled=0` 或线程未建时）。

---

## 3. smollm2 看门狗收口（正确性/安全性）

### 3.1 心跳原子化（修 32-bit 撕裂）

`static volatile double g_wd_hb` → 对齐 qwen 引擎：`static volatile long long g_wd_hb_ns` + `__atomic_store_n/load_n`：

```c
static void wd_kick(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    __atomic_store_n(&g_wd_hb_ns, (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec,
                     __ATOMIC_RELAXED);
}
```

- C906 是 32-bit 内核态？实为 rv64（riscv64-unknown-linux-musl），64-bit double 在 rv64 上可能撕裂（非原子访存指令）。long long + __atomic 保证 8-byte 原子（rv64 支持 AMO/ld.d）。
- 影响面：仅看门狗心跳，不改主流程；`volatile double` → `long long` 语义等价。

### 3.2 看门狗日志改 async-signal-safe

`fprintf(stderr, ...)` → `write(2, buf, len)`（同 qwen 引擎写法）。避免主线程卡在 SDK 持 stdio 锁时看门狗 `fprintf` 二次死锁（违背看门狗初衷）。

### 3.3 其余

- `SM_WD_TIMEOUT` env 已存在，保留。
- 心跳打点位置已覆盖 init/forward/batch/fallback/LM_Head，无需增删。

---

## 4. run_clean.sh（已泛化，无需改）

`run_clean.sh` 用 `PROC=$(basename "$BIN")` 精确匹配传入二进制名，天然覆盖 `qwen_engine_*` / `qwen_lh2_landing` / `smollm2_pool_b2`。仅需在**使用规范**上补充：
- 上板跑 qwen 引擎一律 `sh run_clean.sh ./qwen_engine_int8kv ...` 或先 `sh run_clean.sh --clean <bin>`。
- 保留 "绝不用 pkill -f" 纪律（防误杀 SSH/脚本自身）。

---

## 5. ION 回收策略（CEO 硬约束 1.85MiB 落地文档）

看门狗定位 = **防孤儿泄漏的兜底**（进程退出 → fd 关闭 → ION 释放），**不主动回收**。主动回收优先级（按侵入性升序）：

1. **回收 centroid**：`CENT_ION=0` 强制 centroid 回 DDR fp32+mlock（释放 1.84MB ION；stage1 +15ms 代价）。这是"任何新增 ION 消费者必须先回收 centroid"的标准动作。
2. **收缩 gsc**：`GSC_ION_LAYERS` 22→更少（DDR 层用 mmap page-cache，可回收），每层释放 0.89MB ION。
3. **关 merged pool**：`MERGE=0`（释放 ~0.26MB ION；carveout 预检已有自动回退逻辑）。

约束：**不新增 ION 消费者**。任何未来需要 ION 的新能力（如更大的 gsc 缓存 / B-2 槽扩容）必须先执行 1-2 步回收，否则触发 carveout 预检失败（`free_ion < need` → 干净退出，不崩溃）。

---

## 6. 验收口径（上板后）

1. **qwen 引擎**：人为 `kill -STOP <pid>` 在 init 段（如 gsc load 中）→ 看门狗 ≤90s 强制 `_exit(1)`，ION 归零；decode 段同样验证（既有每层打点已覆盖）。
2. **正常路径零误杀**：完整 CHAT 75-token + 5 步 decode 跑完，无 `[WATCHDOG]` 触发；NEXT/bit-exact 与基线一致。
3. **smollm2**：`SM_HANG_TEST`（既有）复跑 → 看门狗触发、ION 归零；正常回归无变化。
4. `sh run_clean.sh --clean <bin>` 在 ION 有残留时正确清理。
5. 记录数据回报 CEO；改动待 CEO 门禁指示后合入（当前 qwen_engine_lmhead2.c 冻结中，不做生产 commit）。

---

## 7. 落地顺序（建议）

1. 先合 qwen 引擎看门狗提前 + 里程碑打点（独立小 diff，风险最低）。
2. 再收 smollm2 原子心跳 + write(2)（独立小 diff）。
3. 与 INT4 设计A 互不依赖，可并行。
