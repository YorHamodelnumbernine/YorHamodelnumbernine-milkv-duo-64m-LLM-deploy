# ION 孤儿进程清理 + 稳定性修复 — 最小改动方案

日期：2026-08-12
作者：推理引擎工程师
状态：host 侧方案，未上板、未改生产代码
背景：设备测试期间模型偶发 RC=-1 崩溃，崩溃后遗留**存活**进程占住 24MB ION pool，
    毒化后续所有运行（`ion ioctl fail:: Out of memory`），必须 kill -9 清掉；
    期间设备还重启过 2 次。

---

## 0. 结论先行（TL;DR）

- **根因**：ION 是 carveout 内存，绑定在进程的 ion fd / dmabuf 上。**只要进程退出
  （含被 SIGKILL），内核关闭 fd → ION 自动释放**（已实测：kill -9 后 24MB 归零）。
  真正的"孤儿"是一个**应该退出但卡在死循环/重试里不退出的存活进程**。
- **最小改动**（推荐，~30 行）：在 `main()` 最早处启动一个**看门狗线程**，正常时每步
  更新心跳；一旦超过 30–60s 无心跳（说明卡死在 CVI runtime / ION 重试 / SD 读），
  `_exit(1)` 强制退出 → fd 关闭 → ION 释放。**不改库、不改主流程。**
- **零代码临时方案**：包装脚本在跑之前清一次 stale 进程 + 全局 `timeout` 兜底。
- **重启根因**：最可能是 28MB DDR + 8–9MB swap 在同一张 SD 上抖动引发的内核不稳；
  DDR=2MB（去 swap）即为最直接的缓解，与已有结论一致。

---

## 1. 孤儿机制（为什么 SIGKILL 能救）

### 1.1 ION 生命周期

- ION carveout heap 是**系统级共享**内存，分配返回 dmabuf fd。
- 释放路径：`CVI_RT_MemFree` → 或**进程退出时内核自动 close fd**。
- 因此：**任何进程死亡（正常 return、SIGSEGV、SIGKILL）都会释放其占用的 ION**。
- 实测佐证：`/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary` 显示
  `used: 26,214,400`（24MB+1MB），kill -9 全部 stale PID 后归零、运行恢复正常。

### 1.2 为什么会留下"活孤儿"

`main()` 的错误路径（L1986/2001/2038/2078/2100/2132/2161）都会 `return 1` → 进程退出
→ ION 释放，这些路径**不会**留孤儿。真正危险的是**卡住不退出的路径**，最可能在三处
CVI runtime 库内部（本仓库源码看不到）：

1. `CVI_RT_Init` / `CVI_RT_MemAlloc` 在 ION 耗尽时的 **"reopen ion dev" 重试循环**。
2. `CVI_RT_Submit` / `CVI_RT_MemInvld` 在 TPU 异常时**等待硬件超时无上限**。
3. mbox 与 secondary core 通信挂死。

这解释了现象：运行"崩溃 RC=-1"后主线程实际进了某个库内死循环，进程活着、fd 没关，
ION 一直占着；新起的进程申请不到 → `Out of memory` → 也卡在重试 → **多个进程互相叠
加，ION 越占越多**（实测发现多个 stale PID）。

---

## 2. 最小改动方案（推荐）：进程内看门狗线程

### 2.1 设计

一个低开销线程 + 一个全局心跳变量：

```c
/* ---- wd.c：看门狗（~30 行） ---- */
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <pthread.h>
static volatile double g_heartbeat = 0;   /* 单调时钟秒数 */
static double wd_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
                             return ts.tv_sec + ts.tv_nsec/1e9; }

static void *wd_thread(void *arg) {
    (void)arg;
    g_heartbeat = wd_now();                 /* 初始化 */
    for (;;) {
        double last = g_heartbeat;
        sleep(30);                          /* 每 30s 查一次 */
        if (g_heartbeat - last < 1e-6) {    /* 30s 内无心跳 → 判定卡死 */
            fprintf(stderr, "[wd] NO HEARTBEAT >30s — force _exit to release ION\n");
            _exit(1);                       /* 关闭 fd → 内核释放 ION */
        }
    }
}
```

接入点（`smollm2_pool_demo.c`，3 处）：

1. `main()` 开头（`tpu_init` **之前**）：
   ```c
   pthread_t wd; pthread_create(&wd, NULL, wd_thread, NULL);
   ```
   这样即使 `CVI_RT_Init`/`CVI_RT_MemAlloc` 在库内重试死循环，看门狗也会在 30s 后
   强制退出。**心跳必须在这些调用**之前**启动。**

2. 心跳更新点（正常流程每个 >0.5s 的重活阶段打一次点）：
   - `sm_forward_pool` 入口（覆盖 prefill chunk 与 decode step）；
   - `pf_wait()` 返回后（每批权重 load ~600ms）；
   - `pool_load_embed_and_init_layers()` 关键里程碑（LM_Head chunk load）。
   ```c
   g_heartbeat = wd_now();
   ```

### 2.2 为什么 `_exit(1)` 而不是 `exit(1)`

- `exit()` 会跑 atexit / flush stdio，可能在卡死的锁上**二次挂起**。
- `_exit()` 直接系统调用退出：**内核关闭所有 fd → ION 释放**，这是唯一目标。
- 不经过 `pool_free()`，但 pool 里 ION 是 dmabuf fd，进程退出即释放，无需手动 free。

### 2.3 阈值与误杀风险

- decode 单步 ~5.7s，prefill chunk ~1–2s，权重批 ~0.6s → **30s 无心跳 = 真实卡死**，
  误杀余量 >5×。
- 若担心个别环境更慢，可把 `sleep(30)` 调成 `sleep(60)`；不改变设计。

### 2.4 兜底：`CVI_RT_MemAlloc` 失败时的诊断

`pool_init`（L1135）失败分支里，打印 ION debugfs 摘要，指导运维清 stale 进程：

```c
if (!p->ion_mem) {
    fprintf(stderr, "  POOL: ION alloc failed — check stale holders:\n");
    system("cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary 2>/dev/null");
    return -1;
}
```

---

## 3. 零代码临时方案（立刻可上板用）

> 不需要改 C，在设备上直接套壳。

### 3.1 `run_clean.sh` — 跑前清 stale + 全局超时兜底

```sh
#!/bin/sh
# run_clean.sh <args...> — 清 stale ION 持有者后跑精确命令
# 用法: sh run_clean.sh /root/smollm2_pool_b2 /root/smollm2_instruct/ /root/input_tokens.bin 3 3 2

# 1) 若 debugfs 可用且 ION 已被占用，则找 stale 进程杀掉
ION_SUM=/sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
if [ -r "$ION_SUM" ] && grep -q "used:" "$ION_SUM"; then
  used=$(grep "used:" "$ION_SUM" | head -1 | tr -cd 0-9)
  if [ -n "$used" ] && [ "$used" -gt 2000000 ] 2>/dev/null; then
    echo "[clean] ION used=${used}B — killing stale smollm2_pool_b2..."
    # 只杀本模型名进程，绝不用 pkill -f 匹配脚本自身
    for p in $(pgrep -x smollm2_pool_b2); do
      echo "  kill -9 $p"; kill -9 "$p" 2>/dev/null
    done
    sleep 1
  fi
fi

# 2) 全局超时兜底：600s 后 SIGTERM，再 15s 后 SIGKILL（保证 fd 关闭）
#    BusyBox 无 timeout 时退回后台 sleep+kill
if command -v timeout >/dev/null 2>&1; then
  exec timeout -k 15 600 "$@"
else
  "$@" & pid=$!
  ( sleep 600; echo "[wd] 600s timeout, killing $pid"; kill -9 $pid ) &
  wait $pid
fi
```

> 注意：脚本绝不能用 `pkill -f` 匹配自身命令行（此前误杀过 SSH 会话）；
> 用 `pgrep -x smollm2_pool_b2` 精确匹配可执行名。

### 3.2 手动清理命令（应急）

```sh
# 查 ION 占用
cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
# 查存活的本模型进程
pgrep -ax smollm2_pool_b2
# 逐 PID 精确 kill（不要 pkill -f）
kill -9 <pid>
# 确认释放
cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary
```

---

## 4. 设备重启 2 次 — 根因排查

### 4.1 已知事实

- 重启期间 dmesg 为空（无法回溯，可能未持久化）。
- 内存压力真实存在：12MB DDR embed → VmSwap 8–9MB，swap 文件在同一张 SD。
- swap 抖动会同时打满 SD 带宽 + 内存，可能触发 hung task / OOM / watchdog。

### 4.2 假设（按可能性排序）

1. **swap 抖动 → 内核 hung task / watchdog 重启**（最可能）：swap 在 SD 上，
   read 权重与 swap in/out 抢同一控制器，极端时某进程长期 D 状态，内核 watchdog 复位。
2. **ION/TPU 驱动卡死 → 硬件 watchdog**：secondary core 或 TPU 忙等无响应。
3. **DDR OOM 误杀关键内核线程**：28MB 下 `vm.overcommit` + 大 malloc 失败链。

### 4.3 缓解（与既有结论一致，无需新工作）

- **DDR embed=2MB**（去 swap）→ 消除最大压力源，同时 decode 更快（5166ms/tok）。
- **INT4 权重**（本期设计）→ 权重读字节减半，SD/swap 压力进一步下降。
- 这两项同时服务 Wt load、上下文与稳定性，是统一解。

### 4.4 下次上板取证清单（等 TPU 工程师释放设备后）

跑精确命令前/中采集，若再崩溃可定位：

```sh
# 启动前基线
free; swapon -s; cat /proc/sys/vm/swappiness
cat /proc/pressure/memory 2>/dev/null || cat /proc/vmstat | grep -E 'pswpin|pswpout'
# 崩溃前兆（在运行中每 10s 轮询）
while true; do echo "$(date +%T) swap=$(grep VmSwap /proc/*/status 2>/dev/null | tr -d ' ' | tail -3 | tr '\n' ' ')"; sleep 10; done
# 重启后（若可）
cat /var/log/messages 2>/dev/null | tail -50
journalctl -k 2>/dev/null | tail -50
```

---

## 5. 验收口径

1. 人为制造一次 decode 卡死（或用 `kill -STOP` 挂起进程），看门狗 **≤30s 强制退出**，
   ION 归零，后续 run 无需手动 kill。
2. 连续跑精确命令 `/root/smollm2_pool_b2 ... 3 3 2` 多次，`/sys/kernel/debug/ion/...`
   不再残留占用；无"ion ioctl fail"。
3. 正常性能无回归（Wt load / decode 前后对比）。
4. 记录数据回报 CEO，不提交 git。
