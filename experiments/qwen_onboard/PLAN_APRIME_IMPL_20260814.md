# A' 落地实施方案 + 测试计划 — ION 双缓冲（G-ION Gate 收口）

日期：2026-08-14 | 作者：TPU 底层工程师 | 状态：**host 侧就绪，不下板**（等待 Duo 物理断电重启后连通性回归通过）
关联：`REPORT_GATE_ION_20260814.md`（G-ION-1/2 实测）、`REPORT_ION_DOUBLEBUF_FEASIBILITY_20260814.md`（布局/收益）、`REPORT_PHASE7_SIGNOFF_20260814.md`（11.29s 基线）、`qwen_engine_lmhead2.c`（实施载体）

---

## 0. TL;DR

1. **A' 落地为引擎新增 `LW_READ=ion_db` 模式**：背景预读线程把层 l+1 整文件
   （nib+gsc+rms，8.39MB）用 **buffered pread → 对齐 bounce → CPU memcpy** 装入 ION
   `SD_BUF_A/B`，主线程算层 l 时 SD 读与 TIU 重叠。O_DIRECT 分支**在引擎中不存在**
   （仅存于 gate 探针 `gate_ion2_odirect.c`，已判死），因此没有"替换点"，而是新增模式。
2. **ION 布局关键修正（本方案新增结论）**：可行性报告 §1.2 新预算表漏算了现有
   **21.33MiB gsc ION 缓存**。A' 必须**释放 gsc 缓存**——因为整层文件已含 gsc，
   `parse_layer` 指针直接指向 SD_BUF 即可，`gsc_ion_apply()` 在 A' 下关闭。
   释放后预算：**SD_BUF_A/B 2×8.4MiB + neuron 0.25MiB + pools ~1.2MiB ≈ 18.5MiB
   < 24MiB carveout，余 ≥5.5MiB**（若保留 gsc 缓存则 39.6MiB，装不下）。
3. **改动面收敛在 1 个文件**（`qwen_engine_lmhead2.c`）：新宏/新全局/预读线程/
   `layer_io_begin/end` 加分支/main() 初始化/gsc 缓存旁路。~250 行增量，默认行为不变
   （LW_READ 未显式给 `ion_db` 时仍走 mmap 基线）。
4. **bit-exact 由构造保持**：SD_BUF 字节 = 原层文件字节（同源），数值/TIU/LM head 路径
   全部不变；回归仅需确认 bad1=bad2=r_opt=rsh=0、NEXT 3/3。
5. **连通性回归脚本**：`aprime_conn_check.sh`（ping/SSH/SD 速率/ION 状态 + 11.29s 基线
   decode A/B），重启后一键执行。
6. **验收线**：decode A' < 10s 全档、中位数 ~8s、验收线 9.5s（CEO 立项口径）。

---

## 1. 现状核对（host 静态验证）

### 1.1 decode 基线 11.29s/token 分解（decode_e1_v0.log，GSC_ION=1）

| 分量 | s/token | 性质 |
|---|---|---|
| dequant_rvv | 6.12 | ~纯 SD nib 读等待（RVV 数学仅 ~0.02s） |
| runcmdbuf(TIU) | 2.82 | 纯计算 |
| accum(CPU fp32) | 0.54 | 纯计算 |
| blockmax / flush+invld+copy_act / t_head / gap | 0.18/0.43/0.50/0.11 | 其余 |
| **合计** | **~10.7** | ≈11.29 实测 |

- 计算地板（非 SD）≈ 4.1s/layer 流程 + 0.5s head ≈ **4.6s/token**。
- SD nib 读 170.6MiB：现引擎 mmap+readahead ≈ 27.9MiB/s；SD 顺序物理上限 ~20.5–21.5MiB/s。
- A' 双缓冲后 per-token ≈ **max(SD 读, 计算地板) + 气泡** ≈ 6.9–9.1s（G-ION 报告 §5 三档）。

### 1.2 当前 ION 占用（Phase 7e 口径，carveout 实测 ~25MiB=26,214,400B）

| 区域 | 大小 | 说明 |
|---|---|---|
| gsc ION 缓存 | 21.33 MiB | 24 层 × 931,840B，启动时 pread 常驻 |
| pools cmdbuf | ~1.2 MiB | 10 pool + merged pool |
| neuron（DQ/ACT/P1/P2） | 0.25 MiB | NEURON_SZ=262,144B |
| **合计** | **~22.8 MiB** | 余 ~2.2 MiB |

### 1.3 层文件字节构成（parse_layer 逐字段求和，静态复核通过）

| 分量 | B/layer | MiB/layer | ×24 |
|---|---|---|---|
| INT4 nib（K-aligned G32） | 7,454,720 | 7.11 | 170.6 MiB |
| fp16 gsc（per-group scale） | 931,840 | 0.89 | 21.3 MiB |
| rms 表 | 7,168 | 0.007 | 0.17 MiB |
| **合计（layerN_kal.bin）** | **8,393,728** | 8.005 | 192.1 MiB |

---

## 2. A' ION 布局（2×8.4MiB 双缓冲槽位映射）

### 2.1 新预算（释放 gsc 缓存后）

| 区域 | 大小 | 角色 | 来源 |
|---|---|---|---|
| SD_BUF_A | 8,396,800 B（8.01 MiB） | 层 l（nib+gsc+rms 全量） | 预读线程 pread+memcpy 写入 |
| SD_BUF_B | 8,396,800 B | 层 l+1（预读） | 同上 |
| neuron（DQ/ACT/P1/P2） | 262,144 B | 现有单缓冲，**不改** | main 分配 |
| pools | ~1.2 MiB | 现有 cmdbuf | main 分配 |
| KV/INT8 预留 | 1.0 MiB（可选） | 后续 KV 扩展 | 预留 |
| **合计** | **~18.5 MiB** | | **< 24 MiB，余 ≥5.5 MiB** |

- `SD_BUF_SZ = roundup(lsz=8,393,728, 4096) = 8,396,800`（可行性报告写 8,397,312，
  page-align 后 8,396,800 亦可；运行时用 `lsz` 复核 `lsz <= SD_BUF_SZ`）。
- **gsc 缓存 21.33MiB 释放**：A' 下 `gsc_ion_load()` 跳过，`g_gsc_ion=NULL` →
  `gsc_ion_apply()` 自动 no-op（其首行 `if (!g_gsc_ion) return;`）。gsc 字节随层文件进
  SD_BUF，`parse_layer` 指针指向 SD_BUF，与现文件字节一致。

### 2.2 槽位映射（slot = l & 1）

```
层 0:  slot0 ──compute(0)──▶ 同时 issue prefetch(1→slot1)
层 1:  等 done[1] ──compute(1)──▶ issue prefetch(2→slot0)
层 2:  等 done[0] ──compute(2)──▶ issue prefetch(3→slot1)
...
```
- warmup：main 同步装载层 0 → slot0（`ion_prefetch_sync(0,0)`），每步/每 prompt 首个
  层若 slot 里不是层 0 则同步补载（~0.4s，属预算内流水气泡）。
- 预读线程写 `SD_BUF` 后无需 DMA 一致性维护：**SD_BUF 仅被 CPU 消费**
  （dequant 读 nib、accum 读 gsc，均为 CPU 读 ION VA，ION 是 CPU-cacheable）；
  TIU 只读 DQ（走既有 `CVI_RT_MemFlushEx`），不读 SD_BUF。
- 跨层缓存一致性：写线程（预读）与读线程（主）同核域，pthread mutex/cond 提供屏障。

### 2.3 内存（DDR）红线

- SD_BUF_A/B 在 ION carveout（独立物理池），**不占 28MB DDR**。
- bounce 仅 **1 MiB**（复用，`SD_BOUNCE_SZ=1048576`，4096 对齐），非 8.4MB 匿名
  → 不引入 Phase 7 pread 模式的 swap 抖动风险。VmSwap 应维持 ~2.4–2.85MB 稳态。
- 每层 memcpy 8.39MB @ ~753MB/s ≈ **11ms**，可与读重叠，可忽略。

---

## 3. 精确改动点（`qwen_engine_lmhead2.c`，当前文件行号）

| # | 位置（行） | 改动 | 内容 |
|---|---|---|---|
| 1 | L52-56 后 | 新宏 | `#define LW_ION_DB 5`、`#define SD_BUF_SZ 8396800`、`#define SD_BOUNCE_SZ 1048576`、`#define SD_NSLOT 2` |
| 2 | L358-363 后 | 新全局 | `g_sd_ion[2] / g_sd_va[2] / g_sd_slot_layer[2] / g_ion_* 同步原语 / g_bounce / lsz_global / g_t_sd_wait / g_pf_err,g_pf_n,g_t_sd,g_t_memcpy` |
| 3 | L753-789 后 | 新函数 | `ion_prefetch_sync(l,slot)` / `ion_prefetch_thread()` / `ion_prefetch_issue(l,slot)`（骨架见 §4） |
| 4 | L791-868 | 改函数 | `layer_io_begin()` 加 `LW_ION_DB` 分支；`layer_io_end()` 的 munmap 排除 `LW_ION_DB` |
| 5 | L1307-1310 | 加打印 | `run_decode_step` 摘要加 A' 指标行（prefetch_n/sd/memcpy/sd_wait/err） |
| 6 | L1405-1413 | 改条件 | gsc 缓存块：`LW_READ=ion_db` 时跳过 `gsc_ion_load()`（打印 "skip gsc cache"） |
| 7 | L1447-1453 | 加解析 | `LW_READ` env 增加 `ion_db` → `g_lw_mode = LW_ION_DB` |
| 8 | L1484-1500 | 加初始化 | A' 下：`aligned_alloc(4096, 1MiB)` bounce；`CVI_RT_MemAlloc` SD_BUF_A/B；
  `pthread_create(ion_prefetch_thread)`；`ion_prefetch_sync(0,0)` warmup；打印 ION debugfs 摘要（验证布局余量） |
| 9 | main 收尾 | 加清理 | `g_ion_shutdown=1` + broadcast + join 预读线程；`CVI_RT_MemFree` SD_BUF_A/B |

**不改**：`parse_layer`、`dequant_kal_rvv`、`eng_matmul`、pool 构建、LM head、KV cache。
A' 只换"层字节从哪来"（mmap 区域 → ION SD_BUF），其余路径字节级不变。

---

## 4. 代码骨架（可编译参考）

```c
/* ============ 改动 1：新宏（L56 后） ============ */
#define LW_ION_DB 5
#define SD_NSLOT    2
#define SD_BUF_SZ   8396800   /* roundup(layer 8393728B, 4096) */
#define SD_BOUNCE_SZ 1048576  /* 1MiB 对齐 bounce（复用，非 8.4MB 匿名） */

/* ============ 改动 2：新全局（L363 后） ============ */
static CVI_RT_MEM g_sd_ion[SD_NSLOT] = {NULL, NULL};
static uint8_t   *g_sd_va[SD_NSLOT]  = {NULL, NULL};
static volatile int g_sd_slot_layer[SD_NSLOT] = {-1, -1}; /* slot->已装载层 */
static pthread_mutex_t g_ion_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_ion_cv  = PTHREAD_COND_INITIALIZER;
static int g_ion_req_slot = -1, g_ion_req_layer = -1;   /* 预读请求 */
static int g_ion_done[SD_NSLOT] = {0, 0};               /* slot 就绪 */
static int g_ion_shutdown = 0;
static uint8_t *g_bounce = NULL;
static size_t g_bounce_sz = 0, lsz_global = 0;
static int   g_pf_n = 0, g_pf_err = 0;
static double g_t_sd = 0, g_t_memcpy = 0, g_t_sd_wait = 0;

/* ============ 改动 3：预读线程（L789 后） ============ */
static void ion_prefetch_sync(int l, int slot) {
    char path[160]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[ion_db] open %s\n", path); exit(2); }
    long remain = (long)lsz_global, off = 0; uint8_t *dst = g_sd_va[slot];
    while (remain > 0) {
        size_t n = remain < (long)g_bounce_sz ? (size_t)remain : g_bounce_sz;
        if (pread(fd, g_bounce, n, off) != (ssize_t)n) { fprintf(stderr, "[ion_db] short pread\n"); exit(2); }
        memcpy(dst + off, g_bounce, n);
        off += n; remain -= n;
    }
    close(fd); g_sd_slot_layer[slot] = l;
}

static void *ion_prefetch_thread(void *arg) {
    (void)arg;
    for (;;) {
        int slot, l;
        pthread_mutex_lock(&g_ion_mtx);
        while (!g_ion_shutdown && g_ion_req_layer < 0) pthread_cond_wait(&g_ion_cv, &g_ion_mtx);
        if (g_ion_shutdown) { pthread_mutex_unlock(&g_ion_mtx); return NULL; }
        slot = g_ion_req_slot; l = g_ion_req_layer; g_ion_req_layer = -1;
        pthread_mutex_unlock(&g_ion_mtx);

        double t0 = now();
        char path[160]; snprintf(path, sizeof path, "%s/layer%d_kal.bin", WDIR, l);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { g_pf_err++; g_sd_slot_layer[slot] = -1; goto done; }
        long remain = (long)lsz_global, off = 0; uint8_t *dst = g_sd_va[slot];
        while (remain > 0) {
            size_t n = remain < (long)g_bounce_sz ? (size_t)remain : g_bounce_sz;
            if (pread(fd, g_bounce, n, off) != (ssize_t)n) { g_pf_err++; break; }
            memcpy(dst + off, g_bounce, n);
            off += n; remain -= n;
        }
        close(fd);
        g_sd_slot_layer[slot] = l; g_pf_n++;
        g_t_sd += now() - t0;              /* 读+拷贝总耗时（线程侧统计） */
    done:
        pthread_mutex_lock(&g_ion_mtx);
        g_ion_done[slot] = 1;
        pthread_cond_signal(&g_ion_cv);
        pthread_mutex_unlock(&g_ion_mtx);
    }
}

static void ion_prefetch_issue(int l, int slot) {
    pthread_mutex_lock(&g_ion_mtx);
    g_ion_req_slot = slot; g_ion_req_layer = l; g_ion_done[slot] = 0;
    pthread_cond_signal(&g_ion_cv);
    pthread_mutex_unlock(&g_ion_mtx);
}

/* ============ 改动 4：layer_io_begin 新分支（L812 LW_PREAD 后） ============ */
    if (g_lw_mode == LW_ION_DB) {
        int slot = l & 1;
        if (g_sd_slot_layer[slot] != l) {           /* 首个层/跨步冷槽：同步补载 */
            ion_prefetch_sync(l, slot);
        } else if (l > 0) {                          /* 等待预读完成（流水线气泡） */
            double tw = now();
            pthread_mutex_lock(&g_ion_mtx);
            while (!g_ion_done[slot] || g_sd_slot_layer[slot] != l)
                pthread_cond_wait(&g_ion_cv, &g_ion_mtx);
            pthread_mutex_unlock(&g_ion_mtx);
            g_t_sd_wait += now() - tw;
        }
        if (l + 1 < L) ion_prefetch_issue(l + 1, slot ^ 1);
        io->mode = LW_ION_DB; io->src = g_sd_va[slot]; io->lsz = lsz; return 0;
    }
/* layer_io_end：把 munmap 排除条件改为 mode != LW_PREAD && mode != LW_ION_DB */

/* ============ 改动 8：main() A' 初始化（L1500 LW_MMAP_TH 块后） ============ */
    if (g_lw_mode == LW_ION_DB) {
        g_bounce_sz = SD_BOUNCE_SZ;
        if (posix_memalign((void **)&g_bounce, 4096, g_bounce_sz)) { fprintf(stderr, "oom bounce\n"); return 2; }
        for (int s = 0; s < SD_NSLOT; s++) {
            g_sd_ion[s] = CVI_RT_MemAlloc(rt, SD_BUF_SZ);
            if (!g_sd_ion[s]) { fprintf(stderr, "A' SD_BUF%d ION alloc %d B FAILED\n", s, SD_BUF_SZ); return 2; }
            g_sd_va[s] = CVI_RT_MemGetVAddr(g_sd_ion[s]);
        }
        lsz_global = lsz;
        if (pthread_create(&g_pf_thread, NULL, ion_prefetch_thread, NULL) != 0) { fprintf(stderr, "ion pf thread\n"); return 2; }
        ion_prefetch_sync(0, 0);   /* warmup layer0 -> slot0 */
        printf("  [LW_ION_DB] SD_BUF_A/B=%d B x2 in ION, bounce=%zu B, warmup(0) done\n", SD_BUF_SZ, g_bounce_sz);
        system("cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/summary 2>/dev/null | head -8");
    }
/* main 收尾：shutdown + join + MemFree g_sd_ion[0/1] */
```

---

## 5. 测试计划

### 5.1 前置：连通性回归（重启后第一步）

`aprime_conn_check.sh`（本目录，host 侧一键）：
1. **RNDIS**：`ping -c 3 -W 2 192.168.42.1`。
2. **SSH**：`duo_ssh.py "uname -a; free -m | head -2; uptime"`。
3. **SD 速率**：板上 `sync; echo 3 > /proc/sys/vm/drop_caches; time cat /data/qwen/layer*_kal.bin > /dev/null`
   → 速率 = 192.1MiB / real_s，期望 **≥ 20MiB/s**（物理上限 ~21.5）。
4. **ION 状态**：debugfs carveout summary（used/remaining），确认无 stale 占用、余量正常。
5. **decode 基线（B）**：若板上已有 `qwen_engine_lmhead2_phase7e`，跑
   `GSC_ION=1 VERIFY=1 RSH=1 DECODE=1 DECODE_STEPS=6 PROFILE=1`，解析
   `==== decode avg per-token = X.XXs`，与 **11.29s ± 0.5** 比对 → PASS/FAIL。
6. 输出 PASS/FAIL 汇总表；任一 FAIL 即停，修复后再进 A'。

### 5.2 decode A/B 对比（A' 实现后）

| 档 | 配置 | 期望 | 判据 |
|---|---|---|---|
| **B（基线）** | `LW_READ=mmap GSC_ION=1 VERIFY=0` | 11.29s/token | 复现基线（drop_caches 冷启动） |
| **A（A'）** | `LW_READ=ion_db VERIFY=0` | **6.9–9.1s/token** | 全部档 <10s；中位数 ~8s；验收线 **9.5s** |
| **A 正确性** | `LW_READ=ion_db VERIFY=1` | NEXT 3/3 + bit-exact | bad1=bad2=r_opt=rsh=0 |

A/B 编排仿 `phase7e_run.sh`：每档前 drop_caches；`run_clean.sh --clean` 清 ION；
900s/档 timeout；`phase7_analyze_logs.py` 复用解析。

### 5.3 回归项清单（A' VERIFY=1 run 输出核验）

- [ ] 3-prompt NEXT 3/3 = 2130 / 12095 / 99366
- [ ] TIU bit-exact：bad1=bad2=0
- [ ] r_opt mismatch = 0；rsafe 表 vs 运行时扫描 rsh = 0
- [ ] decode 每步 min_gap ≥ 0.05
- [ ] VmSwap 稳态 ~2.4–2.85MB（无 swap 抖动）
- [ ] A' 指标行：`prefetch_n=24×steps`、`sd_wait` 小（≈memcpy ~11ms 级，说明预读追上了）、`err=0`
- [ ] ION debugfs：SD_BUF_A/B 各 8,396,800B，剩余 ≥ 4MB，无 "ion ioctl fail"
- [ ] prefill 3 prompts 时序无回归（per_token 与基线相当）
- [ ] 连续跑 2 次 decode A' 结果稳定（±0.3s 内）

### 5.4 验收口径（CEO 立项）

1. **decode A' 中位数 ~8s，全部档 <10s，验收线 9.5s**（相对 11.29s 基线 -16~29%）。
2. bit-exact / NEXT 3/3 / 红线全过。
3. ION 无泄漏：跑完后 debugfs used 归零，连续 run 无需 kill stale。

---

## 6. 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| buffered pread 直写 ION VA 可能可行（省 memcpy） | 低（正向） | 预留 A'' 探针（30min）：若 buffered pread→ION 直接工作，去掉 bounce memcpy；不影响 A' 主体 |
| 单核 C906B：预读线程与主线程争 CPU | 低 | pread 阻塞在内核（CPU 占用近零）；memcpy ~11ms/层，可忽略；必要时 `sched_yield` 让主线程优先 |
| SD_BUF ION 分配失败（布局余量不足） | 低 | 已释放 gsc 缓存（21.33MiB），预算余 ≥5.5MiB；启动打印 debugfs 复核 |
| 跨步冷槽 sync 补载 0.4s/步 | 低 | 已计入预算气泡（~0.3s）；二期可做跨步 circular prefetch 消除 |
| swap 抖动（bounce 1MiB 匿名） | 低 | bounce 仅 1MiB，远小于 pread 8.4MB 匿名；VmSwap 回归项监控 |
| 预读线程 buf 一致性（写→读） | 低 | 同核 pthread 同步；SD_BUF 无设备访问，无需 flush/invld |

---

## 7. 交付物与回报

- 本文件：`PLAN_APRIME_IMPL_20260814.md`（实施 + 测试计划）
- `aprime_conn_check.sh`（连通性回归 + 基线，重启后直接执行）
- 回报 CEO：方案就绪；等连通性恢复通知后：跑连通性回归 → A' 下板实现 → A/B → 回归 → 数据回报。
