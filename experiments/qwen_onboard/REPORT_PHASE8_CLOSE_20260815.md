# Phase 8 正式收口报告 — Qwen2.5-0.5B decode on CV1800B（TPU 底层视角）

日期：2026-08-15 | 作者：TPU 底层工程师（bmk1822 / CV1800B）
关联：`REPORT_PHASE7_SIGNOFF_20260814.md` / `REPORT_APRIME_INTEGRATION_REVIEW_20260815.md` / `PLAN_APRIME_MEMBUDGET_20260815.md`
结论：**Phase 8 关闭。B-2（per-matrix ion_db 预取）判定 FAIL（>10s 验收线），接受并锁定出货配置。**

---

## 1. 执行摘要

B-2 目标（decode ≤9.5s PASS / ≤10s SOFT-PASS）**未达成**：decode avg 11.29~11.42s，与出货基线
11.18~11.29s **持平（+0.11~+0.24s，噪声内，无回退）**。根因是硬件物理墙，非实现缺陷：
- SD 顺序读天花板 ~20.2MiB/s（实测 19.86–20.57，9 种读法收敛）→ nib-only 地板 8.85s/step
- LM head 0.6s → 理论地板 9.45s = PASS 线零容差
- ION carveout 28.1MB 硬顶 → SD 槽深最多 4 大+1 小，结构气泡（sd_wait ~3.7s + noslot ~1.6s）不可消除
- 7 层 DDR gsc 页错误（accum 2.44s/step）为硬约束：ION 无余量、anon malloc 已实测 crash

**结论：硬件现状下 <10s 不可达，为最终性能结论。** 出货配置锁定 `LW_READ=mmap GSC_ION=1`。

---

## 2. 全链路性能演进（26s → 11.29s，2.3×）

| 阶段 | 杠杆 | decode | 累计 | 备注 |
|---|---|---|---|---|
| Phase 6 首落地 | 裸 mmap demand-paging | **~26.0 s** | — | t_layers≈25.3s，201.4MB/token @7.96MB/s |
| 7b 前 | mmap 读路径修正 | **19.97 s** | -6.0s（-23%） | 生产口径 VERIFY=0 基线 |
| 7c | rsafe 离线预标定（RSH=1 查表） | **13.99 s** | -5.98s（-30%） | 去 wmax 预扫双读 |
| 7d | TIU 提交合并（up/gate 单 cmdbuf） | **13.02 s** | -0.97s（-6.9%） | 8×608 合并 |
| 7e | gsc 全 24 层常驻 ION 缓存 | **11.29 s** | -1.73s（-13.3%） | **出货基线（解码锚点）** |
| Phase 8 | B-2 per-matrix ion_db 预取 | **11.29~11.42 s** | 0（持平） | **FAIL**，物理墙 |

正确性全程零退化：prefill NEXT 3/3、decode NEXT 一致、bad1=bad2=r_opt=rsh=0。

---

## 3. B-2 最终 A/B（本报告实测，DECODE_STEPS=6，冷页缓存公平对比）

| 配置 | decode avg/token | t_layers | 判定 |
|---|---|---|---|
| B 基线 `LW_READ=mmap GSC_ION=1` | 11.18s | — | 锚点 11.29s 复测 |
| A 性能 `LW_READ=ion_db` | 11.29s | 10.65~10.75s | — |
| A 复跑 `LW_READ=ion_db` | 11.42s | — | 差 0.13s（±0.3s 内，稳定） |
| A 正确性 `VERIFY=1` | 通过 | — | 19.18s 为逐位校验开销，非性能 |

**验收线判定：>10s → FAIL。** A/B 持平，无回退。

### 3.1 回归清单（4 档全过）
- prefill P1/P2 NEXT 3/3：2130 / 12095 / 99366 ✓
- decode NEXT：2130 ✓
- bit-exact：bad1=0 bad2=0 r_opt=0 rsh=0 ✓
- ION peak 28,073,984 / 28,102,656（96%，余 28KB，未爆）✓ 运行后 clean usage=0 ✓
- swap-on 强制 ✓；不碰副核 SDHCI，无 kernel lockup ✓；回滚路径零重建 ✓

### 3.2 最终配置（B-2 形态）
SD_NSLOT=5（4×2.18MB 大槽 + 1×0.40MB 小槽，全 ION）、队列 16、lookahead 10、
bounce 2MiB（O_DIRECT→普通对齐 bounce→memcpy 入 ION）、矩阵 fd 复用（168→24 opens/step）、
gsc 17 层 ION + 7 层 DDR mmap、rms 全层 DDR 缓存、prefill pools 回收。

### 3.3 末轮优化实测（全部 wash，接受）
| 优化 | 期望 | 实测 |
|---|---|---|
| fd 复用 168→24 opens/step | 减 open 开销 | 微，t_layers 不变 |
| bounce 1MiB→2MiB | 减 pread 次数 | 微 |
| DDR gsc 预热（层切换时 madvise+touch） | 消 accum 2.42→~1.57s | **无效**：accum 2.44s 不变 |
| anon gsc 常驻（替代 mmap） | 消 DDR gsc 页错误 | **否决**：实测 crash（RC=139，已留痕注释） |

---

## 4. 硬件天花板分析（最终）

- **SD 物理上限**：~20.2MiB/s（HS-4bit，9 种读法全收敛）→ nib 179.17MB/token = 8.85s/step
- **ION 28.1MB 硬顶**：限制 SD 槽深 ≤5（4 大+1 小），无法完全隐藏 SD 延迟；
  4-big + gsc 19 ION 实测超 carveout ~1.8MB，不可行
- **DDR 安全余量**：~5-6MB → gsc 最多 5-6 层安全；7 层即 thrash（accum 2.44s）
- **理论地板 = 8.85s（SD）+ 0.6s（LM head）= 9.45s，PASS 线零容差**；气泡叠加 → 11.3s

<10s 需更快存储介质（SD>20MB/s）或更大 ION/RAM，**当前硬件不可达**。

---

## 5. 出货配置与回滚

- **出货配置**：`LW_READ=mmap GSC_ION=1` → **11.18~11.29s/token**，bit-exact 全绿，一键回滚不动。
- **回滚**：引擎默认即 mmap 路径（ion_db 由 LW_READ 环境变量门控），零重建、零部署即可回退。

---

## 6. 死锁卫生修复（本报告已应用）

推理引擎评审发现的 `ion_prefetch_wait` 死锁（ion_db 实验路径）：预读线程 pread 报错时矩阵槽
置空、队列请求已消费，`g_pushed_max` 不再重推该矩阵 → `layer_io_mat` 永久等 `g_ion_done_cv`。

**已应用最小补丁**（`qwen_engine_lmhead2.c`）：
1. 新增 per-matrix 失败标记 `g_pf_fail[rem]`；预读线程 perr 时置位，成功时清零
2. `layer_io_mat` 等待循环检测到 `g_pf_fail[rem]` 即 break，主线程转 `ion_mat_sync` 同步补载
   （补载失败内部 `exit(2)` 干净退出，不再挂死）
3. 补载清标记（只补一次）

**验证**：补丁后 3 步 smoke——decode avg 11.30s（无回退）、bit-exact 0/0/0/0、ION peak 不变。

---

## 7. 板上状态与归档

- 板上已 `run_clean`（ION usage=0，无 stale holder），二进制为含死锁补丁的当前版本
- 文档归档：本报告 + `REPORT_APRIME_INTEGRATION_REVIEW_20260815.md`（集成侧最终版）+ 预算/设计文档
- 遗留杠杆（CEO 判定默认不做）：gsc 压缩腾 ION 加深槽，理论 ~1-1.5s，收益不确定、有精度风险、需多日

---

## 8. 签核

**Phase 8 关闭。** 出货配置 11.29s/token 为最终交付。TPU 底层视角确认无遗留阻塞。
