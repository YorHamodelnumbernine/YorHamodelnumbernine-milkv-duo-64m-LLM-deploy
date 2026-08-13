# FFN-only INT4 快速验证报告（方向 2 判负）

日期：2026-08-13
作者：推理引擎工程师
状态：验证完成 → 方向 2（FFN-only INT4）**不可行**，INT4 整线搁置，维持 INT8 生产路径。

---

## 0. 结论（TL;DR）

| 判定 | 说明 |
|---|---|
| 性能 | ✅ 达标。Wt load 3.86s→2.17s（−44%，优于 CEO 预期 2.4-2.6s）；Total −25%；Prefill −27%；Decode −23% |
| 质量 | ❌ **全部组大小失败**。next_token 恒≠5021（G64=18164、G32=15247、G16=6271），5021 被推出 top-5 |
| **最终结论** | **方向 2 判负不可行**。G64/G32 下 prefill gap 塌缩（0.0/0.9），G16 下 gap 虽健康（6.1）但 argmax 仍错 |

---

## 1. 验证方法

- **host 侧**：`convert_i4 -f`（FFN-only 模式）转换真实 `layer0.bin`：
  - 层总字节 = **2,299,392 B**（−35.1% vs 3,543,552 B）
  - attn 段（0..887,040，含 rms_attn+Wq/Wk/Wv/Wo）与原始 INT8 **逐字节一致** ✓
  - FFN round-trip rms：up=4.213 / gate=4.235 / down=3.822
- **板端**：30 层 `layerN_ffni4.bin` 由交叉编译的 `convert_i4_rv` 在 Duo 上批量重新生成（G64/G32/G16 各一轮，尺寸全部校验）。
- **二进制**：`smollm2_pool_b2_i4`（分支 experiments/int4-weights 构建，板端 md5 7d9fa308 与本地 worktree 重建一致，完全可复现）。
- **命令**：`WT_FFN_I4=1 [FFN_G=N] /root/smollm2_pool_b2_i4 /root/smollm2_instruct/ /root/input_tokens.bin 3 3 2`（同一二进制，WT_FFN_I4 关闭 = INT8 基线对照）。
- **前置**：ION 泄漏（两次 abort 残留 26MB）已通过重启清除，回归前 ION used=0、swap 64MB 已启用。

---

## 2. 数据对比（精确命令，3 轮中位数）

### 2.1 性能

| 配置 | 文件/层 (B) | 字节节省 | Prefill (ms) | Wt load (ms) | Decode (ms/tok) | Total (ms) |
|---|---|---|---|---|---|---|
| INT8 基线 | 3,543,552 | — | 22,909 | 3,857 | 5,085 | 43,248 |
| FFN-only G64 | 2,299,392 | −35.1% | 16,664 | **2,168** (−44%) | 3,910 | **32,304** (−25%) |
| FFN-only G32 | 2,382,336 | −32.8% | 17,071 | 2,272 | 3,938 | 32,824 |
| FFN-only G16 | 2,548,224 | −28.1% | 16,628 | 2,201 | 3,843 | 32,116 |

### 2.2 质量（3 轮全复现，确定性）

| 配置 | next_token | prefill gap | 验收 |
|---|---|---|---|
| INT8 基线 | **5021** ✓ | 5.8 | — |
| FFN-only G64 | 18164 ✗ | **0.0**（top-1/top-2 并列） | FAIL |
| FFN-only G32 | 15247 ✗ | **0.9** | FAIL |
| FFN-only G16 | 6271 ✗ | 6.1（gap 健康但 argmax 已错） | FAIL |

- 三轮 G64：next_token 均为 18164、gap 均 0.0（完全确定）；两轮 G16：均 6271。
- 5021 在所有组大小下都不在 prefill top-5 内。

### 2.3 字节代价（G 越小 scale 开销越大）

| G | FFN i4 段 (B) | 层总 (B) | 节省 |
|---|---|---|---|
| 64 | 1,410,048 | 2,299,392 | −35.1% |
| 32 | 1,492,992 | 2,382,336 | −32.8% |
| 16 | 1,658,880 | 2,548,224 | −28.1% |

---

## 3. 根因分析

1. **量化数学限制**：per-channel INT8 权重已满幅（std 30-38，max≈127），组 INT4 步长 ≈max/7≈18。G 64→16 仅把 FFN round-trip rms 从 4.2 降到 3.2，仍不足以稳定 top-1。
2. **logit 塌缩**：G64/G32 下 prefill top-2 并列（gap 0.0/0.9），top logits 被 INT4 噪声抹平 → winner 对微扰极敏感。G16 虽 gap 恢复 6.1，但整体 logits 偏移使正确 token 5021 掉出 top-5。
3. **与 §7.2 一致性**：历史数据 G64→6271、G16→34346；本次 G64→18164、G32→15247、G16→6271。具体错误 token 随量化噪声图案不同而异，但判定一致——**全部 ≠5021**。非代码 bug，非复现问题。

---

## 4. 建议

1. **方向 2 判负，INT4 整线搁置**。维持 INT8 生产路径（基线 next_token=5021 已验证未受任何影响）。
2. FFN-only 的性能收益（Wt load −44%）记录在案：未来若从**原始 fp32 权重 + calibration** 出发做 INT4（方向 1 前置条件），此收益上限可作为投入参考；当前 per-channel INT8 二次量化的路已被数学封死。
3. 按 CEO 指示：精力转向**稳定性加固 + fip_4096 受控刷写**。
4. 板上状态：ffni4 已恢复 G64 参考集；生产 INT8 路径与 ION（used=0）干净；未提交任何 git。

---

## 5. 附件

- 二进制一致性：`smollm2_pool_b2_i4` md5 7d9fa308fda0146ee9d6c3219a2a1fde（板端 == 本地 worktree 重建）
- 转换校验：`/tmp/layer0_ffni4_local.bin`（本地转换）attn 段与 `/tmp/layer0.bin` 逐字节一致（cmp -n 887040 通过）
