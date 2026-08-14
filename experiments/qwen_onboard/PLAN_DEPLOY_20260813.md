# ③ 324MB 权重部署方案（Qwen Path A → Duo SD）

日期：2026-08-13 | 作者：推理引擎工程师 | 状态：**方案定稿，待执行**
关联：M2（上板 3-prompt 回归）的前置依赖；weights_kal/（K-aligned INT4 G32，339.53MB，54 文件）

---

## 0. 目标与结论

把 `experiments/qwen_onboard/weights_kal/`（339.53 MB）部署到 Duo 持久存储 `/data/qwen/`，
供 M2 全模型上板回归流式读取。Duo 现状：`/`（ext4）18GB 可用、swap 64MB 已启用、`/data` 已存在
（现有 smollm2 权重）。方案可行，无存储瓶颈。

## 1. 目标布局（与 host weights_kal 镜像）

```
/data/qwen/
  config.bin            32B
  embed_i8.bin          136.13 MB      ← LM head 流式读取
  embed_scales.f32      0.61 MB
  final_rms.f32         3.6 KB
  layer_scales.bin      (参考, 可选)
  scales.bin            (legacy 占位, 可选)
  layer0..23_kal.bin    24 × 8.39 MB = 201.44 MB
  layer0..23_bias.f32   24 × 18.0 KB = 0.43 MB
  合计                  339.53 MB（54 文件）
```

**必须落 `/data`（ext4 持久）**：`/tmp` 是 tmpfs（RAM 背衬，28MB 总量），放不下。

## 2. 传输方法（scp -O，legacy 协议）

- 环境要求 SCP 必须加 `-O`（BusyBox 端不支持 SFTP v3 新语义）。
- 单文件流式推送，逐文件 md5 校验。
- 命令形态：
  ```
  scp -O -r weights_kal/ root@192.168.42.1:/data/qwen/
  ```
  或逐文件 + md5 比对（更稳，可断点续传）：
  ```
  scp -O weights_kal/layerN_kal.bin root@192.168.42.1:/data/qwen/
  ```
- **带宽预估值**：USB RNDIS 网络 ~10-30MB/s；SD 写速（ext4）~20MB/s 级。
  339MB 预计 **15-60s（网络理想）/ 5-10min（实际 SD 写限）**。先跑 1 个 8.39MB layer 实测再批量。
- 校验：传输后 `md5sum /data/qwen/*` 与 host `md5sum weights_kal/*` 全量比对；不一致重传。

## 3. 运行前检查清单

1. **swap 已开**（`free -m` → Swap 64MB，已生效）。M2 全模型 decode 需大内存，swap 必要。
2. **SD 读带宽实测**：M2 引擎按 201MB/layer·token 流式读，实测 ~21.9MB/s（TPU 闸口②）。
   部署后先用 `dd if=/data/qwen/layer0_kal.bin of=/dev/null bs=1M` 复测该文件顺序读速率，
   确认无碎片化拖慢。
3. **ION 24MB**：M2 引擎 ION 布局（DESIGN §9a）≈18.84MB，余 5.16MB。与权重放 SD 无关。
4. **/data 空间**：18GB 可用，339MB 无压力。

## 4. 执行步骤

| 步 | 动作 | 验证 |
|---|---|---|
| 1 | `mkdir -p /data/qwen` | ls 确认 |
| 2 | scp -O 推送 config/embed/scales/bias（小文件，先导） | md5 |
| 3 | scp -O 推送 24 个 layerN_kal.bin（分批，每批 4 个） | md5 + 字节数 |
| 4 | 全量 md5 比对 | 零差异 |
| 5 | `dd` 顺序读 layer0_kal.bin 测带宽 | 记录 MB/s |
| 6 | 在 Duo 上 `qwen_kal_ref`-类校验加载（后续 M2 引擎内做） | — |

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| scp 中断 | 逐文件推送 + md5 重传；断点续传 |
| SD 写慢/碎片 | 先推 layer0 实测写速；大文件连续写，避免穿插小文件 |
| /data 权限 | root 登录，mkdir 后 chmod 755 |
| 传输时间超 watchdog | 用后台任务 + 分文件推送（每文件一次 scp），避免单条命令卡死 |

## 6. 关联

- M2 引擎（SD 流式 + ION 双缓冲）读取 `/data/qwen/`。
- 部署完成后即具备 M2 上板 3-prompt 回归条件。
