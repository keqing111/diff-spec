# DataLoader 负载不均问题说明

**结论**：DSpark 在线训练的 DataLoader 在 **DP 各 rank 之间造出的 pack 负载不均**——
每个 pack 的 **token 数被精确填满且各 rank 一致**，但 pack 里的**样本数相差 26%**
（4.67 vs 5.90）。由于每个样本都要向 server 发一次 hidden-states 请求，
**每个 pack 的生产时间天然相差 26%，整步耗时由最重的那个 rank 决定**。

本文给出：现象的量化 → 原因 → 后果 → 一键复现方法。

```bash
bash repro/run_all.sh            # 全部复现（离线 + 录制数据），不需要 GPU/server/训练
bash repro/run_all.sh --quick    # 只跑离线复现（约 1 分钟）
```

---

## 0. 一页纸

| | 值 |
|---|---|
| 代码位置 | `MultipackDistributedBatchSamplerV2`（`src/speculators/train/distributed_batch_sampler.py`） |
| 打包约束 | `batch_max_length=4096`，**只约束 token 数** |
| 各 rank 的 token 填满率 | 94.6% / 94.7% / 94.6% / 94.7%　**几乎完全一致** |
| 各 rank 的 docs/pack | **4.67 / 5.49 / 5.82 / 5.90**　**相差 26%** |
| 跨 rank 同一步的 docs 极差 | 均值 **1.62**，中位 1 |
| 四个 rank pack 完全相同的步 | **仅 10.4%** |
| 每个 pack 的 server 请求数 | = docs/pack（每样本一次请求）⇒ **相差 26%** |
| 整步耗时 | `≈ docs_max × RTT ÷ num_workers × 1.04` |
| 若 pack 完全均衡的理论收益 | **+7.1% 吞吐**（W=8 实测口径） |

**一句话**：packing 把 token 塞满了，但没管样本数；样本数才是 server 请求数，
才是真正的成本单位。两者不成比例，因为文档长度方差大（CV=0.82）。

---

## 1. 现象

### 1.1 token 填满率一致，但样本数差 26%

真实训练 run（`E3_w8_pf4`，W=8，700k 语料，seed 42）录制值：

| rank | packed_tokens | token 填满率 | **docs/pack** | docs 范围 |
|---|---|---|---|---|
| r0 | 3879 | 94.7% | **4.68** | 2~9 |
| r1 | 3878 | 94.7% | **5.50** | 2~9 |
| r2 | 3875 | 94.6% | **5.83** | 3~9 |
| r3 | 3878 | 94.7% | **5.90** | 2~9 |

`packed_tokens` 四列几乎完全相同（说明打包本身是**有效且公平**的），
但 `docs/pack` 从 4.68 到 5.90，**相差 26.0%**。

### 1.2 不只是平均值不同，每一步都不同

| 指标 | 值 |
|---|---|
| 跨 rank 同步步上 docs 极差 | 均值 **1.62**，中位 1，max 5 |
| 四个 rank pack 完全相同的步 | **10.3%** |

⇒ **89.7% 的 global step 上，四个 rank 拿到的 pack 样本数并不相等。**
即使把语料换成窄带（长度 600~800）后，仍有 ~38% 的步不相等（见 §3）。

---

## 2. 原因

### 2.1 打包约束的是 token，不是样本数

```python
MultipackDistributedBatchSamplerV2(
    batch_max_length=4096,   # ← 唯一的约束
    lengths=dataset.approx_lengths,
    num_replicas=4, rank=rank, seed=42,
)
```

样本数从不参与约束，它只是「**填满 4096 token 之后的副产物**」。

### 2.2 文档长度方差大，副产物就不稳定

700k 语料的长度分布：

| 指标 | 值 |
|---|---|
| mean / median | 711 / **507** token |
| p95 / p99 / max | 1857 / **3072** / 3072 |
| **CV** | **0.82** |

同样填满 4096 token：

- 全命中位长度文档 → 约 **8.1 个**
- 全命 p99 长度文档 → 约 **1.3 个**

一个 pack 里混进几个长文档，样本数就掉一截。各 rank 因为分到的文档组合不同，
样本数就随机游走开——**并且是持久性的偏差**（r3 长期比 r0 多 26%，不是随机涨落）。

### 2.3 离线可复现 ⇒ 这是打包算法 + 语料的性质，与训练无关

`repro/01_pack_imbalance.py` 不训练、不连 server，直接用同一个 sampler 生成 pack：

| rank | packed_tokens | docs/pack |
|---|---|---|
| r0 | 3873 | **4.62** |
| r1 | 3872 | **5.46** |
| r2 | 3873 | **5.80** |
| r3 | 3871 | **5.89** |

与录制值逐项吻合（差 <1.1%，来自训练时的批次对齐起点），**排序与幅度完全一致**。

---

## 3. 后果

### 3.1 每个文档 = 一次 server 请求 ⇒ pack 生产时间相差 26%

`ArrowDataset.__getitem__` 一次处理 **一个文档**，同步调用 server 生成 hidden states，
本地不留缓存（`--on-generate delete`）。所以：

```
一个 pack 的生产时间 = docs/pack × RTT ÷ num_workers
```

`docs/pack` 相差 26% ⇒ **每个 pack 的生产时间相差 26%**。

### 3.2 整步耗时 = 最重 rank 生产一个 pack 的时间

实测 `step_time ≈ docs_max × RTT ÷ num_workers`：

| run | W | docs_max | RTT | 预测 step_time | 实测 | 比值 |
|---|---|---|---|---|---|---|
| E1（W=2） | 2 | 5.90 | 407ms | 1.20s | 1.26s | 1.052 |
| E3_w4 | 4 | 5.90 | 547ms | 0.81s | 0.84s | 1.036 |
| E3_w6 | 6 | 5.90 | 743ms | 0.73s | 0.76s | 1.034 |
| **E3_w8** | 8 | 5.90 | 989ms | 0.73s | 0.76s | **1.037** |
| E7_band_w8（窄带） | 8 | 5.34 | 989ms | 0.66s | 0.68s | 1.036 |

**比值全部在 1.03~1.06**，7 个配置全中。也就是说：
**整步耗时完全由最重 rank 串行生产一个 pack 的时间决定，训练计算被掩盖。**

⇒ **理论收益**：若各 rank pack 完全均衡（W=8 口径下 5.90 → 5.48），
step_time 0.76s → 0.68s，**吞吐 +7.1%**。

### 3.3 更重的一层：掉队的 rank 会把全体拖住

DP 每步要做集合通信，所有 rank 必须等齐。实测（E3_w8）：

| | 值 |
|---|---|
| 各 rank 进入 batch 的时刻差 | p50 **33ms**，p90 **2401ms** |
| 有 rank 停顿的步占比 | 20.4% |
| **仅在这些停顿步上**看时刻差 | p50 **2323ms** |

⇒ 每 5 步有 1 步，最重的 rank 晚到 ~2.3 秒，**其余三个 rank 的 forward 全程在等它**。

### 3.4 `load_ms` 的双峰：队列空 vs 队列非空

**重要：这里没有任何本地缓存。** 每个样本都是 cache miss、都要请求 server。
`load_ms` 的双峰说的是 **DataLoader prefetch 队列**的状态：

| 取批时队列深度 | 步数占比 | `load_ms` |
|---|---|---|
| > 0（批次已提前产好、在队列里） | ~80% | **< 1ms** |
| = 0（队列空，要现场等 worker 生产） | ~20% | **> 200ms（~2.3s）** |

实测队列深度上限 = `prefetch_factor(4) × num_workers` = **32**（W=8），与配置完全一致。

---

## 4. 复现方法

### 4.1 离线复现（推荐，1 分钟，不需要 GPU/server/训练）

```bash
bash repro/run_all.sh --quick
# 或
python3 repro/01_pack_imbalance.py --steps 1500 --compare
```

- 依赖：`numpy`、`datasets`、训练代码 `/home/y50063564/dspark_project/speculators`
- 数据集：`/home/y50063564/data/open_perfectblend_qwen3_4b_700k`（也支持 `--data` 指定）
- 输出：§1.1 的 pack 统计表、§1.2 的逐步极差表、§2.2 的长度分布表；
  `--compare` 会额外跑窄带语料做对照

### 4.2 录制数据验证（秒级）

```bash
bash repro/run_all.sh
# 或
python3 repro/02_validate_on_recorded_run.py data/E3_w8_pf4
```

只读 `data/<RUN>/` 下的 jsonl，输出四节：
[1] 录制的 pack 不均 · [2] step_time 模型验证 · [3] prefetch 队列深度 · [4] 集合点等待

已收录的 run：

| 目录 | 配置 | 用途 |
|---|---|---|
| `data/E3_w8_pf4` | W=8，700k 变长语料 | **生产口径** |
| `data/E1_ref_w2_pf4` | W=2（见 §6 说明） | 参考配置 |
| `data/E7_band_w8` | W=8，窄带 600~800 语料 | 对照 |

### 4.3 关于窄带对照的结论

把文档长度压到 600~800（CV 0.82 → ≈0）后：

| | 700k 变长语料 | 窄带 600~800 |
|---|---|---|
| docs/pack 跨 rank 差异 | **26.0%** | **0.3%** |
| 四个 rank 相等的步占比 | 10.3% | **61.5%** |
| docs/pack 取值 | 2~9 | 5~6 |

⇒ **长度方差是主因**。但即使压掉方差，**仍做不到 100% 相等**——
整数装箱下 `4096 / 中位长度` 不是整数，各 rank 仍会在相邻两值间跳（此处 68% : 32%）。
**要真正对齐到"每步样本数相同"，必须改打包策略本身（见 §6）。**

---

## 5. 数据与口径

- 全部数据来自 `/home/y50063564/dataloader_diag` 的 17 个实验 run（git HEAD `0a5f46c`），
  这里只精选 3 个并做了精简（只保留 `load` / `timeline` / `worker` 三类 jsonl）。
- 所有 run 均为 **单一 seed（42）+ 单机 + 单副本 server**。
- 平台 **run-to-run 非逐位可复现**（同配置两次 |Δloss| 均值 0.0165），
  因此所有结论都用**占比 / 分布**表达，不用单点值。

---

## 6. 需要澄清的三个常见误解

### 6.1 不要用「ρ ≈ 0.95」当阈值 —— 那是同义反复

早期分析算过一个「供需比」 `ρ = 需求 ÷ 能力 = (docs/step_time) ÷ (num_workers/RTT)`，
并发现「所有 run 的 max ρ 都是 0.95~0.97」。**这个数字没有诊断价值**，因为：

```
ρ_max = docs_max × RTT ÷ (step_time × num_workers)
```

而 §3.2 已经证明 `step_time = docs_max × RTT ÷ num_workers × 1.04`，
代入即得 `ρ_max ≡ 1/1.04 ≈ 0.96`。**它恒等于一个常数，与配置无关**——
这是 step_time 由最重 rank 决定的**必然结果**，不是独立测量。

**有诊断价值的是 §3.2 那个等式本身**：整步耗时 ∝ 最重 rank 的 pack 样本数。

### 6.2 `load_ms < 1ms` 不是"缓存命中"

系统**没有本地 hidden-states 缓存**，每个样本都要走一次 server 请求（cache miss）。
`<1ms` 只表示**批次已由 worker 提前生产好、排在 DataLoader 的 prefetch 队列里**。
建议统一用「**队列深度**」而不要用「命中/未命中」描述。

### 6.3 `num_workers=2` 是配置缺陷，不是正常工况

诊断时用的参考配置恰好是 `--num-workers 2 --prefetch-factor 4`。
实际生产中 worker 数应当尽量开大。W=2 时每 rank 只有 2 个在飞请求，
每 rank 供给能力 ≈ `2 / 0.407s ≈ 4.9 docs/s`，恰好被压到需求刀刃上，
server 侧 `running p50=0`（算力被浪费）——**这是配置问题，不是 server 能力问题**。

实测 worker 扫参（同 server、seed 42）：

| W | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| step/s | 0.51 | 0.79 | 1.20 | **1.32** | **1.32** | 1.31 |
| server `waiting` max | 2 | 4 | 9 | 16 | 25 | 41 |

⇒ W ≥ 6 后完全饱和（瓶颈转到 server 侧，见下）。**W=2 那一段不应作为设计依据。**

---

## 7. 可以动的方向

| 方向 | 能否消除 pack 不均 | 需要改什么 | 预期收益 |
|---|---|---|---|
| **1. 提高 worker 数** | ❌ 不消除不均 | 无需改代码（配置问题） | W=2→6 约 +67%（但这是修复配置缺陷，不是优化） |
| **2. 恰好 N 个 doc 组成一个 pack** | ✅ 需求恒定为 N | 需过滤/截断语料到 doc ≤ 819 token，让包预算 < 6×最短 doc | 消除 per-step 抖动；**改变训练分布** |
| **3. 整个 pack 作为一个 server 请求** | ✅ 请求数与尺寸都恒定 | 数据层重构（生成从 `__getitem__` 上移到 pack 级） | 同时摊薄每请求固定开销；受 `max_model_len` 限制 |
| **4. 长度分桶（改语料，不改协议）** | ⚠️ 部分 | 只需重做语料 | 差异 26%→0.3%，但**不等价于"每步相同"**（仍有 38% 的步不等）。反向印证：窄带 run 剩余的均衡收益只剩 **+0.2%** |
| **5. 加 server 副本** | ❌ 不消除不均 | 加卡 | ≤2×（抬高供给侧天花板） |

**注意方向 1 和 5 治的不是这个病**：它们抬高供给上限，但不改变
「一个 pack 的请求数由语料长度随机决定」这个事实。

---

## 8. 文件清单

```
dataloader_imbalance_0915/
├── README.md                          ← 本文
├── repro/
│   ├── run_all.sh                     ← 一键复现入口
│   ├── 01_pack_imbalance.py           ← 离线复现（真实 sampler，无需训练）
│   └── 02_validate_on_recorded_run.py ← 录制数据验证
└── data/
    ├── E3_w8_pf4/                     ← W=8（生产口径）
    ├── E1_ref_w2_pf4/                 ← W=2（参考配置）
    └── E7_band_w8/                    ← 窄带对照
        └── rank{0..3}/{load,timeline,worker}.jsonl
```

## 附：相关背景

本问题是更大范围诊断的一部分（FSDP2 多卡周期性 forward 尖峰）。完整报告、
17 个实验 run 与证据台账在 `/home/y50063564/dataloader_new_0915/`。
本文只保留与 **DataLoader 负载不均**直接相关的部分。
