# dataloader_new_0915

DSpark 在线训练（dp4 + FSDP2）**周期性 forward 尖峰**问题的核对、决定性实验与复现包。

来源：`/home/y50063564/dataloader_diag`（原始报告 + 17 个实验 run）。
本文件夹是**精简重排版**：只保留能影响结论的实验、脚本与数据，并附可一键重算的复现脚本。

---

## 0. 怎么用

```bash
cd /home/y50063564/dataloader_new_0915

# 离线重算（秒级，不启动任何服务，只读本文件夹 data/）
bash repro/01_reproduce_problem.sh        # 复现问题现象
bash repro/02_reproduce_conclusions.sh    # 复现结论链

# 单 run 详细指标 / 多 run 横向对比
python3 repro/print_key_metrics.py data/E1_ref_w2_pf4
python3 repro/print_key_metrics.py data/*

# 真实重跑（各约 35 分钟 / 1.5 小时，需卡 10-14 空闲）
bash repro/01_reproduce_problem.sh --run
bash repro/02_reproduce_conclusions.sh --run
```

**复现的是统计量，不是逐位结果。** `dataloader_diag` 已实测：同代码同配置跑两次，
loss 不一致步占 60%、|Δloss| 均值 0.0165。所有结论都建立在分布/占比上，
`print_key_metrics.py` 输出的也正是这些占比（慢步率、fast 占比、ρ、ready_skew）。

---

## 1. 一句话结论

由四个环节组成，**前三环决定尖峰有多大，第四环决定为什么「平均分配」治不好它**：

1. **需求侧不均**：每步请求数（= pack 里的样本数）不受 token 打包约束，各 rank
   天然相差 ~26%（E1：4.67 vs 5.90）。
2. **供给侧刀刃**：每 rank 能力 = `workers ÷ RTT`，而 worker 用**同步** client（每 worker 只 1 个
   在飞请求）⇒ 能力被钉死在需求附近（ρ≈0.95）。
3. **集合同步放大**：最重的 rank 慢性饥饿、晚进 forward，DP 集合通信把这份迟到 τ
   原样变成全体 rank 的 fwd 尖峰（slope 0.97–0.98、R²=1.00）。
4. **分支锁存**（本文件新增定位）：`load_ms` 是双峰的 —— 要么命中 prefetch 缓冲（<1ms），
   要么当场等一整轮（>1s）。哪个 rank 在开局抢先半步填满缓冲，就在集体同步下被**永久固化**
   在快分支；其余 rank 停在 just-in-time。**这一步与「谁分到更重的 pack」无关，
   只与随机竞争有关** —— 所以均衡 pack 只能换掉受害者，不能消除尖峰。

判据：**ρ = 需求( docs/batch ÷ step_time ) ÷ 能力( workers ÷ RTT )**
实测 16 个 rank-样本：ρ≲0.90 → 慢步 ≤3%；ρ≈0.92–0.95 → 慢步 6%–54%（陡增）。
**ρ 决定刀刃有多薄；第 4 环决定谁站在刀刃上。**

### 1.1 「供需对齐」的定量形式：系统自平衡在 max ρ ≈ 0.95

12 个 run 全部吻合 —— **step_time 自我调整到「最重的那个 rank 恰好贴着 0.95 的产能上限」**：

| W | run | step_time | **max ρ** | mean ρ |
|---|---|---|---|---|
| 2 | E1 自然不均 | 1.26s | **0.951** | 0.906 |
| 2 | E6 流置换 | 1.27s | **0.949** | 0.903 |
| 2 | E5 doc 均衡 | 1.21s | **0.948** | 0.940 |
| 2 | E5' 人为不均衡 | 1.26s | **0.947** | 0.894 |
| 2 | E7b 窄带 | 1.14s | **0.947** | 0.935 |
| 4 | E3_w4 | 0.84s | 0.966 | 0.916 |
| 4 | E7_band_w4 | 0.79s | 0.947 | 0.946 |
| 6 | E3_w6 | 0.76s | 0.967 | 0.918 |
| 8 | E3_w8 | 0.76s | 0.964 | 0.910 |
| 8 | E7_band_w8 | 0.68s | 0.970 | 0.966 |
| 12 | E3_w12 | 0.76s | 0.961 | 0.906 |
| 8 | E4D_w8_seqs32 | 0.76s | 0.965 | 0.898 |

**为什么平衡**：step_time 由「最慢 rank 何时交出数据」决定。最重 rank 一旦跟不上，
step_time 就被拉长（矛盾缓解）；一旦有余量，step_time 又被训练算力下界压回来（矛盾加剧）。
两边夹逼 ⇒ 最重 rank 稳定停在 ρ_max≈0.95。

**这个自平衡点解释了「为什么任何分配手段都消不掉尖峰」**：

| run | max ρ | mean ρ | 谁在顶天花板 | 最差 rank 慢步 | **总慢步** |
|---|---|---|---|---|---|
| E1 自然不均 | 0.951 | 0.906 | **1 个 rank**，其余有余量（r0 仅 0.821） | r3 **53.0%** | 14.9% |
| E5 doc 均衡 | 0.948 | 0.940 | **4 个 rank 一起顶**，全员无余量 | r0 37.8% | **22.4%** |
| E7b 窄带 | 0.947 | 0.935 | 3 个顶 | r1 38.1% | 17.8% |

⇒ **均衡没有抬高天花板（max ρ 仍 0.948），它只是把「一个 rank 顶」换成「全员一起顶」。**
⇒ 副作用：**轻 rank 的余量原本在给系统兜底**（r0 ρ=0.821 意味着它有真实富余去吸收抖动），
   均衡把这部分余量拿掉了 ⇒ 风险从「集中」变「摊开」，总量反而升。

---

## 2. 你的印象 vs 文件夹实测（逐条核对）

| # | 你的表述 | 文件夹实测 | 判定 |
|---|---|---|---|
| 1 | 多 DP 并行时，某个 rank 的 pack 样本数显著多于其他 rank | E1：docs/batch = **4.67 / 5.49 / 5.82 / 5.90**（r3 比 r0 多 26%） | ✅ **一致** |
| 2 | 均衡只是全局角度；落实到每个 global step，各 rank 的 pack 仍可能不同 | 同一步四 rank 的 docs 极差均值 **1.62**，**只有 10.4% 的步完全相等**；E5 做了 doc 均衡后这两项**仍是 10.4% / 1.62（完全没变）** | ✅ **一致，且数据比报告更硬** |
| 3 | worker 给 server 打数据是一次一个样本，导致 pack 完成时间错位 | 机制属实：`__getitem__` = 1 doc = 1 次**同步** HTTP 请求，一个 pack（5–6 docs）要串行 5–6 次。**但「错位→饥饿」作为因果被否定**：互相关在 Δ≥1 无显著格点；rank 内 `corr(docs, log load_ms)=0.067`，按 docs 分组慢步率仅 47%→62% | ⚠️ **机制对，因果不成立** |
| 4 | dp4 下单 server 支持不了过多并发，所以 prefetch 没发挥作用 | **生产区间（W≥6）成立**：W6/W8 吞吐都停在 29 docs/s，`waiting` 堆到 16/25，`running` 只到 3–5（上限 8）⇒ server 是瓶颈。**W2 是不该采用的退化配置**（参考配置恰好落在 W2）：此时 server 没饱和（`running p50=0 / p90=3 / max=6`），瓶颈在 **client 侧在飞请求数**——同步 client ⇒ 每 worker 只 1 个在飞 ⇒ 每 rank 能力 ≈ 2/0.407s ≈ **4.9 docs/s**，重 rank 需求 4.68，**利用率 96%，刀刃上** | ✅ **成立**。W2 段如你所说是 bug（server 算力被浪费）；README 全文以 **W≥6 为生产口径**，W2 数据仅作为「对照/放大镜」 |
| 5 | 某个 rank 周期性出现饥饿 | load_ms **双峰**：`<1ms` 46.3% vs `1–3s` 49.6%，**中间段只有 0.7%**；自相关 **lag1=−0.89、lag2=+0.97** ⇒ 周期恰为 2（快→慢交替） | ✅ **一致** |
| 6 | FSDP 下 allgather，所有 rank 都要等这个饥饿 rank | ready_skew p50 **1067ms**、p90 2294ms；r3 在 **58–63%** 的步里是最后到达者；其余三 rank 的 fwd 时长 p50 ≈ **1.1s**，而 r3 自己只有 **103ms** | ✅ **一致（位置有修正，见下）** |
| 6b | （对**报告自身措辞**的修正，不是对你的表述的修正） | `REPORT.md` §2 的原假设 H 写作「其他 rank 在 FSDP AllGather 等待」。实测 all-gather **调用本身**只耗时 p50 **0.09ms**，等待落在 **`gap_before_ag2`**（root AG 结束 → layer0 AG 开始）：r0/r1/r2 = **1033/1058/966ms**，r3 仅 45ms。即「**首次消费被聚合参数时的设备侧等待**」 | ⚠️ 修正的是报告的措辞。你的表述是「各 rank 拿到数据的时间不同导致掉队」，与此**不冲突** |
| 7 | 把 pack 尽量对齐也解决不了，只是把掉队者从固定 rank 变成随机 rank，反而加剧 | E5（`SAMP_BALANCE=docs`，四 rank 都是 5.47）：慢性受害者消失（p50 897→0.1ms），但**总慢步率 14.9% → 22.4%**；E6 流置换后重流搬到 r0，**饥饿跟着数据流走**（r0 49.5%、原受害者 r3 降到 0.1%）⇒ 不是物理 rank/CPU 固定歧视 | ✅ **一致** |
| 8 | 方案 1：适当降低训练端需求，让 server 有余量去填充不均 | 与 J1 的 ρ 判据一致：**必须让 ρ 明显低于 ~0.9** 才能消除尖峰。注意两点：(a) 在 W2 下要降的是**每 rank 在飞需求**（client 侧），不是 server 负载；(b) 在 W≥6 下 server 已是天花板，降需求 = 直接降吞吐，无余量可填 | ✅ **方向成立，但限定条件比你说的更窄** |
| 9 | 方案 2：把数据集改成 seq 数相同的 pack 分配，让每一步 pack 中样本数相同 | **从未做过**。最接近的 E7/E7b 统一的是请求**尺寸**（600–800 带内），不是样本数。它确实大幅改善了 pack 均齐度（**相同步占比 10.4% → 60.6%**，极差 1.62 → 0.39），**但总量没降**：W8 窄带 6.2% vs 变长 6.8%（没降）、W2 窄带 17.8% vs 变长 14.9%（反而更差） | ⚠️ **未验证；已有数据倾向于不乐观** |

### 关于第 9 条，需要特别说明

你说「统一尺寸之后相当于已经改善了 dataloader 取 pack 的不均情况」——**这句话被数据支持**：

| run | 语料 | 四 rank 相同步占比 | 跨 rank 极差均值 |
|---|---|---|---|
| E1 / E6 / E5 / E5' / E3 | 变长（CV 0.82） | **10.4%** | 1.62 |
| E7 / E7b | 窄带 600–800（CV ≈0） | **60.6%** | 0.39 |

pack 均齐度提升了近 6 倍。**但尖峰总量没有随之下降** —— 这就是「改善 pack 不均」这条路径
目前的实证结论：它改的是**谁**挨饿和**时间分布**，不是**为什么**挨饿。

⚠️ **一个容易搞反的方向**：pack 不均的作用是「**集中**」风险，不是「**加重**」风险。
三次独立的「让 pack 更均」干预（E5 doc 均衡 14.9%→22.4%、E7b 统一尺寸 14.9%→17.8%、
E7_band_w8 6.8%→6.2%）**一次也没降低总慢步率**。原因见 §1.1：不均时轻 rank 的余量
（r0 ρ=0.821）在给系统兜底，均衡把余量拿掉后风险摊到全员，总量反而升。

严格意义上的「每步 pack 样本数完全相同」截至 `dataloader_diag` 仍未实施（E7 窄带下
docs/batch 仍是 `{5: 68%, 6: 32%}`，**32% 的步没对齐**），只出现在 `REPORT.md` §8 的
可行性分析表里。所以它**是一个尚未排除的方向**，但按现有数据，预期收益是
「消灭固定受害者 + 提高 fast 占比」，而不是降低总停顿。

---

## 3. 决定性实验清单（哪个实验改变了哪条结论）

| 实验 | run 目录 | 干预 | **它改变了什么结论** |
|---|---|---|---|
| **P0 前置门** | `p0c/`（未收录原始数据） | 同配置跑两次 vs 开插桩 | 确认插桩影响 ≤ 平台固有噪声；确认跨 rank 时钟可对齐、AG 事件数正确、docs 计数正确。**所有后续结论的可信度前提** |
| **E1 参考基线** | `E1_ref_w2_pf4` | dp4 + W2 + prefetch4 + server dp1 + seed42 | 复现现象：r3 慢性饥饿（53% 慢步）、ready_skew p50 1067ms |
| **E2 滞后耗尽** | `_E2_early_20260910/` + `analysis/legacy/analysis_inorder*` | 互相关 / 队列深度重构 | **否定**「累计 docs → prefetch 耗尽 → 若干步后断粮」（无 Δ≥1 显著格点）；**否定**「当期批次成本解释等待」（R²≈0.013）。把「偶发耗尽」修正为「慢性饥饿」 |
| **E1c 阻塞点定位** | `E1c_agwait_ref` | `AG_PROFILE=1 AG_MAX_PER_FWD=6` | 把阻塞点从「all-gather 调用」**修正**到 `gap_before_ag2`（首次消费被聚合参数）。slope 0.97–0.98、R²=1.00（n=400） |
| **E6 流置换** | `E6_perm_rev` | `SAMP_RANK_PERM=3,2,1,0` | **否定**「物理 rank / CPU 固定歧视」——饥饿跟着数据流走 |
| **E5 均衡** | `E5_balance_docs` | `SAMP_BALANCE=docs` | **否定**「均衡 docs 即可解决」——受害者消失但总慢步率 14.9%→22.4% |
| **E5' 人为不均衡** | `E5b_skew_docs` | `SAMP_BALANCE=skew_docs` | **支持**剂量-反应：docs/batch 最高的 rank 恒为饥饿者（6.19 → 54.0% 慢步） |
| **E3 worker 扫参** | `E3_w1/w4/w6/w8/w12_pf4` | workers ∈ {1,2,4,6,8,12} | 定位**瓶颈切换点**：吞吐 11.3→17.4→26.2→29.0→29.0→28.7 docs/s，W≥6 饱和。W2 段卡在 client 侧在飞数，W≥6 段卡在 server |
| **E4-C token 预算** | `E4C_w8_tok16384/32768` | `--max-num-batched-tokens` 4096→16384→32768 | **否定**「token 预算是天花板」——吞吐完全不变（19243/19243/18981 tok/s），开关已验证生效 |
| **E4-D 并发槽位** | `E4D_w8_seqs16/32` | `--max-num-seqs` 8→16→32 | **否定**「并发槽位是天花板」——并发 3→6、排队 25→0，吞吐仍不变（19243/20115/18972 tok/s） |
| **E7 / E7b 方差控制** | `E7_band_w4/w8`、`E7b_band_w2` | 窄带语料 600–800（请求尺寸 CV 0.82→≈0） | **否定**「压方差 ⇒ 高 ρ 稳定」——只摊平负担、不减总量（W8 6.2% vs 6.8%；W2 17.8% vs 14.9%） |

**非决定性、仅存档**（`scripts/legacy/`、`data/_E2_early_20260910/`）：
早期 `train_diag_*` / `run_ab` / `run_tl` / `run_diag_sequence` / `serve_diag_dp1*` 等探索性脚本，
以及 `in_order=0/1` 的对照。它们没改变最终结论，保留是为了可追溯。

---

## 4. 脚本功能说明

### `scripts/experiment/` —— 实验驱动（会启动 server + 训练）

| 脚本 | 功能 |
|---|---|
| `run_exp.sh` | **通用实验入口**。`run_exp.sh <RUN_ID> [--workers N] [--prefetch N] [--max-steps N] [--server-dp N] [--in-order 0\|1] [--samp-balance off\|docs\|tokens\|skew_docs] [--samp-perm a,b,c,d] [--samp-seed N] [--ag-max N] [--data PATH] [--no-server] ...`。负责：写 `config.txt`（含 git HEAD）+ `diff.patch`、起 server、起两个采集器、`torchrun --nproc_per_node 4` 起训练、结束清理。所有下游实验都调它 |
| `run_E3.sh` | worker 扫参 W=1/4/6/8/12（W=2 复用 E1），server 只起一次复用；每个 run 结束打印 rank 级 load/docs/供给率与 server running/waiting |
| `run_E4C.sh` | 固定 W=8，扫 `--max-num-batched-tokens` = 16384/32768（4096 复用 E3_w8），打印 docs/s、tok/s、tok/step |
| `run_E4D.sh` | 固定 W=8 + budget 32768，扫 `--max-num-seqs` = 16/32，打印并发/排队/吞吐 |
| `run_E6_E5.sh` | 串起三组因果干预：E6 流置换 → E5 doc 均衡 → E5' 人为不均衡，逐个打印 rank 级 load p50/p90/慢步次数 |
| `run_E7.sh` | 窄带语料（600–800）在 W=8 / W=4；打印每 rank 的 ρ（需求/能力）与慢步率、fast 占比 |
| `run_E7b.sh` | 窄带语料在 W=2（与 E1 同档在飞数，做匹配对照） |

### `scripts/server/` —— server 变体（同一 API 端点，训练端无感知）

| 脚本 | 功能 |
|---|---|
| `serve_e1_dp1_1132.sh` | 参考 server：卡 10、dp1、端口 1132、`max-num-seqs 8`、`max-model-len 4096`、`--enforce-eager`、extract_hidden_states 投机配置、HS 落盘到 `_hs_e1_dp1` |
| `serve_e1_dp1_1132_tok.sh` | 同上，额外用 `MAX_NUM_BATCHED_TOKENS` 覆盖 token 预算（E4-C 用） |
| `serve_e1_dp1_1132_seqs.sh` | 同上，额外用 `MAX_NUM_SEQS` 覆盖并发槽位（E4-D 用） |
| `serve_e1_dp2_1132.sh` | **dp2 副本**（卡 10+15）挂在同一个端口后，用于验证「加副本」的提速上限（≤2×） |

### `scripts/analysis/` —— 离线分析（纯读 jsonl，无分布式调用）

| 脚本 | 功能 |
|---|---|
| `analyze_rootcause.py` | 四个子命令：`chain`（逐步跨 rank 对齐、ready_skew、迟到量）、`cost`（按 docs 分桶的样本耗时/collate 耗时）、`xcorr`（K×Δ 互相关，检验「滞后耗尽」）、`queue`（prefetch 队列深度重构） |
| `analyze_ag_align.py` | 按 global_step 与按集合序两种对齐方式，算各 rank 到达集体的迟到量，交叉验证 `chain` 的标签 |
| `analyze_dlprof.py` | 汇总 L1（trainer 等 batch）/L2（worker 样本）/L3（collate）/L4（逐步时间线）四层 |
| `analyze_buffer.py` | 用 `bins_*.jsonl` 重构 prefetch 缓冲深度随步变化（E2c 的原始证据） |
| `extract_steps.py` | 从 train log 抽逐步 loss/耗时 |

### `scripts/tools/`

| 脚本 | 功能 |
|---|---|
| `scrape_metrics.py` | 每秒抓 vLLM `/metrics` 写 jsonl（running/waiting/token 计数等） |
| `scrape_sys.py` | 每秒抓 `/proc/<pid>/schedstat`、`/proc/vmstat` 等，用于 E7 混淆审计（CPU 调度等待、swap、缺页） |
| `build_band_subset.py` | 从 700k 语料构造窄带（600–800 token）子集，供 E7/E7b 用 |
| `test_sampler_equiv.py` | 采样器自检：`SAMP_BALANCE=off` 与原实现逐位一致；`docs` 均衡把 doc 数 spread 442→3；`skew_docs` 拉大到 740；`perm=3,2,1,0` 精确反转流 |

### `repro/`

| 脚本 | 功能 |
|---|---|
| `print_key_metrics.py` | **关键指标打印器**。单 run 打印 6 节（负载与饥饿 / 逐步跨 rank 不均 / 周期性 / 集合同步放大 / server 天花板 / ρ）；多 run 打印横向对比表 |
| `01_reproduce_problem.sh` | 复现问题现象：E1 症状 + E1c 阻塞点 + W8 对照 |
| `02_reproduce_conclusions.sh` | 复现结论链：§1§2 受害者选择与均衡、§3 瓶颈切换、§4 天花板、§5 方差控制 |
| `_slim_data.py` | 生成本文件夹 `data/`（从 `dataloader_diag/exp` 精简：metrics.jsonl 只留 6 个 counter；bins 仅留 E1/E1c；丢弃 sys.jsonl） |

---

## 5. 数据与复现口径

- **17 个 run 的原始 jsonl** 在 `data/<RUN_ID>/`：`rank*/load_rank*.jsonl`（L1）、
  `timeline_rank*.jsonl`（L4）、`ag_rank*.jsonl`（AG 事件）、`worker*_pid*.jsonl`（L2/L3），
  以及 `config.txt`、`run.log`、`metrics_slim.jsonl`。
- **精简规则**：`metrics.jsonl` 70MB→~1MB（只留 `num_requests_running` / `num_requests_waiting` /
  `num_requests_waiting_by_reason` / `prompt_tokens_total` / `iteration_tokens_total_count|sum`）；
  `bins_*.jsonl`（8.6MB/rank）仅 E1/E1c 保留；`sys.jsonl` 丢弃（E7 混淆审计结论已固化在 `EVIDENCE.md` I 节）。
  总体积 1.3GB → **225MB**。
- **已验证**：精简后的数据能逐位重算原报告的全部关键数字（docs/batch、慢步率、ρ、ready_skew、gap_before_ag2）。

**复现的边界（不要越界解读）**：
1. 全部结论为**单一 seed（42）+ 单机 + 单副本 server**。
2. 平台 **run-to-run 非逐位可复现**（同配置两次 |Δloss| 0.0165）⇒ 只能用分布/占比比较。
3. 精确的**每请求生成 token 数不可得**（响应无 usage 字段）；HS 体积为估算。
4. **真实 prefetch 队列深度**为时间戳重构（未读私有变量）。
5. server 侧 **per-request → engine 归因无日志**，「token 速率受限」是聚合层证据。
6. **CPU affinity / NUMA 隔离实验不可做**（容器共享 640 核 cpuset）；以 `schedstat`（run-queue wait 0.1%）间接排除。
7. **GC 停顿**未直接测量。

---

## 6. 未决问题

### 6.1 rank 级残余抖动：机制已定位为「分支锁存」（**本文件新增的分析**）

`E7b` 中四个 rank 的请求尺寸相同、docs/batch 相同（5.31–5.34）、ρ 几乎相同，
慢步率却是 **0.1% vs 38.1%**：

| rank | docs/batch | RTT mean | CPU/doc | ρ | fast(<1ms) | 慢步率 |
|---|---|---|---|---|---|---|
| r0 | 5.32 | 404ms | 121ms | 0.942 | 80.6% | 18.9% |
| r1 | 5.34 | 403ms | 120ms | 0.943 | 61.1% | **38.1%** |
| r2 | 5.34 | 389ms | 112ms | **0.909** | **99.8%** | **0.1%** |
| r3 | 5.31 | 407ms | 124ms | 0.947 | 84.9% | 14.0% |

`REPORT.md` §11 把这条列为「成因未定的 rank 级残余抖动」。**实测数据可以把机制说清楚**：

**(a) 差异不在生产侧。** 四个 rank 的 **RTT（389–407ms）与 CPU/doc（112–124ms）逐项相同**，
样本数也相同（6402–6461）。所以不是「谁的生产慢」。

**(b) 差异在「缓冲里有没有现成批次」。** `load_ms<1ms` 表示该步**命中 prefetch 缓冲**，
`>200ms` 表示**当场等生产**。四 rank 的生产速度相同 ⇒ 快慢只能由「缓冲是否已填满」决定。

**(c) 这是一个双稳态分支，且开局就锁定。** 逐 200 步窗口的 fast 占比：

```
E7b(W2)  r0:  50%  70% 100% 100%  90%  73%      ← 开局慢，中途爬上去
         r1:  56%  68%  50%  50%  48%  94%      ← 长期停在 just-in-time
         r2:  99% 100% 100% 100% 100% 100%      ← 从第 2 步起就锁定在快分支
         r3:  74%  97% 100% 100%  94%  46%      ← 爬上去又掉下来
E1(W2)   r0: 100% 100% 100% 100% 100% 100% 100% 100%   ← 开局 98% fast，锁死
         r3:  44%  50%  50%  36%  46%  46%  50%  50%   ← 全程锁在慢分支
```

看 E7b 开局 25 步的原始 `load_ms`：四个 rank 在前 2 步都同样在等；**第 2 步 r2 拿到 0ms
（命中）而 r0/r1/r3 都在等 300–400ms**。从那一刻起 r2 再没慢过；其余三个落入
「慢一步→快一步」的周期 2 交替（自相关 lag2=+0.96）。

**(d) 为什么优势锁死、追不回来。** DP 训练每步做集合通信，**所有 rank 必须等齐才进下一步**。
快 rank 在集合点等，等待过程**不消耗它已填满的缓冲**；慢 rank 则每步都在消费掉刚产出的那一个批次，
缓冲永远填不满。于是「开局随机领先半步」被集体同步固化成永久分层。

**(e) 领先者会易主 ⇒ 不是固定 rank 属性。** E7_band_w8 的领先者序列是
`r2 → r2 → r1 → r1 → r1 → r3 → r3 → r3 → r1 → r1 → r3 → r3`（每 100 步窗口），
E7b 是 `r2 → r2 → … → r0 → … → r2 → r1`。W2/W4 里是 r2 领先、W8 里变成 r1 —— 
**谁是赢家由早期随机竞争决定**，与 rank 编号、CPU、NUMA 无关。

**这解释了整条争议链**：

| 现象 | 分支锁存给出的解释 |
|---|---|
| E6 置换后饥饿跟着流走 | 换的是「谁的包更重」，重包让该 rank 在开局竞争中更可能落后 ⇒ 更容易被锁在慢分支 |
| E5 均衡后受害者消失、总量反升 | 均衡消除了**偏置**（谁更可能输），但没消除**双稳态**（总会有人输）；四个 rank 都停在 ρ≈0.94 的刀刃上，谁都可能输，总停顿反而更多 |
| 「统一尺寸」改善了 pack 不均却减不了尖峰 | 同上：改的是偏置，不是双稳态 |
| E7b 同 ρ 同尺寸下 0.1% vs 38.1% | 就是分支归属本身；ρ 只决定「刀刃有多薄」，不决定「谁站在刀刃上」 |

⇒ **残余抖动的成因是「随机领先 + 集体同步固化」，不是 server 排队顺序。**
这同时说明：只要 ρ 仍在刀刃附近，**任何只作用于「平均分配」的手段都无法消除尖峰**
—— 这正好回答了下面的 6.2。

诊断脚本：`repro/print_key_metrics.py` 的 **[5] 分支锁存诊断**一节。

### 6.2 源文件夹内部的口径不一致（**已核实并统一**）

`dataloader_diag` 里关于「统一尺寸」有两句**互相矛盾**的话，需要说明：

| 出处 | 原话 | 立场 |
|---|---|---|
| `EVIDENCE.md` L2 判读 1 | 「同档甚至更高 ρ 下，统一尺寸**显著更稳**：E1-r3（ρ=0.95，变长）慢步 53% vs E7 窄带（ρ=0.96–0.97，统一）慢步 1–14% ⇒ 请求尺寸方差是**一阶失稳来源**」 | **支持**「压方差能稳定」 |
| `REPORT.md` §6 结论 2 / §10 闭环表 | 「压方差**不能降低总量**，只能摊平负担：匹配 ρ 与在飞数时（W8）统一尺寸 6.1% ≈ 变长 6.8%；W2 下 17.8% 甚至略高于 14.9%」 | **否定（仅摊平）** |

**两句都对，但比较口径不同 —— L2 判读 1 是未匹配在飞数的中间结论。**

- L2 判读 1 比的是 **E1（W=2，每 rank 在飞 2）** 与 **E7（W=4/8，每 rank 在飞 4/8）**：
  两组**同时变了两个变量**（请求尺寸方差 ↓ **和** 在飞请求数 ↑）。
- `EVIDENCE.md` L2 判读 3 自己指出了这一点：「**第二条等价杠杆 = 在飞请求数**：
  E3_w8（变长、ρ=0.964）12.5% ≪ E1-r3（变长、ρ=0.951）53% —— ρ 与方差相同、
  差别在每 rank 8 个在飞请求把抖动平均掉」。
- `REPORT.md` §6 做的是**匹配对照**：固定 W，只换语料。
  - W8：窄带 **6.2%** vs 变长 **6.8%** → 没降
  - W2：窄带 **17.8%** vs 变长 **14.9%** → 反而更差

我用原始数据重算了两组匹配对照，结论与 `REPORT.md` §6 一致。
**因此本 README 采用 `REPORT.md`（定稿）的口径：压方差只改分布、不改总量。**
`EVIDENCE.md` 的 L2 判读 1 应视为被 §6 与 L2 判读 3 修正过的早期表述。

> 附带一提，源文件夹里同类「先给结论、后被自己修正」的地方不止这一处：
> L1 节也写着「我此前用固定 RTT 算的 W≥4 的 ρ 偏低，已改正」。阅读 `EVIDENCE.md` 时
> 应以 `REPORT.md` 的定稿结论为准。

### 6.3 方案 2 的验证缺口

严格的「每步 pack 样本数完全相同」从未实施。若要补做，按 `REPORT.md` §8：

- 做法：把语料过滤/截断到 doc ≤ 819 token，令包预算收到 < 6 × 最短 doc，从而每包**恰好 N 个 doc**；
- 预期：**消除 per-step 需求跳动**（当前窄带下 docs/batch 仍有 32% 的步是 6、68% 是 5）；
- 代价：**改变训练分布**；且按 §7 的结论，供给受限时平均吞吐与方差无关，
  **预期不提速**，收益只在「消灭固定受害者 + 提高 fast 占比」。
- 判据建议：对比 `pack 相同步占比`（当前 60.6%，目标 100%）与**总慢步率**（当前 W8=6.2%、W2=17.8%）。

### 6.4 已被排除、不必再查的方向

token 预算（E4-C）、并发槽位（E4-D）、CPU 调度/swap/缺页（E7 混淆审计）、
server 产能不足（W2 段 server running p50=0）、物理 rank 歧视（E6）、
存储吞吐（本地 ext4 写 1.4GB/s、读 2.6GB/s，实际用量 ~45%）。
详见 `EVIDENCE.md` K 节与 I 节。

---

## 7. 文件清单

```
dataloader_new_0915/
├── README.md              ← 本文件：核对表 + 实验清单 + 复现指引
├── REPORT.md              ← 原始定稿报告（12 节）
├── EVIDENCE.md            ← 证据台账 A–M 节，逐条主张 → 数据文件与数字
├── repro/
│   ├── 01_reproduce_problem.sh      ← 复现问题现象
│   ├── 02_reproduce_conclusions.sh  ← 复现结论链
│   ├── print_key_metrics.py         ← 关键指标打印器（7 节，含 [5] 分支锁存诊断）
│   └── _slim_data.py                ← 数据精简脚本
├── scripts/
│   ├── experiment/   run_exp.sh, run_E3/E4C/E4D/E6_E5/E7/E7b.sh
│   ├── server/       serve_e1_dp1_1132*.sh, serve_e1_dp2_1132.sh
│   ├── analysis/     analyze_rootcause/ag_align/dlprof/buffer.py, extract_steps.py
│   ├── tools/        scrape_metrics/sys.py, build_band_subset.py, test_sampler_equiv.py
│   └── legacy/       早期探索脚本（非决定性，存档）
├── data/             17 个 run 的精简原始数据 + _E2_early_20260910/
└── analysis/         E1/E1_align/E1_cost/E1_queue/E1_xcorr（CSV/PNG）+ legacy/
```

---

## 附：插桩回退

训练端插桩（`DL_PROFILE` / `AG_PROFILE`）与 sampler 改动都在
`/home/y50063564/dspark_project/speculators`（git HEAD `0a5f46c`），
默认全关，不设环境变量即原行为。回退：

```bash
cd /home/y50063564/dspark_project/speculators
git checkout -- src/speculators/train/distributed_batch_sampler.py
```

改动前的快照与完整 patch 在 `/home/y50063564/dataloader_diag/snapshot/`。
