# dataloader_diag — DSpark 在线训练 FSDP2「周期性 forward 尖峰」根因验证（分支：dspark-fsdp-spike-rootcause）

本分支收录一次**可证伪、可复现、含干预实验**的根因分析：现象 → 原假设 H → 逐边证据 → 因果干预 →
server 天花板定位 → 方差实验 → 结论与剩余不确定性。

## 先读这两个
- **`REPORT.md`** — 定稿报告（现象/逐边判定/干预/天花板/方差实验/工程结论/方案可行性/闭环表/剩余不确定性/工件索引）。
- **`EVIDENCE.md`**（A–M 节）— **证据台账**：每条主张都指向具体数据文件与数字（我此前若干被否定的猜测也在其中，保留过程）。

## 参考配置（问题复现）
train dp4（卡 11–14）、`--num-workers 2`、`--prefetch-factor 4`、server 单副本（卡 10、dp1、端口 1132）、
数据 `open_perfectblend_qwen3_4b_700k`、`SAMP_SEED=42`、`--grad-accum 3`、`--total-seq-len 4096`。
每个 run 目录内含 `config.txt`（git HEAD、worker/prefetch、server DP、train DP、seed、数据路径）与 `diff.patch`（当时的代码改动）。

## 目录
| 路径 | 内容 |
|---|---|
| `exp/<RUN_ID>/` | 每个实验一次采集：`config.txt`/`diff.patch`/`run.log`/`train.log`/`metrics.jsonl`/`sys.jsonl`/`rank*/`（每 rank 的 load/timeline/ag/all-gather 与 worker 事件）/`analysis` 产物 |
| `logs/` | 启动链路日志与 vLLM server 日志 |
| `raw*/`,`p0*/`,`p0c/`,`analysis*` | 早期诊断采集、P0 前置门、缓冲/对齐分析产物 |
| `*.py` | 分析脚本（`analyze_rootcause.py` chain/cost/xcorr/queue、`analyze_ag_align.py`、`analyze_buffer.py`、`extract_steps.py`、`scrape_metrics.py`、`scrape_sys.py`、`build_band_subset.py`）与实验脚本（`run_exp.sh`、`run_E*.sh`、`serve_e1_*.sh`） |
| `snapshot/` | 插桩前 diff（`pre_instrumentation.patch`）、**含插桩的完整 diff**（`full_diff_with_instrumentation.patch`）、`distributed_batch_sampler.py.orig`、`revert.sh` |

## 数据政策（重要）
- **不含任何模型权重/checkpoint**（`checkpoints*`、`*.safetensors`、`*.pt`、`*.npy`、`_hs*`、`__pycache__` 均已排除）。
- 体量较大的原始采集（`*.jsonl`、`*.log`）**已 gzip**（`.gz`）。还原：`find . -name '*.gz' -exec gunzip -k {} \;` 或直接 `zcat x.jsonl.gz | head`。

## 插桩（默认关闭、可一键回退）
- `DL_PROFILE`（每步 load/timeline + worker 每样本耗时 + collate）、`AG_PROFILE`（FSDP2 all-gather 起止与 `wait_for_unshard`）、`DL_MAX_STEPS`、`DL_IN_ORDER`、`SAMP_SEED/SAMP_RANK_PERM/SAMP_BALANCE`。
- 回退：`cd dspark_project/speculators && bash <本目录>/snapshot/revert.sh`（或 `git checkout -- src/speculators/train/distributed_batch_sampler.py`；其余改动均为 env 默认关闭）。

## 结论速览（细节见 REPORT.md）
1. 周期性尖峰 = **集合同步放大器**（任一 rank 迟到 τ ⇒ 全体 fwd 尖峰 ≈τ；slope 0.97–0.98、R²=1.00）放大**每 rank 取数不稳**。
2. 取数不稳的判据是 **ρ = 需求(请求数/步) ÷ 能力(workers ÷ RTT)**：ρ≲0.90 稳；0.92–0.95 慢步 6–54%。
3. **server 天花板 = token 速率 ≈19–20k prompt tok/s**（≈29 docs/s）；与 token 预算、max_num_seqs、排队、存储、CPU/调度**均无关**（E4-C/E4-D/E7 逐一否定）。
4. 压请求尺寸方差（窄带语料）**只摊平受害者、不降低总量**；**在飞数**与**server 副本**才是提速与稳定的主杠杆。
