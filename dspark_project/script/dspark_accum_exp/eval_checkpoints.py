#!/usr/bin/env python3
"""Overfitting / degradation check for accum12 runs on the 50k crop.

For each (run, epoch-checkpoint) load the DSpark checkpoint and measure:
  exp1  accept_len on a fixed 1000-doc TRAIN subset        (train1000)
  exp2  accept_len + tv-loss on the ORIGINAL val 5k        (val)
        draft-entropy (nats/token over draft slots) on 100 val  (val100)
  exp3  accept_len on 1000 docs OUTSIDE the 5w crop        (out1000)

accept_len/tv are the same analytic metrics logged during training. draft
entropy is added at eval time by monkeypatching compute_metrics (no source
change): entropy of the Markov-corrected draft softmax over draft slots.

Usage (hidden-state server on card15 must be running):
  ASCEND_RT_VISIBLE_DEVICES=<card> python eval_checkpoints.py \
      --runs accum12 gqa_ctx_l0_a12 --epochs 0 1 2
"""
import argparse
import csv
import gc
import sys
from pathlib import Path

sys.path.insert(0, "/home/y50063564/dspark_project/speculators/src")

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from safetensors.torch import load_file  # noqa: E402
from speculators.models.dspark.config import DSparkSpeculatorConfig  # noqa: E402
from speculators.models.dspark.core import DSparkDraftModel  # noqa: E402
from speculators.models.metrics import resolve_loss_config  # noqa: E402
from speculators.train.data import ArrowDataset  # noqa: E402
from speculators.train.dataloader import _setup_dataloader  # noqa: E402
from speculators.train.utils import normalize_counted_metrics  # noqa: E402

# ---- entropy: monkeypatch compute_metrics so every forward also reports the
#      mean draft entropy (nats/token) over draft (non-anchor) slots. -------
import speculators.models.dspark.core as _core  # noqa: E402

_orig_compute = _core.compute_metrics


def _compute_with_entropy(logits, targets, confidence_logits, loss_mask,
                          block_size, *args, **kwargs):
    loss, metrics = _orig_compute(logits, targets, confidence_logits, loss_mask,
                                  block_size, *args, **kwargs)
    with torch.no_grad():
        lf = logits.float()
        logp = torch.log_softmax(lf, dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1)               # [1, T]
        nb = logits.shape[1] // block_size
        slot = loss_mask.to(ent.dtype).view(nb, block_size)[:, 1:]
        metrics["draft_entropy_sum"] = (ent.view(nb, block_size)[:, 1:] * slot).sum()
        metrics["draft_entropy_total"] = slot.sum().clamp_min(1.0)
    return loss, metrics


_core.compute_metrics = _compute_with_entropy

VERIFIER = "/home/y50063564/Qwen3-4B"
VLLM = "http://80.48.17.178:1123/v1"
ROOT = Path("/home/y50063564/processed_data/dspark_data/dspark_accum")
EVAL = Path("/home/y50063564/processed_data/dspark_data/dspark_eval")
LOSS_CFG = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')
DEVICE = "npu:0"

_model = None           # current model under evaluation


def build_model(ckpt_dir: Path):
    cfg = DSparkSpeculatorConfig.from_pretrained(str(ckpt_dir))
    cfg.transformer_layer_config._attn_implementation = "sdpa"
    model = DSparkDraftModel(cfg)
    model.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")), strict=False)
    model.load_verifier_weights()
    model.to(DEVICE).eval()
    return model


def make_loader(subset: str):
    assert _model is not None
    hidden = _model.config.transformer_layer_config.hidden_size
    ds = ArrowDataset(
        max_len=4096,
        datapath=str(EVAL / subset),
        vllm_endpoint=VLLM,
        on_missing="generate",
        on_generate="cache",          # generate once, reused across ckpts/runs
        split_ratio=1.0,
        hidden_states_dtype=torch.bfloat16,
        model=VERIFIER,
        request_timeout=900,
        max_retries=5,
    )
    return _setup_dataloader(ds, total_seq_len=4096, hidden_size=hidden,
                             num_workers=4, num_target_layers=5,
                             prefetch_factor=2, preprocess=None)


def eval_subset(subset: str) -> dict:
    loader = make_loader(subset)
    sums: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            gpu = {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor)
                       else v) for k, v in batch.items()}
            _, _loss, metrics = _model(
                **gpu, loss_config=LOSS_CFG, max_anchors=256,
                confidence_head_alpha=1.0,
            )
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v.item()
    return normalize_counted_metrics(sums, world_size=1)


def main():
    global _model
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["accum12", "gqa_ctx_l0_a12"])
    ap.add_argument("--epochs", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--subsets", nargs="+",
                    default=["train1000", "val", "val100", "out1000"])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "overfit_check.csv")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        for ep in args.epochs:
            ckpt = ROOT / run / "checkpoints" / str(ep)
            print(f"\n== {run}  epoch {ep} ==", flush=True)
            _model = build_model(ckpt)
            res = {"run": run, "epoch": ep}
            WANT = {"train1000": ["accept_len"], "val": ["accept_len", "tv_loss"],
                    "val100": ["draft_entropy", "accept_len"],
                    "out1000": ["accept_len"]}
            for subset in args.subsets:
                want = WANT[subset]
                m = eval_subset(subset)
                line = f"  [{subset}] "
                for k in want:
                    v = m.get(k, float("nan"))
                    res[f"{k}_{subset}"] = v
                    line += f"{k}={v:.4f}  "
                print(line, flush=True)
            rows.append(res)

            del _model
            _model = None
            gc.collect()
            torch.npu.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
