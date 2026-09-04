"""干净的 accept_len vs epoch 图: baseline / l0_5l(完成3ep) / ctxonly(1ep)。
横轴固定 0-3 epoch。纵轴 accept_len(滚动均值平滑)。
用法: python3 scripts/plot_acceptlen_epoch.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [
    ("baseline", "/home/y50063564/eval/baseline_qwen3_4b_70w/train_logs.log", 9410.0, "#1f77b4"),
    ("l0_5l", "/tmp/l0_5l_full.log", 127728.0, "#d62728"),
    ("ctxonly", "/tmp/ctxonly_full.log", 127727.0, "#2ca02c"),
]
OUT = Path("/home/y50063564/processed_data/dspark_data/attn_plots/acceptlen_3way_epoch")
OUT.mkdir(parents=True, exist_ok=True)

def parse_acc(log):
    out = []
    cur = None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"train/accept_len=([0-9.]+)", line)
        g = re.search(r"global_step=(\d+)", line)
        if m:
            cur = float(m.group(1))
        elif g and cur is not None:
            out.append((int(g.group(1)), cur))
            cur = None
    return out

def smooth(vals, w=401):
    o = []
    for i in range(len(vals)):
        lo, hi = max(0, i - w // 2), min(len(vals), i + w // 2 + 1)
        o.append(sum(vals[lo:hi]) / (hi - lo))
    return o

fig, ax = plt.subplots(figsize=(10, 6))
for name, log, spe, color in RUNS:
    data = parse_acc(log)
    data = sorted(data)
    xs = [s / spe for s, _ in data]
    ys = smooth([v for _, v in data])
    ax.plot(xs, ys, color=color, lw=1.8, label=name)
    ax.plot([s / spe for s, _ in data], [v for _, v in data], color=color, alpha=0.15, lw=0.5)
    print(f"{name}: {len(data)} 步, 到 epoch {max(xs):.2f}")

ax.axhline(4.758, color="gray", ls=":", lw=1, label="baseline val=4.758")
ax.set_xlim(0, 3.0)
ax.set_xlabel("epochs")
ax.set_ylabel("accept_len")
ax.set_title("accept_len vs epochs (raw thin + smoothed)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "acceptlen_epoch_clean.png", dpi=130)
plt.close(fig)
print(f"saved {OUT}/acceptlen_epoch_clean.png")
