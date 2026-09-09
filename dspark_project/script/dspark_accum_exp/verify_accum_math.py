#!/usr/bin/env python3
"""Pure-CPU numerical check of the grad_accum semantics added to trainer.py.

The trainer's accumulation (option B) divides every micro-batch loss by
``grad_accum`` before backward and takes one AdamW step after ``grad_accum``
micro-batches.  Claim: for equal-size micro-batches this produces *exactly* the
same gradient (and thus the same parameter update) as pooling the
``grad_accum`` micro-batches into one big batch and taking the mean loss over
it — which is the DDP(dp=grad_accum) all-reduce-mean semantics.

We replicate the exact code path (loss/accum -> backward accumulate -> clip ->
step) versus a big-batch reference and compare gradients + post-step params.

Run:  python verify_accum_math.py
"""
import torch

torch.manual_seed(0)

D_IN, D_H, D_OUT = 8, 16, 3
ACCUM = 12
N_BATCH = ACCUM          # micro-batches
BATCH_SAMPLES = 7        # samples per micro-batch (equal size => exact identity)

def make_net():
    return torch.nn.Sequential(
        torch.nn.Linear(D_IN, D_H),
        torch.nn.SiLU(),
        torch.nn.Linear(D_H, D_OUT),
    )

def clone_state(net):
    return [p.detach().clone() for p in net.parameters()]

def set_state(net, state):
    with torch.no_grad():
        for p, s in zip(net.parameters(), state):
            p.copy_(s)

# Fix one stream of micro-batches and targets.
batches = [torch.randn(BATCH_SAMPLES, D_IN) for _ in range(N_BATCH)]
targets = [torch.randn(BATCH_SAMPLES, D_OUT) for _ in range(N_BATCH)]

# Shared initial weights for both subjects (comparability).
template = clone_state(make_net())

def run_ref():
    """Reference: one big batch (all micro-batches concatenated), mean loss."""
    net = make_net()
    set_state(net, template)   # identical init to the accum subject
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    X = torch.cat(batches)
    Y = torch.cat(targets)
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(net(X), Y)      # mean over ALL samples
    loss.backward()
    grad = [p.grad.detach().clone() for p in net.parameters()]
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return net, grad

def run_accum():
    """Subject: grad_accum path (mirrors the trainer.py edit)."""
    net = make_net()
    set_state(net, template)   # identical init to the reference
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    opt.zero_grad()
    for (xb, yb) in zip(batches, targets):
        loss = torch.nn.functional.mse_loss(net(xb), yb)   # mean over THIS micro-batch
        loss = loss / ACCUM                                # trainer: loss / grad_accum
        loss.backward()                                    # accumulate into .grad
    grad = [p.grad.detach().clone() for p in net.parameters()]
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return net, grad

ref_net, ref_grad = run_ref()
acc_net, acc_grad = run_accum()

gmax = max((g - r).abs().max().item() for g, r in zip(acc_grad, ref_grad))
pmax = max(
    (a.detach() - b).abs().max().item()
    for a, b in zip(acc_net.parameters(), ref_net.parameters())
)
print(f"max |grad_accum - grad_pooled|  = {gmax:.3e}")
print(f"max |param_accum - param_pooled| = {pmax:.3e}")

assert gmax < 1e-5, "gradient mismatch: accum-mean != pooled mean"
assert pmax < 1e-5, "parameter mismatch after one step"
print("PASS: grad_accum (loss/accum) == pooled big-batch mean  =>  equals DDP(dp=accum) semantics.")
