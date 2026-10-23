#!/usr/bin/env python3
"""--pad-multiple must not change what either model computes (torch env, .venv-ft).

--pad-multiple K (memory fix of 2026-09-24, see memguard.py and run_ft_v2.set_dnabert_fixed_shapes)
pads each batch to a multiple of K tokens so that only a few tensor shapes ever reach the MPS
backend. The logits of the real tokens must be unchanged:
  HyenaDNA   causal + right-padded + masked mean-pool -> trailing pads cannot reach real tokens.
  DNABERT-2  pad rows are kept instead of unpadded; pad KEYS are still masked out of attention by
             the model's own -10000 additive mask, every other op is per-token, pool is masked.
Checks (eval mode, i.e. dropout off), on CPU and on MPS:
  (a) logits, old path (per-batch padding; DNABERT-2 unpadded) vs --pad-multiple path, 24
      sequences spanning the length range of the dataset the model is trained on;
  (b) a sequence's logits alone vs inside a padded batch;
  (c) DNABERT-2 only: gradients of every trainable (LoRA + head) tensor for a class-weighted CE
      loss on 16 sequences, old vs fixed-shape path;
  (d) both models: those gradients, old full-batch path vs fixed-shape path with --micro-batch
      (exact gradient accumulation, run_ft_v2.accumulate_batch), micro-batches of 3.
Pass criterion (fixed before running): max |logit diff| < 1e-3 and max relative grad diff < 1e-3.

Usage: HF_HUB_OFFLINE=1 bibm/lowcpu.sh .venv-ft/bin/python \
           src/upgrades/fm_finetune/tests/check_pad_invariance.py [--models hyenadna,dnabert2]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run_ft_v2 as R  # noqa: E402

PADK = {"hyenadna": 512, "dnabert2": 16}
DATA = {"hyenadna": "drosophila_enhancers_stark", "dnabert2": "human_enhancers_cohn"}


def set_path(m, model, k):
    R.PAD_MULTIPLE = k
    if model == "dnabert2":
        R.set_dnabert_fixed_shapes(m.backbone.base_model.model, bool(k))


def logits(m, model, seqs, k, chunk=8):
    """Logits in chunks of `chunk` sequences (each chunk padded as the path pads a batch)."""
    import torch
    set_path(m, model, k)
    m.train_mode(False)
    out, width = [], 0
    with torch.no_grad():
        for s in range(0, len(seqs), chunk):
            enc = m.encode(seqs[s:s + chunk])
            out.append(m.logits(enc).float().cpu().numpy())
            width = max(width, int(enc["input_ids"].shape[1]))
    return np.vstack(out), width


def grads(m, model, seqs, y, k, micro=0):
    """Gradients of the class-weighted CE (dropout off, autograd on); micro > 0 uses
    run_ft_v2.accumulate_batch (exact micro-batch accumulation) instead of one full batch."""
    import torch
    set_path(m, model, k)
    m.train_mode(False)
    w = torch.tensor([0.7, 1.3], dtype=torch.float32, device=m.device)
    rows = np.arange(len(seqs))
    if micro:
        R.MICRO_BATCH = micro
        opt = torch.optim.SGD(m.trainable(), lr=0.0)        # only used for zero_grad
        R.accumulate_batch(m, seqs, np.asarray(y), rows, w, opt, m.device)
        R.MICRO_BATCH = 0
    else:
        for p in m.trainable():
            p.grad = None
        yt = torch.tensor(y, dtype=torch.long, device=m.device)
        loss = torch.nn.functional.cross_entropy(m.logits(m.encode(seqs)), yt, weight=w)
        loss.backward()
    return [p.grad.detach().float().cpu().numpy().copy() for p in m.trainable()]


def rel_diff(g0, g1):
    return max(float(np.abs(x - z).max() / max(np.abs(x).max(), 1e-12)) for x, z in zip(g0, g1))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="hyenadna,dnabert2")
    a = ap.parse_args()
    torch.set_num_threads(2)
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    worst_l, worst_g = 0.0, 0.0
    for model in a.models.split(","):
        tr, _ = R.load_split(DATA[model])
        L = tr["sequence"].str.len().to_numpy()
        order = np.argsort(L, kind="stable")
        pick = order[np.linspace(0, len(order) - 1, 24).astype(int)]
        seqs = tr["sequence"].astype(str).to_numpy()[pick].tolist()
        y = tr["label"].to_numpy()[pick]
        for dev in devs:
            R.PAD_MULTIPLE = 0
            m = R.build_model(model, 42, dev)
            a0, l0 = logits(m, model, seqs, 0)
            a1, l1 = logits(m, model, seqs, PADK[model])
            alone = np.vstack([logits(m, model, [s], PADK[model])[0] for s in seqs[:6]])
            d_path = float(np.abs(a0 - a1).max())
            d_alone = float(np.abs(a1[:6] - alone).max())
            worst_l = max(worst_l, d_path, d_alone)
            msg = (f"{model:8s} {dev}: logits old path ({l0} tok) vs pad-to-{PADK[model]} ({l1} tok) "
                   f"max|d|={d_path:.2e}; alone vs batched max|d|={d_alone:.2e}; "
                   f"|logit| max {np.abs(a0).max():.3f}")
            # gradients: shortest 8 sequences for HyenaDNA (drosophila 3 kb x 8 with grads is
            # the memory limit under the MPS cap), 16 for DNABERT-2
            gs = seqs[:8] if model == "hyenadna" else seqs[:16]
            gy = list(y[:len(gs)])
            g0 = grads(m, model, gs, gy, 0)
            nz = sum(float(np.abs(x).max()) > 0 for x in g0)
            if model == "dnabert2":
                rel = rel_diff(g0, grads(m, model, gs, gy, PADK[model]))
                worst_g = max(worst_g, rel)
                msg += f"; grads ({len(g0)} tensors, {nz} non-zero) old vs fixed max rel diff={rel:.2e}"
            relm = rel_diff(g0, grads(m, model, gs, gy, PADK[model], micro=3))
            worst_g = max(worst_g, relm)
            msg += f"; full batch vs micro-batches of 3 (exact accumulation) max rel diff={relm:.2e}"
            print(msg, flush=True)
            set_path(m, model, 0)
            del m
            if dev == "mps":
                torch.mps.empty_cache()
    ok = worst_l < 1e-3 and worst_g < 1e-3
    print(("PASS" if ok else "FAIL") + f": worst max|logit diff| = {worst_l:.2e}, worst relative grad "
          f"diff = {worst_g:.2e} (criteria < 1e-3)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
