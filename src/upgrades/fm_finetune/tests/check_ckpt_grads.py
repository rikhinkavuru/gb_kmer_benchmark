#!/usr/bin/env python3
"""Torch check (run in .venv-ft): activation checkpointing of the DNABERT-2 encoder layers is a
pure memory/compute trade. With dropout ACTIVE (train mode) and a fixed seed, the loss and every
LoRA/head gradient must match the non-checkpointed model up to float error -- which also proves
the RNG state (dropout masks) is replayed in the recomputation. Also checks that only LoRA + head
parameters are trainable and that they receive non-zero gradients.

Run: HF_HUB_OFFLINE=1 nice -n 15 .venv-ft/bin/python src/upgrades/fm_finetune/tests/check_ckpt_grads.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
sys.path.insert(0, MOD)

import torch  # noqa: E402

import run_ft_v2 as rv  # noqa: E402


def grads(m, enc, y, seed):
    m.train_mode(True)
    for p in m.trainable():
        p.grad = None
    torch.manual_seed(seed)
    loss = torch.nn.functional.cross_entropy(m.logits(enc), y)
    loss.backward()
    return float(loss.detach()), [p.grad.detach().float().cpu().clone() for p in m.trainable()]


def main():
    torch.set_num_threads(2)
    dev = sys.argv[1] if len(sys.argv) > 1 else ("mps" if torch.backends.mps.is_available() else "cpu")
    rc = dict(rv.RECIPES["dnabert2"], grad_ckpt=False)
    m = rv.DnabertLoraWrap(42, dev, rc)
    n_base_trainable = sum(p.numel() for n, p in m.backbone.named_parameters()
                           if p.requires_grad and "lora_" not in n)
    assert n_base_trainable == 0, f"{n_base_trainable} non-LoRA backbone params are trainable"
    tr, _ = rv.load_split("human_enhancers_cohn")
    seqs, y, _ = rv.arm_train(tr, "original")
    pick = np.r_[np.where(y == 0)[0][:4], np.where(y == 1)[0][:4]]
    enc = m.encode([seqs[i] for i in pick])
    yt = torch.tensor(y[pick], dtype=torch.long, device=dev)
    # LoRA-B is zero-initialised -> LoRA-A grads are exactly 0 at step 0; take one SGD step so
    # every trainable tensor has a non-trivial gradient before comparing.
    l0, g0 = grads(m, enc, yt, 1)
    with torch.no_grad():
        for p, g in zip(m.trainable(), g0):
            p.add_(-1e-2 * g.to(p.device))
    la, ga = grads(m, enc, yt, 123)
    lb_same, gb_same = grads(m, enc, yt, 123)          # determinism baseline (no ckpt)
    n = rv.enable_layer_checkpointing(m.backbone)
    lb, gb = grads(m, enc, yt, 123)
    ld, gd = grads(m, enc, yt, 999)                    # different dropout masks -> must differ
    rel = max(float((a - b).norm() / (a.norm() + 1e-12)) for a, b in zip(ga, gb))
    rel_same = max(float((a - b).norm() / (a.norm() + 1e-12)) for a, b in zip(ga, gb_same))
    rel_diff = max(float((a - b).norm() / (a.norm() + 1e-12)) for a, b in zip(ga, gd))
    nz = sum(float(g.abs().sum()) > 0 for g in ga)
    print(f"device={dev} layers checkpointed={n} trainable tensors={len(ga)} (non-zero grad: {nz})")
    print(f"loss no-ckpt={la:.6f} ckpt={lb:.6f} other-seed={ld:.6f}")
    print(f"max rel grad diff: ckpt vs no-ckpt={rel:.2e}; no-ckpt rerun={rel_same:.2e}; "
          f"different dropout seed={rel_diff:.2e}")
    assert nz == len(ga), "some trainable tensor got a zero gradient"
    assert abs(la - lb) < 1e-4 and rel < 1e-3, "checkpointing changed the gradients"
    assert rel_diff > 10 * max(rel, 1e-6), "dropout seed has no effect -> test is vacuous"
    print("ok  activation checkpointing is gradient-exact (dropout RNG replayed)")


if __name__ == "__main__":
    main()
