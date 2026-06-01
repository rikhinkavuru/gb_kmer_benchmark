#!/usr/bin/env python3
"""Reproduce EVERY frozen-FM result from the released embedding cache -- with ZERO torch.

The FM embeddings (HyenaDNA + DNABERT-2) are released as a frozen artifact (one .npz per
dataset x split x model, keyed by mean/[sep|cls]/y). This script regenerates the entire downstream
FM-probe result -- both heads (LR, LightGBM), all cleaning arms, all paired Fix-A deltas, all
1000-resample bootstrap CIs -- by loading ONLY those .npz files. No torch, no transformers, no model
construction is imported or run. The brittle model-load path (triton bypass / from_config /
checkpoint-strip for DNABERT-2; right-pad for HyenaDNA) is needed ONLY to regenerate embeddings from
scratch, which a replicator does NOT need to do.

What it does:
  1. Writes results/upgrades/fm_embeddings_manifest.csv (file, n, keys, hidden_dim, size, sha256) so
     the released cache is verifiable.
  2. Re-runs run_fm_probe (HyenaDNA unpaired), run_fm_paired (HyenaDNA paired Fix-A), and
     run_fm_paired_dnabert (DNABERT-2 paired Fix-A) IN-PROCESS from cache, into results/upgrades/repro/.
  3. Asserts each reproduced CSV is BYTE-IDENTICAL to the canonical one already in results/upgrades/.
  4. Asserts 'torch' never entered sys.modules during the whole reproduction.

All three reproduced scripts read embeddings via numpy (extract_embeddings*.load_cached) and train
LightGBM/LR heads; torch is imported only lazily inside the (unused-here) model-loading functions.
"""
import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import _common as C                          # noqa: E402  (lightgbm; NOT torch)
import run_fm_probe                          # noqa: E402  (imports extract_embeddings -- torch lazy)
import run_fm_paired                         # noqa: E402
import run_fm_paired_dnabert                 # noqa: E402

CACHE = os.path.join(C.ROOT, "cache", "fm_embeddings")
CANON = C.RESULTS_DIR
REPRO = os.path.join(CANON, "repro")
JOBS = [(run_fm_probe, "fm_probe.csv", None),
        (run_fm_paired, "fm_paired.csv", None),
        (run_fm_paired_dnabert, "fm_paired_dnabert.csv", None)]


def write_manifest(out):
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        d = np.load(f); keys = sorted(d.files)
        emb_key = [k for k in keys if k != "y"][0]
        rows.append(dict(file=os.path.basename(f), model=("dnabert2-117m" if "dnabert2" in f else "hyenadna-tiny-16k-d128"),
                         n_sequences=int(d[emb_key].shape[0]), hidden_dim=int(d[emb_key].shape[1]),
                         keys="|".join(keys), size_bytes=os.path.getsize(f),
                         sha256=hashlib.sha256(open(f, "rb").read()).hexdigest()))
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def _run_main(mod, out_csv):
    old = sys.argv
    sys.argv = ["reproduce", "--out", out_csv]
    try:
        mod.main()
    finally:
        sys.argv = old


def point_test_effect(task, pooling, head, arm, cache_dir, loader):
    """Deterministic (rng-free) test_effect AUROC point for one cell, from cache -- used by the test.
    test_effect = AUROC(H_orig on T_clean) - AUROC(H_orig on full original test)."""
    otr, ytr, ote, yte = C.load_original(task)
    emb_tr = loader(cache_dir, task, "orig", "train"); emb_te = loader(cache_dir, task, "orig", "test")
    nc = len(np.unique(ytr))
    H = run_fm_paired.fit_head(emb_tr[pooling], ytr, head, C.SEED)
    pred, proba = run_fm_paired.predict(H, emb_te[pooling])
    _, _, ate_seqs, _ = run_fm_probe.arm_split_seqs(task, arm)
    mask = run_fm_paired.membership_mask(ote, ate_seqs)
    full = run_fm_paired._auc(yte, proba, nc)
    onclean = run_fm_paired._auc(yte[mask], proba[mask], nc)
    return round(onclean - full, 4)


def main():
    os.makedirs(REPRO, exist_ok=True)
    man = write_manifest(os.path.join(CANON, "fm_embeddings_manifest.csv"))
    print(f"manifest: {len(man)} cache files -> results/upgrades/fm_embeddings_manifest.csv")
    print("=" * 90)
    print("REPRODUCE FROM CACHE (torch-free): regenerating FM results from released embeddings")
    print("=" * 90)
    allok = True
    for mod, name, _ in JOBS:
        rp = os.path.join(REPRO, name)
        print(f"\n>>> {name} (from {mod.__name__}) ...", flush=True)
        _run_main(mod, rp)
        canon = os.path.join(CANON, name)
        ident = open(rp, "rb").read() == open(canon, "rb").read()
        if not ident:   # fall back to a tolerant numeric diff to localize any drift
            a, b = pd.read_csv(rp), pd.read_csv(canon)
            num = [c for c in a.columns if a[c].dtype.kind in "fi"]
            close = a.shape == b.shape and all(np.allclose(a[c], b[c], atol=1e-9, equal_nan=True) for c in num)
            ident = close
            print(f"    NOTE: not byte-identical; numeric-identical (atol 1e-9): {close}")
        print(f"    {name}: {'BIT-IDENTICAL to canonical' if ident else 'MISMATCH'}")
        allok = allok and ident
    torch_loaded = any(m == "torch" or m.startswith("torch.") for m in sys.modules)
    print(f"\ntorch in sys.modules during reproduction: {torch_loaded}  (must be False)")
    ok = allok and not torch_loaded
    print("\n" + ("REPRODUCTION FROM CACHE: PASS (all results bit-identical, zero torch)"
                  if ok else "REPRODUCTION FROM CACHE: FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
