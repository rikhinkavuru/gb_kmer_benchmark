#!/usr/bin/env python3
"""Cache each task's ORIGINAL split (via _common.load_original -> the patched, lock-protected,
count-checked data.load_dataset; seed 42) as one pickle under external/alt_model/seqcache/<task>.pkl,
so the alt-model workers do not re-read ~290k tiny Genomic-Benchmarks files per process.
Pure I/O; the cached object is exactly (train_seqs, y_train, test_seqs, y_test) as returned by
_common.load_original(task).

DATA INTEGRITY (2026-09-26 rebuild). The first cache (2026-09-23 21:58) was written BEFORE the
22:15 data.py integrity fix, so every pickle is (re)validated here:
  * a pickle is REUSED only if its mtime is after the fix (FIX_EPOCH) AND its train/test counts equal
    results/results.csv (GB) / results/results_nt.csv (NT) n_train/n_test for the LightGBM,
    selected_by_val=1 row (the counts behind every number in the paper); otherwise it is rebuilt;
  * load_cached() re-asserts those counts on EVERY load and raises on mismatch;
  * a manifest (seqcache/manifest.json) records counts, class counts and a sha1 of sequences+labels.
"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402

TASKS = ["demo_human_or_worm", "human_ensembl_regulatory", "human_nontata_promoters",
         "demo_coding_vs_intergenomic_seqs", "human_enhancers_ensembl", "dummy_mouse_enhancers_ensembl",
         "human_enhancers_cohn", "human_ocr_ensembl", "drosophila_enhancers_stark",
         "nt_promoter_all", "nt_promoter_tata", "nt_promoter_no_tata", "nt_enhancers",
         "nt_enhancers_types", "nt_splice_sites_all"]
CACHE = os.path.join(C.ROOT, "external", "alt_model", "seqcache")
MANIFEST = os.path.join(CACHE, "manifest.json")
# 2026-09-23 22:15 EDT (= 02:15 UTC 09-24; epoch 1790216100): data.py lock + completeness + count-check fix.
FIX_EPOCH = 1790216100


def expected_counts():
    """{task: (n_train, n_test)} from the released benchmark CSVs (lgbm, selected_by_val=1)."""
    import pandas as pd
    out = {}
    for f in ("results.csv", "results_nt.csv"):
        df = pd.read_csv(os.path.join(C.ROOT, "results", f))
        df = df[(df["model"] == "lgbm") & (df["selected_by_val"] == 1)]
        for _, r in df.iterrows():
            out[r["dataset"]] = (int(r["n_train"]), int(r["n_test"]))
    return out


def cache_path(task):
    return os.path.join(CACHE, f"{task}.pkl")


def fingerprint(obj):
    tr, ytr, te, yte = obj
    h = hashlib.sha1()
    for seqs, y in ((tr, ytr), (te, yte)):
        h.update("\n".join(seqs).encode())
        h.update(np.asarray(y, np.int64).tobytes())
    return h.hexdigest()


def _check(task, obj, exp):
    tr, ytr, te, yte = obj
    got = (len(tr), len(te))
    if got != exp[task] or len(ytr) != len(tr) or len(yte) != len(te):
        raise RuntimeError(f"{task}: counts {got} != released {exp[task]}")


def load_cached(task):
    """(tr, ytr, te, yte) from the validated pickle cache (falls back to _common.load_original).
    Counts are asserted against results/results{,_nt}.csv on every call."""
    exp = expected_counts()
    p = cache_path(task)
    if os.path.exists(p) and os.path.getmtime(p) > FIX_EPOCH:
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
    else:
        obj = C.load_original(task, C.SEED)
    _check(task, obj, exp)
    return obj


def main():
    os.makedirs(CACHE, exist_ok=True)
    exp = expected_counts()
    tasks = sys.argv[1].split(",") if len(sys.argv) > 1 else TASKS
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    for t in tasks:
        p = cache_path(t)
        old_fp = None
        if os.path.exists(p):
            with open(p, "rb") as fh:
                old = pickle.load(fh)
            ok_counts = (len(old[0]), len(old[2])) == exp[t]
            if os.path.getmtime(p) > FIX_EPOCH and ok_counts and t in man:
                print(f"{t}: cached (post-fix, counts OK)", flush=True)
                continue
            old_fp = fingerprint(old) if ok_counts else f"BAD_COUNTS {(len(old[0]), len(old[2]))}"
        t0 = time.time()
        obj = C.load_original(t, C.SEED)
        obj = (list(obj[0]), np.asarray(obj[1], np.int64), list(obj[2]), np.asarray(obj[3], np.int64))
        _check(t, obj, exp)
        with open(p + ".tmp", "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(p + ".tmp", p)
        fp = fingerprint(obj)
        man[t] = dict(n_train=len(obj[0]), n_test=len(obj[2]),
                      train_class_counts=np.bincount(obj[1]).tolist(),
                      test_class_counts=np.bincount(obj[3]).tolist(), sha1=fp,
                      stale_pre_fix_cache_sha1=old_fp,
                      stale_cache_identical=(None if old_fp is None else old_fp == fp),
                      built=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(MANIFEST + ".tmp", "w") as fh:
            json.dump(man, fh, indent=1)
        os.replace(MANIFEST + ".tmp", MANIFEST)
        print(f"{t}: n_train={len(obj[0])} n_test={len(obj[2])} sha1={fp[:12]} "
              f"(stale cache identical: {man[t]['stale_cache_identical']}) ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
