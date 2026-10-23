#!/usr/bin/env python3
"""Torch-free checks for the reviewer-response fine-tuning pipeline (prepare_splits_v2.py,
run_ft_v2.py batching, eval_ft_v2.py metrics). Nothing here loads a model.

  1. the materialized split files reproduce data.load_dataset() exactly (order, sequence, label)
     -- guards against the partial-download incident of 2026-09-23;
  2. every arm's membership equals its cleaned CSV as a (sequence, label) multiset, positives are
     identical across arms, and the RANDOM arm is an exact size match to comp_equalized, drawn
     from the original negatives with the documented RandomState(42) (train draw, then test);
  3. the index hashes equal the manifest (results/upgrades/fm_finetune_v2_split_manifest.csv);
  4. the fast metrics in eval_ft_v2 equal sklearn (AUROC with ties, AP, balanced acc., MCC);
  5. the training batcher covers every fit position exactly once, mixes classes, is seeded, and
     never emits a size-1 batch (DNABERT-2's squeeze() would break on one).

Run (torch-free venv):  bibm/lowcpu.sh <venv>/bin/python src/upgrades/fm_finetune/tests/test_ft_v2.py
Pass --fast to skip the (slower) dataset reload of check 1.
"""
import collections
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(MOD, "..", "..", ".."))
for p in (MOD, ROOT, os.path.join(ROOT, "src", "upgrades")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_ft_v2 as ev          # noqa: E402
import prepare_splits_v2 as ps   # noqa: E402
import run_ft_v2 as rv           # noqa: E402  (torch is imported lazily, never here)

DATASETS = ("human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark")
EXPECTED = {"human_enhancers_cohn": (20843, 6948), "nt_enhancers": (14968, 400),
            "drosophila_enhancers_stark": (5184, 1730)}


def load_files(ds):
    return {sp: pd.read_csv(os.path.join(ps.OUT_DIR, f"{ds}__{sp}.csv.gz"))
            for sp in ("train", "test")}


def test_files_match_loader(ds):
    import data as gbdata
    d = gbdata.load_dataset(ds, seed=42)
    f = load_files(ds)
    assert (len(d["train_seqs"]), len(d["test_seqs"])) == EXPECTED[ds], "loader counts differ"
    for sp, seqs, y in (("train", d["train_seqs"], d["y_train"]),
                        ("test", d["test_seqs"], d["y_test"])):
        df = f[sp]
        assert len(df) == len(seqs)
        assert np.array_equal(df["idx"].to_numpy(), np.arange(len(seqs)))
        assert df["sequence"].astype(str).tolist() == list(seqs), f"{ds}/{sp}: sequences differ"
        assert np.array_equal(df["label"].to_numpy(), np.asarray(y)), f"{ds}/{sp}: labels differ"
    print(f"ok  {ds}: split files == data.load_dataset() (order, sequences, labels)")


def test_arm_membership(ds):
    f = load_files(ds)
    man = pd.read_csv(ps.MANIFEST)
    rng = np.random.RandomState(ps.RANDOM_SEED)
    for sp in ("train", "test"):              # the random draw consumes train first, then test
        df = f[sp]
        y = df["label"].to_numpy()
        pos = set(np.where(y == 1)[0])
        mem = {}
        for arm in ps.ARMS:
            col = df[f"{arm}_pos"].to_numpy()
            sel = np.where(col >= 0)[0]
            order = sel[np.argsort(col[sel])]
            assert np.array_equal(np.sort(col[sel]), np.arange(len(sel))), f"{arm}: positions"
            mem[arm] = order
            assert set(order[y[order] == 1]) == pos, f"{ds}/{sp}/{arm}: positives differ"
            h = hashlib.sha256(np.sort(order).astype(np.int64).tobytes()).hexdigest()[:16]
            row = man[(man.dataset == ds) & (man.arm == arm) & (man.split == sp)].iloc[0]
            assert h == row["idx_sha256_16"] and len(order) == row["n"], f"{ds}/{sp}/{arm}: hash"
            if arm in ps.CLEAN_STEM:
                c = pd.read_csv(os.path.join(ps.SPLITS_V2, f"{ds}_{ps.CLEAN_STEM[arm]}_{sp}.csv"))
                want = collections.Counter(zip(c["sequence"].astype(str).str.upper(),
                                               c["label"].astype(int)))
                got = collections.Counter(zip(df["sequence"].iloc[order].astype(str),
                                              df["label"].iloc[order].astype(int)))
                assert want == got, f"{ds}/{sp}/{arm}: membership != cleaned CSV multiset"
                # list order reproduces the cleaned CSV row order (labels and sequences)
                assert df["sequence"].iloc[order].astype(str).tolist() == \
                    c["sequence"].astype(str).str.upper().tolist(), f"{arm}: row order"
        n_ce = int((y[mem["comp_equalized"]] == 0).sum())
        assert int((y[mem["random"]] == 0).sum()) == n_ce and len(mem["random"]) == \
            len(mem["comp_equalized"]), "random arm is not size-matched to comp_equalized"
        redo = ps.random_negative_draw(y, n_ce, rng)
        assert np.array_equal(redo, np.sort(mem["random"])), "random draw not reproducible"
        print(f"ok  {ds}/{sp}: arms == cleaned CSVs, positives shared, random arm size-matched "
              f"({n_ce} neg) and reproducible, hashes == manifest")


def test_metrics():
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 matthews_corrcoef, roc_auc_score)
    rng = np.random.RandomState(0)
    for trial in range(20):
        n = rng.randint(30, 400)
        y = (rng.rand(n) < rng.uniform(0.05, 0.95)).astype(np.int64)
        if y.min() == y.max():
            continue
        p = np.round(rng.rand(n) * rng.choice([1, 4, 20, 1000]), 0) / 1000.0   # heavy ties
        p = np.clip(p + 0.3 * y * rng.rand(n), 0, 1)
        m = ev.all_metrics(y, p)
        pred = (p >= 0.5).astype(int)
        assert abs(m["auroc"] - roc_auc_score(y, p)) < 1e-10
        assert abs(m["ap"] - average_precision_score(y, p)) < 1e-10, (m["ap"],
                                                                       average_precision_score(y, p))
        assert abs(m["bal_acc"] - balanced_accuracy_score(y, pred)) < 1e-10
        assert abs(m["mcc"] - matthews_corrcoef(y, pred)) < 1e-10
    d = ev.balanced_draws(np.array([0] * 10 + [1] * 90))
    assert len(d) == ev.N_BAL and all(len(x) == 20 for x in d)
    print("ok  eval_ft_v2 metrics == sklearn (AUROC w/ ties, AP, balanced accuracy, MCC)")


def test_batcher():
    y = np.array([0] * 500 + [1] * 500)            # class-sorted, as the real lists are
    seqs = ["A" * 500] * 1000                       # equal lengths -> ties everywhere
    fit = np.arange(1000)
    for n_fit in (1000, 993, 961):                  # 961 = 30*32 + 1 -> would leave a size-1 batch
        f = fit[:n_fit]
        ch = rv.mixed_batches(seqs, f, 32, np.random.RandomState(1))
        flat = sorted(i for c in ch for i in c)
        assert flat == list(range(n_fit)), "batches do not cover every fit position exactly once"
        assert min(len(c) for c in ch) >= 2, "size-1 batch"
    ch = rv.mixed_batches(seqs, fit, 32, np.random.RandomState(1))
    single = sum(len(set(y[fit[c]])) == 1 for c in ch)
    assert single <= 1, f"{single} single-class batches out of {len(ch)}"
    a = rv.mixed_batches(seqs, fit, 32, np.random.RandomState(7))
    b = rv.mixed_batches(seqs, fit, 32, np.random.RandomState(7))
    assert a == b, "batcher not deterministic under a fixed seed"
    e = rv.eval_batches(["A" * (i % 7 + 1) for i in range(129)], 64)
    assert min(len(c) for c in e) >= 2 and sorted(i for c in e for i in c) == list(range(129))
    print("ok  batcher: full coverage, mixed-class batches, seeded, no size-1 batches")


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    for ds in DATASETS:
        if not fast:
            test_files_match_loader(ds)
        test_arm_membership(ds)
    test_metrics()
    test_batcher()
    print("all fm_finetune v2 checks passed")
