#!/usr/bin/env python3
"""Materialize every fine-tuning arm as index membership over the ORIGINAL split (torch-free).

Why: the torch fine-tuning env (.venv-ft) cannot import genomic_benchmarks / pyarrow, and the
reviewer-response runs need one more arm (a size-matched RANDOM-negative control) plus
per-sequence test predictions that can be re-scored on any arm. Every arm's test set is a
subset of the ORIGINAL test set (positives identical, negatives subsampled; verified below),
so one prediction per original test sequence scores a model on every arm.

Arms (train and test built separately; positives are ALWAYS the full original positives):
  original        released split
  gc_match        cleaned_splits_v2/<task>_gcmatch_{train,test}.csv  (25-bin GC histogram match)
  comp_equalized  cleaned_splits_v2/<task>_comp_equalized_{train,test}.csv (GC + 16-dinucleotide
                  propensity match; the strongest cleaning; ~94%/85% positive by construction)
  random          NEW CONTROL. Negatives drawn uniformly at random WITHOUT replacement from the
                  original negatives of that split, to EXACTLY the comp_equalized arm's negative
                  count in that split (cohn: 399 train / 228 test; nt_enhancers: 343 / 36). No
                  composition criterion. One fixed draw, numpy RandomState(42): train draw
                  first, then test draw from the same generator. It has the same size and class
                  imbalance as comp_equalized but keeps the original composition gap, so it
                  separates "shortcut removed" from "fewer / imbalanced negatives".

Datasets: human_enhancers_cohn and nt_enhancers (composition-UNALIGNED: the released negatives
differ from the positives in GC/dinucleotide composition) and drosophila_enhancers_stark, the
paper's composition-ALIGNED clean control (the equalizer keeps 42.9% of its negatives vs 4.5-4.9%
for the contaminated sets). drosophila was added after the first two were materialized, via
--datasets, so the cohn/nt files the running jobs read were not rewritten (the draw is per dataset,
RandomState(42) re-created for each, so the order of building does not matter).

Cleaned-split rows are mapped back to original indices by exact sequence within (split, label);
duplicated sequences (present in nt_enhancers) are matched as a multiset, so counts are exact and
the matched sequences are identical strings (which index a duplicate maps to is immaterial).

Also written, as a sanity readout of what each arm leaves for a shortcut learner: a
composition-only baseline (standardized logistic regression on the 20-d mono+dinucleotide
signature, class_weight=balanced, fit on the arm's train, AUROC on the arm's test) and the GC
AUROC on the arm's test. Expectation fixed in advance: random ~= original >> comp_equalized ~ 0.5.

Outputs
  external/fm_finetune/splits/<task>__{train,test}.csv.gz   idx,label,sequence,<arm>_pos
      (<arm>_pos = position in that arm's list, -1 = not a member; gc/comp order = the cleaned
       CSV row order, so the validation carve reproduces run_fm_finetune.py's for a given seed)
  results/upgrades/fm_finetune_v2_split_manifest.csv

Usage (torch-free venv):
  python src/upgrades/fm_finetune/prepare_splits_v2.py [--datasets a,b]   (manifest rows of the
      datasets not rebuilt are kept)
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT, os.path.join(ROOT, "src", "upgrades")):
    if p not in sys.path:
        sys.path.insert(0, p)

DATASETS = ("human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark")
ARMS = ("original", "gc_match", "comp_equalized", "random")
CLEAN_STEM = {"gc_match": "gcmatch", "comp_equalized": "comp_equalized"}
SPLITS_V2 = os.path.join(ROOT, "cleaned_splits_v2")
OUT_DIR = os.path.join(ROOT, "external", "fm_finetune", "splits")
MANIFEST = os.path.join(ROOT, "results", "upgrades", "fm_finetune_v2_split_manifest.csv")
RANDOM_SEED = 42


def map_rows_to_index(clean_seqs, clean_y, orig_seqs, orig_y):
    """Multiset-match cleaned rows to original indices within the same label.
    Returns an int array (len = len(clean_seqs)) of original indices, in cleaned-row order."""
    pools = {}
    for i, (s, y) in enumerate(zip(orig_seqs, orig_y)):
        pools.setdefault((s, int(y)), []).append(i)
    for k in pools:
        pools[k].reverse()                       # pop() from the end -> lowest index first
    out = np.empty(len(clean_seqs), dtype=np.int64)
    for j, (s, y) in enumerate(zip(clean_seqs, clean_y)):
        key = (s, int(y))
        if not pools.get(key):
            raise AssertionError(f"cleaned row {j} (label {y}) not found in the original split")
        out[j] = pools[key].pop()
    return out


def random_negative_draw(y, n_neg, rng):
    """All positives + n_neg negatives drawn uniformly without replacement. Original order."""
    neg = np.where(y == 0)[0]
    pick = np.sort(rng.choice(neg, size=n_neg, replace=False))
    return np.sort(np.concatenate([np.where(y == 1)[0], pick]))


def build(dataset):
    import data as gbdata
    d = gbdata.load_dataset(dataset, seed=42)
    orig = {"train": (list(d["train_seqs"]), np.asarray(d["y_train"], np.int64)),
            "test": (list(d["test_seqs"]), np.asarray(d["y_test"], np.int64))}
    members = {sp: {} for sp in orig}             # split -> arm -> ordered original indices
    for sp, (seqs, y) in orig.items():
        members[sp]["original"] = np.arange(len(y))
        for arm, stem in CLEAN_STEM.items():
            df = pd.read_csv(os.path.join(SPLITS_V2, f"{dataset}_{stem}_{sp}.csv"))
            cs = df["sequence"].astype(str).str.upper().tolist()
            cy = df["label"].to_numpy(np.int64)
            members[sp][arm] = map_rows_to_index(cs, cy, seqs, y)
    rng = np.random.RandomState(RANDOM_SEED)
    for sp in ("train", "test"):                  # train draw first, then test (documented)
        y = orig[sp][1]
        n_neg = int((y[members[sp]["comp_equalized"]] == 0).sum())
        members[sp]["random"] = random_negative_draw(y, n_neg, rng)
    return orig, members


def check(orig, members):
    for sp, (seqs, y) in orig.items():
        pos = set(np.where(y == 1)[0].tolist())
        for arm, idx in members[sp].items():
            assert len(set(idx.tolist())) == len(idx), f"{arm}/{sp}: duplicate index"
            assert set(idx[y[idx] == 1].tolist()) == pos, f"{arm}/{sp}: positives differ"
        ce = members[sp]["comp_equalized"]; rd = members[sp]["random"]
        assert (y[ce] == 0).sum() == (y[rd] == 0).sum(), "random arm not size-matched"
        assert len(ce) == len(rd)


def composition_readout(orig, members, arm):
    """Composition-only LR AUROC (fit arm-train, score arm-test) and test GC AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import _common as C
    tr_s, tr_y = orig["train"]; te_s, te_y = orig["test"]
    itr = members["train"][arm]; ite = members["test"][arm]
    Xtr = C.comp_signature([tr_s[i] for i in itr]); Xte = C.comp_signature([te_s[i] for i in ite])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced",
                                                            random_state=42))
    lr.fit(Xtr, tr_y[itr])
    comp_auc = float(roc_auc_score(te_y[ite], lr.predict_proba(Xte)[:, 1]))
    gc = C.gc_content([te_s[i] for i in ite])
    gc_auc = float(roc_auc_score(te_y[ite], gc))
    return comp_auc, gc_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    todo = ap.parse_args().datasets.split(",")
    assert all(d in DATASETS for d in todo), todo
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds in todo:
        orig, members = build(ds)
        check(orig, members)
        for sp, (seqs, y) in orig.items():
            df = pd.DataFrame(dict(idx=np.arange(len(y)), label=y, sequence=seqs))
            for arm in ARMS:
                pos = np.full(len(y), -1, dtype=np.int64)
                pos[members[sp][arm]] = np.arange(len(members[sp][arm]))
                df[f"{arm}_pos"] = pos
            out = os.path.join(OUT_DIR, f"{ds}__{sp}.csv.gz")
            df.to_csv(out, index=False)
            print(f"wrote {out} ({len(df)} rows)")
        for arm in ARMS:
            comp_auc, gc_auc = composition_readout(orig, members, arm)
            for sp, (seqs, y) in orig.items():
                idx = members[sp][arm]
                h = hashlib.sha256(np.sort(idx).astype(np.int64).tobytes()).hexdigest()[:16]
                rows.append(dict(dataset=ds, arm=arm, split=sp, n=len(idx),
                                 n_pos=int((y[idx] == 1).sum()), n_neg=int((y[idx] == 0).sum()),
                                 pos_frac=round(float((y[idx] == 1).mean()), 6),
                                 idx_sha256_16=h,
                                 comp_only_lr_auroc_test=round(comp_auc, 6),
                                 gc_auroc_test=round(gc_auc, 6)))
            print(f"  {ds:22s} {arm:15s} comp-only LR AUROC={comp_auc:.3f}  GC AUROC={gc_auc:.3f}")
    man = pd.DataFrame(rows)
    if os.path.exists(MANIFEST):
        old = pd.read_csv(MANIFEST)
        man = pd.concat([old[~old["dataset"].isin(todo)], man], ignore_index=True)
    man["_o"] = man["dataset"].map({d: i for i, d in enumerate(DATASETS)})
    man = man.sort_values(["_o"], kind="stable").drop(columns="_o")
    man.to_csv(MANIFEST, index=False)
    print(man.to_string(index=False))
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
