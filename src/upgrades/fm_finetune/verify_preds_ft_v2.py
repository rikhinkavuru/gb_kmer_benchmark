#!/usr/bin/env python3
"""Integrity check of every fine-tuning prediction unit before it is scored (torch-free).

Written 2026-09-26 after the local queue (run_queue_v3.sh) finished, because earlier runs were
interrupted (power loss 2026-09-24, network outage 2026-09-25) and nothing on disk is trusted
blindly. For every <model>__<dataset>__<arm>__s<seed>.{json,csv.gz} pair in
results/upgrades/fm_finetune_v2_preds/ it checks:
  1. the file name agrees with the JSON's model / dataset / train_arm / seed;
  2. the test rows cover every ORIGINAL test sequence exactly once (idx 0..n-1) and the labels equal
     the verified split file (external/fm_finetune/splits/<ds>__test.csv.gz, itself checked against
     data.load_dataset() by tests/test_ft_v2.py);
  3. the validation rows are a subset of the arm's TRAIN membership (never test), probabilities are
     finite and in [0, 1], and the validation AUROC recomputed from them equals best_val_auroc;
  4. best_epoch is the arg-max of the per-epoch validation AUROC (model selection never saw test);
  5. the per-arm test AUROC recomputed from the stored probabilities equals the JSON's
     quick_test_auroc (written by the trainer at the time), to 1e-4;
  6. the expected grid is complete: hyenadna x {cohn, nt_enhancers, drosophila} x 4 arms x 3 seeds,
     dnabert2 x {cohn, nt_enhancers} x 3 arms x 3 seeds (54 units).
Output: results/upgrades/fm_finetune_v2_preds_check.csv (one row per unit); exits non-zero on any
failure.
Usage: bibm/lowcpu.sh <torch-free python> src/upgrades/fm_finetune/verify_preds_ft_v2.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_ft_v2 as ev  # noqa: E402

EXPECTED = {("hyenadna", ds): ("original", "gc_match", "comp_equalized", "random")
            for ds in ("human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark")}
EXPECTED.update({("dnabert2", ds): ("original", "comp_equalized", "random")
                 for ds in ("human_enhancers_cohn", "nt_enhancers")})
SEEDS = (42, 43, 44)


def main():
    rows, bad = [], []
    tr_mem, te = {}, {}
    for f in glob.glob(os.path.join(ev.SPLIT_DIR, "*__train.csv.gz")):
        ds = os.path.basename(f).split("__")[0]
        d = pd.read_csv(f, usecols=["idx", "label"] + [f"{a}_pos" for a in ev.ARMS])
        tr_mem[ds] = {a: set(d.loc[d[f"{a}_pos"] >= 0, "idx"]) for a in ev.ARMS}
        tr_mem[ds]["_y"] = d["label"].to_numpy()
    mem = ev.load_membership()
    for js in sorted(glob.glob(os.path.join(ev.PRED_DIR, "*.json"))):
        name = os.path.basename(js)[:-5]
        csv = js[:-5] + ".csv.gz"
        mt = json.load(open(js))
        errs = []
        m, ds, arm, s = name.split("__")
        if (m, ds, arm, int(s[1:])) != (mt["model"], mt["dataset"], mt["train_arm"], mt["seed"]):
            errs.append("name/meta mismatch")
        if not os.path.exists(csv):
            errs.append("missing csv.gz")
            rows.append(dict(unit=name, ok=False, errors=";".join(errs))); bad.append(name); continue
        df = pd.read_csv(csv)
        t = df[df.split == "test"].sort_values("idx")
        v = df[df.split == "val"]
        y = mem[ds]["y"]
        if not np.array_equal(t["idx"].to_numpy(), np.arange(len(y))):
            errs.append("test idx != 0..n-1")
        elif not np.array_equal(t["label"].to_numpy(), y):
            errs.append("test labels != split file")
        p = df["prob"].to_numpy(np.float64)
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            errs.append("prob not finite/in [0,1]")
        if not set(v["idx"]) <= tr_mem[ds][arm]:
            errs.append("val rows outside the arm's train membership")
        if not np.array_equal(v["label"].to_numpy(), tr_mem[ds]["_y"][v["idx"].to_numpy()]):
            errs.append("val labels != train split")
        if len(v) != mt["n_val"]:
            errs.append(f"n_val {len(v)} != meta {mt['n_val']}")
        va = ev.auroc(v["label"].to_numpy(), v["prob"].to_numpy(np.float64))
        if abs(va - mt["best_val_auroc"]) > 1e-4:
            errs.append(f"val AUROC {va:.6f} != best_val_auroc {mt['best_val_auroc']}")
        hist = [h["val_auroc"] for h in mt["history"]]
        if int(np.argmax(hist)) + 1 != mt["best_epoch"]:
            errs.append("best_epoch != argmax(val AUROC)")
        diffs = {}
        for ea, q in (mt.get("quick_test_auroc") or {}).items():
            msk = mem[ds]["mask"][ea]
            a = ev.auroc(y[msk], t["prob"].to_numpy(np.float64)[msk])
            diffs[ea] = abs(a - q)
            if diffs[ea] > 1e-4:
                errs.append(f"test AUROC {ea} {a:.4f} != logged {q}")
        rows.append(dict(unit=name, model=m, dataset=ds, train_arm=arm, seed=mt["seed"],
                         n_test=len(t), n_val=len(v), val_auroc_recomputed=round(va, 6),
                         best_val_auroc=mt["best_val_auroc"], best_epoch=mt["best_epoch"],
                         epochs_run=mt["epochs_run"],
                         max_abs_diff_vs_logged_test_auroc=round(max(diffs.values()), 6) if diffs else np.nan,
                         pad_multiple=mt.get("pad_multiple", 0) or 0,
                         micro_batch=mt.get("micro_batch", 0) or 0,
                         peak_footprint_mb=mt.get("peak_footprint_mb", np.nan),
                         ok=not errs, errors=";".join(errs)))
        if errs:
            bad.append(name)
    out = pd.DataFrame(rows)
    have = {(r.model, r.dataset, r.train_arm, r.seed) for r in out.itertuples()}
    missing = [(m, ds, a, s) for (m, ds), arms in EXPECTED.items() for a in arms for s in SEEDS
               if (m, ds, a, s) not in have]
    out.to_csv(os.path.join(ev.OUT, "fm_finetune_v2_preds_check.csv"), index=False)
    print(out.drop(columns=["errors"]).to_string(index=False))
    print(f"{len(out)} units, {int(out.ok.sum())} ok, {len(bad)} failing, {len(missing)} missing "
          f"from the expected 54-unit grid")
    for b in bad:
        print("FAIL", b, out.loc[out.unit == b, "errors"].iloc[0])
    for mm in missing:
        print("MISSING", mm)
    if bad or missing:
        sys.exit(1)
    print("all prediction units verified")


if __name__ == "__main__":
    main()
