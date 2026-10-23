#!/usr/bin/env python3
"""Build the private sequence cache for all 15 tasks and VERIFY it against the benchmark.

Checks (any failure aborts):
  * n_train / n_test equal results/results.csv / results_nt.csv (the Table-1 source);
  * for the two tasks with a released label cache (cache/<task>__{train,test}__y.npy), the label
    vectors are identical element-by-element -> the sequence ORDER equals data.load_dataset's.
Writes nothing under results/ (cache lives in external/multivar/seqcache/).
Usage: python src/upgrades/multivar/prefetch_tasks.py [--tasks a,b,...]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _mv  # noqa: E402


def expected_counts():
    r = pd.concat([pd.read_csv(os.path.join(_mv.ROOT, "results", f)) for f in ("results.csv", "results_nt.csv")])
    r = r[(r["model"] == "lgbm") & (r["selected_by_val"] == 1)]
    return {t: (int(a), int(b)) for t, a, b in zip(r["dataset"], r["n_train"], r["n_test"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(_mv.TASKS))
    a = ap.parse_args()
    _mv.start_memguard(tag="prefetch")
    exp = expected_counts()
    for task in a.tasks.split(","):
        t0 = time.time()
        tr, ytr, te, yte = _mv.load_task_cached(task)
        ok = (len(tr), len(te)) == exp[task]
        msg = f"[{task}] n_train={len(tr)} n_test={len(te)} expected={exp[task]} -> {'OK' if ok else 'MISMATCH'}"
        for sp, y in (("train", ytr), ("test", yte)):
            fy = os.path.join(_mv.ROOT, "cache", f"{task}__{sp}__y.npy")
            if os.path.exists(fy):
                same = np.array_equal(np.load(fy), y)
                msg += f" | {sp} label order vs cache: {'IDENTICAL' if same else 'DIFFERENT'}"
                ok = ok and same
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            sys.exit(f"verification failed for {task}")
    print("ALL VERIFIED")


if __name__ == "__main__":
    main()
