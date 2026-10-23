#!/usr/bin/env python3
"""Tabulate the memory-fix benches of run_queue_v3.sh from the training log (torch-free).

Each bench trains on the SAME mini-batches (seed-0 batcher, original arm) under one of two input
shapings and logs time and process footprint (phys_footprint, MB = 10^6 bytes) per step:
  fixed  --pad-multiple K (few tensor shapes; DNABERT-2 keeps pad rows instead of unpadding)
  old    per-batch padding (DNABERT-2 unpadded): a new tensor shape nearly every batch
Output: results/upgrades/fm_finetune_v2_memfix_bench.csv (one row per bench step) and a summary
printed to stdout: footprint at step 10 and at the last step, growth per step (least squares over
steps >= 10, i.e. after warm-up), median s/step, and whether memguard aborted the run.
Usage: python src/upgrades/fm_finetune/memfix_evidence.py
"""
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOG = os.path.join(ROOT, "results", "upgrades", "fm_finetune_v2_run.log")
OUT = os.path.join(ROOT, "results", "upgrades", "fm_finetune_v2_memfix_bench.csv")

HDR = re.compile(r"^=== (\S+ \S+) bench: (\w+) (\w+) (\d+) steps, (.*)$")
STEP = re.compile(r"^(\S+)\s+step (\d+) ([\d.]+)s loss=\S+ tokens=(\d+) footprint=(\d+)MB")


def main():
    rows, cur = [], None
    for line in open(LOG, errors="replace"):
        line = line.rstrip("\n")
        h = HDR.match(line)
        if h:
            desc = h.group(5)
            cur = dict(started=h.group(1), model=h.group(2), dataset=h.group(3),
                       shaping="old" if "OLD" in desc else "fixed", config=desc, aborted=False)
            continue
        if line.startswith("=== "):
            cur = None
            continue
        if cur is None:
            continue
        if "MEMGUARD" in line:
            for r in rows:
                if r["started"] == cur["started"]:
                    r["aborted_by_memguard"] = True
            continue
        s = STEP.match(line)
        if s:
            rows.append(dict(started=cur["started"], model=cur["model"], dataset=cur["dataset"],
                             shaping=cur["shaping"], config=cur["config"], step=int(s.group(2)),
                             secs=float(s.group(3)), tokens=int(s.group(4)),
                             footprint_mb=int(s.group(5)), aborted_by_memguard=False))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")
    for (st, model, ds, sh), g in df.groupby(["started", "model", "dataset", "shaping"], sort=False):
        g = g.sort_values("step")
        post = g[g.step >= 10]
        slope = float(np.polyfit(post.step, post.footprint_mb, 1)[0]) if len(post) > 2 else np.nan
        print(f"{model:8s} {ds:27s} {sh:5s} steps {g.step.min()}-{g.step.max()}  "
              f"footprint@10 {int(post.footprint_mb.iloc[0]) if len(post) else -1} MB -> "
              f"@{g.step.max()} {int(g.footprint_mb.iloc[-1])} MB  growth {slope:+.1f} MB/step  "
              f"distinct widths {g.tokens.nunique()}  median {g.secs.median():.2f} s/step  "
              f"aborted_by_memguard={bool(g.aborted_by_memguard.any())}")


if __name__ == "__main__":
    main()
