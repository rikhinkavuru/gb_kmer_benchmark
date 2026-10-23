#!/usr/bin/env python3
"""Housekeeping (2026-09-26): units written before save_unit stored float64 probabilities kept
float32 copies, which can merge near-identical scores into ties (max AUROC shift seen: 0.0008).
This deletes EVERY float32 unit (the mismatching ones -- recomputed MCC/AUROC |diff| > 1e-9 from
the float64 values logged in the JSON -- are listed separately) plus any pos/B1 alias built from one,
so the deterministic worker refits them bit-identically with float64 probabilities."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import analyze_alt_model as A  # noqa: E402

bad = []
for j in sorted(glob.glob(os.path.join(A.UNITS, "*.json"))):
    base = j[:-5]
    z = np.load(base + ".npz", allow_pickle=False)
    if z["proba"].dtype != np.float32:
        continue
    d = json.load(open(j))
    sc = A.Scorer(z["y"].astype(np.int64), z["proba"].astype(np.float64))
    if abs(sc.mcc() - d["mcc"]) > 1e-9 or abs(sc.auroc() - d["auroc"]) > 1e-9:
        print("mismatch", os.path.basename(base), round(sc.auroc() - d["auroc"], 6))
    bad.append(base)
for base in list(bad):
    t, arm, feat, m = os.path.basename(base).split("__")
    if arm == "bench" and feat == "kmer":
        alias = os.path.join(A.UNITS, f"{t}__pos__B1__{m}")
        if os.path.exists(alias + ".json") and json.load(open(alias + ".json")).get("aliased_from"):
            bad.append(alias)
for base in bad:
    for ext in (".json", ".npz"):
        if os.path.exists(base + ext):
            os.remove(base + ext)
    print("deleted", os.path.basename(base))
print(f"{len(bad)} float32 units removed for float64 refit")
