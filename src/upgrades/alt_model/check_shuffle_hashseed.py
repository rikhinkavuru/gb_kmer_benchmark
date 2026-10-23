#!/usr/bin/env python3
"""Reproducibility finding (post hoc, found while validating the alt-model shuffled arms):
src/upgrades/shuffled_neg/dinuc_shuffle.py iterates `verts = set(s)`, and the iteration order of a set
of 1-character strings depends on Python's per-process string-hash salt (PYTHONHASHSEED). The order in
which rng draws are assigned to vertices therefore changes between processes, so the dinucleotide-
shuffled negatives -- and every number computed from them -- are NOT bit-reproducible across runs
(the shuffles remain valid Altschul-Erikson shuffles; only the random draw changes).

This script quantifies how much that matters. For PYTHONHASHSEED in 0..N-1 (one subprocess each) it
rebuilds the shuffled negatives exactly as run_shuffled_neg.py (RandomState(42): train shuffles, then
test shuffles) for cohn / nt_enhancers / drosophila and records
  * a sha1 of the shuffles, the set-iteration order of 'ACGT',
  * the paper's MODEL-FREE readouts (TATA motif-only AUROC at 1.3 bits/pos and GC AUROC, pooled
    train+test, positives vs shuffles), and
  * the paper's k-mer LightGBM MCC/AUROC with shuffled negatives,
and flags the hash seeds that reproduce results/upgrades/shuffled_neg.csv exactly. Output:
results/upgrades/alt_model_shuffle_hashseed.csv. CLI: python check_shuffle_hashseed.py [N=16]
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, ".."))
TASKS = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
K = {"human_enhancers_cohn": 5, "nt_enhancers": 4, "drosophila_enhancers_stark": 4}


def child():
    for d in (UP, HERE, os.path.join(UP, "shuffled_neg")):
        sys.path.insert(0, d)
    import numpy as np
    import _common as C
    import alt_models as AM
    import featurize
    import run_shuffled_neg as SN
    from cache_seqs import load_cached
    tbp = C.load_tbp()
    out = []
    for t in TASKS:
        tr, ytr, te, yte = load_cached(t)
        rng = np.random.RandomState(C.SEED)
        pos_tr = [s for s, y in zip(tr, ytr) if y == 1]; pos_te = [s for s, y in zip(te, yte) if y == 1]
        sh_tr, _ = SN.shuffled_negatives(pos_tr, rng)
        sh_te, _ = SN.shuffled_negatives(pos_te, rng)
        pos = pos_tr + pos_te; neg = sh_tr + sh_te
        ta_p = C.tata_hit_counts(pos, tbp, 1.3).astype(np.int64); ta_n = C.tata_hit_counts(neg, tbp, 1.3).astype(np.int64)
        tata = C.auc_counts(ta_p, ta_n)
        gc = C.auc_float(C.gc_content(pos), C.gc_content(neg))
        Xtr = featurize.kmer_spectrum(pos_tr + sh_tr, K[t]); ytr2 = np.array([1] * len(pos_tr) + [0] * len(sh_tr))
        Xte = featurize.kmer_spectrum(pos_te + sh_te, K[t]); yte2 = np.array([1] * len(pos_te) + [0] * len(sh_te))
        m, _ = AM.fit_lgbm(Xtr, ytr2, n_jobs=1)          # deterministic: thread count does not matter
        p = m.predict_proba(Xte)
        out.append(dict(task=t, hashseed=os.environ.get("PYTHONHASHSEED"), set_order="".join(set("ACGT")),
                        shuffle_sha1=hashlib.sha1("".join(neg).encode()).hexdigest()[:12],
                        tata_auroc=round(float(tata), 4), gc_auroc=round(float(gc), 4),
                        kmer_mcc=round(AM.mcc_from_pred(yte2, m.classes_[p.argmax(1)], 2), 6),
                        kmer_auroc=round(AM.auroc(yte2, p), 6)))
    print("JSON" + json.dumps(out))


def main():
    import pandas as pd
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rel = pd.read_csv(os.path.join(os.path.dirname(UP), "..", "results", "upgrades", "shuffled_neg.csv"))
    rel = rel[rel.arm == "shuffled_neg"].set_index("dataset")
    rows = []
    for h in range(n):
        env = dict(os.environ, PYTHONHASHSEED=str(h))
        r = subprocess.run([sys.executable, __file__, "--child"], env=env, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("JSON")]
        if not line:
            print(r.stderr[-2000:]); continue
        for d in json.loads(line[0][4:]):
            rr = rel.loc[d["task"]]
            d["reproduces_paper_csv"] = bool(abs(d["kmer_mcc"] - rr["kmer_mcc"]) < 1e-6 and
                                             abs(d["tata_auroc"] - rr["tata_auroc"]) < 1e-4)
            d["paper_tata_auroc"] = rr["tata_auroc"]; d["paper_kmer_mcc"] = rr["kmer_mcc"]
            rows.append(d)
            print(d, flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(UP), "..", "results", "upgrades", "alt_model_shuffle_hashseed.csv")
    df.to_csv(os.path.abspath(out), index=False)
    g = df.groupby("task").agg(tata_min=("tata_auroc", "min"), tata_max=("tata_auroc", "max"),
                               gc_min=("gc_auroc", "min"), gc_max=("gc_auroc", "max"),
                               mcc_min=("kmer_mcc", "min"), mcc_max=("kmer_mcc", "max"),
                               n_distinct_shuffles=("shuffle_sha1", "nunique"),
                               n_reproducing=("reproduces_paper_csv", "sum"))
    print(g.to_string())


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        main()
