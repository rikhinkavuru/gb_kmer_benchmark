#!/usr/bin/env python3
"""Step 1b -- linclust arm of the synthetic calibration (added 2026-09-24 ~11:45, POST HOC: the README and
run_cluster.py docstring stated that linclust misses 15-40 % of >= 85 %-identity pairs, but no output file
supported that number; this script produces the evidence or refutes it).

Same synthetic set as run_cluster.calibrate() (identical generator and RandomState(42): 40 random 200-bp and
40 random 500-bp bases, each with one mutant per nominal identity 95..50 % -- odd-numbered bases' mutants
reverse-complemented -- plus 400 random 300-bp decoys). Clustered with `mmseqs linclust` (nucleotide DB,
--min-seq-id T, -c 0.8 --cov-mode 0, 3 threads) at T = 0.8 and 0.5, with the default --kmer-per-seq (21) and
a more sensitive --kmer-per-seq 100. Recall = fraction of planted (mutant, base) pairs placed in the same
cluster; decoys_falsely_merged = decoys sharing a cluster with anything. The search + connected-components
pipeline's rows are in homology_recluster_calibration.csv (method 'search+components').
Output: results/upgrades/homology_recluster_calibration_linclust.csv
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402
import run_cluster as RC  # noqa: E402


def synthetic_set():
    """Verbatim copy of the generator in run_cluster.calibrate() (same seed -> same sequences)."""
    rng = np.random.RandomState(_hc.SEED)
    comp_ = {"A": "T", "C": "G", "G": "C", "T": "A"}
    rc = lambda s: "".join(comp_[c] for c in reversed(s))

    def mut(s, p):
        out = []
        for c in s:
            r = rng.rand()
            if r < p * 0.9:
                out.append(rng.choice([x for x in "ACGT" if x != c]))
            elif r < p * 0.95:
                pass
            elif r < p:
                out.append(c); out.append(rng.choice(list("ACGT")))
            else:
                out.append(c)
        return "".join(out)
    seqs, meta = [], []
    for L in (200, 500):
        for i in range(40):
            b = "".join(rng.choice(list("ACGT"), L)); bi = len(seqs)
            seqs.append(b); meta.append((L, bi, None, None))
            for idv in (95, 90, 85, 80, 75, 70, 65, 60, 55, 50):
                s = mut(b, 1 - idv / 100)
                strand = "-" if i % 2 else "+"
                seqs.append(rc(s) if strand == "-" else s); meta.append((L, bi, idv, strand))
    for _ in range(400):
        seqs.append("".join(rng.choice(list("ACGT"), 300))); meta.append((300, -1, None, None))
    return seqs, meta


def linclust_labels(seqs, thr, kpk, work, log):
    os.makedirs(work, exist_ok=True)
    fa = os.path.join(work, "in.fa")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    db, clu, tmp, tsv = (os.path.join(work, x) for x in ("db", "clu", "tmp", "clu.tsv"))
    RC.run([RC.MMSEQS, "createdb", fa, db, "--dbtype", "2"], log)
    RC.run([RC.MMSEQS, "linclust", db, clu, tmp, "--min-seq-id", str(thr), "-c", "0.8", "--cov-mode", "0",
            "--kmer-per-seq", str(kpk), "--threads", str(_hc.N_THREADS)], log)
    RC.run([RC.MMSEQS, "createtsv", db, db, clu, tsv, "--threads", str(_hc.N_THREADS)], log)
    df = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "mem"])
    lab = np.arange(len(seqs))
    lab[df["mem"].to_numpy()] = df["rep"].to_numpy()
    return lab


def main():
    seqs, meta = synthetic_set()
    log = os.path.join(_hc.EXT, "logs", "mmseqs_calibration_linclust.log"); open(log, "w").close()
    rows = []
    for kpk in (21, 100):
        for tag, thr in RC.THRESHOLDS.items():
            work = os.path.join(RC.MM_DIR, f"_cal_linclust_{tag}_{kpk}")
            if os.path.isdir(work):
                shutil.rmtree(work)
            lab = linclust_labels(seqs, thr, kpk, work, log)
            shutil.rmtree(work)
            sizes = pd.Series(lab).value_counts()
            decoys = [j for j, m in enumerate(meta) if m[1] == -1]
            merged = int(sum(sizes[lab[j]] > 1 for j in decoys))
            for idv in (95, 90, 85, 80, 75, 70, 65, 60, 55, 50):
                for strand in "+-":
                    js = [j for j, m in enumerate(meta) if m[2] == idv and m[3] == strand]
                    rec = float(np.mean([lab[j] == lab[meta[j][1]] for j in js]))
                    rows.append(dict(method=f"linclust (kmer-per-seq {kpk})", threshold=tag, min_seq_id=thr,
                                     nominal_identity=idv, strand=strand, n_pairs=len(js), recall=round(rec, 4),
                                     decoys_falsely_merged=merged, n_decoys=len(decoys)))
    df = pd.DataFrame(rows)
    out = os.path.join(_hc.RES, f"{_hc.PREFIX}_calibration_linclust.csv")
    df.to_csv(out, index=False)
    print(df.pivot_table(index="nominal_identity", columns=["method", "threshold", "strand"], values="recall"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
