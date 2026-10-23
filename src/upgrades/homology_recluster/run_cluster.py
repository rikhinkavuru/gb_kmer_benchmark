#!/usr/bin/env python3
"""Step 1-2 -- MMseqs2 homology graph + clusters for ALL 15 tasks (train+test pooled, both strands).

Reviewer 1, comment 3 asks to "re-cluster all benchmark datasets to remove sequence homology
redundancy". This script builds, per task, the homology graph of the POOLED train+test sequences and
its connected components at two identity thresholds, and measures train/test and within-train
redundancy. It re-fits nothing (run_eval.py does that).

PRE-REGISTERED METHOD (fixed before any real-data run; calibration on synthetic data only)
------------------------------------------------------------------------------------------
Engine. `mmseqs search` ALL-vs-ALL in nucleotide mode (--search-type 3) with --strand 2, i.e. every
query is aligned against every target on BOTH strands (reverse complements included). A hit is a
homology EDGE if its alignment identity (fident, identical columns / alignment length) >= T and it
covers >= 80 % of BOTH sequences (-c 0.8 --cov-mode 0). Clusters = connected components of the edge
graph (single linkage), which is the conservative choice for leakage: two sequences linked through a
chain of near-duplicates can never straddle the split. We use all-vs-all search rather than
`linclust` because the calibration (homology_recluster_calibration.csv) showed that search +
components recovers 100 % of >=85 %-identity pairs on both strands with zero false merges of 400
random decoys; linclust (centre-star alignment, members aligned only to one centre) was not needed.
(Correction 2026-09-24: an earlier version of this docstring quoted a linclust miss rate of 15-40 % that
no output file supported at the time. linclust has since been calibrated on the same synthetic set by
run_linclust_calibration.py -> homology_recluster_calibration_linclust.csv: at id80 it recovers
0.900-0.975 of 95 %-identity pairs, 0.775-0.825 at 90 % and 0.575-0.725 at 85 % (both strands,
kmer-per-seq 21 and 100), i.e. it misses 2.5-42.5 % of >= 85 %-identity pairs, vs 1.00 for search +
components; 0/400 decoys merged by either.)
Timing on the real data showed all-vs-all search is affordable for every task (largest: 289k
sequences), so no task falls back to linclust.
Settings: -k 11 -s 7.5 --max-seqs 1000 -e 1e-3 (MMseqs2 18-8cc5c), tantan masking on, <=3 threads,
--split-memory-limit 2G, run at nice 20 through bibm/lowcpu.sh.

Thresholds (both from the SAME search, filtered on fident):
  T = 0.8, coverage 0.8 (PRIMARY). The conventional nucleotide near-duplicate / redundancy level
       (CD-HIT-EST's recommended range for DNA starts at 0.8; >=80 % identity over >=80 % of both
       sequences is far above chance for 200-500 bp DNA, where unrelated sequences align at ~25-45 %).
  T = 0.5, coverage 0.8 (STRICT). "Every homolog MMseqs2 can detect". For DNA this is bounded by the
       exact-k-mer seeding, not by T: calibration recall is ~95 % at 75 %, ~70-85 % at 70 %, ~25-40 %
       at 65 % and ~0 at <=60 % nominal identity, so the 0.5 setting means effectively >= ~65-70 %.

Per task and threshold we record:
  frac_test_in_train_cluster  fraction of TEST sequences whose component contains >=1 TRAIN sequence
  frac_test_direct_hit        fraction of TEST sequences with a DIRECT edge to a TRAIN sequence
  train_clusters_per_seq      (#components of the TRAIN-only graph) / n_train  (1.0 = no redundancy)
  pooled_clusters_per_seq     (#components of the pooled graph) / n_pooled
  frac_seqs_in_multiseq_cluster, mixed_label_seq_frac (sequence in a component carrying >1 label),
  largest_component
plus threshold-free exact-duplicate statistics (both strands): frac_test_exact_in_train (test
sequence identical to a train sequence or to its reverse complement) and sequence-length summary.

Outputs: external/homology_recluster/clusters/<task>_{id80,id50}.npz (per-sequence component ids,
pooled order = train then test), external/homology_recluster/edges/<task>.npz (all edges with
fident >= 0.5, cov >= 0.8, stored as fident*1000 uint16 so thresholds are exact), and
results/upgrades/homology_recluster_redundancy.csv.
Checkpoint/resume: a task whose outputs exist is skipped. A task whose pooled sequences are identical
(same order) to an already-processed task reuses that task's edges (nt_enhancers_types ==
nt_enhancers sequences with different labels). CPU-only, torch-free.

Usage:  python src/upgrades/homology_recluster/run_cluster.py [--tasks a,b] [--calibrate]
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hc  # noqa: E402

MMSEQS = os.environ.get("MMSEQS", "/opt/homebrew/bin/mmseqs")
THRESHOLDS = {"id80": 0.8, "id50": 0.5}
COV = 0.8
SEARCH_ARGS = ["--search-type", "3", "--strand", "2", "-k", "11", "-s", "7.5", "--max-seqs", "1000",
               "-e", "1e-3", "-c", str(COV), "--cov-mode", "0", "--min-seq-id", "0.5",
               "--threads", str(_hc.N_THREADS), "--split-memory-limit", "2G"]
MM_DIR = os.path.join(_hc.EXT, "mm")
EDGE_DIR = os.path.join(_hc.EXT, "edges")
# processing order: smallest first, so results accumulate and the giant tasks run last
SIZE_ORDER = ["dummy_mouse_enhancers_ensembl", "nt_promoter_tata", "drosophila_enhancers_stark",
              "nt_enhancers", "nt_enhancers_types", "human_enhancers_cohn", "nt_splice_sites_all",
              "human_nontata_promoters", "nt_promoter_no_tata", "nt_promoter_all", "demo_human_or_worm",
              "demo_coding_vs_intergenomic_seqs", "human_enhancers_ensembl", "human_ocr_ensembl",
              "human_ensembl_regulatory"]
assert sorted(SIZE_ORDER) == sorted(_hc.TASKS)


def components(n, a, b):
    """Connected-component label (0..C-1) per node of the undirected graph with edges (a[i], b[i])."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)).tocsr()
    _, lab = connected_components(g, directed=False)
    return lab.astype(np.int64)


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n"); fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)} -- see {log}")


def seq_hash(seqs):
    h = hashlib.sha1()
    for s in seqs:
        h.update(s.encode()); h.update(b"\n")
    return h.hexdigest()


def search_edges(seqs, workdir, log, chunk=4_000_000):
    """All-vs-all both-strand nucleotide search. Returns deduplicated undirected edges as arrays
    (a<b int32, b int32, fid uint16 = round(1000*fident), max over both directions/strands), for every
    non-self hit with fident >= 0.5 and coverage >= 0.8 of both sequences. The m8 file is read in
    chunks so RAM stays bounded on the redundant giant tasks."""
    os.makedirs(workdir, exist_ok=True)
    fa = os.path.join(workdir, "in.fa")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    db = os.path.join(workdir, "db"); aln = os.path.join(workdir, "aln")
    tmp = os.path.join(workdir, "tmp"); m8 = os.path.join(workdir, "aln.m8")
    run([MMSEQS, "createdb", fa, db, "--dbtype", "2"], log)
    run([MMSEQS, "search", db, db, aln, tmp] + SEARCH_ARGS, log)
    run([MMSEQS, "convertalis", db, db, aln, m8, "--format-output",
         "query,target,fident,qcov,tcov", "--threads", str(_hc.N_THREADS)], log)
    keys, fids = [], []
    n_raw = 0
    for df in pd.read_csv(m8, sep="\t", header=None, names=["q", "t", "fident", "qcov", "tcov"],
                          dtype={"q": np.int64, "t": np.int64, "fident": np.float64,
                                 "qcov": np.float32, "tcov": np.float32}, chunksize=chunk):
        n_raw += len(df)
        df = df[(df["q"] != df["t"]) & (df["qcov"] >= COV) & (df["tcov"] >= COV) & (df["fident"] >= 0.5)]
        a = np.minimum(df["q"].to_numpy(), df["t"].to_numpy())
        b = np.maximum(df["q"].to_numpy(), df["t"].to_numpy())
        k = a * np.int64(len(seqs)) + b
        f = np.rint(df["fident"].to_numpy() * 1000).astype(np.uint16)
        # per-chunk dedupe keeping the max fident per pair
        o = np.lexsort((-f.astype(np.int64), k))
        k, f = k[o], f[o]
        first = np.r_[True, k[1:] != k[:-1]]
        keys.append(k[first]); fids.append(f[first])
    if keys:
        k = np.concatenate(keys); f = np.concatenate(fids)
        o = np.lexsort((-f.astype(np.int64), k)); k, f = k[o], f[o]
        first = np.r_[True, k[1:] != k[:-1]]; k, f = k[first], f[first]
    else:
        k = np.zeros(0, np.int64); f = np.zeros(0, np.uint16)
    n = np.int64(len(seqs))
    return (k // n).astype(np.int32), (k % n).astype(np.int32), f, int(n_raw)


_RC = str.maketrans("ACGTN", "TGCAN")


def exact_dup_stats(tr, te):
    """Threshold-free: fraction of test sequences identical to a train sequence or its reverse
    complement; and fraction of train sequences that are exact (either-strand) duplicates of an
    earlier train sequence."""
    canon = lambda s: min(s, s.translate(_RC)[::-1])
    trc = [canon(s) for s in tr]
    trs = set(trc)
    te_in = np.mean([canon(s) in trs for s in te]) if te else float("nan")
    return float(te_in), float(1 - len(trs) / max(len(trc), 1))


def redundancy_row(task, tag, thr, ea, eb, ef, n_tr, y, is_test):
    n = len(y)
    m = ef >= int(round(thr * 1000))
    a, b = ea[m].astype(np.int64), eb[m].astype(np.int64)
    comp = components(n, a, b)
    # TRAIN-only graph components (within-train redundancy)
    mtr = (a < n_tr) & (b < n_tr)
    comp_tr = components(n_tr, a[mtr], b[mtr])
    # test seqs in a component that holds a train seq
    has_tr = np.zeros(comp.max() + 1, bool); has_tr[comp[~is_test]] = True
    te_in = has_tr[comp[is_test]]
    # direct train neighbour
    cross = (is_test[a] != is_test[b])
    te_nodes = np.where(is_test[a], a, b)[cross]
    direct = np.zeros(n, bool); direct[np.unique(te_nodes)] = True
    # mixed-label components
    lab_per = pd.DataFrame({"c": comp, "y": y}).groupby("c")["y"].nunique()
    mixed_c = lab_per[lab_per > 1].index.to_numpy()
    mixed_seq = np.isin(comp, mixed_c)
    sizes = np.bincount(comp)
    row = dict(task=task, threshold=tag, min_seq_id=thr, coverage=COV, n_train=int(n_tr),
               n_test=int(is_test.sum()), n_edges=int(m.sum()),
               frac_test_in_train_cluster=round(float(te_in.mean()), 5),
               n_test_in_train_cluster=int(te_in.sum()),
               frac_test_direct_hit=round(float(direct[is_test].mean()), 5),
               train_clusters=int(comp_tr.max() + 1),
               train_clusters_per_seq=round(float((comp_tr.max() + 1) / n_tr), 5),
               pooled_clusters=int(comp.max() + 1),
               pooled_clusters_per_seq=round(float((comp.max() + 1) / n), 5),
               frac_seqs_in_multiseq_cluster=round(float((sizes[comp] > 1).mean()), 5),
               mixed_label_seq_frac=round(float(mixed_seq.mean()), 5),
               largest_component=int(sizes.max()),
               largest_component_frac=round(float(sizes.max() / n), 5))
    return row, comp, direct


def do_task(task, a, hashes):
    outs = {tag: os.path.join(_hc.CLU_DIR, f"{task}_{tag}.npz") for tag in THRESHOLDS}
    rowf = os.path.join(_hc.CKPT_DIR, f"cluster_{task}.csv")
    edgef = os.path.join(EDGE_DIR, f"{task}.npz")
    if (all(os.path.exists(p) for p in outs.values()) and os.path.exists(rowf)
            and os.path.exists(edgef) and not a.force):
        print(f"[{task}] cached", flush=True)
        hashes[str(np.load(edgef)["seq_sha1"])] = edgef
        return pd.read_csv(rowf)
    t0 = time.time()
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te
    y = np.concatenate([ytr, yte])
    is_test = np.r_[np.zeros(len(tr), bool), np.ones(len(te), bool)]
    h = seq_hash(seqs)
    L = np.array([len(s) for s in seqs])
    if h in hashes:                                   # identical pooled sequences -> reuse edges
        z = np.load(hashes[h])
        ea, eb, ef, n_raw = z["a"], z["b"], z["fid"], int(z["n_raw_hits"])
        src = os.path.basename(hashes[h])[:-4]
        print(f"[{task}] pooled sequences identical to {src}: reusing its edges", flush=True)
    else:
        src = task
        work = os.path.join(MM_DIR, task)
        if os.path.isdir(work):
            shutil.rmtree(work)
        log = os.path.join(_hc.EXT, "logs", f"mmseqs_{task}.log")
        open(log, "w").close()
        ea, eb, ef, n_raw = search_edges(seqs, work, log)
        shutil.rmtree(work)                           # keep only the compact edge list
    np.savez_compressed(edgef + ".tmp.npz", a=ea, b=eb, fid=ef, n=len(seqs), n_train=len(tr),
                        n_raw_hits=n_raw, seq_sha1=h, edge_source=src)
    os.replace(edgef + ".tmp.npz", edgef)
    hashes[h] = edgef
    t_search = time.time() - t0
    ex_te, ex_tr = exact_dup_stats(tr, te)
    rows = []
    for tag, thr in THRESHOLDS.items():
        row, comp, direct = redundancy_row(task, tag, thr, ea, eb, ef, len(tr), y, is_test)
        row.update(frac_test_exact_in_train=round(ex_te, 5), frac_train_exact_dup=round(ex_tr, 5),
                   len_min=int(L.min()), len_median=float(np.median(L)), len_max=int(L.max()),
                   n_raw_hits=n_raw, edge_source=src, search_seconds=round(t_search, 1))
        rows.append(row)
        np.savez_compressed(outs[tag], comp=comp, is_test=is_test, y=y, direct=direct)
        print(f"[{task}] {tag}: test-in-train-cluster {row['frac_test_in_train_cluster']:.4f} "
              f"direct {row['frac_test_direct_hit']:.4f} | train clusters/seq "
              f"{row['train_clusters_per_seq']:.4f} | mixed-label {row['mixed_label_seq_frac']:.4f} "
              f"| largest {row['largest_component']} | exact-dup test {ex_te:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(rowf, index=False)
    print(f"[{task}] done in {time.time() - t0:.0f}s (search {t_search:.0f}s, {n_raw} raw hits, "
          f"{len(ea)} unique edges)", flush=True)
    return df


def calibrate(out_csv):
    """Synthetic recall calibration of the search+components pipeline vs identity and strand."""
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
    n_decoy = 400
    for _ in range(n_decoy):
        seqs.append("".join(rng.choice(list("ACGT"), 300))); meta.append((300, -1, None, None))
    work = os.path.join(MM_DIR, "_calibration")
    if os.path.isdir(work):
        shutil.rmtree(work)
    log = os.path.join(_hc.EXT, "logs", "mmseqs_calibration.log"); open(log, "w").close()
    ea, eb, ef, _ = search_edges(seqs, work, log)
    rows = []
    for tag, thr in THRESHOLDS.items():
        m = ef >= int(round(thr * 1000))
        comp = components(len(seqs), ea[m], eb[m])
        decoy_idx = [j for j, mm in enumerate(meta) if mm[1] == -1]
        sizes = np.bincount(comp)
        decoy_merged = int(sum(sizes[comp[j]] > 1 for j in decoy_idx))
        for idv in (95, 90, 85, 80, 75, 70, 65, 60, 55, 50):
            for strand in "+-":
                js = [j for j, mm in enumerate(meta) if mm[2] == idv and mm[3] == strand]
                rec = np.mean([comp[j] == comp[meta[j][1]] for j in js])
                rows.append(dict(threshold=tag, min_seq_id=thr, nominal_identity=idv, strand=strand,
                                 n_pairs=len(js), recall=round(float(rec), 4),
                                 decoys_falsely_merged=decoy_merged, n_decoys=n_decoy))
    shutil.rmtree(work)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.pivot_table(index=["nominal_identity"], columns=["threshold", "strand"], values="recall"))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(SIZE_ORDER))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=os.path.join(_hc.RES, f"{_hc.PREFIX}_redundancy.csv"))
    a = ap.parse_args()
    for d in (_hc.CLU_DIR, _hc.CKPT_DIR, MM_DIR, EDGE_DIR, os.path.join(_hc.EXT, "logs")):
        os.makedirs(d, exist_ok=True)
    if a.calibrate:
        calibrate(os.path.join(_hc.RES, f"{_hc.PREFIX}_calibration.csv"))
        return
    hashes = {}
    for f in sorted(os.listdir(EDGE_DIR)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            hashes[str(np.load(os.path.join(EDGE_DIR, f))["seq_sha1"])] = os.path.join(EDGE_DIR, f)
    for t in [t for t in a.tasks.split(",") if t]:
        do_task(t, a, hashes)
    # aggregate every task that has a checkpoint (so partial runs still write a table)
    parts = [pd.read_csv(os.path.join(_hc.CKPT_DIR, f"cluster_{t}.csv")) for t in _hc.TASKS
             if os.path.exists(os.path.join(_hc.CKPT_DIR, f"cluster_{t}.csv"))]
    if parts:
        pd.concat(parts).to_csv(a.out, index=False)
        print(f"wrote {a.out} ({len(parts)} tasks)")


if __name__ == "__main__":
    main()
