#!/usr/bin/env python3
"""Map every nt_enhancers sequence (train + test, pinned NT revision) to GRCh38/hg38.

Why this exists (Reviewer 2, comment 1 -- negative-set provenance): the Nucleotide Transformer
`enhancers` task ships sequences only (name column '***|0'), and earlier exact / minimap2 mapping
recovered only ~22-28% of them. The dataset card and NT bioRxiv v1 (A.4.3) state that the training
set is the iEnhancer-2L benchmark (742 strong + 742 weak enhancers, 1484 non-enhancers) PLUS 6000
synthetic enhancers and 6000 synthetic non-enhancers produced by a generative model (Geng et al.
2022, RankGAN). Synthetic sequences have no genomic origin, so a low mappability ceiling is
EXPECTED. This script makes that split explicit, sequence by sequence.

Method (pre-specified before running):
  * reference: UCSC hg38 (soft-masked, upper-cased on read), primary chromosomes chr1-22, X, Y, M
    only (the iEnhancer fragments derive from ENCODE ChromHMM segmentations of the primary
    assembly; alt/random/Un contigs would only create spurious multi-maps).
  * aligner: minimap2 via mappy, preset 'sr'. To keep RAM < ~1.2 GB on a shared 16 GB machine,
    ONE chromosome is indexed at a time (mappy.Aligner(seq=...)); every unique query is mapped to
    each chromosome and the best hit per chromosome is checkpointed (resume-safe).
  * best genome-wide hit = max matching bases (mlen), ties across distinct loci counted (n_best).
  * GENOMIC call (pre-registered): query coverage >= 0.95 AND identity (mlen/blen) >= 0.95.
    Everything else is called NON-GENOMIC (expected: the RankGAN synthetic sequences).
Seed-free (deterministic). CPU only, 1 mapping thread.

Outputs
  external/neg_provenance/nt_map/<chrom>.csv   per-chromosome checkpoints
  results/upgrades/neg_provenance_nt_map.csv   one row per sequence (split, idx, label, ...)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402  (adds repo root to sys.path)
import data_nt  # noqa: E402

EXT = os.path.join(C.ROOT, "external", "neg_provenance")
CKPT = os.path.join(EXT, "nt_map")
HG38 = os.environ.get("HG38_FA", os.path.expanduser("~/.cache/glrb_hg38/hg38.fa"))
OUT = os.path.join(C.RESULTS_DIR, "neg_provenance_nt_map.csv")
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
MIN_COV, MIN_ID = 0.95, 0.95


def load_nt():
    """Rows for every nt_enhancers sequence with its binary label and 3-class type label."""
    rows = []
    for split in ("train", "test"):
        s_bin, y_bin = data_nt._read("enhancers", split)
        s_typ, y_typ = data_nt._read("enhancers_types", split)
        assert s_bin == s_typ, "enhancers / enhancers_types sequence order differs"
        for i, (s, yb, yt) in enumerate(zip(s_bin, y_bin, y_typ)):
            rows.append((split, i, int(yb), int(yt), len(s), s))
    return pd.DataFrame(rows, columns=["split", "idx", "label", "type_label", "length", "sequence"])


def map_chrom(chrom, queries):
    import mappy
    import pyfaidx
    fa = pyfaidx.Fasta(HG38, as_raw=True, sequence_always_upper=True)
    if chrom not in fa:
        return pd.DataFrame(columns=["q", "chr", "mlen", "blen", "nm", "q_st", "q_en", "r_st",
                                     "r_en", "strand", "mapq", "n_hits_eqbest"])
    seq = fa[chrom][:]
    t0 = time.time()
    aln = mappy.Aligner(seq=seq, preset="sr", n_threads=1, best_n=10)
    del seq
    rows = []
    for qi, q in enumerate(queries):
        best, n_eq = None, 0
        for h in aln.map(q):
            if best is None or h.mlen > best.mlen:
                best, n_eq = h, 1
            elif h.mlen == best.mlen:
                n_eq += 1
        if best is not None:
            rows.append((qi, chrom, best.mlen, best.blen, best.NM, best.q_st, best.q_en, best.r_st,
                         best.r_en, "+" if best.strand == 1 else "-", best.mapq, n_eq))
    del aln
    print(f"  {chrom}: {len(rows)} queries with a hit ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows, columns=["q", "chr", "mlen", "blen", "nm", "q_st", "q_en", "r_st",
                                       "r_en", "strand", "mapq", "n_hits_eqbest"])


def main():
    os.makedirs(CKPT, exist_ok=True)
    df = load_nt()
    uniq = pd.unique(df["sequence"])
    qid = {s: i for i, s in enumerate(uniq)}
    df["q"] = df["sequence"].map(qid)
    print(f"nt_enhancers: {len(df)} sequences, {len(uniq)} unique", flush=True)
    for chrom in CHROMS:
        ck = os.path.join(CKPT, f"{chrom}.csv")
        if os.path.exists(ck):
            continue
        res = map_chrom(chrom, list(uniq))
        res.to_csv(ck + ".tmp", index=False)
        os.replace(ck + ".tmp", ck)
    hits = pd.concat([pd.read_csv(os.path.join(CKPT, f"{c}.csv")) for c in CHROMS], ignore_index=True)
    hits = hits.sort_values(["q", "mlen"], ascending=[True, False])
    top = hits.groupby("q").head(1).set_index("q")
    # number of distinct chromosomes/loci achieving the best mlen (multi-mapping indicator)
    bestm = hits.groupby("q")["mlen"].transform("max")
    n_best = hits[hits["mlen"] == bestm].groupby("q")["n_hits_eqbest"].sum()
    out = df.drop(columns=["sequence"]).join(top, on="q")
    out["n_best_loci"] = out["q"].map(n_best).fillna(0).astype(int)
    out["cov"] = (out["q_en"] - out["q_st"]) / out["length"]
    out["ident"] = out["mlen"] / out["blen"]
    out["genomic"] = (out["cov"] >= MIN_COV) & (out["ident"] >= MIN_ID)
    out["cov"] = out["cov"].fillna(0.0)
    out["ident"] = out["ident"].fillna(0.0)
    out["genomic"] = out["genomic"].fillna(False).astype(bool)
    out.to_csv(OUT, index=False)
    g = out.groupby(["split", "label"])["genomic"].agg(["sum", "count", "mean"])
    print(g.to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
