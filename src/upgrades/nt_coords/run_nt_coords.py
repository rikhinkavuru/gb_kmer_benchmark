#!/usr/bin/env python3
"""Upgrade 6 -- extend the TSS-coordinate refutation to nt_enhancers (Route 3).

The cohn coordinate analysis (localization/analyze_cohn.py) REFUTED the core-promoter-leakage
hypothesis: cohn's TATA-flagged negatives are TSS-DEPLETED (1.8% vs 3.3%), with no -25..-30 bp
offset peak -> AT-rich COMPOSITIONAL bias, not TSS-proximal leakage. That test mapped 100% of cohn
but only ~22% of nt_enhancers by EXACT substring match (localization/map_nt.py), leaving an asymmetry.

This module closes the asymmetry with FUZZY/alignment-based coordinate recovery (minimap2 via the
mappy Python binding, CPU) to map more nt_enhancers sequences to GRCh38 (Ensembl release-97, the same
reference as cohn), then repeats the cohn analysis on nt_enhancers:
  * recovered fraction (vs the 22% exact-match baseline);
  * TBP/TATA flagging (>=1.3 bits/pos, the benchmark criterion);
  * TSS overlap (GENCODE v44, +-500 bp) of flagged vs non-flagged NEGATIVES + Fisher exact test;
  * the TATA -> nearest-TSS offset distribution vs the canonical -25..-30 bp core-promoter window.

Goal: confirm (or refute) the SAME "AT-rich background, not TSS-proximal core-promoter leakage"
conclusion on the second dataset. Nulls reported straight. Seed 42; CPU-only; needs the optional
[coords] extra (mappy) + localization/GRCh38.primary_assembly.fa.gz + GENCODE v44. Mapped coords are
cached to localization/nt_coords_mappy.csv so reruns skip the alignment.
"""
import argparse
import bisect
import gzip
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import motif_match

LOC = os.path.join(C.ROOT, "localization")
FASTA = os.path.join(LOC, "GRCh38.primary_assembly.fa.gz")
GENCODE = os.path.join(LOC, "gencode.v44.annotation.gtf.gz")
SEED = 42
THR = 1.3
WIN = 500


def ensembl_to_ucsc(chrom):
    if chrom == "MT":
        return "chrM"
    if chrom in [str(i) for i in range(1, 23)] + ["X", "Y"]:
        return "chr" + chrom
    return chrom


def load_nt_enhancers():
    d = C.gbdata.load_dataset("nt_enhancers", seed=SEED)
    seqs = list(d["train_seqs"]) + list(d["test_seqs"])
    y = np.concatenate([d["y_train"], d["y_test"]])
    return seqs, y


def map_sequences(seqs, cache_csv, min_cov=0.9, min_idfrac=0.95):
    """Fuzzy-map each sequence to GRCh38 with mappy (preset 'sr'); return a DataFrame with chr/
    start/end/strand (UCSC chrom names) or chr='' if unmapped. Cached to cache_csv."""
    if os.path.exists(cache_csv):
        m = pd.read_csv(cache_csv, keep_default_na=False)
        if len(m) == len(seqs):
            print(f"  loaded cached mapping {cache_csv} ({len(m)} rows)")
            return m
    import mappy
    print(f"  building minimap2 index over {os.path.basename(FASTA)} (preset=sr; ~1-2 min, RAM-heavy) ...", flush=True)
    aln = mappy.Aligner(FASTA, preset="sr", n_threads=4)
    if not aln:
        sys.exit("ERROR: failed to build/load the GRCh38 index (check the FASTA).")
    rows = []
    for i, s in enumerate(seqs):
        best = None
        for h in aln.map(s):
            cov = (h.q_en - h.q_st) / max(len(s), 1)
            idfrac = h.mlen / max(h.blen, 1)
            if cov >= min_cov and idfrac >= min_idfrac:
                score = h.mlen
                if best is None or score > best[0]:
                    best = (score, ensembl_to_ucsc(h.ctg), h.r_st, h.r_en, "+" if h.strand == 1 else "-")
        if best:
            rows.append((i, best[1], best[2], best[3], best[4]))
        else:
            rows.append((i, "", -1, -1, ""))
        if (i + 1) % 5000 == 0:
            print(f"    mapped {i+1}/{len(seqs)}", flush=True)
    m = pd.DataFrame(rows, columns=["idx", "chr", "start", "end", "strand"])
    m.to_csv(cache_csv, index=False)
    return m


def best_tata_hit(seqs, tbp):
    """(flagged>=THR, hit_start, best_bits) per sequence -- mirrors localization/analyze_cohn.best_hit."""
    L = tbp["pssm"].shape[1]
    codes = motif_match.encode_sequences(list(seqs)); n, S = codes.shape; npos = S - L + 1
    ni = np.full((1, L), -1e9, np.float32)
    fwd = np.vstack([tbp["pssm"], ni]); rev = np.vstack([tbp["pssm"][::-1, ::-1], ni])
    def scan(a):
        sc = np.zeros((n, npos), np.float32)
        for i in range(L):
            sc += a[codes[:, i:i + npos], i]
        return sc / L
    sf, sr = scan(fwd), scan(rev); bf, br = sf.max(1), sr.max(1); uf = bf >= br
    return np.maximum(bf, br) >= THR, np.where(uf, sf.argmax(1), sr.argmax(1)), np.maximum(bf, br), L


def load_gencode_tss():
    tss = defaultdict(list)
    with gzip.open(GENCODE, "rt") as f:
        for ln in f:
            if ln[0] == "#":
                continue
            c = ln.split("\t")
            if c[2] != "transcript":
                continue
            t0 = (int(c[3]) - 1) if c[6] == "+" else (int(c[4]) - 1)
            tss[c[0]].append((t0, 1 if c[6] == "+" else -1))
    TP, TS = {}, {}
    for c in tss:
        a = sorted(tss[c]); TP[c] = np.array([x[0] for x in a]); TS[c] = np.array([x[1] for x in a])
    return TP, TS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "nt_coords.csv"))
    ap.add_argument("--map-cache", default=os.path.join(LOC, "nt_coords_mappy.csv"))
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    for need in (FASTA, GENCODE):
        if not os.path.exists(need):
            sys.exit(f"ERROR: missing {need}. (FASTA via the [coords] setup; GENCODE ships in localization/.)")

    tbp = C.load_tbp()
    seqs, y = load_nt_enhancers()
    print("=" * 96)
    print(f"UPGRADE 6 -- nt_enhancers coordinate analysis (fuzzy mappy mapping) | THR={THR} | WIN={WIN}")
    print("=" * 96)
    print(f"nt_enhancers total={len(seqs)} (neg={int((y==0).sum())}, pos={int((y==1).sum())})")

    m = map_sequences(seqs, args.map_cache)
    mapped = (m["chr"] != "").to_numpy()
    print(f"\nMAPPED {mapped.sum()}/{len(seqs)} = {mapped.mean():.1%}  (exact-match baseline was ~22%)")
    neg_m = mapped & (y == 0); pos_m = mapped & (y == 1)
    print(f"  negatives mapped {neg_m.sum()}/{int((y==0).sum())} = {neg_m.sum()/max((y==0).sum(),1):.1%}")

    flagged, hit_start, best_bits, L = best_tata_hit(seqs, tbp)
    start = m["start"].to_numpy(); end = m["end"].to_numpy(); strand = m["strand"].to_numpy()
    tata_center = np.where(strand == "+", start + hit_start + L / 2.0, end - hit_start - L / 2.0)
    tata_center = np.where(flagged & mapped, tata_center, np.nan)

    TP, TS = load_gencode_tss()
    print(f"GENCODE: {len(TP)} chroms, {sum(len(v) for v in TP.values())} transcripts; "
          f"naming example: {sorted(TP)[0]!r}")

    def ov(chrom, s, e):
        if chrom not in TP:
            return False
        a = TP[chrom]; i = bisect.bisect_left(a, s - WIN); return i < len(a) and a[i] < e + WIN
    tss_overlap = np.array([ov(c, s, e) if mp else False
                            for c, s, e, mp in zip(m["chr"], start, end, mapped)])

    # NEGATIVES: flagged vs non-flagged TSS overlap + Fisher (among MAPPED negatives)
    negmask = (y == 0) & mapped
    fl = negmask & flagged; nfl = negmask & ~flagged
    fo = tss_overlap[fl].mean() if fl.sum() else float("nan")
    no = tss_overlap[nfl].mean() if nfl.sum() else float("nan")
    po = tss_overlap[(y == 1) & mapped].mean() if ((y == 1) & mapped).sum() else float("nan")
    A = int(tss_overlap[fl].sum()); B = int(fl.sum()) - A
    Cc = int(tss_overlap[nfl].sum()); D = int(nfl.sum()) - Cc
    OR, pv = fisher_exact([[A, B], [Cc, D]]) if (A + B) and (Cc + D) else (float("nan"), float("nan"))

    # TATA -> nearest TSS offset for flagged negatives overlapping a TSS
    def near(chrom, p):
        a = TP[chrom]; i = bisect.bisect_left(a, p); cs = [j for j in (i - 1, i) if 0 <= j < len(a)]
        j = min(cs, key=lambda j: abs(a[j] - p)); return a[j], TS[chrom][j]
    offs = []
    for idx in np.where(fl & tss_overlap)[0]:
        if np.isnan(tata_center[idx]) or m["chr"].iloc[idx] not in TP:
            continue
        t, st = near(m["chr"].iloc[idx], tata_center[idx])
        offs.append((tata_center[idx] - t) if st == 1 else (t - tata_center[idx]))
    offs = np.array(offs)

    print(f"\n=== TSS overlap (+-{WIN}bp), MAPPED negatives ===")
    print(f"  flagged-neg   : {fo:.3f}  ({A}/{int(fl.sum())})")
    print(f"  nonflagged-neg: {no:.3f}  ({Cc}/{int(nfl.sum())})")
    print(f"  positives     : {po:.3f}  [assay positive control]")
    print(f"  enrichment flagged/nonflagged = {fo/no:.2f}x   Fisher OR={OR:.2f} p={pv:.2e}")
    canon = float(np.mean((offs >= -35) & (offs <= -25))) if len(offs) else float("nan")
    print(f"\n=== TATA->nearest-TSS offset (flagged negs overlapping TSS, n={len(offs)}) ===")
    if len(offs):
        print(f"  median={np.median(offs):.0f}  frac in [-35,-25] (canonical core-promoter)={canon:.3f}")

    res = dict(dataset="nt_enhancers", mapped_frac=round(float(mapped.mean()), 4),
               neg_mapped_frac=round(float(neg_m.sum() / max((y == 0).sum(), 1)), 4),
               n_mapped=int(mapped.sum()), n_total=len(seqs),
               flagged_neg_tss=round(float(fo), 4), nonflagged_neg_tss=round(float(no), 4),
               pos_tss=round(float(po), 4),
               enrichment=round(float(fo / no), 3) if no else float("nan"),
               fisher_OR=round(float(OR), 3), fisher_p=float(f"{pv:.2e}"),
               n_offsets=len(offs),
               offset_median=round(float(np.median(offs)), 1) if len(offs) else float("nan"),
               frac_canonical_minus25_30=round(canon, 4) if canon == canon else float("nan"))
    pd.DataFrame([res]).to_csv(args.out, index=False)
    np.save(os.path.join(LOC, "nt_offsets.npy"), offs)

    verdict = ("REFUTES core-promoter leakage (flagged negs TSS-DEPLETED, no -25..-30 peak) -> "
               "AT-rich COMPOSITIONAL bias, replicating cohn"
               if (no == no and fo <= no) else
               "flagged negs TSS-ENRICHED -> partial core-promoter signal (does NOT replicate cohn)")
    rep = [
        "=" * 96,
        "UPGRADE 6 -- nt_enhancers TSS coordinate analysis (fuzzy mapping)",
        "=" * 96,
        f"mapped {mapped.mean():.1%} of nt_enhancers to GRCh38 (exact-match baseline ~22%); "
        f"negatives {neg_m.sum()/max((y==0).sum(),1):.1%}.",
        f"TSS overlap (+-{WIN}bp): flagged-neg {fo:.3f} vs nonflagged-neg {no:.3f} "
        f"({fo/no:.2f}x; Fisher OR={OR:.2f} p={pv:.2e}); positives {po:.3f}.",
        f"TATA->TSS offset: n={len(offs)}, frac in canonical -25..-30 bp = {canon if canon==canon else float('nan'):.3f}.",
        "",
        f"VERDICT: {verdict}.",
        "Seed 42; reference GRCh38 Ensembl release-97 (matches cohn); GENCODE v44; mappy preset 'sr'.",
    ]
    rep = "\n".join(rep)
    print("\n" + rep)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(rep)
    print(f"\nWrote {args.out} + nt_offsets.npy + interpretation.")


if __name__ == "__main__":
    main()
