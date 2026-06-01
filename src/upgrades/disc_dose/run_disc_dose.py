#!/usr/bin/env python3
"""Upgrade 3 -- semi-synthetic discriminability dose-response (Route 1, causal, arbitrary-n).

Replaces the underpowered pooled n=11 Spearman (discriminability -> learnability) with a causal
dose-response: build synthetic data where the motif-only AUROC (discriminability d) is DIALED, and
measure the k-mer+LightGBM benchmark MCC as a function of d.

Construction (seed 42; CPU; reuses featurize/models/motif_match/motif_jaspar):
  * Background = random ACGT sequences, length and base composition matched to the real
    human_nontata_promoters task (L=251; A/C/G/T frequencies measured from that task).
  * Implant PWM-SAMPLED instances (drawn from the motif's PPM, not the consensus) of a chosen
    JASPAR vertebrate TF into a fraction p_pos of POSITIVES and p_neg of NEGATIVES, varying p_pos to
    sweep the realized discriminability. The realized d is MEASURED (not assumed) as the motif-only
    AUROC = auc_counts(pos motif-hit-count, neg motif-hit-count) at 1.0 bits/pos -- the SAME machinery
    as run_discriminability.py.
  * For each cell: standard k-mer spectrum (k=6) -> LightGBM -> TEST MCC (stratified 80/20, seeded).
  * >= 5 seeds x 3 motifs of differing information content (SP1 14.2 bits, MYC 11.0, TBP 8.8).

Output: results/upgrades/disc_dose.csv (one row per motif x seed x p_pos), with measured d and MCC.
The figure script fits MCC ~ d (slope + bootstrap CI) and reports the Spearman monotonicity.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import featurize
import models as gbmodels
import motif_jaspar
import motif_match

BASES = np.array(list("ACGT"))
MOTIFS = ["SP1", "MYC", "TBP"]           # high / medium / low total IC, all > 1 bit/position
# (a long, low-per-position motif like CTCF, 0.63 bits/pos, is excluded: its PWM-sampled instances
#  never cross the 1.0 bits/pos hit threshold, so the motif-only-AUROC d would read ~0.5 even as the
#  k-mer model clearly learns it -- an artifact of the fixed hit threshold, not of discriminability.)
P_POS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
REF_TASK = "human_nontata_promoters"


def ppm_from_pssm(pssm):
    """Reconstruct the (4,L) probability matrix from the log-odds PSSM (uniform 0.25 bg)."""
    p = 0.25 * (2.0 ** pssm.astype(np.float64))
    return p / p.sum(0, keepdims=True)


def motif_ic(ppm):
    return float(np.sum(ppm * np.log2((ppm + 1e-9) / 0.25)))


def ref_profile(task):
    """(length, base_freqs) of the reference real task."""
    tr, _, te, _ = C.load_original(task)
    seqs = tr + te
    L = int(np.median([len(s) for s in seqs]))
    codes = motif_match.encode_sequences(seqs[:3000])
    valid = codes < 4
    freqs = np.array([((codes == b) & valid).sum() / valid.sum() for b in range(4)], dtype=np.float64)
    return L, freqs / freqs.sum()


def make_background(n, L, freqs, rng):
    idx = rng.choice(4, size=(n, L), p=freqs)
    chars = BASES[idx]
    return ["".join(row) for row in chars]


def implant(seqs, ppm, rate, rng):
    """Implant one PWM-sampled instance into a fraction ``rate`` of the sequences (random position)."""
    Lm = ppm.shape[1]
    out = []
    for s in seqs:
        if rng.random() < rate and len(s) >= Lm:
            inst = "".join(BASES[rng.choice(4, p=ppm[:, j])] for j in range(Lm))
            start = rng.randint(0, len(s) - Lm + 1)
            s = s[:start] + inst + s[start + Lm:]
        out.append(s)
    return out


def synth_dataset(ppm, n_per_class, L, freqs, p_pos, p_neg, rng):
    pos = implant(make_background(n_per_class, L, freqs, rng), ppm, p_pos, rng)
    neg = implant(make_background(n_per_class, L, freqs, rng), ppm, p_neg, rng)
    seqs = pos + neg
    y = np.concatenate([np.ones(n_per_class, int), np.zeros(n_per_class, int)])
    return seqs, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "disc_dose.csv"))
    ap.add_argument("--motifs", default=",".join(MOTIFS))
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--p-neg", type=float, default=0.0)
    ap.add_argument("--thr-bits", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    motifs = args.motifs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    L, freqs = ref_profile(REF_TASK)
    pssms = {m["name"].upper(): m for m in motif_jaspar.load_pssms("vertebrates")}
    print("=" * 96)
    print(f"UPGRADE 3 -- DISCRIMINABILITY DOSE-RESPONSE  (synthetic, L={L}, k={args.k}, "
          f"n/class={args.n_per_class})")
    print(f"motifs={motifs}  seeds={seeds}  p_pos grid={P_POS_GRID}  p_neg={args.p_neg}")
    print("=" * 96)

    rows = []
    for mname in motifs:
        m = pssms[mname.upper()]
        ppm = ppm_from_pssm(m["pssm"]); ic = motif_ic(ppm)
        for seed in seeds:
            for p_pos in P_POS_GRID:
                rng = np.random.RandomState(seed * 1000 + int(p_pos * 100))
                seqs, y = synth_dataset(ppm, args.n_per_class, L, freqs, p_pos, args.p_neg, rng)
                # measured discriminability d = motif-only AUROC (pos vs neg hit counts)
                codes = motif_match.encode_sequences(seqs)
                hits = motif_match.count_hits_batch(codes, m["pssm"], args.thr_bits)
                d = C.auc_counts(hits[y == 1].astype(np.int64), hits[y == 0].astype(np.int64))
                # benchmark: k-mer spectrum -> LightGBM, stratified 80/20 (seeded)
                tr_idx, te_idx = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
                                      .split(np.zeros(len(y)), y))
                X = featurize.kmer_spectrum(seqs, args.k)
                model = gbmodels.build_model("lgbm", seed, 2)
                model.fit(X[tr_idx], y[tr_idx])
                pred = model.classes_[np.argmax(model.predict_proba(X[te_idx]), axis=1)]
                mcc = float(matthews_corrcoef(y[te_idx], pred))
                rows.append(dict(motif=mname, motif_ic=round(ic, 2), motif_len=m["length"],
                                 seed=seed, p_pos=p_pos, p_neg=args.p_neg,
                                 motif_auroc_d=round(float(d), 4), bench_mcc=round(mcc, 4),
                                 n_train=len(tr_idx), n_test=len(te_idx)))
            print(f"  {mname:<6} (IC={ic:.1f}) seed={seed} done "
                  f"(d range {min(r['motif_auroc_d'] for r in rows if r['motif']==mname and r['seed']==seed):.2f}"
                  f"-{max(r['motif_auroc_d'] for r in rows if r['motif']==mname and r['seed']==seed):.2f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    # report: pooled Spearman + linear slope MCC~d with bootstrap CI over rows
    from scipy.stats import spearmanr
    d = df["motif_auroc_d"].to_numpy(); mcc = df["bench_mcc"].to_numpy()
    rho, p = spearmanr(d, mcc)
    rng = np.random.RandomState(42)
    slopes = []
    for _ in range(1000):
        idx = rng.randint(0, len(d), len(d))
        slopes.append(np.polyfit(d[idx], mcc[idx], 1)[0])
    slope = float(np.polyfit(d, mcc, 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    report = [
        "=" * 96,
        "DISCRIMINABILITY DOSE-RESPONSE -- benchmark MCC as a function of dialed motif-only AUROC d",
        "=" * 96,
        f"n points = {len(df)} ({len(motifs)} motifs x {len(seeds)} seeds x {len(P_POS_GRID)} doses)",
        f"Spearman(d, MCC) = {rho:+.3f} (p={p:.1e})  -> {'MONOTONE increasing' if rho>0 else 'not increasing'}",
        f"Linear slope dMCC/dd = {slope:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
        f"({'excludes 0' if not (lo<=0<=hi) else 'includes 0'})",
        "",
        "Per motif (information content), d-range and MCC-range:",
    ]
    for mname in motifs:
        g = df[df["motif"] == mname]
        report.append(f"  {mname:<6} IC={g['motif_ic'].iloc[0]:.1f}  d in [{g['motif_auroc_d'].min():.2f},"
                      f"{g['motif_auroc_d'].max():.2f}]  MCC in [{g['bench_mcc'].min():.2f},{g['bench_mcc'].max():.2f}]")
    report += ["",
        "Reading: a positive, significant slope with high Spearman shows learnability is CAUSALLY",
        "driven by motif discriminability -- dial d up (by implanting the motif more in positives than",
        "negatives) and the benchmark MCC rises monotonically, across motifs of differing IC. This is",
        "the arbitrary-n causal replacement for the pooled real-task correlation. Seed 42; synthetic."]
    rep = "\n".join(report)
    print("\n" + rep)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(rep)
    print(f"\nWrote {args.out} ({len(df)} rows) + interpretation.")


if __name__ == "__main__":
    main()
