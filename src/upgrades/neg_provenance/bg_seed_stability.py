#!/usr/bin/env python3
"""POST HOC (fix round 1, 2026-09-26): seed stability of the random and rule-matched backgrounds.

Why: independent verification showed that the "rule background matches the negatives' TATA rate" statement
for cohn (post hoc protein-coding rule, BG_rule_pc) depends on the particular seed-42 draw of 10,000 windows
(seeds 1/2 gave TATA 0.5835/0.5864 vs NEG 0.5741). The pre-registered analysis (seed 42, 10,000 windows per
background) is unchanged; this script adds five further independent draws (seeds 1-5, 10,000 windows each,
same samplers as run_neg_provenance.py) and reports, per background and metric, the per-seed values, their
spread, and the POOLED estimate over all six draws (60,000 windows) with the pooled-background minus NEG
difference and f_rule = (BG_uni - BG_rule)/(BG_uni - NEG), both with 95% percentile bootstrap CIs (B = 1000,
windows resampled independently within each group, numpy RandomState(42)).

Backgrounds: cohn BG_uni, BG_rule (>=15 kb from any GENCODE v44 gene, not overlapping a cohn positive),
BG_rule_pc (protein-coding genes only); nt_enhancers BG_uni, BG_rule (not ChromHMM 4-7 in all nine lines),
BG_rule13 (state 13 in all nine lines). Metrics: local GC, TATA flag (>= 1.3 bits/pos), 100-kb GC, L1+L2.
Decision language fixed in advance of looking at the new draws: a background "matches" NEG on a metric
if the pooled difference CI contains 0; otherwise the difference is reported with its CI.

Outputs: results/upgrades/neg_provenance_bg_seeds.csv        per (dataset, group, seed) means
         results/upgrades/neg_provenance_bg_pooled.csv       pooled estimates, diffs vs NEG, f_rule
         results/upgrades/neg_provenance_bg_seeds_windows.csv.gz   per-window features of the new draws
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402
import genome_utils as G  # noqa: E402
import run_neg_provenance as RP  # noqa: E402

SEEDS = (1, 2, 3, 4, 5)
N_BG = RP.N_BG
PFX = RP.PFX
METRICS = ("gc_local", "tata_rate", "gc_100kb", "iso_AT_L1L2")
RULES = {"cohn": ("BG_uni", "BG_rule", "BG_rule_pc"), "nt_enhancers": ("BG_uni", "BG_rule", "BG_rule13")}
MATCH = {"cohn": "BG_rule_pc", "nt_enhancers": "BG_rule13"}


def features(df, tbp, Bn):
    seqs = df["seq"].tolist()
    out = df[["chr", "start", "end"]].copy()
    out["gc_local"] = C.gc_content(seqs)
    out["tata_flag"] = RP.best_tata(seqs, tbp) >= RP.THR
    mid = ((df["start"] + df["end"]) // 2).to_numpy()
    out["gc_100kb"] = G.regional_gc(df["chr"].to_numpy(), mid, Bn)
    iso = G.iso_class(out["gc_100kb"].to_numpy())
    out["isochore"] = [G.ISO_NAMES[i] if i >= 0 else "NA" for i in iso]
    return out


def metric_values(d):
    iso = d["isochore"].fillna("NA")
    return {"gc_local": d["gc_local"].to_numpy(float), "tata_rate": d["tata_flag"].astype(float).to_numpy(),
            "gc_100kb": d["gc_100kb"].to_numpy(float),
            "iso_AT_L1L2": np.where(iso != "NA", iso.isin(["L1", "L2"]).astype(float), np.nan)}


def main():
    t0 = time.time()
    from pyliftover import LiftOver
    fa = G.fasta(upper=True); sizes = G.chrom_sizes(); tbp = C.load_tbp()
    Bn = G.bins(); gi = G.GeneIndex(); H = G.chromhmm()
    allpos = RP.load_cohn_pos_intervals()
    pos_by_chr = {c: (g["start"].to_numpy(), g["end"].to_numpy()) for c, g in allpos.groupby("chr")}
    whole = lambda L: {c: (np.array([0]), np.array([sizes[c] - L + 1])) for c in sizes}  # noqa: E731
    rule = {c: gi.eligible_starts(c, 500, 15000, sizes[c], extra_block=pos_by_chr.get(c)) for c in sizes}
    rule_pc = {c: gi.eligible_starts(c, 500, 15000, sizes[c], extra_block=pos_by_chr.get(c), protein_coding=True)
               for c in sizes}
    lo19to38 = LiftOver(G.CHAIN_19_38)
    parts = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        draws = [("cohn", "BG_uni", RP.sample_windows(fa, rng, N_BG, 500, whole(500))[0]),
                 ("cohn", "BG_rule", RP.sample_windows(fa, rng, N_BG, 500, rule)[0]),
                 ("cohn", "BG_rule_pc", RP.sample_windows(fa, rng, N_BG, 500, rule_pc)[0]),
                 ("nt_enhancers", "BG_uni", RP.sample_windows(fa, rng, N_BG, 200, whole(200))[0]),
                 ("nt_enhancers", "BG_rule", RP.sample_chromhmm_rule(fa, rng, H, lo19to38, N_BG, 200,
                                                                    lambda st: ~np.isin(st, RP.ENH_STATES))[0]),
                 ("nt_enhancers", "BG_rule13", RP.sample_chromhmm_rule(fa, rng, H, lo19to38, N_BG, 200,
                                                                      lambda st: st == 13)[0])]
        for ds, grp, d in draws:
            f = features(d, tbp, Bn); f["dataset"] = ds; f["group"] = grp; f["seed"] = seed
            parts.append(f)
        print(f"seed {seed} done t={time.time() - t0:.0f}s", flush=True)
    NEW = pd.concat(parts, ignore_index=True)
    NEW.to_csv(PFX + "_bg_seeds_windows.csv.gz", index=False, compression="gzip")

    W = pd.read_csv(PFX + "_intervals.csv.gz", low_memory=False,
                    usecols=["chr", "start", "end", "dataset", "group", "gc_local", "tata_flag", "gc_100kb", "isochore"])
    W42 = W[W.group.str.startswith("BG_") & ~W.group.isin(["BG_chr", "BG_rule_chr"])].copy(); W42["seed"] = 42
    ALL = pd.concat([W42, NEW], ignore_index=True)

    per_seed = []
    for (ds, grp, seed), g in ALL.groupby(["dataset", "group", "seed"]):
        mv = metric_values(g)
        per_seed.append(dict(dataset=ds, group=grp, seed=int(seed), n=len(g), **{k: float(np.nanmean(v)) for k, v in mv.items()}))
    PS = pd.DataFrame(per_seed)
    PS.to_csv(PFX + "_bg_seeds.csv", index=False)

    rows = []
    rs = np.random.RandomState(RP.SEED)
    for ds in ("cohn", "nt_enhancers"):
        neg = metric_values(W[(W.dataset == ds) & (W.group == "NEG")])
        pooled = {grp: metric_values(ALL[(ALL.dataset == ds) & (ALL.group == grp)]) for grp in RULES[ds]}
        for met in METRICS:
            N = neg[met][np.isfinite(neg[met])]
            Nb = N[rs.randint(0, len(N), (RP.B, len(N)))].mean(1)
            U = pooled["BG_uni"][met]; U = U[np.isfinite(U)]
            Ub = U[rs.randint(0, len(U), (RP.B, len(U)))].mean(1)
            for grp in RULES[ds]:
                R = pooled[grp][met]; R = R[np.isfinite(R)]
                Rb = Ub if grp == "BG_uni" else R[rs.randint(0, len(R), (RP.B, len(R)))].mean(1)
                sd = PS[(PS.dataset == ds) & (PS.group == grp)][met]
                d_seed = sd.to_numpy() - N.mean()
                lo, hi = RP.ci(Rb - Nb)
                row = dict(dataset=ds, group=grp, metric=met, n_pooled=len(R), n_seeds=len(sd),
                           pooled=float(R.mean()), pooled_lo=RP.ci(Rb)[0], pooled_hi=RP.ci(Rb)[1],
                           seed_mean=float(sd.mean()), seed_sd=float(sd.std(ddof=1)), seed_min=float(sd.min()),
                           seed_max=float(sd.max()), neg=float(N.mean()), bg_minus_neg=float(R.mean() - N.mean()),
                           bg_minus_neg_lo=lo, bg_minus_neg_hi=hi,
                           seed_diff_min=float(d_seed.min()), seed_diff_max=float(d_seed.max()),
                           matches_neg=bool(lo <= 0 <= hi))
                if grp != "BG_uni":
                    f = (U.mean() - R.mean()) / (U.mean() - N.mean())
                    fl, fh = RP.ci((Ub - Rb) / (Ub - Nb))
                    row.update(f_rule=float(f), f_rule_lo=fl, f_rule_hi=fh, is_match_rule=grp == MATCH[ds])
                rows.append(row)
    P = pd.DataFrame(rows)
    P.to_csv(PFX + "_bg_pooled.csv", index=False)
    print(P.to_string(), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
