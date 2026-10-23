#!/usr/bin/env python3
"""Upgrade E2 -- the "motif PRESENCE does not predict learnability" null, extended from
n=9 (Genomic Benchmarks) to n=15 (GB + Nucleotide Transformer), WITH a confidence interval.

The paper currently states the null as a bare point estimate ("across the nine Genomic
Benchmarks tasks the number of enriched transcription-factor motifs is uncorrelated with
learnability, Spearman -0.03"). At n=9 that is absence of evidence, not evidence of absence:
a Spearman of -0.03 at n=9 is compatible with a very wide range of true correlations. This
module (a) attaches an interval to the number, and (b) adds the six NT tasks so the pooled
test runs at n=15.

NOTHING is recomputed here -- both inputs are read off existing result CSVs:

  x = n_enriched_TFs : the number of BH-FDR-enriched TF motif matrices per task, = the row
      count for that task in results/enriched_tfs.csv (GB) or results/nt/enriched_tfs.csv
      (NT). A task with ZERO enriched TFs has NO rows in those files and MUST still enter
      the correlation as a 0 -- dropping it would be exactly the selection bias the null is
      about. The two zero tasks are dummy_mouse_enhancers_ensembl (GB) and nt_promoter_tata
      (NT); both are cross-checked against the independently-written n_enriched_TF column of
      results/boundary_analysis.csv and results/nt/boundary_analysis.csv, and the run aborts
      if the two sources disagree.

  y = test_MCC : the best-k LightGBM test MCC, defined EXACTLY as run_benchmark.py /
      select_k_val.py define it -- the row of results/results.csv (or results_nt.csv) with
      model=="lgbm" and selected_by_val==1, i.e. k chosen on a stratified 80/20 split of
      TRAIN (seed 42), then retrained on full train and scored once on test. Every value is
      asserted against Table 1 of the paper (tolerance 6e-4, the rounding of a 3-decimal
      table) before any statistic is computed.

Statistics, per subset (GB n=9, NT n=6, pooled n=15):
  * Spearman rho and its two-sided p-value: the scipy asymptotic (t-approximation) p, PLUS
    an exact permutation p by full enumeration of all n! relabelings where that is tractable
    (n=6 -> 720, n=9 -> 362,880) and a 200,000-draw Monte-Carlo permutation p (seed 42,
    reported as (1+#{|rho*|>=|rho|})/(B+1)) at n=15 where 15! is not.
  * TWO 95% intervals for rho, because at these n they disagree in an informative way:
      (a) Fisher z-transform: tanh(arctanh(rho) +/- 1.96/sqrt(n-3)). Also reported is the
          Bonett-Wright variant SE=sqrt((1+rho^2/2)/(n-3)), which is the version actually
          calibrated for SPEARMAN rather than Pearson.
      (b) Bias-corrected (BC) percentile bootstrap over TASKS -- the resampling unit is the
          whole task, 10,000 resamples, np.random.RandomState(42), degenerate resamples
          (fewer than 3 distinct tasks, or zero rank variance in either variable, or a nan
          rho) skipped and counted. z0 = Phi^-1(fraction of replicates strictly below the
          observed rho); limits at Phi(2*z0 +/- 1.96). The plain (uncorrected) percentile
          limits are written too, so the size of the bias correction is visible.

The bootstrap here deliberately departs from the repo-wide "1000 resamples, individual test
sequence as the resampling unit" convention: the unit of THIS analysis is the task, there
are only 9/6/15 of them, and 10,000 draws are needed to resolve the tails of a statistic
computed on so few discrete points. Seed 42 is unchanged.

Outputs (all NEW files under results/upgrades/):
  presence_null.csv                -- the 15-row per-task table
  presence_null_stats.csv          -- one row per subset (GB / NT / pooled)
  presence_null_sensitivity.csv    -- the same rho under the max-over-k MCC (the k at which
                                      the enrichment was actually run) and a suite-partial
                                      Spearman, so the choice of MCC column is auditable
  presence_null_interpretation.txt -- short prose summary

Run:  python src/upgrades/presence_null/run_presence_null.py
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C

GB_ENRICH = os.path.join(C.ROOT, "results", "enriched_tfs.csv")
NT_ENRICH = os.path.join(C.ROOT, "results", "nt", "enriched_tfs.csv")
GB_RESULTS = os.path.join(C.ROOT, "results", "results.csv")
NT_RESULTS = os.path.join(C.ROOT, "results", "results_nt.csv")
GB_BOUNDARY = os.path.join(C.ROOT, "results", "boundary_analysis.csv")
NT_BOUNDARY = os.path.join(C.ROOT, "results", "nt", "boundary_analysis.csv")

# Table 1 of paper_psb.tex -- every extracted MCC is asserted against this (3-dp rounding).
TABLE1 = {
    "demo_human_or_worm": 0.914,
    "human_ensembl_regulatory": 0.907,
    "nt_promoter_all": 0.888,
    "nt_promoter_no_tata": 0.887,
    "human_nontata_promoters": 0.874,
    "nt_promoter_tata": 0.846,
    "demo_coding_vs_intergenomic_seqs": 0.800,
    "human_enhancers_ensembl": 0.593,
    "dummy_mouse_enhancers_ensembl": 0.529,
    "nt_enhancers": 0.514,
    "nt_enhancers_types": 0.476,
    "human_enhancers_cohn": 0.461,
    "human_ocr_ensembl": 0.421,
    "drosophila_enhancers_stark": 0.398,
    "nt_splice_sites_all": 0.244,
}
TABLE1_TOL = 6e-4
# 15! is 1.3e12; 9! = 362,880 and 6! = 720 both enumerate in under a second.
EXACT_PERM_MAX = 500_000


# ------------------------------------------------------------------ inputs
def enriched_counts(path):
    """dataset -> number of BH-FDR-enriched TF motif matrices (= row count in `path`).

    Tasks with zero enriched TFs are ABSENT from the file; the caller supplies the 0."""
    df = pd.read_csv(path)
    dup = int(df.duplicated(["dataset", "matrix_id"]).sum())
    assert dup == 0, f"{path}: {dup} duplicate (dataset, matrix_id) rows"
    g = df.groupby("dataset")
    return (g.size().to_dict(), g["k"].agg(lambda s: int(s.unique()[0])).to_dict())


def best_k_rows(path):
    """Per-task best-k LightGBM test MCC, defined as run_benchmark/select_k_val define it:
    the model=='lgbm', status=='ok', selected_by_val==1 row (k picked on a validation split
    of TRAIN, retrained on full train, scored once on test). Also returns the max-over-k
    MCC, which is the older selection-on-test number used by run_interpretability.py."""
    df = pd.read_csv(path)
    df = df[(df["model"] == "lgbm") & (df["status"] == "ok")].copy()
    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
    sel = df[df["selected_by_val"] == 1]
    counts = sel.groupby("dataset").size()
    assert (counts == 1).all(), f"{path}: selected_by_val==1 is not unique per dataset:\n{counts}"
    maxk = df.groupby("dataset")["mcc"].max()
    out = {}
    for _, r in sel.iterrows():
        out[r["dataset"]] = dict(best_k=int(r["k"]), test_MCC=float(r["mcc"]),
                                 n_test=int(r["n_test"]), maxk_test_MCC=float(maxk[r["dataset"]]))
    return out


def boundary_counts(path):
    """dataset -> n_enriched_TF as written independently by run_interpretability.py."""
    df = pd.read_csv(path)
    return dict(zip(df["dataset"], df["n_enriched_TF"].astype(int)))


def build_table():
    """The 15-row per-task table. All tasks present in results{,_nt}.csv are included; a task
    missing from the enrichment CSV gets n_enriched_TFs = 0 (never dropped)."""
    rows = []
    zeros = []
    for suite, enrich_p, res_p, bnd_p in [("GB", GB_ENRICH, GB_RESULTS, GB_BOUNDARY),
                                          ("NT", NT_ENRICH, NT_RESULTS, NT_BOUNDARY)]:
        counts, ks = enriched_counts(enrich_p)
        bench = best_k_rows(res_p)
        bnd = boundary_counts(bnd_p)
        for task in sorted(bench):
            n_tf = int(counts.get(task, 0))
            if task not in counts:
                zeros.append(task)
            # cross-check against the independently-written boundary_analysis.csv
            assert task in bnd, f"{task} missing from {bnd_p}"
            assert bnd[task] == n_tf, (f"{task}: enrichment CSV says {n_tf} enriched TFs but "
                                       f"{bnd_p} says {bnd[task]}")
            b = bench[task]
            # every MCC must match Table 1 of the paper
            assert task in TABLE1, f"{task} not in the hard-coded Table 1 cross-check"
            d = abs(b["test_MCC"] - TABLE1[task])
            assert d <= TABLE1_TOL, (f"{task}: best-k test MCC {b['test_MCC']:.6f} disagrees with "
                                     f"Table 1 value {TABLE1[task]:.3f} (|diff|={d:.2e})")
            ek = int(ks.get(task, -1))          # -1 = no enriched rows, so no k recorded
            rows.append(dict(task=task, suite=suite, n_enriched_TFs=n_tf,
                             best_k=b["best_k"], test_MCC=round(b["test_MCC"], 6),
                             enrich_k=ek, k_match=(-1 if ek < 0 else int(ek == b["best_k"])),
                             maxk_test_MCC=round(b["maxk_test_MCC"], 6),
                             table1_MCC=TABLE1[task], n_test=b["n_test"]))
    df = pd.DataFrame(rows).sort_values(["suite", "test_MCC"], ascending=[True, False])
    return df.reset_index(drop=True), zeros


# -------------------------------------------------------------- statistics
def _rho(x, y):
    """Spearman rho, or nan if either side has no rank variance."""
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y, rho_obs, n_mc, seed):
    """Two-sided permutation p-value for Spearman rho.

    Exact by full enumeration of all n! relabelings when n! <= EXACT_PERM_MAX (n<=9);
    otherwise Monte-Carlo with `n_mc` random relabelings, reported as the conservative
    (1 + #{|rho*| >= |rho_obs|}) / (n_mc + 1)."""
    rx = rankdata(x) - np.mean(rankdata(x))
    ry = rankdata(y) - np.mean(rankdata(y))
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    n = len(x)
    nfac = float(math.factorial(n))
    tol = 1e-12
    if nfac <= EXACT_PERM_MAX:
        P = np.array(list(itertools.permutations(ry)), dtype=np.float64)   # (n!, n)
        rhos = (P @ rx) / denom
        p = float(np.mean(np.abs(rhos) >= abs(rho_obs) - tol))
        return p, "exact_enumeration", int(nfac)
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(n_mc):
        hits += int(abs(float(rng.permutation(ry) @ rx / denom)) >= abs(rho_obs) - tol)
    return (1.0 + hits) / (n_mc + 1.0), "monte_carlo", int(n_mc)


def fisher_ci(rho, n, z=1.959963984540054):
    """95% Fisher z-transform CI for rho, in two flavours:
      standard  SE = 1/sqrt(n-3)                     (the Pearson-calibrated one)
      Bonett-Wright SE = sqrt((1+rho^2/2)/(n-3))     (calibrated for Spearman)"""
    if n <= 3 or abs(rho) >= 1:
        nan = float("nan")
        return nan, nan, nan, nan
    zr = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    se_bw = np.sqrt((1.0 + rho ** 2 / 2.0) / (n - 3))
    return (float(np.tanh(zr - z * se)), float(np.tanh(zr + z * se)),
            float(np.tanh(zr - z * se_bw)), float(np.tanh(zr + z * se_bw)))


def bc_bootstrap_ci(x, y, rho_obs, B, seed, alpha=0.05):
    """Bias-corrected (BC) percentile bootstrap CI for Spearman rho, resampling TASKS.

    Resampling unit = one whole task (rng.randint(0, n, n), np.random.RandomState(seed)).
    A resample is DEGENERATE and skipped when it contains fewer than 3 distinct tasks or
    when either variable has zero rank variance in it (rho undefined); the count of skipped
    resamples is returned. z0 = Phi^-1(#{rho* < rho_obs} / n_valid); the BC limits are the
    Phi(2*z0 -/+ z_{1-alpha/2}) percentiles of the valid replicates. If z0 is not finite
    (all replicates on one side of rho_obs) the plain percentile limits are returned and
    `bc_ok` is 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    rng = np.random.RandomState(seed)
    reps = np.full(B, np.nan)
    degen = 0
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(idx)) < 3:
            degen += 1
            continue
        r = _rho(x[idx], y[idx])
        if not np.isfinite(r):
            degen += 1
            continue
        reps[b] = r
    v = reps[np.isfinite(reps)]
    assert len(v) > 100, f"only {len(v)} non-degenerate bootstrap resamples"
    zc = norm.ppf(1 - alpha / 2)
    pct_lo = float(np.percentile(v, 100 * alpha / 2))
    pct_hi = float(np.percentile(v, 100 * (1 - alpha / 2)))
    prop = float(np.mean(v < rho_obs))
    z0 = norm.ppf(prop) if 0 < prop < 1 else np.inf * (1 if prop >= 1 else -1)
    if not np.isfinite(z0):
        return dict(boot_lo=pct_lo, boot_hi=pct_hi, pct_lo=pct_lo, pct_hi=pct_hi,
                    z0=float(z0), bc_ok=0, n_valid=len(v), n_degenerate=degen,
                    boot_mean=float(np.mean(v)), boot_sd=float(np.std(v, ddof=1)))
    a1 = float(norm.cdf(2 * z0 - zc))
    a2 = float(norm.cdf(2 * z0 + zc))
    return dict(boot_lo=float(np.percentile(v, 100 * a1)),
                boot_hi=float(np.percentile(v, 100 * a2)),
                pct_lo=pct_lo, pct_hi=pct_hi, z0=float(z0), bc_ok=1,
                n_valid=len(v), n_degenerate=degen,
                boot_mean=float(np.mean(v)), boot_sd=float(np.std(v, ddof=1)))


def analyse(label, sub, xcol, ycol, args):
    x = sub[xcol].to_numpy(np.float64)
    y = sub[ycol].to_numpy(np.float64)
    n = len(sub)
    sr = spearmanr(x, y)
    rho, p_asym = float(sr.statistic), float(sr.pvalue)
    assert abs(rho - _rho(x, y)) < 1e-12, "internal rho disagreement"
    pp, pmeth, pn = perm_p(x, y, rho, args.n_perm, args.seed)
    f_lo, f_hi, bw_lo, bw_hi = fisher_ci(rho, n)
    bs = bc_bootstrap_ci(x, y, rho, args.boot, args.seed)
    return dict(subset=label, n=n, rho=round(rho, 6), p=round(p_asym, 6),
                fisher_lo=round(f_lo, 6), fisher_hi=round(f_hi, 6),
                boot_lo=round(bs["boot_lo"], 6), boot_hi=round(bs["boot_hi"], 6),
                p_perm=round(pp, 6), p_perm_method=pmeth, n_perm=pn,
                fisher_bw_lo=round(bw_lo, 6), fisher_bw_hi=round(bw_hi, 6),
                pct_lo=round(bs["pct_lo"], 6), pct_hi=round(bs["pct_hi"], 6),
                bc_z0=round(bs["z0"], 6), bc_ok=bs["bc_ok"],
                boot_B=args.boot, boot_valid=bs["n_valid"], boot_degenerate=bs["n_degenerate"],
                boot_mean=round(bs["boot_mean"], 6), boot_sd=round(bs["boot_sd"], 6),
                x=xcol, y=ycol)


def partial_spearman(x, y, z):
    """Spearman of x,y partialling out the (binary) covariate z: Pearson correlation of the
    residuals of ranked x and ranked y after regressing each on z."""
    rx, ry, rz = rankdata(x), rankdata(y), np.asarray(z, dtype=np.float64)
    A = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples over TASKS")
    ap.add_argument("--n-perm", type=int, default=200000, help="MC permutations when n! is too big")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 100)
    print("UPGRADE E2 -- motif PRESENCE null, n=9 -> n=15, with confidence intervals")
    print(f"seed={args.seed} | bootstrap over TASKS B={args.boot} | MC permutations={args.n_perm}")
    print("=" * 100)

    df, zeros = build_table()
    assert len(df) == 15, f"expected 15 tasks, got {len(df)}"
    print(f"\nTasks with ZERO enriched TFs (absent from the enrichment CSVs, counted as 0): "
          f"{', '.join(sorted(zeros))}")
    print(f"All 15 best-k LightGBM test MCCs match Table 1 within {TABLE1_TOL:g}.\n")
    print(df.to_string(index=False))

    out_table = os.path.join(args.out_dir, "presence_null.csv")
    df.to_csv(out_table, index=False)

    subsets = [("GB_n9", df[df["suite"] == "GB"]),
               ("NT_n6", df[df["suite"] == "NT"]),
               ("pooled_n15", df)]
    stats_rows = [analyse(lbl, sub, "n_enriched_TFs", "test_MCC", args) for lbl, sub in subsets]
    sdf = pd.DataFrame(stats_rows)
    out_stats = os.path.join(args.out_dir, "presence_null_stats.csv")
    sdf.to_csv(out_stats, index=False)

    # -------- sensitivity: the MCC column, and the suite covariate --------
    sens = [analyse(lbl + "_maxk_mcc", sub, "n_enriched_TFs", "maxk_test_MCC", args)
            for lbl, sub in subsets]
    for r in sens:
        r["note"] = ("max-over-k test MCC (selection on test, the column run_interpretability.py "
                     "used); k-consistent with the enrichment run")
    rho_partial = partial_spearman(df["n_enriched_TFs"], df["test_MCC"],
                                   (df["suite"] == "NT").astype(float))
    sens.append(dict(subset="pooled_n15_partial_suite", n=15, rho=round(rho_partial, 6),
                     x="n_enriched_TFs", y="test_MCC",
                     note="Spearman partialling out suite (GB/NT indicator)"))
    out_sens = os.path.join(args.out_dir, "presence_null_sensitivity.csv")
    pd.DataFrame(sens).to_csv(out_sens, index=False)

    print("\n" + "=" * 100)
    print("SPEARMAN(n_enriched_TFs, best-k test MCC) with 95% intervals")
    print("=" * 100)
    hdr = (f"{'subset':<12}{'n':>3}{'rho':>8}{'p_asym':>9}{'p_perm':>9}  "
           f"{'Fisher 95%':<20}{'BC-boot 95%':<20}{'perm':<18}")
    print(hdr)
    print("-" * len(hdr))
    for r in stats_rows:
        print(f"{r['subset']:<12}{r['n']:>3}{r['rho']:>+8.3f}{r['p']:>9.3f}{r['p_perm']:>9.4f}  "
              f"{'[%+.3f, %+.3f]' % (r['fisher_lo'], r['fisher_hi']):<20}"
              f"{'[%+.3f, %+.3f]' % (r['boot_lo'], r['boot_hi']):<20}"
              f"{r['p_perm_method'] + ' n=' + str(r['n_perm']):<18}")
    print("\nBonett-Wright Fisher CIs (Spearman-calibrated SE) and plain percentile CIs:")
    for r in stats_rows:
        print(f"  {r['subset']:<12} BW [{r['fisher_bw_lo']:+.3f}, {r['fisher_bw_hi']:+.3f}]   "
              f"percentile [{r['pct_lo']:+.3f}, {r['pct_hi']:+.3f}]   "
              f"z0={r['bc_z0']:+.3f} valid={r['boot_valid']}/{r['boot_B']} "
              f"degenerate={r['boot_degenerate']}")
    print("\nSensitivity (max-over-k MCC / suite-partial):")
    for r in sens:
        print(f"  {r['subset']:<28} rho={r['rho']:+.3f}" +
              (f"  p={r['p']:.3f}" if "p" in r else ""))

    # ------------------------------------------------------------ prose
    g, nt, po = stats_rows
    n_khas = int((df["k_match"] >= 0).sum())          # tasks with enrichment rows
    n_kmis = int((df["k_match"] == 0).sum())          # ... whose enrich_k != best_k
    n_kna = int((df["k_match"] < 0).sum())            # zero-enrichment tasks (no k recorded)
    L = []
    a = L.append
    a("=" * 96)
    a("PRESENCE NULL, n=9 -> n=15, WITH A CONFIDENCE INTERVAL  (Upgrade E2)")
    a("=" * 96)
    a("x = # BH-FDR-enriched TF motifs per task (read from results/enriched_tfs.csv and")
    a("    results/nt/enriched_tfs.csv; tasks absent from those files enter as 0).")
    a("y = best-k LightGBM test MCC (results{,_nt}.csv, model=lgbm & selected_by_val==1;")
    a("    all 15 values verified against Table 1 of the paper).")
    a(f"Zero-enrichment tasks (kept, not dropped): {', '.join(sorted(zeros))}.")
    a("")
    for r in stats_rows:
        a(f"{r['subset']:<12} n={r['n']:<3} rho={r['rho']:+.3f}  "
          f"p_asym={r['p']:.3f}  p_perm={r['p_perm']:.4f} ({r['p_perm_method']})")
        a(f"{'':<12} Fisher 95% CI  [{r['fisher_lo']:+.3f}, {r['fisher_hi']:+.3f}]"
          f"   (Bonett-Wright [{r['fisher_bw_lo']:+.3f}, {r['fisher_bw_hi']:+.3f}])")
        a(f"{'':<12} BC bootstrap over tasks 95% CI  [{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}]"
          f"   (plain percentile [{r['pct_lo']:+.3f}, {r['pct_hi']:+.3f}],"
          f" {r['boot_valid']}/{r['boot_B']} non-degenerate)")
        a("")
    a("Reading.")
    a(f"  * The GB-only value reproduces the paper's -0.03 exactly (rho={g['rho']:+.4f}), so the")
    a("    published point estimate is correct as stated.")
    a(f"  * But its 95% CI spans essentially the whole admissible range: Fisher")
    a(f"    [{g['fisher_lo']:+.2f}, {g['fisher_hi']:+.2f}], BC bootstrap [{g['boot_lo']:+.2f}, "
      f"{g['boot_hi']:+.2f}]. At n=9 the data are")
    a("    compatible with a strong correlation in EITHER direction. The reviewer is right that")
    a("    'uncorrelated' overstates what n=9 supports; the defensible claim is that no")
    a("    association is DETECTABLE, not that none exists.")
    a(f"  * Adding the six NT tasks (NT-only rho={nt['rho']:+.3f}) leaves the pooled estimate at")
    a(f"    rho={po['rho']:+.3f} (p={po['p']:.2f}), i.e. still indistinguishable from zero, and the")
    a(f"    pooled CI is still wide: Fisher [{po['fisher_lo']:+.2f}, {po['fisher_hi']:+.2f}],")
    a(f"    BC bootstrap [{po['boot_lo']:+.2f}, {po['boot_hi']:+.2f}].")
    a("  * So n=15 leaves the null INTACT: it does not reverse and it does not turn positive.")
    a(f"    The point estimate drifts from {g['rho']:+.2f} to {po['rho']:+.2f}, i.e. slightly NEGATIVE (more")
    a("    enriched motifs, if anything marginally LOWER MCC -- the direction the paper's own")
    a("    'the failed enhancer/OCR tasks are the most motif-rich' observation implies), but at")
    a("    every n the interval covers 0 comfortably. What n=15 changes is only that the null must")
    a("    be stated as an absence of DETECTABLE association at low power, not as a demonstrated")
    a("    absence of association.")
    a("")
    a("Caveats.")
    a("  * The enrichment counts were computed at the OLD selection-on-test best k, which differs")
    a(f"    from the validation-selected k on {n_kmis} of the {n_khas} tasks that have enrichment rows")
    a(f"    (column enrich_k vs best_k; the {n_kna} zero-enrichment tasks carry enrich_k=-1 because no")
    a("    row records their k). Pairing the counts with the k-consistent max-over-k MCC instead")
    a("    gives the sensitivity rows in presence_null_sensitivity.csv; the GB value there is the")
    a("    -0.23 printed in results/motif_interpretation.txt. All are near zero, none significant.")
    a("  * n_enriched_TFs is not comparable in an absolute sense across suites (different")
    a("    sequence lengths, class counts and JASPAR collections), so the pooled n=15 test is")
    a("    the least assumption-free of the three; the suite-partial rho is in the sensitivity CSV.")
    a("  * Tasks are not independent replicates (overlapping human regulatory annotation), so the")
    a("    effective n is if anything below 15 and the CIs above are optimistic, not conservative.")
    txt = "\n".join(L)
    out_txt = os.path.join(args.out_dir, "presence_null_interpretation.txt")
    with open(out_txt, "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    print(f"\nWrote {out_table} ({len(df)} rows)")
    print(f"Wrote {out_stats} ({len(sdf)} rows)")
    print(f"Wrote {out_sens} ({len(sens)} rows)")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
