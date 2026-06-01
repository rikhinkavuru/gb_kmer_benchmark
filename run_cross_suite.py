#!/usr/bin/env python3
"""run_cross_suite.py -- combine both benchmark suites and test REPLICATION.

Reads the per-suite analysis outputs (bootstrap_cis.csv, motif_discriminability.csv,
contamination_analysis.csv) for Genomic Benchmarks (results/) and the Nucleotide
Transformer downstream tasks (results/nt/) and produces:
  * cross_suite_summary.csv -- every task from both suites: benchmark MCC/AUROC
    (95% CI), max positive-direction motif-only AUROC (95% CI), TATA-neg excess,
    contamination flag.
  * a printed replication verdict -- does 'motif discriminability predicts
    learnability' hold on the independent suite, and is the negative-set TATA
    contamination Genomic-Benchmarks-specific or broader?

Nulls are reported straight. CPU-only; pure pandas/numpy/scipy.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
# tasks that are NOT TF-motif-driven (different motif class / composition) -> boundary cases
NON_TF_TASKS = {"nt_splice_sites_all", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm",
                "dummy_mouse_enhancers_ensembl"}


def _bench_ci(analysis_dir):
    df = pd.read_csv(os.path.join(analysis_dir, "bootstrap_cis.csv"))
    b = df[df["kind"] == "benchmark"]
    out = {}
    for ds, g in b.groupby("dataset"):
        gm = g.set_index("metric")
        out[ds] = dict(mcc=float(gm.loc["mcc", "point"]), mcc_lo=float(gm.loc["mcc", "ci_lo"]),
                       mcc_hi=float(gm.loc["mcc", "ci_hi"]), auroc=float(gm.loc["auroc", "point"]),
                       auroc_lo=float(gm.loc["auroc", "ci_lo"]), auroc_hi=float(gm.loc["auroc", "ci_hi"]))
    return out, df


def _motif_pos(analysis_dir, ci_df):
    """Max positive-direction motif-only AUROC per dataset (representative contrast)."""
    md = pd.read_csv(os.path.join(analysis_dir, "motif_discriminability.csv"))
    md = md[md["representative"] == True]
    mot = ci_df[ci_df["kind"] == "motif_auroc"]
    out = {}
    for ds, g in md.groupby("dataset"):
        r = g.loc[g["motif_auroc"].idxmax()]            # most positive-direction TF
        tf = r["TF"]
        ci = mot[(mot["dataset"] == ds) & (mot["detail2"] == tf)]
        lo = float(ci["ci_lo"].iloc[0]) if len(ci) else np.nan
        hi = float(ci["ci_hi"].iloc[0]) if len(ci) else np.nan
        out[ds] = dict(max_pos_motif_auroc=float(r["motif_auroc"]), best_pos_TF=str(tf),
                       motif_lo=lo, motif_hi=hi)
    return out


def _contam(analysis_dir):
    p = os.path.join(analysis_dir, "contamination_analysis.csv")
    if not os.path.exists(p):
        return {}
    cd = pd.read_csv(p)
    out = {}
    for ds, g in cd.groupby("dataset"):
        d = {}
        tata = g[g["measure"] == "TATA_TBP_hits"]
        if len(tata):
            t = tata.iloc[0]
            d.update(tata_auroc=float(t["auroc_pos_gt_neg"]), tata_lo=float(t["auroc_lo"]),
                     tata_hi=float(t["auroc_hi"]), tata_neg_enriched=bool(t["neg_enriched"]))
        frac = g[g["measure"].astype(str).str.startswith("TATA_box_fraction")]
        if len(frac):
            d["tata_neg_excess"] = -float(frac.iloc[0]["diff_pos_minus_neg"])   # neg - pos
        out[ds] = d
    return out


def _flag(c, task):
    """Contamination flag. A neg-TATA excess only means *negative-set contamination*
    when the positive class is NOT itself promoter-defined; on promoter tasks the
    positives are (wholly or partly) TATA-depleted, so neg>pos TATA is expected and
    is reported as confounded, not contamination."""
    if not c or "tata_auroc" not in c:
        return ""
    is_promoter = "promoter" in task
    neg_tata = c.get("tata_neg_enriched") and c["tata_auroc"] < 0.5 and c.get("tata_hi", 1) < 0.5
    pos_tata = (not c.get("tata_neg_enriched")) and c["tata_auroc"] > 0.5 and c.get("tata_lo", 0) > 0.5
    if pos_tata:
        return "TATA-in-positives (expected; assay control)"
    if neg_tata:
        if is_promoter:
            return "neg-TATA CONFOUNDED (pos class is promoter/TATA-depleted; not contamination)"
        excess = c.get("tata_neg_excess", 0)
        return f"CONTAMINATED-neg-TATA ({'strong' if excess >= 0.15 else 'weak'})"
    return "clean/ambiguous"


def build_suite(label, results_csv, analysis_dir):
    bench, ci_df = _bench_ci(analysis_dir)
    motif = _motif_pos(analysis_dir, ci_df)
    contam = _contam(analysis_dir)
    rdf = pd.read_csv(results_csv)
    rdf = rdf[rdf["status"] == "ok"]
    ncls = {ds: int(g["n_classes"].iloc[0]) for ds, g in rdf.groupby("dataset")}
    rows = []
    for ds in bench:
        b = bench[ds]; m = motif.get(ds, {}); c = contam.get(ds, {})
        rows.append(dict(suite=label, task=ds, n_classes=ncls.get(ds, ""),
            bench_mcc=round(b["mcc"], 4), mcc_ci=f"[{b['mcc_lo']:.3f},{b['mcc_hi']:.3f}]",
            bench_auroc=round(b["auroc"], 4), auroc_ci=f"[{b['auroc_lo']:.3f},{b['auroc_hi']:.3f}]",
            max_pos_motif_auroc=round(m.get("max_pos_motif_auroc", float("nan")), 4),
            motif_ci=(f"[{m['motif_lo']:.3f},{m['motif_hi']:.3f}]"
                      if m.get("motif_lo") == m.get("motif_lo") else ""),
            best_pos_TF=m.get("best_pos_TF", ""),
            tata_neg_excess=(round(c["tata_neg_excess"], 4) if "tata_neg_excess" in c else ""),
            tf_motif_task=(ds not in NON_TF_TASKS),
            contamination_flag=_flag(c, ds)))
    return rows


def _spear(df):
    sub = df.dropna(subset=["max_pos_motif_auroc"])
    if len(sub) < 3 or sub["max_pos_motif_auroc"].nunique() < 2:
        return None
    rho, p = spearmanr(sub["bench_mcc"], sub["max_pos_motif_auroc"])
    return len(sub), rho, p


def build_report(df):
    L = []; add = L.append
    add("=" * 100)
    add("CROSS-SUITE REPLICATION  --  benchmark learnability vs positive-direction motif discriminability")
    add("=" * 100)
    hdr = (f"{'suite':<14}{'task':<34}{'cls':>3}{'MCC':>7} {'MCC 95% CI':<16}"
           f"{'pos-motif AUROC':>16} {'best TF':<10}{'TATAexc':>8}  contamination")
    add(hdr); add("-" * len(hdr))
    for _, r in df.iterrows():
        te = f"{r['tata_neg_excess']:+.1%}" if r["tata_neg_excess"] != "" else "  -"
        add(f"{r['suite'][:13]:<14}{r['task']:<34}{r['n_classes']:>3}{r['bench_mcc']:>7.3f} "
            f"{r['mcc_ci']:<16}{r['max_pos_motif_auroc']:>16.3f} {r['best_pos_TF'][:10]:<10}"
            f"{te:>8}  {r['contamination_flag']}")

    add("\n--- REPLICATION TEST 1: does motif discriminability predict learnability? ---")
    add("Spearman(benchmark MCC, max positive-direction motif-only AUROC):")
    for label in df["suite"].unique():
        s = _spear(df[df["suite"] == label])
        if s:
            add(f"  {label:<24} (n={s[0]}): rho={s[1]:+.2f}  p={s[2]:.2f}")
        tf = df[(df["suite"] == label) & (df["tf_motif_task"])]
        s2 = _spear(tf)
        if s2:
            add(f"  {label+' (TF-motif tasks only)':<24} (n={s2[0]}): rho={s2[1]:+.2f}  p={s2[2]:.2f}")
    s = _spear(df)
    if s:
        add(f"  {'POOLED both suites':<24} (n={s[0]}): rho={s[1]:+.2f}  p={s[2]:.2f}")
    s = _spear(df[df["tf_motif_task"]])
    if s:
        add(f"  {'POOLED, TF-motif only':<24} (n={s[0]}): rho={s[1]:+.2f}  p={s[2]:.2f}")

    add("\n--- REPLICATION TEST 2: is negative-set TATA contamination broader than GB? ---")
    flagged = df[df["contamination_flag"].astype(str).str.startswith("CONTAMINATED")]
    if len(flagged):
        for _, r in flagged.iterrows():
            add(f"  {r['suite']:<22} {r['task']:<32} {r['contamination_flag']} "
                f"(neg TATA excess {r['tata_neg_excess']:+.1%})")
    else:
        add("  no task flagged CONTAMINATED-neg-TATA")
    pos_ctrl = df[df["contamination_flag"].astype(str).str.startswith("TATA-in-positives")]
    if len(pos_ctrl):
        add("  positive controls (TATA correctly concentrated in positives, validating the assay):")
        for _, r in pos_ctrl.iterrows():
            add(f"    {r['suite']:<22} {r['task']}")
    add("\nVerdict is interpreted in the accompanying message / README; nulls reported straight.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gb-results", default=os.path.join(HERE, "results", "results.csv"))
    ap.add_argument("--gb-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--nt-results", default=os.path.join(HERE, "results", "results_nt.csv"))
    ap.add_argument("--nt-dir", default=os.path.join(HERE, "results", "nt"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cross_suite_summary.csv"))
    args = ap.parse_args()
    rows = (build_suite("GenomicBench", args.gb_results, args.gb_dir)
            + build_suite("NucTransformer", args.nt_results, args.nt_dir))
    df = pd.DataFrame(rows).sort_values(["suite", "bench_mcc"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    report = build_report(df)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out} ({len(df)} tasks) + interpretation.")


if __name__ == "__main__":
    main()
