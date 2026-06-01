#!/usr/bin/env python3
"""run_stats.py -- statistical hardening of the two headline claims.

PART A  Confidence intervals + permutation tests
  A.1  For every motif-only AUROC in the discriminability analysis: a bootstrap
       95% CI (resample the scanned sequences within each class, with replacement)
       and a permutation p-value (shuffle class labels, recompute AUROC). The key
       question -- is each failed-task motif-only AUROC distinguishable from 0.5?
       -- is answered by whether the CI excludes 0.5.
  A.2  Bootstrap 95% CIs for the benchmark LightGBM MCC and AUROC per dataset
       (resample the test set), giving the main results table error bars.

PART B  Is the enhancer-negative contamination real? (now a headline claim)
  B.3  For the suspect datasets (human_enhancers_cohn, human_enhancers_ensembl)
       test whether NEGATIVE sequences carry more TSS/promoter signal than
       POSITIVES, on four measures: TATA/TBP hits, Sp1/KLF GC-box hits, GC
       content, CpG O/E. Report pos-vs-neg means, AUROC, bootstrap CI, MW p.
  B.4  Same four measures on clean controls (human_ocr_ensembl,
       drosophila_enhancers_stark) -- the skew should be specific to the suspects.
  B.5  Contamination rate: fraction of each negative set that looks
       promoter-proximal, by a documented TATA-or-CpG-island rule.

CPU-only; reuses motif_jaspar/motif_match (PSSM scan) and the run_discriminability
curated-TF logic. Everything seeded (42). Outputs: bootstrap_cis.csv,
contamination_analysis.csv, stats_summary.txt.
"""
import argparse
import os

import numpy as np
import kselect
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from scipy import sparse
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import data as gbdata
import models
import motif_jaspar
import motif_match
from run_discriminability import (CURATED_VERT, CURATED_INSECT, resolve_curated,
                                  contrasts_for, collection_for)

HERE = os.path.dirname(os.path.abspath(__file__))
SUSPECTS = ["human_enhancers_cohn", "human_enhancers_ensembl"]
CONTROLS = ["human_ocr_ensembl", "drosophila_enhancers_stark"]
DATASET_ORDER = [
    "human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl",
    "human_ocr_ensembl", "human_ensembl_regulatory", "drosophila_enhancers_stark",
    "dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
]
# TBP/TATA is a low-information 7 bp AT-rich PWM, so a 1.0 bits/pos hit mostly
# tracks AT-content (e.g. AT-rich drosophila shows ~21 "hits"/seq). A specificity-
# controlled 1.3 bits/pos threshold (swept: the suspect negative-set enrichment
# persists there while the controls stay flat) is used for the TATA measure and the
# TATA-box contamination rate. GC/CpG are intrinsic composition measures.


# ---------- fast AUROC + resampling -------------------------------------
def auc_counts(pos, neg):
    """Exact AUROC = P(pos>neg)+0.5 P(pos=neg) for small-integer hit counts."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    m = int(max(pos.max(), neg.max())) + 1
    hp = np.bincount(pos, minlength=m).astype(np.float64)
    hn = np.bincount(neg, minlength=m).astype(np.float64)
    lower = np.cumsum(hn) - hn                      # neg strictly below each value
    return float((hp * (lower + 0.5 * hn)).sum() / (len(pos) * len(neg)))


def auc_float(pos, neg):
    """Rank-based AUROC for continuous measures (ties averaged)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(pos, neg, aucfn, B, rng):
    n_p, n_n = len(pos), len(neg)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = aucfn(pos[rng.randint(0, n_p, n_p)], neg[rng.randint(0, n_n, n_n)])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def perm_p(pos, neg, aucfn, B, rng):
    """Two-sided permutation p for 'separation != 0.5': shuffle class labels."""
    obs = aucfn(pos, neg)
    pool = np.concatenate([pos, neg]); n_p, N = len(pos), len(pool)
    eff = abs(obs - 0.5)
    cnt = 0
    for b in range(B):
        perm = pool[rng.permutation(N)]
        if abs(aucfn(perm[:n_p], perm[n_p:]) - 0.5) >= eff:
            cnt += 1
    return (cnt + 1) / (B + 1)


# ---------- sequence composition measures -------------------------------
def gc_content(codes):
    valid = codes < 4
    gc = ((codes == 1) | (codes == 2)) & valid
    return gc.sum(1) / np.maximum(valid.sum(1), 1)


def cpg_oe(codes):
    valid = codes < 4
    nC = (codes == 1).sum(1).astype(np.float64)
    nG = (codes == 2).sum(1).astype(np.float64)
    Lv = np.maximum(valid.sum(1), 1).astype(np.float64)
    cpg = ((codes[:, :-1] == 1) & (codes[:, 1:] == 2)).sum(1).astype(np.float64)
    return cpg * Lv / np.maximum(nC * nG, 1.0)


# ---------- benchmark metric bootstrap (A.2) ----------------------------
def best_lgbm_k(results_csv, cache_dir, seed):
    # k per dataset is selected on a held-out validation split (kselect.best_k_val),
    # NOT by argmax of test-set MCC; results_csv supplies only n_classes.
    df = pd.read_csv(results_csv)
    df = df[(df["model"] == "lgbm") & (df["status"] == "ok")].copy()
    out = {}
    for ds, g in df.groupby("dataset"):
        out[ds] = dict(k=kselect.best_k_val(ds, cache_dir, seed),
                       n_classes=int(g["n_classes"].iloc[0]))
    return out


def bootstrap_benchmark(dataset, k, cache_dir, seed, B, rng):
    Xtr = sparse.load_npz(os.path.join(cache_dir, f"{dataset}__train__k{k}.npz"))
    ytr = np.load(os.path.join(cache_dir, f"{dataset}__train__y.npy"))
    Xte = sparse.load_npz(os.path.join(cache_dir, f"{dataset}__test__k{k}.npz"))
    yte = np.load(os.path.join(cache_dir, f"{dataset}__test__y.npy"))
    nc = len(np.unique(ytr))
    model = models.build_model("lgbm", seed, nc)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = model.classes_[np.argmax(proba, axis=1)]
    mcc_pt = matthews_corrcoef(yte, pred)
    auc_pt = (roc_auc_score(yte, proba[:, 1]) if nc == 2 else
              roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
    n = len(yte)
    mccs = np.empty(B); aucs = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            mccs[b] = np.nan; continue
        mccs[b] = matthews_corrcoef(yte[idx], pred[idx])
        try:
            aucs[b] = (roc_auc_score(yte[idx], proba[idx, 1]) if nc == 2 else
                       roc_auc_score(yte[idx], proba[idx], multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return (mcc_pt, np.nanpercentile(mccs, 2.5), np.nanpercentile(mccs, 97.5),
            auc_pt, np.nanpercentile(aucs, 2.5), np.nanpercentile(aucs, 97.5), n, nc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "results", "results.csv"))
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--datasets", default="", help="comma subset (default: all)")
    ap.add_argument("--max-per-class", type=int, default=4000)
    ap.add_argument("--thr-bits", type=float, default=1.0,
                    help="general motif-hit threshold (bits/pos) for GC-box/curated TFs")
    ap.add_argument("--tata-bits", type=float, default=1.3,
                    help="specificity-controlled TATA/TBP threshold (bits/pos); TBP is low-IC, "
                         "so 1.0 conflates with AT-content while 1.3 is control-validated")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contam-datasets", default="",
                    help="comma list to run the contamination test (PART B) on (default: GB suspects+controls)")
    ap.add_argument("--contam-suspects", default="",
                    help="which --contam-datasets are labeled 'suspect' (the rest 'control')")
    ap.add_argument("--release", default=motif_jaspar.RELEASE)
    ap.add_argument("--pseudocount", type=float, default=motif_jaspar.PSEUDOCOUNT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    boot_rng = np.random.RandomState(args.seed)        # all resampling draws
    bestk = best_lgbm_k(args.results, args.cache_dir, args.seed)
    datasets = ([d for d in DATASET_ORDER if d in bestk]
                + [d for d in sorted(bestk) if d not in DATASET_ORDER])
    if args.datasets:
        want = set(args.datasets.split(","))
        datasets = [d for d in datasets if d in want]
    contam_set = set(args.contam_datasets.split(",")) if args.contam_datasets else set(SUSPECTS + CONTROLS)
    suspect_set = set(args.contam_suspects.split(",")) if args.contam_suspects else set(SUSPECTS)

    pssms = {"vertebrates": motif_jaspar.load_pssms("vertebrates", args.release, pseudocount=args.pseudocount),
             "insects": motif_jaspar.load_pssms("insects", args.release, pseudocount=args.pseudocount)}
    curated = {"vertebrates": resolve_curated(CURATED_VERT, pssms["vertebrates"])[0],
               "insects": resolve_curated(CURATED_INSECT, pssms["insects"])[0]}
    vbyname = {m["name"].upper(): m for m in pssms["vertebrates"]}
    TBP, SP1 = vbyname["TBP"], vbyname["SP1"]            # for the contamination measures

    print("=" * 92)
    print("STATISTICAL HARDENING: bootstrap CIs + permutation tests; contamination test")
    print(f"seed={args.seed} | boot={args.boot} | perm={args.perm} | thr={args.thr_bits} bits/pos")
    print("=" * 92)

    ci_rows, contam_rows = [], []

    # ---- PART A.2: benchmark metric CIs --------------------------------
    print("\nPART A.2 -- benchmark LightGBM MCC / AUROC bootstrap CIs:")
    for d in datasets:
        k = bestk[d]["k"]
        mcc_pt, mlo, mhi, auc_pt, alo, ahi, n, nc = bootstrap_benchmark(
            d, k, args.cache_dir, args.seed, args.boot, boot_rng)
        for metric, pt, lo, hi in [("mcc", mcc_pt, mlo, mhi), ("auroc", auc_pt, alo, ahi)]:
            ci_rows.append(dict(kind="benchmark", dataset=d, detail1="lgbm", detail2=f"k{k}",
                metric=metric, point=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                ci95=f"[{lo:.3f}, {hi:.3f}]", perm_p="", excludes_0p5="", n=n))
        print(f"  {d:<32} MCC={mcc_pt:.3f} [{mlo:.3f},{mhi:.3f}]  AUROC={auc_pt:.3f} [{alo:.3f},{ahi:.3f}]")

    # ---- PART A.1 + PART B: one sequence pass per dataset --------------
    print("\nPART A.1 -- motif-only AUROC bootstrap CIs + permutation p:")
    for d in datasets:
        coll = collection_for(d)
        ds = gbdata.load_dataset(d, seed=args.seed)
        seqs = ds["train_seqs"] + ds["test_seqs"]
        y = np.concatenate([ds["y_train"], ds["y_test"]])
        classes = ds["classes"]
        sub_rng = np.random.RandomState(args.seed)      # per-dataset, deterministic

        for label, pos_set, neg_set, is_rep in contrasts_for(d, classes, y):
            pi = np.where(np.isin(y, list(pos_set)))[0]
            ni = np.where(np.isin(y, list(neg_set)))[0]
            if len(pi) > args.max_per_class:
                pi = sub_rng.choice(pi, args.max_per_class, replace=False)
            if len(ni) > args.max_per_class:
                ni = sub_rng.choice(ni, args.max_per_class, replace=False)
            pos_codes = motif_match.encode_sequences([seqs[i] for i in pi])
            neg_codes = motif_match.encode_sequences([seqs[i] for i in ni])
            for fam, name, m in curated[coll]:
                pc = motif_match.count_hits_batch(pos_codes, m["pssm"], args.thr_bits)
                nc_ = motif_match.count_hits_batch(neg_codes, m["pssm"], args.thr_bits)
                pt = auc_counts(pc, nc_)
                lo, hi = boot_ci(pc, nc_, auc_counts, args.boot, boot_rng)
                pp = perm_p(pc, nc_, auc_counts, args.perm, boot_rng)
                excl = not (lo <= 0.5 <= hi)
                ci_rows.append(dict(kind="motif_auroc", dataset=d, detail1=label, detail2=name,
                    metric="motif_auroc", point=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                    ci95=f"[{lo:.3f}, {hi:.3f}]", perm_p=float(f"{pp:.2e}"),
                    excludes_0p5=excl, n=len(pi) + len(ni)))
            if is_rep:
                print(f"  {d:<32} ({label}) -- {len(curated[coll])} TFs scored")

            # ---- PART B (representative pos-vs-neg contrast only) --------
            if is_rep and d in contam_set:
                role = "suspect" if d in suspect_set else "control"
                measures = {}
                measures["TATA_TBP_hits"] = (motif_match.count_hits_batch(pos_codes, TBP["pssm"], args.tata_bits),
                                             motif_match.count_hits_batch(neg_codes, TBP["pssm"], args.tata_bits), "int")
                measures["GCbox_SP1_hits"] = (motif_match.count_hits_batch(pos_codes, SP1["pssm"], args.thr_bits),
                                              motif_match.count_hits_batch(neg_codes, SP1["pssm"], args.thr_bits), "int")
                measures["GC_content"] = (gc_content(pos_codes), gc_content(neg_codes), "float")
                measures["CpG_OE"] = (cpg_oe(pos_codes), cpg_oe(neg_codes), "float")
                for mname, (pv, nv, kind) in measures.items():
                    fn = auc_counts if kind == "int" else auc_float
                    pos_arr = pv.astype(np.int64) if kind == "int" else pv
                    neg_arr = nv.astype(np.int64) if kind == "int" else nv
                    a = fn(pos_arr, neg_arr)
                    lo, hi = boot_ci(pos_arr, neg_arr, fn, args.boot, boot_rng)
                    try:
                        _, mw = mannwhitneyu(pv, nv, alternative="two-sided")
                    except ValueError:
                        mw = 1.0
                    contam_rows.append(dict(dataset=d, role=role, measure=mname,
                        pos_mean=round(float(np.mean(pv)), 4), neg_mean=round(float(np.mean(nv)), 4),
                        diff_pos_minus_neg=round(float(np.mean(pv) - np.mean(nv)), 4),
                        neg_enriched=bool(np.mean(nv) > np.mean(pv)),
                        auroc_pos_gt_neg=round(a, 4), auroc_lo=round(lo, 4), auroc_hi=round(hi, 4),
                        mannwhitney_p=float(f"{mw:.2e}"), n_pos=len(pi), n_neg=len(ni)))
                # B.5 contamination rate: fraction carrying >=1 specific TATA box
                def tata_box(codes):
                    return motif_match.count_hits_batch(codes, TBP["pssm"], args.tata_bits) >= 1
                fp = float(tata_box(pos_codes).mean()); fn_ = float(tata_box(neg_codes).mean())
                contam_rows.append(dict(dataset=d, role=role, measure="TATA_box_fraction(>=1)",
                    pos_mean=round(fp, 4), neg_mean=round(fn_, 4),
                    diff_pos_minus_neg=round(fp - fn_, 4), neg_enriched=bool(fn_ > fp),
                    auroc_pos_gt_neg="", auroc_lo="", auroc_hi="", mannwhitney_p="",
                    n_pos=len(pi), n_neg=len(ni)))
                print(f"  [B] {d:<30} ({role}) TATA-box fraction: pos={fp:.1%} neg={fn_:.1%} "
                      f"excess(neg-pos)={fn_-fp:+.1%}")

    cidf = pd.DataFrame(ci_rows)
    cidf.to_csv(os.path.join(args.out_dir, "bootstrap_cis.csv"), index=False)
    cdf = pd.DataFrame(contam_rows)
    cdf.to_csv(os.path.join(args.out_dir, "contamination_analysis.csv"), index=False)

    report = build_report(args, cidf, cdf)
    print("\n" + report)
    with open(os.path.join(args.out_dir, "stats_summary.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote: bootstrap_cis.csv ({len(ci_rows)}), contamination_analysis.csv ({len(contam_rows)}), "
          f"stats_summary.txt under {args.out_dir}")


def build_report(args, cidf, cdf):
    L = []; add = L.append
    add("=" * 92)
    add("STATISTICAL SUMMARY")
    add("=" * 92)
    # A.1: are failed-task best motif AUROCs distinguishable from 0.5?
    add("PART A.1 -- best motif-only AUROC per dataset (does its CI exclude 0.5?):")
    mot = cidf[cidf["kind"] == "motif_auroc"].copy()
    if len(mot):
        mot["sep"] = (mot["point"] - 0.5).abs()
        for d, g in mot.groupby("dataset"):
            best = g.loc[g["sep"].idxmax()]
            verdict = "EXCLUDES 0.5" if best["excludes_0p5"] else "includes 0.5 (NS)"
            add(f"  {d:<32} {best['detail2']:<8} AUROC={best['point']:.3f} "
                f"{best['ci95']}  perm_p={best['perm_p']:.1e}  -> {verdict}")
    # A.2 benchmark error bars
    add("\nPART A.2 -- benchmark LightGBM (point [95% CI]):")
    for d, g in cidf[cidf["kind"] == "benchmark"].groupby("dataset"):
        gm = g.set_index("metric")
        add(f"  {d:<32} MCC={gm.loc['mcc','point']:.3f} {gm.loc['mcc','ci95']}  "
            f"AUROC={gm.loc['auroc','point']:.3f} {gm.loc['auroc','ci95']}")
    # PART B contamination
    add("\nPART B -- enhancer-negative TSS/promoter contamination (neg vs pos):")
    add("  (neg_enriched=True + AUROC<0.5 => promoter signal concentrated in the NEGATIVE set)")
    if len(cdf):
        for d, g in cdf.groupby("dataset"):
            role = g["role"].iloc[0]
            add(f"  {d} [{role}]:")
            for _, r in g.iterrows():
                if r["measure"].startswith("TATA_box_fraction"):
                    add(f"      TATA-box fraction (TSS-core contamination estimate): "
                        f"pos={r['pos_mean']:.1%} neg={r['neg_mean']:.1%} excess(neg-pos)={-r['diff_pos_minus_neg']:+.1%}")
                else:
                    arrow = "neg>pos" if r["neg_enriched"] else "pos>neg"
                    add(f"      {r['measure']:<16} pos={r['pos_mean']:.3f} neg={r['neg_mean']:.3f} "
                        f"({arrow})  AUROC(pos>neg)={r['auroc_pos_gt_neg']:.3f} "
                        f"[{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]  p={r['mannwhitney_p']:.1e}")
    add("\nVerdict logic: contamination is supported if the SUSPECT negatives are enriched for")
    add("TATA/GC-box/GC/CpG (neg>pos, AUROC<0.5, CI excluding 0.5) AND the CONTROLs are not.")
    add("If the suspect skew is weak or also present in controls, claim (2) should be reframed")
    add("as 'weak/suggestive'. Thresholds: GC-box(SP1) >= %.1f bits/pos; TATA(TBP) >= %.1f bits/pos"
        % (args.thr_bits, args.tata_bits))
    add("(specificity-controlled, since TBP is a low-information motif).")
    add("Bootstrap: %d resamples (within-class, with replacement). Permutation: %d label shuffles,"
        % (args.boot, args.perm))
    add("two-sided p = fraction with |AUROC-0.5| >= observed. Seed=%d." % args.seed)
    return "\n".join(L)


if __name__ == "__main__":
    main()
