#!/usr/bin/env python3
"""run_interpretability.py -- do LightGBM's discriminative k-mers map to TF motifs?

Scientific question (BOUNDARY ALIGNMENT): is the k-mer<->motif correspondence
stronger on tasks the benchmark SOLVED (human_nontata_promoters,
human_ensembl_regulatory) than on tasks it FAILED (human_enhancers_cohn,
human_enhancers_ensembl, human_ocr_ensembl)?

Method
------
For each dataset we refit the best-k LightGBM (k from results.csv) and take its
full gain-importance vector over all 4^k k-mers. We score every k-mer against
each JASPAR CORE PWM (vertebrates; insects for drosophila) in absolute log-odds
bits (motif_match.py). Then, PER TF, we test whether that motif's score is
enriched among the important k-mers:

    enrichment z(TF) = ( weighted_mean(motif_score | gain) - background_mean )
                       / permutation_sd     (gain-weighted; exact perm. moments)

We do NOT threshold "best match over the whole collection": with ~880 PWMs that
saturates (60-100% of ALL k-mers match something). The per-TF test does not
saturate -- a single PWM only scores high for k-mers resembling its own motif --
and it directly names the TFs the model relies on. Significance is one-sided with
Benjamini-Hochberg FDR across all usable TFs.

A dataset's top-N k-mers are then called "motif-explained" if their best match to
any FDR-significant enriched TF is >= --explained-bits bits/pos. Negative results
(few/no enriched TFs, low explained fraction) are reported straight -- they mean
the signal is diffuse / non-local, not that the analysis failed.

CPU-only; numpy/scipy/scikit-learn/lightgbm/pandas/pyjaspar(Bio). Seeded.
Outputs (under --out-dir): enriched_tfs.csv, motif_results.csv,
boundary_analysis.csv, top_kmers/<dataset>.csv, motif_interpretation.txt.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import motif_importance
import motif_jaspar
import motif_match

HERE = os.path.dirname(os.path.abspath(__file__))

INSECT_DATASETS = {"drosophila_enhancers_stark"}
SOLVED_REG = ["human_nontata_promoters", "human_ensembl_regulatory"]
FAILED_REG = ["human_enhancers_cohn", "human_enhancers_ensembl", "human_ocr_ensembl"]
REGULATORY = SOLVED_REG + FAILED_REG
COMPOSITIONAL = ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm"]
DATASET_ORDER = [
    "human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl",
    "human_ocr_ensembl", "human_ensembl_regulatory", "drosophila_enhancers_stark",
    "dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
]

# canonical proximal-promoter TF families for the nontata sanity check
CANONICAL_TF = {
    "TATA-box (TBP)":     ("TBP",),
    "GC-box / Sp1-KLF":   ("SP1", "SP2", "SP3", "SP4", "KLF", "EGR"),
    "CCAAT (NF-Y/CEBP)":  ("NFYA", "NFYB", "NFYC", "NFY", "CEBP"),
}


def collection_for(ds):
    return "insects" if ds in INSECT_DATASETS else "vertebrates"


def best_lgbm_per_dataset(results_csv):
    df = pd.read_csv(results_csv)
    df = df[(df["model"] == "lgbm") & (df["status"] == "ok")].copy()
    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
    out = {}
    for ds, g in df.groupby("dataset"):
        r = g.loc[g["mcc"].idxmax()]
        out[ds] = dict(best_k=int(r["k"]), mcc=float(r["mcc"]), n_classes=int(r["n_classes"]))
    return out


def enrichment_z(gain, M):
    """Per-TF gain-weighted enrichment z and one-sided p.

    z(TF) = (weighted_mean(M[:,TF] | gain) - mean(M[:,TF])) / perm_sd, where
    perm_sd uses the exact mean/variance of a weighted mean under random
    permutation of the (fixed) importance weights across k-mers.
    """
    N = len(gain)
    s = gain.sum()
    if s <= 0:
        return np.zeros(M.shape[1]), np.ones(M.shape[1])
    w = gain / s
    bg = M.mean(axis=0)
    wm = w @ M
    var_pop = M.var(axis=0)
    sw2 = float((w ** 2).sum())
    perm_var = max(sw2 - 1.0 / N, 0.0) * var_pop * (N / (N - 1))
    sd = np.sqrt(perm_var)
    z = np.where(sd > 1e-12, (wm - bg) / np.where(sd > 1e-12, sd, 1.0), 0.0)
    return z, norm.sf(z)


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "results", "results.csv"))
    ap.add_argument("--datasets", default="", help="comma-separated subset (default: all in results.csv)")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--n-top", type=int, default=50)
    ap.add_argument("--fdr", type=float, default=0.05, help="BH-FDR for an enriched TF")
    ap.add_argument("--explained-bits", type=float, default=1.5,
                    help="bits/pos for a top k-mer to count as explained by an enriched TF")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--release", default=motif_jaspar.RELEASE)
    ap.add_argument("--pseudocount", type=float, default=motif_jaspar.PSEUDOCOUNT)
    ap.add_argument("--max-tf-per-kmer", type=int, default=8)
    args = ap.parse_args()

    np.random.seed(args.seed)
    score_cache = os.path.join(args.cache_dir, "motif_scores")
    topdir = os.path.join(args.out_dir, "top_kmers")
    os.makedirs(topdir, exist_ok=True)

    best = best_lgbm_per_dataset(args.results)
    datasets = ([d for d in DATASET_ORDER if d in best]
                + [d for d in sorted(best) if d not in DATASET_ORDER])
    if args.datasets:
        want = set(args.datasets.split(","))
        datasets = [d for d in datasets if d in want]
    colls = sorted({collection_for(d) for d in datasets})

    print("=" * 92)
    print("INTERPRETABILITY: do LightGBM's discriminative k-mers map to TF motifs?")
    print(f"JASPAR {args.release} CORE | top-N={args.n_top} | FDR<{args.fdr} | "
          f"explained>={args.explained_bits} bits/pos | pseudocount={args.pseudocount} | seed={args.seed}")
    print("=" * 92)
    pssm_by_coll = {c: motif_jaspar.load_pssms(c, release=args.release, pseudocount=args.pseudocount)
                    for c in colls}
    for c in colls:
        print(f"  {c}: {len(pssm_by_coll[c])} CORE PWMs")

    enriched_rows, motif_rows, boundary, top_by_ds, enriched_by_ds = [], [], [], {}, {}
    print("\nper-dataset (refit best-k LightGBM -> gain -> per-TF enrichment):")
    for d in datasets:
        c, k = collection_for(d), best[d]["best_k"]
        tag = f"{c}_{args.release}_pc{args.pseudocount}"
        gain, names, n_used = motif_importance.fit_and_gain(d, k, args.cache_dir, args.seed)
        M, usable = motif_match.tf_score_matrix(k, pssm_by_coll[c], cache_dir=score_cache, tag=tag)
        z, p = enrichment_z(gain, M)
        fdr = bh_fdr(p)
        enr_mask = (fdr < args.fdr) & (z > 0)
        enr_idx = np.where(enr_mask)[0]
        enr_idx = enr_idx[np.argsort(z[enr_idx])[::-1]]
        enriched_by_ds[d] = [(usable[j]["name"], usable[j]["matrix_id"], float(z[j])) for j in enr_idx]
        for j in enr_idx:
            enriched_rows.append(dict(dataset=d, k=k, collection=c, TF=usable[j]["name"],
                matrix_id=usable[j]["matrix_id"], enrich_z=round(float(z[j]), 3),
                p=float(f"{p[j]:.2e}"), fdr=float(f"{fdr[j]:.2e}")))

        # top-N k-mers by gain (rows of M are in vocabulary order, aligned with names/gain)
        order = np.argsort(gain, kind="stable")[::-1][:args.n_top]
        best_all = M[order].max(axis=1)
        best_all_j = M[order].argmax(axis=1)
        if len(enr_idx):
            Menr = M[order][:, enr_idx]
            best_enr = Menr.max(axis=1)
            best_enr_local = enr_idx[Menr.argmax(axis=1)]
        else:
            best_enr = np.full(len(order), -np.inf)
            best_enr_local = np.full(len(order), -1)
        explained = best_enr >= args.explained_bits

        tk = []
        for rank, gi in enumerate(order, 1):
            kmer = str(names[gi]); ii = rank - 1
            tk.append(dict(rank=rank, kmer=kmer, gain=round(float(gain[gi]), 2),
                best_TF=usable[int(best_all_j[ii])]["name"],
                best_bits_per_pos=round(float(best_all[ii]), 3),
                explained_by_enriched=bool(explained[ii]),
                best_enriched_TF=(usable[int(best_enr_local[ii])]["name"] if explained[ii] else "")))
            # motif_results: one row per (kmer, matched enriched TF) >= explained-bits
            if len(enr_idx):
                row = M[gi, enr_idx]
                hit = enr_idx[np.argsort(row)[::-1]]
                hit = [j for j in hit if M[gi, j] >= args.explained_bits][:args.max_tf_per_kmer]
            else:
                hit = []
            if not hit:
                motif_rows.append(dict(dataset=d, k=k, rank=rank, kmer=kmer,
                    gain=round(float(gain[gi]), 2), best_TF_overall=usable[int(best_all_j[ii])]["name"],
                    best_bits_overall=round(float(best_all[ii]), 3), matched_enriched_TF="",
                    matched_matrix_id="", matched_bits_per_pos="", tf_enrichment_z=""))
            else:
                for j in hit:
                    motif_rows.append(dict(dataset=d, k=k, rank=rank, kmer=kmer,
                        gain=round(float(gain[gi]), 2), best_TF_overall=usable[int(best_all_j[ii])]["name"],
                        best_bits_overall=round(float(best_all[ii]), 3),
                        matched_enriched_TF=usable[j]["name"], matched_matrix_id=usable[j]["matrix_id"],
                        matched_bits_per_pos=round(float(M[gi, j]), 3), tf_enrichment_z=round(float(z[j]), 2)))
        pd.DataFrame(tk).to_csv(os.path.join(topdir, f"{d}.csv"), index=False)
        top_by_ds[d] = pd.DataFrame(tk)

        n_enr = int(len(enr_idx))
        frac_expl = float(np.mean(explained))
        mean_enr_bits = float(np.mean(best_enr[np.isfinite(best_enr)])) if n_enr else float("nan")
        top3 = "; ".join(t[0] for t in enriched_by_ds[d][:3]) if n_enr else "-"
        boundary.append(dict(dataset=d, collection=c, best_k=k,
            bench_mcc=round(best[d]["mcc"], 4), n_top=len(order), n_used_features=n_used,
            n_usable_TF=len(usable), n_enriched_TF=n_enr,
            max_enrich_z=round(float(np.nanmax(z)), 3),
            frac_top_explained=round(frac_expl, 4),
            mean_enriched_bits=round(mean_enr_bits, 4) if n_enr else "",
            top_enriched_TFs=top3))
        print(f"  {d:<32} k={k} mcc={best[d]['mcc']:.3f} | enriched_TF={n_enr:3d} "
              f"max_z={np.nanmax(z):5.1f} explained={frac_expl:4.0%}  top: {top3}")

    bdf = pd.DataFrame(boundary).sort_values("bench_mcc", ascending=False).reset_index(drop=True)
    bdf.to_csv(os.path.join(args.out_dir, "boundary_analysis.csv"), index=False)
    pd.DataFrame(enriched_rows).to_csv(os.path.join(args.out_dir, "enriched_tfs.csv"), index=False)
    pd.DataFrame(motif_rows).to_csv(os.path.join(args.out_dir, "motif_results.csv"), index=False)

    report = build_report(args, bdf, enriched_by_ds)
    print("\n" + report)
    with open(os.path.join(args.out_dir, "motif_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote: enriched_tfs.csv ({len(enriched_rows)}), motif_results.csv ({len(motif_rows)}), "
          f"boundary_analysis.csv, top_kmers/<dataset>.csv, motif_interpretation.txt under {args.out_dir}")


def _spear(bdf, subset, col):
    sub = bdf[bdf["dataset"].isin(subset)]
    if len(sub) < 3 or sub[col].nunique() < 2:
        return None
    rho, pp = spearmanr(sub["bench_mcc"], sub[col])
    return len(sub), rho, pp


def build_report(args, bdf, enriched_by_ds):
    L = []
    add = L.append
    add("=" * 92)
    add("BOUNDARY ANALYSIS  --  per-TF motif enrichment of important k-mers vs benchmark MCC")
    add("=" * 92)
    add("Hypothesis: SOLVED tasks rely on a few strong local TF motifs (many FDR-enriched TFs,")
    add("high explained fraction); FAILED tasks have diffuse / non-local signal (few enriched TFs).")
    add("")
    hdr = (f"{'dataset':<32}{'k':>2}{'mcc':>7}{'enrTF':>6}{'maxz':>7}{'expl':>6}"
           f"  {'top enriched TFs':<34}{'coll':>5}")
    add(hdr)
    add("-" * len(hdr))
    for _, r in bdf.iterrows():
        add(f"{r['dataset']:<32}{r['best_k']:>2}{r['bench_mcc']:>7.3f}{r['n_enriched_TF']:>6}"
            f"{r['max_enrich_z']:>7.1f}{r['frac_top_explained']:>6.0%}  "
            f"{(r['top_enriched_TFs'] or '-')[:34]:<34}{r['collection'][:4]:>5}")
    add("")
    add("enrTF = # TFs with BH-FDR<%.2f positive enrichment;  maxz = strongest TF enrichment z;" % args.fdr)
    add(f"expl = fraction of top-{args.n_top} k-mers matching an enriched TF at >= {args.explained_bits} bits/pos.")

    add("\nSpearman rho (benchmark MCC vs motif correspondence):")
    for label, subset in [("all datasets", list(bdf["dataset"])), ("regulatory only", REGULATORY)]:
        parts = []
        for col in ("n_enriched_TF", "frac_top_explained", "max_enrich_z"):
            s = _spear(bdf, subset, col)
            parts.append(f"{col}: rho={s[1]:+.2f} (p={s[2]:.2f})" if s else f"{col}: n/a")
        n = len(bdf[bdf['dataset'].isin(subset)])
        add(f"  {label:<18} (n={n}): " + " | ".join(parts))

    add("\nCore contrast (regulatory tasks):")
    for name, grp in [("SOLVED (nontata, ensembl_regulatory)", SOLVED_REG),
                      ("FAILED (cohn, enhancers_ensembl, ocr)", FAILED_REG),
                      ("CONTROL (demo: composition-driven)", COMPOSITIONAL)]:
        g = bdf[bdf["dataset"].isin(grp)]
        if len(g):
            add(f"  {name:<40} mean enriched_TF={g['n_enriched_TF'].mean():5.1f}  "
                f"mean explained={g['frac_top_explained'].mean():4.0%}  mean mcc={g['bench_mcc'].mean():.3f}")

    add("\nSanity check -- enriched TFs in human_nontata_promoters (canonical promoter elements):")
    enr = dict((nm.upper(), zz) for nm, _, zz in enriched_by_ds.get("human_nontata_promoters", []))
    for elem, prefixes in CANONICAL_TF.items():
        hits = sorted({nm: zz for nm, zz in enr.items() if any(nm.startswith(p) for p in prefixes)}.items(),
                      key=lambda kv: -kv[1])
        present = "YES" if hits else "no "
        shown = ", ".join(f"{nm}(z={zz:.1f})" for nm, zz in hits[:5]) if hits else "-"
        add(f"  [{present}] {elem:<18} {shown}")
    add("  Expectation: this is the NON-TATA promoter set, so TBP/TATA absence is the biologically")
    add("  correct (specificity) result; GC-box/Sp1-KLF is the expected strong positive.")

    add("\nInterpretation:")
    add("  * A high enriched-TF count / explained fraction means the model's most discriminative")
    add("    k-mers really are the cores of specific TF motifs; few/none means diffuse or")
    add("    compositional signal rather than single-motif-local.")
    add("  * The demo sets (human_or_worm, coding_vs_intergenomic) are composition-driven tasks,")
    add("    not TF-motif tasks -- high MCC with low motif enrichment there is expected and supports")
    add("    the refined claim: correspondence tracks whether the TASK is motif-local, not solvability.")
    add("  * Reported straight: enrichment uses an analytic (normal-approx) permutation null with")
    add("    BH-FDR; n is small per-dataset; the background is uniform over k-mers (not GC-matched);")
    add("    datasets use different k (each k is scored against its own PWM-score matrix).")
    return "\n".join(L)


if __name__ == "__main__":
    main()
