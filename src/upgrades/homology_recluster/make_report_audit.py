#!/usr/bin/env python3
"""Interpretation text for the VERIFICATION-AUDIT scripts (run_audit.py, run_e1_reconcile.py,
run_linclust_calibration.py, run_covmode_sensitivity.py), generated ONLY from their CSVs. All of these analyses
are POST HOC (added after independent verification) and are labelled as such.
Writes results/upgrades/homology_recluster_audit_interpretation.txt. No computation beyond reading/formatting.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402

R, P = _hc.RES, _hc.PREFIX


def rd(name):
    p = os.path.join(R, f"{P}_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def ci(lo, hi):
    return "" if lo != lo else f"[{lo:.3f},{hi:.3f}]"


def f3(x):
    return "n/a" if x != x else f"{x:.3f}"


def main():
    L = []; add = L.append
    add("=" * 110)
    add("HOMOLOGY RE-CLUSTERING -- VERIFICATION AUDIT (POST HOC; added 2026-09-24 after independent verification).")
    add("Every number below is read from results/upgrades/homology_recluster_{audit_*,e1_*,calibration*,covmode}.csv.")
    add("=" * 110)

    cl = rd("audit_class")
    if cl is not None:
        add("A. WHAT THE id80 PURGE REMOVES, PER CLASS (audit_class.csv): n test, fraction removed, fraction that are exact")
        add("   (either-strand) copies of a SAME-label train sequence, and the ORIGINAL model's accuracy (= class recall) on")
        add("   removed vs kept test sequences. Tasks where the purge removes >= 5 % of the test set:")
        g = cl[cl.threshold == "id80"]
        tt = g.groupby("task").apply(lambda d: d.n_removed.sum() / d.n_test.sum(), include_groups=False)
        for t in [t for t in _hc.TASKS if t in tt.index and tt[t] >= 0.05]:
            for _, r in g[g.task == t].iterrows():
                add(f"   {t:<34} class {int(r.cls)}: n={int(r.n_test):>6} removed {r.frac_removed:6.1%}  exact same-label copy "
                    f"{r.frac_exact_same_label:6.1%}  acc removed {f3(r.acc_removed)} vs kept {f3(r.acc_kept)}")
        add("")

    at = rd("audit_task")
    if at is not None:
        a8 = at[at.threshold == "id80"].set_index("task")
        add("B. PURGE EVALUABILITY (a class keeping < 10 test sequences = class eliminated; audit_task.csv):")
        ce = a8[a8.class_eliminated]
        add("   id80: " + ("; ".join(f"{t} kept {r.kept_per_class}" for t, r in ce.iterrows()) if len(ce) else "none"))
        a5 = at[at.threshold == "id50"].set_index("task")
        ce5 = a5[a5.class_eliminated]
        add("   id50: " + ("; ".join(f"{t} kept {r.kept_per_class}" for t, r in ce5.iterrows()) if len(ce5) else "none"))
        add("")
        add("C. EXACT (either-strand) TEST/TRAIN DUPLICATES THAT MMseqs2 DID NOT PUT IN A TRAIN COMPONENT (audit_task.csv)")
        add("   and the purge MCC when they are also removed ('purge+exact'); exact twins straddling the seed-42 cluster re-split.")
        for th in ("id80", "id50"):
            g = at[(at.threshold == th) & ((at.n_exact_missed_by_clusters > 0) |
                                           (at.n_pooled_in_exact_groups_straddling_resplit > 0))]
            if not len(g):
                add(f"   {th}: none in any task")
            for _, r in g.iterrows():
                add(f"   {th} {r.task:<34} missed {int(r.n_exact_missed_by_clusters):>4} of {int(r.n_test_exact_dup)} exact test dups "
                    f"(median len {r.missed_len_median if r.missed_len_median == r.missed_len_median else float('nan'):.0f} bp, "
                    f"max {r.missed_len_max if r.missed_len_max == r.missed_len_max else float('nan'):.0f}); purge MCC "
                    f"{f3(r.purge_mcc)} -> purge+exact {f3(r.purge_plus_exact_mcc)} (d {r.purge_plus_exact_minus_purge:+.4f}); "
                    f"re-split test seqs with an exact twin in re-split train: {int(r.n_resplit_test_with_exact_twin_in_train)}")
        s11 = ", ".join(f"{t} {int(r.n_all_len_lt11)}" for t, r in a8.iterrows() if r.n_all_len_lt11 > 0) or "none"
        s50 = ", ".join(f"{t} {int(r.n_test_len_lt50)} (min {int(r.len_min)} bp)" for t, r in a8.iterrows()
                        if r.n_test_len_lt50 > 0) or "none"
        add(f"   pooled sequences shorter than 11 bp (cannot be seeded by MMseqs2, k = 11): {s11}; test sequences < 50 bp: {s50}")
        if "n_pooled_mostly_N" in a8.columns:
            sN = "; ".join(f"{t} {int(r.n_pooled_mostly_N)} pooled (per label {r.mostly_N_per_label}), of them {int(r.n_missed_mostly_N)} "
                           f"missed exact test dups" for t, r in a8.iterrows() if r.n_pooled_mostly_N > 0) or "none"
            add(f"   sequences that are >= 50 % N (unalignable; identical copies are trivially 'duplicates'): {sN}")
        add("")
        add("D. KIND OF HOMOLOGY (id80; purged test sequences with a direct train edge; best train identity)")
        for t, r in a8.iterrows():
            if r.n_test_direct_train_edge >= 20:
                add(f"   {t:<34} n={int(r.n_test_direct_train_edge):>6}  median best identity {r.best_train_identity_median:.3f}  "
                    f">= 0.99: {r.frac_best_train_identity_ge_099:6.1%}  < 0.90: {r.frac_best_train_identity_lt_090:6.1%}  "
                    f"exact same-label test copies: {int(r.n_test_exact_same_label)}")
        add("")

    cb = rd("audit_clusterboot")
    if cb is not None:
        add("E. COMPONENT-LEVEL vs SEQUENCE-LEVEL 95 % CIs (audit_clusterboot.csv; whole id80 components resampled)")
        g = cb[cb.threshold == "id80"]
        for q in ("purge_mcc", "resplit_cluster_mcc", "cluster_minus_random"):
            h = g[g.quantity == q].copy()
            if not len(h):
                continue
            h["w_seq"] = h.seq_hi - h.seq_lo; h["w_comp"] = h.comp_hi - h.comp_lo
            add(f"   {q}: median width ratio component/sequence {np.nanmedian(h.w_comp / h.w_seq):.2f} "
                f"(max {np.nanmax(h.w_comp / h.w_seq):.2f}, {h.loc[(h.w_comp / h.w_seq).idxmax(), 'task']})")
        h = g[g.quantity == "cluster_minus_random"]
        flip = h[(np.sign(h.seq_lo) == np.sign(h.seq_hi)) != (np.sign(h.comp_lo) == np.sign(h.comp_hi))]
        add("   cluster - random: CI excludes 0 at sequence level but not at component level (or vice versa): "
            + (", ".join(f"{r.task} seq {ci(r.seq_lo, r.seq_hi)} comp {ci(r.comp_lo, r.comp_hi)}" for _, r in flip.iterrows())
               if len(flip) else "none"))
        add("")

    c0, c1 = rd("calibration"), rd("calibration_linclust")
    if c0 is not None and c1 is not None:
        add("F. WHY SEARCH + COMPONENTS AND NOT linclust (synthetic calibration; recall of planted pairs, + / - strand)")
        for idv in (95, 90, 85, 80):
            s = c0[(c0.threshold == "id80") & (c0.nominal_identity == idv)].set_index("strand").recall
            parts = [f"search+components {s.get('+', np.nan):.2f}/{s.get('-', np.nan):.2f}"]
            for m in sorted(c1.method.unique()):
                u = c1[(c1.method == m) & (c1.threshold == "id80") & (c1.nominal_identity == idv)].set_index("strand").recall
                parts.append(f"{m} {u.get('+', np.nan):.2f}/{u.get('-', np.nan):.2f}")
            add(f"   id80, {idv}% identity: " + "; ".join(parts))
        add(f"   decoys falsely merged: search {int(c0.decoys_falsely_merged.max())}/400; linclust "
            f"{int(c1.decoys_falsely_merged.max())}/400")
        add("")

    cv = rd("covmode")
    if cv is not None:
        add("G. COVERAGE-RULE SENSITIVITY (variable-length tasks; covmode.csv): shorter sequence >= 80 % covered (cov-mode 1)")
        add("   vs both sequences >= 80 % covered (primary, cov-mode 0); same cached predictions and purge null")
        for _, r in cv.iterrows():
            add(f"   {r.task:<34}{r.threshold:<5} test-with-train-homolog {r.frac_test_in_train_cluster_covmode0:.3f} -> "
                f"{r.frac_test_in_train_cluster:.3f}; purge effect {r.homology_effect_covmode0:+.3f} -> {r.homology_effect:+.3f} "
                f"(robust |d| <= 0.02: {r.robust_to_cov_rule})")
        add("")

    rc, ds = rd("e1_reconcile"), rd("e1_draws_summary")
    if rc is not None:
        add("H. COMPOSITION EQUALIZATION (E1) RECONCILED WITH composition_clean.csv (human_enhancers_cohn, original data)")
        for _, r in rc.iterrows():
            add(f"   {r.arm:<15} k-mer MCC: replay {r.replay_kmer_mcc:.4f} | published {r.published_kmer_mcc:.4f} | this module "
                f"{r.module_kmer_mcc:.4f};  comp-only MCC replay/published/module {r.replay_comp_only_mcc:.4f}/"
                f"{r.published_comp_only_mcc:.4f}/{r.module_comp_only_mcc:.4f};  k-mer AUROC {r.replay_kmer_auroc:.3f}/"
                f"{r.published_kmer_auroc:.3f}/{r.module_kmer_auroc:.3f};  replay == published: {r.replay_equals_published}")
    if ds is not None:
        add("   20 matching draws of the same equalization (e1_draws_summary.csv): median [min, max]")
        for _, r in ds.iterrows():
            add(f"   {r.data:<9} test negatives kept {int(r.n_test_neg_kept_median)}; frac predicted positive "
                f"{r.frac_pred_pos_median:.4f} [{r.frac_pred_pos_min:.4f},{r.frac_pred_pos_max:.4f}] -> MCC-degenerate: "
                f"{r.mcc_degenerate}; k-mer MCC {r.kmer_mcc_median:.3f} [{r.kmer_mcc_min:.3f},{r.kmer_mcc_max:.3f}]; "
                f"k-mer AUROC {r.kmer_auroc_median:.3f} [{r.kmer_auroc_min:.3f},{r.kmer_auroc_max:.3f}]; TATA AUROC "
                f"{r.tata_auroc_median:.3f} [{r.tata_auroc_min:.3f},{r.tata_auroc_max:.3f}]; GC AUROC {r.gc_auroc_median:.3f} "
                f"[{r.gc_auroc_min:.3f},{r.gc_auroc_max:.3f}]")
        add("")
    out = os.path.join(R, f"{P}_audit_interpretation.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
