#!/usr/bin/env python3
"""Step 6 -- interpretation text + compact figure, generated ONLY from the result CSVs.

Reads results/upgrades/homology_recluster_{redundancy,eval,seeds_summary,interventions,positional,routes,
route_labels,route1,table,calibration,labels_addendum,fix_*,extra_*}.csv (whichever exist) and writes
  results/upgrades/homology_recluster_interpretation.txt
  results/upgrades/figures/homology_recluster_mcc.{pdf,png}   (IEEE single column, 3.5 in)
No computation beyond reading/formatting/counting. CPU-only, torch-free.

Revised 2026-09-24 after an independent audit (verifier addendum): the headline leads with the
PRE-REGISTERED homology-inflation calls; the 0.05 materiality cut is labelled post hoc; non-evaluable
purges (class eliminated) are excluded from counts; label counts are given both by the pre-registered
diagnostic rule and strictly by label; re-split contrasts are given as split-seed mean and range; the
Route-3 equalization readout is the AUROC (the equalized-arm MCC is degenerate); the figure shows the
split-seed mean cluster re-split with its range and is saved with a tight bounding box.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402

R = _hc.RES
P = _hc.PREFIX
SHORT = {"demo_human_or_worm": "human_or_worm", "human_ensembl_regulatory": "ensembl_regulatory",
         "nt_promoter_all": "nt_promoter_all", "nt_promoter_no_tata": "nt_promoter_no_tata",
         "human_nontata_promoters": "nontata_promoters", "nt_promoter_tata": "nt_promoter_tata",
         "demo_coding_vs_intergenomic_seqs": "coding_vs_intergen.", "human_enhancers_ensembl": "enhancers_ensembl",
         "dummy_mouse_enhancers_ensembl": "mouse_enhancers", "nt_enhancers": "nt_enhancers",
         "nt_enhancers_types": "nt_enhancers_types", "human_enhancers_cohn": "enhancers_cohn",
         "human_ocr_ensembl": "ocr_ensembl", "drosophila_enhancers_stark": "drosophila_stark",
         "nt_splice_sites_all": "nt_splice_sites"}
MATERIAL = -0.05        # POST-HOC materiality cut for the purge homology effect (effect-size view only)


def rd(name):
    p = os.path.join(R, f"{P}_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def f3(x):
    return "  n/a" if x != x else f"{x:.3f}"


def ci(lo, hi):
    return "" if lo != lo else f"[{lo:.3f},{hi:.3f}]"


def build_text():
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(_hc.CKPT_DIR, "cluster_*.csv")))]
    red = pd.concat(parts, ignore_index=True) if parts else None      # per-task checkpoints = source of truth
    ev, iv, pos = rd("eval"), rd("interventions"), rd("positional")
    rl, r1, tab, cal = rd("route_labels"), rd("route1"), rd("table"), rd("calibration")
    sd, add_, cred = rd("seeds_summary"), rd("labels_addendum"), rd("class_redundancy")
    fx_ex, fx_ed = None, rd("fix_edges")   # C1 exact-duplicate check was replaced by run_audit.py Q3 (audit_task.csv)
    # C1/C4/C5 of an earlier fix_checks.py were removed as duplicates of run_audit.py / run_e1_reconcile.py / run_covmode_sensitivity.py
    fx_e1, e1rec, fx_cov, lin = rd("e1_draws_summary"), rd("e1_reconcile"), rd("covmode"), rd("calibration_linclust")
    xc = rd("extra_cleaning")
    L = []; add = L.append
    add("=" * 110)
    add("HOMOLOGY RE-CLUSTERING (Reviewer 1, comment 3) -- redundancy, homology-clean re-evaluation, interventions,")
    add("route labels. Every number below is read from results/upgrades/homology_recluster_*.csv.")
    add("=" * 110)
    # ---------------- 0. headline (computed) ----------------
    if ev is not None:
        add("0. HEADLINE (id80 primary; all values computed from the CSVs named below)")
        if red is not None:
            r8 = red[red.threshold == "id80"].set_index("task")
            add(f"   test sequences with an id80 train homolog: median {r8.frac_test_in_train_cluster.median():.1%}, "
                f"range {r8.frac_test_in_train_cluster.min():.1%}-{r8.frac_test_in_train_cluster.max():.1%}; "
                f"tasks >= 10%: " + ", ".join(f"{t} {v:.0%}" for t, v in r8.frac_test_in_train_cluster.sort_values(ascending=False).items() if v >= 0.10))
        for th in ("id80", "id50"):
            g = ev[(ev.threshold == th) & (ev.status == "ok")]
            fl = g[g.homology_inflated == True]  # noqa: E712
            tasks = sorted(set(fl.task), key=_hc.TASKS.index)
            add(f"   PRE-REGISTERED HOMOLOGY-INFLATED calls ({th}; eval.csv homology_inflated: purge delta CI<0 AND MCC < null 2.5th pct, "
                f"OR cluster < random re-split with CI<0): {len(tasks)}/15 tasks -- "
                + ", ".join(f"{t} [{'+'.join(sorted(set(fl[fl.task == t].protocol)))}]" for t in tasks))
        pu = ev[(ev.protocol == "test_purge") & (ev.threshold == "id80")].set_index("task")
        ok = pu[pu.status == "ok"]
        he = ok["homology_effect"]
        add("   Effect size of the purge (homology_effect = MCC_purged - mean of the random-removal null; eval.csv):")
        add(f"     >= |{MATERIAL}| (POST-HOC materiality cut): " + ", ".join(f"{t} {v:+.3f}" for t, v in he.sort_values().items() if v <= MATERIAL))
        add(f"     {MATERIAL} < e <= -0.005: " + ", ".join(f"{t} {v:+.3f}" for t, v in he.sort_values().items() if MATERIAL < v <= -0.005))
        add("     |e| < 0.005: " + ", ".join(f"{t} {v:+.3f}" for t, v in he.sort_values().items() if abs(v) < 0.005))
        ne = pu[pu.status != "ok"]
        for t, r in ne.iterrows():
            add(f"     NOT EVALUABLE: {t} purge status '{r.status}' (kept per class {r.get('kept_per_class', '')}); "
                f"accuracy of the same model on the removed test sequences {r.get('acc_removed', np.nan):.3f} vs kept {r.get('acc_kept', np.nan):.3f}")
        if "homology_effect_strat" in pu.columns:
            for t, r in pu[pu.homology_effect_strat.notna()].iterrows():
                add(f"     {t}: binary-class-matched effect {r.homology_effect:+.3f} (p_null={r.p_null:.3f}) but SUBTYPE-matched null "
                    f"effect {r.homology_effect_strat:+.3f} (p={r.p_null_strat:.3f}); kept per subtype {r.kept_per_stratum}; accuracy on "
                    f"removed {r.acc_removed:.3f} vs kept {r.acc_kept:.3f} -> memorisation and loss of the strong subclass are confounded")
        if "acc_removed" in ok.columns:
            add("     accuracy on removed (train-homolog) vs kept test sequences, same model: " + ", ".join(
                f"{t} {r.acc_removed:.3f}/{r.acc_kept:.3f}" for t, r in ok.sort_values("homology_effect").iterrows()
                if r.homology_effect <= -0.005))
        if sd is not None:
            s8 = sd[sd.threshold == "id80"].set_index("task")
            add("   Cluster-aware minus random re-split MCC (id80), split-seed mean [min,max] over seeds 42-44 (seeds_summary.csv): "
                + "; ".join(f"{t} {r.delta_mean:+.3f} [{r.delta_min:+.3f},{r.delta_max:+.3f}]"
                            for t, r in s8.sort_values("delta_mean").iterrows() if r.delta_mean <= -0.02))
            if tab is not None and "resplit_id80_inflated_seed_robust" in tab.columns:
                rob = tab[tab.resplit_id80_inflated_seed_robust == True].task.tolist()  # noqa: E712
                s42 = tab[tab.resplit_id80_inflated_prereg == True].task.tolist()  # noqa: E712
                add(f"   re-split inflation call (id80): seed 42 (pre-registered) {len(s42)}/15 {s42}; seed-robust "
                    f"(delta < 0 for all 3 split seeds) {len(rob)}/15 {rob}")
        if rl is not None:
            for th in ("id80", "id50"):
                g = rl[rl.threshold == th]
                ev_p = g[~g.preservation_purgeMCC.astype(str).str.startswith("n/a")]
                add(f"   route labels ({th}): purge (prevalence-standardised MCC) preserved {int((ev_p.preservation_purgeMCC == 'preserved').sum())}/{len(ev_p)} "
                    f"evaluable (strict, label unchanged: {int((ev_p.preservation_purgeMCC_strict == 'preserved').sum())}/{len(ev_p)}); "
                    f"cluster re-split seed 42: pre-registered diagnostic rule {int((g.preservation == 'preserved').sum())}/{len(g)}, "
                    f"label unchanged {int((g.label_after == g.label_before).sum())}/{len(g)}; split-seed-mean re-split label unchanged "
                    f"{int((g.label_unchanged_seedmean == True).sum())}/{int(g.label_unchanged_seedmean.notna().sum())}")  # noqa: E712
                ch = g[g.label_after != g.label_before]
                add(f"     relabelled at seed 42: " + ", ".join(f"{r.task} ({r.label_before}->{r.label_after})" for _, r in ch.iterrows()))
            if add_ is not None:
                a8 = add_[add_.threshold == "id80"]
                ha = a8[a8.relabel_homology_attributable_seedmean == True]  # noqa: E712
                rp = a8[a8.relabel_repooling_seedmean == True]  # noqa: E712
                add("     split-seed-mean relabels ATTRIBUTABLE TO HOMOLOGY (cluster label != random-re-split label; labels_addendum.csv): "
                    + (", ".join(f"{r.task} ({r.label_before}->{r.label_cluster_seedmean}; cluster {r.mcc_cluster_seedmean:.3f} vs random "
                                 f"{r.mcc_random_seedmean:.3f})" for _, r in ha.iterrows()) or "none"))
                add("     relabels shared with the random re-split (re-pooling / split design, not homology): "
                    + (", ".join(f"{r.task} ({r.label_before}->{r.label_cluster_seedmean}; random {r.mcc_random_seedmean:.3f})"
                                 for _, r in rp.iterrows()) or "none"))
        if r1 is not None and "subset" in r1.columns:
            q = r1[r1.x == "published_disc"]
            pr = q[q.primary == True]  # noqa: E712
            if len(pr):
                p_ = pr.iloc[0]
                add(f"   Route-1 PRIMARY test (published discriminability vs cluster re-split id80 MCC, all 11 TF tasks): "
                    f"rho={p_.rho:+.3f} p={p_.p:.3f}")
            g = q[q.subset == "all_TF"].set_index("y")
            add("   Route-1 (all TF tasks): " + "; ".join(f"{y} n={int(g.loc[y, 'n'])} rho={g.loc[y, 'rho']:+.3f} p={g.loc[y, 'p']:.3f}"
                                                   for y in ["published_MCC", "purge_id80_stdMCC", "resplit_id80_MCC",
                                                             "resplit_random_MCC", "resplit_id80_seedmean_MCC",
                                                             "resplit_random_seedmean_MCC"] if y in g.index))
            g2 = q[q.subset == "excl_NT_enhancer_tasks"].set_index("y")
            add("   Route-1 without the two NT enhancer tasks (provided test split out of distribution by design): " + "; ".join(
                f"{y} n={int(g2.loc[y, 'n'])} rho={g2.loc[y, 'rho']:+.3f} p={g2.loc[y, 'p']:.3f}"
                for y in ["published_MCC", "purge_id80_stdMCC", "resplit_id80_MCC", "resplit_random_MCC"] if y in g2.index))
            g3 = q[q.subset == "paper_9task_subset"].set_index("y")
            if len(g3):
                add("   Route-1 on the SUBMITTED PAPER's 'homology-clean' 9-task subset (Table II: all TF tasks minus human_nontata_promoters and "
                    "nt_enhancers, provided-split MCCs), now with MMseqs2 homology-clean MCCs: " + "; ".join(
                        f"{y} n={int(g3.loc[y, 'n'])} rho={g3.loc[y, 'rho']:+.3f} p={g3.loc[y, 'p']:.4f}"
                        for y in ["published_MCC", "resplit_id80_MCC", "resplit_id80_seedmean_MCC", "resplit_random_MCC",
                                  "purge_id80_MCC", "purge_id80_stdMCC"] if y in g3.index))
            g4 = q[q.subset == "excl_nt_enhancers_types"].set_index("y")
            if "purge_id80_MCC" in g.index:
                add(f"   Route-1 purge (id80) all TF tasks: n={int(g.loc['purge_id80_MCC', 'n'])} rho={g.loc['purge_id80_MCC', 'rho']:+.3f} "
                    f"p={g.loc['purge_id80_MCC', 'p']:.4f} (nt_enhancers_types not evaluable: class eliminated); same 10 tasks, "
                    f"published MCC: rho={g4.loc['published_MCC', 'rho']:+.3f} p={g4.loc['published_MCC', 'p']:.4f}")
        if iv is not None:
            e = iv[(iv.task == "human_enhancers_cohn") & (iv.experiment == "E1_composition")].set_index(["data", "arm"])
            if ("id80", "original") in e.index and ("id80", "comp_equalized") in e.index:
                o, q_ = e.loc[("id80", "original")], e.loc[("id80", "comp_equalized")]
                add(f"   Route-3 on id80 homology-clean cohn (interventions.csv E1): TATA motif-only AUROC {o.tata_auroc:.3f} "
                    f"{ci(o.tata_lo, o.tata_hi)} -> {q_.tata_auroc:.3f} {ci(q_.tata_lo, q_.tata_hi)} after composition equalization; "
                    f"GC AUROC {o.gc_auroc:.3f} -> {q_.gc_auroc:.3f}; k-mer AUROC {o.kmer_auroc:.3f} {ci(o.kmer_auroc_lo, o.kmer_auroc_hi)} -> "
                    f"{q_.kmer_auroc:.3f} {ci(q_.kmer_auroc_lo, q_.kmer_auroc_hi)} (equalized-arm MCC {q_.kmer_mcc:.3f} is degenerate: "
                    f"{q_.retained_neg_frac:.1%} of negatives kept, test ~96% positive)")
        if fx_e1 is not None:
            add("   equalized-arm readout stability over repeated matching draws (e1_draws_summary.csv): " + "; ".join(
                f"{r.task} [{r.data}] {int(r.n_draws)} draws: k-mer MCC {r.kmer_mcc_min:.3f}-{r.kmer_mcc_max:.3f} vs AUROC "
                f"{r.kmer_auroc_min:.3f}-{r.kmer_auroc_max:.3f}, TATA AUROC {r.tata_auroc_min:.3f}-{r.tata_auroc_max:.3f}"
                for _, r in fx_e1.iterrows()))
        if xc is not None:
            for t_ in ("human_enhancers_cohn", "nt_enhancers"):
                g = xc[(xc.task == t_) & (xc.data == "id80")].set_index("arm")
                if not {"uncleaned", "comp_equalized", "placebo_comp_equalized", "tata_flag", "placebo_tata_flag"} <= set(g.index):
                    continue
                u, ce, pce, tf_, ptf = (g.loc[a] for a in ("uncleaned", "comp_equalized", "placebo_comp_equalized",
                                                          "tata_flag", "placebo_tata_flag"))
                kx = lambda r: f"{r.kmer_auroc:.3f}" if r.kmer_auroc == r.kmer_auroc else "n/a"
                add(f"   Route-3 cleaning vs SIZE-MATCHED RANDOM-REMOVAL PLACEBO, id80 homology-clean {t_} (extra_cleaning.csv, run_extra.py X1): "
                    f"TATA AUROC uncleaned {u.tata_auroc:.3f} {ci(u.tata_lo, u.tata_hi)}; TATA-flag {tf_.tata_auroc:.3f} vs placebo "
                    f"{ptf.tata_auroc:.3f}; composition-equalized {ce.tata_auroc:.3f} {ci(ce.tata_lo, ce.tata_hi)} vs placebo "
                    f"{pce.tata_auroc:.3f} {ci(pce.tata_lo, pce.tata_hi)}; k-mer AUROC uncleaned {kx(u)}, equalized {kx(ce)} vs placebo {kx(pce)}")
        # the other baselines the reviewer asked to re-run (fix round 1): frozen-FM probe, LR floor, end-to-end fine-tuning
        fm = rd("extra_fm")
        if fm is not None:
            pc = {"HyenaDNA-tiny": "sep", "DNABERT-2-117M": "mean"}   # the submitted paper's headline cells (fm_paired*.csv)
            for dv in ("original", "id80"):
                c_ = fm[(fm.data == dv) & (fm["head"] == "lgbm") & (fm.arm == "gc_match")]
                c_ = c_[c_.apply(lambda r: r.pooling == pc.get(r.model), axis=1)]
                pl = fm[(fm.data == dv) & (fm.arm == "placebo_gc_match")]
                add(f"   Frozen-FM probe test_effect (gc_match, paper's cells: HyenaDNA sep / DNABERT-2 mean, LightGBM head) [{dv}] "
                    "(extra_fm.csv, run_extra.py X4): " + "; ".join(
                        f"{r.model} {r.task} {r.test_effect_auroc:+.3f}{ci(r.test_effect_auroc_lo, r.test_effect_auroc_hi)}"
                        for _, r in c_.iterrows())
                    + f"; placebo_gc_match CIs covering 0: {int(((pl.test_effect_auroc_lo <= 0) & (pl.test_effect_auroc_hi >= 0)).sum())}/{len(pl)}")
        x7 = rd("extra_lr")
        if x7 is not None:
            add("   LR floor (extra_lr.csv, X7): LR/LightGBM tier agreement " + "; ".join(
                f"{u} {int(g.tier_agree.sum())}/{len(g)}" for u, g in x7.groupby("unit", sort=False)))
        fh = rd("ft_contrasts")
        if fh is not None:
            import make_report_extra as MX
            ppath = os.path.join(R, "fm_finetune_v2_contrasts.csv")
            rh = MX._ft_rules(fh, rd("ft_split_manifest"), "hyenadna", ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"])
            gS = lambda v: (f"{v['g'].loc['S_shortcut_worth'].estimate:.3f}{ci(v['g'].loc['S_shortcut_worth'].ci_lo, v['g'].loc['S_shortcut_worth'].ci_hi)}"
                            if v["ok"]("S_shortcut_worth") else "n/e")
            add("   End-to-end fine-tuning on id80 HC splits (HyenaDNA-tiny, 3 seeds; ft_contrasts.csv, eval_ft_v2.py rules): "
                + "; ".join(f"{ds} S(shortcut worth, AUROC)={gS(v)} rule1 {v['rule1']} rule2 {v['rule2']} rule3 {v['rule3']}"
                            + (f" rule5 {v.get('rule5')}" if ds == "drosophila_enhancers_stark" else "") for ds, v in rh.items())
                + (" [provided-split reference: fm_finetune_v2_contrasts.csv]" if os.path.exists(ppath) else "")
                + "; DNABERT-2 LoRA on HC splits DEFERRED (deferred/README.md)")
        add("")
    add("METHOD. MMseqs2 18-8cc5c all-vs-all nucleotide search, BOTH strands (--strand 2), pooled train+test per task;")
    add("edge = fident >= T over >= 80% of both sequences; clusters = connected components (single linkage).")
    add("T = 0.8 (primary, conventional nucleotide near-duplicate level) and T = 0.5 (strict: all MMseqs2-DETECTABLE homology,")
    add("which in practice means >= ~70% identity because 11-mer seeding limits recall below that; see calibration).")
    if cal is not None:
        c = cal.pivot_table(index="nominal_identity", columns="threshold", values="recall", aggfunc="mean")
        add("Synthetic calibration (recall of planted pairs, mean over strands; 0/400 random decoys merged): " +
            "; ".join(f"{i}%: id80 {c.loc[i,'id80']:.2f} id50 {c.loc[i,'id50']:.2f}" for i in (90, 85, 80, 75, 70, 65)))
        if lin is not None:
            lc = lin[lin.threshold == "id80"].groupby(["method", "nominal_identity"]).recall.mean()
            for m in lc.index.get_level_values(0).unique():
                add(f"linclust calibration (calibration_linclust.csv, {m}; id80, mean over strands): " + "; ".join(
                    f"{i}%: {lc.loc[(m, i)]:.2f}" for i in (95, 90, 85, 80) if (m, i) in lc.index)
                    + f"; decoys merged {int(lin[lin.method == m].decoys_falsely_merged.max())}/{int(lin.n_decoys.max())}")
            add("-> search + connected components recovers >=85%-identity pairs that linclust misses; it was affordable for every task.")
    add("")
    if red is not None:
        add("1. REDUNDANCY (fraction of TEST sequences sharing a component with >=1 TRAIN sequence; exact = identical")
        add("   to a train sequence or its reverse complement; within-train = 1 - train components/train sequences)")
        add(f"   {'task':<34}{'test id80':>10}{'test id50':>10}{'exact':>8}{'withinTr80':>11}{'mixedLab80':>11}{'largest80':>10}")
        for t in _hc.TASKS:
            g = red[red.task == t].set_index("threshold")
            if "id80" not in g.index:
                continue
            a, b = g.loc["id80"], g.loc["id50"] if "id50" in g.index else g.loc["id80"]
            add(f"   {t:<34}{a.frac_test_in_train_cluster:>10.3f}{b.frac_test_in_train_cluster:>10.3f}"
                f"{a.frac_test_exact_in_train:>8.3f}{1-a.train_clusters_per_seq:>11.3f}{a.mixed_label_seq_frac:>11.3f}"
                f"{int(a.largest_component):>10d}")
        if fx_ed is not None:
            add("   KIND OF HOMOLOGY (fix_edges.csv, id80 train-test edges): median identity, share >= 0.99; per test sequence with a")
            add("   direct train edge, share whose best train identity >= 0.99; test sequences < 50 bp")
            for _, r in fx_ed[fx_ed.threshold == "id80"].iterrows():
                if r.n_train_test_edges > 0:
                    add(f"   {r.task:<34} edges {int(r.n_train_test_edges):>8d}  median id {r.tt_edge_fid_median:.3f}  >=0.99 {r.tt_edge_frac_ge099:.2f}"
                        f"  best>=0.99 {r.direct_best_frac_ge099:.2f}  len {int(r.len_min)}-{int(r.len_max)} bp, <50bp test {int(r.n_test_lt50bp)}")
        if fx_ex is not None:
            g = fx_ex[fx_ex.threshold == "id80"]
            add("   EXACT DUPLICATES (fix_exactdup.csv): same-label exact-duplicate share of test, by class; exact duplicates the id80 components MISSED")
            for _, r in g.iterrows():
                if r.exact_dup_test > 0:
                    add(f"   {r.task:<34} exact {r.exact_dup_test:.3f} (same label {r.exact_dup_same_label:.3f}; by class {r.exact_dup_same_label_by_class}); "
                        f"missed {int(r.n_exact_missed_by_components)}"
                        + (f" (median {r.missed_len_median:.0f} bp, max {int(r.missed_len_max)} bp); purge MCC components {r.purge_comp_mcc:.3f} -> "
                           f"components+exact {r.purge_union_mcc:.3f} (effect {r.purge_comp_homology_effect:+.3f} -> {r.purge_union_homology_effect:+.3f})"
                           if r.n_exact_missed_by_components > 0 else ""))
        if fx_cov is not None:
            add("   CONTAINMENT-COVERAGE SENSITIVITY (POST HOC; covmode.csv, run_covmode_sensitivity.py; --cov-mode 1 = an edge needs only the")
            add("   SHORTER sequence >= 80% covered; same search otherwise; purge re-scored with the same cached predictions). Rule fixed")
            add("   before running: ROBUST iff |homology_effect(cov1) - homology_effect(cov0)| <= 0.02.")
            add(f"   {'task':<34}{'thr':<6}{'test w/ homolog cov0 -> cov1':<30}{'purge MCC cov0 -> cov1':<25}{'effect cov0 -> cov1':<24}robust")
            for _, r in fx_cov.iterrows():
                add(f"   {r.task:<34}{r.threshold:<6}{r.frac_test_in_train_cluster_covmode0:>10.3f} -> {r.frac_test_in_train_cluster:<15.3f}"
                    f"{r.purge_mcc_covmode0:>8.3f} -> {r.purge_mcc:<13.3f}{r.homology_effect_covmode0:>+8.3f} -> {r.homology_effect:<+12.3f}{r.robust_to_cov_rule}")
            miss = [t for t in ["dummy_mouse_enhancers_ensembl", "drosophila_enhancers_stark", "human_enhancers_ensembl",
                                "human_ocr_ensembl", "human_ensembl_regulatory"] if t not in set(fx_cov.task)]
            if miss:
                add("   NOT RUN (pending): " + ", ".join(miss))
        add("")
    if ev is not None:
        add("2. BASELINE RE-EVALUATION (best-k LightGBM, paper config; 95% bootstrap CIs, 1000 resamples, seed 42)")
        add("   purge = same trained model scored on test minus sequences whose id80 component holds a train sequence;")
        add("   null = MCC of random removals with identical per-class counts; cluster/random = retrained re-splits (split seed 42).")
        add("   clu-rand CI = independent bootstraps of the two test sets, CONDITIONAL on one split (split variance: section 2b).")
        add(f"   {'task':<34}{'orig':>7}{'purge80':>9} {'d_purge [CI]':<23}{'null':>7}{'clu80':>8}{'rand':>7} "
            f"{'clu-rand [CI]':<22}{'clu50':>7}")
        for t in _hc.TASKS:
            e = ev[ev.task == t]
            if not len(e):
                continue
            o = e[e.protocol == "original"].iloc[0]
            pu = e[(e.protocol == "test_purge") & (e.threshold == "id80")]
            cl = e[(e.protocol == "resplit_cluster") & (e.threshold == "id80")]
            c5 = e[(e.protocol == "resplit_cluster") & (e.threshold == "id50")]
            rn = e[e.protocol == "resplit_random"]
            pu = pu.iloc[0] if len(pu) else None; cl = cl.iloc[0] if len(cl) else None
            s_pu = (f"{pu.mcc:>9.3f} {pu.delta_mcc:+.3f} {ci(pu.delta_lo, pu.delta_hi):<16}{pu.null_mean:>7.3f}"
                    if pu is not None and pu.status == "ok" else f"{'n/e':>9} {('(' + str(pu.status) + ')') if pu is not None else '':<23}{'':>7}")
            s_cl = (f"{cl.mcc:>8.3f}{rn.iloc[0].mcc:>7.3f} {cl.delta_vs_random:+.3f} {ci(cl.delta_vs_random_lo, cl.delta_vs_random_hi):<15}"
                    if cl is not None and cl.status == "ok" else f"{'infeas':>8}{rn.iloc[0].mcc:>7.3f} {'':<22}")
            s5 = f"{c5.iloc[0].mcc:>7.3f}" if len(c5) and c5.iloc[0].status == "ok" else f"{'infeas':>7}"
            add(f"   {t:<34}{o.mcc:>7.3f}{s_pu}{s_cl}{s5}")
        repro = ev[ev.protocol == "original"]
        add(f"   Table-I reproduction: {int(repro.reproduces_table1.sum())}/{len(repro)} tasks within 0.005 "
            f"(max |d| = {np.abs(repro.mcc - repro.published_mcc).max():.4f}).")
        pu_all = ev[(ev.protocol == "test_purge") & (ev.threshold == "id80") & (ev.status == "ok")]
        shift = pu_all[pu_all.get("min_class_kept_frac", pd.Series(1.0, index=pu_all.index)) < 0.25] if "min_class_kept_frac" in pu_all else pu_all.iloc[0:0]
        for _, r in shift.iterrows():
            add(f"   CLASS-MIX CAVEAT: {r.task} purge keeps {r.kept_per_class} (per class kept/full); raw purged MCC {r.mcc:.3f} "
                f"{ci(r.mcc_lo, r.mcc_hi)} vs prevalence-standardised {r.std_mcc:.3f}; the purge-based label uses the standardised value "
                "(run_routes.py addendum, chosen after the first eval results).")
        add("")
    if sd is not None:
        add("2b. SPLIT-SEED ROBUSTNESS of cluster-aware vs random re-split (split seeds 42,43,44; model seed 42)")
        for _, r in sd.iterrows():
            add(f"   {r.task:<34}{r.threshold:<5} cluster {r.mcc_cluster_mean:.3f} [{r.mcc_cluster_min:.3f},{r.mcc_cluster_max:.3f}]  "
                f"random {r.mcc_random_mean:.3f}  delta mean {r.delta_mean:+.3f} range [{r.delta_min:+.3f},{r.delta_max:+.3f}]"
                f"  same sign all seeds: {r.same_sign_all_seeds}")
        if ev is not None:
            fo = ev[(ev.protocol == "resplit_cluster") & (ev.status == "ok") & (ev.threshold == "id80")]
            low = fo[fo.frac_test_from_orig_test < 0.2]
            add("   re-split test sets drawn mostly from the provided TRAIN split (frac_test_from_orig_test < 0.2): "
                + ", ".join(f"{r.task} {r.frac_test_from_orig_test:.3f}" for _, r in low.iterrows())
                + " -- for these the re-split measures a different (in-distribution) task; test-purge answers the homology question.")
        add("")
    if iv is not None:
        add("3. INTERVENTIONS ON HOMOLOGY-CLEAN DATA (HC = cluster-aware re-split + one representative per component x label)")
        add("   Readout for the equalized arm = TATA / GC / k-mer AUROC; its MCC is degenerate on a ~96%-positive test set.")
        for t in iv.task.unique():
            for dv in ["original", "id80", "id50"]:
                g = iv[(iv.task == t) & (iv.data == dv)]
                if not len(g):
                    continue
                e1 = g[g.experiment == "E1_composition"].set_index("arm")
                e2 = g[g.experiment == "E2_shuffled_neg"].set_index("arm")
                if "original" in e1.index:
                    o = e1.loc["original"]
                    add(f"   {t:<28}{dv:<9} TATA {o.tata_auroc:.3f}{ci(o.tata_lo, o.tata_hi)} GC {o.gc_auroc:.3f} "
                        f"kmerMCC {o.kmer_mcc:.3f}{ci(o.kmer_mcc_lo, o.kmer_mcc_hi)} kmerAUROC {o.kmer_auroc:.3f} compOnly {o.comp_only_mcc:.3f}"
                        f"{ci(o.comp_only_mcc_lo, o.comp_only_mcc_hi)}")
                if "comp_equalized" in e1.index:
                    q = e1.loc["comp_equalized"]
                    if q.get("kmer_mcc") != q.get("kmer_mcc"):
                        add(f"   {'':<28}{'':<9} equalized [POST HOC, {q.status}]: kept {q.retained_neg_frac:.1%} negs "
                            f"({int(q.n_test_neg_kept)} in test), maxSMD {q.max_abs_smd:.3f}, TATA {q.tata_auroc:.3f}"
                            f"{ci(q.tata_lo, q.tata_hi)} GC {q.gc_auroc:.3f}{ci(q.gc_lo, q.gc_hi)}")
                    else:
                        add(f"   {'':<28}{'':<9} equalized: kept {q.retained_neg_frac:.1%} negs, maxSMD {q.max_abs_smd:.3f}, TATA "
                            f"{q.tata_auroc:.3f}{ci(q.tata_lo, q.tata_hi)} GC {q.gc_auroc:.3f} kmerAUROC {q.kmer_auroc:.3f}"
                            f"{ci(q.kmer_auroc_lo, q.kmer_auroc_hi)} (kmerMCC {q.kmer_mcc:.3f}, d {q.d_kmer_mcc_vs_original:+.3f}"
                            f"{ci(q.d_kmer_mcc_lo, q.d_kmer_mcc_hi)}) compOnly {q.comp_only_mcc:.3f}")
                if "shuffled_neg" in e2.index:
                    q = e2.loc["shuffled_neg"]
                    add(f"   {'':<28}{'':<9} shuffled negs: TATA {q.tata_auroc:.3f}{ci(q.tata_lo, q.tata_hi)} GC {q.gc_auroc:.3f} "
                        f"kmerMCC {q.kmer_mcc:.3f}")
        if e1rec is not None:
            add("   E1 RECONCILIATION (e1_reconcile.csv, run_e1_reconcile.py): replaying composition_clean.py's RandomState path on cohn")
            for _, r in e1rec.iterrows():
                add(f"   {r.task:<28}{r.arm:<15} k-mer MCC replay {r.replay_kmer_mcc:.4f} / published {r.published_kmer_mcc:.4f} / this module "
                    f"{r.module_kmer_mcc:.4f}; k-mer AUROC {r.replay_kmer_auroc:.3f}/{r.published_kmer_auroc:.3f}/{r.module_kmer_auroc:.3f}; "
                    f"replay == published: {r.replay_equals_published}")
        if fx_e1 is not None:
            add("   Equally valid matching draws (e1_draws_summary.csv) -- the equalized-arm MCC is degenerate (test ~96% positive):")
            for _, r in fx_e1.iterrows():
                add(f"   {r.task:<28}{r.data:<9} {int(r.n_draws)} draws: kmerMCC median {r.kmer_mcc_median:.3f} [{r.kmer_mcc_min:.3f},{r.kmer_mcc_max:.3f}]; "
                    f"kmerAUROC median {r.kmer_auroc_median:.3f} [{r.kmer_auroc_min:.3f},{r.kmer_auroc_max:.3f}]; TATA AUROC median "
                    f"{r.tata_auroc_median:.3f}; GC AUROC median {r.gc_auroc_median:.3f}; predicted positive {r.frac_pred_pos_median:.1%}")
        if xc is not None:
            add("   Route-3 cleaning arms with placebo controls: see homology_recluster_extra_cleaning.csv (run_extra.py X1).")
        add("   PRE-REGISTERED Route-3 verdicts (run_interventions.py docstring; per contaminated task and data):")
        add("   (i) TATA AUROC < 0.5 with CI upper < 0.5; (ii) comp-only MCC CI lower > 0; (iii) dMCC_eq CI upper < 0 [degenerate");
        add("   readout: the equalized test set is ~96% positive]; (iv) equalized AND shuffled TATA AUROC closer to 0.5 than the original arm.")
        for t in ["human_enhancers_cohn", "nt_enhancers"]:
            for dv in ["original", "id80", "id50"]:
                g = iv[(iv.task == t) & (iv.data == dv)]
                if not len(g):
                    continue
                e1 = g[g.experiment == "E1_composition"].set_index("arm"); e2 = g[g.experiment == "E2_shuffled_neg"].set_index("arm")
                o, q, sh = e1.loc["original"], e1.loc["comp_equalized"], e2.loc["shuffled_neg"]
                c1 = bool(o.tata_auroc < 0.5 and o.tata_hi < 0.5)
                c2 = bool(o.comp_only_mcc_lo > 0)
                c3 = (bool(q.d_kmer_mcc_hi < 0) if q.d_kmer_mcc_hi == q.d_kmer_mcc_hi else None)
                c4 = bool(abs(q.tata_auroc - .5) < abs(o.tata_auroc - .5) and abs(sh.tata_auroc - .5) < abs(o.tata_auroc - .5))
                verdict = ("PRESERVED" if (c1 and c2 and c3 and c4) else
                           "NOT EVALUABLE as pre-registered (equalized MCC infeasible); (i),(ii),(iv) hold" if (c3 is None and c1 and c2 and c4)
                           else "NOT preserved")
                add(f"   {t:<28}{dv:<9} (i) {c1} (ii) {c2} (iii) {'n/e' if c3 is None else c3} (iv) {c4} => Route-3 magnitude {verdict}")
        gd = iv[(iv.task == "drosophila_enhancers_stark")]
        for dv in ["original", "id80", "id50"]:
            g = gd[gd.data == dv]
            if len(g):
                o = g[(g.experiment == "E1_composition") & (g.arm == "original")].iloc[0]
                on = g[(g.experiment == "E2_shuffled_neg") & (g.arm == "original_neg")].iloc[0]
                sh = g[(g.experiment == "E2_shuffled_neg") & (g.arm == "shuffled_neg")].iloc[0]
                ok = bool(o.tata_auroc >= 0.5 and abs(sh.tata_auroc - .5) > abs(on.tata_auroc - .5))
                add(f"   drosophila control {dv:<9} TATA {o.tata_auroc:.3f} (>= 0.5), shuffled-arm TATA {on.tata_auroc:.3f} -> {sh.tata_auroc:.3f} "
                    f"(away from 0.5) => control behaves: {ok}")
        add("")
    if pos is not None:
        add("4. POSITIONAL RESCUE dMCC(B8-B1), paired bootstrap (<=20k/split cap as positional_sweep.py)")
        for t in _hc.TASKS:
            g = pos[pos.task == t].set_index("data")
            if not len(g):
                continue
            add(f"   {t:<34}" + "  ".join(f"{dv}: {g.loc[dv,'delta_mcc']:+.3f}{ci(g.loc[dv,'delta_lo'], g.loc[dv,'delta_hi'])}"
                                         for dv in ["original", "id80", "id50"] if dv in g.index))
        sp = pos[pos.task == "nt_splice_sites_all"].set_index("data")
        add("   Pre-registered Route-2: splice dMCC(B8-B1) > 0 with CI lower > 0: " + "; ".join(
            f"{dv} {sp.loc[dv, 'delta_mcc']:+.3f} [{sp.loc[dv, 'delta_lo']:+.3f},{sp.loc[dv, 'delta_hi']:+.3f}] => "
            f"{'PRESERVED' if sp.loc[dv, 'delta_lo'] > 0 else 'NOT preserved'}" for dv in ["original", "id80", "id50"] if dv in sp.index))
        for dv in ["id80", "id50"]:
            ctrl = pos[(pos.data == dv) & (~pos.task.isin(["nt_splice_sites_all", "nt_promoter_tata"]))]
            if not len(ctrl):
                continue
            viol = ctrl[(ctrl.delta_excludes_0 == True) & (ctrl.delta_mcc.abs() > 0.05)]  # noqa: E712
            add(f"   Pre-registered SELECTIVITY (non-positional tasks: CI includes 0 or |dMCC| <= 0.05), {dv}: "
                f"{len(ctrl) - len(viol)}/{len(ctrl)} pass; violations: "
                + (", ".join(f"{r.task} {r.delta_mcc:+.3f} [{r.delta_lo:+.3f},{r.delta_hi:+.3f}]" for _, r in viol.iterrows()) or "none")
                + "; small CI-excluding-0 effects: " + (", ".join(
                    f"{r.task} {r.delta_mcc:+.3f}" for _, r in ctrl[(ctrl.delta_excludes_0 == True) & (ctrl.delta_mcc.abs() <= 0.05)].iterrows()) or "none"))  # noqa: E712
        add("")
    if r1 is not None:
        add("5. ROUTE 1: Spearman(discriminability, MCC) over the 11 TF-motif tasks (PRIMARY = published x vs cluster re-split id80,")
        add("   all TF tasks; everything else is a control or sensitivity -- 3 x-variants x 11 y-variants x 4 subsets, not corrected)")
        for _, r in r1.iterrows():
            add(f"   x={r.x:<24} y={r.y:<28} {str(r.get('subset', '')):<24} n={int(r.n):>2} rho={r.rho:+.3f} p={r.p:.4f} "
                f"jackknife [{r.jack_rho_lo:+.2f},{r.jack_rho_hi:+.2f}] {r.get('role', '')}")
        add("")
    if rl is not None:
        add("6. ROUTE LABELS (pre-registered rule: preserved iff the assigning diagnostic keeps its sign and its CI excludes 0;")
        add("   strict = additionally the full label rule gives the same label). Labels come from a rule RECONSTRUCTED from Table I")
        add("   (it reproduces the published v2 labels on the provided data); diagnose.py verdicts are a second, independent logic.")
        add(f"   label rule reproduces the published label on the provided data: "
            f"{int(rl.drop_duplicates('task').rule_reproduces_published.sum())}/{rl.task.nunique()} tasks; diagnose.py verdict "
            f"disagrees with the published label before cleaning for: " + ", ".join(
                f"{r.task} ({r.diagnose_verdict_before})" for _, r in rl.drop_duplicates("task").iterrows()
                if not _dg_agrees(r.published_label, r.diagnose_verdict_before)))
        for _, r in rl.iterrows():
            add(f"   {r.task:<34}{r.threshold:<5} published={r.published_label:<14} before={r.label_before:<14} "
                f"after(re-split)={r.label_after:<14} [{r.preservation} | strict: {r.preservation_strict}]  "
                f"after(purge)={(r.label_after_purgeMCC if isinstance(r.label_after_purgeMCC, str) else 'n/a'):<14} [{r.preservation_purgeMCC}]"
                f"  seed-mean re-split: {r.label_after_seedmean if isinstance(r.label_after_seedmean, str) else 'n/a'}")
            add(f"   {'':<39}diag before: {r.diagnostic_before}")
            add(f"   {'':<39}diag after : {r.diagnostic_after}" + (f" | purge: {r.diagnostic_after_purgeMCC}"
                                                                   if isinstance(r.diagnostic_after_purgeMCC, str) and r.diagnostic_after_purgeMCC != r.diagnostic_after else ""))
        add("")
    add("7. CAVEATS")
    add("   * Bootstrap CIs resample test sequences, not homology components; within-test families make them somewhat narrow.")
    ab = rd("audit_clusterboot")
    if ab is not None and len(ab):
        ab = ab.assign(r=(ab.comp_hi - ab.comp_lo) / (ab.seq_hi - ab.seq_lo))
        d = ab[ab.quantity == "cluster_minus_random"]
        fl = d[((d.seq_hi < 0) | (d.seq_lo > 0)) != ((d.comp_hi < 0) | (d.comp_lo > 0))]
        add(f"     Component-level bootstrap (audit_clusterboot.csv): CI width ratio component/sequence median "
            + ", ".join(f"{q} {g.r.median():.2f} (max {g.r.max():.1f})" for q, g in ab.groupby("quantity"))
            + "; cluster-minus-random calls that change: " + (", ".join(
                f"{r.task} {r.threshold} {r.point:+.3f} seq [{r.seq_lo:+.3f},{r.seq_hi:+.3f}] comp [{r.comp_lo:+.3f},{r.comp_hi:+.3f}]"
                for _, r in fl.iterrows()) or "none"))
    at = rd("audit_task")
    if at is not None and len(at):
        m = at[(at.threshold == "id80") & (at.n_exact_missed_by_clusters > 0)]
        add("   * Exact twins MMseqs2 cannot seed (mostly-N or < 11 bp; audit_task.csv, id80): " + "; ".join(
            f"{r.task} {int(r.n_exact_missed_by_clusters)} test seqs missed (median {r.missed_len_median:.1f} bp), purge MCC "
            f"{r.purge_mcc:.3f} -> {r.purge_plus_exact_mcc:.3f} when also removed; {int(r.n_resplit_test_with_exact_twin_in_train)} "
            f"re-split test seqs keep an exact train twin" for _, r in m.iterrows()))
    add("   * Multiplicity: 15 tasks x 2 thresholds x 2 protocols of inflation calls, uncorrected; the pre-registered rule is the call.")
    add("   * The 0.05 materiality cut and the prevalence-standardised purge MCC for labels were chosen after results were seen.")
    add("   * Coverage >= 80% of BOTH sequences cannot link contained loci of very different length (see containment sensitivity).")
    add("   * The remaining paper experiments (Route-3 cleaning arms, specificity panel, composition-only baseline on all tasks,")
    add("     frozen-FM probe, graded positional rescue, LR floor, model seeds, motif presence, coordinates) are re-run on")
    add("     homology-clean data by run_extra.py, and end-to-end HyenaDNA fine-tuning by run_stage_ft_hc.sh (DNABERT-2 LoRA deferred);")
    add("     section 8 summarises them, homology_recluster_extra_interpretation.txt has details.")
    L += extra_summary()
    return L


def extra_summary():
    """Section 8/9: one line per run_extra.py experiment (verdicts computed by make_report_extra.py's rules) + audit pointer."""
    import make_report_extra as MX
    L = [""]; add = L.append
    add("8. OTHER PAPER EXPERIMENTS RE-RUN ON HOMOLOGY-CLEAN DATA (run_extra.py; full numbers: homology_recluster_extra_interpretation.txt)")
    txt = MX.build()
    for ln in txt:
        if ln.lstrip().startswith("=>") or ln.startswith("X") or "=> " in ln:
            add("   " + ln.strip())
    pend = [n for n in ("cleaning", "specificity", "composition", "fm", "graded", "lr", "modelseeds", "presence", "coords")
            if rd(f"extra_{n}") is None]
    if pend:
        add("   NOT YET AVAILABLE (still running / pending): " + ", ".join(f"extra_{n}" for n in pend))
    add("")
    add("9. VERIFICATION AUDIT (fix agent; homology_recluster_audit_interpretation.txt + audit_{class,task,clusterboot}.csv,")
    add("   e1_reconcile.csv, e1_draws{,_summary}.csv, calibration_linclust.csv, covmode.csv): per-class purge content, class")
    add("   elimination, exact duplicates missed by MMseqs2, identity of train-test homologs, component-level bootstrap CIs,")
    add("   linclust calibration, E1 reconciliation, containment-coverage sensitivity.")
    return L


def _dg_agrees(pub, verdict):
    v = str(verdict).upper()
    m = {"solved": "SOLVED", "positional": "ROUTE 2", "contamination": "ROUTE 3", "shared-motif": "ROUTE 1"}
    return m.get(pub, "?") in v


def figure(tab, ev, rl, sd, add_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    tasks = [t for t in _hc.TASKS if t in set(tab.task)]
    tabi = tab.set_index("task")
    s8 = sd[sd.threshold == "id80"].set_index("task") if sd is not None else None
    homol = set()
    if add_ is not None:
        a8 = add_[add_.threshold == "id80"]
        homol = set(a8[a8.relabel_homology_attributable_seedmean == True].task)  # noqa: E712
    order = sorted(tasks, key=lambda t: tabi.loc[t, "original_mcc"])
    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    col = {"orig": "#000000", "purge": "#E69F00", "clu": "#0072B2", "rnd": "#999999"}   # Okabe-Ito
    for i, t in enumerate(order):
        r = tabi.loc[t]
        pstd = r.get("purge_id80_std_mcc", np.nan) if r.get("purge_id80_status", "ok") == "ok" else np.nan
        cm = s8.loc[t, "mcc_cluster_mean"] if s8 is not None and t in s8.index else r.get("resplit_id80_mcc", np.nan)
        cmin = s8.loc[t, "mcc_cluster_min"] if s8 is not None and t in s8.index else cm
        cmax = s8.loc[t, "mcc_cluster_max"] if s8 is not None and t in s8.index else cm
        rm = s8.loc[t, "mcc_random_mean"] if s8 is not None and t in s8.index else r.get("resplit_random_mcc", np.nan)
        ax.plot([cmin, cmax], [i, i], color=col["clu"], lw=1.6, alpha=0.45, zorder=1, solid_capstyle="butt")
        ax.scatter(rm, i, marker="|", s=45, color=col["rnd"], zorder=2, lw=1.3)
        ax.scatter(r["original_mcc"], i, s=18, facecolor="white", edgecolor=col["orig"], lw=0.9, zorder=3)
        if pstd == pstd:
            ax.scatter(pstd, i, marker="v", s=18, color=col["purge"], zorder=4)
        else:
            ax.text(0.12, i, "purge n/e", va="center", ha="left", fontsize=7, color=col["purge"])
        ax.scatter(cm, i, marker="o", s=13, color=col["clu"], zorder=5)
    labels = []
    for t in order:
        r = tabi.loc[t]
        labels.append(f"{SHORT.get(t, t)}{'$^\\dagger$' if t in homol else ''} ({100 * r.get('red_test_id80', np.nan):.0f}%)")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.axvline(0.7, color="#555555", ls="--", lw=0.6)
    ax.set_xlabel("MCC (best-k LightGBM)")
    ax.set_xlim(0.1, 1.0)
    h = [Line2D([], [], marker="o", ls="", mfc="white", mec=col["orig"], ms=4, label="provided split"),
         Line2D([], [], marker="v", ls="", color=col["purge"], ms=4, label="homologs purged (prev.-adj.)"),
         Line2D([], [], marker="o", ls="-", color=col["clu"], ms=3.5, lw=1.4, label="cluster re-split (3 seeds)"),
         Line2D([], [], marker="|", ls="", color=col["rnd"], ms=6, mew=1.3, label="random re-split")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.915))   # keep the full 3.5-in width; legend sits above the axes
    os.makedirs(_hc.FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_hc.FIG, f"{P}_mcc.{ext}"), dpi=300)
    plt.close(fig)


def main():
    L = build_text()
    with open(os.path.join(R, f"{P}_interpretation.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    tab, ev, rl, sd, add_ = rd("table"), rd("eval"), rd("route_labels"), rd("seeds_summary"), rd("labels_addendum")
    if tab is not None:
        figure(tab, ev, rl, sd, add_)
        print("figure written")


if __name__ == "__main__":
    main()
