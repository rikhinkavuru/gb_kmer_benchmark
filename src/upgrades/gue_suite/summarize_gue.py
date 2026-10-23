#!/usr/bin/env python3
"""Assemble the GUE result tables from the per-task cards (run_gue.py), the post-hoc files
(run_gue_posthoc.py), the paper-15 reference (paper15_reference.py), the overlap audit
(gue_overlap.py) and the parsed published tables (verify_gue.py). No models are fit here.

Writes (results/upgrades/):
  gue_results.csv               one row per GUE dataset: every pre-registered readout + route label
  gue_published_comparison.csv  our k-mer LightGBM (and composition-only) vs the published GUE numbers
                                (DNABERT-2 Tables 6 and 9; MCC x100, covid macro-F1 x100)
  gue_family_summary.csv        one row per GUE task family
  gue_equalization_decomposition.csv   post hoc PH1 (size-matched random control)
  gue_twins.csv                 post hoc PH2 (dinucleotide-twin audit)
  gue_predictions.csv           the four pre-stated predictions (P1-P4) scored against the results
  gue_suite_comparison.csv      route-label / flag counts: GB and NT (paper Table 1 + gue_paper15_reference)
                                vs GUE
  gue_paper_table.csv           compact per-family table for the paper
  [fix round 1, post hoc]
  gue_results.csv gains DERIVED post-hoc columns (no refits; listed as post hoc in the README):
      independence_class, frac_in_paper_tasks, frac_in_any_gb_nt (gue_overlap_summary.csv),
      route3_signature, route1_signature, hard_reasons, mcc_ci_contains_0.7 (gue_rule_sensitivity.csv)
  gue_independence_routes.csv   route labels / signatures per independence class
  gue_positional_localisation.csv   PH3 per task x bin (run_gue_posthoc_r2.py)
  [fix round 2, post hoc]
  gue_results.csv gains the PH5 columns for the ten EMP rows (emp_provenance.py; no fits):
      emp_intergenic_frac_label1/label0, emp_tata_excess_std[_lo/_hi], emp_location_verdict[_4strata]
  gue_positional_localisation.csv gains full_delta_lo/hi and full_delta_significant; 'carrier' and
      'necessary' are left empty (NA) where the full B=8 delta is not significant (nt_promoter_tata)
  gue_failure_route_audit.csv   the 12 GUE failure-route labels against the post-hoc checks PH4/PH5
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gue_data as G  # noqa: E402

RES = os.path.join(G.ROOT, "results", "upgrades")
CARDS = os.path.join(RES, "gue_cards")
PH = os.path.join(RES, "gue_posthoc")

SHORT = {"SOLVED (class-specific motif discriminability)": "solved (motif-discriminable)",
         "SOLVED": "solved"}


def short_route(v):
    if v in SHORT:
        return SHORT[v]
    if v.startswith("ROUTE 2"):
        return "positional"
    if v.startswith("ROUTE 3"):
        return "contamination"
    if v.startswith("ROUTE 1"):
        return "shared-motif"
    return "hard (no single route)"


FAMILY_ORDER = ["prom_core", "prom_300", "human_tf", "splice", "mouse_tf", "emp", "covid"]
FAMILY_NAME = {"prom_core": "Core promoter (70 bp)", "prom_300": "Promoter (300 bp)",
               "human_tf": "TF binding, human", "splice": "Splice site (3-class)",
               "mouse_tf": "TF binding, mouse", "emp": "Yeast histone marks", "covid": "Covid variant (9-class)"}


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def load_cards():
    rows = []
    for t in G.ORDER:
        p = os.path.join(CARDS, f"{t}.json")
        if not os.path.exists(p):
            continue
        c = json.load(open(p))
        L, CO, CF, P2 = c["learnability"], c["composition_only"], c["composition_fraction"], c["route2_positional"]
        r1 = c.get("route1_motif", {}); r3 = c.get("route3_negatives", {}); sh = c.get("shuffled_negative_control", {})
        ed = c.get("equalized_delta", {}); rd = c["redundancy"]
        row = dict(task=t, family=G.neg_family(t), species=c["species"], n_classes=c["n_classes"],
                   n_train=c["n_train"], n_dev=c["n_dev"], n_test=c["n_test"], pos_frac_test=c["pos_frac_test"], k=c["k"],
                   mcc=r(L["mcc"]), mcc_lo=r(L["mcc_ci"][0]), mcc_hi=r(L["mcc_ci"][1]),
                   auroc=r(L["auroc"]), auroc_lo=r(L["auroc_ci"][0]), auroc_hi=r(L["auroc_ci"][1]),
                   f1_macro=r(L["f1_macro"]), f1_lo=r(L["f1_macro_ci"][0]), f1_hi=r(L["f1_macro_ci"][1]),
                   comp_mcc=r(CO["mcc"]), comp_mcc_lo=r(CO["mcc_ci"][0]), comp_mcc_hi=r(CO["mcc_ci"][1]),
                   comp_auroc=r(CO["auroc"]), comp_f1_macro=r(CO["f1_macro"]),
                   comp_fraction=CF["value"], comp_fraction_lo=CF["ci"][0], comp_fraction_hi=CF["ci"][1],
                   comp_fraction_def=("binary (diagnose.py)" if c["binary"] else "macro-OvR EXTENSION"),
                   pos_spectrum_mcc=P2["spectrum_mcc"], pos_binned_mcc=P2["positional_mcc"],
                   pos_delta=P2["delta_mcc"], pos_delta_lo=P2["delta_ci"][0], pos_delta_hi=P2["delta_ci"][1],
                   pos_significant=P2["significant"], pos_capped=P2["capped"],
                   motif_max_auroc=r1.get("max_pos_motif_auroc"), motif_best_tf=r1.get("best_TF"),
                   motif_richness=r1.get("motif_richness"), motif_panel=r1.get("panel"),
                   tata_pos_rate=r3.get("tata_pos_rate"), tata_neg_rate=r3.get("tata_neg_rate"),
                   tata_neg_excess=r3.get("tata_neg_excess"), tata_auroc=r3.get("tata_motif_auroc"),
                   tata_cleaned_auroc=r3.get("cleaned_tata_auroc"), gc_auroc=r3.get("gc_auroc"),
                   shuffled_tata_auroc=sh.get("tata_auroc_vs_shuffled"), shuffled_gc_auroc=sh.get("gc_auroc_vs_shuffled"),
                   eq_feasible=ed.get("feasible"), eq_retained_neg_frac=ed.get("retained_neg_frac"),
                   eq_mcc_after=ed.get("mcc_after"), eq_auroc_after=ed.get("auroc_after"),
                   eq_comp_auroc_after=ed.get("comp_only_auroc_after"), eq_pos_frac_test_after=ed.get("pos_frac_test_after"),
                   eq_smd_before=ed.get("max_abs_smd_before"), eq_smd_after=ed.get("max_abs_smd_after"),
                   redund_ge07=rd["frac_ge_0_7"], redund_verbatim=rd["frac_verbatim"],
                   tier=c["tier"], verdict=c["verdict"], route=short_route(c["verdict"]),
                   verdict_vertebrate_panel=c.get("verdict_vertebrate_panel"),
                   composition_dominated=c["composition_dominated"], seconds=c["total_s"])
        if "route1_motif_vertebrate_panel" in c:
            row["motif_max_auroc_vertebrate_panel"] = c["route1_motif_vertebrate_panel"]["max_pos_motif_auroc"]
            row["motif_richness_vertebrate_panel"] = c["route1_motif_vertebrate_panel"]["motif_richness"]
        rows.append(row)
    return pd.DataFrame(rows)


def add_posthoc_columns(df):
    """[fix round 1, post hoc] derived columns from the uniform overlap audit and the rule-sensitivity
    table (no refits). Missing inputs leave NaN, never 0."""
    ovp = os.path.join(RES, "gue_overlap_summary.csv")
    if os.path.exists(ovp):
        ov = pd.read_csv(ovp)[["task", "independence_class", "frac_in_paper_tasks", "frac_in_paper_tasks_label1",
                               "frac_in_nt_suite", "frac_in_any_gb_nt"]]
        df = df.merge(ov, on="task", how="left")
    rsp = os.path.join(RES, "gue_rule_sensitivity.csv")
    if os.path.exists(rsp):
        rs = pd.read_csv(rsp)
        rs = rs[rs.suite == "GUE"][["task", "route3_signature", "route1_signature", "route2_signature",
                                     "hard_reasons", "hard_only_because_richness", "mcc_ci_contains_0.7"]]
        df = df.merge(rs, on="task", how="left")
        assert (df.route2_signature == ((df.pos_significant) & (df.pos_delta > 0.2))).all()
    epp = os.path.join(RES, "gue_emp_provenance.csv")
    if os.path.exists(epp):
        ep = pd.read_csv(epp)[["task", "intergenic_frac_label1", "intergenic_frac_label0", "tata_excess_std",
                               "tata_excess_std_lo", "tata_excess_std_hi", "location_verdict",
                               "location_verdict_4strata"]]
        ep = ep.rename(columns={c: "emp_" + c for c in ep.columns if c != "task"})
        df = df.merge(ep, on="task", how="left")
    return df


def failure_route_audit(df):
    """[fix round 2] every GUE failure-route label against the post-hoc checks (no fits):
    positional -> B=8 delta CI (unaffected by PH4/PH5); contamination -> PH5 location verdict;
    shared-motif -> PH4 labels under length-normalised richness."""
    rsp = os.path.join(RES, "gue_rule_sensitivity.csv")
    if not os.path.exists(rsp) or "emp_location_verdict" not in df:
        return pd.DataFrame()
    rs = pd.read_csv(rsp).set_index("task")
    lref = [c for c in rs.columns if c.startswith("route_Lref_")]
    rows = []
    rr = df.route.replace({"solved (motif-discriminable)": "solved"})
    for _, x in df[~rr.isin(["solved", "hard (no single route)"])].iterrows():
        r = dict(task=x.task, family=x.family, route=x.route, frac_in_any_gb_nt=x.frac_in_any_gb_nt,
                 independence_class=x.independence_class)
        if x.route == "positional":
            r.update(check="B=8 delta CI", value=f"{x.pos_delta:+.3f} [{x.pos_delta_lo:+.3f}, {x.pos_delta_hi:+.3f}]",
                     affected_by_posthoc=False,
                     reading="not affected by PH4 (Route 2 has no richness clause) or PH5 (not EMP)")
        elif x.route == "contamination":
            r.update(check="PH5 location-standardised TATA neg-excess (2 strata; 4 strata)",
                     value=(f"{x.tata_neg_excess:+.3f} -> {x.emp_tata_excess_std:+.3f} [{x.emp_tata_excess_std_lo:+.3f}, "
                            f"{x.emp_tata_excess_std_hi:+.3f}]"),
                     affected_by_posthoc=True,
                     reading=(f"label-0 windows are low-signal genomic windows, not sampled negatives; location: "
                              f"{x.emp_location_verdict} ({x.emp_location_verdict_4strata} with 4 strata)"))
        else:
            labs = {c.replace("route_Lref_", "L") + "bp": rs.loc[x.task, c] for c in lref}
            flips = [k for k, v in labs.items() if v != x.route]
            r.update(check="PH4 labels with richness per L_ref bp", value="; ".join(f"{k}:{v}" for k, v in labs.items()),
                     affected_by_posthoc=bool(flips),
                     reading=(f"becomes hard at {', '.join(flips)}" if flips else "stable under length normalisation"))
        rows.append(r)
    return pd.DataFrame(rows)


def localisation_table(df=None):
    """[fix round 1] PH3 per task x bin (run_gue_posthoc_r2.py). [fix round 2] carrier/necessary left NA
    where the full B=8 delta CI includes 0 (the comparator nt_promoter_tata)."""
    full_ci = {}
    if df is not None:
        full_ci.update({t: (lo, hi) for t, lo, hi in df[["task", "pos_delta_lo", "pos_delta_hi"]].values})
    p15p = os.path.join(RES, "gue_paper15_reference.csv")
    if os.path.exists(p15p):
        p15 = pd.read_csv(p15p)
        full_ci.update({t: (lo, hi) for t, lo, hi in p15[["task", "pos_delta_lo", "pos_delta_hi"]].values})
    d = os.path.join(RES, "gue_posthoc_r2")
    rows = []
    if not os.path.isdir(d):
        return pd.DataFrame()
    for t in ["prom_300_tata", "prom_core_tata", "splice_reconstructed", "nt_promoter_tata"]:
        p = os.path.join(d, f"{t}.json")
        if not os.path.exists(p):
            continue
        o = json.load(open(p))
        pr = o["profiles"]; binary = set(pr) == {"0", "1"}
        for b in o["per_bin"]:
            i = b["bin"]
            r = dict(task=t, k=o["k"], spectrum_mcc=o["spectrum_mcc"], positional_mcc=o["positional_mcc"],
                     full_delta=o["full_delta"], bin=i, bin_start_bp=o["bin_edges_bp"][i],
                     bin_end_bp=o["bin_edges_bp"][i + 1], **{kk: vv for kk, vv in b.items() if kk != "bin"},
                     block_gain_share=o["block_gain_share"][f"bin{i}"], top5_kmers=" ".join(o["top5_by_bin"][str(i)]),
                     carrier=i in o["carrier_bins"], necessary=i in o["necessary_bins"],
                     global_block_gain_share=o["block_gain_share"]["global"],
                     reading=o["reading"], tata_placement_is_mechanism=o["tata_placement_is_mechanism"])
            lo, hi = full_ci.get(t, (np.nan, np.nan))
            sig = bool(lo > 0 or hi < 0)
            r.update(full_delta_lo=lo, full_delta_hi=hi, full_delta_significant=sig)
            if not sig:
                r["carrier"] = None; r["necessary"] = None
            if binary:
                for c in ("1", "0"):
                    r[f"tbp_hit_rate_class{c}"] = pr[c]["tbp_hit_rate_by_bin"][i]
                    r[f"tata4_rate_class{c}"] = pr[c]["tata_4mer_rate_by_bin"][i]
                    r[f"gc_class{c}"] = pr[c]["gc_by_bin"][i]
                    r[f"cpg_count_class{c}"] = pr[c]["cpg_count_by_bin"][i]
            rows.append(r)
    return pd.DataFrame(rows)


def independence_routes(df):
    """[fix round 1] route labels and signatures per independence class (gue_overlap_summary.csv)."""
    if "independence_class" not in df:
        return pd.DataFrame()
    rows = []
    for cls, d in df.groupby("independence_class", sort=False):
        rr = d.route.replace({"solved (motif-discriminable)": "solved"})
        fail = d[~rr.isin(["solved", "hard (no single route)"])]
        rows.append(dict(independence_class=cls, n_datasets=len(d), tasks=", ".join(d.task),
                         frac_in_any_gb_nt_range=f"{d.frac_in_any_gb_nt.min():.3f}-{d.frac_in_any_gb_nt.max():.3f}",
                         routes="; ".join(f"{k}:{v}" for k, v in rr.value_counts().items()),
                         n_failure_route=len(fail), failure_route_tasks=", ".join(fail.task),
                         n_route3_signature=int(d.route3_signature.fillna(False).astype(bool).sum()),
                         n_route2_signature=int(d.route2_signature.fillna(False).astype(bool).sum()),
                         n_comp_dominated=int(d.composition_dominated.sum())))
    return pd.DataFrame(rows)


def published_comparison(df):
    pub = pd.read_csv(os.path.join(RES, "gue_published.csv"))
    key = {("DNABERT-2 Table 6", "DNABERT-2"): "DNABERT-2",
           ("DNABERT-2 Table 6", "DNABERT-2♦"): "DNABERT-2 (+GUE pre-training)",
           ("DNABERT-2 Table 6", "NT-2500M-multi"): "NT-2500M-multi",
           ("DNABERT-2 Table 6", "NT-500M-human"): "NT-500M-human",
           ("DNABERT-2 Table 6", "DNABERT (3-mer)"): "DNABERT (3-mer)",
           ("DNABERT-2 Table 9", "HyenaDNA"): "HyenaDNA",
           ("DNABERT-2 Table 9", "DNABERT-2 w/o PT"): "DNABERT-2 (no pre-training)"}
    rows = []
    for _, x in df.iterrows():
        t = x.task
        ours = 100 * (x.f1_macro if t == "covid" else x.mcc)
        lo = 100 * (x.f1_lo if t == "covid" else x.mcc_lo); hi = 100 * (x.f1_hi if t == "covid" else x.mcc_hi)
        comp = 100 * (x.comp_f1_macro if t == "covid" else x.comp_mcc)
        p = pub[pub.task == t]
        row = dict(task=t, family=x.family, metric=("macro-F1 x100" if t == "covid" else "MCC x100"),
                   kmer_lgbm=round(ours, 2), kmer_lgbm_lo=round(lo, 2), kmer_lgbm_hi=round(hi, 2),
                   composition_only=round(comp, 2))
        for (src, m), name in key.items():
            v = p[(p.source == src) & (p.model == m)].value_x100
            row[name] = float(v.iloc[0]) if len(v) else np.nan
        v9 = p[(p.source == "DNABERT-2 Table 9") & (p.model == "DNABERT-2")].value_x100
        row["DNABERT-2 (Table 9 row)"] = float(v9.iloc[0]) if len(v9) else np.nan
        # best published over every parsed row (all Table-6 models + Table-9 HyenaDNA / w/o PT; the
        # Table-9 DNABERT-2 row is excluded because it re-prints DNABERT-2 with different values)
        cand = p[~((p.source == "DNABERT-2 Table 9") & (p.model == "DNABERT-2"))]
        best = cand.loc[cand.value_x100.idxmax()]
        row["best_published"] = float(best.value_x100); row["best_published_model"] = f"{best.model} ({best.source[-7:]})"
        row["gap_to_DNABERT2"] = round(ours - row["DNABERT-2"], 2)
        row["gap_to_best"] = round(ours - row["best_published"], 2)
        row["comp_gap_to_DNABERT2"] = round(comp - row["DNABERT-2"], 2)
        row["DNABERT2_in_our_CI"] = bool(lo <= row["DNABERT-2"] <= hi)
        allv = list(cand.value_x100) + [ours]
        row["kmer_rank_among_published"] = int(1 + sum(v > ours for v in cand.value_x100))
        row["n_published_models"] = int(len(cand))
        row["kmer_beats_models"] = int(sum(v < ours for v in cand.value_x100))
        row["comp_beats_models"] = int(sum(v < comp for v in cand.value_x100))
        row["note"] = ("covid: the archive's train/dev/test partition (73335/9166/9168) differs from Table 12 "
                       "(77669/7000/7000); published numbers may be on a different test split"
                       if t == "covid" else "")
        rows.append(row)
    return pd.DataFrame(rows)


def posthoc_tables():
    eq, tw = [], []
    for t in G.ORDER:
        p = os.path.join(PH, f"{t}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        w = d["PH2_twins"]; w = dict(task=t, family=G.neg_family(t), **w); tw.append(w)
        e = d.get("PH1_equalization")
        if e and e.get("feasible"):
            row = dict(task=t, family=G.neg_family(t), k=e["k"], n_test=e["n_test"],
                       retained_neg_frac_train=e["retained_neg_frac_train"],
                       retained_neg_frac_test=e["retained_neg_frac_test"], n_test_neg_matched=e["n_test_neg_matched"])
            for m in ("kmer", "comp"):
                b = e[m]
                for q in ("A0", "Ar_mean", "Am", "size_effect", "matched_vs_random", "total_drop", "rel_gap"):
                    row[f"{m}_{q}"] = b[q]
                    if f"{q}_ci" in b:
                        row[f"{m}_{q}_lo"], row[f"{m}_{q}_hi"] = b[f"{q}_ci"]
                row[f"{m}_Ar_sd_over_arms"] = b["Ar_sd_over_arms"]
            row["R1_supported"] = e["R1_supported"]; row["reading"] = e["reading"]
            eq.append(row)
        elif e is not None:
            eq.append(dict(task=t, family=G.neg_family(t), reading="infeasible: " + e.get("reason", "")))
    return pd.DataFrame(eq), pd.DataFrame(tw)


def predictions(df, tw):
    rows = []
    m = df[df.family == "mouse_tf"]
    if len(m):
        a = m.comp_auroc.between(0.45, 0.55)
        rows.append(dict(prediction="P1a", statement="mouse TF (dinucleotide-shuffled negatives): composition-only "
                         "AUROC within 0.05 of 0.5", n_tasks=len(m), n_hold=int(a.sum()),
                         values="; ".join(f"{t}={v:.3f}" for t, v in zip(m.task, m.comp_auroc)),
                         outcome="HOLDS" if a.all() else ("PARTLY" if a.any() else "FAILS")))
        b = m.comp_fraction.abs() <= 0.10
        rows.append(dict(prediction="P1b", statement="mouse TF: composition fraction ~0 (read as |fraction| <= 0.10)",
                         n_tasks=len(m), n_hold=int(b.sum()),
                         values="; ".join(f"{t}={v:+.3f}" for t, v in zip(m.task, m.comp_fraction)),
                         outcome="HOLDS" if b.all() else ("PARTLY" if b.any() else "FAILS")))
        c = m.eq_retained_neg_frac > 0.5
        rows.append(dict(prediction="P1c", statement="mouse TF: equalized split retains most (>50%) negatives",
                         n_tasks=len(m), n_hold=int(c.sum()),
                         values="; ".join(f"{t}={v:.3f}" for t, v in zip(m.task, m.eq_retained_neg_frac)),
                         outcome="HOLDS" if c.all() else ("PARTLY" if c.any() else "FAILS")))
    h = df[df.family == "human_tf"]
    if len(h):
        a = (h.gc_auroc - 0.5).abs() <= 0.05
        rows.append(dict(prediction="P2", statement="human TF (GC-matched negatives): GC AUROC within 0.05 of 0.5",
                         n_tasks=len(h), n_hold=int(a.sum()),
                         values="; ".join(f"{t}={v:.3f}" for t, v in zip(h.task, h.gc_auroc)),
                         outcome="HOLDS" if a.all() else ("PARTLY" if a.any() else "FAILS")))
    s = df[df.task == "splice_reconstructed"]
    if len(s):
        x = s.iloc[0]
        ok = bool(x.pos_significant and x.pos_delta > 0)
        rows.append(dict(prediction="P3a", statement="splice/reconstructed: significant positive positional delta",
                         n_tasks=1, n_hold=int(ok), values=f"delta={x.pos_delta:+.4f} [{x.pos_delta_lo:+.4f},{x.pos_delta_hi:+.4f}]",
                         outcome="HOLDS" if ok else "FAILS"))
        r2 = df[(df.pos_significant) & (df.pos_delta > 0.2)]
        other = [t for t in r2.task if t != "splice_reconstructed"]
        rows.append(dict(prediction="P3b", statement="splice is the only GUE task that can meet the Route-2 rule "
                         "(significant delta > 0.2)", n_tasks=len(df), n_hold=int(len(other) == 0),
                         values="tasks meeting the delta rule: " + ", ".join(f"{t} ({v:+.3f})" for t, v in zip(r2.task, r2.pos_delta)),
                         outcome="HOLDS" if not other else "FAILS"))
    rows.append(dict(prediction="P4", statement="no magnitude prediction for the promoter tasks", n_tasks=0, n_hold=0,
                     values="none", outcome="not applicable"))
    return pd.DataFrame(rows)


def main():
    df = load_cards()
    df = add_posthoc_columns(df)
    df.to_csv(os.path.join(RES, "gue_results.csv"), index=False)
    print(f"gue_results.csv: {len(df)} tasks")
    cmp_ = published_comparison(df)
    cmp_.to_csv(os.path.join(RES, "gue_published_comparison.csv"), index=False)
    eq, tw = posthoc_tables()
    if len(eq):
        eq.to_csv(os.path.join(RES, "gue_equalization_decomposition.csv"), index=False)
    if len(tw):
        tw.to_csv(os.path.join(RES, "gue_twins.csv"), index=False)
    pr = predictions(df, tw)
    pr.to_csv(os.path.join(RES, "gue_predictions.csv"), index=False)
    ir = independence_routes(df)
    if len(ir):
        ir.to_csv(os.path.join(RES, "gue_independence_routes.csv"), index=False)
    lo = localisation_table(df)
    if len(lo):
        lo.to_csv(os.path.join(RES, "gue_positional_localisation.csv"), index=False)
    fa = failure_route_audit(df)
    if len(fa):
        fa.to_csv(os.path.join(RES, "gue_failure_route_audit.csv"), index=False)

    # family summary
    fam = []
    for f in FAMILY_ORDER:
        d = df[df.family == f]
        if not len(d):
            continue
        c = cmp_[cmp_.family == f]
        fam.append(dict(family=f, name=FAMILY_NAME[f], n_datasets=len(d),
                        n_test_range=f"{d.n_test.min()}-{d.n_test.max()}",
                        mean_kmer=round(c.kmer_lgbm.mean(), 1), mean_comp_only=round(c.composition_only.mean(), 1),
                        mean_DNABERT2=round(c["DNABERT-2"].mean(), 1), mean_NT2500M_multi=round(c["NT-2500M-multi"].mean(), 1),
                        mean_HyenaDNA=round(c["HyenaDNA"].mean(), 1),
                        mean_gap_to_DNABERT2=round(c.gap_to_DNABERT2.mean(), 1),
                        n_kmer_ge_DNABERT2=int((c.gap_to_DNABERT2 >= 0).sum()),
                        comp_fraction_range=f"{d.comp_fraction.min():.2f}-{d.comp_fraction.max():.2f}",
                        n_comp_dominated=int(d.composition_dominated.sum()),
                        routes="; ".join(f"{k}:{v}" for k, v in d.route.value_counts().items()),
                        tiers="; ".join(f"{k}:{v}" for k, v in d.tier.value_counts().items()),
                        max_pos_delta=round(d.pos_delta.max(), 3),
                        n_route3_signature=(int(d.route3_signature.fillna(False).astype(bool).sum())
                                            if "route3_signature" in d else None),
                        independence=("; ".join(sorted(set(d.independence_class))) if "independence_class" in d else None),
                        frac_in_any_gb_nt_range=(f"{d.frac_in_any_gb_nt.min():.2f}-{d.frac_in_any_gb_nt.max():.2f}"
                                                 if "frac_in_any_gb_nt" in d else None),
                        neg_construction=G.NEG_CONSTRUCTION.get(f, "")))
    fam = pd.DataFrame(fam)
    fam.to_csv(os.path.join(RES, "gue_family_summary.csv"), index=False)

    # suite comparison
    p15p = os.path.join(RES, "gue_paper15_reference.csv")
    rows = []
    if os.path.exists(p15p):
        p15 = pd.read_csv(p15p)
        rsp = os.path.join(RES, "gue_rule_sensitivity.csv")
        rs = pd.read_csv(rsp) if os.path.exists(rsp) else None
        for s in ("GB", "NT"):
            d = p15[p15.suite == s]
            rows.append(dict(suite=s, n_datasets=len(d), routes="; ".join(f"{k}:{v}" for k, v in d.paper_route.value_counts().items()),
                             rule_routes="; ".join(f"{k}:{v}" for k, v in d.rule_route.value_counts().items()),
                             n_solved_tier=int((d.mcc >= 0.7).sum()), n_partial_tier=int(((d.mcc >= 0.5) & (d.mcc < 0.7)).sum()),
                             n_hard_tier=int((d.mcc < 0.5).sum()),
                             n_binary=int((d.n_classes == 2).sum()),
                             n_comp_dominated_binary=int(d[(d.n_classes == 2)].composition_dominated.sum()),
                             n_comp_dominated_solved_binary=int(d[(d.n_classes == 2) & (d.mcc >= 0.7)].composition_dominated.sum()),
                             median_comp_fraction_binary=round(d[d.n_classes == 2].comp_fraction.median(), 3),
                             n_route3_signature=(int(rs[rs.task.isin(d.task)].route3_signature.fillna(False).astype(bool).sum())
                                                 if rs is not None else None),
                             source="gue_paper15_reference.csv (routes: submitted Table 1)"))
    gr = df.route.replace({"solved (motif-discriminable)": "solved"}).value_counts()
    rows.append(dict(suite="GUE", n_datasets=len(df), routes="n/a (no hand labels)",
                     rule_routes="; ".join(f"{k}:{v}" for k, v in gr.items()),
                     n_solved_tier=int((df.mcc >= 0.7).sum()), n_partial_tier=int(((df.mcc >= 0.5) & (df.mcc < 0.7)).sum()),
                     n_hard_tier=int((df.mcc < 0.5).sum()), n_binary=int((df.n_classes == 2).sum()),
                     n_comp_dominated_binary=int(df[df.n_classes == 2].composition_dominated.sum()),
                     n_comp_dominated_solved_binary=int(df[(df.n_classes == 2) & (df.mcc >= 0.7)].composition_dominated.sum()),
                     median_comp_fraction_binary=round(df[df.n_classes == 2].comp_fraction.median(), 3),
                     n_route3_signature=(int(df.route3_signature.fillna(False).astype(bool).sum())
                                         if "route3_signature" in df else None),
                     source="gue_results.csv (rule_routes: diagnose._verdict, pre-registered)"))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "gue_suite_comparison.csv"), index=False)

    # compact paper table (per family)
    pt = fam[["name", "n_datasets", "frac_in_any_gb_nt_range", "mean_kmer", "mean_comp_only", "mean_DNABERT2",
              "comp_fraction_range", "n_comp_dominated", "max_pos_delta", "n_route3_signature", "routes"]].copy()
    pt["routes"] = pt.routes.str.replace("solved (motif-discriminable)", "solved", regex=False)
    loc_note = {}
    if "emp_location_verdict" in df:
        e = df[(df.family == "emp") & (df.route3_signature.fillna(False).astype(bool))]
        vc_ = e.emp_location_verdict.value_counts()
        loc_note["emp"] = (f"Route-3 signature {len(e)}: location explains {vc_.get('explained by location', 0)}, "
                           f"partly {vc_.get('partly explained', 0)}, not {vc_.get('not explained', 0)} (PH5); "
                           "negatives are low-signal genomic windows")
    pt["note"] = ["composition-only inverted by shuffle twins split across train/test (gue_twins.csv)"
                  if f == "mouse_tf" else ("covid: macro-F1 x100; archive partition differs from Table 12" if f == "covid"
                                           else loc_note.get(f, ""))
                  for f in fam.family]
    pt.to_csv(os.path.join(RES, "gue_paper_table.csv"), index=False)
    print(fam.to_string(index=False))
    print(pr.to_string(index=False))


if __name__ == "__main__":
    main()
