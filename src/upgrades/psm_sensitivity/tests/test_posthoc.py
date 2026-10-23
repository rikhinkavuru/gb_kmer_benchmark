#!/usr/bin/env python3
"""Sanity tests for the POST HOC outputs (run_posthoc.py + summarize_psm.py). Needs the outputs.

1. dup_masks on a toy context.
2. Leak-free test: recomputed INDEPENDENTLY from the frozen snapshot + prediction cache with sklearn
   (no module code): the nt original-model AUROC without the test sequences that occur in train must
   equal the nt_enhancers_leakfreeTest row, and that row must have 100 test positives / 200 negatives.
3. Leak-free refit: no training row of any refit arm equals a test sequence (checked on the rebuilt arms
   is expensive, so the check is on the leakage table + the refit original row's training size).
4. between_dataset: the point share equals (placebo - overlap) / (placebo - 0.5) and the contrast
   row equals the difference of the two dataset rows.
5. dissociation_stats: every arm enters once (n_F1 + n_F2 + n_F3 == n_arms).
6. weighting diagnostic and leak-free label-noise estimate (decision.csv vs sweep rows / formula).
7. P6 composition-normalised contrast: the composition-only point AUROCs under the overlap / placebo weights
   equal the pre-registered sweep rows (a different code path), and the share / difference-of-contrasts /
   ratio-difference columns follow from the dataset rows; the P6 draws share their resamples with P4
   (same n_test; the bit-identity of the k-mer draws is asserted inside run_posthoc.py).
8. P0b interval audit: the nt overlapping test sequences are exactly the 100 verbatim duplicates (same AUROC
   without them as the leak-free test), and cohn's overlapping test sequences are all negatives.
9. P7 decomposition CIs (fix round 2): the point A0 / Ar_mean / Am of every dataset equal decision.csv (k-mer
   within = A0 / Ar_mean / Am; composition-only within = comp0 / comp_r_mean / comp_m; k-mer train_only =
   trainonly_* columns; these come from the summarize_psm code path), the decomposition arithmetic holds, the
   kmer_minus_comp rows are the differences of the model rows, every CI brackets its point estimate and no
   bootstrap draw is NaN.
10. Caveat (11): on the published cohn split both the unweighted and the class-balanced-training k-mer models
   call at most a handful of test sequences negative at 0.5 and have the SAME MCC@0.5 (independent sklearn
   recompute from the prediction caches and the published split file; threshold p > 0.5 as in psm_core).
11. Fix round 3: per-design F2 gaps, the P8 F2 band construction (2q of positives outside the band) with one
   independent sklearn recompute, the caliper sweep and the per-design dissociation rows. Test 9 also covers the
   P7 fixed_model rows and the composition-only paired-design rows.

Run:  bibm/lowcpu.sh <python> src/upgrades/psm_sensitivity/tests/test_posthoc.py
"""
import os
import sys
import types

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import _common as C  # noqa: E402
import psm_data as D  # noqa: E402

RES = C.RESULTS_DIR


def test_dup_masks():
    import run_posthoc as RP
    ctx = types.SimpleNamespace(tr=["AAA", "CCC", "GGG", "CCC"], te=["CCC", "TTT", "GGG"])
    dte, dtr = RP.dup_masks(ctx)
    assert dte.tolist() == [True, False, True]
    assert dtr.tolist() == [False, True, True, True]


def test_leakfree_test_independent():
    df = pd.read_csv(os.path.join(D.SNAP_DIR, "snap_nt_enhancers.csv.gz"),
                     dtype={"split": str, "label": np.int64, "file_id": str, "sequence": str})
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    keep = ~te["sequence"].isin(set(tr["sequence"])).to_numpy()
    y = te["label"].to_numpy()
    assert keep.sum() == 300 and (y[~keep] == 1).all()
    p = np.load(os.path.join(D.SNAP_DIR, "preds", "nt_enhancers", "original.npz"))["kmer"]
    auc = roc_auc_score(y[keep], p[keep])
    sw = pd.read_csv(os.path.join(RES, "psm_sensitivity_posthoc_sweep.csv"))
    r = sw[(sw["dataset"] == "nt_enhancers_leakfreeTest") & (sw["family"] == "original") & (sw["design"] == "within")
           & (sw["model"] == "kmer")].iloc[0]
    assert abs(r["auroc"] - auc) < 1e-9, (r["auroc"], auc)
    assert int(r["n_pos"]) == 100 and int(r["n_neg"]) == 200
    lk = pd.read_csv(os.path.join(RES, "psm_sensitivity_leakage_check.csv")).set_index("dataset")
    assert abs(lk.loc["nt_enhancers", "orig_auroc_leakfree_test"] - auc) < 1e-9
    assert lk.loc["human_enhancers_cohn", "n_test_verbatim_in_train"] == 0
    assert lk.loc["drosophila_enhancers_stark", "n_test_verbatim_in_train"] == 0
    print(f"PASS leak-free test: nt original AUROC {auc:.6f} on 100 pos / 200 neg")


def test_leakfree_refit_sizes():
    sw = pd.read_csv(os.path.join(RES, "psm_sensitivity_posthoc_sweep.csv"))
    lk = pd.read_csv(os.path.join(RES, "psm_sensitivity_leakage_check.csv")).set_index("dataset")
    q = sw[(sw["dataset"] == "nt_enhancers_leakfreeRefit") & (sw["family"] == "original") & (sw["design"] == "within")
           & (sw["model"] == "kmer")]
    if not len(q):
        print("SKIP leak-free refit (not run yet)")
        return
    r = q.iloc[0]
    assert int(r["n_train"]) == int(lk.loc["nt_enhancers", "n_train"]) - int(lk.loc["nt_enhancers", "n_train_rows_equal_a_test_seq"])
    assert int(r["n_pos"]) == 200 and int(r["n_neg"]) == 200
    print(f"PASS leak-free refit: original trained on {int(r['n_train'])} rows, full 400-sequence test")


def test_between_dataset_arithmetic():
    bt = pd.read_csv(os.path.join(RES, "psm_sensitivity_between_dataset.csv"))
    ds = bt[bt["kind"] == "dataset"]
    for _, r in ds.iterrows():
        s = (r["auroc_placebo_mean"] - r["auroc_overlap"]) / (r["auroc_placebo_mean"] - 0.5)
        assert abs(s - r["share_above_chance_lost"]) < 1e-12
        assert r["share_lo"] <= r["share_above_chance_lost"] + 0.05 and r["share_hi"] >= r["share_above_chance_lost"] - 0.05
    for _, r in bt[bt["kind"] == "contrast"].iterrows():
        a, b = r["contrast"].split(" - ")
        ra = ds[(ds["dataset"] == a) & (ds["design"] == r["design"])].iloc[0]
        rb = ds[(ds["dataset"] == b) & (ds["design"] == r["design"])].iloc[0]
        assert abs(ra["share_above_chance_lost"] - rb["share_above_chance_lost"] - r["share_above_chance_lost"]) < 1e-12
    print(f"PASS between-dataset arithmetic ({len(ds)} dataset rows)")


def test_dissociation_unique_arms():
    d = pd.read_csv(os.path.join(RES, "psm_sensitivity_dissociation_stats.csv"))
    assert ((d["n_F1"] + d["n_F2"] + d["n_F3"]) == d["n_arms"]).all()
    print("PASS dissociation_stats uses each arm once")


def test_weighting_diagnostic_and_labelnoise():
    """(h) the composition-only AUROC under overlap weights in decision.csv equals the sweep rows (matched arm,
    fixed_model and within designs) and the over-correction flag is exactly (value < 0.5); (j) the leak-free nt
    label-noise estimate is rho * (Ar_mean - 0.5) with rho from the raw nt ELS rates."""
    dec = pd.read_csv(os.path.join(RES, "psm_sensitivity_decision.csv")).set_index("dataset")
    sw = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in
                    ("psm_sensitivity_sweep.csv", "psm_sensitivity_posthoc_sweep.csv")], ignore_index=True)
    n = 0
    for ds, r in dec.iterrows():
        for design, tag in (("within", "overlap"), ("fixed_model", "fixedoverlap")):
            kindcol = "train_kind" if design == "within" else "test_kind"
            q = sw[(sw["dataset"] == ds) & (sw["family"] == "W_overlap") & (sw["design"] == design)
                   & (sw["model"] == "comp") & (sw[kindcol] == "matched")]
            if not len(q) or pd.isna(r.get(f"{tag}_comp_auroc", np.nan)):
                continue
            assert abs(q["auroc"].iloc[0] - r[f"{tag}_comp_auroc"]) < 1e-12, (ds, design)
            assert bool(r[f"{tag}_weights_overcorrect"]) == bool(r[f"{tag}_comp_auroc"] < 0.5)
            n += 1
    nt = dec.loc["nt_enhancers"]
    rho = (nt["els_rate_retained"] - nt["els_rate_neg_all"]) / (nt["els_rate_pos"] - nt["els_rate_neg_all"])
    for ds in ("nt_enhancers_leakfreeTest", "nt_enhancers_leakfreeRefit"):
        r = dec.loc[ds]
        assert abs(r["rho_hidden_pos"] - rho) < 1e-12
        assert abs(r["labelnoise_max_auroc_drop"] - rho * (r["Ar_mean"] - 0.5)) < 1e-12
        assert abs(r["labelnoise_bound_over_shortcut_drop"] - r["labelnoise_max_auroc_drop"] / r["drop_shortcut"]) < 1e-12
    print(f"PASS weighting diagnostic ({n} rows) and leak-free label-noise estimate")


def test_between_comp_p6():
    bt = pd.read_csv(os.path.join(RES, "psm_sensitivity_between_dataset.csv"))
    dec = pd.read_csv(os.path.join(RES, "psm_sensitivity_decision.csv")).set_index("dataset")
    ds = bt[bt["kind"] == "dataset"]
    assert ds["comp_share_lost"].notna().all(), "P6 columns missing"
    n = 0
    for _, r in ds.iterrows():
        tag = "fixedoverlap" if r["design"] == "fixed_model_overlap" else "overlap"
        d = dec.loc[r["dataset"]]
        assert abs(r["comp_auroc_overlap"] - d[f"{tag}_comp_auroc"]) < 1e-9, (r["dataset"], r["design"])
        assert abs(r["comp_auroc_placebo_mean"] - d[f"{tag}_comp_placebo"]) < 1e-9, (r["dataset"], r["design"])
        cs = (r["comp_auroc_placebo_mean"] - r["comp_auroc_overlap"]) / (r["comp_auroc_placebo_mean"] - 0.5)
        assert abs(cs - r["comp_share_lost"]) < 1e-12
        assert abs(r["kmer_minus_comp_share"] - (r["share_above_chance_lost"] - cs)) < 1e-12
        assert abs(r["kmer_over_comp_share"] - r["share_above_chance_lost"] / cs) < 1e-12
        assert r["comp_share_lo"] <= cs + 0.05 and r["comp_share_hi"] >= cs - 0.05
        assert bool(r["share_interpretable"]) == bool(r["auroc_overlap"] >= 0.5 and r["comp_auroc_overlap"] >= 0.5)
        n += 1
    for _, r in bt[bt["kind"] == "contrast"].iterrows():
        a, b = r["contrast"].split(" - ")
        ra = ds[(ds["dataset"] == a) & (ds["design"] == r["design"])].iloc[0]
        rb = ds[(ds["dataset"] == b) & (ds["design"] == r["design"])].iloc[0]
        assert abs(r["comp_share_diff"] - (ra["comp_share_lost"] - rb["comp_share_lost"])) < 1e-12
        assert abs(r["diff_of_contrasts"] - (r["share_above_chance_lost"] - r["comp_share_diff"])) < 1e-12
        assert abs(r["ratio_diff"] - (ra["kmer_over_comp_share"] - rb["kmer_over_comp_share"])) < 1e-12
        assert r["diff_of_contrasts_lo"] <= r["diff_of_contrasts_hi"] and r["ratio_diff_lo"] <= r["ratio_diff_hi"]
    parts = os.path.join(D.SNAP_DIR, "parts_posthoc")
    for lab in ds["dataset"].unique():
        k, c = np.load(os.path.join(parts, f"between_{lab}.npz")), np.load(os.path.join(parts, f"between_comp_{lab}.npz"))
        assert int(k["n_test"]) == int(c["n_test"]) and len(k["fixed_model_overlap_aw"]) == len(c["fixed_model_overlap_aw"])
    print(f"PASS P6 composition-normalised contrast ({n} dataset rows match the sweep; contrast arithmetic)")


def test_interval_audit_p0b():
    li = pd.read_csv(os.path.join(RES, "psm_sensitivity_leakage_interval.csv")).set_index("dataset")
    lk = pd.read_csv(os.path.join(RES, "psm_sensitivity_leakage_check.csv")).set_index("dataset")
    nt = li.loc["nt_enhancers"]
    assert int(nt["n_test_overlap_any"]) == int(lk.loc["nt_enhancers", "n_test_verbatim_in_train"]) == 100
    assert abs(nt["orig_auroc_without_overlapping"] - lk.loc["nt_enhancers", "orig_auroc_leakfree_test"]) < 1e-12
    co = li.loc["human_enhancers_cohn"]
    assert int(co["n_test_overlap_any_pos"]) == 0 and int(co["n_test_overlap_any"]) == int(co["n_test_overlap_any_neg"])
    assert int(co["n_test_overlap_half"]) <= int(co["n_test_overlap_any"])
    print(f"PASS P0b interval audit (nt overlaps == the 100 verbatim duplicates; cohn {int(co['n_test_overlap_any'])} "
          f"overlapping test sequences, all negatives)")


def test_decomposition_p7():
    dc = pd.read_csv(os.path.join(RES, "psm_sensitivity_decomposition_ci.csv"))
    dec = pd.read_csv(os.path.join(RES, "psm_sensitivity_decision.csv")).set_index("dataset")
    q = dc.set_index(["dataset", "design", "model", "matched_seeds"])
    n = 0
    for ds, d in dec.iterrows():
        checks = [(("within", "kmer"), ("A0", "Ar_mean", "Am")),
                  (("within", "comp"), ("comp0", "comp_r_mean", "comp_m")),
                  (("train_only", "kmer"), ("A0", "trainonly_A_randomTrain_mean", "trainonly_A_matchedTrain")),
                  # fix round 3: P7 fixed_model rows and the composition-only paired-design rows
                  (("fixed_model", "kmer"), ("A0", "fixed_A_randomTest_mean", "fixed_A_matchedTest")),
                  (("fixed_model", "comp"), ("comp0", None, "comp_fixed_Am")),
                  (("train_only", "comp"), ("comp0", "comp_trainonly_Ar_mean", "comp_trainonly_Am"))]
        for (design, model), cols in checks:
            r = q.loc[(ds, design, model, "42")]
            for mine, theirs in zip(("A0", "Ar_mean", "Am"), cols):
                if theirs is None:
                    continue
                assert abs(r[mine] - d[theirs]) < 1e-9, (ds, design, model, mine, r[mine], d[theirs])
            n += 1
        assert abs(q.loc[(ds, "within", "kmer", "42"), "size_effect"] - d["drop_size"]) < 1e-9
        assert abs(q.loc[(ds, "within", "kmer", "42"), "matched_vs_random"] - d["drop_shortcut"]) < 1e-9
    for _, r in dc.iterrows():
        cols = ["size_effect", "matched_vs_random", "rel_gap"]
        if r["model"] != "kmer_minus_comp":
            assert abs(r["size_effect"] - (r["A0"] - r["Ar_mean"])) < 1e-12
            assert abs(r["matched_vs_random"] - (r["Ar_mean"] - r["Am"])) < 1e-12
            assert abs(r["rel_gap"] - (r["Ar_mean"] - r["Am"]) / (r["Ar_mean"] - 0.5)) < 1e-12
            assert int(r["n_boot_nan"]) == 0
        else:
            k = q.loc[(r["dataset"], r["design"], "kmer", r["matched_seeds"])]
            c = q.loc[(r["dataset"], r["design"], "comp", r["matched_seeds"])]
            for col in cols:
                assert abs(r[col] - (k[col] - c[col])) < 1e-12
        for col in cols:
            assert r[f"{col}_lo"] <= r[col] + 1e-12 and r[f"{col}_hi"] >= r[col] - 1e-12, (r["dataset"], col)
    print(f"PASS P7 decomposition ({n} point triples equal decision.csv; arithmetic; CIs bracket points)")


def test_bw_mcc_published_split():
    import run_psm_sensitivity as R
    from sklearn.metrics import matthews_corrcoef
    task = "human_enhancers_cohn"
    df = pd.read_csv(os.path.join(D.SNAP_DIR, f"snap_{task}.csv.gz"),
                     dtype={"split": str, "label": np.int64, "file_id": str, "sequence": str})
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    ctx = types.SimpleNamespace(task=task, tr=tr["sequence"].tolist(), ytr=tr["label"].to_numpy(),
                                te=te["sequence"].tolist(), yte=te["label"].to_numpy())
    a = R.paper_anchor(ctx)
    y = ctx.yte[a.te_idx]
    pt = pd.read_csv(os.path.join(RES, "psm_sensitivity_paper_table.csv"))
    pt = pt[pt["dataset"] == task].set_index("condition")
    out = []
    for cache, name, cond in ((task, "paper_equalized_matched", "published comp-equalized split (paper)"),
                              (task + "_balw", "paperbw_equalized_matched",
                               "[post hoc] published split, class-balanced training weights")):
        p = np.load(os.path.join(D.SNAP_DIR, "preds", cache, name + ".npz"))["kmer"][a.te_idx]
        pred = (p > 0.5).astype(int)            # same threshold rule as psm_core._all_metrics
        mcc = matthews_corrcoef(y, pred)
        assert abs(mcc - pt.loc[cond, "mcc05"]) < 1e-9, (cond, mcc, pt.loc[cond, "mcc05"])
        assert abs(roc_auc_score(y, p) - pt.loc[cond, "auroc"]) < 1e-9
        out.append((int((pred == 0).sum()), len(y), mcc))
    assert all(k <= 5 for k, _, _ in out) and abs(out[0][2] - out[1][2]) < 1e-12
    print("PASS caveat 11: predicted negatives at 0.5 on the published cohn split: "
          + "; ".join(f"{k} of {n} (MCC {m:.6f})" for k, n, m in out))


def test_f2_designs_band_caliper():
    """11 (fix round 3). f2_selection_difficulty per-design gaps equal summary.csv; P8: the fraction of training and
    test positives outside the band is 2q (construction; nt leak-free test excepted, whose band is defined on all
    test positives), every kept test negative lies in the band (asserted in run_posthoc.py), and one P8 AUROC is
    recomputed independently with sklearn (drosophila q0.3, within design, composition-only, in-band positives);
    caliper_sweep equals summary.csv; dissociation_stats has one row per design."""
    import run_psm_sensitivity as R
    summ = pd.read_csv(os.path.join(RES, "psm_sensitivity_summary.csv"))
    f2 = pd.read_csv(os.path.join(RES, "psm_sensitivity_f2_selection_difficulty.csv"))
    sk = summ.set_index(["dataset", "family", "level_label", "design", "model"])
    n = 0
    for _, r in f2.iterrows():
        for design, tag in (("within", ""), ("fixed_model", "fixed_"), ("train_only", "trainonly_")):
            for model in ("kmer", "comp"):
                q = sk.loc[(r["dataset"], "F2_trim", r["level_label"], design, model)]
                assert abs(r[f"{tag}{model}_gap"] - (q["auroc_random_mean"] - q["auroc_matched"])) < 1e-12
                n += 1
    band = pd.read_csv(os.path.join(RES, "psm_sensitivity_f2_band.csv"))
    for _, r in band.iterrows():
        assert abs(r["frac_train_pos_outside_band"] - 2 * r["q"]) < 0.005, r
        if r["dataset"] != "nt_enhancers_leakfreeTest":
            assert abs(r["frac_test_pos_outside_band"] - 2 * r["q"]) < 0.01, r
    task = "drosophila_enhancers_stark"
    ctx = R.Ctx(task, dict(R.DATASETS)[task])
    by = {a.name: a for a in R.build_arms(ctx)}
    lte = ctx.lte
    pos = ctx.yte == 1
    lo, hi = np.quantile(lte[pos], 0.3), np.quantile(lte[pos], 0.7)
    inb = pos & (lte >= lo) & (lte <= hi)
    a = by["F2_trim_q0.3_matched42"]
    t = np.zeros(len(ctx.yte), bool)
    t[a.te_idx] = True
    sel = t & ((ctx.yte == 0) | inb)
    p = np.load(os.path.join(D.SNAP_DIR, "preds", task, "F2_trim_q0.3_matched42.npz"))["comp"]
    mine = roc_auc_score(ctx.yte[sel], p[sel])
    theirs = band[(band["dataset"] == task) & (band["level_label"] == "q0.3")].iloc[0]["within_comp_inband_matched"]
    assert abs(mine - theirs) < 1e-12, (mine, theirs)
    cal = pd.read_csv(os.path.join(RES, "psm_sensitivity_caliper_sweep.csv"))
    for _, r in cal.iterrows():
        q = sk.loc[(r["dataset"], "F3_1to1", r["level_label"], "within", "kmer")]
        assert abs(r["kmer_matched"] - q["auroc_matched"]) < 1e-12 and abs(r["kmer_random_mean"] - q["auroc_random_mean"]) < 1e-12
    dst = pd.read_csv(os.path.join(RES, "psm_sensitivity_dissociation_stats.csv"))
    assert set(dst["design"]) == {"within", "fixed_model", "train_only"}
    assert not dst.duplicated(["dataset", "design"]).any()
    print(f"PASS F2 per-design gaps ({n}), P8 band construction + sklearn recompute ({mine:.6f}), caliper sweep, "
          f"dissociation per design")


if __name__ == "__main__":
    test_dup_masks()
    print("PASS dup_masks")
    test_leakfree_test_independent()
    test_leakfree_refit_sizes()
    test_between_dataset_arithmetic()
    test_dissociation_unique_arms()
    test_weighting_diagnostic_and_labelnoise()
    test_between_comp_p6()
    test_interval_audit_p0b()
    test_decomposition_p7()
    test_bw_mcc_published_split()
    test_f2_designs_band_caliper()
