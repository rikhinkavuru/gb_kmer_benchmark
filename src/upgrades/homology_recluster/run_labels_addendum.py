#!/usr/bin/env python3
"""Step 8 -- label ADDENDA (POST HOC, flagged as such) + the compact paper table. No model fitting; reads CSVs
and the cluster files only.

POST-HOC ADDENDA (written 2026-09-24 ~11:15, AFTER run_routes.py's pre-registered results were seen; they refine
the interpretation and do not replace the pre-registered preservation calls, which are carried unchanged):
A1 RE-POOLING CONTROL FOR RELABELS. The cluster-aware re-split pools the provided train and test splits, so a
   relabel under it can come from homology OR from re-pooling itself (e.g. removing a deliberately
   out-of-distribution provided test split). The label rule of run_routes.py is therefore also applied to the
   RANDOM re-split MCC (identical per-label targets; run_eval.py, split seed 42) and to the split-seed mean
   (seeds 42-44, run_seeds.py), with every non-MCC diagnostic identical to the homology-clean 'after' row.
   A relabel is HOMOLOGY-ATTRIBUTABLE iff label(cluster re-split) != label(random re-split); a relabel with
   label(cluster) == label(random) != label(before) is attributed to re-pooling / split design.
   Primary: split-seed means (a single split can be dominated by where a few giant families fall).
A2 SUBMITTED-PAPER LABEL SET. The submitted (June / BIBM) manuscript labels three non-TF tasks 'composition'
   (demo_human_or_worm, demo_coding_vs_intergenomic_seqs, dummy_mouse_enhancers_ensembl) and
   human_enhancers_ensembl 'contamination (weak)'; the working-tree v2 labels them solved / shared-motif.
   'composition' is a task-TYPE label (non-TF class contrast) and cannot be changed by homology cleaning; its
   supporting diagnostic, the composition fraction (run_extra.py X3), must keep sign with CI lower > 0.
   'contamination (weak)' is assessed with diagnose.py's Route-3 criterion (TATA neg-excess >= 0.10, TATA
   AUROC < 0.5, TATA-flag cleaning moves it toward 0.5) plus the pre-registered sign/CI rule (TATA AUROC CI
   upper < 0.5 and neg-excess CI lower > 0) on the NR data.
A3 WHERE THE HOMOLOGY IS. Per task and threshold: fraction of test sequences of each class whose component holds
   a train sequence, split into SAME-label homology (a train homolog with the test sequence's label -- the
   kind that can inflate a memorising model) and CROSS-label-only homology (can only deflate).
A4 ROUTE-1 PREDICTIVE CHECK WITH HOMOLOGY-CLEAN MCCs. route1_predict.py's leave-one-task-out OLS of MCC on the
   published discriminability (11 TF-motif tasks), MAE with a task-bootstrap CI (seed 42) and R^2, for the
   published MCC and each homology-clean MCC.

Outputs: results/upgrades/homology_recluster_labels_addendum.csv, homology_recluster_class_redundancy.csv,
homology_recluster_route1_loto.csv, homology_recluster_paper_table.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402
from run_routes import label_rule, preserved  # noqa: E402

R = _hc.RES
P = _hc.PREFIX
SUBMITTED = dict(_hc.PAPER_ROUTE)
SUBMITTED.update({"demo_human_or_worm": "composition", "demo_coding_vs_intergenomic_seqs": "composition",
                  "dummy_mouse_enhancers_ensembl": "composition", "human_enhancers_ensembl": "contamination (weak)"})


def rd(name):
    p = os.path.join(R, f"{P}_{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def a1_a2(ev, rt, rl, sd, comp):
    rows = []
    for _, lb in rl.iterrows():
        task, th = lb.task, lb.threshold
        aft = rt[(rt.task == task) & (rt.data == th)].iloc[0]
        bef = rt[(rt.task == task) & (rt.data == "original")].iloc[0]
        e = ev[ev.task == task]
        rn = e[e.protocol == "resplit_random"].iloc[0]
        cl = e[(e.protocol == "resplit_cluster") & (e.threshold == th)].iloc[0]
        s = sd[(sd.task == task) & (sd.threshold == th)].iloc[0]
        promoter = "promoter" in task
        lab = lambda m: label_rule(m, aft.pos_delta, aft.pos_lo, bool(aft.binary), promoter,
                                   aft.get("tata_neg_excess", np.nan), aft.get("tata_hi", np.nan))
        r = dict(task=task, threshold=th, published_v2=_hc.PAPER_ROUTE[task], published_submitted=SUBMITTED[task],
                 label_before=lb.label_before,
                 mcc_cluster_s42=cl.mcc, mcc_random_s42=rn.mcc, mcc_cluster_seedmean=s.mcc_cluster_mean,
                 mcc_random_seedmean=s.mcc_random_mean,
                 label_cluster_s42=lb.label_after, label_random_s42=lab(rn.mcc),
                 label_cluster_seedmean=lab(s.mcc_cluster_mean), label_random_seedmean=lab(s.mcc_random_mean))
        a_r = aft.copy(); a_r["mcc"], a_r["mcc_lo"], a_r["mcc_hi"] = rn.mcc, rn.mcc_lo, rn.mcc_hi
        r["preservation_random_s42"] = preserved(lb.label_before, a_r)[0]
        r["preservation_cluster_s42"] = lb.preservation
        r["preservation_purge_std"] = lb.preservation_purgeMCC
        for v in ("s42", "seedmean"):
            c_, n_ = r[f"label_cluster_{v}"], r[f"label_random_{v}"]
            r[f"relabel_{v}"] = c_ != lb.label_before
            r[f"relabel_homology_attributable_{v}"] = c_ != n_
            r[f"relabel_repooling_{v}"] = (c_ == n_) and (c_ != lb.label_before)
        # A2: submitted labels
        sub = SUBMITTED[task]
        if sub == "composition":
            c = comp[comp.task == task].set_index("unit") if comp is not None else None
            if c is not None and "original" in c.index and f"resplit_{th}" in c.index:
                b, a = c.loc["original"], c.loc[f"resplit_{th}"]
                pu = c.loc[f"test_purge_{th}"] if f"test_purge_{th}" in c.index else None
                ok = lambda x: bool(x.comp_fraction > 0 and x.comp_fraction_lo > 0)
                r.update(sub_diag_before=f"comp_fraction={b.comp_fraction:.3f} [{b.comp_fraction_lo:.3f},{b.comp_fraction_hi:.3f}]",
                         sub_diag_after=f"comp_fraction={a.comp_fraction:.3f} [{a.comp_fraction_lo:.3f},{a.comp_fraction_hi:.3f}]"
                         + (f" | purge {pu.comp_fraction:.3f} [{pu.comp_fraction_lo:.3f},{pu.comp_fraction_hi:.3f}]" if pu is not None else ""),
                         sub_preservation=("task-type label (unchanged by construction); supporting diagnostic "
                                           + ("preserved" if ok(a) and (pu is None or ok(pu)) else "NOT preserved")))
            else:
                r.update(sub_preservation="pending (run_extra.py X3 not finished)")
        elif sub == "contamination (weak)":
            crit = lambda x: bool(x.tata_neg_excess >= 0.10 and x.tata_auroc < 0.5 and bool(x.cleaning_moves_to_chance))
            st, dg = preserved("contamination", aft)
            _, dg0 = preserved("contamination", bef)
            r.update(sub_diag_before=f"{dg0}; diagnose.py R3 criterion {crit(bef)}",
                     sub_diag_after=f"{dg}; diagnose.py R3 criterion {crit(aft)}",
                     sub_preservation=st if crit(aft) else "NOT preserved (diagnose.py criterion fails)")
        else:
            r.update(sub_preservation="same label as v2 (see preservation columns)")
        r["diagnose_verdict_before"] = lb.diagnose_verdict_before
        r["diagnose_verdict_after"] = lb.diagnose_verdict_after
        rows.append(r)
    return pd.DataFrame(rows)


def a3():
    rows = []
    for task in _hc.TASKS:
        for tag in ("id80", "id50"):
            f = os.path.join(_hc.CLU_DIR, f"{task}_{tag}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            comp, is_test, y = z["comp"], z["is_test"], z["y"]
            df = pd.DataFrame({"c": comp, "t": is_test, "y": y})
            tr = df[~df.t]
            lab_sets = tr.groupby("c")["y"].agg(lambda s: frozenset(s.unique()))
            te = df[df.t].copy()
            te["train_labels"] = te["c"].map(lab_sets)
            te["any"] = te["train_labels"].notna()
            te["same"] = [bool(isinstance(s, frozenset) and yy in s) for s, yy in zip(te["train_labels"], te["y"])]
            te["cross_only"] = te["any"] & ~te["same"]
            r = dict(task=task, threshold=tag, n_test=len(te),
                     frac_test_any_train_homolog=round(float(te["any"].mean()), 5),
                     frac_test_same_label_homolog=round(float(te["same"].mean()), 5),
                     frac_test_cross_label_only=round(float(te["cross_only"].mean()), 5))
            for lab in sorted(np.unique(y)):
                g = te[te.y == lab]
                r[f"class{lab}_n_test"] = len(g)
                r[f"class{lab}_frac_any"] = round(float(g["any"].mean()), 5) if len(g) else np.nan
                r[f"class{lab}_frac_same"] = round(float(g["same"].mean()), 5) if len(g) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def a4(ev, sd):
    from sklearn.linear_model import LinearRegression
    cs = pd.read_csv(os.path.join(_hc.ROOT, "results", "cross_suite_summary.csv")).set_index("task")
    tf = [t for t in _hc.TASKS if t not in _hc.NON_TF]
    x = cs.loc[tf, "max_pos_motif_auroc"].astype(float).to_numpy()

    def pick(proto, th, col):
        out = []
        for t in tf:
            r = ev[(ev.task == t) & (ev.protocol == proto) & (ev.threshold == th)]
            out.append(float(r.iloc[0][col]) if len(r) and r.iloc[0]["status"] == "ok" else np.nan)
        return np.array(out)

    ys = {"published_MCC": cs.loc[tf, "bench_mcc"].astype(float).to_numpy(),
          "purge_id80_stdMCC": pick("test_purge", "id80", "std_mcc"),
          "purge_id50_stdMCC": pick("test_purge", "id50", "std_mcc"),
          "resplit_id80_MCC": pick("resplit_cluster", "id80", "mcc"),
          "resplit_id50_MCC": pick("resplit_cluster", "id50", "mcc"),
          "resplit_id80_seedmean_MCC": np.array([float(sd[(sd.task == t) & (sd.threshold == "id80")].iloc[0].mcc_cluster_mean) for t in tf]),
          "resplit_id50_seedmean_MCC": np.array([float(sd[(sd.task == t) & (sd.threshold == "id50")].iloc[0].mcc_cluster_mean) for t in tf]),
          # re-pooling CONTROL (A1 logic): the random re-split has no homology control, so a change it shares
          # with the cluster re-split is a re-pooling / split-design effect, not homology
          "resplit_random_MCC": np.array(
              [float(ev[(ev.task == t) & (ev.protocol == "resplit_random")].iloc[0].mcc) for t in tf]),
          "resplit_random_seedmean_MCC": np.array([float(sd[(sd.task == t) & (sd.threshold == "id80")].iloc[0].mcc_random_mean) for t in tf])}
    rows = []
    for name, y in ys.items():
        m = np.isfinite(y)
        xx, yy = x[m], y[m]; n = len(xx)
        pred = np.empty(n)
        for i in range(n):
            k = np.ones(n, bool); k[i] = False
            pred[i] = LinearRegression().fit(xx[k].reshape(-1, 1), yy[k]).predict(xx[i].reshape(1, -1))[0]
        res = yy - pred
        r2 = 1 - np.sum(res ** 2) / np.sum((yy - yy.mean()) ** 2)
        rng = np.random.RandomState(_hc.SEED); maes, r2s = [], []
        for _ in range(1000):
            i = rng.randint(0, n, n); st = np.sum((yy[i] - yy[i].mean()) ** 2)
            if st > 0:
                maes.append(np.mean(np.abs(res[i]))); r2s.append(1 - np.sum(res[i] ** 2) / st)
        rho, p = spearmanr(xx, yy)
        slope = LinearRegression().fit(xx.reshape(-1, 1), yy).coef_[0]
        rows.append(dict(y=name, n=n, spearman_rho=round(float(rho), 4), spearman_p=round(float(p), 4),
                         ols_slope=round(float(slope), 4), loto_mae=round(float(np.mean(np.abs(res))), 4),
                         loto_mae_lo=round(float(np.percentile(maes, 2.5)), 4), loto_mae_hi=round(float(np.percentile(maes, 97.5)), 4),
                         loto_r2=round(float(r2), 4), loto_r2_lo=round(float(np.percentile(r2s, 2.5)), 4),
                         loto_r2_hi=round(float(np.percentile(r2s, 97.5)), 4)))
    return pd.DataFrame(rows)


def paper_table(ev, tab, add, sd, cred):
    rows = []
    for task in _hc.TASKS:
        t = tab[tab.task == task].iloc[0]
        e = ev[ev.task == task]
        pu8 = e[(e.protocol == "test_purge") & (e.threshold == "id80")].iloc[0]
        cl8 = e[(e.protocol == "resplit_cluster") & (e.threshold == "id80")].iloc[0]
        s8 = sd[(sd.task == task) & (sd.threshold == "id80")].iloc[0]
        s5 = sd[(sd.task == task) & (sd.threshold == "id50")].iloc[0]
        a8 = add[(add.task == task) & (add.threshold == "id80")].iloc[0]
        a5 = add[(add.task == task) & (add.threshold == "id50")].iloc[0]
        c8 = cred[(cred.task == task) & (cred.threshold == "id80")].iloc[0]
        rows.append(dict(
            task=task, suite=t.suite, red_test_id80=t.red_test_id80, red_test_id50=t.red_test_id50,
            red_test_same_label_id80=c8.frac_test_same_label_homolog, within_train_red_id80=t.within_train_red_id80,
            exact_dup_test=t.exact_dup_test,
            original_mcc=t.original_mcc, original_lo=t.original_mcc_lo, original_hi=t.original_mcc_hi,
            purge_id80_status=pu8.status,
            purge_id80_mcc=pu8.mcc, purge_id80_std_mcc=pu8.std_mcc, purge_id80_delta=pu8.delta_mcc,
            purge_id80_delta_lo=pu8.delta_lo, purge_id80_delta_hi=pu8.delta_hi, purge_id80_null_mean=pu8.null_mean,
            purge_id80_homology_effect=pu8.homology_effect, purge_id80_kept=pu8.frac_test_kept,
            purge_id50_mcc=t.purge_id50_mcc, purge_id50_std_mcc=t.purge_id50_std_mcc,
            resplit_id80_mcc=cl8.mcc, resplit_id80_lo=cl8.mcc_lo, resplit_id80_hi=cl8.mcc_hi,
            resplit_random_mcc=t.resplit_random_mcc, resplit_id80_minus_random=cl8.delta_vs_random,
            resplit_id80_minus_random_lo=cl8.delta_vs_random_lo, resplit_id80_minus_random_hi=cl8.delta_vs_random_hi,
            resplit_id80_seedmean=s8.mcc_cluster_mean, random_seedmean=s8.mcc_random_mean,
            resplit_id80_minus_random_seedmean=s8.delta_mean, resplit_id80_dvr_same_sign_all_seeds=s8.same_sign_all_seeds,
            resplit_id50_mcc=t.resplit_id50_mcc, resplit_id50_seedmean=s5.mcc_cluster_mean,
            label_published_v2=a8.published_v2, label_published_submitted=a8.published_submitted,
            label_before=a8.label_before,
            label_after_purge_id80=t.label_after_purge_id80, preservation_purge_id80=a8.preservation_purge_std,
            label_after_resplit_id80_s42=a8.label_cluster_s42, preservation_resplit_id80_s42=a8.preservation_cluster_s42,
            label_after_resplit_id80_seedmean=a8.label_cluster_seedmean,
            label_random_resplit_seedmean=a8.label_random_seedmean,
            relabel_homology_attributable_id80=a8.relabel_homology_attributable_seedmean,
            relabel_repooling_id80=a8.relabel_repooling_seedmean,
            label_after_resplit_id50_seedmean=a5.label_cluster_seedmean,
            relabel_homology_attributable_id50=a5.relabel_homology_attributable_seedmean,
            submitted_label_preservation_id80=a8.sub_preservation,
            diagnose_verdict_before=a8.diagnose_verdict_before, diagnose_verdict_after_id80=a8.diagnose_verdict_after))
    return pd.DataFrame(rows)


def main():
    ev, rt, rl, sd, tab = rd("eval"), rd("routes"), rd("route_labels"), rd("seeds_summary"), rd("table")
    comp = rd("extra_composition")
    add = a1_a2(ev, rt, rl, sd, comp)
    add.to_csv(os.path.join(R, f"{P}_labels_addendum.csv"), index=False)
    cred = a3()
    cred.to_csv(os.path.join(R, f"{P}_class_redundancy.csv"), index=False)
    lo = a4(ev, sd)
    lo.to_csv(os.path.join(R, f"{P}_route1_loto.csv"), index=False)
    pt = paper_table(ev, tab, add, sd, cred)
    pt.to_csv(os.path.join(R, f"{P}_paper_table.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(add[["task", "threshold", "label_before", "label_cluster_s42", "label_random_s42", "label_cluster_seedmean",
               "label_random_seedmean", "relabel_homology_attributable_seedmean", "relabel_repooling_seedmean",
               "preservation_random_s42", "sub_preservation"]].to_string())
    print(cred[["task", "threshold", "frac_test_any_train_homolog", "frac_test_same_label_homolog",
                "frac_test_cross_label_only", "class0_frac_any", "class1_frac_any"]].to_string())
    print(lo.to_string())


if __name__ == "__main__":
    main()
