#!/usr/bin/env python3
"""POST HOC sensitivity analyses of the end-to-end fine-tuning result (torch-free). Added in
camera-ready fix round 1 (2026-09-26), after two independent verifiers reviewed the v2 report.
Nothing here changes a pre-registered output (eval_ft_v2.py / summarize_ft_v2.py) or a post hoc
output of posthoc_ft_v2.py; every file below is new and every text built from it says "post hoc".
No decision rule is attached to any quantity here.

Honesty note. Before this script was written, the verifiers had computed scratch versions of Q2 (G-A
for four fine-tuned/frozen pairs), Q3b (cCRE-free negatives, fine-tuned cohn/nt models only), Q4
(random-draw distribution, fine-tuned models only) and Q5 (within-class R^2, fine-tuned models only),
and those numbers were seen. The definitions below are the ones they used, written down here unchanged
and extended to every model family and dataset variant; nothing was tuned. The `degenerate` flag of Q1
was defined here after the HyenaDNA cohn comp_eq units that motivated it were already known.

Q1  Unit diagnostics (fm_finetune_v2_unit_diagnostics.csv), one row per fine-tuning unit: SD, min and
    max of the test probabilities, share of test sequences with p >= 0.5, one_class (every test
    sequence on one side of 0.5), near_constant (test-probability SD < 0.01), max validation AUROC
    over epochs, own-arm test AUROC; `degenerate` = one_class AND near_constant AND max validation
    AUROC < 0.6. Validation-carve leakage: validation rows whose exact sequence also occurs among the
    unit's fitting rows (the arm's train set minus the validation carve).
Q2  GC-matched test effect, fine-tuned vs frozen / k-mer (fm_finetune_v2_gc_paired.csv). G - A (the
    paper's test_effect sign: the same weights on the GC-matched test subset minus the released test)
    for every model, and fine-tuned minus baseline differences on the same joint resamples
    (eval_ft_v2.joint_boot_auc: RandomState(42), 1000 resamples of the test sequences).
Q3  Latent-positive (ENCODE cCRE) sensitivity. ENCODE SCREEN cCRE v4 overlap of every sequence,
    read-only from psm_sensitivity's annotation cache (external/psm_sensitivity/genomic_annot_*.csv.gz;
    its row order is snap_*.csv.gz, asserted equal to the fine-tuning split files sequence by
    sequence). ELS = pELS | dELS, PLS = promoter-like.
    (a) fm_finetune_v2_ccre_enrichment.csv: per dataset x split x arm, the share of negatives (and of
        positives) overlapping any cCRE / an ELS / a PLS, with Fisher exact tests of the equalized
        negatives vs the size-matched random negatives and vs the released negatives the equalizer
        did not keep.
    (b) fm_finetune_v2_ccre_sensitivity.csv (+ long form _ccre_contrasts.csv): derived test sets that
        drop every negative overlapping any cCRE (`ccre_any_free`) or an ELS (`ccre_els_free`) from
        every eval arm (positives untouched; training unchanged), for cohn, nt_enhancers and
        nt_enhancers_leakfree and every fine-tuned and baseline model. Identical computations:
        eval_ft_v2.contrasts (A-H, S, T, C-A, U, R with joint-bootstrap CIs) and
        summarize_ft_v2.ratio_boot (share lost with CIs). Rule columns are the pre-registered rules
        RE-APPLIED post hoc to a derived test set (prefix `reapplied_`), not verdicts.
        drosophila has no human cCRE annotation (dm6) and is not included.
Q4  Random-draw reference for B (fm_finetune_v2_randdraw.csv). Per model x dataset: 1000 random draws
    (RandomState(42) per dataset, identical for every model) of as many test negatives as the
    equalized test arm holds, without replacement from all test negatives, plus all positives; the
    released-split weights' seed-mean AUROC on each draw. Reported: mean, 2.5/97.5 percentiles, min,
    max, the observed B and C, and the one-sided empirical p = (1 + #draws <= B) / 1001. This turns
    the single seeded random arm (C) into its sampling distribution.
Q5  Within-class score-composition coupling (fm_finetune_v2_score_composition_withinclass.csv and
    _summary). The pre-registered pooled CV R^2 could partly reflect the label-composition correlation;
    here each run's logit on the released test split is also regressed on composition within
    positives, within negatives, on the label alone, and on label + composition (5-fold CV OLS, KFold
    seed 42, the same standardized 20-d signature as eval_ft_v2.score_composition).
    partial_gain = R^2(label + composition) - R^2(label). Every run (fine-tuned and baselines).
Q6  Multiplicity (fm_finetune_v2_multiplicity.csv). The pre-registered rules were evaluated on 5
    fine-tuned cells without correction. S, T, B and the share lost with 95% and Bonferroni (0.05/5,
    i.e. 99%) percentile CIs from the same joint bootstrap (1000 resamples; a 99% bound rests on 5 tail
    draws). The 95% CIs must equal the pre-registered ones (asserted).

Usage: bibm/lowcpu.sh <torch-free python> src/upgrades/fm_finetune/sensitivity_ft_v2.py
       (after posthoc_ft_v2.py; before make_fig_ft_v2.py and write_interpretation_ft_v2.py)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baselines_ft_v2 as bl  # noqa: E402
import eval_ft_v2 as ev  # noqa: E402
import summarize_ft_v2 as sm  # noqa: E402

PRE = os.path.join(ev.OUT, "fm_finetune_v2")
PSM = os.path.join(ev.ROOT, "external", "psm_sensitivity")
O, CE, RD, GC = "original", "comp_equalized", "random", "gc_match"
COHN, NT, NTL, DROS = "human_enhancers_cohn", "nt_enhancers", "nt_enhancers_leakfree", "drosophila_enhancers_stark"
N_DRAW = 1000
NEAR_CONST_SD = 0.01
DEGEN_VAL = 0.6


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ shared inputs
def leakfree_keep():
    dups = pd.read_csv(PRE + "_leakfree_dups.csv")
    assert len(dups) == 100 and (dups.label == 1).all()
    keep = np.ones(400, bool)
    keep[dups.test_idx.to_numpy()] = False
    return keep


def derive(runs, mem, base_ds, keep, name):
    """Derived dataset `name`: the test sequences of `base_ds` where keep is True (training unchanged)."""
    mem[name] = dict(y=mem[base_ds]["y"][keep], mask={a: m[keep] for a, m in mem[base_ds]["mask"].items()})
    out = []
    for r in runs:
        if r["meta"]["dataset"] != base_ds:
            continue
        meta = dict(r["meta"]); meta["dataset"] = name
        out.append(dict(meta=meta, p=r["p"][keep], idx=np.arange(int(keep.sum())), y=r["y"][keep]))
    return out


def annotation(ds, split):
    """cCRE flags for the fine-tuning split file `ds`/`split`, aligned row by row (asserted)."""
    snap = pd.read_csv(os.path.join(PSM, f"snap_{ds}.csv.gz"))
    ann = pd.read_csv(os.path.join(PSM, f"genomic_annot_{ds}.csv.gz"))
    assert len(snap) == len(ann)
    sel = (snap["split"] == split).to_numpy()
    s, a = snap[sel].reset_index(drop=True), ann[sel].reset_index(drop=True)
    f = pd.read_csv(os.path.join(ev.SPLIT_DIR, f"{ds}__{split}.csv.gz"))
    assert len(s) == len(f)
    assert (s["sequence"].str.upper().to_numpy() == f["sequence"].str.upper().to_numpy()).all(), (ds, split)
    assert (s["label"].to_numpy() == f["label"].to_numpy()).all(), (ds, split)
    return f, a


# ------------------------------------------------------------------ Q1
def unit_diagnostics(ft, mem):
    rows = []
    trains = {}
    for r in ft:
        mt = r["meta"]; ds, ta = mt["dataset"], mt["train_arm"]
        if ds not in trains:
            tr = pd.read_csv(os.path.join(ev.SPLIT_DIR, f"{ds}__train.csv.gz"))
            trains[ds] = (tr, tr["sequence"].astype(str).str.strip().str.upper().to_numpy())
        tr, seq = trains[ds]
        pr = pd.read_csv(os.path.join(ev.PRED_DIR, f"{mt['model']}__{ds}__{ta}__s{mt['seed']}.csv.gz"))
        vidx = pr.loc[pr["split"] == "val", "idx"].to_numpy()
        arm = tr[f"{ta}_pos"].to_numpy() >= 0
        assert arm[vidx].all(), "validation rows must lie inside the arm's train set"
        fit = arm.copy(); fit[vidx] = False
        fitset = set(seq[fit])
        v_copy = int(np.sum([s in fitset for s in seq[vidx]]))
        p = r["p"]; y = mem[ds]["y"]; own = mem[ds]["mask"][ta]
        vh = [h["val_auroc"] for h in mt["history"]]
        one = bool((p >= 0.5).all() or (p < 0.5).all())
        nc = bool(p.std() < NEAR_CONST_SD)
        rows.append(dict(model=mt["model"], dataset=ds, train_arm=ta, seed=mt["seed"],
                         test_prob_sd=float(p.std()), test_prob_min=float(p.min()),
                         test_prob_max=float(p.max()), frac_test_p_ge_0_5=float((p >= 0.5).mean()),
                         one_class=one, near_constant=nc, max_val_auroc=float(np.max(vh)),
                         best_val_auroc=float(mt["best_val_auroc"]), best_epoch=mt["best_epoch"],
                         own_arm_test_auroc=float(ev.auroc(y[own], p[own])),
                         degenerate=bool(one and nc and np.max(vh) < DEGEN_VAL),
                         n_val=len(vidx), n_val_verbatim_in_fit=v_copy,
                         frac_val_verbatim_in_fit=v_copy / max(len(vidx), 1)))
    df = pd.DataFrame(rows).sort_values(["model", "dataset", "train_arm", "seed"])
    for ds, (tr, seq) in trains.items():
        dup = pd.Series(seq).duplicated(keep=False).to_numpy()
        log(f"[Q1] {ds}: {int(dup.sum())} of {len(seq)} training rows share their exact sequence with "
            f"another training row")
    return df


# ------------------------------------------------------------------ Q2
def gc_paired(g, mem):
    lev, prs = [], []
    boots = {}
    for (model, ds) in sorted({(k[0], k[1]) for k in g if k[2] == O}):
        bt, seeds = ev.joint_boot_auc(g, mem, model, ds, [(O, O), (O, GC)])
        if not seeds:
            continue
        pt = {pr: ev.point_auc(g, mem, model, ds, pr[0], pr[1], seeds).mean() for pr in [(O, O), (O, GC)]}
        b = bt[(O, GC)] - bt[(O, O)]
        boots[(model, ds)] = (pt[(O, GC)] - pt[(O, O)], b)
        lev.append(dict(kind="level", dataset=ds, model=model, family=sm.model_family(model),
                        baseline="", G_minus_A=pt[(O, GC)] - pt[(O, O)],
                        ci_lo=float(np.percentile(b, 2.5)), ci_hi=float(np.percentile(b, 97.5)),
                        A=pt[(O, O)], G=pt[(O, GC)], n_seeds=len(seeds)))
    for (m, ds), (v, b) in boots.items():
        if m not in sm.FT_MODELS:
            continue
        for (m2, ds2), (v2, b2) in sorted(boots.items()):
            if ds2 != ds or m2 in sm.FT_MODELS:
                continue
            d = b - b2
            prs.append(dict(kind="paired_diff", dataset=ds, model=m, family="fine-tuned", baseline=m2,
                            G_minus_A=v - v2, ci_lo=float(np.percentile(d, 2.5)),
                            ci_hi=float(np.percentile(d, 97.5)), A=np.nan, G=np.nan, n_seeds=np.nan,
                            finetuned_value=v, baseline_value=v2))
    out = pd.DataFrame(lev + prs)
    out["note"] = ("post hoc; G-A = AUROC(same weights, GC-matched test subset) - AUROC(released test); "
                   "paired_diff = fine-tuned G-A minus baseline G-A on identical resamples (negative = "
                   "the fine-tuned model loses more on the GC-matched test)")
    return out


# ------------------------------------------------------------------ Q3
def ccre_enrichment():
    rows = []
    flags = ("ccre_any", "ccre_els", "ccre_pls")
    for ds in (COHN, NT):
        for split in ("train", "test"):
            f, a = annotation(ds, split)
            y = f["label"].to_numpy()
            mapped = a["genomic_mapped"].to_numpy() == 1
            keeps = {"": np.ones(len(y), bool)}
            if ds == NT and split == "test":
                keeps["leakfree"] = leakfree_keep()
            for tag, kp in keeps.items():
                dsn = NTL if tag else ds
                for fl in flags:
                    v = a[fl].to_numpy()
                    ok = mapped & np.isfinite(v)
                    pos = kp & (y == 1) & ok
                    arm_neg = {arm: kp & (y == 0) & ok & (f[f"{arm}_pos"].to_numpy() >= 0)
                               for arm in (O, GC, CE, RD)}
                    for arm, neg in arm_neg.items():
                        n_all = int((kp & (y == 0) & (f[f"{arm}_pos"].to_numpy() >= 0)).sum())
                        r = dict(dataset=dsn, split=split, feature=fl, arm=arm, n_neg=n_all,
                                 n_neg_annotated=int(neg.sum()), n_neg_overlap=int(v[neg].sum()),
                                 rate_neg=float(v[neg].mean()) if neg.sum() else np.nan,
                                 n_pos_annotated=int(pos.sum()), rate_pos=float(v[pos].mean()) if pos.sum() else np.nan)
                        if arm == CE:
                            for other, lab in ((arm_neg[RD], "vs_random"),
                                               (arm_neg[O] & ~neg, "vs_released_not_kept")):
                                t = [[int(v[neg].sum()), int(neg.sum() - v[neg].sum())],
                                     [int(v[other].sum()), int(other.sum() - v[other].sum())]]
                                orr, p = fisher_exact(t, alternative="two-sided")
                                r[f"rate_{lab}"] = float(v[other].mean()) if other.sum() else np.nan
                                r[f"n_{lab}"] = int(other.sum())
                                r[f"odds_ratio_{lab}"] = float(orr)
                                r[f"fisher_p_{lab}"] = float(p)
                        rows.append(r)
    return pd.DataFrame(rows)


def ccre_sensitivity(runs, mem, compare):
    lf = leakfree_keep()
    derived, names = [], []
    for ds in (COHN, NT):
        f, a = annotation(ds, "test")
        y = f["label"].to_numpy()
        assert (a["genomic_mapped"].to_numpy() == 1).all() and a[["ccre_any", "ccre_els"]].notna().all().all()
        for fl in ("ccre_any", "ccre_els"):
            free = a[fl].to_numpy() == 0
            base_keep = (y == 1) | free
            variants = [(ds, base_keep)]
            if ds == NT:
                variants.append((NTL, base_keep & lf))
            for src, kp in variants:
                name = f"{src}|{fl}_free"
                derived += derive(runs, mem, ds, kp, name)
                names.append((name, src, fl))
    t0 = time.time()
    ct, vd = ev.contrasts(derived, mem)
    log(f"[Q3] contrasts on {len(names)} derived test sets, {len(derived)} runs: {time.time() - t0:.0f} s")
    g = ev.group_runs(derived)
    rows = []
    for name, src, fl in names:
        for model in sorted({k[0] for k in g if k[1] == name}):
            c = ct[(ct.model == model) & (ct.dataset == name)].set_index("contrast")
            r = dict(dataset=src, filter=f"{fl}_free", model=model, family=sm.model_family(model))
            y = mem[name]["y"]
            for key, arm in (("A", O), ("B", CE), ("C", RD)):
                r[f"{key}_n_neg"] = int(((y == 0) & mem[name]["mask"][arm]).sum())
                r[f"{key}_n_pos"] = int(((y == 1) & mem[name]["mask"][arm]).sum())
            for key, cn in (("A", "A_orig_orig"), ("B", "B_orig_compeq"), ("C", "C_orig_random"),
                            ("D", "D_compeq_compeq"), ("E", "E_random_random"), ("G", "G_orig_gcmatch"),
                            ("S", "S_shortcut_worth"), ("T", "T_test_imbalance_ctl"), ("CmA", "C_minus_A"),
                            ("U", "U_train_size_ctl"), ("R", "R_retention")):
                if cn in c.index:
                    r[f"{key}_est"] = c.loc[cn, "estimate"]; r[f"{key}_lo"] = c.loc[cn, "ci_lo"]
                    r[f"{key}_hi"] = c.loc[cn, "ci_hi"]
            r.update(sm.ratio_boot(g, mem, model, name))
            v = vd[(vd.model == model) & (vd.dataset == name)]
            for col in [k for k in v.columns if k.startswith("rule")]:
                r["reapplied_" + col] = v.iloc[0][col]
            ref = compare[(compare.model == model) & (compare.dataset == src)]
            if len(ref):
                q = ref.iloc[0]
                for k in ("A_auroc_mean", "B_auroc_mean", "C_auroc_mean", "S_est", "frac_lost",
                          "frac_lost_lo", "frac_lost_hi"):
                    r[f"unfiltered_{k}"] = q[k]
            rows.append(r)
    ct["posthoc_derived_test_set"] = True
    return pd.DataFrame(rows), ct


# ------------------------------------------------------------------ Q4
def randdraw(g, mem, datasets):
    rows = []
    for ds in datasets:
        y = mem[ds]["y"]
        neg = np.where(y == 0)[0]; pos = np.where(y == 1)[0]
        k = int(((y == 0) & mem[ds]["mask"][CE]).sum())
        rng = np.random.RandomState(ev.SEED)
        draws = [np.r_[pos, rng.choice(neg, k, replace=False)] for _ in range(N_DRAW)]
        for model in sorted({kk[0] for kk in g if kk[1] == ds and kk[2] == O}):
            P = g[(model, ds, O)]
            seeds = sorted(P)
            vals = np.array([np.mean([ev.auroc(y[d], P[s][d]) for s in seeds]) for d in draws])
            Bv = ev.point_auc(g, mem, model, ds, O, CE, seeds).mean()
            Cv = ev.point_auc(g, mem, model, ds, O, RD, seeds).mean()
            Av = ev.point_auc(g, mem, model, ds, O, O, seeds).mean()
            rows.append(dict(dataset=ds, model=model, family=sm.model_family(model), n_seeds=len(seeds),
                             n_pos=len(pos), k_neg=k, n_neg_pool=len(neg), n_draws=N_DRAW, A=Av, B=Bv, C=Cv,
                             draw_mean=vals.mean(), draw_p2_5=np.percentile(vals, 2.5),
                             draw_p97_5=np.percentile(vals, 97.5), draw_min=vals.min(), draw_max=vals.max(),
                             n_draws_le_B=int((vals <= Bv).sum()),
                             p_one_sided_B=(1 + int((vals <= Bv).sum())) / (N_DRAW + 1),
                             C_percentile_in_draws=float((vals <= Cv).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Q5
def cv_r2(X, z):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    zh = np.zeros_like(z)
    for a, b in KFold(5, shuffle=True, random_state=ev.SEED).split(X):
        zh[b] = LinearRegression().fit(X[a], z[a]).predict(X[b])
    return 1.0 - np.sum((z - zh) ** 2) / np.sum((z - z.mean()) ** 2)


def withinclass(runs, mem):
    sys.path.insert(0, os.path.join(ev.ROOT, "src", "upgrades")); sys.path.insert(0, ev.ROOT)
    import _common as C
    sig, rows = {}, []
    for r in runs:
        mt = r["meta"]; ds = mt["dataset"]; y = mem[ds]["y"]
        if ds not in sig:
            te = pd.read_csv(os.path.join(ev.SPLIT_DIR, f"{ds}__test.csv.gz"))
            X = C.comp_signature(te["sequence"].astype(str).tolist()).astype(np.float64)
            sig[ds] = (X - X.mean(0)) / (X.std(0) + 1e-12)
        X = sig[ds]
        p = np.clip(r["p"], 1e-6, 1 - 1e-6); z = np.log(p / (1.0 - p))
        if z.std() < 1e-9:
            continue
        lab = y[:, None].astype(np.float64)
        rows.append(dict(model=mt["model"], dataset=ds, train_arm=mt["train_arm"], seed=mt["seed"],
                         family=sm.model_family(mt["model"]),
                         r2_pooled=cv_r2(X, z), r2_within_pos=cv_r2(X[y == 1], z[y == 1]),
                         r2_within_neg=cv_r2(X[y == 0], z[y == 0]), r2_label_only=cv_r2(lab, z),
                         r2_label_plus_comp=cv_r2(np.c_[X, lab], z)))
    df = pd.DataFrame(rows)
    df["partial_gain"] = df.r2_label_plus_comp - df.r2_label_only
    num = ["r2_pooled", "r2_within_pos", "r2_within_neg", "r2_label_only", "r2_label_plus_comp", "partial_gain"]
    gg = df.groupby(["model", "dataset", "train_arm"])[num]
    summ = gg.mean().add_suffix("_mean").join(gg.std(ddof=1).add_suffix("_sd"))
    summ = summ.join(df.groupby(["model", "dataset", "train_arm"])["seed"].size().rename("n_seeds")).reset_index()
    summ["family"] = summ.model.map(sm.model_family)
    return df, summ


# ------------------------------------------------------------------ Q6
def multiplicity(g, mem, ct_pre):
    rows = []
    cells = sorted({(k[0], k[1]) for k in g if k[0] in sm.FT_MODELS})
    m = len(cells)
    alpha = 0.05 / m
    for model, ds in cells:
        bt, seeds = ev.joint_boot_auc(g, mem, model, ds, [(O, O), (O, CE), (O, RD)])
        pt = {pr: ev.point_auc(g, mem, model, ds, pr[0], pr[1], seeds) for pr in [(O, O), (O, CE), (O, RD)]}
        A, Bv, Cv = (pt[(O, O)].mean(), pt[(O, CE)].mean(), pt[(O, RD)].mean())
        q = {"S": (A - Bv, bt[(O, O)] - bt[(O, CE)], "S_shortcut_worth"),
             "T": (Cv - Bv, bt[(O, RD)] - bt[(O, CE)], "T_test_imbalance_ctl"),
             "B": (Bv, bt[(O, CE)], "B_orig_compeq"),
             "frac_lost": ((A - Bv) / (A - 0.5), (bt[(O, O)] - bt[(O, CE)]) / (bt[(O, O)] - 0.5), None)}
        for name, (est, b, cname) in q.items():
            lo95, hi95 = np.percentile(b, [2.5, 97.5])
            lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if cname is not None:
                ref = ct_pre[(ct_pre.model == model) & (ct_pre.dataset == ds) & (ct_pre.contrast == cname)].iloc[0]
                assert abs(ref.ci_lo - lo95) < 1e-6 and abs(ref.ci_hi - hi95) < 1e-6, (model, ds, name)
            rows.append(dict(model=model, dataset=ds, quantity=name, estimate=est, ci95_lo=lo95, ci95_hi=hi95,
                             n_cells=m, bonferroni_level=1 - alpha, ci_bonf_lo=lo, ci_bonf_hi=hi,
                             excludes_0_bonf=bool(lo > 0 or hi < 0) if name != "B" else np.nan,
                             covers_0_5_bonf=bool(lo <= 0.5 <= hi) if name == "B" else np.nan))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    mem = ev.load_membership()
    ft = ev.load_runs(ev.PRED_DIR)
    base = ev.load_runs(bl.OUT_DIR)
    assert len(ft) >= 54 and len(base) == 52, (len(ft), len(base))
    lf = derive(ft + base, mem, NT, leakfree_keep(), NTL)
    allruns = ft + base + lf
    g = ev.group_runs(allruns)
    compare = pd.read_csv(PRE + "_compare.csv")

    ud = unit_diagnostics(ft, mem)
    ud.round(6).to_csv(PRE + "_unit_diagnostics.csv", index=False)
    log(ud[ud.degenerate | ud.one_class][["model", "dataset", "train_arm", "seed", "test_prob_sd",
                                          "frac_test_p_ge_0_5", "max_val_auroc", "degenerate"]].round(4).to_string(index=False))
    log(ud.groupby(["model", "dataset"])[["frac_val_verbatim_in_fit", "best_val_auroc", "own_arm_test_auroc"]]
        .mean().round(3).to_string())

    gp = gc_paired(g, mem)
    gp.round(6).to_csv(PRE + "_gc_paired.csv", index=False)
    log(gp[["kind", "dataset", "model", "baseline", "G_minus_A", "ci_lo", "ci_hi"]].round(4).to_string(index=False))

    en = ccre_enrichment()
    en.round(6).to_csv(PRE + "_ccre_enrichment.csv", index=False)
    log(en[en.arm == CE][["dataset", "split", "feature", "rate_neg", "rate_vs_random", "rate_vs_released_not_kept",
                           "fisher_p_vs_random", "fisher_p_vs_released_not_kept", "rate_pos"]].round(4).to_string(index=False))

    cs, cct = ccre_sensitivity(ft + base, mem, compare)
    cs.round(6).to_csv(PRE + "_ccre_sensitivity.csv", index=False)
    cct.round(6).to_csv(PRE + "_ccre_contrasts.csv", index=False)
    log(cs[["dataset", "filter", "model", "B_n_neg", "A_est", "B_est", "B_lo", "B_hi", "C_est", "S_est", "S_lo",
            "frac_lost", "frac_lost_lo", "frac_lost_hi", "frac_lost_random", "unfiltered_frac_lost"]]
        .round(3).to_string(index=False))

    rd = randdraw(g, mem, (COHN, NT, NTL, DROS))
    rd.round(6).to_csv(PRE + "_randdraw.csv", index=False)
    log(rd[["dataset", "model", "k_neg", "B", "C", "draw_p2_5", "draw_p97_5", "draw_min", "p_one_sided_B",
            "C_percentile_in_draws"]].round(3).to_string(index=False))

    wc, wcs = withinclass(ft + base, mem)
    wc.round(6).to_csv(PRE + "_score_composition_withinclass.csv", index=False)
    wcs.round(6).to_csv(PRE + "_score_composition_withinclass_summary.csv", index=False)
    pre_sc = pd.read_csv(PRE + "_score_composition.csv")
    chk = wc.merge(pre_sc, on=["model", "dataset", "train_arm", "seed"])
    assert len(chk) == len(pre_sc) and (abs(chk.r2_pooled - chk.r2_cv) < 1e-5).all(), "pooled R^2 != pre-registered"
    log(wcs[wcs.train_arm == O][["model", "dataset", "r2_pooled_mean", "r2_within_pos_mean", "r2_within_neg_mean",
                                 "r2_label_only_mean", "partial_gain_mean"]].round(3).to_string(index=False))

    mu = multiplicity(ev.group_runs(ft), mem, pd.read_csv(PRE + "_contrasts.csv"))
    mu.round(6).to_csv(PRE + "_multiplicity.csv", index=False)
    log(mu.round(3).to_string(index=False))
    log(f"[sensitivity_ft_v2] done in {time.time() - t0:.0f} s; wrote {PRE}_{{unit_diagnostics,gc_paired,"
        "ccre_enrichment,ccre_sensitivity,ccre_contrasts,randdraw,score_composition_withinclass{,_summary},"
        "multiplicity}.csv")


if __name__ == "__main__":
    main()
