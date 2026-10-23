#!/usr/bin/env python3
"""Independent cross-check of the fix-round-1 post hoc sensitivity outputs (sensitivity_ft_v2.py)
against the stored per-sequence predictions, with sklearn / scipy and an independent sequence join.
Torch-free, about a minute. Changes no result.

  1. unit diagnostics: probability SD / one-class flags recomputed from the prediction files;
     every degenerate unit has MCC 0 and balanced accuracy 0.5 on every eval arm
     (fm_finetune_v2_metrics.csv); validation-carve copies recomputed for the nt_enhancers units.
  2. cCRE enrichment: counts re-derived by joining the annotation to the split files through a
     sequence dictionary (not by row position); Fisher p recomputed.
  3. cCRE-free re-scoring: A, B, C (seed-mean sklearn AUROC), S, share lost and the negative counts
     of every row of fm_finetune_v2_ccre_sensitivity.csv.
  4. random draws: B and C recomputed; the full 1000-draw distribution of one cell per dataset
     re-generated with sklearn from the same RandomState(42) draws.
  5. within-class R^2: an independently written 20-d composition featurizer gives the same R^2
     (|diff| < 0.02; featurizers may differ in N handling) for the released-trained fine-tuned runs.
  6. G - A levels equal -Sgc of the paper tables; paired rows are differences of levels.
  7. multiplicity: 95% CIs equal the pre-registered ones; the Bonferroni CI contains the 95% CI.
Run: bibm/lowcpu.sh <torch-free python> src/upgrades/fm_finetune/tests/test_sensitivity_ft_v2.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RES = os.path.join(ROOT, "results", "upgrades")
PRE = os.path.join(RES, "fm_finetune_v2")
SPLITS = os.path.join(ROOT, "external", "fm_finetune", "splits")
PSM = os.path.join(ROOT, "external", "psm_sensitivity")
PRED = {"ft": os.path.join(RES, "fm_finetune_v2_preds"),
        "base": os.path.join(ROOT, "external", "fm_finetune", "baseline_preds")}
NT, NTL = "nt_enhancers", "nt_enhancers_leakfree"
O, CE, RD, GC = "original", "comp_equalized", "random", "gc_match"


def test_df(ds):
    return pd.read_csv(os.path.join(SPLITS, f"{ds}__test.csv.gz"))


def preds():
    P = {}
    for fam, d in PRED.items():
        for js in sorted(glob.glob(os.path.join(d, "*.json"))):
            mt = json.load(open(js))
            df = pd.read_csv(js[:-5] + ".csv.gz")
            te = df[df.split == "test"].sort_values("idx")
            P[(mt["model"], mt["dataset"], mt["train_arm"], int(mt["seed"]))] = te.prob.to_numpy(float)
    return P


def seedmean(P, model, ds, ta, sel, y):
    ks = sorted(k for k in P if k[0] == model and k[1] == ds and k[2] == ta)
    return np.mean([roc_auc_score(y[sel], P[k][sel]) for k in ks])


def leakfree_keep():
    d = pd.read_csv(PRE + "_leakfree_dups.csv")
    keep = np.ones(400, bool); keep[d.test_idx.to_numpy()] = False
    return keep


def ann_by_seq(ds):
    """Annotation joined to the split files through a sequence dictionary (independent of row order)."""
    snap = pd.read_csv(os.path.join(PSM, f"snap_{ds}.csv.gz"))
    ann = pd.read_csv(os.path.join(PSM, f"genomic_annot_{ds}.csv.gz"))
    out = {}
    for split in ("train", "test"):
        m = (snap.split == split).to_numpy()
        seqs = snap.sequence[m].str.upper().to_numpy()
        a = ann[m].reset_index(drop=True)
        look = {}
        for i, s_ in enumerate(seqs):
            look.setdefault(s_, i)
        f = pd.read_csv(os.path.join(SPLITS, f"{ds}__{split}.csv.gz"))
        idx = np.array([look[s_] for s_ in f.sequence.str.upper()])
        out[split] = (f, a.iloc[idx].reset_index(drop=True))
    return out


def t_units(P):
    ud = pd.read_csv(PRE + "_unit_diagnostics.csv")
    met = pd.read_csv(PRE + "_metrics.csv")
    for r in ud.itertuples():
        p = P[(r.model, r.dataset, r.train_arm, int(r.seed))]
        assert abs(p.std() - r.test_prob_sd) < 1e-6 and abs((p >= 0.5).mean() - r.frac_test_p_ge_0_5) < 1e-6
        assert bool(r.one_class) == bool((p >= 0.5).all() or (p < 0.5).all())
        if r.degenerate:
            q = met[(met.model == r.model) & (met.dataset == r.dataset) & (met.train_arm == r.train_arm)
                    & (met.seed == r.seed)]
            assert (q.mcc.abs() < 1e-9).all() and (abs(q.bal_acc - 0.5) < 1e-9).all(), q[["eval_arm", "mcc", "bal_acc"]]
    tr = pd.read_csv(os.path.join(SPLITS, f"{NT}__train.csv.gz"))
    seq = tr.sequence.str.strip().str.upper().to_numpy()
    n = 0
    for r in ud[ud.dataset == NT].itertuples():
        df = pd.read_csv(os.path.join(PRED["ft"], f"{r.model}__{NT}__{r.train_arm}__s{r.seed}.csv.gz"))
        v = df[df.split == "val"].idx.to_numpy()
        fit = (tr[f"{r.train_arm}_pos"].to_numpy() >= 0); fit[v] = False
        s_fit = set(seq[fit])
        assert sum(x in s_fit for x in seq[v]) == r.n_val_verbatim_in_fit
        n += 1
    print(f"ok  unit diagnostics: {len(ud)} units; {int(ud.degenerate.sum())} degenerate (all MCC 0, balanced "
          f"accuracy 0.5 on every eval arm); {n} nt_enhancers validation-copy counts reproduced")


def t_enrichment():
    en = pd.read_csv(PRE + "_ccre_enrichment.csv")
    n = 0
    for ds in ("human_enhancers_cohn", NT):
        A = ann_by_seq(ds)
        for split in ("train", "test"):
            f, a = A[split]
            y = f.label.to_numpy()
            for fl in ("ccre_any", "ccre_els"):
                v = a[fl].to_numpy()
                ok = (a.genomic_mapped.to_numpy() == 1) & np.isfinite(v)
                negs = {arm: (y == 0) & ok & (f[f"{arm}_pos"].to_numpy() >= 0) for arm in (O, CE, RD)}
                r = en[(en.dataset == ds) & (en.split == split) & (en.feature == fl) & (en.arm == CE)].iloc[0]
                assert int(v[negs[CE]].sum()) == r.n_neg_overlap and int(negs[CE].sum()) == r.n_neg_annotated
                t = [[int(v[negs[CE]].sum()), int(negs[CE].sum() - v[negs[CE]].sum())],
                     [int(v[negs[RD]].sum()), int(negs[RD].sum() - v[negs[RD]].sum())]]
                assert abs(fisher_exact(t)[1] - r.fisher_p_vs_random) < 1e-6
                n += 1
    print(f"ok  cCRE enrichment: {n} equalized-arm rows re-derived through a sequence join (counts, Fisher p)")


def t_ccre_rescoring(P):
    cs = pd.read_csv(PRE + "_ccre_sensitivity.csv")
    lf = leakfree_keep()
    n = 0
    for ds in ("human_enhancers_cohn", NT):
        f, a = ann_by_seq(ds)["test"]
        y = f.label.to_numpy()
        for fl in ("ccre_any", "ccre_els"):
            base = (y == 1) | (a[fl].to_numpy() == 0)
            for src, kp in ((ds, base),) + (((NTL, base & lf),) if ds == NT else ()):
                for r in cs[(cs.dataset == src) & (cs["filter"] == f"{fl}_free")].itertuples():
                    sel = {k: kp & (f[f"{arm}_pos"].to_numpy() >= 0) for k, arm in (("A", O), ("B", CE), ("C", RD))}
                    v = {k: seedmean(P, r.model, ds, O, s_, y) for k, s_ in sel.items()}
                    assert int(((y == 0) & sel["B"]).sum()) == r.B_n_neg
                    for k in "ABC":
                        assert abs(v[k] - getattr(r, f"{k}_est")) < 1e-6, (src, fl, r.model, k)
                    assert abs((v["A"] - v["B"]) - r.S_est) < 1e-6
                    assert abs((v["A"] - v["B"]) / (v["A"] - 0.5) - r.frac_lost) < 1e-5
                    assert r.frac_lost_lo <= r.frac_lost <= r.frac_lost_hi and r.S_lo <= r.S_est <= r.S_hi
                    n += 1
    print(f"ok  cCRE-free re-scoring: {n} model x test-set rows (A, B, C, S, share lost, n_neg) == sklearn")


def t_randdraw(P):
    rd = pd.read_csv(PRE + "_randdraw.csv")
    lf = leakfree_keep()
    done = set()
    for r in rd.itertuples():
        src = NT if r.dataset == NTL else r.dataset
        t = test_df(src)
        keep = lf if r.dataset == NTL else np.ones(len(t), bool)
        y = t.label.to_numpy()[keep]
        ce = (t[f"{CE}_pos"].to_numpy() >= 0)[keep]; rdm = (t[f"{RD}_pos"].to_numpy() >= 0)[keep]
        Pk = {k: v[keep] for k, v in P.items() if k[0] == r.model and k[1] == src and k[2] == O}
        B = np.mean([roc_auc_score(y[ce], p[ce]) for p in Pk.values()])
        C = np.mean([roc_auc_score(y[rdm], p[rdm]) for p in Pk.values()])
        assert abs(B - r.B) < 1e-6 and abs(C - r.C) < 1e-6 and int(((y == 0) & ce).sum()) == r.k_neg
        if r.dataset not in done:                 # full distribution for the first model of each dataset
            neg = np.where(y == 0)[0]; pos = np.where(y == 1)[0]
            rng = np.random.RandomState(42)
            vals = []
            for _ in range(int(r.n_draws)):
                d = np.r_[pos, rng.choice(neg, int(r.k_neg), replace=False)]
                vals.append(np.mean([roc_auc_score(y[d], p[d]) for p in Pk.values()]))
            vals = np.array(vals)
            assert abs(vals.mean() - r.draw_mean) < 2e-6 and abs(vals.min() - r.draw_min) < 2e-6  # CSV holds 6 dp
            assert abs((1 + (vals <= r.B).sum()) / (len(vals) + 1) - r.p_one_sided_B) < 2e-6
            done.add(r.dataset)
    print(f"ok  random draws: B, C, k for {len(rd)} rows; full 1000-draw distribution re-generated for "
          f"{len(done)} cells ({', '.join(sorted(done))})")


def comp20(seqs):
    L = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(seqs), 20))
    for i, s_ in enumerate(seqs):
        c = np.array([L.get(ch, 4) for ch in s_.upper()])
        v = c[c < 4]
        m = np.bincount(v, minlength=4).astype(float); X[i, :4] = m / max(m.sum(), 1)
        a, b = c[:-1], c[1:]; ok = (a < 4) & (b < 4)
        d = np.bincount(a[ok] * 4 + b[ok], minlength=16).astype(float); X[i, 4:] = d / max(d.sum(), 1)
    return X


def r2(X, z):
    zh = np.zeros_like(z)
    for a, b in KFold(5, shuffle=True, random_state=42).split(X):
        zh[b] = LinearRegression().fit(X[a], z[a]).predict(X[b])
    return 1 - np.sum((z - zh) ** 2) / np.sum((z - z.mean()) ** 2)


def t_withinclass(P):
    wc = pd.read_csv(PRE + "_score_composition_withinclass.csv")
    n, worst = 0, 0.0
    for ds in ("human_enhancers_cohn", NT, "drosophila_enhancers_stark"):
        t = test_df(ds); y = t.label.to_numpy()
        X = comp20(t.sequence.astype(str).tolist()); X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        for r in wc[(wc.dataset == ds) & (wc.train_arm == O) & wc.model.isin(["hyenadna", "dnabert2"])].itertuples():
            p = np.clip(P[(r.model, ds, O, int(r.seed))], 1e-6, 1 - 1e-6); z = np.log(p / (1 - p))
            for col, v in (("r2_within_pos", r2(X[y == 1], z[y == 1])), ("r2_within_neg", r2(X[y == 0], z[y == 0])),
                           ("r2_label_only", r2(y[:, None].astype(float), z))):
                worst = max(worst, abs(v - getattr(r, col)))
                assert abs(v - getattr(r, col)) < 0.02, (r.model, ds, r.seed, col, v, getattr(r, col))
            n += 1
    assert (abs(wc.partial_gain - (wc.r2_label_plus_comp - wc.r2_label_only)) < 2e-6).all()  # 6-dp CSV
    print(f"ok  within-class R^2: {n} released-trained fine-tuned runs reproduced with an independent featurizer "
          f"(max |diff| {worst:.4f})")


def t_gc_and_multiplicity():
    gp = pd.read_csv(PRE + "_gc_paired.csv")
    tabs = pd.concat([pd.read_csv(PRE + "_paper_table.csv"), pd.read_csv(PRE + "_posthoc_paper_table.csv")])
    lv = gp[gp.kind == "level"]
    for r in lv.itertuples():
        t = tabs[(tabs.model == r.model) & (tabs.dataset == r.dataset)]
        assert len(t) == 1 and abs(r.G_minus_A + t.Sgc_est.iloc[0]) < 2e-6, (r.model, r.dataset)
        assert abs(r.ci_lo + t.Sgc_hi.iloc[0]) < 2e-6 and abs(r.ci_hi + t.Sgc_lo.iloc[0]) < 2e-6
    pdf = gp[gp.kind == "paired_diff"]
    for r in pdf.itertuples():
        a = lv[(lv.model == r.model) & (lv.dataset == r.dataset)].G_minus_A.iloc[0]
        b = lv[(lv.model == r.baseline) & (lv.dataset == r.dataset)].G_minus_A.iloc[0]
        assert abs(r.G_minus_A - (a - b)) < 2e-6 and r.ci_lo <= r.ci_hi
    mu = pd.read_csv(PRE + "_multiplicity.csv")
    ct = pd.read_csv(PRE + "_contrasts.csv")
    cmap = {"S": "S_shortcut_worth", "T": "T_test_imbalance_ctl", "B": "B_orig_compeq"}
    for r in mu.itertuples():
        assert r.ci_bonf_lo <= r.ci95_lo + 1e-12 and r.ci95_hi <= r.ci_bonf_hi + 1e-12
        if r.quantity in cmap:
            c = ct[(ct.model == r.model) & (ct.dataset == r.dataset) & (ct.contrast == cmap[r.quantity])].iloc[0]
            assert abs(c.ci_lo - r.ci95_lo) < 1e-6 and abs(c.ci_hi - r.ci95_hi) < 1e-6 and abs(c.estimate - r.estimate) < 1e-6
    print(f"ok  G-A: {len(lv)} levels == -Sgc of the paper tables, {len(pdf)} paired rows consistent; "
          f"multiplicity: {len(mu)} rows, 95% CIs == pre-registered, Bonferroni CIs nest them")


if __name__ == "__main__":
    P = preds()
    t_units(P)
    t_enrichment()
    t_ccre_rescoring(P)
    t_randdraw(P)
    t_withinclass(P)
    t_gc_and_multiplicity()
    print("all sensitivity cross-checks passed")
