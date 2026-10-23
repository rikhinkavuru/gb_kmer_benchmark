#!/usr/bin/env python3
"""POST HOC additions to the pre-registered GUE analysis (run_gue.py). Nothing here changes a
pre-registered readout or a route label; every quantity below is labelled post hoc in the outputs.

Written 2026-09-27 after the first four cards (prom_core_all/notata/tata, prom_300_all) and the
test-suite run, and BEFORE any of the analyses below was run. Motivation, stated before running:
 (a) In the pre-registered equalized-split arm the equalizer keeps only 2.6-6.9 % of the core-promoter
     negatives, so the refit model sees ~97 % positives and predicts one class: the equalized MCC is
     0.000 by class imbalance, whether or not a composition shortcut was removed. This is the same size
     artifact found for the paper's cohn set (bibm/HANDOFF.md section 4), for which the camera-ready
     reads equalization as an AUROC decomposition against size-matched random removal
     (src/upgrades/psm_sensitivity/run_posthoc.py, part P7). PH1 applies that decomposition to GUE.
 (b) In the sanity test, mouse_tf_3 gave a NEGATIVE composition fraction (-1.13): the composition-only
     model is anti-predictive. GUE's mouse-TF negatives are dinucleotide shuffles of the positives,
     i.e. exact composition twins; if a test positive's shuffle twin sits in train with label 0 the
     composition-only model learns the opposite label for that composition. PH2 measures this.

PH1  EQUALIZATION DECOMPOSITION (binary tasks). Same validation-selected k (from the card), same
     LightGBM (models.LGBM_PARAMS). Arms, each refit on its own train split and scored on its own test
     split (within design):
       A0  the pre-registered learnability model on the full test split;
       Am  the pre-registered composition-equalized arm (run_composition_clean propensity matching,
           RandomState(42)), recomputed bit-identically -- its AUROC is asserted equal to the card's
           equalized_delta.auroc_after;
       Ar  10 size-matched RANDOM arms: in each split keep the same NUMBER of negatives as Am, drawn
           uniformly without replacement (RandomState(42 + r), r = 0..9); positives untouched.
     The same arms are also fit with the composition-only LightGBM (20-d signature).
     Quantities (psm_sensitivity P7): size effect A0 - mean(Ar); matched-vs-random effect
     mean(Ar) - Am; relative gap (mean(Ar) - Am) / (mean(Ar) - 0.5) = share of the random arm's
     above-chance AUROC lost when the removed negatives are chosen by composition.
     Two-level paired bootstrap: level 1 resamples the task's test sequences (B = 1000,
     RandomState(42)); every arm is scored on its own members of the resample; level 2 resamples the
     10 random arms with replacement (RandomState(43)). 95 % percentile CIs.
     Reading rule (fixed now). R1: the composition shortcut is called SUPPORTED on a task if the
     matched-vs-random CI excludes 0 with a positive sign AND the composition-only AUROC of the matched
     arm is closer to 0.5 than that of the random arms (i.e. equalization did remove composition
     signal). R2: otherwise 'not distinguishable'. Paper reference values (psm_sensitivity_decomposition_ci.csv,
     within / kmer / matched seed 42): cohn rel_gap 0.545 [0.398, 0.683]; drosophila 0.317 [0.156, 0.470].
     Caveat stated in advance: where the equalizer keeps < ~10 % of negatives the matched arm is a small,
     extreme-propensity subset; the rel_gap is then read as 'how far the composition-matched negatives
     are from the typical negatives', not as a clean effect size.

PH2  COMPOSITION-TWIN AUDIT (all 28 tasks). A 'twin' of sequence s is a DIFFERENT sequence of equal
     length with an identical 16-dinucleotide COUNT vector (every dinucleotide-preserving shuffle is a
     twin; unrelated genomic windows essentially never are). Per task: fraction of test sequences with
     a twin in train, split by same-label and cross-label twins; fraction of all label-1 sequences with
     a label-0 twin anywhere in the released task (confirms shuffle construction). Binary tasks: the
     pre-registered composition-only model (refit identically) is scored on the test subsets WITH and
     WITHOUT a cross-label twin in train. Descriptive; no rule.

Output: results/upgrades/gue_posthoc/<task>.json (checkpoint), assembled by summarize_gue.py into
gue_equalization_decomposition.csv and gue_twins.csv. CPU only, LightGBM on GUE_THREADS threads.
Run: GUE_THREADS=3 bibm/lowcpu.sh python src/upgrades/gue_suite/run_gue_posthoc.py [--tasks a,b]
"""
import os

N_THREADS = int(os.environ.get("GUE_THREADS", "3"))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["GUE_THREADS"] = str(N_THREADS)
import run_gue as R  # noqa: E402  (paths, thread limits, helpers)
import gue_data as G  # noqa: E402
import run_composition_clean as CC  # noqa: E402

C, DG = R.C, R.DG
SEED, BOOT, N_RAND = 42, 1000, 10
OUT_DIR = os.path.join(C.RESULTS_DIR, "gue_posthoc")


# ----------------------------------------------------------------------------- PH1
def arm_indices(y, keep_neg_local):
    """Original indices of an arm's split: all positives + the kept negatives (sorted), exactly the
    order CC.negatives_kept_split uses."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    return np.sort(np.concatenate([pos, neg[np.asarray(keep_neg_local, dtype=np.int64)]]))


def fit_p1(Xtr, ytr, Xte):
    cls, pr = R.fit_proba(Xtr, ytr, Xte, SEED)
    return pr[:, int(np.where(cls == 1)[0][0])]


def auc(y, p, w=None):
    if len(np.unique(y[w > 0] if w is not None else y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p, sample_weight=w))


def ph1(task, card, d):
    tr, ytr, te, yte = d["train_seqs"], d["y_train"], d["test_seqs"], d["y_test"]
    k = card["k"]
    rng = np.random.RandomState(SEED)
    comp_tr = C.comp_signature(tr, ks=(1, 2)); comp_te = C.comp_signature(te, ks=(1, 2))
    prop = CC.fit_propensity(np.vstack([comp_tr, comp_te]), np.concatenate([ytr, yte]), SEED)
    p_tr = prop.predict_proba(comp_tr)[:, 1]; p_te = prop.predict_proba(comp_te)[:, 1]
    keep_tr = CC.match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], DG.PROP_BINS, rng)
    keep_te = CC.match_proportional(p_te[yte == 1], p_te[yte == 0], DG.PROP_BINS, rng)
    if len(keep_tr) < 10 or len(keep_te) < 10:
        return dict(feasible=False, reason="equalization infeasible (pre-registered arm)")
    n_neg_tr, n_neg_te = int((ytr == 0).sum()), int((yte == 0).sum())
    arms = {"matched": (arm_indices(ytr, keep_tr), arm_indices(yte, keep_te))}
    for r in range(N_RAND):
        rr = np.random.RandomState(SEED + r)
        ktr = np.sort(rr.choice(n_neg_tr, len(keep_tr), replace=False))
        kte = np.sort(rr.choice(n_neg_te, len(keep_te), replace=False))
        arms[f"random{r}"] = (arm_indices(ytr, ktr), arm_indices(yte, kte))
    # sanity: matched arm == CC.negatives_kept_split
    ce_te, _ = CC.negatives_kept_split(te, yte, keep_te)
    assert ce_te == [te[i] for i in arms["matched"][1]]

    Xtr_full = R.kspec(tr, k); Xte_full = R.kspec(te, k)
    preds = {}   # arm -> (test idx, p_kmer, p_comp)
    p0 = fit_p1(Xtr_full, ytr, Xte_full)
    p0c = fit_p1(comp_tr, ytr, comp_te)
    preds["orig"] = (np.arange(len(yte)), p0, p0c)
    for name, (itr, ite) in arms.items():
        pk = fit_p1(Xtr_full[itr], ytr[itr], Xte_full[ite])
        pc = fit_p1(comp_tr[itr], ytr[itr], comp_te[ite])
        preds[name] = (ite, pk, pc)
    del Xtr_full, Xte_full
    am_pt = auc(yte[preds["matched"][0]], preds["matched"][1])
    assert abs(am_pt - card["equalized_delta"]["auroc_after"]) < 1e-5, (am_pt, card["equalized_delta"]["auroc_after"])

    # level-1 resample weights (counts of each original test index), shared by every arm
    n = len(yte); rng1 = np.random.RandomState(SEED)
    W = np.zeros((BOOT, n), dtype=np.float64)
    for b in range(BOOT):
        W[b] = np.bincount(rng1.randint(0, n, n), minlength=n)
    rnames = [f"random{r}" for r in range(N_RAND)]
    rng2 = np.random.RandomState(SEED + 1)
    L2 = rng2.randint(0, N_RAND, size=(BOOT, N_RAND))
    out = dict(feasible=True, k=k, n_rand=N_RAND, B=BOOT,
               retained_neg_frac_train=round(len(keep_tr) / n_neg_tr, 4),
               retained_neg_frac_test=round(len(keep_te) / n_neg_te, 4),
               n_test_neg_matched=int(len(keep_te)), n_test=int(n))
    for model, j in (("kmer", 1), ("comp", 2)):
        pts = {a: auc(yte[preds[a][0]], preds[a][j]) for a in preds}
        draws = {}
        for a in preds:
            ite, p = preds[a][0], preds[a][j]
            draws[a] = np.array([auc(yte[ite], p, W[b, ite]) for b in range(BOOT)])
        A0, Am = pts["orig"], pts["matched"]
        Ar = float(np.mean([pts[a] for a in rnames]))
        Rd = np.stack([draws[a] for a in rnames], axis=1)            # (B, 10)
        Ar_d = np.take_along_axis(Rd, L2, axis=1).mean(axis=1)
        A0_d, Am_d = draws["orig"], draws["matched"]
        q = dict(size_effect=(A0 - Ar, A0_d - Ar_d), matched_vs_random=(Ar - Am, Ar_d - Am_d),
                 total_drop=(A0 - Am, A0_d - Am_d),
                 rel_gap=((Ar - Am) / (Ar - 0.5), (Ar_d - Am_d) / (Ar_d - 0.5)))
        blk = dict(A0=round(A0, 6), Ar_mean=round(Ar, 6), Am=round(Am, 6),
                   Ar_sd_over_arms=round(float(np.std([pts[a] for a in rnames], ddof=1)), 6),
                   A0_ci=[round(float(np.nanpercentile(A0_d, x)), 6) for x in (2.5, 97.5)],
                   Ar_ci=[round(float(np.nanpercentile(Ar_d, x)), 6) for x in (2.5, 97.5)],
                   Am_ci=[round(float(np.nanpercentile(Am_d, x)), 6) for x in (2.5, 97.5)])
        for key, (pt, dr) in q.items():
            blk[key] = round(float(pt), 6)
            blk[key + "_ci"] = [round(float(np.nanpercentile(dr, 2.5)), 6), round(float(np.nanpercentile(dr, 97.5)), 6)]
        blk["n_boot_nan"] = int(np.isnan(q["matched_vs_random"][1]).sum())
        out[model] = blk
    km, cm = out["kmer"], out["comp"]
    lo = km["matched_vs_random_ci"][0]
    comp_removed = abs(cm["Am"] - 0.5) < abs(cm["Ar_mean"] - 0.5)
    out["R1_supported"] = bool(lo > 0 and comp_removed)
    out["reading"] = ("SUPPORTED: composition-chosen removal costs more AUROC than random removal of the "
                      "same number of negatives, and equalization removed composition signal"
                      if out["R1_supported"] else
                      "NOT DISTINGUISHABLE" + ("" if comp_removed else
                                               " (equalization did not move composition-only AUROC toward 0.5)"))
    return out


# ----------------------------------------------------------------------------- PH2
def dinuc_keys(seqs):
    """(len, 16-dinucleotide count vector) bytes key per sequence."""
    codes = C.motif_match.encode_sequences(seqs).astype(np.int16)     # (n, S), 4 = pad / non-ACGT
    a, b = codes[:, :-1], codes[:, 1:]
    ok = (a < 4) & (b < 4)
    idx = np.where(ok, a * 4 + b, 16)
    keys = []
    for i, s in enumerate(seqs):
        cnt = np.bincount(idx[i], minlength=17)[:16].astype(np.int32)
        keys.append(len(s).to_bytes(4, "little") + cnt.tobytes())
    return keys


def ph2(task, card, d, binary):
    tr, ytr, te, yte = d["train_seqs"], d["y_train"], d["test_seqs"], d["y_test"]
    dv, ydv = d["dev_seqs"], d["y_dev"]
    ktr = []
    for i in range(0, len(tr), 20000):
        ktr += dinuc_keys(tr[i:i + 20000])
    kte = dinuc_keys(te)
    idx = {}
    for key, s, y in zip(ktr, tr, ytr):
        idx.setdefault(key, {}).setdefault(int(y), set()).add(hash(s))
    same = np.zeros(len(te), bool); cross = np.zeros(len(te), bool)
    for i, (key, s, y) in enumerate(zip(kte, te, yte)):
        ent = idx.get(key)
        if not ent:
            continue
        hs = hash(s)
        for lab, hset in ent.items():
            if len(hset - {hs}) > 0:
                if lab == int(y):
                    same[i] = True
                else:
                    cross[i] = True
    out = dict(n_test=int(len(te)), frac_test_twin_in_train=round(float((same | cross).mean()), 6),
               frac_test_crosslabel_twin_in_train=round(float(cross.mean()), 6),
               frac_test_samelabel_twin_in_train=round(float(same.mean()), 6))
    if binary:
        # construction check over the whole released task
        allk = ktr + dinuc_keys(dv) + kte
        ally = np.concatenate([ytr, ydv, yte]); alls = tr + dv + te
        neg = {}
        for key, s, y in zip(allk, alls, ally):
            if y == 0:
                neg.setdefault(key, set()).add(hash(s))
        pos_i = np.where(ally == 1)[0]
        has = [len(neg.get(allk[i], set()) - {hash(alls[i])}) > 0 for i in pos_i]
        out["frac_label1_with_label0_twin_anywhere"] = round(float(np.mean(has)), 6)
        # pre-registered composition-only model, refit identically, scored on twin subsets
        p = fit_p1(C.comp_signature(tr, ks=(1, 2)), ytr, C.comp_signature(te, ks=(1, 2)))
        full = auc(yte, p)
        assert abs(full - card["composition_fraction"]["comp_only_auroc"]) < 1e-3, (full, card["composition_fraction"])
        out["comp_only_auroc_all"] = round(full, 6)
        for tag, m in (("with_crosslabel_twin", cross), ("without_crosslabel_twin", ~cross)):
            out[f"n_test_{tag}"] = int(m.sum())
            out[f"comp_only_auroc_{tag}"] = (round(auc(yte[m], p[m]), 6) if m.sum() > 1 else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(G.ORDER))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for task in [t for t in args.tasks.split(",") if t]:
        cpath = os.path.join(R.CARD_DIR, f"{task}.json")
        out = os.path.join(OUT_DIR, f"{task}.json")
        if not os.path.exists(cpath):
            print(f"[{task}] no card yet, skipping", flush=True); continue
        if os.path.exists(out) and not args.force:
            print(f"[{task}] post-hoc exists, skipping", flush=True); continue
        card = json.load(open(cpath))
        d = G.load_gue(task)
        t0 = time.time()
        res = dict(task=task, posthoc=True, written="2026-09-27", n_jobs=N_THREADS)
        res["PH2_twins"] = ph2(task, card, d, card["binary"])
        tw = res["PH2_twins"]
        print(f"[{task}] PH2 twin-in-train {tw['frac_test_twin_in_train']:.3f} (cross-label "
              f"{tw['frac_test_crosslabel_twin_in_train']:.3f})" +
              (f"; label1 with label0 twin anywhere {tw['frac_label1_with_label0_twin_anywhere']:.3f}; "
               f"comp-only AUROC all {tw['comp_only_auroc_all']:.3f} | with x-twin "
               f"{tw['comp_only_auroc_with_crosslabel_twin']} | without {tw['comp_only_auroc_without_crosslabel_twin']}"
               if card["binary"] else ""), flush=True)
        if card["binary"] and card.get("equalized_delta", {}).get("feasible"):
            res["PH1_equalization"] = ph1(task, card, d)
            e = res["PH1_equalization"]
            print(f"[{task}] PH1 kmer A0 {e['kmer']['A0']:.3f} Ar {e['kmer']['Ar_mean']:.3f} Am {e['kmer']['Am']:.3f} "
                  f"mvr {e['kmer']['matched_vs_random']:+.3f} {e['kmer']['matched_vs_random_ci']} rel_gap "
                  f"{e['kmer']['rel_gap']:.3f} | comp Ar {e['comp']['Ar_mean']:.3f} Am {e['comp']['Am']:.3f} | "
                  f"{e['reading']}", flush=True)
        elif card["binary"]:
            res["PH1_equalization"] = dict(feasible=False, reason="pre-registered equalized arm infeasible")
        res["total_s"] = round(time.time() - t0, 1)
        tmp = out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=2)
        os.replace(tmp, out)
        print(f"[{task}] wrote {out} ({res['total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
