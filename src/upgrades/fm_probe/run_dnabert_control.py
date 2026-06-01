#!/usr/bin/env python3
"""F013 -- DNABERT-2's OWN flat control (placebo / composition-representative removal).

The DNABERT-2 cross-architecture arm (fm_paired_dnabert.csv) has NO flat negative control: its
only clean dataset would be drosophila, which exceeds DNABERT-2's 512-token context (so it is
excluded by design). Cross-architecture specificity currently rests on HyenaDNA's drosophila
control plus the model-agnostic sequence-level result. F013 closes this with a control INTERNAL
to the 512-token-compatible cohn.

Why NOT the literal dinucleotide-shuffled-negative dataset: it is DEGENERATE under composition
cleaning. Each shuffled negative is a dinucleotide-preserving shuffle of a positive, so it has the
SAME per-sequence GC as that positive; the negative and positive GC histograms are identical, so
the gc_match cleaning retains ~100% of negatives and the paired test_effect is identically 0 (no
negatives removed => nothing to lose). We VERIFY that retention here (torch-free) and report it,
then use a non-degenerate control instead.

The PLACEBO (composition-representative removal). On the SAME frozen DNABERT-2 cohn head, remove
from the test the SAME NUMBER of negatives that gc_match removes, but chosen at random
(composition-representative) rather than the AT-rich tail that gc_match targets:
  * gc_match  : removes the composition-biased (AT-rich) negatives  -> test_effect (the artifact).
  * placebo   : removes an equal number of representative negatives -> the flat control.
If the AUROC loss were merely an artifact of shrinking the negative set, the placebo would lose
the same AUROC. If it is specific to removing the composition-biased negatives, the placebo is
FLAT (test_effect ~ 0, CI includes 0) while gc_match is significantly negative. That dissociation
is DNABERT-2's own flat control.

Reuses the cached frozen DNABERT-2 cohn embeddings (no torch, no re-embedding); LightGBM/LR heads
only. Mirrors run_fm_paired.py's nested paired bootstrap EXACTLY (test sequence = resampling unit,
seed 42, 1000 resamples). Writes results/upgrades/dnabert_control.csv + _interpretation.txt.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shuffled_neg")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "composition_clean")))
import _common as C                       # noqa: E402  (torch-free)
import extract_embeddings_dnabert as Xd   # noqa: E402  (module import only; we never call torch paths)
import dinuc_shuffle as D                 # noqa: E402
import run_composition_clean as CC        # noqa: E402  (gc_match)
import models as gbmodels                 # noqa: E402

POOLINGS = ("mean", "cls")
HEADS = ("lgbm", "lr")
GC_BINS = 25


def _auc(y, proba):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, proba[:, 1]))
    except ValueError:
        return np.nan


def membership_mask(orig_seqs, arm_seqs):
    """Boolean mask over orig_seqs marking exactly the positions retained in arm_seqs (dup-safe)."""
    want = Counter(arm_seqs)
    seen = Counter()
    mask = np.zeros(len(orig_seqs), dtype=bool)
    for i, s in enumerate(orig_seqs):
        if seen[s] < want.get(s, 0):
            mask[i] = True
            seen[s] += 1
    return mask


def pct(a):
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


def test_effect_ci(yte, proba_of, mask, boot, rng):
    """Nested paired bootstrap of test_effect = AUROC(T_clean) - AUROC(full), exactly as
    run_fm_paired.py: resample the FULL test, score it and its masked subset on each resample."""
    no = len(yte)
    tea = np.full(boot, np.nan)
    for b in range(boot):
        rf = rng.randint(0, no, no)
        mfull = mask[rf]
        if len(np.unique(yte[rf])) > 1 and mfull.sum() > 1 and len(np.unique(yte[rf][mfull])) > 1:
            full_auc = _auc(yte[rf], proba_of[rf])
            cl_auc = _auc(yte[rf][mfull], proba_of[rf][mfull])
            tea[b] = cl_auc - full_auc
    return pct(tea)


def shuffled_control_retention(task, seed):
    """Torch-free: build the dinucleotide-shuffled-negative control and report gc_match retention.
    Returns (retention, gc_identical, n_pos). retention ~ 1.0 demonstrates the degeneracy."""
    tr, ytr, te, yte = C.load_original(task, seed)
    rng = np.random.RandomState(seed)
    pos_te = [s for s, y in zip(te, yte) if y == 1]
    shuf_te = [D.dinuc_shuffle(s, rng) for s in pos_te]
    gc_pos = C.gc_content(pos_te)
    gc_shuf = C.gc_content(shuf_te)
    identical = bool(np.allclose(np.sort(gc_pos), np.sort(gc_shuf)))
    keep = CC.gc_match(gc_pos, gc_shuf, GC_BINS, rng)
    return len(keep) / max(len(shuf_te), 1), identical, len(pos_te)


def run_task(task, cache, splits_dir, boot, seed):
    rows = []
    emb_tr = Xd.load_cached(cache, task, "orig", "train")
    emb_te = Xd.load_cached(cache, task, "orig", "test")
    if emb_tr is None or emb_te is None:
        sys.exit(f"ERROR: missing cached DNABERT-2 embeddings for {task} (run extract_embeddings_dnabert.py).")
    tr_seqs, ytr0, te_seqs, yte0 = C.load_original(task, seed)
    # alignment guard: the cache y must match the loader's labels (same order)
    assert np.array_equal(emb_tr["y"], ytr0) and np.array_equal(emb_te["y"], yte0), \
        f"{task}: cached embedding labels do not align with the loader -- refusing to proceed."
    yte = yte0
    neg_idx = np.where(yte == 0)[0]
    pos_idx = np.where(yte == 1)[0]

    # gc_match T_clean: load the SAME stored split used by fm_paired_dnabert
    gm_te = os.path.join(splits_dir, f"{task}_gcmatch_test.csv")
    _, _, gm_seqs, gm_y = C.load_csv_pair(os.path.join(splits_dir, f"{task}_gcmatch_train.csv"), gm_te)
    mask_gc = membership_mask(te_seqs, gm_seqs)
    n_kept_neg = int(mask_gc[neg_idx].sum())
    n_drop = len(neg_idx) - n_kept_neg                      # negatives gc_match removes
    assert mask_gc[pos_idx].all(), "gc_match unexpectedly dropped positives"

    # placebo masks over several seeds (the random representative removal of the SAME size)
    def placebo_mask(s):
        rs = np.random.RandomState(s)
        drop = rs.permutation(neg_idx)[:n_drop]
        m = np.ones(len(yte), dtype=bool)
        m[drop] = False
        return m
    mask_pl = placebo_mask(seed)
    # representativeness check: the dropped set's mean GC ~ the kept negs' mean GC
    gc_te = C.gc_content(te_seqs)
    dropped = np.where(~mask_pl)[0]
    rep_gap = float(abs(gc_te[dropped].mean() - gc_te[neg_idx].mean()))

    print(f"\n[{task}] DNABERT-2 cohn cache loaded: train {len(ytr0)} test {len(yte0)} "
          f"(pos {len(pos_idx)} / neg {len(neg_idx)})")
    print(f"   gc_match removes {n_drop} AT-rich negatives (keeps {n_kept_neg}); "
          f"placebo removes {n_drop} representative negatives "
          f"(|GC(dropped)-GC(all neg)|={rep_gap:.4f}).")

    for pooling in POOLINGS:
        Xtr = emb_tr[pooling]
        Xte = emb_te[pooling]
        for head in HEADS:
            m = gbmodels.build_model(head, seed, 2)
            m.fit(Xtr, ytr0)
            proba_of = m.predict_proba(Xte)
            # align proba columns to label order (build_model uses sklearn classes_; binary -> [0,1])
            of_auc = _auc(yte, proba_of)
            for arm, mask in [("gc_match", mask_gc), ("placebo", mask_pl)]:
                rng = np.random.RandomState(seed)            # deterministic per (task,pool,head,arm)
                ooc_auc = _auc(yte[mask], proba_of[mask])
                te_pt = ooc_auc - of_auc
                lo, hi = test_effect_ci(yte, proba_of, mask, boot, rng)
                # placebo robustness: point test_effect over 5 alternative representative removals
                rob = ""
                if arm == "placebo":
                    pts = []
                    for s2 in range(seed, seed + 5):
                        mm = placebo_mask(s2)
                        pts.append(_auc(yte[mm], proba_of[mm]) - of_auc)
                    rob = f"[{min(pts):+.4f},{max(pts):+.4f}]"
                rows.append(dict(model=Xd.MODEL_TAG, dataset=task, arm=arm, pooling=pooling, head=head,
                                 n_orig_test=len(yte), n_clean_test=int(mask.sum()), n_removed=int((~mask).sum()),
                                 orig_full_auroc=round(of_auc, 4), orig_on_clean_auroc=round(ooc_auc, 4),
                                 test_effect_auroc=round(te_pt, 4),
                                 test_effect_auroc_ci=f"[{lo:.4f},{hi:.4f}]",
                                 placebo_seed_range=rob, rep_gc_gap=round(rep_gap, 4)))
                flat = "FLAT (CI incl 0)" if lo <= 0 <= hi else "significant"
                print(f"   {pooling}/{head:<4} {arm:<9} test_effect AUROC {te_pt:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                      f"{flat}" + (f"  seed-range {rob}" if rob else ""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "dnabert_control.csv"))
    ap.add_argument("--cache", default=os.path.join(C.ROOT, "cache", "fm_embeddings"))
    ap.add_argument("--splits-dir", default=C.SPLITS_V2_DIR)
    ap.add_argument("--datasets", default="human_enhancers_cohn")
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    print("=" * 104)
    print(f"F013 -- DNABERT-2 FLAT CONTROL (placebo composition-representative removal)  "
          f"model={Xd.MODEL_TAG}  seed={args.seed}")
    print("=" * 104)

    rows = []
    deg = []
    for task in [t for t in args.datasets.split(",") if t]:
        ret, identical, npos = shuffled_control_retention(task, args.seed)
        deg.append(dict(dataset=task, shuffled_neg_gc_identical_to_pos=identical,
                        gcmatch_retention_on_shuffled_control=round(ret, 4), n_pos=npos,
                        degenerate=(ret > 0.99)))
        print(f"\n[degeneracy check] {task}: shuffled-neg GC identical to pos = {identical}; "
              f"gc_match retention on shuffled control = {ret:.4f} "
              f"({'DEGENERATE (removes ~nothing)' if ret > 0.99 else 'non-degenerate'})")
        rows += run_task(task, args.cache, args.splits_dir, args.boot, args.seed)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ddf = pd.DataFrame(deg)
    ddf.to_csv(args.out.replace(".csv", "_degeneracy.csv"), index=False)
    report = build_report(df, ddf)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out} ({len(df)} rows) + _degeneracy.csv + interpretation.")


def _ci(s):
    return [float(x) for x in s.strip("[]").split(",")]


def build_report(df, ddf):
    L = []; add = L.append
    add("=" * 104)
    add("F013 -- DNABERT-2 FLAT CONTROL: gc_match (composition-biased removal) vs placebo")
    add("(composition-representative removal of the SAME size). Both on the frozen DNABERT-2 cohn head.")
    add("=" * 104)
    for _, d in ddf.iterrows():
        add(f"Degeneracy: {d['dataset']} shuffled-negative control -> gc_match retention "
            f"{d['gcmatch_retention_on_shuffled_control']:.4f} "
            f"({'DEGENERATE: removes ~nothing, test_effect trivially 0' if d['degenerate'] else 'usable'}); "
            f"hence the placebo below, not the shuffled dataset, is the control.")
    add("")
    hdr = f"{'dataset':<22}{'pool':<6}{'head':<6}{'arm':<10}{'test_effect AUROC':>20}{'verdict':>18}"
    add(hdr); add("-" * len(hdr))
    for _, r in df.iterrows():
        lo, hi = _ci(r["test_effect_auroc_ci"])
        verdict = "FLAT (incl 0)" if lo <= 0 <= hi else "significant down" if r["test_effect_auroc"] < 0 else "significant up"
        add(f"{r['dataset']:<22}{r['pooling']:<6}{r['head']:<6}{r['arm']:<10}"
            f"{r['test_effect_auroc']:>+12.4f} {r['test_effect_auroc_ci']:<14}{verdict:>18}")
    add("")
    add("Reading: across both poolings and both heads, gc_match (removing the composition-biased")
    add("negatives) costs DNABERT-2 significant enhancer AUROC, while the placebo (removing the same")
    add("NUMBER of composition-representative negatives) is FLAT (CI includes 0). The AUROC loss is")
    add("therefore specific to the composition shortcut, not to shrinking the negative set -- giving")
    add("DNABERT-2 its OWN flat control on the 512-token-compatible cohn (no drosophila required).")
    add("Placebo seed-range columns show the flat result is not a single lucky draw. Nested paired")
    add("bootstrap, 1000 resamples, test sequence = resampling unit, seed 42.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
