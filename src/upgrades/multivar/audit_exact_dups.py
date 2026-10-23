#!/usr/bin/env python3
"""POST-HOC (verification round, 2026-09-24): verbatim train/test duplicates + class-identity checks.

Why (results-motivated, labelled post hoc): the 8-mer-Jaccard homology covariate
(multivar_homology.csv) puts human_enhancers_ensembl at 38.5% and nt_enhancers_types at 25% of test
sequences with a >=0.7 near-duplicate in train, although the paper's "homology-clean" 9-task subset
keeps both.  An independent check found the human_enhancers_ensembl hits to be mostly VERBATIM copies.
This script measures that directly on the FULL test split of every task (no subsampling):

  per task: fraction of test sequences whose exact string occurs in train (forward strand), or whose
  reverse complement does, the fraction of those duplicates whose train copies all carry the SAME label
  as the test copy, and the per-class duplicate fraction.  Per-test flags are saved to
  external/multivar/dups/<task>.npz for dedup_eval.py.

It also writes the class-identity evidence that the post-hoc orientation analysis (P4/P8 in
run_multivar.py) relies on, from the sequences themselves:
  * nt_enhancers_types vs nt_enhancers: identical train/test sequence lists (same order), and whether
    nt_enhancers_types label 0 marks exactly the nt_enhancers label-0 (non-enhancer) sequences;
  * nt_splice_sites_all: per class, the most frequent position of the GT and AG dinucleotides and
    their frequency (acceptor = AG at the junction, donor = GT, background = neither).

No thresholds, no model fitting.  Outputs: results/upgrades/multivar_posthoc_exact_dup.csv,
results/upgrades/multivar_posthoc_class_identity.csv.  Every load asserts Table-1 counts (_mv).
Run: bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/multivar/audit_exact_dups.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mv  # noqa: E402

OUT_DIR = os.path.join(_mv.EXT_DIR, "dups")
RD = _mv.RESULTS_DIR
_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(_COMP)[::-1]


def exact_dup_flags(tr, ytr, te, yte):
    """Per test sequence: (fwd, rc, any, concordant).  concordant = a duplicate whose train copies
    (either strand) all carry the test label."""
    labs = {}
    for s, y in zip(tr, ytr):
        labs.setdefault(s, set()).add(int(y))
    n = len(te)
    fwd, rc, conc = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for i, (s, y) in enumerate(zip(te, yte)):
        a = labs.get(s)
        r = revcomp(s)
        b = labs.get(r) if r != s else None
        fwd[i] = a is not None
        rc[i] = b is not None
        if a is not None or b is not None:
            u = (a or set()) | (b or set())
            conc[i] = u == {int(y)}
    return fwd, rc, fwd | rc, conc


def any_copy_concordant(tr, ytr, te, yte):
    """Per test sequence (fix round 2, looser definition for transparency): a duplicate for which AT LEAST
    ONE train copy (either strand) carries the test label.  exact_dup_flags' `concordant` requires ALL
    train copies to carry it."""
    labs = {}
    for s, y in zip(tr, ytr):
        labs.setdefault(s, set()).add(int(y))
    out = np.zeros(len(te), bool)
    for i, (s, y) in enumerate(zip(te, yte)):
        u = labs.get(s, set()) | labs.get(revcomp(s), set())
        out[i] = int(y) in u
    return out


def dinuc_peak(seqs, din):
    """(position, frequency) of the most frequent start position of dinucleotide `din`."""
    L = min(len(s) for s in seqs)
    A = np.frombuffer("".join(s[:L] for s in seqs).encode("ascii"), np.uint8).reshape(len(seqs), L)
    hit = (A[:, :-1] == ord(din[0])) & (A[:, 1:] == ord(din[1]))
    f = hit.mean(axis=0)
    j = int(f.argmax())
    return j, float(f[j])


def main():
    _mv.start_memguard(tag="audit_exact_dups")
    os.makedirs(OUT_DIR, exist_ok=True)
    cov = pd.read_csv(os.path.join(RD, "multivar_covariates.csv")).set_index("task")
    rows = []
    for task in _mv.TASKS:
        tr, ytr, te, yte = _mv.load_task_cached(task)
        fwd, rc, anyd, conc = exact_dup_flags(tr, ytr, te, yte)
        conc_any = any_copy_concordant(tr, ytr, te, yte)
        np.savez_compressed(os.path.join(OUT_DIR, f"{task}.npz"), fwd=fwd, rc=rc, any=anyd, concordant=conc,
                            yte=yte)
        ncls = int(max(ytr.max(), yte.max()) + 1)
        row = dict(task=task, S11=bool(cov.loc[task, "S11"]), n_train=len(tr), n_test=len(te), n_classes=ncls,
                   n_exact_any=int(anyd.sum()), frac_exact_fwd=float(fwd.mean()),
                   frac_exact_rc_only=float((rc & ~fwd).mean()), frac_exact_any=float(anyd.mean()),
                   frac_dup_label_concordant=float(conc[anyd].mean()) if anyd.any() else np.nan,
                   frac_dup_label_any_copy_concordant=float(conc_any[anyd].mean()) if anyd.any() else np.nan,
                   concordance_definition=("frac_dup_label_concordant: ALL train copies (either strand) carry the "
                                           "test label; _any_copy_: at least one does"),
                   homology_jaccard07=float(cov.loc[task, "homology"]))
        for c in range(3):
            m = yte == c
            row[f"n_test_class{c}"] = int(m.sum()) if c < ncls else np.nan
            row[f"dup_frac_class{c}"] = float(anyd[m].mean()) if c < ncls and m.any() else np.nan
            row[f"dup_share_class{c}"] = float((yte[anyd] == c).mean()) if c < ncls and anyd.any() else np.nan
        rows.append(row)
        print(f"{task:<34} test {len(te):>6}  verbatim dup (either strand) {anyd.mean():.4f}  "
              f"label-concordant {row['frac_dup_label_concordant']:.3f}  per-class "
              + " ".join(f"c{c}={row[f'dup_frac_class{c}']:.3f}" for c in range(ncls)), flush=True)
        del tr, te
    df = pd.DataFrame(rows)
    tmp = os.path.join(RD, "multivar_posthoc_exact_dup.csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(RD, "multivar_posthoc_exact_dup.csv"))

    # ---------------------------------------------------------------- class identity
    ci = []
    a_tr, a_ytr, a_te, a_yte = _mv.load_task_cached("nt_enhancers")
    b_tr, b_ytr, b_te, b_yte = _mv.load_task_cached("nt_enhancers_types")
    ci.append(dict(task="nt_enhancers_types", check="train_sequences_identical_in_order_to_nt_enhancers",
                   value=float(a_tr == b_tr), note="1 = identical lists"))
    ci.append(dict(task="nt_enhancers_types", check="test_sequences_identical_in_order_to_nt_enhancers",
                   value=float(a_te == b_te), note="1 = identical lists"))
    ci.append(dict(task="nt_enhancers_types", check="label0_set_equals_nt_enhancers_label0_train",
                   value=float(np.array_equal(a_ytr == 0, b_ytr == 0)),
                   note="nt_enhancers label 0 = non-enhancer (binary, label 1 = positive; data_nt.py)"))
    ci.append(dict(task="nt_enhancers_types", check="label0_set_equals_nt_enhancers_label0_test",
                   value=float(np.array_equal(a_yte == 0, b_yte == 0)), note=""))
    for c in range(3):
        ci.append(dict(task="nt_enhancers_types", check=f"train_class{c}_fraction_that_is_nt_enhancers_positive",
                       value=float((a_ytr[b_ytr == c] == 1).mean()), note=""))
    tr, ytr, te, yte = _mv.load_task_cached("nt_splice_sites_all")
    for c in range(3):
        S = [s for s, y in zip(tr, ytr) if y == c]
        for din in ("GT", "AG"):
            j, f = dinuc_peak(S, din)
            ci.append(dict(task="nt_splice_sites_all", check=f"class{c}_{din}_peak_position", value=j,
                           note=f"0-based start; n={len(S)} train seqs of length {len(S[0])}"))
            ci.append(dict(task="nt_splice_sites_all", check=f"class{c}_{din}_peak_frequency", value=f, note=""))
    cdf = pd.DataFrame(ci)
    cdf.to_csv(os.path.join(RD, "multivar_posthoc_class_identity.csv"), index=False)
    print(cdf.to_string())
    print(f"wrote multivar_posthoc_exact_dup.csv, multivar_posthoc_class_identity.csv; peak footprint "
          f"{_mv._PEAK[0]:.0f} MB")


if __name__ == "__main__":
    main()
