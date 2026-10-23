#!/usr/bin/env python3
"""POST HOC (added 2026-09-26, after the --quick smoke run printed the hg19 extent of the two pre-registered
nt rule regions): which ChromHMM definition reproduces the iEnhancer-2L candidate counts?

Liu et al. 2016 (iEnhancer-2L) report "742 strong enhancers, 370 517 weak enhancers, and 5 257 994
non-enhancers" before random down-sampling, with strong enhancers and non-enhancers "clearly annotated in
all the nine tissues" and all samples "divided into 200 bp fragments". The paper does not enumerate the
states. For candidate definitions (applied to the same nine hg19 Broad/ENCODE 15-state segmentations) we
report the merged region size and the number of whole 200-bp fragments it contains (sum over merged
regions of floor(len/200)), to compare with the published counts. Descriptive only: a count match is
circumstantial evidence for the definition, not proof. Autosomes + chrX (chrY is absent in female lines).

Output: results/upgrades/neg_provenance_chromhmm_rule_counts.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402
import genome_utils as G  # noqa: E402

RULES = {
    "state13_all9 (Heterochrom/lo in all nine)": lambda st: st == 13,
    "nonenhancer_all9 (not states 4-7 in all nine)": lambda st: ~np.isin(st, (4, 5, 6, 7)),
    "nonenh_nonprom_all9 (not states 1-7 in all nine)": lambda st: ~np.isin(st, (1, 2, 3, 4, 5, 6, 7)),
    "state13or12_all9 (Heterochrom/lo or Repressed in all nine)": lambda st: np.isin(st, (12, 13)),
    "state13to15_all9 (Heterochrom/lo or Repetitive/CNV in all nine)": lambda st: np.isin(st, (13, 14, 15)),
    "strongenh_all9 (states 4-5 in all nine)": lambda st: np.isin(st, (4, 5)),
    "enhancer_all9 (states 4-7 in all nine)": lambda st: np.isin(st, (4, 5, 6, 7)),
}
ANY_RULES = {   # (predicate, exclusion-predicate applied in ALL nine or None)
    "weakenh_any (states 6-7 in >=1 of nine)": (lambda st: np.isin(st, (6, 7)), None),
    "enhancer_any (states 4-7 in >=1 of nine)": (lambda st: np.isin(st, (4, 5, 6, 7)), None),
    "enhancer_any_not_strong_all9 (4-7 in >=1, not 4-5 in all nine)": (lambda st: np.isin(st, (4, 5, 6, 7)),
                                                                       lambda st: np.isin(st, (4, 5))),
}
PUBLISHED = {"state13_all9 (Heterochrom/lo in all nine)": 5257994,
             "nonenhancer_all9 (not states 4-7 in all nine)": 5257994,
             "nonenh_nonprom_all9 (not states 1-7 in all nine)": 5257994,
             "state13or12_all9 (Heterochrom/lo or Repressed in all nine)": 5257994,
             "state13to15_all9 (Heterochrom/lo or Repetitive/CNV in all nine)": 5257994,
             "strongenh_all9 (states 4-5 in all nine)": 742, "enhancer_all9 (states 4-7 in all nine)": 742,
             "weakenh_any (states 6-7 in >=1 of nine)": 370517, "enhancer_any (states 4-7 in >=1 of nine)": 370517,
             "enhancer_any_not_strong_all9 (4-7 in >=1, not 4-5 in all nine)": 370517}


def rule_regions(H, pred, mode="all", excl_all=None):
    out = {}
    for c in [f"chr{i}" for i in range(1, 23)] + ["chrX"]:
        if not all(c in H[cell] for cell in G.CHROMHMM_CELLS):
            continue
        pts = np.unique(np.concatenate([np.concatenate([H[cell][c][0], H[cell][c][1]]) for cell in G.CHROMHMM_CELLS]))
        a, b = pts[:-1], pts[1:]
        good = np.ones(len(a), bool) if mode == "all" else np.zeros(len(a), bool)
        excl = np.ones(len(a), bool)
        for cell in G.CHROMHMM_CELLS:
            s, e, st = H[cell][c]
            i = np.searchsorted(s, a, side="right") - 1
            cov = (i >= 0) & (a < e[np.clip(i, 0, None)])
            v = np.where(cov, st[np.clip(i, 0, None)], 0)
            if mode == "all":
                good &= cov & pred(v)
            else:
                good |= cov & pred(v)
            if excl_all is not None:
                excl &= cov & excl_all(v)
        if excl_all is not None:
            good &= ~excl
        out[c] = G.merge_intervals(a[good], b[good])
    return out


def main():
    H = G.chromhmm()
    rows = []
    todo = [(n, p, "all", None) for n, p in RULES.items()] + [(n, p, "any", x) for n, (p, x) in ANY_RULES.items()]
    for name, pred, mode, excl in todo:
        R = rule_regions(H, pred, mode, excl)
        L = np.concatenate([e - s for s, e in R.values()])
        rows.append(dict(rule=name, merged_regions=int(len(L)), region_bp=int(L.sum()),
                         n_200bp_fragments=int((L // 200).sum()), regions_ge200bp=int((L >= 200).sum()),
                         published_count=PUBLISHED[name],
                         ratio_fragments_to_published=float((L // 200).sum() / PUBLISHED[name])))
        print(rows[-1], flush=True)
    out = os.path.join(C.RESULTS_DIR, "neg_provenance_chromhmm_rule_counts.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
