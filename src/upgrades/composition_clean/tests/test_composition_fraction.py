#!/usr/bin/env python3
"""Sanity checks for Depth 2 (composition-fraction decomposition). Fast, synthetic, no torch.

Two controlled regimes verify the metric behaves as defined:
  composition_fraction = (AUROC_comp_only - 0.5) / (AUROC_full - 0.5)

  (A) COMPOSITION fully separates the classes (positives GC-rich, negatives AT-rich): the
      comp-only model already captures the signal, so AUROC_comp ~ AUROC_full and the
      fraction sits near 1.
  (B) only a HIGHER-ORDER motif separates the classes, with mononucleotide+dinucleotide
      composition held ~matched between classes: the comp-only model is near chance while a
      full k-mer model separates, so the fraction sits near 0 (and well below regime A).

Run:  python src/upgrades/composition_clean/tests/test_composition_fraction.py  (prints PASS)
Also discoverable by pytest (def test_*)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))           # the module dir (run_composition_fraction)
sys.path.insert(0, os.path.join(HERE, "..", ".."))     # src/upgrades (for _common)
import _common as C
import featurize
import run_composition_fraction as R

SEED = 42
BOOT = 200            # small bootstrap keeps the test fast; point fraction is exact


# ----------------------------------------------------------------- generators
def _gc_seq(gc, length, rng):
    """Random ACGT sequence with expected GC fraction ``gc`` (A,C,G,T probs)."""
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choice(list("ACGT"), size=length, p=p))


def _composition_separable(n=300, length=150, seed=0):
    """Positives GC-rich (~0.70), negatives AT-rich (~0.30) -> composition alone separates."""
    rng = np.random.RandomState(seed)
    pos = [_gc_seq(0.70, length, rng) for _ in range(n)]
    neg = [_gc_seq(0.30, length, rng) for _ in range(n)]
    seqs = pos + neg
    y = np.array([1] * n + [0] * n, dtype=np.int64)
    return seqs, y


def _dinuc_shuffle(seq, rng):
    """Altschul-Erikson dinucleotide-preserving shuffle: returns a permutation of ``seq`` with the
    EXACT same first letter, last letter, mononucleotide counts AND dinucleotide (adjacent-pair)
    counts -- a random Eulerian walk of the dinucleotide de Bruijn (multi)graph. This is the
    textbook negative control that destroys k>=3 structure while holding k<=2 composition fixed."""
    s = seq
    n = len(s)
    if n < 3:
        return s
    last = s[-1]
    # outgoing edges per node (letter), in order of appearance
    edges = {}
    for a, b in zip(s[:-1], s[1:]):
        edges.setdefault(a, []).append(b)
    # build a random arborescence into `last`: pick a random last-outgoing edge per node
    while True:
        for a in edges:
            rng.shuffle(edges[a])
        # choose, for each node != last, the edge used to (eventually) reach `last`
        intree = {last}
        ok = True
        last_edge = {}
        for a in list(edges):
            if a == last:
                continue
            # walk forward until we hit the in-tree set; mark the FIRST edge that does
            cur = a; path = [a]
            seen = {a}
            while cur not in intree:
                nxts = [b for b in edges[cur]]
                if not nxts:
                    ok = False; break
                nxt = nxts[0]
                if nxt in seen:                       # cycle not reaching last -> retry shuffle
                    ok = False; break
                last_edge[cur] = nxt
                cur = nxt; path.append(cur); seen.add(cur)
            if not ok:
                break
            intree.update(path)
        if ok:
            break
    # order each node's edge list so the arborescence edge is used LAST
    order = {}
    for a in edges:
        lst = list(edges[a])
        if a in last_edge:
            le = last_edge[a]
            lst.remove(le); rng.shuffle(lst); lst.append(le)
        else:
            rng.shuffle(lst)
        order[a] = lst
    # Eulerian walk from the original first letter
    out = [s[0]]
    cur = s[0]
    ptr = {a: 0 for a in order}
    for _ in range(n - 1):
        nxt = order[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def _motif_only_pairs(n=400, length=150, seed=0):
    """Return (pos, neg) lists where neg[i] is the dinucleotide-preserving shuffle of pos[i].

    Each positive is a GC-neutral i.i.d. background with several non-overlapping copies of a
    discriminative 6-mer implanted. Because neg[i] shares pos[i]'s EXACT mono+dinucleotide counts,
    the discriminative information lives strictly at k>=3 (the implanted 6-mer) -- invisible to the
    comp-only (k=1,2) model but visible to the full k-mer model. Returned as matched pairs so the
    caller can keep each pair on the SAME side of the train/test split: then the test-set positive
    and negative composition distributions are identical and comp-only AUROC is exactly chance."""
    rng = np.random.RandomState(seed)
    motif = "ACGTAC"
    m = len(motif)
    reps = 6

    def make_pos():
        s = list(rng.choice(list("ACGT"), size=length))
        for _ in range(reps):
            j = rng.randint(0, length - m)
            s[j:j + m] = list(motif)
        return "".join(s)

    pos = [make_pos() for _ in range(n)]
    neg = [_dinuc_shuffle(p, rng) for p in pos]
    return pos, neg


# --------------------------------------------------------------------- driver
def _fraction_on_split(tr_seqs, ytr, te_seqs, yte, k_full, seed=SEED, boot=BOOT):
    """Fit comp-only (k1,2) and full (k-mer at k_full) ONCE on train; return the paired-bootstrap
    composition-fraction result dict from run_composition_fraction (the production code path)."""
    Ctr = C.comp_signature(tr_seqs, ks=(1, 2)); Cte = C.comp_signature(te_seqs, ks=(1, 2))
    p_comp = R._proba1("lgbm", Ctr, ytr, Cte, seed)
    Xtr = featurize.kmer_spectrum(tr_seqs, k_full); Xte = featurize.kmer_spectrum(te_seqs, k_full)
    p_full = R._proba1("lgbm", Xtr, ytr, Xte, seed)
    return R.paired_fraction(yte, p_full, p_comp, seed, boot)


def _fraction(seqs, y, k_full, seed=SEED, boot=BOOT):
    """Composition-separable regime: ordinary stratified 70/30 split."""
    from sklearn.model_selection import StratifiedShuffleSplit
    tr, te = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=seed)
                  .split(np.zeros(len(y)), y))
    tr_seqs = [seqs[i] for i in tr]; te_seqs = [seqs[i] for i in te]
    return _fraction_on_split(tr_seqs, y[tr], te_seqs, y[te], k_full, seed, boot)


def _fraction_motif_only(n=400, k_full=4, seed=SEED, boot=BOOT):
    """Motif-only regime: split by matched PAIR (pos[i] and its dinuc-shuffle neg[i] go to the same
    side), so test pos/neg composition is identical and comp-only AUROC is exactly chance."""
    pos, neg = _motif_only_pairs(n=n, seed=seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n); cut = int(0.70 * n)
    tri, tei = perm[:cut], perm[cut:]
    tr_seqs = [pos[i] for i in tri] + [neg[i] for i in tri]
    te_seqs = [pos[i] for i in tei] + [neg[i] for i in tei]
    ytr = np.array([1] * len(tri) + [0] * len(tri), dtype=np.int64)
    yte = np.array([1] * len(tei) + [0] * len(tei), dtype=np.int64)
    return _fraction_on_split(tr_seqs, ytr, te_seqs, yte, k_full, seed, boot)


# ----------------------------------------------------------------------- tests
def test_composition_separable_fraction_near_one():
    seqs, y = _composition_separable(seed=1)
    res = _fraction(seqs, y, k_full=4)
    assert res["full_auroc"] > 0.9, f"GC-separable full AUROC should be high ({res['full_auroc']:.3f})"
    assert res["comp_auroc"] > 0.9, f"comp-only should also separate ({res['comp_auroc']:.3f})"
    assert res["frac"] > 0.8, f"composition fraction should be near 1 ({res['frac']:.3f})"


def test_motif_only_fraction_near_zero():
    res = _fraction_motif_only(n=400, k_full=4, seed=2)
    assert res["full_auroc"] > 0.8, f"k-mer model should see the 6-mer ({res['full_auroc']:.3f})"
    assert abs(res["comp_auroc"] - 0.5) < 0.05, \
        f"comp-only should be at chance (matched k<=2 composition), got {res['comp_auroc']:.3f}"
    assert abs(res["frac"]) < 0.10, f"motif-only composition fraction should be near 0 ({res['frac']:.3f})"


def test_composition_clearly_exceeds_motif_only():
    """The metric must ORDER the two regimes: composition-driven >> motif-only."""
    fa = _fraction(*_composition_separable(seed=3), k_full=4)["frac"]
    fb = _fraction_motif_only(n=400, k_full=4, seed=3)["frac"]
    assert fa - fb > 0.5, f"composition fraction must dominate motif-only ({fa:.3f} vs {fb:.3f})"


def test_fraction_ci_is_two_element_and_brackets_definition():
    """CI is a 2.5/97.5 percentile pair and the point fraction matches the stated formula."""
    seqs, y = _composition_separable(seed=4)
    res = _fraction(seqs, y, k_full=4)
    assert res["frac_lo"] <= res["frac_hi"], "CI must be ordered lo<=hi"
    recomputed = (res["comp_auroc"] - 0.5) / (res["full_auroc"] - 0.5)
    assert abs(recomputed - res["frac"]) < 1e-9, "point fraction must equal (comp-0.5)/(full-0.5)"


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} composition_fraction sanity tests PASSED")
