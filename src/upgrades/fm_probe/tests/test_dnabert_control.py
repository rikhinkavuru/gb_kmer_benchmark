"""Sanity tests for F013 DNABERT-2 placebo control -- mask mechanics + degeneracy logic.

Torch-free (the control reuses cached embeddings; this test exercises only the masking / bootstrap
/ degeneracy helpers, not any FM forward pass)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import run_dnabert_control as R  # noqa: E402


def test_membership_mask_dup_safe():
    orig = ["A", "B", "A", "C", "A"]
    m = R.membership_mask(orig, ["A", "A", "C"])     # want 2 A's + 1 C
    assert m.tolist() == [True, False, True, True, False]   # first two A's + the C
    assert m.sum() == 3


def test_test_effect_ci_brackets_zero_when_subset_is_representative():
    # full test: perfectly separable; a RANDOM half-subset keeps separability -> test_effect ~ 0.
    rng = np.random.RandomState(0)
    n = 400
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    proba = np.zeros((n, 2))
    proba[y == 0, 1] = rng.uniform(0.0, 0.45, (y == 0).sum())
    proba[y == 1, 1] = rng.uniform(0.55, 1.0, (y == 1).sum())
    proba[:, 0] = 1 - proba[:, 1]
    mask = np.ones(n, bool)
    drop = rng.permutation(np.where(y == 0)[0])[:50]   # representative removal
    mask[drop] = False
    lo, hi = R.test_effect_ci(y, proba, mask, boot=400, rng=np.random.RandomState(42))
    assert lo <= 0 <= hi, (lo, hi)                     # flat control brackets 0


def test_shuffled_control_is_degenerate():
    # dinucleotide shuffles preserve per-sequence GC -> gc_match retains ~everything.
    ret, identical, npos = R.shuffled_control_retention("human_enhancers_cohn", seed=42)
    assert identical is True
    assert ret > 0.99, ret                              # degenerate: removes ~nothing
    assert npos > 1000


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL F013 SANITY TESTS PASSED")
