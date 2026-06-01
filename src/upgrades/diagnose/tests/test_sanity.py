#!/usr/bin/env python3
"""End-to-end sanity check for Upgrade 10 (the three-route diagnostic CLI), on a small bundled toy
dataset. Fast, no torch. Run:  python src/upgrades/diagnose/tests/test_sanity.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import diagnose as DG

TOY = os.path.join(HERE, "..", "toy_dataset.csv")


def _make_toy(n=200, L=150, seed=0):
    """A clearly motif-discriminable toy: positives carry a GC-box (Sp1) consensus, negatives do not."""
    rng = np.random.RandomState(seed)
    motif = "GGGGCGGGGC"
    def bg():
        return "".join(rng.choice(list("ACGT"), size=L))
    pos, neg = [], []
    for _ in range(n):
        s = list(bg())
        j = rng.randint(0, L - len(motif)); s[j:j + len(motif)] = list(motif)
        pos.append("".join(s))
        neg.append(bg())
    seqs = pos + neg
    y = np.array([1] * n + [0] * n)
    return seqs, y


def test_diagnose_card_structure_and_verdict():
    seqs, y = _make_toy()
    card = DG.diagnose(seqs, y, boot=100, seed=42)
    for key in ("learnability", "route1_motif", "route2_positional", "route3_negatives",
                "composition_fraction", "verdict"):
        assert key in card, f"missing report-card section: {key}"
    assert card["learnability"]["mcc"] > 0.5, f"implanted motif should be learnable, got {card['learnability']['mcc']}"
    assert card["route1_motif"]["max_pos_motif_auroc"] > 0.6, "Sp1-in-positives should be discriminable"
    assert "SOLVED" in card["verdict"], f"expected SOLVED verdict, got {card['verdict']}"


def test_composition_fraction_present_and_well_formed():
    seqs, y = _make_toy()
    card = DG.diagnose(seqs, y, boot=100, seed=42)
    cf = card["composition_fraction"]
    assert isinstance(cf["value"], (int, float)) and np.isfinite(cf["value"]), \
        f"composition_fraction value must be numeric, got {cf['value']!r}"
    assert isinstance(cf["ci"], list) and len(cf["ci"]) == 2, \
        f"composition_fraction ci must be a 2-element list, got {cf['ci']!r}"
    assert cf["ci"][0] <= cf["ci"][1], "composition_fraction CI must be ordered lo<=hi"
    assert "definition" in cf and "0.5" in cf["definition"], "composition_fraction must carry its definition"


def test_markdown_and_json_serializable():
    import json
    seqs, y = _make_toy(n=120)
    card = DG.diagnose(seqs, y, boot=50, seed=42)
    md = DG._to_markdown(card, "toy")
    assert "VERDICT" in md and "Route 2" in md
    json.dumps(card)                                    # must be JSON-serializable (no numpy types)


def test_bundled_toy_csv_loads():
    if not os.path.exists(TOY):
        # write the bundled toy on first run so the CLI example is reproducible
        import pandas as pd
        seqs, y = _make_toy(n=150)
        pd.DataFrame({"sequence": seqs, "label": y}).to_csv(TOY, index=False)
    s, yy, coords = DG._read_input(TOY)
    assert len(s) == len(yy) and coords is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} diagnose sanity tests PASSED")
