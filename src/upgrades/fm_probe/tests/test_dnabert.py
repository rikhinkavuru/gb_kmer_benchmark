#!/usr/bin/env python3
"""Sanity checks for the DNABERT-2 extractor (Depth 1) -- the parts that need no model download / no
torch: pinned constants, the .npz cache round-trip, and the state-dict key transform + load guard that
prevents a silent random-weight load. Run:
  python src/upgrades/fm_probe/tests/test_dnabert.py"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import extract_embeddings_dnabert as XD       # torch-free at import time


def test_constants_pinned():
    assert XD.MODEL_REVISION and len(XD.MODEL_REVISION) == 40, "DNABERT-2 revision must be a 40-char sha"
    assert XD.HIDDEN_DIM == 768 and XD.POOLINGS == ("mean", "cls") and XD.MAX_TOKENS == 512
    assert XD.MODEL_TAG == "dnabert2-117m"


def test_cache_roundtrip():
    mean = np.arange(2 * XD.HIDDEN_DIM, dtype=np.float32).reshape(2, XD.HIDDEN_DIM)
    cls = mean[::-1].copy(); y = np.array([0, 1])
    with tempfile.TemporaryDirectory() as d:
        XD.save_cached(d, "toy", "orig", "train", mean, cls, y)
        got = XD.load_cached(d, "toy", "orig", "train")
        assert got is not None and np.array_equal(got["mean"], mean) and np.array_equal(got["cls"], cls)
        assert np.array_equal(got["y"], y)
        assert XD.load_cached(d, "toy", "orig", "test") is None


def test_state_dict_key_transform_and_guard():
    """Mimic the bert.-strip + cls.-drop transform used in load_model, and the guard that flags a
    random-weight load. A complete checkpoint must leave ONLY 'pooler.*' missing."""
    # a realistic (mini) DNABERT-2 checkpoint: 'bert.' prefix on encoder/embeddings, 'cls.' MLM head
    ckpt = {"bert.embeddings.word_embeddings.weight": 1, "bert.encoder.layer.0.attn.weight": 1,
            "cls.predictions.decoder.weight": 1}
    sd = {(k[len("bert."):] if k.startswith("bert.") else k): v
          for k, v in ckpt.items() if not k.startswith("cls.")}
    assert set(sd) == {"embeddings.word_embeddings.weight", "encoder.layer.0.attn.weight"}, sd
    # model expects embeddings/encoder/pooler; only pooler is allowed to be missing
    model_keys = {"embeddings.word_embeddings.weight", "encoder.layer.0.attn.weight",
                  "pooler.dense.weight", "pooler.dense.bias"}
    missing = model_keys - set(sd)
    bad = [m for m in missing if not m.startswith("pooler.")]
    assert bad == [], f"guard should pass when only pooler is missing, got {bad}"
    # a BROKEN checkpoint (missing the encoder) must trip the guard
    sd_bad = {"embeddings.word_embeddings.weight": 1}
    missing2 = model_keys - set(sd_bad)
    bad2 = [m for m in missing2 if not m.startswith("pooler.")]
    assert "encoder.layer.0.attn.weight" in bad2, "guard must flag a missing encoder (random-init risk)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"ALL {len(fns)} dnabert sanity tests PASSED")
