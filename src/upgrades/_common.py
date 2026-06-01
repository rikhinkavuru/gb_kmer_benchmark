"""Shared helpers for the upgrade modules. CPU-only, torch-free.

Everything here reuses the existing root-level pipeline (data, featurize, models,
motif_jaspar, motif_match, run_stats) so the upgrades inherit the same fixed seed,
fixed k-mer vocabulary, deterministic LightGBM, and 1000-resample percentile
bootstrap. Importing this module adds the repo root to sys.path; it imports NO
torch/transformers (the FM extractor isolates those behind an optional extra).
"""
import os
import sys

import numpy as np

# --- repo root on sys.path so `import data, featurize, ...` resolve -------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data as gbdata            # noqa: E402
import featurize                 # noqa: E402
import models as gbmodels        # noqa: E402
import motif_jaspar             # noqa: E402
import motif_match             # noqa: E402
from run_stats import auc_counts, auc_float, boot_ci  # noqa: E402,F401  (re-exported)

from scipy import sparse                               # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: E402

SEED = 42
BOOT = 1000
RESULTS_DIR = os.path.join(ROOT, "results", "upgrades")
SPLITS_V2_DIR = os.path.join(ROOT, "cleaned_splits_v2")


# ---------------------------------------------------------------- loaders
def load_original(task, seed=SEED):
    """(tr_seqs, ytr, te_seqs, yte) for an original GB/NT task (via data.load_dataset)."""
    d = gbdata.load_dataset(task, seed=seed)
    return d["train_seqs"], d["y_train"], d["test_seqs"], d["y_test"]


def load_csv_pair(train_csv, test_csv):
    """Load a cleaned split written as <task>_{train,test}.csv (columns sequence,label)."""
    import pandas as pd
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    return (tr["sequence"].astype(str).str.upper().tolist(), tr["label"].to_numpy(np.int64),
            te["sequence"].astype(str).str.upper().tolist(), te["label"].to_numpy(np.int64))


# ----------------------------------------------------- composition features
def gc_content(seqs):
    """Per-sequence GC fraction over A/C/G/T positions (non-ACGT ignored)."""
    codes = motif_match.encode_sequences(seqs)        # (n,S) int8, non-ACGT/pad = 4
    valid = codes < 4
    gc = ((codes == 1) | (codes == 2)) & valid
    return gc.sum(1) / np.maximum(valid.sum(1), 1)


def comp_signature(seqs, ks=(1, 2)):
    """Dense composition signature = concatenated L1-normalized k-mer spectra for k in ks.
    ks=(1,2) -> 4 mononucleotide + 16 dinucleotide frequencies (20-dim); GC is implied by
    the mononucleotide block. Reuses featurize.kmer_spectrum (the SAME deterministic
    featurizer used by the benchmark)."""
    blocks = [featurize.kmer_spectrum(seqs, k).toarray().astype(np.float32) for k in ks]
    return np.hstack(blocks)


def dinuc_freqs(seqs):
    """(n, 16) dinucleotide frequency matrix (L1-normalized), column order AA,AC,...,TT.
    Pairs touching a non-ACGT base are skipped (so an internal N breaks the dinucleotide)."""
    return featurize.kmer_spectrum(seqs, 2).toarray().astype(np.float64)


# ------------------------------------------------- fit / eval / bootstrap
def _auroc(y, proba, nc):
    return (roc_auc_score(y, proba[:, 1]) if nc == 2 else
            roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def fit_eval_boot(Xtr, ytr, Xte, yte, model_name, seed, B, rng):
    """Fit `model_name` on (Xtr,ytr), evaluate once on (Xte,yte), return point metrics +
    1000-resample percentile bootstrap CIs (test sequence = resampling unit). Works on
    dense or sparse X. Mirrors run_stats.bootstrap_benchmark exactly so CIs are comparable.

    Returns dict(acc, mcc, mcc_lo, mcc_hi, auroc, auroc_lo, auroc_hi, n_test, n_classes).
    """
    nc = len(np.unique(ytr))
    m = gbmodels.build_model(model_name, seed, nc)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    pred = m.classes_[np.argmax(proba, axis=1)]
    acc = float((pred == yte).mean())
    mcc_pt = float(matthews_corrcoef(yte, pred))
    try:
        auc_pt = float(_auroc(yte, proba, nc))
    except ValueError:
        auc_pt = float("nan")
    n = len(yte)
    mm = np.empty(B); aa = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            mm[b] = np.nan; continue
        mm[b] = matthews_corrcoef(yte[idx], pred[idx])
        try:
            aa[b] = _auroc(yte[idx], proba[idx], nc)
        except ValueError:
            pass
    return dict(acc=round(acc, 6), mcc=round(mcc_pt, 6),
                mcc_lo=round(float(np.nanpercentile(mm, 2.5)), 6),
                mcc_hi=round(float(np.nanpercentile(mm, 97.5)), 6),
                auroc=(round(auc_pt, 6) if auc_pt == auc_pt else float("nan")),
                auroc_lo=round(float(np.nanpercentile(aa, 2.5)), 6),
                auroc_hi=round(float(np.nanpercentile(aa, 97.5)), 6),
                n_test=int(n), n_classes=int(nc))


def kmer_fit_eval_boot(tr_seqs, ytr, te_seqs, yte, k, model_name, seed, B, rng):
    """Convenience: featurize with the standard k-mer spectrum at k, then fit_eval_boot."""
    Xtr = featurize.kmer_spectrum(tr_seqs, k)
    Xte = featurize.kmer_spectrum(te_seqs, k)
    return fit_eval_boot(Xtr, ytr, Xte, yte, model_name, seed, B, rng)


def load_tbp():
    """The vertebrate TBP/TATA PSSM (used for all species, per the paper's convention)."""
    return next(m for m in motif_jaspar.load_pssms("vertebrates") if m["name"].upper() == "TBP")


def load_sp1():
    """The vertebrate SP1 GC-box PSSM."""
    return next(m for m in motif_jaspar.load_pssms("vertebrates") if m["name"].upper() == "SP1")


def tata_hit_counts(seqs, tbp, thr_bits=1.3):
    """Per-sequence TATA(TBP) motif-hit count at >= thr_bits bits/pos, both strands."""
    return motif_match.count_hits_batch(motif_match.encode_sequences(list(seqs)), tbp["pssm"], thr_bits)


def motif_only_auroc(pos_counts, neg_counts):
    """Directed motif-only AUROC = P(pos hit-count > neg) for integer counts (auc_counts)."""
    return auc_counts(np.asarray(pos_counts, np.int64), np.asarray(neg_counts, np.int64))
