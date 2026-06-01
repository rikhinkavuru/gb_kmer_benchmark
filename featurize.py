"""k-mer spectrum featurization (fixed 4^k vocabulary, L1 frequency-normalized).

CPU-only; depends on numpy + scikit-learn (CountVectorizer) + scipy.sparse (which
ships with scikit-learn). No torch/tensorflow, no pretrained embeddings.

Design choices that matter for reproducibility / leakage-safety
--------------------------------------------------------------
* Fixed, data-INDEPENDENT vocabulary: the full set of 4^k ACGT k-mers, generated
  by ``itertools.product`` in a deterministic order. The feature space is
  therefore identical for every dataset and every split, so columns always align
  and there is no possibility of train/test vocabulary leakage.
* Overlapping k-mer counts via ``CountVectorizer(analyzer="char")``; any k-mer
  containing a non-ACGT character (e.g. ``N``) simply falls outside the fixed
  vocabulary and is ignored.
* The count vector is L1-normalized into a frequency *spectrum* whose entries sum
  to 1 (an all-zero row -- a sequence shorter than k, or all-N -- is left at 0).
* Matrices stay sparse (scipy CSR, float32) so the large datasets (100k x 4096)
  fit comfortably in RAM, and are cached to disk as ``.npz``.
"""
import itertools

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

BASES = "ACGT"


def vocabulary(k):
    """Deterministic list of all 4^k ACGT k-mers (fixed feature order)."""
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def feature_dim(k):
    """Dimensionality of the k-mer spectrum = 4**k."""
    return 4 ** k


def kmer_spectrum(seqs, k):
    """Return an (n_seqs x 4^k) CSR matrix of L1-normalized k-mer frequencies.

    Parameters
    ----------
    seqs : list[str]   uppercase DNA sequences (A/C/G/T plus possibly N etc.)
    k    : int         k-mer length

    The matrix is float32 and sparse. Rows are frequency spectra summing to 1
    (zero rows are left as zero, never NaN).
    """
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k),
                          vocabulary=vocabulary(k), lowercase=False)
    counts = vec.fit_transform(seqs).astype(np.float32)   # fixed vocab => deterministic
    # L1 normalization -> relative k-mer frequencies; zero-norm rows stay zero.
    return normalize(counts, norm="l1", axis=1, copy=False).tocsr()


def binned_kmer_spectrum(seqs, k, n_bins):
    """Position-aware featurization: split each sequence into ``n_bins`` equal-
    fraction contiguous bins, take the L1-normalized k-mer spectrum WITHIN each bin,
    and concatenate the bins -> (n_seqs, n_bins * 4^k) CSR.

    ``n_bins == 1`` is exactly ``kmer_spectrum`` (the position-blind baseline), so a
    B=1 vs B>1 comparison at the same k isolates the effect of resolving position.
    Coarse positional information (which k-mers occur in which part of the sequence)
    is preserved, so a fixed-position motif (e.g. a splice GT-AG at a set offset)
    becomes learnable, while a positionally-diffuse motif is unaffected. Bin width is
    a fraction of each sequence's own length, so variable-length inputs are fine.
    """
    if n_bins <= 1:
        return kmer_spectrum(seqs, k)
    from scipy import sparse
    blocks = []
    for b in range(n_bins):
        subs = [s[(b * len(s)) // n_bins:((b + 1) * len(s)) // n_bins] for s in seqs]
        blocks.append(kmer_spectrum(subs, k))
    return sparse.hstack(blocks, format="csr")
