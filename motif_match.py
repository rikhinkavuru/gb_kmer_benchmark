"""Score k-mers against JASPAR PWMs (absolute log-odds, bits) for enrichment tests.

Per-(k-mer, PWM) score
----------------------
For a k-mer and a PWM of length >= k, over every gapless window and BOTH strands,
the absolute log-odds score (uniform 0.25 background) is

    S = sum_i log2( PPM[base_i, window_i] / 0.25 )   (bits),

and the (k-mer, PWM) score is the maximum S / k (bits-per-position, max ~2.0).

Why we do NOT threshold "best match over the whole collection"
-------------------------------------------------------------
With ~880 PWMs x many windows x 2 strands, the maximum score over the WHOLE
collection saturates: 60-100% of all 4^k k-mers are the near-perfect consensus of
SOME motif window (measured), so "does this k-mer match any TF" is ~always yes and
cannot discriminate datasets. Instead we score each k-mer against each TF
separately (a single TF's PWM only scores high for k-mers resembling ITS motif,
so it does not saturate) and ask, per TF, whether that motif score is correlated
with k-mer importance (see run_interpretability.enrichment_z). PWMs shorter than
k cannot contain a k-mer and are excluded for that k.
"""
import itertools
import os

import numpy as np

BASES = "ACGT"
_CODE = {b: i for i, b in enumerate(BASES)}


def encode(kmers):
    """List of k-mer strings -> (n, k) int8 base-code array (A=0,C=1,G=2,T=3)."""
    return np.array([[_CODE[c] for c in s] for s in kmers], dtype=np.int8)


def all_kmer_codes(k):
    """(4^k, k) codes in the same order as featurize.vocabulary(k)."""
    return np.array(list(itertools.product(range(4), repeat=k)), dtype=np.int8)


def usable_pssms(pssms, k):
    """PWMs long enough to contain a k-mer (length >= k), order preserved."""
    return [m for m in pssms if m["length"] >= k]


def _score_vs_pssm(codes, pssm, k):
    """Best absolute log-odds S (bits, total) of each k-mer vs one PSSM, over all
    windows and both strands."""
    L = pssm.shape[1]
    n = codes.shape[0]
    rc = (3 - codes)[:, ::-1]               # reverse complement at the k-mer level
    best = np.full(n, -np.inf, dtype=np.float32)
    for strand in (codes, rc):
        for o in range(L - k + 1):
            S = np.zeros(n, dtype=np.float32)
            for i in range(k):
                S += pssm[strand[:, i], o + i]
            np.maximum(best, S, out=best)
    return best


def score_matrix(codes, pssms_usable, k):
    """(n_kmers, n_pwms) matrix of best bits-per-position for the given PWMs
    (which must already be filtered to length >= k)."""
    M = np.empty((codes.shape[0], len(pssms_usable)), dtype=np.float32)
    for j, m in enumerate(pssms_usable):
        M[:, j] = _score_vs_pssm(codes, m["pssm"], k) / k
    return M


def tf_score_matrix(k, pssms, cache_dir=None, tag=""):
    """(4^k x n_usable_TF) bits-per-position matrix over ALL k-mers, plus the list
    of usable PWMs (length >= k). Cached to disk; rows are in vocabulary order."""
    usable = usable_pssms(pssms, k)
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"tfscores__{tag}__k{k}.npy")
        if os.path.exists(path):
            return np.load(path), usable
    M = score_matrix(all_kmer_codes(k), usable, k)
    if path:
        np.save(path, M)
    return M, usable


# --- whole-sequence PWM scanning (for motif-discriminability analysis) -----
_LUT5 = np.full(256, 4, dtype=np.int8)            # default 4 = non-ACGT / pad
for _b, _i in _CODE.items():
    _LUT5[ord(_b)] = _i


def encode_sequences(seqs, pad=4):
    """List of sequences -> (n, S_max) int8 codes (A0 C1 G2 T3, non-ACGT/pad = 4),
    right-padded to the longest sequence. Windows touching code 4 never score (see
    count_hits_batch), so padding is harmless and lets us batch variable lengths."""
    S = max(len(s) for s in seqs)
    out = np.full((len(seqs), S), pad, dtype=np.int8)
    for r, s in enumerate(seqs):
        a = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
        out[r, :len(a)] = _LUT5[a]
    return out


def count_hits_batch(codes, pssm, thr_bits):
    """Per-sequence count of PWM matches at >= thr_bits bits/pos, scanning every
    window on BOTH strands. ``codes`` is (n, S) from encode_sequences. A window is
    a hit if it scores >= thr_bits on either strand; either-strand hits at the same
    position count once. Non-ACGT/pad positions (code 4) get -inf and never hit.
    """
    n, S = codes.shape
    L = pssm.shape[1]
    if S < L:
        return np.zeros(n, dtype=np.int32)
    npos = S - L + 1
    neg_inf = np.full((1, L), -1e9, dtype=np.float32)
    fwd = np.vstack([pssm, neg_inf])                       # (5, L), row 4 = -inf
    rev = np.vstack([pssm[::-1, ::-1], neg_inf])           # reverse-complement PWM

    def _scan(a):
        sc = np.zeros((n, npos), dtype=np.float32)
        for i in range(L):
            sc += a[codes[:, i:i + npos], i]
        return sc / L                                       # bits per position

    hit = (_scan(fwd) >= thr_bits) | (_scan(rev) >= thr_bits)
    return hit.sum(axis=1).astype(np.int32)
