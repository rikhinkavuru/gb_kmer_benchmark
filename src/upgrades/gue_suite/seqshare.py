"""Exact sequence-sharing primitives for the GUE independence audit (gue_overlap.py).

containment_scan(short, long_) answers, for every sequence s in `short`, whether s occurs verbatim
(forward strand OR reverse complement) as a substring of some sequence in `long_`, and where.
It is exact: candidate positions come from a 32-mer anchor index, and every candidate is confirmed by
a full string comparison, so there are no false positives and (for any s with at least one N-free
32-mer) no false negatives.

Method. For each s we take the first N-free 32-mer of s (offset o) and of rc(s) (offset o') as its two
anchors, packed 2 bits/base into a uint64. Every N-free 32-mer window of every long sequence is packed
the same way (vectorised, chunked) and looked up in the sorted anchor array; a window at position p that
matches the forward anchor of s is a containment iff long[p-o : p-o+len(s)] == s, and likewise for the
rc anchor with rc(s). Deterministic, seed-free, pure numpy + python; single thread.
"""
import numpy as np

K = 32
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i
_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rc(s):
    return s.translate(_RC)[::-1]


def encode(seqs):
    """(n, S_max) uint8 codes A0 C1 G2 T3, anything else / padding = 4."""
    S = max(len(s) for s in seqs)
    out = np.full((len(seqs), S), 4, dtype=np.uint8)
    for r, s in enumerate(seqs):
        a = np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)
        out[r, :len(a)] = _LUT[a]
    return out


def windows(codes):
    """Packed K-mer value and validity (no code 4 inside) for every window start of every row."""
    n, S = codes.shape
    npos = S - K + 1
    if npos <= 0:
        return np.zeros((n, 0), np.uint64), np.zeros((n, 0), bool)
    val = np.zeros((n, npos), np.uint64)
    bad = np.zeros((n, npos), bool)
    two = np.uint64(2); three = np.uint64(3)
    for o in range(K):
        c = codes[:, o:o + npos]
        bad |= (c == 4)
        val = (val << two) | (c.astype(np.uint64) & three)
    return val, ~bad


def _chunks_by_length(seqs, max_cells=4_000_000):
    """Index chunks of `seqs`, length-sorted, each with n * S_max <= max_cells."""
    order = np.argsort(np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs)), kind="stable")
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= max_cells:
            j += 1
        yield order[i:j]
        i = j


def build_anchors(seqs):
    """Anchor table for `seqs`: sorted uint64 values with (seq index, strand, offset).
    Returns (vals, sid, strand, off, n_unanchored)."""
    V, I, ST, OF = [], [], [], []
    unanch = 0
    for strand in (0, 1):
        for idx in _chunks_by_length(seqs):
            batch = [seqs[i] if strand == 0 else rc(seqs[i]) for i in idx]
            val, ok = windows(encode(batch))
            if val.shape[1] == 0:
                unanch += len(idx) if strand == 0 else 0
                continue
            has = ok.any(axis=1)
            first = np.argmax(ok, axis=1)
            rows = np.nonzero(has)[0]
            if strand == 0:
                unanch += int((~has).sum())
            V.append(val[rows, first[rows]]); I.append(idx[rows])
            ST.append(np.full(len(rows), strand, np.int8)); OF.append(first[rows].astype(np.int32))
    vals = np.concatenate(V) if V else np.zeros(0, np.uint64)
    sid = np.concatenate(I) if I else np.zeros(0, np.int64)
    st = np.concatenate(ST) if ST else np.zeros(0, np.int8)
    of = np.concatenate(OF) if OF else np.zeros(0, np.int32)
    o = np.argsort(vals, kind="stable")
    return vals[o], sid[o], st[o], of[o], unanch


def containment_scan(short, long_, anchors=None, on_hit=None):
    """For every short[i]: is it a substring (fwd or rc) of some long_[j]?

    Returns dict(hit=bool[n_short], first_long=int[n_short] (-1 if none), first_start=int[n_short],
    first_strand=int8[n_short], n_containing=int[n_short] (number of distinct long sequences that
    contain it), n_unanchored=int). `on_hit(i, j, start, strand)` is called once per (short, long)
    pair found (optional)."""
    vals, sid, st, of, unanch = anchors if anchors is not None else build_anchors(short)
    n = len(short)
    hit = np.zeros(n, bool)
    first_long = np.full(n, -1, np.int64); first_start = np.full(n, -1, np.int64)
    first_strand = np.full(n, -1, np.int8); n_cont = np.zeros(n, np.int64)
    if len(vals) == 0:
        return dict(hit=hit, first_long=first_long, first_start=first_start, first_strand=first_strand,
                    n_containing=n_cont, n_unanchored=unanch)
    rc_cache = {}
    for idx in _chunks_by_length(long_):
        batch = [long_[i] for i in idx]
        val, ok = windows(encode(batch))
        if val.shape[1] == 0:
            continue
        rr, cc = np.nonzero(ok)
        v = val[rr, cc]
        lo = np.searchsorted(vals, v, "left"); hi = np.searchsorted(vals, v, "right")
        m = hi > lo
        if not m.any():
            continue
        rr, cc, lo, hi = rr[m], cc[m], lo[m], hi[m]
        seen = set()
        for r, c, a, b in zip(rr.tolist(), cc.tolist(), lo.tolist(), hi.tolist()):
            g = batch[r]; j = int(idx[r])
            for t in range(a, b):
                i = int(sid[t]); s = short[i]
                start = c - int(of[t])
                if start < 0 or start + len(s) > len(g):
                    continue
                strand = int(st[t])
                if strand == 0:
                    target = s
                else:
                    target = rc_cache.get(i)
                    if target is None:
                        target = rc_cache[i] = rc(s)
                if g[start:start + len(s)] != target:
                    continue
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                n_cont[i] += 1
                if not hit[i]:
                    hit[i] = True; first_long[i] = j; first_start[i] = start; first_strand[i] = strand
                if on_hit is not None:
                    on_hit(i, j, start, strand)
    return dict(hit=hit, first_long=first_long, first_start=first_start, first_strand=first_strand,
                n_containing=n_cont, n_unanchored=unanch)
