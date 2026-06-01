"""Load JASPAR CORE PWMs via pyjaspar and convert them to log-odds PSSMs (numpy).

Pure-Python / CPU only. Motifs come from the JASPAR SQLite DB bundled with
pyjaspar (offline after install). Each count matrix (4 x L) becomes a (4, L)
float32 log-odds PSSM under a uniform 0.25 background with a fixed pseudocount:

    PPM[b, j]  = (count[b, j] + pc) / (sum_b count[b, j] + 4*pc)
    PSSM[b, j] = log2( PPM[b, j] / 0.25 )

Reverse-complement matching is handled at the k-mer level in motif_match.py, so
only the forward PSSM is stored here.
"""
import numpy as np

BASES = "ACGT"
RELEASE = "JASPAR2024"
PSEUDOCOUNT = 0.5
BACKGROUND = 0.25


def counts_to_pssm(counts_4xL, pseudocount=PSEUDOCOUNT, background=BACKGROUND):
    c = counts_4xL.astype(np.float64) + pseudocount
    ppm = c / c.sum(axis=0, keepdims=True)
    return np.log2(ppm / background).astype(np.float32)


def load_pssms(tax_group, release=RELEASE, collection="CORE", pseudocount=PSEUDOCOUNT):
    """Return a list of dicts {name, matrix_id, length, pssm(4xL)} for one tax group.

    tax_group: 'vertebrates' or 'insects' (or any JASPAR tax_group string).
    """
    from pyjaspar import jaspardb
    jdb = jaspardb(release=release)
    motifs = jdb.fetch_motifs(collection=collection, tax_group=tax_group)
    out = []
    for m in motifs:
        counts = np.array([list(m.counts[b]) for b in BASES], dtype=np.float64)  # (4, L)
        if counts.shape[1] == 0:
            continue
        out.append(dict(name=str(m.name), matrix_id=str(m.matrix_id),
                        length=int(counts.shape[1]),
                        pssm=counts_to_pssm(counts, pseudocount)))
    return out
