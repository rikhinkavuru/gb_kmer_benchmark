#!/usr/bin/env python3
"""Cheap sanity checks for the neg_provenance module.

Run:  bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/neg_provenance/tests/test_sanity.py
(needs the verified caches in external/neg_provenance/cache and hg38.fa; ~1 min)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import genome_utils as G  # noqa: E402
import run_neg_provenance as RP  # noqa: E402
import _common as C  # noqa: E402


def test_interval_distance():
    ms, me = G.merge_intervals(np.array([100, 150, 400]), np.array([200, 180, 500]))
    assert list(ms) == [100, 400] and list(me) == [200, 500]
    d = G.dist_to_intervals(np.array([0, 150, 200, 250, 390, 600]), np.array([50, 160, 210, 260, 400, 700]), ms, me)
    # [0,50)->gap 50 to 100; overlap; [200,210) adjacent to end 200 -> 0; [250,260): 50 from 200, 140 to 400
    assert list(d) == [50, 0, 0, 50, 0, 100], d


def test_eligible_starts():
    class Fake(G.GeneIndex):
        def __init__(self):
            self.g_all = {"chrT": (np.array([1000]), np.array([2000]))}
            self.g_pc = self.g_all
    s, e = Fake().eligible_starts("chrT", 10, 100, 5000)
    # blocked window for [900,2100) -> starts 891..2099 blocked; admissible [0,891) and [2100,4991)
    assert list(s) == [0, 2100] and list(e) == [891, 4991], (s, e)
    q = np.array([890, 2100]); dd = G.dist_to_intervals(q, q + 10, *Fake().g_all["chrT"])
    assert (dd >= 100).all()


def test_isochores():
    assert list(G.iso_class([0.30, 0.37, 0.40, 0.45, 0.50, 0.60, np.nan])) == [0, 1, 1, 2, 3, 4, -1]


def test_sample_windows_uniform_over_nfree():
    """Global rejection: chromosome weights must follow N-free length, not raw length."""
    fa = {"chrA": "A" * 1000 + "N" * 9000, "chrB": "C" * 1000}      # equal N-free length, chrA 10x longer

    class F(dict):
        def __getitem__(self, k):
            return dict.__getitem__(self, k)
    elig = {c: (np.array([0]), np.array([len(s) - 10 + 1])) for c, s in fa.items()}
    df, tries = RP.sample_windows(F(fa), np.random.RandomState(0), 2000, 10, elig)
    assert df["seq"].str.contains("N").sum() == 0
    frac_a = (df["chr"] == "chrA").mean()
    assert 0.45 < frac_a < 0.55, frac_a                                   # old per-chrom rejection gave ~0.91
    assert tries > 2000


def test_gc_standardised_tata():
    edges = np.array([-1, 0.3, 0.5, 2.0])
    ref_gc = np.array([0.2, 0.2, 0.4, 0.4, 0.6, 0.6]); ref_f = np.array([1, 1, 1, 0, 0, 0], float)
    tgt_gc = np.array([0.2, 0.2, 0.2, 0.6]); tgt_f = np.array([1, 1, 1, 0], float)
    exp_, obs = RP.gc_standardised_tata(tgt_gc, tgt_f, ref_gc, ref_f, edges)
    assert abs(exp_ - 0.75) < 1e-9 and abs(obs - 0.75) < 1e-9


def test_best_tata_equals_paper_criterion():
    """best_tata >= 1.3 bits/pos  <=>  tata_hit_counts >= 1 (the paper's TATA-box flag)."""
    tbp = C.load_tbp()
    d = pd.read_csv(os.path.join(G.ROOT, "localization", "cohn", "test_negative.csv.gz")).head(300)
    fa = G.fasta(upper=True)
    seqs = [fa[r][int(s):int(e)] for r, s, e in zip(d.region, d.start, d.end)]
    a = RP.best_tata(seqs, tbp) >= RP.THR
    b = C.tata_hit_counts(seqs, tbp) >= 1
    assert (a == b).all() and 0 < a.sum() < len(a)


def test_ienhancer_originals_published_counts():
    import verify_inputs as V
    s = V.ienhancer_originals()
    assert 3268 <= len(s) <= 3368, len(s)       # 2968 benchmark + 400 independent (100 overlap)


def test_genome_gc():
    B = G.bins()
    acgt = sum(float(v[0].sum()) for v in B.values()); gc = sum(float(v[1].sum()) for v in B.values())
    assert 2.9e9 < acgt < 3.1e9, acgt
    assert 0.40 < gc / acgt < 0.42, gc / acgt          # hg38 genome-wide GC ~ 40.9%


def test_regional_gc_matches_direct_count():
    B = G.bins(); fa = G.fasta(upper=True)
    mid = 50_000_000
    v = G.regional_gc(np.array(["chr2"]), np.array([mid]), B)[0]
    s = fa["chr2"][mid - 50_000: mid + 50_000]
    direct = (s.count("G") + s.count("C")) / max(1, sum(s.count(x) for x in "ACGT"))
    assert abs(v - direct) < 0.002, (v, direct)


def test_chromhmm_lookup():
    H = G.chromhmm()
    s, e, st = H["Gm12878"]["chr1"]
    M = G.chromhmm_state_at(H, np.array(["chr1"]), np.array([float(s[1] + 1)]))
    assert M[0, 0] == st[1]


def test_header_parsers():
    """fix round 1: hg19 header coordinates and stripped UCSC custom-track 'range=' headers."""
    import nt_split_provenance as NS
    assert NS.parse_hdr_coord("CHRX_10705600_10705800") == ("chrX", 10705600, 10705800)
    assert NS.parse_hdr_coord("Chr11_6627824_6628024") == ("chr11", 6627824, 6628024)
    assert NS.parse_hdr_coord("hg19_ct_UserTrack_3545_1210") is None
    h = "hg19_ct_UserTrack_3545_158rangechr1278319870-783938695pad0"
    assert NS.parse_parent_range(h, "chr12") == ("chr12", 78319869, 78393869)   # 1-based closed -> BED
    assert NS.parse_parent_range(h, "chr1") == ("chr12", 78319869, 78393869)   # chr1 reading invalid -> unique chr12
    assert NS.parse_parent_range("hg19_ct_UserTrack_3545_7245 range=chr10:99167611-99168610 5'pad=0", None) == \
        ("chr10", 99167610, 99168610)
    assert NS.parse_parent_range("hg19_ct_UserTrack_3545_61rangechr1251538-2675375pad03pad0", "chr12") == \
        ("chr12", 51537, 267537)
    assert NS.parse_parent_range("hg19_ct_UserTrack_3545_321", "chr1") is None


def test_exact_segment_match_and_bp_fraction():
    import nt_split_provenance as NS
    H = {c: {"chrT": (np.array([0, 1000, 5000]), np.array([1000, 5000, 9000]), np.array([13, 13, 7], np.int8))}
         for c in G.CHROMHMM_CELLS}
    assert len(NS.exact_segment_matches(H, "chrT", 1000, 5000)) == 9
    assert len(NS.exact_segment_matches(H, "chrT", 1000, 4000)) == 0
    assert abs(NS.state_bp_fraction(H, "Gm12878", "chrT", 4000, 6000, 13) - 0.5) < 1e-12
    assert NS.window_states(H, "chrT", 4900, 5100)[0] == (7, 13)


def test_log10_chi2_sf_and_ci_columns():
    from scipy.stats import chi2
    for st, df in ((300.0, 23), (1400.0, 23)):              # finite region: identical to scipy
        assert abs(RP.log10_chi2_sf(st, df) - chi2.logsf(st, df) / np.log(10)) < 1e-9
    v = RP.log10_chi2_sf(2400.0, 23)                         # underflow region: finite and monotone
    assert np.isfinite(v) and v < RP.log10_chi2_sf(1400.0, 23)
    r = RP.put_ci({}, "x_ci", (0.1, 0.2))
    assert r["x_ci_lo"] == 0.1 and r["x_ci_hi"] == 0.2 and r["x_ci"] == "(0.1, 0.2)"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f(); print("PASS", name, flush=True)
