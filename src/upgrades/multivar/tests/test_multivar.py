#!/usr/bin/env python3
"""Cheap sanity tests for the multivar module (torch-free; < 30 s).  test_outputs_consistent reads the
result CSVs, so run the tests after run_multivar.py (run_pipeline.sh does).
Run:  bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/multivar/tests/test_multivar.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
sys.path.insert(0, MOD)
import _mv  # noqa: E402
import run_multivar as RM  # noqa: E402
import measure_homology_all as MH  # noqa: E402


def rand_seqs(rng, n, L, with_n=True):
    out = []
    for _ in range(n):
        s = "".join(rng.choice(list("ACGT"), L))
        if with_n and rng.rand() < 0.3:
            i = rng.randint(0, L)
            s = s[:i] + "N" + s[i + 1:]
        out.append(s)
    return out


def test_gc_fraction_matches_common():
    rng = np.random.RandomState(0)
    seqs = rand_seqs(rng, 200, 57) + ["NNNN", "GGCC", "ATAT"]
    a = _mv.gc_fraction(seqs)
    b = np.asarray(_mv.C.gc_content(seqs), float)
    assert np.allclose(a, b, atol=1e-12), np.max(np.abs(a - b))


def test_loto_press_equals_bruteforce():
    rng = np.random.RandomState(1)
    for trial in range(30):
        n = rng.choice([11, 15])
        p = rng.randint(1, 4)
        X = rng.randn(n, p)
        if trial % 3 == 0:                       # binary column with a single 1 (constant in one fold)
            X[:, 0] = 0.0
            X[rng.randint(n), 0] = 1.0
        if trial % 3 == 1:                       # binary column with two 1s
            X[:, -1] = 0.0
            X[rng.choice(n, 2, replace=False), -1] = 1.0
        y = rng.randn(n)
        a = RM.loto_pred(X, y)
        b = RM.loto_pred_bruteforce(X, y)
        assert np.allclose(a, b, atol=1e-8), (trial, np.max(np.abs(a - b)))


def test_route1_reproduction():
    cs = pd.read_csv(os.path.join(_mv.ROOT, "results", "cross_suite_summary.csv"))
    tf = cs[cs["tf_motif_task"]].reset_index(drop=True)
    y = tf["bench_mcc"].to_numpy(float)
    pr = RM.loto_pred(tf[["max_pos_motif_auroc"]].to_numpy(float), y)
    r2, mae = RM.loto_metrics(y, pr)
    assert abs(r2 - 0.439) < 5e-4 and abs(mae - 0.140) < 5e-4, (r2, mae)


def test_partial_spearman_matches_rank_residuals():
    rng = np.random.RandomState(2)
    x, z = rng.randn(15), rng.randn(15)
    y = x + 0.5 * z + rng.randn(15)
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    assert abs(RM.partial_spearman(x, y, z) - np.corrcoef(ex, ey)[0, 1]) < 1e-10


def test_lmg_sums_to_r2_and_aicc():
    rng = np.random.RandomState(3)
    X = rng.randn(11, 3)
    y = X @ np.array([1.0, 0.5, 0.0]) + rng.randn(11)
    sh, r2 = RM.lmg(X, y)
    assert abs(sh.sum() - r2) < 1e-12
    _, _, rss = RM.fit(X, y)
    n, p = 11, 3
    K = p + 2
    assert abs(RM.aicc(rss, n, p) - (n * np.log(rss / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1))) < 1e-12
    assert len(RM.subsets(list(range(10)))) == 175
    assert len(RM.subsets(list(range(11)))) == 231
    assert len(RM.subsets(list(range(10)), anchor=0)) == 46


def test_sparse_max_jaccard_equals_dense_reference():
    rng = np.random.RandomState(4)
    tr = rand_seqs(rng, 300, 60)
    te = rand_seqs(rng, 50, 60) + [tr[5], tr[7][:40] + "".join(rng.choice(list("ACGT"), 20))]
    Q, T = MH.H.presence_matrix(te), MH.H.presence_matrix(tr)
    dense = MH.H.max_jaccard_to_train(Q, T)
    # chunked running max, exactly as run_task does
    sp = np.zeros(len(te))
    for c0 in range(0, len(tr), 70):
        sp = np.maximum(sp, MH.max_jaccard_sparse(Q, MH.H.presence_matrix(tr[c0:c0 + 70]), batch=16))
    assert np.allclose(dense, sp, atol=1e-12), np.max(np.abs(dense - sp))
    assert abs(sp[-2] - 1.0) < 1e-12


def test_fisher_z_loto_bounded():
    rng = np.random.RandomState(5)
    x = rng.rand(11, 1)
    y = np.clip(2.0 * x[:, 0] - 0.4 + 0.1 * rng.randn(11), -0.98, 0.98)
    pr = RM.fisher_z_loto(x, y)
    assert np.all(np.abs(pr) < 1)
    assert np.allclose(pr, np.tanh(RM.loto_pred_bruteforce(x, np.arctanh(y))), atol=1e-10)


def test_exact_dup_flags():
    import audit_exact_dups as AD
    tr = ["AACCGGTT", "ACGTACGA", "GGGGCCCC", "TTTTAAAC"]
    ytr = np.array([1, 0, 1, 0])
    te = ["AACCGGTT", "TCGTACGT", "GGGGCCCC", "CATGCATG", "GTTTAAAA"]
    yte = np.array([1, 0, 0, 1, 1])
    fwd, rc, anyd, conc = AD.exact_dup_flags(tr, ytr, te, yte)
    assert fwd.tolist() == [True, False, True, False, False]
    assert rc.tolist() == [False, True, False, False, True]       # palindromes count as forward matches only
    assert anyd.tolist() == [True, True, True, False, True]
    assert conc.tolist() == [True, True, False, False, False]      # GGGGCCCC label differs; GTTTAAAA rc label 0
    assert AD.revcomp("ACGTN") == "NACGT"


def test_prevalence_std_mcc_equals_mcc_on_full_set():
    import dedup_eval as DE
    rng = np.random.RandomState(6)
    y = rng.randint(0, 3, 300)
    proba = rng.dirichlet([1, 1, 1], 300)
    proba[np.arange(300), y] += 0.3
    full = DE.subset_metrics(y, proba, np.ones(300, bool), 3)
    assert abs(full["mcc"] - full["mcc_prevalence_std"]) < 1e-12
    m = (y != 1) | (rng.rand(300) < 0.3)                         # subset with class 1 under-represented
    sub = DE.subset_metrics(y, proba, m, 3)
    assert sub["n_class1"] < full["n_class1"] and np.isfinite(sub["mcc_prevalence_std"])


def test_preregistration_hash_unchanged():
    import hashlib
    with open(os.path.join(MOD, "run_multivar.py"), "rb") as fh:
        head = b"".join(fh.readlines()[:99])
    with open(os.path.join(_mv.RESULTS_DIR, "multivar_preregistration.txt")) as fh:
        txt = fh.read()
    assert hashlib.sha256(head).hexdigest() in txt


def test_partial_boot_degenerate_is_nan_not_crash():
    x = np.arange(7.0)
    y = x ** 2                                   # y a perfect monotone function of the control x
    z = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0])
    v, p, lo, hi = RM.partial_boot(z, y, x)      # rho(z, y | x) undefined
    assert np.isnan(v) and np.isnan(p) and np.isnan(lo) and np.isnan(hi)
    v2, p2, lo2, hi2 = RM.partial_boot(x, y, z)  # rho(x, y | z) = 1: degenerate, p undefined (fix round 2)
    assert abs(v2 - 1) < 1e-12 and np.isnan(p2)
    assert np.isnan(lo2) and np.isnan(hi2)       # fix round 3: and no [+1, +1] percentile CI
    assert np.isnan(RM.perm_partial_p(x, y, z)[0]) and np.isnan(RM.perm_partial_p(z, y, x)[0])


def test_perm_spearman_p_exact_and_mc():
    """fix rounds 2-3: the exact permutation p (DP over all n! orderings) == the brute-force loop over all
    orderings, with and without ties; == the round-2 enumeration (perm_corr_p on ranks) at n = 9; rho = 1 at
    n = 7 gives 2/7!; the Monte Carlo branch agrees with the exact value at n = 9 within MC error."""
    import itertools
    rng = np.random.RandomState(7)
    for n in (4, 5, 6, 7, 8):
        for ties in (False, True):
            a, b = rng.randn(n), rng.randn(n)
            if ties:
                a, b = np.round(a), np.round(b * 0.7)
            r0 = stats.spearmanr(a, b)[0]
            if r0 != r0:
                continue
            rs = np.array([stats.spearmanr(a, b[list(q)])[0] for q in itertools.permutations(range(n))])
            p_loop = float(np.mean(np.abs(rs) >= abs(r0) - 1e-12))
            p_vec, meth = RM.perm_spearman_p(a, b)
            assert meth.startswith("exact") and abs(p_vec - p_loop) < 1e-12, (n, ties, p_vec, p_loop)
    a, b = rng.randn(9), rng.randn(9)
    assert abs(RM.perm_spearman_p(a, b)[0] - RM.perm_corr_p(stats.rankdata(a), stats.rankdata(b))[0]) < 1e-12
    x = np.arange(7.0)
    assert abs(RM.perm_spearman_p(x, x ** 3)[0] - 2 / 5040) < 1e-15
    assert np.isnan(RM.spearman_t(x, x ** 3)[1])                 # scipy's p = 0 is not reported
    rng = np.random.RandomState(7)                                 # the round-2 draws (44 normals, then n = 9)
    rng.randn(44)
    a, b = rng.randn(9), rng.randn(9)
    b = a + 2.0 * b                                                # moderate association: exact p ~ 0.037
    pe, me = RM.perm_spearman_p(a, b)
    pm, mm = RM.perm_corr_p(stats.rankdata(a), stats.rankdata(b), n_mc=50000, exact_max=8)
    assert me.startswith("exact") and mm.startswith("Monte Carlo")
    assert abs(pe - pm) < 0.005 and 0.01 < pe < 0.1, (pe, pm)   # MC SE at p ~ 0.04, 50,000 draws: 0.0009


def test_perm_partial_residuals_and_holm():
    rng = np.random.RandomState(8)
    x, z = rng.randn(8), rng.randn(8)
    y = x + 0.5 * z + rng.randn(8)
    ex, ey = RM.rank_residuals(x, z), RM.rank_residuals(y, z)
    assert abs(np.corrcoef(ex, ey)[0, 1] - RM.partial_spearman(x, y, z)) < 1e-10
    p, meth = RM.perm_partial_p(x, y, z)
    assert 0 < p <= 1 and meth.startswith("exact")
    adj = RM.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_table1_routes_and_distinct_sets():
    r = RM.table1_routes()
    assert set(r) == set(_mv.TASKS)
    assert {t for t, v in r.items() if v == "solved"} == {"human_ensembl_regulatory", "nt_promoter_all",
                                                          "nt_promoter_no_tata", "human_nontata_promoters",
                                                          "nt_promoter_tata"}
    cov = pd.read_csv(os.path.join(_mv.RESULTS_DIR, "multivar_covariates.csv"))
    s11 = list(cov.loc[cov["S11"], "task"])
    assert RM.n_distinct_sets(s11) == (9, 8)
    # fix round 3: the union is redundant only with BOTH components present
    assert RM.n_distinct_sets(["nt_promoter_all", "nt_promoter_tata", "nt_enhancers"]) == (3, 2)
    assert RM.n_distinct_sets(["nt_promoter_all", "nt_promoter_tata", "nt_promoter_no_tata"]) == (2, 2)
    assert RM.n_distinct_sets(["human_ensembl_regulatory", "human_ocr_ensembl", "nt_enhancers",
                               "nt_enhancers_types"]) == (3, 2)
    assert sorted(map(sorted, RM.max_independent_sets(["a", "b", "c"], [("a", "b"), ("b", "c")]))) == [["a", "c"]]


def test_homology_clean_labels_and_overlap():
    """fix round 2: only task sets without a >= 0.05 homology task carry homology_clean=True; the task
    identities that P12 relies on hold."""
    rd = _mv.RESULTS_DIR
    p5 = pd.read_csv(os.path.join(rd, "multivar_posthoc_homology_clean.csv"))
    for _, r in p5.iterrows():
        kept = r["kept_homology_ge_0_05"] if isinstance(r["kept_homology_ge_0_05"], str) else ""
        assert bool(r["homology_clean"]) == (kept == ""), (r["subset"], kept)
    assert list(p5.loc[p5["homology_clean"], "subset"]) == ["rule_clean_homology_lt_0.05"]
    p12 = pd.read_csv(os.path.join(rd, "multivar_posthoc_distinct_datasets.csv"))
    cov = pd.read_csv(os.path.join(rd, "multivar_covariates.csv")).set_index("task")
    for _, r in p12.iterrows():
        assert bool(r["homology_clean"]) == bool((cov.loc[r["tasks"].split(";"), "homology"] < 0.05).all())
    idf = pd.read_csv(os.path.join(rd, "multivar_posthoc_task_identity.csv"))
    assert (idf["value"] == 1).all() and len(idf) == 8
    fam = pd.read_csv(os.path.join(rd, "multivar_posthoc_family.csv"))
    b = fam[fam["analysis"].str.startswith("between")]
    assert b["indicator_equals_table1_solved"].all() and b["indicator_is_mcc_dichotomy"].all()


def test_outputs_consistent():
    """Run after run_multivar.py: A1 reverse partial == P7 (same function), nested fold counts match the
    selected-model strings, and the decision row is unique."""
    rd = _mv.RESULTS_DIR
    pa = pd.read_csv(os.path.join(rd, "multivar_partial.csv"))
    g7 = pd.read_csv(os.path.join(rd, "multivar_posthoc_gc_vs_disc.csv"))
    for cfg in ("S11/Y1_MCC", "S15/Y1_MCC"):
        a = pa[(pa["config"] == cfg) & (pa["covariate"] == "gc_mean")].iloc[0]
        b = g7[(g7["config"] == cfg) & (g7["variant"] == "as_published")].iloc[0]
        for ca, cb in (("posthoc_rev_partial_rho_c_given_disc", "partial_gc_given_disc"),
                       ("posthoc_rev_partial_boot_lo", "partial_gc_given_disc_lo"),
                       ("posthoc_rev_partial_boot_hi", "partial_gc_given_disc_hi"),
                       ("partial_rho", "partial_disc_given_gc"), ("partial_boot_lo", "partial_disc_given_gc_lo"),
                       ("partial_boot_hi", "partial_disc_given_gc_hi")):
            assert abs(a[ca] - b[cb]) < 1e-12, (cfg, ca, a[ca], b[cb])
    ne = pd.read_csv(os.path.join(rd, "multivar_nested.csv"))
    assert ne["in_decision_rule"].sum() == 1
    # fix round 3: the S11 Spearman p is exact (DP over all 11! orderings); a verifier's independent DP gave 0.012774
    p5 = pd.read_csv(os.path.join(rd, "multivar_posthoc_homology_clean.csv")).set_index("subset")
    assert abs(p5.loc["S11_all", "spearman_p_perm"] - 0.012774) < 5e-7, p5.loc["S11_all", "spearman_p_perm"]
    assert p5["spearman_p_perm_method"].str.startswith("exact").all()
    # degenerate |rho| = 1 partials carry neither p nor CI
    q7 = p5.loc["rule_clean_homology_lt_0.05"]
    assert np.isnan(q7["partial_given_gc_mean_p"]) and np.isnan(q7["partial_given_gc_mean_lo"]) and np.isnan(q7["partial_given_gc_mean_hi"])
    p13 = pd.read_csv(os.path.join(rd, "multivar_posthoc_locus_independent.csv"))
    lk = pd.read_csv(os.path.join(rd, "multivar_posthoc_task_locus_links.csv"))
    for tau, g in p13.groupby("tau"):                  # every reported set is mutually unlinked at its threshold
        E = {frozenset((a, b)) for a, b, v in zip(lk["task_a"], lk["task_b"], lk["locus_link"]) if v >= tau}
        for t_ in g["tasks"]:
            ts = t_.split(";")
            assert not any(frozenset((a, b)) in E for i, a in enumerate(ts) for b in ts[i + 1:]), (tau, t_)
    for _, r in ne.iterrows():
        parts = [p.rsplit(" x", 1) for p in r["selected_models"].split("; ")]
        n_disc = sum(int(k) for m, k in parts if "disc" in m.split(" + "))
        n_gc = sum(int(k) for m, k in parts if "gc_mean" in m.split(" + "))
        assert sum(int(k) for _, k in parts) == r["n"]
        assert n_disc == r["n_folds_selected_with_disc"] and n_gc == r["n_folds_selected_with_gc_mean"], r["selected_models"]


def test_locus_sketch_equals_bruteforce():
    """fix round 3: audit_locus_overlap's vectorized canonical, 1/4-sampled 31-mer sets and per-query
    containment == a plain-Python reference (both strands, N-containing k-mers skipped)."""
    import audit_locus_overlap as AL
    rng = np.random.RandomState(9)
    comp = str.maketrans("ACGT", "TGCA")

    def rs(L):
        return "".join(rng.choice(list("ACGT"), L))

    def code(w):
        return int("".join("0123"["ACGT".index(c)] for c in w), 4)

    def kmers_py(q):
        out = set()
        for i in range(len(q) - AL.K + 1):
            w = q[i:i + AL.K]
            if set(w) - set("ACGT"):
                continue
            c = min(code(w), code(w.translate(comp)[::-1]))
            if ((c * 0x9E3779B97F4A7C15) % (1 << 64)) >> AL.SAMPLE_SHIFT == 0:
                out.add(c)
        return out
    ref = [rs(rng.randint(40, 300)) for _ in range(40)]
    qs = [ref[3][10:200], rs(150), ref[5][::-1].translate(comp), "N" * 50 + ref[9][:120], rs(20),
          ref[1][:60] + rs(90)]
    R = AL.reference_set(ref)
    Rpy = set().union(*[kmers_py(q) for q in ref])
    assert set(R.tolist()) == Rpy
    km, own, cnt = AL.query_sketches(qs)
    c = AL.containment(AL.member(R, km), own, cnt)
    for q, v in zip(qs, c):
        k = kmers_py(q)
        assert (np.isnan(v) and not k) or abs(v - len(k & Rpy) / len(k)) < 1e-12


def test_gc_disc_readout():
    """fix round 3: the MAE ordering and its single-task reversals are what a direct recomputation gives."""
    rng = np.random.RandomState(10)
    x, g = rng.rand(9), rng.rand(9)
    y = 0.5 * x + 0.5 * g + 0.1 * rng.randn(9)
    tk = [f"t{i}" for i in range(8)] + [RM.REG]
    r = RM.gc_disc_readout(tk, x, g, y)
    md = RM.loto_metrics(y, RM.loto_pred(x[:, None], y))[1]
    mg = RM.loto_metrics(y, RM.loto_pred(g[:, None], y))[1]
    assert abs(r["loto_MAE_disc"] - md) < 1e-12 and r["loto_MAE_lower"] == ("gc" if mg < md else "disc")
    for t in r["loto_MAE_order_reversed_by_dropping"].split(";"):
        if t:
            k = np.array([u != t for u in tk])
            a = RM.loto_metrics(y[k], RM.loto_pred(x[k][:, None], y[k]))[1]
            b = RM.loto_metrics(y[k], RM.loto_pred(g[k][:, None], y[k]))[1]
            assert ("gc" if b < a else "disc") != r["loto_MAE_lower"]
    assert r["rho_gc_minus_rho_disc_lo"] <= r["rho_gc_minus_rho_disc"] + 0.5
    assert abs(r["spearman_gc_mcc"] - stats.spearmanr(g, y)[0]) < 1e-12


def test_class_gc_stats():
    x = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert abs(_mv.pooled_within_sd(x, y) - 0.1) < 1e-12
    assert abs(_mv.max_pairwise_cohens_d(x, y) - 4.0) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"all {len(fns)} tests passed")
