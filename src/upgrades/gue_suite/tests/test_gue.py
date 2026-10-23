#!/usr/bin/env python3
"""Sanity tests for the GUE module (CPU, no torch; about 30-60 s at GUE_THREADS=3).

  1. the analysed archive is the pinned one, and every GUE split size equals DNABERT-2 Table 12
     (covid is the one documented exception: same total, different train/dev/test partition);
  2. run_gue.select_k == diagnose._select_k, run_gue.comp_fraction == diagnose._composition_fraction,
     and run_gue.equalized_delta reproduces diagnose._equalized_delta's point estimates (on the small
     mouse_tf_3 task) -- i.e. the driver is the shipped tool, not a re-implementation that drifts;
  3. the driver reproduces a number printed in the paper: nt_promoter_tata, validation-selected
     k = 4, test MCC 0.8463 (results/results_nt.csv, lgbm, selected_by_val=1), at GUE_THREADS threads;
  4. post-hoc helpers (shuffle twins, identity/containment, arm indices);
  5. [fix round 1] seqshare.containment_scan == brute-force substring search (fwd + rc, with N and
     variable lengths), and the rule-sensitivity table reproduces every stored verdict (S0);
  6. [fix round 2] PH5 helpers (emp_provenance.py): exact genome location (fwd/rc, absent, repeated),
     midpoint location classes on a toy ORF table, the location-standardised excess (0 when two classes
     differ only in their stratum mix), the reading rule, and the archived Pham #positive/#negative table.
Run:  python src/upgrades/gue_suite/tests/test_gue.py
"""
import os
import sys

os.environ.setdefault("GUE_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_gue as R  # noqa: E402  (sets thread limits, paths)
import gue_data as G  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def test_archive_and_rowcounts():
    assert G.sha256(G.GUE_ZIP) == G.GUE_ZIP_SHA256, "GUE.zip differs from the analysed archive"
    for task in G.ORDER:
        n = tuple(len(pd.read_csv(os.path.join(G.task_dir(task), f"{s}.csv"))) for s in ("train", "dev", "test"))
        if task == "covid":
            assert sum(n) == sum(G.PAPER_SPLITS[task]), (task, n)
        else:
            assert n == G.PAPER_SPLITS[task], (task, n, G.PAPER_SPLITS[task])
    print("ok  archive SHA-256 pinned; 27/28 split sizes == Table 12, covid total == Table 12")


def test_driver_equals_shipped_tool():
    d = G.load_gue("mouse_tf_3")
    tr, ytr, te, yte = d["train_seqs"], d["y_train"], d["test_seqs"], d["y_test"]
    k, _ = R.select_k(tr, ytr)
    assert k == R.DG._select_k(tr, ytr, R.SEED), "select_k drifted from diagnose._select_k"
    cls, pr = R.fit_proba(R.kspec(tr, k), ytr, R.kspec(te, k))
    ccls, cpr = R.fit_proba(R.C.comp_signature(tr), ytr, R.C.comp_signature(te))
    mine = R.comp_fraction(yte, pr[:, 1], cpr[:, 1], 2, R.SEED, 200)
    ref = R.DG._composition_fraction(tr, ytr, te, yte, k, R.SEED, 200)
    assert abs(mine["value"] - ref["frac"]) < 1e-4 and mine["ci"] == [ref["frac_lo"], ref["frac_hi"]], (mine, ref)
    learn, _ = R.metrics_boot(yte, cls, pr, 2, R.SEED, 200)
    ref_fe = R.C.fit_eval_boot(R.kspec(tr, k), ytr, R.kspec(te, k), yte, "lgbm", R.SEED, 200,
                               np.random.RandomState(R.SEED))
    assert abs(learn["mcc"] - ref_fe["mcc"]) < 1e-6 and abs(learn["mcc_ci"][0] - ref_fe["mcc_lo"]) < 1e-6
    eq = R.equalized_delta(tr, ytr, te, yte, k, learn, R.SEED, 200)
    eq_ref = R.DG._equalized_delta(tr, ytr, te, yte, k, R.SEED, 200, np.random.RandomState(R.SEED))
    assert eq["feasible"] == eq_ref["feasible"]
    if eq["feasible"]:
        for key in ("mcc_before", "mcc_after", "retained_neg_frac", "max_abs_smd_after"):
            assert abs(eq[key] - eq_ref[key]) < 1e-3, (key, eq[key], eq_ref[key])
    print(f"ok  driver == diagnose.py on mouse_tf_3 (k={k}, comp fraction {mine['value']:+.4f}, "
          f"equalized MCC {eq.get('mcc_after')})")


def test_reproduces_paper_number():
    import data_nt
    tr, ytr = data_nt._read("promoter_tata", "train")
    te, yte = data_nt._read("promoter_tata", "test")
    k, vals = R.select_k(tr, ytr)
    cls, pr = R.fit_proba(R.kspec(tr, k), ytr, R.kspec(te, k))
    m, _ = R.metrics_boot(yte, cls, pr, 2, R.SEED, 200)
    ref = pd.read_csv(os.path.join(R.C.ROOT, "results", "results_nt.csv"))
    ref = ref[(ref.dataset == "nt_promoter_tata") & (ref.model == "lgbm") & (ref.selected_by_val == 1)]
    assert k == int(ref.k.iloc[0]), (k, vals)
    assert abs(m["mcc"] - float(ref.mcc.iloc[0])) < 5e-4, (m["mcc"], float(ref.mcc.iloc[0]))
    print(f"ok  nt_promoter_tata reproduced: k={k}, MCC {m['mcc']:.4f} vs paper {float(ref.mcc.iloc[0]):.4f}")


def test_posthoc_helpers():
    """Cheap unit tests (added 2026-09-27) for the post-hoc / overlap helpers."""
    import run_gue_posthoc as P
    import gue_overlap as O
    sys.path.insert(0, os.path.join(R.UP, "shuffled_neg"))
    import dinuc_shuffle as D
    rng = np.random.RandomState(0)
    seqs = ["".join(rng.choice(list("ACGT"), 101)) for _ in range(50)]
    shuf = [D.dinuc_shuffle(s, np.random.RandomState(i)) for i, s in enumerate(seqs)]
    ks, kh = P.dinuc_keys(seqs), P.dinuc_keys(shuf)
    assert all(a == b for a, b in zip(ks, kh)), "a dinucleotide shuffle must be a composition twin"
    assert len(set(ks)) == len(ks), "independent random sequences must not be twins"
    assert P.dinuc_keys(["ACGTN"])[0] != P.dinuc_keys(["ACGTA"])[0]
    # identity / containment, forward and reverse complement
    a = ["ACGTAC", "GGGCCC", "TTTTAA"]
    b = ["GTACGT", "GGGCCC"]                                   # rc(ACGTAC) = GTACGT
    assert list(O.identical(a, b)) == [True, True, False]
    assert list(O.contained(["CGTA", "AAAA", "TACG"], ["ACGTAC"], 4)) == [True, False, True]
    # arm indices == CC.negatives_kept_split order
    y = np.array([1, 0, 0, 1, 0, 1]); keep = np.array([0, 2])
    idx = P.arm_indices(y, keep)
    s_ = [str(i) for i in range(6)]
    ref, _ = P.CC.negatives_kept_split(s_, y, keep)
    assert [s_[i] for i in idx] == ref
    print("ok  post-hoc helpers: shuffle twins, identity/containment (fwd+rc), arm indices")


def test_seqshare_and_rule_reproduction():
    """[fix round 1] exact containment scan vs brute force; stored verdicts reproduced by diagnose._verdict."""
    import seqshare as S
    rng = np.random.RandomState(1)

    def rs(n):
        return "".join(rng.choice(list("ACGT"), n))
    longs = [rs(int(rng.randint(40, 500))) for _ in range(800)] + ["ACGT" * 10 + "N" + "ACGTTTGCA" * 10]
    shorts = []
    for i in range(600):
        L = int(rng.choice([50, 70, 101, 300]))
        g = longs[int(rng.randint(0, 800))]
        if len(g) >= L and i % 3 == 0:
            st = int(rng.randint(0, len(g) - L + 1)); shorts.append(g[st:st + L])
        elif len(g) >= L and i % 3 == 1:
            st = int(rng.randint(0, len(g) - L + 1)); shorts.append(S.rc(g[st:st + L]))
        else:
            shorts.append(rs(L))
    shorts += [longs[-1][2:80], S.rc(longs[-1][5:70])]
    bf = np.array([any((s in g) or (S.rc(s) in g) for g in longs) for s in shorts])
    res = S.containment_scan(shorts, longs)
    assert np.array_equal(bf, res["hit"]), "containment_scan disagrees with brute force"
    hit = np.where(res["hit"])[0]
    for i in hit[:50]:
        g = longs[res["first_long"][i]]; st = res["first_start"][i]; s = shorts[i]
        assert g[st:st + len(s)] == (s if res["first_strand"][i] == 0 else S.rc(s))
    rsp = os.path.join(R.C.RESULTS_DIR, "gue_rule_sensitivity.csv")
    if os.path.exists(rsp):
        import json
        import diagnose as DG
        rs_ = pd.read_csv(rsp).set_index("task")
        for t in G.ORDER:
            c = json.load(open(os.path.join(R.CARD_DIR, f"{t}.json")))
            assert DG._verdict(c) == c["verdict"], t
        print(f"ok  containment_scan == brute force ({int(bf.sum())}/{len(bf)} contained); "
              f"28/28 stored GUE verdicts reproduced ({len(rs_)} rows in gue_rule_sensitivity.csv)")
    else:
        print(f"ok  containment_scan == brute force ({int(bf.sum())}/{len(bf)} contained)")


def test_emp_provenance_helpers():
    """[fix round 2] PH5 helpers on toy data (no network, no GUE data needed except the Pham page)."""
    import emp_provenance as E
    rng = np.random.RandomState(3)
    chrI = "".join(rng.choice(list("ACGT"), 3000)); chrII = "".join(rng.choice(list("ACGT"), 2000))
    rep = chrII[100:160]
    chrII = chrII[:1500] + rep + chrII[1560:]                       # a 60-bp repeat -> 2 hits
    g, offs, keys, poss, lut = E.build_index(["chrI", "chrII"], [chrI, chrII])
    q = {"fwd": chrI[200:700], "rc": E.rc(chrII[900:1400]), "absent": "".join(rng.choice(list("ACGT"), 500)),
         "repeat": rep}
    loc = E.locate(list(q.values()), g, offs, ["chrI", "chrII"], keys, poss, lut)
    assert loc[q["fwd"]] == ("chrI", 200, "+", 1), loc[q["fwd"]]
    assert loc[q["rc"]] == ("chrII", 900, "-", 1), loc[q["rc"]]
    assert loc[q["absent"]][0] is None and loc[q["absent"]][3] == 0
    assert loc[q["repeat"]][0] is None and loc[q["repeat"]][3] == 2
    orfs = {"chrI": (np.array([1000, 5000]), np.array([2000, 6000]), np.array(["+", "-"]))}
    cases = {1500: ("ORF", "ORF rest"), 1200: ("ORF", "ORF 5'"), 800: ("intergenic", "promoter"),
             6300: ("intergenic", "promoter"), 3000: ("intergenic", "intergenic other"), 5900: ("ORF", "ORF 5'"),
             5300: ("ORF", "ORF rest")}
    for mid, want in cases.items():
        assert E.classify("chrI", mid, orfs) == want, (mid, E.classify("chrI", mid, orfs), want)
    # same within-stratum TBP rates, different stratum mix -> marginal excess > 0, standardised excess 0
    s1 = np.array(["ORF"] * 80 + ["intergenic"] * 20); s0 = np.array(["ORF"] * 20 + ["intergenic"] * 80)
    h1 = np.r_[np.r_[np.ones(40), np.zeros(40)], np.r_[np.ones(16), np.zeros(4)]]
    h0 = np.r_[np.r_[np.ones(10), np.zeros(10)], np.r_[np.ones(64), np.zeros(16)]]
    marg = h0.mean() - h1.mean()
    assert abs(marg - 0.18) < 1e-12 and abs(E.std_excess(h1, s1, h0, s0, ["ORF", "intergenic"])) < 1e-12
    assert E.verdict(0.01, 0.09) == "explained by location" and E.verdict(0.10, 0.2) == "not explained"
    assert E.verdict(0.05, 0.12) == "partly explained"
    if os.path.exists(os.path.join(E.PHAM, "members_wayback_20070715.htm")):
        pc = E.pham_counts()
        assert set(pc) == set(E.MARKS) and pc["H3"] == (7667, 7298) and pc["H3K79me3"] == (15337, 13500), pc
    print("ok  PH5 helpers: exact genome location (fwd/rc/absent/repeat), location classes, standardised excess, "
          "reading rule, Pham counts")


if __name__ == "__main__":
    test_archive_and_rowcounts()
    test_driver_equals_shipped_tool()
    test_reproduces_paper_number()
    test_posthoc_helpers()
    test_seqshare_and_rule_reproduction()
    test_emp_provenance_helpers()
    print("all GUE sanity tests passed")
