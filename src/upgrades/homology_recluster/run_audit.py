#!/usr/bin/env python3
"""Step 9 -- VERIFICATION AUDIT of the homology results (written 2026-09-24 ~11:30, after independent
verifiers reviewed homology_recluster_{eval,route_labels,route1}.csv; everything here is POST HOC and
labelled as such in the report). No model is fitted: it reads the cached predictions of run_eval.py /
run_seeds.py, the component files of run_cluster.py and the sequences.

Questions and fixed readouts
----------------------------
Q1 WHAT DOES THE PURGE REMOVE, PER CLASS? For every task, threshold and class: n test sequences, fraction
   removed by the purge, fraction that are EXACT (either-strand) copies of a SAME-label training sequence,
   and the ORIGINAL model's accuracy (= per-class recall) on the removed vs the kept sequences. A purge that
   removes every sequence of a class (or leaves < 10) is flagged: its purged MCC scores a K-class model on a
   test set that lacks a class (run_eval.py now marks such rows status='class_eliminated').
Q2 (the nt_enhancers subtype-matched null lives in run_eval.py, NULL_STRATA; not duplicated here.)
Q3 EXACT DUPLICATES MISSED BY MMseqs2. MMseqs2 cannot seed sequences shorter than k = 11 or made of N (masked),
   so identical (either-strand) test/train pairs can sit in different components. Per task and
   threshold: number of test sequences that are exact copies of a train sequence but NOT purged, the MCC of
   the purge when they are also removed ("purge+exact"), the number of pooled sequences whose exact twin
   sits on the other side of the seed-42 cluster-aware re-split, length statistics, and the number of sequences
   that are >= 50 % N (per label) -- such sequences cannot be aligned but are trivially identical.
Q4 KIND OF HOMOLOGY. For purged test sequences with a direct train edge: median best train identity and the
   fraction whose best train hit is >= 0.99 identity (near-identical loci / shifted windows) vs family-level
   homology (0.8-0.9).
Q5 COMPONENT-LEVEL BOOTSTRAP. Test sequences in one homology component are not independent. 95 % percentile
   CIs (1000 resamples, seed 42) resampling whole id80/id50 components (test sequences grouped by component)
   for: original MCC, purged MCC, cluster-aware re-split MCC, random re-split MCC and cluster - random
   (independent component resamples of the two test sets), next to the sequence-level CIs of run_eval.py.

Outputs: results/upgrades/homology_recluster_audit_{class,task,clusterboot}.csv
Checkpoints: external/homology_recluster/ckpt/audit/<task>.pkl. Usage: python run_audit.py [--tasks a,b]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402
from run_eval import PRED_DIR, pred_of, std_mcc  # noqa: E402
from run_cluster import EDGE_DIR, SIZE_ORDER  # noqa: E402

CK = os.path.join(_hc.CKPT_DIR, "audit")
THS = ("id80", "id50")
MIN_CLASS_KEPT = 10
_RC = str.maketrans("ACGTN", "TGCAN")


def canon(s):
    return min(s, s.translate(_RC)[::-1])


def fast_mcc(y, p, K):
    """Multiclass MCC (Gorodkin) from the confusion matrix; equals sklearn.matthews_corrcoef."""
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(np.float64)
    t = cm.sum(1); q = cm.sum(0); c = np.trace(cm); s = cm.sum()
    num = c * s - np.dot(t, q)
    den = np.sqrt(s * s - np.dot(q, q)) * np.sqrt(s * s - np.dot(t, t))
    return float(num / den) if den > 0 else 0.0


def comp_groups(idx, comp):
    """Group the positions `idx` by component: returns (order, starts, sizes) for fast resampling."""
    c = comp[idx]
    o = np.argsort(c, kind="stable")
    cs = c[o]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(cs)])
    return idx[o], starts, sizes


def comp_boot(y, p, K, groups, rng, B=_hc.BOOT):
    """MCC percentile CI resampling whole components (y, p aligned with groups[0])."""
    order, starts, sizes = groups
    yo, po = y, p
    out = np.full(B, np.nan)
    G = len(starts)
    for b in range(B):
        g = rng.randint(0, G, G)
        ln = sizes[g]
        pos = np.repeat(starts[g] - np.r_[0, np.cumsum(ln)[:-1]], ln) + np.arange(ln.sum())
        yy, pp = yo[pos], po[pos]
        if len(np.unique(yy)) < 2:
            continue
        out[b] = fast_mcc(yy, pp, K)
    return out


def q(a, p):
    return float(np.nanpercentile(a, p)) if np.isfinite(a).any() else float("nan")


def load_pred(task, unit):
    f = os.path.join(PRED_DIR, f"{task}__{unit}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return z["te_idx"], pred_of(z["proba"], z["classes"])


def do_task(task):
    f = os.path.join(CK, f"{task}.pkl")
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    tr, ytr, te, yte = _hc.load_task(task)
    n_tr, n = len(tr), len(tr) + len(te)
    y = np.concatenate([ytr, yte]); K = int(y.max()) + 1
    labs = np.unique(y)
    cz = [canon(s) for s in tr + te]
    L = np.array([len(s) for s in tr + te])
    nfrac = np.array([s.count("N") / max(len(s), 1) for s in tr + te])
    del tr, te
    trset = set(cz[:n_tr])
    tr_lab = {}
    for c, l in zip(cz[:n_tr], ytr):
        tr_lab.setdefault(c, set()).add(int(l))
    ex_any = np.array([c in trset for c in cz[n_tr:]])
    ex_same = np.array([int(l) in tr_lab.get(c, ()) for c, l in zip(cz[n_tr:], yte)])
    gid = pd.factorize(pd.Series(cz))[0]
    del cz, trset, tr_lab
    te_idx0, pred0 = load_pred(task, "original")
    assert np.array_equal(te_idx0, np.arange(n_tr, n))
    correct = pred0 == yte
    ez = np.load(os.path.join(EDGE_DIR, f"{task}.npz"))
    ea, eb, ef = ez["a"].astype(np.int64), ez["b"].astype(np.int64), ez["fid"].astype(np.int64)
    ev = pd.read_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_eval.csv"))
    ev = ev[ev.task == task]
    cls_rows, task_rows, boot_rows = [], [], []
    test_frac = (n - n_tr) / n
    for th in THS:
        comp, is_test = _hc.load_clusters(task, th)
        has_tr = np.zeros(comp.max() + 1, bool); has_tr[comp[:n_tr]] = True
        removed = has_tr[comp[n_tr:]]
        keep = ~removed
        # ---------- Q1 per class
        kept_per_class = {}
        for l in labs:
            m = yte == l
            kept_per_class[int(l)] = int((m & keep).sum())
            cls_rows.append(dict(
                task=task, threshold=th, cls=int(l), n_test=int(m.sum()), n_removed=int((m & removed).sum()),
                frac_removed=round(float(removed[m].mean()), 5),
                frac_exact_same_label=round(float(ex_same[m].mean()), 5),
                frac_exact_any=round(float(ex_any[m].mean()), 5),
                acc_removed=round(float(correct[m & removed].mean()), 5) if (m & removed).any() else np.nan,
                acc_kept=round(float(correct[m & keep].mean()), 5) if (m & keep).any() else np.nan,
                acc_all=round(float(correct[m].mean()), 5)))
        class_eliminated = bool(min(kept_per_class.values()) < MIN_CLASS_KEPT)
        # ---------- Q3 exact duplicates missed
        missed = ex_any & keep
        keep2 = keep & ~ex_any
        m_purge = fast_mcc(yte[keep], pred0[keep], K) if keep.sum() and len(np.unique(yte[keep])) > 1 else np.nan
        m_purge2 = (fast_mcc(yte[keep2], pred0[keep2], K)
                    if missed.any() and len(np.unique(yte[keep2])) > 1 else m_purge)
        ref = {int(l): float(np.mean(yte == l)) for l in labs}
        s_purge2 = std_mcc(yte[keep2], pred0[keep2], ref) if missed.any() else np.nan
        cl_is_test = _hc.cluster_resplit(comp, y, test_frac, _hc.SEED)
        gt = pd.DataFrame({"g": gid, "t": cl_is_test}).groupby("g")["t"].nunique()
        strad_groups = gt[gt > 1].index.to_numpy()
        strad = np.isin(gid, strad_groups)
        # ---------- Q4 kind of homology (direct train edges of purged test sequences)
        mth = ef >= int(round(1000 * (0.8 if th == "id80" else 0.5)))
        a, b, fi = ea[mth], eb[mth], ef[mth]
        cross = (a < n_tr) != (b < n_tr)
        tnode = np.where(a >= n_tr, a, b)[cross] - n_tr
        best = np.zeros(n - n_tr, np.int64)
        np.maximum.at(best, tnode, fi[cross])
        hit = best > 0
        eb_all = best[hit] / 1000.0
        task_rows.append(dict(
            task=task, threshold=th, n_test=int(n - n_tr), n_removed=int(removed.sum()),
            frac_removed=round(float(removed.mean()), 5), kept_per_class=str(kept_per_class),
            class_eliminated=class_eliminated,
            n_test_exact_dup=int(ex_any.sum()), n_test_exact_same_label=int(ex_same.sum()),
            n_exact_missed_by_clusters=int(missed.sum()),
            missed_len_median=float(np.median(L[n_tr:][missed])) if missed.any() else np.nan,
            missed_len_max=int(L[n_tr:][missed].max()) if missed.any() else np.nan,
            purge_mcc=m_purge, purge_plus_exact_mcc=m_purge2, purge_plus_exact_std_mcc=s_purge2,
            purge_plus_exact_minus_purge=(m_purge2 - m_purge) if missed.any() else 0.0,
            n_pooled_in_exact_groups_straddling_resplit=int(strad.sum()),
            n_resplit_test_with_exact_twin_in_train=int((strad & cl_is_test).sum()),
            n_test_direct_train_edge=int(hit.sum()),
            best_train_identity_median=float(np.median(eb_all)) if hit.any() else np.nan,
            frac_best_train_identity_ge_099=round(float((eb_all >= 0.99).mean()), 5) if hit.any() else np.nan,
            frac_best_train_identity_lt_090=round(float((eb_all < 0.90).mean()), 5) if hit.any() else np.nan,
            n_pooled_mostly_N=int((nfrac >= 0.5).sum()),
            mostly_N_per_label=str({int(l): int(((nfrac >= 0.5) & (y == l)).sum()) for l in labs}),
            n_missed_mostly_N=int((missed & (nfrac[n_tr:] >= 0.5)).sum()),
            len_min=int(L.min()), len_median=float(np.median(L)), len_max=int(L.max()),
            n_test_len_lt50=int((L[n_tr:] < 50).sum()), n_all_len_lt11=int((L < 11).sum())))
        # ---------- Q5 component-level bootstrap
        rng = np.random.RandomState(_hc.SEED)
        te_pool = np.arange(n_tr, n)

        def ci_row(quant, idx, pred_by_pool, point_row, lo_col="mcc_lo", hi_col="mcc_hi"):
            grp = comp_groups(idx, comp)
            yy = y[grp[0]]; pp = pred_by_pool(grp[0])
            bt = comp_boot(yy, pp, K, grp, rng)
            boot_rows.append(dict(task=task, threshold=th, quantity=quant, n_seqs=len(idx), n_components=len(grp[1]),
                                  point=fast_mcc(y[idx], pred_by_pool(idx), K),
                                  seq_lo=point_row.get(lo_col, np.nan) if point_row is not None else np.nan,
                                  seq_hi=point_row.get(hi_col, np.nan) if point_row is not None else np.nan,
                                  comp_lo=q(bt, 2.5), comp_hi=q(bt, 97.5)))
            return bt, grp

        p0 = np.full(n, -1, np.int64); p0[n_tr:] = pred0
        o_row = ev[ev.protocol == "original"].iloc[0].to_dict()
        ci_row("original_mcc", te_pool, lambda i: p0[i], o_row)
        pu = ev[(ev.protocol == "test_purge") & (ev.threshold == th)]
        if keep.sum() >= 10 and len(np.unique(yte[keep])) > 1 and not class_eliminated:
            ci_row("purge_mcc", te_pool[keep], lambda i: p0[i], pu.iloc[0].to_dict() if len(pu) else None)
        rc_ = load_pred(task, f"resplit_{th}")
        rr_ = load_pred(task, "resplit_random")
        cl_row = ev[(ev.protocol == "resplit_cluster") & (ev.threshold == th)]
        if rc_ is not None and rr_ is not None and len(cl_row) and cl_row.iloc[0]["status"] == "ok":
            pc = np.full(n, -1, np.int64); pc[rc_[0]] = rc_[1]
            pr = np.full(n, -1, np.int64); pr[rr_[0]] = rr_[1]
            bc, _ = ci_row("resplit_cluster_mcc", rc_[0], lambda i: pc[i], cl_row.iloc[0].to_dict())
            rn_row = ev[ev.protocol == "resplit_random"].iloc[0].to_dict()
            br, _ = ci_row("resplit_random_mcc", rr_[0], lambda i: pr[i], rn_row)
            d = bc - br
            cr = cl_row.iloc[0]
            boot_rows.append(dict(task=task, threshold=th, quantity="cluster_minus_random", n_seqs=len(rc_[0]),
                                  n_components=np.nan, point=float(cr["mcc"] - rn_row["mcc"]),
                                  seq_lo=cr["delta_vs_random_lo"], seq_hi=cr["delta_vs_random_hi"],
                                  comp_lo=q(d, 2.5), comp_hi=q(d, 97.5)))
    res = dict(cls=cls_rows, task=task_rows, boot=boot_rows)
    os.makedirs(CK, exist_ok=True)
    with open(f + ".tmp", "wb") as fh:
        pickle.dump(res, fh)
    os.replace(f + ".tmp", f)
    print(f"[{task}] audit done in {time.time()-t0:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(SIZE_ORDER))
    a = ap.parse_args()
    for t in [t for t in a.tasks.split(",") if t]:
        do_task(t)
    out = {"cls": [], "task": [], "boot": []}
    for t in _hc.TASKS:
        f = os.path.join(CK, f"{t}.pkl")
        if os.path.exists(f):
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            for k in out:
                out[k].extend(r.get(k, []))
    names = {"cls": "audit_class", "task": "audit_task", "boot": "audit_clusterboot"}
    for k, nm in names.items():
        if out[k]:
            p = os.path.join(_hc.RES, f"{_hc.PREFIX}_{nm}.csv")
            pd.DataFrame(out[k]).to_csv(p, index=False)
            print(f"wrote {p} ({len(out[k])} rows)")


if __name__ == "__main__":
    main()
