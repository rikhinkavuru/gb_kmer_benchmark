#!/usr/bin/env python3
"""Step 1c -- COVERAGE-MODE SENSITIVITY for the variable-length tasks (added 2026-09-24 ~11:50, POST HOC, in
response to a verifier's note). The primary graph (run_cluster.py) requires >= 80 % coverage of BOTH sequences
(-c 0.8 --cov-mode 0). In tasks whose sequences vary in length (dummy_mouse 331-4776 bp, drosophila 236-3237,
human_enhancers_ensembl 2-573, human_ocr_ensembl 71-593, human_ensembl_regulatory 71-802) a short sequence
contained in (or overlapping most of) a longer one from the same locus fails that rule, so the reported
redundancy of those tasks could be an under-estimate.

Sensitivity rule (fixed before running): the same all-vs-all both-strand MMseqs2 search as run_cluster.py
except `-c 0.8 --cov-mode 1` (>= 80 % of the TARGET covered; because the search is all-vs-all, every pair is
tried in both directions, so a pair is kept whenever the SHORTER sequence is >= 80 % covered). Edges with
fident >= T (T = 0.8 / 0.5), connected components, then -- with the SAME cached original-split predictions
and run_eval.purge_block (random-removal null, 1000 draws) -- the test-purge. No re-split / refit.
Readout per task and T: frac_test_in_train_cluster and purge MCC / homology_effect under cov-mode 1 next to
the primary cov-mode 0 values (homology_recluster_redundancy.csv / _eval.csv). The primary analysis is
called ROBUST to the coverage rule for a task if |homology_effect(cov1) - homology_effect(cov0)| <= 0.02.
Tasks run smallest first; human_ensembl_regulatory (a ~48-min search with cov-mode 0) is last.
AS RUN (2026-09-24): cov-mode 1 searches took 3.0-3.5x their cov-mode-0 time (enhancers_ensembl 1781 s vs 514 s,
ocr 3649 s vs 1237 s), so human_ensembl_regulatory (~2.5-3 h projected at nice 20 on a shared laptop) was NOT run;
the job was stopped after ocr. Re-run with --tasks human_ensembl_regulatory to add it (checkpointed).

Outputs: results/upgrades/homology_recluster_covmode.csv; edges in external/homology_recluster/edges_covmode1/.
Usage: python run_covmode_sensitivity.py [--tasks a,b]
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402
import run_cluster as RC  # noqa: E402
import run_eval as RE  # noqa: E402

TASKS = ["dummy_mouse_enhancers_ensembl", "drosophila_enhancers_stark", "human_enhancers_ensembl",
         "human_ocr_ensembl", "human_ensembl_regulatory"]
ARGS = [a if a != "0" or prev != "--cov-mode" else "1"
        for prev, a in zip([""] + RC.SEARCH_ARGS[:-1], RC.SEARCH_ARGS)]
assert "--cov-mode" in ARGS and ARGS[ARGS.index("--cov-mode") + 1] == "1"
EDGE2 = os.path.join(_hc.EXT, "edges_covmode1")


def search_edges_cov1(seqs, workdir, log, chunk=4_000_000):
    """run_cluster.search_edges with --cov-mode 1; keeps hits with max(qcov, tcov) >= 0.8, fident >= 0.5."""
    os.makedirs(workdir, exist_ok=True)
    fa = os.path.join(workdir, "in.fa")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    db, aln, tmp, m8 = (os.path.join(workdir, x) for x in ("db", "aln", "tmp", "aln.m8"))
    RC.run([RC.MMSEQS, "createdb", fa, db, "--dbtype", "2"], log)
    RC.run([RC.MMSEQS, "search", db, db, aln, tmp] + ARGS, log)
    RC.run([RC.MMSEQS, "convertalis", db, db, aln, m8, "--format-output", "query,target,fident,qcov,tcov",
            "--threads", str(_hc.N_THREADS)], log)
    keys, fids = [], []
    n = np.int64(len(seqs))
    for df in pd.read_csv(m8, sep="\t", header=None, names=["q", "t", "fident", "qcov", "tcov"],
                          dtype={"q": np.int64, "t": np.int64, "fident": np.float64, "qcov": np.float32,
                                 "tcov": np.float32}, chunksize=chunk):
        df = df[(df.q != df.t) & (np.maximum(df.qcov, df.tcov) >= RC.COV) & (df.fident >= 0.5)]
        a = np.minimum(df.q.to_numpy(), df.t.to_numpy()); b = np.maximum(df.q.to_numpy(), df.t.to_numpy())
        k = a * n + b; f = np.rint(df.fident.to_numpy() * 1000).astype(np.uint16)
        o = np.lexsort((-f.astype(np.int64), k)); k, f = k[o], f[o]
        first = np.r_[True, k[1:] != k[:-1]]
        keys.append(k[first]); fids.append(f[first])
    k = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    f = np.concatenate(fids) if fids else np.zeros(0, np.uint16)
    o = np.lexsort((-f.astype(np.int64), k)); k, f = k[o], f[o]
    first = np.r_[True, k[1:] != k[:-1]] if len(k) else np.zeros(0, bool)
    k, f = k[first], f[first]
    return (k // n).astype(np.int32), (k % n).astype(np.int32), f


def do_task(task):
    ck = os.path.join(_hc.CKPT_DIR, f"covmode_{task}.csv")
    if os.path.exists(ck):
        return pd.read_csv(ck)
    t0 = time.time()
    tr, ytr, te, yte = _hc.load_task(task)
    n_tr, n = len(tr), len(tr) + len(te)
    ef_path = os.path.join(EDGE2, f"{task}.npz")
    if os.path.exists(ef_path):
        z = np.load(ef_path); ea, eb, ef = z["a"], z["b"], z["fid"]
    else:
        os.makedirs(EDGE2, exist_ok=True)
        work = os.path.join(RC.MM_DIR, f"cov1_{task}")
        if os.path.isdir(work):
            shutil.rmtree(work)
        log = os.path.join(_hc.EXT, "logs", f"mmseqs_cov1_{task}.log"); open(log, "w").close()
        ea, eb, ef = search_edges_cov1(tr + te, work, log)
        shutil.rmtree(work)
        np.savez_compressed(ef_path + ".tmp.npz", a=ea, b=eb, fid=ef)
        os.replace(ef_path + ".tmp.npz", ef_path)
    t_search = time.time() - t0
    z = np.load(os.path.join(RE.PRED_DIR, f"{task}__original.npz"))
    assert np.array_equal(z["te_idx"], np.arange(n_tr, n))
    red = pd.read_csv(os.path.join(_hc.CKPT_DIR, f"cluster_{task}.csv")).set_index("threshold")
    ev = pd.read_csv(os.path.join(_hc.RES, f"{_hc.PREFIX}_eval.csv"))
    rows = []
    for tag, thr in RC.THRESHOLDS.items():
        m = ef >= int(round(thr * 1000))
        comp = RC.components(n, ea[m], eb[m])
        has_tr = np.zeros(comp.max() + 1, bool); has_tr[comp[:n_tr]] = True
        keep = ~has_tr[comp[n_tr:]]
        _, pb = RE.purge_block(yte, z["proba"], z["classes"], keep, _hc.SEED)
        p0 = ev[(ev.task == task) & (ev.protocol == "test_purge") & (ev.threshold == tag)].iloc[0]
        rows.append(dict(task=task, threshold=tag, cov_rule="max(qcov,tcov)>=0.8 (cov-mode 1)",
                         n_edges=int(m.sum()), n_edges_covmode0=int(red.loc[tag, "n_edges"]),
                         frac_test_in_train_cluster=round(float((~keep).mean()), 5),
                         frac_test_in_train_cluster_covmode0=float(red.loc[tag, "frac_test_in_train_cluster"]),
                         purge_status=pb.get("status"), purge_mcc=pb.get("mcc", np.nan),
                         purge_null_mean=pb.get("null_mean", np.nan), homology_effect=pb.get("homology_effect", np.nan),
                         p_null=pb.get("p_null", np.nan), purge_mcc_covmode0=p0.get("mcc", np.nan),
                         homology_effect_covmode0=p0.get("homology_effect", np.nan),
                         search_seconds=round(t_search, 1)))
        r = rows[-1]
        r["robust_to_cov_rule"] = (bool(abs(r["homology_effect"] - r["homology_effect_covmode0"]) <= 0.02)
                                   if r["homology_effect"] == r["homology_effect"] and
                                   r["homology_effect_covmode0"] == r["homology_effect_covmode0"] else np.nan)
        print(f"[{task}] {tag}: test-in-train cov1 {r['frac_test_in_train_cluster']:.4f} vs cov0 "
              f"{r['frac_test_in_train_cluster_covmode0']:.4f} | effect cov1 {r['homology_effect']:+.4f} vs cov0 "
              f"{r['homology_effect_covmode0']:+.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ck, index=False)
    print(f"[{task}] done in {time.time()-t0:.0f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    for t in [t for t in a.tasks.split(",") if t]:
        do_task(t)
    parts = [pd.read_csv(os.path.join(_hc.CKPT_DIR, f"covmode_{t}.csv")) for t in TASKS
             if os.path.exists(os.path.join(_hc.CKPT_DIR, f"covmode_{t}.csv"))]
    if parts:
        out = os.path.join(_hc.RES, f"{_hc.PREFIX}_covmode.csv")
        pd.concat(parts, ignore_index=True).to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
