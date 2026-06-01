#!/usr/bin/env python3
"""run_benchmark.py -- CPU-only k-mer + LightGBM benchmark over Genomic Benchmarks.

Sweeps every (dataset x k x model) cell and writes a single tidy ``results.csv``
plus a printed summary. CPU-only by construction: numpy + scipy(ships with
scikit-learn) + scikit-learn + lightgbm + genomic-benchmarks. No torch/tensorflow,
no pretrained models or embeddings, no GPU.

Two phases
----------
  Phase 1 (prepare): load each dataset once, fix the train/test partition once,
    featurize every k into an L1-normalized sparse k-mer spectrum, and cache the
    matrices + labels + meta to ``--cache-dir``. Reruns reuse the cache, so they
    are fast (featurization is the expensive part).
  Phase 2 (run): for every (k, model) spawn an isolated child process that loads
    the cached matrices, fits the model, evaluates, and reports metrics + train
    time + peak RAM as a JSON line.

Why a child process per cell
----------------------------
  Peak RAM is measured with the stdlib ``resource`` module (psutil is not a
  permitted dependency). ``ru_maxrss`` is a per-process high-water mark, so a
  fresh process per cell yields a clean, comparable peak for that cell. It also
  makes the suite robust: a cell that fails (odd data, OOM, ...) is recorded and
  the sweep continues. Use ``--in-process`` to disable (faster, but peak RAM
  then reflects the whole run's high-water mark).

Reproducibility
---------------
  A single ``--seed`` (default 42) drives numpy, Python's ``random``,
  ``PYTHONHASHSEED`` (child env), each model's ``random_state``, and any
  synthesized split. The k-mer vocabulary is fixed and data-independent.
  LightGBM runs in deterministic mode. Exact versions are pinned in
  ``requirements.txt``.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
from scipy import sparse

import data as gbdata
import featurize
import metrics as gbmetrics
import models as gbmodels

ALL_DATASETS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
    "human_ensembl_regulatory",        # 3 classes -> macro-averaged metrics
    "drosophila_enhancers_stark",
    "dummy_mouse_enhancers_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
]
DEFAULT_KS = [3, 4, 5, 6]

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_SENTINEL = "__RESULT__ "
CSV_FIELDS = ["dataset", "n_classes", "k", "feature_dim", "model", "n_train",
              "n_test", "accuracy", "mcc", "auroc", "train_time_s",
              "peak_ram_mb", "split_source", "seed", "status", "error"]


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def peak_ram_mb():
    """Process peak RSS in MB. ru_maxrss is bytes on macOS, kilobytes on Linux."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024.0


# --- cache paths ---------------------------------------------------------
def _x_path(cache, ds, split, k):
    return os.path.join(cache, f"{ds}__{split}__k{k}.npz")


def _y_path(cache, ds, split):
    return os.path.join(cache, f"{ds}__{split}__y.npy")


def _meta_path(cache, ds):
    return os.path.join(cache, f"{ds}__meta.json")


def _stratified_cap(seqs, y, cap, seed):
    """Deterministic stratified subsample to <= cap sequences (quick mode)."""
    if cap is None or len(seqs) <= cap:
        return seqs, y
    rng = np.random.RandomState(seed)
    chunks = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        take = min(len(ci), max(1, int(round(cap * len(ci) / len(y)))))
        chunks.append(rng.choice(ci, size=take, replace=False))
    idx = np.sort(np.concatenate(chunks))
    return [seqs[i] for i in idx], y[idx]


# --- Phase 1: prepare + cache --------------------------------------------
def prepare(ds, ks, seed, cache, max_seqs_per_split=None):
    os.makedirs(cache, exist_ok=True)
    d = gbdata.load_dataset(ds, seed=seed)
    tr_seqs, y_tr = d["train_seqs"], d["y_train"]
    te_seqs, y_te = d["test_seqs"], d["y_test"]
    if max_seqs_per_split:
        tr_seqs, y_tr = _stratified_cap(tr_seqs, y_tr, max_seqs_per_split, seed)
        te_seqs, y_te = _stratified_cap(te_seqs, y_te, max_seqs_per_split, seed + 1)

    np.save(_y_path(cache, ds, "train"), y_tr)
    np.save(_y_path(cache, ds, "test"), y_te)
    meta = dict(dataset=ds, n_classes=d["n_classes"], classes=d["classes"],
                n_train=int(len(y_tr)), n_test=int(len(y_te)),
                split_source=d["split_source"], seed=seed,
                max_seqs_per_split=max_seqs_per_split)

    built = reused = 0
    for k in ks:
        xtr, xte = _x_path(cache, ds, "train", k), _x_path(cache, ds, "test", k)
        if os.path.exists(xtr) and os.path.exists(xte) and os.path.exists(_meta_path(cache, ds)):
            reused += 1
            continue
        sparse.save_npz(xtr, featurize.kmer_spectrum(tr_seqs, k))
        sparse.save_npz(xte, featurize.kmer_spectrum(te_seqs, k))
        built += 1
    with open(_meta_path(cache, ds), "w") as fh:
        json.dump(meta, fh, indent=2)
    meta["_built"], meta["_reused"] = built, reused
    return meta


# --- one cell (run inside a child process) -------------------------------
def run_cell(ds, k, model_name, seed, cache):
    set_seeds(seed)
    with open(_meta_path(cache, ds)) as fh:
        meta = json.load(fh)
    Xtr = sparse.load_npz(_x_path(cache, ds, "train", k))
    Xte = sparse.load_npz(_x_path(cache, ds, "test", k))
    ytr = np.load(_y_path(cache, ds, "train"))
    yte = np.load(_y_path(cache, ds, "test"))

    model = gbmodels.build_model(model_name, seed, meta["n_classes"])
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0

    proba = model.predict_proba(Xte)
    pred = model.classes_[np.argmax(proba, axis=1)]
    m = gbmetrics.compute_metrics(yte, pred, proba, meta["n_classes"])

    return dict(dataset=ds, n_classes=meta["n_classes"], k=k,
                feature_dim=featurize.feature_dim(k), model=model_name,
                n_train=int(len(ytr)), n_test=int(len(yte)),
                accuracy=round(m["accuracy"], 6), mcc=round(m["mcc"], 6),
                auroc=(round(m["auroc"], 6) if m["auroc"] == m["auroc"] else ""),
                train_time_s=round(train_time, 4),
                peak_ram_mb=round(peak_ram_mb(), 1),
                split_source=meta["split_source"], seed=seed,
                status="ok", error="")


# --- orchestration -------------------------------------------------------
def _fail_row(ds, k, model_name, seed, status, error):
    return dict(dataset=ds, n_classes="", k=k, feature_dim=featurize.feature_dim(k),
                model=model_name, n_train="", n_test="", accuracy="", mcc="",
                auroc="", train_time_s="", peak_ram_mb="", split_source="",
                seed=seed, status=status, error=error)


def _spawn_cell(ds, k, model_name, seed, cache, timeout):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    cmd = [sys.executable, os.path.abspath(__file__), "--cell",
           "--dataset", ds, "--k", str(k), "--model", model_name,
           "--seed", str(seed), "--cache-dir", cache]
    try:
        p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _fail_row(ds, k, model_name, seed, "timeout", f">{timeout}s")
    if p.returncode != 0:
        tail = ((p.stderr or "").strip().splitlines() or [""])[-1]
        return _fail_row(ds, k, model_name, seed, "error", tail[:300])
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])
    return _fail_row(ds, k, model_name, seed, "error", "no result line emitted")


def _print_cell(r):
    if r["status"] != "ok":
        print(f"    k={r['k']:<2} dim={r['feature_dim']:<5} {r['model']:<5} "
              f"{r['status'].upper()}: {r['error']}", flush=True)
        return
    auroc = r["auroc"] if r["auroc"] != "" else float("nan")
    print(f"    k={r['k']:<2} dim={r['feature_dim']:<5} {r['model']:<5} "
          f"acc={r['accuracy']:.4f} mcc={r['mcc']:.4f} auroc={auroc if isinstance(auroc,str) else f'{auroc:.4f}'} "
          f"fit={r['train_time_s']:.2f}s ram={r['peak_ram_mb']:.0f}MB", flush=True)


def print_summary(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    print("\n" + "=" * 86)
    print("SUMMARY  --  best k per (dataset, model), ranked by MCC")
    print("=" * 86)
    hdr = (f"{'dataset':<32}{'model':<6}{'k':>2}{'dim':>6}  "
           f"{'acc':>6} {'mcc':>6} {'auroc':>6} {'fit_s':>7} {'ram_MB':>7}")
    print(hdr)
    print("-" * len(hdr))
    by_ds = {}
    for r in ok:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds in sorted(by_ds):
        for mname in ("lr", "lgbm"):
            cand = [r for r in by_ds[ds] if r["model"] == mname]
            if not cand:
                continue
            best = max(cand, key=lambda r: (r["mcc"] if isinstance(r["mcc"], (int, float)) else -9))
            a = best["auroc"]
            astr = f"{a:>6.3f}" if isinstance(a, (int, float)) else f"{'n/a':>6}"
            print(f"{ds:<32}{mname:<6}{best['k']:>2}{best['feature_dim']:>6}  "
                  f"{best['accuracy']:>6.3f} {best['mcc']:>6.3f} {astr} "
                  f"{best['train_time_s']:>7.2f} {best['peak_ram_mb']:>7.0f}")
    nfail = len(rows) - len(ok)
    if nfail:
        print(f"\n{nfail} cell(s) did not complete -- see status/error columns in the CSV.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS),
                    help="comma-separated dataset names (default: all 9)")
    ap.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)),
                    help="comma-separated k values (default: 3,4,5,6)")
    ap.add_argument("--models", default=",".join(gbmodels.MODELS),
                    help="comma-separated models (default: lr,lgbm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.csv"))
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-cell timeout in seconds (default 3600)")
    ap.add_argument("--max-seqs-per-split", type=int, default=None,
                    help="quick mode: stratified-cap each split to N sequences")
    ap.add_argument("--in-process", action="store_true",
                    help="run cells in-process (faster; peak RAM less isolated)")
    # internal single-cell mode (used by the child processes)
    ap.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dataset")
    ap.add_argument("--k", type=int)
    ap.add_argument("--model")
    args = ap.parse_args()

    if args.cell:
        res = run_cell(args.dataset, args.k, args.model, args.seed, args.cache_dir)
        print(RESULT_SENTINEL + json.dumps(res))
        return

    set_seeds(args.seed)
    datasets = [d for d in args.datasets.split(",") if d]
    ks = [int(x) for x in args.ks.split(",") if x]
    model_list = [m for m in args.models.split(",") if m]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 86)
    print("CPU-only k-mer spectrum + LightGBM benchmark over Genomic Benchmarks")
    print(f"datasets={len(datasets)}  ks={ks}  models={model_list}  seed={args.seed}"
          f"  mode={'in-process' if args.in_process else 'subprocess-per-cell'}")
    print(f"cache={args.cache_dir}")
    print(f"out  ={args.out}")
    if args.max_seqs_per_split:
        print(f"QUICK MODE: capping each split to {args.max_seqs_per_split} sequences")
    print("=" * 86)

    fh = open(args.out, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
    fh.flush()

    rows = []
    t_start = time.time()
    for di, ds in enumerate(datasets, 1):
        try:
            meta = prepare(ds, ks, args.seed, args.cache_dir, args.max_seqs_per_split)
            print(f"\n[{di}/{len(datasets)}] {ds} | classes={meta['n_classes']} "
                  f"train={meta['n_train']} test={meta['n_test']} "
                  f"split={meta['split_source']} | cache: {meta['_built']} built, "
                  f"{meta['_reused']} reused", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad dataset must not kill the suite
            print(f"\n[{di}/{len(datasets)}] {ds} | PREPARE FAILED: {e}", flush=True)
            for k in ks:
                for mname in model_list:
                    r = _fail_row(ds, k, mname, args.seed, "prepare_error", str(e)[:300])
                    rows.append(r)
                    writer.writerow(r)
                    fh.flush()
            continue

        for k in ks:
            for mname in model_list:
                if args.in_process:
                    try:
                        r = run_cell(ds, k, mname, args.seed, args.cache_dir)
                    except Exception as e:  # noqa: BLE001
                        r = _fail_row(ds, k, mname, args.seed, "error", str(e)[:300])
                else:
                    r = _spawn_cell(ds, k, mname, args.seed, args.cache_dir, args.timeout)
                rows.append(r)
                writer.writerow(r)
                fh.flush()
                _print_cell(r)
    fh.close()

    print(f"\nWrote {len(rows)} rows to {args.out} in {time.time() - t_start:.1f}s")
    print_summary(rows)


if __name__ == "__main__":
    main()
