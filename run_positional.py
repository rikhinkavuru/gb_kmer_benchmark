#!/usr/bin/env python3
"""run_positional.py -- causal test of the POSITIONAL failure route.

The failure taxonomy has three routes: shared-motif (motifs present in both
classes), contamination (negatives carry the positive class's signal), and
POSITIONAL (the signal is a fixed-position motif that the position-blind k-mer
spectrum discards -- splice sites fail this way on both suites). This tests
causality with a DOUBLE DISSOCIATION: add minimal positional features
(position-binned k-mers) and check that
  * the positional task (splice) MCC rises substantially,
  * a motif-local promoter barely moves (signal already positionally diffuse),
  * a contaminated / shared-motif task barely moves (position can't fix a signal
    that is shared or points the wrong way).
If positional features recover exactly the positional failure and nothing else,
the three routes are mechanistically distinct.

Method: for each task, featurize the train/test sequences two ways at the SAME k
(the benchmark's best k) -- the position-blind spectrum, and that SAME spectrum
AUGMENTED with a position-binned spectrum (B bins) concatenated alongside it --
fit the identical LightGBM, and compare test MCC with a PAIRED bootstrap (resample
test indices once, score both models on it) so the delta has a proper CI. Adding
(not replacing) means a task whose signal needs no position can simply ignore the
extra bins, so the delta isolates the *benefit* of position. CPU-only; reuses
featurize/models + the loaders.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import matthews_corrcoef

import data as gbdata
import featurize
import models
import kselect

HERE = os.path.dirname(os.path.abspath(__file__))

# (task, predicted failure route) -- splice is positional; the rest are controls
DEFAULT_TASKS = [
    ("nt_splice_sites_all", "POSITIONAL"),
    ("nt_promoter_no_tata", "motif-local control"),
    ("nt_enhancers", "contamination/shared-motif control"),
    ("human_nontata_promoters", "motif-local control (GB)"),
    ("human_enhancers_cohn", "contamination/shared-motif control (GB)"),
]


def best_k(task, cache_dir, seed):
    # k is selected on a held-out validation split carved from TRAIN (kselect),
    # never on the test split. (Former gb_csv/nt_csv args are no longer needed.)
    return kselect.best_k_val(task, cache_dir, seed)


def _fit_predict(Xtr, ytr, Xte, seed):
    m = models.build_model("lgbm", seed, len(np.unique(ytr)))
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    return m.classes_[np.argmax(proba, axis=1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gb-results", default=os.path.join(HERE, "results", "results.csv"))
    ap.add_argument("--nt-results", default=os.path.join(HERE, "results", "results_nt.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "positional_features.csv"))
    ap.add_argument("--n-bins", type=int, default=8, help="positional bins per sequence (B)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default="", help="comma subset of tasks (default: the 5 dissociation tasks)")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    tasks = DEFAULT_TASKS
    if args.tasks:
        want = set(args.tasks.split(","))
        tasks = [(t, r) for t, r in DEFAULT_TASKS if t in want]

    print("=" * 96)
    print("POSITIONAL FEATURE TEST -- does resolving position recover the positional failure only?")
    print(f"position-binned k-mers, B={args.n_bins} bins | boot={args.boot} | seed={args.seed}")
    print("=" * 96)

    rows = []
    for task, route in tasks:
        k = best_k(task, os.path.join(HERE, "cache"), args.seed)
        d = gbdata.load_dataset(task, seed=args.seed)
        ytr, yte = d["y_train"], d["y_test"]
        suite = "NucTransformer" if task.startswith("nt_") else "GenomicBench"

        Xtr_s = featurize.kmer_spectrum(d["train_seqs"], k)
        Xte_s = featurize.kmer_spectrum(d["test_seqs"], k)
        pred_s = _fit_predict(Xtr_s, ytr, Xte_s, args.seed)

        # positional arm = global spectrum AUGMENTED with the position-binned spectrum
        # (concatenated "alongside"), so motif-local tasks keep the global view and the
        # model only gains by USING position where it helps.
        Xtr_p = sparse.hstack([Xtr_s, featurize.binned_kmer_spectrum(d["train_seqs"], k, args.n_bins)], format="csr")
        Xte_p = sparse.hstack([Xte_s, featurize.binned_kmer_spectrum(d["test_seqs"], k, args.n_bins)], format="csr")
        pred_p = _fit_predict(Xtr_p, ytr, Xte_p, args.seed)

        mcc_s = matthews_corrcoef(yte, pred_s)
        mcc_p = matthews_corrcoef(yte, pred_p)

        n = len(yte)
        ss = np.empty(args.boot); ps = np.empty(args.boot); dd = np.empty(args.boot)
        for b in range(args.boot):
            idx = rng.randint(0, n, n)
            a = matthews_corrcoef(yte[idx], pred_s[idx])
            c = matthews_corrcoef(yte[idx], pred_p[idx])
            ss[b] = a; ps[b] = c; dd[b] = c - a
        slo, shi = np.percentile(ss, [2.5, 97.5])
        plo, phi = np.percentile(ps, [2.5, 97.5])
        dlo, dhi = np.percentile(dd, [2.5, 97.5])
        excl0 = not (dlo <= 0 <= dhi)
        rows.append(dict(task=task, suite=suite, route=route, k=k, n_bins=args.n_bins,
            feat_dim_spectrum=featurize.feature_dim(k),
            feat_dim_positional=(1 + args.n_bins) * featurize.feature_dim(k),
            spectrum_mcc=round(mcc_s, 4), spectrum_ci=f"[{slo:.3f},{shi:.3f}]",
            positional_mcc=round(mcc_p, 4), positional_ci=f"[{plo:.3f},{phi:.3f}]",
            delta_mcc=round(mcc_p - mcc_s, 4), delta_ci=f"[{dlo:.3f},{dhi:.3f}]",
            delta_excludes_0=excl0))
        print(f"  {task:<28} [{route:<34}] k={k} | spectrum MCC={mcc_s:.3f} -> "
              f"positional MCC={mcc_p:.3f}  delta={mcc_p-mcc_s:+.3f} [{dlo:+.3f},{dhi:+.3f}] "
              f"{'(sig)' if excl0 else '(ns)'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    report = build_report(df, args)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out} ({len(df)} tasks) + interpretation.")


def build_report(df, args):
    L = []; add = L.append
    add("=" * 96)
    add("POSITIONAL FEATURE DOUBLE-DISSOCIATION  (position-binned k-mers, B=%d)" % args.n_bins)
    add("=" * 96)
    hdr = (f"{'task':<28}{'route':<36}{'spectrum MCC':>14}{'positional MCC':>16}"
           f"{'delta':>9} {'delta 95% CI':<18}")
    add(hdr); add("-" * len(hdr))
    for _, r in df.iterrows():
        add(f"{r['task']:<28}{r['route']:<36}{r['spectrum_mcc']:>14.3f}{r['positional_mcc']:>16.3f}"
            f"{r['delta_mcc']:>+9.3f} {r['delta_ci']:<18}{' SIG' if r['delta_excludes_0'] else ' ns'}")
    add("")
    add("delta = positional_MCC - spectrum_MCC, paired bootstrap (same resampled test set).")
    add("SIG = delta 95% CI excludes 0 (positional features changed performance).")

    pos = df[df["route"].str.startswith("POSITIONAL")]
    ctrl = df[~df["route"].str.startswith("POSITIONAL")]
    add("\nDouble-dissociation verdict:")
    if len(pos):
        p = pos.iloc[0]
        recov = p["delta_excludes_0"] and p["delta_mcc"] > 0
        add(f"  POSITIONAL task ({p['task']}): delta={p['delta_mcc']:+.3f} {p['delta_ci']} "
            f"-> {'RECOVERS' if recov else 'does NOT recover'} with positional features.")
    sig_ctrl = ctrl[ctrl["delta_excludes_0"]]
    add(f"  control tasks: {len(ctrl)} total, {len(sig_ctrl)} with a significant delta; "
        f"max |delta| among controls = {ctrl['delta_mcc'].abs().max():.3f} "
        f"(vs positional task delta {pos['delta_mcc'].iloc[0]:+.3f})." if len(pos) else "")
    add("")
    add("Reading: a large, significant positive delta ONLY on the positional task -- with")
    add("motif-local and contamination/shared-motif controls barely moving -- confirms the")
    add("positional failure is CAUSALLY due to discarding position, and that it is a distinct")
    add("route from shared-motif and contamination (which positional features cannot fix).")
    add("If the positional task does NOT recover, the positional story is incomplete -- reported")
    add("straight. Bin width is a fraction of each sequence length; B=%d; seed=%d." % (args.n_bins, args.seed))
    return "\n".join(L)


if __name__ == "__main__":
    main()