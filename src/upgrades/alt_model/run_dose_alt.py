#!/usr/bin/env python3
"""Route 1 (causal) under alternative interpretable models: the semi-synthetic discriminability
dose-response of run_disc_dose.py, refit with tuned L1/L2-LR and the EBM (alt_models.py).

Identical data: the 135 synthetic datasets (SP1/MYC/TBP x seeds 42-46 x 9 implant rates; random
ACGT backgrounds with human_nontata_promoters' length and base composition; PWM-sampled instances
implanted in positives only) are regenerated with run_disc_dose.py's own functions and RNG seeding
(RandomState(seed*1000 + int(p_pos*100))), the realized motif-only AUROC d is re-measured and
ASSERTED equal to results/upgrades/disc_dose.csv:motif_auroc_d, and the same stratified 80/20 split
(StratifiedShuffleSplit(test_size=0.2, random_state=seed)) and k=6 spectrum are used. Only the
classifier changes. The lgbm model is refit too, as a check that it reproduces disc_dose.csv:bench_mcc.

Pre-registered readout (see run_alt_model.py): OLS slope of test MCC on d over the 135 settings with a
1000-resample percentile bootstrap CI (rows resampled, RandomState(42), as run_disc_dose.py), plus the
three per-motif slopes and Spearman(d, MCC). Route 1's causal half is preserved under a model if its
pooled slope is > 0 with a CI excluding 0.

Checkpointed per model to external/alt_model/dose_<model>.csv (resumable).
CLI: python run_dose_alt.py --models lgbm,lr,ebm      (through bibm/lowcpu.sh)
2026-09-26 relaunch: 3 threads (LightGBM deterministic -> identical to 1 thread; LR grid in worker
processes; EBM bags seeded). The interrupted 09-23 run wrote no dose rows.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, ".."))
for _d in (UP, os.path.join(UP, "disc_dose"), HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _common as C                   # noqa: E402
import featurize                      # noqa: E402
import motif_jaspar                   # noqa: E402
import motif_match                    # noqa: E402
import run_disc_dose as DD            # noqa: E402
import alt_models as AM               # noqa: E402
from cache_seqs import load_cached    # noqa: E402

OUT_DIR = os.path.join(C.ROOT, "external", "alt_model")
K = 6


def ref_profile_cached(task=DD.REF_TASK):
    """== run_disc_dose.ref_profile, reading the cached original split."""
    tr, _, te, _ = load_cached(task)
    seqs = list(tr) + list(te)
    L = int(np.median([len(s) for s in seqs]))
    codes = motif_match.encode_sequences(seqs[:3000])
    valid = codes < 4
    freqs = np.array([((codes == b) & valid).sum() / valid.sum() for b in range(4)], dtype=np.float64)
    return L, freqs / freqs.sum()


def fit_predict(model, Xtr, ytr, Xte, seqs_tr, seqs_te):
    if model == "lgbm":
        m, _ = AM.fit_lgbm(Xtr, ytr, n_jobs=3)
        return m.classes_[np.argmax(m.predict_proba(Xte), axis=1)], {}
    if model == "lr":
        pp, info, m, _ = AM.fit_lr_tuned(Xtr, ytr, n_jobs=3)
        return m.classes_[np.argmax(pp(Xte), axis=1)], dict(lr_penalty=info["penalty"], lr_C=info["C"],
                                                            lr_degenerate=info["degenerate_all_zero"])
    if model == "ebm":
        voc = featurize.vocabulary(K)
        Etr, Ete, fn = AM.ebm_design(Xtr, Xte, C.comp_signature(seqs_tr), C.comp_signature(seqs_te), ytr, voc)
        m, _ = AM.fit_ebm(Etr, ytr, n_jobs=3, feature_names=fn)
        return m.classes_[np.argmax(m.predict_proba(Ete), axis=1)], {}
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="lr")
    ap.add_argument("--released", default=os.path.join(C.RESULTS_DIR, "disc_dose.csv"))
    args = ap.parse_args()
    rel = pd.read_csv(args.released)
    L, freqs = ref_profile_cached()
    pssms = {m["name"].upper(): m for m in motif_jaspar.load_pssms("vertebrates")}
    for model in [m for m in args.models.split(",") if m]:
        out = os.path.join(OUT_DIR, f"dose_{model}.csv")
        done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
        done_keys = set(zip(done.get("motif", []), done.get("seed", []), done.get("p_pos", [])))
        for _, r in rel.iterrows():
            key = (r["motif"], int(r["seed"]), float(r["p_pos"]))
            if key in done_keys:
                continue
            t0 = time.time()
            m = pssms[r["motif"].upper()]
            ppm = DD.ppm_from_pssm(m["pssm"])
            seed, p_pos = int(r["seed"]), float(r["p_pos"])
            rng = np.random.RandomState(seed * 1000 + int(p_pos * 100))
            seqs, y = DD.synth_dataset(ppm, int((r["n_train"] + r["n_test"]) // 2), L, freqs, p_pos,
                                       float(r["p_neg"]), rng)
            codes = motif_match.encode_sequences(seqs)
            hits = motif_match.count_hits_batch(codes, m["pssm"], 1.0)
            d = C.auc_counts(hits[y == 1].astype(np.int64), hits[y == 0].astype(np.int64))
            d_match = abs(round(float(d), 4) - float(r["motif_auroc_d"])) < 1e-9
            tr_idx, te_idx = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
                                  .split(np.zeros(len(y)), y))
            X = featurize.kmer_spectrum(seqs, K)
            pred, extra = fit_predict(model, X[tr_idx], y[tr_idx], X[te_idx],
                                      [seqs[i] for i in tr_idx], [seqs[i] for i in te_idx])
            mcc = AM.mcc_from_pred(y[te_idx], pred, 2)
            row = dict(motif=r["motif"], motif_ic=r["motif_ic"], seed=seed, p_pos=p_pos, model=model,
                       motif_auroc_d=round(float(d), 4), d_matches_released=bool(d_match),
                       bench_mcc=round(mcc, 4), released_lgbm_mcc=float(r["bench_mcc"]),
                       n_train=len(tr_idx), n_test=len(te_idx), seconds=round(time.time() - t0, 1), **extra)
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"  dose {model} {r['motif']} seed={seed} p={p_pos:.2f} d={d:.4f} (match={d_match}) "
                  f"MCC={mcc:.3f} [lgbm released {r['bench_mcc']:.2f}] {time.time()-t0:.0f}s", flush=True)
    print("[dose] done", flush=True)


if __name__ == "__main__":
    main()
