#!/usr/bin/env python3
"""Upgrade 1 / Depth 1 -- PAIRED FM-probe for the SECOND architecture (DNABERT-2).

Runs the EXACT Fix-A paired evaluation used for HyenaDNA (run_fm_paired.py: same test_effect /
paired_delta definitions, same 1000-resample paired bootstrap, same membership-mask T_clean), but on
the cached frozen DNABERT-2 embeddings, for the SHORT datasets only (human_enhancers_cohn,
nt_enhancers). drosophila is excluded (3.2 kb > 512 tokens) -- so DNABERT-2 has no flat-control arm;
the cross-architecture claim is the SIGN + SIGNIFICANCE of the composition-removal delta on the two
contaminated sets, compared against HyenaDNA. Pooling = mean / [CLS] (DNABERT-2's analogues of
HyenaDNA's mean / [SEP]); arms = tata_flag, gc_match, comp_equalized; heads = lgbm, lr.

LightGBM-only process (reads the .npz DNABERT-2 cache via numpy; never imports torch). Reuses the
verbatim helpers from run_fm_paired.py so the methodology is identical. Seed 42. Writes
results/upgrades/fm_paired_dnabert.csv (same schema as fm_paired.csv + a 'model' column).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import run_fm_paired as PP                 # verbatim helpers (lightgbm-only; no torch loaded)
import run_fm_probe as P                   # arm_split_seqs, ROLE
import extract_embeddings_dnabert as XD    # DNABERT-2 cache loader (numpy; torch lazy/unused here)

MODEL = "DNABERT-2-117M"
DATASETS = ["human_enhancers_cohn", "nt_enhancers"]
POOLINGS = ["mean", "cls"]
ARMS = ["tata_flag", "gc_match", "comp_equalized"]
HEADS = ["lgbm", "lr"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "fm_paired_dnabert.csv"))
    ap.add_argument("--emb-cache", default=os.path.join(C.ROOT, "cache", "fm_embeddings"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tasks = [t for t in args.datasets.split(",") if t]

    print("=" * 104)
    print(f"DEPTH 1 -- PAIRED FM-PROBE (DNABERT-2)  model={XD.MODEL_TAG}  seed={args.seed}  poolings={POOLINGS}")
    print("=" * 104)
    rows = []
    for task in tasks:
        otr_seqs, ytr0, ote_seqs, yte0 = C.load_original(task)
        emb_tr = XD.load_cached(args.emb_cache, task, "orig", "train")
        emb_te = XD.load_cached(args.emb_cache, task, "orig", "test")
        if emb_tr is None or emb_te is None:
            sys.exit(f"ERROR: missing DNABERT-2 embeddings for {task}; run extract_embeddings_dnabert.py first.")
        assert emb_tr["mean"].shape[0] == len(otr_seqs) and emb_te["mean"].shape[0] == len(ote_seqs), \
            f"{task}: cached DNABERT-2 rows != split sizes"
        idx_tr = {s: i for i, s in enumerate(otr_seqs)}
        idx_te = {s: i for i, s in enumerate(ote_seqs)}
        nc = len(np.unique(ytr0))
        print(f"\n[{task}] ({P.ROLE[task]})  orig train={len(ytr0)} test={len(yte0)}")

        for pooling in POOLINGS:
            Xtr_o = emb_tr[pooling]; Xte_o = emb_te[pooling]
            for head in HEADS:
                rng = np.random.RandomState(args.seed)
                H_orig = PP.fit_head(Xtr_o, ytr0, head, args.seed)
                pred_of, proba_of = PP.predict(H_orig, Xte_o)
                of_mcc = PP._mcc(yte0, pred_of); of_auc = PP._auc(yte0, proba_of, nc)
                for arm in ARMS:
                    try:
                        atr_seqs, atr_y, ate_seqs, ate_y = P.arm_split_seqs(task, arm)
                    except FileNotFoundError:
                        continue
                    Xtr_c = Xtr_o[[idx_tr[s] for s in atr_seqs]]
                    H_clean = PP.fit_head(Xtr_c, atr_y, head, args.seed)
                    pred_cf, proba_cf = PP.predict(H_clean, Xte_o)
                    mask = PP.membership_mask(ote_seqs, ate_seqs)
                    pos = np.where(mask)[0]; y_c = yte0[pos]
                    ooc_mcc = PP._mcc(y_c, pred_of[pos]); ooc_auc = PP._auc(y_c, proba_of[pos], nc)
                    coc_mcc = PP._mcc(y_c, pred_cf[pos]); coc_auc = PP._auc(y_c, proba_cf[pos], nc)
                    n = len(pos); no = len(yte0)
                    pdm = np.full(args.boot, np.nan); pda = np.full(args.boot, np.nan)
                    tem = np.full(args.boot, np.nan); tea = np.full(args.boot, np.nan)
                    for b in range(args.boot):
                        r = rng.randint(0, n, n); sub = pos[r]
                        if len(np.unique(y_c[r])) > 1:
                            pdm[b] = matthews_corrcoef(y_c[r], pred_cf[sub]) - matthews_corrcoef(y_c[r], pred_of[sub])
                            try:
                                pda[b] = PP._auc(y_c[r], proba_cf[sub], nc) - PP._auc(y_c[r], proba_of[sub], nc)
                            except ValueError:
                                pass
                        rf = rng.randint(0, no, no); mfull = mask[rf]
                        if len(np.unique(yte0[rf])) > 1 and mfull.sum() > 1 and len(np.unique(yte0[rf][mfull])) > 1:
                            tem[b] = matthews_corrcoef(yte0[rf][mfull], pred_of[rf][mfull]) - matthews_corrcoef(yte0[rf], pred_of[rf])
                            try:
                                tea[b] = PP._auc(yte0[rf][mfull], proba_of[rf][mfull], nc) - PP._auc(yte0[rf], proba_of[rf], nc)
                            except ValueError:
                                pass
                    pdm_lo, pdm_hi = PP.pct(pdm); pda_lo, pda_hi = PP.pct(pda)
                    tem_lo, tem_hi = PP.pct(tem); tea_lo, tea_hi = PP.pct(tea)
                    rows.append(dict(model=MODEL, dataset=task, role=P.ROLE[task], arm=arm, pooling=pooling, head=head,
                        n_orig_test=int(no), n_clean_test=int(n),
                        orig_full_mcc=round(of_mcc, 4), orig_full_auroc=round(of_auc, 4),
                        orig_on_clean_mcc=round(ooc_mcc, 4), orig_on_clean_auroc=round(ooc_auc, 4),
                        clean_on_clean_mcc=round(coc_mcc, 4), clean_on_clean_auroc=round(coc_auc, 4),
                        unpaired_delta_mcc=round(coc_mcc - of_mcc, 4), unpaired_delta_auroc=round(coc_auc - of_auc, 4),
                        paired_delta_mcc=round(coc_mcc - ooc_mcc, 4), paired_delta_mcc_ci=f"[{pdm_lo:.3f},{pdm_hi:.3f}]",
                        paired_delta_auroc=round(coc_auc - ooc_auc, 4), paired_delta_auroc_ci=f"[{pda_lo:.3f},{pda_hi:.3f}]",
                        test_effect_mcc=round(ooc_mcc - of_mcc, 4), test_effect_mcc_ci=f"[{tem_lo:.3f},{tem_hi:.3f}]",
                        test_effect_auroc=round(ooc_auc - of_auc, 4), test_effect_auroc_ci=f"[{tea_lo:.3f},{tea_hi:.3f}]"))
                    print(f"   {pooling}/{head:<4} {arm:<15} orig_full={of_auc:.3f} -> orig_on_clean={ooc_auc:.3f} "
                          f"(test_effect AUROC {ooc_auc-of_auc:+.3f} [{tea_lo:+.3f},{tea_hi:+.3f}])", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} rows).")


if __name__ == "__main__":
    main()
