#!/usr/bin/env python3
"""POST HOC PH3 (fix round 1): WHERE is the positional signal in the GUE tasks that meet the Route-2 rule?

Written 2026-09-27 (fix round 1), after the pre-registered cards and BEFORE this script was run.
Motivation, stated before running. The first report explained the two GUE TATA-promoter Route-2 labels
by 'GUE's TATA negatives carry a TATA motif at arbitrary positions, so only its fixed offset separates
the classes'. A verifier scan found the negatives' TBP hits peak at the SAME canonical offset as the
positives' (about -30 from the TSS), which refutes that mechanism. The prom_300_tata positives are
identical to nt_promoter_tata's, where the same code gives a non-significant delta (+0.019), so the
route is set by GUE's negative set -- but WHICH positional feature carries the B=8 gain is unknown.
PH3 localises it. It changes no pre-registered readout or label.

Tasks: the three GUE tasks that meet the Route-2 rule (prom_300_tata, prom_core_tata,
splice_reconstructed -- splice is the positive control: its signal must sit at the centre, where the
donor/acceptor dinucleotide is) and the paper's nt_promoter_tata (identical positives to prom_300_tata;
comparator). Same data and arms as the pre-registered Route-2 readout: the card's validation-selected k,
train/test capped at 20,000 with cap_split(seed 42 / 43), LightGBM (models.LGBM_PARAMS), features =
global k-mer spectrum (+) the B=8 position-binned spectrum (featurize.binned_kmer_spectrum; bin b covers
[b*L//8, (b+1)*L//8) of each sequence). Both reference arms are re-fitted and asserted equal to the
card's spectrum_mcc / positional_mcc.

Readouts (paired 1000-resample bootstrap over test sequences, RandomState(42), the SAME resamples for
every arm; 95 % percentile CIs):
  add_b   global + bin b only;           gain_b = MCC(add_b) - MCC(global)
  drop_b  global + all bins except b;    loss_b = MCC(B=8) - MCC(drop_b)
  gain share of each block (global, bin 0..7) in the B=8 model (LightGBM split gain), and the top
  features by gain with their class means on train;
  descriptive per-bin class profiles on the pooled released sequences: fraction with >= 1 TBP hit
  (paper's TBP PSSM, 1.3 bits/pos, both strands, hit assigned to the bin of its centre), fraction with
  'TATA' in the bin, mean GC fraction and mean CpG count per bin.
Reading rule (fixed now):
  L1 carrier bins = bins with gain_b >= 0.5 x the full B=8 delta AND gain_b's CI excluding 0; if none,
     the positional signal is 'distributed' (no single bin carries it).
  L2 a bin is necessary if loss_b's CI excludes 0 (positive).
  M  TATA placement is named as the mechanism only if (i) a carrier bin is the bin holding the class-1
     TBP-hit peak, (ii) in that bin the class-1 minus class-0 TBP-hit rate is >= 0.10 and (iii) one of
     that bin's top-5 gain features contains 'TATA'. Otherwise TATA placement is NOT claimed and the
     carrier is described by its bins and top features only.
Output: results/upgrades/gue_posthoc_r2/<task>.json (checkpoint), assembled by summarize_gue.py into
results/upgrades/gue_positional_localisation.csv. CPU only, LightGBM on GUE_THREADS threads.
Run: GUE_THREADS=3 bibm/lowcpu.sh python src/upgrades/gue_suite/run_gue_posthoc_r2.py [--tasks a,b] [--force]
"""
import os

N_THREADS = int(os.environ.get("GUE_THREADS", "3"))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["GUE_THREADS"] = str(N_THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_gue as R  # noqa: E402  (paths, thread limits, helpers)
import gue_data as G  # noqa: E402

C, DG = R.C, R.DG
import featurize  # noqa: E402

SEED, BOOT, B = 42, 1000, 8
TASKS = ["prom_300_tata", "prom_core_tata", "splice_reconstructed", "nt_promoter_tata"]
OUT_DIR = os.path.join(C.RESULTS_DIR, "gue_posthoc_r2")
P15_PARTS = os.path.join(C.RESULTS_DIR, "gue_paper15_reference_parts")


def load(task):
    """(tr, ytr, te, yte, pooled released seqs, pooled y, reference card values)."""
    if task.startswith("nt_"):
        tr, ytr, te, yte = C.load_original(task)
        ref = pd.concat([pd.read_csv(os.path.join(C.ROOT, "results", f)) for f in ("results.csv", "results_nt.csv")])
        ref = ref[(ref.model == "lgbm") & (ref.selected_by_val == 1)].set_index("dataset")
        assert len(ytr) == int(ref.loc[task, "n_train"]) and len(yte) == int(ref.loc[task, "n_test"]), task
        p = json.load(open(os.path.join(P15_PARTS, f"{task}.json")))
        refv = dict(k=p["k"], spectrum_mcc=p["pos_spectrum_mcc"], positional_mcc=p["pos_binned_mcc"],
                    delta=p["pos_delta"], source="gue_paper15_reference_parts")
        pool_s, pool_y = list(tr) + list(te), np.concatenate([ytr, yte])
    else:
        d = G.load_gue(task)
        tr, ytr, te, yte = d["train_seqs"], d["y_train"], d["test_seqs"], d["y_test"]
        c = json.load(open(os.path.join(R.CARD_DIR, f"{task}.json")))
        p2 = c["route2_positional"]
        refv = dict(k=c["k"], spectrum_mcc=p2["spectrum_mcc"], positional_mcc=p2["positional_mcc"],
                    delta=p2["delta_mcc"], source="gue_cards")
        pool_s = tr + d["dev_seqs"] + te
        pool_y = np.concatenate([ytr, d["y_dev"], yte])
    return tr, ytr, te, yte, pool_s, pool_y, refv


def blocks(seqs, k):
    """[global, bin0, ..., bin7] CSR blocks (chunked like run_gue.pos_arm)."""
    out = [[] for _ in range(B + 1)]
    for i in range(0, len(seqs), R.CHUNK):
        ss = seqs[i:i + R.CHUNK]
        out[0].append(featurize.kmer_spectrum(ss, k))
        for b in range(B):
            subs = [s[(b * len(s)) // B:((b + 1) * len(s)) // B] for s in ss]
            out[b + 1].append(featurize.kmer_spectrum(subs, k))
    return [sparse.vstack(x, format="csr") for x in out]


def fit_pred(blk_tr, ytr, blk_te, use):
    Xtr = sparse.hstack([blk_tr[j] for j in use], format="csr")
    Xte = sparse.hstack([blk_te[j] for j in use], format="csr")
    m = R.models.build_model("lgbm", SEED, len(np.unique(ytr)))
    m.fit(Xtr, ytr)
    return m.classes_[np.argmax(m.predict_proba(Xte), axis=1)], m


def profiles(seqs, y, tbp):
    """Per-class, per-bin descriptive profiles on the pooled released sequences."""
    L_m = tbp["pssm"].shape[1]
    codes = C.motif_match.encode_sequences(seqs)
    n, S = codes.shape
    npos = S - L_m + 1
    neg_inf = np.full((1, L_m), -1e9, dtype=np.float32)
    fwd = np.vstack([tbp["pssm"], neg_inf]); rev = np.vstack([tbp["pssm"][::-1, ::-1], neg_inf])

    def scan(a):
        sc = np.zeros((n, npos), np.float32)
        for i in range(L_m):
            sc += a[codes[:, i:i + npos], i]
        return sc / L_m
    hits = (scan(fwd) >= DG.TATA_BITS) | (scan(rev) >= DG.TATA_BITS)      # (n, npos) window starts
    lens = np.array([len(s) for s in seqs])
    out = {}
    for c in sorted(np.unique(y).tolist()):
        m = y == c
        tbp_rate, tata_rate, gc, cpg = [], [], [], []
        for b in range(B):
            lo = (b * lens[m]) // B; hi = ((b + 1) * lens[m]) // B
            centre = np.arange(npos)[None, :] + L_m // 2
            inb = (centre >= lo[:, None]) & (centre < hi[:, None])
            tbp_rate.append(round(float((hits[m] & inb).any(axis=1).mean()), 4))
            subs = [s[(b * len(s)) // B:((b + 1) * len(s)) // B] for s, keep in zip(seqs, m) if keep]
            tata_rate.append(round(float(np.mean(["TATA" in s for s in subs])), 4))
            gc.append(round(float(np.mean([(s.count("G") + s.count("C")) / max(len(s), 1) for s in subs])), 4))
            cpg.append(round(float(np.mean([s.count("CG") for s in subs])), 4))
        out[str(c)] = dict(n=int(m.sum()), tbp_hit_rate_by_bin=tbp_rate, tata_4mer_rate_by_bin=tata_rate,
                           gc_by_bin=gc, cpg_count_by_bin=cpg,
                           tbp_any_hit=round(float(hits[m].any(axis=1).mean()), 4))
    edges = [int((b * int(np.median(lens))) // B) for b in range(B + 1)]
    return out, edges


def run(task):
    T0 = time.time()
    tr, ytr, te, yte, pool_s, pool_y, refv = load(task)
    k = refv["k"]
    ptr, pytr, cap1 = R.cap_split(tr, ytr, R.POS_CAP, SEED)
    pte, pyte, cap2 = R.cap_split(te, yte, R.POS_CAP, SEED + 1)
    btr, bte = blocks(ptr, k), blocks(pte, k)
    vocab = featurize.vocabulary(k)
    preds = {}
    preds["global"], _ = fit_pred(btr, pytr, bte, [0])
    preds["B8"], m8 = fit_pred(btr, pytr, bte, list(range(B + 1)))
    mg, m8v = matthews_corrcoef(pyte, preds["global"]), matthews_corrcoef(pyte, preds["B8"])
    assert abs(mg - refv["spectrum_mcc"]) < 1e-4 and abs(m8v - refv["positional_mcc"]) < 1e-4, (task, mg, m8v, refv)
    for b in range(B):
        preds[f"add{b}"], _ = fit_pred(btr, pytr, bte, [0, b + 1])
        preds[f"drop{b}"], _ = fit_pred(btr, pytr, bte, [0] + [j + 1 for j in range(B) if j != b])
        print(f"  [{task}] bin {b}: add MCC {matthews_corrcoef(pyte, preds[f'add{b}']):.4f}  drop MCC "
              f"{matthews_corrcoef(pyte, preds[f'drop{b}']):.4f}", flush=True)
    # paired bootstrap, identical resamples for every arm
    rng = np.random.RandomState(SEED); n = len(pyte)
    idxs = [rng.randint(0, n, n) for _ in range(BOOT)]

    def mcc_b(p):
        return np.array([matthews_corrcoef(pyte[i], p[i]) for i in idxs])
    bg, b8 = mcc_b(preds["global"]), mcc_b(preds["B8"])
    full = m8v - mg
    per_bin = []
    for b in range(B):
        ma, md = matthews_corrcoef(pyte, preds[f"add{b}"]), matthews_corrcoef(pyte, preds[f"drop{b}"])
        ga = mcc_b(preds[f"add{b}"]) - bg; ld = b8 - mcc_b(preds[f"drop{b}"])
        glo, ghi = np.percentile(ga, [2.5, 97.5]); llo, lhi = np.percentile(ld, [2.5, 97.5])
        per_bin.append(dict(bin=b, add_mcc=round(float(ma), 4), gain=round(float(ma - mg), 4),
                            gain_lo=round(float(glo), 4), gain_hi=round(float(ghi), 4),
                            gain_share_of_full=round(float((ma - mg) / full), 4) if full else None,
                            drop_mcc=round(float(md), 4), loss=round(float(m8v - md), 4),
                            loss_lo=round(float(llo), 4), loss_hi=round(float(lhi), 4)))
    # split-gain importance per block and top features of the B=8 model
    imp = m8.booster_.feature_importance(importance_type="gain")
    V = len(vocab)
    blk_gain = [float(imp[j * V:(j + 1) * V].sum()) for j in range(B + 1)]
    tot = sum(blk_gain) or 1.0
    Xtr8 = [btr[j] for j in range(B + 1)]
    classes = sorted(np.unique(pytr).tolist())
    top = []
    for f in np.argsort(-imp)[:15]:
        j, v = divmod(int(f), V)
        col = Xtr8[j][:, v].toarray().ravel()
        top.append(dict(block=("global" if j == 0 else f"bin{j - 1}"), kmer=vocab[v],
                        gain_share=round(float(imp[f] / tot), 4),
                        class_means={str(c): round(float(col[pytr == c].mean()), 5) for c in classes}))
    top_by_bin = {}
    for b in range(B):
        sl = imp[(b + 1) * V:(b + 2) * V]
        top_by_bin[str(b)] = [vocab[int(v)] for v in np.argsort(-sl)[:5] if sl[int(v)] > 0]
    binary = set(np.unique(pool_y).tolist()) == {0, 1}
    prof, edges = profiles(pool_s, pool_y, C.load_tbp())
    # reading rule
    carriers = [r["bin"] for r in per_bin if r["gain_share_of_full"] is not None and r["gain_share_of_full"] >= 0.5
                and r["gain_lo"] > 0]
    necessary = [r["bin"] for r in per_bin if r["loss_lo"] > 0]
    tata_mech = None
    if binary:
        t1 = prof["1"]["tbp_hit_rate_by_bin"]; t0 = prof["0"]["tbp_hit_rate_by_bin"]
        peak = int(np.argmax(t1))
        tata_mech = bool(peak in carriers and (t1[peak] - t0[peak]) >= 0.10
                         and any("TATA" in km for km in top_by_bin[str(peak)]))
        tata_detail = dict(class1_tbp_peak_bin=peak, tbp_rate_diff_at_peak=round(t1[peak] - t0[peak], 4),
                           peak_bin_top5=top_by_bin[str(peak)])
    else:
        tata_detail = {}
    out = dict(task=task, k=k, reference=refv, capped=bool(cap1 or cap2), n_train_used=len(pytr),
               n_test_used=n, bin_edges_bp=edges, spectrum_mcc=round(float(mg), 4),
               positional_mcc=round(float(m8v), 4), full_delta=round(float(full), 4),
               per_bin=per_bin, block_gain_share={("global" if j == 0 else f"bin{j - 1}"): round(g / tot, 4)
                                                  for j, g in enumerate(blk_gain)},
               top_features=top, top5_by_bin=top_by_bin, profiles=prof,
               carrier_bins=carriers, necessary_bins=necessary,
               reading=("distributed (no single carrier bin)" if not carriers else
                        "carrier bin(s) " + ",".join(map(str, carriers))),
               tata_placement_is_mechanism=tata_mech, tata_check=tata_detail,
               seconds=round(time.time() - T0, 1), posthoc="PH3 (fix round 1)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in [x for x in a.tasks.split(",") if x]:
        p = os.path.join(OUT_DIR, f"{t}.json")
        if os.path.exists(p) and not a.force:
            print(f"[{t}] exists, skipping", flush=True)
            continue
        print(f"[{t}] start", flush=True)
        o = run(t)
        tmp = p + ".tmp"
        json.dump(o, open(tmp, "w"), indent=1)
        os.replace(tmp, p)
        print(f"[{t}] carriers={o['carrier_bins']} necessary={o['necessary_bins']} "
              f"tata_mechanism={o['tata_placement_is_mechanism']} ({o['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
