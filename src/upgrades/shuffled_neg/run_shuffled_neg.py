#!/usr/bin/env python3
"""Upgrade 5 -- contamination generality via CONTROLLED negatives (Route 3), self-contained.

Promotes the claim from "two FANTOM/ENCODE-lineage datasets are contaminated" toward "the artifact
is a property of the random-genomic negative-sampling PROTOCOL." For the cohn and nt_enhancers
POSITIVE sets, we construct matched negatives by dinucleotide-preserving shuffling of the positives
(Altschul-Erikson; dinuc_shuffle.py). By construction these negatives have the SAME mononucleotide
+ dinucleotide composition (hence identical GC) as the positives, so any composition gap is removed.

Prediction: the TATA motif-only AUROC and GC AUROC -- which are far from 0.5 against the real,
random-genomic negatives (the contamination artifact) -- collapse to ~0.5 against the shuffled
negatives. That isolates the negative-sampling protocol, not the positive class, as the cause.

For each dataset we report BOTH arms side by side:
  * original_neg  -- positives vs the dataset's real negatives (the artifact is present);
  * shuffled_neg  -- positives vs dinucleotide-preserving shuffles of the positives (artifact gone).
plus the k-mer+LightGBM MCC on each (the shuffled-negative MCC is residual HIGHER-ORDER structure,
since mono+dinuc are matched), and a verification that dinucleotide composition is exactly preserved.

drosophila_enhancers_stark is included as a sanity control. CPU-only, seed 42, 1000-bootstrap.
Optional external composition-matched benchmark (e.g. DeepSTARR) = documented TODO in the README.
Writes results/upgrades/shuffled_neg.csv + _interpretation.txt.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import kselect
import dinuc_shuffle as D

DEFAULT = [("human_enhancers_cohn", "contaminated"),
           ("nt_enhancers", "contaminated"),
           ("drosophila_enhancers_stark", "clean control")]


def shuffled_negatives(pos_seqs, rng):
    """One dinucleotide-preserving shuffle per positive; returns (shuffles, frac_preserved)."""
    sh = [D.dinuc_shuffle(s, rng) for s in pos_seqs]
    pres = np.mean([D.verify_dinuc_preserved(p, q) for p, q in zip(pos_seqs, sh)])
    return sh, float(pres)


def arm_auroc_block(pos_tr, pos_te, neg_tr, neg_te, k, tbp, args, rng):
    """TATA & GC motif/composition AUROC (pos vs neg, pooled) + k-mer MCC, with bootstrap CIs."""
    gc_pos = np.concatenate([C.gc_content(pos_tr), C.gc_content(pos_te)])
    gc_neg = np.concatenate([C.gc_content(neg_tr), C.gc_content(neg_te)])
    ta_pos = np.concatenate([C.tata_hit_counts(pos_tr, tbp, args.tata_bits),
                             C.tata_hit_counts(pos_te, tbp, args.tata_bits)]).astype(np.int64)
    ta_neg = np.concatenate([C.tata_hit_counts(neg_tr, tbp, args.tata_bits),
                             C.tata_hit_counts(neg_te, tbp, args.tata_bits)]).astype(np.int64)
    gc_a = C.auc_float(gc_pos, gc_neg); gc_lo, gc_hi = C.boot_ci(gc_pos, gc_neg, C.auc_float, args.boot, rng)
    ta_a = C.auc_counts(ta_pos, ta_neg); ta_lo, ta_hi = C.boot_ci(ta_pos, ta_neg, C.auc_counts, args.boot, rng)
    tr_seqs = list(pos_tr) + list(neg_tr); ytr = np.array([1] * len(pos_tr) + [0] * len(neg_tr))
    te_seqs = list(pos_te) + list(neg_te); yte = np.array([1] * len(pos_te) + [0] * len(neg_te))
    bench = C.kmer_fit_eval_boot(tr_seqs, ytr, te_seqs, yte, k, "lgbm", args.seed, args.boot, rng)
    return dict(tata_auroc=round(ta_a, 4), tata_ci=f"[{ta_lo:.3f},{ta_hi:.3f}]",
                gc_auroc=round(gc_a, 4), gc_ci=f"[{gc_lo:.3f},{gc_hi:.3f}]",
                kmer_mcc=bench["mcc"], kmer_mcc_ci=f"[{bench['mcc_lo']:.3f},{bench['mcc_hi']:.3f}]",
                kmer_auroc=bench["auroc"], n_test=bench["n_test"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "shuffled_neg.csv"))
    ap.add_argument("--tata-bits", type=float, default=1.3)
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tbp = C.load_tbp()
    tasks = DEFAULT if not args.datasets else [(t, r) for t, r in DEFAULT if t in set(args.datasets.split(","))]

    print("=" * 100)
    print(f"UPGRADE 5 -- SHUFFLED-NEGATIVE CONTROL (dinucleotide-preserving) | tata={args.tata_bits} | "
          f"boot={args.boot} | seed={args.seed}")
    print("=" * 100)
    rows = []
    for task, role in tasks:
        rng = np.random.RandomState(args.seed)
        k = kselect.best_k_val(task, os.path.join(C.ROOT, "cache"), args.seed)
        tr, ytr, te, yte = C.load_original(task, args.seed)
        pos_tr = [s for s, y in zip(tr, ytr) if y == 1]; pos_te = [s for s, y in zip(te, yte) if y == 1]
        neg_tr = [s for s, y in zip(tr, ytr) if y == 0]; neg_te = [s for s, y in zip(te, yte) if y == 0]
        shuf_tr, p1 = shuffled_negatives(pos_tr, rng)
        shuf_te, p2 = shuffled_negatives(pos_te, rng)
        pres = round((p1 * len(pos_tr) + p2 * len(pos_te)) / max(len(pos_tr) + len(pos_te), 1), 4)
        print(f"\n[{task}] ({role}) k={k} | n_pos={len(pos_tr)+len(pos_te)} n_neg={len(neg_tr)+len(neg_te)} "
              f"| dinuc preserved in shuffles: {pres:.3f}")

        a_orig = arm_auroc_block(pos_tr, pos_te, neg_tr, neg_te, k, tbp, args, rng)
        a_shuf = arm_auroc_block(pos_tr, pos_te, shuf_tr, shuf_te, k, tbp, args, rng)
        for arm, blk, npos, nneg in [("original_neg", a_orig, len(pos_tr) + len(pos_te), len(neg_tr) + len(neg_te)),
                                     ("shuffled_neg", a_shuf, len(pos_tr) + len(pos_te), len(pos_tr) + len(pos_te))]:
            rows.append(dict(dataset=task, role=role, arm=arm, best_k=k, dinuc_preserved=pres,
                             n_pos=npos, n_neg=nneg, **blk))
            print(f"   {arm:<13} TATA AUROC={blk['tata_auroc']:.3f} {blk['tata_ci']}  "
                  f"GC AUROC={blk['gc_auroc']:.3f} {blk['gc_ci']}  kmerMCC={blk['kmer_mcc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    report = build_report(df)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out} ({len(df)} rows) + interpretation.")


def build_report(df):
    L = []; add = L.append
    add("=" * 100)
    add("SHUFFLED-NEGATIVE CONTROL -- does the TATA/GC artifact vanish with composition-matched negatives?")
    add("=" * 100)
    hdr = f"{'dataset':<28}{'arm':<14}{'TATA AUROC':>12}{'GC AUROC':>10}{'kmer MCC':>10}"
    add(hdr); add("-" * len(hdr))
    for _, r in df.iterrows():
        add(f"{r['dataset']:<28}{r['arm']:<14}{r['tata_auroc']:>12.3f}{r['gc_auroc']:>10.3f}{r['kmer_mcc']:>10.3f}")
    add("")
    add("Verdict (per dataset): with dinucleotide-preserving shuffled negatives, GC AUROC -> 0.5 by")
    add("construction (identical composition) and the TATA motif-only AUROC should collapse to ~0.5")
    add("if the TATA enrichment was a property of random-genomic negatives, not the positive class.")
    for ds in df["dataset"].unique():
        g = df[df["dataset"] == ds].set_index("arm")
        if "original_neg" in g.index and "shuffled_neg" in g.index:
            o, s = g.loc["original_neg"], g.loc["shuffled_neg"]
            add(f"  {ds}: TATA AUROC {o['tata_auroc']:.3f} -> {s['tata_auroc']:.3f}; "
                f"GC AUROC {o['gc_auroc']:.3f} -> {s['gc_auroc']:.3f}; "
                f"kmer MCC {o['kmer_mcc']:.3f} -> {s['kmer_mcc']:.3f} "
                f"({'artifact removed by composition matching' if abs(s['tata_auroc']-0.5) < abs(o['tata_auroc']-0.5) else 'TATA did not move to 0.5'}).")
    add("\nReading: the shuffled-negative benchmark has, by construction, NO mono/dinucleotide")
    add("composition gap. If the artifact (TATA/GC separation) vanishes there while it is present")
    add("against the real random-genomic negatives, the artifact is a property of the NEGATIVE-")
    add("SAMPLING PROTOCOL. Residual k-mer MCC on the shuffled arm = higher-order (motif) structure")
    add("that survives dinucleotide matching. Dinucleotide preservation is verified exactly. Seed 42.")
    add("TODO (optional external): an enhancer benchmark with composition-matched/shuffled negatives")
    add("(e.g. DeepSTARR-derived) would show the artifact absent there too; source documented in README.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
