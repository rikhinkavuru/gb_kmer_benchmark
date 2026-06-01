#!/usr/bin/env python3
"""Upgrade 1 -- frozen FM-probe: does a pretrained genomic foundation model lose enhancer
score when the composition artifact is removed?

For human_enhancers_cohn, nt_enhancers, and drosophila_enhancers_stark (clean control), and
for each cleaning arm {original, tata_flag, gc_match, comp_equalized}, we:
  1. Extract frozen HyenaDNA embeddings (mean-pool + [SEP]; see extract_embeddings.py) for the
     arm's train/test sequences -- CPU forward passes, cached to .npz.
  2. Train a LightGBM head AND a LogisticRegression head on the frozen embeddings (same
     hyperparameters as the k-mer pipeline; dense path).
  3. Report test accuracy / MCC / AUROC with 1000-resample percentile bootstrap CIs.

Headline = the cohn / nt_enhancers enhancer MCC (and AUROC) DROP from original -> cleaned for
the LightGBM head on mean-pooled embeddings; drosophila (clean control) should barely move.
This converts the paper's conditional claim ("a spurious signal exists and our k-mer
classifier exploits it") into a measured one about an independent, pretrained model family.

Efficiency: every arm's sequences are a subset of the ORIGINAL split's sequences, so each
unique sequence is embedded exactly once (per dataset+split), cached, and every arm is
assembled by lookup. Reruns are free (embeddings cached; heads are cheap).

The MCC deltas are UNPAIRED (each arm has its own, differently-sized test set, since cleaning
removes negatives); like Upgrade 2 the composition-equalized arm retains few negatives, so its
benchmark MCC is underpowered -- read alongside the Upgrade-2 AUROC collapse. CPU-only, seed 42.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _common as C
import extract_embeddings as X

DATASETS = ["human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark"]
ROLE = {"human_enhancers_cohn": "contaminated", "nt_enhancers": "contaminated",
        "drosophila_enhancers_stark": "clean control"}
ARMS = ["original", "tata_flag", "gc_match", "comp_equalized"]
POOLINGS = ["mean", "sep"]
HEADS = ["lgbm", "lr"]


def arm_split_seqs(task, arm):
    """(tr_seqs, ytr, te_seqs, yte) for one cleaning arm. Every arm subsets the original."""
    if arm == "original":
        return C.load_original(task)
    if arm == "tata_flag":
        d = os.path.join(C.ROOT, "cleaned_splits")
        return C.load_csv_pair(os.path.join(d, f"{task}_train.csv"), os.path.join(d, f"{task}_test.csv"))
    if arm in ("gc_match", "comp_equalized"):
        tag = "gcmatch" if arm == "gc_match" else "comp_equalized"
        d = C.SPLITS_V2_DIR
        return C.load_csv_pair(os.path.join(d, f"{task}_{tag}_train.csv"),
                               os.path.join(d, f"{task}_{tag}_test.csv"))
    raise ValueError(arm)


def original_embeddings(task, split, n_expected, cache_dir):
    """Load cached mean/sep embeddings for the ORIGINAL split (NO torch). Errors if the
    extract step has not been run, since this process must not load torch alongside
    LightGBM (double-OpenMP segfault on macOS)."""
    cached = X.load_cached(cache_dir, task, "orig", split)
    if cached is None:
        sys.exit(f"ERROR: no cached embeddings for {task} orig_{split} in {cache_dir}.\n"
                 f"Run the torch-only extract step first:\n"
                 f"  python src/upgrades/fm_probe/extract_embeddings.py --datasets {task} --cache {cache_dir}")
    if cached["mean"].shape[0] != n_expected:
        sys.exit(f"ERROR: cached {task} orig_{split} has {cached['mean'].shape[0]} rows, expected "
                 f"{n_expected}. Delete the stale cache and re-extract.")
    return cached["mean"], cached["sep"]


def gather(seqs, idx_of, mean, sep):
    """Look up each sequence's row in the original-split embedding arrays."""
    rows = [idx_of[s] for s in seqs]
    return mean[rows], sep[rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "fm_probe.csv"))
    ap.add_argument("--emb-cache", default=os.path.join(C.ROOT, "cache", "fm_embeddings"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--boot", type=int, default=C.BOOT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tasks = [t for t in args.datasets.split(",") if t]
    print("=" * 100)
    print(f"UPGRADE 1 -- FROZEN FM-PROBE  |  model={X.MODEL_TAG} (rev {X.MODEL_REVISION[:10]})  seed={args.seed}")
    print(f"datasets={tasks}  arms={ARMS}  poolings={POOLINGS}  heads={HEADS}  boot={args.boot}")
    print("LightGBM-only process: embeddings are read from cache (extract step runs torch separately).")
    print("=" * 100)

    rows = []
    for task in tasks:
        rng = np.random.RandomState(args.seed)        # per-dataset deterministic stream
        otr_seqs, _, ote_seqs, _ = C.load_original(task)
        mtr, str_ = original_embeddings(task, "train", len(otr_seqs), args.emb_cache)
        mte, ste = original_embeddings(task, "test", len(ote_seqs), args.emb_cache)
        idx_tr = {s: i for i, s in enumerate(otr_seqs)}
        idx_te = {s: i for i, s in enumerate(ote_seqs)}
        emb_tr = {"mean": mtr, "sep": str_}; emb_te = {"mean": mte, "sep": ste}
        print(f"\n[{task}] ({ROLE[task]}) embeddings ready "
              f"(train {len(otr_seqs)}, test {len(ote_seqs)})")

        for arm in ARMS:
            try:
                tr_seqs, ytr, te_seqs, yte = arm_split_seqs(task, arm)
            except FileNotFoundError:
                print(f"   {arm:<15} SKIPPED (split not found; run run_composition_clean.py / run_cleaning.py)")
                continue
            # coverage check: every arm sequence must be in the original split
            miss = sum(1 for s in tr_seqs if s not in idx_tr) + sum(1 for s in te_seqs if s not in idx_te)
            assert miss == 0, f"{task}/{arm}: {miss} sequences not in original split (cannot reuse embeddings)"
            for pooling in POOLINGS:
                Xtr = gather(tr_seqs, idx_tr, emb_tr["mean"], emb_tr["sep"])[0 if pooling == "mean" else 1]
                Xte = gather(te_seqs, idx_te, emb_te["mean"], emb_te["sep"])[0 if pooling == "mean" else 1]
                for head in HEADS:
                    r = C.fit_eval_boot(Xtr, ytr, Xte, yte, head, args.seed, args.boot, rng)
                    rows.append(dict(dataset=task, role=ROLE[task], arm=arm, pooling=pooling, head=head,
                        n_train=len(ytr), n_test=r["n_test"], accuracy=r["acc"],
                        mcc=r["mcc"], mcc_ci=f"[{r['mcc_lo']:.3f},{r['mcc_hi']:.3f}]",
                        auroc=r["auroc"], auroc_ci=f"[{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]"))
            # concise line for the primary (mean-pool, lgbm)
            pr = next(x for x in rows if x["dataset"] == task and x["arm"] == arm
                      and x["pooling"] == "mean" and x["head"] == "lgbm")
            print(f"   {arm:<15} [mean/lgbm] MCC={pr['mcc']:.3f} {pr['mcc_ci']}  "
                  f"AUROC={pr['auroc']:.3f} {pr['auroc_ci']}  (n_test={pr['n_test']})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    report = build_report(df)
    print("\n" + report)
    with open(args.out.replace(".csv", "_interpretation.txt"), "w") as fh:
        fh.write(report)
    print(f"\nWrote {args.out} ({len(df)} rows) + interpretation. Embeddings cached in {args.emb_cache}.")


def build_report(df):
    L = []; add = L.append
    add("=" * 100)
    add(f"UPGRADE 1 -- FROZEN FM-PROBE (HyenaDNA-tiny-16k-d128) : enhancer score vs cleaning arm")
    add("=" * 100)
    prim = df[(df["pooling"] == "mean") & (df["head"] == "lgbm")]
    hdr = f"{'dataset':<28}{'arm':<16}{'n_test':>7}{'MCC':>8} {'MCC 95% CI':<16}{'AUROC':>7} {'AUROC 95% CI':<16}"
    add(hdr); add("-" * len(hdr))
    for _, r in prim.iterrows():
        add(f"{r['dataset']:<28}{r['arm']:<16}{r['n_test']:>7}{r['mcc']:>8.3f} {r['mcc_ci']:<16}"
            f"{r['auroc']:>7.3f} {r['auroc_ci']:<16}")
    add("\nHeadline (mean-pool, LightGBM head) -- original -> cleaned delta:")
    for ds in prim["dataset"].unique():
        g = prim[prim["dataset"] == ds].set_index("arm")
        if "original" not in g.index:
            continue
        o = g.loc["original"]
        add(f"  {ds} [{g['role'].iloc[0]}]: original MCC={o['mcc']:.3f} AUROC={o['auroc']:.3f}")
        for arm in ["tata_flag", "gc_match", "comp_equalized"]:
            if arm in g.index:
                c = g.loc[arm]
                add(f"      -> {arm:<15} MCC={c['mcc']:.3f} (Δ{c['mcc']-o['mcc']:+.3f})  "
                    f"AUROC={c['auroc']:.3f} (Δ{c['auroc']-o['auroc']:+.3f})  n_test={c['n_test']}")
    add("\nReading: if the FROZEN pretrained genomic FM loses enhancer MCC/AUROC when the")
    add("composition artifact is removed (cohn, nt_enhancers) while the clean control")
    add("(drosophila) barely moves, then a pretrained model -- not just our k-mer classifier --")
    add("was exploiting the composition shortcut. Deltas are UNPAIRED (cleaned test sets are")
    add("smaller); the comp_equalized arm retains few negatives so its MCC is underpowered --")
    add("read with the Upgrade-2 TATA/GC AUROC collapse. Mean-pool + [SEP], LR + LightGBM heads,")
    add("all poolings/heads in fm_probe.csv. No PyTorch in the diagnostic pipeline; PyTorch used")
    add("only for this frozen FM feature extraction. Seed 42, 1000-resample bootstrap.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
