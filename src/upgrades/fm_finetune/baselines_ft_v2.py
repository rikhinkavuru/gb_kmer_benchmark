#!/usr/bin/env python3
"""POST HOC (added 2026-09-26, after every fine-tuning unit had been scored): the paper's two
non-fine-tuned model families re-fit on EXACTLY the fine-tuning arms, so fine-tuned, frozen and
k-mer models can be compared on identical test sequences with identical bootstrap resamples.

Why: Reviewer 1 (comment 2) asks whether fully fine-tuned FMs "still exploit negative-set
compositional shortcuts". The paper's evidence so far is a frozen-embedding probe (fm_probe) and the
k-mer LightGBM (composition_clean, psm_sensitivity). Those were scored on related but not identical
test sets (e.g. psm_sensitivity's matched arm is a re-drawn seed-42 match, and neither module has
the size-matched random arm used here). Re-fitting them on the fine-tuning arms makes the
comparison exact. Nothing in the pre-registered analysis (eval_ft_v2.py) changes; these runs are
scored by the same code in a SEPARATE output set (posthoc_ft_v2.py).

Models (all LightGBM with the benchmark's fixed hyperparameters, models.LGBM_PARAMS, seed 42,
num_threads 2, deterministic; one seed, as in the paper -- LightGBM is deterministic given data):
  kmer_lgbm             k-mer spectrum at the validation-selected k (results/val_selection.csv
                        new_k: cohn 5, nt_enhancers 4, drosophila 4) -- the paper's classifier
  hyenadna_frozen_mean  frozen HyenaDNA-tiny-16k-d128 mean-pool embedding + LightGBM head
  hyenadna_frozen_sep   frozen HyenaDNA [SEP]-token embedding + LightGBM head (the paper's headline cell)
  dnabert2_frozen_mean  frozen DNABERT-2-117M mean-pool embedding + LightGBM head (paper headline cell)
  dnabert2_frozen_cls   frozen DNABERT-2 [CLS] embedding + LightGBM head
Frozen embeddings: the released cache (Zenodo 10.5281/zenodo.20494982) as fetched by
homology_recluster/fetch_fm_cache.py into external/homology_recluster/fm_embeddings/; every file is
re-verified here against the sha256 in results/upgrades/fm_embeddings_manifest.csv, and its row count
and labels against the verified split files. DNABERT-2 has no drosophila embeddings (512-token
context), as in the paper.
Train arms: original, gc_match, comp_equalized, random (the fine-tuning arms, from
external/fm_finetune/splits/, membership verified by tests/test_ft_v2.py). Each model is fit on the
arm's FULL train membership (no 10% validation carve: nothing is early-stopped), in the arm's list
order, and scores every ORIGINAL test sequence, so it can be evaluated on every eval arm.
Outputs (same format as run_ft_v2.py, so eval_ft_v2.py's functions score them unchanged):
  external/fm_finetune/baseline_preds/<model>__<dataset>__<arm>__s42.{csv.gz,json}
  results/upgrades/fm_finetune_v2_baselines_repro.csv  reproduction check against the published
      numbers (psm_sensitivity_decision.csv A0 / paper_anchor_auroc; fm_paired*.csv)
Usage: bibm/lowcpu.sh <torch-free python> src/upgrades/fm_finetune/baselines_ft_v2.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, ROOT, os.path.join(ROOT, "src", "upgrades")):
    if p not in sys.path:
        sys.path.insert(0, p)
import eval_ft_v2 as ev  # noqa: E402
import featurize  # noqa: E402
import models as gbmodels  # noqa: E402

SEED = 42
OUT_DIR = os.path.join(ROOT, "external", "fm_finetune", "baseline_preds")
EMB_DIR = os.path.join(ROOT, "external", "homology_recluster", "fm_embeddings")
MANIFEST = os.path.join(ROOT, "results", "upgrades", "fm_embeddings_manifest.csv")
DATASETS = ("human_enhancers_cohn", "nt_enhancers", "drosophila_enhancers_stark")
ARMS = ("original", "gc_match", "comp_equalized", "random")
EMB_TAG = {"hyenadna": "hyenadna-tiny-16k-d128", "dnabert2": "dnabert2-117m"}
POOLS = {"hyenadna": ("mean", "sep"), "dnabert2": ("mean", "cls")}


def val_k(ds):
    v = pd.read_csv(os.path.join(ROOT, "results", "val_selection.csv")).set_index("dataset")
    return int(v.loc[ds, "new_k"])


def load_split(ds):
    out = {}
    for sp in ("train", "test"):
        out[sp] = pd.read_csv(os.path.join(ev.SPLIT_DIR, f"{ds}__{sp}.csv.gz"))
        assert np.array_equal(out[sp]["idx"].to_numpy(), np.arange(len(out[sp])))
    return out


def arm_order(df, arm):
    col = df[f"{arm}_pos"].to_numpy()
    sel = np.where(col >= 0)[0]
    return sel[np.argsort(col[sel])]


def load_emb(ds, fm, split, y_expected, man):
    f = f"emb__{ds}__orig__{split}__{EMB_TAG[fm]}.npz"
    p = os.path.join(EMB_DIR, f)
    if not os.path.exists(p):
        return None
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h == man.loc[f, "sha256"], f"sha256 mismatch for {f}"
    d = np.load(p)
    assert int(man.loc[f, "n_sequences"]) == len(y_expected) == d["y"].shape[0], f"{f}: row count"
    assert np.array_equal(d["y"].astype(np.int64), y_expected), f"{f}: labels != split file"
    return {k: d[k].astype(np.float32) for k in d.files if k != "y"}


def fit_predict(Xtr, ytr, Xte):
    m = gbmodels.build_model("lgbm", SEED, 2)
    m.set_params(n_jobs=2)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def write_unit(model, tag, ds, arm, y_te, prob, n_fit, n_fit_neg, secs, extra):
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"{model}__{ds}__{arm}__s{SEED}")
    pd.DataFrame(dict(split="test", idx=np.arange(len(y_te)), label=y_te,
                      prob=np.round(prob.astype(np.float64), 7))).to_csv(stem + ".csv.gz", index=False)
    meta = dict(model=model, model_tag=tag, dataset=ds, train_arm=arm, seed=SEED, n_fit=int(n_fit),
                n_val=0, n_fit_neg=int(n_fit_neg), n_fit_pos=int(n_fit - n_fit_neg),
                best_epoch=None, best_val_auroc=None, epochs_run=None, history=[],
                n_params_total=None, n_params_trainable=None, device="cpu",
                recipe=dict(tag=tag, eval_batch=None, head="lgbm", lgbm_params="models.LGBM_PARAMS",
                            num_threads=2, seed=SEED),
                train_secs=round(secs, 1), posthoc=True, **extra)
    json.dump(meta, open(stem + ".json", "w"), indent=1)


def main():
    man = pd.read_csv(MANIFEST).set_index("file")
    t0 = time.time()
    for ds in DATASETS:
        sp = load_split(ds)
        tr, te = sp["train"], sp["test"]
        ytr_all = tr["label"].to_numpy(np.int64); yte = te["label"].to_numpy(np.int64)
        feats = {}
        k = val_k(ds)
        feats[("kmer_lgbm", f"kmer-k{k}-lgbm")] = (
            featurize.kmer_spectrum(tr["sequence"].astype(str).tolist(), k).tocsr(),
            featurize.kmer_spectrum(te["sequence"].astype(str).tolist(), k).tocsr(),
            dict(k=k, features=f"k-mer spectrum k={k} (val-selected)"))
        for fm in ("hyenadna", "dnabert2"):
            etr = load_emb(ds, fm, "train", ytr_all, man)
            ete = load_emb(ds, fm, "test", yte, man)
            if etr is None or ete is None:
                print(f"[{ds}] no {fm} embeddings (expected only for dnabert2 x drosophila)")
                continue
            for pool in POOLS[fm]:
                feats[(f"{fm}_frozen_{pool}", f"{EMB_TAG[fm]}-frozen-{pool}-lgbm")] = (
                    etr[pool], ete[pool], dict(features=f"frozen {EMB_TAG[fm]} {pool} embedding",
                                               emb_sha256_train=man.loc[
                                                   f"emb__{ds}__orig__train__{EMB_TAG[fm]}.npz", "sha256"]))
        for (model, tag), (Xtr_all, Xte, extra) in feats.items():
            for arm in ARMS:
                stem = os.path.join(OUT_DIR, f"{model}__{ds}__{arm}__s{SEED}")
                if os.path.exists(stem + ".json") and os.path.exists(stem + ".csv.gz"):
                    print(f"skip {os.path.basename(stem)} (done)")
                    continue
                o = arm_order(tr, arm)
                s = time.time()
                prob = fit_predict(Xtr_all[o], ytr_all[o], Xte)
                secs = time.time() - s
                write_unit(model, tag, ds, arm, yte, prob, len(o), int((ytr_all[o] == 0).sum()), secs,
                           extra)
                msk = {a: te[f"{a}_pos"].to_numpy() >= 0 for a in ARMS}
                q = {a: round(ev.auroc(yte[msk[a]], prob[msk[a]]), 4) for a in ARMS}
                print(f"{time.strftime('%H:%M:%S')} {model:22s} {ds:27s} train={arm:15s} "
                      f"n_fit={len(o):6d} {secs:6.1f}s  test AUROC by eval arm {q}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    repro()


def repro():
    """Reproduction check: the refits must reproduce the published numbers they correspond to."""
    runs = ev.load_runs(OUT_DIR)
    mem = ev.load_membership()
    P = {(r["meta"]["model"], r["meta"]["dataset"], r["meta"]["train_arm"]): r["p"] for r in runs}

    def auc(model, ds, ta, ea):
        y = mem[ds]["y"]; m = mem[ds]["mask"][ea]
        return ev.auroc(y[m], P[(model, ds, ta)][m])
    rows = []
    psm = pd.read_csv(os.path.join(ev.OUT, "psm_sensitivity_decision.csv")).set_index("dataset")
    for ds in DATASETS:
        rows.append(dict(check="kmer A (orig->orig) vs psm_sensitivity A0", dataset=ds,
                         ours=auc("kmer_lgbm", ds, "original", "original"),
                         published=psm.loc[ds, "A0"], source="psm_sensitivity_decision.csv:A0"))
        rows.append(dict(check="kmer D (comp_eq->comp_eq) vs psm_sensitivity paper_anchor_auroc",
                         dataset=ds, ours=auc("kmer_lgbm", ds, "comp_equalized", "comp_equalized"),
                         published=psm.loc[ds, "paper_anchor_auroc"],
                         source="psm_sensitivity_decision.csv:paper_anchor_auroc"))
    fp = pd.read_csv(os.path.join(ev.OUT, "fm_paired.csv"))
    fd = pd.read_csv(os.path.join(ev.OUT, "fm_paired_dnabert.csv"))
    for fm, tab in (("hyenadna", fp), ("dnabert2", fd)):
        for _, r in tab[(tab["head"] == "lgbm") & tab["arm"].isin(["gc_match", "comp_equalized"])].iterrows():
            model = f"{fm}_frozen_{r['pooling']}"
            if (model, r["dataset"], "original") not in P:
                continue
            for lab, ta, ea, col in (("A orig->orig", "original", "original", "orig_full_auroc"),
                                     ("orig->arm (fixed head)", "original", r["arm"], "orig_on_clean_auroc"),
                                     ("arm->arm (retrained)", r["arm"], r["arm"], "clean_on_clean_auroc")):
                if (model, r["dataset"], ta) not in P:
                    continue
                rows.append(dict(check=f"{model} {lab} [{r['arm']}] vs fm_paired", dataset=r["dataset"],
                                 ours=auc(model, r["dataset"], ta, ea), published=r[col],
                                 source=f"{'fm_paired.csv' if fm == 'hyenadna' else 'fm_paired_dnabert.csv'}:{col}"))
    df = pd.DataFrame(rows).drop_duplicates(["check", "dataset"])
    df["abs_diff"] = (df["ours"] - df["published"]).abs()
    df["match_1e-3"] = df["abs_diff"] < 1e-3
    df.round(6).to_csv(os.path.join(ev.OUT, "fm_finetune_v2_baselines_repro.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(df.round(4).to_string(index=False))
    print(f"reproduced {int(df['match_1e-3'].sum())}/{len(df)} published numbers to 1e-3")


if __name__ == "__main__":
    if "--repro-only" in sys.argv:
        repro()
    else:
        main()
