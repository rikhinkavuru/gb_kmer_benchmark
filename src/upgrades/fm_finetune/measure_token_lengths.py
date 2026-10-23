#!/usr/bin/env python3
"""DNABERT-2 token lengths of the fine-tuning splits (camera-ready item D9; post hoc, descriptive).

Question: the submitted paper said DNABERT-2 "cannot use the long drosophila control (sequences
exceed its 512-token context)". That was never measured. This script measures it.

What it does (tokenizer only; no model weights, no training, no network):
  * loads the DNABERT-2 tokenizer from the SAME pinned local snapshot the fine-tuning code uses
    (run_ft_v2.dnabert_snapshot(): external/fm_finetune/hf_cache/models--zhihan1996--DNABERT-2-117M/
    snapshots/7bce263b15377fc15361f52cfab88f8b586abda0), exactly as run_ft_v2.DnabertLoraWrap does:
    AutoTokenizer.from_pretrained(<snapshot>, trust_remote_code=True), offline;
  * tokenizes every sequence of the fine-tuning splits (external/fm_finetune/splits/
    <dataset>__{train,test}.csv.gz, i.e. the released train/test splits the models were trained and
    scored on) WITHOUT truncation and WITH the tokenizer's special tokens ([CLS] ... [SEP]), which is
    the length the model would see before run_ft_v2's truncation at max_length=512;
  * counts, per dataset and split (and pooled train+test), the sequence length in bp and in tokens
    (median, quartiles, min, max) and the number / fraction of sequences with more than 512 tokens.

Output: results/upgrades/fm_finetune_v2_token_lengths.csv (one row per dataset x split, split in
{train, test, all}). The drosophila rows are the ones the paper quotes; cohn and nt_enhancers are
listed for context (their sequences are 500 bp and 200 bp).

Run (low CPU, offline):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bibm/lowcpu.sh .venv-ft/bin/python \
      src/upgrades/fm_finetune/measure_token_lengths.py
"""
import gzip
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HF_CACHE = os.path.join(ROOT, "external", "fm_finetune", "hf_cache")
DNABERT_REV = "7bce263b15377fc15361f52cfab88f8b586abda0"   # = run_ft_v2.DNABERT_REV
SNAPSHOT = os.path.join(HF_CACHE, "models--zhihan1996--DNABERT-2-117M", "snapshots", DNABERT_REV)
SPLITS = os.path.join(ROOT, "external", "fm_finetune", "splits")
OUT = os.path.join(ROOT, "results", "upgrades", "fm_finetune_v2_token_lengths.csv")
DATASETS = ["drosophila_enhancers_stark", "human_enhancers_cohn", "nt_enhancers"]
CONTEXT = 512   # run_ft_v2.DnabertLoraWrap.encode: truncation=True, max_length=512


def main():
    from transformers import AutoTokenizer
    if not os.path.isfile(os.path.join(SNAPSHOT, "tokenizer.json")):
        sys.exit(f"pinned DNABERT-2 tokenizer not found at {SNAPSHOT}")
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True)
    probe = tok("ACGT")["input_ids"]
    assert probe[0] == tok.cls_token_id and probe[-1] == tok.sep_token_id, \
        "expected the [CLS] ... [SEP] template"

    rows = []
    for ds in DATASETS:
        per_split = {}
        for split in ("train", "test"):
            path = os.path.join(SPLITS, f"{ds}__{split}.csv.gz")
            with gzip.open(path, "rt") as fh:
                df = pd.read_csv(fh, usecols=["label", "sequence"])
            seqs = df["sequence"].astype(str).tolist()
            enc = tok(seqs, add_special_tokens=True, truncation=False)["input_ids"]
            ntok = np.array([len(x) for x in enc], dtype=np.int64)
            nbp = np.array([len(s) for s in seqs], dtype=np.int64)
            per_split[split] = (nbp, ntok)
        per_split["all"] = (np.concatenate([per_split["train"][0], per_split["test"][0]]),
                            np.concatenate([per_split["train"][1], per_split["test"][1]]))
        for split in ("train", "test", "all"):
            nbp, ntok = per_split[split]
            over = int((ntok > CONTEXT).sum())
            rows.append(dict(
                dataset=ds, split=split, n=len(ntok),
                len_bp_median=float(np.median(nbp)), len_bp_q1=float(np.percentile(nbp, 25)),
                len_bp_q3=float(np.percentile(nbp, 75)), len_bp_min=int(nbp.min()),
                len_bp_max=int(nbp.max()),
                tokens_median=float(np.median(ntok)), tokens_q1=float(np.percentile(ntok, 25)),
                tokens_q3=float(np.percentile(ntok, 75)), tokens_min=int(ntok.min()),
                tokens_max=int(ntok.max()),
                bp_per_nonspecial_token=round(float(nbp.sum() / (ntok - 2).sum()), 3),
                context_tokens=CONTEXT, n_over_context=over,
                frac_over_context=round(over / len(ntok), 5),
                tokens_include_special="yes ([CLS] and [SEP])",
                tokenizer=f"zhihan1996/DNABERT-2-117M@{DNABERT_REV} (local snapshot, offline)",
                source=f"external/fm_finetune/splits/{ds}__{{train,test}}.csv.gz"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out[["dataset", "split", "n", "len_bp_median", "tokens_median", "tokens_q1",
                   "tokens_q3", "tokens_min", "tokens_max", "n_over_context",
                   "frac_over_context"]].to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
