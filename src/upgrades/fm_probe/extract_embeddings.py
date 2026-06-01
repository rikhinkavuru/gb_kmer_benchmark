#!/usr/bin/env python3
"""Upgrade 1 -- frozen genomic-FM feature extraction (the ONLY torch in the repo).

ALL torch/transformers usage is isolated in this file behind the optional ``[fm]`` extra.
The diagnostic pipeline never imports it, so the honesty claim is precise: "no PyTorch in
the diagnostic pipeline; PyTorch is used only for frozen FM feature extraction."

Model (confirmed from the live HF card, not assumed):
  LongSafari/hyenadna-tiny-16k-seqlen-d128-hf  --  HyenaDNA, a SMALL pretrained genomic
  foundation model. d_model=128, n_layer=2, 436,096 backbone params, single-nucleotide
  character tokenizer, max context 16,386 nt. The 16k-context tiny variant is used (not the
  1k one) because drosophila_enhancers_stark sequences run to 3,237 nt (median 2,142) -- the
  1k model would truncate ~97% of the clean-control sequences and confound the comparison;
  cohn (500 nt) and nt_enhancers (200 nt) fit either. The model revision is PINNED for
  reproducibility (trust_remote_code loads the repo's modeling code at that commit).

Pooling (forward pass only, frozen weights, CPU):
  * mean  -- mean of the last hidden state over the NUCLEOTIDE tokens (special tokens [SEP]
             and [PAD] excluded). The default.
  * sep   -- the hidden state at the final [SEP] token, which causally summarizes the whole
             sequence (HyenaDNA is a causal conv model, so this is the CLS-equivalent; there
             is no prepended CLS/BOS to use as a first-token vector).

HyenaDNA is causal, so RIGHT padding is mandatory and safe (verified: a real token's hidden
state is bit-identical whether the sequence is batched with right padding or run alone, max
|delta| ~3e-6; LEFT padding -- the tokenizer default -- corrupts every real token via the
causal convolution, so we force padding_side='right').

Embeddings are a deterministic function of the input sequence (eval mode, no dropout) and are
cached to .npz so reruns are free. CPU-only; no fine-tuning; no GPU.
"""
import os
import sys

import numpy as np

MODEL_REPO = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
MODEL_REVISION = "d79fa37e2cd62dd338103c630f95be8f90812d46"  # pinned (reproducibility)
MODEL_TAG = "hyenadna-tiny-16k-d128"
HIDDEN_DIM = 128
POOLINGS = ("mean", "sep")
SEED = 42


def available():
    """True iff torch + transformers import (the optional [fm] extra is installed)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def load_model(repo=MODEL_REPO, revision=MODEL_REVISION, num_threads=None):
    """Load the frozen backbone + tokenizer (right-padding forced). Returns (model, tok)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if num_threads:
        torch.set_num_threads(int(num_threads))
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
    tok.padding_side = "right"                      # MANDATORY for the causal model
    model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
    model.eval()
    for p in model.parameters():                    # frozen: no grad, no training
        p.requires_grad_(False)
    return model, tok


def _pool_batch(last_hidden, special_mask, pad_mask):
    """last_hidden (B,L,H) torch -> (mean (B,H), sep (B,H)) numpy float32.
    special_mask (B,L) True at [SEP]/[PAD]; pad_mask (B,L) True at real+SEP (non-pad)."""
    import torch
    nuc = (~special_mask) & pad_mask                # nucleotide positions only
    w = nuc.unsqueeze(-1).to(last_hidden.dtype)
    summ = (last_hidden * w).sum(1)
    cnt = w.sum(1).clamp_min(1.0)
    mean = (summ / cnt).cpu().numpy().astype(np.float32)
    # sep = hidden at the last non-pad position (the [SEP] token)
    last_idx = pad_mask.to(torch.int64).sum(1) - 1  # index of final non-pad token
    sep = last_hidden[torch.arange(last_hidden.shape[0]), last_idx].cpu().numpy().astype(np.float32)
    return mean, sep


def embed_sequences(seqs, model, tok, batch_size=32, max_len=None, log_every=0):
    """Return dict(mean=(n,H), sep=(n,H)) float32 embeddings for ``seqs``.

    Sequences are sorted by length so each batch pads minimally (then results are unsorted
    back to the input order). Deterministic; CPU forward passes only.
    """
    import torch
    n = len(seqs)
    order = sorted(range(n), key=lambda i: len(seqs[i]))     # short->long for tight batches
    mean = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
    sep = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
    pad_id = tok.pad_token_id
    done = 0
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        chunk = [seqs[i] for i in idx]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=(max_len is not None),
                  max_length=max_len, return_special_tokens_mask=True)
        ids = enc["input_ids"]
        special = enc["special_tokens_mask"].bool()
        padm = ids != pad_id
        with torch.no_grad():
            out = model(input_ids=ids)
        m, sp = _pool_batch(out.last_hidden_state, special, padm)
        for j, i in enumerate(idx):
            mean[i] = m[j]; sep[i] = sp[j]
        done += len(idx)
        if log_every and (done % log_every < batch_size or done == n):
            print(f"      embedded {done}/{n}", flush=True)
    return dict(mean=mean, sep=sep)


# --------------------------------------------------------- disk cache
def cache_path(cache_dir, task, arm, split, model_tag=MODEL_TAG):
    return os.path.join(cache_dir, f"emb__{task}__{arm}__{split}__{model_tag}.npz")


def load_cached(cache_dir, task, arm, split, model_tag=MODEL_TAG):
    p = cache_path(cache_dir, task, arm, split, model_tag)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return dict(mean=d["mean"], sep=d["sep"], y=d["y"])


def save_cached(cache_dir, task, arm, split, mean, sep, y, model_tag=MODEL_TAG):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path(cache_dir, task, arm, split, model_tag),
                        mean=mean, sep=sep, y=np.asarray(y))


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def cli():
    """Standalone embedding+caching step (torch ONLY -- never imports LightGBM).

    LightGBM and PyTorch each load their own OpenMP runtime; on macOS having both in one
    process segfaults. So embedding (torch) is a SEPARATE process from head-training
    (run_fm_probe.py, LightGBM). This CLI embeds the ORIGINAL train/test split of each
    dataset once and caches it; every cleaning arm is a subset of the original split, so
    run_fm_probe assembles all arms from this cache by lookup, with no torch.
    """
    import argparse
    ap = argparse.ArgumentParser(description="Cache frozen HyenaDNA embeddings (torch-only step).")
    ap.add_argument("--datasets", default="human_enhancers_cohn,nt_enhancers,drosophila_enhancers_stark")
    ap.add_argument("--cache", default=os.path.join(_ROOT, "cache", "fm_embeddings"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if not available():
        print("torch/transformers not installed (the [fm] extra). Install with:\n"
              "  uv pip install --python <venv> torch transformers einops")
        sys.exit(2)
    model, tok = load_model(num_threads=(args.threads or None))
    if args.selftest:
        emb = embed_sequences(["ACGTACGT", "TTTTAAAACCCCGGGG", "ACGT"], model, tok, batch_size=2)
        ok = bool(np.isfinite(emb["mean"]).all() and np.isfinite(emb["sep"]).all())
        print("selftest mean", emb["mean"].shape, "sep", emb["sep"].shape, "finite:", ok)
        return

    sys.path.insert(0, _ROOT)
    import data as gbdata                      # loaders only -- does NOT import LightGBM
    os.makedirs(args.cache, exist_ok=True)
    print(f"caching {MODEL_TAG} (rev {MODEL_REVISION[:10]}) embeddings -> {args.cache}")
    print(f"backbone params: {sum(p.numel() for p in model.parameters())}  hidden={HIDDEN_DIM}")
    for task in [t for t in args.datasets.split(",") if t]:
        d = gbdata.load_dataset(task, seed=args.seed)
        for split, seqs in [("train", d["train_seqs"]), ("test", d["test_seqs"])]:
            if load_cached(args.cache, task, "orig", split) is not None:
                print(f"  {task} orig_{split}: cached ({len(seqs)} seqs) -- skip")
                continue
            print(f"  {task} orig_{split}: embedding {len(seqs)} sequences ...", flush=True)
            emb = embed_sequences(seqs, model, tok, batch_size=args.batch_size,
                                  log_every=max(2000, args.batch_size))
            y = d["y_train"] if split == "train" else d["y_test"]
            save_cached(args.cache, task, "orig", split, emb["mean"], emb["sep"], y=y)
            print(f"    saved {cache_path(args.cache, task, 'orig', split)}", flush=True)
    print("done.")


if __name__ == "__main__":
    cli()
