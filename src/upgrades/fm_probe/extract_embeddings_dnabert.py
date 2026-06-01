#!/usr/bin/env python3
"""Upgrade 1 / Depth 1 -- frozen DNABERT-2 feature extraction (SECOND FM architecture).

A second, independent pretrained genomic FM to show the composition-riding result (U1) is not a
HyenaDNA-specific artifact. ALL torch/transformers usage is isolated here behind the optional [fm]
extra; the diagnostic pipeline never imports it, and this runs as a SEPARATE process from any
LightGBM (libomp).

Model (confirmed from the live HF card, not assumed):
  zhihan1996/DNABERT-2-117M -- MosaicBERT-style bidirectional encoder, BPE tokenizer (vocab 4096),
  hidden_size 768, ~117M params, max_position 512 tokens. Revision PINNED.

Datasets: ONLY the short sets human_enhancers_cohn (500 bp -> ~129 BPE tokens) and nt_enhancers
(200 bp -> ~54 tokens). drosophila_enhancers_stark is DELIBERATELY EXCLUDED: its 3.2 kb sequences
exceed the 512-token context and would truncate ~97% of the clean control, reintroducing exactly the
confound the HyenaDNA-16k choice avoided. (This is a reasoned exclusion, documented in the README.)

Loading DNABERT-2 on CPU / transformers 5.x / torch 2.x / Python 3.14 requires a documented,
reproducible workaround (the MosaicBERT remote code predates this stack); each step is legitimate --
the resulting embeddings are REAL frozen DNABERT-2 forward passes, weights fully loaded:
  1. Bypass transformers' STATIC check_imports for the GPU-only deps (triton/flash_attn). The model's
     own code wraps that import in try/except (bert_layers.py) and falls back to standard attention
     on CPU when the flash/triton kernel is absent -- so no stub is needed, only the static check
     is too strict.
  2. Build the module on CPU via AutoModel.from_config (normal nn.Module init), NOT from_pretrained
     (whose meta-device lazy init is incompatible with this custom model on torch 2.x).
  3. Load the published checkpoint, stripping the 'bert.' prefix and dropping the MLM 'cls.' head;
     this fully populates the embeddings + encoder. The ONLY unmatched parameter is the BertPooler
     (pooler.dense), which we DO NOT use (we pool the last hidden state directly). A guard asserts
     that the embeddings/encoder loaded and the pooler is the only missing block, so a silent
     random-weight load can never pass unnoticed.
  4. config.pad_token_id is set explicitly (transformers 5.x removed the 4.x default).

Pooling (frozen, CPU forward only): mean over the real (attention-masked) tokens [default], and the
[CLS] first token [alternative] -- the BERT analogues of HyenaDNA's mean / [SEP]. Embeddings are a
deterministic function of the input (eval, no dropout) and cached to .npz so reruns are free.
"""
import os
import sys

import numpy as np

MODEL_REPO = "zhihan1996/DNABERT-2-117M"
MODEL_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"   # pinned (reproducibility)
MODEL_TAG = "dnabert2-117m"
HIDDEN_DIM = 768
MAX_TOKENS = 512
POOLINGS = ("mean", "cls")
SEED = 42
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _bypass_static_triton_check():
    """Make transformers' static import check tolerate the GPU-only deps; the model's own
    try/except provides the CPU fallback at runtime."""
    import transformers.dynamic_module_utils as dmu
    orig = dmu.check_imports

    def patched(fn):
        try:
            return orig(fn)
        except ImportError as e:
            if "triton" in str(e) or "flash_attn" in str(e):
                return []
            raise
    dmu.check_imports = patched


def load_model(repo=MODEL_REPO, revision=MODEL_REVISION, num_threads=None):
    """Load the frozen DNABERT-2 backbone + tokenizer on CPU (see module docstring for the recipe).
    Returns (model, tok). Asserts the trained weights actually loaded (no silent random init)."""
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    from huggingface_hub import hf_hub_download
    if num_threads:
        torch.set_num_threads(int(num_threads))
    torch.manual_seed(SEED)
    _bypass_static_triton_check()
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(repo, revision=revision, trust_remote_code=True)
    cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 3
    with torch.device("cpu"):
        model = AutoModel.from_config(cfg, trust_remote_code=True)
    sd = torch.load(hf_hub_download(repo, "pytorch_model.bin", revision=revision),
                    map_location="cpu", weights_only=True)
    sd = {(k[len("bert."):] if k.startswith("bert.") else k): v
          for k, v in sd.items() if not k.startswith("cls.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # GUARD against a silent random-weight load: the ONLY thing allowed to be missing is the
    # (unused) pooler; nothing in embeddings/encoder may be missing.
    bad = [m for m in missing if not m.startswith("pooler.")]
    assert not bad, f"DNABERT-2 weights did NOT load (random init risk): missing {bad[:5]}"
    assert any(k.startswith("encoder.") for k in sd) and any(k.startswith("embeddings.") for k in sd), \
        "checkpoint is missing the encoder/embeddings -- refusing to produce embeddings"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _last_hidden(out):
    return out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state


def embed_sequences(seqs, model, tok, batch_size=16, max_tokens=MAX_TOKENS, log_every=0):
    """Return dict(mean=(n,768), cls=(n,768)) float32. Mean-pool over attention-masked tokens;
    cls = first token. Sequences sorted by length for tight batches; CPU forward only."""
    import torch
    n = len(seqs)
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    mean = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
    cls = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
    done = 0
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        enc = tok([seqs[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_tokens)
        with torch.no_grad():
            h = _last_hidden(model(**enc))            # (b, L, 768)
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        mp = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        mp = mp.cpu().numpy().astype(np.float32)
        cl = h[:, 0].cpu().numpy().astype(np.float32)
        for j, i in enumerate(idx):
            mean[i] = mp[j]; cls[i] = cl[j]
        done += len(idx)
        if log_every and (done % log_every < batch_size or done == n):
            print(f"      embedded {done}/{n}", flush=True)
    return dict(mean=mean, cls=cls)


# ---- disk cache (own keys mean/cls; mirrors extract_embeddings.py's path scheme) ----
def cache_path(cache_dir, task, arm, split, model_tag=MODEL_TAG):
    return os.path.join(cache_dir, f"emb__{task}__{arm}__{split}__{model_tag}.npz")


def load_cached(cache_dir, task, arm, split, model_tag=MODEL_TAG):
    p = cache_path(cache_dir, task, arm, split, model_tag)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return dict(mean=d["mean"], cls=d["cls"], y=d["y"])


def save_cached(cache_dir, task, arm, split, mean, cls, y, model_tag=MODEL_TAG):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path(cache_dir, task, arm, split, model_tag),
                        mean=mean, cls=cls, y=np.asarray(y))


def cli():
    import argparse
    ap = argparse.ArgumentParser(description="Cache frozen DNABERT-2 embeddings (torch-only step).")
    ap.add_argument("--datasets", default="human_enhancers_cohn,nt_enhancers",
                    help="SHORT datasets only; drosophila is excluded (3.2kb > 512 tokens).")
    ap.add_argument("--cache", default=os.path.join(_ROOT, "cache", "fm_embeddings"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not available():
        print("torch/transformers not installed (the [fm] extra)."); sys.exit(2)
    model, tok = load_model(num_threads=(args.threads or None))
    if args.selftest:
        emb = embed_sequences(["ACGTACGTAC", "TTTTAAAACCCCGGGG", "ACGT"], model, tok, batch_size=2)
        print("selftest mean", emb["mean"].shape, "cls", emb["cls"].shape,
              "finite", bool(np.isfinite(emb["mean"]).all()))
        return
    sys.path.insert(0, _ROOT)
    import data as gbdata                 # loaders only -- NOT LightGBM
    os.makedirs(args.cache, exist_ok=True)
    print(f"caching {MODEL_TAG} (rev {MODEL_REVISION[:10]}) -> {args.cache}; "
          f"params={sum(p.numel() for p in model.parameters())} hidden={HIDDEN_DIM}")
    for task in [t for t in args.datasets.split(",") if t]:
        assert task != "drosophila_enhancers_stark", "drosophila is excluded for DNABERT-2 (512-token limit)."
        d = gbdata.load_dataset(task, seed=args.seed)
        for split, seqs in [("train", d["train_seqs"]), ("test", d["test_seqs"])]:
            if load_cached(args.cache, task, "orig", split) is not None:
                print(f"  {task} orig_{split}: cached ({len(seqs)}) -- skip"); continue
            print(f"  {task} orig_{split}: embedding {len(seqs)} sequences ...", flush=True)
            emb = embed_sequences(seqs, model, tok, batch_size=args.batch_size,
                                  log_every=max(2000, args.batch_size))
            y = d["y_train"] if split == "train" else d["y_test"]
            save_cached(args.cache, task, "orig", split, emb["mean"], emb["cls"], y=y)
            print(f"    saved {cache_path(args.cache, task, 'orig', split)}", flush=True)
    print("done.")


if __name__ == "__main__":
    cli()
