#!/usr/bin/env python3
"""End-to-end fine-tuning of genomic FMs on original vs composition-cleaned vs size-matched random
negatives (reviewer 1, comment 2). Torch-only; run in .venv-ft. Never import LightGBM here.

Question: once every weight (HyenaDNA, full fine-tune) or a LoRA adapter + head (DNABERT-2) is
trained end-to-end, does the leaderboard score still ride the negative-set composition shortcut?
And is the collapse on the composition-equalized arm caused by removing the shortcut, or merely
by the arm having ~25x fewer, heavily outnumbered negatives?

ARMS (materialized by prepare_splits_v2.py; positives identical in every arm)
  original        released split
  gc_match        25-bin GC-histogram-matched negatives            (HyenaDNA only)
  comp_equalized  GC + 16-dinucleotide propensity-matched negatives
  random          CONTROL: uniformly random negatives, EXACTLY the comp_equalized negative count
                  in train and in test (cohn 399/228, nt_enhancers 343/36), fixed draw (seed 42)
Every model predicts EVERY sequence of the ORIGINAL test split; each arm's test set is a subset of
it, so one prediction file scores a model on all arms (orig->X = same weights, cleaner test set).

MODELS / RECIPES (fixed before any new result was seen; identical across arms)
  hyenadna  LongSafari/hyenadna-tiny-16k-seqlen-d128-hf @ d79fa37e (the frozen-probe revision),
            FULL fine-tune (436k params), masked mean-pool + linear head; 6 epochs, batch 32, AdamW
            lr 6e-4 wd 0.1, 10% warmup + cosine, grad clip 1.0, class-weighted CE. = the recipe of
            run_fm_finetune.py (which produced results/upgrades/fm_finetune.csv) with ONE fix below.
  dnabert2  zhihan1996/DNABERT-2-117M @ 7bce263b (the frozen-embedding revision), LoRA (peft) on
            every attention Wqkv + attention-output dense + GLU MLP (gated_layers, wo) projection,
            r=8, alpha=16, lora_dropout=0.05; trainable linear head on the masked mean-pool (the
            frozen probe's pooling); backbone weights frozen. AdamW lr 3e-4 wd 0.01, 10% warmup +
            cosine, grad clip 1.0, batch 32, class-weighted CE, up to 4 epochs with early stopping
            (patience 2 epochs on validation AUROC). PyTorch attention (no triton on macOS).
            Per-encoder-layer activation checkpointing (memory only; gradient-exact incl. dropout
            RNG replay on MPS -- tests/check_ckpt_grads.py). Inference batch 32.
  Both: a stratified 10% validation carve of the arm's TRAIN split (seeded by the training seed,
  identical to run_fm_finetune.stratified_val_carve) selects the epoch; the test split is never
  used for selection. Seeds 42, 43, 44.

  FIX vs run_fm_finetune.py: its batcher sorts by length with a STABLE sort, and every training
  list is class-sorted (all negatives, then all positives), while cohn sequences are all 500 bp, so
  nearly every mini-batch was single-class. Here the list is randomly permuted (training-seeded)
  before the stable length sort, so equal-length ties are mixed-class. Applied to all arms and both
  models; the old numbers are therefore re-generated, not reused. (The old run also saved no
  per-sequence predictions, so its imbalance-robust metrics could not be recomputed anyway.)
  Units run seed-major (all arms of seed 42, then 43, then 44) so a partial run has whole seeds.

MEMORY (added 2026-09-24 after a unit grew to 17.6 GB on the 16 GB laptop; see memguard.py)
  Every run has a hard footprint cap (--mem-cap-mb, default 4500 or $FT_MEM_CAP_MB; exceeded ->
  exit 86). Units trained after the incident also use:
  --pad-multiple K  pad to a multiple of K tokens so only a few tensor shapes occur (the MPS
                    backend caches graphs per shape without bound); DNABERT-2 then keeps pad rows
                    instead of unpadding (set_dnabert_fixed_shapes). Real-token logits unchanged.
  --micro-batch N   exact gradient accumulation over micro-batches of N (optimizer batch stays 32).
  --eval-batch N    smaller inference batches (per-sequence outputs do not depend on it).
  All three are numerically no-ops (tests/check_pad_invariance.py: max |logit diff| <= 9e-8, max
  relative grad diff <= 1.4e-4); they are recorded per unit in the metadata JSON (pad_multiple,
  micro_batch, eval_batch_used, peak_footprint_mb). Units trained before 11:00 on 2026-09-24 ran
  with 0/0/recipe; their numbers are unaffected (only memory differed).

OUTPUTS (resumable: a unit whose prediction file + meta exist is skipped)
  results/upgrades/fm_finetune_v2_preds/<model>__<dataset>__<arm>__s<seed>.csv.gz
      split(val|test), idx (row index in the original split), label, prob
  results/upgrades/fm_finetune_v2_preds/<model>__<dataset>__<arm>__s<seed>.json   run metadata
  external/fm_finetune/checkpoints/<unit>.pt   best-epoch trainable weights (+ head)
The analysis (all metrics, CIs, contrasts, decision rules) is in eval_ft_v2.py (torch-free).

Usage (the full plan, in priority order, is run_queue_v2.sh):
  .venv-ft/bin/python src/upgrades/fm_finetune/run_ft_v2.py --model hyenadna \
      --dataset human_enhancers_cohn --arms original,comp_equalized,random --seeds 42,43,44
  .venv-ft/bin/python src/upgrades/fm_finetune/run_ft_v2.py --model dnabert2 --bench 20
  .venv-ft/bin/python src/upgrades/fm_finetune/run_ft_v2.py --model dnabert2 \
      --dataset nt_enhancers --arms random --seeds 42 --smoke 5 --pred-dir /tmp/ftv2_smoke
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
warnings.filterwarnings("ignore", message=".*Unable to import Triton.*")
warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_fm_finetune as ft          # noqa: E402  (HyenaClassifier, stratified_val_carve; torch lazily)
import memguard                       # noqa: E402  (hard footprint cap; torch-free)

GUARD = None                          # memguard.MemGuard, started in main()

SPLIT_DIR = os.path.join(ROOT, "external", "fm_finetune", "splits")
CKPT_DIR = os.path.join(ROOT, "external", "fm_finetune", "checkpoints")
HF_CACHE = os.path.join(ROOT, "external", "fm_finetune", "hf_cache")
PRED_DIR = os.path.join(ROOT, "results", "upgrades", "fm_finetune_v2_preds")
ARMS = ("original", "gc_match", "comp_equalized", "random")
SEEDS = (42, 43, 44)
VAL_FRAC = 0.1

DNABERT_REPO = "zhihan1996/DNABERT-2-117M"
DNABERT_REV = "7bce263b15377fc15361f52cfab88f8b586abda0"
LORA_TARGETS = ["Wqkv", "attention.output.dense", "gated_layers", "wo"]

RECIPES = {
    "hyenadna": dict(tag=ft.MODEL_TAG, epochs=6, batch=32, lr=ft.LR, weight_decay=ft.WEIGHT_DECAY,
                     warmup_frac=ft.WARMUP_FRAC, clip=1.0, patience=None, eval_batch=64,
                     finetune="full"),
    "dnabert2": dict(tag="dnabert2-117m-lora", epochs=4, batch=32, lr=3e-4, weight_decay=0.01,
                     warmup_frac=0.1, clip=1.0, patience=2, eval_batch=32, finetune="lora",
                     lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_targets=LORA_TARGETS,
                     grad_ckpt=True),
}
# grad_ckpt: per-encoder-layer activation checkpointing (non-reentrant, RNG-preserving). Pure
# memory/compute trade (identical gradients up to float error; see tests/check_ckpt_grads.py):
# without it a batch of 32 x ~116 tokens needs >3 GB of MPS memory on this 16 GB laptop.


# --------------------------------------------------------------------- data
def load_split(dataset):
    import pandas as pd
    tr = pd.read_csv(os.path.join(SPLIT_DIR, f"{dataset}__train.csv.gz"))
    te = pd.read_csv(os.path.join(SPLIT_DIR, f"{dataset}__test.csv.gz"))
    return tr, te


def arm_train(tr, arm):
    """(seqs, labels, original_idx) of one arm's TRAIN split, in the arm's list order."""
    sub = tr[tr[f"{arm}_pos"] >= 0].sort_values(f"{arm}_pos")
    return (sub["sequence"].astype(str).tolist(), sub["label"].to_numpy(np.int64),
            sub["idx"].to_numpy(np.int64))


def _chunk(order, batch):
    """Consecutive chunks of `batch`; a trailing size-1 chunk is merged into the previous one
    (DNABERT-2's attention squeezes the batch axis and breaks on a single sequence)."""
    chunks = [order[s:s + batch] for s in range(0, len(order), batch)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + last
    return chunks


def mixed_batches(seqs, idx, batch, rng):
    """Random permutation, then STABLE length sort (tight padding, mixed-class ties), then the
    batch order is shuffled. Returns lists of positions into `idx`."""
    perm = rng.permutation(len(idx))
    order = sorted(perm.tolist(), key=lambda i: len(seqs[idx[i]]))
    chunks = _chunk(order, batch)
    rng.shuffle(chunks)
    return chunks


def eval_batches(seqs, batch):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    return _chunk(order, batch)


# ------------------------------------------------------------------- models
def _bypass_static_triton_check():
    """transformers' static import check rejects the GPU-only triton import that DNABERT-2's own
    code already guards with try/except (falls back to PyTorch attention). Same recipe as
    fm_probe/extract_embeddings_dnabert.py."""
    import transformers.dynamic_module_utils as dmu
    if getattr(dmu.check_imports, "_gbkb_patched", False):
        return
    orig = dmu.check_imports

    def patched(fn):
        try:
            return orig(fn)
        except ImportError as e:
            if "triton" in str(e) or "flash_attn" in str(e):
                return []
            raise
    patched._gbkb_patched = True
    dmu.check_imports = patched


def dnabert_snapshot():
    path = os.path.join(HF_CACHE, "models--zhihan1996--DNABERT-2-117M", "snapshots", DNABERT_REV)
    if not os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        from huggingface_hub import snapshot_download
        path = snapshot_download(DNABERT_REPO, revision=DNABERT_REV, cache_dir=HF_CACHE,
                                 allow_patterns=["*.json", "*.py", "*.txt", "pytorch_model.bin"])
    assert path.rstrip("/").endswith(DNABERT_REV), path
    return path


PAD_MULTIPLE = 0   # set from --pad-multiple; 0 = pad each batch to its own longest (as before)
MICRO_BATCH = 0    # set from --micro-batch; 0 = whole batch in one forward/backward (as before)
EVAL_BATCH = 0     # set from --eval-batch; 0 = the recipe's eval_batch


class HyenaWrap(ft.HyenaClassifier):
    """run_fm_finetune.HyenaClassifier (all weights trainable) + the uniform interface.

    --pad-multiple K pads every batch to the next multiple of K tokens instead of to its own
    longest sequence. HyenaDNA is causal and right-padded and the pool is masked, so the logits of
    the real tokens do not depend on how many pad tokens follow (tests/check_pad_invariance.py).
    It bounds the number of distinct tensor shapes, and therefore the MPS graph cache: without it
    drosophila_enhancers_stark (535 distinct lengths) grew the process to 17.6 GB (memguard.py)."""

    def encode(self, seqs):
        if not PAD_MULTIPLE:
            return super().encode(seqs)
        return self.tok(list(seqs), return_tensors="pt", padding=True,
                        pad_to_multiple_of=PAD_MULTIPLE, return_special_tokens_mask=True)

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_total(self):
        return self.n_params()

    def best_state(self):
        return dict(backbone={k: v.detach().clone() for k, v in self.backbone.state_dict().items()},
                    head={k: v.detach().clone() for k, v in self.head.state_dict().items()})

    def load_best(self, st):
        self.backbone.load_state_dict(st["backbone"])
        self.head.load_state_dict(st["head"])


_DNABERT_UNPAD_ORIG = {}


def set_dnabert_fixed_shapes(base_model, on):
    """Fixed-shape execution for DNABERT-2 (memory fix, 2026-09-24; used when --pad-multiple > 0).

    DNABERT-2's encoder "unpads" every batch to a (total_nnz, 768) tensor, where total_nnz = number
    of real tokens in the batch, which differs for nearly every batch; each new shape makes the MPS
    backend compile and cache new graphs, so the process grew ~175 MB/min without bound
    (fm_finetune_v2_memwatch_76231.log). Here the encoder keeps every position instead (unpad =
    reshape, indices = all positions) and the tokenizer pads to a multiple of --pad-multiple, so
    only a few (batch, length) shapes ever occur. The real tokens' outputs are unchanged: pad KEYS
    are still removed from attention by the -10000 additive mask the model builds from the real
    attention_mask, every other op is per-token, and the mean-pool is masked; pad rows are computed
    and discarded. tests/check_pad_invariance.py verifies logits and LoRA gradients vs the unpadded
    path. Module-level patch of the remote-code module (affects every instance in the process)."""
    import sys
    import torch
    mod = sys.modules[type(base_model.encoder).__module__]
    if not _DNABERT_UNPAD_ORIG:
        _DNABERT_UNPAD_ORIG.update(unpad_input=mod.unpad_input,
                                   unpad_input_only=mod.unpad_input_only)
    if not on:
        mod.unpad_input = _DNABERT_UNPAD_ORIG["unpad_input"]
        mod.unpad_input_only = _DNABERT_UNPAD_ORIG["unpad_input_only"]
        return

    def unpad_keep_all(hidden_states, attention_mask):
        b, s = attention_mask.shape[:2]
        dev = hidden_states.device
        idx = torch.arange(b * s, device=dev)
        cu = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=dev)
        return hidden_states.reshape(b * s, *hidden_states.shape[2:]), idx, cu, s

    def unpad_only_keep_all(hidden_states, attention_mask):
        return hidden_states.reshape(-1, *hidden_states.shape[2:])

    mod.unpad_input = unpad_keep_all
    mod.unpad_input_only = unpad_only_keep_all


def enable_layer_checkpointing(peft_model):
    """Wrap every DNABERT-2 encoder layer's forward in torch.utils.checkpoint (non-reentrant).
    Only active when grad is enabled; inference is unchanged.

    Dropout masks must be identical in the forward and in the backward-time recomputation.
    torch.utils.checkpoint's preserve_rng_state does NOT stash the MPS generator (torch.mps has
    no `_initialized` flag, so the device-state stash is skipped; verified: gradients differed by
    ~100% on MPS). So on MPS the RNG is replayed explicitly: the MPS generator state at the start
    of the forward is restored for the recomputation, and the global stream is put back after it.
    tests/check_ckpt_grads.py asserts the result is gradient-exact with dropout active."""
    import torch
    import torch.utils.checkpoint as cp
    layers = peft_model.base_model.model.encoder.layer
    for layer in layers:
        fwd = layer.forward

        def wrapped(*a, _fwd=fwd, **k):
            if not torch.is_grad_enabled():
                return _fwd(*a, **k)
            if a[0].device.type != "mps":
                return cp.checkpoint(_fwd, *a, use_reentrant=False, preserve_rng_state=True, **k)
            start = torch.mps.get_rng_state()
            calls = [0]

            def run(*a2, **k2):
                calls[0] += 1
                if calls[0] == 1:                       # the real forward
                    return _fwd(*a2, **k2)
                cur = torch.mps.get_rng_state()         # backward-time recomputation
                torch.mps.set_rng_state(start)
                try:
                    return _fwd(*a2, **k2)
                finally:
                    torch.mps.set_rng_state(cur)
            return cp.checkpoint(run, *a, use_reentrant=False, preserve_rng_state=False, **k)
        layer.forward = wrapped
    return len(layers)


class DnabertLoraWrap:
    """DNABERT-2 backbone (frozen) + LoRA adapters + masked-mean-pool linear head."""

    def __init__(self, seed, device, rc):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        torch.manual_seed(seed)
        _bypass_static_triton_check()
        local = dnabert_snapshot()
        self.tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(local, trust_remote_code=True)
        cfg.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 3
        with torch.device("cpu"):
            base = AutoModel.from_config(cfg, trust_remote_code=True)
        sd = torch.load(os.path.join(local, "pytorch_model.bin"), map_location="cpu",
                        weights_only=True)
        sd = {(k[len("bert."):] if k.startswith("bert.") else k): v
              for k, v in sd.items() if not k.startswith("cls.")}
        missing, _ = base.load_state_dict(sd, strict=False)
        bad = [m for m in missing if not m.startswith("pooler.")]
        assert not bad, f"DNABERT-2 weights did NOT load (random-init risk): {bad[:5]}"
        del sd
        self.fixed_shapes = bool(PAD_MULTIPLE)
        set_dnabert_fixed_shapes(base, self.fixed_shapes)
        torch.manual_seed(seed)                   # LoRA-A init + head init depend only on seed
        lcfg = LoraConfig(r=rc["lora_r"], lora_alpha=rc["lora_alpha"],
                          lora_dropout=rc["lora_dropout"], target_modules=rc["lora_targets"],
                          bias="none")
        self.backbone = get_peft_model(base, lcfg)
        n_lora_mods = sum(1 for n, _ in self.backbone.named_modules() if n.endswith("lora_A"))
        assert n_lora_mods == 12 * 4, f"expected 48 LoRA-wrapped linears, got {n_lora_mods}"
        if rc.get("grad_ckpt"):
            enable_layer_checkpointing(self.backbone)
        self.head = torch.nn.Linear(768, 2)
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        self.device = device
        self.backbone.to(device)
        self.head.to(device)
        self.n_lora_modules = n_lora_mods

    def trainable(self):
        return ([p for p in self.backbone.parameters() if p.requires_grad]
                + list(self.head.parameters()))

    def n_total(self):
        return (sum(p.numel() for p in self.backbone.parameters())
                + sum(p.numel() for p in self.head.parameters()))

    def encode(self, seqs):
        return self.tok(list(seqs), return_tensors="pt", padding=True, truncation=True,
                        max_length=512, pad_to_multiple_of=PAD_MULTIPLE or None)

    def logits(self, enc):
        ids = enc["input_ids"].to(self.device)
        am = enc["attention_mask"].to(self.device)
        out = self.backbone(input_ids=ids, attention_mask=am)
        h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(pooled)

    def train_mode(self, flag):
        self.backbone.train(flag)
        self.head.train(flag)

    def best_state(self):
        return dict(lora={n: p.detach().clone() for n, p in self.backbone.named_parameters()
                          if p.requires_grad},
                    head={k: v.detach().clone() for k, v in self.head.state_dict().items()})

    def load_best(self, st):
        import torch
        params = dict(self.backbone.named_parameters())
        with torch.no_grad():
            for n, v in st["lora"].items():
                params[n].copy_(v)
        self.head.load_state_dict(st["head"])


def build_model(model, seed, device):
    if model == "hyenadna":
        return HyenaWrap(seed, device)
    return DnabertLoraWrap(seed, device, RECIPES["dnabert2"])


def accumulate_batch(m, seqs, y, rows, cw, opt, device):
    """One optimizer batch (rows) as micro-batches of MICRO_BATCH with gradient accumulation.
    Exact: CrossEntropyLoss(weight=cw, reduction='mean') = sum_i w_{y_i} l_i / sum_i w_{y_i} over
    the WHOLE batch, so every micro-batch contributes its weighted SUM divided by the full-batch
    weight total; the accumulated gradient equals the full-batch gradient (HyenaDNA and DNABERT-2
    have no cross-sequence ops, e.g. no batch norm). Only memory changes. Returns the batch loss
    as a 0-d tensor (for logging)."""
    import torch
    wsum = cw[torch.as_tensor(y[rows], dtype=torch.long, device=device)].sum()
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for s in range(0, len(rows), MICRO_BATCH):
        r = rows[s:s + MICRO_BATCH]
        yb = torch.tensor(y[r], dtype=torch.long, device=device)
        lo = torch.nn.functional.cross_entropy(m.logits(m.encode([seqs[i] for i in r])), yb,
                                               weight=cw, reduction="sum") / wsum
        lo.backward()
        total += float(lo.detach())
    return torch.tensor(total)


def predict(m, seqs, batch):
    import torch
    batch = EVAL_BATCH or batch
    m.train_mode(False)
    out = np.zeros(len(seqs), dtype=np.float64)
    with torch.no_grad():
        for chunk in eval_batches(seqs, batch):
            p = torch.softmax(m.logits(m.encode([seqs[i] for i in chunk])).float(), dim=-1)[:, 1]
            out[chunk] = p.cpu().numpy()
    return out


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


# ------------------------------------------------------------------ training
def unit_name(model, dataset, arm, seed):
    return f"{model}__{dataset}__{arm}__s{seed}"


def train_unit(model, dataset, arm, seed, device, tr, te, log, max_steps=None):
    import torch
    rc = RECIPES[model]
    seqs, y, oidx = arm_train(tr, arm)
    fit_pos, val_pos = ft.stratified_val_carve(y, VAL_FRAC, seed)
    t_start = time.time()
    m = build_model(model, seed, device)
    n_tr = sum(p.numel() for p in m.trainable())
    log(f"  [{model}/{dataset}/{arm}/s{seed}] n_fit={len(fit_pos)} n_val={len(val_pos)} "
        f"neg_fit={(y[fit_pos] == 0).sum()} trainable={n_tr:,} total={m.n_total():,}")
    y_fit = y[fit_pos]
    cnt = np.bincount(y_fit, minlength=2).astype(np.float64)
    cw = torch.tensor(len(y_fit) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(m.trainable(), lr=rc["lr"], weight_decay=rc["weight_decay"])
    steps_per_epoch = int(np.ceil(len(fit_pos) / rc["batch"]))
    total = steps_per_epoch * rc["epochs"]
    warm = max(1, int(rc["warmup_frac"] * total))

    def lr_at(step):
        if step < warm:
            return step / warm
        return 0.5 * (1.0 + np.cos(np.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.RandomState(seed)
    val_seqs = [seqs[i] for i in val_pos]
    hist, best, bad_epochs, step = [], dict(auc=-1.0, epoch=-1, state=None), 0, 0
    for ep in range(rc["epochs"]):
        m.train_mode(True)
        t0 = time.time(); tot = 0.0; seen = 0
        for bi, chunk in enumerate(mixed_batches(seqs, fit_pos, rc["batch"], rng)):
            rows = fit_pos[chunk]
            if MICRO_BATCH and len(rows) > MICRO_BATCH:
                loss = accumulate_batch(m, seqs, y, rows, cw, opt, device)
            else:                                           # the original path, unchanged
                enc = m.encode([seqs[i] for i in rows])
                yb = torch.tensor(y[rows], dtype=torch.long, device=device)
                loss = lossf(m.logits(enc), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(m.trainable(), rc["clip"])
            opt.step(); sched.step(); step += 1
            tot += float(loss.item()) * len(rows); seen += len(rows)
            if bi % 100 == 0:
                log(f"      step {bi}/{steps_per_epoch} loss={tot / max(seen, 1):.4f} "
                    f"({time.time() - t0:.0f}s)")
            if max_steps and step >= max_steps:
                break
            if str(device) == "mps" and bi % 200 == 199:
                torch.mps.empty_cache()
        pv = predict(m, val_seqs, rc["eval_batch"])
        if str(device) == "mps":
            torch.mps.empty_cache()
        va = _auc(y[val_pos], pv)
        hist.append(dict(epoch=ep + 1, loss=round(tot / max(seen, 1), 6), val_auroc=round(va, 6),
                         secs=round(time.time() - t0, 1)))
        log(f"    epoch {ep + 1}/{rc['epochs']} loss={tot / max(seen, 1):.4f} val_auroc={va:.4f} "
            f"({time.time() - t0:.0f}s)")
        if va == va and va > best["auc"]:
            best = dict(auc=va, epoch=ep + 1, state=m.best_state()); bad_epochs = 0
        else:
            bad_epochs += 1
        if max_steps and step >= max_steps:
            break
        if rc["patience"] and bad_epochs >= rc["patience"]:
            log(f"    early stop: no val improvement for {bad_epochs} epochs")
            break
    if best["state"] is not None:
        m.load_best(best["state"])
    meta = dict(model=model, model_tag=rc["tag"], dataset=dataset, train_arm=arm, seed=seed,
                n_fit=int(len(fit_pos)), n_val=int(len(val_pos)),
                n_fit_neg=int((y[fit_pos] == 0).sum()), n_fit_pos=int((y[fit_pos] == 1).sum()),
                best_epoch=best["epoch"], best_val_auroc=round(best["auc"], 6),
                epochs_run=len(hist), history=hist, n_params_total=int(m.n_total()),
                n_params_trainable=int(n_tr), device=str(device), recipe=rc,
                train_secs=round(time.time() - t_start, 1), torch=torch.__version__)
    return m, meta, (val_seqs, y[val_pos], oidx[val_pos]), best


def run_unit(model, dataset, arm, seed, device, tr, te, log, pred_dir=PRED_DIR, ckpt_dir=CKPT_DIR,
             max_steps=None):
    import resource
    import pandas as pd
    import torch
    name = unit_name(model, dataset, arm, seed)
    pred_csv = os.path.join(pred_dir, name + ".csv.gz")
    meta_js = os.path.join(pred_dir, name + ".json")
    if os.path.exists(pred_csv) and os.path.exists(meta_js):
        log(f"  skip {name} (done)")
        return
    m, meta, (vs, vy, vidx), best = train_unit(model, dataset, arm, seed, device, tr, te, log,
                                               max_steps=max_steps)
    rc = RECIPES[model]
    t0 = time.time()
    te_seqs = te["sequence"].astype(str).tolist()
    pt = predict(m, te_seqs, rc["eval_batch"])
    pv = predict(m, vs, rc["eval_batch"])
    meta["predict_secs"] = round(time.time() - t0, 1)
    yt = te["label"].to_numpy(np.int64)
    quick = {}
    for ea in ARMS:
        mask = te[f"{ea}_pos"].to_numpy() >= 0
        quick[ea] = round(_auc(yt[mask], pt[mask]), 4)
    meta["quick_test_auroc"] = quick
    meta["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    if str(device) == "mps":
        meta["mps_driver_mb_end"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    meta["env"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "LOWCPU_THREADS",
                                                  "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                                                  "PYTORCH_MPS_LOW_WATERMARK_RATIO")}
    meta["niceness"] = os.getpriority(os.PRIO_PROCESS, 0)
    meta["max_steps"] = max_steps
    meta["pad_multiple"] = PAD_MULTIPLE          # 0 = per-batch padding (+ unpadding for DNABERT-2)
    meta["micro_batch"] = MICRO_BATCH            # 0 = none (exact accumulation otherwise)
    meta["eval_batch_used"] = EVAL_BATCH or rc["eval_batch"]
    if GUARD is not None:
        meta["peak_footprint_mb"] = round(GUARD.peak_mb, 1)
        meta["mem_cap_mb"] = GUARD.cap_mb
    log(f"  -> {name} best_epoch={meta['best_epoch']} test AUROC by eval arm: {quick}")
    os.makedirs(pred_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    df = pd.concat([pd.DataFrame(dict(split="val", idx=vidx, label=vy, prob=pv)),
                    pd.DataFrame(dict(split="test", idx=te["idx"].to_numpy(), label=yt, prob=pt))])
    df.to_csv(pred_csv + ".tmp", index=False, compression="gzip", float_format="%.7g")
    torch.save(best["state"], os.path.join(ckpt_dir, name + ".pt"))
    with open(meta_js + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(pred_csv + ".tmp", pred_csv)
    os.replace(meta_js + ".tmp", meta_js)
    del m, best
    gc.collect()
    if str(device) == "mps":
        torch.mps.empty_cache()


def bench(model, dataset, device, steps, log):
    """Time `steps` optimizer steps + one full-test prediction pass; extrapolate the plan."""
    import torch
    tr, te = load_split(dataset)
    rc = RECIPES[model]
    seqs, y, _ = arm_train(tr, "original")
    fit_pos, _ = ft.stratified_val_carve(y, VAL_FRAC, 42)
    m = build_model(model, 42, device)
    n_tr = sum(p.numel() for p in m.trainable())
    log(f"[bench] {model} trainable={n_tr:,} total={m.n_total():,}")
    opt = torch.optim.AdamW(m.trainable(), lr=rc["lr"])
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.RandomState(0)
    m.train_mode(True)
    times = []
    for i, chunk in enumerate(mixed_batches(seqs, fit_pos, rc["batch"], rng)[:steps]):
        t0 = time.time()
        rows = fit_pos[chunk]
        enc = m.encode([seqs[j] for j in rows])
        if MICRO_BATCH and len(rows) > MICRO_BATCH:
            loss = accumulate_batch(m, seqs, y, rows, torch.ones(2, device=device), opt, device)
        else:
            loss = lossf(m.logits(enc), torch.tensor(y[rows], dtype=torch.long, device=device))
            opt.zero_grad(); loss.backward()
        opt.step()
        if str(device) == "mps":
            torch.mps.synchronize()
        times.append(time.time() - t0)
        log(f"  step {i} {times[-1]:.2f}s loss={float(loss):.4f} "
            f"tokens={int(enc['input_ids'].shape[1])} footprint={memguard.footprint_mb():.0f}MB")
    t0 = time.time()
    predict(m, te["sequence"].astype(str).tolist()[:512], rc["eval_batch"])
    t_inf = (time.time() - t0) / 512
    sps = float(np.median(times[2:] if len(times) > 4 else times))
    ep = int(np.ceil(len(fit_pos) / rc["batch"])) * sps
    log(f"[bench] median {sps:.3f}s/step -> {ep / 60:.1f} min/epoch on original ({len(fit_pos)} "
        f"fit seqs); inference {t_inf * 1000:.1f} ms/seq")
    if str(device) == "mps":
        log(f"[bench] mps allocated {torch.mps.current_allocated_memory() / 1e6:.0f} MB, "
            f"driver {torch.mps.driver_allocated_memory() / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(RECIPES), required=True)
    ap.add_argument("--dataset", default="human_enhancers_cohn")
    ap.add_argument("--arms", default="original,comp_equalized,random")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bench", type=int, default=0, help="time N steps and exit")
    ap.add_argument("--smoke", type=int, default=0,
                    help="wiring check: stop training after N steps; writes to --pred-dir")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--pad-multiple", type=int, default=0,
                    help="fixed-shape batches: pad to a multiple of this many tokens (and, for "
                         "dnabert2, keep pad rows instead of unpadding); 0 = old behaviour")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="split each training batch into micro-batches of this size with exact "
                         "gradient accumulation (memory only; effective batch unchanged)")
    ap.add_argument("--eval-batch", type=int, default=0, help="override the recipe's eval batch")
    ap.add_argument("--mem-cap-mb", type=float,
                    default=float(os.environ.get("FT_MEM_CAP_MB", "4500")),
                    help="hard cap on this process's phys_footprint; exceeded -> exit 86")
    a = ap.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    global GUARD, PAD_MULTIPLE, MICRO_BATCH, EVAL_BATCH
    GUARD = memguard.MemGuard(a.mem_cap_mb, log).start()     # before torch is even imported
    PAD_MULTIPLE = a.pad_multiple
    MICRO_BATCH = a.micro_batch
    EVAL_BATCH = a.eval_batch
    import torch
    torch.set_num_threads(a.threads)
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"[ft_v2] memguard cap {a.mem_cap_mb:.0f} MB (footprint now {GUARD.sample():.0f} MB); "
        f"pad_multiple={PAD_MULTIPLE} micro_batch={MICRO_BATCH} eval_batch={EVAL_BATCH or 'recipe'}")

    if a.bench:
        bench(a.model, a.dataset, dev, a.bench, log)
        return
    tr, te = load_split(a.dataset)
    log(f"[ft_v2] model={a.model} dataset={a.dataset} arms={a.arms} seeds={a.seeds} device={dev}")
    arms = a.arms.split(",")
    assert all(arm in ARMS for arm in arms), arms
    if a.smoke:
        assert a.pred_dir, "--smoke needs --pred-dir (never writes smoke output to results/)"
    pdir = a.pred_dir or PRED_DIR
    cdir = os.path.join(pdir, "ckpt") if a.smoke else CKPT_DIR
    for seed in [int(s) for s in a.seeds.split(",")]:          # seed-major: complete seeds first
        for arm in arms:
            run_unit(a.model, a.dataset, arm, seed, dev, tr, te, log, pred_dir=pdir,
                     ckpt_dir=cdir, max_steps=a.smoke or None)
    log("[ft_v2] done")


if __name__ == "__main__":
    main()
