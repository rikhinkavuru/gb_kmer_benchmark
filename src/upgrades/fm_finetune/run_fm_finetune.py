#!/usr/bin/env python3
"""Fine-tuned genomic-FM arm: how much of a fine-tuned FM's enhancer score is composition?

The frozen probe (src/upgrades/fm_probe/) is an *existence proof* -- it shows a pretrained
representation carries the composition shortcut. It cannot say how much a *fine-tuned*
leaderboard number is inflated, because a fine-tuned model may or may not keep using the
shortcut once the head and the backbone are both free to move. This module answers that
directly: the SAME model, the SAME recipe, trained and evaluated on the original benchmark
split versus the composition-cleaned splits.

  arm `original`       train on the released split, test on the released test split
                       -> this is the number a leaderboard reports
  arm `gc_match`       negatives subsampled to the positives' GC histogram
  arm `comp_equalized` negatives resampled to match the positives' joint GC+16-dinucleotide
                       propensity (the strongest cleaning)
  arm `original->X`    the model trained on the ORIGINAL split, evaluated on arm X's test
                       split (same fixed weights, cleaner test set)

Model: LongSafari/hyenadna-tiny-16k-seqlen-d128-hf at the SAME pinned revision the frozen
probe uses, so the frozen and fine-tuned arms differ only in whether gradients flow. The
backbone is mean-pooled over nucleotide tokens (identical pooling to the frozen `mean` arm)
and a linear head is trained on top; everything is trainable.

Protocol (mirrors the rest of the pipeline):
  * a stratified 10% validation carve of the TRAIN split selects the epoch; the test split is
    never used for model selection;
  * class-weighted cross-entropy (the cleaned arms are class-imbalanced by construction);
  * three seeds (42, 43, 44) per arm, reported as mean +/- SD;
  * 1000-resample percentile bootstrap CIs with the individual test sequence as the
    resampling unit, exactly as in run_stats.py.

Torch is confined to this file (like extract_embeddings.py); the diagnostic pipeline still
imports zero torch. Runs on Apple MPS or CPU; ~436k backbone parameters.

Usage:
    python src/upgrades/fm_finetune/run_fm_finetune.py --dataset human_enhancers_cohn
    python src/upgrades/fm_finetune/run_fm_finetune.py --smoke     # 2-min wiring check
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
warnings.filterwarnings("ignore", message=".*single label was found.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_REPO = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
MODEL_REVISION = "d79fa37e2cd62dd338103c630f95be8f90812d46"   # pinned (same as the frozen probe)
MODEL_TAG = "hyenadna-tiny-16k-d128"
HIDDEN_DIM = 128

SEEDS = (42, 43, 44)
EPOCHS = 3
BATCH = 32
LR = 6e-4                 # HyenaDNA's downstream fine-tuning LR for the tiny models
WEIGHT_DECAY = 0.1
WARMUP_FRAC = 0.1
VAL_FRAC = 0.1
BOOT = 1000

RESULTS_DIR = os.path.join(ROOT, "results", "upgrades")
SPLITS_V2 = os.path.join(ROOT, "cleaned_splits_v2")
ARMS = ("original", "gc_match", "comp_equalized")
ARM_FILE = {"gc_match": "gcmatch", "comp_equalized": "comp_equalized"}


# --------------------------------------------------------------------- data
def load_arm(dataset, arm):
    """(train_seqs, y_train, test_seqs, y_test) for one arm of one dataset."""
    import pandas as pd
    if arm == "original":
        import data as gbdata
        d = gbdata.load_dataset(dataset, seed=42)
        return (d["train_seqs"], np.asarray(d["y_train"], np.int64),
                d["test_seqs"], np.asarray(d["y_test"], np.int64))
    stem = ARM_FILE[arm]
    out = []
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(SPLITS_V2, f"{dataset}_{stem}_{split}.csv"))
        out.append(df["sequence"].astype(str).str.upper().tolist())
        out.append(df["label"].to_numpy(np.int64))
    return out[0], out[1], out[2], out[3]


def _subsample(seqs, y, n, seed=0):
    """Class-stratified subsample of n sequences (smoke mode only)."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        keep.extend(idx[:max(1, n // len(np.unique(y)))].tolist())
    keep = np.sort(np.asarray(keep))
    return [seqs[i] for i in keep], y[keep]


def stratified_val_carve(y, frac, seed):
    """Deterministic per-class hold-out index pair (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    val = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(frac * len(idx))))
        val[idx[:n]] = True
    return np.where(~val)[0], np.where(val)[0]


# -------------------------------------------------------------------- model
class HyenaClassifier:
    """Frozen-probe backbone + mean-pool + linear head, all trainable."""

    def __init__(self, seed, device):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION,
                                                 trust_remote_code=True)
        self.tok.padding_side = "right"          # MANDATORY: HyenaDNA is causal
        self.backbone = AutoModel.from_pretrained(MODEL_REPO, revision=MODEL_REVISION,
                                                  trust_remote_code=True)
        self.head = torch.nn.Linear(HIDDEN_DIM, 2)
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        self.device = device
        self.backbone.to(device)
        self.head.to(device)

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, seqs):
        enc = self.tok(list(seqs), return_tensors="pt", padding=True,
                       return_special_tokens_mask=True)
        return enc

    def logits(self, enc):
        import torch
        ids = enc["input_ids"].to(self.device)
        special = enc["special_tokens_mask"].bool().to(self.device)
        padm = (ids != self.tok.pad_token_id)
        hidden = self.backbone(input_ids=ids).last_hidden_state
        nuc = (~special) & padm                     # nucleotide positions only
        w = nuc.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * w).sum(1) / w.sum(1).clamp_min(1.0)
        return self.head(pooled)

    def train_mode(self, flag):
        self.backbone.train(flag)
        self.head.train(flag)


def _length_sorted_batches(seqs, n, batch, shuffle_rng=None):
    """Batch indices: sorted by length so padding is tight; batch ORDER shuffled for SGD."""
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    chunks = [order[s:s + batch] for s in range(0, n, batch)]
    if shuffle_rng is not None:
        shuffle_rng.shuffle(chunks)
    return chunks


def predict_proba(model, seqs, batch=64):
    """Positive-class probability for every sequence, in input order."""
    import torch
    model.train_mode(False)
    n = len(seqs)
    out = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for idx in _length_sorted_batches(seqs, n, batch):
            enc = model.encode([seqs[i] for i in idx])
            p = torch.softmax(model.logits(enc), dim=-1)[:, 1].float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = p[j]
    return out


def train_one(dataset, arm, seed, device, epochs=EPOCHS, batch=BATCH, log=print,
              max_train=None):
    """Fine-tune one (dataset, arm, seed). Returns (model, meta, splits)."""
    import torch
    tr_seqs, ytr, te_seqs, yte = load_arm(dataset, arm)
    if max_train:                                   # smoke mode only: stratified subsample
        tr_seqs, ytr = _subsample(tr_seqs, ytr, max_train)
        te_seqs, yte = _subsample(te_seqs, yte, max_train)
    tr_idx, va_idx = stratified_val_carve(ytr, VAL_FRAC, seed)
    fit_seqs = [tr_seqs[i] for i in tr_idx]; y_fit = ytr[tr_idx]
    val_seqs = [tr_seqs[i] for i in va_idx]; y_val = ytr[va_idx]

    model = HyenaClassifier(seed, device)
    cnt = np.bincount(y_fit, minlength=2).astype(np.float64)
    cw = torch.tensor(len(y_fit) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32,
                      device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = len(fit_seqs)
    steps_per_epoch = int(np.ceil(n / batch))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(WARMUP_FRAC * total_steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.RandomState(seed)
    best = dict(val_auroc=-1.0, epoch=-1, state=None)
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        model.train_mode(True)
        t0 = time.time(); tot = 0.0
        for chunk in _length_sorted_batches(fit_seqs, n, batch, rng):
            enc = model.encode([fit_seqs[i] for i in chunk])
            yb = torch.tensor(y_fit[chunk], dtype=torch.long, device=device)
            loss = lossf(model.logits(enc), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.item()) * len(chunk)
        pv = predict_proba(model, val_seqs)
        va = float(roc_auc_score(y_val, pv)) if len(np.unique(y_val)) > 1 else float("nan")
        log(f"      [{dataset}/{arm}/seed{seed}] epoch {ep + 1}/{epochs} "
            f"loss={tot / n:.4f} val_auroc={va:.4f} ({time.time() - t0:.0f}s)")
        if va == va and va > best["val_auroc"]:
            best = dict(val_auroc=va, epoch=ep + 1,
                        state={k: v.detach().clone() for k, v in
                               list(model.backbone.state_dict().items())},
                        head={k: v.detach().clone() for k, v in
                              list(model.head.state_dict().items())})
    if best["state"] is not None:
        model.backbone.load_state_dict(best["state"])
        model.head.load_state_dict(best["head"])
    meta = dict(n_train=len(fit_seqs), n_val=len(val_seqs),
                best_epoch=best["epoch"], val_auroc=round(best["val_auroc"], 6),
                n_params=model.n_params())
    return model, meta, (te_seqs, yte)


# ------------------------------------------------------------------ metrics
def eval_boot(y, proba, seed=42, B=BOOT):
    """Point AUROC/MCC/acc + 1000-resample percentile bootstrap CIs (test sequence = unit)."""
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    y = np.asarray(y, np.int64)
    pred = (proba >= 0.5).astype(np.int64)
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    mcc = float(matthews_corrcoef(y, pred))
    acc = float((pred == y).mean())
    rng = np.random.RandomState(seed)
    n = len(y)
    aa = np.full(B, np.nan); mm = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aa[b] = roc_auc_score(y[idx], proba[idx])
        mm[b] = matthews_corrcoef(y[idx], pred[idx])
    q = lambda a, p: round(float(np.nanpercentile(a, p)), 6)
    return dict(auroc=round(auc, 6), auroc_lo=q(aa, 2.5), auroc_hi=q(aa, 97.5),
                mcc=round(mcc, 6), mcc_lo=q(mm, 2.5), mcc_hi=q(mm, 97.5),
                acc=round(acc, 6), n_test=int(n),
                pos_frac=round(float((y == 1).mean()), 6))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="human_enhancers_cohn")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    epochs = 1 if a.smoke else a.epochs
    max_train = 256 if a.smoke else None
    out_csv = a.out or os.path.join(RESULTS_DIR, "fm_finetune.csv")
    if a.smoke:
        out_csv = os.path.join(RESULTS_DIR, "fm_finetune_smoke.csv")
        arms, seeds = arms[:2], seeds[:1]

    print(f"[fm_finetune] model={MODEL_TAG} device={dev} dataset={a.dataset} "
          f"arms={arms} seeds={seeds} epochs={epochs}", flush=True)
    rows = []
    # test splits of every arm, so a model trained on one arm can be scored on another
    test_of = {arm: load_arm(a.dataset, arm)[2:] for arm in arms}
    if max_train:
        test_of = {k: _subsample(v[0], v[1], max_train) for k, v in test_of.items()}

    for arm in arms:
        for seed in seeds:
            t0 = time.time()
            model, meta, (te_seqs, yte) = train_one(a.dataset, arm, seed, dev,
                                                    epochs=epochs, batch=a.batch,
                                                    max_train=max_train)
            # in-arm evaluation (the headline number for that arm)
            for eval_arm in arms:
                if eval_arm != arm and arm != "original":
                    continue          # cross-eval only from the original-trained model
                ev_seqs, ev_y = test_of[eval_arm]
                m = eval_boot(ev_y, predict_proba(model, ev_seqs), seed=42)
                rows.append(dict(model=MODEL_TAG, dataset=a.dataset, train_arm=arm,
                                 eval_arm=eval_arm, seed=seed, **meta, **m,
                                 train_secs=round(time.time() - t0, 1)))
                print(f"   -> train={arm} eval={eval_arm} seed={seed} "
                      f"AUROC={m['auroc']:.4f} [{m['auroc_lo']:.3f},{m['auroc_hi']:.3f}] "
                      f"MCC={m['mcc']:.4f}", flush=True)
            del model
    import pandas as pd
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[fm_finetune] wrote {out_csv} ({len(df)} rows)")
    # seed-aggregated summary
    agg = (df.groupby(["dataset", "train_arm", "eval_arm"])
             .agg(n_seeds=("seed", "size"), auroc_mean=("auroc", "mean"),
                  auroc_sd=("auroc", "std"), auroc_min=("auroc", "min"),
                  auroc_max=("auroc", "max"), mcc_mean=("mcc", "mean"),
                  mcc_sd=("mcc", "std"), n_test=("n_test", "first"),
                  n_train=("n_train", "first")).reset_index())
    agg_csv = out_csv.replace(".csv", "_summary.csv")
    agg.round(6).to_csv(agg_csv, index=False)
    print(agg.round(4).to_string(index=False))
    print(f"[fm_finetune] wrote {agg_csv}")


if __name__ == "__main__":
    main()
