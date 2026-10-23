#!/usr/bin/env python3
"""Runner for the deferred GPU part of the fine-tuning experiments (Reviewer 1, comment 2). These
are the units that cannot run well on the 16 GB laptop; see README.md for the ranked job list, the
environment and how results are handed back.

It uses run_ft_v2.py's code path: it imports run_ft_v2 read-only and calls its load_split /
run_unit / train_unit / predict. So the splits, the per-seed stratified 10% validation carve, the
mixed-class batcher, class-weighted CE, warmup + cosine, clipping, best-epoch selection /
early stopping, the seed-major order, skip-if-done resumption and the prediction-file format are
all the same. What this runner adds:
  * device auto-selection cuda > mps > cpu (--device overrides);
  * SHA-256 verification of the split files against SHA256SUMS (refuses to train on other data);
  * DNABERT-2: PyTorch attention on every device (triton import blocked before torch is imported,
    plus the module-level flash flag forced to None and asserted). Every laptop run used PyTorch
    attention, since there is no triton on macOS;
  * the new pre-registered recipes of extra_models.py (dnabert2full, ntv2_500m; ntv2_50m is for the
    wiring test only);
  * telemetry in each unit's JSON: GPU name, CUDA peak allocated/reserved memory, TF32 flags, host
    peak RSS in MB on every OS, runner/code hashes;
  * --bench N: time N optimizer steps (CUDA-synchronized) and extrapolate per-unit / per-job time;
  * --prefetch: download all pinned model snapshots, then exit (for offline compute nodes);
  * --smoke N [--crop BP --subset K --smoke-layers L]: tiny wiring test. It writes ONLY to --pred-dir.
    Cropping, subsetting and layer truncation are refused outside --smoke.

Usage
  python run_gpu.py --prefetch
  python run_gpu.py --model dnabert2full --dataset human_enhancers_cohn --bench 30
  python run_gpu.py --model dnabert2full --dataset human_enhancers_cohn \
      --arms original,comp_equalized,random --seeds 42,43,44
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FT_DIR = os.path.dirname(HERE)
SUMS = os.path.join(HERE, "SHA256SUMS")
SMOKE_ONLY_MODELS = ("ntv2_50m",)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("hyenadna", "dnabert2", "dnabert2full", "ntv2_500m",
                                        "ntv2_50m"))
    ap.add_argument("--dataset", default="human_enhancers_cohn")
    ap.add_argument("--arms", default="original,comp_equalized,random")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (torch.set_num_threads)")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 matmuls on Ampere+ (default OFF = plain fp32, as on the "
                         "laptop). If used, use it for EVERY unit of that model.")
    ap.add_argument("--allow-triton", action="store_true",
                    help="do not block triton (DNABERT-2 would then try its old flash kernel)")
    ap.add_argument("--mem-cap-mb", type=float, default=0.0,
                    help="hard host-memory cap (memguard; 0 = off)")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--pred-dir", default=None, help="default: run_ft_v2.PRED_DIR")
    ap.add_argument("--smoke", type=int, default=0, help="stop training after N steps (wiring)")
    ap.add_argument("--crop", type=int, default=0, help="SMOKE ONLY: crop sequences to BP")
    ap.add_argument("--subset", type=int, default=0,
                    help="SMOKE ONLY: keep K train sequences per class per arm")
    ap.add_argument("--smoke-layers", type=int, default=0,
                    help="SMOKE ONLY: keep the first L encoder layers (dnabert2*/ntv2*)")
    a = ap.parse_args()
    if not a.prefetch and not a.model:
        ap.error("--model is required")
    if (a.crop or a.subset or a.smoke_layers) and not a.smoke:
        ap.error("--crop / --subset / --smoke-layers are wiring-test options; they need --smoke")
    if a.smoke and not a.pred_dir:
        ap.error("--smoke needs --pred-dir (smoke output never goes to results/)")
    if a.model in SMOKE_ONLY_MODELS and not (a.smoke or a.bench):
        ap.error(f"{a.model} is the wiring-test stand-in; use ntv2_500m for real runs")
    return a


def select_device(pref, torch):
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_splits(V2, dataset):
    want = {}
    with open(SUMS) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                digest, rel = line.split()
                want[os.path.basename(rel)] = digest
    for sp in ("train", "test"):
        name = f"{dataset}__{sp}.csv.gz"
        path = os.path.join(V2.SPLIT_DIR, name)
        assert name in want, f"{name} not in SHA256SUMS"
        got = sha256(path)
        assert got == want[name], f"split file {path} differs from SHA256SUMS ({got} != {want[name]})"
    return True


def smoke_subset(tr, te, arms, crop, k, rng_seed=0):
    import numpy as np
    rng = np.random.RandomState(rng_seed)
    if k:
        keep = set()
        for arm in arms:
            sub = tr[tr[f"{arm}_pos"] >= 0]
            for lab in (0, 1):
                rows = sub.index[sub["label"] == lab].to_numpy()
                keep.update(rng.choice(rows, min(k, len(rows)), replace=False).tolist())
        tr = tr.loc[sorted(keep)].copy()
    if crop:
        tr = tr.copy(); te = te.copy()
        tr["sequence"] = tr["sequence"].astype(str).str[:crop]
        te["sequence"] = te["sequence"].astype(str).str[:crop]
    return tr, te


def code_hashes(V2):
    out = {}
    for f in (V2.__file__, V2.ft.__file__, os.path.join(HERE, "extra_models.py"), __file__):
        out[os.path.basename(f)] = sha256(f)[:16]
    return out


def host_max_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3      # bytes on macOS, KiB on Linux


def main():
    a = parse()
    if a.model and a.model.startswith("dnabert2") and not a.allow_triton:
        sys.modules["triton"] = None        # -> ImportError -> DNABERT-2's PyTorch attention path
    if FT_DIR not in sys.path:
        sys.path.insert(0, FT_DIR)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import run_ft_v2 as V2
    import extra_models as X
    need = ("load_split", "arm_train", "run_unit", "train_unit", "predict", "mixed_batches",
            "RECIPES", "build_model", "HyenaWrap", "DnabertLoraWrap", "dnabert_snapshot",
            "enable_layer_checkpointing", "set_dnabert_fixed_shapes", "ARMS", "VAL_FRAC",
            "SPLIT_DIR", "PRED_DIR", "CKPT_DIR", "HF_CACHE", "PAD_MULTIPLE", "ft")
    missing = [n for n in need if not hasattr(V2, n)]
    assert not missing, f"run_ft_v2.py API changed; missing {missing}"
    assert V2.PAD_MULTIPLE == 0, "the GPU runs use run_ft_v2's default per-batch padding"
    X.register(V2, smoke_layers=a.smoke_layers)

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    if a.mem_cap_mb:
        import memguard
        V2.GUARD = memguard.MemGuard(a.mem_cap_mb, log).start()
    import torch
    torch.set_num_threads(a.threads)
    if a.smoke:      # memory only: memory-map checkpoint loads (identical tensors), keeps smoke < 1 GB
        _load = torch.load

        def _mmap_load(*args, **kw):
            kw.setdefault("mmap", True)
            return _load(*args, **kw)
        torch.load = _mmap_load
    dev = select_device(a.device, torch)
    cuda = dev.startswith("cuda")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
        torch.backends.cudnn.allow_tf32 = bool(a.tf32)     # HyenaDNA's short conv: fp32 too
        # HyenaDNA on drosophila meets hundreds of FFT sizes (2 x padded length); bound the cuFFT
        # plan cache (a cache only; results are identical). On MPS the analogous per-shape graph
        # cache is what grew to 17.6 GB.
        torch.backends.cuda.cufft_plan_cache[0].max_size = 128
    gpu = torch.cuda.get_device_name(0) if cuda else None
    log(f"[run_gpu] device={dev} gpu={gpu} torch={torch.__version__} "
        f"cuda={torch.version.cuda} tf32={a.tf32} triton_blocked={sys.modules.get('triton', 1) is None}")

    if a.prefetch:
        prefetch(V2, X, log)
        return
    verify_splits(V2, a.dataset)
    if a.bench:
        bench(V2, a.model, a.dataset, dev, a.bench, log)
        return

    tr, te = V2.load_split(a.dataset)
    arms = a.arms.split(",")
    assert all(arm in V2.ARMS for arm in arms), arms
    if a.smoke:
        tr, te = smoke_subset(tr, te, arms, a.crop, a.subset)
    pdir = a.pred_dir or V2.PRED_DIR
    cdir = os.path.join(pdir, "ckpt") if a.smoke else V2.CKPT_DIR
    hashes = code_hashes(V2)
    extra = dict(runner="gpu_deferred/run_gpu.py", code_sha256_16=hashes, gpu=gpu,
                 cuda_version=torch.version.cuda, tf32=bool(a.tf32),
                 triton_blocked=sys.modules.get("triton", 1) is None, python=platform.python_version(),
                 host=platform.platform(), smoke=dict(steps=a.smoke, crop=a.crop, subset=a.subset,
                                                      layers=a.smoke_layers) if a.smoke else None)
    orig_train = V2.train_unit

    def train_unit(*args, **kw):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        out = orig_train(*args, **kw)
        out[1].update(extra)
        return out

    V2.train_unit = train_unit
    log(f"[run_gpu] model={a.model} dataset={a.dataset} arms={arms} seeds={a.seeds} "
        f"pred_dir={pdir} code={hashes}")
    for seed in [int(s) for s in a.seeds.split(",")]:          # seed-major, as run_ft_v2.main
        for arm in arms:
            name = V2.unit_name(a.model, a.dataset, arm, seed)
            js = os.path.join(pdir, name + ".json")
            existed = os.path.exists(js)
            V2.run_unit(a.model, a.dataset, arm, seed, dev, tr, te, log, pred_dir=pdir,
                        ckpt_dir=cdir, max_steps=a.smoke or None)
            if not existed and os.path.exists(js):              # post-hoc telemetry for this unit
                meta = json.load(open(js))
                meta["max_rss_mb"] = round(host_max_rss_mb(), 1)
                meta["max_rss_note"] = "host peak RSS of the runner process so far, MB"
                if cuda:
                    meta["cuda_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                    meta["cuda_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
                with open(js + ".tmp", "w") as fh:
                    json.dump(meta, fh, indent=1)
                os.replace(js + ".tmp", js)
    log("[run_gpu] done")


def prefetch(V2, X, log):
    from transformers import AutoModel, AutoTokenizer
    ft = V2.ft
    AutoTokenizer.from_pretrained(ft.MODEL_REPO, revision=ft.MODEL_REVISION, trust_remote_code=True)
    AutoModel.from_pretrained(ft.MODEL_REPO, revision=ft.MODEL_REVISION, trust_remote_code=True)
    log(f"[prefetch] {ft.MODEL_REPO} @ {ft.MODEL_REVISION}")
    log(f"[prefetch] {V2.dnabert_snapshot()}")
    for key in ("ntv2_500m",):
        repo, rev = X.NTV2[key]
        log(f"[prefetch] {X.hf_snapshot(repo, rev, 'model.safetensors')}")


def bench(V2, model, dataset, dev, steps, log):
    """Time `steps` optimizer steps on the original arm (synchronized) and one inference pass,
    then extrapolate an upper bound per unit (all epochs, no early stop) for every arm."""
    import numpy as np
    import torch
    cuda = dev.startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    tr, te = V2.load_split(dataset)
    rc = V2.RECIPES[model]
    seqs, y, _ = V2.arm_train(tr, "original")
    fit_pos, _ = V2.ft.stratified_val_carve(y, V2.VAL_FRAC, 42)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    m = V2.build_model(model, 42, dev)
    n_tr = sum(p.numel() for p in m.trainable())
    log(f"[bench] {model} {dataset} trainable={n_tr:,} total={m.n_total():,}")
    opt = torch.optim.AdamW(m.trainable(), lr=rc["lr"], weight_decay=rc["weight_decay"])
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.RandomState(0)
    m.train_mode(True)
    times = []
    for i, chunk in enumerate(V2.mixed_batches(seqs, fit_pos, rc["batch"], rng)[:steps]):
        sync(); t0 = time.time()
        rows = fit_pos[chunk]
        enc = m.encode([seqs[j] for j in rows])
        loss = lossf(m.logits(enc), torch.tensor(y[rows], dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.trainable(), rc["clip"]); opt.step()
        sync(); times.append(time.time() - t0)
        if i % 10 == 0:
            log(f"  step {i} {times[-1]:.3f}s loss={float(loss):.4f} "
                f"tokens={int(enc['input_ids'].shape[1])}")
    n_inf = min(512, len(te))
    sync(); t0 = time.time()
    V2.predict(m, te["sequence"].astype(str).tolist()[:n_inf], rc["eval_batch"])
    sync(); t_inf = (time.time() - t0) / n_inf
    sps = float(np.median(times[3:] if len(times) > 6 else times))
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan")
    log(f"[bench] median {sps:.3f} s/step, inference {t_inf * 1e3:.2f} ms/seq, "
        f"peak CUDA allocated {peak:.2f} GB")
    total = 0.0
    for arm in V2.ARMS:
        n = int((tr[f"{arm}_pos"] >= 0).sum())
        n_val = int(round(V2.VAL_FRAC * n)); n_fit = n - n_val
        spe = int(np.ceil(n_fit / rc["batch"]))
        unit = rc["epochs"] * (spe * sps + n_val * t_inf) + (len(te) + n_val) * t_inf
        total += unit if arm != "gc_match" else 0.0
        log(f"[bench]   arm {arm:15s} n_fit={n_fit:6d} {spe:4d} steps/epoch -> <= {unit / 60:6.1f} "
            f"min per unit ({rc['epochs']} epochs, no early stop)")
    log(f"[bench] original+comp_equalized+random, one seed: <= {total / 3600:.2f} h; "
        f"three seeds: <= {3 * total / 3600:.2f} h")


if __name__ == "__main__":
    main()
