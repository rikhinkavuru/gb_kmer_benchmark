#!/usr/bin/env python3
"""Torch checks for gpu_deferred/extra_models.py (run in .venv-ft on the laptop, CPU, < 1 GB).

  1. select_device prefers cuda > mps > cpu, and an explicit --device always wins.
  2. Blocking triton through sys.modules makes `import triton` raise ImportError. This is how
     run_gpu.py keeps DNABERT-2 on its PyTorch attention path on a CUDA machine.
  3. NT-v2 through the transformers-5 shims (50M checkpoint; its modeling code is byte-identical to
     the 500M one, and this is asserted):
       a. NTv2LoraWrap loads every checkpoint tensor (asserted in the wrapper) and wraps exactly
          12 x 6 = 72 linears with LoRA;
       b. LoRA is a no-op at init (B = 0): the wrapped backbone's hidden states equal the plain
          pretrained EsmModel's;
       c. its hidden states and masked-LM logits on 16 real (ragged, padded) enhancer sequences
          equal those of transformers 4.57 running the unmodified remote code (reference written
          by tests/ntv2_tf4_reference.py in a throwaway 4.57 venv): the 5.x port is an equivalence;
       d. the pretrained weights are in use (the MLM head reconstructs its input tokens; a
          random-init model of the same shape does not).
  4. The NT-v2 500M checkpoint at the pinned revision matches the model this code builds from its
     config. Every tensor name and shape is compared against the safetensors header, fetched with
     an HTTP range request (no 2 GB download). Exact total and LoRA-trainable parameter counts of
     NT-v2 500M are reported from a model built on the meta device (no memory). (DNABERT-2
     full-FT counts are logged by the smoke run, before its layer truncation.)
Run: HF_HUB_OFFLINE=0 bibm/lowcpu.sh .venv-ft/bin/python \
         src/upgrades/fm_finetune/gpu_deferred/tests/check_extra_models.py
"""
import filecmp
import json
import os
import struct
import sys
import types
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GD = os.path.dirname(HERE)
sys.path.insert(0, GD)
sys.path.insert(0, os.path.dirname(GD))

import numpy as np  # noqa: E402

import extra_models as X  # noqa: E402
import run_gpu as R  # noqa: E402


def check_device():
    def fake(cuda, mps):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda),
                                     backends=types.SimpleNamespace(
                                         mps=types.SimpleNamespace(is_available=lambda: mps)))
    assert R.select_device(None, fake(True, True)) == "cuda"
    assert R.select_device(None, fake(False, True)) == "mps"
    assert R.select_device(None, fake(False, False)) == "cpu"
    assert R.select_device("cpu", fake(True, True)) == "cpu"
    print("ok  1. device selection cuda > mps > cpu, explicit --device wins")


def check_triton_block():
    saved = sys.modules.get("triton", "absent")
    sys.modules["triton"] = None
    try:
        import triton  # noqa: F401
        raise AssertionError("triton import was not blocked")
    except ImportError:
        pass
    finally:
        if saved == "absent":
            del sys.modules["triton"]
        else:
            sys.modules["triton"] = saved
    print("ok  2. sys.modules['triton'] = None -> ImportError (DNABERT-2 PyTorch attention path)")


def check_ntv2_50m():
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors.torch import load_file
    torch.manual_seed(0)
    rc = X.NEW_RECIPES["ntv2_50m"]
    m = X.NTv2LoraWrap(42, "cpu", rc)
    assert m.n_lora_modules == 72, m.n_lora_modules
    local = X.hf_snapshot(rc["repo"], rc["revision"], "model.safetensors")
    big = os.path.join(X.V2.HF_CACHE, "models--" + X.NTV2["ntv2_500m"][0].replace("/", "--"),
                       "snapshots", X.NTV2["ntv2_500m"][1])
    for f in ("modeling_esm.py", "esm_config.py"):
        assert filecmp.cmp(os.path.join(local, f), os.path.join(big, f), shallow=False), f
    n_tr = sum(p.numel() for p in m.trainable())
    print(f"ok  3a. NT-v2-50M: all {m.n_ckpt_tensors} checkpoint tensors loaded, 72 LoRA modules, "
          f"trainable {n_tr:,} of {m.n_total():,}; modeling code byte-identical to 500M")

    # real sequences: first 16 cohn test sequences (LoRA no-op check on a 120 bp crop)
    import pandas as pd
    te = pd.read_csv(os.path.join(X.V2.SPLIT_DIR, "human_enhancers_cohn__test.csv.gz"), nrows=16)
    seqs = [s[:120] for s in te["sequence"].astype(str)]
    enc = m.tok(seqs, return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]

    esm_cls = get_class_from_dynamic_module("modeling_esm.EsmModel", local)
    head_cls = get_class_from_dynamic_module("modeling_esm.EsmLMHead", local)
    cfg = X.ntv2_config(local)
    sd = load_file(os.path.join(local, "model.safetensors"))
    plain = esm_cls(cfg, add_pooling_layer=False)
    plain.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("esm.")}, strict=False)
    plain.eval(); m.train_mode(False)
    with torch.no_grad():
        h_lora = m.backbone(input_ids=ids, attention_mask=am).last_hidden_state
        h_plain = plain(input_ids=ids, attention_mask=am).last_hidden_state
    d = float((h_lora - h_plain).abs().max())
    assert d < 1e-5, d
    print(f"ok  3b. LoRA wrapper is a no-op at init (max |dh| = {d:.2e})")
    special = set(m.tok.all_special_ids); mask_id = m.tok.mask_token_id; tok = m.tok
    del m, h_lora, h_plain                                   # keep the check under 1 GB

    head = head_cls(cfg)
    head.load_state_dict({k[len("lm_head."):]: v for k, v in sd.items()
                          if k.startswith("lm_head.")}, strict=True)
    head.eval()
    del sd
    import gc; gc.collect()
    # (i) the transformers-4.57 reference (tests/ntv2_tf4_reference.py): same inputs, same weights
    ref_path = os.path.join(X.V2.ROOT, "external", "fm_finetune", "gpu_smoke",
                            "ntv2_50m_tf4_reference.npz")
    ref = np.load(ref_path)
    ids, am, masked = (torch.tensor(ref[k]) for k in ("ids", "am", "masked"))
    pick = ref["pick"]; pi, pj = torch.tensor(pick[:, 0]), torch.tensor(pick[:, 1])
    te16 = pd.read_csv(os.path.join(X.V2.SPLIT_DIR, "human_enhancers_cohn__test.csv.gz"), nrows=16)
    seqs = [s[:300 - 7 * i] for i, s in enumerate(te16["sequence"].astype(str))]
    assert torch.equal(tok(seqs, return_tensors="pt", padding=True)["input_ids"], ids), \
        "tokenization differs from the transformers-4 reference"
    with torch.no_grad():
        h5 = plain(input_ids=ids, attention_mask=am).last_hidden_state
        lg5 = head(plain(input_ids=masked, attention_mask=am).last_hidden_state)[pi, pj]
    real = (am == 1) & torch.tensor(~np.isin(ids.numpy(), list(special)))
    dh = float((h5 - torch.tensor(ref["hidden"]))[am.bool()].abs().max())
    dl = float((lg5 - torch.tensor(ref["masked_logits"])).abs().max())
    scale = float(torch.tensor(ref["hidden"])[am.bool()].abs().max())
    print(f"    vs transformers {ref['transformers']} reference: max |dh| = {dh:.2e} (|h| max "
          f"{scale:.1f}) over {int(am.sum())} real tokens; masked-LM logits max |d| = {dl:.2e}")
    assert dh < 1e-3 and dl < 1e-3, (dh, dl)
    print("ok  3c. NT-v2 under transformers 5.9 + shims == transformers 4.57 (hidden states and "
          "masked-LM logits, identical tokenization)")

    # (ii) descriptive: the pretrained model reconstructs its input; masked 6-mers vs random init
    def readout(model):
        with torch.no_grad():
            recon = head(model(input_ids=ids, attention_mask=am).last_hidden_state).argmax(-1)
            lg = head(model(input_ids=masked, attention_mask=am).last_hidden_state)
        r = float((recon[real] == ids[real]).float().mean())
        lp = torch.log_softmax(lg[pi, pj], -1).gather(-1, ids[pi, pj].unsqueeze(-1)).squeeze(-1)
        return r, float(lp.mean())

    torch.manual_seed(1)
    rand = esm_cls(cfg, add_pooling_layer=False).eval()
    (r_pre, lp_pre), (r_rand, lp_rand) = readout(plain), readout(rand)
    uni = float(np.log(1.0 / cfg.vocab_size))
    print(f"    unmasked-token reconstruction ({int(real.sum())} tokens): pretrained {r_pre:.3f} vs "
          f"random-init {r_rand:.3f}; masked tokens ({len(pick)}): mean log p(true 6-mer) "
          f"pretrained {lp_pre:.3f}, random-init {lp_rand:.3f}, uniform {uni:.3f}")
    assert r_pre > 0.9 and r_rand < 0.05, (r_pre, r_rand)
    print("ok  3d. pretrained weights are in use (input reconstruction 0.9+ vs ~0 at random init)")
    del plain, rand


def safetensors_header(repo, rev, fname="model.safetensors"):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=60) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        hdr = json.loads(r.read(n))
    hdr.pop("__metadata__", None)
    return {k: tuple(v["shape"]) for k, v in hdr.items()}


def check_500m_and_counts():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    X.apply_transformers5_shims()
    repo, rev = X.NTV2["ntv2_500m"]
    rc = X.NEW_RECIPES["ntv2_500m"]
    local = os.path.join(X.V2.HF_CACHE, "models--" + repo.replace("/", "--"), "snapshots", rev)
    if not os.path.isfile(os.path.join(local, "config.json")):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo, revision=rev, cache_dir=X.V2.HF_CACHE,
                                  allow_patterns=["*.json", "*.py", "*.txt"])
    cfg = X.ntv2_config(local)
    esm_cls = get_class_from_dynamic_module("modeling_esm.EsmModel", local)
    with torch.device("meta"):
        base = esm_cls(cfg, add_pooling_layer=False)
    shapes = {k: tuple(v.shape) for k, v in base.state_dict().items()}
    try:
        hdr = safetensors_header(repo, rev)
    except Exception as e:                                  # offline: report, do not pass
        print(f"SKIP 4. could not fetch the 500M safetensors header ({e})")
        hdr = None
    if hdr is not None:
        esm = {k[4:]: s for k, s in hdr.items() if k.startswith("esm.")}
        miss = [k for k in shapes if k not in esm and not k.endswith("inv_freq")]
        extra = [k for k in esm if k not in shapes]
        wrong = [k for k in shapes if k in esm and esm[k] != shapes[k]]
        assert not miss and not extra and not wrong, (miss[:3], extra[:3], wrong[:3])
        print(f"ok  4. NT-v2-500M @ {rev[:8]}: all {len(esm)} esm.* checkpoint tensors match the "
              f"model built from its config (names and shapes)")
    n_base = sum(p.numel() for p in base.parameters())
    lcfg = LoraConfig(r=rc["lora_r"], lora_alpha=rc["lora_alpha"], lora_dropout=rc["lora_dropout"],
                      target_modules=rc["lora_targets"], bias="none")
    pm = get_peft_model(base, lcfg)
    n_mod = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A"))
    n_lora = sum(p.numel() for n, p in pm.named_parameters() if p.requires_grad)
    head = cfg.hidden_size * 2 + 2
    assert n_mod == cfg.num_hidden_layers * 6
    print(f"    NT-v2-500M LoRA: {n_mod} modules, trainable {n_lora + head:,} "
          f"(LoRA {n_lora:,} + head {head:,}) of {n_base + n_lora + head:,} total")

    return dict(ntv2_500m_trainable=n_lora + head, ntv2_500m_total=n_base + n_lora + head,
                ntv2_500m_lora_modules=n_mod)


if __name__ == "__main__":
    import memguard                      # fm_finetune/memguard.py (read-only import)
    guard = memguard.MemGuard(1100, print, interval=1.0).start()   # hard stop above 1.1 GB
    import torch
    torch.set_num_threads(2)
    check_device()
    check_triton_block()
    check_ntv2_50m()
    counts = check_500m_and_counts()
    guard.sample()
    counts["peak_footprint_mb"] = round(guard.peak_mb)
    print("all gpu_deferred model checks passed", json.dumps(counts))
