#!/usr/bin/env python3
"""Reference forward of NT-v2-50M under transformers 4.57, the 4.x line the remote code was written
for, loaded the standard way (AutoModelForMaskedLM.from_pretrained, trust_remote_code, no shims).
check_extra_models.py (transformers 5.9, with the shims) must reproduce these hidden states and
masked-LM logits. That makes the transformers-5 port of NT-v2 a tested equivalence, not an
assumption.

Env (throwaway, CPU): uv venv external/fm_finetune/venv-tf4 --python .venv-ft/bin/python
     uv pip install --python external/fm_finetune/venv-tf4/bin/python torch==2.12.0 \
         transformers==4.57.6 safetensors==0.8.0 numpy==2.5.1 pandas==3.0.5
Run: HF_HUB_OFFLINE=1 bibm/lowcpu.sh external/fm_finetune/venv-tf4/bin/python \
         src/upgrades/fm_finetune/gpu_deferred/tests/ntv2_tf4_reference.py
Writes external/fm_finetune/gpu_smoke/ntv2_50m_tf4_reference.npz
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
REPO, REV = ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
             "81b29e5786726d891dbf929404ef20adca5b36f1")
LOCAL = os.path.join(ROOT, "external", "fm_finetune", "hf_cache",
                     "models--" + REPO.replace("/", "--"), "snapshots", REV)
OUT = os.path.join(ROOT, "external", "fm_finetune", "gpu_smoke", "ntv2_50m_tf4_reference.npz")


def inputs(tok):
    te = pd.read_csv(os.path.join(ROOT, "external", "fm_finetune", "splits",
                                  "human_enhancers_cohn__test.csv.gz"), nrows=16)
    seqs = [s[:300 - 7 * i] for i, s in enumerate(te["sequence"].astype(str))]  # ragged -> padding
    enc = tok(seqs, return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]
    rng = np.random.RandomState(0)
    special = set(tok.all_special_ids)
    real = (am == 1) & torch.tensor(~np.isin(ids.numpy(), list(special)))
    cand = torch.nonzero(real).tolist()
    pick = np.array([cand[k] for k in rng.choice(len(cand), int(0.15 * len(cand)), replace=False)])
    masked = ids.clone()
    masked[pick[:, 0], pick[:, 1]] = tok.mask_token_id
    return seqs, ids, am, masked, pick


def main():
    assert transformers.__version__.startswith("4."), transformers.__version__
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(LOCAL, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(LOCAL, trust_remote_code=True).eval()
    seqs, ids, am, masked, pick = inputs(tok)
    with torch.no_grad():
        h = model.esm(input_ids=ids, attention_mask=am).last_hidden_state
        lg = model(input_ids=masked, attention_mask=am).logits
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, ids=ids.numpy(), am=am.numpy(), masked=masked.numpy(), pick=pick,
                        hidden=h.numpy(), masked_logits=lg[pick[:, 0], pick[:, 1]].numpy(),
                        transformers=transformers.__version__, torch=torch.__version__)
    print(f"wrote {OUT}: transformers {transformers.__version__}, hidden {tuple(h.shape)}, "
          f"{len(pick)} masked positions")


if __name__ == "__main__":
    sys.exit(main())
