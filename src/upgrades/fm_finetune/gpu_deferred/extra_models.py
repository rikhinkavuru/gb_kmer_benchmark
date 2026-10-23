#!/usr/bin/env python3
"""Model wrappers and recipes for the deferred GPU runs (driven by run_gpu.py in this directory).

Everything except the model wrapper is run_ft_v2.py's code, imported read-only: the same split files,
the same stratified 10% validation carve per seed, the same mixed-class batcher, the same
class-weighted cross-entropy, the same warmup + cosine schedule and gradient clipping, the same
best-epoch selection on validation AUROC, and the same per-sequence prediction files. This module
only adds models, which are registered into run_ft_v2.RECIPES / run_ft_v2.build_model at runtime.

PRE-REGISTERED RECIPES (fixed 2026-09-24, before any unit of these models was trained)
  dnabert2full  zhihan1996/DNABERT-2-117M @ 7bce263b (the frozen-embedding / LoRA revision),
                FULL-parameter fine-tune: all 117M backbone weights plus the linear head on the
                masked mean-pool are trainable. AdamW lr 3e-5 (the learning rate the DNABERT-2
                authors used for GUE fine-tuning), weight decay 0.01, 10% warmup + cosine, grad clip
                1.0, batch 32, class-weighted CE, at most 4 epochs, early stopping with patience 2 on
                validation AUROC, per-layer activation checkpointing (saves memory only). PyTorch
                attention is used, as in every local DNABERT-2 run. Compared with the LoRA recipe
                (run_ft_v2 "dnabert2"), only the trainable set and the learning rate differ.
  ntv2_500m     InstaDeepAI/nucleotide-transformer-v2-500m-multi-species @ 06615c16. LoRA (r=8,
                alpha=16, dropout 0.05) on the self-attention query/key/value, the attention-output
                dense and both SwiGLU FFN projections (intermediate.dense, output.dense) of all 29
                layers (174 modules). Backbone frozen; trainable linear head on the attention-mask
                mean-pool of the last layer, pooled exactly as for DNABERT-2. Optimizer, schedule,
                batch, epochs and early stopping are identical to the DNABERT-2 LoRA recipe
                (lr 3e-4), with no per-model tuning. Per-layer activation checkpointing. The 6-mer
                tokenizer has a 2048-token limit, so no sequence of the three datasets is truncated
                (the longest, 3237 bp, is ~540 tokens).
  ntv2_50m      The same recipe on the 50M checkpoint, whose modeling code is byte-identical to
                the 500M one (same revision family). Used ONLY for the local wiring test. It is
                never reported.

transformers 5.x compatibility of the NT-v2 remote code (modeling_esm.py was written for 4.32):
two helpers it imports were removed in 5.x. apply_transformers5_shims() puts back exact copies with
unchanged behaviour: find_pruneable_heads_and_indices (called only by prune_heads, which this
pipeline never uses) and PreTrainedModel.get_head_mask, which for head_mask=None returns
[None] * n_layers as 4.x did. Any other head_mask raises. Two PretrainedConfig defaults the code
reads (is_decoder, add_cross_attention) are no longer set by 5.x, so ntv2_config() sets them to their
4.x values (False, False: an encoder). Weights are loaded explicitly from the
checkpoint, and every model key must be matched, so a silent random initialisation is impossible.
tests/check_extra_models.py checks that the pretrained NT-v2 masked-LM head predicts masked 6-mers
far above chance through this code path, i.e. the forward is numerically sound under 5.x.
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
FT_DIR = os.path.dirname(HERE)
if FT_DIR not in sys.path:
    sys.path.insert(0, FT_DIR)

import run_ft_v2 as V2  # noqa: E402  (read-only import; torch is imported lazily there)

NTV2 = {
    "ntv2_500m": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                  "06615c1660c892fc199840c18123f8385b3542a8"),
    "ntv2_50m": ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                 "81b29e5786726d891dbf929404ef20adca5b36f1"),
}
# peft full-matches this against module names relative to the EsmModel
NTV2_LORA_TARGETS = (r"encoder\.layer\.\d+\.(attention\.self\.(query|key|value)"
                     r"|attention\.output\.dense|intermediate\.dense|output\.dense)")
NTV2_LORA_PER_LAYER = 6

_LORA_COMMON = dict(epochs=4, batch=32, lr=3e-4, weight_decay=0.01, warmup_frac=0.1, clip=1.0,
                    patience=2, eval_batch=32, finetune="lora", lora_r=8, lora_alpha=16,
                    lora_dropout=0.05, grad_ckpt=True)
NEW_RECIPES = {
    "dnabert2full": dict(tag="dnabert2-117m-full", epochs=4, batch=32, lr=3e-5, weight_decay=0.01,
                         warmup_frac=0.1, clip=1.0, patience=2, eval_batch=32, finetune="full",
                         grad_ckpt=True, repo=V2.DNABERT_REPO, revision=V2.DNABERT_REV),
    "ntv2_500m": dict(_LORA_COMMON, tag="nt-v2-500m-multi-species-lora",
                      lora_targets=NTV2_LORA_TARGETS, repo=NTV2["ntv2_500m"][0],
                      revision=NTV2["ntv2_500m"][1], max_tokens=2048),
    "ntv2_50m": dict(_LORA_COMMON, tag="nt-v2-50m-multi-species-lora-WIRING-TEST-ONLY",
                     lora_targets=NTV2_LORA_TARGETS, repo=NTV2["ntv2_50m"][0],
                     revision=NTV2["ntv2_50m"][1], max_tokens=2048),
}
MODELS = ("hyenadna", "dnabert2", "dnabert2full", "ntv2_500m", "ntv2_50m")
SMOKE_LAYERS = 0     # set by register(); >0 ONLY in the laptop wiring test (see DnabertFullWrap)


# ----------------------------------------------------------------- helpers
def force_torch_attention(module):
    """Disable DNABERT-2's triton flash-attention path in every loaded copy of its remote code.
    The forward reads the module-level `flash_attn_qkvpacked_func` at call time and falls back to
    PyTorch attention when it is None. That is the path every local run took, because macOS has no
    triton. On a CUDA machine with triton installed the flash path would otherwise be taken (the
    checkpoint sets attention dropout to 0.0), and its 2023 triton kernel does not compile under
    current triton."""
    names = sorted({type(sm).__module__ for sm in module.modules()
                    if type(sm).__name__ == "BertUnpadSelfAttention"})
    assert names, "DNABERT-2 attention module not found"
    for n in names:
        mod = sys.modules[n]
        assert hasattr(mod, "flash_attn_qkvpacked_func"), n
        mod.flash_attn_qkvpacked_func = None
    return names


def apply_transformers5_shims():
    import torch
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            # verbatim from transformers 4.x pytorch_utils (only used by prune_heads)
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(mu.PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                raise NotImplementedError("head_mask is not supported in this pipeline")
            return [None] * num_hidden_layers
        mu.PreTrainedModel.get_head_mask = get_head_mask


V4_CONFIG_DEFAULTS = dict(is_decoder=False, add_cross_attention=False)


def ntv2_config(local):
    """NT-v2 config under transformers 5.x. The remote EsmConfig relies on two PretrainedConfig
    defaults that 5.x no longer sets, so they are filled in with their 4.x values (an encoder, no
    cross-attention). That is what the checkpoint is: config.json predates 5.x and does not set
    them."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(local, trust_remote_code=True)
    for k, v in V4_CONFIG_DEFAULTS.items():
        if getattr(cfg, k, None) is None:
            setattr(cfg, k, v)
    assert cfg.is_decoder is False and cfg.add_cross_attention is False
    return cfg


def ntv2_tokenizer(local):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    assert type(tok).__name__ == "EsmTokenizer", type(tok)       # native class, no remote code
    return tok


def hf_snapshot(repo, revision, weights):
    """Local snapshot of `repo` @ `revision` under run_ft_v2.HF_CACHE (downloaded if missing)."""
    path = os.path.join(V2.HF_CACHE, "models--" + repo.replace("/", "--"), "snapshots", revision)
    if not os.path.isfile(os.path.join(path, weights)):
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, revision=revision, cache_dir=V2.HF_CACHE,
                                 allow_patterns=["*.json", "*.py", "*.txt", weights])
    assert path.rstrip("/").endswith(revision), path
    return path


def encoder_of(m):
    """The nn.Module that holds `.layer` (the encoder stack) for any wrapper here."""
    bb = m.backbone
    base = bb.base_model.model if hasattr(bb, "peft_config") else bb
    return base.encoder


def truncate_layers(m, n):
    """SMOKE TEST ONLY: keep the first n encoder layers, to bound memory on the laptop. Applied
    after the full pretrained weights have been loaded and checked."""
    import torch
    enc = encoder_of(m)
    enc.layer = torch.nn.ModuleList(list(enc.layer)[:n])


# ----------------------------------------------------------------- wrappers
class DnabertFullWrap(V2.DnabertLoraWrap):
    """DNABERT-2, every weight trainable, with the masked-mean-pool linear head. The loading code is
    copied from run_ft_v2.DnabertLoraWrap.__init__ (identical checks). The one change is
    torch.load(mmap=True), which only saves memory. trainable / encode / logits / best_state /
    load_best are inherited unchanged; best_state therefore stores every trainable weight."""

    def __init__(self, seed, device, rc):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        torch.manual_seed(seed)
        V2._bypass_static_triton_check()
        local = V2.dnabert_snapshot()
        self.tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(local, trust_remote_code=True)
        cfg.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 3
        if SMOKE_LAYERS:          # wiring test only: build the first L layers, keeps the laptop < 1 GB
            cfg.num_hidden_layers = SMOKE_LAYERS
        with torch.device("cpu"):
            base = AutoModel.from_config(cfg, trust_remote_code=True)
        sd = torch.load(os.path.join(local, "pytorch_model.bin"), map_location="cpu",
                        weights_only=True, mmap=True)
        sd = {(k[len("bert."):] if k.startswith("bert.") else k): v
              for k, v in sd.items() if not k.startswith("cls.")}
        missing, unexpected = base.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("pooler.")]
        assert not bad, f"DNABERT-2 weights did NOT load (random-init risk): {bad[:5]}"
        if SMOKE_LAYERS:          # the only checkpoint tensors left out are those of layers >= L
            assert all(k.startswith("encoder.layer.") and int(k.split(".")[2]) >= SMOKE_LAYERS
                       for k in unexpected), unexpected[:5]
        del sd
        self.fixed_shapes = bool(V2.PAD_MULTIPLE)
        V2.set_dnabert_fixed_shapes(base, self.fixed_shapes)
        torch.manual_seed(seed)                       # head init depends only on seed
        for p in base.parameters():
            p.requires_grad_(True)
        self.backbone = base
        if rc.get("grad_ckpt"):
            V2.enable_layer_checkpointing(SimpleNamespace(base_model=SimpleNamespace(model=base)))
        self.head = torch.nn.Linear(768, 2)
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        self.device = device
        self.backbone.to(device)
        self.head.to(device)
        self.n_lora_modules = 0


class NTv2LoraWrap(V2.DnabertLoraWrap):
    """Nucleotide Transformer v2 (ESM-style, rotary, SwiGLU) with LoRA adapters and the
    masked-mean-pool linear head. trainable / train_mode / best_state / load_best are inherited
    from the DNABERT-2 LoRA wrapper; encode and logits are specific to NT-v2."""

    def __init__(self, seed, device, rc):
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        apply_transformers5_shims()
        torch.manual_seed(seed)
        local = hf_snapshot(rc["repo"], rc["revision"], "model.safetensors")
        self.tok = ntv2_tokenizer(local)                         # native EsmTokenizer (6-mer)
        cfg = ntv2_config(local)
        esm_cls = get_class_from_dynamic_module("modeling_esm.EsmModel", local)
        with torch.device("cpu"):
            base = esm_cls(cfg, add_pooling_layer=False)
        sd = load_file(os.path.join(local, "model.safetensors"))
        sd = {k[len("esm."):]: v for k, v in sd.items() if k.startswith("esm.")}
        missing, unexpected = base.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("inv_freq")]   # deterministic buffers
        assert not missing and not unexpected, \
            f"NT-v2 weights did NOT load cleanly: missing={missing[:5]} unexpected={unexpected[:5]}"
        self.n_ckpt_tensors = len(sd)
        del sd
        torch.manual_seed(seed)                       # LoRA-A init + head init depend only on seed
        lcfg = LoraConfig(r=rc["lora_r"], lora_alpha=rc["lora_alpha"],
                          lora_dropout=rc["lora_dropout"], target_modules=rc["lora_targets"],
                          bias="none")
        self.backbone = get_peft_model(base, lcfg)
        n_lora = sum(1 for n, _ in self.backbone.named_modules() if n.endswith("lora_A"))
        want = cfg.num_hidden_layers * NTV2_LORA_PER_LAYER
        assert n_lora == want, f"expected {want} LoRA-wrapped linears, got {n_lora}"
        if rc.get("grad_ckpt"):
            V2.enable_layer_checkpointing(self.backbone)
        self.hidden = cfg.hidden_size
        self.max_tokens = int(rc.get("max_tokens", 2048))
        self.head = torch.nn.Linear(cfg.hidden_size, 2)
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        self.device = device
        self.backbone.to(device)
        self.head.to(device)
        self.n_lora_modules = n_lora
        self.fixed_shapes = False

    def encode(self, seqs):
        return self.tok(list(seqs), return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_tokens)

    def logits(self, enc):
        ids = enc["input_ids"].to(self.device)
        am = enc["attention_mask"].to(self.device)
        out = self.backbone(input_ids=ids, attention_mask=am)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        m = am.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(pooled)


# ----------------------------------------------------------------- registry
def register(v2=V2, smoke_layers=0):
    """Add the new recipes to run_ft_v2.RECIPES and route run_ft_v2.build_model (which train_unit
    and bench look up at call time) through this module. For hyenadna and dnabert2 the original
    classes are used unchanged; the only addition for DNABERT-2 is force_torch_attention."""
    global SMOKE_LAYERS
    SMOKE_LAYERS = int(smoke_layers)
    v2.RECIPES.update(NEW_RECIPES)

    def build_model(model, seed, device):
        if model == "hyenadna":
            m = v2.HyenaWrap(seed, device)
        elif model == "dnabert2":
            m = v2.DnabertLoraWrap(seed, device, v2.RECIPES["dnabert2"])
        elif model == "dnabert2full":
            m = DnabertFullWrap(seed, device, v2.RECIPES["dnabert2full"])
        elif model in NTV2:
            m = NTv2LoraWrap(seed, device, v2.RECIPES[model])
        else:
            raise ValueError(model)
        if model.startswith("dnabert2"):
            m.attention_modules = force_torch_attention(m.backbone)
        m.full_counts = (sum(p.numel() for p in m.trainable()), m.n_total())
        print(f"[extra_models] {model}: trainable {m.full_counts[0]:,} of {m.full_counts[1]:,} "
              f"params (full model)", flush=True)
        if smoke_layers and model != "hyenadna":
            truncate_layers(m, smoke_layers)
        return m

    v2.build_model = build_model
    return build_model
