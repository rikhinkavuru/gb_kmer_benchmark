"""Second benchmark suite: Nucleotide Transformer downstream tasks (HuggingFace).

CPU-only. Reads the per-task parquet files from
InstaDeepAI/nucleotide_transformer_downstream_tasks via hf_hub_download (locally
cached; no torch/tensorflow). Returns the SAME dict shape as data.load_dataset, so
the entire existing pipeline (featurize -> LightGBM -> enrichment -> discriminability
-> bootstrap/contamination) runs unchanged via the ``nt_`` dataset-name prefix.

These tasks carry published foundation-model numbers (Nucleotide Transformer,
DNABERT-2, HyenaDNA), so they are an independent replication suite. Sequences are
fixed length per task (promoters 300 bp, enhancers 200 bp, splice 400 bp); labels
are already 0-indexed ints (binary or, for enhancers_types / splice_sites_all,
3-class). Label 1 is the positive class for the binary tasks.
"""
import numpy as np

REPO = "InstaDeepAI/nucleotide_transformer_downstream_tasks"
REVISION = "96d86d567d4cd33536e49b429dc7983121619a08"  # pinned dataset commit (for reproducibility)
# the sequence-classification tasks used for replication (4 binary + 2 multiclass)
TASKS = ["promoter_all", "promoter_tata", "promoter_no_tata",
         "enhancers", "enhancers_types", "splice_sites_all"]
NT_DATASETS = [f"nt_{t}" for t in TASKS]


def _read(task, split):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(REPO, f"{task}/{split}.parquet", repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(path, columns=["sequence", "label"])
    seqs = [s.strip().upper() for s in df["sequence"].tolist()]
    y = np.asarray(df["label"].tolist(), dtype=np.int64)
    return seqs, y


def load_nt_task(task, seed=42):
    """Return the standard dataset dict for one NT downstream task."""
    if task not in TASKS:
        raise ValueError(f"unknown NT task {task!r}; choose from {TASKS}")
    tr_seqs, y_tr = _read(task, "train")
    te_seqs, y_te = _read(task, "test")
    classes = sorted(set(y_tr.tolist()) | set(y_te.tolist()))   # already 0..n-1
    return dict(name=f"nt_{task}", base=REPO, classes=[str(c) for c in classes],
                n_classes=len(classes), train_seqs=tr_seqs, test_seqs=te_seqs,
                y_train=y_tr, y_test=y_te, split_source="provided")
