"""Load Genomic Benchmarks datasets as raw sequences + integer labels.

Uses ``genomic_benchmarks.loc2seq.download_dataset`` (already-downloaded data is
reused from ``~/.genomic_benchmarks``; the first ever call downloads it). The
on-disk layout is ``<base>/<split>/<class>/<id>.txt``, one sequence per file.

Robustness (per the spec, the loader may return empty/odd splits):
* Missing ``train`` or ``test`` directories are skipped, not fatal.
* Empty sequence files are dropped.
* Class labels are the sorted union of class folder names across both splits, so
  integer labels are stable and multiclass datasets (e.g.
  ``human_ensembl_regulatory`` with 3 classes) work transparently.
* If the provided ``test`` split is missing/empty, or a class is absent from it,
  a deterministic stratified hold-out (20%, driven by ``seed``) is carved out of
  the available data instead, and ``split_source`` records that this happened.
"""
import fcntl
import glob
import os
import time

import numpy as np

# Expected (train, test) sequence counts per Genomic Benchmarks task (version 0), as
# used for every result in the paper (Table I). genomic_benchmarks' cloud-cache path
# calls download_from_cloud_cache(force_download=True), i.e. it DELETES and re-downloads
# ~/.genomic_benchmarks/<task> on every download_dataset() call, so concurrent processes
# can read a half-unzipped directory without any error. We therefore only download when
# the local copy is incomplete, under an exclusive file lock, and verify counts on read.
GB_EXPECTED = {
    "demo_human_or_worm": (75000, 25000),
    "human_ensembl_regulatory": (231348, 57713),
    "human_nontata_promoters": (27097, 9034),
    "demo_coding_vs_intergenomic_seqs": (75000, 25000),
    "human_enhancers_ensembl": (123872, 30970),
    "dummy_mouse_enhancers_ensembl": (968, 242),
    "human_enhancers_cohn": (20843, 6948),
    "human_ocr_ensembl": (139804, 34952),
    "drosophila_enhancers_stark": (5184, 1730),
}
GB_ROOT = os.path.join(os.path.expanduser("~"), ".genomic_benchmarks")


def _n_files(base, split):
    return len(glob.glob(os.path.join(base, split, "*", "*.txt")))


def _gb_complete(name, base):
    exp = GB_EXPECTED.get(name)
    if exp is None:
        return os.path.isdir(base)
    return _n_files(base, "train") >= exp[0] and _n_files(base, "test") >= exp[1]


def _gb_base(name):
    """Path to a complete local copy of a GB task; downloads once, under a lock."""
    base = os.path.join(GB_ROOT, name)
    if _gb_complete(name, base):
        return base
    os.makedirs(GB_ROOT, exist_ok=True)
    with open(os.path.join(GB_ROOT, f".{name}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not _gb_complete(name, base):
                from genomic_benchmarks.loc2seq import download_dataset
                download_dataset(name, version=0)
            if not _gb_complete(name, base):
                raise RuntimeError(f"{name}: download incomplete under {base}")
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return base


def _read_split_dir(base, split):
    """Return (seqs, class_names) for one split dir; ([], []) if it is absent."""
    sd = os.path.join(base, split)
    seqs, labs = [], []
    if not os.path.isdir(sd):
        return seqs, labs
    for cls in sorted(os.listdir(sd)):
        cd = os.path.join(sd, cls)
        if not os.path.isdir(cd):
            continue
        for fp in sorted(glob.glob(os.path.join(cd, "*.txt"))):
            with open(fp) as fh:
                s = fh.read().strip().upper()
            if s:
                seqs.append(s)
                labs.append(cls)
    return seqs, labs


def _stratified_split(y, test_frac, seed):
    """Deterministic per-class hold-out; returns (train_idx, test_idx)."""
    rng = np.random.RandomState(seed)
    test_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(test_frac * len(idx))))
        test_mask[idx[:n]] = True
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def load_dataset(name, seed=42, test_frac=0.2, min_test_per_class=1):
    """Load one dataset. Returns a dict with sequences, int labels and metadata.

    Dispatches on the name: an ``nt_`` prefix routes to the Nucleotide Transformer
    downstream-task loader (second suite); otherwise the Genomic Benchmarks loader.
    """
    if name.startswith("nt_"):
        import data_nt
        return data_nt.load_nt_task(name[3:], seed=seed)
    exp = GB_EXPECTED.get(name)
    for attempt in range(5):
        base = _gb_base(name)
        tr_seqs, tr_labs = _read_split_dir(base, "train")
        te_seqs, te_labs = _read_split_dir(base, "test")
        if exp is None or (len(tr_seqs), len(te_seqs)) == exp:
            break
        time.sleep(30)   # another process was rewriting the directory; retry
    else:
        raise RuntimeError(f"{name}: read {len(tr_seqs)}/{len(te_seqs)} train/test "
                           f"sequences, expected {exp[0]}/{exp[1]} (partial dataset)")

    classes = sorted(set(tr_labs) | set(te_labs))
    if not classes:
        raise ValueError(f"{name}: no sequences found under {base}")
    label_of = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([label_of[c] for c in tr_labs], dtype=np.int64)
    y_te = np.array([label_of[c] for c in te_labs], dtype=np.int64)

    n_classes = len(classes)

    def _test_usable():
        if len(y_te) < max(2, n_classes):
            return False
        return all((y_te == c).sum() >= min_test_per_class for c in range(n_classes))

    split_source = "provided"
    if not _test_usable():
        # Synthesize a clean, reproducible test split from all available data.
        split_source = f"stratified_holdout(test_frac={test_frac}, seed={seed})"
        all_seqs = tr_seqs + te_seqs
        y_all = np.concatenate([y_tr, y_te]) if len(y_te) else y_tr
        tr_idx, te_idx = _stratified_split(y_all, test_frac, seed)
        tr_seqs = [all_seqs[i] for i in tr_idx]
        te_seqs = [all_seqs[i] for i in te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    return dict(name=name, base=base, classes=classes, n_classes=n_classes,
                train_seqs=tr_seqs, test_seqs=te_seqs,
                y_train=y_tr, y_test=y_te, split_source=split_source)
