"""Race-safe, read-only data snapshots for the PSM-sensitivity module.

Why this exists: the installed ``genomic_benchmarks`` calls ``download_from_cloud_cache``
with ``force_download=True``, so EVERY ``data.load_dataset(<GB task>)`` call rmtree's
``~/.genomic_benchmarks/<task>`` and re-downloads it. With several agents on this machine
that is a read/delete race (a reader can silently get a partially-extracted directory).
This module therefore never calls ``download_dataset``: it reads the existing cache with the
IDENTICAL ordering rules as ``data._read_split_dir`` (sorted class dirs, sorted *.txt files,
``strip().upper()``, empty files skipped, provided train/test split), verifies the exact
per-split/per-class counts, retries if the directory is mid-rewrite, and freezes the result
to ``external/psm_sensitivity/snap_<task>.csv.gz`` (columns split, label, file_id, sequence).
All later runs read the frozen snapshot. The file_id (GB file stem, e.g. ``train_negative_17``)
is what lets us join GRCh38 coordinates for cohn (localization/cohn/*.csv.gz).

NT tasks come from the pinned HF parquet via ``data_nt`` (hf_hub cache; no rmtree race).

UPDATE (after data.py was patched with a file lock + completeness check): ``load_task`` now
takes the sequences from ``data.load_dataset`` (the sanctioned path) and uses the snapshot only
for file ids, after asserting element-wise identity (see load_task).
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import _common as C  # noqa: E402

SNAP_DIR = os.path.join(C.ROOT, "external", "psm_sensitivity")
GB_CACHE = os.path.expanduser("~/.genomic_benchmarks")

# exact expected sizes (train_neg, train_pos, test_neg, test_pos); verified against the
# paper's results (composition_clean.csv n_total / n_test) before first use.
EXPECTED = {
    "human_enhancers_cohn": (10422, 10421, 3474, 3474),
    "drosophila_enhancers_stark": (2592, 2592, 865, 865),
    "nt_enhancers": (7484, 7484, 200, 200),
}


def _read_gb_dir(task):
    base = os.path.join(GB_CACHE, task)
    rows = []
    for split in ("train", "test"):
        sd = os.path.join(base, split)
        classes = sorted(c for c in os.listdir(sd) if os.path.isdir(os.path.join(sd, c)))
        assert classes == ["negative", "positive"], f"{task}/{split}: classes {classes}"
        for lab, cls in enumerate(classes):
            for fp in sorted(glob.glob(os.path.join(sd, cls, "*.txt"))):
                with open(fp) as fh:
                    s = fh.read().strip().upper()
                if s:
                    rows.append((split, lab, os.path.basename(fp)[:-4], s))
    return pd.DataFrame(rows, columns=["split", "label", "file_id", "sequence"])


def _counts(df):
    g = df.groupby(["split", "label"]).size()
    return (int(g.get(("train", 0), 0)), int(g.get(("train", 1), 0)),
            int(g.get(("test", 0), 0)), int(g.get(("test", 1), 0)))


def snapshot(task, retries=10, wait=15):
    """Return the frozen snapshot DataFrame for `task`, creating it on first use."""
    os.makedirs(SNAP_DIR, exist_ok=True)
    path = os.path.join(SNAP_DIR, f"snap_{task}.csv.gz")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"split": str, "label": np.int64, "file_id": str, "sequence": str})
        assert _counts(df) == EXPECTED[task], f"corrupt snapshot {path}: {_counts(df)}"
        return df
    if task.startswith("nt_"):
        tr, ytr, te, yte = C.load_original(task)
        df = pd.DataFrame({"split": ["train"] * len(tr) + ["test"] * len(te),
                           "label": np.concatenate([ytr, yte]).astype(np.int64),
                           "file_id": [f"train_{i}" for i in range(len(tr))] + [f"test_{i}" for i in range(len(te))],
                           "sequence": list(tr) + list(te)})
    else:
        df = None
        for attempt in range(retries):
            try:
                cand = _read_gb_dir(task)
                if _counts(cand) == EXPECTED[task]:
                    df = cand
                    break
                print(f"[psm_data] {task}: counts {_counts(cand)} != expected (dir mid-rewrite?); retry")
            except (FileNotFoundError, AssertionError, OSError) as e:
                print(f"[psm_data] {task}: read failed ({e}); retry")
            time.sleep(wait)
        if df is None:
            raise RuntimeError(f"could not obtain a complete {task} directory after {retries} tries")
    assert _counts(df) == EXPECTED[task], f"{task}: {_counts(df)} != {EXPECTED[task]}"
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return df


def _results_counts(task):
    """(n_train, n_test) of the paper's benchmark row (model=lgbm, selected_by_val=1)."""
    f = "results_nt.csv" if task.startswith("nt_") else "results.csv"
    r = pd.read_csv(os.path.join(C.ROOT, "results", f))
    r = r[(r["dataset"] == task) & (r["model"] == "lgbm") & (r["selected_by_val"] == 1)].iloc[0]
    return int(r["n_train"]), int(r["n_test"])


def load_task(task, check_log=None):
    """(tr_seqs, ytr, te_seqs, yte, tr_ids, te_ids).

    The sequences/labels returned are the ones from the SANCTIONED loader
    (_common.load_original -> data.load_dataset, lock-protected + count-checked). The frozen
    snapshot is used only to attach file ids (GB file stems, for the GRCh38 coordinate join),
    and only after asserting it is element-wise IDENTICAL (same order, sequences and labels) to
    the loader's output and that the counts equal results/results{,_nt}.csv n_train/n_test."""
    tr, ytr, te, yte = C.load_original(task)
    ytr, yte = np.asarray(ytr, np.int64), np.asarray(yte, np.int64)
    ntr, nte = _results_counts(task)
    assert (len(tr), len(te)) == (ntr, nte), f"{task}: loader {len(tr)}/{len(te)} != results.csv {ntr}/{nte}"
    df = snapshot(task)
    s_tr = df[df["split"] == "train"]
    s_te = df[df["split"] == "test"]
    same = (s_tr["sequence"].tolist() == list(tr) and s_te["sequence"].tolist() == list(te)
            and np.array_equal(s_tr["label"].to_numpy(np.int64), ytr)
            and np.array_equal(s_te["label"].to_numpy(np.int64), yte))
    assert same, f"{task}: snapshot differs from data.load_dataset output -- delete {SNAP_DIR}/snap_{task}.csv.gz"
    if check_log is not None:
        import hashlib
        h = hashlib.sha256()
        for s, y in zip(list(tr) + list(te), np.r_[ytr, yte]):
            h.update(f"{s}\t{int(y)}\n".encode())
        check_log.append(dict(dataset=task, n_train=len(tr), n_test=len(te), results_csv_n_train=ntr,
                              results_csv_n_test=nte, n_train_pos=int(ytr.sum()), n_test_pos=int(yte.sum()),
                              snapshot_identical_to_loader=True, sha256_seq_label=h.hexdigest()))
    return (list(tr), ytr, list(te), yte, s_tr["file_id"].tolist(), s_te["file_id"].tolist())


if __name__ == "__main__":
    log = []
    for t in EXPECTED:
        a = load_task(t, check_log=log)
        print(t, "train", len(a[0]), "test", len(a[2]), "pos-rate", a[1].mean().round(3), a[3].mean().round(3))
    out = os.path.join(C.RESULTS_DIR, "psm_sensitivity_data_check.csv")
    pd.DataFrame(log).to_csv(out, index=False)
    print("wrote", out)
