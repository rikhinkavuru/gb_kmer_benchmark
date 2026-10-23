"""Shared helpers for the multivar module (torch-free; never imports lightgbm).

* TASKS: the 15 benchmark tasks (9 Genomic Benchmarks + 6 Nucleotide Transformer).
* load_task_cached(): the same (train, y_train, test, y_test) that C.load_original() returns, with a
  pickle cache under external/multivar/seqcache/.
  WHY NOT C.load_original FOR GENOMIC-BENCHMARKS TASKS: data.load_dataset -> genomic_benchmarks
  download_dataset -> download_from_cloud_cache(force_download=True) rmtree()s and re-extracts
  ~/.genomic_benchmarks/<task> on EVERY call. With several agents loading concurrently, a reader can
  see a half-extracted directory and silently get a partial dataset (observed 2026-09-23:
  human_ensembl_regulatory came back with 100,651 instead of 231,348 training sequences). We
  therefore read GB tasks from a private copy of the SAME Zenodo cloud-cache zip
  (external/multivar/gb_zips/<task>_v0.zip, record 16605299) and replicate data._read_split_dir
  exactly: classes = sorted class folders; files in sorted path order; strip().upper(); empty files
  dropped; labels = index in sorted(union of class names). All nine GB tasks have a provided test
  split, so data.load_dataset's stratified-holdout fallback never applies (asserted below).
  NT tasks go through C.load_original (HF hub cache, read-only; not affected by the race).
* gc_fraction(): per-sequence GC fraction over A/C/G/T positions -- same definition as
  _common.gc_content (non-ACGT ignored) but string-based, so it never materialises an (n x maxlen)
  array (tests/test_multivar.py checks the two agree to 1e-12).
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UPG = os.path.abspath(os.path.join(HERE, ".."))
if UPG not in sys.path:
    sys.path.insert(0, UPG)
import _common as C  # noqa: E402  (adds repo root to sys.path; torch-free)

ROOT = C.ROOT
RESULTS_DIR = C.RESULTS_DIR
EXT_DIR = os.path.join(ROOT, "external", "multivar")
SEQCACHE = os.path.join(EXT_DIR, "seqcache")

GB_TASKS = ["demo_human_or_worm", "human_ensembl_regulatory", "human_nontata_promoters",
            "demo_coding_vs_intergenomic_seqs", "human_enhancers_ensembl",
            "dummy_mouse_enhancers_ensembl", "human_enhancers_cohn", "human_ocr_ensembl",
            "drosophila_enhancers_stark"]
NT_TASKS = ["nt_promoter_all", "nt_promoter_tata", "nt_promoter_no_tata", "nt_enhancers",
            "nt_enhancers_types", "nt_splice_sites_all"]
TASKS = GB_TASKS + NT_TASKS


GB_ZIP_DIR = os.path.join(EXT_DIR, "gb_zips")
GB_ZIP_URL = "https://zenodo.org/records/16605299/files/{task}_v0.zip?download=1"


def _read_gb_zip(task):
    """Replicates data.load_dataset for a GB task from the private zip snapshot (see module doc)."""
    import zipfile
    fp = os.path.join(GB_ZIP_DIR, f"{task}_v0.zip")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} missing -- download it: curl -L -o {fp} "
                                f"'{GB_ZIP_URL.format(task=task)}'")
    with zipfile.ZipFile(fp) as zf:
        split = {"train": {}, "test": {}}
        for n in zf.namelist():
            parts = n.split("/")
            # layout: <task>/<split>/<class>/<id>.txt
            if len(parts) != 4 or parts[0] != task or parts[1] not in split or not n.endswith(".txt"):
                continue
            split[parts[1]].setdefault(parts[2], []).append(n)
        out = {}
        for sp in ("train", "test"):
            seqs, labs = [], []
            for cls in sorted(split[sp]):
                for n in sorted(split[sp][cls]):          # == sorted(glob) within one class dir
                    s = zf.read(n).decode("ascii", "replace").strip().upper()
                    if s:
                        seqs.append(s)
                        labs.append(cls)
            out[sp] = (seqs, labs)
    classes = sorted(set(out["train"][1]) | set(out["test"][1]))
    lab = {c: i for i, c in enumerate(classes)}
    ytr = np.array([lab[c] for c in out["train"][1]], np.int64)
    yte = np.array([lab[c] for c in out["test"][1]], np.int64)
    # data.load_dataset would fall back to a stratified hold-out only if a class were missing
    assert all((yte == c).sum() >= 1 for c in range(len(classes))), f"{task}: unusable test split"
    return out["train"][0], ytr, out["test"][0], yte


def expected_counts():
    """{task: (n_train, n_test)} from the Table-1 source (results.csv / results_nt.csv)."""
    import pandas as pd
    r = pd.concat([pd.read_csv(os.path.join(ROOT, "results", f)) for f in ("results.csv", "results_nt.csv")])
    r = r[(r["model"] == "lgbm") & (r["selected_by_val"] == 1)]
    return {t: (int(a), int(b)) for t, a, b in zip(r["dataset"], r["n_train"], r["n_test"])}


def _assert_counts(task, out):
    exp = expected_counts()[task]
    got = (len(out[0]), len(out[2]))
    if got != exp or len(out[1]) != got[0] or len(out[3]) != got[1]:
        raise RuntimeError(f"{task}: loaded n_train/n_test {got} != Table-1 {exp} -- refusing a partial dataset")


def load_task_cached(task, seed=C.SEED):
    """(tr_seqs, ytr, te_seqs, yte) as C.load_original(task, seed) returns them, pickle-cached.
    Train/test counts are asserted against results.csv / results_nt.csv on EVERY load (cache hit or
    miss), per the data-integrity policy for private copies."""
    os.makedirs(SEQCACHE, exist_ok=True)
    fp = os.path.join(SEQCACHE, f"{task}__seed{seed}.pkl")
    if os.path.exists(fp):
        with open(fp, "rb") as fh:
            out = pickle.load(fh)
        _assert_counts(task, out)
        return out
    if task.startswith("nt_"):
        out = C.load_original(task, seed=seed)
    else:
        out = _read_gb_zip(task)
    out = (list(out[0]), np.asarray(out[1], np.int64), list(out[2]), np.asarray(out[3], np.int64))
    _assert_counts(task, out)
    tmp = fp + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, fp)
    return out


# ------------------------------------------------------------------ memory guard (machine sharing)
# Copied (not imported) from src/upgrades/fm_finetune/memguard.py so this module never depends on
# another lane's file.  Watches phys_footprint (what Activity Monitor shows; includes compressed and
# swapped pages, unlike RSS) and hard-exits the process above the cap, so a runaway allocation can
# never push the shared 16 GB laptop into swap (cf. the 17.6 GB PID-62701 incident, 2026-09-24).
MEM_CAP_MB = float(os.environ.get("MULTIVAR_MEM_CAP_MB", "2800"))
_libc = None


def footprint_mb(pid=None):
    """phys_footprint of `pid` (default: this process) in MB (1e6 bytes); nan if unreadable."""
    global _libc
    import ctypes
    pid = os.getpid() if pid is None else pid
    if sys.platform != "darwin":
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024 / 1e6
        except OSError:
            pass
        return float("nan")
    if _libc is None:
        _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    buf = ctypes.create_string_buffer(512)
    if _libc.proc_pid_rusage(ctypes.c_int(int(pid)), ctypes.c_int(2), buf) != 0:   # RUSAGE_INFO_V2
        return float("nan")
    return int.from_bytes(buf.raw[72:80], "little") / 1e6                           # ri_phys_footprint


_PEAK = [0.0]


def start_memguard(cap_mb=None, interval=3.0, tag="multivar"):
    """Daemon thread: os._exit(86) if this process's footprint exceeds cap_mb (outputs are always
    written .tmp -> rename, so an abort never leaves a half-written file). Returns a peak() getter."""
    import threading
    import time
    cap = MEM_CAP_MB if cap_mb is None else float(cap_mb)

    def run():
        while True:
            fp = footprint_mb()
            if fp == fp:
                _PEAK[0] = max(_PEAK[0], fp)
                if fp > cap:
                    print(f"!!! [{tag}] MEMGUARD: footprint {fp:.0f} MB > cap {cap:.0f} MB -- aborting "
                          f"(exit 86) before the shared machine swaps", flush=True)
                    os._exit(86)
            time.sleep(interval)

    threading.Thread(target=run, name="memguard", daemon=True).start()
    return lambda: _PEAK[0]


def gc_fraction(seqs):
    """Per-sequence GC fraction over A/C/G/T positions (0 for a sequence with no ACGT)."""
    out = np.empty(len(seqs), dtype=np.float64)
    for i, s in enumerate(seqs):
        g = s.count("G") + s.count("C")
        v = g + s.count("A") + s.count("T")
        out[i] = g / v if v > 0 else 0.0
    return out


def pooled_within_sd(x, y):
    """Within-class pooled SD: sqrt(sum_c (n_c - 1) var_c / (N - K))."""
    classes = np.unique(y)
    num = sum((np.sum(y == c) - 1) * np.var(x[y == c], ddof=1) for c in classes)
    den = len(x) - len(classes)
    return float(np.sqrt(num / den))


def max_pairwise_cohens_d(x, y):
    """|Cohen's d| between classes (pooled SD of the pair); max over class pairs if >2 classes."""
    classes = np.unique(y)
    best = 0.0
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = x[y == classes[i]], x[y == classes[j]]
            sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                         / (len(a) + len(b) - 2))
            d = abs(a.mean() - b.mean()) / sp if sp > 0 else 0.0
            best = max(best, float(d))
    return best
