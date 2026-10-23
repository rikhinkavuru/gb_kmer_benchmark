"""Shared helpers for the homology re-clustering module (Reviewer 1, comment 3).

Torch-free. Everything here either re-uses the repo's shared code by IMPORT (featurize, models,
_common) or is a local copy; no shared file is modified.

Why a private Genomic Benchmarks loader?
  `data.load_dataset()` calls genomic_benchmarks' `download_from_cloud_cache`, whose default
  `force_download=True` DELETES and re-downloads ~/.genomic_benchmarks/<task> on every call. With
  several agents loading concurrently this silently yields partial datasets (data.py skips missing
  files without error; observed 2026-09-23 on human_ensembl_regulatory). We therefore fetch the SAME
  Zenodo archive (record 16605299, dataset version 0) once into
  external/homology_recluster/gb/<task>/ and parse it with the repo's own `data._read_split_dir`
  and the identical label mapping, so sequences, labels and ORDER are byte-identical to
  data.load_dataset() on an uncorrupted cache. Split sizes are then asserted against the
  n_train/n_test recorded in results/results.csv (the Table-1 source), so a partial load cannot
  pass silently.
NT tasks use data_nt (HuggingFace hub cache at the pinned revision; no deletion race).
"""
import os
import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, ".."))
for _d in (UP, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import _common as C          # noqa: E402  (adds repo root to sys.path; torch-free)
import data as gbdata        # noqa: E402
import featurize             # noqa: E402

ROOT = C.ROOT
SEED = C.SEED
BOOT = C.BOOT
# machine-sharing limit: HC_THREADS if set, else the bibm/lowcpu.sh thread budget (LOWCPU_THREADS, 2 since the
# 2026-09-24 13:10 CPU policy), else 3 (the value used for every unit computed before that change). Only units
# that are not cached yet are affected; each run_extra experiment compares before/after at one thread count.
N_THREADS = int(os.environ.get("HC_THREADS", os.environ.get("LOWCPU_THREADS", "3")))
# The shared model factory (models.py) hard-codes n_jobs=-1. Every helper we REUSE by import
# (_common.fit_eval_boot, run_composition_clean, positional_sweep, diagnose) builds its model
# through that factory, so we cap threads for THIS PROCESS ONLY by overriding the parameter dicts
# at import time. No shared file is modified; LightGBM stays deterministic (deterministic=True,
# force_row_wise=True) and every arm in this module runs with the same thread count.
C.gbmodels.LGBM_PARAMS["n_jobs"] = N_THREADS
C.gbmodels.LR_PARAMS["n_jobs"] = 1
EXT = os.path.join(ROOT, "external", "homology_recluster")
GB_DIR = os.path.join(EXT, "gb")
CLU_DIR = os.path.join(EXT, "clusters")
CKPT_DIR = os.path.join(EXT, "ckpt")
RES = C.RESULTS_DIR
FIG = os.path.join(RES, "figures")
PREFIX = "homology_recluster"

ZENODO = "https://zenodo.org/records/16605299"

TASKS = [  # Table-1 order (MCC descending)
    "demo_human_or_worm", "human_ensembl_regulatory", "nt_promoter_all", "nt_promoter_no_tata",
    "human_nontata_promoters", "nt_promoter_tata", "demo_coding_vs_intergenomic_seqs",
    "human_enhancers_ensembl", "dummy_mouse_enhancers_ensembl", "nt_enhancers",
    "nt_enhancers_types", "human_enhancers_cohn", "human_ocr_ensembl",
    "drosophila_enhancers_stark", "nt_splice_sites_all",
]
# published route label per task (working-tree paper_psb.tex Table tab:land, v2 labels; '' = solved)
PAPER_ROUTE = {
    "demo_human_or_worm": "solved", "human_ensembl_regulatory": "solved",
    "nt_promoter_all": "solved", "nt_promoter_no_tata": "solved",
    "human_nontata_promoters": "solved", "nt_promoter_tata": "solved",
    "demo_coding_vs_intergenomic_seqs": "solved", "human_enhancers_ensembl": "shared-motif",
    "dummy_mouse_enhancers_ensembl": "shared-motif", "nt_enhancers": "contamination",
    "nt_enhancers_types": "shared-motif", "human_enhancers_cohn": "contamination",
    "human_ocr_ensembl": "shared-motif", "drosophila_enhancers_stark": "shared-motif",
    "nt_splice_sites_all": "positional",
}
# Route-1 TF-motif tasks (cross_suite_summary.csv tf_motif_task == True; n = 11)
NON_TF = {"nt_splice_sites_all", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm",
          "dummy_mouse_enhancers_ensembl"}


# ------------------------------------------------------------------ loading
def _expected_sizes():
    out = {}
    for f in ("results.csv", "results_nt.csv"):
        d = pd.read_csv(os.path.join(ROOT, "results", f))
        d = d[(d["model"] == "lgbm") & (d["selected_by_val"] == 1)]
        for _, r in d.iterrows():
            out[r["dataset"]] = (int(r["n_train"]), int(r["n_test"]))
    return out


def best_k(task):
    vs = pd.read_csv(os.path.join(ROOT, "results", "val_selection.csv")).set_index("dataset")
    return int(vs.loc[task, "new_k"])


def table1(task):
    """Published Table-1 MCC/AUROC (+CI) for the task (results/val_selection.csv new_* columns)."""
    vs = pd.read_csv(os.path.join(ROOT, "results", "val_selection.csv")).set_index("dataset")
    r = vs.loc[task]
    return dict(mcc=float(r["new_test_mcc"]), mcc_lo=float(r["new_mcc_lo"]), mcc_hi=float(r["new_mcc_hi"]),
                auroc=float(r["new_auroc"]), auroc_lo=float(r["new_auroc_lo"]),
                auroc_hi=float(r["new_auroc_hi"]))


def _fetch_gb_zip(task):
    """Download the Zenodo archive ONCE into our private dir (atomic rename); never extracted --
    extracting ~10^5 one-sequence files is very slow on the shared disk, so we read the zip."""
    os.makedirs(GB_DIR, exist_ok=True)
    z = os.path.join(GB_DIR, f"{task}.zip")
    if os.path.exists(z):
        return z
    url = f"{ZENODO}/files/{task}_v0.zip?download=1"
    tmp = z + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, z)
    return z


def _read_split_zip(zf, task, split):
    """Zip analogue of data._read_split_dir: classes in sorted order, files in sorted order within
    a class, content .strip().upper(), empty files dropped -> identical sequences and ORDER."""
    by_cls = {}
    pre = f"{task}/{split}/"
    for name in zf.namelist():
        if not name.startswith(pre) or not name.endswith(".txt"):
            continue
        rest = name[len(pre):]
        if "/" not in rest:
            continue
        cls, fn = rest.split("/", 1)
        if "/" in fn:
            continue
        by_cls.setdefault(cls, []).append(name)
    seqs, labs = [], []
    for cls in sorted(by_cls):
        for name in sorted(by_cls[cls]):
            s = zf.read(name).decode().strip().upper()
            if s:
                seqs.append(s); labs.append(cls)
    return seqs, labs


def load_task(task):
    """(tr_seqs, ytr, te_seqs, yte) identical to C.load_original(task) on an intact cache, with
    split sizes asserted against the Table-1 record. GB tasks are parsed from a private copy of the
    same Zenodo archive; a pooled-sequence cache (npz) makes repeat loads fast."""
    cache = os.path.join(GB_DIR, f"{task}.seqs.npz")
    if task.startswith("nt_"):
        tr, ytr, te, yte = C.load_original(task)
    elif os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        tr = z["tr"].tobytes().decode().split("\n")
        te = z["te"].tobytes().decode().split("\n")
        ytr, yte = z["ytr"], z["yte"]
    else:
        import zipfile
        with zipfile.ZipFile(_fetch_gb_zip(task)) as zf:
            tr, trl = _read_split_zip(zf, task, "train")
            te, tel = _read_split_zip(zf, task, "test")
        classes = sorted(set(trl) | set(tel))
        lab = {c: i for i, c in enumerate(classes)}
        ytr = np.array([lab[c] for c in trl], dtype=np.int64)
        yte = np.array([lab[c] for c in tel], dtype=np.int64)
        exp = _expected_sizes()[task]
        assert (len(tr), len(te)) == exp, f"{task}: loaded {(len(tr), len(te))} != Table-1 {exp}"
        enc = lambda L: np.frombuffer("\n".join(L).encode(), dtype=np.uint8)
        np.savez(cache + ".tmp.npz", tr=enc(tr), te=enc(te), ytr=ytr, yte=yte)
        os.replace(cache + ".tmp.npz", cache)
    exp = _expected_sizes()[task]
    assert (len(tr), len(te)) == exp, f"{task}: loaded {(len(tr), len(te))} != Table-1 {exp}"
    return list(tr), np.asarray(ytr, np.int64), list(te), np.asarray(yte, np.int64)


# ------------------------------------------------------------------ models
def build_lgbm(n_classes, seed=SEED):
    """The paper's fixed LightGBM config (models.LGBM_PARAMS), capped at N_THREADS threads."""
    m = C.gbmodels.build_model("lgbm", seed, n_classes)
    m.set_params(n_jobs=N_THREADS)
    return m


def build_lr(n_classes, seed=SEED):
    m = C.gbmodels.build_model("lr", seed, n_classes)
    try:
        m.set_params(logisticregression__n_jobs=1)
    except ValueError:
        pass
    return m


def fit_proba(Xtr, ytr, Xte, model="lgbm", seed=SEED):
    nc = len(np.unique(ytr))
    m = build_lgbm(nc, seed) if model == "lgbm" else build_lr(nc, seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte), m.classes_


def auroc(y, proba):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    nc = proba.shape[1]
    try:
        if nc == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        present = np.unique(y)
        if len(present) < nc:
            return float("nan")
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def boot_metrics(y, pred, proba, seed=SEED, B=BOOT):
    """Point MCC/AUROC + percentile bootstrap CIs (test sequence = resampling unit), same scheme as
    _common.fit_eval_boot / homology/run_purged_eval.boot_metrics."""
    from sklearn.metrics import matthews_corrcoef
    rng = np.random.RandomState(seed)
    y = np.asarray(y); pred = np.asarray(pred)
    n = len(y)
    mm = np.full(B, np.nan); aa = np.full(B, np.nan)
    for b in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        mm[b] = matthews_corrcoef(y[idx], pred[idx])
        aa[b] = auroc(y[idx], proba[idx])
    q = lambda a, p: float(np.nanpercentile(a, p)) if np.isfinite(a).any() else float("nan")
    return dict(mcc=float(matthews_corrcoef(y, pred)), mcc_lo=q(mm, 2.5), mcc_hi=q(mm, 97.5),
                auroc=auroc(y, proba), auroc_lo=q(aa, 2.5), auroc_hi=q(aa, 97.5), n_test=int(n),
                pos_frac=float(np.mean(y == 1)) if len(np.unique(y)) <= 2 else float("nan"))


# ------------------------------------------------------------------ clusters
def load_clusters(task, tag):
    """Per-sequence component ids over the POOLED order (train then test) for a threshold tag
    ('id80' or 'id50'), written by run_cluster.py."""
    z = np.load(os.path.join(CLU_DIR, f"{task}_{tag}.npz"))
    return z["comp"], z["is_test"]


def cluster_resplit(comp, y, test_frac, seed=SEED):
    """Cluster-aware, label-stratified re-split (pre-registered in run_eval.py).

    Whole connected components are assigned to train or test. Components are visited in a seeded
    random order; a component joins TEST iff doing so reduces the L1 distance between the current
    per-label test counts and the per-label targets round(test_frac * n_label). Everything else goes
    to TRAIN. Deterministic given seed. Returns a boolean is_test mask over the pooled sequences."""
    y = np.asarray(y)
    labels = np.unique(y)
    lab_idx = {l: i for i, l in enumerate(labels)}
    target = np.array([round(test_frac * np.sum(y == l)) for l in labels], dtype=float)
    uc, inv = np.unique(comp, return_inverse=True)
    counts = np.zeros((len(uc), len(labels)))
    np.add.at(counts, (inv, np.array([lab_idx[v] for v in y])), 1)
    order = np.random.RandomState(seed).permutation(len(uc))
    cur = np.zeros(len(labels))
    in_test = np.zeros(len(uc), bool)
    for c in order:
        new = cur + counts[c]
        if np.abs(new - target).sum() < np.abs(cur - target).sum():
            in_test[c] = True
            cur = new
    return in_test[inv]


def random_resplit(y, test_frac, seed=SEED):
    """Label-stratified RANDOM (not cluster-aware) re-split of the pooled data with the same
    per-label test targets -- the control that isolates homology from 'any re-split'."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    is_test = np.zeros(len(y), bool)
    for l in np.unique(y):
        idx = np.where(y == l)[0]
        rng.shuffle(idx)
        is_test[idx[:int(round(test_frac * len(idx)))]] = True
    return is_test


def nr_representatives(comp, y, seed=SEED):
    """Non-redundant subset: ONE sequence per (component, label) pair, chosen at random (seeded).
    Removes within-class near-duplicate families (the pseudo-replication that could bias the
    pooled within-dataset diagnostics) while keeping cross-label near-duplicates visible."""
    rng = np.random.RandomState(seed)
    key = comp.astype(np.int64) * 16 + np.asarray(y, np.int64)
    order = rng.permutation(len(key))
    _, first = np.unique(key[order], return_index=True)
    return np.sort(order[first])


def hc_data(task, tag, nr=True, seed=SEED):
    """HOMOLOGY-CLEAN data for the intervention / route analyses (pre-registered definition).

    The cluster-aware re-split of run_eval.py (whole components on one side, label-stratified, original
    test fraction, `seed`), then -- if `nr` -- NON-REDUNDANT de-duplication: one representative per
    (component, label), chosen at random (seeded). Components never straddle the split, so the
    representatives of a (component, label) pair all live in one split and the NR step is simultaneously
    a within-split dedupe. Returns dict(tr, ytr, te, yte, comp_tr, comp_te, n_pooled, n_nr)."""
    tr, ytr, te, yte = load_task(task)
    comp, _ = load_clusters(task, tag)
    seqs = tr + te
    y = np.concatenate([ytr, yte])
    test_frac = len(te) / len(seqs)
    is_test = cluster_resplit(comp, y, test_frac, seed)
    keep = np.ones(len(seqs), bool)
    if nr:
        keep[:] = False
        keep[nr_representatives(comp, y, seed)] = True
    itr = np.where(~is_test & keep)[0]
    ite = np.where(is_test & keep)[0]
    return dict(tr=[seqs[i] for i in itr], ytr=y[itr], te=[seqs[i] for i in ite], yte=y[ite],
                comp_tr=comp[itr], comp_te=comp[ite], n_pooled=len(seqs), n_nr=int(keep.sum()))


def original_data(task):
    """The paper's setting (provided split, all sequences) in the same dict layout as hc_data."""
    tr, ytr, te, yte = load_task(task)
    return dict(tr=tr, ytr=ytr, te=te, yte=yte, comp_tr=None, comp_te=None,
                n_pooled=len(tr) + len(te), n_nr=len(tr) + len(te))
