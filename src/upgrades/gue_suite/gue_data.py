"""GUE (Genome Understanding Evaluation; DNABERT-2, Zhou et al., ICLR 2024) loader + reference tables.

Data source (official): the Google Drive archive linked from the DNABERT-2 README
(https://github.com/MAGICS-LAB/DNABERT_2 -> https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-),
downloaded with gdown to external/gue/GUE.zip and unzipped to external/gue/GUE/.
SHA-256 of the archive we analysed is pinned below (GUE_ZIP_SHA256) and re-checked by the tests.

Each GUE dataset is <dir>/{train,dev,test}.csv with columns sequence,label. Label 1 is the positive
class for every binary task (promoter / TF-bound / histone-marked). Sequences are upper-cased; the
featurizer ignores any k-mer containing a non-ACGT symbol (IUPAC codes occur only in covid).

Reference tables (transcribed by hand from DNABERT-2 arXiv:2306.15006v2 = ICLR 2024 camera-ready,
full text saved at external/gue/refs/dnabert2_arxiv_2306.15006_fulltext.txt):
  * PAPER_SPLITS  -- Table 12 train/dev/test sizes.
  * PUBLISHED     -- Table 6 (all models, 3-seed means, x100) and Table 9 (HyenaDNA row, from the
                     DNABERT-2 authors' run of hyenadna-medium-450k). Metric = MCC x100 for every
                     dataset except covid (F1 x100). Table 9 re-prints a DNABERT-2 row whose TF-H /
                     TF-M values differ from Table 6; we use Table 6 for DNABERT-2 and record the
                     Table-9 DNABERT-2 values separately (PUBLISHED_T9_DNABERT2) so the inconsistency
                     is visible. The Table 9 CNN row is omitted: its extracted TF-M/virus/splice row
                     has 8 numbers for 7 columns, so it cannot be transcribed unambiguously.
"""
import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GUE_DIR = os.path.join(ROOT, "external", "gue", "GUE")
GUE_ZIP = os.path.join(ROOT, "external", "gue", "GUE.zip")
GUE_ZIP_SHA256 = "581ba5a69843769f4cbc34470e2fe970f8ee371280914bd66574af01ee60cd79"

# task_id -> (relative dir, species, task family, DNABERT-2 metric)
TASKS = {
    # ---- human
    "prom_core_all":    ("prom/prom_core_all",    "human", "core promoter (70 bp)", "mcc"),
    "prom_core_notata": ("prom/prom_core_notata", "human", "core promoter (70 bp)", "mcc"),
    "prom_core_tata":   ("prom/prom_core_tata",   "human", "core promoter (70 bp)", "mcc"),
    "prom_300_all":     ("prom/prom_300_all",     "human", "promoter (300 bp)", "mcc"),
    "prom_300_notata":  ("prom/prom_300_notata",  "human", "promoter (300 bp)", "mcc"),
    "prom_300_tata":    ("prom/prom_300_tata",    "human", "promoter (300 bp)", "mcc"),
    "human_tf_0":       ("tf/0", "human", "TF binding (101 bp)", "mcc"),
    "human_tf_1":       ("tf/1", "human", "TF binding (101 bp)", "mcc"),
    "human_tf_2":       ("tf/2", "human", "TF binding (101 bp)", "mcc"),
    "human_tf_3":       ("tf/3", "human", "TF binding (101 bp)", "mcc"),
    "human_tf_4":       ("tf/4", "human", "TF binding (101 bp)", "mcc"),
    "splice_reconstructed": ("splice/reconstructed", "human", "splice site (400 bp, 3-class)", "mcc"),
    # ---- mouse
    "mouse_tf_0": ("mouse/0", "mouse", "TF binding (101 bp)", "mcc"),
    "mouse_tf_1": ("mouse/1", "mouse", "TF binding (101 bp)", "mcc"),
    "mouse_tf_2": ("mouse/2", "mouse", "TF binding (101 bp)", "mcc"),
    "mouse_tf_3": ("mouse/3", "mouse", "TF binding (101 bp)", "mcc"),
    "mouse_tf_4": ("mouse/4", "mouse", "TF binding (101 bp)", "mcc"),
    # ---- yeast epigenetic marks
    "emp_H3":       ("EMP/H3",       "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H3K14ac":  ("EMP/H3K14ac",  "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H3K36me3": ("EMP/H3K36me3", "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H3K4me1":  ("EMP/H3K4me1",  "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H3K4me2":  ("EMP/H3K4me2",  "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H3K4me3":  ("EMP/H3K4me3",  "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H3K79me3": ("EMP/H3K79me3", "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H3K9ac":   ("EMP/H3K9ac",   "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H4":       ("EMP/H4",       "yeast", "epigenetic mark (500 bp)", "mcc"),
    "emp_H4ac":     ("EMP/H4ac",     "yeast", "epigenetic mark (500 bp)", "mcc"),
    # ---- virus
    "covid": ("virus/covid", "virus", "covid variant (1000 bp, 9-class)", "f1"),
}
# pre-registered processing order: human -> mouse -> yeast EMP -> covid (heaviest last)
ORDER = list(TASKS)

# GUE+ datasets that ship in the same archive but are OUT of scope (not part of the 28-dataset GUE):
SKIPPED = {
    "EPI/{GM12878,HUVEC,HeLa-S3,IMR90,K562,NHEK}": "GUE+ enhancer-promoter interaction: a sequence-PAIR "
        "task (two 3 kb inputs); the single-sequence k-mer pipeline and diagnose.py do not apply",
    "fungi/species_20": "GUE+ species classification (10 kb, 20-class): not part of the 28-dataset GUE",
    "virus/species_40": "GUE+ species classification (5 kb, 25-class): not part of the 28-dataset GUE",
}

# DNABERT-2 Table 12 (train, dev, test)
PAPER_SPLITS = {
    "prom_core_tata": (4904, 613, 613), "prom_core_notata": (42452, 5307, 5307),
    "prom_core_all": (47356, 5920, 5920),
    "prom_300_tata": (4904, 613, 613), "prom_300_notata": (42452, 5307, 5307),
    "prom_300_all": (47356, 5920, 5920),
    "human_tf_0": (32378, 1000, 1000), "human_tf_1": (30672, 1000, 1000),
    "human_tf_2": (19000, 1000, 1000), "human_tf_3": (27294, 1000, 1000),
    "human_tf_4": (19000, 1000, 1000),
    "splice_reconstructed": (36496, 4562, 4562),
    "mouse_tf_0": (6478, 810, 810), "mouse_tf_1": (53952, 6745, 6745),
    "mouse_tf_2": (2620, 328, 328), "mouse_tf_3": (1904, 239, 239),
    "mouse_tf_4": (15064, 1883, 1883),
    "emp_H3": (11971, 1497, 1497), "emp_H3K14ac": (26438, 3305, 3305),
    "emp_H3K36me3": (27904, 3488, 3488), "emp_H3K4me1": (25341, 3168, 3168),
    "emp_H3K4me2": (24545, 3069, 3069), "emp_H3K4me3": (29439, 3680, 3680),
    "emp_H3K79me3": (23069, 2884, 2884), "emp_H3K9ac": (22224, 2779, 2779),
    "emp_H4": (11679, 1461, 1461), "emp_H4ac": (27275, 3410, 3410),
    "covid": (77669, 7000, 7000),
}

# ENCODE identifiers given in Table 12 (for the README / table)
ENCODE_ID = {
    "human_tf_0": "wgEncodeEH000552", "human_tf_1": "wgEncodeEH000606",
    "human_tf_2": "wgEncodeEH001546", "human_tf_3": "wgEncodeEH001776",
    "human_tf_4": "wgEncodeEH002829",
    "mouse_tf_0": "Ch12Nrf2Iggrab", "mouse_tf_1": "Ch12Znf384hpa004051Iggrab",
    "mouse_tf_2": "MelJundIggrab", "mouse_tf_3": "MelMafkDm2p5dStd", "mouse_tf_4": "MelNelfeIggrab",
}

# Negative-class construction documented in DNABERT-2 App. B.2 (paraphrase, for interpretation; EMP from its source)
NEG_CONSTRUCTION = {
    "prom_300": "non-TATA: randomly substituted promoter sequences; TATA: random non-promoter genomic "
                "sequences carrying a TATA motif",
    "prom_core": "as prom_300, centre -34..+35 bp around the TSS",
    "human_tf": "non-overlapping genomic sequences of the same length and GC content (GC-matched)",
    "mouse_tf": "dinucleotide-shuffled positives (dinucleotide frequencies preserved)",
    "splice": "Wang et al. 2019 donor/acceptor/non-site, reconstructed with adversarial negatives",
    # [fix round 1] provenance corrected: DNABERT-2 App. B.2 only names the download site; the source is
    # Pham et al. 2005 (Genome Inform 16(2):3-11, PMID 16901084), whose abstract describes the classes.
    # [fix round 2] PH5 (emp_provenance.py): GUE label counts equal Pham's #positive/#negative on 10/10 marks;
    # the ten marks and the 'relative to H3' normalisation match Pokholok et al. 2005, which the Pham
    # abstract does not name (full text not accessed), so Pokholok is not stated as the source here.
    "emp": "Pham et al. 2005 datasets (jaist.ac.jp/~tran/nucleosome): yeast genomic windows (ORF and intergenic "
           "ChIP-chip regions) with high (label 1) vs low (label 0) relative histone occupancy/modification; "
           "no sampled negatives; GUE split 8:1:1",
    "covid": "9 SARS-CoV-2 variants (GISAID), 1 kb segments",
}


def neg_family(task):
    for fam in ("prom_300", "prom_core", "human_tf", "mouse_tf", "splice", "emp", "covid"):
        if task.startswith(fam):
            return fam
    return ""


# -------------------------------------------------------------- published numbers (x100)
_COLS = (["emp_H3", "emp_H3K14ac", "emp_H3K36me3", "emp_H3K4me1", "emp_H3K4me2", "emp_H3K4me3",
          "emp_H3K79me3", "emp_H3K9ac", "emp_H4", "emp_H4ac",
          "prom_300_all", "prom_300_notata", "prom_300_tata",
          "human_tf_0", "human_tf_1", "human_tf_2", "human_tf_3", "human_tf_4",
          "prom_core_all", "prom_core_notata", "prom_core_tata",
          "mouse_tf_0", "mouse_tf_1", "mouse_tf_2", "mouse_tf_3", "mouse_tf_4",
          "covid", "splice_reconstructed"])
_T6 = {  # DNABERT-2 Table 6, in _COLS order
    "DNABERT (3-mer)": [74.15, 42.07, 48.49, 42.95, 31.34, 28.92, 60.12, 50.48, 78.27, 38.60,
                        90.44, 93.61, 69.83, 67.95, 70.90, 60.51, 53.03, 69.76, 70.92, 69.82, 78.15,
                        42.31, 79.10, 69.90, 55.40, 41.97, 62.23, 84.14],
    "DNABERT (6-mer)": [73.10, 40.06, 47.25, 41.44, 32.27, 27.81, 61.17, 51.22, 79.26, 37.43,
                        90.48, 93.05, 61.56, 66.84, 70.14, 61.03, 51.89, 70.97, 68.90, 70.47, 76.06,
                        44.42, 78.94, 71.44, 44.89, 42.48, 55.50, 84.07],
    "NT-500M-human": [69.67, 33.55, 44.14, 37.15, 30.87, 24.06, 58.35, 45.81, 76.17, 33.74,
                      87.71, 90.75, 78.07, 61.59, 66.75, 53.58, 42.95, 60.81, 63.45, 64.82, 71.34,
                      31.04, 75.04, 61.67, 29.17, 29.27, 50.82, 79.71],
    "NT-2500M-multi": [78.77, 56.20, 61.99, 55.30, 36.49, 40.34, 64.70, 56.01, 81.67, 49.13,
                       91.01, 94.00, 79.43, 66.64, 70.28, 58.72, 51.65, 69.34, 70.33, 71.58, 72.97,
                       63.31, 83.76, 71.52, 69.44, 47.07, 73.04, 89.35],
    "DNABERT-2": [78.27, 52.57, 56.88, 50.52, 31.13, 36.27, 67.39, 55.63, 80.71, 50.43,
                  86.77, 94.27, 71.59, 71.99, 76.06, 66.52, 58.54, 77.43, 69.37, 68.04, 74.17,
                  56.76, 84.77, 79.32, 66.47, 52.66, 71.02, 84.99],
    "DNABERT-2 (further PT)": [80.17, 57.42, 61.90, 53.00, 39.89, 41.20, 65.46, 57.07, 81.86, 50.35,
                               88.31, 94.34, 68.79, 69.12, 71.87, 62.96, 55.35, 74.94, 67.50, 69.53,
                               76.18, 64.23, 86.28, 81.28, 73.49, 50.80, 68.49, 85.93],
}
_T9_HYENA = [67.17, 31.98, 48.27, 35.83, 25.81, 23.15, 54.09, 50.84, 73.69, 38.44,
             47.38, 52.24, 5.34, 62.3, 67.86, 46.85, 41.78, 61.23, 36.95, 35.38, 72.87,
             35.62, 80.50, 65.34, 54.20, 19.17, 23.27, 72.67]
_T9_DNABERT2 = [78.27, 52.57, 56.88, 50.52, 31.13, 36.27, 67.39, 55.63, 80.71, 50.43,
                86.77, 94.27, 71.59, 84.38, 87.16, 83.99, 79.85, 87.80, 69.47, 68.04, 74.17,
                81.60, 92.67, 92.38, 84.91, 76.10, 71.02, 84.99]

PUBLISHED = {m: dict(zip(_COLS, v)) for m, v in _T6.items()}
PUBLISHED["HyenaDNA (T9)"] = dict(zip(_COLS, _T9_HYENA))
PUBLISHED_T9_DNABERT2 = dict(zip(_COLS, _T9_DNABERT2))
for _m, _v in list(_T6.items()) + [("HyenaDNA", _T9_HYENA), ("T9 DNABERT-2", _T9_DNABERT2)]:
    assert len(_v) == len(_COLS), (_m, len(_v))


# ---------------------------------------------------------------------- loading
def task_dir(task):
    return os.path.join(GUE_DIR, TASKS[task][0])


def _read(task, split):
    import pandas as pd
    df = pd.read_csv(os.path.join(task_dir(task), f"{split}.csv"))
    seqs = df["sequence"].astype(str).str.strip().str.upper().tolist()
    y = df["label"].to_numpy(np.int64)
    return seqs, y


def load_gue(task):
    """Return dict(train_seqs, y_train, dev_seqs, y_dev, test_seqs, y_test, n_classes) for one task."""
    if task not in TASKS:
        raise KeyError(f"unknown GUE task {task!r}")
    out = {}
    for s in ("train", "dev", "test"):
        seqs, y = _read(task, s)
        out[f"{s}_seqs"], out[f"y_{s}"] = seqs, y
    out["n_classes"] = int(len(np.unique(np.concatenate([out["y_train"], out["y_test"]]))))
    out["name"] = task
    return out


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
