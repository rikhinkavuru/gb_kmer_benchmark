#!/usr/bin/env python3
"""POST-HOC (fix round 2, 2026-09-24 evening): are the 15 benchmark tasks distinct sequence collections?

Why (results-motivated, labelled post hoc): a verifier found that nt_promoter_all is the union of
nt_promoter_tata and nt_promoter_no_tata, and nt_enhancers_types re-labels the nt_enhancers sequences.
Every cross-task statistic in this module treats tasks as exchangeable units, so the overlap has to be
measured, not assumed.  This script measures it for ALL pairs of the 15 tasks, from the sequences:

  * per task: the set of distinct sequences (train + test), each reduced to its canonical strand
    (min of the string and its reverse complement), hashed to 64 bits (blake2b; collision probability
    for ~10^6 sequences ~1e-7);
  * per ordered pair (A, B): the fraction of A's distinct sequences that also occur in B (either strand);
  * exact identity checks by string comparison (no hashing): is nt_promoter_all's train (test) split,
    as a multiset of (sequence, label) pairs, exactly the concatenation of nt_promoter_tata's and
    nt_promoter_no_tata's train (test) splits, and are those two disjoint; are nt_enhancers_types'
    train/test sequence lists identical to nt_enhancers'; (fix round 3) is the OCR class of
    human_ensembl_regulatory, per split, the same multiset of strings as human_ocr_ensembl's positives.
Exact strings only; the same locus in a shifted window is measured by audit_locus_overlap.py.

No thresholds, no model fitting.  Every load asserts the Table-1 counts (_mv.load_task_cached).
Outputs: results/upgrades/multivar_posthoc_task_overlap.csv (210 ordered pairs of distinct tasks),
         results/upgrades/multivar_posthoc_task_identity.csv (the exact identity checks),
         results/upgrades/multivar_posthoc_task_overlap_classes.csv (for every other pair sharing >= 1% of
         either task's sequences: the shared sequences by split:label in each task, plus class sizes and,
         for Genomic-Benchmarks tasks, the class folder names).
Run: bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/multivar/audit_task_overlap.py
(about 1-2 min; peak footprint < 1 GB -- one task's sequences in memory at a time.)
"""
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mv  # noqa: E402

RD = _mv.RESULTS_DIR
_COMP = str.maketrans("ACGTN", "TGCAN")


def canon(s):
    r = s.translate(_COMP)[::-1]
    return s if s <= r else r


def hash64(seqs):
    """sorted unique uint64 hashes of the canonical strand of each sequence."""
    h = np.fromiter((int.from_bytes(hashlib.blake2b(canon(s).encode("ascii"), digest_size=8).digest(), "little")
                     for s in seqs), dtype=np.uint64, count=len(seqs))
    return np.unique(h)


def gb_class_names(task):
    """Class folder names of a Genomic-Benchmarks task in label order (labels = index in the sorted class
    names, exactly as _mv._read_gb_zip / data._read_split_dir assign them); None for NT tasks."""
    if task.startswith("nt_"):
        return None
    import zipfile
    with zipfile.ZipFile(os.path.join(_mv.GB_ZIP_DIR, f"{task}_v0.zip")) as zf:
        return sorted({n.split("/")[2] for n in zf.namelist() if len(n.split("/")) == 4 and n.endswith(".txt")})


def multiset_equal(pairs_a, pairs_b):
    return sorted(pairs_a) == sorted(pairs_b)


def main():
    _mv.start_memguard(tag="audit_task_overlap")
    t0 = time.time()
    cov = pd.read_csv(os.path.join(RD, "multivar_covariates.csv")).set_index("task")
    H = {}
    for task in _mv.TASKS:
        tr, ytr, te, yte = _mv.load_task_cached(task)
        H[task] = hash64(tr + te)
        print(f"{task:<34} {len(tr) + len(te):>7} sequences, {len(H[task]):>7} distinct (canonical strand)  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del tr, te
    rows = []
    for a in _mv.TASKS:
        for b in _mv.TASKS:
            if a == b:
                continue
            n_sh = int(np.intersect1d(H[a], H[b], assume_unique=True).size)
            rows.append(dict(task_a=a, task_b=b, S11_a=bool(cov.loc[a, "S11"]), S11_b=bool(cov.loc[b, "S11"]),
                             n_distinct_a=int(H[a].size), n_distinct_b=int(H[b].size), n_shared=n_sh,
                             frac_a_in_b=n_sh / H[a].size))
    ov = pd.DataFrame(rows)
    tmp = os.path.join(RD, "multivar_posthoc_task_overlap.csv.tmp")
    ov.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(RD, "multivar_posthoc_task_overlap.csv"))
    nz = ov[ov["n_shared"] > 0].sort_values("frac_a_in_b", ascending=False)
    print("\nordered pairs with any shared sequence (fraction of A's distinct sequences found in B):")
    print(nz.to_string(index=False) if len(nz) else "  none")

    # ---------------------------------------------------------------- exact identity checks (strings)
    ci = []
    al = _mv.load_task_cached("nt_promoter_all")
    ta = _mv.load_task_cached("nt_promoter_tata")
    nt = _mv.load_task_cached("nt_promoter_no_tata")
    for sp, (i_s, i_y) in (("train", (0, 1)), ("test", (2, 3))):
        pa = list(zip(al[i_s], al[i_y].tolist()))
        pt = list(zip(ta[i_s], ta[i_y].tolist()))
        pn = list(zip(nt[i_s], nt[i_y].tolist()))
        ci.append(dict(task="nt_promoter_all", check=f"{sp}_seq_label_multiset_equals_tata_plus_no_tata",
                       value=float(multiset_equal(pa, pt + pn)),
                       note=f"n {len(pa)} = {len(pt)} + {len(pn)}; exact (sequence, label) pairs"))
        ci.append(dict(task="nt_promoter_all", check=f"{sp}_tata_and_no_tata_sequence_sets_disjoint",
                       value=float(not (set(ta[i_s]) & set(nt[i_s]))), note="exact strings"))
    en = _mv.load_task_cached("nt_enhancers")
    et = _mv.load_task_cached("nt_enhancers_types")
    for sp, i_s in (("train", 0), ("test", 2)):
        ci.append(dict(task="nt_enhancers_types", check=f"{sp}_sequence_list_identical_to_nt_enhancers",
                       value=float(en[i_s] == et[i_s]), note="same order; only the labels differ"))
    del al, ta, nt, en, et
    # fix round 3: the regulatory task's OCR class vs human_ocr_ensembl's positive class, as MULTISETS of exact
    # strings per split (the round-2 wording '69,901 of 69,902 shared' counted distinct canonical sequences;
    # one train sequence occurs twice in both tasks)
    rg = _mv.load_task_cached("human_ensembl_regulatory")
    oc = _mv.load_task_cached("human_ocr_ensembl")
    lr, lo = gb_class_names("human_ensembl_regulatory").index("ocr"), gb_class_names("human_ocr_ensembl").index("positive")
    for sp, (i_s, i_y) in (("train", (0, 1)), ("test", (2, 3))):
        a = [q for q, y in zip(rg[i_s], rg[i_y].tolist()) if y == lr]
        b = [q for q, y in zip(oc[i_s], oc[i_y].tolist()) if y == lo]
        ci.append(dict(task="human_ensembl_regulatory", check=f"{sp}_ocr_class_multiset_equals_human_ocr_ensembl_positives",
                       value=float(multiset_equal(a, b)),
                       note=f"n {len(a)} = {len(b)} ({len(set(a))} distinct); exact strings, same split"))
    del rg, oc
    # ---------------------------------------------------------------- partial overlaps: which classes/splits
    # For every unordered pair sharing >= 1% of either task's distinct sequences that is NOT one of the two
    # identities above, break the shared sequences down by (split, label) in each task.
    ident = {frozenset(p) for p in (("nt_promoter_all", "nt_promoter_tata"), ("nt_promoter_all", "nt_promoter_no_tata"),
                                    ("nt_enhancers", "nt_enhancers_types"))}
    big = ov[(ov["frac_a_in_b"] >= 0.01)]
    pairs = sorted({tuple(sorted((a, b))) for a, b in zip(big["task_a"], big["task_b"])} - {tuple(sorted(p)) for p in ident})
    brk = []

    def split_label_keys(task):
        tr, ytr, te, yte = _mv.load_task_cached(task)
        d = {}
        for sp, S, Y in (("train", tr, ytr), ("test", te, yte)):
            for s, y in zip(S, Y.tolist()):
                d.setdefault(canon(s), set()).add(f"{sp}:{y}")
        return d

    for a, b in pairs:
        da, db = split_label_keys(a), split_label_keys(b)
        sh = set(da) & set(db)
        cnt = {}
        for s in sh:
            k = ("|".join(sorted(da[s])), "|".join(sorted(db[s])))
            cnt[k] = cnt.get(k, 0) + 1
        for (ka, kb), c in sorted(cnt.items(), key=lambda kv: -kv[1]):
            brk.append(dict(task_a=a, task_b=b, split_label_in_a=ka, split_label_in_b=kb, n_shared=c,
                            frac_of_shared=c / len(sh)))
        del da, db
    for a, b in pairs:
        for t in (a, b):
            tr, ytr, te, yte = _mv.load_task_cached(t)
            cn = gb_class_names(t)
            for sp, Y in (("train", ytr), ("test", yte)):
                for y in np.unique(Y):
                    brk.append(dict(task_a=t, task_b="(class sizes)", split_label_in_a=f"{sp}:{int(y)}",
                                    split_label_in_b=(f"class name: {cn[int(y)]}" if cn else ""),
                                    n_shared=int((Y == y).sum()), frac_of_shared=np.nan))
    bdf = pd.DataFrame(brk)
    tmp = os.path.join(RD, "multivar_posthoc_task_overlap_classes.csv.tmp")
    bdf.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(RD, "multivar_posthoc_task_overlap_classes.csv"))
    print("\npartial overlaps by split:label (canonical-strand sequences shared by the pair):")
    print(bdf.to_string(index=False) if len(bdf) else "  none")
    cdf = pd.DataFrame(ci)
    tmp = os.path.join(RD, "multivar_posthoc_task_identity.csv.tmp")
    cdf.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(RD, "multivar_posthoc_task_identity.csv"))
    print("\n" + cdf.to_string(index=False))
    print(f"\nwrote multivar_posthoc_task_overlap.csv ({len(ov)} rows), multivar_posthoc_task_identity.csv "
          f"({time.time() - t0:.0f}s; peak footprint {_mv._PEAK[0]:.0f} MB)")


if __name__ == "__main__":
    main()
