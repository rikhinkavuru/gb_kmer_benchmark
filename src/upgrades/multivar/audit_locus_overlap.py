#!/usr/bin/env python3
"""POST-HOC (fix round 3, 2026-09-24 night): do the 11 TF-motif tasks share genomic LOCI, beyond the exact
sequence identities of audit_task_overlap.py?

Why (results-motivated, labelled post hoc): the fix-round-2 audit compared EXACT canonical-strand strings
only.  Tasks drawn from the same source (the promoter tasks of both suites and the promoter class of the
3-class Ensembl regulatory task; the Ensembl enhancer / OCR tasks) can contain the same locus through a
shifted or resized window, which exact matching cannot see.  A claims verifier measured this on a few
pairs (e.g. 73% of human_nontata_promoters positives mostly contained in nt_promoter_all positives); this
script measures it systematically, for every ordered pair of CLASSES of the 11 S11 tasks.

Method (fixed before any output of this script existed):
  * unit = class = (task, label), train + test pooled;
  * k-mers: canonical 31-mers (min of the forward and reverse-complement 2-bit codes, 62 bits), k-mers
    containing a non-ACGT base skipped;
  * sampling (FracMinHash-style): keep a k-mer iff the top 2 bits of (code * 0x9E3779B97F4A7C15 mod 2^64)
    are 0, i.e. 1/4 of the hash space.  The same k-mers are kept in every task, so containment is unbiased;
  * reference = the set of sampled k-mers of ALL sequences of a class (and, as the union over its classes,
    of a whole task);
  * query = up to 2,000 sequences per class (numpy default_rng(42), without replacement); containment of a
    query sequence in a reference = |its distinct sampled k-mers found in the reference| / |its distinct
    sampled k-mers| (sequences with no sampled k-mer are not scored);
  * reported per (query class, reference class) and (query class, whole reference task): mean
    containment, the fraction of query sequences with containment >= 0.5 ("mostly contained": more than half
    of the query window lies in some reference sequence(s)) with a Wilson 95% CI, and the fraction >= 0.8.
  Pairs within the same task are skipped.  Negative (background) classes are reported like any other class,
  so the chance level of a random genomic window against a large reference is visible in the same table.

Outputs (results/upgrades/):
  multivar_posthoc_task_locus_overlap.csv  one row per (query class, reference class or 'all') of different
                                           S11 tasks
  multivar_posthoc_task_locus_links.csv    one row per unordered S11 task pair: the largest 'mostly
                                           contained' fraction over both directions and all query classes
                                           (query class vs the whole other task), where it comes from, the
                                           same restricted to FUNCTIONAL (non-background) query classes, and
                                           the exact-string shared fraction of audit_task_overlap.py
Checkpoints: external/multivar/locus/ref_<task>__<params>.csv (one per reference task; resumable).
Run: bibm/lowcpu.sh /Users/rikhinkavuru/Downloads/venv/bin/python src/upgrades/multivar/audit_locus_overlap.py
(~10-25 min on efficiency cores; peak footprint well under 1.5 GB: one reference class at a time.)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _mv  # noqa: E402

RD = _mv.RESULTS_DIR
K = 31
SAMPLE_SHIFT = 62                     # keep iff (hash >> 62) == 0  ->  1/4 of k-mers
MIX = np.uint64(0x9E3779B97F4A7C15)
N_QUERY = 2000
QSEED = 42
CHUNK_BASES = 3_000_000               # padded bases per vectorized chunk (~100 MB of temporaries)
PARAMS = f"k{K}_s{1 << (64 - SAMPLE_SHIFT)}_q{N_QUERY}_seed{QSEED}"
CK = os.path.join(_mv.EXT_DIR, "locus")
LUT = np.full(256, 4, np.uint8)
for _c, _v in zip("ACGTacgt", (0, 1, 2, 3, 0, 1, 2, 3)):
    LUT[ord(_c)] = _v
# background (non-functional) classes, read from the class names (audit_task_overlap.gb_class_names) and,
# for NT tasks, from the NT label convention (0 = negative / non-element; multivar_posthoc_class_identity.csv
# shows nt_enhancers_types' class 0 is nt_enhancers' negative class)
BACKGROUND_NAMES = {"negative", "negatives", "noncoding", "non-enhancer", "nonenhancer", "random", "background"}


def s11_tasks():
    cov = pd.read_csv(os.path.join(RD, "multivar_covariates.csv"))
    return list(cov.loc[cov["S11"], "task"])


def class_names(task, labels):
    if task.startswith("nt_"):
        return {int(y): ("background" if int(y) == 0 else f"label{int(y)}") for y in labels}
    import audit_task_overlap as AO
    cn = AO.gb_class_names(task)
    return {int(y): cn[int(y)] for y in labels}


def _sketch_block(seqs):
    """seqs of similar length -> (can, keep): (n, L-K+1) canonical 62-bit codes and the mask of valid
    (all-ACGT) AND sampled k-mers."""
    L = max(len(s) for s in seqs)
    n = L - K + 1
    if n <= 0:
        return np.zeros((len(seqs), 0), np.uint64), np.zeros((len(seqs), 0), bool)
    arr = np.full((len(seqs), L), 4, np.uint8)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = LUT[np.frombuffer(s.encode("ascii"), np.uint8)]
    valid = np.ones((len(seqs), n), bool)
    fwd = np.zeros((len(seqs), n), np.uint64)
    rc = np.zeros((len(seqs), n), np.uint64)
    two, three = np.uint64(2), np.uint64(3)
    for j in range(K):
        b = arr[:, j:j + n]
        valid &= b < 4
        bb = np.where(b < 4, b, 0).astype(np.uint64)
        fwd = (fwd << two) | bb
        rc |= (three - bb) << np.uint64(2 * j)
    can = np.minimum(fwd, rc)
    del fwd, rc
    with np.errstate(over="ignore"):
        h = can * MIX
    keep = valid & ((h >> np.uint64(SAMPLE_SHIFT)) == 0)
    return can, keep


def _blocks(seqs):
    """indices of seqs grouped into length-sorted blocks of <= CHUNK_BASES padded bases."""
    order = np.argsort([len(s) for s in seqs], kind="stable")
    blk, cur, curmax = [], [], 0
    for i in order:
        L = len(seqs[i])
        if cur and max(curmax, L) * (len(cur) + 1) > CHUNK_BASES:
            blk.append(cur)
            cur, curmax = [], 0
        cur.append(int(i))
        curmax = max(curmax, L)
    if cur:
        blk.append(cur)
    return blk


def reference_set(seqs):
    """sorted unique sampled canonical 31-mers of all seqs."""
    parts = []
    for b in _blocks(seqs):
        can, keep = _sketch_block([seqs[i] for i in b])
        parts.append(np.unique(can[keep]))
        del can, keep
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.uint64)


def query_sketches(seqs):
    """per-sequence distinct sampled k-mers, concatenated: (kmers, owner index, per-sequence count)."""
    out = [None] * len(seqs)
    for b in _blocks(seqs):
        can, keep = _sketch_block([seqs[i] for i in b])
        for r, i in enumerate(b):
            out[i] = np.unique(can[r][keep[r]])
        del can, keep
    cnt = np.array([len(o) for o in out], np.int64)
    km = np.concatenate(out) if len(out) else np.zeros(0, np.uint64)
    own = np.repeat(np.arange(len(out)), cnt)
    return km, own, cnt


def member(ref, q):
    if len(ref) == 0 or len(q) == 0:
        return np.zeros(len(q), bool)
    pos = np.searchsorted(ref, q)
    pos[pos >= len(ref)] = len(ref) - 1
    return ref[pos] == q


def containment(hit, own, cnt):
    """per query sequence: fraction of its sampled k-mers that hit; NaN for sequences without k-mers."""
    num = np.bincount(own, weights=hit.astype(float), minlength=len(cnt))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def summarize(c):
    ok = c[~np.isnan(c)]
    n = len(ok)
    k5 = int((ok >= 0.5).sum())
    lo, hi = wilson(k5, n)
    return dict(n_query_scored=n, mean_containment=float(ok.mean()) if n else np.nan,
                frac_ge_0_5=k5 / n if n else np.nan, frac_ge_0_5_lo=lo, frac_ge_0_5_hi=hi,
                frac_ge_0_8=float((ok >= 0.8).mean()) if n else np.nan)


def main():
    _mv.start_memguard(tag="audit_locus_overlap")
    t0 = time.time()
    os.makedirs(CK, exist_ok=True)
    tasks = s11_tasks()
    assert len(tasks) == 11, tasks
    # ---------------------------------------------------------------- query sketches (all classes)
    Q = {}
    for t in tasks:
        tr, ytr, te, yte = _mv.load_task_cached(t)
        seqs, y = tr + te, np.concatenate([ytr, yte])
        names = class_names(t, np.unique(y))
        for lab in np.unique(y):
            idx = np.flatnonzero(y == lab)
            rng = np.random.default_rng(QSEED)
            pick = np.sort(rng.choice(idx, size=min(N_QUERY, len(idx)), replace=False))
            km, own, cnt = query_sketches([seqs[i] for i in pick])
            Q[(t, int(lab))] = dict(km=km, own=own, cnt=cnt, n_class=len(idx), n_query=len(pick), name=names[int(lab)])
        del tr, te, seqs
        print(f"query sketches {t:<30} ({time.time() - t0:.0f}s)", flush=True)
    # ---------------------------------------------------------------- references, one task at a time
    for rt in tasks:
        fp = os.path.join(CK, f"ref_{rt}__{PARAMS}.csv")
        if os.path.exists(fp):
            print(f"ref {rt}: checkpoint found, skipped", flush=True)
            continue
        tr, ytr, te, yte = _mv.load_task_cached(rt)
        seqs, y = tr + te, np.concatenate([ytr, yte])
        del tr, te
        names = class_names(rt, np.unique(y))
        hits = {q: [] for q in Q if q[0] != rt}
        nref = {}
        for lab in np.unique(y):
            t1 = time.time()
            ref = reference_set([seqs[i] for i in np.flatnonzero(y == lab)])
            nref[int(lab)] = (int((y == lab).sum()), len(ref))
            for q in hits:
                hits[q].append(member(ref, Q[q]["km"]))
            print(f"ref {rt}:{int(lab)} ({names[int(lab)]}) {nref[int(lab)][0]} seqs, {len(ref):,} sampled k-mers "
                  f"({time.time() - t1:.0f}s; footprint {_mv.footprint_mb():.0f} MB)", flush=True)
            del ref
        del seqs
        rows = []
        for q, hl in hits.items():
            qq = Q[q]
            base = dict(query_task=q[0], query_label=q[1], query_class=qq["name"], query_class_n=qq["n_class"],
                        n_query=qq["n_query"], ref_task=rt)
            for lab, h in zip(sorted(nref), hl):
                rows.append(dict(base, ref_label=str(lab), ref_class=names[lab], ref_n_seqs=nref[lab][0],
                                 ref_n_sampled_kmers=nref[lab][1], **summarize(containment(h, qq["own"], qq["cnt"]))))
            anyh = np.logical_or.reduce(hl)
            rows.append(dict(base, ref_label="all", ref_class="(whole task)", ref_n_seqs=sum(v[0] for v in nref.values()),
                             ref_n_sampled_kmers=np.nan, **summarize(containment(anyh, qq["own"], qq["cnt"]))))
        tmp = fp + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, fp)
        print(f"ref {rt}: done ({time.time() - t0:.0f}s)", flush=True)
    # ---------------------------------------------------------------- merge + task-level links
    df = pd.concat([pd.read_csv(os.path.join(CK, f"ref_{rt}__{PARAMS}.csv"), dtype={"ref_label": str}) for rt in tasks],
                   ignore_index=True)
    df.insert(0, "params", PARAMS)
    df["query_is_background"] = df["query_class"].str.lower().isin(BACKGROUND_NAMES)
    tmp = os.path.join(RD, "multivar_posthoc_task_locus_overlap.csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(RD, "multivar_posthoc_task_locus_overlap.csv"))
    ex = pd.read_csv(os.path.join(RD, "multivar_posthoc_task_overlap.csv"))
    exf = {(a, b): f for a, b, f in zip(ex["task_a"], ex["task_b"], ex["frac_a_in_b"])}
    whole = df[df["ref_label"] == "all"]
    links = []
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            g = whole[((whole["query_task"] == a) & (whole["ref_task"] == b)) |
                      ((whole["query_task"] == b) & (whole["ref_task"] == a))]
            top = g.sort_values("frac_ge_0_5", ascending=False).iloc[0]
            gf = g[~g["query_is_background"]]
            topf = gf.sort_values("frac_ge_0_5", ascending=False).iloc[0]
            links.append(dict(task_a=a, task_b=b, locus_link=float(top["frac_ge_0_5"]),
                              locus_link_from=f"{top['query_task']}:{top['query_label']} ({top['query_class']}) in "
                                              f"{top['ref_task']}",
                              locus_link_lo=float(top["frac_ge_0_5_lo"]), locus_link_hi=float(top["frac_ge_0_5_hi"]),
                              locus_link_functional=float(topf["frac_ge_0_5"]),
                              locus_link_functional_from=f"{topf['query_task']}:{topf['query_label']} "
                                                         f"({topf['query_class']}) in {topf['ref_task']}",
                              exact_shared_frac_max=max(exf[(a, b)], exf[(b, a)])))
    lk = pd.DataFrame(links).sort_values("locus_link", ascending=False)
    tmp = os.path.join(RD, "multivar_posthoc_task_locus_links.csv.tmp")
    lk.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(RD, "multivar_posthoc_task_locus_links.csv"))
    pd.set_option("display.width", 250)
    print("\ntask pairs by locus link (largest share of a query class 'mostly contained' (>= 0.5) in the other task):")
    print(lk.to_string(index=False))
    print(f"\nwrote multivar_posthoc_task_locus_overlap.csv ({len(df)} rows), multivar_posthoc_task_locus_links.csv "
          f"({len(lk)} pairs) ({time.time() - t0:.0f}s; peak footprint {_mv._PEAK[0]:.0f} MB)")


if __name__ == "__main__":
    main()
