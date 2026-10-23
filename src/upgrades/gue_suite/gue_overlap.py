#!/usr/bin/env python3
"""Independence audit: how much of GUE is the SAME data as the suites already in the paper?

The reviewer asked for an *independently curated* third suite. GUE was assembled by different
authors (DNABERT-2, Zhou et al. 2024), but some of its tasks are drawn from the same upstream sources
as the Nucleotide Transformer (NT) downstream tasks and Genomic Benchmarks (GB) -- EPDnew promoters
in particular. Whether GUE adds independent evidence is therefore an empirical question about the
sequences, answered here with exact string matching (no models).

Checks (exact string identity, forward strand OR reverse complement, upper-cased):
  1. identical-length sequence identity between each GUE promoter / splice split and every split of
     the NT tasks of the same length (promoter_all/no_tata/tata = 300 bp; splice_sites_all = 400 bp),
     with label agreement on the shared sequences;
  2. containment of GB human_nontata_promoters (251 bp) sequences inside GUE prom_300 sequences
     (every 251-bp window of every GUE sequence is hashed), and of GUE prom_core (70 bp) sequences
     inside GB human_nontata_promoters sequences;
  3. nesting of GUE prom_core (70 bp) inside GUE prom_300 (300 bp).
  [added 2026-09-27, post hoc -- descriptive audit, no hypothesis attached]
  4. GUE prom_core (70 bp) contained in the NT promoter tasks (300 bp) that are in the paper;
  5. GUE yeast EMP (10 histone-mark datasets) vs the NT suite's own 10 yeast histone tasks
     (H3 ... H4ac; same upstream source: the Pham et al. 2005 datasets built from the Pokholok et al.
     2005 ChIP-chip maps, distributed at jaist.ac.jp/~tran/nucleosome). These NT tasks are NOT among
     the paper's 15 tasks, but they are part of the NT downstream suite, so they decide whether GUE-EMP
     is independent of "the NT suite" as a whole (pinned revision data_nt.REVISION).
  (The first version's per-task summary audited only the pairs above and wrote 0.0 for the GUE tasks
   it did not audit -- human TF, mouse TF, covid. That summary is superseded by check 6.)
  [added 2026-09-27, fix round 1, post hoc -- descriptive audit, no hypothesis attached. Motivation: a
   verifier found 20-45 % of the (unaudited) human-TF sequences verbatim inside GB/NT sequences.
   Every GUE task is now audited against every task, in both directions; a cell that was not computed
   is NaN, never 0.]
  6. UNIFORM containment audit: all 28 GUE datasets x every task of the paper's 15 (9 GB + 6 NT) and
     the rest of the NT downstream suite at the pinned revision (10 yeast histone tasks +
     splice_sites_acceptors / splice_sites_donors; 18 NT tasks in total). Exact, fwd or rc
     (seqshare.containment_scan: 32-mer anchor index + full string confirmation):
       (a) the GUE sequence occurs inside a sequence of the other task (identity included);
       (b) the GUE sequence contains a sequence of the other task of length >= MIN_LEN_B = 50 bp
           (a shorter exact match is not counted as shared data).
     'shared' = (a) or (b). Partial overlaps (two windows of one locus shifted against each other) are
     NOT counted, so every fraction is a LOWER bound on shared genomic loci.
     -> gue_overlap_matrix.csv (one row per GUE task x other task; label and split breakdowns),
        gue_overlap_other_tasks.csv (one row per other task: size, length profile, and how many of its
        sequences occur inside any GUE sequence),
        gue_overlap_summary.csv (one row per GUE task: fraction of released sequences shared with the
        paper's 15 tasks / with the NT suite / with anything, by label and for the test split, largest
        contributors, and an independence class by the rule below, fixed before this audit was run).
     Independence class (descriptive): 'near-disjoint' if < 1 % of the released sequences are shared
     with any GB/NT task; 'shares with NT suite only (not the paper's 15)' if < 1 % with the paper's
     tasks but >= 1 % with the NT suite; 'shares with the paper's tasks' otherwise.
     Consistency checks (asserted): where check 6 and checks 1-2 measure the same relation, the hit
     vectors must be identical.
  7. [fix round 1] start offset of the GUE prom_core (70 bp) sequences inside the GUE prom_300 and NT
     promoter sequences that contain them (modal offset and the fraction at it), to state exactly which
     window of the 300-bp promoter the core task uses.

Outputs: results/upgrades/gue_overlap.csv (one row per targeted comparison, checks 1-5 and 7),
         results/upgrades/gue_overlap_matrix.csv, results/upgrades/gue_overlap_other_tasks.csv,
         results/upgrades/gue_overlap_summary.csv (check 6).
Seed-free, deterministic. CPU only, single thread; peak RAM ~2 GB (one batch of GB/NT sequences + the
unique GUE sequences + the anchor index).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import gue_data as G  # noqa: E402
import _common as C   # noqa: E402  (repo root on sys.path)
import seqshare as SS  # noqa: E402

OUT = os.path.join(C.RESULTS_DIR, "gue_overlap.csv")
OUT_SUM = os.path.join(C.RESULTS_DIR, "gue_overlap_summary.csv")
OUT_MAT = os.path.join(C.RESULTS_DIR, "gue_overlap_matrix.csv")
OUT_OTH = os.path.join(C.RESULTS_DIR, "gue_overlap_other_tasks.csv")
_RC = str.maketrans("ACGTN", "TGCAN")
NT_HISTONE = ["H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", "H3K9ac",
              "H4", "H4ac"]
PAPER_GB = ["demo_human_or_worm", "human_ensembl_regulatory", "human_nontata_promoters",
            "demo_coding_vs_intergenomic_seqs", "human_enhancers_ensembl", "dummy_mouse_enhancers_ensembl",
            "human_enhancers_cohn", "human_ocr_ensembl", "drosophila_enhancers_stark"]
PAPER_NT = ["promoter_all", "promoter_tata", "promoter_no_tata", "enhancers", "enhancers_types",
            "splice_sites_all"]
NT_REST = NT_HISTONE + ["splice_sites_acceptors", "splice_sites_donors"]
MIN_LEN_B = 50
NEAR_DISJOINT = 0.01
BATCH_CHARS = 260_000_000      # direction (b): GB/NT sequences indexed per batch (bounded RAM)


def rc(s):
    return s.translate(_RC)[::-1]


def nt_split(task, split):
    import data_nt
    return data_nt._read(task, split)


def gb_split(task, split):
    d = C.gbdata.load_dataset(task, seed=42)
    exp = pd.read_csv(os.path.join(C.ROOT, "results", "results.csv"))
    exp = exp[(exp.dataset == task)].iloc[0]
    assert len(d["y_train"]) == int(exp.n_train) and len(d["y_test"]) == int(exp.n_test), "GB count mismatch"
    return (d["train_seqs"], d["y_train"]) if split == "train" else (d["test_seqs"], d["y_test"])


def identical(a, b):
    """Boolean hit vector: does each sequence of `a` occur in `b` exactly (fwd or rc)?"""
    sb = set(b)
    return np.array([(s in sb) or (rc(s) in sb) for s in a], dtype=bool)


def contained(short, long_, L):
    """Does each `short` sequence (length L) occur as a substring of any `long_` sequence (fwd or rc)?
    Hashes every length-L window of every long sequence (64-bit str hash; collision-free in practice)."""
    win = set()
    for s in long_:
        for i in range(len(s) - L + 1):
            win.add(hash(s[i:i + L]))
    hit = np.array([(len(s) == L) and ((hash(s) in win) or (hash(rc(s)) in win)) for s in short], dtype=bool)
    del win
    return hit


def label_agreement(a_seqs, a_y, b_seqs, b_y):
    """Among sequences of a that occur (fwd or rc) in b, fraction with the same label."""
    lab = {}
    for s, y in zip(b_seqs, b_y):
        lab[s] = int(y)
    pairs = []
    for s, y in zip(a_seqs, a_y):
        o = lab.get(s, lab.get(rc(s)))
        if o is not None:
            pairs.append(int(y) == o)
    return (float(np.mean(pairs)) if pairs else float("nan")), len(pairs)


def load_other(suite, task):
    """(seqs upper-cased, labels, split vector) of one GB / NT task, train+test."""
    if suite == "GB":
        tr, te = gb_split(task, "train"), gb_split(task, "test")
    else:
        tr, te = nt_split(task, "train"), nt_split(task, "test")
    seqs = [s.strip().upper() for s in list(tr[0]) + list(te[0])]
    y = np.concatenate([np.asarray(tr[1]), np.asarray(te[1])]).astype(np.int64)
    sv = np.array(["train"] * len(tr[1]) + ["test"] * len(te[1]))
    return seqs, y, sv


def uniform_audit(pool, old_hits):
    """Check 6. `pool(task)` gives the GUE task's (seqs, y, split vector); `old_hits` holds hit vectors
    from checks 1-2 that the uniform audit must reproduce exactly."""
    t0 = time.time()
    # ---- unique GUE sequences (identical strings shared across GUE tasks are indexed once)
    uid_of, uniq, task_uids = {}, [], {}
    task_bit = {t: i for i, t in enumerate(G.ORDER)}
    for t in G.ORDER:
        seqs, _, _ = pool(t)
        u = np.empty(len(seqs), np.int64)
        for i, s in enumerate(seqs):
            j = uid_of.get(s)
            if j is None:
                j = uid_of[s] = len(uniq); uniq.append(s)
            u[i] = j
        task_uids[t] = u
    gue_taskmask = np.zeros(len(uniq), np.int64)
    for t, u in task_uids.items():
        gue_taskmask[u] |= np.int64(1) << np.int64(task_bit[t])
    del uid_of
    max_gue_len = max(len(s) for s in uniq)
    print(f"[6] {len(uniq)} unique GUE sequences (max length {max_gue_len}); building anchors", flush=True)
    anchors = SS.build_anchors(uniq)
    print(f"    {len(anchors[0])} anchors, {anchors[4]} unanchorable ({time.time() - t0:.0f}s)", flush=True)

    others = ([("GB", t, True) for t in PAPER_GB] + [("NT", t, True) for t in PAPER_NT] +
              [("NT", t, False) for t in NT_REST])
    hitA, labA, hitB = {}, {}, {}
    oth_rows = []
    # ---- direction (a): GUE sequence inside a GB/NT sequence (one other task in memory at a time)
    for suite, ot, in_paper in others:
        t1 = time.time()
        seqs, y, sv = load_other(suite, ot)
        r = SS.containment_scan(uniq, seqs, anchors=anchors)
        hitA[ot] = r["hit"]
        lab = np.full(len(uniq), -1, np.int64); lab[r["hit"]] = y[r["first_long"][r["hit"]]]
        labA[ot] = lab
        L = np.fromiter((len(s) for s in seqs), np.int64, len(seqs))
        oth_rows.append(dict(other_suite=suite, other_task=("nt_" + ot if suite == "NT" else ot),
                             in_paper_15=in_paper, n_other=len(seqs), n_train=int((sv == "train").sum()),
                             n_test=int((sv == "test").sum()), len_min=int(L.min()), len_median=float(np.median(L)),
                             len_mean=round(float(L.mean()), 2), len_max=int(L.max()),
                             n_eligible_b=int(((L >= MIN_LEN_B) & (L <= max_gue_len)).sum())))
        print(f"  (a) {suite}:{ot:<32} n={len(seqs):>7}  GUE-unique contained: {int(r['hit'].sum()):>6}"
              f"  ({time.time() - t1:.0f}s)", flush=True)
        del seqs, y, sv
    # ---- direction (b): a GB/NT sequence (>= MIN_LEN_B bp) inside a GUE sequence; batched
    batch, chars = [], 0
    batches = []
    for o in others:
        batch.append(o)
        chars += next(r["len_mean"] * r["n_other"] for r in oth_rows
                      if r["other_task"] == ("nt_" + o[1] if o[0] == "NT" else o[1]))
        if chars >= BATCH_CHARS:
            batches.append(batch); batch, chars = [], 0
    if batch:
        batches.append(batch)
    n_in_gue = {}
    for bi, bat in enumerate(batches):
        t1 = time.time()
        ouid, ouniq, omask = {}, [], []
        per_task_uids = {}
        for bit, (suite, ot, _) in enumerate(bat):
            seqs, _, _ = load_other(suite, ot)
            us = []
            for s in seqs:
                if not (MIN_LEN_B <= len(s) <= max_gue_len):
                    continue
                j = ouid.get(s)
                if j is None:
                    j = ouid[s] = len(ouniq); ouniq.append(s); omask.append(0)
                omask[j] |= (1 << bit)
                us.append(j)
            per_task_uids[ot] = np.array(us, np.int64)
            del seqs
        del ouid
        omask = np.array(omask, np.int64)
        gmask = np.zeros(len(uniq), np.int64)            # which batch tasks each GUE sequence contains
        o_gtask = np.zeros(len(ouniq), np.int64)         # which GUE tasks contain each other sequence

        def on_hit(i, j, start, strand):
            gmask[j] |= omask[i]
            o_gtask[i] |= gue_taskmask[j]
        r = SS.containment_scan(ouniq, uniq, on_hit=on_hit)
        for bit, (suite, ot, _) in enumerate(bat):
            hitB[ot] = (gmask >> np.int64(bit)) & np.int64(1) == 1
            us = per_task_uids[ot]
            n_in_gue[ot] = {"any": int(r["hit"][us].sum())}
            for gt in G.ORDER:
                n_in_gue[ot][gt] = int(((o_gtask[us] >> np.int64(task_bit[gt])) & np.int64(1)).sum())
        print(f"  (b) batch {bi + 1}/{len(batches)}: {len(ouniq)} unique GB/NT sequences >= {MIN_LEN_B} bp;"
              f" GUE-unique containing one: {int((gmask != 0).sum())} ({time.time() - t1:.0f}s)", flush=True)
        del ouniq, omask, gmask, o_gtask, r

    # ---- consistency with checks 1-2 (same relation measured twice must agree exactly)
    for (gtask, ot, kind), old in old_hits.items():
        new = (hitA if kind == "a" else hitB)[ot][task_uids[gtask]]
        assert np.array_equal(new, old), f"uniform audit disagrees with targeted check: {gtask} vs {ot} ({kind})"
    print(f"  consistency: {len(old_hits)} targeted hit vectors reproduced exactly", flush=True)

    # ---- matrix rows
    mat = []
    for suite, ot, in_paper in others:
        oname = "nt_" + ot if suite == "NT" else ot
        for gt in G.ORDER:
            seqs, y, sv = pool(gt)
            u = task_uids[gt]
            a, b = hitA[ot][u], hitB[ot][u]
            s_ = a | b
            pos = y == 1
            te = sv == "test"
            ct = {}
            if a.any():
                for gy, oy in zip(y[a], labA[ot][u][a]):
                    key = f"{int(gy)}->{int(oy)}"
                    ct[key] = ct.get(key, 0) + 1
            mat.append(dict(gue_task=gt, family=G.neg_family(gt), other_suite=suite, other_task=oname,
                            other_in_paper_15=in_paper, audited=True, n_gue=len(u),
                            n_contained_in_other=int(a.sum()), n_contains_other=int(b.sum()),
                            n_shared=int(s_.sum()), frac_shared=round(float(s_.mean()), 6),
                            n_shared_label1=int(s_[pos].sum()), n_shared_other_labels=int(s_[~pos].sum()),
                            n_test=int(te.sum()), n_test_shared=int(s_[te].sum()),
                            n_other_seqs_inside_this_gue_task=n_in_gue[ot][gt],
                            label_gue_to_containing_other=json.dumps(dict(sorted(ct.items())))))
    mat = pd.DataFrame(mat)
    mat.to_csv(OUT_MAT, index=False)
    oth = pd.DataFrame(oth_rows)
    oth["n_inside_any_gue"] = [n_in_gue[(r.other_task[3:] if r.other_suite == "NT" else r.other_task)]["any"]
                               for r in oth.itertuples()]
    oth.to_csv(OUT_OTH, index=False)
    print(f"wrote {OUT_MAT}\nwrote {OUT_OTH}", flush=True)

    # ---- per-GUE-task summary
    paper = [o for s, o, p in others if p]; suite_nt = [o for s, o, p in others if s == "NT"]
    name = {o: ("nt_" + o if s == "NT" else o) for s, o, p in others}
    fam_note = {"human_tf": "ENCODE TF ChIP-seq peaks (new assay/labels); sequences audited vs every GB/NT task",
                "mouse_tf": "mouse ENCODE TF ChIP-seq peaks vs dinucleotide shuffles; audited vs every GB/NT task",
                "covid": "SARS-CoV-2 genome segments; no viral GB/NT task; audited vs every GB/NT task",
                "emp": "Pham et al. 2005 yeast datasets; NT suite has the same 10 histone tasks (not in the paper's 15)",
                "splice": "counterpart: nt_splice_sites_all (Wang et al. 2019 source)",
                "prom_300": "counterparts: nt_promoter_* (EPDnew), GB human_nontata_promoters (EPDnew)",
                "prom_core": "counterparts: nt_promoter_* / GB human_nontata_promoters (EPDnew)"}
    srows = []
    for gt in G.ORDER:
        seqs, y, sv = pool(gt)
        u = task_uids[gt]

        def union(ts):
            h = np.zeros(len(u), bool)
            for o in ts:
                h |= hitA[o][u] | hitB[o][u]
            return h
        hp, hn = union(paper), union(suite_nt)
        ha = hp | hn
        binary = set(np.unique(y).tolist()) <= {0, 1}
        pos, te = (y == 1) if binary else np.zeros(len(y), bool), sv == "test"
        contrib = sorted(((int((hitA[o][u] | hitB[o][u]).sum()), name[o]) for o in paper + NT_REST), reverse=True)
        contrib = [f"{n_}:{c}" for c, n_ in contrib if c > 0][:4]
        if ha.mean() < NEAR_DISJOINT:
            cls = "near-disjoint"
        elif hp.mean() < NEAR_DISJOINT:
            cls = "shares with NT suite only (not the paper's 15)"
        else:
            cls = "shares with the paper's tasks"
        srows.append(dict(task=gt, family=G.neg_family(gt), n_released=len(u), audited_vs_paper=True,
                          audited_vs_nt_suite=True,
                          n_in_paper_tasks=int(hp.sum()), frac_in_paper_tasks=round(float(hp.mean()), 6),
                          frac_in_paper_tasks_label1=(round(float(hp[pos].mean()), 6) if pos.any() else np.nan),
                          frac_in_paper_tasks_label0=(round(float(hp[~pos].mean()), 6) if binary else np.nan),
                          frac_test_in_paper_tasks=round(float(hp[te].mean()), 6),
                          frac_contained_in_paper_tasks=round(float(np.any([hitA[o][u] for o in paper], axis=0).mean()), 6),
                          frac_in_nt_suite=round(float(hn.mean()), 6),
                          frac_in_any_gb_nt=round(float(ha.mean()), 6),
                          frac_label1_in_any_gb_nt=(round(float(ha[pos].mean()), 6) if pos.any() else np.nan),
                          independence_class=cls, largest_contributors="; ".join(contrib),
                          note=fam_note.get(G.neg_family(gt), "")))
    sdf = pd.DataFrame(srows)
    sdf.to_csv(OUT_SUM, index=False)
    print(sdf.drop(columns=["note"]).to_string(index=False))
    print(f"wrote {OUT_SUM}  ({time.time() - t0:.0f}s)")


def main():
    rows = []
    pools = {}

    def pool(task):
        if task not in pools:
            d = G.load_gue(task)
            pools[task] = (d["train_seqs"] + d["dev_seqs"] + d["test_seqs"],
                           np.concatenate([d["y_train"], d["y_dev"], d["y_test"]]),
                           np.array(["train"] * len(d["y_train"]) + ["dev"] * len(d["y_dev"]) +
                                    ["test"] * len(d["y_test"])))
        return pools[task]

    old_hits = {}   # (gue task, other task, direction) -> hit vector, re-checked by the uniform audit

    def add(gtask, suite, otask, osplit, rel, hit, split_vec, y, agree=None, n_agree=None, extra=None):
        for gs in ("train", "dev", "test", "all"):
            m = np.ones(len(hit), bool) if gs == "all" else (split_vec == gs)
            pos = (y == 1) & m
            row = dict(gue_task=gtask, gue_split=gs, other_suite=suite, other_task=otask,
                       other_split=osplit, relation=rel, n_gue=int(m.sum()), n_overlap=int(hit[m].sum()),
                       frac_overlap=round(float(hit[m].mean()) if m.any() else float("nan"), 6),
                       n_overlap_label1=int(hit[pos].sum()), n_overlap_other_labels=int(hit[m & ~pos].sum()),
                       label_agreement=(round(agree, 6) if (gs == "all" and agree is not None) else None),
                       n_label_pairs=(n_agree if gs == "all" else None))
            if extra and gs == "all":
                row.update(extra)
            rows.append(row)
        print(f"  {gtask:<18} vs {suite}:{otask}/{osplit:<9} [{rel}] {int(hit.sum())}/{len(hit)} = "
              f"{hit.mean():.4f}" + (f"  label agreement {agree:.4f} (n={n_agree})" if agree is not None and
                                     n_agree else ""), flush=True)

    # ---- 1. identity with NT (same length), paper tasks
    pairs = [("prom_300_all", "promoter_all"), ("prom_300_notata", "promoter_no_tata"),
             ("prom_300_tata", "promoter_tata"), ("splice_reconstructed", "splice_sites_all")]
    nt_cache = {}
    for gtask, ntask in pairs:
        seqs, y, sv = pool(gtask)
        if ntask not in nt_cache:
            tr = nt_split(ntask, "train"); te = nt_split(ntask, "test")
            nt_cache[ntask] = (tr[0] + te[0], np.concatenate([tr[1], te[1]]))
        ns, ny = nt_cache[ntask]
        hit = identical(seqs, ns)
        ag, na = label_agreement(seqs, y, ns, ny)
        add(gtask, "NT", "nt_" + ntask, "train+test", "identical", hit, sv, y, ag, na)
        if len(set(len(s) for s in seqs) | set(len(s) for s in ns)) == 1:   # same fixed length: identity == containment
            old_hits[(gtask, ntask, "a")] = hit
    # cross-family: every GUE prom_300 task against the NT promoter_all pool (superset)
    for gtask in ("prom_300_notata", "prom_300_tata"):
        seqs, y, sv = pool(gtask)
        ns, ny = nt_cache["promoter_all"]
        hit = identical(seqs, ns); ag, na = label_agreement(seqs, y, ns, ny)
        add(gtask, "NT", "nt_promoter_all", "train+test", "identical", hit, sv, y, ag, na)
        old_hits[(gtask, "promoter_all", "a")] = hit

    # ---- 2. containment with GB human_nontata_promoters (251 bp)
    gb = {s: gb_split("human_nontata_promoters", s) for s in ("train", "test")}
    gb_seqs = gb["train"][0] + gb["test"][0]; gb_y = np.concatenate([gb["train"][1], gb["test"][1]])
    L_gb = int(np.median([len(s) for s in gb_seqs]))
    assert all(len(s) == L_gb for s in gb_seqs), "GB human_nontata_promoters is not fixed-length"
    s300, y300, _ = pool("prom_300_all")
    hit_gb = contained(gb_seqs, s300, L_gb)
    gsv = np.array(["train"] * len(gb["train"][1]) + ["test"] * len(gb["test"][1]))
    add("GB:human_nontata_promoters", "GUE", "prom_300_all", "train+dev+test",
        f"GB {L_gb}bp contained in GUE 300bp", hit_gb, gsv, gb_y)
    # reverse direction for the GUE prom_300 tasks: does a GUE 300-bp sequence CONTAIN a GB sequence?
    win_gb = set(hash(s) for s in gb_seqs) | set(hash(rc(s)) for s in gb_seqs)
    for gtask in ("prom_300_all", "prom_300_notata", "prom_300_tata"):
        seqs, y, sv = pool(gtask)
        hit = np.array([any(hash(s[i:i + L_gb]) in win_gb for i in range(len(s) - L_gb + 1)) for s in seqs])
        add(gtask, "GB", "human_nontata_promoters", "train+test", f"GUE 300bp contains a GB {L_gb}bp sequence",
            hit, sv, y)
        old_hits[(gtask, "human_nontata_promoters", "b")] = hit
    del win_gb
    for gtask in ("prom_core_all", "prom_core_notata", "prom_core_tata"):
        seqs, y, sv = pool(gtask)
        hit = contained(seqs, gb_seqs, 70)
        add(gtask, "GB", "human_nontata_promoters", "train+test", "GUE 70bp contained in GB 251bp", hit, sv, y)
        old_hits[(gtask, "human_nontata_promoters", "a")] = hit

    # ---- 3. GUE prom_core nested inside GUE prom_300 (+ check 7: offset of the core window)
    def offsets(short, long_, hit_ref):
        r = SS.containment_scan(short, long_)
        assert np.array_equal(r["hit"], hit_ref), "seqshare disagrees with the window-hash containment"
        st = r["first_start"][r["hit"]]
        fw = r["first_strand"][r["hit"]] == 0
        if not len(st):
            return {}
        vals, cnt = np.unique(st[fw], return_counts=True) if fw.any() else (np.array([-1]), np.array([0]))
        return dict(modal_start_offset_fwd=int(vals[np.argmax(cnt)]),
                    frac_at_modal_offset=round(float(cnt.max() / len(st)), 6),
                    frac_found_on_forward_strand=round(float(fw.mean()), 6))
    for gtask in ("prom_core_all",):
        seqs, y, sv = pool(gtask)
        hit = contained(seqs, s300, 70)
        add(gtask, "GUE", "prom_300_all", "train+dev+test", "GUE core 70bp contained in GUE 300bp", hit, sv, y,
            extra=offsets(seqs, s300, hit))

    # ---- 4. [post hoc] GUE prom_core contained in the NT promoter tasks (paper tasks)
    nt300 = nt_cache["promoter_all"][0]
    for gtask in ("prom_core_all", "prom_core_notata", "prom_core_tata"):
        seqs, y, sv = pool(gtask)
        hit = contained(seqs, nt300, 70)
        add(gtask, "NT", "nt_promoter_all", "train+test", "GUE 70bp contained in NT 300bp", hit, sv, y,
            extra=offsets(seqs, nt300, hit))
        old_hits[(gtask, "promoter_all", "a")] = hit

    # ---- 5. [post hoc] GUE EMP vs NT histone tasks (NT suite, NOT in the paper's 15)
    for h in NT_HISTONE:
        gtask = f"emp_{h}"
        seqs, y, sv = pool(gtask)
        tr = nt_split(h, "train"); te = nt_split(h, "test")
        ns = tr[0] + te[0]; ny = np.concatenate([tr[1], te[1]])
        Ls = sorted(set(len(s) for s in ns))
        hit = identical(seqs, ns)
        ag, na = label_agreement(seqs, y, ns, ny)
        add(gtask, "NT(suite, not in paper)", h, "train+test", f"identical (NT lengths {Ls[0]}-{Ls[-1]})",
            hit, sv, y, ag, na)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    del nt_cache, gb, gb_seqs, s300, nt300

    # ---- 6. [fix round 1] uniform audit + summary
    uniform_audit(pool, old_hits)


if __name__ == "__main__":
    main()
