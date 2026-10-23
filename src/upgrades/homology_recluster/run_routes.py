#!/usr/bin/env python3
"""Step 4d/4e/5 -- route diagnostics before vs after homology cleaning, the Route-1 correlation with
homology-clean MCCs, and the final per-task table.

PRE-REGISTERED ANALYSIS PLAN (written 2026-09-24 before any run of this file)
----------------------------------------------------------------------------
Inputs: homology_recluster_{redundancy,eval,interventions,positional}.csv (run_cluster / run_eval /
run_interventions) and results/cross_suite_summary.csv (published discriminability, Route-1 x-axis).

Per task we compute every route-assigning diagnostic twice with IDENTICAL code: BEFORE on the paper's
data (provided split, all sequences) and AFTER on homology-clean data (id80 primary; id50 strict):
  * MCC: BEFORE = our reproduction of Table I; AFTER = cluster-aware re-split MCC (primary) and the
    test-purge MCC (secondary), from homology_recluster_eval.csv.
  * Route 2: dMCC(B8 - B1) with paired-bootstrap CI (homology_recluster_positional.csv; for tasks
    without an original-data E3 row the published positional_sweep.csv row is used for BEFORE).
  * Route 3 (binary tasks): TATA (TBP >= 1.3 bits/pos) box rate in positives/negatives, neg-excess
    (neg - pos, bootstrap CI), TATA motif-only AUROC (CI) and GC AUROC (CI); TATA-flag cleaning
    (diagnose.py's rule). AFTER = the same on the NON-REDUNDANT representatives (one per component x
    label); <= 15000 sequences per class, seeded, same sampler before/after.
  * Route 1: max positive-direction motif-only AUROC over the curated TF panel (run_discriminability:
    1.0 bits/pos, representative contrast, <= 4000 per class, seeded) and motif richness;
    BEFORE on all sequences, AFTER on the NR representatives.

Label rule (reconstructed from the paper's Table I and applied identically BEFORE and AFTER):
  solved         if MCC >= 0.70
  positional     elif dMCC(B8-B1) > 0.20 with CI lower > 0
  contamination  elif binary, non-promoter task, TATA neg-excess >= 0.15 and TATA AUROC CI upper < 0.5
  shared-motif   otherwise (failing, no positional rescue, no robust negative-set contamination)
  (the rule must reproduce the published labels on the BEFORE data; any mismatch is reported.)

PRESERVATION RULE (from the task specification, fixed in advance): a route label is PRESERVED if the
diagnostic that assigns it keeps its sign and its 95 % CI still excludes 0 on homology-clean data:
  solved         MCC - 0.70           > 0, CI lower bound of MCC > 0.70
  positional     dMCC(B8 - B1)        > 0, CI lower > 0
  contamination  TATA AUROC - 0.5     < 0, CI upper < 0.5   AND  neg-excess > 0, CI lower > 0
  shared-motif   0.70 - MCC           > 0, CI upper bound of MCC < 0.70
and we additionally report whether the full label rule re-assigns the task on homology-clean data
(e.g. a shared-motif task acquiring a positional rescue). A label whose diagnostic keeps its sign but
whose CI now touches 0 is reported as "weakened", not preserved.

ADDENDUM (added 2026-09-24 after the first run_eval results, BEFORE run_routes.py was run): the
cluster-aware re-split pools the provided train and test splits, so for a benchmark whose provided test
split is deliberately out-of-distribution (the random re-split control reveals this, e.g. nt_enhancers)
the re-split also removes that shift. The MCC-based rules (tier; solved / shared-motif preservation) are
therefore ALSO evaluated with the test-purge MCC (same trained model, provided test split minus homologs),
reported side by side as a secondary protocol; the primary protocol stays the re-split as pre-registered.
For that secondary protocol the PREVALENCE-STANDARDISED purge MCC (run_eval.py's std_mcc, CI from the
same bootstrap) is used, because run_eval.py's plan already states that purging can shift the class mix and
that this alone moves MCC (human_nontata_promoters: purge keeps a 91 %-positive test set; raw purged MCC
equals its random-removal null); the raw purged MCC is carried alongside (purge_raw_mcc).

SEED ROBUSTNESS (added with run_seeds.py, before its first run): if homology_recluster_seeds_summary.csv
exists, the label rule is also applied to the MEAN re-split MCC over split seeds {42,43,44}
(label_after_seedmean), reported as a robustness column next to the pre-registered seed-42 label.

Route-1 correlation: Spearman(max_pos_motif_auroc, MCC) over the 11 TF-motif tasks with the published
x and y in {published MCC, purge-id80 MCC, re-split-id80 MCC, re-split-id50 MCC}, plus x recomputed
on NR representatives; rho, p, leave-one-task-out jackknife range (as clean_subset_correlation.py).

The diagnose.py CLI verdict (diagnose._verdict) is also evaluated on a card assembled from the same
BEFORE / AFTER numbers, as a second, independent labelling logic.

VERIFIER ADDENDUM (2026-09-24 ~11:30, POST HOC, after an independent audit; the pre-registered columns
above are unchanged and still reported):
  * STRICT LABEL COUNT. preserved() tests only the diagnostic that ASSIGNED the label, so a task can be
    'preserved' while the full label rule re-assigns it (nt_enhancers: contamination diagnostics keep
    their sign, but the re-split MCC >= 0.7 makes it 'solved'). preservation_strict = 'preserved' only
    if the pre-registered test passes AND label_after == label_before; both counts are reported.
  * PURGE EVALUABILITY. A purge whose status is not 'ok' (run_eval.py: 'class_eliminated',
    'too_few_kept') is 'n/a (not evaluable)' and is excluded from the purge counts and from the purge
    Route-1 correlation (it is no longer silently scored on a test set missing a class).
  * ROUTE-1 CONTROLS. The PRIMARY Route-1 test (named in advance by the plan above: MCC protocol =
    cluster-aware re-split) is x = published discriminability, y = re-split-id80 MCC, all 11 TF tasks.
    Added as controls/sensitivities: y = RANDOM re-split MCC (same pooling, no homology control -- a
    change shared with it is a re-pooling effect, not homology), split-seed means (seeds 42-44), and
    subsets excluding nt_enhancers_types, both NT enhancer tasks (whose provided test split is out of
    distribution by design, so re-pooling changes them), and the submitted paper's 9-task subset.
  * TABLE. Adds the pre-registered HOMOLOGY-INFLATED call per protocol, the purge status / per-class
    kept counts / accuracy on removed vs kept test sequences / subtype-matched effect, and split-seed
    mean + range of cluster - random with a seed-robust inflation call (seed-42 call AND the delta is
    negative for all three split seeds).

Outputs: results/upgrades/homology_recluster_routes.csv, homology_recluster_route1.csv,
         homology_recluster_table.csv, homology_recluster_interpretation.txt.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _hc  # noqa: E402
C = _hc.C
sys.path.insert(0, os.path.join(_hc.UP, "diagnose"))
sys.path.insert(0, _hc.ROOT)
from run_discriminability import CURATED_VERT, CURATED_INSECT, resolve_curated  # noqa: E402

TATA_BITS = 1.3
THR_BITS = 1.0
CAP_R3 = 15000
CAP_R1 = 4000
INSECT = {"drosophila_enhancers_stark"}
MULTI_REP = {"human_ensembl_regulatory": "promoter", "nt_enhancers_types": 0, "nt_splice_sites_all": 0}


def sub(idx, cap, rng):
    return idx if len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False))


def gb_class_names(task):
    import zipfile
    with zipfile.ZipFile(_hc._fetch_gb_zip(task)) as zf:
        names = {n.split("/")[2] for n in zf.namelist() if n.count("/") >= 3 and n.startswith(f"{task}/train/")}
    return sorted(names)


def rep_contrast(task, y):
    """(pos_mask, neg_mask) for the representative contrast of run_discriminability."""
    labs = np.unique(y)
    if len(labs) == 2:
        return y == 1, y == 0
    rep = MULTI_REP[task]
    if isinstance(rep, str):
        rep = gb_class_names(task).index(rep)
    return y == rep, y != rep


def pooled(task, dv):
    """Pooled (seqs, y) for BEFORE ('original': all sequences) or AFTER (NR representatives at dv)."""
    tr, ytr, te, yte = _hc.load_task(task)
    seqs = tr + te; y = np.concatenate([ytr, yte])
    if dv == "original":
        return seqs, y
    comp, _ = _hc.load_clusters(task, dv)
    keep = _hc.nr_representatives(comp, y, _hc.SEED)
    return [seqs[i] for i in keep], y[keep]


def route3_block(seqs, y, tbp, rng):
    pos = sub(np.where(y == 1)[0], CAP_R3, rng); neg = sub(np.where(y == 0)[0], CAP_R3, rng)
    P = [seqs[i] for i in pos]; N = [seqs[i] for i in neg]
    tp = C.tata_hit_counts(P, tbp, TATA_BITS).astype(np.int64); tn = C.tata_hit_counts(N, tbp, TATA_BITS).astype(np.int64)
    gp = C.gc_content(P); gn = C.gc_content(N)
    ta = C.auc_counts(tp, tn); tlo, thi = C.boot_ci(tp, tn, C.auc_counts, _hc.BOOT, rng)
    ga = C.auc_float(gp, gn); glo, ghi = C.boot_ci(gp, gn, C.auc_float, _hc.BOOT, rng)
    pr, nr = float((tp >= 1).mean()), float((tn >= 1).mean())
    ex = np.empty(_hc.BOOT)
    fp, fn = (tp >= 1).astype(float), (tn >= 1).astype(float)
    for b in range(_hc.BOOT):
        ex[b] = fn[rng.randint(0, len(fn), len(fn))].mean() - fp[rng.randint(0, len(fp), len(fp))].mean()
    # diagnose.py TATA-flag cleaning: drop excess flagged negatives, re-measure TATA AUROC
    flagged = tn >= 1; nidx = np.arange(len(tn)); keep_fl = int(round(pr * len(tn)))
    fl = nidx[flagged]
    keep = (np.sort(np.concatenate([nidx[~flagged], fl[rng.permutation(len(fl))[:keep_fl]]]))
            if len(fl) > keep_fl else nidx)
    cl = C.auc_counts(tp, tn[keep])
    return dict(n_pos_scanned=len(P), n_neg_scanned=len(N), tata_pos_rate=round(pr, 4), tata_neg_rate=round(nr, 4),
                tata_neg_excess=round(nr - pr, 4), tata_excess_lo=round(float(np.percentile(ex, 2.5)), 4),
                tata_excess_hi=round(float(np.percentile(ex, 97.5)), 4),
                tata_auroc=round(ta, 4), tata_lo=round(tlo, 4), tata_hi=round(thi, 4),
                gc_auroc=round(ga, 4), gc_lo=round(glo, 4), gc_hi=round(ghi, 4),
                cleaned_tata_auroc=round(cl, 4), cleaning_moves_to_chance=bool(abs(cl - .5) < abs(ta - .5)))


def route1_block(task, seqs, y, curated, rng):
    pm, nm = rep_contrast(task, y)
    pos = sub(np.where(pm)[0], CAP_R1, rng); neg = sub(np.where(nm)[0], CAP_R1, rng)
    pc_ = C.motif_match.encode_sequences([seqs[i] for i in pos]); nc_ = C.motif_match.encode_sequences([seqs[i] for i in neg])
    best, best_tf, rich = 0.0, "", []
    for fam, name, m in curated:
        a = C.motif_match.count_hits_batch(pc_, m["pssm"], THR_BITS)
        b = C.motif_match.count_hits_batch(nc_, m["pssm"], THR_BITS)
        auc = C.auc_counts(a.astype(np.int64), b.astype(np.int64))
        rich.append((a.mean() + b.mean()) / 2)
        if auc > best:
            best, best_tf = auc, name
    return dict(max_pos_motif_auroc=round(float(best), 4), best_TF=best_tf, motif_richness=round(float(np.mean(rich)), 4))


def diag_block(task, dv, tbp, curated_by):
    f = os.path.join(_hc.CKPT_DIR, f"routes_{task}_{dv}.csv")
    if os.path.exists(f):
        return pd.read_csv(f).iloc[0].to_dict()
    seqs, y = pooled(task, dv)
    rng = np.random.RandomState(_hc.SEED)
    out = dict(task=task, data=dv, n_seqs=len(seqs))
    coll = "insects" if task in INSECT else "vertebrates"
    out.update(route1_block(task, seqs, y, curated_by[coll], np.random.RandomState(_hc.SEED)))
    if coll == "insects":   # diagnose.py always uses the vertebrate panel -> also keep that for its verdict
        v = route1_block(task, seqs, y, curated_by["vertebrates"], np.random.RandomState(_hc.SEED))
        out.update(vert_max_pos_motif_auroc=v["max_pos_motif_auroc"], vert_motif_richness=v["motif_richness"])
    if len(np.unique(y)) == 2:
        out.update(route3_block(seqs, y, tbp, rng))
    pd.DataFrame([out]).to_csv(f, index=False)
    print(f"  diag {task} [{dv}] done", flush=True)
    return out


# ------------------------------------------------------------------ labels
def label_rule(mcc, pdelta, plo, binary, promoter, excess, tata_hi):
    if mcc >= 0.7:
        return "solved"
    if pdelta == pdelta and pdelta > 0.2 and plo > 0:
        return "positional"
    if binary and not promoter and excess == excess and excess >= 0.15 and tata_hi < 0.5:
        return "contamination"
    return "shared-motif"


def preserved(label, a):
    """Pre-registered preservation test on AFTER numbers `a`. Returns (status, diagnostic_str)."""
    if label == "solved":
        s = a["mcc"] - 0.7; ok = s > 0 and a["mcc_lo"] > 0.7
        weak = s > 0 and not ok
        d = f"MCC-0.7={s:+.3f} (MCC CI [{a['mcc_lo']:.3f},{a['mcc_hi']:.3f}])"
    elif label == "shared-motif":
        s = 0.7 - a["mcc"]; ok = s > 0 and a["mcc_hi"] < 0.7
        weak = s > 0 and not ok
        d = f"0.7-MCC={s:+.3f} (MCC CI [{a['mcc_lo']:.3f},{a['mcc_hi']:.3f}])"
    elif label == "positional":
        ok = a["pos_delta"] > 0 and a["pos_lo"] > 0
        weak = a["pos_delta"] > 0 and not ok
        d = f"dMCC(B8-B1)={a['pos_delta']:+.3f} [{a['pos_lo']:+.3f},{a['pos_hi']:+.3f}]"
    elif label == "contamination":
        ok = (a["tata_auroc"] < 0.5 and a["tata_hi"] < 0.5 and a["tata_neg_excess"] > 0 and a["tata_excess_lo"] > 0)
        weak = a["tata_auroc"] < 0.5 and a["tata_neg_excess"] > 0 and not ok
        d = (f"TATA AUROC-0.5={a['tata_auroc']-0.5:+.3f} [{a['tata_lo']-0.5:+.3f},{a['tata_hi']-0.5:+.3f}]; "
             f"neg-excess={a['tata_neg_excess']:+.3f} [{a['tata_excess_lo']:+.3f},{a['tata_excess_hi']:+.3f}]")
    else:
        return "n/a", ""
    return ("preserved" if ok else ("weakened" if weak else "NOT preserved")), d


def diagnose_verdict(mcc, pdelta, plo, phi, r1, r3):
    import diagnose as DG
    card = dict(learnability=dict(mcc=mcc),
                route2_positional=dict(delta_mcc=pdelta if pdelta == pdelta else 0.0,
                                       significant=bool(pdelta == pdelta and not (plo <= 0 <= phi))))
    if r3 is not None:
        card["route1_motif"] = r1
        card["route3_negatives"] = r3
    return DG._verdict(card).split(" (")[0]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", default="id80,id50")
    a = ap.parse_args()
    ths = [t for t in a.thresholds.split(",") if t]
    R = _hc.RES
    # per-task clustering checkpoints are the source of truth (the aggregate CSV is written only at the
    # end of a full run_cluster.py pass)
    import glob
    red = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(_hc.CKPT_DIR, "cluster_*.csv")))],
                    ignore_index=True)
    ev = pd.read_csv(os.path.join(R, f"{_hc.PREFIX}_eval.csv"))
    pos_p = os.path.join(R, f"{_hc.PREFIX}_positional.csv")
    posr = pd.read_csv(pos_p) if os.path.exists(pos_p) else pd.DataFrame(columns=["task", "data"])
    sweep = pd.read_csv(os.path.join(R, "positional_sweep.csv")).set_index("task")
    cs = pd.read_csv(os.path.join(_hc.ROOT, "results", "cross_suite_summary.csv")).set_index("task")
    tbp = C.load_tbp()
    curated_by = {"vertebrates": resolve_curated(CURATED_VERT, C.motif_jaspar.load_pssms("vertebrates"))[0],
                  "insects": resolve_curated(CURATED_INSECT, C.motif_jaspar.load_pssms("insects"))[0]}

    def pos_of(task, dv):
        g = posr[(posr["task"] == task) & (posr["data"] == dv)]
        if len(g):
            r = g.iloc[0]
            return float(r["delta_mcc"]), float(r["delta_lo"]), float(r["delta_hi"]), "E3"
        if dv == "original" and task in sweep.index:
            r = sweep.loc[task]; lo, hi = [float(x) for x in r["delta_ci"].strip("[]").split(",")]
            return float(r["delta_mcc"]), lo, hi, "positional_sweep.csv"
        return np.nan, np.nan, np.nan, "missing"

    rows = []
    for task in _hc.TASKS:
        e = ev[ev["task"] == task]
        if not len(e):
            print(f"[{task}] no eval rows -> skipped"); continue
        o = e[e["protocol"] == "original"].iloc[0]
        binary = bool(len(np.unique(_hc.load_task(task)[1])) == 2)
        promoter = "promoter" in task
        for dv in ["original"] + ths:
            if dv != "original" and not os.path.exists(os.path.join(_hc.CLU_DIR, f"{task}_{dv}.npz")):
                continue
            d = diag_block(task, dv, tbp, curated_by)
            if dv == "original":
                m = dict(mcc=o["mcc"], mcc_lo=o["mcc_lo"], mcc_hi=o["mcc_hi"], mcc_source="original (our repro)")
            else:
                rs = e[(e["protocol"] == "resplit_cluster") & (e["threshold"] == dv)]
                if not len(rs) or rs.iloc[0]["status"] != "ok":
                    m = dict(mcc=np.nan, mcc_lo=np.nan, mcc_hi=np.nan, mcc_source=f"resplit {dv} infeasible")
                else:
                    r = rs.iloc[0]; m = dict(mcc=r["mcc"], mcc_lo=r["mcc_lo"], mcc_hi=r["mcc_hi"], mcc_source=f"resplit {dv}")
            if dv == "original":
                pm = dict(purge_mcc=o["mcc"], purge_mcc_lo=o["mcc_lo"], purge_mcc_hi=o["mcc_hi"])
            else:
                pu = e[(e["protocol"] == "test_purge") & (e["threshold"] == dv)]
                pm = (dict(purge_mcc=pu.iloc[0]["std_mcc"], purge_mcc_lo=pu.iloc[0]["std_mcc_lo"],
                           purge_mcc_hi=pu.iloc[0]["std_mcc_hi"], purge_raw_mcc=pu.iloc[0]["mcc"])
                      if len(pu) and pu.iloc[0]["status"] == "ok" else
                      dict(purge_mcc=np.nan, purge_mcc_lo=np.nan, purge_mcc_hi=np.nan, purge_raw_mcc=np.nan))
            m.update(pm)
            pdl, plo, phi, psrc = pos_of(task, dv)
            row = dict(task=task, data=dv, binary=binary, **m, pos_delta=pdl, pos_lo=plo, pos_hi=phi, pos_source=psrc,
                       **{k: v for k, v in d.items() if k not in ("task", "data")})
            row["label"] = (label_rule(row["mcc"], pdl, plo, binary, promoter, row.get("tata_neg_excess", np.nan),
                                       row.get("tata_hi", np.nan)) if row["mcc"] == row["mcc"] else "n/a")
            row["label_purgeMCC"] = (label_rule(row["purge_mcc"], pdl, plo, binary, promoter, row.get("tata_neg_excess", np.nan),
                                                row.get("tata_hi", np.nan)) if row["purge_mcc"] == row["purge_mcc"] else "n/a")
            r1 = dict(max_pos_motif_auroc=row.get("vert_max_pos_motif_auroc", row["max_pos_motif_auroc"]),
                      motif_richness=row.get("vert_motif_richness", row["motif_richness"]))
            r3 = (dict(tata_neg_excess=row["tata_neg_excess"], tata_motif_auroc=row["tata_auroc"],
                       cleaning_moves_to_chance=bool(row["cleaning_moves_to_chance"])) if binary else None)
            row["diagnose_verdict"] = (diagnose_verdict(row["mcc"], pdl, plo, phi, r1, r3)
                                       if row["mcc"] == row["mcc"] else "n/a")
            rows.append(row)
    rt = pd.DataFrame(rows)
    sdp = os.path.join(R, f"{_hc.PREFIX}_seeds_summary.csv")
    rt["resplit_seedmean_mcc"] = np.nan; rt["label_after_seedmean"] = ""
    if os.path.exists(sdp):
        sd = pd.read_csv(sdp)
        for i, r in rt.iterrows():
            g = sd[(sd.task == r["task"]) & (sd.threshold == r["data"])]
            if len(g) and int(g.iloc[0]["n_seeds"]) >= 3:
                mm = float(g.iloc[0]["mcc_cluster_mean"])
                rt.at[i, "resplit_seedmean_mcc"] = mm
                rt.at[i, "label_after_seedmean"] = label_rule(
                    mm, r["pos_delta"], r["pos_lo"], r["binary"], "promoter" in r["task"],
                    r.get("tata_neg_excess", np.nan), r.get("tata_hi", np.nan))
    # before-label vs published + preservation
    out = []
    for task in rt["task"].unique():
        g = rt[rt["task"] == task].set_index("data")
        before = g.loc["original"]
        for dv in ths:
            if dv not in g.index:
                continue
            aft = g.loc[dv]
            if aft["mcc"] != aft["mcc"]:
                st, dg = "n/a (re-split infeasible)", ""
            else:
                st, dg = preserved(before["label"], aft)
            _, dg0 = preserved(before["label"], before)
            aft_p = aft.copy(); aft_p["mcc"], aft_p["mcc_lo"], aft_p["mcc_hi"] = aft["purge_mcc"], aft["purge_mcc_lo"], aft["purge_mcc_hi"]
            pu_row = ev[(ev["task"] == task) & (ev["protocol"] == "test_purge") & (ev["threshold"] == dv)]
            pu_status = pu_row.iloc[0]["status"] if len(pu_row) else "missing"
            if aft_p["mcc"] == aft_p["mcc"]:
                st_p, dg_p = preserved(before["label"], aft_p)
            else:
                st_p, dg_p = f"n/a (purge not evaluable: {pu_status})", ""

            def strict(st_, lab_after):   # verifier addendum: pre-registered test AND label unchanged
                if st_.startswith("n/a"):
                    return st_
                if lab_after != before["label"]:
                    return f"NOT preserved (relabelled {before['label']}->{lab_after})"
                return st_
            sm_lab = aft["label_after_seedmean"] if isinstance(aft["label_after_seedmean"], str) else ""
            out.append(dict(task=task, threshold=dv, published_label=_hc.PAPER_ROUTE[task],
                            label_before=before["label"], rule_reproduces_published=(before["label"] == _hc.PAPER_ROUTE[task]),
                            label_after=aft["label"], relabelled=(aft["label"] != before["label"]),
                            preservation=st, preservation_strict=strict(st, aft["label"]),
                            diagnostic_before=dg0, diagnostic_after=dg,
                            purge_status=pu_status,
                            label_after_purgeMCC=aft["label_purgeMCC"], preservation_purgeMCC=st_p,
                            preservation_purgeMCC_strict=strict(st_p, aft["label_purgeMCC"]),
                            resplit_seedmean_mcc=aft["resplit_seedmean_mcc"], label_after_seedmean=aft["label_after_seedmean"],
                            label_unchanged_seedmean=(sm_lab == before["label"]) if sm_lab else np.nan,
                            diagnostic_after_purgeMCC=dg_p,
                            diagnose_verdict_before=before["diagnose_verdict"], diagnose_verdict_after=aft["diagnose_verdict"]))
    pres = pd.DataFrame(out)
    rt.to_csv(os.path.join(R, f"{_hc.PREFIX}_routes.csv"), index=False)
    pres.to_csv(os.path.join(R, f"{_hc.PREFIX}_route_labels.csv"), index=False)

    # ---------------- Route-1 correlation with homology-clean MCCs ----------------
    tf = [t for t in _hc.TASKS if t not in _hc.NON_TF]
    xs_pub = cs.loc[tf, "max_pos_motif_auroc"].astype(float)
    ycols = {"published_MCC": cs.loc[tf, "bench_mcc"].astype(float)}
    for proto, th, name, col in [("test_purge", "id80", "purge_id80_MCC", "mcc"),
                                 ("test_purge", "id80", "purge_id80_stdMCC", "std_mcc"),
                                 ("resplit_cluster", "id80", "resplit_id80_MCC", "mcc"),
                                 ("test_purge", "id50", "purge_id50_MCC", "mcc"),
                                 ("test_purge", "id50", "purge_id50_stdMCC", "std_mcc"),
                                 ("resplit_cluster", "id50", "resplit_id50_MCC", "mcc")]:
        vals = {}
        for t in tf:
            r = ev[(ev["task"] == t) & (ev["protocol"] == proto) & (ev["threshold"] == th)]
            vals[t] = float(r.iloc[0][col]) if len(r) and r.iloc[0]["status"] == "ok" else np.nan
        ycols[name] = pd.Series(vals)
    # verifier addendum: re-pooling control (random re-split, no homology control) and split-seed means
    rn_ = {t: float(ev[(ev["task"] == t) & (ev["protocol"] == "resplit_random")].iloc[0]["mcc"]) for t in tf}
    ycols["resplit_random_MCC"] = pd.Series(rn_)
    if os.path.exists(sdp):
        sd_ = pd.read_csv(sdp)
        for th in ths:
            g_ = sd_[sd_.threshold == th].set_index("task")
            ycols[f"resplit_{th}_seedmean_MCC"] = g_.reindex(tf)["mcc_cluster_mean"].astype(float)
        ycols["resplit_random_seedmean_MCC"] = sd_[sd_.threshold == ths[0]].set_index("task").reindex(tf)[
            "mcc_random_mean"].astype(float)
    xs_nr = rt[(rt["data"] == "id80")].set_index("task").reindex(tf)["max_pos_motif_auroc"].astype(float)
    xs_ori = rt[(rt["data"] == "original")].set_index("task").reindex(tf)["max_pos_motif_auroc"].astype(float)
    SUBSETS = {"all_TF": [],
               "excl_nt_enhancers_types": ["nt_enhancers_types"],
               "excl_NT_enhancer_tasks": ["nt_enhancers", "nt_enhancers_types"],
               "paper_9task_subset": ["human_nontata_promoters", "nt_enhancers"]}

    def spear(xs, ys, drop):
        dfc = pd.DataFrame({"x": xs, "y": ys.reindex(tf)}).drop(index=drop).dropna()
        if len(dfc) < 4:
            return None
        rho, p = spearmanr(dfc["x"], dfc["y"])
        jk = [spearmanr(dfc["x"].drop(t), dfc["y"].drop(t))[0] for t in dfc.index]
        jp = [spearmanr(dfc["x"].drop(t), dfc["y"].drop(t))[1] for t in dfc.index]
        return dict(n=len(dfc), rho=round(float(rho), 4), p=round(float(p), 4),
                    jack_rho_lo=round(float(min(jk)), 4), jack_rho_hi=round(float(max(jk)), 4),
                    jack_all_sig=bool(max(jp) < 0.05),
                    tasks_missing=",".join(sorted(set(tf) - set(dfc.index) - set(drop))))
    r1rows = []
    for xname, xs in [("published_disc", xs_pub), ("recomputed_disc_all", xs_ori), ("recomputed_disc_NR_id80", xs_nr)]:
        for sub, drop in SUBSETS.items():
            if xname != "published_disc" and sub != "all_TF":
                continue          # sensitivity subsets only for the published x (limits multiplicity)
            for yname, ys in ycols.items():
                s = spear(xs, ys, drop)
                if s is None:
                    continue
                r1rows.append(dict(x=xname, y=yname, subset=sub, **s,
                                   primary=bool(xname == "published_disc" and yname == "resplit_id80_MCC"
                                                and sub == "all_TF"),
                                   role=("PRIMARY (pre-registered MCC protocol)"
                                         if (xname == "published_disc" and yname == "resplit_id80_MCC" and sub == "all_TF")
                                         else "re-pooling control" if yname.startswith("resplit_random")
                                         else "reference" if yname == "published_MCC" else "sensitivity")))
    r1 = pd.DataFrame(r1rows)
    r1.to_csv(os.path.join(R, f"{_hc.PREFIX}_route1.csv"), index=False)

    # ---------------- final table ----------------
    trows = []
    for task in _hc.TASKS:
        e = ev[ev["task"] == task]
        if not len(e):
            continue
        rd = red[red["task"] == task].set_index("threshold")
        o = e[e["protocol"] == "original"].iloc[0]
        rr = dict(task=task, suite=("NT" if task.startswith("nt_") else "GB"),
                  n_train=int(o["n_train_orig"]), n_test=int(o["n_test_orig"]), k=int(o["k"]),
                  exact_dup_test=(rd.iloc[0]["frac_test_exact_in_train"] if len(rd) else np.nan),
                  published_mcc=o["published_mcc"], original_mcc=o["mcc"], original_mcc_lo=o["mcc_lo"], original_mcc_hi=o["mcc_hi"])
        for th in ths:
            if th in rd.index:
                rr[f"red_test_{th}"] = rd.loc[th, "frac_test_in_train_cluster"]
                rr[f"within_train_red_{th}"] = round(1 - rd.loc[th, "train_clusters_per_seq"], 5)
            pu = e[(e["protocol"] == "test_purge") & (e["threshold"] == th)]
            if len(pu):
                p = pu.iloc[0]
                rr.update({f"purge_{th}_status": p["status"], f"purge_{th}_kept": p["frac_test_kept"],
                           f"purge_{th}_kept_per_class": p.get("kept_per_class", np.nan),
                           f"purge_{th}_acc_removed": p.get("acc_removed", np.nan),
                           f"purge_{th}_acc_kept": p.get("acc_kept", np.nan)})
            if len(pu) and pu.iloc[0]["status"] == "ok":
                p = pu.iloc[0]
                rr.update({f"purge_{th}_mcc": p["mcc"], f"purge_{th}_lo": p["mcc_lo"], f"purge_{th}_hi": p["mcc_hi"],
                           f"purge_{th}_delta": p["delta_mcc"], f"purge_{th}_delta_lo": p["delta_lo"],
                           f"purge_{th}_delta_hi": p["delta_hi"], f"purge_{th}_homology_effect": p["homology_effect"],
                           f"purge_{th}_p_null": p["p_null"],
                           f"purge_{th}_homology_effect_strat": p.get("homology_effect_strat", np.nan),
                           f"purge_{th}_p_null_strat": p.get("p_null_strat", np.nan),
                           f"purge_{th}_inflated_prereg": bool(p["homology_inflated"]),
                           f"purge_{th}_std_mcc": p["std_mcc"],
                           f"purge_{th}_std_lo": p["std_mcc_lo"], f"purge_{th}_std_hi": p["std_mcc_hi"]})
            rs = e[(e["protocol"] == "resplit_cluster") & (e["threshold"] == th)]
            if len(rs) and rs.iloc[0]["status"] == "ok":
                s = rs.iloc[0]
                rr.update({f"resplit_{th}_mcc": s["mcc"], f"resplit_{th}_lo": s["mcc_lo"], f"resplit_{th}_hi": s["mcc_hi"],
                           f"resplit_{th}_delta_vs_orig": s["delta_vs_original"],
                           f"resplit_{th}_delta_vs_random": s["delta_vs_random"],
                           f"resplit_{th}_dvr_lo": s["delta_vs_random_lo"], f"resplit_{th}_dvr_hi": s["delta_vs_random_hi"],
                           f"resplit_{th}_inflated_prereg": bool(s["homology_inflated"]),
                           f"resplit_{th}_frac_test_from_orig_test": s["frac_test_from_orig_test"]})
                if os.path.exists(sdp):
                    g_ = sd[(sd.task == task) & (sd.threshold == th)]
                    if len(g_):
                        q_ = g_.iloc[0]
                        rr.update({f"resplit_{th}_dvr_seedmean": q_["delta_mean"], f"resplit_{th}_dvr_seedmin": q_["delta_min"],
                                   f"resplit_{th}_dvr_seedmax": q_["delta_max"],
                                   f"resplit_{th}_seedmin_mcc": q_["mcc_cluster_min"],
                                   f"resplit_{th}_seedmax_mcc": q_["mcc_cluster_max"],
                                   f"resplit_{th}_inflated_seed_robust": bool(s["homology_inflated"]) and float(q_["delta_max"]) < 0})
            elif len(rs):
                rr[f"resplit_{th}_mcc"] = np.nan; rr[f"resplit_{th}_status"] = rs.iloc[0]["status"]
            rr[f"inflated_prereg_{th}"] = bool(rr.get(f"purge_{th}_inflated_prereg", False)
                                               or rr.get(f"resplit_{th}_inflated_prereg", False))
            lb = pres[(pres["task"] == task) & (pres["threshold"] == th)]
            if len(lb):
                rr[f"label_after_{th}"] = lb.iloc[0]["label_after"]; rr[f"preservation_{th}"] = lb.iloc[0]["preservation"]
                rr[f"preservation_strict_{th}"] = lb.iloc[0]["preservation_strict"]
                rr[f"label_after_purge_{th}"] = lb.iloc[0]["label_after_purgeMCC"]
                rr[f"resplit_{th}_seedmean_mcc"] = lb.iloc[0]["resplit_seedmean_mcc"]
                rr[f"label_after_seedmean_{th}"] = lb.iloc[0]["label_after_seedmean"]
                rr[f"preservation_purge_{th}"] = lb.iloc[0]["preservation_purgeMCC"]
                rr[f"preservation_purge_strict_{th}"] = lb.iloc[0]["preservation_purgeMCC_strict"]
        rn = e[e["protocol"] == "resplit_random"]
        if len(rn):
            rr["resplit_random_mcc"] = rn.iloc[0]["mcc"]
        lb = pres[pres["task"] == task]
        rr["published_label"] = _hc.PAPER_ROUTE[task]
        rr["label_before"] = lb.iloc[0]["label_before"] if len(lb) else ""
        trows.append(rr)
    tab = pd.DataFrame(trows)
    tab.to_csv(os.path.join(R, f"{_hc.PREFIX}_table.csv"), index=False)
    print(pres.to_string())
    print(r1.to_string())
    print(f"wrote routes/route_labels/route1/table csv under {R}")


if __name__ == "__main__":
    main()
