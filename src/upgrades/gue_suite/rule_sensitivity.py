#!/usr/bin/env python3
"""POST HOC PH4 (fix round 1): how fragile are the route labels to the rule's precedence and to its
length-dependent Route-1 threshold? No model is fitted: diagnose._verdict is re-applied to the stored
readouts (GUE cards; the paper's 15 tasks from gue_paper15_reference_parts).

Written 2026-09-27 (fix round 1) BEFORE it was run. Motivation: the first report attributed all 8 GUE
'hard' labels to the Route-1 richness threshold (> 0.2 hits/sequence) being unreachable on 70-101 bp
sequences; a verifier showed that explains at most 7, and that length-rescaling would also demote EMP
'shared-motif' labels. It also counted the Route-3 SIGNATURE only where the Route-3 LABEL fired,
hiding tasks where the MCC >= 0.7 branch pre-empts it.

S0  Reproduce: diagnose._verdict on the stored readouts must return the stored verdict (asserted).
S1  Signatures without precedence (per task, binary only): Route-3 signature = TATA neg-excess >= 0.10
    AND TATA AUROC < 0.5 AND cleaning moves it toward chance (the exact Route-3 clause of _verdict);
    Route-1 signature = richness > 0.2 AND max motif AUROC < 0.6; Route-2 signature = significant
    delta > 0.2. Reported next to the label.
S2  Why 'hard': for every task labelled hard, which clauses fail (MCC < 0.7; Route 2; each Route-3
    condition; Route-1 richness; Route-1 motif AUROC).
S3  Length-normalised Route-1 threshold: richness is re-expressed per L_ref bp,
    r' = r x L_ref / mean length of the task's released sequences, the 0.2 threshold is kept, and
    _verdict is re-applied. L_ref in {200, 250, 300, 400, 500} and L_ref = the median of the paper's 15
    task mean lengths (the development suites). Reported: per task label under each L_ref; per suite the
    number of labels that change (by direction) and, for the paper's 15, agreement with the submitted
    Table 1. Descriptive; no L_ref is adopted.
S4  Solved-boundary fragility: tasks whose MCC 95 % CI contains 0.7 (the label could flip between
    'solved' and the next rule under resampling).
Mean lengths: GUE from the released CSVs (train+dev+test); paper tasks from
gue_overlap_other_tasks.csv (len_mean over train+test, written by gue_overlap.py). The richness pool is
a <= 4,000-per-class subsample of the same sequences, so its mean length can differ slightly (fixed-length
tasks: identical).
Outputs: results/upgrades/gue_rule_sensitivity.csv (one row per task, GUE + paper 15),
         results/upgrades/gue_rule_sensitivity_summary.csv (one row per suite x variant).
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UP = os.path.abspath(os.path.join(HERE, ".."))
for _d in (UP, os.path.join(UP, "diagnose")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import gue_data as G  # noqa: E402
import diagnose as DG  # noqa: E402

RES = os.path.join(G.ROOT, "results", "upgrades")
L_REFS = [200, 250, 300, 400, 500]


def short(v):
    if v.startswith("SOLVED"):
        return "solved"
    if v.startswith("ROUTE 2"):
        return "positional"
    if v.startswith("ROUTE 3"):
        return "contamination"
    if v.startswith("ROUTE 1"):
        return "shared-motif"
    return "hard (no single route)"


TABLE1_MAP = {"contamination (weak)": "contamination"}


def gue_cards():
    out = []
    for t in G.ORDER:
        c = json.load(open(os.path.join(RES, "gue_cards", f"{t}.json")))
        lens = []
        for s in ("train", "dev", "test"):
            df = pd.read_csv(os.path.join(G.task_dir(t), f"{s}.csv"), usecols=["sequence"])
            lens.append(df.sequence.astype(str).str.strip().str.len().to_numpy())
        out.append(dict(suite="GUE", task=t, card=c, len_mean=float(np.concatenate(lens).mean()),
                        mcc_lo=c["learnability"]["mcc_ci"][0], mcc_hi=c["learnability"]["mcc_ci"][1],
                        stored=c["verdict"], table1=None))
    return out


def paper_cards():
    oth = pd.read_csv(os.path.join(RES, "gue_overlap_other_tasks.csv")).set_index("other_task")
    out = []
    pdir = os.path.join(RES, "gue_paper15_reference_parts")
    for f in sorted(os.listdir(pdir)):
        p = json.load(open(os.path.join(pdir, f)))
        card = dict(learnability=dict(mcc=p["mcc"]),
                    route2_positional=dict(significant=p["pos_significant"], delta_mcc=p["pos_delta"]))
        if p.get("motif_richness") is not None:
            card["route1_motif"] = dict(max_pos_motif_auroc=p["motif_max_auroc"], motif_richness=p["motif_richness"])
            card["route3_negatives"] = dict(tata_neg_excess=p["tata_neg_excess"], tata_motif_auroc=p["tata_auroc"],
                                            cleaning_moves_to_chance=p["cleaning_moves_to_chance"])
        out.append(dict(suite=p["suite"], task=p["task"], card=card, len_mean=float(oth.loc[p["task"], "len_mean"]),
                        mcc_lo=p["mcc_lo"], mcc_hi=p["mcc_hi"], stored=p["verdict"],
                        table1=TABLE1_MAP.get(p["paper_route"], p["paper_route"])))
    return out


def main():
    items = gue_cards() + paper_cards()
    paper_len = [x["len_mean"] for x in items if x["suite"] in ("GB", "NT")]
    l_dev = float(np.median(paper_len))
    refs = L_REFS + ([round(l_dev, 1)] if round(l_dev, 1) not in L_REFS else [])   # no duplicate variant
    rows = []
    for x in items:
        c = x["card"]
        v0 = DG._verdict(c)
        assert v0 == x["stored"], (x["task"], v0, x["stored"])                      # S0
        r1, r3, p2 = c.get("route1_motif"), c.get("route3_negatives"), c["route2_positional"]
        mcc = c["learnability"]["mcc"]
        row = dict(suite=x["suite"], task=x["task"], len_mean=round(x["len_mean"], 2), mcc=round(mcc, 4),
                   mcc_lo=round(x["mcc_lo"], 4), mcc_hi=round(x["mcc_hi"], 4), route=short(v0),
                   table1_route=x["table1"],
                   route2_signature=bool(p2.get("significant") and p2.get("delta_mcc", 0) > 0.2))
        if r3:
            row.update(tata_neg_excess=r3["tata_neg_excess"], tata_auroc=r3["tata_motif_auroc"],
                       cleaning_moves_to_chance=bool(r3["cleaning_moves_to_chance"]),
                       route3_signature=bool(r3["tata_neg_excess"] >= 0.10 and r3["tata_motif_auroc"] < 0.5
                                             and r3["cleaning_moves_to_chance"]))
        else:
            row.update(route3_signature=None)
        if r1:
            row.update(motif_richness=r1["motif_richness"], motif_max_auroc=r1["max_pos_motif_auroc"],
                       route1_signature=bool(r1["motif_richness"] > 0.2 and r1["max_pos_motif_auroc"] < 0.6))
        else:
            row.update(route1_signature=None)
        # S2
        if row["route"].startswith("hard"):
            why = ["MCC<0.7"]
            why.append("R2: delta not >0.2 & significant" if not row["route2_signature"] else "")
            if r3:
                if r3["tata_neg_excess"] < 0.10:
                    why.append("R3: neg-excess<0.10")
                if r3["tata_motif_auroc"] >= 0.5:
                    why.append("R3: TATA AUROC>=0.5")
                if not r3["cleaning_moves_to_chance"]:
                    why.append("R3: cleaning not toward chance")
            else:
                why.append("R1/R3: not evaluable (multiclass)")
            if r1:
                if r1["motif_richness"] <= 0.2:
                    why.append("R1: richness<=0.2")
                if r1["max_pos_motif_auroc"] >= 0.6:
                    why.append("R1: motif AUROC>=0.6")
            row["hard_reasons"] = "; ".join(w for w in why if w)
            row["hard_only_because_richness"] = bool(r1 and r1["motif_richness"] <= 0.2
                                                     and r1["max_pos_motif_auroc"] < 0.6)
        # S3
        for L in refs:
            cc = copy.deepcopy(c)
            if cc.get("route1_motif"):
                cc["route1_motif"]["motif_richness"] = cc["route1_motif"]["motif_richness"] * L / x["len_mean"]
                row[f"richness_per_{L}bp"] = round(cc["route1_motif"]["motif_richness"], 4)
            row[f"route_Lref_{L}"] = short(DG._verdict(cc))
        # S4
        row["mcc_ci_contains_0.7"] = bool(x["mcc_lo"] <= 0.7 <= x["mcc_hi"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "gue_rule_sensitivity.csv"), index=False)

    srows = []
    for suite_name, d in (("GUE", df[df.suite == "GUE"]), ("GB+NT (paper 15)", df[df.suite != "GUE"])):
        base = dict(suite=suite_name, n=len(d),
                    n_route3_signature=int(d.route3_signature.fillna(False).astype(bool).sum()),
                    n_route3_label=int((d.route == "contamination").sum()),
                    route3_signature_tasks=", ".join(d[d.route3_signature.fillna(False).astype(bool)].task),
                    n_route1_signature=int(d.route1_signature.fillna(False).astype(bool).sum()),
                    n_route1_label=int((d.route == "shared-motif").sum()),
                    n_hard=int(d.route.str.startswith("hard").sum()),
                    n_hard_only_because_richness=int(d.get("hard_only_because_richness", pd.Series(dtype=bool))
                                                     .fillna(False).astype(bool).sum()),
                    n_mcc_ci_contains_07=int(d["mcc_ci_contains_0.7"].sum()),
                    mcc_ci_contains_07_tasks=", ".join(d[d["mcc_ci_contains_0.7"]].task),
                    L_dev_median_paper_len=round(l_dev, 1))
        for L in ["as registered"] + refs:
            col = "route" if L == "as registered" else f"route_Lref_{L}"
            r = dict(base, variant=("as registered" if L == "as registered" else f"richness per {L} bp"
                                    + (" (= median paper-15 length)" if L != "as registered" and abs(L - l_dev) < 0.05 else "")))
            ch = d[d[col] != d.route]
            r["n_changed"] = len(ch)
            r["changes"] = "; ".join(f"{t}: {a}->{b}" for t, a, b in zip(ch.task, ch.route, ch[col]))
            r["n_hard_to_shared_motif"] = int(((d.route.str.startswith("hard")) & (d[col] == "shared-motif")).sum())
            r["n_shared_motif_to_hard"] = int(((d.route == "shared-motif") & (d[col].str.startswith("hard"))).sum())
            r["counts"] = "; ".join(f"{k}:{v}" for k, v in d[col].value_counts().items())
            if suite_name != "GUE":
                sc = d[d.table1_route != "composition"]
                r["table1_agree"] = int((sc[col] == sc.table1_route).sum()); r["table1_scorable"] = len(sc)
            srows.append(r)
    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(RES, "gue_rule_sensitivity_summary.csv"), index=False)
    print(df.to_string(index=False, max_colwidth=60))
    print(sdf.drop(columns=["route3_signature_tasks", "mcc_ci_contains_07_tasks"]).to_string(index=False, max_colwidth=90))


if __name__ == "__main__":
    main()
