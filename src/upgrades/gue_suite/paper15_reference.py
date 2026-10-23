#!/usr/bin/env python3
"""POST HOC reference on the paper's own 15 tasks (9 Genomic Benchmarks + 6 NT), computed with EXACTLY
the GUE driver's code (run_gue.fit_proba / metrics_boot / comp_fraction / pos_arm / cap_split /
motif_block and diagnose._verdict), so GUE can be read against the two suites already in the paper on
the same scale.

Written 2026-09-27, after the first GUE cards (human + mouse families) and BEFORE this script was run.
Two descriptive questions, no decision rule:
 Q1  COMPOSITION FLAG. How often does the pre-registered composition-dominance flag (composition
     fraction >= 0.80 and full AUROC >= 0.60) fire on GB / NT, including tasks the paper labels
     'solved'? (If it fires on solved tasks, the flag marks composition-solvability, not contamination.)
 Q2  RULE CALIBRATION. The GUE route labels come from diagnose._verdict with the paper's thresholds.
     Does that rule, fed the same driver's readouts, reproduce the paper's Table-1 route labels on the
     paper's own tasks? Agreement is scored against the submitted Table 1 (bibm/paper_bibm_submitted.tex,
     tab:land), mapping 'solved' <- SOLVED*, 'positional' <- ROUTE 2, 'contamination' / 'contamination
     (weak)' <- ROUTE 3, 'shared-motif' <- ROUTE 1; the Table-1 label 'composition' is a model-free
     task-type label that diagnose._verdict cannot emit (reported separately, not scored as a miss).

Per task: provided train/test split (data.load_dataset; train/test counts asserted against
results/results*.csv), k = the paper's validation-selected k (results/val_selection.csv new_k; identical
to what run_gue.select_k would pick, which is asserted by tests/test_gue.py on nt_promoter_tata).
 * learnability: k-mer LightGBM on the full train split; its MCC must reproduce results*.csv
   (lgbm, selected_by_val=1) to 5e-4 (flag column mcc_reproduces_paper);
 * composition-only LightGBM + composition fraction (binary: diagnose definition; multiclass: the
   macro-OvR extension), paired 1000-resample bootstrap;
 * Route 2: B=1 vs B=8 at k, train/test capped at 20,000 exactly as run_positional_sweep / run_gue;
 * Route 1 / Route 3 (binary): run_gue.motif_block on the pooled train+test sequences capped at 4,000
   per class (RandomState(42)); panel = curated vertebrate (diagnose default) for every task except
   drosophila_enhancers_stark, which gets the curated INSECT panel as in the paper (vertebrate panel kept
   as a sensitivity column);
 * verdict = diagnose._verdict on those readouts (equalized split and redundancy are not verdict inputs
   and are not recomputed here; the paper reports them separately).
Output: results/upgrades/gue_paper15_reference.csv (checkpoint per task under
results/upgrades/gue_paper15_reference_parts/). CPU only, LightGBM on GUE_THREADS threads, one task in
memory at a time (peak ~2.5 GB on human_ensembl_regulatory).
"""
import os

os.environ.setdefault("GUE_THREADS", "3")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_gue as R  # noqa: E402

C, DG = R.C, R.DG
from run_discriminability import CURATED_INSECT, CURATED_VERT, resolve_curated  # noqa: E402

TASKS = ["demo_human_or_worm", "human_ensembl_regulatory", "human_nontata_promoters",
         "demo_coding_vs_intergenomic_seqs", "human_enhancers_ensembl", "dummy_mouse_enhancers_ensembl",
         "human_enhancers_cohn", "human_ocr_ensembl", "drosophila_enhancers_stark",
         "nt_promoter_all", "nt_promoter_tata", "nt_promoter_no_tata", "nt_enhancers",
         "nt_enhancers_types", "nt_splice_sites_all"]
PAPER_ROUTE = {  # submitted BIBM Table 1 (bibm/paper_bibm_submitted.tex, tab:land)
    "demo_human_or_worm": "composition", "human_ensembl_regulatory": "solved",
    "nt_promoter_all": "solved", "nt_promoter_no_tata": "solved", "human_nontata_promoters": "solved",
    "nt_promoter_tata": "solved", "demo_coding_vs_intergenomic_seqs": "composition",
    "human_enhancers_ensembl": "contamination (weak)", "dummy_mouse_enhancers_ensembl": "composition",
    "nt_enhancers": "contamination", "nt_enhancers_types": "shared-motif",
    "human_enhancers_cohn": "contamination", "human_ocr_ensembl": "shared-motif",
    "drosophila_enhancers_stark": "shared-motif", "nt_splice_sites_all": "positional"}
OUT = os.path.join(C.RESULTS_DIR, "gue_paper15_reference.csv")
PARTS = os.path.join(C.RESULTS_DIR, "gue_paper15_reference_parts")


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


def agree(paper, rule):
    if paper == "composition":
        return None                      # model-free label; not emitted by diagnose._verdict
    return (paper.split(" ")[0] == rule)


def main():
    os.makedirs(PARTS, exist_ok=True)
    ref = pd.concat([pd.read_csv(os.path.join(C.ROOT, "results", f)) for f in ("results.csv", "results_nt.csv")])
    ref = ref[(ref.model == "lgbm") & (ref.selected_by_val == 1)].set_index("dataset")
    kmap = pd.read_csv(os.path.join(C.ROOT, "results", "val_selection.csv")).set_index("dataset").new_k
    vert = None; ins = None
    rows = []
    for task in TASKS:
        part = os.path.join(PARTS, f"{task}.json")
        if os.path.exists(part):
            rows.append(json.load(open(part))); print(f"[{task}] cached", flush=True); continue
        t0 = time.time()
        tr, ytr, te, yte = C.load_original(task)
        assert len(ytr) == int(ref.loc[task, "n_train"]) and len(yte) == int(ref.loc[task, "n_test"]), task
        k = int(kmap[task]); nc = len(np.unique(ytr)); binary = nc == 2
        cls, pr = R.fit_proba(R.kspec(tr, k), ytr, R.kspec(te, k))
        learn, pred_g = R.metrics_boot(yte, cls, pr, nc)
        ccls, cpr = R.fit_proba(C.comp_signature(tr, ks=(1, 2)), ytr, C.comp_signature(te, ks=(1, 2)))
        comp, _ = R.metrics_boot(yte, ccls, cpr, nc)
        if binary:
            c1 = int(np.where(cls == 1)[0][0]); cc1 = int(np.where(ccls == 1)[0][0])
            cf = R.comp_fraction(yte, pr[:, c1], cpr[:, cc1], 2)
        else:
            cf = R.comp_fraction(yte, pr, cpr, nc)
        # Route 2 (as run_gue)
        ptr, pytr, cap1 = R.cap_split(tr, ytr, R.POS_CAP, R.SEED)
        pte, pyte, cap2 = R.cap_split(te, yte, R.POS_CAP, R.SEED + 1)
        if cap1 or cap2:
            c_, p_ = R.fit_proba(R.kspec(ptr, k), pytr, R.kspec(pte, k)); pg = c_[np.argmax(p_, axis=1)]
        else:
            pg = pred_g
        c_, p_ = R.fit_proba(R.pos_arm(ptr, k, R.N_BINS), pytr, R.pos_arm(pte, k, R.N_BINS))
        pp = c_[np.argmax(p_, axis=1)]
        rng = np.random.RandomState(R.SEED); n = len(pyte); dd = np.empty(R.BOOT)
        for b in range(R.BOOT):
            idx = rng.randint(0, n, n)
            dd[b] = matthews_corrcoef(pyte[idx], pp[idx]) - matthews_corrcoef(pyte[idx], pg[idx])
        mg, mp = float(matthews_corrcoef(pyte, pg)), float(matthews_corrcoef(pyte, pp))
        dlo, dhi = np.percentile(dd, [2.5, 97.5])
        card = dict(learnability=dict(mcc=learn["mcc"]),
                    route2_positional=dict(delta_mcc=round(mp - mg, 4), delta_ci=[round(float(dlo), 4), round(float(dhi), 4)],
                                           significant=bool(not (dlo <= 0 <= dhi))))
        row = dict(task=task, suite="NT" if task.startswith("nt_") else "GB", paper_route=PAPER_ROUTE[task],
                   n_classes=nc, n_train=len(ytr), n_test=len(yte), k=k,
                   mcc=round(learn["mcc"], 6), mcc_lo=round(learn["mcc_ci"][0], 6), mcc_hi=round(learn["mcc_ci"][1], 6),
                   auroc=round(learn["auroc"], 6), paper_mcc=float(ref.loc[task, "mcc"]),
                   mcc_reproduces_paper=bool(abs(learn["mcc"] - float(ref.loc[task, "mcc"])) < 5e-4),
                   comp_mcc=round(comp["mcc"], 6), comp_mcc_lo=round(comp["mcc_ci"][0], 6),
                   comp_mcc_hi=round(comp["mcc_ci"][1], 6), comp_auroc=round(comp["auroc"], 6),
                   comp_fraction=cf["value"], comp_fraction_lo=cf["ci"][0], comp_fraction_hi=cf["ci"][1],
                   comp_fraction_definition=cf["definition"],
                   composition_dominated=bool(cf["full_auroc"] >= R.COMP_FLAG_MIN_AUROC and cf["value"] >= R.COMP_FLAG_FRAC),
                   pos_spectrum_mcc=round(mg, 4), pos_binned_mcc=round(mp, 4), pos_delta=round(mp - mg, 4),
                   pos_delta_lo=round(float(dlo), 4), pos_delta_hi=round(float(dhi), 4),
                   pos_significant=card["route2_positional"]["significant"], pos_capped=bool(cap1 or cap2))
        if binary:
            seqs_all = tr + te; y_all = np.concatenate([ytr, yte]); mrng = np.random.RandomState(R.SEED)
            pos = [seqs_all[i] for i in np.where(y_all == 1)[0]]; neg = [seqs_all[i] for i in np.where(y_all == 0)[0]]
            if len(pos) > R.MOTIF_CAP:
                pos = [pos[i] for i in mrng.choice(len(pos), R.MOTIF_CAP, replace=False)]
            if len(neg) > R.MOTIF_CAP:
                neg = [neg[i] for i in mrng.choice(len(neg), R.MOTIF_CAP, replace=False)]
            if vert is None:
                vert, _ = resolve_curated(CURATED_VERT, C.motif_jaspar.load_pssms("vertebrates"))
            r1v, r3, _ = R.motif_block(pos, neg, vert, np.random.RandomState(R.SEED))
            if task == "drosophila_enhancers_stark":
                if ins is None:
                    ins, _ = resolve_curated(CURATED_INSECT, C.motif_jaspar.load_pssms("insects"))
                r1, _, _ = R.motif_block(pos, neg, ins, np.random.RandomState(R.SEED)); panel = "curated insect"
                alt = dict(card, route1_motif=r1v, route3_negatives=r3)
                row["verdict_vertebrate_panel"] = DG._verdict(alt)
            else:
                r1, panel = r1v, "curated vertebrate"
            card["route1_motif"] = r1; card["route3_negatives"] = r3
            row.update(motif_panel=panel, motif_max_auroc=r1["max_pos_motif_auroc"], motif_best_tf=r1["best_TF"],
                       motif_richness=r1["motif_richness"], tata_neg_excess=r3["tata_neg_excess"],
                       tata_auroc=r3["tata_motif_auroc"], tata_cleaned_auroc=r3["cleaned_tata_auroc"],
                       cleaning_moves_to_chance=r3["cleaning_moves_to_chance"], gc_auroc=r3["gc_auroc"])
        v = DG._verdict(card)
        row.update(verdict=v, rule_route=short(v), rule_agrees_with_table1=agree(PAPER_ROUTE[task], short(v)),
                   tier="solved" if learn["mcc"] >= 0.7 else ("partial" if learn["mcc"] >= 0.5 else "hard"),
                   seconds=round(time.time() - t0, 1), posthoc=True)
        json.dump(row, open(part, "w"), indent=2)
        rows.append(row)
        print(f"[{task}] k={k} MCC {row['mcc']:.4f} (paper {row['paper_mcc']:.4f}, repro={row['mcc_reproduces_paper']}) "
              f"comp frac {row['comp_fraction']:+.3f} flag={row['composition_dominated']} pos delta {row['pos_delta']:+.4f} "
              f"| rule: {row['rule_route']} vs Table 1: {row['paper_route']} ({row['seconds']:.0f}s)", flush=True)
        del tr, te
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    sc = df[df.rule_agrees_with_table1.notna()]
    print(f"rule agrees with Table 1 on {int(sc.rule_agrees_with_table1.sum())}/{len(sc)} scorable tasks")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
