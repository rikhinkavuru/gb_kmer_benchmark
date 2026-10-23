#!/usr/bin/env python3
"""Figure for the multivariate covariate analysis (Reviewer 1, comment 1) -- supplement / rebuttal.
Reads ONLY results/upgrades/multivar_{partial,covariates,posthoc_family,posthoc_orientation_robustness,
posthoc_task_identity,posthoc_distinct_datasets,
posthoc_task_locus_links}.csv.
(a) partial Spearman rho(discriminability, MCC | c) on the 11 TF-motif tasks for each covariate c, 95%
    task-bootstrap CI (pre-registered A1), and beside it the reverse partial rho(c, MCC | disc) (post-hoc
    column) so the collinearity with GC mean / composition is visible;
(b) benchmark MCC vs discriminability for the 11 tasks, colour + marker = task family (promoter incl. the
    3-class regulatory task vs enhancer/OCR; post hoc P11 -- the same partition as Table I's 'solved' vs
    failing split), open markers = train/test near-duplicate fraction >= 0.05 (homology-leaky, P5),
    in-sample OLS line; the arrow moves nt_enhancers_types to its enhancer-oriented discriminability under
    the uniform functional-class rule (post hoc P8); a note states that the 11 tasks are 9 exact-duplicate-
    free sequence sets (P12) and how many promoter-task pairs share most promoter loci (P13; fix round 3;
    read from multivar_posthoc_task_identity.csv / _distinct_datasets.csv / _task_locus_links.csv).
IEEE single column (3.5 in), fonts 7 pt (panel-b footnote 6.5 pt), Okabe-Ito blue/vermillion (validated with the dataviz palette
checker: CVD dE 21.9) + marker-shape redundancy. Writes results/upgrades/figures/multivar_covariates.{pdf,png}."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RD = os.path.join(ROOT, "results", "upgrades")
OUT = os.path.join(RD, "figures")
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#6b6b6b"
COV_LABEL = {"len_log": "log length", "len_sd_log": "length SD", "gc_mean": "GC mean",
             "gc_sd": "GC SD (within class)", "gc_gap": "GC class gap |d|",
             "homology": "train/test homology", "ntrain_log": "log n_train", "multiclass": "3-class task",
             "pos_delta": "positional rescue", "comp_mcc": "composition-only MCC"}
TASK_LABEL = {"human_ensembl_regulatory": "ens. regulatory", "human_nontata_promoters": "nonTATA prom.",
              "human_enhancers_ensembl": "enh. Ensembl", "human_enhancers_cohn": "enh. Cohn",
              "drosophila_enhancers_stark": "Dros. Stark", "human_ocr_ensembl": "OCR Ensembl",
              "nt_promoter_all": "NT prom. all", "nt_promoter_no_tata": "NT prom. noTATA",
              "nt_promoter_tata": "NT prom. TATA", "nt_enhancers": "NT enh.",
              "nt_enhancers_types": "NT enh. types"}
LABELLED = {"human_enhancers_ensembl": (-0.004, 0.035, "right"), "nt_enhancers_types": (0.005, 0.0, "left"),
            "nt_enhancers": (-0.004, 0.03, "right"), "human_nontata_promoters": (-0.004, -0.06, "right"),
            "human_ensembl_regulatory": (0.0, 0.035, "center"), "drosophila_enhancers_stark": (0.004, -0.045, "left")}


def main():
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    pa = pd.read_csv(os.path.join(RD, "multivar_partial.csv"))
    pa = pa[pa["config"] == "S11/Y1_MCC"].sort_values("partial_rho").reset_index(drop=True)
    cov = pd.read_csv(os.path.join(RD, "multivar_covariates.csv"))
    cov = cov[cov["S11"]].reset_index(drop=True)
    fam = pd.read_csv(os.path.join(RD, "multivar_posthoc_family.csv"))
    fam = fam[fam["variant"] == "as_published"].set_index("analysis")
    prom = set(fam.loc["within_promoter_plus_regulatory_n5", "tasks"].split(";"))
    enh = set(fam.loc["within_enhancer_ocr_n6", "tasks"].split(";"))
    assert prom | enh == set(cov["task"]) and not prom & enh
    rob = pd.read_csv(os.path.join(RD, "multivar_posthoc_orientation_robustness.csv"))
    u1 = rob[(rob["config"] == "S11/Y1_MCC") & (rob["variant"] == "U1_functional_representative")].iloc[0]
    m = re.match(r"nt_enhancers_types=([0-9.]+) \(([^)]*)\)$", u1["changed"])
    assert m, u1["changed"]
    d_u1 = float(m.group(1))

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 5.3), gridspec_kw=dict(height_ratios=[1.3, 1.15], hspace=0.42))
    # (a) forest: disc | c (circles) and c | disc (triangles); filled = 95% CI excludes 0
    raw = float(pa["spearman_disc_y"].iloc[0])
    a.axvline(0, color="0.75", lw=0.6, zorder=0)
    a.axvline(raw, color=GREY, lw=0.8, ls="--", zorder=0)
    a.text(raw + 0.03, -0.75, f"raw $\\rho$ = {raw:+.2f}", color=GREY, ha="left", va="center", fontsize=7)
    off = 0.17
    for i, r in pa.iterrows():
        excl = r["partial_boot_lo"] > 0 or r["partial_boot_hi"] < 0
        a.plot([r["partial_boot_lo"], r["partial_boot_hi"]], [i + off] * 2, color=BLUE, lw=1.0, alpha=1 if excl else 0.55)
        a.plot(r["partial_rho"], i + off, "o", ms=4.0, mfc=BLUE if excl else "white", mec=BLUE, mew=0.9)
        rex = bool(r["posthoc_rev_partial_boot_excludes_0"])
        a.plot([r["posthoc_rev_partial_boot_lo"], r["posthoc_rev_partial_boot_hi"]], [i - off] * 2, color=VERM, lw=1.0,
               alpha=1 if rex else 0.55)
        a.plot(r["posthoc_rev_partial_rho_c_given_disc"], i - off, "^", ms=4.2, mfc=VERM if rex else "white", mec=VERM,
               mew=0.9)
    a.legend(handles=[Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, ms=4.0, lw=1.0,
                             label="$\\rho$(disc, MCC | covariate)"),
                      Line2D([], [], marker="^", ls="-", color=VERM, mfc=VERM, ms=4.2, lw=1.0,
                             label="$\\rho$(covariate, MCC | disc)"),
                      Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=4.0,
                             label="open = 95% CI includes 0")],
             loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.4,
             columnspacing=0.8, borderaxespad=0.1)
    a.set_yticks(range(len(pa)))
    a.set_yticklabels([COV_LABEL.get(c, c) for c in pa["covariate"]])
    a.set_xlim(-1.02, 1.02)
    a.set_ylim(-1.1, len(pa) + 1.7)
    a.set_xlabel("partial Spearman $\\rho$ with benchmark MCC, 95% task-bootstrap CI")
    a.set_title("(a) discriminability vs each covariate, each controlling for the other (n = 11)", fontsize=7,
                loc="left")
    # (b) MCC vs disc by family; open = homology-leaky
    x, y = cov["disc"].to_numpy(float), cov["bench_mcc"].to_numpy(float)
    sl, ic = np.polyfit(x, y, 1)
    xx = np.linspace(0.52, 0.77, 50)
    b.plot(xx, sl * xx + ic, color="0.6", lw=0.8, zorder=0)
    b.text(0.705, 0.765, "OLS fit (n = 11)", color=GREY, ha="left", va="top", fontsize=7)
    for _, r in cov.iterrows():
        pr = r["task"] in prom
        col, mk = (BLUE, "o") if pr else (VERM, "s")
        leaky = r["homology"] >= 0.05
        b.plot(r["disc"], r["bench_mcc"], mk, ms=5.0, mfc="white" if leaky else col, mec=col, mew=1.0, zorder=3)
        if r["task"] in LABELLED:
            dx, dy, ha = LABELLED[r["task"]]
            b.text(r["disc"] + dx, r["bench_mcc"] + dy, TASK_LABEL[r["task"]], fontsize=7, ha=ha, va="center",
                   color="0.2")
    k = cov.index[cov["task"] == "nt_enhancers_types"][0]
    b.annotate("", xy=(d_u1, y[k]), xytext=(x[k] - 0.004, y[k]),
               arrowprops=dict(arrowstyle="->", color=VERM, lw=0.8, ls="--"), zorder=2)
    b.text(0.615, y[k] + 0.012, "enhancer-oriented\n(post hoc)", color=VERM, fontsize=7, ha="center",
           va="bottom")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, mfc=BLUE, ms=5, label="promoter / regulatory"),
                      Line2D([], [], marker="s", ls="", color=VERM, mfc=VERM, ms=5, label="enhancer / OCR"),
                      Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=5,
                             label="open: $\\geq$5% near-dups")],
             loc="upper left", frameon=False, handlelength=1.0, borderaxespad=0.1)
    # fix round 2: the 11 points are not 11 independent sequence collections (audit_task_overlap.py, P12)
    ident = pd.read_csv(os.path.join(RD, "multivar_posthoc_task_identity.csv"))
    assert (ident["value"] == 1).all()
    d12 = pd.read_csv(os.path.join(RD, "multivar_posthoc_distinct_datasets.csv")).set_index("subset")
    nd = int(d12.loc["S11_all", "n_exact_distinct_sets"])
    b.set_xlabel("discriminability (max motif-only AUROC)")
    # fix round 3: exact-duplicate terminology, plus the locus-level sharing of the promoter tasks
    # (audit_locus_overlap.py; the sentence is printed only if the promoter tasks' locus links are >= 0.5)
    lk = pd.read_csv(os.path.join(RD, "multivar_posthoc_task_locus_links.csv"))
    prom_pairs = lk[lk["task_a"].isin(prom) & lk["task_b"].isin(prom)
                    & (lk["exact_shared_frac_max"] < 0.001)]            # promoter pairs WITHOUT exact sharing
    frac_hi = int((prom_pairs["locus_link"] >= 0.5).sum())
    assert frac_hi >= 3, prom_pairs
    b.annotate(f"The 11 tasks are {nd} exact-duplicate-free sequence sets (NT prom. all = TATA + noTATA\n"
               "tasks; NT enh. types relabels NT enh.; ens. regulatory's OCR class = OCR Ensembl positives),\n"
               f"and {frac_hi} of the {len(prom_pairs)} other promoter-task pairs share most promoter loci (31-mer audit).",
               xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -30), textcoords="offset points",
               ha="center", va="top", fontsize=6.5, color=GREY)
    b.set_ylabel("benchmark MCC (Table 1)")
    b.set_xlim(0.52, 0.77)
    b.set_ylim(0.3, 1.0)
    b.set_title("(b) MCC vs discriminability by task family (= Table I solved / failing)", fontsize=7, loc="left")
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"multivar_covariates.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}/multivar_covariates.pdf/.png")


if __name__ == "__main__":
    main()
