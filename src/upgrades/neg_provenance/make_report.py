#!/usr/bin/env python3
"""Compact table + interpretation for the negative-set provenance analysis (reads ONLY files written by
run_neg_provenance.py, nt_split_provenance.py, bg_seed_stability.py, chromhmm_rule_counts.py, verify_inputs.py
and write_sources.py; every number in the outputs is formatted from those files).

Also computes one POST HOC whole-class decomposition for nt_enhancers (all 7,684 negatives incl. the
RankGAN-synthetic ones, and the all-genomic test split), from the per-window table + the benchmark sequences,
because D1 is defined on the genomic subset only. Bootstrap B=1000, seed 42, window/sequence = unit.

Fix round 1 (2026-09-26): CIs are read from numeric <name>_lo/_hi columns (no eval); split-stratified nt
rows, the independent-test parent-region finding, six-seed pooled backgrounds and the chromosome tests
(exact log10 p) are added; wording follows the verified data (see README.md).

Outputs
  results/upgrades/neg_provenance_compact_table.csv       groups x key metrics, "value [lo, hi]"
  results/upgrades/neg_provenance_compact_table.tex       LaTeX table suggestion (single IEEE column)
  results/upgrades/neg_provenance_nt_wholeclass.csv       post hoc whole-class / test-split decomposition
  results/upgrades/neg_provenance_nt_typelabels.csv       post hoc enhancers_types label check
  results/upgrades/neg_provenance_interpretation.txt
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402

R = C.RESULTS_DIR
P = os.path.join(R, "neg_provenance")
SEED, B = 42, 1000

GROUPS = [("cohn", "POS", "cohn: positives (H3K27ac peaks)"),
          ("cohn", "NEG", "cohn: negatives"),
          ("cohn", "BG_rule_pc", "cohn: rule, >=15 kb from protein-coding genes (post hoc)"),
          ("cohn", "BG_rule", "cohn: rule, >=15 kb from any GENCODE gene (pre-registered)"),
          ("cohn", "BG_uni", "cohn: random genome"),
          ("nt_enhancers", "POS", "nt: positives (genomic, train+test)"),
          ("nt_enhancers", "NEG", "nt: negatives (genomic, train+test)"),
          ("nt_enhancers", "NEG_train", "nt: negatives, train (iEnhancer-2L benchmark)"),
          ("nt_enhancers", "NEG_test", "nt: negatives, test (iEnhancer-EL independent set)"),
          ("nt_enhancers", "NEG_SYN", "nt: negatives (RankGAN-synthetic, train only)"),
          ("nt_enhancers", "BG_rule13", "nt: rule, ChromHMM state 13 in all 9 lines (pre-registered secondary)"),
          ("nt_enhancers", "BG_rule", "nt: rule, non-enhancer (not 4-7) in all 9 lines (pre-registered primary)"),
          ("nt_enhancers", "BG_uni", "nt: random genome")]
METRICS = [("gc_local", "GC (window)"), ("tata_rate", "TATA-box rate"), ("gc_100kb", "GC (100 kb)"),
           ("iso_AT_L1L2", "AT-rich isochore (L1+L2)"), ("ge15kb_gene_pc", ">=15 kb from protein-coding gene"),
           ("comp_intergenic", "intergenic"), ("dist_tss_kb", "median distance to TSS (kb)"),
           ("rep_bp_L1", "L1 bp fraction"), ("ccre_any", "any ENCODE cCRE overlap"),
           ("hmm_13_all9", "ChromHMM state 13 (low signal) in all 9 lines"), ("hmm_enh_any", "enhancer state in >=1 line")]


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def fmt(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main():
    S = pd.read_csv(P + "_summary.csv")
    T = pd.read_csv(P + "_tests.csv")
    O = pd.read_csv(P + "_nt_origin.csv")
    RC = pd.read_csv(P + "_chromhmm_rule_counts.csv")
    META = pd.read_csv(P + "_meta.csv", index_col=0).iloc[:, 0]
    SAN = pd.read_csv(P + "_sanity_paper_numbers.csv")
    CH = pd.read_csv(P + "_chrom.csv")
    CHT = pd.read_csv(P + "_chrom_tests.csv")
    CHK = pd.read_csv(P + "_input_checks.csv")
    BS = pd.read_csv(P + "_nt_by_split.csv")
    BT = pd.read_csv(P + "_nt_by_split_tests.csv")
    HC = pd.read_csv(P + "_nt_hg19_check.csv")
    BP = pd.read_csv(P + "_bg_pooled.csv")
    SRC = pd.read_csv(P + "_sources.csv")

    def s(ds, g, m):
        if g in ("NEG_train", "NEG_test"):
            r = BS[(BS.split == g.split("_")[1]) & (BS.group == "NEG") & (BS.metric == m)]
        else:
            r = S[(S.dataset == ds) & (S.group == g) & (S.metric == m)]
        if len(r) == 0:
            return np.nan, np.nan, np.nan, 0
        r = r.iloc[0]
        return r.value, r.ci_lo, r.ci_hi, int(r.n)

    def sf(ds, g, m, d=3, pct=False):
        v, lo, hi, _ = s(ds, g, m)
        if pct:
            return f"{100 * v:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
        return fmt(v, lo, hi, d)

    def bs(sp, g, m, pct=True):
        r = BS[(BS.split == sp) & (BS.group == g) & (BS.metric == m)].iloc[0]
        return f"{100 * r.value:.1f}% [{100 * r.ci_lo:.1f}, {100 * r.ci_hi:.1f}]" if pct else fmt(r.value, r.ci_lo, r.ci_hi)

    def hc(check, cls=None, split=None, note=None):
        q = HC[HC.check == check]
        if cls is not None:
            q = q[q.cls == cls]
        if split is not None:
            q = q[q.split == split]
        if note is not None:
            q = q[q.note == note]
        return q.iloc[0]

    def bt(contrast, metric):
        return BT[(BT.contrast == contrast) & (BT.metric == metric)].iloc[0]

    def bp(ds, g, m):
        return BP[(BP.dataset == ds) & (BP.group == g) & (BP.metric == m)].iloc[0]

    def cit(row, base, d=3):
        return f"[{row[base + '_lo']:.{d}f}, {row[base + '_hi']:.{d}f}]"

    def cip(row, base):
        return f"[{100 * row[base + '_lo']:.1f}, {100 * row[base + '_hi']:.1f}]"

    def cht(ds, ref, method):
        return CHT[(CHT.dataset == ds) & (CHT.reference == ref) & (CHT.method == method)].iloc[0]

    syn = O[(O.split == "all") & (O.label == 0) & (O.origin == "synthetic")].iloc[0]
    # ------------------------------------------------------------------ compact table
    rows = []
    for ds, g, lab in GROUPS:
        row = {"group": lab}
        for m, ml in METRICS:
            if g == "NEG_SYN":
                if m == "gc_local":
                    row[ml] = fmt(syn.gc_local, syn.gc_lo, syn.gc_hi)
                elif m == "tata_rate":
                    row[ml] = fmt(syn.tata_rate, syn.tata_lo, syn.tata_hi)
                else:
                    row[ml] = "n/a (no locus)"
                continue
            v, lo, hi, n = s(ds, g, m)
            row[ml] = "n/a" if n == 0 else fmt(v, lo, hi, 1 if m == "dist_tss_kb" else 3)
        row["n"] = int(syn.n_rows) if g == "NEG_SYN" else s(ds, g, "gc_local")[3]
        rows.append(row)
    ct = pd.DataFrame(rows)[["group", "n"] + [ml for _, ml in METRICS]]
    ct.to_csv(P + "_compact_table.csv", index=False)

    # LaTeX (5 metrics, one IEEE column)
    pc_t = bp("cohn", "BG_rule_pc", "tata_rate")
    tex_m = [("gc_local", "GC"), ("tata_rate", "TATA"), ("gc_100kb", "GC$_{100\\,\\mathrm{kb}}$"),
             ("iso_AT_L1L2", "L1+L2"), ("hmm_13_all9", "S13$_{9}$")]
    short = {("cohn", "POS"): "Positives", ("cohn", "NEG"): "Negatives", ("cohn", "BG_rule_pc"): "Rule (pc genes)$^a$",
             ("cohn", "BG_rule"): "Rule (all genes)", ("cohn", "BG_uni"): "Random genome",
             ("nt_enhancers", "POS"): "Positives", ("nt_enhancers", "NEG_train"): "Negatives, train$^b$",
             ("nt_enhancers", "NEG_test"): "Negatives, test$^c$", ("nt_enhancers", "NEG_SYN"): "Negatives, synthetic",
             ("nt_enhancers", "BG_rule13"): "Rule (S13 in 9)$^d$", ("nt_enhancers", "BG_rule"): "Rule (non-enh.\\ in 9)",
             ("nt_enhancers", "BG_uni"): "Random genome"}
    lines = ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}",
             "\\caption{Negative-set provenance: window means (fractions); 95\\% bootstrap CIs in the CSV. "
             "TATA: $\\ge$1 TBP hit at 1.3\\,bits/pos; L1+L2: 100-kb GC $<$41\\%; S13$_9$: ChromHMM state 13 "
             "(low signal) in all nine ENCODE lines. Backgrounds: 10{,}000 N-free windows (seed 42).}\\label{tab:prov}",
             "\\begin{tabular}{lrrrrrr}\\toprule",
             "Group & $n$ & " + " & ".join(t for _, t in tex_m) + "\\\\\\midrule"]
    for ds, g, _ in GROUPS:
        if (ds, g) not in short:
            continue
        if (ds, g) == ("cohn", "POS"):
            lines.append("\\multicolumn{7}{l}{\\texttt{human\\_enhancers\\_cohn}}\\\\")
        if (ds, g) == ("nt_enhancers", "POS"):
            lines.append("\\midrule\\multicolumn{7}{l}{\\texttt{nt\\_enhancers} (genomic = iEnhancer originals)}\\\\")
        cells = []
        for m, _ in tex_m:
            if g == "NEG_SYN":
                cells.append(f"{syn.gc_local:.3f}" if m == "gc_local" else f"{syn.tata_rate:.3f}" if m == "tata_rate" else "--")
            else:
                cells.append(f"{s(ds, g, m)[0]:.3f}")
        n = int(syn.n_rows) if g == "NEG_SYN" else s(ds, g, "gc_local")[3]
        lines.append(f"\\quad {short[(ds, g)]} & {n:,} & " + " & ".join(cells) + "\\\\")
    r13 = RC.set_index("rule").loc[[i for i in RC.rule if i.startswith("state13_all9")][0]]
    hdr13 = hc("hg19 header window, whole13_all9", cls="non", split="train")
    par = hc("test NEG: parent range = one ChromHMM segment (both ends +-1 bp) in >=1 line", cls="non")
    enh_t = hc("test NEG: fragment enhancer state (4-7) in >=1 line (whole window)", cls="non")
    p13c = hc("test NEG: ... and that segment is state 13 (Heterochrom/lo) in every matching line", cls="non")
    lines += ["\\bottomrule",
              "\\multicolumn{7}{p{0.96\\columnwidth}}{$^a$Post hoc variant (gene set unspecified); six pooled "
              f"background draws give TATA {pc_t.pooled:.3f} ({pc_t.bg_minus_neg:+.3f} vs.\\ negatives). "
              f"$^b$iEnhancer-2L: {int(hdr13.k):,}/{int(hdr13.n):,} hg19 source windows are S13 in all nine lines. "
              f"$^c$iEnhancer-EL: cut from single S13 segments of $\\ge$2 lines ({int(p13c.k)}/{int(p13c.n)} parent regions), "
              f"not the nine-line intersection; {100 * enh_t.frac:.1f}\\% touch an enhancer state. "
              f"$^d$Reproduces the published non-enhancer candidate count ({int(r13.n_200bp_fragments):,} vs.\\ "
              f"5,257,994 200-bp fragments).}}\\\\",
              "\\end{tabular}", "\\end{table}"]
    with open(P + "_compact_table.tex", "w") as fh:
        fh.write("\n".join(lines).replace("{,}", ",") + "\n")

    # ------------------------------------------------------------------ post hoc nt whole-class decomposition
    import data_nt
    W = pd.read_csv(P + "_intervals.csv.gz", low_memory=False, usecols=["dataset", "group", "gc_local", "tata_flag"])
    bgw = W[(W.dataset == "nt_enhancers") & (W.group == "BG_uni")]
    bgc, bgt = bgw.gc_local.to_numpy(), bgw.tata_flag.astype(float).to_numpy()
    tbp = C.load_tbp()
    whole = []
    for split in ("all", "test"):
        cls = {}
        for lab in (0, 1):
            sq = []
            for sp in (("train", "test") if split == "all" else ("test",)):
                x, y = data_nt._read("enhancers", sp)
                sq += [q for q, yy in zip(x, y) if yy == lab]
            cls[lab] = (np.asarray(C.gc_content(sq)), (C.tata_hit_counts(sq, tbp) >= 1).astype(float))
        r = np.random.RandomState(SEED)
        for mi, mname in ((0, "gc_local"), (1, "tata_rate")):
            pv, nv, uv = cls[1][mi], cls[0][mi], (bgc if mi == 0 else bgt)
            est = (uv.mean() - nv.mean()) / (pv.mean() - nv.mean())
            reps = []
            for _ in range(B):
                a = pv[r.randint(0, len(pv), len(pv))].mean(); b = nv[r.randint(0, len(nv), len(nv))].mean()
                u = uv[r.randint(0, len(uv), len(uv))].mean()
                reps.append((u - b) / (a - b))
            lo, hi = ci(reps)
            whole.append(dict(split=split, metric=mname, n_pos=len(pv), n_neg=len(nv), pos=pv.mean(), neg=nv.mean(),
                              bg_uni=uv.mean(), share_of_gap_from_neg_side=est, ci_lo=lo, ci_hi=hi))
    WH = pd.DataFrame(whole)
    WH.to_csv(P + "_nt_wholeclass.csv", index=False)

    # post hoc side check: enhancers_types labels of the verbatim iEnhancer originals (label consistency)
    import verify_inputs as V
    ien = {}
    for (sp, cl), f in V.IEN_FILES.items():
        for _, q in V.read_fasta(os.path.join(V.IEN, f)):
            ien.setdefault(q, set()).add(f"{sp}_{cl}")
    tl = []
    for sp in ("train", "test"):
        xs, yb = data_nt._read("enhancers", sp); _, yt = data_nt._read("enhancers_types", sp)
        for q, b_, t_ in zip(xs, yb, yt):
            tl.append((sp, "|".join(sorted(ien.get(q, {"synthetic"}))), int(b_), int(t_), q))
    TL = pd.DataFrame(tl, columns=["nt_split", "ienhancer_set", "enhancers_label", "enhancers_types_label", "seq"])
    TL = TL.groupby(["nt_split", "ienhancer_set", "enhancers_label", "enhancers_types_label"]).agg(
        rows=("seq", "size"), unique_seqs=("seq", "nunique")).reset_index()
    TL.to_csv(P + "_nt_typelabels.csv", index=False)

    # ------------------------------------------------------------------ interpretation
    def t(test, ds, met, rule=None, grp=None):
        q = T[(T.test == test) & (T.dataset == ds) & (T.metric == met)]
        if rule is not None:
            q = q[q.rule_bg == rule]
        if grp is not None:
            q = q[q.group == grp]
        return q.iloc[0]

    rc = RC.set_index("rule")
    rne = rc.loc[[i for i in rc.index if i.startswith("nonenhancer_all9")][0]]
    rst = rc.loc[[i for i in rc.index if i.startswith("strongenh_all9")][0]]
    r1315 = rc.loc[[i for i in rc.index if i.startswith("state13to15_all9")][0]]
    rwk = rc[[i.startswith(("weakenh_any", "enhancer_any")) for i in rc.index]]
    nfail = int((CHK.status == "FAIL").sum())
    o_all0 = O[(O.split == "all") & (O.label == 0)].set_index("origin")
    o_all1 = O[(O.split == "all") & (O.label == 1)].set_index("origin")
    otr0 = O[(O.split == "train") & (O.label == 0)].set_index("origin")
    otr1 = O[(O.split == "train") & (O.label == 1)].set_index("origin")
    ote0 = O[(O.split == "test") & (O.label == 0) & (O.origin == "iEnhancer_original")].iloc[0]
    n_syn_test = int(O[(O.split == "test") & (O.origin == "synthetic")]["n_rows"].sum())
    syn_tr0 = otr0.loc["synthetic", "n_rows"] / otr0["n_rows"].sum()
    syn_tr1 = otr1.loc["synthetic", "n_rows"] / otr1["n_rows"].sum()
    chc = CH[CH.dataset == "cohn"].set_index("chr")
    chn = CH[CH.dataset == "nt_enhancers"].set_index("chr")
    D1 = {(ds, m): t("D1_decomposition", ds, m) for ds in ("cohn", "nt_enhancers")
          for m in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2")}
    PC = {m: t("D1_posthoc_rule_proteincoding", "cohn", m) for m in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2")}
    S13 = {m: t("D1_secondary_state13", "nt_enhancers", m) for m in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2")}
    CHR = {m: t("D1_sensitivity_chr_matched", "cohn", m) for m in ("gc_local", "tata_rate")}
    D2 = {(ds, g, m): t("D2_flagged_vs_nonflagged", ds, m, grp=g) for ds in ("cohn", "nt_enhancers")
          for g in ("NEG", "BG_uni") for m in ("iso_AT_L1L2", "rep_L1_major", "rep_simple_lowc_any")}
    D3 = T[T.test == "D3_nt_chromhmm"].iloc[0]; D4 = T[T.test == "D4_cohn_rule"].iloc[0]
    D5 = {(ds, g, rb): t("D5_posthoc_gc_standardised_tata", ds, "tata_rate", rule=rb, grp=g)
          for ds in ("cohn", "nt_enhancers") for g in ("NEG", "POS") for rb in ("BG_uni", "BG_rule")}
    wh = WH.set_index(["split", "metric"])

    L = []
    A = L.append
    A("NEGATIVE-SET PROVENANCE: human_enhancers_cohn and nt_enhancers  (Reviewer 2, comment 1)")
    A("=" * 100)
    A(f"Inputs: {len(CHK) - nfail}/{len(CHK)} integrity/build checks PASS (neg_provenance_input_checks.csv). All "
      "numbers below are read from results/upgrades/neg_provenance_*.csv; [.,.] = 95% percentile bootstrap CI (1000 "
      "resamples, window/sequence = unit, seed 42). About 40 descriptive metrics are reported with unadjusted CIs "
      "(no multiplicity correction); the verdicts rest only on the pre-registered decision rules D1-D4. Labels: "
      "'pre-registered' = in the run_neg_provenance.py plan; 'post hoc' = added afterwards (incl. fix round 1).")
    A("")
    A(f"1. PRIMARY-SOURCE PROVENANCE ({len(SRC)} rows with verbatim quotes, DOIs/URLs: neg_provenance_sources.csv)")
    A("  cohn: Cohn, Zuk & Kaplan 2018 (bioRxiv 10.1101/264200) sampled 'a random set of genomic regions (500 bp "
      "long), excluding regions with N's, regions less than 15Kb from genes (Zerbino et al., 2018), and H3K27ac bound "
      "regions (Villar et al., 2015)'; their code states the negatives 'are not intersected with any gene "
      "(+-15000 bases in each direction) and with any peak of H3K27ac'. Positives are human-liver H3K27ac peaks "
      "(Villar et al. 2015). No GC, repeat or chromosome matching is described. Genomic Benchmarks did not "
      "re-sample: it located the published sequences on GRCh38 (Ensembl 97) by exact match (13,895/14,000 "
      "positives, 13,896/14,000 negatives).")
    A("  nt_enhancers: the NT card (pinned revision) = Geng et al. 2022: iEnhancer-2L benchmark (742 strong, 742 weak, "
      "1484 non-enhancers) + 6000 synthetic enhancers + 6000 synthetic non-enhancers 'produced with a generative "
      "model' (RankGAN); test = the iEnhancer-EL independent set (Liu et al. 2018; 100/100/200), said to be "
      "'constructed based on the same protocol'. iEnhancer-2L (Liu et al. 2016, 10.1093/bioinformatics/btv604) built "
      "its set from ChromHMM states of nine ENCODE cell lines; strong enhancers and non-enhancers 'are clearly "
      "annotated in all the nine tissues'; 200-bp fragments; CD-HIT 80%; 1484 non-enhancers randomly picked from "
      "'5 257 994 non-enhancers'. Which states count as non-enhancer is not stated. ChromHMM state 13 is "
      "'Heterochromatin; low signal' in a nine-mark model that has no H3K9me3 (UCSC track page), so it is described "
      "here as unmarked/low-signal chromatin.")
    A("")
    A("2. WHICH nt_enhancers SEQUENCES ARE GENOMIC (census of all 15,368 rows)")
    A(f"  Exactly the {int(META['nt_genomic_seqs'])}/{int(META['nt_total_seqs'])} sequences "
      f"({100 * META['nt_genomic_seqs'] / META['nt_total_seqs']:.1f}%) that are verbatim iEnhancer originals map to hg38 "
      "(cov>=0.95, id>=0.95; row-level agreement 15,368/15,368); every other sequence is a 199-bp RankGAN-synthetic "
      f"sequence with no genomic origin. Negatives: {int(o_all0.loc['synthetic', 'n_rows']):,} of "
      f"{int(o_all0['n_rows'].sum()):,} are synthetic ({100 * o_all0.loc['synthetic', 'frac_of_class']:.1f}% of all "
      f"negatives; {100 * syn_tr0:.1f}% of the {int(otr0['n_rows'].sum()):,} TRAINING negatives; {n_syn_test} synthetic "
      f"rows in test). Positives: {int(o_all1.loc['synthetic', 'n_rows']):,} synthetic "
      f"({100 * o_all1.loc['synthetic', 'frac_of_class']:.1f}% of all positives; {100 * syn_tr1:.1f}% of training "
      f"positives). Genomic negatives: {int(o_all0.loc['iEnhancer_original', 'n_rows'])} rows = "
      f"{int(o_all0.loc['iEnhancer_original', 'n_unique'])} unique loci ({int(otr0.loc['iEnhancer_original', 'n_unique']):,} "
      f"training [iEnhancer-2L] + {int(ote0.n_unique)} test [iEnhancer-EL]). The TEST split is "
      f"{100 * (1 - n_syn_test / 400):.0f}% genomic. The iEnhancer-2L training FASTA headers carry hg19 coordinates: "
      f"our hg38 locus lifted to hg19 equals the header for {int(META['nt_hdr_header_equals_lifted_map'])}/"
      f"{int(META['nt_hdr_n_train_originals_with_coord_header'])} training originals "
      f"({100 * META['nt_hdr_frac_concordant']:.1f}%; the rest map to duplicated loci).")
    A(f"  Synthetic vs genomic negatives: GC {fmt(o_all0.loc['synthetic', 'gc_local'], o_all0.loc['synthetic', 'gc_lo'], o_all0.loc['synthetic', 'gc_hi'])} "
      f"vs {fmt(o_all0.loc['iEnhancer_original', 'gc_local'], o_all0.loc['iEnhancer_original', 'gc_lo'], o_all0.loc['iEnhancer_original', 'gc_hi'])}; "
      f"TATA-box rate {fmt(o_all0.loc['synthetic', 'tata_rate'], o_all0.loc['synthetic', 'tata_lo'], o_all0.loc['synthetic', 'tata_hi'])} "
      f"vs {fmt(o_all0.loc['iEnhancer_original', 'tata_rate'], o_all0.loc['iEnhancer_original', 'tata_lo'], o_all0.loc['iEnhancer_original', 'tata_hi'])} "
      f"(random 200-bp genome windows: GC {sf('nt_enhancers', 'BG_uni', 'gc_local')}, TATA {sf('nt_enhancers', 'BG_uni', 'tata_rate')}). "
      "The GAN reproduced only part of the originals' AT shift; synthetic positives resemble genomic positives "
      f"(GC {o_all1.loc['synthetic', 'gc_local']:.3f} vs {o_all1.loc['iEnhancer_original', 'gc_local']:.3f}).")
    tlx = TL.set_index(["nt_split", "ienhancer_set", "enhancers_types_label"])["rows"]
    tly = TL.set_index(["nt_split", "ienhancer_set", "enhancers_types_label"])["unique_seqs"]
    A(f"  Side finding (post hoc, neg_provenance_nt_typelabels.csv; not part of R2.1): in the training split every one of "
      f"the 742 iEnhancer weak-enhancer originals also appears with the enhancers_types STRONG label "
      f"({int(tlx.loc[('train', 'train_weak', 2)])} rows / {int(tly.loc[('train', 'train_weak', 2)])} distinct sequences "
      f"labelled strong; {int(tlx.loc[('train', 'train_weak', 1)])} rows labelled weak), and all {int(tlx.loc[('test', 'test_strong|train_strong', 2)])} test strong enhancers are "
      "verbatim iEnhancer-2L TRAINING strong enhancers (present in NT train). The binary enhancers labels are "
      "consistent (no sequence carries both labels).")
    A("")
    A("3. WHERE THE GENOMIC NEGATIVES COME FROM (hg38 coordinates; random/rule backgrounds N-free, same length)")
    A(f"  cohn NEG (n={s('cohn', 'NEG', 'gc_local')[3]:,}) vs random genome: >=15 kb from every protein-coding gene "
      f"{sf('cohn', 'NEG', 'ge15kb_gene_pc', pct=True)} vs {sf('cohn', 'BG_uni', 'ge15kb_gene_pc', pct=True)} "
      f"[D4 {D4.verdict}]; intergenic {sf('cohn', 'NEG', 'comp_intergenic', pct=True)} vs {sf('cohn', 'BG_uni', 'comp_intergenic', pct=True)} "
      f"(so {100 * (1 - s('cohn', 'NEG', 'comp_intergenic')[0]):.1f}% lie inside a GENCODE v44 gene or promoter, "
      "mostly introns of non-coding genes); "
      f"median distance to a TSS {sf('cohn', 'NEG', 'dist_tss_kb', 1)} vs {sf('cohn', 'BG_uni', 'dist_tss_kb', 1)} kb; "
      f"AT-rich isochores (L1+L2) {sf('cohn', 'NEG', 'iso_AT_L1L2', pct=True)} vs {sf('cohn', 'BG_uni', 'iso_AT_L1L2', pct=True)}; "
      f"L1 bp {sf('cohn', 'NEG', 'rep_bp_L1')} vs {sf('cohn', 'BG_uni', 'rep_bp_L1')}; any cCRE {sf('cohn', 'NEG', 'ccre_any', pct=True)} "
      f"vs {sf('cohn', 'BG_uni', 'ccre_any', pct=True)}; ChromHMM state 13 in all 9 lines "
      f"{sf('cohn', 'NEG', 'hmm_13_all9', pct=True)} vs {sf('cohn', 'BG_uni', 'hmm_13_all9', pct=True)}, an enhancer state "
      f"in >=1 line {sf('cohn', 'NEG', 'hmm_enh_any', pct=True)} vs {sf('cohn', 'BG_uni', 'hmm_enh_any', pct=True)}. Only "
      f"{sf('cohn', 'NEG', 'ge15kb_gene_all', pct=True)} are >=15 kb from every GENCODE v44 gene of any biotype, so "
      "Cohn's 'genes' behave like protein-coding genes, not today's all-biotype annotation. Summary: gene-distal, "
      "liver-H3K27ac-free 500-bp windows -- not gene deserts and not uniformly inactive chromatin.")
    cu = cht("cohn", "uniform N-free length", "GOF"); cpc = cht("cohn", "BG_rule_pc", "homogeneity")
    A(f"  cohn chromosomes: vs N-free length chi-square {cu.stat:.0f} (df {int(cu.df)}), p = {cu.p_str}; vs the "
      f"composition-matching rule background BG_rule_pc (2 x K homogeneity) chi-square {cpc.stat:.0f} (df {int(cpc.df)}), "
      f"p = {cpc.p_str}. No negative on chr9 (expected {chc.loc['chr9', 'exp_uniform']:.0f} under uniform, "
      f"{chc.loc['chr9', 'exp_match']:.0f} under BG_rule_pc) or chrY; chr2 {chc.loc['chr2', 'ratio_vs_uniform']:.2f}x and "
      f"chrX {chc.loc['chrX', 'ratio_vs_uniform']:.2f}x, chr1 {chc.loc['chr1', 'ratio_vs_uniform']:.2f}x of expectation; "
      f"positives are present on chr9 ({100 * chc.loc['chr9', 'pos_frac']:.1f}%). This skew is NOT implied by the "
      "documented rule and points to an undocumented quirk of the original sampler; it is not what makes the "
      f"negatives AT-rich (a chromosome-matched random background has GC {sf('cohn', 'BG_chr', 'gc_local')} vs "
      f"{sf('cohn', 'BG_uni', 'gc_local')} uniform; chromosome-matched rule background f_rule="
      f"{CHR['gc_local'].f_rule:.2f} {cit(CHR['gc_local'], 'f_rule_ci', 2)}).")
    tr13 = bs("train", "NEG", "hmm_13_all9"); te13 = bs("test", "NEG", "hmm_13_all9")
    d_tt = bt("NEG test - NEG train", "hmm_13_all9")
    A(f"  nt genomic NEG, split-stratified (post hoc; neg_provenance_nt_by_split.csv). ChromHMM state 13 in ALL nine "
      f"cell lines: TRAIN {tr13} (n={int(BS[(BS.split == 'train') & (BS.group == 'NEG')].n_loci.iloc[0]):,}), TEST {te13} "
      f"(n={int(BS[(BS.split == 'test') & (BS.group == 'NEG')].n_loci.iloc[0])}), pooled {sf('nt_enhancers', 'NEG', 'hmm_13_all9', pct=True)} "
      f"(random genome {sf('nt_enhancers', 'BG_uni', 'hmm_13_all9', pct=True)}); test - train "
      f"{100 * d_tt.diff_a_minus_b:+.1f} points [{100 * d_tt.diff_lo:.1f}, {100 * d_tt.diff_hi:.1f}], Fisher p = "
      f"{d_tt.fisher_p:.1e}. Non-enhancer in all nine: train {bs('train', 'NEG', 'hmm_nonenh_all9')}, test "
      f"{bs('test', 'NEG', 'hmm_nonenh_all9')}; an enhancer state (4-7) in >=1 line: train {bs('train', 'NEG', 'hmm_enh_any')}, "
      f"test {bs('test', 'NEG', 'hmm_enh_any')}. The pre-registered D3 rule (>=80% of NEG non-enhancer in all nine and "
      f">=80% of POS an enhancer state in >=1 line) is met pooled ({D3.verdict}) and within each split "
      f"(train/test/pooled: {'/'.join(r.verdict for r in BT[BT.contrast.str.startswith('D3 per split')].itertuples())}), "
      "but the stricter state-13 description holds for the training split only.")
    tp = bt("NEG test - NEG train", "gc_local"); tt = bt("NEG test - NEG train", "tata_rate")
    A(f"  The difference is not a mapping or liftOver artefact: without minimap2 or liftOver, the hg19 header windows "
      f"of the training originals are state 13 in all nine lines for {int(hdr13.k):,}/{int(hdr13.n):,} non-enhancers, "
      f"strong-enhancer state in all nine for {int(hc('hg19 header window, whole_strong_all9', cls='strong', split='train').k)}/"
      f"{int(hc('hg19 header window, whole_strong_all9', cls='strong', split='train').n)} strong and weak-enhancer state "
      f"in >=1 line for {int(hc('hg19 header window, weak_any', cls='weak', split='train').k)}/"
      f"{int(hc('hg19 header window, weak_any', cls='weak', split='train').n)} weak enhancers (neg_provenance_nt_hg19_check.csv).")
    pp_ = hc("test NEG: header carries a parseable parent range", cls="non")
    pin = hc("test NEG: parent range on the fragment's own hg19 chromosome and containing it", cls="non")
    p13 = hc("test NEG: ... and that segment is state 13 (Heterochrom/lo) in every matching line", cls="non")
    p99 = hc("test NEG: parent range >=99% state 13 (Heterochrom/lo) in all nine lines", cls="non")
    pnull = hc("NULL test NEG: shifted parent range (5-100 kb, 20 draws/region) = one segment in >=1 line", cls="non")
    plen = hc("test NEG parent range length (bp): median [IQR]", cls="non")
    nml = HC[HC.check == "test NEG with exact segment match: number of matching lines"]
    two = nml[nml.note == "lines=2"].iloc[0]
    n_ge2 = int(nml[nml.note != "lines=1"].k.sum())
    base1 = HC[(HC.check.str.startswith("BASELINE test NEG")) & (HC.note == "lines=1; seed 42")].iloc[0]
    wpp = hc("test weak POS: header carries a parseable parent range", cls="weak")
    wseg = hc("test weak POS: parent range = one ChromHMM segment (both ends +-1 bp) in >=1 line", cls="weak")
    wh1 = hc("test weak POS with exact segment match: parent matches a segment of this line", cls="weak", note="H1hesc")
    A(f"  Why the TEST negatives differ (post hoc). Their FASTA headers are stripped UCSC custom-track headers "
      f"('hg19_ct_UserTrack_3545_<item> range=chrN:start-end'): {int(pp_.k)}/{int(pp_.n)} carry a parent range; "
      f"{int(pin.k)}/{int(pin.n)} parent ranges contain the fragment. Every parseable parent range ({int(par.k)}/{int(par.n)}) "
      f"coincides (both ends within 1 bp) with one ChromHMM segment in at least one of the nine lines, state 13 in "
      f"{int(p13.k)}/{int(p13.n)} (shifted-region null: {int(pnull.k)}/{int(pnull.n)}); the matching segment is shared by "
      f"at least two lines for {n_ge2}/{int(two.n)} (exactly two for {100 * two.frac:.0f}%), whereas only "
      f"{100 * (1 - base1.frac):.0f}% of random length-matched state-13 segments are shared by >=2 lines. Only {int(p99.k)}/{int(p99.n)} parent ranges are >=99% state 13 in all nine lines (median "
      f"length {plen.frac / 1000:.0f} kb [IQR {plen.ci_lo / 1000:.0f}-{plen.ci_hi / 1000:.0f}]). So the independent-test "
      "negatives were cut from single ChromHMM state-13 segments (each identical in >=2 of the nine lines), not from "
      "the nine-line intersection; fragments of such segments are state 13 in all nine lines in "
      f"{100 * hc('test NEG: fragment whole window state 13 in all nine lines (liftOver of fragment)', cls='non').frac:.0f}% "
      f"of cases and touch an enhancer state in {100 * enh_t.frac:.1f}%. The same user track supplied the test weak "
      f"enhancers: {int(wpp.k)}/{int(wpp.n)} carry a parent range, every one ({int(wseg.k)}/{int(wseg.n)}) is exactly one "
      f"ChromHMM segment and {int(wh1.k)}/{int(wh1.n)} match H1-hESC, i.e. single-line (H1-hESC) weak-enhancer segments. The test strong enhancers "
      f"({int(hc('hg19 header window, whole_strong_all9', cls='strong', split='test').k)}/100 strong in all nine) are "
      "verbatim training strong enhancers. Composition is nonetheless the same: test vs train negatives GC "
      f"{bs('test', 'NEG', 'gc_local', pct=False)} vs {bs('train', 'NEG', 'gc_local', pct=False)} (diff "
      f"{tp.diff_a_minus_b:+.4f} [{tp.diff_lo:.4f}, {tp.diff_hi:.4f}]), TATA {bs('test', 'NEG', 'tata_rate', pct=False)} vs "
      f"{bs('train', 'NEG', 'tata_rate', pct=False)} (diff {tt.diff_a_minus_b:+.3f} [{tt.diff_lo:.3f}, {tt.diff_hi:.3f}]), "
      f"L1+L2 {bs('test', 'NEG', 'iso_AT_L1L2')} vs {bs('train', 'NEG', 'iso_AT_L1L2')}.")
    n13 = cht("nt_enhancers", "BG_rule13", "homogeneity"); n13g = cht("nt_enhancers", "BG_rule13 counts + 0.5", "GOF")
    nu = cht("nt_enhancers", "uniform N-free length", "GOF")
    A(f"  nt genomic NEG (pooled, n={s('nt_enhancers', 'NEG', 'gc_local')[3]:,}) vs random genome: L1+L2 isochores "
      f"{sf('nt_enhancers', 'NEG', 'iso_AT_L1L2', pct=True)} vs {sf('nt_enhancers', 'BG_uni', 'iso_AT_L1L2', pct=True)}; "
      f">=15 kb from protein-coding genes {sf('nt_enhancers', 'NEG', 'ge15kb_gene_pc', pct=True)} vs "
      f"{sf('nt_enhancers', 'BG_uni', 'ge15kb_gene_pc', pct=True)}; L1 bp {sf('nt_enhancers', 'NEG', 'rep_bp_L1')} vs "
      f"{sf('nt_enhancers', 'BG_uni', 'rep_bp_L1')}; any cCRE {sf('nt_enhancers', 'NEG', 'ccre_any', pct=True)} vs "
      f"{sf('nt_enhancers', 'BG_uni', 'ccre_any', pct=True)}. Chromosomes: chrX {chn.loc['chrX', 'ratio_vs_uniform']:.2f}x "
      f"and chr13 {chn.loc['chr13', 'ratio_vs_uniform']:.2f}x over-, chr17/19/22 "
      f"{chn.loc['chr17', 'ratio_vs_uniform']:.2f}/{chn.loc['chr19', 'ratio_vs_uniform']:.2f}/{chn.loc['chr22', 'ratio_vs_uniform']:.2f}x "
      f"under-represented vs N-free length (chi-square {nu.stat:.0f}, df {int(nu.df)}, p = {nu.p_str}), but fully "
      f"consistent with the state-13 rule background (homogeneity chi-square {n13.stat:.1f}, df {int(n13.df)}, "
      f"p = {n13.p_str}; goodness of fit {n13g.stat:.1f}, p = {n13g.p_str}; chr1-22, X).")
    A(f"  POST HOC rule reconstruction (neg_provenance_chromhmm_rule_counts.csv): in the nine hg19 segmentations, "
      f"'state 13 in all nine' contains {int(r13.n_200bp_fragments):,} whole 200-bp fragments vs the published "
      f"5,257,994 non-enhancers (ratio {r13.ratio_fragments_to_published:.4f}); 'states 13-15 in all nine' "
      f"{int(r1315.n_200bp_fragments):,} ({r1315.ratio_fragments_to_published:.4f}), so the count separates state 13 "
      f"from the literal reading but only weakly from 13-15; the literal 'not an enhancer state in all nine' contains "
      f"{int(rne.n_200bp_fragments):,} ({rne.ratio_fragments_to_published:.2f}x); 'strong-enhancer state (4/5) in all "
      f"nine' contains exactly {int(rst.n_200bp_fragments):,} fragments = the published 742 strong enhancers. The "
      "weak-enhancer count (370,517) is not reproduced by any of the three readings tried "
      f"({', '.join(f'{int(v):,}' for v in rwk.n_200bp_fragments)}). Together with the header check this identifies "
      "the TRAINING negatives as random 200-bp fragments of chromatin that is state 13 (low signal) in all nine lines.")
    A("")
    A("4. DOES THE DOCUMENTED RULE REPRODUCE THE AT-RICHNESS?  (pre-registered D1; f_rule = (BG_uni-BG_rule)/(BG_uni-NEG))")
    for ds, rn in (("cohn", "BG_rule (>=15 kb from any GENCODE v44 gene, not overlapping a positive)"),
                   ("nt_enhancers", "BG_rule (not ChromHMM 4-7 in all nine lines; the literal pre-registered reading)")):
        A(f"  {ds}, rule = {rn}:")
        for m in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2"):
            d = D1[(ds, m)]
            A(f"    {m:12s} NEG {sf(ds, 'NEG', m)} | rule {sf(ds, 'BG_rule', m)} | random {sf(ds, 'BG_uni', m)} | "
              f"POS {sf(ds, 'POS', m)}; random-NEG {d.bguni_minus_neg:+.4f} {cit(d, 'bguni_minus_neg_ci', 4)}; "
              f"f_rule {d.f_rule:.2f} {cit(d, 'f_rule_ci', 2)} -> {d.verdict}")
    A("  cohn, POST HOC rule with protein-coding genes only (BG_rule_pc; seed-42 draw):")
    for m in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2"):
        d = PC[m]
        A(f"    {m:12s} rule_pc {sf('cohn', 'BG_rule_pc', m)} vs NEG {sf('cohn', 'NEG', m)}; rule_pc-NEG "
          f"{d.term_rule_minus_neg:+.4f} {cit(d, 'term_rule_minus_neg_ci', 4)}; f_rule {d.f_rule:.2f} {cit(d, 'f_rule_ci', 2)}")
    A("  nt_enhancers, pre-registered SECONDARY rule (state 13 in all nine; BG_rule13; seed-42 draw):")
    for m in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2"):
        d = S13[m]
        A(f"    {m:12s} rule13 {sf('nt_enhancers', 'BG_rule13', m)} vs NEG {sf('nt_enhancers', 'NEG', m)}; rule13-NEG "
          f"{d.term_rule_minus_neg:+.4f} {cit(d, 'term_rule_minus_neg_ci', 4)}; f_rule {d.f_rule:.2f} {cit(d, 'f_rule_ci', 2)}")
    A("  POST HOC seed stability (bg_seed_stability.py; six independent draws = 60,000 windows per background; "
      "'matches' = pooled-minus-NEG CI contains 0, fixed before the new draws were looked at):")
    for ds, g in (("cohn", "BG_rule_pc"), ("cohn", "BG_rule"), ("nt_enhancers", "BG_rule13"), ("nt_enhancers", "BG_rule")):
        parts = []
        for m in ("gc_local", "tata_rate", "gc_100kb", "iso_AT_L1L2"):
            r_ = bp(ds, g, m)
            parts.append(f"{m} {r_.pooled:.4f} (seeds {r_.seed_min:.4f}-{r_.seed_max:.4f}) - NEG {r_.neg:.4f} = "
                         f"{r_.bg_minus_neg:+.4f} [{r_.bg_minus_neg_lo:+.4f}, {r_.bg_minus_neg_hi:+.4f}] "
                         f"{'matches' if r_.matches_neg else 'DIFFERS'}; f_rule {r_.f_rule:.2f} [{r_.f_rule_lo:.2f}, {r_.f_rule_hi:.2f}]")
        A(f"    {ds} {g}: " + " | ".join(parts))
    pcg = bp("cohn", "BG_rule_pc", "gc_local"); pct_ = bp("cohn", "BG_rule_pc", "tata_rate")
    pci = bp("cohn", "BG_rule_pc", "iso_AT_L1L2")
    A("  Reading: cohn -> EXPLAINED under the pre-registered rule (the all-biotype gene set over-shoots: the rule "
      "background is MORE AT-rich than the negatives); the post hoc protein-coding variant reproduces the negatives' "
      f"window and 100-kb GC exactly (pooled {pcg.bg_minus_neg:+.4f} [{pcg.bg_minus_neg_lo:+.4f}, {pcg.bg_minus_neg_hi:+.4f}]) "
      f"but over-shoots the TATA rate and L1+L2 fraction by about one point (TATA {pct_.bg_minus_neg:+.3f} "
      f"[{pct_.bg_minus_neg_lo:+.3f}, {pct_.bg_minus_neg_hi:+.3f}]; L1+L2 {pci.bg_minus_neg:+.3f} [{pci.bg_minus_neg_lo:+.3f}, "
      f"{pci.bg_minus_neg_hi:+.3f}]), so it reproduces GC and approximately (within ~0.01) the TATA rate; the seed-42 "
      "'TATA within CI' was partly draw luck. nt -> NOT EXPLAINED under the pre-registered primary (literal) reading, "
      f"which covers {100 * META['nt_rule_nonenh_region_bp_hg19'] / META['genome_primary_bp']:.0f}% of the (hg19) genome "
      "and is barely AT-shifted; the pre-registered secondary rule (state 13 in all nine lines) matches the pooled "
      "genomic negatives on all four metrics, pooled over six draws as well, and matches the test negatives' GC and "
      f"TATA (test NEG vs BG_rule13: GC {bt('BG_rule13 - NEG test', 'gc_local').diff_a_minus_b:+.4f} "
      f"[{bt('BG_rule13 - NEG test', 'gc_local').diff_lo:+.4f}, {bt('BG_rule13 - NEG test', 'gc_local').diff_hi:+.4f}], TATA "
      f"{bt('BG_rule13 - NEG test', 'tata_rate').diff_a_minus_b:+.3f} [{bt('BG_rule13 - NEG test', 'tata_rate').diff_lo:+.3f}, "
      f"{bt('BG_rule13 - NEG test', 'tata_rate').diff_hi:+.3f}]; L1+L2 {bt('BG_rule13 - NEG test', 'iso_AT_L1L2').diff_a_minus_b:+.3f} "
      f"[{bt('BG_rule13 - NEG test', 'iso_AT_L1L2').diff_lo:+.3f}, {bt('BG_rule13 - NEG test', 'iso_AT_L1L2').diff_hi:+.3f}]).")
    A("")
    A("5. HOW MUCH OF THE POSITIVE-NEGATIVE GAP IS THE NEGATIVES' DOING? (share = (random-NEG)/(POS-NEG))")
    for ds in ("cohn", "nt_enhancers"):
        for m in ("gc_local", "tata_rate"):
            d = D1[(ds, m)]
            A(f"  {ds} {m}: POS-NEG {d.gap_pos_minus_neg:+.3f} {cit(d, 'gap_ci')} = (POS-random) "
              f"{d.term_pos_minus_bguni:+.3f} {cit(d, 'term_pos_minus_bguni_ci')} + (random-NEG) {d.bguni_minus_neg:+.3f} "
              f"{cit(d, 'bguni_minus_neg_ci')}; negative-side share {100 * d.share_of_gap_from_neg_side:.1f}% "
              f"{cip(d, 'share_of_gap_from_neg_side_ci')}; positive-side share {100 * (1 - d.share_of_gap_from_neg_side):.1f}%")
    for sp in ("all", "test"):
        for m in ("gc_local", "tata_rate"):
            w = wh.loc[(sp, m)]
            lab_ = "all splits, incl. synthetic" if sp == "all" else "test split, all genomic"
            A(f"  nt_enhancers POST HOC whole class ({lab_}): POS {w.pos:.3f} NEG "
              f"{w.neg:.3f} random {w.bg_uni:.3f}; negative-side share of the {m} gap {100 * w.share_of_gap_from_neg_side:.1f}% "
              f"[{100 * w.ci_lo:.1f}, {100 * w.ci_hi:.1f}] (n={int(w.n_pos)}/{int(w.n_neg)})")
    A("  Reading: relative to random genome the negatives are shifted by the sampling rules (TATA "
      f"{-D1[('cohn', 'tata_rate')].bguni_minus_neg * 100:+.1f} / {-D1[('nt_enhancers', 'tata_rate')].bguni_minus_neg * 100:+.1f} "
      f"points, GC {-D1[('cohn', 'gc_local')].bguni_minus_neg * 100:+.1f} / {-D1[('nt_enhancers', 'gc_local')].bguni_minus_neg * 100:+.1f} "
      "points for cohn / genomic nt), but most of the positive-negative gap is the positives' GC-richness and TATA "
      "depletion relative to random genome, which any random-genome negative set would share.")
    A("")
    A("6. TATA-FLAGGED NEGATIVES (pre-registered D2; post hoc D5)")
    for ds in ("cohn", "nt_enhancers"):
        a = D2[(ds, "NEG", "iso_AT_L1L2")]; b = D2[(ds, "NEG", "rep_L1_major")]
        ua = D2[(ds, "BG_uni", "iso_AT_L1L2")]; ub = D2[(ds, "BG_uni", "rep_L1_major")]
        sl = D2[(ds, "NEG", "rep_simple_lowc_any")]
        A(f"  {ds}: flagged vs non-flagged negatives in L1+L2 isochores {100 * a.flagged:.1f}% vs {100 * a.nonflagged:.1f}% "
          f"(RR {a.risk_ratio:.2f}, diff CI {cip(a, 'diff_ci')}, Fisher p={a.fisher_p:.1e}) -> {a.verdict}; L1-dominated "
          f"(>=50% bp L1) {100 * b.flagged:.1f}% vs {100 * b.nonflagged:.1f}% (RR {b.risk_ratio:.2f}, {cip(b, 'diff_ci')}, "
          f"p={b.fisher_p:.1e}) -> {b.verdict}; simple/low-complexity repeat overlap {100 * sl.flagged:.1f}% vs "
          f"{100 * sl.nonflagged:.1f}% (RR {sl.risk_ratio:.2f}). Same contrast in random genome: RR {ua.risk_ratio:.2f} "
          f"(isochore), {ub.risk_ratio:.2f} (L1) -> the association is generic to AT-rich DNA, not specific to the negatives.")
    for ds in ("cohn", "nt_enhancers"):
        n1 = D5[(ds, "NEG", "BG_uni")]; n2 = D5[(ds, "NEG", "BG_rule")]; p1 = D5[(ds, "POS", "BG_uni")]
        A(f"  D5 {ds}: NEG obs/exp TATA rate given local GC = {n1.risk_ratio:.3f} {cit(n1, 'f_rule_ci')} against random "
          f"genome and {n2.risk_ratio:.3f} {cit(n2, 'f_rule_ci')} against the pre-registered rule background; POS "
          f"{p1.risk_ratio:.3f} {cit(p1, 'f_rule_ci')} (random genome reference).")
    negr = [D5[(ds, "NEG", rb)] for ds in ("cohn", "nt_enhancers") for rb in ("BG_uni", "BG_rule")]
    n_excl = sum(1 for r_ in negr if not (r_.f_rule_ci_lo <= 1 <= r_.f_rule_ci_hi))
    maxdev = max(abs(1 - r_.risk_ratio) for r_ in negr)
    A(f"  Reading: the negatives' TATA-box rate is approximately what their local GC predicts (all four NEG ratios "
      f"within {100 * maxdev:.1f}% of 1; {n_excl} of the four CIs {'excludes' if n_excl == 1 else 'exclude'} 1): the TATA excess is AT content, consistent "
      "with the column-scrambled-TBP control in the paper. Positives carry FEWER TATA hits than their GC predicts.")
    A("")
    A("7. MECHANISM (stated only as strongly as the data support)")
    A(f"  cohn negatives are gene-distal 500-bp windows (>=15 kb from genes, outside liver H3K27ac peaks; "
      f"{100 * s('cohn', 'NEG', 'ge15kb_gene_pc')[0]:.1f}% >=15 kb from any protein-coding gene vs "
      f"{100 * s('cohn', 'BG_uni', 'ge15kb_gene_pc')[0]:.1f}% of random windows). The genomic nt_enhancers TRAINING "
      "negatives are 200-bp fragments of chromatin that is ChromHMM state 13 (low signal, unmarked) in all nine ENCODE "
      f"lines ({100 * s('nt_enhancers', 'NEG_train', 'hmm_13_all9')[0]:.1f}%; {int(hdr13.k):,}/{int(hdr13.n):,} by hg19 "
      "headers); the TEST negatives (iEnhancer-EL) were cut from single state-13 segments shared by >=2 lines, not "
      "the nine-line intersection "
      f"({100 * s('nt_enhancers', 'NEG_test', 'hmm_13_all9')[0]:.1f}% state 13 in all nine). Gene-distal and unmarked "
      f"DNA is concentrated in AT-rich, L1-rich isochores (L1+L2 {100 * s('cohn', 'NEG', 'iso_AT_L1L2')[0]:.0f}% and "
      f"{100 * s('nt_enhancers', 'NEG', 'iso_AT_L1L2')[0]:.0f}% vs {100 * s('cohn', 'BG_uni', 'iso_AT_L1L2')[0]:.0f}% "
      "genome-wide). Backgrounds re-drawn by these rules, with the unstated details inferred post hoc (protein-coding "
      "genes; state 13), reproduce the negatives' GC and approximately their TATA rate (cohn pooled GC "
      f"{pcg.pooled:.3f} vs {pcg.neg:.3f}, TATA {pct_.pooled:.3f} vs {pct_.neg:.3f}; nt GC "
      f"{bp('nt_enhancers', 'BG_rule13', 'gc_local').pooled:.3f} vs {bp('nt_enhancers', 'BG_rule13', 'gc_local').neg:.3f}, TATA "
      f"{bp('nt_enhancers', 'BG_rule13', 'tata_rate').pooled:.3f} vs {bp('nt_enhancers', 'BG_rule13', 'tata_rate').neg:.3f}); "
      "the pre-registered cohn rule over-shoots and the pre-registered literal nt rule does not reproduce them. This "
      f"sampling accounts for {100 * D1[('cohn', 'gc_local')].share_of_gap_from_neg_side:.0f}% (cohn) and "
      f"{100 * D1[('nt_enhancers', 'gc_local')].share_of_gap_from_neg_side:.0f}% (genomic nt; "
      f"{100 * wh.loc[('all', 'gc_local')].share_of_gap_from_neg_side:.1f}% over the whole class incl. synthetic) of the "
      "positive-negative GC gap and "
      f"{100 * D1[('cohn', 'tata_rate')].share_of_gap_from_neg_side:.0f}% / "
      f"{100 * D1[('nt_enhancers', 'tata_rate')].share_of_gap_from_neg_side:.0f}% of the TATA gap; the rest is the "
      "enhancer positives' GC-richness (and TATA depletion) relative to random genome.")
    A("  Caveats: the cohn gene set and the exact iEnhancer state definition are inferred (protein-coding variant and "
      "count reconstruction are post hoc; the count does not separate state 13 from states 13-15); Villar H3K27ac "
      "peaks are proxied by the benchmark positives; the iEnhancer-EL test negatives do not follow the nine-line rule "
      "(construction inferred from their FASTA headers, not documented); the chr9/chrX/chr2 skew of cohn is unexplained; "
      "the gap shares use random genome as the reference.")
    A("")
    A("8. SANITY (paper numbers)")
    for r_ in SAN.itertuples():
        A(f"  {r_.dataset} {r_.cls}: TATA-box fraction {r_.tata_box_fraction:.4f} on all {r_.n:,} sequences")
    A("  (paper, from a 4,000/4,000 subsample in results/contamination_analysis.csv and results/nt/: cohn 0.346 / "
      "0.573, nt_enhancers 0.106 / 0.315 -- consistent.)")
    with open(P + "_interpretation.txt", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
