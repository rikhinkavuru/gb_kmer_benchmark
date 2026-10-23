#!/usr/bin/env python3
"""Negative-set provenance for human_enhancers_cohn and nt_enhancers (Reviewer 2, comment 1).

"The coordinate analysis convincingly refutes core-promoter leakage, but the paper does not yet explain
why the negatives are AT-rich or which genomic regions they come from."

=====================================================================================================
PRE-REGISTERED ANALYSIS PLAN  (written before any statistic below was computed; seed 42; B = 1000)
=====================================================================================================
PRIMARY-SOURCE PROVENANCE (read before this script was written; quotes in neg_provenance_sources.csv)
  cohn : Cohn, Zuk & Kaplan 2018 (bioRxiv 10.1101/264200), sec 2.3.1: "we sampled a random set of
         genomic regions (500 bp long), excluding regions with N's, regions less than 15Kb from genes
         (Zerbino et al., 2018), and H3K27ac bound regions (Villar et al., 2015)"; the H3K27ac loader
         (github cohnDikla/enhancer_CNN, DataLoaderH3K27acvsNeg.py) re-states "not intersected with any
         gene (+-15000 bases in each direction) and with any peak of H3K27ac". Positives = human liver
         H3K27ac (distal) peaks of Villar et al. 2015. Genomic Benchmarks re-located the sequences on
         GRCh38 by exact matching (seq2loc); 13,896/14,000 negatives found.
  nt   : NT downstream `enhancers` (HF revision 96d86d5; NT bioRxiv v1 A.4.3) = Geng et al. 2022
         (Biophys Chem 286:106822): the iEnhancer-2L benchmark (Liu et al. 2016, Bioinformatics
         32:362) -- 742 strong + 742 weak enhancers + 1484 non-enhancers, 200-bp fragments from ENCODE
         ChromHMM segmentations of nine cell lines (Ernst et al. 2011), non-enhancers "clearly annotated
         in all the nine tissues" -- PLUS 6000 synthetic enhancers + 6000 synthetic non-enhancers from a
         generative model (RankGAN). Test = iEnhancer independent set (100 strong, 100 weak, 200 non).

GROUPS (each interval is a genomic window; background draws are N-free, same length)
  cohn : POS (13,895), NEG (13,896)
         BG_uni     10,000 windows uniform over the N-free primary assembly (chr1-22,X,Y)
         BG_chr     1 per NEG, same chromosome & length (the "matched random background")
         BG_rule    10,000 windows obeying the DOCUMENTED Cohn rule: >= 15,000 bp from every GENCODE
                    v44 gene (all biotypes; the literal analogue of the Ensembl gene set) and not
                    overlapping a cohn positive (proxy for Villar H3K27ac peaks, unavailable on hg38)
         BG_rule_chr 1 per NEG, same chromosome, rule-obeying (sensitivity: removes chromosome skew)
  nt   : POS / NEG = nt_enhancers sequences called GENOMIC by map_nt_enhancers.py (cov>=0.95 &
         identity>=0.95), de-duplicated by locus; NON-GENOMIC (synthetic) sequences summarised
         separately (local GC, TATA) but have no provenance to map.
         BG_uni (10,000), BG_chr (1 per NEG) as above, L = 200
         BG_rule    10,000 200-bp windows whose hg19 location (UCSC liftOver chain) is in a
                    non-enhancer ChromHMM state (not 4/5/6/7) in ALL nine cell lines (literal reading)
         BG_rule13  secondary: state 13 (Heterochrom/lo) in all nine cell lines

FEATURES (per window): local GC; TATA flag (JASPAR TBP best hit >= 1.3 bits/pos, the benchmark
  criterion); 100-kb regional GC (window centred on the midpoint, A/C/G/T only) and isochore family
  (L1<37, L2 37-41, H1 41-46, H2 46-53, H3>53 %GC; "AT-rich isochore" = L1+L2); GENCODE v44
  compartment of the midpoint (promoter = within 1 kb of any transcript TSS > exon > intron > intergenic);
  distance to nearest TSS; gap to nearest gene (all biotypes / protein-coding); RepeatMasker (UCSC hg38
  rmsk) base fractions by class/family; ENCODE SCREEN V4 cCRE overlap; hg19 ChromHMM state in 9 cell
  lines at the lifted midpoint.
STATISTICS: fractions/means with 95% percentile bootstrap CIs (1000 resamples, window = resampling unit,
  groups resampled independently, numpy RandomState(42)); chromosome distributions vs expectation by
  chi-square goodness of fit; binary 2x2 contrasts by Fisher's exact test.

DECISION RULES (applied verbatim to the numbers; nulls reported straight)
  D1 (does the documented negative-sampling rule explain the AT shift?), per dataset, metric = local GC
     (secondary: 100-kb GC, TATA rate, L1+L2 fraction). Decompose POS-NEG = (POS-BG_uni) +
     (BG_uni-BG_rule) + (BG_rule-NEG). f_rule = (BG_uni-BG_rule)/(BG_uni-NEG).
       - if the BG_uni-NEG CI includes 0 -> "negatives are not AT-shifted relative to the genome; the gap
         is carried by GC-rich positives".
       - else EXPLAINED if f_rule >= 0.67 and its CI lower bound > 0.33; PARTIAL if 0.33 <= f_rule < 0.67
         (or f_rule >= 0.67 with CI lower bound <= 0.33); NOT EXPLAINED if f_rule < 0.33.
  D2 (do TATA-flagged negatives concentrate in AT-rich isochores / L1 repeats?): within NEG, flagged vs
     non-flagged: L1+L2-isochore fraction and L1-dominated (>=50% bp L1) fraction. CONCENTRATED if the
     risk ratio >= 1.2 and the bootstrap CI of the difference excludes 0; the same contrast inside BG_uni
     is reported as the genome-wide reference (is the association generic to AT-rich DNA?).
  D3 (nt provenance verification): VERIFIED if >= 80% of liftable genomic NEG are non-enhancer in all nine
     cell lines AND >= 80% of liftable genomic POS are an enhancer state (4-7) in >= 1 cell line.
  D4 (cohn rule verification): VERIFIED if >= 80% of NEG are >= 15 kb from every protein-coding gene
     (GENCODE v44 all-biotype annotation is newer/larger than Cohn's Ensembl-2017 set, so the
     all-biotype fraction is reported but not used for the verdict) and BG_uni is far lower.
Outputs: results/upgrades/neg_provenance_{summary,table,tests,chrom,sources}.csv,
         neg_provenance_intervals.csv.gz, neg_provenance_interpretation.txt

=====================================================================================================
RESUME NOTES + POST HOC ADDITIONS (2026-09-26). The plan above is unchanged. The first run (2026-09-23)
was interrupted before ANY statistic was computed (no summary/test file existed). Everything below was
added on resume, after verify_inputs.py passed and BEFORE the main run; it is labelled post hoc because
it is not part of the original pre-registration.
  * Implementation fix (restores the plan's stated "uniform over the N-free primary assembly"): genome-
    wide background draws now reject N-containing windows GLOBALLY (chromosome re-drawn) instead of
    re-drawing within the first-drawn chromosome, which over-weighted N-rich chromosomes (chrY).
  * nt provenance, sequence level: every nt_enhancers sequence is matched verbatim against the published
    iEnhancer-2L benchmark (742/742/1484) and independent test set (100/100/200) (two GitHub copies,
    external/neg_provenance/ienhancer). verify_inputs.py showed the pre-registered GENOMIC call and
    verbatim iEnhancer identity coincide for all 15,368 rows, so GENOMIC == iEnhancer original and
    NON-GENOMIC == RankGAN synthetic. Reported: origin census (neg_provenance_nt_origin.csv) and the
    concordance of the iEnhancer FASTA-header coordinates (hg19) with our hg38 mapping lifted to hg19.
  * cohn sensitivity BG_rule_pc: the documented rule with PROTEIN-CODING genes only (Cohn's Ensembl-2018
    gene set is not specified by biotype); D1 reported for it as a sensitivity row (no verdict change rule).
  * ChromHMM state composition (mean fraction of the nine cell lines in each of the 15 states) for every
    group: descriptive only.
  * D2 extra metrics: any-repeat bp fraction and simple-repeat/low-complexity overlap (AT-rich simple
    repeats such as (TA)n create TBP hits directly). Descriptive; the D2 verdict metrics are unchanged.
  * D5 (GC-standardised TATA rate): expected TATA-flag rate of NEG (and POS) if each window had the
    TATA-flag rate of a reference group (BG_uni, BG_rule) at the same local GC (20 pooled-quantile GC
    bins; bootstrap over windows, B=1000). ratio = observed/expected; a ratio CI containing 1 means local
    GC alone accounts for the group's TATA rate. Descriptive, no verdict.
  * Paper-number sanity: TATA-box fractions of all cohn / nt_enhancers sequences by class are recomputed
    from these data (paper: cohn 34.6% vs 57.3%; nt 10.6% vs 31.5%).

FIX ROUND 1 (2026-09-26, after independent verification; POST HOC, no pre-registered rule changed). The
sampling, annotation and every statistic above are unchanged (same seed-42 random stream; outputs were diffed
against the previous run). Additions:
  * Every bootstrap-CI tuple in neg_provenance_tests.csv is also written as numeric <name>_lo / <name>_hi columns.
  * Chromosome distributions (neg_provenance_chrom_tests.csv): besides the pre-registered goodness-of-fit vs
    uniform N-free length and vs the empirical BG_rule counts, NEG is compared with the rule background that
    matches its composition (cohn BG_rule_pc, nt BG_rule13), both by goodness of fit (expected = rule counts
    + 0.5, restricted to chromosomes the rule sampler covers) and by a 2 x K chi-square test of homogeneity
    (NEG counts vs background counts; accounts for the finite background sample). log10 p is computed with
    chi2.logsf so p-values below the float range are reported exactly instead of as 0.
  * Split-stratified nt provenance, hg19-header (liftover-free) checks, the parent regions of the
    independent-test negatives, and multi-seed background stability are separate post hoc scripts
    (nt_split_provenance.py, bg_seed_stability.py); see README.md.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, chisquare, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402
import data_nt  # noqa: E402
import motif_match  # noqa: E402
import genome_utils as G  # noqa: E402
import verify_inputs as V  # noqa: E402

SEED, B = 42, 1000
THR = 1.3
N_BG = 10_000
RES = C.RESULTS_DIR
PFX = os.path.join(RES, "neg_provenance")
COHN_DIR = os.path.join(C.ROOT, "localization", "cohn")
NT_MAP = PFX + "_nt_map.csv"
ENH_STATES = (4, 5, 6, 7)
N_GCBINS = 20


# ------------------------------------------------------------------------------ sequence utils
def best_tata(seqs, tbp):
    """Per-sequence best TBP PSSM score (bits/pos, both strands); flag = best >= 1.3 is identical to
    C.tata_hit_counts(seq) >= 1 (the paper's TATA-box criterion; tested in tests/test_sanity.py)."""
    L = tbp["pssm"].shape[1]
    out = np.zeros(len(seqs), np.float32)
    for lo in range(0, len(seqs), 4000):                       # chunk to bound RAM
        codes = motif_match.encode_sequences(list(seqs[lo:lo + 4000])); n, S = codes.shape
        npos = S - L + 1
        ni = np.full((1, L), -1e9, np.float32)
        fwd = np.vstack([tbp["pssm"], ni]); rev = np.vstack([tbp["pssm"][::-1, ::-1], ni])
        best = np.full(n, -1e9, np.float32)
        for a in (fwd, rev):
            sc = np.zeros((n, npos), np.float32)
            for i in range(L):
                sc += a[codes[:, i:i + npos], i]
            best = np.maximum(best, sc.max(1) / L)
        out[lo:lo + n] = best
    return out


def gc_frac(seqs):
    return C.gc_content(list(seqs))


def fetch(fa, chroms, starts, ends):
    return [fa[c][int(s):int(e)] for c, s, e in zip(chroms, starts, ends)]


def is_acgt(s):
    return len(s) > 0 and s.count("N") == 0 and set(s) <= set("ACGT")


# ------------------------------------------------------------------------------ backgrounds
def _uniform_start(rng, elig):
    s, e = elig
    w = (e - s).astype(float)
    cw = np.cumsum(w)
    r = rng.uniform(0, cw[-1])
    k = min(int(np.searchsorted(cw, r, side="right")), len(s) - 1)
    st = int(s[k] + np.floor(r - (cw[k] - w[k])))
    return min(max(st, int(s[k])), int(e[k]) - 1)


def sample_windows(fa, rng, n, L, eligible):
    """n windows of length L, uniform over all N-free admissible windows of the union of `eligible` =
    {chrom: (start_lo, start_hi)} (admissible START intervals). N-containing windows are rejected and the
    chromosome is re-drawn (global rejection -> exactly uniform over N-free admissible windows)."""
    cs = [c for c in eligible if float(np.sum(eligible[c][1] - eligible[c][0])) > 0]
    p = np.array([float(np.sum(eligible[c][1] - eligible[c][0])) for c in cs]); p = p / p.sum()
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        c = cs[rng.choice(len(cs), p=p)]
        st = _uniform_start(rng, eligible[c])
        seq = fa[c][st:st + L]
        if len(seq) == L and is_acgt(seq):
            out.append((c, st, st + L, seq))
    return pd.DataFrame(out, columns=["chr", "start", "end", "seq"]), tries


def sample_matched(fa, rng, chroms, Ls, eligible_fn):
    """One window per (chrom, length): uniform over N-free admissible windows on that chromosome."""
    out = []
    for c, L in zip(chroms, Ls):
        elig = eligible_fn(c, int(L))
        for _ in range(100000):
            st = _uniform_start(rng, elig)
            seq = fa[c][st:st + int(L)]
            if len(seq) == L and is_acgt(seq):
                out.append((c, st, st + int(L), seq)); break
        else:
            raise RuntimeError(f"could not draw an N-free window on {c}")
    return pd.DataFrame(out, columns=["chr", "start", "end", "seq"])


def sample_chromhmm_rule(fa, rng, H, lo19to38, n, L, pred):
    """Draw n N-free hg38 windows of length L whose hg19 source window lies entirely inside a region
    where `pred(state)` holds in all nine ChromHMM cell lines (sampled in hg19, lifted to hg38)."""
    elig = {}
    for c in [f"chr{i}" for i in range(1, 23)] + ["chrX"]:
        bnds = [np.concatenate([H[cell][c][0], H[cell][c][1]]) for cell in G.CHROMHMM_CELLS if c in H[cell]]
        if len(bnds) < len(G.CHROMHMM_CELLS):
            continue
        pts = np.unique(np.concatenate(bnds))
        a, b = pts[:-1], pts[1:]
        good = np.ones(len(a), bool)
        for cell in G.CHROMHMM_CELLS:
            s, e, st = H[cell][c]
            i = np.searchsorted(s, a, side="right") - 1
            cov = (i >= 0) & (a < e[np.clip(i, 0, None)])
            good &= cov & pred(np.where(cov, st[np.clip(i, 0, None)], 0))
        ms, me = G.merge_intervals(a[good], b[good])
        tot_bp = float(np.sum(me - ms))
        keep = (me - ms) >= L
        ms, me = ms[keep], me[keep] - L + 1          # admissible starts
        if len(ms):
            elig[c] = (ms, me, tot_bp)
    tot = {c: float(np.sum(v[1] - v[0])) for c, v in elig.items()}
    region_bp = sum(v[2] for v in elig.values())
    cs = list(elig); p = np.array([tot[c] for c in cs]); p /= p.sum()
    out = []
    tries = 0
    while len(out) < n:
        tries += 1
        c = cs[rng.choice(len(cs), p=p)]
        st19 = _uniform_start(rng, elig[c][:2])
        a1 = lo19to38.convert_coordinate(c, st19); a2 = lo19to38.convert_coordinate(c, st19 + L - 1)
        if not a1 or not a2 or len(a1) != 1 or len(a2) != 1:
            continue
        (c1, p1, s1, _), (c2, p2, s2, _) = a1[0], a2[0]
        if c1 != c2 or s1 != s2 or abs(p2 - p1) != L - 1 or c1 not in G.PRIMARY:
            continue
        st38 = min(p1, p2)
        seq = fa[c1][st38:st38 + L]
        if len(seq) == L and is_acgt(seq):
            out.append((c1, st38, st38 + L, seq))
    print(f"    ChromHMM-rule background: {n} windows from {tries} hg19 draws; rule-region hg19 bp="
          f"{region_bp:.4e}", flush=True)
    return pd.DataFrame(out, columns=["chr", "start", "end", "seq"]), dict(tries=tries, region_bp_hg19=region_bp)


# ------------------------------------------------------------------------------ annotation
def annotate(df, gi, Bn, rm, cc, sizes, tbp, lo38to19, H):
    """Add all per-window features to df (columns chr,start,end,seq)."""
    df = df.copy().reset_index(drop=True)
    df["gc_local"] = gc_frac(df["seq"].tolist())
    df["tata_bits"] = best_tata(df["seq"].tolist(), tbp)
    df["tata_flag"] = df["tata_bits"] >= THR
    mid = ((df["start"] + df["end"]) // 2).to_numpy()
    df["gc_100kb"] = G.regional_gc(df["chr"].to_numpy(), mid, Bn)
    iso = G.iso_class(df["gc_100kb"].to_numpy())
    df["isochore"] = [G.ISO_NAMES[i] if i >= 0 else "NA" for i in iso]
    ga = gi.annotate(df["chr"].to_numpy(), df["start"].to_numpy(), df["end"].to_numpy())
    for c in ga.columns:
        df[c] = ga[c].to_numpy()
    R = G.base_code_fractions(rm, df["chr"], df["start"], df["end"], len(G.RMSK_CODES), sizes)
    for k, v in G.RMSK_CODES.items():
        df[f"rm_{k.replace('/', '_')}"] = R[:, v]
    Cc = G.base_code_fractions(cc, df["chr"], df["start"], df["end"], len(G.CCRE_CODES), sizes)
    for k, v in G.CCRE_CODES.items():
        df[f"cc_{k}"] = Cc[:, v]
    c19, p19, _ = G.liftover_points(lo38to19, df["chr"].to_numpy(), mid)
    M = G.chromhmm_state_at(H, c19, p19)
    df["hmm_lifted"] = ~np.isnan(p19) & (M > 0).all(1)
    for j, cell in enumerate(G.CHROMHMM_CELLS):
        df[f"hmm_{cell}"] = M[:, j]
    return df


def derived(df):
    """Binary / continuous metric columns used in the summary (NaN = not evaluable)."""
    m = pd.DataFrame(index=df.index)
    m["gc_local"] = df["gc_local"]
    m["tata_rate"] = df["tata_flag"].astype(float)
    m["gc_100kb"] = df["gc_100kb"]
    ok_iso = df["isochore"] != "NA"
    for nme in G.ISO_NAMES:
        m[f"iso_{nme}"] = np.where(ok_iso, (df["isochore"] == nme).astype(float), np.nan)
    m["iso_AT_L1L2"] = np.where(ok_iso, df["isochore"].isin(["L1", "L2"]).astype(float), np.nan)
    for comp in ("promoter", "exon", "intron", "intergenic"):
        m[f"comp_{comp}"] = (df["compartment"] == comp).astype(float)
    m["dist_tss_kb"] = df["dist_tss"] / 1000.0
    m["dist_gene_all_kb"] = df["dist_gene_all"] / 1000.0
    m["dist_gene_pc_kb"] = df["dist_gene_pc"] / 1000.0
    m["ge15kb_gene_all"] = (df["dist_gene_all"] >= 15000).astype(float)
    m["ge15kb_gene_pc"] = (df["dist_gene_pc"] >= 15000).astype(float)
    m["rep_bp_any"] = 1.0 - df["rm_none"]
    m["rep_bp_L1"] = df["rm_LINE_L1"]
    m["rep_bp_Alu"] = df["rm_SINE_Alu"]
    m["rep_L1_any"] = (df["rm_LINE_L1"] > 0).astype(float)
    m["rep_L1_major"] = (df["rm_LINE_L1"] >= 0.5).astype(float)
    m["rep_Alu_any"] = (df["rm_SINE_Alu"] > 0).astype(float)
    m["rep_LTR_any"] = (df["rm_LTR"] > 0).astype(float)
    m["rep_simple_lowc_any"] = ((df["rm_Simple_repeat"] + df["rm_Low_complexity"]) > 0).astype(float)
    m["ccre_any"] = (df["cc_none"] < 1.0).astype(float)
    m["ccre_ELS"] = ((df["cc_pELS"] + df["cc_dELS"]) > 0).astype(float)
    m["ccre_PLS"] = (df["cc_PLS"] > 0).astype(float)
    hc = df[[f"hmm_{c}" for c in G.CHROMHMM_CELLS]].to_numpy()
    lift = df["hmm_lifted"].to_numpy()
    enh = np.isin(hc, ENH_STATES)
    m["hmm_nonenh_all9"] = np.where(lift, (~enh).all(1).astype(float), np.nan)
    m["hmm_enh_any"] = np.where(lift, enh.any(1).astype(float), np.nan)
    m["hmm_strongenh_all9"] = np.where(lift, np.isin(hc, (4, 5)).all(1).astype(float), np.nan)
    m["hmm_13_all9"] = np.where(lift, (hc == 13).all(1).astype(float), np.nan)
    m["hmm_13_majority"] = np.where(lift, ((hc == 13).sum(1) >= 5).astype(float), np.nan)
    m["hmm_repressed_any"] = np.where(lift, (hc == 12).any(1).astype(float), np.nan)
    m["hmm_txn_any"] = np.where(lift, np.isin(hc, (9, 10, 11)).any(1).astype(float), np.nan)
    m["hmm_promoter_any"] = np.where(lift, np.isin(hc, (1, 2, 3)).any(1).astype(float), np.nan)   # post hoc
    for k in range(1, 16):                                                                         # post hoc
        m[f"hmm_state{k:02d}_frac"] = np.where(lift, (hc == k).mean(1), np.nan)
    return m


MEDIAN_METRICS = {"dist_tss_kb", "dist_gene_all_kb", "dist_gene_pc_kb"}


def boot_group(M, rng):
    """Point estimates + bootstrap replicate vectors for every metric column of M."""
    n = len(M)
    idx = rng.randint(0, n, size=(B, n)).astype(np.int32)
    est, reps = {}, {}
    for col in M.columns:
        v = M[col].to_numpy(dtype=float)
        if not np.isfinite(v).any():
            est[col] = np.nan; reps[col] = np.full(B, np.nan); continue
        if col in MEDIAN_METRICS:
            est[col] = float(np.nanmedian(v))
            reps[col] = np.nanmedian(v[idx], axis=1)
        else:
            est[col] = float(np.nanmean(v))
            reps[col] = np.nanmean(v[idx], axis=1)
    return est, reps


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def put_ci(row, name, tup):
    """Store a CI both as the legacy string column and as numeric <name>_lo / <name>_hi columns (fix round 1)."""
    lo, hi = float(tup[0]), float(tup[1])
    row[name] = str((lo, hi)); row[name + "_lo"] = lo; row[name + "_hi"] = hi
    return row


def p_str(p, log10p):
    """Printable p-value; below the double range give the exact exponent from log10 p."""
    if p > 0:
        return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
    return f"10^{log10p:.0f}"


def log10_chi2_sf(stat, df):
    """log10 of the chi-square upper tail; below the double range scipy's logsf returns -inf, so use the
    asymptotic expansion of the regularised upper incomplete gamma Q(a, x), a = df/2, x = stat/2:
    Q ~ x^(a-1) e^(-x) / Gamma(a) * sum_k (a-1)(a-2)...(a-k) / x^k   (accurate for x >> a)."""
    from scipy.special import gammaln
    lp = float(chi2.logsf(stat, df))
    if np.isfinite(lp):
        return lp / np.log(10)
    a, x = df / 2.0, stat / 2.0
    term, s = 1.0, 1.0
    for k in range(1, 60):
        term *= (a - k) / x
        s += term
        if abs(term) < 1e-17:
            break
    return float(((a - 1) * np.log(x) - x - gammaln(a) + np.log(s)) / np.log(10))


def gof_test(obs, exp_counts):
    """Chi-square goodness of fit of obs against expected counts (rescaled to obs total)."""
    e = np.asarray(exp_counts, float); e = e / e.sum() * obs.sum()
    stat = float(((obs - e) ** 2 / e).sum()); df = len(obs) - 1
    return stat, df, float(chi2.sf(stat, df)), log10_chi2_sf(stat, df), len(obs)


def homogeneity_test(a, b):
    """2 x K chi-square test of homogeneity (columns with zero total dropped)."""
    tab = np.vstack([a, b]).astype(float); tab = tab[:, tab.sum(0) > 0]
    stat, p, df, _ = chi2_contingency(tab, correction=False)
    return float(stat), int(df), float(p), log10_chi2_sf(stat, df), tab.shape[1]


def gc_standardised_tata(tgt_gc, tgt_flag, ref_gc, ref_flag, edges):
    """Expected TATA-flag rate of the target windows if each had the reference group's TATA-flag rate
    at its local-GC bin. Bins without reference windows borrow the nearest non-empty bin."""
    nb = len(edges) - 1
    rb = np.clip(np.digitize(ref_gc, edges[1:-1]), 0, nb - 1)
    tb = np.clip(np.digitize(tgt_gc, edges[1:-1]), 0, nb - 1)
    cnt = np.bincount(rb, minlength=nb).astype(float); hit = np.bincount(rb, weights=ref_flag, minlength=nb)
    ok = np.flatnonzero(cnt > 0)
    rate = np.full(nb, np.nan); rate[ok] = hit[ok] / cnt[ok]
    for j in np.flatnonzero(cnt == 0):
        rate[j] = rate[ok[np.argmin(np.abs(ok - j))]]
    return float(np.mean(rate[tb])), float(np.mean(tgt_flag))


# ------------------------------------------------------------------------------ dataset loaders
def load_cohn(fa):
    parts = []
    for sp in ("train", "test"):
        for cl in ("positive", "negative"):
            d = pd.read_csv(os.path.join(COHN_DIR, f"{sp}_{cl}.csv.gz"))
            d["split"] = sp; d["group"] = "POS" if cl == "positive" else "NEG"
            parts.append(d)
    d = pd.concat(parts, ignore_index=True).rename(columns={"region": "chr"})
    d["seq"] = fetch(fa, d["chr"], d["start"], d["end"])   # strand-invariant features only
    return d[["id", "split", "group", "chr", "start", "end", "strand", "seq"]]


def load_cohn_pos_intervals():
    parts = [pd.read_csv(os.path.join(COHN_DIR, f"{sp}_positive.csv.gz")) for sp in ("train", "test")]
    return pd.concat(parts, ignore_index=True).rename(columns={"region": "chr"})


def load_nt_mapped():
    """nt_map rows + benchmark sequence + verbatim iEnhancer-2L origin (class, published split, header)."""
    m = pd.read_csv(NT_MAP, keep_default_na=False, na_values=[""])
    seqs = {sp: data_nt._read("enhancers", sp)[0] for sp in ("train", "test")}
    m["seq_ds"] = [seqs[sp][i] for sp, i in zip(m["split"], m["idx"])]
    info = {}
    for (sp, cl), f in V.IEN_FILES.items():
        for h, s in V.read_fasta(os.path.join(V.IEN, f)):
            info.setdefault(s, []).append((sp, cl))
    hdr = {}
    for f in V.ELM_FILES.values():
        for h, s in V.read_fasta(os.path.join(V.IEN, f)):
            hdr.setdefault(s, []).append(h)
    m["ien_class"] = m["seq_ds"].map(lambda s: "|".join(sorted({c for _, c in info.get(s, [])})))
    m["ien_split"] = m["seq_ds"].map(lambda s: "|".join(sorted({p for p, _ in info.get(s, [])})))
    m["ien_header"] = m["seq_ds"].map(lambda s: "|".join(sorted(hdr.get(s, []))))
    m["origin"] = np.where(m["ien_class"] != "", "iEnhancer_original", "synthetic")
    assert (m["genomic"].astype(bool) == (m["origin"] == "iEnhancer_original")).all(), \
        "GENOMIC call != verbatim iEnhancer identity (run verify_inputs.py)"
    return m


def header_concordance(m, lo38to19):
    """For iEnhancer training originals (header 'CHRn_start_end', hg19 per the source files), compare the
    header start with our hg38 minimap2 locus lifted to hg19. -> summary dict."""
    g = m[m["genomic"].astype(bool)].drop_duplicates("seq_ds")
    n = same = diffchr = diffpos = unlift = 0
    for r in g.itertuples():
        hs = [h for h in r.ien_header.split("|") if h.upper().startswith("CHR") and h.count("_") == 2]
        if not hs:
            continue
        c, s, _ = hs[0].split("_"); c = "chr" + (c[3:].upper() if c[3:].upper() in ("X", "Y") else c[3:])
        n += 1
        res = lo38to19.convert_coordinate(r.chr, int(r.r_st))
        if not res:
            unlift += 1; continue
        c19, p19, st19, _ = res[0]
        s19 = p19 if st19 == "+" else p19 - (int(r.r_en) - int(r.r_st) - 1)
        if c19 != c:
            diffchr += 1
        elif int(s19) != int(s):
            diffpos += 1
        else:
            same += 1
    return dict(n_train_originals_with_coord_header=n, header_equals_lifted_map=same,
                same_chrom_other_pos=diffpos, other_chrom=diffchr, unliftable=unlift,
                frac_concordant=same / n if n else np.nan)


# ------------------------------------------------------------------------------ main
def val(summ, ds, g, k):
    r = summ[(summ["dataset"] == ds) & (summ["group"] == g) & (summ["metric"] == k)]
    return r.iloc[0]


def d1_row(test, ds, met, REPS, summ, g_pos, g_neg, g_uni, g_rule, verdict_rule=True):
    P = REPS.get((ds, g_pos, met)) if g_pos else None
    N, U, R = (REPS[(ds, g, met)] for g in (g_neg, g_uni, g_rule))
    e = {g: val(summ, ds, g, met).value for g in (g_neg, g_uni, g_rule)}
    un = e[g_uni] - e[g_neg]; b = e[g_uni] - e[g_rule]
    f = b / un if un != 0 else np.nan
    un_ci = ci(U - N); f_ci = ci((U - R) / (U - N))
    row = dict(test=test, dataset=ds, metric=met, neg=g_neg, bg_uni=g_uni, rule_bg=g_rule,
               term_bguni_minus_rule=b, term_rule_minus_neg=e[g_rule] - e[g_neg], bguni_minus_neg=un, f_rule=f)
    put_ci(row, "term_bguni_minus_rule_ci", ci(U - R)); put_ci(row, "term_rule_minus_neg_ci", ci(R - N))
    put_ci(row, "bguni_minus_neg_ci", un_ci); put_ci(row, "f_rule_ci", f_ci)
    if P is not None:
        ep = val(summ, ds, g_pos, met).value
        row.update(gap_pos_minus_neg=ep - e[g_neg], term_pos_minus_bguni=ep - e[g_uni],
                   share_of_gap_from_neg_side=un / (ep - e[g_neg]) if ep != e[g_neg] else np.nan)
        put_ci(row, "gap_ci", ci(P - N)); put_ci(row, "term_pos_minus_bguni_ci", ci(P - U))
        put_ci(row, "share_of_gap_from_neg_side_ci", ci((U - N) / (P - N)))
    if verdict_rule:
        if un_ci[0] <= 0 <= un_ci[1]:
            v = "NEG_NOT_SHIFTED_VS_GENOME"
        elif f >= 0.67 and f_ci[0] > 0.33:
            v = "EXPLAINED"
        elif f >= 0.33:
            v = "PARTIAL"
        else:
            v = "NOT_EXPLAINED"
        row["verdict"] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="subsample for a smoke test (outputs *_QUICKTEST*)")
    args = ap.parse_args()
    global PFX
    if args.quick:
        PFX = PFX + "_QUICKTEST"
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    from pyliftover import LiftOver
    fa = G.fasta(upper=True)
    sizes = G.chrom_sizes()
    tbp = C.load_tbp()
    print("loading annotation caches ...", flush=True)
    Bn = G.bins(); gi = G.GeneIndex()
    H = G.chromhmm()
    nbg = 400 if args.quick else N_BG
    meta = {}

    raw = []   # (dataset, group, frame with chr,start,end,seq [+id,split])
    # ================================================================= COHN
    cohn = load_cohn(fa)
    if args.quick:
        cohn = pd.concat([g.sample(500, random_state=SEED) for _, g in cohn.groupby("group")], ignore_index=True)
    neg = cohn[cohn["group"] == "NEG"].reset_index(drop=True)
    pos = cohn[cohn["group"] == "POS"].reset_index(drop=True)
    print(f"[cohn] POS={len(pos)} NEG={len(neg)}; sampling backgrounds ...", flush=True)
    whole = lambda L: {c: (np.array([0]), np.array([sizes[c] - L + 1])) for c in sizes}  # noqa: E731
    whole_fn = lambda c, L: (np.array([0]), np.array([sizes[c] - L + 1]))  # noqa: E731
    Ls = (neg["end"] - neg["start"]).to_numpy()
    assert (Ls == 500).all()
    bg_uni, meta["cohn_bg_uni_tries"] = sample_windows(fa, rng, nbg, 500, whole(500))
    bg_chr = sample_matched(fa, rng, neg["chr"].to_numpy(), Ls, whole_fn)
    # the positives used for the exclusion are ALL cohn positives, also in --quick mode
    allpos = load_cohn_pos_intervals()
    pos_by_chr = {c: (g["start"].to_numpy(), g["end"].to_numpy()) for c, g in allpos.groupby("chr")}
    rule = {c: gi.eligible_starts(c, 500, 15000, sizes[c], extra_block=pos_by_chr.get(c)) for c in sizes}
    rule_pc = {c: gi.eligible_starts(c, 500, 15000, sizes[c], extra_block=pos_by_chr.get(c), protein_coding=True)
               for c in sizes}
    bg_rule, meta["cohn_bg_rule_tries"] = sample_windows(fa, rng, nbg, 500, rule)
    bg_rule_chr = sample_matched(fa, rng, neg["chr"].to_numpy(), Ls, lambda c, L: rule[c])
    bg_rule_pc, meta["cohn_bg_rule_pc_tries"] = sample_windows(fa, rng, nbg, 500, rule_pc)   # post hoc
    for name, d in (("POS", pos), ("NEG", neg), ("BG_uni", bg_uni), ("BG_chr", bg_chr),
                    ("BG_rule", bg_rule), ("BG_rule_chr", bg_rule_chr), ("BG_rule_pc", bg_rule_pc)):
        raw.append(("cohn", name, d))
    meta["cohn_rule_admissible_start_bp"] = sum(float(np.sum(e - s)) for s, e in rule.values())
    meta["cohn_rule_pc_admissible_start_bp"] = sum(float(np.sum(e - s)) for s, e in rule_pc.values())
    meta["genome_primary_bp"] = float(sum(sizes.values()))

    # ================================================================= NT_ENHANCERS
    m = load_nt_mapped()
    lo38to19 = LiftOver(G.CHAIN_38_19)
    meta.update({f"nt_hdr_{k}": v for k, v in header_concordance(m, lo38to19).items()})
    gen = m[m["genomic"].astype(bool)].copy()
    gen["start"] = gen["r_st"].astype(int); gen["end"] = gen["r_en"].astype(int)
    gen["group"] = np.where(gen["label"] == 1, "POS", "NEG")
    gen = gen.sort_values(["split", "idx"])
    nlab = gen.groupby(["chr", "start", "end"])["label"].nunique()
    n_conflict = int((nlab > 1).sum())
    gen_u = gen.drop_duplicates(["chr", "start", "end", "label"]).reset_index(drop=True)
    gen_u["seq"] = gen_u["seq_ds"]
    meta["nt_genomic_unique_loci_POS"] = int((gen_u.group == "POS").sum())
    meta["nt_genomic_unique_loci_NEG"] = int((gen_u.group == "NEG").sum())
    meta["nt_genomic_len_not200"] = int(((gen_u.end - gen_u.start) != 200).sum())
    meta["nt_genomic_mapq0_or_multibest"] = int(((gen_u.mapq == 0) | (gen_u.n_best_loci > 1)).sum())
    if args.quick:
        gen_u = pd.concat([g.sample(min(300, len(g)), random_state=SEED) for _, g in gen_u.groupby("group")], ignore_index=True)
    npos, nneg = gen_u[gen_u.group == "POS"].reset_index(drop=True), gen_u[gen_u.group == "NEG"].reset_index(drop=True)
    print(f"[nt] genomic unique loci POS={len(npos)} NEG={len(nneg)} (label-conflict loci={n_conflict})", flush=True)
    nt_bg_uni, meta["nt_bg_uni_tries"] = sample_windows(fa, rng, nbg, 200, whole(200))
    nt_bg_chr = sample_matched(fa, rng, nneg["chr"].to_numpy(), (nneg["end"] - nneg["start"]).to_numpy(), whole_fn)
    lo19to38 = LiftOver(G.CHAIN_19_38)
    nt_bg_rule, mr = sample_chromhmm_rule(fa, rng, H, lo19to38, nbg, 200, lambda st: ~np.isin(st, ENH_STATES))
    nt_bg_rule13, mr13 = sample_chromhmm_rule(fa, rng, H, lo19to38, nbg, 200, lambda st: st == 13)
    meta.update({f"nt_rule_nonenh_{k}": v for k, v in mr.items()})
    meta.update({f"nt_rule_state13_{k}": v for k, v in mr13.items()})
    del lo19to38
    for name, d in (("POS", npos), ("NEG", nneg), ("BG_uni", nt_bg_uni), ("BG_chr", nt_bg_chr),
                    ("BG_rule", nt_bg_rule), ("BG_rule13", nt_bg_rule13)):
        raw.append(("nt_enhancers", name, d))

    # ================================================================= ANNOTATE (one pass)
    parts = []
    for ds, name, d in raw:
        x = d[["chr", "start", "end", "seq"]].copy()
        x["dataset"] = ds; x["group"] = name
        if "id" in d:
            x["id"] = d["id"].to_numpy(); x["split"] = d["split"].to_numpy()
        elif "idx" in d:
            x["id"] = (d["split"] + "_" + d["idx"].astype(str)).to_numpy(); x["split"] = d["split"].to_numpy()
            x["type_label"] = d["type_label"].to_numpy(); x["ien_class"] = d["ien_class"].to_numpy()
        parts.append(x)
    allw = pd.concat(parts, ignore_index=True)
    print(f"annotating {len(allw)} windows ... t={time.time()-t0:.0f}s", flush=True)
    rm = G.rmsk_table(); cc = G.ccre_table()
    allw = annotate(allw, gi, Bn, rm, cc, sizes, tbp, lo38to19, H)
    del rm, cc, lo38to19
    allw.drop(columns=["seq"]).to_csv(PFX + "_intervals.csv.gz", index=False, compression="gzip")
    print(f"annotation done t={time.time()-t0:.0f}s", flush=True)

    # ================================================================= nt ORIGIN CENSUS (all 15,368 sequences)
    syn_rows = []
    for (split, lab, org), g in m.groupby(["split", "label", "origin"]):
        sq = g["seq_ds"].tolist()
        gcv = gc_frac(sq); tb = (best_tata(sq, tbp) >= THR).astype(float)
        r = np.random.RandomState(SEED)
        idx = r.randint(0, len(sq), size=(B, len(sq)))
        gci, tbi = ci(gcv[idx].mean(1)), ci(tb[idx].mean(1))
        syn_rows.append(dict(split=split, label=int(lab), origin=org, n_rows=len(sq), n_unique=int(g["seq_ds"].nunique()),
                             len_200=int((g["length"] == 200).sum()), len_199=int((g["length"] == 199).sum()),
                             ien_strong_rows=int(g["ien_class"].str.contains("strong").sum()),
                             ien_weak_rows=int(g["ien_class"].str.contains("weak").sum()),
                             ien_non_rows=int(g["ien_class"].str.contains("non").sum()),
                             from_published_test=int(g["ien_split"].str.contains("test").sum()),
                             gc_local=float(gcv.mean()), gc_lo=gci[0], gc_hi=gci[1],
                             tata_rate=float(tb.mean()), tata_lo=tbi[0], tata_hi=tbi[1]))
    for lab in (0, 1):   # pooled over splits: synthetic fraction (census) + composition by origin
        g = m[m["label"] == lab]
        for org in ("iEnhancer_original", "synthetic"):
            gg = g[g["origin"] == org]; sq = gg["seq_ds"].tolist()
            gcv = gc_frac(sq); tb = (best_tata(sq, tbp) >= THR).astype(float)
            r = np.random.RandomState(SEED); idx = r.randint(0, len(sq), size=(B, len(sq)))
            fr = (g["origin"] == org).to_numpy(float); ridx = r.randint(0, len(fr), size=(B, len(fr)))
            syn_rows.append(dict(split="all", label=lab, origin=org, n_rows=len(sq), n_unique=int(gg["seq_ds"].nunique()),
                                 len_200=int((gg["length"] == 200).sum()), len_199=int((gg["length"] == 199).sum()),
                                 frac_of_class=float(fr.mean()), frac_lo=ci(fr[ridx].mean(1))[0], frac_hi=ci(fr[ridx].mean(1))[1],
                                 gc_local=float(gcv.mean()), gc_lo=ci(gcv[idx].mean(1))[0], gc_hi=ci(gcv[idx].mean(1))[1],
                                 tata_rate=float(tb.mean()), tata_lo=ci(tb[idx].mean(1))[0], tata_hi=ci(tb[idx].mean(1))[1]))
    syn_df = pd.DataFrame(syn_rows)
    syn_df.to_csv(PFX + "_nt_origin.csv", index=False)
    print(syn_df.to_string(), flush=True)

    # ================================================================= paper-number sanity (TATA-box fractions)
    san = []
    cohn_all = load_cohn(fa) if args.quick else cohn
    for grp in ("POS", "NEG"):
        sq = cohn_all[cohn_all.group == grp]["seq"].tolist()
        san.append(dict(dataset="cohn", cls=grp, n=len(sq), tata_box_fraction=float((C.tata_hit_counts(sq, tbp) >= 1).mean()),
                        gc_mean=float(np.mean(gc_frac(sq)))))
    for lab, grp in ((1, "POS"), (0, "NEG")):
        sq = m[m.label == lab]["seq_ds"].tolist()
        san.append(dict(dataset="nt_enhancers", cls=grp, n=len(sq), tata_box_fraction=float((C.tata_hit_counts(sq, tbp) >= 1).mean()),
                        gc_mean=float(np.mean(gc_frac(sq)))))
    pd.DataFrame(san).to_csv(PFX + "_sanity_paper_numbers.csv", index=False)

    # ================================================================= SUMMARY + BOOTSTRAP
    rows, REPS = [], {}
    for (ds, grp), g in allw.groupby(["dataset", "group"], sort=False):
        Md = derived(g.reset_index(drop=True))
        est, reps = boot_group(Md, rng)
        for k in Md.columns:
            lo, hi = ci(reps[k])
            nn = int(np.isfinite(Md[k].to_numpy(dtype=float)).sum())
            rows.append(dict(dataset=ds, group=grp, metric=k, value=est[k], ci_lo=lo, ci_hi=hi, n=nn))
            REPS[(ds, grp, k)] = reps[k]
    summ = pd.DataFrame(rows)
    summ.to_csv(PFX + "_summary.csv", index=False)

    key = ["gc_local", "tata_rate", "gc_100kb", "iso_AT_L1L2", "iso_L1", "comp_intergenic", "comp_promoter",
           "dist_tss_kb", "ge15kb_gene_pc", "ge15kb_gene_all", "rep_bp_any", "rep_bp_L1", "rep_L1_major",
           "ccre_any", "ccre_ELS", "hmm_nonenh_all9", "hmm_13_all9", "hmm_enh_any"]
    tab = summ[summ.metric.isin(key)].copy()
    tab["cell"] = [f"{v:.3f} [{a:.3f}, {b:.3f}]" for v, a, b in zip(tab.value, tab.ci_lo, tab.ci_hi)]
    wide = tab.pivot_table(index="metric", columns=["dataset", "group"], values="cell", aggfunc="first").reindex(key)
    wide.to_csv(PFX + "_table.csv")

    # ================================================================= TESTS
    tests = []
    for ds in ("cohn", "nt_enhancers"):
        for met in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2"):
            tests.append(d1_row("D1_decomposition", ds, met, REPS, summ, "POS", "NEG", "BG_uni", "BG_rule"))
    for met in ("gc_local", "gc_100kb", "tata_rate", "iso_AT_L1L2"):
        r_ = d1_row("D1_sensitivity_chr_matched", "cohn", met, REPS, summ, None, "NEG", "BG_chr", "BG_rule_chr",
                    verdict_rule=False)
        tests.append(r_)
        tests.append(d1_row("D1_posthoc_rule_proteincoding", "cohn", met, REPS, summ, None, "NEG", "BG_uni",
                            "BG_rule_pc", verdict_rule=False))
        tests.append(d1_row("D1_secondary_state13", "nt_enhancers", met, REPS, summ, None, "NEG", "BG_uni",
                            "BG_rule13", verdict_rule=False))

    d2_metrics = ("iso_AT_L1L2", "rep_L1_major", "rep_L1_any", "gc_100kb", "gc_local",
                  "rep_simple_lowc_any", "rep_bp_any")       # last two: post hoc, descriptive
    for ds in ("cohn", "nt_enhancers"):
        grps = ("NEG", "BG_uni", "BG_rule", "POS") + (("BG_rule13",) if ds == "nt_enhancers" else ())
        for grp in grps:
            g = allw[(allw.dataset == ds) & (allw.group == grp)].reset_index(drop=True)
            Md = derived(g)
            fl = g["tata_flag"].to_numpy(bool)
            for met in d2_metrics:
                v = Md[met].to_numpy(dtype=float)
                a_, b_ = v[fl & np.isfinite(v)], v[~fl & np.isfinite(v)]
                if len(a_) == 0 or len(b_) == 0:
                    continue
                r1 = np.random.RandomState(SEED)
                d_rep = (a_[r1.randint(0, len(a_), (B, len(a_)))].mean(1)
                         - b_[r1.randint(0, len(b_), (B, len(b_)))].mean(1))
                binary = met not in ("gc_100kb", "gc_local", "rep_bp_any")
                rr = a_.mean() / b_.mean() if b_.mean() > 0 else np.nan
                p = np.nan
                if binary:
                    p = fisher_exact([[int(a_.sum()), int(len(a_) - a_.sum())],
                                      [int(b_.sum()), int(len(b_) - b_.sum())]])[1]
                verdict = ""
                lo, hi = ci(d_rep)
                if binary and grp == "NEG" and met in ("iso_AT_L1L2", "rep_L1_major"):
                    verdict = "CONCENTRATED" if (rr >= 1.2 and (lo > 0 or hi < 0)) else "NOT_CONCENTRATED"
                tests.append(put_ci(dict(test="D2_flagged_vs_nonflagged", dataset=ds, group=grp, metric=met,
                                         flagged_n=len(a_), nonflagged_n=len(b_), flagged=a_.mean(),
                                         nonflagged=b_.mean(), diff=a_.mean() - b_.mean(), risk_ratio=rr,
                                         fisher_p=p, verdict=verdict), "diff_ci", (lo, hi)))

    a = val(summ, "nt_enhancers", "NEG", "hmm_nonenh_all9"); b = val(summ, "nt_enhancers", "POS", "hmm_enh_any")
    tests.append(dict(test="D3_nt_chromhmm", dataset="nt_enhancers", group="NEG|POS",
                      metric="NEG hmm_nonenh_all9 | POS hmm_enh_any", flagged=a.value, nonflagged=b.value,
                      diff_ci=f"NEG [{a.ci_lo:.3f}, {a.ci_hi:.3f}]; POS [{b.ci_lo:.3f}, {b.ci_hi:.3f}]",
                      flagged_n=a.n, nonflagged_n=b.n,
                      verdict="VERIFIED" if (a.value >= 0.8 and b.value >= 0.8) else "NOT_VERIFIED"))
    n_ = val(summ, "cohn", "NEG", "ge15kb_gene_pc"); u_ = val(summ, "cohn", "BG_uni", "ge15kb_gene_pc")
    al = val(summ, "cohn", "NEG", "ge15kb_gene_all")
    tests.append(dict(test="D4_cohn_rule", dataset="cohn", group="NEG|BG_uni",
                      metric="ge15kb_gene_pc NEG | BG_uni (diff = NEG ge15kb_gene_all)",
                      flagged=n_.value, nonflagged=u_.value, diff=al.value,
                      diff_ci=f"NEG [{n_.ci_lo:.3f}, {n_.ci_hi:.3f}]; BG_uni [{u_.ci_lo:.3f}, {u_.ci_hi:.3f}]",
                      verdict="VERIFIED" if (n_.value >= 0.8 and u_.value < n_.value - 0.2) else "NOT_VERIFIED"))

    # D5 (post hoc): GC-standardised TATA rate
    for ds in ("cohn", "nt_enhancers"):
        sub = allw[allw.dataset == ds]
        edges = np.quantile(sub["gc_local"].to_numpy(), np.linspace(0, 1, N_GCBINS + 1))
        edges[0], edges[-1] = -1.0, 2.0
        for tgt in ("NEG", "POS"):
            for ref in ("BG_uni", "BG_rule"):
                T = sub[sub.group == tgt]; R = sub[sub.group == ref]
                tg, tf = T["gc_local"].to_numpy(), T["tata_flag"].to_numpy(float)
                rg, rf = R["gc_local"].to_numpy(), R["tata_flag"].to_numpy(float)
                exp_, obs = gc_standardised_tata(tg, tf, rg, rf, edges)
                r1 = np.random.RandomState(SEED)
                reps_e, reps_o = [], []
                for _ in range(B):
                    it = r1.randint(0, len(tg), len(tg)); ir = r1.randint(0, len(rg), len(rg))
                    e1, o1 = gc_standardised_tata(tg[it], tf[it], rg[ir], rf[ir], edges)
                    reps_e.append(e1); reps_o.append(o1)
                reps_e, reps_o = np.array(reps_e), np.array(reps_o)
                r5 = dict(test="D5_posthoc_gc_standardised_tata", dataset=ds, group=tgt, metric="tata_rate",
                          rule_bg=ref, flagged=obs, nonflagged=exp_, diff=obs - exp_, risk_ratio=obs / exp_,
                          ref_raw_rate=float(rf.mean()))
                put_ci(r5, "diff_ci", ci(reps_o - reps_e)); put_ci(r5, "f_rule_ci", ci(reps_o / reps_e))
                tests.append(r5)
    pd.DataFrame(tests).to_csv(PFX + "_tests.csv", index=False)

    # ================================================================= CHROMOSOMES
    crow, ctests = [], []
    acgt = {c: float(Bn[c][0].sum()) for c in sizes}
    MATCH = {"cohn": "BG_rule_pc", "nt_enhancers": "BG_rule13"}          # fix round 1 (post hoc)
    for ds in ("cohn", "nt_enhancers"):
        sub = allw[allw.dataset == ds]
        cnt = {g: sub[sub.group == g]["chr"].value_counts() for g in sub.group.unique()}
        chroms = list(G.PRIMARY)
        negc = np.array([cnt["NEG"].get(c, 0) for c in chroms], float)
        posc = np.array([cnt["POS"].get(c, 0) for c in chroms], float)
        eu = np.array([acgt[c] for c in chroms]); eu = eu / eu.sum() * negc.sum()
        er = np.array([cnt["BG_rule"].get(c, 0) for c in chroms], float) + 0.5   # empirical rule territory
        er = er / er.sum() * negc.sum()
        mc = np.array([cnt[MATCH[ds]].get(c, 0) for c in chroms], float)
        em = (mc + 0.5) / (mc + 0.5).sum() * negc.sum()
        for i, c in enumerate(chroms):
            row = dict(dataset=ds, chr=c, NEG=int(negc[i]), exp_uniform=eu[i], exp_rule=er[i],
                       ratio_vs_uniform=negc[i] / eu[i], ratio_vs_rule=negc[i] / er[i],
                       neg_frac=negc[i] / negc.sum(), pos_frac=posc[i] / posc.sum(),
                       genome_frac=acgt[c] / sum(acgt.values()),
                       match_bg=MATCH[ds], exp_match=em[i], ratio_vs_match=negc[i] / em[i])
            for g in cnt:
                row[f"n_{g}"] = int(cnt[g].get(c, 0))
            crow.append(row)
        chi_u = chisquare(negc, eu); chi_r = chisquare(negc, er)
        crow.append(dict(dataset=ds, chr="CHISQ_stat|p", NEG=int(negc.sum()),
                         exp_uniform=chi_u.statistic, exp_rule=chi_r.statistic,
                         ratio_vs_uniform=chi_u.pvalue, ratio_vs_rule=chi_r.pvalue))
        # fix round 1: explicit test table (GOF + homogeneity), exact log10 p
        ms_ = np.array([c in ("chrY",) for c in chroms]) if ds == "nt_enhancers" else np.zeros(len(chroms), bool)
        keep = ~ms_                                            # ChromHMM rule sampler covers chr1-22, X only
        for comp, method, fn, args_ in (
                ("uniform N-free length", "GOF", gof_test, (negc, eu)),
                ("BG_rule counts + 0.5", "GOF", gof_test,
                 (negc, np.array([cnt["BG_rule"].get(c, 0) for c in chroms], float) + 0.5)),
                (f"{MATCH[ds]} counts + 0.5", "GOF", gof_test, (negc[keep], mc[keep] + 0.5)),
                ("BG_uni", "homogeneity", homogeneity_test, (negc, np.array([cnt["BG_uni"].get(c, 0) for c in chroms], float))),
                ("BG_rule", "homogeneity", homogeneity_test, (negc, np.array([cnt["BG_rule"].get(c, 0) for c in chroms], float))),
                (MATCH[ds], "homogeneity", homogeneity_test, (negc[keep], mc[keep]))):
            st_, df_, p_, lp_, k_ = fn(*args_)
            ctests.append(dict(dataset=ds, reference=comp, method=method, n_neg=int(negc.sum()), stat=st_, df=df_,
                               p=p_, log10_p=lp_, p_str=p_str(p_, lp_), chroms_used=k_))
    pd.DataFrame(crow).to_csv(PFX + "_chrom.csv", index=False)
    pd.DataFrame(ctests).to_csv(PFX + "_chrom_tests.csv", index=False)
    meta.update(dict(nt_label_conflict_loci=n_conflict, runtime_s=round(time.time() - t0),
                     nt_genomic_seqs=int(m["genomic"].astype(bool).sum()), nt_total_seqs=len(m)))
    pd.Series(meta).to_csv(PFX + "_meta.csv")
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
