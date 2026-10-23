#!/usr/bin/env python3
"""POST HOC PH5 (fix round 2): is the yeast-EMP Route-3 signature a negative-sampling artifact, or does
it follow from WHERE in the genome each histone mark's high/low windows sit? No model is fitted.

Written 2026-09-27 (fix round 2) BEFORE it was run; the reading rules below are fixed.
Motivation. PH4 found the Route-3 signature (TATA neg-excess >= 0.10, TATA AUROC < 0.5, cleaning toward
chance) on 5 of the 10 GUE yeast EMP datasets. The first report called this 'may be biological
(untested)'. A verifier pointed out that it is testable: the EMP classes are genomic windows of high vs
low ChIP-chip signal (Pham et al. 2005), and nucleosome-depleted, AT/TATA-rich promoter regions should
fall in the LOW class of occupancy and gene-body marks but in the HIGH class of promoter/5' marks.
If that explains the signature, the four EMP Route-3 labels are rule false positives with respect to
negative-set contamination (a specificity limit of Route 3), not evidence of contamination.

Data
 D1 Pham et al. 2005 (Genome Inform 16(2):3-11, PMID 16901084) release, the source that GUE (DNABERT-2
    App. B.2) and the NT histone tasks cite: jaist.ac.jp/~tran/nucleosome (now offline). Internet-Archive
    snapshot of its datasets page (members.htm, 2007-07-15: #positive / #negative per mark) and of
    whichever datasetNN.txt files the archive holds ('>probe id' / sequence / label). Fetched by
    external/gue/pham2005/fetch_pham.sh; SHA-256 recorded in the output.
 D2 S. cerevisiae genome sacCer3 and the SGD ORF table sgdGene, both from UCSC (SHA-256 recorded).

S0 Provenance and label direction.
   (a) per mark, GUE label-1 / label-0 totals (train+dev+test) vs Pham's #positive / #negative; exact
       equality on all 10 marks => GUE label 1 = Pham's 'positive' class.
   (b) for each archived Pham file: share of GUE sequences present in it with the same label.
   Pham's abstract describes the two classes as 'DNA areas with high and low relative occupancy,
   acetylation, or methylation'; with (a) this reads label 1 = high. S1 checks that reading against
   genomic location.
S1 Genomic location. Every unique EMP sequence (union over the 10 marks) is located exactly (forward or
   reverse complement) in sacCer3 by a 32-mer anchor index plus full-string confirmation; only unique
   hits are used. Location class of the window MIDPOINT: 'ORF' (inside any sgdGene ORF, either strand)
   or 'intergenic'. Secondary (4 strata): intergenic with an ORF start (ATG) <= 500 bp downstream in
   that ORF's orientation ('promoter'); other intergenic; ORF first 500 bp from the ATG ('ORF 5''); ORF
   rest. Validated against the Pham probe names where the file is archived ('i' prefix = intergenic
   probe; agreement reported).
   Label-direction check (predictions fixed now, from Pokholok et al. 2005 and nucleosome depletion at
   promoters): if label 1 = high signal, label-1 windows are LESS often intergenic than label-0
   windows for H3 and H4 (occupancy) and H3K36me3 (transcribed-region methylation), and MORE often for
   H3K9ac, H3K14ac and H4ac (acetylation 'predominantly at the beginning of genes'). Supported if >= 5
   of these 6 hold. H3K4me1/2/3 and H3K79me3 are reported without a prediction.
S2 (primary) Location-standardised TATA neg-excess. On exactly the pooled, capped sample of run_gue's
   Route-3 readout (train+dev+test, <= 4,000 per class drawn with RandomState(42); the stored
   tata_pos_rate / tata_neg_rate are reproduced, asserted), the same vertebrate TBP PSSM at 1.3 bits/pos:
     marginal excess e = r0 - r1 (share of label-0 minus label-1 windows with >= 1 TBP hit);
     within-stratum excess e_s = r0(s) - r1(s);
     standardised excess e_std = sum_s w1(s) r0(s) - r1, i.e. the negatives re-weighted to the
     positives' ORF/intergenic mix (w1 = share of located label-1 windows in stratum s);
     95 % percentile CI from 1000 class-stratified bootstrap resamples of the located windows (seed 42).
   Reading rule: 'explained by location' if the CI upper bound of e_std is < 0.10 (the Route-3
   threshold); 'not explained' if the CI lower bound is >= 0.10; otherwise 'partly explained'. The
   Route-3 clause (excess >= 0.10 and TATA AUROC < 0.5) is also re-evaluated within each stratum.
   Windows not located uniquely are excluded (counts reported; e on the located subset is reported
   next to the full-sample e). Sensitivity: the same with the 4 secondary strata.
S3 Across the 10 marks (they share 34-92 % of their windows pairwise, so these are NOT independent
   observations; descriptive p-values): Spearman rho between the intergenic-fraction difference
   (label 0 minus label 1) and the marginal TATA neg-excess, permutation p (100,000 permutations,
   seed 42); the Route-3 signature split by the sign of that difference (Fisher exact, two-sided); and
   the verifier's literature grouping (promoter/5'-enriched: H3K9ac, H3K14ac, H4ac, H3K4me3, H3K4me2 vs
   promoter-depleted/gene-body: H3, H4, H3K36me3, H3K79me3, H3K4me1): exact one-sided Mann-Whitney on
   the neg-excess and two-sided Fisher exact on the signature, labelled as the verifier's grouping.
Outputs: results/upgrades/gue_emp_provenance.csv (one row per mark), gue_emp_provenance_strata.csv
(mark x stratum x class), gue_emp_provenance_tests.csv (S0/S1/S3 summaries);
external/gue/yeast_genome/emp_window_locations.csv.gz (every unique EMP window: chrom, start, strand,
class). CPU only, single process, < 1 GB.

Run log (disclosed). First run 2026-09-27 ~13:10: a coding bug kept windows that were not located
uniquely as a spurious stratum (stratum weights summed to ~0.97, biasing e_std); found on inspection
of the stratum table, fixed (only uniquely located windows are stratified; weights asserted to sum to
1), and re-run before any number was used. share_of_excess_explained is reported only where the
marginal excess is >= 0.10 (it is unstable for small excesses). Reading rules unchanged.
"""
import gzip
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UP = os.path.abspath(os.path.join(HERE, ".."))
for _d in (UP, os.path.join(UP, "diagnose")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import gue_data as G  # noqa: E402

RES = os.path.join(G.ROOT, "results", "upgrades")
EXT = os.path.join(G.ROOT, "external", "gue")
PHAM = os.path.join(EXT, "pham2005")
GEN = os.path.join(EXT, "yeast_genome")
SEED = 42
BOOT = 1000
MOTIF_CAP = 4000          # run_gue.MOTIF_CAP
ANCHOR = 32
PROM_BP = 500
MARKS = ["H3", "H4", "H3K9ac", "H3K14ac", "H4ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K36me3", "H3K79me3"]
PHAM_FILE = {"H3": "01", "H4": "02", "H3K9ac": "04", "H3K14ac": "05", "H4ac": "08", "H3K4me1": "10",
             "H3K4me2": "11", "H3K4me3": "12", "H3K36me3": "13", "H3K79me3": "14"}
# S1 pre-stated direction (label 1 = high): intergenic share of label 1 LOWER (-1) or HIGHER (+1)
PREDICTED_SIGN = {"H3": -1, "H4": -1, "H3K36me3": -1, "H3K9ac": +1, "H3K14ac": +1, "H4ac": +1}
# S3 verifier's literature grouping
LIT_PROMOTER = ["H3K9ac", "H3K14ac", "H4ac", "H3K4me3", "H3K4me2"]
LIT_GENEBODY = ["H3", "H4", "H3K36me3", "H3K79me3", "H3K4me1"]
COMP = str.maketrans("ACGTN", "TGCAN")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rc(s):
    return s.translate(COMP)[::-1]


# ------------------------------------------------------------------------------------------ D1
def pham_counts():
    """#positive / #negative per mark from the archived members.htm table."""
    t = open(os.path.join(PHAM, "members_wayback_20070715.htm"), encoding="latin-1").read()
    out = {}
    for row in re.findall(r"<tr.*?</tr>", t, flags=re.S | re.I):
        txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", row)).strip()
        m = re.match(r"^(H\S+) (\d+) (\d+) ", txt)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def read_pham_file(mark):
    p = os.path.join(PHAM, f"dataset{PHAM_FILE[mark]}.txt")
    if not os.path.exists(p) or open(p, "rb").read(1) != b">":
        return None, None
    lines = [x.strip() for x in open(p).read().split("\n")]
    recs = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            j = i + 1
            seq = []
            while j < len(lines) and lines[j] and lines[j][0] in "ACGTNacgtn":
                seq.append(lines[j]); j += 1
            lab = int(lines[j])
            recs.append((lines[i][1:], "".join(seq).upper(), lab))
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(recs, columns=["probe", "sequence", "label"]), sha256(p)


# ------------------------------------------------------------------------------------------ D2
def load_genome():
    chroms, seqs = [], []
    name, buf = None, []
    with gzip.open(os.path.join(GEN, "sacCer3.fa.gz"), "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    chroms.append(name); seqs.append("".join(buf).upper())
                name, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.strip())
    chroms.append(name); seqs.append("".join(buf).upper())
    return chroms, seqs


def build_index(chroms, seqs):
    sep = "N" * ANCHOR
    g = sep.join(seqs)
    offs = []
    o = 0
    for s in seqs:
        offs.append(o); o += len(s) + len(sep)
    lut = np.full(256, 4, np.uint8)
    for i, c in enumerate(b"ACGT"):
        lut[c] = i
    b = lut[np.frombuffer(g.encode(), np.uint8)]
    n = len(b) - ANCHOR + 1
    code = np.zeros(n, np.uint64)
    bad = np.zeros(n, bool)
    for j in range(ANCHOR):
        bj = b[j:j + n]
        bad |= bj == 4
        code = (code << np.uint64(2)) | (bj & 3).astype(np.uint64)
    pos = np.nonzero(~bad)[0]
    code = code[pos]
    order = np.argsort(code, kind="stable")
    return g, np.asarray(offs), code[order], pos[order], lut


def anchor_code(s, lut):
    b = lut[np.frombuffer(s[:ANCHOR].encode(), np.uint8)]
    if (b == 4).any() or len(b) < ANCHOR:
        return None
    c = 0
    for x in b:
        c = (c << 2) | int(x)
    return np.uint64(c)


def locate(seqs, g, offs, chroms, keys, poss, lut):
    """Exact unique location of each sequence: (chrom, 0-based start, strand) or None."""
    out = {}
    for s in seqs:
        hits = []
        for strand, q in (("+", s), ("-", rc(s))):
            a = anchor_code(q, lut)
            if a is None:
                continue
            lo = np.searchsorted(keys, a, "left"); hi = np.searchsorted(keys, a, "right")
            for p in poss[lo:hi]:
                p = int(p)
                if g[p:p + len(q)] == q:
                    hits.append((p, strand))
        if len(hits) == 1:
            p, strand = hits[0]
            ci = int(np.searchsorted(offs, p, "right") - 1)
            out[s] = (chroms[ci], p - int(offs[ci]), strand, 1)
        else:
            out[s] = (None, -1, "", len(hits))
    return out


def load_orfs():
    cols = ["bin", "name", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount",
            "exonStarts", "exonEnds", "proteinID"]
    d = pd.read_csv(os.path.join(GEN, "sgdGene.txt.gz"), sep="\t", header=None, names=cols)
    return d


def classify(chrom, mid, orfs_by_chrom):
    """Return (class2, class4) for a window midpoint."""
    o = orfs_by_chrom.get(chrom)
    if o is None:
        return "intergenic", "intergenic other"
    s, e, st = o
    inside = (s <= mid) & (mid < e)
    if inside.any():
        # distance from the ATG along the ORF
        d = np.where(st[inside] == "+", mid - s[inside], e[inside] - 1 - mid)
        return "ORF", ("ORF 5'" if (d < PROM_BP).any() else "ORF rest")
    up_plus = (st == "+") & (s - PROM_BP <= mid) & (mid < s)
    up_minus = (st == "-") & (e <= mid) & (mid < e + PROM_BP)
    return "intergenic", ("promoter" if (up_plus | up_minus).any() else "intergenic other")


# ------------------------------------------------------------------------------------------ S2
def pooled_sample(task):
    """Exactly run_gue's pooled, capped Route-3 sample (same order, same RandomState draws)."""
    d = G.load_gue(task)
    seqs_all = d["train_seqs"] + d["dev_seqs"] + d["test_seqs"]
    y_all = np.concatenate([d["y_train"], d["y_dev"], d["y_test"]])
    mrng = np.random.RandomState(SEED)
    pos = [seqs_all[i] for i in np.where(y_all == 1)[0]]
    neg = [seqs_all[i] for i in np.where(y_all == 0)[0]]
    if len(pos) > MOTIF_CAP:
        pos = [pos[i] for i in mrng.choice(len(pos), MOTIF_CAP, replace=False)]
    if len(neg) > MOTIF_CAP:
        neg = [neg[i] for i in mrng.choice(len(neg), MOTIF_CAP, replace=False)]
    return d, seqs_all, y_all, pos, neg


def std_excess(h1, s1, h0, s0, strata):
    """e_std = sum_s w1(s) r0(s) - r1; returns nan if a stratum with w1>0 has no negatives."""
    r1 = h1.mean()
    assert abs(sum((s1 == k).mean() for k in strata) - 1) < 1e-9, "every located window must be in one stratum"
    tot = 0.0
    for k in strata:
        w = (s1 == k).mean()
        if w == 0:
            continue
        m0 = s0 == k
        if not m0.any():
            return np.nan
        tot += w * h0[m0].mean()
    return tot - r1


def boot_std(h1, s1, h0, s0, strata, rng):
    out = np.empty(BOOT)
    n1, n0 = len(h1), len(h0)
    for b in range(BOOT):
        i1 = rng.randint(0, n1, n1); i0 = rng.randint(0, n0, n0)
        out[b] = std_excess(h1[i1], s1[i1], h0[i0], s0[i0], strata)
    return np.nanpercentile(out, [2.5, 97.5])


def verdict(lo, hi):
    if hi < 0.10:
        return "explained by location"
    if lo >= 0.10:
        return "not explained"
    return "partly explained"


def main():
    import _common as C
    import diagnose as DG

    # ---------------- S0
    pc = pham_counts()
    assert set(pc) == set(MARKS), pc
    tbp = C.load_tbp()
    manifest = {"members_wayback_20070715.htm": sha256(os.path.join(PHAM, "members_wayback_20070715.htm")),
                "sacCer3.fa.gz": sha256(os.path.join(GEN, "sacCer3.fa.gz")),
                "sgdGene.txt.gz": sha256(os.path.join(GEN, "sgdGene.txt.gz"))}

    # ---------------- genome + ORFs
    chroms, gseqs = load_genome()
    g, offs, keys, poss, lut = build_index(chroms, gseqs)
    orfs = load_orfs()
    obc = {c: (d.cdsStart.to_numpy(), d.cdsEnd.to_numpy(), d.strand.to_numpy()) for c, d in orfs.groupby("chrom")}
    print(f"genome {sum(map(len, gseqs))} bp, {len(chroms)} sequences; {len(orfs)} ORFs", flush=True)

    loaded = {}
    union = set()
    for m in MARKS:
        loaded[m] = pooled_sample(f"emp_{m}")
        union.update(loaded[m][1])
    union = sorted(union)
    loc = locate(union, g, offs, chroms, keys, poss, lut)
    lrows = []
    for s in union:
        ch, st, strand, nh = loc[s]
        c2, c4 = (classify(ch, st + len(s) // 2, obc) if ch else (None, None))
        lrows.append((s, ch, st, strand, nh, len(s), c2, c4))
    ldf = pd.DataFrame(lrows, columns=["sequence", "chrom", "start", "strand", "n_hits", "length", "class2", "class4"])
    os.makedirs(GEN, exist_ok=True)
    ldf.to_csv(os.path.join(GEN, "emp_window_locations.csv.gz"), index=False, compression="gzip")
    ok = ldf[ldf.n_hits == 1]                     # only uniquely located windows get a class
    c2map = dict(zip(ok.sequence, ok.class2)); c4map = dict(zip(ok.sequence, ok.class4))
    print(f"located uniquely {int((ldf.n_hits == 1).sum())}/{len(ldf)} unique EMP windows; "
          f"0 hits {int((ldf.n_hits == 0).sum())}, >1 hits {int((ldf.n_hits > 1).sum())}", flush=True)

    rows, srows = [], []
    rs = pd.read_csv(os.path.join(RES, "gue_rule_sensitivity.csv")).set_index("task")
    rng = np.random.RandomState(SEED)
    val_rows = []
    for m in MARKS:
        task = f"emp_{m}"
        d, seqs_all, y_all, pos, neg = loaded[m]
        card = json.load(open(os.path.join(RES, "gue_cards", f"{task}.json")))
        r3 = card["route3_negatives"]
        n1, n0 = int((y_all == 1).sum()), int((y_all == 0).sum())
        row = dict(task=task, mark=m, pham_n_positive=pc[m][0], pham_n_negative=pc[m][1],
                   gue_n_label1=n1, gue_n_label0=n0, counts_match_pham=bool((n1, n0) == pc[m]))
        # S0(b)
        pf, psha = read_pham_file(m)
        row["pham_file_archived"] = pf is not None
        if pf is not None:
            manifest[f"dataset{PHAM_FILE[m]}.txt"] = psha
            key = dict(zip(pf.sequence, pf.label))
            same = np.array([key.get(s, -1) == int(y) for s, y in zip(seqs_all, y_all)])
            row["frac_gue_in_pham_same_label"] = round(float(same.mean()), 5)
            row["pham_n_records"] = len(pf)
            inter_probe = dict(zip(pf.sequence, pf.probe.str.startswith("i")))
            v = [(inter_probe[s], c2map[s]) for s in set(seqs_all) if s in inter_probe and s in c2map]
            agree = np.mean([(a and b == "intergenic") or ((not a) and b == "ORF") for a, b in v])
            row["probe_name_vs_midpoint_class_agreement"] = round(float(agree), 4)
            row["probe_name_n_compared"] = len(v)
            val_rows.append((m, len(v), agree))
        # Route-3 readout reproduction (asserted)
        pcodes = C.motif_match.encode_sequences(pos); ncodes = C.motif_match.encode_sequences(neg)
        ph = C.motif_match.count_hits_batch(pcodes, tbp["pssm"], DG.TATA_BITS)
        nh = C.motif_match.count_hits_batch(ncodes, tbp["pssm"], DG.TATA_BITS)
        r1_all, r0_all = float((ph >= 1).mean()), float((nh >= 1).mean())
        assert abs(round(r1_all, 4) - r3["tata_pos_rate"]) < 1e-9 and abs(round(r0_all, 4) - r3["tata_neg_rate"]) < 1e-9, \
            (task, r1_all, r0_all, r3)
        row.update(tata_pos_rate=round(r1_all, 4), tata_neg_rate=round(r0_all, 4),
                   tata_neg_excess=round(r0_all - r1_all, 4), stored_tata_neg_excess=r3["tata_neg_excess"],
                   tata_auroc=r3["tata_motif_auroc"],
                   route3_signature=bool(rs.loc[task, "route3_signature"]), route=rs.loc[task, "route"])
        # located subset
        k1 = np.array([s in c2map for s in pos]); k0 = np.array([s in c2map for s in neg])
        s1 = np.array([c2map.get(s, "") for s in pos], dtype=object); s0 = np.array([c2map.get(s, "") for s in neg], dtype=object)
        q1 = np.array([c4map.get(s, "") for s in pos], dtype=object); q0 = np.array([c4map.get(s, "") for s in neg], dtype=object)
        h1 = (ph >= 1)[k1].astype(float); h0 = (nh >= 1)[k0].astype(float)
        c1, c0 = ph[k1].astype(np.int64), nh[k0].astype(np.int64)
        s1, s0, q1, q0 = s1[k1].astype(str), s0[k0].astype(str), q1[k1].astype(str), q0[k0].astype(str)
        row.update(n_pool_label1=len(pos), n_pool_label0=len(neg), n_located_label1=int(k1.sum()),
                   n_located_label0=int(k0.sum()), frac_located=round(float((k1.sum() + k0.sum()) / (len(pos) + len(neg))), 4),
                   tata_neg_excess_located=round(float(h0.mean() - h1.mean()), 4))
        # S1
        f1, f0 = float((s1 == "intergenic").mean()), float((s0 == "intergenic").mean())
        row.update(intergenic_frac_label1=round(f1, 4), intergenic_frac_label0=round(f0, 4),
                   intergenic_diff_0_minus_1=round(f0 - f1, 4),
                   promoter_frac_label1=round(float((q1 == "promoter").mean()), 4),
                   promoter_frac_label0=round(float((q0 == "promoter").mean()), 4))
        if m in PREDICTED_SIGN:
            row["predicted_sign_label1_minus_label0"] = PREDICTED_SIGN[m]
            row["direction_prediction_holds"] = bool(np.sign(f1 - f0) == PREDICTED_SIGN[m])
        # S2 primary + secondary
        for tag, a1, a0, strata in (("", s1, s0, ["ORF", "intergenic"]),
                                    ("_4strata", q1, q0, ["promoter", "intergenic other", "ORF 5'", "ORF rest"])):
            e = std_excess(h1, a1, h0, a0, strata)
            lo, hi = boot_std(h1, a1, h0, a0, strata, rng)
            row[f"tata_excess_std{tag}"] = round(float(e), 4)
            row[f"tata_excess_std{tag}_lo"] = round(float(lo), 4)
            row[f"tata_excess_std{tag}_hi"] = round(float(hi), 4)
            row[f"location_verdict{tag}"] = verdict(lo, hi)
            row[f"share_of_excess_explained{tag}"] = (round(float(1 - e / (h0.mean() - h1.mean())), 3)
                                                     if (h0.mean() - h1.mean()) >= 0.10 else np.nan)   # only where e >= 0.10
            for st in strata:
                m1, m0 = a1 == st, a0 == st
                if m1.sum() == 0 or m0.sum() == 0:
                    continue
                r1s, r0s = float(h1[m1].mean()), float(h0[m0].mean())
                au = float(C.auc_counts(c1[m1], c0[m0]))
                srows.append(dict(task=task, mark=m, strata_set=("2" if not tag else "4"), stratum=st,
                                  n_label1=int(m1.sum()), n_label0=int(m0.sum()),
                                  share_label1=round(float(m1.mean()), 4), share_label0=round(float(m0.mean()), 4),
                                  tbp_rate_label1=round(r1s, 4), tbp_rate_label0=round(r0s, 4),
                                  tata_neg_excess=round(r0s - r1s, 4), tata_auroc=round(au, 4),
                                  route3_clause_within=bool((r0s - r1s) >= 0.10 and au < 0.5)))
        rows.append(row)
        print(f"[{task}] match={row['counts_match_pham']} located={row['frac_located']:.3f} "
              f"interg 1/0 {f1:.3f}/{f0:.3f} e={row['tata_neg_excess']:+.3f} e_loc={row['tata_neg_excess_located']:+.3f} "
              f"e_std={row['tata_excess_std']:+.3f} [{row['tata_excess_std_lo']:+.3f},{row['tata_excess_std_hi']:+.3f}] "
              f"{row['location_verdict']} | 4strata {row['tata_excess_std_4strata']:+.3f} {row['location_verdict_4strata']}",
              flush=True)
    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(srows)
    df["literature_group_verifier"] = df.mark.map(lambda x: "promoter/5'-enriched" if x in LIT_PROMOTER else "promoter-depleted/gene-body")
    df.to_csv(os.path.join(RES, "gue_emp_provenance.csv"), index=False)
    sdf.to_csv(os.path.join(RES, "gue_emp_provenance_strata.csv"), index=False)

    # ---------------- S0/S1/S3 tests
    t = []
    t.append(dict(test="S0a counts match Pham #positive/#negative", value=f"{int(df.counts_match_pham.sum())}/10",
                  detail="; ".join(f"{m} {a}/{b}" for m, a, b in df[["mark", "gue_n_label1", "gue_n_label0"]].values)))
    arch = df[df.pham_file_archived]
    t.append(dict(test="S0b archived Pham files: GUE sequences present with the same label",
                  value="; ".join(f"{m} {v:.4f}" for m, v in arch[["mark", "frac_gue_in_pham_same_label"]].values) or "none archived",
                  detail=f"{len(arch)}/10 files archived"))
    if val_rows:
        t.append(dict(test="S1 validation: Pham probe name ('i' = intergenic) vs midpoint class",
                      value="; ".join(f"{m} {a:.4f} (n={n})" for m, n, a in val_rows), detail=""))
    t.append(dict(test="S1 located uniquely (union of EMP windows)",
                  value=f"{int((ldf.n_hits == 1).sum())}/{len(ldf)}",
                  detail=f"0 hits {int((ldf.n_hits == 0).sum())}; >1 hits {int((ldf.n_hits > 1).sum())}"))
    pr = df[df.direction_prediction_holds.notna()]
    nh_ = int(pr.direction_prediction_holds.astype(bool).sum())
    t.append(dict(test="S1 label-direction predictions (label 1 = high) hold", value=f"{nh_}/{len(pr)}",
                  detail="; ".join(f"{m} predicted {'+' if s_ > 0 else '-'}, observed label1-label0 {a - b:+.3f}"
                                   for m, s_, a, b in pr[["mark", "predicted_sign_label1_minus_label0",
                                                          "intergenic_frac_label1", "intergenic_frac_label0"]].values),
                  reading=("label 1 = high supported" if nh_ >= 5 else "label direction NOT supported by location")))
    x = df.intergenic_diff_0_minus_1.to_numpy(); yv = df.tata_neg_excess.to_numpy()
    rho = stats.spearmanr(x, yv).correlation
    rx, ry = stats.rankdata(x), stats.rankdata(yv)
    prng = np.random.RandomState(SEED)
    idx = np.argsort(prng.rand(100000, len(ry)), axis=1)
    pry = ry[idx]
    rxc = rx - rx.mean(); pyc = pry - pry.mean(axis=1, keepdims=True)
    perm = (pyc @ rxc) / (np.sqrt((rxc ** 2).sum()) * np.sqrt((pyc ** 2).sum(axis=1)))
    p_perm = float((np.abs(perm) >= abs(rho) - 1e-12).mean())
    t.append(dict(test="S3 Spearman(intergenic diff 0-1, TATA neg-excess), 10 marks", value=f"rho={rho:.3f}",
                  detail=f"two-sided permutation p={p_perm:.5f} (100,000 perms, seed 42); marks share 34-92% of windows"))
    grp = df.intergenic_diff_0_minus_1 > 0
    tab = [[int((grp & df.route3_signature).sum()), int((grp & ~df.route3_signature).sum())],
           [int((~grp & df.route3_signature).sum()), int((~grp & ~df.route3_signature).sum())]]
    t.append(dict(test="S3 Route-3 signature vs sign of intergenic diff (label 0 more intergenic)",
                  value=f"{tab[0][0]}/{sum(tab[0])} vs {tab[1][0]}/{sum(tab[1])}",
                  detail=f"Fisher exact two-sided p={stats.fisher_exact(tab)[1]:.4f}; table={tab}"))
    gb = df[df.mark.isin(LIT_GENEBODY)]; gp = df[df.mark.isin(LIT_PROMOTER)]
    mw = stats.mannwhitneyu(gb.tata_neg_excess, gp.tata_neg_excess, alternative="greater", method="exact")
    tab2 = [[int(gb.route3_signature.sum()), len(gb) - int(gb.route3_signature.sum())],
            [int(gp.route3_signature.sum()), len(gp) - int(gp.route3_signature.sum())]]
    t.append(dict(test="S3 verifier's literature grouping: gene-body/promoter-depleted vs promoter/5'",
                  value=f"signature {tab2[0][0]}/5 vs {tab2[1][0]}/5",
                  detail=f"exact one-sided Mann-Whitney p={mw.pvalue:.4f} (U={mw.statistic:.0f}); Fisher two-sided "
                         f"p={stats.fisher_exact(tab2)[1]:.4f}; agrees with the measured intergenic-diff sign on "
                         f"{int(((df.mark.isin(LIT_GENEBODY)) == grp).sum())}/10 marks"))
    sig = df[df.route3_signature]
    t.append(dict(test="S2 location verdict on the Route-3-signature marks (primary, 2 strata)",
                  value="; ".join(f"{m} {v}" for m, v in sig[["mark", "location_verdict"]].values),
                  detail="; ".join(f"{m} e={e:+.3f} -> e_std={s:+.3f} [{lo:+.3f},{hi:+.3f}]" for m, e, s, lo, hi in
                                   sig[["mark", "tata_neg_excess_located", "tata_excess_std", "tata_excess_std_lo",
                                        "tata_excess_std_hi"]].values)))
    t.append(dict(test="S2 location verdict (4 strata, sensitivity)",
                  value="; ".join(f"{m} {v}" for m, v in sig[["mark", "location_verdict_4strata"]].values),
                  detail="; ".join(f"{m} e_std={s:+.3f} [{lo:+.3f},{hi:+.3f}]" for m, s, lo, hi in
                                   sig[["mark", "tata_excess_std_4strata", "tata_excess_std_4strata_lo",
                                        "tata_excess_std_4strata_hi"]].values)))
    w2 = sdf[(sdf.strata_set == "2") & sdf.route3_clause_within]
    t.append(dict(test="S2 Route-3 clause (excess>=0.10 & TATA AUROC<0.5) within a stratum (2 strata)",
                  value=f"{len(w2)} mark x stratum cells",
                  detail="; ".join(f"{m}/{s} {e:+.3f}/{a:.3f}" for m, s, e, a in w2[["mark", "stratum", "tata_neg_excess", "tata_auroc"]].values) or "none"))
    t.append(dict(test="manifest (SHA-256)", value=json.dumps(manifest), detail=""))
    tdf = pd.DataFrame(t)
    tdf.to_csv(os.path.join(RES, "gue_emp_provenance_tests.csv"), index=False)
    print(df.drop(columns=[c for c in df.columns if c.startswith("pham_n")]).to_string(index=False))
    print(tdf.to_string(index=False, max_colwidth=160))


if __name__ == "__main__":
    main()
