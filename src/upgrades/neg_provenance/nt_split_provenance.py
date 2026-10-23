#!/usr/bin/env python3
"""POST HOC (fix round 1, 2026-09-26): split-stratified provenance of the genomic nt_enhancers sequences.

Why: independent verification of the first report found that the pooled "95.5% of genomic nt negatives are
ChromHMM state 13 (Heterochrom/lo) in all nine cell lines" hides a train/test difference. The training
negatives are the iEnhancer-2L benchmark (Liu et al. 2016); the test negatives are the iEnhancer-EL
independent set (Liu et al. 2018), which the source says was built with "the same protocol". This script is
descriptive and post hoc; no pre-registered decision rule is changed. The pre-registered D3 rule is applied
per split as a stratified re-reading, labelled as such.

What it does (seed 42, B = 1000 percentile bootstrap, sequence/locus = unit, groups resampled independently)
  (1) Split-stratified metrics. Loci are de-duplicated WITHIN each split (a locus present in both splits
      counts in both; pooled = de-duplicated across splits = the main analysis). Annotations are joined from
      results/upgrades/neg_provenance_intervals.csv.gz (no re-annotation). -> neg_provenance_nt_by_split.csv
      Contrasts (test - train; rule background - split) with bootstrap CIs and Fisher p for binary metrics;
      D3 per split. -> neg_provenance_nt_by_split_tests.csv
  (2) Liftover-free hg19 check. The iEnhancer-2L training FASTA headers ('CHRX_10705600_10705800') and the
      independent-set enhancer headers ('Chr11_6627824_6628024') carry hg19 0-based half-open coordinates;
      each whole 200-bp window is looked up directly in the nine hg19 ChromHMM segmentations (no minimap2,
      no liftOver). -> neg_provenance_nt_hg19_check.csv
  (3) Parent regions of the independent-test negatives. Their headers are UCSC custom-track FASTA headers
      with punctuation stripped ('hg19_ct_UserTrack_3545_158rangechr1278319870-783938695pad0' =
      'range=chr12:78319870-78393869 5'pad=0'). The chromosome is disambiguated with the fragment's own hg19
      position (hg38 minimap2 locus lifted to hg19). Per parent region: does it contain the fragment; does it
      coincide (both ends within 1 bp) with a single ChromHMM segment of any of the nine lines, and in which
      state; bp fraction of the region in state 13 per line. Null: the same regions shifted by a random
      offset of 5-100 kb (either direction; 20 draws per region, seed 42).
      -> neg_provenance_nt_test_parent.csv (per sequence) + summary rows in neg_provenance_nt_hg19_check.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import genome_utils as G  # noqa: E402
import run_neg_provenance as RP  # noqa: E402
import verify_inputs as V  # noqa: E402

SEED, B = RP.SEED, RP.B
PFX = RP.PFX
METRICS = ["gc_local", "tata_rate", "gc_100kb", "iso_AT_L1L2", "ge15kb_gene_pc", "comp_intergenic", "rep_bp_L1",
           "ccre_any", "hmm_13_all9", "hmm_13_ge7", "hmm_13_majority", "hmm_nonenh_all9", "hmm_enh_any",
           "hmm_strongenh_all9", "hmm_13_nlines"]
BINARY = {"iso_AT_L1L2", "ge15kb_gene_pc", "comp_intergenic", "ccre_any", "hmm_13_all9", "hmm_13_ge7",
          "hmm_13_majority", "hmm_nonenh_all9", "hmm_enh_any", "hmm_strongenh_all9", "tata_rate"}


def ci(x):
    return RP.ci(x)


def boot_mean(v, seed_offset):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    r = np.random.RandomState(SEED + seed_offset)
    idx = r.randint(0, len(v), size=(B, len(v)))
    return float(v.mean()), v[idx].mean(1), len(v)


def metric_frame(g):
    M = RP.derived(g.reset_index(drop=True))
    hc = g[[f"hmm_{c}" for c in G.CHROMHMM_CELLS]].to_numpy()
    lift = g["hmm_lifted"].to_numpy(bool)
    n13 = (hc == 13).sum(1).astype(float)
    M["hmm_13_ge7"] = np.where(lift, (n13 >= 7).astype(float), np.nan)
    M["hmm_13_nlines"] = np.where(lift, n13, np.nan)
    return M


def parse_hdr_coord(h):
    """'CHRX_10705600_10705800' / 'Chr11_6627824_6628024' -> ('chrX', 10705600, 10705800) (hg19, 0-based)."""
    m = re.fullmatch(r"[Cc][Hh][Rr]([0-9]+|[XxYy])_(\d+)_(\d+)", h.strip())
    if not m:
        return None
    c = m.group(1).upper() if m.group(1).upper() in ("X", "Y") else m.group(1)
    return "chr" + c, int(m.group(2)), int(m.group(3))


CHR_NAMES = [str(i) for i in range(1, 23)] + ["X", "Y"]


def parse_parent_range(h, chrom_hint):
    """UCSC Table-Browser header -> (chrom, start0, end) (1-based closed -> 0-based half-open), or None.
    Unstripped form 'range=chr10:99167611-99168610' is unambiguous. In the stripped form
    ('rangechr1278319870-783938695pad0') chromosome and start digits are fused: every chromosome name that
    prefixes the digits and gives a valid range (end > start, < 10 Mb) is a candidate; the fragment's own hg19
    chromosome wins if it is a candidate, otherwise a unique candidate is used, otherwise None."""
    if not h:
        return None
    m = re.search(r"range=chr([0-9XY]+):(\d+)-(\d+)", h)
    if m:
        return "chr" + m.group(1), int(m.group(2)) - 1, int(m.group(3))
    if "rangechr" not in h:
        return None
    body = h[h.index("rangechr") + len("rangechr"):]
    cands = {}
    for cd in CHR_NAMES:
        if body.startswith(cd):
            mm = re.match(r"(\d+)-(\d+)5pad0", body[len(cd):])
            if mm:
                a, b = int(mm.group(1)), int(mm.group(2))
                if a >= 1 and a < b and b - a < 10_000_000:
                    cands["chr" + cd] = (a - 1, b)
    if chrom_hint in cands:
        return (chrom_hint,) + cands[chrom_hint]
    if len(cands) == 1:
        c = next(iter(cands))
        return (c,) + cands[c]
    return None


def window_states(H, c, s, e):
    """Per cell line: sorted tuple of ChromHMM states overlapping [s, e) (hg19); () if not covered."""
    out = []
    for cell in G.CHROMHMM_CELLS:
        if c not in H[cell]:
            out.append(()); continue
        ss, ee, st = H[cell][c]
        i0 = max(np.searchsorted(ss, s, side="right") - 1, 0)
        i1 = np.searchsorted(ss, e - 1, side="right") - 1
        seg = [k for k in range(i0, i1 + 1) if ee[k] > s and ss[k] < e]
        out.append(tuple(sorted({int(st[k]) for k in seg})))
    return out


def exact_segment_matches(H, c, s, e, tol=1):
    out = []
    for cell in G.CHROMHMM_CELLS:
        if c not in H[cell]:
            continue
        ss, ee, st = H[cell][c]
        i = np.searchsorted(ss, s - tol)
        for k in range(max(i - 1, 0), min(i + 3, len(ss))):
            if abs(ss[k] - s) <= tol and abs(ee[k] - e) <= tol:
                out.append((cell, int(st[k]), int(ss[k] - s), int(ee[k] - e)))
    return out


def state_bp_fraction(H, cell, c, s, e, state):
    ss, ee, st = H[cell][c]
    i0 = max(np.searchsorted(ss, s, side="right") - 1, 0)
    i1 = np.searchsorted(ss, e - 1, side="right") - 1
    bp = 0
    for k in range(i0, i1 + 1):
        if st[k] == state:
            bp += max(0, min(ee[k], e) - max(ss[k], s))
    return bp / (e - s)


def segment_catalogue(H):
    """state -> dict(chr, s, e, len) of every ChromHMM segment of every line (for the sharing baseline)."""
    cat = {}
    for cell in G.CHROMHMM_CELLS:
        for c, (ss, ee, st) in H[cell].items():
            for k in np.unique(st):
                sel = st == k
                d = cat.setdefault(int(k), {"chr": [], "s": [], "e": []})
                d["chr"].append(np.full(int(sel.sum()), c, dtype=object)); d["s"].append(ss[sel]); d["e"].append(ee[sel])
    out = {}
    for k, d in cat.items():
        s_, e_ = np.concatenate(d["s"]), np.concatenate(d["e"])
        out[k] = {"chr": np.concatenate(d["chr"]), "s": s_, "e": e_, "len": e_ - s_}
    return out


def main():
    from pyliftover import LiftOver
    m = RP.load_nt_mapped()
    W = pd.read_csv(PFX + "_intervals.csv.gz", low_memory=False)
    W = W[W.dataset == "nt_enhancers"].copy()
    W["isochore"] = W["isochore"].fillna("NA")           # read_csv turns the literal "NA" class into NaN
    ann = W[W.group.isin(["POS", "NEG"])].drop_duplicates(["chr", "start", "end", "group"])
    gen = m[m["genomic"].astype(bool)].copy()
    gen["start"] = gen["r_st"].astype(int); gen["end"] = gen["r_en"].astype(int)
    gen["group"] = np.where(gen["label"] == 1, "POS", "NEG")
    key = ["chr", "start", "end", "group"]
    cols = [c for c in ann.columns if c not in ("dataset", "id", "split", "type_label", "ien_class")]

    sets = {}
    for sp in ("train", "test"):
        sets[sp] = gen[gen.split == sp].drop_duplicates(key)[key]
    sets["pooled"] = gen.drop_duplicates(key)[key]
    meta = pd.read_csv(PFX + "_meta.csv", index_col=0).iloc[:, 0]
    for grp in ("POS", "NEG"):
        assert (sets["pooled"].group == grp).sum() == int(meta[f"nt_genomic_unique_loci_{grp}"]), grp
    rows, parts = [], {}
    for j, (sp, loci) in enumerate(sets.items()):
        x = loci.merge(ann[cols], on=key, how="left", validate="one_to_one")
        assert x["gc_local"].notna().all(), f"{sp}: loci missing from intervals.csv.gz"
        for k, grp in enumerate(("NEG", "POS")):
            g = x[x.group == grp].reset_index(drop=True)
            Mg = metric_frame(g)
            parts[(sp, grp)] = Mg
            for q, met in enumerate(METRICS):
                est, reps, n = boot_mean(Mg[met], 100 * j + 10 * k + q)
                lo, hi = ci(reps)
                rows.append(dict(split=sp, group=grp, metric=met, value=est, ci_lo=lo, ci_hi=hi, n=n,
                                 n_loci=len(g)))
    # rule backgrounds (seed-42 draws of the main run) for reference
    for k, bg in enumerate(("BG_rule13", "BG_rule", "BG_uni")):
        g = W[W.group == bg].reset_index(drop=True)
        Mg = metric_frame(g); parts[("bg", bg)] = Mg
        for q, met in enumerate(METRICS):
            est, reps, n = boot_mean(Mg[met], 1000 + 10 * k + q)
            lo, hi = ci(reps)
            rows.append(dict(split="background", group=bg, metric=met, value=est, ci_lo=lo, ci_hi=hi, n=n, n_loci=len(g)))
    S = pd.DataFrame(rows)
    S.to_csv(PFX + "_nt_by_split.csv", index=False)

    # ---------------------------------------------------------------- contrasts
    T = []

    def contrast(name, a, b, met, off):
        va = np.asarray(a[met], float); va = va[np.isfinite(va)]
        vb = np.asarray(b[met], float); vb = vb[np.isfinite(vb)]
        r = np.random.RandomState(SEED + off)
        d = va[r.randint(0, len(va), (B, len(va)))].mean(1) - vb[r.randint(0, len(vb), (B, len(vb)))].mean(1)
        lo, hi = ci(d)
        p = np.nan
        if met in BINARY:
            p = fisher_exact([[int(va.sum()), int(len(va) - va.sum())], [int(vb.sum()), int(len(vb) - vb.sum())]])[1]
        T.append(dict(contrast=name, metric=met, a=float(va.mean()), b=float(vb.mean()), diff_a_minus_b=float(va.mean() - vb.mean()),
                      diff_lo=lo, diff_hi=hi, fisher_p=p, n_a=len(va), n_b=len(vb)))

    cm = ["hmm_13_all9", "hmm_13_ge7", "hmm_nonenh_all9", "hmm_enh_any", "gc_local", "tata_rate", "gc_100kb",
          "iso_AT_L1L2", "ge15kb_gene_pc", "rep_bp_L1", "ccre_any"]
    for q, met in enumerate(cm):
        contrast("NEG test - NEG train", parts[("test", "NEG")], parts[("train", "NEG")], met, 2000 + q)
        contrast("POS test - POS train", parts[("test", "POS")], parts[("train", "POS")], met, 2100 + q)
    for j, sp in enumerate(("train", "test", "pooled")):
        for q, met in enumerate(("gc_local", "tata_rate", "gc_100kb", "iso_AT_L1L2", "hmm_13_all9")):
            contrast(f"BG_rule13 - NEG {sp}", parts[("bg", "BG_rule13")], parts[(sp, "NEG")], met, 3000 + 10 * j + q)
            contrast(f"BG_uni - NEG {sp}", parts[("bg", "BG_uni")], parts[(sp, "NEG")], met, 3500 + 10 * j + q)
    for sp in ("train", "test", "pooled"):
        a = S[(S.split == sp) & (S.group == "NEG") & (S.metric == "hmm_nonenh_all9")].iloc[0]
        b = S[(S.split == sp) & (S.group == "POS") & (S.metric == "hmm_enh_any")].iloc[0]
        c13 = S[(S.split == sp) & (S.group == "NEG") & (S.metric == "hmm_13_all9")].iloc[0]
        T.append(dict(contrast=f"D3 per split (post hoc stratification): {sp}", metric="NEG hmm_nonenh_all9 | POS hmm_enh_any",
                      a=a.value, b=b.value, diff_lo=np.nan, diff_hi=np.nan, n_a=a.n, n_b=b.n,
                      verdict="VERIFIED" if (a.value >= 0.8 and b.value >= 0.8) else "NOT_VERIFIED",
                      note=f"NEG nonenh_all9 [{a.ci_lo:.3f}, {a.ci_hi:.3f}]; POS enh_any [{b.ci_lo:.3f}, {b.ci_hi:.3f}]; "
                           f"NEG state13_all9 {c13.value:.3f} [{c13.ci_lo:.3f}, {c13.ci_hi:.3f}]"))
    # distribution of the number of lines in state 13 (NEG, per split)
    for sp in ("train", "test"):
        v = parts[(sp, "NEG")]["hmm_13_nlines"].dropna().astype(int)
        for k in range(10):
            T.append(dict(contrast=f"NEG {sp}: number of lines in state 13 (midpoint)", metric=f"n13=={k}",
                          a=float((v == k).mean()), n_a=int((v == k).sum()), b=np.nan))

    # ---------------------------------------------------------------- (2) liftover-free hg19 header check
    H = G.chromhmm()
    cls_of = {}
    for (sp, cl), f in V.IEN_FILES.items():
        for _, s in V.read_fasta(os.path.join(V.IEN, f)):
            cls_of.setdefault((sp, s), set()).add(cl)
    hrows = []
    for (sp, kind), f in V.ELM_FILES.items():
        for h, s in V.read_fasta(os.path.join(V.IEN, f)):
            pc = parse_hdr_coord(h)
            cl = "|".join(sorted(cls_of.get((sp, s), {"?"})))
            if pc is None:
                continue
            c, a, b = pc
            stt = window_states(H, c, a, b)
            hrows.append(dict(split=sp, cls=cl, chr=c, start=a, end=b, length=b - a,
                              whole13_all9=all(x == (13,) for x in stt),
                              whole_strong_all9=all(len(x) > 0 and set(x) <= {4, 5} for x in stt),
                              weak_any=any(set(x) & {6, 7} for x in stt),
                              enh_any=any(set(x) & {4, 5, 6, 7} for x in stt),
                              nonenh_all9=all(len(x) > 0 and not (set(x) & {4, 5, 6, 7}) for x in stt),
                              n_lines_13=sum(x == (13,) for x in stt)))
    HD = pd.DataFrame(hrows)
    chk = []

    def addchk(check, sp, cl, v, note=""):
        v = np.asarray(v, float)
        est, reps, n = boot_mean(v, 5000 + len(chk))
        lo, hi = ci(reps)
        chk.append(dict(check=check, split=sp, cls=cl, n=n, k=int(v.sum()), frac=est, ci_lo=lo, ci_hi=hi, note=note))

    for (sp, cl), g in HD.groupby(["split", "cls"]):
        for met in ("whole13_all9", "nonenh_all9", "whole_strong_all9", "weak_any", "enh_any"):
            addchk(f"hg19 header window, {met}", sp, cl, g[met], "iEnhancer FASTA header coordinates; no liftOver")

    # ---------------------------------------------------------------- (3) parent regions of the independent set
    lo = LiftOver(G.CHAIN_38_19)
    hdr = {}
    for kind in ("non", "enh"):
        for h, s in V.read_fasta(os.path.join(V.IEN, V.ELM_FILES[("test", kind)])):
            hdr[s] = h
    ien_test = {}
    for cl in ("strong", "weak", "non"):
        for _, s in V.read_fasta(os.path.join(V.IEN, V.IEN_FILES[("test", cl)])):
            ien_test[s] = cl
    t = m[(m.split == "test")].copy()
    t["ien_test_cls"] = t["seq_ds"].map(ien_test)
    t = t[t["ien_test_cls"].isin(["non", "weak"])]            # test strong carry their own 200-bp hg19 window
    CLS_STATES = {"non": {13}, "weak": {6, 7}}
    catalogue = segment_catalogue(H)
    prow = []
    rng = np.random.RandomState(SEED)
    null = {"non": [0, 0], "weak": [0, 0]}
    share_base = {"non": [], "weak": []}
    for r in t.itertuples():
        cl = r.ien_test_cls
        h = hdr.get(r.seq_ds)
        a1 = lo.convert_coordinate(r.chr, int(r.r_st)); a2 = lo.convert_coordinate(r.chr, int(r.r_en) - 1)
        c19 = s19 = e19 = None
        if a1 and a2 and len(a1) == 1 and len(a2) == 1 and a1[0][0] == a2[0][0]:
            c19 = a1[0][0]; s19 = min(a1[0][1], a2[0][1]); e19 = max(a1[0][1], a2[0][1]) + 1
        pr = parse_parent_range(h, c19)
        row = dict(cls=cl, idx=int(r.idx), hg38=f"{r.chr}:{int(r.r_st)}-{int(r.r_en)}", header=h, hg19_chr=c19,
                   hg19_start=s19, hg19_end=e19, parent_parsed=pr is not None)
        if s19 is not None:
            stt = window_states(H, c19, s19, e19)
            row.update(frag_whole13_all9=all(x == (13,) for x in stt), frag_n_lines_13=sum(x == (13,) for x in stt),
                       frag_enh_any=any(set(x) & {4, 5, 6, 7} for x in stt),
                       frag_weak_any=any(set(x) & {6, 7} for x in stt),
                       frag_strong_all9=all(len(x) > 0 and set(x) <= {4, 5} for x in stt),
                       frag_states="; ".join(f"{c}:{','.join(map(str, x))}" for c, x in zip(G.CHROMHMM_CELLS, stt)))
        if pr is not None:
            c, ps, pe = pr
            mt = exact_segment_matches(H, c, ps, pe)
            frc = [sum(state_bp_fraction(H, cell, c, ps, pe, k) for k in CLS_STATES[cl]) if c in H[cell] else np.nan
                   for cell in G.CHROMHMM_CELLS]
            sts = sorted({x[1] for x in mt})
            row.update(parent_chr=c, parent_start=ps, parent_end=pe, parent_len=pe - ps,
                       parent_same_chr_as_fragment=c == c19,
                       frag_inside_parent=bool(s19 is not None and c == c19 and ps <= s19 and e19 <= pe),
                       n_lines_exact_segment=len({x[0] for x in mt}),
                       exact_segment_states="|".join(map(str, sts)),
                       exact_segment_class_state=bool(len(mt) > 0 and set(sts) <= CLS_STATES[cl]),
                       exact_segment_lines="|".join(f"{x[0]}:{x[1]}" for x in mt),
                       exact_offsets="|".join(f"{x[2]},{x[3]}" for x in mt),
                       parent_n_lines_ge99pct_class_state=int(np.nansum(np.array(frc) >= 0.99)),
                       parent_mean_frac_class_state=float(np.nanmean(frc)))
            for _ in range(20):          # null 1: shifted region, same length
                off = int(rng.randint(5000, 100001)) * (1 if rng.rand() < 0.5 else -1)
                if ps + off < 0:
                    off = -off
                null[cl][1] += 1
                null[cl][0] += len(exact_segment_matches(H, c, ps + off, pe + off)) > 0
            if len(mt):                  # baseline 2: how many lines share a random same-state, length-matched segment
                stt0 = max(set(x[1] for x in mt), key=[x[1] for x in mt].count)
                cat = catalogue[stt0]
                ok = np.flatnonzero(np.abs(cat["len"] - (pe - ps)) <= 0.2 * (pe - ps))
                for k in rng.choice(ok, size=min(10, len(ok)), replace=False) if len(ok) else []:
                    cc_, s_, e_ = cat["chr"][k], int(cat["s"][k]), int(cat["e"][k])
                    share_base[cl].append(len({x[0] for x in exact_segment_matches(H, cc_, s_, e_)}))
        prow.append(row)
    PR = pd.DataFrame(prow)
    PR.to_csv(PFX + "_nt_test_parent.csv", index=False)
    for cl, lab, st_lab in (("non", "test NEG", "state 13 (Heterochrom/lo)"), ("weak", "test weak POS", "state 6/7 (weak enhancer)")):
        P1 = PR[PR.cls == cl]; pp = P1[P1.parent_parsed]
        addchk(f"{lab}: header carries a parseable parent range", "test", cl, P1["parent_parsed"])
        addchk(f"{lab}: parent range on the fragment's own hg19 chromosome and containing it", "test", cl,
               pp["frag_inside_parent"])
        addchk(f"{lab}: parent range = one ChromHMM segment (both ends +-1 bp) in >=1 line", "test", cl,
               pp["n_lines_exact_segment"] >= 1)
        addchk(f"{lab}: ... and that segment is {st_lab} in every matching line", "test", cl,
               pp["exact_segment_class_state"])
        addchk(f"{lab}: parent range >=99% {st_lab} in all nine lines", "test", cl,
               pp["parent_n_lines_ge99pct_class_state"] == 9)
        chk.append(dict(check=f"NULL {lab}: shifted parent range (5-100 kb, 20 draws/region) = one segment in >=1 line",
                        split="test", cls=cl, n=null[cl][1], k=int(null[cl][0]), frac=null[cl][0] / max(null[cl][1], 1),
                        ci_lo=np.nan, ci_hi=np.nan, note="seed 42"))
        em = pp[pp.n_lines_exact_segment >= 1]
        for k, v in em["n_lines_exact_segment"].value_counts().sort_index().items():
            chk.append(dict(check=f"{lab} with exact segment match: number of matching lines", split="test", cls=cl,
                            n=len(em), k=int(v), frac=v / len(em), ci_lo=np.nan, ci_hi=np.nan, note=f"lines={int(k)}"))
        sb = np.array(share_base[cl])
        for k in sorted(set(sb.tolist())):
            chk.append(dict(check=f"BASELINE {lab}: random same-state, length-matched (+-20%) segment: number of lines "
                                  "with an identical segment", split="test", cls=cl, n=len(sb), k=int((sb == k).sum()),
                            frac=float((sb == k).mean()), ci_lo=np.nan, ci_hi=np.nan, note=f"lines={k}; seed 42"))
        per_cell = {c: int(em["exact_segment_lines"].str.contains(c + ":").sum()) for c in G.CHROMHMM_CELLS}
        for c, v in per_cell.items():
            chk.append(dict(check=f"{lab} with exact segment match: parent matches a segment of this line", split="test",
                            cls=cl, n=len(em), k=v, frac=v / max(len(em), 1), ci_lo=np.nan, ci_hi=np.nan, note=c))
        chk.append(dict(check=f"{lab} parent range length (bp): median [IQR]", split="test", cls=cl, n=len(pp),
                        k=np.nan, frac=float(pp.parent_len.median()), ci_lo=float(pp.parent_len.quantile(0.25)),
                        ci_hi=float(pp.parent_len.quantile(0.75)), note="quartiles, not a CI"))
    P0 = PR[PR.cls == "non"]
    addchk("test NEG: fragment whole window state 13 in all nine lines (liftOver of fragment)", "test", "non",
           P0["frag_whole13_all9"].astype(float))
    addchk("test NEG: fragment enhancer state (4-7) in >=1 line (whole window)", "test", "non",
           P0["frag_enh_any"].astype(float))
    P1 = PR[PR.cls == "weak"]
    addchk("test weak POS: fragment weak-enhancer state (6/7) in >=1 line (whole window)", "test", "weak",
           P1["frag_weak_any"].astype(float))
    addchk("test weak POS: fragment strong state (4/5) in all nine lines (whole window)", "test", "weak",
           P1["frag_strong_all9"].astype(float))
    pd.DataFrame(chk).to_csv(PFX + "_nt_hg19_check.csv", index=False)
    pd.DataFrame(T).to_csv(PFX + "_nt_by_split_tests.csv", index=False)
    print(S[S.metric.isin(["hmm_13_all9", "hmm_nonenh_all9", "hmm_enh_any", "gc_local", "tata_rate", "iso_AT_L1L2"])]
          .to_string(), flush=True)
    print(pd.DataFrame(chk).to_string(), flush=True)


if __name__ == "__main__":
    main()
