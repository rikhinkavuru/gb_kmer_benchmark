"""Low-memory hg38 annotation helpers for the negative-set provenance analysis.

Everything is streamed one chromosome at a time and cached as compact .npz files under
external/neg_provenance/cache/, so the peak RAM stays < ~1 GB on a shared machine.

Coordinates everywhere are 0-based, half-open (BED convention), UCSC chromosome names (chr1...).
"""
import gzip
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXT = os.path.join(ROOT, "external", "neg_provenance")
CACHE = os.path.join(EXT, "cache")
HG38 = os.environ.get("HG38_FA", os.path.expanduser("~/.cache/glrb_hg38/hg38.fa"))
GENCODE_GTF = os.path.join(EXT, "gencode.v44.annotation.gtf.gz")
RMSK = os.path.join(EXT, "rmsk.txt.gz")
CCRE = os.environ.get("CCRE_BED", os.path.join(EXT, "GRCh38-cCREs.bed"))
CCRE_FALLBACK = os.path.expanduser("~/mlcbfinal/data/clinvar/GRCh38-cCREs.bed")
CHROMHMM_DIR = os.path.join(EXT, "chromhmm_hg19")
CHAIN_38_19 = os.path.join(EXT, "hg38ToHg19.over.chain.gz")
CHAIN_19_38 = os.path.join(EXT, "hg19ToHg38.over.chain.gz")
PRIMARY = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
BIN = 1000
CHROMHMM_CELLS = ["Gm12878", "H1hesc", "Hepg2", "Hmec", "Hsmm", "Huvec", "K562", "Nhek", "Nhlf"]

# Isochore families by GC (Costantini et al. 2006, Genome Res 16:536): L1<37, L2 37-41, H1 41-46,
# H2 46-53, H3>53 (%). Bin edges are right-open on the lower family.
ISO_EDGES = [0.37, 0.41, 0.46, 0.53]
ISO_NAMES = ["L1", "L2", "H1", "H2", "H3"]

# RepeatMasker codes (per-base arrays). Family-level for the AT-rich-relevant classes.
RMSK_CODES = {"none": 0, "LINE/L1": 1, "LINE/L2": 2, "LINE/other": 3, "SINE/Alu": 4,
              "SINE/MIR": 5, "SINE/other": 6, "LTR": 7, "DNA": 8, "Simple_repeat": 9,
              "Low_complexity": 10, "Satellite": 11, "other": 12}
RMSK_NAMES = {v: k for k, v in RMSK_CODES.items()}

CCRE_CODES = {"none": 0, "PLS": 1, "pELS": 2, "dELS": 3, "CA-H3K4me3": 4, "CA-CTCF": 5,
              "CA-TF": 6, "CA": 7, "TF": 8}
COMPARTMENTS = {0: "intergenic", 1: "intron", 2: "exon", 3: "promoter"}


def _cache(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)


def iso_class(gc):
    """Isochore family index (0..4) for GC fraction(s); NaN -> -1."""
    gc = np.asarray(gc, dtype=float)
    out = np.digitize(gc, ISO_EDGES)
    return np.where(np.isnan(gc), -1, out)


def fasta(upper=True):
    import pyfaidx
    return pyfaidx.Fasta(HG38, as_raw=True, sequence_always_upper=upper)


def chrom_sizes():
    fai = pd.read_csv(HG38 + ".fai", sep="\t", header=None, usecols=[0, 1], names=["chr", "len"])
    return {c: int(n) for c, n in zip(fai["chr"], fai["len"]) if c in PRIMARY}


# ----------------------------------------------------------------- 1-kb GC / N / softmask bins
def bins():
    """dict chrom -> (acgt, gc, lower) uint16 arrays of per-1kb-bin counts (cached)."""
    path = _cache("hg38_bins1kb.npz")
    if not os.path.exists(path):
        fa = fasta(upper=False)
        arrs = {}
        step = 10_000_000
        for c, n in chrom_sizes().items():
            nb = (n + BIN - 1) // BIN
            acgt = np.zeros(nb, np.uint16); gc = np.zeros(nb, np.uint16); low = np.zeros(nb, np.uint16)
            for s in range(0, n, step):
                e = min(n, s + step)
                b = np.frombuffer(fa[c][s:e].encode("ascii"), dtype=np.uint8)
                up = b & 0xDF
                is_acgt = (up == 65) | (up == 67) | (up == 71) | (up == 84)
                is_gc = (up == 67) | (up == 71)
                is_low = (b >= 97) & is_acgt
                pad = (-len(b)) % BIN
                for arr, v in ((acgt, is_acgt), (gc, is_gc), (low, is_low)):
                    vv = np.concatenate([v, np.zeros(pad, bool)]) if pad else v
                    arr[s // BIN: s // BIN + len(vv) // BIN] = vv.reshape(-1, BIN).sum(1)
            arrs[f"{c}_acgt"], arrs[f"{c}_gc"], arrs[f"{c}_low"] = acgt, gc, low
            print(f"    bins {c} done", flush=True)
        np.savez_compressed(path, **arrs)
    z = np.load(path)
    return {c: (z[f"{c}_acgt"], z[f"{c}_gc"], z[f"{c}_low"]) for c in PRIMARY if f"{c}_acgt" in z}


def regional_gc(chroms, mids, B, half_bins=50, min_acgt_frac=0.5):
    """GC fraction (over A/C/G/T only) of the (2*half_bins)-kb window centred on each midpoint.
    Windows with < min_acgt_frac sequenced (non-N) bases -> NaN."""
    out = np.full(len(mids), np.nan)
    chroms = np.asarray(chroms); mids = np.asarray(mids, dtype=np.int64)
    for c in np.unique(chroms):
        if c not in B:
            continue
        acgt, gc, _ = B[c]
        ca = np.concatenate([[0], np.cumsum(acgt, dtype=np.int64)])
        cg = np.concatenate([[0], np.cumsum(gc, dtype=np.int64)])
        m = chroms == c
        b = mids[m] // BIN
        lo = np.clip(b - half_bins, 0, len(acgt)); hi = np.clip(b + half_bins, 0, len(acgt))
        na = ca[hi] - ca[lo]; ng = cg[hi] - cg[lo]
        v = np.where(na >= min_acgt_frac * (hi - lo) * BIN, ng / np.maximum(na, 1), np.nan)
        out[m] = v
    return out


# ----------------------------------------------------------------------------- GENCODE v44
def gencode():
    """Cached GENCODE v44 tables: genes (chr,start,end,strand,gene_type), tss (chr,pos,strand),
    exons (chr,start,end). 0-based half-open."""
    path = _cache("gencode_v44.npz")
    if not os.path.exists(path):
        genes, tss, exons = [], [], []
        cols = [0, 2, 3, 4, 6, 8]
        for ch in pd.read_csv(GENCODE_GTF, sep="\t", comment="#", header=None, usecols=cols,
                              names=["chr", "feat", "s", "e", "strand", "attr"], chunksize=400_000):
            ch = ch[ch["chr"].isin(PRIMARY)]
            g = ch[ch["feat"] == "gene"]
            genes.append(pd.DataFrame({"chr": g["chr"], "start": g["s"] - 1, "end": g["e"],
                                       "strand": g["strand"],
                                       "gene_type": g["attr"].str.extract(r'gene_type "([^"]+)"')[0]}))
            t = ch[ch["feat"] == "transcript"]
            tss.append(pd.DataFrame({"chr": t["chr"],
                                     "pos": np.where(t["strand"] == "+", t["s"] - 1, t["e"] - 1),
                                     "strand": t["strand"]}))
            x = ch[ch["feat"] == "exon"]
            exons.append(pd.DataFrame({"chr": x["chr"], "start": x["s"] - 1, "end": x["e"]}))
        G = pd.concat(genes); T = pd.concat(tss).drop_duplicates(); X = pd.concat(exons).drop_duplicates()
        np.savez_compressed(path,
                            g_chr=G["chr"].to_numpy(str), g_s=G["start"].to_numpy(np.int64),
                            g_e=G["end"].to_numpy(np.int64), g_type=G["gene_type"].to_numpy(str),
                            t_chr=T["chr"].to_numpy(str), t_pos=T["pos"].to_numpy(np.int64),
                            x_chr=X["chr"].to_numpy(str), x_s=X["start"].to_numpy(np.int64),
                            x_e=X["end"].to_numpy(np.int64))
    z = np.load(path, allow_pickle=False)
    genes = pd.DataFrame({"chr": z["g_chr"], "start": z["g_s"], "end": z["g_e"], "gene_type": z["g_type"]})
    tss = pd.DataFrame({"chr": z["t_chr"], "pos": z["t_pos"]})
    exons = pd.DataFrame({"chr": z["x_chr"], "start": z["x_s"], "end": z["x_e"]})
    return genes, tss, exons


def merge_intervals(s, e):
    """Merge sorted-or-not half-open intervals; returns (starts, ends) numpy arrays."""
    if len(s) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    o = np.argsort(s, kind="stable"); s = np.asarray(s)[o]; e = np.asarray(e)[o]
    emax = np.maximum.accumulate(e)
    new = np.ones(len(s), bool); new[1:] = s[1:] > emax[:-1]
    idx = np.flatnonzero(new)
    starts = s[idx]
    ends = np.append(emax[idx[1:] - 1], emax[-1])
    return starts, ends


def dist_to_intervals(q_s, q_e, m_s, m_e):
    """Gap (bp) between each query interval [q_s,q_e) and the nearest merged interval
    (0 if overlapping). m_* must be merged/sorted. Returns int64 array (large if none)."""
    q_s = np.asarray(q_s, np.int64); q_e = np.asarray(q_e, np.int64)
    if len(m_s) == 0:
        return np.full(len(q_s), 10**12, np.int64)
    BIG = np.int64(10**12)
    # i = last merged interval with m_s < q_e (the only candidate for overlap / left neighbour,
    # because merged intervals are disjoint and sorted, so m_e is increasing)
    i = np.searchsorted(m_s, q_e, side="left") - 1
    has_l = i >= 0
    le = m_e[np.clip(i, 0, None)]
    dl = np.where(le > q_s, 0, q_s - le)          # 0 = overlap; else #bases between them
    dl = np.where(has_l, dl, BIG)
    j = i + 1
    has_r = j < len(m_s)
    dr = np.where(has_r, m_s[np.clip(j, 0, len(m_s) - 1)] - q_e, BIG)
    return np.minimum(dl, dr)


def nearest_point_dist(q, pts_sorted):
    """|q - nearest point| for sorted points."""
    q = np.asarray(q, np.int64)
    if len(pts_sorted) == 0:
        return np.full(len(q), 10**12, np.int64)
    i = np.searchsorted(pts_sorted, q)
    a = np.abs(q - pts_sorted[np.clip(i - 1, 0, len(pts_sorted) - 1)])
    b = np.abs(pts_sorted[np.clip(i, 0, len(pts_sorted) - 1)] - q)
    return np.minimum(a, b)


class GeneIndex:
    """Per-chromosome merged gene spans (all biotypes / protein-coding), exons, TSS."""

    def __init__(self):
        genes, tss, exons = gencode()
        self.g_all, self.g_pc, self.tss, self.exon = {}, {}, {}, {}
        for c in PRIMARY:
            g = genes[genes["chr"] == c]
            self.g_all[c] = merge_intervals(g["start"].to_numpy(), g["end"].to_numpy())
            gp = g[g["gene_type"] == "protein_coding"]
            self.g_pc[c] = merge_intervals(gp["start"].to_numpy(), gp["end"].to_numpy())
            self.tss[c] = np.sort(tss.loc[tss["chr"] == c, "pos"].to_numpy())
            x = exons[exons["chr"] == c]
            self.exon[c] = merge_intervals(x["start"].to_numpy(), x["end"].to_numpy())

    def annotate(self, chroms, starts, ends):
        """DataFrame: compartment (by interval midpoint), dist_tss, dist_gene_all, dist_gene_pc."""
        chroms = np.asarray(chroms); s = np.asarray(starts, np.int64); e = np.asarray(ends, np.int64)
        mid = (s + e) // 2
        n = len(s)
        comp = np.zeros(n, np.int8); dt = np.full(n, 10**12, np.int64)
        dga = np.full(n, 10**12, np.int64); dgp = np.full(n, 10**12, np.int64)
        for c in np.unique(chroms):
            if c not in self.g_all:
                continue
            m = chroms == c
            dt[m] = nearest_point_dist(mid[m], self.tss[c])
            dga[m] = dist_to_intervals(s[m], e[m], *self.g_all[c])
            dgp[m] = dist_to_intervals(s[m], e[m], *self.g_pc[c])
            in_gene = dist_to_intervals(mid[m], mid[m] + 1, *self.g_all[c]) == 0
            in_exon = dist_to_intervals(mid[m], mid[m] + 1, *self.exon[c]) == 0
            cc = np.where(in_gene, 1, 0)
            cc = np.where(in_exon, 2, cc)
            cc = np.where(dt[m] <= 1000, 3, cc)
            comp[m] = cc
        return pd.DataFrame({"compartment": [COMPARTMENTS[int(v)] for v in comp],
                             "dist_tss": dt, "dist_gene_all": dga, "dist_gene_pc": dgp})

    def eligible_starts(self, c, L, flank, chrom_len, extra_block=None, protein_coding=False):
        """Merged intervals of admissible START positions for a length-L window that stays >= flank
        bp away from every gene span (and optionally outside extra_block intervals)."""
        gs, ge = (self.g_pc if protein_coding else self.g_all)[c]
        bs = [gs - flank, ]; be = [ge + flank, ]
        if extra_block is not None and len(extra_block[0]):
            bs.append(np.asarray(extra_block[0])); be.append(np.asarray(extra_block[1]))
        bs = np.concatenate(bs); be = np.concatenate(be)
        # a start p is blocked if [p, p+L) intersects [bs, be) -> p in (bs-L, be)
        ms, me = merge_intervals(np.maximum(bs - L + 1, 0), np.minimum(be, chrom_len))
        # complement within [0, chrom_len-L]
        free_s = np.concatenate([[0], me]); free_e = np.concatenate([ms, [chrom_len - L + 1]])
        ok = free_e > free_s
        return free_s[ok], free_e[ok]


# ------------------------------------------------------------------ per-base code arrays
def _rmsk_code(cls, fam):
    if cls == "LINE":
        return RMSK_CODES["LINE/L1"] if fam == "L1" else RMSK_CODES["LINE/L2"] if fam == "L2" else RMSK_CODES["LINE/other"]
    if cls == "SINE":
        return RMSK_CODES["SINE/Alu"] if fam == "Alu" else RMSK_CODES["SINE/MIR"] if fam == "MIR" else RMSK_CODES["SINE/other"]
    if cls in ("LTR", "DNA", "Simple_repeat", "Low_complexity", "Satellite"):
        return RMSK_CODES[cls]
    return RMSK_CODES["other"]


def rmsk_table():
    path = _cache("rmsk_hg38.npz")
    if not os.path.exists(path):
        parts = []
        for ch in pd.read_csv(RMSK, sep="\t", header=None, usecols=[5, 6, 7, 11, 12],
                              names=["chr", "s", "e", "cls", "fam"], chunksize=1_000_000):
            ch = ch[ch["chr"].isin(PRIMARY)]
            keys = ch["cls"].astype(str) + "|" + ch["fam"].astype(str)
            u = {k: _rmsk_code(*k.split("|", 1)) for k in keys.unique()}
            parts.append(pd.DataFrame({"chr": ch["chr"].to_numpy(str), "s": ch["s"].to_numpy(np.int64),
                                       "e": ch["e"].to_numpy(np.int64),
                                       "code": keys.map(u).to_numpy(np.uint8)}))
        R = pd.concat(parts, ignore_index=True)
        np.savez_compressed(path, chr=R["chr"].to_numpy(str), s=R["s"].to_numpy(), e=R["e"].to_numpy(),
                            code=R["code"].to_numpy())
    z = np.load(path)
    return pd.DataFrame({"chr": z["chr"], "s": z["s"], "e": z["e"], "code": z["code"]})


def ccre_table():
    path = _cache("ccre_v4.npz")
    if not os.path.exists(path):
        src = CCRE if os.path.exists(CCRE) else CCRE_FALLBACK
        if not os.path.exists(src):
            raise FileNotFoundError("SCREEN cCRE BED missing: download "
                                    "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed to " + CCRE)
        d = pd.read_csv(src, sep="\t", header=None, usecols=[0, 1, 2, 5], names=["chr", "s", "e", "cls"])
        d = d[d["chr"].isin(PRIMARY)]
        code = d["cls"].map(lambda x: CCRE_CODES.get(x.split(",")[0], 0)).to_numpy(np.uint8)
        np.savez_compressed(path, chr=d["chr"].to_numpy(str), s=d["s"].to_numpy(np.int64),
                            e=d["e"].to_numpy(np.int64), code=code, src=np.array([src]))
    z = np.load(path)
    return pd.DataFrame({"chr": z["chr"], "s": z["s"], "e": z["e"], "code": z["code"]})


def base_code_fractions(table, chroms, starts, ends, ncodes, sizes):
    """For each query interval, fraction of its bases carrying each code (per-base array painted
    one chromosome at a time; later rows overwrite earlier ones on overlap). -> (n, ncodes)."""
    chroms = np.asarray(chroms); s = np.asarray(starts, np.int64); e = np.asarray(ends, np.int64)
    out = np.zeros((len(s), ncodes), np.float32)
    for c in np.unique(chroms):
        if c not in sizes:
            continue
        arr = np.zeros(sizes[c], np.uint8)
        t = table[table["chr"] == c]
        for a, b, k in zip(t["s"].to_numpy(), t["e"].to_numpy(), t["code"].to_numpy()):
            arr[a:b] = k
        idx = np.flatnonzero(chroms == c)
        for i in idx:
            v = arr[s[i]:e[i]]
            out[i] = np.bincount(v, minlength=ncodes)[:ncodes] / max(len(v), 1)
        del arr
    return out


# ------------------------------------------------------------------------ ChromHMM (hg19)
def chromhmm():
    """dict cell -> dict chrom -> (starts, ends, state int8) for the nine ENCODE/Broad cell lines
    (Ernst et al. 2011; UCSC hg19 wgEncodeBroadHmm, originally hg18)."""
    out = {}
    for cell in CHROMHMM_CELLS:
        d = pd.read_csv(os.path.join(CHROMHMM_DIR, f"wgEncodeBroadHmm{cell}HMM.bed.gz"), sep="\t",
                        header=None, usecols=[0, 1, 2, 3], names=["chr", "s", "e", "name"])
        d["state"] = d["name"].str.split("_").str[0].astype(np.int8)
        per = {}
        for c, g in d.groupby("chr"):
            g = g.sort_values("s")
            per[c] = (g["s"].to_numpy(np.int64), g["e"].to_numpy(np.int64), g["state"].to_numpy(np.int8))
        out[cell] = per
    return out


def chromhmm_state_at(H, chroms19, pos19):
    """(n, 9) int8 matrix of ChromHMM state at hg19 positions (0 = not covered / unlifted)."""
    n = len(pos19)
    M = np.zeros((n, len(CHROMHMM_CELLS)), np.int8)
    chroms19 = np.asarray(chroms19, dtype=object); pos19 = np.asarray(pos19, dtype=np.float64)
    for j, cell in enumerate(CHROMHMM_CELLS):
        for c in set(chroms19[~pd.isna(chroms19)]):
            if c not in H[cell]:
                continue
            s, e, st = H[cell][c]
            m = (chroms19 == c) & ~np.isnan(pos19)
            p = pos19[m].astype(np.int64)
            i = np.searchsorted(s, p, side="right") - 1
            ok = (i >= 0) & (p < e[np.clip(i, 0, None)])
            v = np.where(ok, st[np.clip(i, 0, None)], 0)
            M[np.flatnonzero(m), j] = v
    return M


def liftover_points(lo, chroms, pos):
    """pyliftover point conversion -> (chrom, pos, strand) arrays (None/NaN if unmapped/multi)."""
    oc, op, os_ = [], [], []
    for c, p in zip(chroms, pos):
        r = lo.convert_coordinate(c, int(p))
        if r and len(r) == 1:
            oc.append(r[0][0]); op.append(r[0][1]); os_.append(r[0][2])
        else:
            oc.append(None); op.append(np.nan); os_.append(None)
    return np.array(oc, dtype=object), np.array(op, dtype=float), np.array(os_, dtype=object)
