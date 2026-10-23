#!/usr/bin/env python3
"""Integrity / build verification of every input used by the negative-set provenance analysis.

Re-run after any interruption; writes results/upgrades/neg_provenance_input_checks.csv (one row per check,
PASS/FAIL + evidence) and exits non-zero if any check FAILs. Nothing is trusted from the first
(interrupted, 2026-09-23) run without passing here.

Checks
  files   : every .gz decompresses end-to-end; GENCODE header = v44/GRCh38; chain headers = hg38<->hg19
            chromosome sizes; ChromHMM files = 9 cell lines, 15 states, hg19 coordinates.
  build   : hg38.fa .fai sizes == hg38.chrom.sizes (primary chromosomes); RepeatMasker (UCSC rmsk)
            intervals are soft-masked in hg38.fa (random sample) and never exceed hg38 sizes;
            gap.txt N-gaps match the N content of hg38.fa per chromosome.
  caches  : cached npz tables (GENCODE genes/TSS/exons, rmsk, SCREEN cCRE, 1-kb bins) have exactly the
            row counts of their source files (primary chromosomes); genome-wide GC.
  data    : human_enhancers_cohn interval files: counts == results/results.csv n_train/n_test, and the
            strand-aware hg38 sequence at EVERY interval == the benchmark sequence loaded with the
            patched loader data.load_dataset (id order reconstructed from the sorted file names);
            nt_enhancers: counts == results/results_nt.csv, nt_map covers every sequence, and the
            pre-registered GENOMIC call (cov>=0.95 & id>=0.95) coincides with exact identity to the
            published iEnhancer-2L benchmark / independent-test sequences (two independent GitHub copies).
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import _common as C  # noqa: E402
import genome_utils as G  # noqa: E402

OUT = os.path.join(C.RESULTS_DIR, "neg_provenance_input_checks.csv")
HG19_SIZES = {"chr1": 249250621, "chr2": 243199373, "chrX": 155270560, "chr21": 48129895}
rows = []


def check(name, ok, evidence):
    rows.append(dict(check=name, status="PASS" if ok else "FAIL", evidence=str(evidence)))
    print(("PASS " if ok else "FAIL ") + name + " :: " + str(evidence), flush=True)


def gz_ok(path):
    n = 0
    with gzip.open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            n += len(b)
    return n


def main():
    E = G.EXT
    # ------------------------------------------------------------------ files decompress
    gz = [os.path.join(E, f) for f in ("gencode.v44.annotation.gtf.gz", "rmsk.txt.gz", "gap.txt.gz",
                                         "hg38ToHg19.over.chain.gz", "hg19ToHg38.over.chain.gz")]
    gz += [os.path.join(G.CHROMHMM_DIR, f"wgEncodeBroadHmm{c}HMM.bed.gz") for c in G.CHROMHMM_CELLS]
    for p in gz:
        try:
            n = gz_ok(p)
            check(f"gzip_integrity:{os.path.basename(p)}", n > 0, f"{os.path.getsize(p)} B -> {n} B")
        except Exception as ex:  # noqa: BLE001
            check(f"gzip_integrity:{os.path.basename(p)}", False, repr(ex))
    with gzip.open(os.path.join(E, "gencode.v44.annotation.gtf.gz"), "rt") as fh:
        h = fh.readline().strip()
    check("gencode_header", "GRCh38" in h and "version 44" in h, h)
    for f, t, q in (("hg38ToHg19.over.chain.gz", 248956422, 249250621),
                    ("hg19ToHg38.over.chain.gz", 249250621, 248956422)):
        with gzip.open(os.path.join(E, f), "rt") as fh:
            first = fh.readline().split()
        check(f"chain_header:{f}", first[2] == "chr1" and int(first[3]) == t and int(first[8]) == q,
              " ".join(first[:9]))

    # ------------------------------------------------------------------ hg38 reference
    sizes = G.chrom_sizes()
    cs = pd.read_csv(os.path.join(E, "hg38.chrom.sizes"), sep="\t", header=None, names=["c", "n"])
    cs = {c: int(n) for c, n in zip(cs.c, cs.n) if c in G.PRIMARY}
    check("hg38_fai_vs_chrom_sizes", sizes == cs and sizes["chr1"] == 248956422,
          f"{len(sizes)} primary chromosomes; chr1={sizes.get('chr1')}")

    # ------------------------------------------------------------------ ChromHMM hg19
    states_all = set()
    for cell in G.CHROMHMM_CELLS:
        d = pd.read_csv(os.path.join(G.CHROMHMM_DIR, f"wgEncodeBroadHmm{cell}HMM.bed.gz"), sep="\t",
                        header=None, usecols=[0, 1, 2, 3], names=["chr", "s", "e", "name"])
        st = d["name"].str.split("_").str[0].astype(int)
        states_all |= set(st.unique())
        mx = d.groupby("chr")["e"].max()
        ok = all(mx.get(c, 0) <= n for c, n in HG19_SIZES.items()) and mx.get("chr1", 0) > 248956422
        check(f"chromhmm_hg19:{cell}", ok and st.between(1, 15).all(),
              f"rows={len(d)} states={st.nunique()} chr1_max_end={mx.get('chr1')} (hg19 chr1=249250621, "
              f"hg38=248956422) covered_bp={int((d.e - d.s).sum())}")
    check("chromhmm_15_states", states_all == set(range(1, 16)), sorted(states_all))

    # ------------------------------------------------------------------ gap vs N content
    Bn = G.bins()
    gap = pd.read_csv(os.path.join(E, "gap.txt.gz"), sep="\t", header=None, usecols=[1, 2, 3],
                      names=["chr", "s", "e"])
    bad = []
    for c in G.PRIMARY:
        nN = sizes[c] - int(Bn[c][0].sum())
        ng = int((gap[gap.chr == c].e - gap[gap.chr == c].s).sum())
        if abs(nN - ng) > max(1000, 0.01 * max(nN, 1)):
            bad.append((c, nN, ng))
    check("gap_table_matches_hg38_N_content", not bad, f"mismatching chromosomes: {bad[:5]}")
    acgt = sum(float(v[0].sum()) for v in Bn.values()); gcn = sum(float(v[1].sum()) for v in Bn.values())
    check("genome_gc", 0.40 < gcn / acgt < 0.42 and 2.9e9 < acgt < 3.1e9,
          f"GC={gcn / acgt:.4f} over {acgt:.4e} ACGT bases")

    # ------------------------------------------------------------------ rmsk build + cache
    rm_src = 0; over = 0
    samp = []
    rng = np.random.RandomState(42)
    for ch in pd.read_csv(G.RMSK, sep="\t", header=None, usecols=[5, 6, 7, 11],
                          names=["chr", "s", "e", "cls"], chunksize=1_000_000):
        ch = ch[ch["chr"].isin(G.PRIMARY)]
        rm_src += len(ch)
        over += int((ch["e"].to_numpy() > ch["chr"].map(sizes).to_numpy()).sum())
        samp.append(ch.sample(min(400, len(ch)), random_state=rng))
    samp = pd.concat(samp)
    samp = samp[~samp["cls"].isin(["Simple_repeat", "Low_complexity"])].sample(2000, random_state=42)
    fa_soft = G.fasta(upper=False)
    low = tot = 0
    for c, s, e in zip(samp.chr, samp.s, samp.e):
        q = fa_soft[c][int(s):int(e)]
        low += sum(ch_.islower() for ch_ in q); tot += len(q)
    check("rmsk_is_hg38_softmask_concordance", low / tot > 0.95 and over == 0,
          f"{low / tot:.4f} of bases in 2000 random rmsk intervals are soft-masked in hg38.fa; "
          f"intervals past hg38 chromosome end={over}")
    rm = G.rmsk_table()
    check("cache_rmsk_rowcount", len(rm) == rm_src, f"cache={len(rm)} source(primary)={rm_src}")

    # ------------------------------------------------------------------ GENCODE cache
    ng = nt_ = nx = 0
    import re
    tss_set, ex_set = set(), set()
    with gzip.open(G.GENCODE_GTF, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            if f[0] not in G.PRIMARY:
                continue
            if f[2] == "gene":
                ng += 1
            elif f[2] == "transcript":   # cache de-duplicates TSS on (chr, pos, strand)
                tss_set.add((f[0], int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1, f[6]))
            elif f[2] == "exon":
                ex_set.add((f[0], int(f[3]) - 1, int(f[4])))
    genes, tss, exons = G.gencode()
    check("cache_gencode_counts", len(genes) == ng and len(tss) == len(tss_set) and len(exons) == len(ex_set),
          f"genes {len(genes)}/{ng}; unique TSS {len(tss)}/{len(tss_set)}; unique exons {len(exons)}/{len(ex_set)}")
    del tss_set, ex_set, re

    # ------------------------------------------------------------------ cCRE cache
    src = G.CCRE if os.path.exists(G.CCRE) else G.CCRE_FALLBACK
    n_src = 0; n_acc = 0
    with open(src) as fh:
        for line in fh:
            f = line.split("\t")
            if f[0] in G.PRIMARY:
                n_src += 1
                n_acc += f[4].startswith("EH38E")
    cc = G.ccre_table()
    check("cache_ccre_rowcount_and_version", len(cc) == n_src and n_acc == n_src and n_src > 2.3e6,
          f"cache={len(cc)} source(primary)={n_src} (SCREEN Registry V4 GRCh38 lists 2,348,854 cCREs); "
          f"EH38E accessions={n_acc}; source={src}")
    # fix round 1: the BED is kept inside the repo's external/ dir, with its md5 recorded (README)
    import hashlib
    h = hashlib.md5()
    with open(src, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    check("ccre_source_in_repo_md5", src == G.CCRE and os.path.exists(G.CCRE),
          f"source={src}; md5={h.hexdigest()} (downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed)")

    # ------------------------------------------------------------------ cohn intervals vs benchmark
    import data
    res = pd.read_csv(os.path.join(C.ROOT, "results", "results.csv"))
    r = res[(res.dataset == "human_enhancers_cohn") & (res.model == "lgbm")].iloc[0]
    ds = data.load_dataset("human_enhancers_cohn")
    check("cohn_counts_vs_results_csv", (len(ds["train_seqs"]), len(ds["test_seqs"])) == (r.n_train, r.n_test),
          f"loaded {len(ds['train_seqs'])}/{len(ds['test_seqs'])}; results.csv {r.n_train}/{r.n_test}")
    fa = G.fasta(upper=True)
    rc = str.maketrans("ACGT", "TGCA")
    mism = 0; n_int = 0
    for sp, seqs, y in (("train", ds["train_seqs"], ds["y_train"]), ("test", ds["test_seqs"], ds["y_test"])):
        for ci_, cl in enumerate(ds["classes"]):
            d = pd.read_csv(os.path.join(C.ROOT, "localization", "cohn", f"{sp}_{cl}.csv.gz"))
            d = d.iloc[np.argsort(d["id"].astype(str).to_numpy() + ".txt", kind="stable")]   # sorted glob order
            bench = [s for s, yy in zip(seqs, y) if yy == ci_]
            assert len(bench) == len(d), (sp, cl, len(bench), len(d))
            for (reg, s, e, strand), b in zip(d[["region", "start", "end", "strand"]].itertuples(index=False), bench):
                g = fa[reg][int(s):int(e)]
                g = g if strand == "+" else g[::-1].translate(rc)
                mism += g != b; n_int += 1
    check("cohn_hg38_sequence_equals_benchmark_all", mism == 0 and n_int == r.n_train + r.n_test,
          f"{n_int - mism}/{n_int} intervals reproduce the benchmark sequence exactly")

    # ------------------------------------------------------------------ nt_enhancers
    import data_nt
    rn = pd.read_csv(os.path.join(C.ROOT, "results", "results_nt.csv"))
    rn = rn[(rn.dataset == "nt_enhancers") & (rn.model == "lgbm")].iloc[0]
    m = pd.read_csv(os.path.join(C.RESULTS_DIR, "neg_provenance_nt_map.csv"))
    ntr = int((m.split == "train").sum()); nte = int((m.split == "test").sum())
    check("nt_counts_vs_results_nt_csv", (ntr, nte) == (rn.n_train, rn.n_test),
          f"nt_map train/test {ntr}/{nte}; results_nt.csv {rn.n_train}/{rn.n_test}")
    orig = ienhancer_originals()
    seqs = {sp: data_nt._read("enhancers", sp)[0] for sp in ("train", "test")}
    m["seq"] = [seqs[sp][i] for sp, i in zip(m.split, m.idx)]
    m["is_orig"] = m.seq.isin(orig)
    agree = int((m.is_orig == m.genomic.astype(bool)).sum())
    check("nt_genomic_call_equals_iEnhancer_original", agree == len(m),
          f"{agree}/{len(m)} rows agree; genomic={int(m.genomic.sum())}, iEnhancer-original={int(m.is_orig.sum())}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    nfail = sum(r_["status"] == "FAIL" for r_ in rows)
    print(f"wrote {OUT}: {len(rows) - nfail} PASS, {nfail} FAIL")
    sys.exit(1 if nfail else 0)


def read_fasta(path):
    out, h, s = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if h is not None:
                    out.append((h, "".join(s).upper()))
                h, s = line[1:], []
            else:
                s.append(line)
    if h is not None:
        out.append((h, "".join(s).upper()))
    return out


IEN = os.path.join(G.EXT, "ienhancer")
IEN_FILES = {  # (published split, class) -> file ; two independent GitHub copies (see ienhancer/SOURCES.txt)
    ("train", "strong"): "bert_layer2_train_strong_enhancers.fa", ("train", "weak"): "bert_layer2_train_weak_enhancers.fa",
    ("train", "non"): "bert_layer1_train_nonenhancers.fa", ("test", "strong"): "bert_layer2_test_strong_enhancers.fa",
    ("test", "weak"): "bert_layer2_test_weak_enhancers.fa", ("test", "non"): "bert_layer1_test_nonenhancers.fa"}
ELM_FILES = {("train", "enh"): "elm_original_enhancer_train.txt", ("train", "non"): "elm_original_non_enhancer_train.txt",
             ("test", "enh"): "elm_original_enhancer_ind.txt", ("test", "non"): "elm_original_non_enhancer_ind.txt"}


def ienhancer_originals():
    """Set of all published iEnhancer-2L benchmark + independent-test sequences; asserts the two
    GitHub copies are identical and have the published counts (742/742/1484; 100/100/200)."""
    a = {k: read_fasta(os.path.join(IEN, f)) for k, f in IEN_FILES.items()}
    b = {k: read_fasta(os.path.join(IEN, f)) for k, f in ELM_FILES.items()}
    exp = {("train", "strong"): 742, ("train", "weak"): 742, ("train", "non"): 1484,
           ("test", "strong"): 100, ("test", "weak"): 100, ("test", "non"): 200}
    for k, n in exp.items():
        assert len(a[k]) == n, (k, len(a[k]))
    for sp in ("train", "test"):
        assert {s for _, s in a[(sp, "strong")]} | {s for _, s in a[(sp, "weak")]} == {s for _, s in b[(sp, "enh")]}
        assert {s for _, s in a[(sp, "non")]} == {s for _, s in b[(sp, "non")]}
    return {s for v in a.values() for _, s in v}


if __name__ == "__main__":
    main()
