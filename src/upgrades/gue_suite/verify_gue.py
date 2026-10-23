#!/usr/bin/env python3
"""GUE data integrity + reference-table verification (run BEFORE any model is fit; no models here).

1. Archive identity: SHA-256 of external/gue/GUE.zip == gue_data.GUE_ZIP_SHA256 (the archive linked
   from the DNABERT-2 README, Google-Drive id 1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-).
2. Extraction integrity: every analysed train/dev/test.csv on disk has the same size and CRC-32 as
   its member in the zip (so the files read by run_gue.py ARE the archive's files).
3. Row counts vs DNABERT-2 Table 12 (gue_data.PAPER_SPLITS).
4. Per-split data profile: class counts, sequence-length range, non-ACGT symbol rate, within-split
   exact duplicates (forward strand, n_dup_within_split; and strand-aware, n_dup_within_split_canonical,
   added in fix round 1), and exact (forward or reverse-complement) sequence identity train<->test,
   train<->dev and dev<->test (verbatim leakage; the 8-mer-Jaccard redundancy is in run_gue.py).
5. Published numbers: re-parses DNABERT-2 Tables 6 and 9 from the saved arXiv full text
   (external/gue/refs/dnabert2_arxiv_2306.15006_fulltext.txt) and checks every hand-transcribed value
   in gue_data (Table 6 rows, Table 9 HyenaDNA and DNABERT-2 rows) against the parse.
   Also writes the full parsed tables (all Table 6 models, Table 9 HyenaDNA / DNABERT-2 w/o PT /
   DNABERT-2) to results/upgrades/gue_published.csv, which summarize_gue.py uses.

Outputs: results/upgrades/gue_integrity.csv, results/upgrades/gue_published.csv. Exit code != 0 on
any mismatch. Deterministic, no models, < 1 GB RAM.
"""
import os
import re
import sys
import zipfile
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gue_data as G  # noqa: E402

ROOT = G.ROOT
RES = os.path.join(ROOT, "results", "upgrades")
REF_TXT = os.path.join(ROOT, "external", "gue", "refs", "dnabert2_arxiv_2306.15006_fulltext.txt")
_RC = str.maketrans("ACGTN", "TGCAN")


def crc32_file(path, chunk=1 << 20):
    c = 0
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def canon(s):
    r = s.translate(_RC)[::-1]
    return s if s <= r else r


# ------------------------------------------------------------------ published-table parser
T6_MODELS = ["DNABERT (3-mer)", "DNABERT (4-mer)", "DNABERT (5-mer)", "DNABERT (6-mer)",
             "NT-500M-human", "NT-500M-1000g", "NT-2500M-1000g", "NT-2500M-multi",
             "DNABERT-2", "DNABERT-2♦"]
T9_MODELS = ["HyenaDNA", "CNN", "DNABERT-2 w/o PT", "DNABERT-2"]
NUM = r"\d+(?:\.\d+)?"


def _parse_block(text, models):
    """Return {model: [values in order of appearance]} for rows '<model> v1 v2 ...' in text."""
    out = {m: [] for m in models}
    # longest names first so 'DNABERT-2 w/o PT' / 'DNABERT-2♦' are not read as 'DNABERT-2'
    alts = sorted(models, key=len, reverse=True)
    pat = re.compile("(" + "|".join(re.escape(m) for m in alts) + r")(?![♦\w/-]| w/o)\s+((?:" + NUM +
                     r"\s+)+)")
    for mt in pat.finditer(text + " "):
        out[mt.group(1)].extend(float(v) for v in mt.group(2).split())
    return out


def parse_published():
    txt = open(REF_TXT, encoding="utf-8").read()
    a = txt.index("A.1 ALL EXPERIMENT RESULTS"); b = txt.index("Table 6: This table presents")
    t6 = _parse_block(txt[a:b], T6_MODELS)
    c = txt.index("HyenaDNA 67.17"); d = txt.index("Table 9: Benchmark DNABERT-2")
    t9 = _parse_block(txt[c - 200:d], T9_MODELS)
    return t6, t9


def main():
    ok = True
    # 1. archive
    h = G.sha256(G.GUE_ZIP)
    arch_ok = h == G.GUE_ZIP_SHA256
    ok &= arch_ok
    print(f"[archive] sha256 {h} {'== pinned' if arch_ok else '!= PINNED ' + G.GUE_ZIP_SHA256}")
    z = zipfile.ZipFile(G.GUE_ZIP)
    members = {i.filename: i for i in z.infolist()}

    rows = []
    for task in G.ORDER:
        rel = G.TASKS[task][0]
        seqs = {}
        for s in ("train", "dev", "test"):
            p = os.path.join(G.task_dir(task), f"{s}.csv")
            m = members.get(f"GUE/{rel}/{s}.csv")
            crc_ok = bool(m is not None and m.file_size == os.path.getsize(p) and m.CRC == crc32_file(p))
            df = pd.read_csv(p)
            sq = df["sequence"].astype(str).str.strip().str.upper().tolist()
            y = df["label"].to_numpy(np.int64)
            seqs[s] = sq
            L = np.array([len(x) for x in sq])
            nonacgt = sum(len(x) - sum(x.count(c) for c in "ACGT") for x in sq) / max(L.sum(), 1)
            cls = {int(c): int((y == c).sum()) for c in np.unique(y)}
            rows.append(dict(task=task, split=s, n=len(sq), paper_n=G.PAPER_SPLITS[task][("train", "dev", "test").index(s)],
                             crc_matches_zip=crc_ok, n_classes=len(cls), class_counts=str(cls),
                             len_min=int(L.min()), len_median=float(np.median(L)), len_max=int(L.max()),
                             nonacgt_rate=round(float(nonacgt), 6),
                             n_dup_within_split=int(len(sq) - len(set(sq))),
                             # [fix round 1] strand-aware count (a sequence and its reverse complement are one)
                             n_dup_within_split_canonical=int(len(sq) - len(set(canon(x) for x in sq)))))
            ok &= crc_ok
        can = {s: set(canon(x) for x in seqs[s]) for s in seqs}
        for r in rows[-3:]:
            s = r["split"]
            r["n_rows_equal_table12"] = bool(r["n"] == r["paper_n"])
        te_c = [canon(x) for x in seqs["test"]]; dv_c = [canon(x) for x in seqs["dev"]]
        rows[-1]["test_in_train_exact"] = int(sum(x in can["train"] for x in te_c))
        rows[-1]["test_in_dev_exact"] = int(sum(x in can["dev"] for x in te_c))
        rows[-2]["dev_in_train_exact"] = int(sum(x in can["train"] for x in dv_c))
        tot = sum(r["n"] for r in rows[-3:]); ptot = sum(G.PAPER_SPLITS[task])
        cnt_ok = all(r["n_rows_equal_table12"] for r in rows[-3:])
        note = "" if cnt_ok else (" (total equal)" if tot == ptot else " (TOTAL DIFFERS)")
        if task != "covid":
            ok &= cnt_ok
        else:
            ok &= tot == ptot
        print(f"[{task:<20}] n={[r['n'] for r in rows[-3:]]} table12={list(G.PAPER_SPLITS[task])} "
              f"{'OK' if cnt_ok else 'DIFF' + note} crc={'OK' if all(r['crc_matches_zip'] for r in rows[-3:]) else 'FAIL'} "
              f"len={rows[-1]['len_min']}-{rows[-1]['len_max']} test_in_train={rows[-1]['test_in_train_exact']}",
              flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(RES, "gue_integrity.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    # 5. published tables
    t6, t9 = parse_published()
    pub_rows = []
    for m, v in t6.items():
        if len(v) != len(G._COLS):
            print(f"[published] Table 6 {m}: parsed {len(v)} values (expected 28)"); ok = False; continue
        for c, x in zip(G._COLS, v):
            pub_rows.append(dict(source="DNABERT-2 Table 6", model=m, task=c, value_x100=x))
    for m, v in t9.items():
        if len(v) != len(G._COLS):
            print(f"[published] Table 9 {m}: parsed {len(v)} values (expected 28) -> not used "
                  f"(ambiguous row)")
            continue
        for c, x in zip(G._COLS, v):
            pub_rows.append(dict(source="DNABERT-2 Table 9", model=m, task=c, value_x100=x))
    pub = pd.DataFrame(pub_rows)
    pub["metric"] = np.where(pub.task == "covid", "macro-F1 x100", "MCC x100")
    # cross-check the hand transcription in gue_data
    n_chk = n_bad = 0
    for m, d in G.PUBLISHED.items():
        src, mm = (("DNABERT-2 Table 9", "HyenaDNA") if m.startswith("HyenaDNA") else
                   ("DNABERT-2 Table 6", "DNABERT-2♦" if "further" in m else m))
        for c, x in d.items():
            got = pub[(pub.source == src) & (pub.model == mm) & (pub.task == c)].value_x100
            n_chk += 1
            if len(got) != 1 or abs(float(got.iloc[0]) - x) > 1e-9:
                n_bad += 1; print(f"  MISMATCH {m} {c}: gue_data {x} vs parsed {list(got)}")
    for c, x in G.PUBLISHED_T9_DNABERT2.items():
        got = pub[(pub.source == "DNABERT-2 Table 9") & (pub.model == "DNABERT-2") & (pub.task == c)].value_x100
        n_chk += 1
        if len(got) != 1 or abs(float(got.iloc[0]) - x) > 1e-9:
            n_bad += 1; print(f"  MISMATCH T9 DNABERT-2 {c}: gue_data {x} vs parsed {list(got)}")
    ok &= n_bad == 0
    print(f"[published] {n_chk} hand-transcribed values checked against the parsed text: {n_bad} mismatches")
    t6d2 = pub[(pub.source == "DNABERT-2 Table 6") & (pub.model == "DNABERT-2")].set_index("task").value_x100
    t9d2 = pub[(pub.source == "DNABERT-2 Table 9") & (pub.model == "DNABERT-2")].set_index("task").value_x100
    diff = [c for c in G._COLS if abs(t6d2[c] - t9d2[c]) > 1e-9]
    print(f"[published] DNABERT-2 Table 6 vs Table 9 rows differ on {len(diff)} datasets: {diff}")
    out2 = os.path.join(RES, "gue_published.csv")
    pub.to_csv(out2, index=False)
    print(f"wrote {out2}")
    print("VERIFY", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
