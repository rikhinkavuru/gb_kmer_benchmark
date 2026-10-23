#!/usr/bin/env python3
"""diagnose.py -- the three-route diagnostic, shipped as a standalone CLI / report card.

Takes (sequences, labels) [+ optional GRCh38-style coordinates] and emits a route-labeled report
card (JSON + human-readable Markdown) covering:
  * LEARNABILITY        -- k-mer spectrum + LightGBM MCC/AUROC (validation-selected k), bootstrap CI;
  * ROUTE 1 motif       -- max positive-direction motif-only AUROC (discriminability) + motif richness;
  * ROUTE 2 positional  -- B=1 vs B=8 position-binned delta-MCC (paired bootstrap);
  * ROUTE 3 negatives   -- TATA neg-excess + GC AUROC + the cleaning effect (remove excess TATA
                           negatives, re-measure the TATA motif-only AUROC);
  * a single ROUTE VERDICT per dataset.

It also emits the four checks the paper recommends every regulatory benchmark ship with:
  1. COMPOSITION FRACTION       -- the share of above-chance AUROC a 20-d composition signature
                                   (4 mono + 16 dinucleotide frequencies) reaches on its own;
  2. SHUFFLED-NEGATIVE CONTROL  -- the same motif/GC measures against dinucleotide-preserving
                                   shuffles of the positives, which have no composition gap by
                                   construction (the control that behaves *differently* on clean
                                   and contaminated data);
  3. EQUALIZED-SPLIT DELTA      -- the score lost when the negatives' joint GC + 16-dinucleotide
                                   distribution is matched to the positives' (positives untouched);
  4. TRAIN/TEST REDUNDANCY      -- the >=0.7 8-mer-Jaccard near-duplicate rate across the split.

Packaging mirrors the companion homology-leakage splitter: standalone, numpy/scipy/scikit-learn/
LightGBM + pyjaspar core, DETERMINISTIC (seed 42), CPU-only, documented thresholds. No torch.

CLI:
  python diagnose.py --input seqs.csv --out report     # seqs.csv: columns sequence,label[,chr,start,end,strand]
  -> writes report.json and report.md
Programmatic:
  from diagnose import diagnose
  card = diagnose(sequences, labels)                    # dict report card

Thresholds (documented, fixed): TATA/TBP flag >= 1.3 bits/pos (low-IC, specificity-controlled);
curated-TF motif hit >= 1.0 bits/pos; positional bins B=8; k in {3,4,5,6} by validation MCC;
bootstrap = 1000 resamples (test sequence = unit); seed = 42.
"""
import argparse
import json
import os
import sys

import numpy as np

_UP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _d in (_UP, os.path.join(_UP, "shuffled_neg"), os.path.join(_UP, "composition_clean"),
           os.path.join(_UP, "homology")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _common as C
import featurize
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from run_discriminability import CURATED_VERT, resolve_curated

SEED = 42
TATA_BITS = 1.3
THR_BITS = 1.0
N_BINS = 8
PROP_BINS = 20          # propensity histogram bins, matching run_composition_clean.py
JACCARD_THR = 0.7       # 8-mer-Jaccard near-duplicate threshold, matching measure_homology.py
KS = (3, 4, 5, 6)


def _select_k(seqs, y, seed):
    """Validation-selected k (stratified 80/20 carve of the data, max val MCC)."""
    from sklearn.metrics import matthews_corrcoef
    tr, va = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(np.zeros(len(y)), y))
    best_k, best = KS[0], -2.0
    for k in KS:
        X = featurize.kmer_spectrum(seqs, k)
        m = C.gbmodels.build_model("lgbm", seed, len(np.unique(y)))
        m.fit(X[tr], y[tr])
        pred = m.classes_[np.argmax(m.predict_proba(X[va]), axis=1)]
        mcc = matthews_corrcoef(y[va], pred)
        if mcc > best:
            best, best_k = mcc, k
    return best_k


def _composition_fraction(tr_seqs, ytr, te_seqs, yte, k, seed, boot):
    """composition_fraction = (AUROC_comp_only - 0.5)/(AUROC_full - 0.5) on the held-out test split.

    full = LightGBM on the k-mer spectrum at the selected k; comp_only = LightGBM on the 20-d
    composition signature (GC + 16 dinuc, _common.comp_signature). Both fit ONCE on train; AUROC is
    the class-1 binary AUROC. Point fraction from the full-data AUROCs; CI from a 1000-resample (or
    `boot`) percentile bootstrap with the test sequence as the resampling unit (seed 42), paired so
    full and comp AUROCs share each resample. Torch-free (sklearn roc_auc_score)."""
    from sklearn.metrics import roc_auc_score
    yte = np.asarray(yte, dtype=np.int64)

    def proba1(X_tr, X_te):
        m = C.gbmodels.build_model("lgbm", seed, len(np.unique(ytr)))
        m.fit(X_tr, ytr)
        proba = m.predict_proba(X_te)
        return proba[:, int(np.where(m.classes_ == 1)[0][0])]

    def auroc1(yy, p1):
        if len(np.unique(yy)) < 2:
            return float("nan")
        try:
            return float(roc_auc_score(yy, p1))
        except ValueError:
            return float("nan")

    p_full = proba1(featurize.kmer_spectrum(tr_seqs, k), featurize.kmer_spectrum(te_seqs, k))
    p_comp = proba1(C.comp_signature(tr_seqs, ks=(1, 2)), C.comp_signature(te_seqs, ks=(1, 2)))
    full_pt, comp_pt = auroc1(yte, p_full), auroc1(yte, p_comp)
    frac_pt = (comp_pt - 0.5) / (full_pt - 0.5) if (full_pt - 0.5) != 0 else float("nan")

    n = len(yte); rng = np.random.RandomState(seed); fr = np.full(boot, np.nan)
    for b in range(boot):
        idx = rng.randint(0, n, n)
        yb = yte[idx]
        if len(np.unique(yb)) < 2:
            continue
        d = auroc1(yb, p_full[idx]) - 0.5
        if d != 0:
            fr[b] = (auroc1(yb, p_comp[idx]) - 0.5) / d
    return dict(frac=round(float(frac_pt), 4),
                frac_lo=round(float(np.nanpercentile(fr, 2.5)), 4),
                frac_hi=round(float(np.nanpercentile(fr, 97.5)), 4),
                full_auroc=round(float(full_pt), 4), comp_auroc=round(float(comp_pt), 4))


def _shuffled_negative_control(pos_seqs, tbp, seed):
    """Control negatives with NO composition gap: dinucleotide-preserving shuffles of the positives.

    If the real negatives give a TATA/GC AUROC far from 0.5 but these give ~0.5, the class skew is
    compositional and comes from how the negatives were sampled. On a genuinely clean set, where the
    motif belongs to the positive class, the shuffled-negative AUROC instead moves AWAY from 0.5.
    This is the one control that behaves *differently* on clean and contaminated data.
    """
    import dinuc_shuffle as D
    r = np.random.RandomState(seed)
    sh = [D.dinuc_shuffle(s, r) for s in pos_seqs]
    preserved = float(np.mean([D.verify_dinuc_preserved(a, b) for a, b in zip(pos_seqs, sh)]))
    tata = C.motif_only_auroc(C.tata_hit_counts(pos_seqs, tbp, TATA_BITS),
                              C.tata_hit_counts(sh, tbp, TATA_BITS))
    gc = C.auc_float(C.gc_content(pos_seqs), C.gc_content(sh))
    return dict(tata_auroc_vs_shuffled=round(float(tata), 4),
                gc_auroc_vs_shuffled=round(float(gc), 4),
                dinuc_preserved_frac=round(preserved, 4), n_shuffled=len(sh))


def _equalized_delta(tr_seqs, ytr, te_seqs, yte, k, seed, boot, rng):
    """Composition-equalize the NEGATIVES within each split, refit, and report the score delta.

    Identical procedure to run_composition_clean.py's `comp_equalized` arm: one LR propensity
    p = P(positive | 4 mono + 16 dinucleotide frequencies) fit on the pooled splits, then negatives
    resampled per split so their propensity histogram matches the positives'. Positives are never
    dropped. The delta is the part of the score that the negative set's composition was supplying.
    """
    import run_composition_clean as CC
    comp_tr = C.comp_signature(tr_seqs, ks=(1, 2))
    comp_te = C.comp_signature(te_seqs, ks=(1, 2))
    prop = CC.fit_propensity(np.vstack([comp_tr, comp_te]), np.concatenate([ytr, yte]), seed)
    p_tr = prop.predict_proba(comp_tr)[:, 1]
    p_te = prop.predict_proba(comp_te)[:, 1]
    keep_tr = CC.match_proportional(p_tr[ytr == 1], p_tr[ytr == 0], PROP_BINS, rng)
    keep_te = CC.match_proportional(p_te[yte == 1], p_te[yte == 0], PROP_BINS, rng)
    if len(keep_tr) < 10 or len(keep_te) < 10:
        return dict(feasible=False, reason="fewer than 10 negatives survive equalization in a split",
                    retained_neg_frac=round(float((len(keep_tr) + len(keep_te)) /
                                                  max(int((ytr == 0).sum() + (yte == 0).sum()), 1)), 4))
    ce_tr, ce_ytr = CC.negatives_kept_split(tr_seqs, ytr, keep_tr)
    ce_te, ce_yte = CC.negatives_kept_split(te_seqs, yte, keep_te)
    before = C.kmer_fit_eval_boot(tr_seqs, ytr, te_seqs, yte, k, "lgbm", seed, boot, rng)
    after = C.kmer_fit_eval_boot(ce_tr, ce_ytr, ce_te, ce_yte, k, "lgbm", seed, boot, rng)
    neg_before = np.vstack([comp_tr[ytr == 0], comp_te[yte == 0]])
    neg_after = np.vstack([comp_tr[ytr == 0][keep_tr], comp_te[yte == 0][keep_te]])
    pos_all = np.vstack([comp_tr[ytr == 1], comp_te[yte == 1]])
    return dict(feasible=True,
                mcc_before=before["mcc"], mcc_after=after["mcc"],
                delta_mcc=round(after["mcc"] - before["mcc"], 4),
                auroc_before=before["auroc"], auroc_after=after["auroc"],
                retained_neg_frac=round(float((len(keep_tr) + len(keep_te)) /
                                              max(int((ytr == 0).sum() + (yte == 0).sum()), 1)), 4),
                max_abs_smd_before=round(float(np.nanmax(np.abs(CC.smd(pos_all, neg_before)))), 4),
                max_abs_smd_after=round(float(np.nanmax(np.abs(CC.smd(pos_all, neg_after)))), 4))


def _redundancy(tr_seqs, te_seqs):
    """Train/test near-duplicate rate: fraction of test sequences with a >=0.7 8-mer Jaccard
    neighbour in train (the audit run in measure_homology.py, applied to this split)."""
    import measure_homology as H
    jac = H.max_jaccard_to_train(H.presence_matrix(te_seqs), H.presence_matrix(tr_seqs))
    return dict(frac_ge_0_7=round(float((jac >= JACCARD_THR).mean()), 4),
                frac_verbatim=round(float((jac >= 0.999).mean()), 4),
                median_max_jaccard=round(float(np.median(jac)), 4),
                verdict=H.verdict({0.7: float((jac >= JACCARD_THR).mean())}), n_test=len(te_seqs))


def diagnose(seqs, y, coords=None, seed=SEED, boot=1000, verbose=False):
    """Return the three-route report-card dict for one (sequences, labels) dataset."""
    seqs = [s.strip().upper() for s in seqs]
    y = np.asarray(y, dtype=np.int64)
    classes = sorted(set(y.tolist())); nc = len(classes)
    binary = (nc == 2 and set(classes) == {0, 1})
    rng = np.random.RandomState(seed)
    tr, te = next(StratifiedShuffleSplit(1, test_size=0.25, random_state=seed).split(np.zeros(len(y)), y))
    k = _select_k([seqs[i] for i in tr], y[tr], seed)
    if verbose:
        print(f"  selected k={k}; n={len(seqs)} classes={nc}", flush=True)

    # LEARNABILITY (global k-mer spectrum)
    tr_seqs = [seqs[i] for i in tr]; te_seqs = [seqs[i] for i in te]
    learn = C.kmer_fit_eval_boot(tr_seqs, y[tr], te_seqs, y[te], k, "lgbm", seed, boot, rng)

    card = dict(n=len(seqs), n_classes=nc, k=k, binary=binary,
                learnability=dict(mcc=learn["mcc"], mcc_ci=[learn["mcc_lo"], learn["mcc_hi"]],
                                  auroc=learn["auroc"], n_test=learn["n_test"]))

    # COMPOSITION FRACTION (binary only): share of ABOVE-CHANCE AUROC explained by composition
    # (GC + 16 dinuc) alone. full = k-mer spectrum at the selected k; comp_only = comp_signature;
    # both fit once on train, evaluated on the SAME held-out test resamples (paired bootstrap).
    if binary:
        cf = _composition_fraction(tr_seqs, y[tr], te_seqs, y[te], k, seed, boot)
        card["composition_fraction"] = dict(
            value=cf["frac"], ci=[cf["frac_lo"], cf["frac_hi"]],
            full_auroc=cf["full_auroc"], comp_only_auroc=cf["comp_auroc"],
            definition="(AUROC_comp_only-0.5)/(AUROC_full-0.5)")
        # EQUALIZED-SPLIT DELTA: what the score loses once the negatives' joint GC + 16-dinucleotide
        # distribution is matched to the positives'.
        card["equalized_delta"] = _equalized_delta(tr_seqs, y[tr], te_seqs, y[te], k, seed, boot,
                                                   np.random.RandomState(seed))

    # TRAIN/TEST REDUNDANCY: 8-mer-Jaccard near-duplicate rate across this split.
    card["redundancy"] = _redundancy(tr_seqs, te_seqs)

    # ROUTE 2 positional (B=1 vs B=8), paired bootstrap on the held-out test set
    from sklearn.metrics import matthews_corrcoef
    def fit_pred(Xtr, Xte):
        m = C.gbmodels.build_model("lgbm", seed, nc); m.fit(Xtr, y[tr])
        return m.classes_[np.argmax(m.predict_proba(Xte), axis=1)]
    Xtr_g = featurize.kmer_spectrum(tr_seqs, k); Xte_g = featurize.kmer_spectrum(te_seqs, k)
    pred_g = fit_pred(Xtr_g, Xte_g)
    Xtr_p = sparse.hstack([Xtr_g, featurize.binned_kmer_spectrum(tr_seqs, k, N_BINS)], format="csr")
    Xte_p = sparse.hstack([Xte_g, featurize.binned_kmer_spectrum(te_seqs, k, N_BINS)], format="csr")
    pred_p = fit_pred(Xtr_p, Xte_p)
    yte = y[te]; n = len(yte); dd = np.empty(boot)
    for b in range(boot):
        idx = rng.randint(0, n, n)
        dd[b] = matthews_corrcoef(yte[idx], pred_p[idx]) - matthews_corrcoef(yte[idx], pred_g[idx])
    d_mcc = float(matthews_corrcoef(yte, pred_p) - matthews_corrcoef(yte, pred_g))
    dlo, dhi = np.percentile(dd, [2.5, 97.5])
    card["route2_positional"] = dict(delta_mcc=round(d_mcc, 4), delta_ci=[round(float(dlo), 4), round(float(dhi), 4)],
                                     significant=bool(not (dlo <= 0 <= dhi)))

    # Route 1 + Route 3 need a positive/negative contrast (binary only)
    if binary:
        pos = [seqs[i] for i in np.where(y == 1)[0]]; neg = [seqs[i] for i in np.where(y == 0)[0]]
        cap = 4000
        if len(pos) > cap:
            pos = [pos[i] for i in rng.choice(len(pos), cap, replace=False)]
        if len(neg) > cap:
            neg = [neg[i] for i in rng.choice(len(neg), cap, replace=False)]
        pcodes = C.motif_match.encode_sequences(pos); ncodes = C.motif_match.encode_sequences(neg)
        # ROUTE 1: max positive-direction motif-only AUROC over curated TFs
        curated, _ = resolve_curated(CURATED_VERT, C.motif_jaspar.load_pssms("vertebrates"))
        best_auc, best_tf, rich = 0.5, "", []
        for fam, name, m in curated:
            pc = C.motif_match.count_hits_batch(pcodes, m["pssm"], THR_BITS)
            ncc = C.motif_match.count_hits_batch(ncodes, m["pssm"], THR_BITS)
            auc = C.auc_counts(pc.astype(np.int64), ncc.astype(np.int64))
            rich.append((pc.mean() + ncc.mean()) / 2)
            if auc > best_auc:
                best_auc, best_tf = auc, name
        card["route1_motif"] = dict(max_pos_motif_auroc=round(float(best_auc), 4), best_TF=best_tf,
                                    motif_richness=round(float(np.mean(rich)), 3))
        # ROUTE 3: TATA neg-excess + GC AUROC + cleaning effect
        tbp = C.load_tbp()
        ptata = C.motif_match.count_hits_batch(pcodes, tbp["pssm"], TATA_BITS)
        ntata = C.motif_match.count_hits_batch(ncodes, tbp["pssm"], TATA_BITS)
        pos_rate = float((ptata >= 1).mean()); neg_rate = float((ntata >= 1).mean())
        gc_pos = C.gc_content(pos); gc_neg = C.gc_content(neg)
        tata_auc = C.auc_counts(ptata.astype(np.int64), ntata.astype(np.int64))
        gc_auc = C.auc_float(gc_pos, gc_neg)
        # cleaning: drop excess flagged negatives so neg flag-rate ~ pos flag-rate; re-measure TATA AUROC
        neg_idx = np.arange(len(neg)); flagged = ntata >= 1
        keep_fl = int(round(pos_rate * len(neg)))
        fl_idx = neg_idx[flagged]
        if len(fl_idx) > keep_fl:
            sel = rng.permutation(len(fl_idx))[:keep_fl]
            keep = np.sort(np.concatenate([neg_idx[~flagged], fl_idx[sel]]))
        else:
            keep = neg_idx
        cleaned_tata_auc = C.auc_counts(ptata.astype(np.int64), ntata[keep].astype(np.int64))
        card["route3_negatives"] = dict(
            tata_pos_rate=round(pos_rate, 4), tata_neg_rate=round(neg_rate, 4),
            tata_neg_excess=round(neg_rate - pos_rate, 4),
            tata_motif_auroc=round(float(tata_auc), 4), gc_auroc=round(float(gc_auc), 4),
            cleaned_tata_auroc=round(float(cleaned_tata_auc), 4),
            cleaning_moves_to_chance=bool(abs(cleaned_tata_auc - 0.5) < abs(tata_auc - 0.5)),
            frac_neg_removed=round(1 - len(keep) / max(len(neg), 1), 4))
        # SHUFFLED-NEGATIVE CONTROL: the same two measures against composition-matched negatives.
        card["shuffled_negative_control"] = _shuffled_negative_control(list(pos), tbp, seed)
        # optional coordinate hook (Route 3 localization) -- only summarized if coords given
        if coords is not None:
            mapped = sum(1 for c in coords if c and c[0])
            card["route3_negatives"]["coords_provided"] = True
            card["route3_negatives"]["coords_mapped_frac"] = round(mapped / max(len(coords), 1), 4)

    card["verdict"] = _verdict(card)
    return card


def _verdict(card):
    """Single route verdict from the documented decision rules."""
    mcc = card["learnability"]["mcc"]
    pos = card.get("route2_positional", {})
    r1 = card.get("route1_motif"); r3 = card.get("route3_negatives")
    if mcc >= 0.7:
        if r1 and r1["max_pos_motif_auroc"] >= 0.65:
            return "SOLVED (class-specific motif discriminability)"
        return "SOLVED"
    if pos.get("significant") and pos.get("delta_mcc", 0) > 0.2:
        return "ROUTE 2: POSITIONAL (fixed-position signal discarded by the position-blind spectrum)"
    if r3 and r3["tata_neg_excess"] >= 0.10 and r3["tata_motif_auroc"] < 0.5 and r3["cleaning_moves_to_chance"]:
        return "ROUTE 3: NEGATIVE-SET CONTAMINATION (AT-rich/TATA compositional bias in the negatives)"
    if r1 and r1["motif_richness"] > 0.2 and r1["max_pos_motif_auroc"] < 0.6:
        return "ROUTE 1: SHARED-MOTIF (motif-rich but non-discriminative; motifs shared across classes)"
    return "HARD (no single route dominates; inspect the per-route metrics)"


# ----------------------------------------------------------------- CLI / IO
def _read_input(path):
    import pandas as pd
    df = pd.read_csv(path)
    seqs = df["sequence"].astype(str).tolist(); y = df["label"].to_numpy()
    coords = None
    if {"chr", "start", "end", "strand"} <= set(df.columns):
        coords = [(str(r.chr), int(r.start), int(r.end), str(r.strand)) if str(r.chr) else None
                  for r in df.itertuples()]
    return seqs, y, coords


def _to_markdown(card, name):
    L = [f"# Three-route diagnostic report card -- {name}", "",
         f"- **n** = {card['n']} sequences, {card['n_classes']} classes, selected **k = {card['k']}**",
         f"- **Learnability** (k-mer + LightGBM): MCC = **{card['learnability']['mcc']:.3f}** "
         f"{card['learnability']['mcc_ci']}, AUROC = {card['learnability']['auroc']}",
         f"- **Route 2 (positional)**: delta-MCC(B=8 - B=1) = **{card['route2_positional']['delta_mcc']:+.3f}** "
         f"{card['route2_positional']['delta_ci']} ({'significant' if card['route2_positional']['significant'] else 'ns'})"]
    if "composition_fraction" in card:
        cf = card["composition_fraction"]
        L.append(f"- **Composition fraction** = **{cf['value']:+.3f}** {cf['ci']} "
                 f"(comp-only AUROC {cf['comp_only_auroc']:.3f} / full AUROC {cf['full_auroc']:.3f}; "
                 f"{cf['definition']}; share of above-chance AUROC from GC+16 dinuc, can exceed 1 / go <0)")
    if "route1_motif" in card:
        r1 = card["route1_motif"]
        L.append(f"- **Route 1 (motif)**: max positive motif-only AUROC = **{r1['max_pos_motif_auroc']:.3f}** "
                 f"(best TF {r1['best_TF']}), motif richness = {r1['motif_richness']:.2f} hits/seq")
    if "route3_negatives" in card:
        r3 = card["route3_negatives"]
        L.append(f"- **Route 3 (negatives)**: TATA neg-excess = **{r3['tata_neg_excess']:+.3f}**, "
                 f"TATA motif AUROC = {r3['tata_motif_auroc']:.3f} -> cleaned {r3['cleaned_tata_auroc']:.3f}, "
                 f"GC AUROC = {r3['gc_auroc']:.3f}")
    if "shuffled_negative_control" in card:
        sc = card["shuffled_negative_control"]
        L.append(f"- **Shuffled-negative control** (dinucleotide-preserving shuffles of the positives, "
                 f"no composition gap by construction): TATA AUROC = **{sc['tata_auroc_vs_shuffled']:.3f}**, "
                 f"GC AUROC = {sc['gc_auroc_vs_shuffled']:.3f}. Near 0.5 => the real-negative skew was "
                 f"compositional; away from 0.5 => the motif genuinely belongs to the positive class.")
    if "equalized_delta" in card:
        ed = card["equalized_delta"]
        if ed.get("feasible"):
            L.append(f"- **Equalized-split delta**: MCC {ed['mcc_before']:.3f} -> "
                     f"**{ed['mcc_after']:.3f}** (**{ed['delta_mcc']:+.3f}**) after matching the "
                     f"negatives' joint GC + 16-dinucleotide distribution to the positives'; "
                     f"{ed['retained_neg_frac']:.1%} of negatives retained, max|SMD| "
                     f"{ed['max_abs_smd_before']:.3f} -> {ed['max_abs_smd_after']:.3f}")
        else:
            L.append(f"- **Equalized-split delta**: not computable ({ed.get('reason')}; "
                     f"{ed.get('retained_neg_frac', 0):.1%} of negatives would survive) — itself a "
                     f"strong contamination signal, since almost no negative matches any positive's "
                     f"composition")
    if "redundancy" in card:
        rd = card["redundancy"]
        L.append(f"- **Train/test redundancy**: {rd['frac_ge_0_7']:.1%} of test sequences have a "
                 f">=0.7 8-mer-Jaccard near-duplicate in train "
                 f"({rd['frac_verbatim']:.1%} verbatim) — {rd['verdict']}")
    L += ["", f"## VERDICT: {card['verdict']}", "",
          "_Thresholds: TATA>=1.3 bits/pos, curated-TF>=1.0 bits/pos, B=8, k by validation MCC, "
          "1000-resample bootstrap, seed 42. CPU-only; no PyTorch._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with columns sequence,label[,chr,start,end,strand]")
    ap.add_argument("--out", default="report", help="output prefix (writes <out>.json and <out>.md)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    seqs, y, coords = _read_input(args.input)
    name = args.name or os.path.basename(args.input)
    print(f"diagnosing {name}: {len(seqs)} sequences ...", flush=True)
    card = diagnose(seqs, y, coords=coords, seed=args.seed, boot=args.boot, verbose=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(card, fh, indent=2)
    with open(args.out + ".md", "w") as fh:
        fh.write(_to_markdown(card, name))
    print("\n" + _to_markdown(card, name))
    print(f"\nWrote {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
