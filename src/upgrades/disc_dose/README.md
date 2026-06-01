# Upgrade 3 — Semi-synthetic discriminability dose-response (Route 1)

Replaces the underpowered pooled n=11 real-task Spearman (discriminability → learnability) with a
**causal, arbitrary-n dose-response**: dial the motif-only AUROC (discriminability `d`) by implanting
a TF motif more in positives than negatives, and measure the k-mer+LightGBM benchmark MCC.

## Method
- Background = random ACGT, length 251 and base composition matched to `human_nontata_promoters`.
- Implant **PWM-sampled** instances (drawn from the motif PPM, not the consensus) into a fraction
  `p_pos` of positives (and `p_neg=0` of negatives); sweep `p_pos`.
- `d` is **measured** (not assumed) as the motif-only AUROC = `auc_counts(pos hits, neg hits)` at
  1.0 bits/pos — the same definition as `run_discriminability.py`.
- 3 motifs of differing information content (SP1 14.2 bits, MYC 11.0, TBP 8.8; all > 1 bit/position
  so the fixed-threshold hit definition is sensitive) × 5 seeds (42–46) × 9 doses.

## Run
```
python src/upgrades/disc_dose/run_disc_dose.py            # results/upgrades/disc_dose.csv
python src/upgrades/disc_dose/make_fig_disc_dose.py       # figures/disc_dose.{pdf,png}
python src/upgrades/disc_dose/tests/test_sanity.py        # 4 sanity checks
```

## Measured result (seed 42; n = 135 = 3 motifs × 5 seeds × 9 doses)
- **Spearman(d, MCC) = +0.924 (p = 2.4e-57)** — monotone increasing.
- **Linear slope dMCC/dd = +1.99, 95% bootstrap CI [+1.92, +2.06]** (excludes 0).
- Per motif (higher IC reaches higher d and higher MCC):

| motif | total IC | d range | MCC range |
|---|---|---|---|
| SP1 | 14.2 bits | 0.49 – 0.96 | −0.05 – 0.88 |
| MYC | 11.0 bits | 0.50 – 0.87 | −0.06 – 0.72 |
| TBP | 8.8 bits | 0.49 – 0.81 | −0.09 – 0.60 |

**Interpretation:** learnability is **causally** driven by motif discriminability — dial `d` up and the
benchmark MCC rises monotonically, across three motifs of differing information content. This is the
arbitrary-n causal replacement for the real-task correlation; the slope is tight and excludes 0.

## Notes
- A long, low-per-position motif (CTCF, 0.63 bits/pos) is deliberately excluded: its PWM-sampled
  instances never cross the 1.0 bits/pos hit threshold, so `d` would read ~0.5 even as the k-mer model
  clearly learns it — an artifact of the fixed hit threshold, not of discriminability. (Documented in
  the run script.) k=6, n/class=1000, seed 42; purely synthetic, CPU-only.
