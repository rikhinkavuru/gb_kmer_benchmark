# Upgrade 5 — Contamination generality via controlled negatives (Route 3)

Promotes the claim from "two FANTOM/ENCODE-lineage datasets are contaminated" toward "the artifact
is a property of the random-genomic negative-sampling **protocol**." Self-contained (no external data):
for the cohn and nt_enhancers POSITIVE sets, build matched negatives by **dinucleotide-preserving
shuffling of the positives** (Altschul–Erikson; `dinuc_shuffle.py`). These negatives have, by
construction, the same mono+dinucleotide composition (hence identical GC) as the positives, so any
composition gap is removed.

## Run
```
python src/upgrades/shuffled_neg/run_shuffled_neg.py        # results/upgrades/shuffled_neg.csv
python src/upgrades/shuffled_neg/make_fig_shuffled_neg.py   # figures/shuffled_neg.{pdf,png}
python src/upgrades/shuffled_neg/tests/test_sanity.py       # 5 sanity checks (dinuc shuffle exactness)
```

## Measured result (seed 42, 1000-bootstrap; dinucleotide composition preserved EXACTLY, frac=1.000)
| dataset | arm | TATA motif-only AUROC (95% CI) | GC AUROC (95% CI) | k-mer MCC |
|---|---|---|---|---|
| human_enhancers_cohn | original_neg | 0.371 `[.365,.376]` | 0.737 `[.732,.743]` | 0.461 |
| human_enhancers_cohn | **shuffled_neg** | **0.551** `[.546,.557]` | **0.500** `[.494,.507]` | 0.769 |
| nt_enhancers | original_neg | 0.393 `[.387,.399]` | 0.822 `[.816,.829]` | 0.514 |
| nt_enhancers | **shuffled_neg** | **0.507** `[.502,.511]` | **0.500** `[.491,.509]` | 0.545 |
| drosophila (control) | original_neg | 0.539 `[.526,.552]` | 0.487 `[.474,.501]` | 0.398 |
| drosophila (control) | **shuffled_neg** | **0.749** `[.738,.760]` | 0.500 `[.487,.514]` | 0.922 |

## Interpretation
- **cohn & nt_enhancers (contaminated):** against the real random-genomic negatives the negatives are
  TATA-enriched (AUROC 0.37, 0.39 < 0.5) and GC-separated (0.74, 0.82). Against dinucleotide-preserving
  shuffled negatives **the artifact vanishes**: GC → 0.500 (by construction) and the TATA motif-only
  AUROC collapses to ~0.5 (0.371→0.551, 0.393→0.507). The composition/TATA separation was therefore a
  property of the **negative-sampling protocol**, not of the positive class.
- **drosophila (clean control, TATA-in-positives):** the TATA AUROC RISES to 0.749 against shuffled
  negatives — because drosophila's positives genuinely contain TATA boxes (an assay control), and
  shuffling destroys them in the negatives. This correctly distinguishes a real positive-class motif
  from negative-set contamination, and validates that the test is not just "shuffling forces 0.5."
- Residual k-mer MCC on the shuffled arm (cohn 0.769, drosophila 0.922, nt_enhancers 0.545) is the
  higher-order (≥3-mer / motif) structure of the positives that survives dinucleotide matching.

## Optional external benchmark (documented TODO)
A second, composition-matched-by-design enhancer benchmark would show the artifact absent wherever
random-genomic negatives are NOT used. Exact source to add if pursued:
**DeepSTARR** (Almeida et al. 2022, *Nat. Genet.*), Drosophila S2 STARR-seq, GEO **GSE183939** /
the de Almeida lab `DeepSTARR` repo — its activity-regression construction avoids random-genomic
negatives. CPU-only; left as a TODO since the in-environment shuffled-negative control already isolates
the protocol.
