"""The two CPU-only classifiers, with fixed and documented hyperparameters.

  lr    LogisticRegression  -- linear floor (L2 penalty, C=1.0, lbfgs solver)
  lgbm  LGBMClassifier      -- gradient-boosted trees, the main model

Both are seeded by the shared ``--seed`` and consume the cached L1-normalized
k-mer spectrum. Linear models are scale-sensitive, so ``lr`` additionally
L2-normalizes each row (``Normalizer``) inside its pipeline -- without this the
tiny per-feature frequencies at high k (4^6 = 4096 dims, each ~1e-3) make the
L2-regularized solver underfit and collapse to the majority class. LightGBM is
scale-invariant (tree splits), so it consumes the raw frequency spectrum.
LightGBM is run in deterministic mode so a fixed seed + fixed config reproduces
results bit-for-bit. No GPU is ever requested.

Hyperparameters are intentionally fixed (not tuned per dataset) so the benchmark
measures the featurization x model-family axes cleanly and stays reproducible.
"""
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# --- fixed hyperparameters (see README for rationale) --------------------
LR_PARAMS = dict(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    n_jobs=-1,
)

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
    n_jobs=-1,
    verbose=-1,
    deterministic=True,     # reproducible given a fixed seed...
    force_row_wise=True,    # ...required by deterministic mode
)

MODELS = ["lr", "lgbm"]


def build_model(name, seed, n_classes):
    """Return a fresh, unfitted estimator. ``n_classes`` is accepted for API
    symmetry; both estimators handle binary and multiclass automatically."""
    key = name.lower()
    if key in ("lr", "logreg", "logisticregression"):
        # Row L2-normalization makes the linear floor scale-robust across k.
        return make_pipeline(
            Normalizer(norm="l2"),
            LogisticRegression(random_state=seed, **LR_PARAMS),
        )
    if key in ("lgbm", "lightgbm"):
        return LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    raise ValueError(f"unknown model {name!r} (choose from {MODELS})")
