"""Per-cell evaluation metrics.

accuracy, MCC (Matthews correlation -- natively multiclass), and AUROC. For
multiclass datasets AUROC is one-vs-rest, macro-averaged, matching the spec.
AUROC is returned as NaN whenever it is undefined (e.g. only one class present
in the test set), never raising.
"""
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Return dict(accuracy, mcc, auroc). ``y_proba`` is (n_samples, n_classes),
    column order matching the estimator's sorted ``classes_`` (0..n_classes-1)."""
    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    auroc = float("nan")
    try:
        if len(np.unique(y_true)) < 2:
            auroc = float("nan")
        elif n_classes == 2:
            auroc = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            auroc = float(roc_auc_score(y_true, y_proba,
                                        multi_class="ovr", average="macro"))
    except Exception:
        auroc = float("nan")

    return dict(accuracy=acc, mcc=mcc, auroc=auroc)
