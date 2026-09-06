"""Patient-level metrics and joint patient bootstrap across partitions/seeds/models."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def scores(y, predictions):
    y, predictions = np.asarray(y), np.asarray(predictions)
    if len(np.unique(y)) != 2 or not np.isfinite(predictions).all():
        raise ValueError("AUROC requires both classes and finite predictions")
    return {"auroc": float(roc_auc_score(y, predictions)),
            "average_precision": float(average_precision_score(y, predictions)), "patients": len(y)}


def paired_bootstrap(y, baseline, candidate, samples=2000, seed=20260906):
    """Inputs [runs, patients]; same patients resampled jointly in every run/arm.

    The estimand is mean per-run AUROC/AP, not the score of an ensemble. Training
    variability is reported separately; patient intervals do not estimate it.
    """
    y = np.asarray(y)
    a, b = np.atleast_2d(baseline), np.atleast_2d(candidate)
    if a.shape != b.shape or a.shape[1] != len(y):
        raise ValueError("Matched runs and patient ordering required")
    def statistic(indices):
        ma = np.array([[scores(y[indices], row[indices])[k] for k in ("auroc", "average_precision")] for row in a]).mean(0)
        mb = np.array([[scores(y[indices], row[indices])[k] for k in ("auroc", "average_precision")] for row in b]).mean(0)
        return np.stack((ma, mb, mb - ma))
    point = statistic(np.arange(len(y)))
    rng, boot = np.random.default_rng(seed), []
    while len(boot) < samples:
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) == 2:
            boot.append(statistic(indices))
    low, high = np.quantile(np.stack(boot), [0.025, 0.975], axis=0)
    return {arm: {metric: {"estimate": float(point[i, j]), "ci95": [float(low[i, j]), float(high[i, j])]}
                  for j, metric in enumerate(("auroc", "average_precision"))}
            for i, arm in enumerate(("baseline", "candidate", "difference"))}
