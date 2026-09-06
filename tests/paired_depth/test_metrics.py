import numpy as np
import pytest

from ultrai.paired_depth.metrics import paired_bootstrap, scores


def test_patient_bootstrap_resamples_matched_runs_jointly():
    labels = [0, 0, 1, 1, 0, 1]
    predictions = np.array([[0.1, 0.7, 0.6, 0.9, 0.4, 0.5], [0.2, 0.6, 0.5, 0.7, 0.1, 0.9]])
    result = paired_bootstrap(labels, predictions, predictions, samples=50)
    for metric in ("auroc", "average_precision"):
        assert result["difference"][metric] == {"estimate": 0.0, "ci95": [0.0, 0.0]}
        expected = np.mean([scores(labels, p)[metric] for p in predictions])
        assert result["baseline"][metric]["estimate"] == expected
    with pytest.raises(ValueError):
        paired_bootstrap(labels, predictions[:, :5], predictions, samples=10)
