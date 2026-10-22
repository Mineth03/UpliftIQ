"""Tests for uncertainty-adjusted benefit scores."""

import numpy as np
import pandas as pd
import pytest

from causaiq.causal.scoring import attach_risk_scores, risk_adjusted_score


def test_lambda_zero_equals_effect():
    effect = np.array([10.0, 5.0, -1.0])
    se = np.array([2.0, 1.0, 0.5])
    scores = risk_adjusted_score(effect, se, lambda_risk=0.0)
    np.testing.assert_allclose(scores, effect)


def test_lambda_penalizes_uncertainty():
    effect = np.array([10.0, 10.0])
    se = np.array([1.0, 4.0])
    scores = risk_adjusted_score(effect, se, lambda_risk=1.5)
    assert scores[0] > scores[1]
    np.testing.assert_allclose(scores, np.array([8.5, 4.0]))


def test_rejects_negative_lambda():
    with pytest.raises(ValueError, match="lambda_risk"):
        risk_adjusted_score(np.array([1.0]), np.array([0.1]), lambda_risk=-0.1)


def test_rejects_negative_se():
    with pytest.raises(ValueError, match="non-negative"):
        risk_adjusted_score(np.array([1.0]), np.array([-0.1]), lambda_risk=0.5)


def test_attach_risk_scores_column():
    df = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "effect": [3.0, 4.0],
            "se": [1.0, 0.5],
        }
    )
    out = attach_risk_scores(df, lambda_risk=2.0)
    np.testing.assert_allclose(out["score"].to_numpy(), np.array([1.0, 3.0]))
    # original unchanged
    assert "score" not in df.columns
