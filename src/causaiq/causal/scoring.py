"""Risk-adjusted benefit scores from causal effect estimates.

score = estimated_causal_effect - lambda * standard_error

lambda = 0 recovers causal-only optimization for experimental comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_adjusted_score(
    effect: np.ndarray | pd.Series | float,
    se: np.ndarray | pd.Series | float,
    lambda_risk: float,
) -> np.ndarray:
    """Compute uncertainty-penalized scores.

    Parameters
    ----------
    effect :
        Estimated causal effects (e.g. incremental units or revenue).
    se :
        Standard errors of those estimates (non-negative).
    lambda_risk :
        Risk aversion. Must be >= 0. Higher values penalize uncertain effects more.
    """
    if lambda_risk < 0:
        raise ValueError(f"lambda_risk must be >= 0, got {lambda_risk}")

    effect_arr = np.asarray(effect, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    if effect_arr.shape != se_arr.shape:
        raise ValueError(
            f"effect and se shape mismatch: {effect_arr.shape} vs {se_arr.shape}"
        )
    if np.any(se_arr < 0):
        raise ValueError("standard errors must be non-negative")

    return effect_arr - float(lambda_risk) * se_arr


def attach_risk_scores(
    candidates: pd.DataFrame,
    *,
    lambda_risk: float,
    effect_col: str = "effect",
    se_col: str = "se",
    score_col: str = "score",
) -> pd.DataFrame:
    """Return a copy of candidates with a risk-adjusted score column."""
    out = candidates.copy()
    out[score_col] = risk_adjusted_score(
        out[effect_col].to_numpy(),
        out[se_col].to_numpy(),
        lambda_risk,
    )
    return out
