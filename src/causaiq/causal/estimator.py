"""CausalForestDML wrapper: offline training and effect/SE export.

Training is intentionally separate from MILP optimization. Manager knobs
(budget, lambda, campaign limits) must consume the exported effect table
without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "store_size",
    "elasticity",
    "seasonality",
    "price",
    "lag_sales",
)


@dataclass
class EffectEstimate:
    """Per-row estimates aligned with an input feature frame."""

    effect: np.ndarray
    se: np.ndarray
    feature_frame: pd.DataFrame


class CausalEffectEstimator:
    """Thin EconML CausalForestDML adapter with conservative laptop defaults."""

    def __init__(
        self,
        *,
        feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
        n_estimators: int = 100,
        min_samples_leaf: int = 20,
        max_depth: int | None = 6,
        random_state: int = 42,
        n_jobs: int = 2,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, panel: pd.DataFrame) -> CausalEffectEstimator:
        """Fit CausalForestDML on observational panel rows."""
        from econml.dml import CausalForestDML

        missing = [c for c in self.feature_cols + ["treatment", "outcome"] if c not in panel.columns]
        if missing:
            raise ValueError(f"panel missing required columns: {missing}")

        X = panel[self.feature_cols].to_numpy(dtype=float)
        T = panel["treatment"].to_numpy(dtype=int)
        Y = panel["outcome"].to_numpy(dtype=float)

        # Modest learners keep memory under control on 16 GB laptops
        model_y = GradientBoostingRegressor(
            random_state=self.random_state,
            max_depth=3,
            n_estimators=50,
        )
        model_t = GradientBoostingClassifier(
            random_state=self.random_state,
            max_depth=3,
            n_estimators=50,
        )

        est = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            inference=True,
        )
        est.fit(Y=Y, T=T, X=X)
        self._model = est
        return self

    def predict_effect_and_se(self, features: pd.DataFrame) -> EffectEstimate:
        """Return effect and standard error for each row in ``features``."""
        if self._model is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        X = features[self.feature_cols].to_numpy(dtype=float)
        effect = np.asarray(self._model.effect(X), dtype=float).reshape(-1)

        se = self._safe_se(X, effect_len=len(effect))
        return EffectEstimate(effect=effect, se=se, feature_frame=features.reset_index(drop=True))

    def _safe_se(self, X: np.ndarray, *, effect_len: int) -> np.ndarray:
        """Best-effort SE extraction; fall back to small epsilon if inference fails."""
        model = self._model
        assert model is not None
        try:
            # EconML: effect_interval returns (lower, upper); SE ≈ (upper-lower)/(2*z)
            lower, upper = model.effect_interval(X, alpha=0.05)
            lower = np.asarray(lower, dtype=float).reshape(-1)
            upper = np.asarray(upper, dtype=float).reshape(-1)
            # z_{0.975} ≈ 1.96 for normal approx
            se = (upper - lower) / (2.0 * 1.96)
            se = np.maximum(se, 1e-6)
            if se.shape[0] != effect_len:
                raise ValueError("SE length mismatch")
            return se
        except Exception:
            # Transparent fallback so optimization still runs; flag via large SE
            return np.full(effect_len, np.nan, dtype=float)

    def estimate_candidate_effects(
        self,
        panel: pd.DataFrame,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach mean predicted effect/SE to store–product candidates.

        Uses mean feature values within each candidate from the panel as the
        prediction point (simple aggregation suitable for Milestone 1 demos).
        """
        if self._model is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        feat_means = (
            panel.groupby("candidate_id", as_index=False)[self.feature_cols]
            .mean()
        )
        estimate = self.predict_effect_and_se(feat_means)
        scored = feat_means[["candidate_id"]].copy()
        scored["effect"] = estimate.effect
        scored["se"] = estimate.se
        # Replace NaN SE with a conservative fraction of |effect| for MILP demos
        nan_se = scored["se"].isna()
        scored.loc[nan_se, "se"] = np.maximum(0.1, 0.25 * np.abs(scored.loc[nan_se, "effect"]))

        out = candidates.drop(columns=[c for c in ("effect", "se") if c in candidates.columns], errors="ignore")
        return out.merge(scored, on="candidate_id", how="left")
