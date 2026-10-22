"""Smoke tests for CausalForestDML wrapper on tiny synthetic data."""

import numpy as np

from causaiq.causal.estimator import CausalEffectEstimator
from causaiq.data.synthetic import (
    SyntheticRetailConfig,
    build_promotion_candidates,
    generate_synthetic_retail,
)


def test_fit_and_candidate_effects_shapes():
    panel = generate_synthetic_retail(
        SyntheticRetailConfig(n_stores=6, n_products=5, n_days=20, seed=42, noise_scale=2.0)
    )
    candidates = build_promotion_candidates(panel)
    est = CausalEffectEstimator(
        n_estimators=40,
        min_samples_leaf=15,
        max_depth=4,
        n_jobs=2,
        random_state=42,
    )
    est.fit(panel)
    out = est.estimate_candidate_effects(panel, candidates)

    assert len(out) == len(candidates)
    assert out["effect"].notna().all()
    assert out["se"].notna().all()
    assert (out["se"] >= 0).all()
    # Effects should have some dispersion (not a constant collapse)
    assert float(np.std(out["effect"])) > 0.0
