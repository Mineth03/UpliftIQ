"""Tests for synthetic retail data generation."""

from causaiq.data.synthetic import (
    SyntheticRetailConfig,
    build_promotion_candidates,
    generate_synthetic_retail,
)


def test_reproducible_with_seed():
    cfg = SyntheticRetailConfig(n_stores=5, n_products=4, n_days=10, seed=123)
    a = generate_synthetic_retail(cfg)
    b = generate_synthetic_retail(cfg)
    assert a.equals(b)


def test_panel_shape_and_columns():
    cfg = SyntheticRetailConfig(n_stores=3, n_products=2, n_days=5, seed=1)
    df = generate_synthetic_retail(cfg)
    assert len(df) == 3 * 2 * 5
    for col in (
        "treatment",
        "outcome",
        "true_cate",
        "promo_cost",
        "inventory_use",
        "candidate_id",
    ):
        assert col in df.columns
    assert set(df["treatment"].unique()).issubset({0, 1})
    assert (df["true_cate"] > 0).all()


def test_different_seeds_differ():
    a = generate_synthetic_retail(SyntheticRetailConfig(n_stores=4, n_products=3, n_days=8, seed=1))
    b = generate_synthetic_retail(SyntheticRetailConfig(n_stores=4, n_products=3, n_days=8, seed=2))
    assert not a["outcome"].equals(b["outcome"])


def test_build_candidates():
    panel = generate_synthetic_retail(
        SyntheticRetailConfig(n_stores=2, n_products=3, n_days=5, seed=7)
    )
    cands = build_promotion_candidates(panel)
    assert len(cands) == 2 * 3
    assert "true_effect" in cands.columns
    assert cands["candidate_id"].is_unique
