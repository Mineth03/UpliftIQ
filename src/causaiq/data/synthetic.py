"""Reproducible synthetic retail panel with known promotional treatment effects.

This data is for pipeline verification and experimental evaluation only.
It must never be presented as real business sales, costs, or inventory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticRetailConfig:
    """Controls size and strength of the synthetic observational panel."""

    n_stores: int = 20
    n_products: int = 30
    n_days: int = 60
    treatment_rate: float = 0.25
    noise_scale: float = 3.0
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _true_cate(store_size: np.ndarray, elasticity: np.ndarray, seasonality: np.ndarray) -> np.ndarray:
    """Ground-truth incremental units from promotion (known only in synthetic data)."""
    return np.maximum(0.5, 2.0 + 1.5 * store_size + 2.0 * elasticity + 0.8 * seasonality)


def generate_synthetic_retail(config: SyntheticRetailConfig | None = None) -> pd.DataFrame:
    """Generate store–product–day rows with binary promotion treatment and known CATE.

    Columns
    -------
    store_id, product_id, day, store_size, elasticity, seasonality, price,
    lag_sales, treatment, outcome, true_cate, candidate_id, promo_cost,
    inventory_use
    """
    cfg = config or SyntheticRetailConfig()
    rng = np.random.default_rng(cfg.seed)

    stores = pd.DataFrame(
        {
            "store_id": np.arange(cfg.n_stores),
            "store_size": rng.uniform(0.0, 1.0, cfg.n_stores),
        }
    )
    products = pd.DataFrame(
        {
            "product_id": np.arange(cfg.n_products),
            "elasticity": rng.uniform(0.0, 1.0, cfg.n_products),
            "base_price": rng.uniform(5.0, 40.0, cfg.n_products),
            "promo_cost": rng.uniform(20.0, 120.0, cfg.n_products),
            "inventory_use": rng.uniform(5.0, 40.0, cfg.n_products),
        }
    )

    index = pd.MultiIndex.from_product(
        [stores["store_id"], products["product_id"], np.arange(cfg.n_days)],
        names=["store_id", "product_id", "day"],
    )
    df = index.to_frame(index=False)
    df = df.merge(stores, on="store_id", how="left")
    df = df.merge(products, on="product_id", how="left")

    # Seasonality in [-1, 1] peaking mid-week-ish pattern by day index
    df["seasonality"] = np.sin(2.0 * np.pi * df["day"].to_numpy() / 7.0)

    # Confounders that affect both treatment propensity and outcome
    propensity_logits = (
        -1.0
        + 1.2 * df["store_size"].to_numpy()
        + 0.8 * df["elasticity"].to_numpy()
        + 0.5 * df["seasonality"].to_numpy()
    )
    propensity = 1.0 / (1.0 + np.exp(-propensity_logits))
    # Rescale toward target treatment rate while keeping dependence on confounders
    propensity = propensity * (cfg.treatment_rate / float(np.mean(propensity)))
    propensity = np.clip(propensity, 0.02, 0.85)
    df["treatment"] = rng.binomial(1, propensity)

    df["true_cate"] = _true_cate(
        df["store_size"].to_numpy(),
        df["elasticity"].to_numpy(),
        df["seasonality"].to_numpy(),
    )

    baseline = (
        20.0
        + 8.0 * df["store_size"].to_numpy()
        + 5.0 * df["elasticity"].to_numpy()
        + 3.0 * df["seasonality"].to_numpy()
        - 0.15 * df["base_price"].to_numpy()
    )
    df["lag_sales"] = baseline + rng.normal(0.0, cfg.noise_scale, len(df))
    df["price"] = df["base_price"] * (1.0 - 0.15 * df["treatment"])

    noise = rng.normal(0.0, cfg.noise_scale, len(df))
    df["outcome"] = (
        baseline
        + 0.35 * df["lag_sales"].to_numpy()
        + df["treatment"].to_numpy() * df["true_cate"].to_numpy()
        + noise
    )
    df["outcome"] = np.maximum(0.0, df["outcome"])

    df["candidate_id"] = (
        "S" + df["store_id"].astype(str) + "_P" + df["product_id"].astype(str)
    )

    ordered = [
        "store_id",
        "product_id",
        "day",
        "candidate_id",
        "store_size",
        "elasticity",
        "seasonality",
        "price",
        "base_price",
        "lag_sales",
        "treatment",
        "outcome",
        "true_cate",
        "promo_cost",
        "inventory_use",
    ]
    return df[ordered].reset_index(drop=True)


def build_promotion_candidates(
    panel: pd.DataFrame,
    *,
    effect_col: str = "effect",
    se_col: str = "se",
    agg_true_cate: bool = True,
) -> pd.DataFrame:
    """Collapse panel to one optimization candidate per store–product.

    If ``effect_col`` / ``se_col`` are absent, placeholders are left for the
    estimator to fill. When ``agg_true_cate`` is True, mean ground-truth CATE
    is retained for evaluation only.
    """
    agg: dict[str, Any] = {
        "promo_cost": "mean",
        "inventory_use": "mean",
        "store_size": "mean",
        "elasticity": "mean",
        "store_id": "first",
        "product_id": "first",
    }
    if agg_true_cate and "true_cate" in panel.columns:
        agg["true_cate"] = "mean"

    candidates = (
        panel.groupby("candidate_id", as_index=False).agg(agg).rename(columns={"true_cate": "true_effect"})
        if "true_cate" in agg
        else panel.groupby("candidate_id", as_index=False).agg(agg)
    )

    if effect_col not in candidates.columns:
        candidates[effect_col] = np.nan
    if se_col not in candidates.columns:
        candidates[se_col] = np.nan
    return candidates
