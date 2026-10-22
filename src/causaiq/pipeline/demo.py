"""End-to-end Milestone 1 demonstration on synthetic data.

Offline: generate panel → fit CausalForestDML → export candidate effect table.
Online: attach risk scores → MILP select (no retraining).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from causaiq.causal.estimator import CausalEffectEstimator
from causaiq.causal.scoring import attach_risk_scores
from causaiq.data.synthetic import (
    SyntheticRetailConfig,
    build_promotion_candidates,
    generate_synthetic_retail,
)
from causaiq.optimization.milp import OptimizationConstraints, select_promotions_milp


@dataclass
class DemoConfig:
    synthetic: SyntheticRetailConfig
    budget: float = 800.0
    inventory_capacity: float = 250.0
    max_promotions: int = 12
    lambda_risk: float = 1.0
    n_jobs: int = 2
    artifact_dir: Path | None = None


def run_milestone1_demo(config: DemoConfig | None = None) -> dict[str, Any]:
    """Run generate → estimate → score → optimize; optionally write Parquet artifacts."""
    cfg = config or DemoConfig(synthetic=SyntheticRetailConfig())

    panel = generate_synthetic_retail(cfg.synthetic)
    candidates = build_promotion_candidates(panel)

    estimator = CausalEffectEstimator(
        random_state=cfg.synthetic.seed,
        n_jobs=cfg.n_jobs,
        n_estimators=80,
        min_samples_leaf=25,
    )
    estimator.fit(panel)
    effect_table = estimator.estimate_candidate_effects(panel, candidates)

    scored = attach_risk_scores(effect_table, lambda_risk=cfg.lambda_risk)
    scored_causal_only = attach_risk_scores(effect_table, lambda_risk=0.0)

    constraints = OptimizationConstraints(
        budget=cfg.budget,
        inventory_capacity=cfg.inventory_capacity,
        max_promotions=cfg.max_promotions,
    )
    result_risk = select_promotions_milp(scored, constraints)
    result_causal = select_promotions_milp(scored_causal_only, constraints)

    if cfg.artifact_dir is not None:
        art = Path(cfg.artifact_dir)
        art.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(art / "synthetic_panel.parquet", index=False)
        effect_table.to_parquet(art / "effect_table.parquet", index=False)
        scored.to_parquet(art / "scored_candidates.parquet", index=False)
        result_risk.selected_table.to_parquet(art / "selected_risk_aware.parquet", index=False)
        result_causal.selected_table.to_parquet(art / "selected_causal_only.parquet", index=False)

    # Evaluation aids (synthetic only — known true effects)
    def true_value(selected_ids: list[str]) -> float:
        if not selected_ids:
            return 0.0
        sub = effect_table.set_index("candidate_id").loc[selected_ids]
        if "true_effect" not in sub.columns:
            return float("nan")
        return float(sub["true_effect"].sum())

    summary = {
        "n_panel_rows": int(len(panel)),
        "n_candidates": int(len(effect_table)),
        "lambda_risk": cfg.lambda_risk,
        "constraints": {
            "budget": cfg.budget,
            "inventory_capacity": cfg.inventory_capacity,
            "max_promotions": cfg.max_promotions,
        },
        "effect_stats": {
            "mean_effect": float(effect_table["effect"].mean()),
            "mean_se": float(effect_table["se"].mean()),
            "mean_true_effect": float(effect_table["true_effect"].mean())
            if "true_effect" in effect_table.columns
            else None,
        },
        "risk_aware": result_risk.to_summary()
        | {"true_effect_sum": true_value(result_risk.selected_ids)},
        "causal_only": result_causal.to_summary()
        | {"true_effect_sum": true_value(result_causal.selected_ids)},
        "panel": panel,
        "effect_table": effect_table,
        "scored": scored,
        "result_risk": result_risk,
        "result_causal": result_causal,
    }
    return summary


def effect_table_from_artifacts(artifact_dir: Path) -> pd.DataFrame:
    """Load a previously exported effect table for interactive optimization."""
    path = Path(artifact_dir) / "effect_table.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing effect table at {path}")
    return pd.read_parquet(path)
