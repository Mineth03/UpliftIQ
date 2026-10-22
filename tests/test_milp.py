"""Tests for MILP promotion selection."""

import pandas as pd
import pytest

from causaiq.optimization.milp import OptimizationConstraints, select_promotions_milp


def _toy_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C", "D"],
            "score": [10.0, 9.0, 8.0, 1.0],
            "promo_cost": [50.0, 50.0, 40.0, 10.0],
            "inventory_use": [30.0, 25.0, 20.0, 5.0],
        }
    )


def test_respects_budget_and_inventory_and_cardinality():
    candidates = _toy_candidates()
    constraints = OptimizationConstraints(budget=100.0, inventory_capacity=50.0, max_promotions=2)
    result = select_promotions_milp(candidates, constraints)

    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.n_selected <= 2
    assert result.total_cost <= 100.0 + 1e-6
    assert result.total_inventory <= 50.0 + 1e-6
    # Best feasible pair under inventory 50: B+C (scores 9+8, inv 25+20=45, cost 90)
    assert set(result.selected_ids) == {"B", "C"}


def test_max_promotions_zero_selects_none():
    result = select_promotions_milp(
        _toy_candidates(),
        OptimizationConstraints(budget=1000.0, inventory_capacity=1000.0, max_promotions=0),
    )
    assert result.n_selected == 0
    assert result.selected_ids == []


def test_infeasible_tiny_budget():
    # Positive costs but zero budget → no promotion affordable; still OPTIMAL with empty set
    result = select_promotions_milp(
        _toy_candidates(),
        OptimizationConstraints(budget=0.0, inventory_capacity=1000.0, max_promotions=5),
    )
    assert result.n_selected == 0
    assert result.status in {"OPTIMAL", "FEASIBLE"}


def test_rejects_negative_constraints():
    with pytest.raises(ValueError):
        OptimizationConstraints(budget=-1.0, inventory_capacity=1.0, max_promotions=1)
