"""MILP promotion selection under budget, inventory, and campaign limits.

Maximizes sum(score_i * x_i) subject to:
  - sum(cost_i * x_i) <= budget
  - sum(inventory_use_i * x_i) <= inventory_capacity
  - sum(x_i) <= max_promotions
  - x_i binary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from ortools.linear_solver import pywraplp


@dataclass(frozen=True)
class OptimizationConstraints:
    budget: float
    inventory_capacity: float
    max_promotions: int

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError("budget must be >= 0")
        if self.inventory_capacity < 0:
            raise ValueError("inventory_capacity must be >= 0")
        if self.max_promotions < 0:
            raise ValueError("max_promotions must be >= 0")


@dataclass
class OptimizationResult:
    selected_ids: list[str]
    objective_value: float
    total_cost: float
    total_inventory: float
    n_selected: int
    status: str
    selected_table: pd.DataFrame

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_selected": self.n_selected,
            "objective_value": self.objective_value,
            "total_cost": self.total_cost,
            "total_inventory": self.total_inventory,
            "selected_ids": list(self.selected_ids),
        }


def select_promotions_milp(
    candidates: pd.DataFrame,
    constraints: OptimizationConstraints,
    *,
    id_col: str = "candidate_id",
    score_col: str = "score",
    cost_col: str = "promo_cost",
    inventory_col: str = "inventory_use",
    time_limit_seconds: float = 30.0,
) -> OptimizationResult:
    """Solve binary promotion selection with CBC via OR-Tools."""
    required = [id_col, score_col, cost_col, inventory_col]
    missing = [c for c in required if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidates missing columns: {missing}")
    if candidates.empty:
        raise ValueError("candidates frame is empty")

    frame = candidates.reset_index(drop=True)
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver could not be created")

    solver.SetTimeLimit(int(time_limit_seconds * 1000))

    x = [solver.BoolVar(f"x_{i}") for i in range(len(frame))]

    solver.Maximize(
        solver.Sum(float(frame.at[i, score_col]) * x[i] for i in range(len(frame)))
    )
    solver.Add(
        solver.Sum(float(frame.at[i, cost_col]) * x[i] for i in range(len(frame)))
        <= float(constraints.budget)
    )
    solver.Add(
        solver.Sum(float(frame.at[i, inventory_col]) * x[i] for i in range(len(frame)))
        <= float(constraints.inventory_capacity)
    )
    solver.Add(solver.Sum(x) <= int(constraints.max_promotions))

    status_code = solver.Solve()
    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status = status_map.get(status_code, f"UNKNOWN_{status_code}")

    if status_code not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        empty = frame.iloc[0:0].copy()
        return OptimizationResult(
            selected_ids=[],
            objective_value=float("nan"),
            total_cost=0.0,
            total_inventory=0.0,
            n_selected=0,
            status=status,
            selected_table=empty,
        )

    chosen_mask = [var.solution_value() > 0.5 for var in x]
    selected = frame.loc[chosen_mask].copy()
    selected_ids = selected[id_col].astype(str).tolist()
    total_cost = float(selected[cost_col].sum()) if len(selected) else 0.0
    total_inv = float(selected[inventory_col].sum()) if len(selected) else 0.0
    objective = float(solver.Objective().Value())

    return OptimizationResult(
        selected_ids=selected_ids,
        objective_value=objective,
        total_cost=total_cost,
        total_inventory=total_inv,
        n_selected=len(selected_ids),
        status=status,
        selected_table=selected.reset_index(drop=True),
    )
