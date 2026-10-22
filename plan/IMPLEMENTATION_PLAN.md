# IMPLEMENTATION_PLAN.md

## Milestone 1 — Verified local research core (CURRENT)

**Goal:** Runnable, tested pipeline on synthetic data: generate → estimate → score → optimize → demo UI.

| Step | Deliverable | Done when |
|------|-------------|-----------|
| 1.1 | Project memory + package layout + venv | Imports work; docs present |
| 1.2 | Synthetic retail generator | Seeded runs reproducible; ground-truth τ column present |
| 1.3 | Risk scoring | Unit tests for λ=0 and λ>0 |
| 1.4 | MILP selector | Feasible under budget/inventory/max campaigns; tests |
| 1.5 | CausalForestDML wrapper | Trains on synthetic; exports effect + SE |
| 1.6 | Demo script | Prints selection summary without Streamlit |
| 1.7 | Streamlit skeleton | Loads effect table; sliders for budget/λ; shows selection |
| 1.8 | Pytest suite green | Record real output in `CURRENT_STATUS.md` |

## Milestone 2 — Semi-synthetic evaluation harness

- Inject known effects into Favorita-like or synthetic covariates
- Implement four comparison policies with shared constraints
- Metrics: policy value, regret, downside risk, CI coverage, stability, constraint violation rate
- Document that results are experimental, not live retail KPIs

## Milestone 3 — Favorita observational case study (conditional)

- Acquire / document dataset paths (Kaggle Corporación Favorita)
- Define treatment, outcome, timing, confounders
- Overlap / positivity diagnostics; limitation write-up
- Only then report observational estimates with explicit caveats

## Milestone 4 — BI polish and thesis artifacts

- Scenario comparison views (λ=0 vs λ>0)
- Exportable recommendation tables
- Experiment notebooks + figure generation for the dissertation
- Reproducibility checklist (seeds, versions, hardware)

## Milestone 5 — Robustness and stretch (optional)

- Alternative estimators / scoring rules
- Larger candidate sets / solve-time profiling
- GPU use only if clearly beneficial (not required for CausalForestDML)

## Working agreements

- Prefer small PRs / commits when the user requests them
- Update `CURRENT_STATUS.md` after each meaningful step
- Never mark unfinished features complete
- Never invent evaluation numbers
