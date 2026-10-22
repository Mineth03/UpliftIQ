# AGENTS.md — CausaIQ Project Memory for AI Assistants

This file is the permanent operating guide for any AI model or IDE continuing **CausaIQ**.

## Project identity

- **Working name / package:** CausaIQ (`causaiq`)
- **Repository folder:** UpliftIQ (legacy/local folder name; do not rename unless asked)
- **Research title:** An Uncertainty-Aware Causal-Prescriptive Business Intelligence Framework for Resource-Constrained Retail Decision Making
- **Owner:** Final-year undergraduate AI and Data Science research project
- **Hardware target:** Windows laptop — Intel i9-13900H, RTX 4060, 16 GB RAM; Python 3.11

## Mandatory reading order (every new session)

1. `AGENTS.md` (this file)
2. `PROJECT_CONTEXT.md`
3. `CURRENT_STATUS.md`
4. `DECISIONS.md`
5. `plan/IMPLEMENTATION_PLAN.md`
6. `docs/REFERENCES.md` when methods or citations are needed

Do **not** ask the user to repeat information already documented in these files.

## Research problem (do not dilute)

Retail promotion selection from historical sales or forecasts does not identify whether a promotion **causes** incremental sales. Causal ML can estimate incremental effects, but those estimates are uncertain. Optimizing on point estimates alone can yield risky promotions under budget and inventory limits.

**Research question:** Does incorporating uncertainty in causal effect estimates improve constrained retail promotion decisions compared with conventional causal optimization?

We do **not** claim causal inference, uncertainty quantification, or MILP are novel methods. We experimentally evaluate their joint value for this retail decision problem.

## Architecture rules

1. Offline causal training is separate from interactive optimization. **Never retrain** when budget, lambda (risk), or campaign limits change.
2. Pipeline stages: data prep → CausalForestDML (EconML) → SE / CIs → risk-adjusted score `τ̂ − λ·SE` → MILP (OR-Tools) → BI dashboard (Streamlit).
3. Keep ML, optimization, and UI modular.
4. Prefer Parquet for processed data; pytest for tests.
5. Use ~2–4 CPU workers for memory-intensive jobs on this laptop.
6. No cloud auth, no complex React frontend in the initial research implementation.

## Data and evaluation rules

- Corporación Favorita is the **candidate** observational case study; verify treatment, confounders, overlap, and limitations before causal claims.
- Semi-synthetic / synthetic data with **known** treatment effects is required for proper experimental evaluation.
- Four comparison policies (same candidates and constraints where applicable):
  1. Historical sales-based selection
  2. Predictive ML-based selection
  3. Causal optimization **without** uncertainty adjustment
  4. Proposed uncertainty-aware causal optimization
- Metrics: policy value, regret, downside risk, CI coverage, decision stability, constraint satisfaction.
- **Never invent** experimental results or present simulated costs/inventory/outcomes as real business data.
- Do not mark unfinished work as completed. Do not fabricate model accuracy or causal findings.

## Documentation rules (after every meaningful step)

- Update `CURRENT_STATUS.md`: completed work, files touched, commands run, test results, blockers, next task.
- Update `DECISIONS.md` when technical or methodological choices change.
- Update `PROJECT_CONTEXT.md` when scope or architecture changes.
- Keep these files version-controlled so another assistant can continue without context loss.

## Implementation style

- Make small, testable changes.
- Explain important implementation decisions in `DECISIONS.md` and briefly in chat.
- Prefer verified runnable milestones over large unfinished scaffolds.
