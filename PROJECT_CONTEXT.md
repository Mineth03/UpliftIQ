# PROJECT_CONTEXT.md — CausaIQ Scope and Architecture

## Research title

An Uncertainty-Aware Causal-Prescriptive Business Intelligence Framework for Resource-Constrained Retail Decision Making

## One-sentence goal

Estimate heterogeneous promotional causal effects with uncertainty, score candidates with a risk penalty, and select feasible promotions under budget, inventory, and campaign limits — then evaluate whether uncertainty-aware selection outperforms causal-only optimization.

## In scope (current)

- Synthetic (and later semi-synthetic) retail data with known ground-truth treatment effects
- Observational Favorita pipeline **after** treatment/confounder/overlap verification (not yet started)
- CausalForestDML (EconML) for CATE estimation with inference (SE / CIs)
- Risk-adjusted scores: `score = estimated_effect - lambda * standard_error`
- MILP promotion selection (OR-Tools) under budget, inventory capacity, and max-campaign constraints
- Offline train / online optimize separation (artifact-based effect table)
- Streamlit + Plotly BI prototype for recommendations and scenario comparison
- Automated pytest coverage for scoring, MILP, and data generation
- Four-policy experimental comparison design (implementation progressive)

## Out of scope (initial research implementation)

- Cloud deployment, authentication, multi-tenant SaaS
- Complex React / Next.js frontend
- Claiming causal identification on Favorita without documented assumptions checks
- Fabricated “business KPI” results from synthetic runs presented as real retail outcomes

## Core modules (target package: `causaiq`)

| Module | Responsibility |
|--------|----------------|
| `causaiq.data` | Synthetic / later Favorita prep; Parquet I/O |
| `causaiq.causal` | CausalForestDML wrapper, inference export, risk scoring |
| `causaiq.optimization` | MILP selection under constraints |
| `causaiq.pipeline` | Offline train + online optimize orchestration |
| `app/` | Streamlit dashboard |

## Decision interface (manager knobs)

Changed without retraining:

- Promotional budget
- Risk aversion `lambda`
- Max number of simultaneous promotions
- Inventory / capacity limits (as encoded in candidate table)

## Data strategy

1. **Synthetic first** — known τ for unit tests and pipeline verification.
2. **Semi-synthetic** — Favorita-like covariates with injected known effects for evaluation.
3. **Observational Favorita** — case study only after treatment definition and overlap diagnostics are documented.

## Evaluation policy set

1. Historical sales-based selection  
2. Predictive ML-based selection  
3. Causal optimization (λ = 0)  
4. Uncertainty-aware causal optimization (λ > 0)  

Policies 3 and 4 must share identical candidates and hard constraints.

## Tech stack (locked for Milestone 1)

- Python 3.11
- pandas, numpy, scikit-learn
- econml (CausalForestDML)
- ortools (MILP)
- streamlit, plotly
- pytest, pyarrow (Parquet)
- Local venv; no Docker required for Milestone 1
