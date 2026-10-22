# DECISIONS.md — Technical and Methodological Log

Format: date — decision — rationale — alternatives considered.

---

## 2026-10-22 — Package name `causaiq` inside folder `UpliftIQ`

- **Decision:** Python import package is `causaiq`; keep local/repo folder name `UpliftIQ` for now.
- **Rationale:** Research branding is CausaIQ; renaming the GitHub remote/folder is out of scope for Milestone 1.
- **Alternatives:** Rename repo to CausaIQ immediately (deferred to user preference).

## 2026-10-22 — Synthetic-first development before Favorita claims

- **Decision:** Milestone 1 uses fully synthetic retail panel data with known ground-truth CATE.
- **Rationale:** Enables correctness tests for scoring, MILP, and estimator plumbing without false causal claims.
- **Alternatives:** Jump straight to Favorita (rejected: treatment/overlap not verified).

## 2026-10-22 — Risk score `score = effect - lambda * se`

- **Decision:** Linear mean–standard-error penalty with manager-tunable `lambda >= 0`.
- **Rationale:** Transparent, matches research framing; `lambda = 0` recovers causal-only optimization for fair comparison.
- **Alternatives:** CVaR / lower confidence bound only / utility of full posterior (deferred).

## 2026-10-22 — Offline effect table, online MILP

- **Decision:** Train CausalForestDML once; export candidate-level `effect`, `se`, costs, inventory use; optimize from that table.
- **Rationale:** Manager knobs must not trigger retraining; matches BI interactivity needs on a 16 GB laptop.
- **Alternatives:** Retrain on every slider change (rejected).

## 2026-10-22 — OR-Tools CBC for MILP

- **Decision:** Use `ortools.linear_solver` with CBC for binary promotion selection.
- **Rationale:** Free, adequate for small–medium candidate sets in research demos.
- **Alternatives:** PuLP+CBC, Gurobi (license), greedy heuristic (kept as optional baseline later).

## 2026-10-22 — Streamlit for initial BI

- **Decision:** Streamlit + Plotly for the first dashboard.
- **Rationale:** Fast research UI; React deferred per project scope.
- **Alternatives:** Dash, Gradio, full React SPA (out of scope for Milestone 1).

## 2026-10-22 — EconML CausalForestDML as primary estimator

- **Decision:** Primary CATE estimator is EconML `CausalForestDML` with inference enabled when available.
- **Rationale:** Specified in research design; supports heterogeneous effects and uncertainty export.
- **Alternatives:** Meta-learners only, DoWhy, custom forests (may add as robustness checks later).

## 2026-10-22 — Candidate unit = store–product promotion opportunity

- **Decision:** Optimization variables are binary flags on discrete promotion candidates (e.g., store–SKU or product group), each with cost, inventory impact, estimated effect, and SE.
- **Rationale:** Maps cleanly to MILP and BI recommendations.
- **Alternatives:** Continuous discount depth optimization (deferred).
