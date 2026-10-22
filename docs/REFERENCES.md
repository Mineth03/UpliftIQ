# REFERENCES.md — Methods and Dataset Anchors

This is a working bibliography for implementation and thesis writing. Prefer primary sources when citing.

## Causal ML / EconML

- EconML documentation — CausalForestDML and inference APIs: https://www.pywhy.org/econml/
- Athey, S., & Wager, S. — Generalized Random Forests / causal forests (heterogeneous treatment effects).
- Chernozhukov et al. — Double/debiased machine learning (DML) for treatment effects.

## Uncertainty and decision making

- Mean–standard-error or lower-confidence-bound style decision rules (document exact score used: `τ̂ − λ·SE`).
- Decision-focused / prescriptive analytics literature connecting estimates to constrained optimization (cite carefully; we evaluate, we do not claim novelty of MILP).

## Optimization

- Google OR-Tools — linear/MIP solvers: https://developers.google.com/optimization

## Retail data

- Corporación Favorita Grocery Sales Forecasting (Kaggle) — observational candidate; treatment definition and overlap must be verified before causal claims.

## Software stack versions (record after install)

Record pinned versions from the active venv in `CURRENT_STATUS.md` after Milestone 1 install:

- Python, pandas, numpy, scikit-learn, econml, ortools, streamlit, plotly, pyarrow, pytest

## Integrity note

Do not copy substantial copyrighted text into the repo. Summarize and cite. Do not invent citations.
