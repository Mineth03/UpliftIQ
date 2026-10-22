# CausaIQ

**Research title:** An Uncertainty-Aware Causal-Prescriptive Business Intelligence Framework for Resource-Constrained Retail Decision Making

Local research codebase for estimating heterogeneous promotional effects (EconML `CausalForestDML`), adjusting for estimate uncertainty, and selecting promotions with MILP under budget, inventory, and campaign limits.

> Synthetic demo outputs are **not** real business results.

## Quick start (Windows / Python 3.11)

```powershell
cd C:\Users\asus\OneDrive\Desktop\UpliftIQ
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
python scripts\run_demo.py
streamlit run app\streamlit_app.py
```

## Project memory (read first)

- `AGENTS.md`
- `PROJECT_CONTEXT.md`
- `CURRENT_STATUS.md`
- `DECISIONS.md`
- `plan/IMPLEMENTATION_PLAN.md`

## Layout

```
src/causaiq/     # data, causal, optimization, pipeline
app/             # Streamlit BI prototype
tests/           # pytest
scripts/         # CLI demos
data/            # raw / processed / synthetic / artifacts (gitignored contents)
```
