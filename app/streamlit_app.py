"""CausaIQ Milestone 1 — local Streamlit dashboard.

Interactive knobs (budget, lambda, campaign limits) re-run MILP only.
They do not retrain CausalForestDML.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causaiq.causal.scoring import attach_risk_scores
from causaiq.data.synthetic import SyntheticRetailConfig
from causaiq.optimization.milp import OptimizationConstraints, select_promotions_milp
from causaiq.pipeline.demo import DemoConfig, run_milestone1_demo

ARTIFACT = ROOT / "data" / "artifacts" / "effect_table.parquet"


@st.cache_data(show_spinner="Fitting CausalForestDML on synthetic data (once)…")
def load_or_train_effect_table() -> pd.DataFrame:
    if ARTIFACT.exists():
        return pd.read_parquet(ARTIFACT)

    summary = run_milestone1_demo(
        DemoConfig(
            synthetic=SyntheticRetailConfig(
                n_stores=10,
                n_products=12,
                n_days=30,
                seed=42,
            ),
            artifact_dir=ROOT / "data" / "artifacts",
            n_jobs=2,
            lambda_risk=1.0,
        )
    )
    return summary["effect_table"]


def main() -> None:
    st.set_page_config(page_title="CausaIQ", layout="wide")
    st.title("CausaIQ")
    st.caption(
        "Uncertainty-aware causal-prescriptive promotion selection "
        "(synthetic demo — not real business outcomes)."
    )

    effect_table = load_or_train_effect_table()

    with st.sidebar:
        st.header("Decision knobs")
        st.markdown("Changing these **does not retrain** the causal model.")
        budget = st.slider("Budget", min_value=50.0, max_value=2000.0, value=800.0, step=25.0)
        inventory = st.slider(
            "Inventory capacity", min_value=20.0, max_value=800.0, value=250.0, step=10.0
        )
        max_promos = st.slider("Max promotions", min_value=1, max_value=40, value=12)
        lambda_risk = st.slider("Risk aversion λ", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    scored = attach_risk_scores(effect_table, lambda_risk=lambda_risk)
    constraints = OptimizationConstraints(
        budget=budget,
        inventory_capacity=inventory,
        max_promotions=max_promos,
    )
    result = select_promotions_milp(scored, constraints)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", result.status)
    c2.metric("Selected", result.n_selected)
    c3.metric("Total cost", f"{result.total_cost:.1f}")
    c4.metric("Objective (risk-adjusted)", f"{result.objective_value:.2f}")

    st.subheader("Selected promotions")
    if result.n_selected == 0:
        st.info("No promotions selected under current constraints.")
    else:
        show_cols = [
            c
            for c in [
                "candidate_id",
                "effect",
                "se",
                "score",
                "promo_cost",
                "inventory_use",
                "true_effect",
            ]
            if c in result.selected_table.columns
        ]
        st.dataframe(result.selected_table[show_cols], use_container_width=True)

    st.subheader("Candidate scores")
    plot_df = scored.copy()
    plot_df["selected"] = plot_df["candidate_id"].isin(result.selected_ids)
    fig = px.scatter(
        plot_df,
        x="se",
        y="effect",
        color="selected",
        hover_data=["candidate_id", "score", "promo_cost"],
        title="Estimated effect vs uncertainty (SE)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"Mean estimated effect: **{effect_table['effect'].mean():.3f}** · "
        f"Mean SE: **{effect_table['se'].mean():.3f}** · "
        f"Candidates: **{len(effect_table)}**"
    )
    st.warning(
        "Synthetic data only. Do not interpret these figures as real retail campaign results."
    )


if __name__ == "__main__":
    main()
