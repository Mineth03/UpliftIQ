#!/usr/bin/env python
"""CLI entry for Milestone 1 demo (no Streamlit required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causaiq.data.synthetic import SyntheticRetailConfig
from causaiq.pipeline.demo import DemoConfig, run_milestone1_demo


def main() -> None:
    artifact_dir = ROOT / "data" / "artifacts"
    cfg = DemoConfig(
        synthetic=SyntheticRetailConfig(
            n_stores=12,
            n_products=15,
            n_days=40,
            seed=42,
        ),
        budget=800.0,
        inventory_capacity=250.0,
        max_promotions=12,
        lambda_risk=1.0,
        n_jobs=2,
        artifact_dir=artifact_dir,
    )
    summary = run_milestone1_demo(cfg)

    printable = {
        k: v
        for k, v in summary.items()
        if k
        not in {
            "panel",
            "effect_table",
            "scored",
            "result_risk",
            "result_causal",
        }
    }
    print(json.dumps(printable, indent=2))
    print(f"\nArtifacts written under: {artifact_dir}")


if __name__ == "__main__":
    main()
