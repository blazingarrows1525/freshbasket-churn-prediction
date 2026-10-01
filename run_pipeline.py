"""Run the whole project end to end: raw workbook -> cleaned data -> features -> models -> reports.

Each step runs in its own process, in this order:
    data_validation      Phase-0 inventory of the raw workbook
    preprocessing        data-quality checks, cleaning, member x month grid
    feature_engineering  snapshots, modelling dataset, leakage test, feature definitions
    eda                  exploratory figures and tables
    modeling             baseline + candidate models, out-of-time validation, stability checks
    final_model          selected model, threshold, calibration, gains, prediction files
    explainability       global and local drivers, risk by segment
    ablation             feature-group ablation
    scenarios            retention what-ifs with sensitivity
    summaries            feature catalogue, experiment log, model-selection summary

Usage:
    python run_pipeline.py                 # pipeline + notebooks + deck  (~3 minutes)
    python run_pipeline.py --no-notebooks  # pipeline only                (~1.5 minutes)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["data_validation", "preprocessing", "feature_engineering", "eda", "modeling", "final_model",
         "explainability", "ablation", "scenarios", "summaries"]


def run(cmd: list[str]) -> None:
    t0 = time.time()
    print(f"\n>>> {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"step failed: {' '.join(cmd)}")
    print(f"    done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-notebooks", action="store_true", help="skip executing the notebooks and rebuilding the deck")
    a = ap.parse_args()
    t0 = time.time()
    for step in STEPS:
        run([sys.executable, "-m", f"churn.{step}"])
    if not a.no_notebooks:
        run([sys.executable, "notebooks/build_notebooks.py"])
        run([sys.executable, "presentation/build_deck.py"])
    print(f"\nall done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
