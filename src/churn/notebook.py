"""Shared imports and helpers for the notebooks, so each notebook's setup cell is one line."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import warnings

import lightgbm  # noqa: F401  (import before scikit-learn: the reverse order crashes LightGBM on Windows)
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from churn.paths import P, ROOT

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)

__all__ = ["np", "pd", "display", "Markdown", "Image", "fig", "tab", "run_step", "P", "ROOT"]


def fig(name: str, width: int = 900) -> None:
    """Show a figure saved by the pipeline (outputs/figures/<name>.png)."""
    display(Image(filename=str(P.figures / f"{name}.png"), width=width))


def tab(name: str) -> pd.DataFrame:
    """Load a table saved by the pipeline (outputs/tables/<name>.csv)."""
    return pd.read_csv(P.tables / f"{name}.csv")


def run_step(module: str, *args: str, tail: int = 5) -> None:
    """Run one pipeline step exactly as run_pipeline.py does and print its last log messages."""
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).replace(str(ROOT), ".").splitlines()
    msgs = [ln for ln in out if re.match(r"\d\d:\d\d:\d\d \w+ ", ln) and ln.split(": ", 1)[-1].strip()]
    if r.returncode != 0:
        msgs = out
    if msgs:
        print("\n".join(msgs[-tail:]))
    written = sorted(p.relative_to(ROOT).as_posix() for d in (P.tables, P.figures, P.predictions) for p in d.rglob("*")
                     if p.is_file() and p.suffix in {".csv", ".png", ".json"} and p.stat().st_mtime >= t0 - 1)
    if written:
        print(f"files regenerated ({len(written)}): " + ", ".join(written[:8]) + (" ..." if len(written) > 8 else ""))
    print(f"--> python -m {' '.join([module, *args])}: finished in {time.time() - t0:.1f}s (exit code {r.returncode})")
