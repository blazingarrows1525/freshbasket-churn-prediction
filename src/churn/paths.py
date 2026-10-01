"""Project paths, resolved from the repository root so the code runs from any working directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw: Path = ROOT / "data" / "raw"                       # the supplied workbook (not committed)
    interim: Path = ROOT / "data" / "interim"               # parquet caches of the raw sheets
    processed: Path = ROOT / "data" / "processed"           # cleaned tables + customer-level modelling dataset
    figures: Path = ROOT / "outputs" / "figures"
    tables: Path = ROOT / "outputs" / "tables"              # every result table, CSV + Markdown
    predictions: Path = ROOT / "outputs" / "predictions"    # churn prediction / scoring / scenario files
    models: Path = ROOT / "outputs" / "models"
    reports: Path = ROOT / "reports"

    def ensure(self) -> "Paths":
        for p in (self.interim, self.processed, self.figures, self.tables, self.predictions, self.models):
            p.mkdir(parents=True, exist_ok=True)
        return self


P = Paths().ensure()
