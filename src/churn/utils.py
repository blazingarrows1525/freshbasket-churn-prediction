"""Small shared helpers: cached Excel reads, table export, logging, schema profiling and the data-quality log."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------- IO
def read_excel_sheet_cached(xlsx: Path, sheet: str, cache: Path, header_row: int = 1) -> pd.DataFrame:
    """Read one worksheet, caching it as parquet (the cache is invalidated when the workbook changes).

    The workbook puts a title in row 1 and the column header in row 2 (header_row=1). Fully empty
    columns (formatting artefacts) are dropped.
    """
    if not xlsx.exists():
        raise FileNotFoundError(f"{xlsx} not found - copy FreshBasket_Loyalty_Churn_Dataset.xlsx into data/raw/")
    stamp = cache.with_suffix(".stamp.json")
    src_meta = {"size": xlsx.stat().st_size, "mtime": xlsx.stat().st_mtime, "sheet": sheet}
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text()) == src_meta:
        return pd.read_parquet(cache)
    log.info("Parsing %s :: %s (cached afterwards)", xlsx.name, sheet)
    df = pd.read_excel(xlsx, sheet_name=sheet, header=header_row)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    stamp.write_text(json.dumps(src_meta))
    return df


def write_table(df: pd.DataFrame, path: Path, index: bool = False, floatfmt: str = ".3f", md: bool = True) -> Path:
    """Write `df` to `<path>.csv` and (optionally) `<path>.md`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=index)
    if md:
        path.with_suffix(".md").write_text(df.to_markdown(index=index, floatfmt=floatfmt), encoding="utf-8")
    return path.with_suffix(".csv")


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


# ----------------------------------------------------------------------------------------------- profiling
def profile_frame(df: pd.DataFrame, table: str, max_examples: int = 3) -> pd.DataFrame:
    """One row per column: dtype, missing %, distinct count, min/max, example values."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        entry = {"table": table, "column": col, "dtype": str(s.dtype), "rows": n,
                 "missing_pct": round(100 * (1 - len(non_null) / n), 2) if n else np.nan,
                 "n_unique": int(non_null.nunique()), "min": None, "max": None,
                 "examples": ", ".join(map(str, non_null.drop_duplicates().head(max_examples).tolist()))}
        if (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)) and len(non_null):
            entry["min"], entry["max"] = non_null.min(), non_null.max()
        rows.append(entry)
    out = pd.DataFrame(rows)
    for c in ("min", "max"):
        out[c] = out[c].map(lambda v: v.date() if isinstance(v, pd.Timestamp) and v == v.normalize() else v)
    return out


def table_summary(df: pd.DataFrame, table: str, key: list[str] | None = None, date_col: str | None = None) -> dict:
    """Headline facts for a table: shape, duplicates, key uniqueness, date range."""
    out = {"table": table, "rows": len(df), "columns": df.shape[1], "exact_duplicate_rows": int(df.duplicated().sum())}
    if key:
        out["candidate_key"] = " + ".join(key)
        out["duplicate_keys"] = int(df.duplicated(key).sum())
    if date_col:
        out["date_column"] = date_col
        out["date_min"] = pd.Timestamp(df[date_col].min()).date()
        out["date_max"] = pd.Timestamp(df[date_col].max()).date()
    return out


SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}


@dataclass
class QualityLog:
    """Collects data-quality findings as they are detected: issue, evidence, severity, treatment, reason."""
    findings: list[dict] = field(default_factory=list)

    def add(self, issue: str, evidence: str, severity: str, treatment: str, reason: str, table: str = "") -> None:
        assert severity in SEVERITY_ORDER, severity
        self.findings.append({"table": table, "issue": issue, "evidence": evidence, "severity": severity,
                              "treatment": treatment, "reason": reason})

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.findings, columns=["table", "issue", "evidence", "severity", "treatment", "reason"])
        if df.empty:
            return df
        df["_o"] = df.severity.map(SEVERITY_ORDER)
        return df.sort_values(["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
