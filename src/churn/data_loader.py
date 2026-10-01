"""Raw loaders for the FreshBasket loyalty workbook (each sheet is cached to parquet on first read)."""
from __future__ import annotations

import pandas as pd

from churn.paths import P
from churn.utils import read_excel_sheet_cached

XLSX = P.raw / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
SHEETS = {"profile": "fb Customer Profile", "activity": "fb Monthly Activity", "labels": "fb Churn Label"}


def load_profile() -> pd.DataFrame:
    return read_excel_sheet_cached(XLSX, SHEETS["profile"], P.interim / "raw_profile.parquet")


def load_activity() -> pd.DataFrame:
    return read_excel_sheet_cached(XLSX, SHEETS["activity"], P.interim / "raw_activity.parquet")


def load_labels() -> pd.DataFrame:
    return read_excel_sheet_cached(XLSX, SHEETS["labels"], P.interim / "raw_labels.parquet")


def load_all() -> dict[str, pd.DataFrame]:
    return {SHEETS["profile"]: load_profile(), SHEETS["activity"]: load_activity(), SHEETS["labels"]: load_labels()}
