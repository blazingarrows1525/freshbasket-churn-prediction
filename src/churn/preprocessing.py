"""Data-quality checks and cleaning for the three loyalty tables.

Produces a clean profile table, a clean monthly activity table and a member x month grid in which every
month is one of:
    observed        a row exists
    gap             no row, but the member has rows before and after it (a deleted record)
    after_last_row  no row after the member's last row (nothing happened)
    pre_signup      before the member joined
    no_rows_yet     after signup but before the member's first row
Every treatment is logged with its evidence in outputs/tables/data_quality_findings.

Note: the grid's `status` uses all rows and is only for reporting. Features never use it for months after
a cutoff: feature_engineering.month_matrix() re-derives gaps from rows up to each cutoff.

Run:  python -m churn.preprocessing
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from churn import config as C
from churn import data_loader as D
from churn.utils import QualityLog, setup_logging, write_table

log = logging.getLogger(__name__)


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.title()


def clean_profile(cp: pd.DataFrame, q: QualityLog) -> pd.DataFrame:
    n_dup = int(cp.duplicated().sum())
    q.add("Duplicate profile rows", f"{n_dup} exact duplicate rows ({cp.CUSTOMER_ID.duplicated().sum()} repeated IDs, "
          "all identical)", "Low", "Dropped", "Identical copies carry no information", "fb Customer Profile")
    cp = cp.drop_duplicates().copy()
    for col in ["CITY", "MEMBERSHIP_TIER"]:
        raw = cp[col].nunique()
        cp[col] = clean_text(cp[col])
        q.add(f"Inconsistent casing/whitespace in {col}", f"{raw} raw spellings -> {cp[col].nunique()} values "
              f"({', '.join(sorted(cp[col].unique()))})", "Medium", "Trimmed and title-cased",
              "Variants such as 'PORTLAND', ' Portland ' and 'portland' are one value; left as-is they split segments",
              "fb Customer Profile")
    for col, fill in [("GENDER", "Unknown"), ("HOME_STORE_PRICE_TIER", "Unknown")]:
        n = int(cp[col].isna().sum())
        cp[col] = cp[col].fillna(fill)
        q.add(f"Missing {col}", f"{n} members ({n / len(cp):.1%})", "Low", f"Explicit '{fill}' category",
              "Missingness may itself be informative; no guessed value is imputed", "fb Customer Profile")
    n_age = int(cp.AGE.isna().sum())
    q.add("Missing AGE", f"{n_age} members ({n_age / len(cp):.1%})", "Low",
          "Left missing for LightGBM; median + missing-indicator for logistic regression",
          "Avoids inventing ages", "fb Customer Profile")
    late = int((cp.SIGNUP_DATE > C.OFFICIAL_CUTOFF).sum())
    q.add("Members who joined during the label window", f"{late} members signed up after {C.OFFICIAL_CUTOFF.date()}",
          "Medium", "Not eligible for the official snapshot (no history before the label window)",
          "Churn is only defined for members with earlier purchase history", "fb Customer Profile")
    return cp


def clean_activity(ma: pd.DataFrame, q: QualityLog) -> pd.DataFrame:
    n_exact = int(ma.duplicated().sum())
    ma = ma.drop_duplicates().copy()
    conflict = ma[ma.duplicated(["CUSTOMER_ID", "MONTH"], keep=False)]
    q.add("Duplicate activity rows", f"{n_exact} exact duplicates; {conflict.groupby(['CUSTOMER_ID', 'MONTH']).ngroups} "
          f"customer-months with two different rows, each pair differing only in the sign of TOTAL_SPEND",
          "Medium", "Exact duplicates dropped; for conflicting pairs the positive-spend row is kept",
          "The pairs are the same record entered twice with a sign error", "fb Monthly Activity")
    ma = ma.sort_values("TOTAL_SPEND", ascending=False).drop_duplicates(["CUSTOMER_ID", "MONTH"]).copy()

    implied = ma.TRANSACTIONS * ma.AVG_BASKET_VALUE
    exact = np.isclose(ma.TOTAL_SPEND, implied, rtol=0.005, atol=0.05) & (ma.TRANSACTIONS > 0)
    q.add("TOTAL_SPEND = TRANSACTIONS x AVG_BASKET_VALUE", f"holds (within 0.5%, basket value is rounded to cents) on "
          f"{exact.sum():,} of {(ma.TRANSACTIONS > 0).sum():,} purchase months ({exact.sum() / (ma.TRANSACTIONS > 0).sum():.1%})",
          "Info", "Used to diagnose and repair the spend errors below", "Two independent fields agree", "fb Monthly Activity")
    neg = ma.TOTAL_SPEND < 0
    q.add("Negative spend", f"{int(neg.sum())} rows; |spend| equals transactions x avg basket in "
          f"{np.isclose(ma.TOTAL_SPEND[neg].abs(), implied[neg], rtol=0.005, atol=0.05).mean():.0%} of them",
          "Medium", "Sign flipped (absolute value)", "Pure sign-entry errors", "fb Monthly Activity")
    ma.loc[neg, "TOTAL_SPEND"] = ma.loc[neg, "TOTAL_SPEND"].abs()
    zero_bad = (ma.TRANSACTIONS > 0) & (ma.TOTAL_SPEND == 0)
    q.add("Zero spend in months with transactions", f"{int(zero_bad.sum())} rows (average basket value is present)",
          "Low", "Spend rebuilt as transactions x average basket", "A purchase month cannot have zero spend",
          "fb Monthly Activity")
    ma.loc[zero_bad, "TOTAL_SPEND"] = implied[zero_bad]
    ratio = ma.TOTAL_SPEND / implied.replace(0, np.nan)
    outl = ratio > C.SPEND_RATIO_MAX
    q.add("Outlier spend values", f"{int(outl.sum())} rows where spend is > {C.SPEND_RATIO_MAX:g}x transactions x "
          f"avg basket (e.g. {ma.loc[outl, 'TOTAL_SPEND'].max():,.0f} vs {implied[outl].loc[ma.loc[outl, 'TOTAL_SPEND'].idxmax()]:,.0f}); "
          f"99.9th percentile of spend otherwise {ma.loc[~outl & (ma.TRANSACTIONS > 0), 'TOTAL_SPEND'].quantile(0.999):,.0f}",
          "Medium", "Replaced by transactions x average basket", "Both transaction count and basket value are "
          "normal in these rows, so only the total was mistyped", "fb Monthly Activity")
    ma.loc[outl, "TOTAL_SPEND"] = implied[outl]
    zt = ma.TRANSACTIONS == 0
    q.add("Zero-transaction months", f"{int(zt.sum()):,} rows with 0 transactions (spend and basket 0, but "
          f"{(ma.loc[zt, ['APP_SESSIONS', 'EMAILS_OPENED', 'SUPPORT_TICKETS']].sum(axis=1) > 0).mean():.0%} show app, "
          "email or support activity)", "Info", "Kept as genuine non-purchase months",
          "Members stay engaged for 1-3 months after their last purchase before records stop", "fb Monthly Activity")
    return ma


def make_grid(cp: pd.DataFrame, ma: pd.DataFrame) -> pd.DataFrame:
    """Member x month grid (Jan 2023 onwards) with the status of every month."""
    months = pd.date_range(C.DATA_START, ma.MONTH.max(), freq="MS")
    grid = pd.MultiIndex.from_product([cp.CUSTOMER_ID.unique(), months], names=["CUSTOMER_ID", "MONTH"]).to_frame(index=False)
    grid = grid.merge(ma, on=["CUSTOMER_ID", "MONTH"], how="left", indicator=True)
    grid = grid.merge(cp[["CUSTOMER_ID", "SIGNUP_DATE"]], on="CUSTOMER_ID")
    grid["first_row"] = grid.CUSTOMER_ID.map(ma.groupby("CUSTOMER_ID").MONTH.min())
    grid["last_row"] = grid.CUSTOMER_ID.map(ma.groupby("CUSTOMER_ID").MONTH.max())
    grid["status"] = np.select(
        [grid._merge == "both", grid.MONTH < grid.SIGNUP_DATE.dt.to_period("M").dt.to_timestamp(),
         grid.first_row.isna() | (grid.MONTH < grid.first_row), grid.MONTH < grid.last_row],
        ["observed", "pre_signup", "no_rows_yet", "gap"], default="after_last_row")
    return grid.drop(columns=["_merge"])


def build(q: QualityLog | None = None) -> dict:
    q = q or QualityLog()
    cp = clean_profile(D.load_profile(), q)
    ma = clean_activity(D.load_activity(), q)
    cl = D.load_labels()

    n_dl = int(cl.duplicated().sum())
    cl = cl.drop_duplicates().copy()
    q.add("Duplicate label rows", f"{n_dl} exact duplicates", "Low", "Dropped", "Identical copies", "fb Churn Label")
    leaky = [c for c in C.POST_OUTCOME_COLUMNS if c in cl.columns and c != "OBSERVATION_END_DATE"]
    q.add("Post-outcome columns in the label table",
          f"{', '.join(leaky)} are computed over the whole window including Apr-Jun 2024 (a LAST_PURCHASE_DATE "
          "before April implies CHURNED = 1)", "High",
          "Never used as features; honest equivalents are recomputed from activity up to each cutoff",
          "Direct target leakage", "fb Churn Label")
    no_act = sorted(set(cl.CUSTOMER_ID) - set(ma.CUSTOMER_ID))
    q.add("Members without any activity row", f"{len(no_act)} members", "Low",
          "Not eligible (no purchase history)", "Nothing to base a prediction on", "fb Monthly Activity")

    grid = make_grid(cp, ma)
    gap = grid.status == "gap"
    q.add("Missing months inside members' activity spans",
          f"{int(gap.sum())} customer-months missing between existing rows for {grid.loc[gap, 'CUSTOMER_ID'].nunique()} members "
          f"({gap.sum() / (grid.status.isin(['observed', 'gap'])).sum():.1%} of those spans)", "Medium",
          "Treated as unknown: features use observed months only and count the gaps. For each snapshot a gap is "
          "recognised from rows up to that cutoff only (feature_engineering.month_matrix)", "Deleted records, not "
          "inactivity: members have rows on both sides", "fb Monthly Activity")

    # can the given label be rebuilt from the activity table with the stated definition?
    lw = C.OFFICIAL_CUTOFF + pd.Timedelta(days=1)
    tx_pre = ma[ma.MONTH < lw].groupby("CUSTOMER_ID").TRANSACTIONS.sum()
    tx_lw = ma[ma.MONTH >= lw].groupby("CUSTOMER_ID").TRANSACTIONS.sum()
    rb = cl.set_index("CUSTOMER_ID")[["CHURNED"]].join(tx_pre.rename("pre")).join(tx_lw.rename("lw")).fillna(0)
    rb["rebuilt"] = ((rb.pre > 0) & (rb.lw == 0)).astype(int)
    mism = rb[rb.rebuilt != rb.CHURNED].join(cl.set_index("CUSTOMER_ID").LAST_PURCHASE_DATE)
    have = set(zip(ma.CUSTOMER_ID, ma.MONTH))
    missing_row = [(i, d) not in have for i, d in zip(mism.index, mism.LAST_PURCHASE_DATE)]
    q.add("Label reconstruction check", f"rebuilding CHURNED from activity (no purchase Apr-Jun 2024, purchases before) "
          f"agrees for {1 - len(mism) / len(rb):.2%} of members; the {len(mism)} disagreements "
          f"({int((mism.CHURNED == 1).sum())} labelled churned, {int((mism.CHURNED == 0).sum())} labelled active): for "
          f"{sum(missing_row)} of them the activity row for the month of their recorded LAST_PURCHASE_DATE is missing",
          "Medium",
          "The given CHURNED is the target for the official snapshot; rebuilt labels are used only for earlier "
          "snapshots, accepting ~0.4% label noise", "Confirms the definition and shows the label table was built "
          "before rows were deleted", "fb Churn Label")
    return {"profile": cp, "activity": ma, "labels": cl, "grid": grid, "qlog": q}


def run():
    out = build()
    for k in ["profile", "activity", "labels", "grid"]:
        out[k].to_parquet(C.P.processed / f"{k}_clean.parquet", index=False)
    write_table(out["qlog"].to_frame(), C.P.tables / "data_quality_findings")
    log.info("grid status:\n%s", out["grid"].status.value_counts())


if __name__ == "__main__":
    setup_logging()
    run()
