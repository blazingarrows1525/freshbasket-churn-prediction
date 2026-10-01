"""Exploratory analysis. Each function returns (figure, table) and answers one question (see the notebook).

Run:  python -m churn.eda
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn import config as C
from churn import plotting as pl
from churn.utils import setup_logging, write_table

log = logging.getLogger(__name__)
P = C.P


def load():
    ds = pd.read_parquet(P.processed / "customer_modelling_dataset.parquet")
    grid = pd.read_parquet(P.processed / "grid_clean.parquet")
    return ds, grid


def timeline(ds, grid):
    """FEATURE WINDOW -> CUTOFF -> LABEL WINDOW for every snapshot."""
    snaps = sorted(ds.snapshot.unique())
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for i, s in enumerate(snaps):
        c = pd.Timestamp(s)
        start = max(C.DATA_START, (c - pd.DateOffset(months=C.LONG)).to_period("M").to_timestamp() + pd.DateOffset(months=1))
        y = len(snaps) - 1 - i
        ax.barh(y, (c - start).days, left=start, height=0.45, color=pl.ACCENT, alpha=0.85)
        end = c + pd.DateOffset(months=C.LABEL_MONTHS)
        lab_col = pl.SECOND if s != str(C.SCORING_CUTOFF.date()) else pl.DEEMPH
        ax.barh(y, (end - c).days, left=c, height=0.45, color=lab_col, alpha=0.9)
        role = ("TEST (given CHURNED)" if s == str(C.OFFICIAL_CUTOFF.date()) else
                "SCORING (future unknown)" if s == str(C.SCORING_CUTOFF.date()) else "training (label rebuilt)")
        ax.text(end + pd.Timedelta(days=8), y, f"cutoff {s}: {role}", va="center", fontsize=8.5, color=pl.INK_2)
    ax.axvline(pd.Timestamp("2024-06-30"), color=pl.MUTED, lw=1)
    ax.set_yticks([])
    ax.set_xlim(pd.Timestamp("2022-12-15"), pd.Timestamp("2025-06-01"))
    ax.barh([], [], color=pl.ACCENT, label="feature window (6 months, only data <= cutoff)")
    ax.barh([], [], color=pl.SECOND, label="label window (3 months after cutoff)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_title("Snapshot design: features before the cutoff, churn measured after it")
    ax.grid(axis="y", visible=False)
    return fig, pd.DataFrame({"snapshot": snaps})


def churn_by_population(ds, grid):
    d = ds[ds.churn.notna()]
    t = d.groupby(["snapshot", "population"]).churn.agg(["mean", "size"]).unstack()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(t))
    pops = t["mean"].columns
    for i, p in enumerate(pops):
        ax.bar(x + (i - 0.5) * 0.36, t["mean"][p] * 100, width=0.34, color=[pl.ACCENT, pl.SECOND][i], label=p)
        for xi, v in zip(x, t["mean"][p] * 100):
            ax.text(xi + (i - 0.5) * 0.36, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=pl.INK_2)
    ax.set_xticks(x, t.index)
    ax.set_ylabel("churn rate in the next 3 months (%)")
    ax.set_title("Lapsed members almost all 'churn'; active members churn at ~10% every quarter")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    return fig, t.round(3).reset_index()


def segment_rates(ds, grid):
    """Official snapshot: churn rate by tier, city, signup cohort and other attributes."""
    d = ds[ds.snapshot == str(C.OFFICIAL_CUTOFF.date())]
    rows = []
    for col in ["MEMBERSHIP_TIER", "CITY", "signup_cohort", "GENDER", "HOME_STORE_PRICE_TIER", "PREFERRED_CATEGORY", "MARKETING_OPT_IN"]:
        for pop, g in [("all eligible", d), ("active", d[d.active_at_cutoff])]:
            t = g.groupby(col, observed=True).churn.agg(["mean", "size"]).reset_index()
            for _, r in t.iterrows():
                rows.append({"attribute": col, "value": str(r[col]), "population": pop, "churn_rate": r["mean"], "n": r["size"]})
    tab = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, col in zip(axes, ["MEMBERSHIP_TIER", "signup_cohort", "CITY"]):
        t = tab[(tab.attribute == col)].pivot(index="value", columns="population", values="churn_rate")
        order = t["active"].sort_values().index
        y = np.arange(len(order))
        ax.barh(y + 0.2, t.loc[order, "all eligible"] * 100, height=0.38, color=pl.DEEMPH, label="all eligible")
        ax.barh(y - 0.2, t.loc[order, "active"] * 100, height=0.38, color=pl.ACCENT, label="active")
        ax.set_yticks(y, order, fontsize=8.5)
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("churn rate (%)")
        ax.grid(axis="y", visible=False)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.7, 10)
    fig.suptitle("Churn rate by segment, Apr-Jun 2024 label", x=0.07, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return fig, tab.round(4)


def engagement_support(ds, grid):
    """Active members: churn rate by engagement and support behaviour (last 3 months)."""
    d = ds[(ds.snapshot != str(C.SCORING_CUTOFF.date())) & ds.active_at_cutoff & ds.churn.notna()]
    specs = [("app_avg_3m", [-0.01, 0.5, 1.5, 3, 20], "app sessions / month"),
             ("email_avg_3m", [-0.01, 0.2, 1, 2, 20], "emails opened / month"),
             ("coupons_avg_3m", [-0.01, 0.2, 0.7, 20], "coupons redeemed / month"),
             ("tickets_avg_3m", [-0.01, 0.01, 0.34, 20], "support tickets / month"),
             ("txn_trend", [-99, -2, -0.5, 0.5, 99], "change in transactions/month (last 3m vs prior 3m)")]
    rows = []
    fig, axes = plt.subplots(1, len(specs), figsize=(15, 3.6), sharey=True)
    for ax, (col, bins, lab) in zip(axes, specs):
        b = pd.cut(d[col], bins)
        t = d.groupby(b, observed=True).churn.agg(["mean", "size"])
        ax.bar(range(len(t)), t["mean"] * 100, color=pl.ACCENT, width=0.6)
        ax.set_xticks(range(len(t)), [str(i) for i in t.index], fontsize=7, rotation=20)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="x", visible=False)
        for i, (v, n) in enumerate(zip(t["mean"], t["size"])):
            ax.text(i, v * 100 + 0.8, f"{v:.0%}\nn={n}", ha="center", fontsize=7, color=pl.INK_2)
        for idx, r in t.iterrows():
            rows.append({"feature": col, "bin": str(idx), "churn_rate": r["mean"], "n": r["size"]})
    axes[0].set_ylabel("churn rate, active members (%)")
    fig.suptitle("Active members: churn vs recent engagement, support and purchase trend (4 labelled quarters pooled)",
                 x=0.05, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig, pd.DataFrame(rows).round(4)


def trajectories(ds, grid):
    """Monthly activity in the 6 months before the official cutoff: churners vs retained (active members)."""
    d = ds[(ds.snapshot == str(C.OFFICIAL_CUTOFF.date())) & ds.active_at_cutoff]
    g = grid[(grid.status == "observed") & grid.CUSTOMER_ID.isin(d.CUSTOMER_ID)]
    g = g.merge(d[["CUSTOMER_ID", "churn"]], on="CUSTOMER_ID")
    last = C.OFFICIAL_CUTOFF.to_period("M").to_timestamp()
    g["m_before"] = (last.year - g.MONTH.dt.year) * 12 + (last.month - g.MONTH.dt.month)
    g = g[g.m_before.between(0, 5)]
    cols = ["TRANSACTIONS", "APP_SESSIONS", "EMAILS_OPENED", "SUPPORT_TICKETS"]
    t = g.groupby(["churn", "m_before"])[cols].mean()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, c in zip(axes, cols):
        for i, (lab, k) in enumerate([("retained", 0), ("churned in Apr-Jun", 1)]):
            s = t.loc[k][c].sort_index(ascending=False)
            ax.plot(-s.index, s.values, marker="o", ms=4, color=[pl.DEEMPH, pl.ACCENT][i], label=lab)
        ax.set_title(c.replace("_", " ").lower() + " / month", fontsize=10)
        ax.set_xlabel("months before cutoff (0 = March 2024)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Members who churned were already fading: activity of active members before the cutoff",
                 x=0.05, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig, t.round(3).reset_index()


ALL = {"eda_timeline": timeline, "eda_churn_by_population": churn_by_population, "eda_segment_rates": segment_rates,
       "eda_engagement_support": engagement_support, "eda_trajectories": trajectories}


def run():
    ds, grid = load()
    for name, fn in ALL.items():
        fig, tab = fn(ds, grid)
        pl.save(fig, P.figures / f"{name}.png")
        write_table(tab, P.tables / name)
        log.info("saved %s", name)


if __name__ == "__main__":
    setup_logging()
    run()
