"""Member snapshots: features at a cutoff + the churn label for the 3 months after it.

FEATURE WINDOW (months <= cutoff)  ->  CUTOFF  ->  LABEL WINDOW (3 months after)

Leakage rules, enforced here and tested by `leakage_test`:
* only activity rows with MONTH <= cutoff are read for features;
* whether a missing month is a gap (deleted record) or inactivity is decided from rows up to the cutoff
  only: a month without a row is a gap if the member has a later row that is still <= cutoff;
* the label table's LAST_PURCHASE_DATE / MONTHS_OBSERVED / INSUFFICIENT_HISTORY_FLAG are never used
  (they are computed over the label window).
Windows are fixed (3 and 6 months) and recency is capped at 6 months, so a snapshot with 6 months of
history and one with 15 months produce comparable features.

Run:  python -m churn.feature_engineering
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from churn import config as C
from churn import preprocessing as PP
from churn.utils import setup_logging, write_table

log = logging.getLogger(__name__)
P = C.P

ACT = ["TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE", "DISTINCT_CATEGORIES", "PROMO_TXN_PCT", "APP_SESSIONS",
       "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS", "COMPLAINT_FLAG"]

GROUPS = {
    "demographics": ["AGE", "GENDER", "CITY", "MEMBERSHIP_TIER", "MARKETING_OPT_IN", "PREFERRED_CATEGORY",
                     "HOME_STORE_PRICE_TIER", "tenure_months"],
    "behavioural": ["recency_months", "no_purchase_6m", "txn_avg_3m", "txn_avg_6m", "txn_trend", "spend_avg_3m",
                    "spend_avg_6m", "spend_trend", "basket_avg_6m", "active_share_6m", "cats_avg_3m",
                    "promo_pct_avg_6m", "history_months", "gap_months_6m"],
    "engagement": ["app_avg_3m", "app_avg_6m", "app_trend", "email_avg_3m", "email_avg_6m", "email_trend",
                   "coupons_avg_3m", "coupons_avg_6m", "coupon_trend", "months_since_any_activity"],
    "support": ["tickets_avg_3m", "tickets_avg_6m", "tickets_trend", "complaint_6m"],
}
CATEGORICAL = ["GENDER", "CITY", "MEMBERSHIP_TIER", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"]
ALL_FEATURES = [f for g in GROUPS.values() for f in g]

# feature -> (type, definition, business interpretation, leakage note)
_W = "months <= cutoff only"
FEATURE_INFO = {
    "AGE": ("numeric", "age from the profile", "life stage", "static attribute"),
    "GENDER": ("categorical", "gender ('Unknown' if missing)", "demographic segment", "static attribute"),
    "CITY": ("categorical", "home city (cleaned casing)", "regional differences", "static attribute"),
    "MEMBERSHIP_TIER": ("categorical", "Silver / Gold / Platinum", "does tier protect against churn?", "static attribute"),
    "MARKETING_OPT_IN": ("binary", "opted in to marketing", "reachable by email campaigns", "static attribute"),
    "PREFERRED_CATEGORY": ("categorical", "preferred shopping category", "shopping mission", "static attribute"),
    "HOME_STORE_PRICE_TIER": ("categorical", "price tier of the home store", "price positioning of the store", "static attribute"),
    "tenure_months": ("numeric", "months from signup to cutoff", "loyalty length", "uses cutoff date only"),
    "recency_months": ("numeric", "months since the last purchase month, capped at 6", "how long since they last shopped", _W),
    "no_purchase_6m": ("binary", "no purchase in the 6 months to the cutoff", "already lapsed", _W),
    "txn_avg_3m": ("numeric", "mean transactions per observed month, last 3 months", "current shopping frequency", _W),
    "txn_avg_6m": ("numeric", "mean transactions per observed month, last 6 months", "habitual shopping frequency", _W),
    "txn_trend": ("numeric", "txn_avg_3m minus the mean of the 3 months before", "shopping more or less than before", _W),
    "spend_avg_3m": ("numeric", "mean monthly spend, last 3 months (cleaned spend)", "current value", _W),
    "spend_avg_6m": ("numeric", "mean monthly spend, last 6 months", "habitual value", _W),
    "spend_trend": ("numeric", "spend_avg_3m minus the 3 months before", "spending more or less than before", _W),
    "basket_avg_6m": ("numeric", "mean basket value in purchase months, last 6 months", "basket size", _W),
    "active_share_6m": ("numeric", "share of observed months with a purchase, last 6 months", "regularity", _W),
    "cats_avg_3m": ("numeric", "mean distinct categories per month, last 3 months", "breadth of the relationship", _W),
    "promo_pct_avg_6m": ("numeric", "mean share of promotional transactions in purchase months, last 6 months", "deal seeking", _W),
    "history_months": ("numeric", "months of known history in the 6-month window", "how much we know about the member", _W),
    "gap_months_6m": ("numeric", "missing (deleted) months in the 6-month window", "data completeness", "gaps decided from rows <= cutoff"),
    "app_avg_3m": ("numeric", "mean app sessions per month, last 3 months", "digital engagement now", _W),
    "app_avg_6m": ("numeric", "mean app sessions per month, last 6 months", "habitual digital engagement", _W),
    "app_trend": ("numeric", "app_avg_3m minus the 3 months before", "engagement fading or growing", _W),
    "email_avg_3m": ("numeric", "mean emails opened per month, last 3 months", "responsiveness to marketing now", _W),
    "email_avg_6m": ("numeric", "mean emails opened per month, last 6 months", "habitual responsiveness", _W),
    "email_trend": ("numeric", "email_avg_3m minus the 3 months before", "responsiveness fading or growing", _W),
    "coupons_avg_3m": ("numeric", "mean coupons redeemed per month, last 3 months", "offer usage now", _W),
    "coupons_avg_6m": ("numeric", "mean coupons redeemed per month, last 6 months", "habitual offer usage", _W),
    "coupon_trend": ("numeric", "coupons_avg_3m minus the 3 months before", "offer usage fading or growing", _W),
    "months_since_any_activity": ("numeric", "months since the last month with any row, capped at 6", "silent members", _W),
    "tickets_avg_3m": ("numeric", "mean support tickets per month, last 3 months", "current service problems", _W),
    "tickets_avg_6m": ("numeric", "mean support tickets per month, last 6 months", "habitual service problems", _W),
    "tickets_trend": ("numeric", "tickets_avg_3m minus the 3 months before", "problems escalating", _W),
    "complaint_6m": ("binary", "any complaint flag in the last 6 months", "a formal complaint was raised", _W),
}
assert set(FEATURE_INFO) == set(ALL_FEATURES)
assert not set(ALL_FEATURES) & set(C.POST_OUTCOME_COLUMNS), "post-outcome label columns must never be features"


def month_matrix(grid: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """member x month tables of each activity field, as known at `cutoff`."""
    last_m = cutoff.to_period("M").to_timestamp()
    g = grid[grid.MONTH <= last_m].copy()
    obs = g[g.status == "observed"]
    g["fr"] = g.CUSTOMER_ID.map(obs.groupby("CUSTOMER_ID").MONTH.min())
    g["lr"] = g.CUSTOMER_ID.map(obs.groupby("CUSTOMER_ID").MONTH.max())
    g["state"] = np.select([g.status == "observed", g.fr.isna() | (g.MONTH < g.fr), g.MONTH < g.lr],
                           ["observed", "before_first_row", "gap"], default="inactive")
    out = {}
    for col in ACT:
        v = g[col].where(g.state == "observed")
        v = v.where(g.state != "inactive", 0.0)
        out[col] = g.assign(v=v).pivot(index="CUSTOMER_ID", columns="MONTH", values="v")
    out["state"] = g.pivot(index="CUSTOMER_ID", columns="MONTH", values="state")
    out["any_row"] = g.assign(v=(g.state == "observed").astype(float)).pivot(index="CUSTOMER_ID", columns="MONTH", values="v")
    return out


def _win(m: pd.DataFrame, last_m: pd.Timestamp, start_back: int, n: int) -> pd.DataFrame:
    """Columns for months (last_m - start_back - n + 1) .. (last_m - start_back)."""
    return m.reindex(columns=[last_m - pd.DateOffset(months=k) for k in range(start_back, start_back + n)])


def snapshot(grid: pd.DataFrame, profile: pd.DataFrame, cutoff: pd.Timestamp,
             given_labels: pd.DataFrame | None = None, with_label: bool = True) -> pd.DataFrame:
    last_m = cutoff.to_period("M").to_timestamp()
    M = month_matrix(grid, cutoff)
    tx = M["TRANSACTIONS"]
    f = pd.DataFrame(index=tx.index)
    f["eligible"] = tx.fillna(0).sum(axis=1) > 0

    bought = tx.fillna(0) > 0
    last_buy = bought.apply(lambda r: r[r].index.max() if r.any() else pd.NaT, axis=1)
    rec = ((last_m.year - last_buy.dt.year) * 12 + (last_m.month - last_buy.dt.month)).astype(float)
    f["recency_months"] = rec.clip(upper=C.LONG).fillna(C.LONG)
    f["no_purchase_6m"] = (rec.fillna(99) >= C.LONG).astype(int)
    anyrow = M["any_row"]
    last_any = anyrow.apply(lambda r: r[r > 0].index.max() if (r > 0).any() else pd.NaT, axis=1)
    f["months_since_any_activity"] = ((last_m.year - last_any.dt.year) * 12 + (last_m.month - last_any.dt.month)
                                      ).astype(float).clip(upper=C.LONG).fillna(C.LONG)

    def avg(col, back, n):
        return _win(M[col], last_m, back, n).mean(axis=1, skipna=True)

    f["txn_avg_3m"], f["txn_avg_6m"] = avg("TRANSACTIONS", 0, 3), avg("TRANSACTIONS", 0, 6)
    f["txn_trend"] = f.txn_avg_3m - avg("TRANSACTIONS", 3, 3)
    f["spend_avg_3m"], f["spend_avg_6m"] = avg("TOTAL_SPEND", 0, 3), avg("TOTAL_SPEND", 0, 6)
    f["spend_trend"] = f.spend_avg_3m - avg("TOTAL_SPEND", 3, 3)
    w6 = _win(tx, last_m, 0, 6)
    f["basket_avg_6m"] = _win(M["AVG_BASKET_VALUE"], last_m, 0, 6).where(w6 > 0).mean(axis=1)
    f["active_share_6m"] = (w6 > 0).sum(axis=1) / w6.notna().sum(axis=1).replace(0, np.nan)
    f["cats_avg_3m"] = avg("DISTINCT_CATEGORIES", 0, 3)
    f["promo_pct_avg_6m"] = _win(M["PROMO_TXN_PCT"], last_m, 0, 6).where(w6 > 0).mean(axis=1)
    st = _win(M["state"], last_m, 0, 6)
    f["gap_months_6m"] = (st == "gap").sum(axis=1)
    f["history_months"] = (st.isin(["observed", "gap", "inactive"])).sum(axis=1)
    f["app_avg_3m"], f["app_avg_6m"] = avg("APP_SESSIONS", 0, 3), avg("APP_SESSIONS", 0, 6)
    f["app_trend"] = f.app_avg_3m - avg("APP_SESSIONS", 3, 3)
    f["email_avg_3m"], f["email_avg_6m"] = avg("EMAILS_OPENED", 0, 3), avg("EMAILS_OPENED", 0, 6)
    f["email_trend"] = f.email_avg_3m - avg("EMAILS_OPENED", 3, 3)
    f["coupons_avg_3m"], f["coupons_avg_6m"] = avg("COUPONS_REDEEMED", 0, 3), avg("COUPONS_REDEEMED", 0, 6)
    f["coupon_trend"] = f.coupons_avg_3m - avg("COUPONS_REDEEMED", 3, 3)
    f["tickets_avg_3m"], f["tickets_avg_6m"] = avg("SUPPORT_TICKETS", 0, 3), avg("SUPPORT_TICKETS", 0, 6)
    f["tickets_trend"] = f.tickets_avg_3m - avg("SUPPORT_TICKETS", 3, 3)
    f["complaint_6m"] = (_win(M["COMPLAINT_FLAG"], last_m, 0, 6).fillna(0).sum(axis=1) > 0).astype(int)

    prof = profile.set_index("CUSTOMER_ID")
    f = f.join(prof[["SIGNUP_DATE", "AGE", "GENDER", "CITY", "MEMBERSHIP_TIER", "MARKETING_OPT_IN",
                     "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER", "PREFERRED_STORE_ID"]])
    f["tenure_months"] = ((cutoff - f.SIGNUP_DATE).dt.days / 30.44).round(1)
    f["signup_cohort"] = f.SIGNUP_DATE.dt.year.astype("Int64").astype(str)
    f["active_at_cutoff"] = (_win(tx, last_m, 0, C.ACTIVE_MONTHS).fillna(0).sum(axis=1) > 0)
    f["population"] = np.where(f.active_at_cutoff, "active (bought in last 3 months)", "lapsed (no purchase in last 3 months)")
    f["cutoff"] = cutoff
    f = f[f.SIGNUP_DATE <= cutoff]
    if not with_label:
        return f.reset_index()

    # label: no purchase in the 3 months after the cutoff
    if given_labels is not None:
        f = f.join(given_labels.set_index("CUSTOMER_ID")["CHURNED"].rename("churn"))
    else:
        nxt = [last_m + pd.DateOffset(months=k) for k in range(1, C.LABEL_MONTHS + 1)]
        g = grid[grid.MONTH.isin(nxt) & (grid.status == "observed")]
        future = g.groupby("CUSTOMER_ID").TRANSACTIONS.sum()
        if grid.MONTH.max() >= nxt[-1]:
            f["churn"] = (future.reindex(f.index).fillna(0) == 0).astype(int)
        else:
            f["churn"] = np.nan                     # scoring snapshot: the future is unknown
    return f.reset_index()


def build_all(grid, profile, labels) -> dict[str, pd.DataFrame]:
    out = {str(c.date()): snapshot(grid, profile, c) for c in C.TRAIN_CUTOFFS}
    out[str(C.OFFICIAL_CUTOFF.date())] = snapshot(grid, profile, C.OFFICIAL_CUTOFF, labels)
    out[str(C.SCORING_CUTOFF.date())] = snapshot(grid, profile, C.SCORING_CUTOFF)
    return out


def load_snapshots() -> dict[str, pd.DataFrame]:
    """All snapshots, eligible members only, with categories aligned across snapshots."""
    grid = pd.read_parquet(P.processed / "grid_clean.parquet")
    prof = pd.read_parquet(P.processed / "profile_clean.parquet")
    lab = pd.read_parquet(P.processed / "labels_clean.parquet")
    snaps = build_all(grid, prof, lab)
    cats = {c: sorted(pd.concat([s[c] for s in snaps.values()]).dropna().astype(str).unique()) for c in CATEGORICAL}
    for k, s in snaps.items():
        s = s[s.eligible].copy()
        for c in CATEGORICAL:
            s[c] = pd.Categorical(s[c].astype(str), categories=cats[c])
        snaps[k] = s.reset_index(drop=True)
    return snaps


def leakage_test() -> pd.DataFrame:
    """Rebuild each labelled snapshot from activity truncated at its cutoff and compare every feature.

    If any feature read a month after the cutoff (directly, or through the gap/inactivity decision), the
    truncated rebuild would differ. Expected maximum difference: 0.
    """
    prof = pd.read_parquet(P.processed / "profile_clean.parquet")
    act = pd.read_parquet(P.processed / "activity_clean.parquet")
    full_grid = PP.make_grid(prof, act)
    rows = []
    for cutoff in C.TRAIN_CUTOFFS + [C.OFFICIAL_CUTOFF]:
        last_m = cutoff.to_period("M").to_timestamp()
        trunc_grid = PP.make_grid(prof, act[act.MONTH <= last_m])
        a = snapshot(full_grid, prof, cutoff, with_label=False).set_index("CUSTOMER_ID")
        b = snapshot(trunc_grid, prof, cutoff, with_label=False).set_index("CUSTOMER_ID").reindex(a.index)
        num = [f for f in ALL_FEATURES if f not in CATEGORICAL]
        diff = (a[num].astype(float) - b[num].astype(float)).abs()
        both_nan = a[num].isna() & b[num].isna()
        max_diff = diff.where(~both_nan, 0).max().max()
        nan_mismatch = int((a[num].isna() != b[num].isna()).sum().sum())
        cat_mismatch = int(sum((a[c].astype(str) != b[c].astype(str)).sum() for c in CATEGORICAL))
        rows.append({"cutoff": str(cutoff.date()), "members": len(a), "activity_rows_removed": int((act.MONTH > last_m).sum()),
                     "features_compared": len(ALL_FEATURES), "max_abs_difference": float(max_diff),
                     "missing_value_mismatches": nan_mismatch, "categorical_mismatches": cat_mismatch,
                     "passed": bool(max_diff == 0 and nan_mismatch == 0 and cat_mismatch == 0)})
    return pd.DataFrame(rows)


def run():
    snaps = load_snapshots()
    snap_tab = pd.DataFrame([{
        "cutoff": k, "label_window": f"{(pd.Timestamp(k) + pd.Timedelta(days=1)).strftime('%b %Y')} +3 months",
        "eligible_members": len(s), "churn_rate_all": s.churn.mean(), "active_members": int(s.active_at_cutoff.sum()),
        "churn_rate_active": s[s.active_at_cutoff].churn.mean(), "lapsed_members": int((~s.active_at_cutoff).sum()),
        "churn_rate_lapsed": s[~s.active_at_cutoff].churn.mean(),
        "role": "test (given CHURNED)" if k == str(C.OFFICIAL_CUTOFF.date()) else
                ("scoring (future unknown)" if k == str(C.SCORING_CUTOFF.date()) else "training (label rebuilt)")}
        for k, s in snaps.items()])
    write_table(snap_tab, P.tables / "snapshots_target_distribution")

    full = pd.concat([s.assign(snapshot=k) for k, s in snaps.items()], ignore_index=True)
    full.drop(columns=["SIGNUP_DATE"]).to_csv(P.processed / "customer_modelling_dataset.csv", index=False)
    full.to_parquet(P.processed / "customer_modelling_dataset.parquet", index=False)

    leak = leakage_test()
    write_table(leak, P.tables / "leakage_check")
    assert leak.passed.all(), f"leakage test failed:\n{leak}"

    cat = pd.DataFrame([{"feature": f, "group": g, "type": FEATURE_INFO[f][0], "definition": FEATURE_INFO[f][1],
                         "business_interpretation": FEATURE_INFO[f][2], "leakage_control": FEATURE_INFO[f][3]}
                        for g, fs in GROUPS.items() for f in fs])
    write_table(cat, P.tables / "feature_definitions")
    log.info("modelling dataset: %d rows x %d columns; leakage test passed for %d snapshots",
             len(full), full.shape[1], len(leak))


if __name__ == "__main__":
    setup_logging()
    run()
