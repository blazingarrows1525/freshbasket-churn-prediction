"""Final churn model, calibration, bootstrap comparison, prediction files and gains.

Selected model: logistic regression without class weights (see model_comparison):
it ties LightGBM on the active population (bootstrap below) while being fully
transparent - every member's risk decomposes exactly into feature contributions.

Run:  python -m churn.final
"""
from __future__ import annotations

import logging

import lightgbm  # noqa: F401  load before scikit-learn: on Windows the reverse order crashes LightGBM (OpenMP DLL clash)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from churn import config as C
from churn import feature_engineering as F
from churn import modeling as Md
from churn import plotting as pl
from churn.utils import setup_logging, write_json, write_table

log = logging.getLogger(__name__)
P = C.P

NAMES = {
    "AGE": "age", "GENDER": "gender", "CITY": "city", "MEMBERSHIP_TIER": "membership tier",
    "MARKETING_OPT_IN": "marketing opt-in", "PREFERRED_CATEGORY": "preferred category",
    "HOME_STORE_PRICE_TIER": "home store price tier", "tenure_months": "tenure (months)",
    "recency_months": "months since last purchase", "no_purchase_6m": "no purchase in 6 months",
    "txn_avg_3m": "transactions/month (3m)", "txn_avg_6m": "transactions/month (6m)",
    "txn_trend": "transaction trend (3m vs prior 3m)", "spend_avg_3m": "spend/month (3m)", "spend_avg_6m": "spend/month (6m)",
    "spend_trend": "spend trend (3m vs prior 3m)", "basket_avg_6m": "average basket (6m)",
    "active_share_6m": "share of months with a purchase (6m)", "cats_avg_3m": "categories bought/month (3m)",
    "promo_pct_avg_6m": "share of promo transactions (6m)", "history_months": "months of history",
    "gap_months_6m": "missing months in records (6m)", "app_avg_3m": "app sessions/month (3m)",
    "app_avg_6m": "app sessions/month (6m)", "app_trend": "app session trend", "email_avg_3m": "emails opened/month (3m)",
    "email_avg_6m": "emails opened/month (6m)", "email_trend": "email-open trend", "coupons_avg_3m": "coupons redeemed/month (3m)",
    "coupons_avg_6m": "coupons redeemed/month (6m)", "coupon_trend": "coupon trend",
    "months_since_any_activity": "months since any recorded activity", "tickets_avg_3m": "support tickets/month (3m)",
    "tickets_avg_6m": "support tickets/month (6m)", "tickets_trend": "support ticket trend", "complaint_6m": "complaint in last 6 months",
}


def lr_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Exact logit contributions of each original feature relative to the training average."""
    pre = model.named_steps["pre"]
    lr = model.named_steps["lr"]
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    names = pre.get_feature_names_out()
    coef = lr.coef_[0]
    # numeric parts are standardised (mean 0 in training) -> contribution = coef * z;
    # one-hot parts: contribution relative to the reference category = coef * indicator
    contrib = pd.DataFrame(Z * coef, columns=names, index=X.index)
    orig = {}
    for n in names:
        base = n.split("__", 1)[1]
        if n.startswith("num__missingindicator_"):
            key = base.replace("missingindicator_", "")
        elif n.startswith("cat__"):
            key = next(c for c in F.CATEGORICAL if base.startswith(c + "_"))
        else:
            key = base
        orig[n] = key
    return contrib.T.groupby(pd.Series(orig)).sum().T


def top_drivers(contrib: pd.DataFrame, X: pd.DataFrame, k: int = 3) -> pd.Series:
    out = []
    for i, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(k)
        parts = []
        for f, v in top.items():
            val = X.loc[i, f]
            val = f"{val:.1f}" if isinstance(val, (float, np.floating)) else str(val)
            parts.append(f"{NAMES.get(f, f)} = {val} (+{v:.2f} logit)")
        out.append("; ".join(parts) if parts else "no risk-raising factor above average")
    return pd.Series(out, index=contrib.index)


def bootstrap_compare(y, p1, p2, n=1000, seed=C.SEED) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    d_auc, d_ap = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        d_auc.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
        d_ap.append(average_precision_score(y[idx], p1[idx]) - average_precision_score(y[idx], p2[idx]))
    q = lambda a: (float(np.mean(a)), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975)))
    return {"ROC-AUC diff (LR - LGBM)": q(d_auc), "PR-AUC diff (LR - LGBM)": q(d_ap)}


def reliability(y, p, bins=10) -> pd.DataFrame:
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(mean_predicted=("p", "mean"),
                                                                      observed_rate=("y", "mean"), n=("y", "size"))


def plot_calibration(tabs: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pop in zip(axes, ["all", "active"]):
        ax.plot([0, 1], [0, 1], color=pl.BASELINE, lw=1)
        for i, (name, t) in enumerate(tabs[pop].items()):
            ax.plot(t.mean_predicted, t.observed_rate, marker="o", ms=5, color=pl.CATEGORICAL[i], label=name)
        ax.set_xlabel("mean predicted churn probability (decile)")
        ax.set_ylabel("observed churn rate")
        ax.set_title(f"Calibration, official test snapshot ({pop} members)", fontsize=10.5)
        ax.legend(fontsize=8)
        lim = 1 if pop == "all" else max(0.8, max(t.observed_rate.max() for t in tabs[pop].values()) + 0.05)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.tight_layout()
    return fig


def plot_gains(g: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.r_[0, g.targeted_share.values] * 100
    y = np.r_[0, g.share_of_churners_captured.values] * 100
    ax.plot(x, y, marker="o", color=pl.ACCENT, label="target by model risk")
    ax.plot([0, 100], [0, 100], color=pl.BASELINE, lw=1, label="random / blanket targeting")
    for xi, yi in zip(x[1:], y[1:]):
        ax.text(xi + 1, yi - 4, f"{yi:.0f}%", fontsize=8, color=pl.INK_2)
    ax.set_xlabel("% of active members contacted (highest risk first)")
    ax.set_ylabel("% of the quarter's churners reached")
    ax.set_title("Cumulative gains on the Apr-Jun 2024 test quarter (active members)")
    ax.set_xlim(0, 55)
    ax.legend(fontsize=8, loc="lower right")
    return fig


def run():
    snaps = F.load_snapshots()
    keys = [str(c.date()) for c in C.TRAIN_CUTOFFS]
    val_train, val = Md.stack(snaps, keys[:2]), snaps[keys[2]]
    train, test = Md.stack(snaps, keys), snaps[str(C.OFFICIAL_CUTOFF.date())]
    feats = F.ALL_FEATURES

    # threshold chosen on the validation snapshot, active members (the population a campaign would target)
    m_val = Md.make_lr(feats).fit(val_train[feats], val_train.churn)
    pv = m_val.predict_proba(val[feats])[:, 1]
    thr = Md.best_threshold(val.churn.values[val.active_at_cutoff.values], pv[val.active_at_cutoff.values])

    model = Md.make_lr(feats).fit(train[feats], train.churn)
    p = model.predict_proba(test[feats])[:, 1]
    # LightGBM challenger for the bootstrap/calibration comparison
    lg_val = Md.fit_lgb(val_train, feats, valid=val)
    n_trees = max(int(lg_val.best_iteration or 200), 50)
    lg = Md.fit_lgb(train, feats, n_estimators=n_trees)
    p_lgb = lg.predict(test[feats])

    act = test.active_at_cutoff.values
    y = test.churn.values.astype(int)
    boot = bootstrap_compare(y[act], p[act], p_lgb[act])
    write_json({"threshold_from_validation": thr, "bootstrap_active_population": boot, "lgbm_trees": n_trees}, P.tables / "final_model_selection.json")
    cal = {pop: {"logistic regression": reliability(y[m], p[m]), "LightGBM": reliability(y[m], p_lgb[m])}
           for pop, m in [("all", np.ones(len(y), bool)), ("active", act)]}
    for pop in cal:
        for name, t in cal[pop].items():
            write_table(t.reset_index(drop=True), P.tables / f"calibration_{pop}_{name.split()[0].lower()}")
    pl.save(plot_calibration(cal), P.figures / "calibration.png")
    rows = [{"population": pop, **Md.metrics(y[m], p[m], thr)} for pop, m in [("all eligible", np.ones(len(y), bool)), ("active", act)]]
    write_table(pd.DataFrame(rows), P.tables / "final_model_test_metrics")
    g = Md.gains(y[act], p[act])
    write_table(g, P.tables / "gains_active")
    pl.save(plot_gains(g), P.figures / "gains_active.png")

    # prediction file for the held-out (official) cohort
    X = test[feats]
    contrib = lr_contributions(model, X)
    contrib.index = test.CUSTOMER_ID.values
    Xi = X.set_index(test.CUSTOMER_ID.values)
    out = pd.DataFrame({"customer_id": test.CUSTOMER_ID.values, "churn_probability": p.round(4),
                        "predicted_churn": (p >= thr).astype(int), "actual_churned": y,
                        "population": test.population.values, "membership_tier": test.MEMBERSHIP_TIER.astype(str).values,
                        "city": test.CITY.astype(str).values, "signup_cohort": test.signup_cohort.values})
    out["risk_band"] = pd.cut(out.churn_probability, [-0.01, 0.1, thr, 1.0], labels=["low", "medium", "high"]).astype(str)
    out["key_drivers"] = top_drivers(contrib, Xi).values
    out.sort_values("churn_probability", ascending=False).to_csv(P.predictions / "churn_predictions_apr_jun_2024.csv", index=False)
    contrib.to_parquet(P.predictions / "lr_contributions_test.parquet")

    # forward scoring: retrain on all four labelled snapshots, score members at 2024-06-30
    all_train = Md.stack(snaps, keys + [str(C.OFFICIAL_CUTOFF.date())])
    prod = Md.make_lr(feats).fit(all_train[feats], all_train.churn)
    sc = snaps[str(C.SCORING_CUTOFF.date())]
    ps = prod.predict_proba(sc[feats])[:, 1]
    c2 = lr_contributions(prod, sc[feats])
    c2.index = sc.CUSTOMER_ID.values
    fwd = pd.DataFrame({"customer_id": sc.CUSTOMER_ID.values, "churn_probability_jul_sep_2024": ps.round(4),
                        "predicted_churn": (ps >= thr).astype(int), "population": sc.population.values,
                        "membership_tier": sc.MEMBERSHIP_TIER.astype(str).values, "city": sc.CITY.astype(str).values})
    fwd["key_drivers"] = top_drivers(c2, sc[feats].set_index(sc.CUSTOMER_ID.values)).values
    fwd.sort_values("churn_probability_jul_sep_2024", ascending=False).to_csv(P.predictions / "churn_scores_jul_sep_2024.csv", index=False)

    import joblib
    P.models.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "features": feats}, P.models / "logreg_test.joblib")
    joblib.dump({"model": prod, "threshold": thr, "features": feats}, P.models / "logreg_production.joblib")
    log.info("threshold %.3f; bootstrap %s", thr, boot)
    log.info("\n%s", pd.DataFrame(rows)[["population", "ROC-AUC", "PR-AUC", "precision", "recall", "F1", "accuracy"]].round(3))


if __name__ == "__main__":
    setup_logging()
    run()
