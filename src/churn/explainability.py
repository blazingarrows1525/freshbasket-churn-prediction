"""Churn drivers: logistic-regression coefficients (primary, exact) and SHAP on the
LightGBM challenger (cross-check), plus predicted vs actual risk by segment.

Importance describes what the model uses to rank members; it does not say that
changing a feature would change a member's behaviour.

Run:  python -m churn.explain
"""
from __future__ import annotations

import logging

import lightgbm  # noqa: F401  (import before scikit-learn, see final_model.py)
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shap

from churn import config as C
from churn import feature_engineering as F
from churn import modeling as Md
from churn.final_model import NAMES
from churn import plotting as pl
from churn.utils import setup_logging, write_json, write_table

log = logging.getLogger(__name__)
P = C.P
GROUP_OF = {f: g for g, fs in F.GROUPS.items() for f in fs}


def lr_coefficients(model) -> pd.DataFrame:
    pre, lr = model.named_steps["pre"], model.named_steps["lr"]
    names = pre.get_feature_names_out()
    t = pd.DataFrame({"term": names, "coef_per_sd_or_level": lr.coef_[0]})
    t["odds_ratio"] = np.exp(t.coef_per_sd_or_level)
    t["feature"] = [n.split("__", 1)[1] for n in names]
    t["abs"] = t.coef_per_sd_or_level.abs()
    return t.sort_values("abs", ascending=False).drop(columns="abs")


def run():
    snaps = F.load_snapshots()
    keys = [str(c.date()) for c in C.TRAIN_CUTOFFS]
    train, test = Md.stack(snaps, keys), snaps[str(C.OFFICIAL_CUTOFF.date())]
    art = joblib.load(P.models / "logreg_test.joblib")
    model, feats = art["model"], art["features"]

    coefs = lr_coefficients(model)
    write_table(coefs, P.tables / "lr_coefficients")

    # mean |contribution| per original feature on the test snapshot, by population
    contrib = pd.read_parquet(P.predictions / "lr_contributions_test.parquet")
    act = test.active_at_cutoff.values
    imp = pd.DataFrame({"all eligible": contrib.abs().mean(), "active": contrib[act].abs().mean()})
    imp["feature_name"] = [NAMES.get(i, i) for i in imp.index]
    imp["group"] = [GROUP_OF.get(i, "?") for i in imp.index]
    imp = imp.sort_values("active", ascending=False)
    write_table(imp.reset_index(names="feature"), P.tables / "lr_driver_importance")
    grp = imp.groupby("group")[["all eligible", "active"]].sum()
    grp = grp / grp.sum()
    write_table(grp.reset_index(), P.tables / "lr_driver_group_share")

    # SHAP for LightGBM challenger (same training data, early-stopped tree count)
    sel = pd.read_json(P.tables / "final_model_selection.json", typ="series")
    lg = Md.fit_lgb(train, feats, n_estimators=int(sel["lgbm_trees"]))
    sv = shap.TreeExplainer(lg).shap_values(test[feats])
    sv = sv[1] if isinstance(sv, list) else sv
    shap_imp = pd.DataFrame({"feature": feats, "mean_abs_shap_active": np.abs(sv[act]).mean(axis=0),
                             "mean_abs_shap_all": np.abs(sv).mean(axis=0)})
    shap_imp["feature_name"] = shap_imp.feature.map(lambda f: NAMES.get(f, f))
    shap_imp = shap_imp.sort_values("mean_abs_shap_active", ascending=False)
    write_table(shap_imp, P.tables / "lgbm_shap_importance")
    rank_corr = imp["active"].rank().corr(shap_imp.set_index("feature").mean_abs_shap_active.rank(), method="spearman")
    top_lr, top_shap = set(imp.head(10).index), set(shap_imp.head(10).feature)
    write_json({"spearman_rank_correlation_active": float(rank_corr), "top10_overlap": len(top_lr & top_shap),
                "top10_common_features": sorted(top_lr & top_shap)}, P.tables / "driver_rank_agreement.json")

    # local explanations: why did the model score these members the way it did?
    preds_file = pd.read_csv(P.predictions / "churn_predictions_apr_jun_2024.csv").set_index("customer_id")
    thr = float(art["threshold"])
    ids = test.CUSTOMER_ID.values
    pa = preds_file.loc[ids, "churn_probability"].values
    is_act = test.active_at_cutoff.values
    cases = {
        "highest-risk active member": ids[is_act][np.argmax(pa[is_act])],
        "active member closest to the decision threshold": ids[is_act][np.argmin(np.abs(pa[is_act] - thr))],
        "lowest-risk active member": ids[is_act][np.argmin(pa[is_act])],
        "lapsed member (no purchase in the last 3 months)": ids[~is_act][np.argmax(pa[~is_act])],
    }
    Xi = test[feats].set_index(ids)
    rows = []
    for case, cid in cases.items():
        c = contrib.loc[cid]
        for f in c.abs().sort_values(ascending=False).index[:5]:
            v = Xi.loc[cid, f]
            rows.append({"case": case, "customer_id": int(cid), "churn_probability": float(preds_file.loc[cid, "churn_probability"]),
                         "actual_churned": int(preds_file.loc[cid, "actual_churned"]), "feature": f,
                         "feature_name": NAMES.get(f, f),
                         "value": round(float(v), 2) if isinstance(v, (int, float, np.number)) else str(v),
                         "contribution_logit": round(float(c[f]), 3)})
    write_table(pd.DataFrame(rows), P.tables / "local_explanations")

    # figure: top drivers for active members, LR vs SHAP side by side
    top = imp.head(12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {g: pl.CATEGORICAL[i] for i, g in enumerate(F.GROUPS)}
    y = np.arange(len(top))[::-1]
    axes[0].barh(y, top["active"], color=[colors[g] for g in top.group], height=0.6)
    axes[0].set_yticks(y, top.feature_name, fontsize=8.5)
    axes[0].set_title("Logistic regression: mean |contribution| (logit), active members", fontsize=10)
    axes[0].legend(handles=[Patch(color=c, label=g) for g, c in colors.items()], fontsize=8, loc="lower right")
    axes[0].grid(axis="y", visible=False)
    st = shap_imp.head(12)
    y = np.arange(len(st))[::-1]
    axes[1].barh(y, st.mean_abs_shap_active, color=[colors[GROUP_OF[f]] for f in st.feature], height=0.6)
    axes[1].set_yticks(y, st.feature_name, fontsize=8.5)
    axes[1].set_title(f"LightGBM: mean |SHAP| (log-odds), active members\nrank agreement with LR (Spearman) = {rank_corr:.2f}", fontsize=10)
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    pl.save(fig, P.figures / "drivers.png")

    # predicted vs actual churn by segment (active members, test quarter)
    preds = pd.read_csv(P.predictions / "churn_predictions_apr_jun_2024.csv")
    a = preds[preds.population.str.startswith("active")]
    rows = []
    for col in ["membership_tier", "city", "signup_cohort"]:
        t = a.groupby(col).agg(members=("customer_id", "size"), actual_rate=("actual_churned", "mean"),
                               mean_predicted=("churn_probability", "mean"),
                               share_high_risk=("predicted_churn", "mean")).reset_index().rename(columns={col: "segment"})
        t.insert(0, "dimension", col)
        rows.append(t)
    seg = pd.concat(rows, ignore_index=True)
    write_table(seg, P.tables / "risk_by_segment_active")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, dim in zip(axes, ["membership_tier", "signup_cohort", "city"]):
        t = seg[seg.dimension == dim].sort_values("actual_rate")
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.actual_rate * 100, width=0.38, color=pl.DEEMPH, label="actual churn")
        ax.bar(x + 0.2, t.mean_predicted * 100, width=0.38, color=pl.ACCENT, label="mean predicted risk")
        ax.set_xticks(x, t.segment.astype(str), rotation=30, fontsize=8)
        ax.set_title(dim.replace("_", " "), fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8)
    fig.suptitle("Active members, Apr-Jun 2024: predicted vs actual churn by segment", x=0.05, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    pl.save(fig, P.figures / "risk_by_segment.png")
    log.info("group share:\n%s\nrank corr LR vs SHAP: %.2f", grp.round(3), rank_corr)


if __name__ == "__main__":
    setup_logging()
    run()
