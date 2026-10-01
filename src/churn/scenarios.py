"""Retention what-if scenarios (model-based sensitivity, not causal effects).

BASELINE      the production model's churn risk for members active at 2024-06-30 (Jul-Sep 2024)
TARGET        the riskiest 20% of those members
INTERVENTION  each action is written as an explicit change to the member's recent behaviour
MODEL         the same logistic regression re-scores the changed members
CHANGE        mean predicted risk before vs after, and expected churners in the target group

How big is each change? Not invented: it is a share of the observed gap between retained and churning
active members in the four labelled quarters (e.g. retained members open ~0.7 more emails a month than
churners). The central case closes 50% of that gap; 25% and 100% are reported as sensitivity. A member is
never pushed beyond the retained-member average, and only the named features move (transactions are left
unchanged, which is conservative because engagement and purchases move together).

Why "not causal": the model learned that engaged members churn less. It did not learn what happens when
the business *causes* engagement - that needs a randomised hold-out test.

Run:  python -m churn.scenarios
"""
from __future__ import annotations

import logging

import joblib
import lightgbm  # noqa: F401  (import before scikit-learn: the reverse order crashes LightGBM on Windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn import config as C
from churn import feature_engineering as F
from churn import plotting as pl
from churn.utils import setup_logging, write_table

log = logging.getLogger(__name__)
P = C.P
TARGET_SHARE = 0.20
CENTRAL, SENSITIVITY = 0.5, (0.25, 0.5, 1.0)

# behaviour -> (3-month feature, 6-month feature, trend feature)
BEHAVIOURS = {"app sessions": ("app_avg_3m", "app_avg_6m", "app_trend"),
              "emails opened": ("email_avg_3m", "email_avg_6m", "email_trend"),
              "coupons redeemed": ("coupons_avg_3m", "coupons_avg_6m", "coupon_trend"),
              "support tickets": ("tickets_avg_3m", "tickets_avg_6m", "tickets_trend")}
ACTIONS = {
    "A. proactive support outreach": dict(moves=["support tickets"],
                                          who="members with at least one support ticket in the last 3 months",
                                          eligible=lambda d, ref: d.tickets_avg_3m > 0),
    "B. targeted coupon": dict(moves=["coupons redeemed"],
                               who="members redeeming fewer coupons than the retained-member average",
                               eligible=lambda d, ref: d.coupons_avg_3m < ref["coupons_avg_3m"]),
    "C. engagement outreach (app + email)": dict(moves=["app sessions", "emails opened"],
                                                 who="members below the retained-member average for app sessions or emails opened",
                                                 eligible=lambda d, ref: (d.app_avg_3m < ref["app_avg_3m"]) | (d.email_avg_3m < ref["email_avg_3m"])),
}


def behaviour_gaps(snaps) -> tuple[pd.DataFrame, pd.Series]:
    """Mean 3-month behaviour of retained vs churning active members, four labelled quarters pooled."""
    keys = [str(c.date()) for c in C.TRAIN_CUTOFFS + [C.OFFICIAL_CUTOFF]]
    d = pd.concat([snaps[k] for k in keys], ignore_index=True)
    d = d[d.active_at_cutoff]
    cols = [v[0] for v in BEHAVIOURS.values()]
    means = d.groupby("churn")[cols].mean().rename(index={0: "retained", 1: "churned"})
    tab = means.T.assign(gap_retained_minus_churned=lambda t: t.retained - t.churned)
    tab.insert(0, "behaviour", list(BEHAVIOURS))
    return tab.reset_index(names="feature"), means.loc["retained"]


def apply_action(d: pd.DataFrame, action: dict, gaps: pd.DataFrame, ref: pd.Series, share: float) -> tuple[pd.DataFrame, pd.Series]:
    d = d.copy()
    elig = action["eligible"](d, ref)
    for b in action["moves"]:
        f3, f6, ftr = BEHAVIOURS[b]
        gap = float(gaps.set_index("feature").loc[f3, "gap_retained_minus_churned"])
        step = share * gap
        x = d.loc[elig, f3]
        if step >= 0:   # raise towards the retained average, never beyond it
            new = x + np.minimum(step, np.maximum(ref[f3] - x, 0))
        else:           # reduce towards the retained average, never below it
            new = x - np.minimum(-step, np.maximum(x - ref[f3], 0))
        delta = new - x
        d.loc[elig, f3] = new
        d.loc[elig, f6] = np.maximum(d.loc[elig, f6] + delta / 2, 0)     # 3 of the 6 months change
        d.loc[elig, ftr] = d.loc[elig, ftr] + delta
    return d, elig


def run():
    snaps = F.load_snapshots()
    art = joblib.load(P.models / "logreg_production.joblib")
    model, feats, thr = art["model"], art["features"], art["threshold"]
    gaps, ref = behaviour_gaps(snaps)
    write_table(gaps, P.tables / "scenario_behaviour_gaps")

    sc = snaps[str(C.SCORING_CUTOFF.date())]
    act = sc[sc.active_at_cutoff].copy()
    act["p"] = model.predict_proba(act[feats])[:, 1]
    k = int(round(TARGET_SHARE * len(act)))
    order = act.sort_values("p", ascending=False)
    target, rest = order.head(k).copy(), order.iloc[k:]
    write_table(pd.DataFrame({
        "segment": ["all active members", f"targeted top {TARGET_SHARE:.0%}", "rest of active members"],
        "members": [len(act), k, len(rest)],
        "mean_predicted_risk": [act.p.mean(), target.p.mean(), rest.p.mean()],
        "expected_churners": [act.p.sum(), target.p.sum(), rest.p.sum()]}), P.tables / "retention_target_group")

    rows = []
    for name, action in ACTIONS.items():
        for share in SENSITIVITY:
            mod, elig = apply_action(target[feats], action, gaps, ref, share)
            p_new = model.predict_proba(mod)[:, 1]
            e = elig.values
            steps = "; ".join(f"{b} {share * float(gaps.set_index('behaviour').loc[b, 'gap_retained_minus_churned']):+.2f}/month"
                              for b in action["moves"])
            rows.append({"scenario": name, "gap_share_closed": share, "who": action["who"], "change_per_member": steps,
                         "members_targeted": k, "members_eligible_for_action": int(e.sum()),
                         "mean_risk_before_eligible": target.p.values[e].mean(), "mean_risk_after_eligible": p_new[e].mean(),
                         "change_pp_eligible": (p_new[e].mean() - target.p.values[e].mean()) * 100,
                         "expected_churners_before_target": target.p.sum(), "expected_churners_after_target": p_new.sum(),
                         "expected_churners_avoided_model_based": target.p.sum() - p_new.sum(),
                         "moved_below_threshold": int(((target.p.values >= thr) & (p_new < thr))[e].sum())})
    allres = pd.DataFrame(rows)
    write_table(allres, P.tables / "retention_scenario_sensitivity")
    res = allres[allres.gap_share_closed == CENTRAL].drop(columns="gap_share_closed").reset_index(drop=True)
    write_table(res, P.tables / "retention_scenarios")
    res.to_csv(P.predictions / "retention_scenarios.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(res))[::-1]
    ax.barh(y + 0.2, res.mean_risk_before_eligible * 100, height=0.38, color=pl.DEEMPH, label="predicted risk before")
    ax.barh(y - 0.2, res.mean_risk_after_eligible * 100, height=0.38, color=pl.ACCENT, label="under the assumption (50% of gap)")
    for yy, b, a, n in zip(y, res.mean_risk_before_eligible * 100, res.mean_risk_after_eligible * 100, res.members_eligible_for_action):
        ax.text(max(a, b) + 1, yy, f"{a - b:+.1f} pp  (n={n})", va="center", fontsize=8.5, color=pl.INK_2)
    ax.set_yticks(y, res.scenario)
    ax.set_xlabel("mean predicted churn probability of members the action applies to (%)")
    ax.set_title(f"What-ifs for the riskiest {TARGET_SHARE:.0%} of active members (Jul-Sep 2024 scores)", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(res.mean_risk_before_eligible.max() * 100 + 22, 60))
    ax = axes[1]
    for i, name in enumerate(ACTIONS):
        t = allres[allres.scenario == name]
        ax.plot(t.gap_share_closed * 100, t.expected_churners_avoided_model_based, marker="o", color=pl.CATEGORICAL[i],
                label=name.split(" (")[0])
    ax.set_xlabel("share of the retained-vs-churned behaviour gap closed (%)")
    ax.set_ylabel("expected churners avoided (model-based)")
    ax.set_title("Sensitivity to the size of the assumed change", fontsize=10.5)
    ax.set_xticks([25, 50, 100])
    ax.legend(fontsize=8)
    fig.tight_layout()
    pl.save(fig, P.figures / "retention_scenarios.png")
    log.info("\n%s", res.drop(columns=["who", "change_per_member"]).round(3).to_string())


if __name__ == "__main__":
    setup_logging()
    run()
