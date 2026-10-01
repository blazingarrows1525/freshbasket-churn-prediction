"""Builds and executes the six notebooks from the pipeline outputs.

The notebooks are the readable walk-through. All logic lives in src/churn; notebooks 01-05 only read the
tables, figures and files the pipeline wrote (all committed), so they re-run without the raw data.
Notebook 06 is the live demo and needs the pipeline to have run once (it loads the trained models).

Run:  python notebooks/build_notebooks.py          (all)
      python notebooks/build_notebooks.py 03 06    (some)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
KERNEL = "freshbasket-churn"
SETUP = """%matplotlib inline
from churn.notebook import *   # pandas as pd, numpy as np, display, Markdown, fig(), tab(), run_step(), paths P"""


def md(s):
    return nbformat.v4.new_markdown_cell(s.strip())


def code(s):
    return nbformat.v4.new_code_cell(s.strip())


def chart_notes(shows, why, learned, implies):
    return md(f"""**What does this show?** {shows}

**Why is it relevant?** {why}

**What did we learn?** {learned}

**What does it imply?** {implies}""")


NB = {}

# ================================================================================================ 01
NB["01_data_quality_eda"] = [
    md("""# 01 - Data quality and exploratory analysis
FreshBasket loyalty workbook: member profiles, 18 months of monthly activity (Jan 2023 - Jun 2024) and the churn
label (no purchase in Apr-Jun 2024 despite earlier purchases). Code: `src/churn/data_validation.py`,
`src/churn/preprocessing.py`, `src/churn/eda.py`.

> **At a glance**
> * 2,608 profile rows, 28,073 activity rows, 2,605 label rows for 2,600 members; all three tables join on `CUSTOMER_ID`.
> * Spend errors could be repaired exactly, because `TOTAL_SPEND = TRANSACTIONS x AVG_BASKET_VALUE` on 99.8% of
>   purchase months.
> * Three label-table columns are computed over the label window and would leak the answer; they are never used.
> * **80% of the Apr-Jun 2024 churners had already stopped buying before the label window.** Among members still
>   buying, churn is a steady ~10% per quarter. That reframes the whole problem."""),
    code(SETUP),
    md("## 1. What was supplied (before any cleaning)"),
    code("""display(tab("table_summary"))
tab("integrity_checks")"""),
    md("""The three tables share `CUSTOMER_ID` and contain exactly the same 2,600 members. Profile and label are one row
per member once exact duplicates are dropped; activity is one row per member-month. Six members have no activity at
all, and only 400 members have a row in every month (late sign-ups, lapsed members and deleted rows). So the
modelling dataset is built at **member x snapshot** grain rather than with a raw merge, which would repeat member
attributes once per month."""),
    md("## 2. Data-quality findings (every treatment is logged, nothing is cleaned silently)"),
    code("""tab("data_quality_findings")"""),
    md("""The brief lists the issues to look for (missing months, duplicates, invalid/negative spend, zero-transaction
anomalies, inconsistent casing, outlier spend). All of them are present. The most useful discovery is the spend
identity: spend equals transactions x average basket on 99.8% of purchase months. That made it possible to **diagnose**
each spend error rather than guess:
* the 22 negative spends are pure sign errors (their absolute value matches the identity);
* the 13 extreme values are mistyped totals (e.g. 3,731 where the identity gives 297), not big shoppers.

The label table's `LAST_PURCHASE_DATE`, `MONTHS_OBSERVED` and `INSUFFICIENT_HISTORY_FLAG` are computed over the label
window (a last purchase before April implies churn), so they are excluded. Rebuilding `CHURNED` from the activity with
the brief's definition agrees for 99.65% of members; all 9 disagreements are members whose activity row for their
last-purchase month was deleted."""),
    md("## 3. Who churns? Two very different populations"),
    code("""fig("eda_churn_by_population", 750)
tab("snapshots_target_distribution")"""),
    chart_notes("Churn rate in the 3 months after each quarterly cutoff, split by whether the member bought anything in "
                "the 3 months before the cutoff.",
                "The brief's label counts every member who stops buying, including members who stopped long ago.",
                "Members who were already lapsed 'churn' at 97-99% in every quarter. Members still buying churn at "
                "9.6-10.3%, every quarter. The headline churn rate rises (18% -> 34%) only because lapsed members accumulate.",
                "Predicting the lapsed group is trivial and too late for a retention offer. The real early-warning problem "
                "is the active members, so every result is reported for both populations."),
    md("## 4. Does churn differ by tier, city or signup cohort?"),
    code("""fig("eda_segment_rates", 1000)
s = tab("eda_segment_rates"); s[s.population == "active"].sort_values(["attribute", "churn_rate"])"""),
    chart_notes("Apr-Jun 2024 churn rate by membership tier, signup cohort and city, for all eligible and for active members.",
                "The brief asks whether tier and other segments affect churn risk, and the business currently segments by tier.",
                "Among active members tier makes almost no difference (Silver 9.5%, Gold 9.8%, Platinum 9.4%). Cities range "
                "7.3-11.4% on 150-200 members each (small differences). Members who joined in 2024 churn least (3.2%).",
                "Tier-based retention budgets are not supported by the data; behaviour has to do the targeting."),
    md("## 5. Engagement, support and purchase trend vs churn (active members)"),
    code("""fig("eda_engagement_support", 1100)
tab("eda_engagement_support")"""),
    chart_notes("Churn rate of active members by their last-3-month app sessions, emails opened, coupons redeemed, support "
                "tickets and change in transactions (4 labelled quarters pooled).",
                "The brief asks for the observed relationship between engagement, support experience and churn.",
                "Strong gradients: 43% churn with <= 0.5 app sessions a month vs 0.5% above 3; 30% churn with more than one "
                "ticket every three months vs 5.6% with none; 26% churn when transactions fall by more than 2 a month.",
                "Engagement, support and purchase-trend features should carry the signal. These are associations: engaged "
                "members may simply be the ones who were going to stay."),
    md("## 6. Do churners fade before they leave?"),
    code("""fig("eda_trajectories", 1100)
tab("eda_trajectories")"""),
    chart_notes("Average monthly activity of members active at the March 2024 cutoff, in the 6 months before it, split by "
                "whether they churned in Apr-Jun.",
                "If churn builds up gradually, a model using only pre-cutoff data can see it coming.",
                "Churners fade for months: transactions 3.6 -> 0.9 a month, app sessions 1.8 -> 0.3, emails 1.0 -> 0.3, while "
                "support tickets rise 0.17 -> about 0.5. Retained members stay flat (about 4.7 transactions a month).",
                "There is a window to act before the last purchase, and trend features (last 3 months vs the 3 before) "
                "should help."),
    md("""## Takeaways for modelling
* Build features only from months up to the cutoff; exclude the label-window columns.
* Report lapsed and active members separately; the active members are the retention problem.
* Recency, purchase trend, app/email engagement and support tickets are the signals to model; demographics and tier
  look weak."""),
]

# ================================================================================================ 02
NB["02_feature_engineering"] = [
    md("""# 02 - Feature engineering and leakage control
Code: `src/churn/feature_engineering.py`.

> **At a glance**
> * One row per **member x quarterly snapshot**: features from the 6 months up to the cutoff, label from the 3 months
>   after it. Five snapshots: three training quarters (labels rebuilt), the official test quarter (the given
>   `CHURNED`), and one scoring quarter (future unknown).
> * 36 features in 4 groups: demographics (8), behavioural (14), engagement (10), support (4).
> * A leakage test rebuilds every snapshot from data truncated at its cutoff: **no feature changes** (maximum
>   difference 0.0)."""),
    code(SETUP),
    md("## 1. Snapshot design"),
    code("""fig("eda_timeline", 1000)
tab("snapshots_target_distribution")"""),
    chart_notes("The feature window (blue, only data up to the cutoff) and label window (orange, the next 3 months) of each "
                "snapshot.",
                "This is how the model will be used: on the last day of a quarter, score every member for the next quarter.",
                "Only one quarter has the given label. Rebuilding the same label for three earlier quarters (99.65% agreement "
                "where both exist) gives honest out-of-time training data whose label windows all end before the test cutoff.",
                "Training on earlier quarters and testing on Apr-Jun 2024 mimics deployment; no random split is needed."),
    md("## 2. Feature catalogue"),
    code("""tab("feature_catalogue")"""),
    md("""Design choices:
* **Fixed windows** (3 and 6 months) and recency capped at 6 months, so the first snapshot (6 months of history) and
  the last (15 months) produce comparable features.
* **Trends** (last 3 months minus the 3 before) capture fading, which the EDA showed is the key pattern.
* **Missing months are unknown, not zero.** A month without a row is a deleted record (gap) if the member has a later
  row up to the cutoff; otherwise it is inactivity. That decision uses rows up to the cutoff only.
* **Usefulness** columns come from the fitted models (mean absolute logistic contribution and LightGBM SHAP); the
  ablation in notebook 04 tests the groups formally."""),
    md("## 3. Leakage test"),
    code("""tab("leakage_check")"""),
    md("""For each labelled snapshot the features are rebuilt after deleting **every activity row after the cutoff** (up to
18,975 rows). If any feature had looked into the future, directly or through the gap-vs-inactivity decision, the values
would change. None does. In addition, the code asserts that the label table's post-outcome columns are never in the
feature list."""),
    md("## 4. The modelling dataset (deliverable)"),
    code("""ds = pd.read_csv(P.processed / "customer_modelling_dataset.csv")
print(ds.shape)
display(ds.groupby("snapshot").agg(members=("CUSTOMER_ID", "size"), churn_rate=("churn", "mean"),
                                   active_share=("active_at_cutoff", "mean")).round(3))
ds.head()"""),
]

# ================================================================================================ 03
NB["03_modeling"] = [
    md("""# 03 - Baseline, candidate models, validation and final model
Code: `src/churn/modeling.py`, `src/churn/final_model.py`.

> **At a glance**
> * Out-of-time validation: tune on the Jan-Mar 2024 quarter, test on Apr-Jun 2024 (the given label), never a random split.
> * Baseline recency rule: active PR-AUC 0.514. Logistic regression: **0.828** (95% CI 0.775-0.873). LightGBM 0.823 -
>   a statistical tie (bootstrap PR-AUC difference +0.005, CI -0.019 to +0.029).
> * Stable over time: across three consecutive test quarters the logistic regression averages PR-AUC 0.824 +/- 0.019 and
>   beats every other model in each quarter.
> * Selected: logistic regression - equally accurate, best calibrated, and every score splits into exact per-member reasons.
> * Contacting the riskiest 10% of active members reaches **76%** of the quarter's churners at 73% precision."""),
    code(SETUP),
    md("""## 1. Validation design
| Step | Train on snapshots | Evaluate on | Used for |
|---|---|---|---|
| Tuning | 2023-06 + 2023-09 | 2023-12 (label Jan-Mar 2024) | thresholds, tree counts, regularisation check |
| Test | 2023-06 + 2023-09 + 2023-12 | 2024-03 (label Apr-Jun 2024, the given `CHURNED`) | all reported results |
| Production scoring | all four labelled snapshots | 2024-06-30 members (Jul-Sep 2024) | forward scores, scenarios |

* **Available at prediction time:** everything up to the cutoff month. **Not available:** anything after it,
  including the label table's last-purchase and months-observed columns.
* **Why this mimics the business:** the model is retrained on past quarters and scores the next one.
* **What could still be optimistic:** one test quarter; rebuilt training labels (~0.4% noise); the data look synthetic
  (signals are unusually clean), so live performance should be re-checked."""),
    md("## 2. Baseline and candidate models on the test quarter"),
    code("""m = tab("model_comparison")
m[["population", "model", "val PR-AUC", "ROC-AUC", "PR-AUC", "PR-AUC CI low", "PR-AUC CI high", "Brier", "threshold",
   "precision", "recall", "F1", "accuracy"]].round(3)"""),
    md("""* The **recency rule** (risk rises with months since the last purchase) is the baseline. It is already excellent
  on all eligible members (ROC-AUC 0.971) because lapsed members are easy, but weak on active members (PR-AUC 0.514).
* **Accuracy is not used to choose:** predicting "no churn" for every active member is already 90.4% accurate.
* **Class imbalance** (about 1 churner in 10 active members) is handled by choosing the decision threshold on the
  validation quarter. Class weights were tried: no ranking gain and worse calibration (higher Brier). No resampling."""),
    code("""tab("model_selection_summary")"""),
    md("## 3. Is the result stable over time? (rolling origin, 3 test quarters)"),
    code("""ro = tab("rolling_origin_stability")
display(ro.pivot_table(index="model", columns="test_snapshot", values="PR-AUC active").round(3))
tab("rolling_origin_summary").round(3)"""),
    md("""Each quarter is predicted by a model trained only on the quarters before it. The logistic regression is the best
model in every quarter (PR-AUC 0.803-0.841) and the most stable; the gap to the rule baseline is large and consistent."""),
    md("## 4. Hyperparameters: does the regularisation strength matter?"),
    code("""tab("lr_regularisation_sweep").round(4)"""),
    md("""Checked on the validation quarter only: validation PR-AUC moves by less than 0.012 across a 1,000-fold range of C,
much less than the bootstrap uncertainty (about +/- 0.05). The pre-set C = 0.3 is kept; heavier tuning on ~1,700 active
members per quarter would mostly fit noise. LightGBM's tree count comes from early stopping on the validation quarter."""),
    md("## 5. Logistic regression vs LightGBM, and a specialist model"),
    code("""import json
print(json.load(open(P.tables / "final_model_selection.json")))
tab("model_comparison_active_trained")[["model", "ROC-AUC", "PR-AUC", "F1", "Brier"]].round(3)"""),
    md("""The paired bootstrap puts the PR-AUC difference between the two models at +0.005 with a 95% interval from -0.019 to
+0.029: a tie. When two models tie, the simpler, calibrated and explainable one wins. A model trained only on active
members did not do better (PR-AUC 0.818), so one model serves both populations."""),
    md("## 6. Final model on the held-out quarter"),
    code("""display(tab("final_model_test_metrics").round(3))
fig("calibration", 950)"""),
    chart_notes("Mean predicted churn probability vs observed churn rate by decile, for all eligible and active members.",
                "A retention list is only useful if a 70% score really means about 70% churn.",
                "Both models track the diagonal; for the top decile of active members the logistic regression predicts 72.4% "
                "and 72.7% churn. Class weighting would have pushed probabilities up (Brier 0.043 vs 0.034).",
                "Scores can be read as probabilities, which the scenario analysis relies on."),
    code("""display(tab("gains_active").round(3))
fig("gains_active", 620)"""),
    chart_notes("Share of the test quarter's churners reached when contacting the riskiest x% of active members.",
                "The business decision is how many members to contact, not which threshold maximises F1.",
                "The riskiest 5% contain 49% of churners, the riskiest 10% contain 76% (precision 73%, 7.6x random), the "
                "riskiest 20% contain 91%.",
                "A monthly list of the top 10-20% of active members replaces the blanket offer."),
    md("## 7. The prediction file (deliverable)"),
    code("""p = pd.read_csv(P.predictions / "churn_predictions_apr_jun_2024.csv")
print(p.shape); p.head(8)"""),
    md("## 8. Experiment log (including rejected experiments)"),
    code("""tab("experiment_log")"""),
]

# ================================================================================================ 04
NB["04_explainability"] = [
    md("""# 04 - Explainability: global drivers, individual members, segments and ablation
Code: `src/churn/explainability.py`, `src/churn/ablation.py`.

> **At a glance**
> * Behaviour (40%) and engagement (37%) drive the logistic regression's scores for active members; demographics
>   15%, support 7%.
> * LightGBM SHAP agrees on the main drivers (rank correlation 0.59 across all 36 features).
> * Every member's score splits exactly into feature contributions, so each prediction comes with its reasons.
> * Ablation: demographics alone are useless for active members (ROC-AUC 0.51); behaviour lifts it to 0.93,
>   engagement to 0.96."""),
    code(SETUP),
    md("## 1. Global explanation: what generally drives the predictions?"),
    code("""fig("drivers", 1100)
display(tab("lr_driver_group_share").round(3))
import json; print(json.load(open(P.tables / "driver_rank_agreement.json")))"""),
    chart_notes("Left: average absolute contribution of each feature to the logistic regression's log-odds for active "
                "members. Right: LightGBM mean absolute SHAP values on the same members.",
                "The brief asks for the strongest churn drivers, and the business needs to know what to watch.",
                "Months since the last purchase, transactions in the last 3 months and their trend, emails opened, app "
                "sessions and support tickets lead in both models.",
                "The signals are behavioural and observable every month, so they can drive a monthly risk list."),
    code("""c = tab("lr_coefficients"); c.head(15).round(3)"""),
    md("""Odds ratios are per standard deviation (numeric features) or vs the reference level (categories): a falling
transaction trend, fewer emails and app sessions raise risk; more support tickets raise it (OR 1.73 per sd).
**Marketing opt-in** has an odds ratio above 1 (1.51) even though opted-in members churn less in the raw data (8.5% vs
11.4%). The coefficient is conditional on emails opened: an opted-in member who stops opening emails is disengaging. A
coefficient describes an association inside the model, not the effect of changing the variable."""),
    md("## 2. Local explanation: why did the model score these members the way it did?"),
    code("""le = tab("local_explanations")
for case, g in le.groupby("case", sort=False):
    r = g.iloc[0]
    display(Markdown(f"**{case}** - member {r.customer_id}: predicted {r.churn_probability:.1%}, actually churned: {bool(r.actual_churned)}"))
    display(g[["feature_name", "value", "contribution_logit"]])"""),
    md("""Each contribution is in log-odds relative to an average member: positive pushes towards churn. The prediction file
lists the three largest risk-raising contributions for every member (`key_drivers`), so a retention agent sees *why*
someone is on the list, e.g. "support tickets/month (3m) = 1.3; transaction trend = -6.3"."""),
    md("## 3. Risk across tier, signup cohort and city"),
    code("""fig("risk_by_segment", 1100)
tab("risk_by_segment_active").round(3)"""),
    chart_notes("Actual churn vs mean predicted risk by segment, active members, test quarter.",
                "The brief asks to compare risk across tiers, cities and signup cohorts.",
                "Predicted and actual churn match closely by tier and city. The model over-predicts the small 2024 cohort "
                "(actual 3.2%, predicted 6.1%, 124 members).",
                "The model is not biased towards any tier; segment differences are small compared with behavioural ones."),
    md("## 4. Ablation: do behaviour, engagement and support add value over demographics?"),
    code("""fig("ablation", 800)
tab("ablation").round(3)"""),
    chart_notes("Test-quarter ROC-AUC and PR-AUC as feature groups are added (logistic regression; LightGBM in the table).",
                "The brief asks whether engagement, support and behavioural features improve accuracy over demographics alone.",
                "Demographics alone: active PR-AUC 0.095, i.e. the 9.6% base rate - no signal. + behavioural 0.687, "
                "+ engagement 0.812, + support 0.828. Engagement alone (with demographics) reaches 0.711.",
                "Collecting and refreshing behavioural and engagement data matters far more than demographic detail."),
]

# ================================================================================================ 05
NB["05_scenario_analysis"] = [
    md("""# 05 - Retention scenarios (model-based, not causal)
Code: `src/churn/scenarios.py`.

> **At a glance**
> * Population: the 1,657 members active on 2024-06-30, scored for Jul-Sep 2024. The riskiest 20% (331 members) hold
>   131 of the 140 expected churners.
> * Each action changes the member's recent behaviour by **half the observed gap between retained and churning
>   members** (sensitivity: 25% and 100%), never beyond the retained-member average.
> * Central case: engagement outreach -10.2 pp predicted risk (about 34 fewer expected churners in the target group);
>   targeted coupon -2.7 pp; proactive support -3.4 pp.
> * These numbers rank what to **test**; they are not what the actions will achieve."""),
    code(SETUP),
    md("## 1. Baseline: where the risk sits"),
    code("""tab("retention_target_group").round(3)"""),
    md("## 2. How big is each assumed change? Grounded in the data"),
    code("""tab("scenario_behaviour_gaps").round(3)"""),
    md("""Retained active members open about 0.65 more emails, have about 1.6 more app sessions and redeem about 0.36 more
coupons a month than members who churn, and raise fewer tickets (0.13 vs 0.35 a month). The central scenario assumes an
action closes **half** of that gap for the members it applies to. This replaces arbitrary magnitudes ("+1 coupon")
with a stated, data-based assumption that can be changed."""),
    md("## 3. Results"),
    code("""r = tab("retention_scenarios")
display(r[["scenario", "who", "change_per_member", "members_eligible_for_action", "mean_risk_before_eligible",
           "mean_risk_after_eligible", "change_pp_eligible", "expected_churners_avoided_model_based"]].round(3))
fig("retention_scenarios", 1100)"""),
    chart_notes("Left: mean predicted churn risk of the members each action applies to, before and under the assumption. "
                "Right: expected churners avoided (model-based) as the assumed share of the behaviour gap changes.",
                "The brief asks for a retention scenario and the estimated change in churn risk.",
                "Engagement outreach moves the model's risk most (-10.2 pp for all 331 targeted members). Coupons and support "
                "outreach move it less (-2.7 and -3.4 pp), partly because they apply to fewer members and the gaps are smaller.",
                "Test engagement outreach first, with support outreach for members with open tickets as a second arm."),
    code("""s = tab("retention_scenario_sensitivity")
s.pivot_table(index="scenario", columns="gap_share_closed", values="change_pp_eligible").round(2)"""),
    md("""## 4. What these numbers do and do not mean
* **Do:** they show how the fitted model's risk responds if a member's recent behaviour looked like a more engaged
  member's, under explicit assumptions. That is enough to rank which actions to test first and on whom.
* **Do not:** they are not causal effects. The model learned that engaged members churn less; it did not learn what
  happens when the business *causes* engagement (a coupon might be redeemed by people who would have stayed anyway).
* **How to measure the real effect:** a randomised test inside the top-risk group with a 20-30% hold-out, comparing
  actual churn after one quarter. No offer costs or member margins were supplied, so no ROI is claimed."""),
]

# ================================================================================================ 06
NB["06_live_demo"] = [
    md("""# 06 - Live demo (2-3 minutes)
1. Retrain and score the held-out quarter from code. 2. Recompute the targeting result from the prediction file.
3. Explain one member's score. 4. Ask the model a retention what-if.

**Before recording:** run `python run_pipeline.py --no-notebooks` once (about 1.5 minutes): this notebook loads the
trained models. Then *Kernel -> Restart Kernel and Run All Cells*. Values in CAPITALS can be changed live."""),
    code(SETUP + """
import joblib
import matplotlib.pyplot as plt
from churn import config as C, feature_engineering as FE, final_model as FM, plotting as pl, scenarios as SC"""),
    md("## 1. Retrain on three earlier quarters and score Apr-Jun 2024"),
    code("""run_step("churn.final_model")
tab("final_model_test_metrics")[["population", "n", "churn_rate", "ROC-AUC", "PR-AUC", "precision", "recall", "F1"]].round(3)"""),
    md("## 2. Targeting value, recomputed from the prediction file (active members)"),
    code("""TOP_SHARE = 0.10
p = pd.read_csv(P.predictions / "churn_predictions_apr_jun_2024.csv")
act = p[p.population.str.startswith("active")].sort_values("churn_probability", ascending=False)
top = act.head(round(TOP_SHARE * len(act)))
print(f"{len(act):,} active members, churn rate {act.actual_churned.mean():.1%}")
print(f"contacting the riskiest {TOP_SHARE:.0%} ({len(top)} members) reaches {top.actual_churned.sum() / act.actual_churned.sum():.0%} "
      f"of churners at {top.actual_churned.mean():.0%} precision")
act.head(5)[["customer_id", "churn_probability", "risk_band", "membership_tier", "key_drivers"]]"""),
    md("## 3. Why is this member at risk? Exact logistic-regression contributions"),
    code("""CUSTOMER_ID = int(act.customer_id.iloc[0])
snap = FE.load_snapshots()[str(C.OFFICIAL_CUTOFF.date())].set_index("CUSTOMER_ID")
art = joblib.load(P.models / "logreg_test.joblib")
X = snap.loc[[CUSTOMER_ID], art["features"]]
c = FM.lr_contributions(art["model"], X).iloc[0]
c = c[c.abs().sort_values(ascending=False).index[:8]].iloc[::-1]
fig_, ax = plt.subplots(figsize=(8, 3.6))
ax.barh([FM.NAMES.get(k, k) for k in c.index], c.values, color=[pl.SECOND if v > 0 else pl.ACCENT for v in c.values])
ax.axvline(0, color=pl.BASELINE, lw=1)
ax.set_xlabel("contribution to churn log-odds (right = raises risk)")
ax.set_title(f"Member {CUSTOMER_ID}: churn probability {art['model'].predict_proba(X)[:, 1][0]:.1%}")
plt.show()"""),
    md("""## 4. Live what-if for the riskiest active members (Jul-Sep 2024 scores)
The model learned that engaged members churn less; it did not learn what happens when we *cause* engagement. These
numbers rank what to test with a hold-out group."""),
    code("""ACTION = "C. engagement outreach (app + email)"   # or "A. proactive support outreach", "B. targeted coupon"
GAP_SHARE = 0.5                                   # share of the retained-vs-churned behaviour gap closed (0.25 / 0.5 / 1.0)
TARGET_SHARE = 0.20
snaps = FE.load_snapshots()
prod = joblib.load(P.models / "logreg_production.joblib")
gaps, ref = SC.behaviour_gaps(snaps)
sc = snaps[str(C.SCORING_CUTOFF.date())]
now = sc[sc.active_at_cutoff].copy()
now["p"] = prod["model"].predict_proba(now[prod["features"]])[:, 1]
target = now.sort_values("p", ascending=False).head(round(TARGET_SHARE * len(now)))
changed, eligible = SC.apply_action(target[prod["features"]], SC.ACTIONS[ACTION], gaps, ref, GAP_SHARE)
p_new = prod["model"].predict_proba(changed)[:, 1]
e = eligible.values
print(f"targeted {len(target)} of {len(now)} active members; the action applies to {e.sum()} ({SC.ACTIONS[ACTION]['who']})")
print(f"mean predicted risk of those members: {target.p.values[e].mean():.1%} -> {p_new[e].mean():.1%}")
print(f"expected churners in the target group: {target.p.sum():.0f} -> {p_new.sum():.0f} (model-based, not causal)")"""),
]


def build(names=None):
    for name, cells in NB.items():
        if names and not any(name.startswith(n) for n in names):
            continue
        nb = nbformat.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "Python (freshbasket-churn)", "language": "python"}
        NotebookClient(nb, timeout=900, kernel_name=KERNEL, resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("built", name)


if __name__ == "__main__":
    build(sys.argv[1:] or None)
