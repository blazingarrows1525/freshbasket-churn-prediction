# FreshBasket loyalty churn prediction

**Demo video:** [presentation and live notebook demo (Google Drive)](https://drive.google.com/file/d/1pNMlP2WHOChuOY9t0UF2CBDmSilBRGpK/view?usp=sharing)

Tailwyndz Propel 2026 data-science assessment - **Assessment 4: Customer Churn Prediction for a Retail Loyalty
Program**. Python, reproducible end to end with one command (`python run_pipeline.py`, about 3 minutes).

| | |
|---|---|
| **Question** | Which loyalty members will stop shopping next quarter, why, and which retention actions are worth testing? |
| **Key finding** | 80% of the members labelled as churned had already stopped buying before the label window. For members still buying, churn is a steady ~10% per quarter, and that is the real early-warning problem. |
| **Model** | Logistic regression on 36 leakage-safe features, chosen over LightGBM after a statistical tie |
| **Result (held-out quarter, active members)** | PR-AUC **0.83** (95% CI 0.78-0.87) vs 0.51 for a recency rule; ROC-AUC 0.96; well calibrated |
| **Business value** | Contacting the riskiest **10%** of active members reaches **76%** of the quarter's churners (precision 73%, 7.6x random) |
| **Report / slides** | [`reports/final_report.md`](reports/final_report.md) - [`presentation/`](presentation/) |

---

## 1. Project title

**Finding loyalty members before they stop shopping: churn prediction, drivers and retention scenarios for
FreshBasket Retail.**

## 2. Business problem

FreshBasket runs a three-tier loyalty programme (Silver, Gold, Platinum). Churn is rising and every member receives
the same retention offer. That is expensive, and it mostly reaches members who were never going to leave. The
business has no systematic way to spot at-risk members early or to know which action might keep them.

## 3. Objective

1. Predict, for every member, the probability of churning in the next 3 months (churn = no purchase in the window
   despite earlier purchases - the brief's definition).
2. Explain the drivers globally and for each member, and compare risk across tier, city and signup cohort.
3. Test whether behaviour, engagement and support data add value over demographics (ablation).
4. Estimate, under explicit assumptions, how retention actions would change predicted risk.
5. Turn this into retention and marketing recommendations.

Success is measured on how well churners are found (PR-AUC, precision, recall, F1), not accuracy: among active
members, predicting "nobody churns" is already 90% accurate.

## 4. Dataset description

One workbook, three sheets, joined on `CUSTOMER_ID` (full inventory: [`reports/data_inventory.md`](reports/data_inventory.md)):

| Sheet | Rows | Members | Content |
|---|---|---|---|
| fb Customer Profile | 2,608 | 2,600 | signup date, age, gender, city, tier, marketing opt-in, preferred store/category, home-store price tier |
| fb Monthly Activity | 28,073 | 2,594 | Jan 2023 - Jun 2024 (18 months): transactions, spend, basket value, categories, promo %, app sessions, emails opened, coupons, support tickets, complaints |
| fb Churn Label | 2,605 | 2,600 | `CHURNED` (target, Apr-Jun 2024) plus observation end, last purchase date, months observed, insufficient-history flag |

The same 2,600 members appear in all three tables (0 orphan IDs). Six members have no activity at all. Activity is
one row per member-month, so the modelling dataset is built at **member x quarter-end snapshot** grain (10,776 rows,
5 snapshots) instead of a raw merge, which would repeat each profile up to 18 times.

## 5. Data-quality handling

Every issue is logged with its evidence, severity, treatment and reason
([`outputs/tables/data_quality_findings.md`](outputs/tables/data_quality_findings.md)). Nothing is cleaned silently.

| Issue | Evidence | Treatment |
|---|---|---|
| Spend identity | `TOTAL_SPEND = TRANSACTIONS x AVG_BASKET_VALUE` on 99.8% of purchase months | used to diagnose the spend errors below |
| Negative spend | 22 rows; the absolute value matches the identity in 100% of them | sign corrected |
| Outlier spend | 13 rows > 2x the identity (e.g. 3,731 vs 297; 99.9th percentile otherwise 634) | rebuilt from transactions x basket (a mistyped total, not a big shopper) |
| Zero spend in purchase months | 20 rows | rebuilt from transactions x basket |
| Duplicates | 8 profile, 58 activity, 5 label exact duplicates; 2 member-months entered twice with opposite spend signs | dropped; positive-spend row kept |
| Inconsistent text | CITY 40 spellings -> 10 cities; tier 12 -> 3 | trimmed and title-cased |
| Missing demographics | age 3.0%, gender 2.0%, home-store tier 5.0% | explicit "Unknown"; age median + missing indicator for the logistic regression |
| Missing months | 953 member-months missing between existing rows (769 members) | treated as unknown (deleted records), decided from data up to each cutoff only |
| Zero-transaction months | 2,330 rows, 74% still show app/email/support activity | kept: genuine non-purchase months |
| **Leaky label columns** | `LAST_PURCHASE_DATE`, `MONTHS_OBSERVED`, `INSUFFICIENT_HISTORY_FLAG` are computed over the label window | **never used**; honest equivalents recomputed up to each cutoff |
| Label check | rebuilding `CHURNED` from activity agrees for 99.65% of members; all 9 mismatches are deleted rows | confirms the definition; the given label stays the ground truth |

## 6. Methodology

```
raw workbook -> Phase-0 inventory -> cleaning + quality log -> member x month grid
  -> quarterly snapshots (features <= cutoff | label = next 3 months) -> leakage test
  -> baseline + candidate models -> out-of-time validation + rolling stability check
  -> final model, threshold, calibration, gains -> drivers, ablation -> retention what-ifs
```

Every step is a module in `src/churn/` and runs in its own process from `run_pipeline.py`. Every table and figure
cited in the report is written by the pipeline (`outputs/tables`, `outputs/figures`).

## 7. Feature engineering

36 features in four groups; full catalogue with definitions, business meaning, leakage control and measured
usefulness: [`outputs/tables/feature_catalogue.md`](outputs/tables/feature_catalogue.md).

| Group | n | Examples |
|---|---|---|
| Behavioural | 14 | months since last purchase (capped at 6), transactions and spend per month over 3/6 months, **trends** (last 3 months minus the 3 before), basket, share of months with a purchase, promo share, missing months |
| Engagement | 10 | app sessions, emails opened, coupons redeemed (3/6 months and trends), months since any activity |
| Support | 4 | support tickets (3/6 months), ticket trend, complaint in the last 6 months |
| Demographics | 8 | age, gender, city, tier, marketing opt-in, preferred category, home-store price tier, tenure |

Fixed windows make every quarter's features comparable; trends capture the "fading" seen in the EDA. **Leakage
test:** each labelled snapshot is rebuilt after deleting all activity after its cutoff (up to 18,975 rows); every
feature is identical (maximum difference 0.0, [`outputs/tables/leakage_check.md`](outputs/tables/leakage_check.md)).

## 8. Validation strategy

Out-of-time, mirroring production (learn from past quarters, score the next one). No random splits: they would let a
member's future quarter help predict their past one.

| Step | Train on snapshots | Evaluate on |
|---|---|---|
| Tuning (threshold, tree counts, regularisation check) | 2023-06 + 2023-09 | 2023-12 (label Jan-Mar 2024) |
| **Test** (all reported results) | 2023-06 + 2023-09 + 2023-12 | **2024-03 (label Apr-Jun 2024 = the given `CHURNED`)** |
| Stability (rolling origin) | all earlier quarters | each of 2023-09, 2023-12, 2024-03 |
| Production scoring | all four labelled snapshots | 2024-06-30 members (Jul-Sep 2024) |

Labels for the three earlier quarters are rebuilt with the brief's definition (99.65% agreement where both exist), so
every training label window ends before the test cutoff. Results are always reported for **all eligible** and
**active** members.

## 9. Models tested

Experiment log including rejected experiments: [`outputs/tables/experiment_log.md`](outputs/tables/experiment_log.md).

| Model (test quarter, active members) | PR-AUC (95% CI) | ROC-AUC | F1 | Brier | PR-AUC over 3 quarters |
|---|---|---|---|---|---|
| Recency rule (baseline) | 0.514 (0.437-0.589) | 0.861 | 0.576 | 0.069 | 0.502 +/- 0.038 |
| **Logistic regression** | **0.828 (0.775-0.873)** | **0.962** | **0.756** | **0.034** | **0.824 +/- 0.019** |
| Logistic regression, class-weighted | 0.816 (0.756-0.865) | 0.961 | 0.740 | 0.043 | - |
| Random forest | 0.817 (0.764-0.863) | 0.955 | 0.711 | 0.038 | 0.765 +/- 0.045 |
| LightGBM | 0.823 (0.769-0.872) | 0.960 | 0.749 | 0.034 | 0.804 +/- 0.025 |
| LightGBM, class-weighted | 0.827 (0.774-0.873) | 0.960 | 0.744 | 0.037 | - |

Also tested and rejected: a model trained on active members only (PR-AUC 0.818). The logistic-regression
regularisation strength was checked on the validation quarter: performance is flat across C = 0.01-10.

## 10. Final model

**Logistic regression** (standardised features, L2, C = 0.3), decision threshold 0.384 chosen on the validation
quarter.

* **Tie with LightGBM:** paired bootstrap PR-AUC difference +0.005 (95% CI -0.019 to +0.029).
* **Better calibrated:** class weights raised the Brier score without improving ranking.
* **Most stable:** best model in each of the three rolling test quarters.
* **Exact reasons for every member's score.**

Imbalance (about 1 churner in 10 active members) is handled by the validation-tuned threshold, not by resampling.

## 11. Results

Held-out quarter (features up to March 2024, label April-June 2024):

| Population | n | Churn rate | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| All eligible | 2,368 | 34.3% | 0.992 | 0.989 | 0.956 | 0.945 | 0.950 | 0.966 |
| **Active members** | 1,715 | 9.6% | 0.962 | **0.828** | 0.788 | 0.726 | 0.756 | 0.955 |

* **Targeting:** the riskiest 5% of active members contain 49% of churners; the riskiest 10% contain 76% (precision
  73%, 7.6x random); the riskiest 20% contain 91%.
* **Calibration:** in the top risk decile the model predicts 72.4% churn and 72.7% is observed.
* **Deliverables:**
  * prediction file: [`outputs/predictions/churn_predictions_apr_jun_2024.csv`](outputs/predictions/churn_predictions_apr_jun_2024.csv)
    (customer_id, churn_probability, predicted_churn, risk_band, key_drivers, actual);
  * forward scores for Jul-Sep 2024: [`outputs/predictions/churn_scores_jul_sep_2024.csv`](outputs/predictions/churn_scores_jul_sep_2024.csv);
  * modelling dataset: [`data/processed/customer_modelling_dataset.csv`](data/processed/customer_modelling_dataset.csv).

## 12. Explainability

* **Global:** behaviour (40%) and engagement (37%) dominate the logistic-regression attribution for active members;
  demographics account for 15% and support for 7%.
  * Strongest drivers: falling transaction trend, fewer emails opened, fewer transactions and app sessions, more
    support tickets (odds ratio 1.73 per sd), and months since the last purchase.
  * LightGBM SHAP agrees (Spearman 0.59 across all 36 features).
* **Local:** every member's score splits exactly into feature contributions. The prediction file lists the top three
  risk-raising reasons per member, e.g. "support tickets/month (3m) = 1.3; transaction trend = -6.3".
* **Segments:** tier makes no difference among active members (Silver 9.5%, Gold 9.8%, Platinum 9.4%). Predicted and
  actual churn match by tier and city.
* **Ablation** (active members, logistic regression):

  | Features | ROC-AUC | PR-AUC |
  |---|---|---|
  | demographics only | 0.508 | 0.095 (= base rate) |
  | + behavioural | 0.932 | 0.687 |
  | + engagement | 0.955 | 0.812 |
  | + support (full) | 0.962 | 0.828 |

## 13. Scenario analysis

* **Population:** the 1,657 members active on 2024-06-30, scored for Jul-Sep 2024.
* **Target:** the riskiest 20% (331 members) hold 131 of the 140 expected churners.
* **Assumption:** each action closes **half the observed gap** in recent behaviour between retained and churning
  members (sensitivity 25-100%), never beyond the retained-member average. The same model re-scores the members.

| Action | Applies to | Assumed change per member | Mean predicted risk | Expected churners avoided |
|---|---|---|---|---|
| Engagement outreach (app + email) | 331 | +0.79 app sessions, +0.32 emails / month | 39.6% -> 29.4% (-10.2 pp) | 34 |
| Proactive support outreach | 157 with a recent ticket | -0.11 tickets / month | 50.4% -> 47.1% (-3.4 pp) | 5 |
| Targeted coupon | 247 below-average redeemers | +0.18 coupons / month | 47.3% -> 44.6% (-2.7 pp) | 7 |

These are **model-based sensitivities, not causal effects**: the model learned that engaged members churn less, not
what happens when the business causes engagement. They rank what to test; a randomised hold-out test measures the real
effect. No costs were supplied, so no ROI is claimed.

## 14. Limitations

* **Data size:** one data source, 18 months and one quarter of given labels; earlier labels are rebuilt (~0.4% noise).
* **Signal strength:** the signals are unusually clean, which suggests synthetic data. Re-validate on live data.
* **Causality:** observational data only. Drivers and what-ifs are associations; no action effect is proven.
* **Missing months** are treated as deleted records. If some were real inactivity, a few features are understated.
* **No economics:** no offer costs, response rates or member value; the threshold optimises F1, not profit.

## 15. How to run the project

```bash
python -m venv .venv
.venv\Scripts\activate                      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --prefix .venv --name freshbasket-churn --display-name "Python (freshbasket-churn)"
```

Put the supplied workbook at `data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx`, then:

```bash
python run_pipeline.py                  # everything: pipeline, notebooks, slides (~3 min)
python run_pipeline.py --no-notebooks   # pipeline only (~1.5 min)
python -m churn.final_model             # any single step (order listed in run_pipeline.py)
jupyter lab notebooks/                  # kernel "Python (freshbasket-churn)"
```

| Notebook | Content |
|---|---|
| `01_data_quality_eda` | inventory, quality findings, two populations, segments, engagement and support, fading |
| `02_feature_engineering` | snapshot design, feature catalogue, leakage test, modelling dataset |
| `03_modeling` | validation design, baseline vs candidates, stability, bootstrap, calibration, gains, experiment log |
| `04_explainability` | global drivers, local member explanations, segments, ablation |
| `05_scenario_analysis` | target group, data-grounded assumptions, results, sensitivity, caveats |
| `06_live_demo` | re-runs a step, recomputes results from the prediction file, live what-if (needs the pipeline run once) |

Notebooks 01-05 only read committed outputs, so they re-run without the raw data.

## 16. Requirements

Python 3.10 (tested on 3.10.11, Windows 11). Pinned in [`requirements.txt`](requirements.txt): numpy, pandas,
pyarrow, scipy, scikit-learn, lightgbm, shap, matplotlib, openpyxl, tabulate, joblib, nbformat, nbclient, ipykernel,
jupyterlab, python-pptx.

## 17. Repository structure

```
.
|-- README.md, requirements.txt, pyproject.toml, run_pipeline.py, .gitignore
|-- data/
|   |-- raw/               <- FreshBasket_Loyalty_Churn_Dataset.xlsx goes here (not committed)
|   |-- interim/           <- parquet caches of the raw sheets (rebuilt)
|   `-- processed/         <- cleaned tables + customer_modelling_dataset.csv (deliverable)
|-- src/churn/
|   |-- config.py, paths.py, utils.py, plotting.py, notebook.py
|   |-- data_loader.py         read the three sheets (cached)
|   |-- data_validation.py     Phase-0 inventory: schema, keys, relationships
|   |-- preprocessing.py       quality checks, cleaning, member x month grid, label check
|   |-- feature_engineering.py snapshots, 36 features, leakage test, modelling dataset
|   |-- eda.py                 exploratory figures and tables
|   |-- modeling.py            baseline, candidates, out-of-time validation, stability, regularisation sweep
|   |-- final_model.py         selected model, threshold, calibration, gains, prediction files
|   |-- explainability.py      coefficients, contributions, SHAP cross-check, local explanations, segments
|   |-- ablation.py            feature-group ablation
|   |-- scenarios.py           retention what-ifs with data-grounded assumptions and sensitivity
|   `-- summaries.py           feature catalogue, experiment log, model-selection summary
|-- notebooks/            01-06 (executed, with outputs) + build_notebooks.py
|-- outputs/
|   |-- figures/           every chart
|   |-- tables/            every result table (CSV + Markdown)
|   |-- predictions/       churn predictions, forward scores, retention scenarios
|   `-- models/            saved models (rebuilt, not committed)
|-- reports/              final_report.md, findings.md, data_inventory.md, requirement_traceability.md, quality_audit.md
`-- presentation/         churn_prediction_presentation.pptx, presentation_outline.md, build_deck.py
```

## 18. Reproducibility notes

* **One command:** `python run_pipeline.py` rebuilds every table, figure, prediction file, notebook and the slide deck
  from the raw workbook.
* **Pinned versions** in `requirements.txt`; all paths relative to the repository root; random seeds fixed (42).
* **Verified:**
  * a fresh environment built from `requirements.txt` ran the full pipeline;
  * the leakage test and every notebook execute without errors;
  * the deck passes an automatic text-fit check.
* **Windows note:** LightGBM is imported before scikit-learn in several modules. The reverse order crashed LightGBM
  with an OpenMP DLL clash on the development machine.
* The raw workbook is not committed (company data); models and parquet caches are rebuilt in seconds.
