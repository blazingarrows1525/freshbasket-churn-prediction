# Requirement traceability

Every requirement of the Assessment 4 brief, mapped to code, evidence, validation and the slide where it is presented.
Slide numbers refer to `presentation/churn_prediction_presentation.pptx`.

## Core tasks

| Requirement | Source in brief | Implementation | Evidence / output | Validation | Slide |
|---|---|---|---|---|---|
| Data-quality checks and EDA: missing months, duplicates, invalid/negative spend, zero-transaction anomalies, casing, outlier spend | Core task 1 | `data_validation.py`, `preprocessing.py`, `eda.py` | `data_quality_findings`, `integrity_checks`, `reports/data_inventory.md`, notebook 01 | every issue logged with evidence; spend repairs checked against the spend identity | 4 |
| Join the three tables into one customer-level modelling dataset | Core task 2 | `preprocessing.make_grid`, `feature_engineering.build_all` | `data/processed/customer_modelling_dataset.csv` (10,776 member-snapshots) | key/cardinality checks; label rebuild agrees 99.65% | 3 |
| Leakage-safe features: recency, spend and transaction trends, rolling engagement, tenure, ticket trend, promo usage, demographics/tier; only pre-label-window data | Core task 3 | `feature_engineering.py` (36 features) | `feature_catalogue`, `leakage_check` | automated truncation test: max difference 0.0; post-outcome columns asserted out | 7 |
| Relationship between engagement, support and churn | Core task 4 | `eda.engagement_support`, `eda.trajectories` | `eda_engagement_support`, `eda_trajectories`, notebook 01 | pooled over 4 labelled quarters | 6 |
| Time-based or cohort-based validation; no random split | Core task 5 | `modeling.py` (out-of-time snapshots, rolling origin) | `model_comparison`, `rolling_origin_stability` | train label windows end before the test cutoff | 8 |
| At least two approaches: rule/logistic baseline and an ML classifier | Core task 6 | `modeling.MODEL_SPECS`: recency rule, logistic regression (+/- weights), random forest, LightGBM (+/- weights) | `model_comparison`, `model_selection_summary`, `experiment_log` | bootstrap CIs; 3-quarter stability | 9 |
| Predict churn probability for a held-out cohort | Core task 7 | `final_model.py` | `outputs/predictions/churn_predictions_apr_jun_2024.csv` | scored on the Apr-Jun 2024 quarter never used for training | 9-10 |
| Address class imbalance; report precision, recall, F1, PR-AUC and accuracy | Core task 8 | threshold tuned on the validation quarter; class weights compared | `final_model_test_metrics`, `model_comparison` | Brier and calibration show weights hurt; no resampling | 9 |
| Explain drivers (importance or SHAP); compare risk across tier, city, signup cohort | Core task 9 | `explainability.py` | `lr_coefficients`, `lr_driver_importance`, `lgbm_shap_importance`, `local_explanations`, `risk_by_segment_active` | LR vs SHAP rank agreement (Spearman 0.59) | 11 |
| Ablation: engagement, support and behaviour vs demographics alone | Core task 10 | `ablation.py` | `ablation`, `outputs/figures/ablation.png` | same out-of-time design; LR and LightGBM both | 11 |
| At least one retention scenario and the estimated change in churn risk | Core task 11 | `scenarios.py` | `retention_scenarios`, `retention_scenario_sensitivity`, `scenario_behaviour_gaps` | assumptions grounded in observed gaps; 25-100% sensitivity; labelled model-based | 12 |
| Assumptions, limitations and business recommendations | Core task 12 | report sections 13-15; README 13-14 | `reports/final_report.md` | each limitation linked to its consequence | 13 |

## What the solution should do

| Requirement | Source in brief | Implementation | Evidence / output | Slide |
|---|---|---|---|---|
| Analyse monthly activity and engagement | Problem objective | `eda.py` | notebook 01 | 5-6 |
| Predict customer-level churn risk for the next period | Problem objective | `final_model.py` | prediction file + `churn_scores_jul_sep_2024.csv` (forward scores) | 10 |
| Strongest drivers across behavioural, demographic, engagement and support features | Problem objective | `explainability.py`, `ablation.py` | `lr_driver_group_share`, `ablation` | 11 |
| Compare risk across tier, city, signup cohort | Problem objective | `eda.segment_rates`, `explainability.py` | `eda_segment_rates`, `risk_by_segment_active` | 11 |
| Simulate a retention intervention and estimate its impact | Problem objective | `scenarios.py` | `retention_scenarios` | 12 |
| Translate findings into retention and marketing recommendations | Problem objective | report section 14 | `reports/final_report.md`, `reports/findings.md` | 13 |

## Required deliverables

| Deliverable | Where |
|---|---|
| Reproducible Python project | `src/churn/`, `run_pipeline.py`, `requirements.txt`, `pyproject.toml` |
| README with setup, assumptions and execution steps | `README.md` |
| Data-quality and EDA notebook | `notebooks/01_data_quality_eda.ipynb` (+ 02-06) |
| Final customer-level modelling dataset | `data/processed/customer_modelling_dataset.csv` |
| Model comparison and validation results | `outputs/tables/model_comparison.*`, `model_selection_summary.*`, `rolling_origin_stability.*`, notebook 03 |
| Churn prediction file with customer ID, churn probability, predicted label and key drivers | `outputs/predictions/churn_predictions_apr_jun_2024.csv` |
| Driver-attribution and feature-impact analysis | notebook 04, `lr_*`, `lgbm_shap_importance`, `local_explanations`, `ablation` |
| Retention scenario analysis | notebook 05, `retention_scenarios`, `retention_scenario_sensitivity` |
| Final report with findings, limitations and recommendations | `reports/final_report.md`, `reports/findings.md` |
| 10-15 minute presentation | `presentation/churn_prediction_presentation.pptx` (14 slides, scripts in the notes), `presentation/presentation_outline.md` |
