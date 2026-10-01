# Final quality audit

Checklist run before publishing. Each item points to the evidence.

## Data

| Check | Status | Evidence |
|---|---|---|
| All files inspected | done | `reports/data_inventory.md`: every sheet profiled (rows, dtypes, missingness, distinct counts, ranges) |
| Schema documented | done | `outputs/tables/schema_inventory.csv` (one row per column), `table_summary.csv` |
| Data quality checked | done | `outputs/tables/data_quality_findings.md`: 18 findings with evidence, severity, treatment, reason |
| Joins validated | done | `integrity_checks.csv`: 0 orphan IDs, identical member sets, key uniqueness after de-duplication |
| Grain validated | done | member x quarter-end snapshot (10,776 rows); no raw merge of monthly rows onto profiles |

## Modelling

| Check | Status | Evidence |
|---|---|---|
| Baseline built | done | recency rule: active PR-AUC 0.514 |
| Multiple approaches tested | done | logistic regression, random forest, LightGBM, with and without class weights, plus an active-only model (`experiment_log`) |
| Validation appropriate | done | out-of-time snapshots (tune Jan-Mar 2024, test Apr-Jun 2024) + rolling origin over 3 quarters; no random split |
| Leakage audited | done | post-outcome label columns excluded and asserted; automated truncation test, max feature difference 0.0 (`leakage_check`) |
| Metrics appropriate | done | PR-AUC, precision, recall, F1, ROC-AUC, Brier, accuracy, bootstrap CIs; reported for all eligible and active members |
| Advanced model not better than the simple one: reported honestly | done | LightGBM ties the logistic regression; the simpler model is selected (`model_selection_summary`) |

## Explainability

| Check | Status | Evidence |
|---|---|---|
| Feature importance / explanation | done | exact logistic contributions (global + per member), LightGBM SHAP cross-check (`lr_driver_importance`, `lgbm_shap_importance`, `local_explanations`) |
| Business interpretation | done | report sections 11 and 14, `reports/findings.md` |
| No unsupported causal claims | done | drivers and scenarios labelled as associations / model-based throughout; the experiment needed is named |

## Business

| Check | Status | Evidence |
|---|---|---|
| Scenario analysis | done | `retention_scenarios`, assumptions grounded in observed behaviour gaps, 25-100% sensitivity |
| Recommendations | done | report section 14 |
| Limitations | done | report section 15, README section 14 |

## Engineering

| Check | Status | Evidence |
|---|---|---|
| Modular Python | done | `src/churn/` (one module per pipeline stage) |
| README | done | 18 sections: problem to reproducibility |
| Requirements | done | pinned `requirements.txt`; a fresh virtual environment built from it ran the whole project |
| Reproducible execution | done | `python run_pipeline.py` rebuilds everything from the raw workbook in about 2 minutes; regenerated predictions match the development run exactly |
| Clean repository | done | see below |

## GitHub pre-publish check (automated)

| Check | Result |
|---|---|
| Personal machine paths in committed text, tables or notebook outputs | none found |
| Credentials or tokens | none found |
| Broken links / missing files referenced in Markdown | none |
| Lint (pyflakes) | clean (the deliberate `import lightgbm` lines are the Windows import-order fix) |
| Notebooks | all 6 executed top to bottom without errors; outputs kept on purpose so reviewers can read results without running anything |
| Raw data | not committed (`data/raw/README.md` explains where to put the workbook) |
| Intermediate files (parquet caches, saved models, previews, virtual environment) | excluded by `.gitignore`; rebuilt by the pipeline |
| Repository size | 127 files, 8.4 MB |

## Presentation

| Check | Status | Evidence |
|---|---|---|
| 10-15 minute deck | done | 14 slides, word-for-word scripts in the speaker notes (about 12 minutes) |
| Storyline | done | business problem -> data -> data issues -> insights -> features -> method/validation -> results -> explanation -> scenario -> decision/limitations -> conclusion |
| Per-slide objective, content, visual, script, likely question and answer | done | `presentation/presentation_outline.md` |
| Text fits, numbers match the tables | done | automatic layout check passes; slide numbers are read from `outputs/tables` |
