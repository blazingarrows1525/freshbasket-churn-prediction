# Key findings

Each finding is backed by a generated table in `outputs/tables/`.

| # | Finding | Evidence | So what |
|---|---|---|---|
| 1 | Most measured churn is members who had already left: 649 of the 813 labelled churners (80%) bought nothing in Jan-Mar 2024 | `snapshots_target_distribution`: lapsed members churn at 97-99% in every quarter | a retention offer is too late for them; win-back is a separate programme |
| 2 | Churn among members still buying is a steady ~10% per quarter (9.6-10.3%) | `snapshots_target_distribution` | the real early-warning problem; report it as the KPI |
| 3 | Churners fade for months before leaving: transactions 3.6 -> 0.9 a month, app sessions 1.8 -> 0.3, tickets 0.17 -> ~0.5 | `eda_trajectories` | there is a window to act; trend features carry signal |
| 4 | Spend errors are diagnosable: spend = transactions x basket on 99.8% of purchase months; 22 sign errors and 13 mistyped totals repaired | `data_quality_findings` | clean spend features without guessing |
| 5 | Three label-table columns are computed over the label window and would leak the answer | `data_quality_findings` | excluded; an automated test proves the features are leakage-safe (`leakage_check`, max difference 0.0) |
| 6 | Demographics carry no signal for active members (ROC-AUC 0.51, PR-AUC = base rate); tier churn is 9.4-9.8% | `ablation`, `eda_segment_rates` | do not budget retention by tier |
| 7 | Behaviour and engagement drive risk (40% and 37% of attribution); support tickets add a smaller, consistent signal | `lr_driver_group_share`, `ablation` | monitor monthly purchases, app/email engagement and tickets |
| 8 | A logistic regression ties LightGBM (PR-AUC +0.005, CI -0.019 to +0.029) and is the most stable over three quarters (0.824 +/- 0.019) | `final_model_selection.json`, `rolling_origin_summary` | the simpler, explainable model is enough |
| 9 | The riskiest 10% of active members contain 76% of next quarter's churners (precision 73%, 7.6x random) | `gains_active` | a short monthly list replaces blanket offers |
| 10 | Probabilities are calibrated (top decile 72.4% predicted vs 72.7% observed); class weights would have worsened this | `calibration_active_logistic`, `model_comparison` | scores can be read as probabilities |
| 11 | Model-based what-ifs rank engagement outreach first (-10.2 pp for the riskiest 20%), then support and coupons (~-3 pp) | `retention_scenarios`, `retention_scenario_sensitivity` | test engagement outreach first, with a random hold-out |
| 12 | Opted-in members who stop opening emails are higher risk (opt-in odds ratio 1.51 given email opens), although opted-in members churn less overall | `lr_coefficients`, `eda_segment_rates` | "opted in but silent" is an early warning sign |
