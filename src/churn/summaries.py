"""Summary tables assembled from the generated results (no number is typed by hand):

    outputs/tables/feature_catalogue.*         definition, interpretation, leakage control, measured usefulness
    outputs/tables/experiment_log.*            every experiment, including the rejected ones
    outputs/tables/model_selection_summary.*   performance, stability, complexity, interpretability, decision

Run:  python -m churn.summaries
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from churn import config as C
from churn.utils import setup_logging, write_table

log = logging.getLogger(__name__)
P = C.P


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(P.tables / f"{name}.csv")


def feature_catalogue() -> pd.DataFrame:
    defs = t("feature_definitions")
    imp = t("lr_driver_importance").set_index("feature")
    shap = t("lgbm_shap_importance").set_index("feature")
    defs["mean_abs_lr_contribution_active"] = defs.feature.map(imp["active"]).round(3)
    defs["rank_lr_active"] = defs.feature.map(imp["active"].rank(ascending=False)).astype(int)
    defs["mean_abs_shap_active"] = defs.feature.map(shap["mean_abs_shap_active"]).round(3)
    defs["used_in_final_model"] = "yes"
    return defs


def experiment_log() -> pd.DataFrame:
    mc = t("model_comparison").set_index(["population", "model"])
    act = mc.xs("active")
    at = t("model_comparison_active_trained").set_index("model")
    sw = t("lr_regularisation_sweep")
    ro = t("rolling_origin_summary").set_index("model")
    ab = t("ablation")
    ab = ab[ab.model == "logistic regression"].set_index("variant")
    sel = json.loads((P.tables / "final_model_selection.json").read_text())
    boot = sel["bootstrap_active_population"]["PR-AUC diff (LR - LGBM)"]
    fm = t("final_model_test_metrics").set_index("population")
    lk = t("leakage_check")
    val = "out-of-time: train 2023-06/09/12 snapshots, test 2024-03 (Apr-Jun 2024 label)"

    def m(name):
        r = act.loc[name]
        return f"active PR-AUC {r['PR-AUC']:.3f} (95% CI {r['PR-AUC CI low']:.3f}-{r['PR-AUC CI high']:.3f}), ROC-AUC {r['ROC-AUC']:.3f}, Brier {r['Brier']:.3f}"

    rows = [
        ("E01", "recency + transaction-trend rule", "rule (no training)", "-", val, m("rule_recency"),
         "baseline to beat"),
        ("E02", "all 36 features", "logistic regression", f"L2, C={C.LR_C}, no class weights", val, m("logreg"),
         "SELECTED: best PR-AUC, best calibration, exact per-member reasons"),
        ("E03", "all 36 features", "logistic regression", f"C={C.LR_C}, class_weight=balanced", val, m("logreg_balanced"),
         "rejected: no ranking gain, worse calibration (higher Brier)"),
        ("E04", "all 36 features", "random forest", "500 trees, min_samples_leaf=20, max_features=0.4", val, m("random_forest"),
         "rejected: not better than LR, less interpretable"),
        ("E05", "all 36 features", "LightGBM", f"{int(act.loc['lightgbm', 'n_trees'])} trees (early stopping on validation quarter)",
         val, m("lightgbm"), "kept as challenger (tie with LR; used for the SHAP cross-check)"),
        ("E06", "all 36 features", "LightGBM", f"{int(act.loc['lightgbm_balanced', 'n_trees'])} trees, scale_pos_weight", val,
         m("lightgbm_balanced"), "rejected: no ranking gain, worse calibration"),
        ("E07", "all 36 features, trained on active members only", "logistic regression", f"C={C.LR_C}", val,
         f"active PR-AUC {at.loc['logreg (trained on active only)', 'PR-AUC']:.3f}",
         "rejected: a specialist model is not better; one model serves both populations"),
        ("E08", "all 36 features", "logistic regression", f"C in {sw.C.tolist()}", "validation quarter only (2023-12 snapshot)",
         f"validation PR-AUC active {sw['val PR-AUC active'].min():.3f}-{sw['val PR-AUC active'].max():.3f}",
         f"flat: result does not depend on C; kept C={C.LR_C}"),
        ("E09", "all 36 features", "rule / LR / RF / LightGBM", "LightGBM fixed 200 trees", "rolling origin: 3 consecutive test quarters",
         f"LR active PR-AUC mean {ro.loc['logreg', 'PR-AUC active mean']:.3f} (min {ro.loc['logreg', 'PR-AUC active min']:.3f}); "
         f"LightGBM {ro.loc['lightgbm', 'PR-AUC active mean']:.3f}; rule {ro.loc['rule_recency', 'PR-AUC active mean']:.3f}",
         "LR is the most stable and best in every quarter"),
        ("E10", "demographics only (8)", "logistic regression", f"C={C.LR_C}", val,
         f"active PR-AUC {ab.loc['1. demographics only', 'PR-AUC active']:.3f}, ROC-AUC {ab.loc['1. demographics only', 'ROC-AUC active']:.3f}",
         "demographics alone carry no signal for active members"),
        ("E11", "+ behavioural (22)", "logistic regression", f"C={C.LR_C}", val,
         f"active PR-AUC {ab.loc['2. + behavioural', 'PR-AUC active']:.3f}", "behaviour carries most of the signal"),
        ("E12", "+ engagement (32)", "logistic regression", f"C={C.LR_C}", val,
         f"active PR-AUC {ab.loc['3. + engagement', 'PR-AUC active']:.3f}", "engagement adds clearly"),
        ("E13", "+ support (36, full)", "logistic regression", f"C={C.LR_C}", val,
         f"active PR-AUC {ab.loc['4. + support (full)', 'PR-AUC active']:.3f}", "support adds a smaller, consistent gain"),
        ("E14", "selected model", "logistic regression", "F1-maximising threshold on the validation quarter", val,
         f"threshold {sel['threshold_from_validation']:.3f}: active precision {fm.loc['active', 'precision']:.3f}, "
         f"recall {fm.loc['active', 'recall']:.3f}, F1 {fm.loc['active', 'F1']:.3f}", "operating point (imbalance handled here, not by resampling)"),
        ("E15", "LR vs LightGBM", "paired bootstrap (1,000 resamples)", "-", "test quarter, active members",
         f"PR-AUC difference {boot[0]:+.3f} (95% CI {boot[1]:+.3f} to {boot[2]:+.3f})", "statistical tie -> prefer the simpler, explainable model"),
        ("E16", "all features", "leakage test", "rebuild each snapshot from data truncated at its cutoff", "4 labelled snapshots",
         f"max feature difference {lk.max_abs_difference.max():g}; all passed: {bool(lk.passed.all())}", "features are leakage-safe"),
    ]
    return pd.DataFrame(rows, columns=["id", "features", "model", "parameters", "validation", "result", "decision"])


def model_selection_summary() -> pd.DataFrame:
    act = t("model_comparison").set_index(["population", "model"]).xs("active")
    ro = t("rolling_origin_summary").set_index("model")
    info = {
        "rule_recency": ("none", "fully transparent", "baseline"),
        "logreg": ("low (36 coefficients)", "exact per-member contributions", "SELECTED"),
        "logreg_balanced": ("low", "exact contributions", "rejected (worse calibration)"),
        "random_forest": ("high (500 trees)", "needs SHAP / permutation", "rejected (not better)"),
        "lightgbm": ("medium-high (boosted trees)", "needs SHAP", "challenger (statistical tie)"),
        "lightgbm_balanced": ("medium-high", "needs SHAP", "rejected (worse calibration)"),
    }
    rows = []
    for k, (cx, interp, dec) in info.items():
        r = act.loc[k]
        stab = (f"{ro.loc[k, 'PR-AUC active mean']:.3f} +/- {ro.loc[k, 'PR-AUC active std']:.3f}" if k in ro.index else "not run")
        rows.append({"model": k, "PR-AUC active (test)": round(r["PR-AUC"], 3),
                     "95% CI": f"{r['PR-AUC CI low']:.3f}-{r['PR-AUC CI high']:.3f}",
                     "ROC-AUC active": round(r["ROC-AUC"], 3), "F1 active": round(r["F1"], 3), "Brier active": round(r["Brier"], 3),
                     "stability: PR-AUC over 3 quarters (mean +/- sd)": stab, "complexity": cx,
                     "interpretability": interp, "decision": dec})
    return pd.DataFrame(rows)


def run():
    write_table(feature_catalogue(), P.tables / "feature_catalogue")
    write_table(experiment_log(), P.tables / "experiment_log")
    write_table(model_selection_summary(), P.tables / "model_selection_summary")
    log.info("summary tables written")


if __name__ == "__main__":
    setup_logging()
    run()
