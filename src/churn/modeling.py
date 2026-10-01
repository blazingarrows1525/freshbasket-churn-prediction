"""Baselines, candidate models, out-of-time validation, imbalance handling and stability checks.

Validation design
  tuning:   train on snapshots 2023-06 + 2023-09  ->  validate on 2023-12 (label Jan-Mar 2024)
  test:     train on 2023-06 + 2023-09 + 2023-12  ->  test on 2024-03 (the given CHURNED label, Apr-Jun 2024)
  scoring:  train on all four labelled snapshots  ->  score 2024-06-30 for Jul-Sep 2024
Every training label window ends on or before the test cutoff, so the test is a genuine "predict the next
quarter" exercise. Random splits are not used: they would mix members' past and future quarters.

Two populations are always reported:
  all eligible  every member with purchase history before the cutoff (the brief's label)
  active        members who bought in the 3 months before the cutoff - the ones a retention offer can
                still reach (churn ~10% in every quarter)

Run:  python -m churn.modeling
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config as C
from churn import feature_engineering as FE
from churn.utils import setup_logging, write_table

log = logging.getLogger(__name__)
P = C.P
TRAIN_KEYS = [str(c.date()) for c in C.TRAIN_CUTOFFS]
TEST_KEY = str(C.OFFICIAL_CUTOFF.date())
SCORE_KEY = str(C.SCORING_CUTOFF.date())


def stack(snaps, keys) -> pd.DataFrame:
    return pd.concat([snaps[k] for k in keys], ignore_index=True)


# ----------------------------------------------------------------------------------------------- models
def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Transparent baseline rule: the longer since the last purchase, the higher the risk (0..1)."""
    return (df.recency_months.values + 0.1 * (df.txn_trend.values < 0)) / (C.LONG + 0.1)


def make_lr(features, class_weight=None, Creg=C.LR_C) -> Pipeline:
    num = [f for f in features if f not in FE.CATEGORICAL]
    cat = [f for f in features if f in FE.CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=Creg, class_weight=class_weight, max_iter=2000))])


def fit_lgb(train, features, weight_balanced=False, n_estimators=None, valid=None):
    params = dict(C.LGB_PARAMS)
    n = n_estimators or params.pop("n_estimators")
    params.pop("n_estimators", None)
    if weight_balanced:
        params["scale_pos_weight"] = (train.churn == 0).sum() / max((train.churn == 1).sum(), 1)
    cats = [f for f in features if f in FE.CATEGORICAL]
    lab = lambda df: np.ascontiguousarray(df.churn.to_numpy(dtype=np.float64))  # noqa: E731
    d = lgb.Dataset(train[features], label=lab(train), categorical_feature=cats)
    kw = {}
    if valid is not None:
        kw = dict(valid_sets=[lgb.Dataset(valid[features], label=lab(valid), categorical_feature=cats, reference=d)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return lgb.train(params, d, num_boost_round=n, **kw)


def make_rf(features, class_weight=None):
    num = [f for f in features if f not in FE.CATEGORICAL]
    cat = [f for f in features if f in FE.CATEGORICAL]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features=0.4,
                                                                 class_weight=class_weight, random_state=C.SEED, n_jobs=8))])


MODEL_SPECS = {
    "rule_recency": dict(kind="rule"),
    "logreg": dict(kind="lr", class_weight=None),
    "logreg_balanced": dict(kind="lr", class_weight="balanced"),
    "random_forest": dict(kind="rf", class_weight=None),
    "lightgbm": dict(kind="lgb", balanced=False),
    "lightgbm_balanced": dict(kind="lgb", balanced=True),
}


def train_predict(spec, train, test, features, n_trees=None):
    k = spec["kind"]
    if k == "rule":
        return None, rule_score(test)
    if k == "lr":
        m = make_lr(features, spec.get("class_weight")).fit(train[features], train.churn)
        return m, m.predict_proba(test[features])[:, 1]
    if k == "rf":
        m = make_rf(features, spec.get("class_weight")).fit(train[features], train.churn)
        return m, m.predict_proba(test[features])[:, 1]
    m = fit_lgb(train, features, spec.get("balanced", False), n_estimators=n_trees)
    return m, m.predict(test[features])


# ----------------------------------------------------------------------------------------------- metrics
def best_threshold(y, p) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    f1 = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1))])


def metrics(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"n": len(y), "churn_rate": float(np.mean(y)), "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p), "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
            "threshold": thr, "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "F1": f1_score(y, yhat, zero_division=0),
            "accuracy": accuracy_score(y, yhat), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def bootstrap_ci(y, p, metric=average_precision_score, n=1000, seed=C.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(metric(y[idx], p[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evaluate_models(snaps, features=None, populations=("all", "active"), train_active_only=False):
    """Tune on the validation snapshot, then score the official test snapshot."""
    features = features or FE.ALL_FEATURES
    val_train, val = stack(snaps, TRAIN_KEYS[:2]), snaps[TRAIN_KEYS[2]]
    train, test = stack(snaps, TRAIN_KEYS), snaps[TEST_KEY]
    if train_active_only:
        val_train, val, train = (d[d.active_at_cutoff] for d in (val_train, val, train))
    rows, preds = [], {}
    for name, spec in MODEL_SPECS.items():
        n_trees = None
        if spec["kind"] == "lgb":
            m = fit_lgb(val_train, features, spec["balanced"], valid=val)
            n_trees = max(int(m.best_iteration or 200), 50)
        _, p_val = train_predict(spec, val_train, val, features, n_trees)
        _, p_test = train_predict(spec, train, test, features, n_trees)
        preds[name] = p_test
        for pop in populations:
            mv = np.ones(len(val), bool) if pop == "all" else val.active_at_cutoff.values
            mt = np.ones(len(test), bool) if pop == "all" else test.active_at_cutoff.values
            thr = best_threshold(val.churn.values[mv], p_val[mv])          # threshold chosen on validation only
            yt = test.churn.values[mt].astype(int)
            lo, hi = bootstrap_ci(yt, p_test[mt])
            rows.append({"population": pop, "model": name, "n_trees": n_trees,
                         "val ROC-AUC": roc_auc_score(val.churn.values[mv], p_val[mv]),
                         "val PR-AUC": average_precision_score(val.churn.values[mv], p_val[mv]),
                         **metrics(yt, p_test[mt], thr), "PR-AUC CI low": lo, "PR-AUC CI high": hi})
    return pd.DataFrame(rows), preds, test


def rolling_origin(snaps, features=None, n_trees=200) -> pd.DataFrame:
    """Stability: train on all earlier labelled quarters, test on the next one (3 test quarters).

    The LightGBM tree count is fixed (no tuning) so no quarter is used for both a choice and a test.
    """
    features = features or FE.ALL_FEATURES
    order = TRAIN_KEYS + [TEST_KEY]
    rows = []
    for i in range(1, len(order)):
        tr, te = stack(snaps, order[:i]), snaps[order[i]]
        a = te.active_at_cutoff.values
        y = te.churn.values.astype(int)
        preds = {"rule_recency": rule_score(te),
                 "logreg": make_lr(features).fit(tr[features], tr.churn).predict_proba(te[features])[:, 1],
                 "random_forest": make_rf(features).fit(tr[features], tr.churn).predict_proba(te[features])[:, 1],
                 "lightgbm": fit_lgb(tr, features, n_estimators=n_trees).predict(te[features])}
        for k, p in preds.items():
            rows.append({"train_snapshots": " + ".join(order[:i]), "test_snapshot": order[i], "model": k,
                         "active_members": int(a.sum()), "active_churn_rate": y[a].mean(),
                         "ROC-AUC active": roc_auc_score(y[a], p[a]), "PR-AUC active": average_precision_score(y[a], p[a]),
                         "ROC-AUC all": roc_auc_score(y, p), "PR-AUC all": average_precision_score(y, p)})
    return pd.DataFrame(rows)


def regularisation_sweep(snaps, features=None, grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)) -> pd.DataFrame:
    """Logistic-regression L2 strength on the validation quarter only (the test quarter is not touched)."""
    features = features or FE.ALL_FEATURES
    vt, v = stack(snaps, TRAIN_KEYS[:2]), snaps[TRAIN_KEYS[2]]
    a = v.active_at_cutoff.values
    rows = []
    for Creg in grid:
        p = make_lr(features, Creg=Creg).fit(vt[features], vt.churn).predict_proba(v[features])[:, 1]
        rows.append({"C": Creg, "val PR-AUC active": average_precision_score(v.churn[a], p[a]),
                     "val ROC-AUC active": roc_auc_score(v.churn[a], p[a]), "val PR-AUC all": average_precision_score(v.churn, p),
                     "used": Creg == C.LR_C})
    return pd.DataFrame(rows)


def gains(y, p, fractions=(0.05, 0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    out = []
    for f in fractions:
        k = int(round(f * len(ys)))
        out.append({"targeted_share": f, "members_targeted": k, "churners_captured": int(ys[:k].sum()),
                    "share_of_churners_captured": ys[:k].sum() / ys.sum(), "precision_in_target": ys[:k].mean(),
                    "lift_vs_random": ys[:k].mean() / ys.mean()})
    return pd.DataFrame(out)


def run():
    snaps = FE.load_snapshots()
    res, preds, test = evaluate_models(snaps)
    write_table(res, P.tables / "model_comparison")
    res_a, _, _ = evaluate_models(snaps, populations=("active",), train_active_only=True)
    res_a["model"] = res_a.model + " (trained on active only)"
    write_table(res_a, P.tables / "model_comparison_active_trained")
    ro = rolling_origin(snaps)
    write_table(ro, P.tables / "rolling_origin_stability")
    summ = ro.groupby("model")[["ROC-AUC active", "PR-AUC active"]].agg(["mean", "std", "min", "max"])
    summ.columns = [" ".join(c) for c in summ.columns]
    write_table(summ.reset_index(), P.tables / "rolling_origin_summary")
    write_table(regularisation_sweep(snaps), P.tables / "lr_regularisation_sweep")

    test_out = test[["CUSTOMER_ID", "population", "MEMBERSHIP_TIER", "CITY", "signup_cohort", "churn"]].copy()
    for k, v in preds.items():
        test_out[f"p_{k}"] = v
    test_out.to_parquet(P.predictions / "test_predictions_all_models.parquet", index=False)
    log.info("\n%s", res[["population", "model", "ROC-AUC", "PR-AUC", "precision", "recall", "F1", "Brier"]].round(3).to_string())
    log.info("rolling origin:\n%s", summ.round(3).to_string())


if __name__ == "__main__":
    setup_logging()
    run()
