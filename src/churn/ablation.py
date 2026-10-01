"""Feature-group ablation: does behaviour, engagement and support add to demographics?

Same out-of-time design as the main model (train on the three earlier quarters,
test on Apr-Jun 2024). Both the logistic regression and LightGBM are scored so
the conclusion does not depend on one model family.

Run:  python -m churn.ablation
"""
from __future__ import annotations

import logging

import lightgbm  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from churn import config as C
from churn import feature_engineering as F
from churn import modeling as Md
from churn import plotting as pl
from churn.utils import setup_logging, write_table

log = logging.getLogger(__name__)
P = C.P
G = F.GROUPS
VARIANTS = {
    "1. demographics only": G["demographics"],
    "2. + behavioural": G["demographics"] + G["behavioural"],
    "3. + engagement": G["demographics"] + G["behavioural"] + G["engagement"],
    "4. + support (full)": G["demographics"] + G["behavioural"] + G["engagement"] + G["support"],
    "demographics + engagement only": G["demographics"] + G["engagement"],
    "demographics + support only": G["demographics"] + G["support"],
    "full - behavioural": G["demographics"] + G["engagement"] + G["support"],
}


def run():
    snaps = F.load_snapshots()
    keys = [str(c.date()) for c in C.TRAIN_CUTOFFS]
    vt, v = Md.stack(snaps, keys[:2]), snaps[keys[2]]
    train, test = Md.stack(snaps, keys), snaps[str(C.OFFICIAL_CUTOFF.date())]
    y = test.churn.values.astype(int)
    act = test.active_at_cutoff.values
    rows = []
    for name, feats in VARIANTS.items():
        lr = Md.make_lr(feats).fit(train[feats], train.churn)
        p_lr = lr.predict_proba(test[feats])[:, 1]
        b = Md.fit_lgb(vt, feats, valid=v)
        n = max(int(b.best_iteration or 100), 50)
        p_gb = Md.fit_lgb(train, feats, n_estimators=n).predict(test[feats])
        for model, p in [("logistic regression", p_lr), ("LightGBM", p_gb)]:
            rows.append({"variant": name, "model": model, "n_features": len(feats),
                         "ROC-AUC all": roc_auc_score(y, p), "PR-AUC all": average_precision_score(y, p),
                         "ROC-AUC active": roc_auc_score(y[act], p[act]), "PR-AUC active": average_precision_score(y[act], p[act])})
        log.info("%s done", name)
    res = pd.DataFrame(rows)
    write_table(res, P.tables / "ablation")
    t = res[res.model == "logistic regression"].set_index("variant")
    fig, ax = plt.subplots(figsize=(9, 4))
    order = list(VARIANTS)
    y_ = np.arange(len(order))[::-1]
    ax.barh(y_ + 0.2, t.loc[order, "PR-AUC active"], height=0.38, color=pl.ACCENT, label="PR-AUC, active members (base rate 9.6%)")
    ax.barh(y_ - 0.2, t.loc[order, "ROC-AUC active"], height=0.38, color=pl.DEEMPH, label="ROC-AUC, active members")
    for yy, a, r in zip(y_, t.loc[order, "PR-AUC active"], t.loc[order, "ROC-AUC active"]):
        ax.text(a + 0.01, yy + 0.2, f"{a:.2f}", va="center", fontsize=8, color=pl.INK_2)
        ax.text(r + 0.01, yy - 0.2, f"{r:.2f}", va="center", fontsize=8, color=pl.INK_2)
    ax.set_yticks(y_, order)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Ablation (logistic regression, Apr-Jun 2024 test quarter)")
    pl.save(fig, P.figures / "ablation.png")
    log.info("\n%s", res.round(3).to_string())


if __name__ == "__main__":
    setup_logging()
    run()
