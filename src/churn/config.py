"""Project configuration: churn definition, snapshot dates, feature windows and model settings."""
from __future__ import annotations

import pandas as pd

from churn import paths

P = paths.P               # modules use C.P

SEED = 42

# Churn definition (brief): no purchase transaction in the 3 months after the cutoff, for a member who
# purchased at some point before the cutoff.
LABEL_MONTHS = 3
OFFICIAL_CUTOFF = pd.Timestamp("2024-03-31")          # label window Apr-Jun 2024 = the given CHURNED column
# Earlier cutoffs whose labels are rebuilt from the activity table with the same definition. Their label
# windows end on or before the official cutoff, so every training label is known when the official
# snapshot is scored (out-of-time validation).
TRAIN_CUTOFFS = [pd.Timestamp(d) for d in ["2023-06-30", "2023-09-30", "2023-12-31"]]
SCORING_CUTOFF = pd.Timestamp("2024-06-30")           # forward scoring for Jul-Sep 2024 (no labels exist yet)
DATA_START = pd.Timestamp("2023-01-01")

# Feature windows are fixed-length so every snapshot has comparable features (the first training
# snapshot only has 6 months of history).
SHORT, LONG = 3, 6

# "Active at cutoff" = at least one purchase in the last ACTIVE_MONTHS months. Members who lapsed earlier
# are already gone; retention offers can only reach the active ones.
ACTIVE_MONTHS = 3

SPEND_RATIO_MAX = 2.0     # spend above 2x (transactions x avg basket) is treated as an entry error

LR_C = 0.3                # L2 strength of the logistic regression (validation sweep: flat between 0.01 and 10)
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  n_estimators=400, verbose=-1, seed=SEED, n_jobs=8)

# Columns of the label table that are computed over the label window itself: never used as features.
POST_OUTCOME_COLUMNS = ["LAST_PURCHASE_DATE", "MONTHS_OBSERVED", "INSUFFICIENT_HISTORY_FLAG", "OBSERVATION_END_DATE"]
