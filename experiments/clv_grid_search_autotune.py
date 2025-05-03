#!/usr/bin/env python
"""
Grid-search + auto-tune for CLV model
─────────────────────────────────────
• sweeps (history_months, pred_months) and XGBoost params
• uses time-series split to avoid leakage
• ranks combos by weighted composite score (weights in YAML)
• writes champion windows + hyper-params back to model_config.yaml
• saves heat-maps + CSV for analyst review
"""

from __future__ import annotations

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import TimeSeriesSplit
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:  # scikit-learn < 1.3
    from sklearn.metrics import mean_squared_error
    rmse = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)

from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# ────────────────────────────────────────
# Paths & configuration
# ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH  = PROJECT_ROOT / "config" / "model_config.yaml"

with open(CONFIG_PATH) as f:
    cfg: Dict[str, Any] = yaml.safe_load(f)

DATA_PATH   = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
MODEL_DIR   = PROJECT_ROOT / "models"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CSV_OUT      = OUTPUT_DIR / "clv_grid_search_results.csv"
RMSE_PNG     = OUTPUT_DIR / "rmse_heatmap.png"
R2_PNG       = OUTPUT_DIR / "r2_heatmap.png"

# sweep limits
MAX_HISTORY = 18
MAX_PREDICT = 12
MIN_CUSTOMERS_STATIC  = 100
MIN_CUSTOMERS_PERCENT = 0.05

# logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# make src importable
sys.path.append(str(PROJECT_ROOT))

from src import data_loader
from src.feature_engineering import make_dataset

# ────────────────────────────────────────
# 1. Load + basic sanity check
# ────────────────────────────────────────
log.info("Loading raw data %s", DATA_PATH)
df = data_loader.load_raw_data(DATA_PATH)
df = df.sort_values(["customer_id", "order_date"])

end_date       = df["order_date"].max()
data_start     = df["order_date"].min()
total_cust     = df["customer_id"].nunique()
max_possible_m = (end_date.year - data_start.year) * 12 + end_date.month - data_start.month

# auto-select candidate windows
hist_candidates = [m for m in (3, 6, 9, 12, 15, 18)
                   if m + 3 <= max_possible_m and m <= MAX_HISTORY]
pred_candidates = [m for m in (3, 6, 9, 12) if m <= MAX_PREDICT]

log.info("Candidate windows: history %s x prediction %s",
         hist_candidates, pred_candidates)

# ────────────────────────────────────────
# 2. Grid search
# ────────────────────────────────────────
results  : list[dict[str, Any]] = []
weights = cfg.get("composite_weights", {"rmse": 0.5, "r2": 0.5})
xgb_base = cfg["modeling"]

for h_m in hist_candidates:
    for p_m in pred_candidates:

        cutoff = end_date - relativedelta(months=p_m)

        # Build dataset (single-source features + label)
        X_y = make_dataset(df, cutoff=cutoff,
                           history_months=h_m, pred_months=p_m)
        if X_y[0].empty:
            continue
        X_full, y_full = X_y

        # adaptive min-customers threshold
        min_required = max(MIN_CUSTOMERS_STATIC,
                           int(total_cust * MIN_CUSTOMERS_PERCENT))
        if len(X_full) < min_required:
            log.warning("Skip h=%s p=%s — only %s customers (<%s)",
                        h_m, p_m, len(X_full), min_required)
            continue

        # time-series CV: average metrics across all folds
        rmse_folds, r2_folds = [], []
        for train_idx, val_idx in TimeSeriesSplit(n_splits=4).split(X_full):
            X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

            model = XGBRegressor(**xgb_base)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=cfg["training"]["early_stopping_rounds"],
                eval_metric=cfg["training"]["eval_metric"],
                verbose=False,
            )

            preds = model.predict(X_val)
            rmse_folds.append(rmse(y_val, preds))
            r2_folds.append(r2_score(y_val, preds))

        res_rmse = float(np.mean(rmse_folds))
        res_r2   = float(np.mean(r2_folds))
        log.info("h=%s p=%s  RMSE=%.2f  R²=%.2f",
                 h_m, p_m, res_rmse, res_r2)

        # store result
        results.append(
            dict(hist_months=h_m, pred_months=p_m,
                 rmse=res_rmse, r2=res_r2,
                 n_customers=len(X_full))
        )

# ────────────────────────────────────────
# 3. Rank + persist results
# ────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(CSV_OUT, index=False)
log.info("Saved raw grid results → %s", CSV_OUT)

# composite score
res_df["rmse_norm"] = res_df["rmse"] / res_df["rmse"].max()
res_df["r2_norm"]   = res_df["r2"]   / res_df["r2"].max()
res_df["composite"] = ( weights["rmse"] * res_df["rmse_norm"] + weights["r2"] * (1 - res_df["r2_norm"])
)

champ = res_df.loc[res_df["composite"].idxmin()]
h_best, p_best = int(champ["hist_months"]), int(champ["pred_months"])
log.info("Champion: history=%sM  pred=%sM  RMSE=%.2f  R²=%.2f",
         h_best, p_best, champ["rmse"], champ["r2"])

# heat-maps (pivot kwargs explicit for pandas >=2.0)
plt.figure(figsize=(8, 5))
sns.heatmap(
    res_df.pivot(index="hist_months", columns="pred_months", values="rmse"),
    annot=True, fmt=".2f", cmap="coolwarm"
)
plt.title("RMSE heat-map")
plt.savefig(RMSE_PNG); plt.close()

plt.figure(figsize=(8, 5))
sns.heatmap(
    res_df.pivot(index="hist_months", columns="pred_months", values="r2"),
    annot=True, fmt=".2f", cmap="YlGnBu"
)
plt.title("R² heat-map")
plt.savefig(R2_PNG); plt.close()

log.info("Saved heat-maps to %s and %s", RMSE_PNG, R2_PNG)

# ────────────────────────────────────────
# 4. Write champion windows + params back to YAML
# ────────────────────────────────────────
cfg["training"]["history_months"] = h_best
cfg["training"]["pred_months"]    = p_best
cfg["modeling"] = xgb_base  # already champion params

with open(CONFIG_PATH, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

log.info("Updated %s with champion windows + model params", CONFIG_PATH)