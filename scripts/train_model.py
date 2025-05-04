
"""
Train the production CLV model
──────────────────────────────
Reads champion windows + hyper-params from model_config.yaml,
fits one XGBoost regressor on the full history window, and
saves:

• models/clv_model_<timestamp>.joblib  (timestamped artefact)
• models/clv_model_latest.joblib       (symlink / copy)
• models/clv_model_<timestamp>.metrics.json
• models/clv_model_latest.metrics.json
"""

from __future__ import annotations

import sys, yaml, logging
from pathlib import Path
from typing import Dict, Any


import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor

# ─────────────────────── Setup ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

CONFIG_PATH  = PROJECT_ROOT / "config" / "model_config.yaml"
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

from src import data_loader
from src.feature_engineering import make_dataset, select_training_features
from src.modeling import save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# ───────────────────────── 1 ▸ load data ────────────────────────────
with open(CONFIG_PATH) as f:
    cfg: Dict[str, Any] = yaml.safe_load(f)

# choose snapshot date
df_raw = data_loader.load_raw_data(DATA_PATH)
cutoff = pd.Timestamp(cfg["training"].get("cutoff_date")) \
         if cfg["training"].get("cutoff_date") else df_raw["order_date"].max()

log.info("Training snapshot cutoff → %s", cutoff.date())

# ───────────────────────── 2 ▸ build dataset ────────────────────────
X, y = make_dataset(
    df_raw,
    cutoff=cutoff,
    history_months = cfg["training"]["history_months"],
    pred_months    = cfg["training"]["pred_months"],
)
X = select_training_features(X)

if X.empty or y is None:
    raise ValueError("Training data is empty or label is missing")

# ─────────────────────── 3. Train final model ───────────────────────
model = XGBRegressor(**cfg["modeling"])
model.fit(X, y)

y_pred = model.predict(X)
rmse = root_mean_squared_error(y, y_pred)
r2   = r2_score(y, y_pred)
metrics = {"rmse": float(rmse), "r2": float(r2)}

log.info("Training RMSE = %.3f | R² = %.3f", rmse, r2)

# ─────────────────────── 4. Save artefacts ───────────────────────
save_model(model, MODEL_DIR, metrics=metrics)

log.info("Training complete.")