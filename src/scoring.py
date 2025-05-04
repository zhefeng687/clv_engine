"""
src/scoring.py
────────────────────────────────────────────────────────────
Reusable helpers for batch inference.

Public entry point
------------------
score_customers(raw_csv_path, cutoff, cfg, model_dir)
    • loads raw transactions
    • builds feature matrix for the chosen snapshot date
    • loads clv_model_latest.joblib
    • returns a DataFrame that contains
        customer_id,
        predicted_clv,
        tenure_days,
        recency_days,
        mean_days_between_orders,
        std1_days_between_orders,
        median_days_between_orders
      ready for downstream ranking or clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src import data_loader
from src.feature_engineering import make_dataset, select_training_features
from src.modeling import load_latest_model

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Main high-level helper
# ─────────────────────────────────────────────────────────────
def score_customers(
    raw_csv_path: str | Path,
    *,
    cutoff: pd.Timestamp,
    cfg: Dict[str, Any],
    model_dir: str | Path = "models",
) -> pd.DataFrame:
    """
    End-to-end scoring for one snapshot date.

    Parameters
    ----------
    raw_csv_path : str | Path
        Path to transactions CSV.
    cutoff       : pd.Timestamp
        Reference date that separates history from the future window.
    cfg          : dict
        Parsed YAML config (expects training.* keys).
    model_dir    : str | Path, optional
        Folder containing clv_model_latest.joblib.

    Returns
    -------
    pd.DataFrame
        Columns:
            customer_id
            predicted_clv
            tenure_days
            recency_days
            mean_days_between_orders
            std1_days_between_orders
            median_days_between_orders
    """
    logger.info("Scoring snapshot cutoff → %s", cutoff.date())

    # 1 ▸ Load raw data and build features (no labels at inference)
    df_raw = data_loader.load_raw_data(raw_csv_path)
    X, _ = make_dataset(
        df_raw,
        cutoff=cutoff,
        history_months=cfg["training"]["history_months"],
        pred_months=None,
    )
    X = select_training_features(X)

    if X.empty:
        raise ValueError("No customers found for the requested cutoff window.")

    # 2 ▸ Load production model
    model = load_latest_model(model_dir)

    # 3 ▸ Predict CLV
    preds = predict_clv(model, X)

    # 4 ▸ Package results with needed behavioural columns
    df_pred = X.reset_index()  # brings customer_id out of the index
    df_pred["predicted_clv"] = preds

    keep_cols = [
        "customer_id",
        "predicted_clv",
        "tenure_days",
        "recency_days",
        "mean_days_between_orders",
        "std1_days_between_orders",
        "median_days_between_orders",
    ]
    df_pred = df_pred[[c for c in keep_cols if c in df_pred.columns]]

    logger.info(
        "Prepared prediction DataFrame with %s customers, %s feature columns",
        len(df_pred),
        len(df_pred.columns) - 1,  # exclude customer_id
    )
    return df_pred


# ─────────────────────────────────────────────────────────────
# Low-level helper
# ─────────────────────────────────────────────────────────────
def predict_clv(model, X_features: pd.DataFrame) -> np.ndarray:
    """Wrapper around model.predict with logging."""
    y_pred = model.predict(X_features)
    logger.info("Predicted CLV for %s customers", len(y_pred))
    return y_pred

