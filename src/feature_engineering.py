
"""
src/feature_engineering.py
--------------------------------------------------------------------
Single source of customer-level features and (optionally) CLV labels.

Design
------
• Parameterised on `history_months` and `pred_months`.
• Cut-off date separates "history" and "future" windows                │
      ┌──────────── history_months ───────────┐┌─ pred_months ───┐
 timeline: ──┬───────────────(cutoff)─────────┬──────────────────┬──>
             ↓                               ↓                  ↓
       features built here            target summed here   unseen future
• No data leakage: every row used for the label lies strictly *after*
  features in time.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------#
# Core feature builder
# ------------------------------------------------------------------#
def build_features(
    df_raw: pd.DataFrame,
    *,
    cutoff: Optional[pd.Timestamp] = None,
    history_months: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create customer-level R/F/M features, behaviour flags, and cadence stats.

    Parameters
    ----------
    df_raw : DataFrame with columns [customer_id, order_date, revenue]
    cutoff : Snapshot date separating history from future.  If None, the
             latest order_date in df_raw is used (with a warning).
    history_months : How many months of history to keep.  If None, use all.

    Returns
    -------
    DataFrame indexed by customer_id with engineered features.
    """

    # 1 ─ ensure order_date is datetime (idempotent)
    if not np.issubdtype(df_raw["order_date"].dtype, np.datetime64):
        df_raw = df_raw.copy()  # avoid mutating caller
        df_raw["order_date"] = pd.to_datetime(df_raw["order_date"])

    # 2 ----- choose cutoff -----------------------------------------------------
    if cutoff is None:
        cutoff = df_raw["order_date"].max()
        logger.warning(
            "No 'cutoff' supplied — defaulting to latest order_date %s. "
            "For reproducible back-tests and to avoid future data leakage, "
            "pass an explicit cutoff.",
            cutoff.date(),
        )

    # 3 - restrict to rolling history window if requested
    if history_months is not None:
        start_date = cutoff - relativedelta(months=history_months)
        mask = (df_raw["order_date"] >= start_date) & (df_raw["order_date"] < cutoff)
        df_hist = df_raw.loc[mask].copy()
    else:
        df_hist = df_raw.loc[df_raw["order_date"] < cutoff].copy()

    if df_hist.empty:
        raise ValueError("No transactions in the selected history window")

    # 4 ─ aggregate to customer level
    features = (
        df_hist.groupby("customer_id")
        .agg(
            revenue_sum=("revenue", "sum"),
            revenue_count=("revenue", "count"),
            avg_order_value=("revenue", "mean"),
            first_purchase=("order_date", "min"),
            last_purchase=("order_date", "max"),
            order_dates=("order_date", list),
        )
        .reset_index()
        .set_index("customer_id")
    )

    # 5 ─ lifecycle metrics
    features["tenure_days"] = (cutoff - features["first_purchase"]).dt.days
    features["recency_days"] = (cutoff - features["last_purchase"]).dt.days
    features["log_revenue_sum"] = np.log1p(features["revenue_sum"])

    # 6 ─ behaviour flags
    features["is_repeat_buyer"] = features["revenue_count"] > 1
    features["active_last_30d"] = features["recency_days"] <= 30

    # 7 ─ cadence statistics
    cadence_stats = features["order_dates"].apply(_calculate_cadence_features)
    cadence_stats.columns = [
        "mean_days_between_orders",
        "std_days_between_orders",
        "median_days_between_orders",
    ]
    features = pd.concat([features, cadence_stats], axis=1)

    # 8 ─ drop helper columns
    features.drop(
        columns=["first_purchase", "last_purchase", "order_dates"], inplace=True
    )

    logger.info(
        "Feature engineering complete: %s customers | %s features",
        *features.shape,
    )
    return features

# ------------------------------------------------------------------#
# Helper: cadence stats
# ------------------------------------------------------------------#
def _calculate_cadence_features(order_dates: list[pd.Timestamp]) -> pd.Series:
    """
    Return mean, std (sample), median of gaps (days) between consecutive orders.
    """
    if len(order_dates) < 2:
        return pd.Series([np.nan, np.nan, np.nan], dtype="float64")

    dates_sorted = np.sort(np.array(order_dates, dtype="datetime64[ns]"))
    gaps = np.diff(dates_sorted).astype("timedelta64[D]").astype(float)
    return pd.Series(
        [gaps.mean(), gaps.std(ddof=1), np.median(gaps)], dtype="float64"
    )

# ------------------------------------------------------------------#
# Label builder
# ------------------------------------------------------------------#
def build_label(
    df_raw: pd.DataFrame,
    cutoff: pd.Timestamp,
    pred_months: int,
) -> pd.Series:
    """
    Total revenue in the forward window [cutoff, cutoff + pred_months).
    """
    end_date = cutoff + relativedelta(months=pred_months)
    fut = df_raw.loc[
        (df_raw["order_date"] >= cutoff) & (df_raw["order_date"] < end_date)
    ]
    y = fut.groupby("customer_id")["revenue"].sum()
    y.name = "future_revenue"
    return y

# ------------------------------------------------------------------#
# Convenience wrapper
# ------------------------------------------------------------------#
def make_dataset(
    df_raw: pd.DataFrame,
    cutoff: pd.Timestamp,
    history_months: int,
    pred_months: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
     Build X (and y if pred_months is provided) in one call.
    """
    X = build_features(
        df_raw, cutoff=cutoff, history_months=history_months
    )

    if pred_months is None:
        return X, None

    y = build_label(df_raw, cutoff, pred_months)
    y = y.reindex(X.index, fill_value=0)
    return X, y


# ------------------------------------------------------------------#
# Train-time selector
# ------------------------------------------------------------------#
def select_training_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Drop label and any lifecycle buckets before modelling.
    """
    return df_features.drop(columns=["future_revenue"], errors="ignore")




'''
def generate_lifecycle_segments(df_features):
    """
    Assign customer_stage and status_segment based on tenure_days and recency_days.

    Args:
        df_features (pd.DataFrame): Feature-engineered customer-level data.

    Returns:
        pd.DataFrame: DataFrame with lifecycle segments.
    """

    # Define customer_stage based on tenure_days
    df_features['customer_stage'] = pd.cut(
        df_features['tenure_days'],
        bins=[-np.inf, 90, 180, 365, np.inf],
        labels=["New", "Growing", "Established", "Loyal"]
    )

    # Define status_segment based on recency_days
    df_features['status_segment'] = pd.cut(
        df_features['recency_days'],
        bins=[-np.inf, 30, 90, 180, np.inf],
        labels=["Active", "Lagging", "Dormant", "Churn-risk"]
    )

    print("Lifecycle segmentation completed (customer_stage and status_segment assigned).")

    return df_features

def select_training_features(df_features):
    """
    Select training features by dropping non-predictive columns.

    Args:
        df_features (pd.DataFrame): DataFrame with full feature set.

    Returns:
        pd.DataFrame: DataFrame with only features for model training.
    """
    drop_cols = ["customer_id", "future_revenue", "customer_stage", "status_segment"]
    X_features = df_features.drop(columns=drop_cols, errors="ignore")
    
    return X_features
'''