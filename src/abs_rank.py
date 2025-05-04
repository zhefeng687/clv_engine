"""
AAssign absolute rank, percentile, and segment labels
to each customer's predicted CLV.

Used in: scripts/score_and_rank_customers.py
Typical usage:
    df_scores = score_customers(...)
    df_ranked = add_absolute_rank(df_scores, cfg)
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────
# Main ranking function
# ───────────────────────────────────────────────────────────
def add_absolute_rank(
    df_predictions: pd.DataFrame,
    cfg: dict | None = None,
    *,
    bins: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Add CLV-based rank, percentile, and segment label columns.

    Parameters
    ----------
    df_predictions : DataFrame
        Must contain columns ['customer_id', 'predicted_clv'].
    cfg : dict, optional
        YAML config. Used to extract scoring.bins / labels if provided.
    bins : list of percentile thresholds (overrides config)
    labels : list of human-readable segment names (overrides config)

    Returns
    -------
    DataFrame with additional columns:
        • clv_rank         (1 = highest CLV)
        • clv_percentile   (0-1)
        • clv_segment      (label from `labels`)
    """

    required_cols = {"customer_id", "predicted_clv"}
    if not required_cols.issubset(df_predictions.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df_predictions.columns)}")
    
    # pull bins/labels from config if not passed explicitly
    if bins is None or labels is None:
        if cfg and "scoring" in cfg:
            bins   = bins   or cfg["scoring"].get("percentile_bins")
            labels = labels or cfg["scoring"].get("segment_labels")

        # fallback defaults
        bins   = bins   or [0, 0.01, 0.05, 0.10, 0.20, 1.0]
        labels = labels or ["Top 1%", "Top 5%", "Top 10%", "Top 20%", "Others"]

    df_ranked = df_predictions.copy()

    # absolute rank (1 = highest CLV)
    df_ranked["clv_rank"] = df_ranked["predicted_clv"] \
        .rank(method="dense", ascending=False).astype(int) 

    # percentile (0-1, lower is better customer)
    df_ranked["clv_percentile"] = df_ranked["clv_rank"] / len(df_ranked)

    # segment label
    df_ranked["clv_segment"] = pd.cut(
        df_ranked["clv_percentile"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    logger.info("Assigned rank + segment to %s customers", len(df_ranked))
    return df_ranked
