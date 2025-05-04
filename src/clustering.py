"""
src/clustering.py
────────────────────────────────────────────────────────────
Assign behavioural clusters to customers.

Default feature mix (value + behaviour):
    • predicted_clv
    • tenure_days
    • recency_days
    • mean_days_between_orders
    • std1_days_between_orders
    • median_days_between_orders
"""

from __future__ import annotations

import logging
from typing import Sequence, Dict, Any, Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Main clustering helper
# ─────────────────────────────────────────────────────────────
def cluster_customers(
    df_predictions: pd.DataFrame,
    *,
    cfg: Dict[str, Any] | None = None,
    features: Sequence[str] = (
        "predicted_clv",
        "tenure_days",
        "recency_days",
        "mean_days_between_orders",
        "std1_days_between_orders",
        "median_days_between_orders",
    ),
    n_clusters: Optional[int] = None,
    random_state: Optional[int] = None,
    scale: bool = True,
) -> pd.DataFrame:
    """
    K-means cluster assignment.

    Parameters
    ----------
    df_predictions : DataFrame containing customer_id, predicted_clv,
                     and all columns named in `features`.
    cfg            : Parsed YAML config - pulls defaults from cfg["clustering"].
    features       : Feature columns for clustering.  Defaults to a hybrid
                     value-plus-behaviour set (see module docstring).
    n_clusters     : Overrides cfg value or default (5).
    random_state   : Overrides cfg value or default (42).
    scale          : Z-score features before K-means so no column dominates.

    Returns
    -------
    DataFrame — copy of input + cluster_id column (int).
    """

    # ─── default overrides from config ──────────────────────
    if cfg and "clustering" in cfg:
        n_clusters   = n_clusters   or cfg["clustering"].get("n_clusters", 5)
        random_state = random_state or cfg["clustering"].get("random_state", 42)

    n_clusters   = n_clusters   or 5
    random_state = random_state or 42

    # ─── check feature presence ─────────────────────────────
    missing = set(features) - set(df_predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns for clustering: {missing}")

    # ─── build feature matrix ───────────────────────────────
    X = df_predictions[list(features)].values
    if scale and X.shape[1] > 0:
        X = StandardScaler().fit_transform(X)

    # ─── run K-means ────────────────────────────────────────
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",           # safe for scikit-learn ≥1.4
    )
    labels = kmeans.fit_predict(X)

    # ─── output copy + cluster_id ───────────────────────────
    df_out = df_predictions.copy()
    df_out["cluster_id"] = labels.astype(int)

    logger.info(
        "Clustered %s customers into %s clusters using %s features",
        len(df_out), n_clusters, len(features)
    )
    return df_out
