"""
Utility to order behavioural clusters by economic value
(median predicted CLV).

Typical usage
-------------
>>> df_clustered = cluster_customers(...)
>>> rank_table   = rank_clusters_by_median(df_clustered)
or
>>> df_enriched  = rank_clusters_by_median(df_clustered, inplace=True)
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def rank_clusters_by_median(
    df_clustered: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute each cluster's median predicted CLV and assign a dense rank
    (1 = highest median).

    Parameters
    ----------
    df_clustered : DataFrame
        Must include 'cluster_id' and 'predicted_clv'.
    inplace : bool, optional (default False)
        • False  → return a three-column summary table
                   [cluster_id, median_clv, cluster_rank].
        • True   → merge those two new columns back into *df_clustered*
                   and return the enriched DataFrame.

    Returns
    -------
    pd.DataFrame
        Either the summary table or the enriched original DF,
        depending on *inplace*.
    """
    required = {"cluster_id", "predicted_clv"}
    missing = required - set(df_clustered.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = (
        df_clustered
        .groupby("cluster_id", as_index=False)["predicted_clv"]
        .median()
        .rename(columns={"predicted_clv": "median_clv"})
        .sort_values("median_clv", ascending=False, ignore_index=True)
    )

    # Dense ranking: if two clusters share a median, they get the same rank.
    summary["cluster_rank"] = (
        summary["median_clv"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    logger.info("Calculated median CLV and ranked %s clusters", len(summary))

    if inplace:
        return df_clustered.merge(summary, on="cluster_id", how="left")

    return summary
