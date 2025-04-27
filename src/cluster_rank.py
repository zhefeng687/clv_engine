import pandas as pd

def rank_clusters_by_median(df_clustered):
    """
    Calculate median predicted CLV per cluster and rank clusters.

    Args:
        df_clustered (pd.DataFrame): DataFrame with at least customer_id, predicted_clv, and cluster_id columns.

    Returns:
        pd.DataFrame: DataFrame with cluster_id, median_clv, cluster_rank.
    """
    # Calculate median predicted CLV per cluster
    cluster_medians = (
        df_clustered.groupby("cluster_id")["predicted_clv"]
        .median()
        .reset_index()
        .rename(columns={"predicted_clv": "median_clv"})
    )

    # Rank clusters: highest median CLV gets rank 1
    cluster_medians["cluster_rank"] = cluster_medians["median_clv"].rank(method="first", ascending=False)

    print(f"Calculated median CLV and ranked {cluster_medians.shape[0]} clusters.")

    return cluster_medians
