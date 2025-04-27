import pandas as pd

def rank_customers(df_predictions):
    """
    Rank customers individually based on their predicted CLV.

    Args:
        df_predictions (pd.DataFrame): DataFrame with at least customer_id and predicted_clv columns.

    Returns:
        pd.DataFrame: DataFrame with customer_id, predicted_clv, clv_rank, clv_percentile, clv_segment.
    """
    df_ranked = df_predictions.copy()

    # Rank customers by predicted CLV descending
    df_ranked["clv_rank"] = df_ranked["predicted_clv"].rank(method="first", ascending=False)

    # Normalize ranks into percentiles
    df_ranked["clv_percentile"] = df_ranked["clv_rank"] / df_ranked.shape[0]

    # Assign CLV segment labels based on percentile bins
    df_ranked["clv_segment"] = pd.cut(
        df_ranked["clv_percentile"],
        bins=[0, 0.01, 0.05, 0.10, 0.20, 1.0],
        labels=["Top 1%", "Top 5%", "Top 10%", "Top 20%", "Others"]
    )

    print(f"Ranked {df_ranked.shape[0]} customers by predicted CLV.")

    return df_ranked
