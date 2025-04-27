from sklearn.cluster import KMeans

def cluster_customers(df_predictions, n_clusters=5, random_state=42):
    """
    Cluster customers based on their predicted CLV only.

    Args:
        df_predictions (pd.DataFrame): DataFrame with at least customer_id and predicted_clv columns.
        n_clusters (int): Number of clusters to form.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Original df_predictions plus a cluster_id column.
    """
    df_clustered = df_predictions.copy()

    # Use predicted CLV only for clustering
    X_cluster = df_clustered[["predicted_clv"]]

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_cluster)

    df_clustered["cluster_id"] = cluster_labels

    print(f"Clustered {df_clustered.shape[0]} customers into {n_clusters} clusters.")

    return df_clustered
