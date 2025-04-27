# src/feature_engineering.py

import pandas as pd
import numpy as np

def create_features(df_raw):
    """
    Create feature set from raw transaction data.
    
    Args:
        df_raw (pd.DataFrame): Raw transactional data with at least customer_id, order_date, revenue columns.

    Returns:
        pd.DataFrame: Feature-engineered customer-level data.
    """

    # Ensure date format
    df_raw['order_date'] = pd.to_datetime(df_raw['order_date'])

    # Reference date for recency, tenure calculations (latest transaction date)
    reference_date = df_raw['order_date'].max()

    # Group by customer_id to create aggregated features
    features = (
        df_raw.groupby('customer_id')
        .agg(
            revenue_sum=('revenue', 'sum'),
            revenue_count=('revenue', 'count'),
            avg_order_value=('revenue', 'mean'),
            first_purchase=('order_date', 'min'),
            last_purchase=('order_date', 'max'),
            order_dates=('order_date', list)
        )
        .reset_index()
    )

    # Create lifecycle features
    features['tenure_days'] = (reference_date - features['first_purchase']).dt.days
    features['recency_days'] = (reference_date - features['last_purchase']).dt.days
    features['log_revenue_sum'] = np.log1p(features['revenue_sum'])  # Log transform

    # Binary flags
    features['is_repeat_buyer'] = features['revenue_count'] > 1
    features['active_last_30d'] = features['recency_days'] <= 30

    # Cadence statistics: Mean, Median, Std days between orders
    features[['mean_days_between_orders', 'std_days_between_orders', 'median_days_between_orders']] = features['order_dates'].apply(
        lambda dates: _calculate_cadence_features(dates)
    )

    # Drop intermediate columns
    features.drop(columns=['first_purchase', 'last_purchase', 'order_dates'], inplace=True)

    print(f"Feature engineering completed: {features.shape[0]} customers, {features.shape[1]} features.")

    return features

def _calculate_cadence_features(order_dates):
    """
    Calculate mean, std, median of days between orders.
    
    Args:
        order_dates (list of datetime): List of order dates for a customer.

    Returns:
        pd.Series: [mean_gap, std_gap, median_gap]
    """
    if len(order_dates) < 2:
        return pd.Series([np.nan, np.nan, np.nan])

    order_dates_sorted = sorted(order_dates)
    gaps = [(order_dates_sorted[i] - order_dates_sorted[i-1]).days for i in range(1, len(order_dates_sorted))]

    return pd.Series([np.mean(gaps), np.std(gaps), np.median(gaps)])

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
