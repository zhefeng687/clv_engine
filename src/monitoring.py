import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import root_mean_squared_error, r2_score

def calculate_feature_drift(current_features, reference_features, threshold=0.1):
    """
    Detect feature drift using Kolmogorov-Smirnov (KS) test.
    """
    drift_report = {}
    for col in current_features.columns:
        stat, p_value = ks_2samp(reference_features[col].dropna(), current_features[col].dropna())
        drift_report[col] = p_value < threshold  # True if drift detected
    print(f"Calculated feature drift for {len(current_features.columns)} features.")
    return drift_report

def calculate_prediction_drift(y_true, y_pred, baseline_rmse, baseline_r2, rmse_threshold=0.20, r2_threshold=0.15):
    """
    Detect prediction drift by comparing RMSE and R² to baseline.
    """
    current_rmse = root_mean_squared_error(y_true, y_pred)
    current_r2 = r2_score(y_true, y_pred)

    rmse_drift = (current_rmse - baseline_rmse) / baseline_rmse > rmse_threshold
    r2_drift = (baseline_r2 - current_r2) / baseline_r2 > r2_threshold

    print(f"Calculated prediction drift: Current RMSE={current_rmse:.4f}, Current R²={current_r2:.4f}")
    return {
        "rmse_drift": rmse_drift,
        "r2_drift": r2_drift,
        "current_rmse": current_rmse,
        "current_r2": current_r2
    }
