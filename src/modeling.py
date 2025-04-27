# src/modeling.py

import os
import json
import joblib
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score

def train_clv_model(X_train, y_train, config):
    """
    Train a CLV prediction model (XGBoost Regressor) using config parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target variable (future revenue).
        config (dict): Model hyperparameters from config file.

    Returns:
        model: Trained XGBoost model object.
    """

    model = xgb.XGBRegressor(**config["modeling"])

    model.fit(
        X_train, 
        y_train,
        eval_metric=config["training"]["eval_metric"],
        verbose=True
    )

    print("Model training completed.")

    return model

def save_model(model, filepath):
    """
    Save trained model as a .joblib file.

    Args:
        model: Trained model object.
        filepath (str): Destination filepath.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}.")

def load_model(filepath):
    """
    Load a trained model from a .joblib file.

    Args:
        filepath (str): Path to the saved model.

    Returns:
        model: Loaded model object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}.")
    return model

def track_model_performance(y_true, y_pred):
    """
    Track basic model performance metrics (RMSE, R2).

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        dict: Dictionary of metrics.
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Model Performance: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    return {
        "RMSE": rmse,
        "R2": r2
    }

def save_model_metadata(metadata_dict, filepath):
    """
    Save model training metadata (metrics, hyperparameters, timestamp) as JSON.

    Args:
        metadata_dict (dict): Metadata to save.
        filepath (str): Destination JSON file path.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"Model metadata saved to {filepath}.")
