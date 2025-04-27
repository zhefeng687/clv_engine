import pandas as pd

def predict_clv(model, X_features):
    """
    Predict customer CLV using a trained model.

    Args:
        model: Trained XGBoost model (or compatible regressor).
        X_features (pd.DataFrame): Features for prediction.

    Returns:
        np.array: Predicted CLV values.
    """
    y_pred = model.predict(X_features)
    print(f"Predicted CLV for {len(y_pred)} customers.")
    return y_pred

def prepare_prediction_df(customer_ids, y_pred):
    """
    Assemble a DataFrame with customer_id and predicted CLV.

    Args:
        customer_ids (array-like): List of customer IDs.
        y_pred (array-like): Predicted CLV values.

    Returns:
        pd.DataFrame: DataFrame with customer_id and predicted_clv.
    """
    df_pred = pd.DataFrame({
        "customer_id": customer_ids,
        "predicted_clv": y_pred
    })
    print(f"Prepared prediction DataFrame with {df_pred.shape[0]} customers.")
    return df_pred
