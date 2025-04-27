"""
Predict Customer Lifetime Value (CLV) and Rank Customers by Predicted CLV.
"""

import os
import sys
import yaml
import pandas as pd

# Import custom modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import modeling
from src import data_loader
from src import scoring
from src import abs_rank
from src import utils

# ==== Load Configuration ====

CONFIG_PATH = "config/model_config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# ==== Step 1: Load Model and Features ====

model_path = "models/clv_model_latest.joblib"  # Update filename if needed
features_path = "data/processed/processed_features.csv"

model = modeling.load_model(model_path)
df_features = data_loader.load_processed_data(features_path)

# ==== Step 2: Prepare X_features for Prediction ====

X_features = df_features.drop(columns=["customer_id", "future_revenue", "customer_stage", "status_segment"], errors="ignore")

# ==== Step 3: Predict CLV ====

y_pred = scoring.predict_clv(model, X_features)

# ==== Step 4: Prepare Prediction DataFrame ====

df_predictions = scoring.prepare_prediction_df(df_features["customer_id"], y_pred)

# ==== Step 5: Rank Customers ====

df_ranked = abs_rank.rank_customers(df_predictions)

# ==== Step 6: Save Predictions ====

output_path = "outputs/clv_ranked_predictions.csv"
utils.save_dataframe(df_ranked, output_path)

print(f"Customer CLV predictions and rankings saved to {output_path}")
