# scripts/train_model.py

import os
import sys
import yaml
import pandas as pd
import xgboost as xgb

# Import custom modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_loader
from src import feature_engineering
from src import modeling
from src import utils
from src.feature_engineering import select_training_features

# ==== Load Configuration ====

CONFIG_PATH = "config/model_config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# ==== Step 1: Load Raw Transaction Data ====

raw_data_path = "data/raw/transactions.csv"
df_raw = data_loader.load_raw_data(raw_data_path)

# ==== Step 2: Feature Engineering ====

df_features = feature_engineering.create_features(df_raw)
df_features = feature_engineering.generate_lifecycle_segments(df_features)

# Save processed features for scoring later
processed_data_path = "data/processed/processed_features.csv"
data_loader.save_processed_data(df_features, processed_data_path)

# ==== Step 3: Prepare Training Dataset ====

target_column = config["modeling"]["target_column"]  # e.g., "future_revenue"

# Features and target
X_train = select_training_features(df_features)
y_train = df_features[target_column]

# ==== Step 4: Train the Model ====

model = modeling.train_clv_model(X_train, y_train, config["modeling"])

# ==== Step 5: Save Trained Model and Metadata ====

model_output_path = f"models/clv_model_{utils.timestamp_now()}.joblib"
modeling.save_model(model, model_output_path)

# Save model training metadata (metrics, config snapshot)
metrics = modeling.track_model_performance(y_train, model.predict(X_train))
metadata = {
    "timestamp": utils.timestamp_now(),
    "config": config["modeling"],
    "metrics": metrics
}
metadata_output_path = "models/model_metadata.json"
modeling.save_model_metadata(metadata, metadata_output_path)

print("Model training completed and artifacts saved.")
