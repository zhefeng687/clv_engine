# scripts/monitoring_check.py

import os
import sys
import yaml
import pandas as pd

# Import custom modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import monitoring
from src import data_loader
from src import modeling
from src import scoring
from src import utils

# ==== Load Configuration ====

CONFIG_PATH = "config/model_config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# ==== Helper: Create Monitoring Folder ====

monitoring_output_dir = "outputs/monitoring/"
utils.create_folder_if_not_exists(monitoring_output_dir)

# Generate timestamp
timestamp = utils.timestamp_now()

# ==== Step 1: Load Reference and Current Features ====
# training time
reference_path = "data/processed/processed_features_baseline.csv"  

# latest batch
current_path = "data/processed/processed_features.csv"

df_reference = data_loader.load_processed_data(reference_path)
df_current = data_loader.load_processed_data(current_path)

drop_cols = ["customer_id", "future_revenue", "customer_stage", "status_segment"]

reference_features = df_reference.drop(columns=drop_cols, errors="ignore")
current_features = df_current.drop(columns=drop_cols, errors="ignore")

# ==== Step 2: Feature Drift Monitoring ====

print("\nRunning Feature Drift Monitoring...")
drift_report = monitoring.calculate_feature_drift(current_features, reference_features, threshold=0.1)

# Print Report
print("\nFeature Drift Report:")
for feature, drifted in drift_report.items():
    status = "Drift Detected" if drifted else "Stable"
    print(f" - {feature}: {status}")

# Save Feature Drift Report
feature_drift_df = pd.DataFrame({
    "feature": list(drift_report.keys()),
    "drift_detected": list(drift_report.values())
})
feature_drift_path = os.path.join(monitoring_output_dir, f"feature_drift_report_{timestamp}.csv")
feature_drift_df.to_csv(feature_drift_path, index=False)
print(f"Feature Drift Report saved to {feature_drift_path}")

# ==== Step 3: Prediction Drift Monitoring (Optional) ====
true_labels_available = "future_revenue" in df_current.columns

if true_labels_available:
    print("\nRunning Prediction Drift Monitoring...")

    model_path = "models/clv_model_latest.joblib"
    model = modeling.load_model(model_path)

    X_features = df_current.drop(columns=["customer_id", "future_revenue", "customer_stage", "status_segment"], errors="ignore")
    y_true = df_current["future_revenue"]

    y_pred = scoring.predict_clv(model, X_features)

    baseline_metadata_path = "models/model_metadata.json"
    with open(baseline_metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    baseline_rmse = metadata["metrics"]["RMSE"]
    baseline_r2 = metadata["metrics"]["R2"]

    prediction_drift_report = monitoring.calculate_prediction_drift(
        y_true,
        y_pred,
        baseline_rmse,
        baseline_r2,
        rmse_threshold=config["monitoring"]["drift_threshold_rmse"],
        r2_threshold=config["monitoring"]["drift_threshold_r2"]
    )

    # Print Report
    print("\nPrediction Drift Report:")
    for key, value in prediction_drift_report.items():
        print(f" - {key}: {value}")

    # Save Prediction Drift Report
    import json
    prediction_drift_path = os.path.join(monitoring_output_dir, f"prediction_drift_report_{timestamp}.json")
    with open(prediction_drift_path, "w") as f:
        json.dump(prediction_drift_report, f, indent=4)
    print(f"Prediction Drift Report saved to {prediction_drift_path}")

else:
    print("\nSkipping Prediction Drift Monitoring (no true labels available in latest data).")

# ==== Step 4: Monitoring Completed ====

print("\nMonitoring Check Completed and Reports Saved!")
