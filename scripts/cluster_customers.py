import os
import sys
import yaml
import pandas as pd

# Import custom modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_loader
from src import clustering
from src import utils

# ==== Load Configuration ====

CONFIG_PATH = "config/model_config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# ==== Step 1: Load Predicted Customer CLV Data ====

predictions_path = "outputs/clv_ranked_predictions.csv"  # Predictions + customer_id + predicted_clv
df_predictions = data_loader.load_processed_data(predictions_path)

# ==== Step 2: Cluster Customers Based on Predicted CLV ====

n_clusters = config["clustering"]["n_clusters"]
random_state = config["clustering"]["random_state"]

df_clustered = clustering.cluster_customers(
    df_predictions=df_predictions,
    n_clusters=n_clusters,
    random_state=random_state
)

# ==== Step 3: Save Clustered Customer Data ====

output_path = "outputs/clv_clustered_customers.csv"
utils.save_dataframe(df_clustered, output_path)

print(f"Clustered customers saved to {output_path}")
