import os
import sys
import yaml
import pandas as pd

# Import custom modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_loader
from src import cluster_rank
from src import utils

# ==== Load Configuration ====

CONFIG_PATH = "config/model_config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# ==== Step 1: Load Clustered Customer Data ====

clustered_path = "outputs/clv_clustered_customers.csv"  # After clustering customers
df_clustered = data_loader.load_processed_data(clustered_path)

# ==== Step 2: Rank Clusters by Median CLV ====

df_cluster_medians = cluster_rank.rank_clusters_by_median(df_clustered)

# ==== Step 3: Save Cluster Rankings ====

output_path = "outputs/clv_cluster_rankings.csv"
utils.save_dataframe(df_cluster_medians, output_path)

print(f"Cluster median CLV rankings saved to {output_path}")
