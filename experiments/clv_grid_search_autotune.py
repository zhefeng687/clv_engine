
import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from joblib import dump

from xgboost import XGBRegressor

# Import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_loader
from src.feature_engineering import select_training_features

# ==== Config ====

CONFIG_PATH = "config/model_config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

DATA_PATH = "data/raw/transactions.csv"
MODEL_DIR = "models/"
OUTPUT_CSV = "outputs/clv_grid_search_results.csv"
RMSE_HEATMAP = "outputs/rmse_heatmap.png"
R2_HEATMAP = "outputs/r2_heatmap.png"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("outputs/", exist_ok=True)

MIN_CUSTOMERS_PERCENT = 0.05
MIN_CUSTOMERS_STATIC = 100
MAX_HISTORY = 18
MAX_PREDICT = 12

# ==== Load Raw Data ====

print("Loading raw data...")
df = data_loader.load_raw_data(DATA_PATH)
df['order_date'] = pd.to_datetime(df['order_date'])
df = df.sort_values(by=["customer_id", "order_date"])
end_date = df['order_date'].max()
total_customers = df["customer_id"].nunique()

# ==== Auto-Select Time Windows ====

data_start = df["order_date"].min()
max_possible_months = (end_date.year - data_start.year) * 12 + end_date.month - data_start.month

hist_months_list = [m for m in [3, 6, 9, 12, 15, 18] if m + 3 <= max_possible_months and m <= MAX_HISTORY]
pred_months_list = [m for m in [3, 6, 9, 12] if m <= MAX_PREDICT]

results = []

print(f"Auto-selected windows: History {hist_months_list} × Prediction {pred_months_list}")

# === Utility: Cadence Feature Calculation
def order_gap_stats(order_dates):
    if len(order_dates) < 2:
        return pd.Series([np.nan, np.nan, np.nan])
    order_dates_sorted = sorted(order_dates)
    deltas = [(order_dates_sorted[i] - order_dates_sorted[i - 1]).days for i in range(1, len(order_dates_sorted))]
    return pd.Series([np.mean(deltas), np.std(deltas), np.median(deltas)])

# ==== Grid Search ====

results = []

for hist_months in hist_months_list:
    for pred_months in pred_months_list:
        
        cutoff = end_date - relativedelta(months=pred_months)
        hist_start = cutoff - relativedelta(months=hist_months)

        hist_df = df[(df['order_date'] >= hist_start) & (df['order_date'] <= cutoff)]
        future_df = df[(df['order_date'] > cutoff) & (df['order_date'] <= cutoff + relativedelta(months=pred_months))]

        if hist_df.empty or future_df.empty:
            continue

        # ==== Feature Engineering ====
        features = (
            hist_df.groupby("customer_id")
            .agg(
                revenue_sum=("revenue", "sum"),
                revenue_count=("revenue", "count"),
                avg_order_value=("revenue", "mean"),
                last_purchase=("order_date", "max"),
                first_purchase=("order_date", "min"),
                order_dates=("order_date", list)
            )
            .reset_index()
        )

        # Calculate lifecycle features
        features["recency_days"] = (cutoff - features["last_purchase"]).dt.days
        features["tenure_days"] = (cutoff - features["first_purchase"]).dt.days

        # Log-transformed revenue_sum
        features["log_revenue_sum"] = np.log1p(features["revenue_sum"])

        # Binary flags
        features["is_repeat_buyer"] = features["revenue_count"] > 1
        features["active_last_30d"] = features["recency_days"] <= 30

        # Cadence features
        features[['mean_days_between_orders', 'std_days_between_orders', 'median_days_between_orders']] = features['order_dates'].apply(order_gap_stats)

        # Lifecycle Segmentation
        features['customer_stage'] = pd.cut(
            features['tenure_days'],
            bins=[-np.inf, 90, 180, 365, np.inf],
            labels=["New", "Growing", "Established", "Loyal"]
        )

        features['status_segment'] = pd.cut(
            features['recency_days'],
            bins=[-np.inf, 30, 90, 180, np.inf],
            labels=["Active", "Lagging", "Dormant", "Churn-risk"]
        )

        features.drop(columns=["order_dates", "first_purchase", "last_purchase"], inplace=True)

        # ==== Targets ====
        targets = (
            future_df.groupby("customer_id")
            .agg(future_revenue=("revenue", "sum"))
            .reset_index()
        )

        data = pd.merge(features, targets, on="customer_id", how="left")
        data["future_revenue"] = data["future_revenue"].fillna(0)

        # Adaptive minimum customers filter
        min_customers_required = max(MIN_CUSTOMERS_STATIC, int(total_customers * MIN_CUSTOMERS_PERCENT))
        if len(data) < min_customers_required:
            print(f"Skipping hist={hist_months}, pred={pred_months} (only {len(data)} customers)")
            continue

        X = select_training_features(data)
        y = data["future_revenue"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=config["training"]["test_size"], 
            random_state=config["modeling"]["random_state"]
        )

        model = XGBRegressor(**config["modeling"])

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=config["training"]["early_stopping_rounds"],
            eval_metric=config["training"]["eval_metric"],
            verbose=False
        )

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        results.append({
            "hist_months": hist_months,
            "pred_months": pred_months,
            "rmse": rmse,
            "r2": r2,
            "n_customers": len(data)
        })

        print(f"Done hist={hist_months} → pred={pred_months} | RMSE={rmse:.2f} | R²={r2:.2f}")


# ==== Save Grid Search Results ====

results_df = pd.DataFrame(results)

# Normalize RMSE and R² for composite scoring
rmse_max = results_df["rmse"].max()
r2_max = results_df["r2"].max()

results_df["rmse_norm"] = results_df["rmse"] / rmse_max
results_df["r2_norm"] = results_df["r2"] / r2_max

# Weighted Sum Composite Score
results_df["composite"] = 0.5 * results_df["rmse_norm"] + 0.5 * (1 - results_df["r2_norm"])

# Save results
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Grid search results saved to {OUTPUT_CSV}")

# ==== Create Heatmaps ====

print("Generating heatmaps...")

# RMSE Heatmap
plt.figure(figsize=(8, 5))
pivot_rmse = results_df.pivot("hist_months", "pred_months", "rmse")
sns.heatmap(pivot_rmse, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("RMSE Heatmap (History vs Prediction Window)")
plt.xlabel("Prediction Window (Months)")
plt.ylabel("History Window (Months)")
plt.savefig(RMSE_HEATMAP)
plt.close()

# R² Score Heatmap
plt.figure(figsize=(8, 5))
pivot_r2 = results_df.pivot("hist_months", "pred_months", "r2")
sns.heatmap(pivot_r2, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("R² Score Heatmap (History vs Prediction Window)")
plt.xlabel("Prediction Window (Months)")
plt.ylabel("History Window (Months)")
plt.savefig(R2_HEATMAP)
plt.close()

print(f"Heatmaps saved to:\n- {RMSE_HEATMAP}\n- {R2_HEATMAP}")

# Pick best config based on lowest composite score
best_config = results_df.sort_values("composite").iloc[0]
h_best = int(best_config['hist_months'])
p_best = int(best_config['pred_months'])

print("\nBest Config Found:")
print(f"History: {h_best} months | Prediction: {p_best} months")
print(f"RMSE: {best_config['rmse']:.2f} | R²: {best_config['r2']:.2f}")

# ==== Prepare Final Training Data Based on Best Config ====

cutoff = end_date - relativedelta(months=p_best)
hist_start = cutoff - relativedelta(months=h_best)

hist_df = df[(df['order_date'] >= hist_start) & (df['order_date'] <= cutoff)]
future_df = df[(df['order_date'] > cutoff) & (df['order_date'] <= cutoff + relativedelta(months=p_best))]

# ==== Final Feature Engineering (Same as earlier, Strategic Plan Features) ====

features = (
    hist_df.groupby("customer_id")
    .agg(
        revenue_sum=("revenue", "sum"),
        revenue_count=("revenue", "count"),
        avg_order_value=("revenue", "mean"),
        last_purchase=("order_date", "max"),
        first_purchase=("order_date", "min"),
        order_dates=("order_date", list)
    )
    .reset_index()
)

features["recency_days"] = (cutoff - features["last_purchase"]).dt.days
features["tenure_days"] = (cutoff - features["first_purchase"]).dt.days

features["log_revenue_sum"] = np.log1p(features["revenue_sum"])
features["is_repeat_buyer"] = features["revenue_count"] > 1
features["active_last_30d"] = features["recency_days"] <= 30

features[['mean_days_between_orders', 'std_days_between_orders', 'median_days_between_orders']] = features['order_dates'].apply(order_gap_stats)

features['customer_stage'] = pd.cut(
    features['tenure_days'],
    bins=[-np.inf, 90, 180, 365, np.inf],
    labels=["New", "Growing", "Established", "Loyal"]
)

features['status_segment'] = pd.cut(
    features['recency_days'],
    bins=[-np.inf, 30, 90, 180, np.inf],
    labels=["Active", "Lagging", "Dormant", "Churn-risk"]
)

features.drop(columns=["order_dates", "first_purchase", "last_purchase"], inplace=True)

# ==== Final Target Construction ====

targets = (
    future_df.groupby("customer_id")
    .agg(future_revenue=("revenue", "sum"))
    .reset_index()
)

data = pd.merge(features, targets, on="customer_id", how="left")
data["future_revenue"] = data["future_revenue"].fillna(0)

X_final = data.drop(columns=["customer_id", "future_revenue"])
y_final = data["future_revenue"]

# ==== Final Model Training ====

final_model = XGBRegressor(**config["modeling"])

final_model.fit(
    X_final, y_final,
    eval_metric=config["training"]["eval_metric"],
    verbose=False
)

# ==== Save Final Model ====

model_path = f"models/clv_model_hist{h_best}_pred{p_best}.joblib"
dump(final_model, model_path)

print(f"Final model saved to {model_path}")
