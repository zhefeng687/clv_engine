# Architecture Overview
- Folder Structure
- Module Breakdown
- Core Scripts Plan 


# Folder Structure

This document outlines the final approved folder structure for the Customer Lifetime Value (CLV) Engine MVP.  
Designed for local execution first, with future extension to production systems.

```bash
clv_engine/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── strategic_plan.md  
├── config/
│   └── model_config.yaml          # Hyperparameters, training configs, segment rules
├── data/
│   ├── raw/                       # Raw transactional data (input CSVs)
│   ├── processed/                 # Feature-engineered datasets
│   ├── outputs/                   # Prediction outputs (CLV predictions, business segments)
│   └── monitoring/                # Monitoring artifacts (drift metrics, alerts)
├── outputs/
│   ├── feature_importances.png    # SHAP global explanation plot
│   ├── segment_metrics.csv        # RMSE, MAE per customer_stage
│   ├── interactions.csv           # Top SHAP interactions
│   ├── cluster_summary.csv        # Clustering results summary
│   └── business_segments.csv      # Lifecycle segments for marketing and CX teams
├── models/
│   ├── clv_model_TIMESTAMP.joblib # Trained model artifacts
│   └── model_metadata.json        # Model training metadata (timestamp, config, metrics)
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Optional: EDA, SHAP plots, quick visualizations
├── experiments/                   # For research code, hyperparameter tuning
│   └── clv_grid_search_autotune.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Loading and saving data
│   ├── feature_engineering.py     # Creating tenure_days, recency_days, cadence stats
│   ├── modeling.py                # Training and saving the CLV model
│   ├── scoring.py                 # Batch scoring of customers
│   ├── clustering.py              # KMeans clustering and profile generation
│   ├── monitoring.py              # Drift detection, model health checks
│   └── utils.py                   # Helper functions (timestamping, saving, etc.)
├── scripts/
│   ├── train_model.py             # Orchestration script: Train model + save outputs
│   ├── score_customers.py         # Batch scoring script
│   ├── cluster_customers.py       # Clustering and segment generation
│   └── monitoring_check.py        # Monitoring and alert script
└── requirements.txt               # Python package dependencies
```


# Module Blueprint

This section details the functional design of each Python module inside the `src/` directory.  
Each module has a **single responsibility**, clean inputs/outputs, and is aligned with professional production practices.

---

## 1. src/data_loader.py
**Purpose:** Handle loading and saving of data (raw, processed, outputs).

| Function | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `load_raw_data(filepath)` | Load customer transactions from CSV | File path | Pandas DataFrame |
| `save_processed_data(df, filepath)` | Save feature-engineered data | DataFrame, path | CSV file |
| `load_processed_data(filepath)` | Load processed features for modeling or scoring | File path | Pandas DataFrame |

---

## 2. src/feature_engineering.py
**Purpose:** Build all lifecycle, revenue, and cadence features.

| Function | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `create_features(df_raw)` | Create tenure, recency, revenue_sum, cadence stats | Raw transactions | Feature DataFrame |
| `generate_lifecycle_segments(df_features)` | Assign customer_stage and status_segment | Feature DataFrame | Feature DataFrame with new segment columns |

---

## 3. src/modeling.py
**Purpose:** Train, save, and load the CLV prediction model.

| Function | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `train_clv_model(X_train, y_train, config)` | Train XGBoost regressor | Feature set, labels, config | Trained model object |
| `save_model(model, filepath)` | Save model artifact with timestamp | Model object | .joblib file |
| `save_model_metadata(metadata, filepath)` | Save training metrics and configs | Metadata dictionary | JSON file |
| `load_model(filepath)` | Load saved model for scoring | File path | Trained model object |

---

## 4. src/scoring.py
**Purpose:** Score customers in batch and rank by predicted CLV.

| Function | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `predict_clv(model, X_features)` | Generate CLV predictions | Model, features | Predicted CLV values |
| `rank_customers(df_predicted)` | Rank customers by CLV and assign percentiles | DataFrame with predictions | DataFrame with `clv_rank`, `clv_percentile`, `clv_segment` |

---

## 5. src/clustering.py
**Purpose:** Perform clustering to discover customer behavior groups.

| Function | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `perform_clustering(df_features, config)` | Apply KMeans clustering | Feature set, config parameters | DataFrame with `cluster_id` assigned |
| `generate_cluster_summary(df_clustered)` | Summarize clusters (size, average CLV) | Clustered DataFrame | Summary DataFrame |

---

## 6. src/monitoring.py
**Purpose:** Monitor feature drift, model performance, and alert triggers.

| Function | Purpose | Inputs | Outputs |
|----------|---------|--------|---------|
| `calculate_feature_drift(current_features, reference_features)` | Compare current features to historical baseline | Two DataFrames | Drift metrics (simple stats) |
| `track_model_performance(y_true, y_pred)` | Calculate RMSE, R2 metrics | True labels, predicted values | Metrics dictionary |
| `save_monitoring_report(metrics, filepath)` | Save monitoring outputs | Metrics dictionary | JSON or CSV file |

---

## 7. src/utils.py
**Purpose:** Common utilities for timestamping, saving, and folder management.

| Function | Purpose |
|----------|---------|
| `timestamp_now()` | Generate current timestamp for versioning |
| `create_folder_if_not_exists(path)` | Ensure target folders exist |
| `save_dataframe(df, filepath)` | Save a DataFrame to CSV cleanly |

---

## Module Responsibility Matrix

| Module | Main Responsibility |
|--------|----------------------|
| data_loader.py | Handle raw and processed data I/O |
| feature_engineering.py | Build lifecycle-aware features |
| modeling.py | Train, save, load CLV prediction models |
| scoring.py | Predict and rank customers |
| clustering.py | Behavioral segmentation via KMeans |
| monitoring.py | Feature/model drift, monitoring reports |
| utils.py | Helper functions for reusability |

---

# Core Scripts Plan

This section outlines the runnable scripts inside the `scripts/` directory,  
detailing their purposes, inputs, main steps, and outputs.  
All scripts are aligned with the Production-Ready CLV Engine Strategic Plan.

---

## Script: train_model.py

**Purpose:**  
Train the CLV prediction model from raw transactions.

**Inputs:**  
- `data/raw/transactions.csv`
- Configurations from `config/model_config.yaml`

**Steps:**
1. Load raw transactions.
2. Perform feature engineering (build lifecycle, revenue, cadence features).
3. Generate customer lifecycle segments (`customer_stage`, `status_segment`).
4. Prepare training dataset (features, labels).
5. Train XGBoost regressor model.
6. Save trained model artifact (`models/clv_model_TIMESTAMP.joblib`).
7. Save model training metadata (`models/model_metadata.json`).
8. Save processed features for scoring and clustering.

**Outputs:**  
- Trained model artifact (.joblib)
- Model metadata (.json)
- Feature-engineered dataset (.csv)

---

## Script: score_customers.py

**Purpose:**  
Score customers using trained model and generate CLV predictions.

**Inputs:**  
- Trained model from `models/`
- Processed feature dataset from `data/processed/`

**Steps:**
1. Load processed customer features.
2. Load trained CLV model.
3. Predict CLV scores for each customer.
4. Rank customers by predicted CLV.
5. Assign CLV percentile and segment labels (`Top 1%`, `Top 5%`, etc.).
6. Save ranked customer predictions.

**Outputs:**  
- `outputs/clv_predictions.csv` with CLV scores, ranks, percentiles, and segments.

---

## Script: cluster_customers.py

**Purpose:**  
Cluster customers by behavioral features for exploratory segmentation.

**Inputs:**  
- Processed feature dataset from `data/processed/`

**Steps:**
1. Load processed customer features.
2. Perform KMeans clustering based on selected features (e.g., tenure, recency, AOV).
3. Assign `cluster_id` to each customer.
4. Generate cluster summary (size, average CLV, behavioral profile).
5. Save clustered customer data and cluster summary.

**Outputs:**  
- Cluster assignments added to `outputs/clv_predictions.csv`
- Cluster summary saved as `outputs/cluster_summary.csv`

---

## Script: monitoring_check.py

**Purpose:**  
Perform basic monitoring checks on feature drift, prediction drift, and segment distribution.

**Inputs:**  
- Latest feature datasets
- Previous baseline feature datasets
- Latest model outputs

**Steps:**
1. Load current and reference feature datasets.
2. Compare feature distributions (e.g., recency_days, tenure_days).
3. Track model prediction statistics (e.g., mean CLV, variance).
4. Check for segment size drift (e.g., lifecycle stage balance).
5. Save monitoring report and flag major drifts.

**Outputs:**  
- Monitoring metrics saved in `data/monitoring/`
- Alert flags (if needed for retraining triggers)

---

## Summary of Scripts

| Script | Main Function |
|--------|----------------|
| train_model.py | Train CLV model, save model + metadata |
| score_customers.py | Predict and rank customers by CLV |
| cluster_customers.py | Behavioral clustering of customers |
| monitoring_check.py | Monitor drift, stability, model health |

---



# Scripts-to-Modules Execution Flow

This section describes how each orchestration script (`scripts/`) interacts with the reusable modules (`src/`).  
The flow follows the natural execution order: **train ➔ score ➔ cluster ➔ monitor**.

Each script calls specific modules and functions to complete its job, ensuring modularity, testability, and clarity.

---

## Script: train_model.py

**Purpose:** Train the CLV prediction model from raw transaction data.

**Execution Flow:**
- **data_loader.py**
  - `load_raw_data(filepath)` → Load raw transaction data into a DataFrame
- **feature_engineering.py**
  - `create_features(df_raw)` → Engineer lifecycle, revenue, and cadence features
  - `generate_lifecycle_segments(df_features)` → Assign customer lifecycle stages
- **data_loader.py**
  - `save_processed_data(df_features, filepath)` → Save processed features for scoring and clustering
- **modeling.py**
  - `train_clv_model(X_train, y_train, config)` → Train the XGBoost regression model
  - `save_model(model, filepath)` → Save trained model artifact (.joblib)
  - `track_model_performance(y_true, y_pred)` → Evaluate RMSE, R²
  - `save_model_metadata(metadata, filepath)` → Save training metrics and configs (.json)
- **utils.py** 
  - `timestamp_now()` → Create timestamp for model filenames

**Outputs Generated:**
- Trained model artifact (`models/clv_model_TIMESTAMP.joblib`)
- Model metadata (`models/model_metadata.json`)
- Feature-engineered dataset (`data/processed/processed_features.csv`)

---

## Script: score_customers.py

**Purpose:** Score customers using the trained model and generate ranked CLV predictions.

**Execution Flow:**
- **data_loader.py**
  - `load_processed_data(filepath)` → Load processed customer features
- **modeling.py**
  - `load_model(filepath)` → Load trained CLV model
- **scoring.py**
  - `predict_clv(model, X_features)` → Predict future CLV for each customer
  - `rank_customers(df_predicted)` → Assign CLV ranks and percentiles
- **utils.py**
  - `save_dataframe(df, filepath)` → Save CLV predictions
- **modeling.py** (optional, if explainability included)
  - `generate_shap_summary(model, X_features)` → Save SHAP feature importance plot (feature_importances.png)

**Outputs Generated:**
- CLV scores and rankings (`outputs/clv_predictions.csv`)
- SHAP feature importance summary (`outputs/feature_importances.png`)
- SHAP top interactions (`outputs/interactions.csv`)

---

## Script: cluster_customers.py

**Purpose:** Cluster customers by behavioral features for exploratory segmentation.

**Execution Flow:**
- **data_loader.py**
  - `load_processed_data(filepath)` → Load processed customer features
- **clustering.py**
  - `perform_clustering(df_features, config)` → Perform KMeans clustering and assign `cluster_id`
  - `generate_cluster_summary(df_clustered)` → Summarize clusters by size, CLV, behavior
- **utils.py**
  - `save_dataframe(df, filepath)` → Save clustered customers and cluster summary

**Outputs Generated:**
- Clustered customer file (`outputs/clv_predictions.csv` with `cluster_id`)
- Cluster summary statistics (`outputs/cluster_summary.csv`)

---

## Script: monitoring_check.py

**Purpose:** Monitor feature drift, prediction drift, and segment stability for retraining triggers.

**Execution Flow:**
- **data_loader.py**
  - `load_processed_data(filepath)` → Load current feature dataset
  - (Optional) Load reference dataset for baseline comparison
- **monitoring.py**
  - `calculate_feature_drift(current_features, reference_features)` → Detect feature distribution changes
  - `track_model_performance(y_true, y_pred)` → Check prediction performance stability
  - `save_monitoring_report(metrics, filepath)` → Save drift and health report

**Outputs Generated:**
- Monitoring report (`data/monitoring/monitoring_report.json` or `.csv`)
- Drift flags for retraining review

---



