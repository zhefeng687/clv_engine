# Customer Lifetime Value (CLV) Engine

Predict, Rank, Cluster, Monitor, and Govern Customer Lifetime Value using production-grade ML pipelines.

---

## Overview

This CLV Engine system:

- Predicts Customer Lifetime Value (CLV) using historical customer behaviors.  
- Ranks customers individually by predicted CLV.  
- Clusters customers based only on predicted CLV.  
- Ranks clusters based on median predicted CLV.  
- Monitors feature drift and prediction drift automatically.  
- Follows governance principles for retraining decisions.  
- Fully orchestrated with one-click `run_full_pipeline.py`.

---

## System Overview Diagram
data/raw/transactions.csv
    ↓
(src/feature_engineering.py)
Feature Engineering (tenure, recency, revenue_sum, cadence stats)
    ↓
(scripts/train_model.py)
Train CLV Model (XGBoost)
    ↓
(scripts/score_and_rank_customers.py)
Predict CLV → Rank Customers Individually
    ↓
(scripts/cluster_customers.py)
Cluster Customers by Predicted CLV
    ↓
(scripts/cluster_rank_customers.py)
Rank Clusters by Median CLV
    ↓
(scripts/monitoring_check.py)
Monitor Feature Drift and Prediction Drift
    ↓
(outputs/)
    - models/
    - outputs/clv_ranked_predictions.csv
    - outputs/clv_clustered_customers.csv
    - outputs/clv_cluster_rankings.csv
    - outputs/monitoring/ (drift reports)

---

## Project Structure

```
clv_engine/
├── config/
│   └── model_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── (trained model artifacts)
├── outputs/
│   ├── clv_ranked_predictions.csv
│   ├── clv_clustered_customers.csv
│   ├── clv_cluster_rankings.csv
│   └── monitoring/
│       ├── feature_drift_report_TIMESTAMP.csv
│       └── prediction_drift_report_TIMESTAMP.json
├── scripts/
│   ├── train_model.py
│   ├── score_and_rank_customers.py
│   ├── cluster_customers.py
│   ├── cluster_rank_customers.py
│   └── monitoring_check.py
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── scoring.py
│   ├── abs_rank.py
│   ├── clustering.py
│   ├── cluster_rank.py
│   ├── monitoring.py
│   └── utils.py
├── run_full_pipeline.py
├── launch_readiness_checklist.md
├── readme.md
└── requirements.txt
```


## How to Run (One-Click Full Pipeline)

To train, predict, cluster, rank, and monitor all in one shot:

```bash
python run_full_pipeline.py
```

run_full_pipeline.py orchestrates the entire system cleanly:
- Train model
- Predict and rank customers
- Cluster customers
- Rank clusters
- Monitor feature drift and prediction drift
- Save all reports to outputs/

## Inputs and Outputs

### Input

| Input | Description |
|-------|-------------|
| `data/raw/transactions.csv` | Raw transaction history (customer_id, order_date, revenue) |

### Output

| Output | Description |
|--------|-------------|
| `outputs/clv_ranked_predictions.csv` | Individual customer CLV scores and rankings |
| `outputs/clv_clustered_customers.csv` | Clustered customers with cluster_id assigned |
| `outputs/clv_cluster_rankings.csv` | Ranked clusters based on median predicted CLV |
| `outputs/monitoring/feature_drift_report_TIMESTAMP.csv` | Feature drift report |
|` outputs/monitoring/prediction_drift_report_TIMESTAMP.json` | Prediction drift report
---

## Key Modules (src/)
| Module                 | Purpose                                                                         |
|------------------------|---------------------------------------------------------------------------------|
| `data_loader.py`       | Load/save data                                                                  |
| `feature_engineering.py`| Create features, lifecycle segmentation, select features for training           |
| `modeling.py`          | Train/save/load models                                                          |
| `scoring.py`           | Predict CLV                                                                     |
| `abs_rank.py`          | Rank customers individually by predicted CLV                                    |
| `clustering.py`        | Cluster customers based only on predicted CLV                                   |
| `cluster_rank.py`      | Rank clusters based on median CLV                                               |
| `monitoring.py`        | Detect feature drift and prediction drift                                       |
| `utils.py`             | Common helpers (folder creation, timestamps)                                    |

---

## Configuration

Edit model, clustering, and monitoring settings inside:

```bash
config/model_config.yaml
```

## Monitoring and Governance

### Monitoring:

- Feature Drift Monitoring:
  - Compares current features vs baseline using Kolmogorov-Smirnov tests.
- Prediction Drift Monitoring:
  - Compares current RMSE and R² against training baselines.

### Governance:

- Monitoring reports are saved into outputs/monitoring/.
- **Retraining is NOT automatic after drift detection.**
- Human review is required after drift detection.
- If drift is serious, Strategic Plan must be reviewed before deciding to retrain or redesign.
- Monitoring thresholds configured in config/model_config.yaml.

---

## System Principles

- Predict → Rank → Cluster → Rank Clusters → Monitor — clean modular pipeline.
- Median CLV used for cluster ranking (robust to outliers).
- Config-driven modeling, clustering, monitoring settings.
- Modular `src/` modules and clean `scripts/` orchestrators.
- Monitoring outputs (feature drift, prediction drift) saved automatically for auditability.
- Manual retrain decisions after monitoring, not automatic.

---

## Future Enhancements

- Add Segment Drift Monitoring (lifecycle stage distribution shifts).  
- Add SHAP Drift Monitoring (feature importance shifts).  
- Migrate pipelines to Airflow DAG for scheduled retraining.  
- Add lightweight email/Slack alerting based on monitoring reports.


