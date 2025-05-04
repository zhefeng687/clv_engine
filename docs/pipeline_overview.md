# Purpose

This document walks through the complete CLV pipeline: who runs which script, in what order, and why. It reflects all code files, including the new auto-driver, model-card persistence, min-max grid-search scaling, and the human-in-the-loop governance model.

# 1 Primary flows

## 1-1 Model-build flow (quarterly or on sustained drift alert)

| Step | Script | Key outputs | Human gate |
|------|--------|-------------|------------|
| **Window tuning** | `experiments/clv_grid_search_autotune.py` | `outputs/grid_search/clv_grid_search_results_<TS>.csv`  +  2 heat-map PNGs | Data-science review of champion window |
| **Champion training** | `scripts/train_model.py` | `models/clv_model_<TS>.joblib`(large, auto-purged), `models/cards/clv_model_<TS>.json`(model card), `models/champion_manifest.csv`, `models/clv_model_latest.metrics.json` (baseline) | PR / ticket approval — deploy after sign-off |

### 1-2 Operational scoring flow (scheduled driver)

### 1·2 Operational scoring flow (scheduled)

| Step | Script(s) (inside `run_full_pipeline.py`) | Output CSV |
|------|-------------------------------------------|------------|
| **Score + value tiers** | `score_and_rank_customers.py` | `outputs/clv_predictions_<DATE>_predXm.csv` |
| **Behaviour clustering** | `cluster_customers.py` | `outputs/clv_clusters_<DATE>_predXm.csv` |
| **Cluster CLV-Index ranking** | `cluster_rank_customers.py` | `outputs/clv_cluster_ranking_<DATE>_predXm.csv` |
| **Cut-off bump** | driver logic | Updates `run.last_score_cutoff` in `config/model_config.yaml` (+14 days for first 12 weeks, then +30 days) |
| **Merge realised revenue + drift monitor** | `merge_actual_clv.py` → `monitor_drift_simple.py` | `outputs/clv_predictions_actual_<DATE>_predXm.csv` + Markdown drift alert |

The driver (run_full_pipeline.py) decides whether a given prediction file is ready for merge/monitor by checking:
today >= <prediction_cutoff> + relativedelta(months=pred_months).

# 2 Cadence

| Phase | Frequency | Driver / Script(s) | Rationale |
|-------|-----------|--------------------|-----------|
| **Model infancy** (first 12 weeks after launch) | **Every 14 days** – Monday 06:00 | `run_full_pipeline.py` | Recency & cadence can shift sharply after each promo wave; fortnightly refresh surfaces logic or data bugs quickly. |
| **Steady state** | **Monthly** – first Monday 06:00 | `run_full_pipeline.py` | Typical house-ware customers order a handful of times per year; monthly scoring keeps segments current without over-compute. |
| **Merge realised revenue + drift monitor** | *Implicit* – executed by `run_full_pipeline.py` **whenever** it finds a prediction file whose forward window has finished and has not yet been labelled | `merge_actual_clv.py` → `monitor_drift_simple.py` | Accuracy can only be calculated once `pred_months` of real behaviour has elapsed. |
| **Grid-search window tuning** | **Quarterly** or on two consecutive drift alerts | `experiments/clv_grid_search_autotune.py` (manual) | Re-evaluates whether a new history/pred window outperforms the incumbent. |
| **Champion training** | Immediately after grid-search is approved | `scripts/train_model.py` | Fits the selected window, writes baseline metrics JSON, model card JSON, and updates champion manifest. |

# 3 Artefact lifecycle

| Asset | Retention policy | Location | Notes |
|-------|-----------------|----------|-------|
| **Prediction CSVs** `clv_predictions_<DATE>_predXm.csv` and `_actual_` versions | Keep **≥ 1 year** (at least until their forward window matures and drift is evaluated). Older files may be archived to cold storage. | `outputs/` | Provide direct evidence of what marketing acted on and enable ROI analysis. |
| **Model card JSONs** `clv_model_<TS>.json` | Keep **7 years** (audit / finance). | `models/cards/` | Tiny (1–2 kB) files capturing windows, hyper-params, code commit, and baseline metrics. |
| **Champion manifest** `champion_manifest.csv` | Keep **7 years**; append-only. | `models/` | Ledger of every deployed champion; one row per model ID. |
| **Baseline metrics JSON** `clv_model_latest.metrics.json` | Keep until next champion training overwrites it. | `models/` | Read by `monitor_drift_simple.py` to detect accuracy drift. |
| **.joblib binaries** | Optional purge **after 9 months** via `cleanup_old_models.py` or S3 lifecycle rule. | `models/` | Heavy artefacts (5–20 MB). Can be recreated from a model card and code commit if ever needed. |
| **Grid-search result CSVs / PNG heat-maps** | Keep until the next grid-search supersedes them. | `outputs/grid_search/` | Evidence for champion window decision; lightweight (<200 kB). |
| **Logs & Markdown drift alerts** | Retain per org logging policy (e.g., 1 year in Splunk). | `outputs/` or centralized log store | Drift alerts are also logged at WARN level for SIEM capture. |

# 4 End-to-end diagram

```markdown
```text
                    ┌───────────────────────────────────────────┐
                    │  Grid-search (manual, quarterly)          │
                    │  experiments/clv_grid_search_autotune.py  │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │  Champion training (manual)               │
                    │  scripts/train_model.py                   │
                    │    → model card JSON + baseline metrics   │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════╧═════════════════════════════════════════╗
║              scheduled scripts/run_full_pipeline.py                         ║
║                                                                             ║
║ 1. score_and_rank_customers.py                                              ║
║    ├─ build features → predict CLV                                          ║
║    │    → outputs/clv_predictions_<DATE>_predXm.csv   (plain)               ║
║    └─ add absolute-rank tiers (Top-1 %, Top-5 %, …)                         ║
║         → outputs/clv_ranked_predictions_<DATE>_predXm.csv                  ║
║                                                                             ║
║ 2a. (value lens) — abs-rank columns already present                         ║
║                                                                             ║
║ 2b. (behaviour lens)                                                        ║
║     cluster_customers.py            → outputs/clv_clusters_<DATE>_predXm.csv║
║     cluster_rank_customers.py →outputs/clv_cluster_ranking_<DATE>_predXm.csv║
║                                                                             ║
║ 3. bump YAML run.last_score_cutoff  (+14 d first 12 wks, else +30 d)        ║
║                                                                             ║
║ 4. for each older prediction file whose window is finished & unlabeled:     ║
║        merge_actual_clv.py      → adds actual_clv                           ║
║        monitor_drift_simple.py  → alert if RMSE / R² drift                  ║
╚═══════════════════════════════════╤═════════════════════════════════════════╝
                                    │
                                    ▼
                   🚨  Alert triggers human RCA → may rerun grid-search
```
