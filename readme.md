# House-Ware CLV Engine

Predict customer lifetime value, produce two independent segmentation lenses (value tiers & behavioural clusters), and monitor accuracy drift—without automated retraining.

---

## 1. Repo structure

## 2. Quick-start commands

- **One-time model build (run manually)**
python scripts/clv_grid_search_autotune.py   # pick best history / pred windows
python scripts/train_model.py                # fit champion + save baseline metrics

- **Recurring scoring job (cron / Airflow)**
python scripts/run_full_pipeline.py          # score → rank → cluster
                                             # driver updates YAML cutoff

- **When a prediction window matures (handled automatically):**
- the same driver detects it → merges actuals → runs drift monitor
- and drops a Markdown alert if RMSE or R² drift beyond thresholds

- **Configuration lives in config/model_config.yaml:**
training:
  history_months: 9          # auto-updated by grid search
  pred_months:    6
run:
  last_score_cutoff: "2021-12-31"   # driver bumps this after each scoring batch
monitoring:
  drift_threshold_rmse: 0.20
  drift_threshold_r2:   0.15

Override any run with --cutoff YYYY-MM-DD.

## 3. Operational cadence

| Phase | Frequency | Script(s) | Key artefact |
|-------|-----------|-----------|--------------|
| **Scoring + segmentation** | Every 2 weeks (first 3 mo) → monthly | `run_full_pipeline.py` | `outputs/clv_predictions_<DATE>_predXm.csv` + cluster files |
| **Label merge + drift monitor** | Automatically (driver detects matured window) | `merge_actual_clv.py` → `monitor_drift_simple.py` | Markdown alert saved to `outputs/` |
| **Grid-search window tuning** | Quarterly **or** on sustained drift alert | `experiments/clv_grid_search_autotune.py` | Results CSV + heat-maps |
| **Champion training** | After grid-search approval | `scripts/train_model.py` | `.joblib` model, baseline metrics JSON, model-card JSON, updated champion manifest |

## 4. Model governance essentials

- **Model card JSONs**  
  Stored in `models/cards/`.  
  They capture:
  - model ID and timestamp
  - Git commit hash of the code
  - training cutoff, history_months, pred_months
  - full hyper-parameter set
  - baseline RMSE and R²

- **Champion manifest**  
  Append-only CSV at `models/champion_manifest.csv`.  
  One row per deployed champion with the same fields as the model card; acts as a ledger of model history.

- **Baseline metrics JSON**  
  `models/clv_model_latest.metrics.json`  
  Read by `monitor_drift_simple.py` to detect accuracy drift.

- **Prediction CSVs**  
  `outputs/clv_predictions_<DATE>_predXm.csv` (and `_actual_` once revenue attaches).  
  Retained at least until the forward window matures and drift has been evaluated.

- **Reproducibility procedure**

1. Locate the desired row in `champion_manifest.csv`; note the `code_commit` and `training_cutoff`.
2. Check out that code version:

    ```bash
    git checkout <code_commit>
    ```

3. Re-train with the same cutoff (windows are in the manifest row or model card):

    ```bash
    python scripts/train_model.py --cutoff <training_cutoff>
    ```

4. Confirm that the new RMSE and R² match the values in the manifest.

## 5. Lifecycle in one diagram

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



* **Score + value tiers** (`score_and_rank_customers.py`)
  – writes plain predictions **and** absolute-rank segments
* **Behaviour clustering** (`cluster_customers.py`)
  – consumes the plain prediction file, writes `cluster_id`
* **Cluster CLV ranking** (`cluster_rank_customers.py`)
  – computes median CLV, CLV-Index, cluster_rank
```

## 6. Governance quick-ref

- **Drift alert thresholds**  
  RMSE +20 %  •  R² −15 %  (see config/model_config.yaml)
- **No automatic retrain.**  
  Alerts open a JIRA ticket; DS runs grid-search + training only after review.
- **Artefact retention**  
  Model card JSONs & champion manifest kept 7 yrs, .joblib purged ≥ 9 mo.
- **Reproduce a past model**  
  1. Find model_id in champion_manifest.csv  
  2. `git checkout <code_commit>` • `python scripts/train_model.py --cutoff <date>`  
  3. Verify RMSE/R² match manifest row.
