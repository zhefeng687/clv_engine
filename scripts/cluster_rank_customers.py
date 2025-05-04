#!/usr/bin/env python
"""
Rank behavioural clusters by economic value
────────────────────────────────────────────
1. Scores (already clustered) customers' median CLV
2. Builds CLV Index  = cluster median / overall median
3. Ranks clusters (1 = highest median)
4. Writes outputs/clv_cluster_rankings_<YYYYMMDD>.csv
"""

from __future__ import annotations

import logging
import sys
import yaml
from pathlib import Path

import pandas as pd

# project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.cluster_rank import rank_clusters_by_median

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# ── load config (timestamp format only) ───────────────────
with open(PROJECT_ROOT / "config" / "model_config.yaml") as f:
    cfg = yaml.safe_load(f)
ts_fmt = cfg["general"]["timestamp_format"]

# ── locate latest clustered file ──────────────────────────
out_dir = PROJECT_ROOT / "outputs"
cluster_files = sorted(out_dir.glob("clv_clusters_*_pred*m.csv"))
if not cluster_files:
    raise FileNotFoundError(
        "No clustered-customer CSVs found in outputs/. "
        "Run cluster_customers.py first."
    )

cluster_path = cluster_files[-1]
log.info("Loading clustered customers from %s", cluster_path.name)
df_clustered = pd.read_csv(cluster_path)

# ── build cluster ranking (median CLV) ─────────────────────
rank_table = rank_clusters_by_median(df_clustered)
overall_median = df_clustered["predicted_clv"].median()
rank_table["clv_index"] = rank_table["median_clv"] / overall_median

# reorder for readability
rank_table = rank_table[["cluster_id", "median_clv", "clv_index", "cluster_rank"]]

log.info("Ranked %s clusters; overall customer median = %.2f",
         len(rank_table), overall_median)

# ──  Writes outputs/clv_cluster_ranking_<YYYYMMDD>_pred<X>m.csv ───
parts = cluster_path.stem.split("_")   # ['clv','clusters','YYYYMMDD','pred6m']
snapshot_str = parts[2]     # YYYYMMDD
pred_tag      = parts[3] 
rank_path = out_dir / f"clv_cluster_ranking_{snapshot_str}_{pred_tag}.csv"
log.info("Cluster ranking + CLV index saved → %s", rank_path)
