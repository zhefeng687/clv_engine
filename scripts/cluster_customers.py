#!/usr/bin/env python
"""
1. Scores customers for a snapshot date (predicted CLV + behaviour cols)
2. Runs K-means clustering on the hybrid feature set
3. Saves outputs/clv_clustered_customers_<YYYYMMDD>_pred<X>m.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path

import pandas as pd

# project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.scoring    import score_customers
from src.clustering import cluster_customers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# ───────────────────────── CLI args ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--cutoff", type=str,
    help="Snapshot date YYYY-MM-DD (default = today)"
)
args = parser.parse_args()

# load config early because it might contain the cutoff
with open(PROJECT_ROOT / "config" / "model_config.yaml") as f:
     cfg = yaml.safe_load(f)

if args.cutoff:
    snapshot = pd.Timestamp(args.cutoff)
else:
    snapshot = pd.Timestamp(
        cfg.get("run", {}).get("last_score_cutoff", pd.Timestamp.today())
    )

log.info("Clustering snapshot → %s", snapshot.date())

# ───────────────────────── Config ───────────────────────────
raw_csv   = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
model_dir = PROJECT_ROOT / "models"
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

# ───────────────────────── 1 ▸ Score  ───────────────────────
df_scores = score_customers(
    raw_csv_path = raw_csv,
    cutoff       = snapshot,
    cfg          = cfg,
    model_dir    = model_dir,
)

# FIRST-RUN GUARD
if df_scores.empty:
    log.warning(
        "No customers scored for cutoff %s — check data availability or cutoff date.",
        snapshot.date()
    )
    sys.exit(0)

# ──── 2 ▸ Cluster ───────
df_clustered = cluster_customers(
    df_scores,
    cfg=cfg,                # supplies default feature list & scaling
)

# ───── 3 ▸ Saves outputs/clv_clusters_<YYYYMMDD>_pred<X>m.csv ─────
pred_m  = cfg["training"]["pred_months"]
out_csv = output_dir / f"clv_clusters_{snapshot:%Y%m%d}_pred{pred_m}m.csv"
df_clustered.to_csv(out_csv, index=False)
log.info("Clustered customers saved → %s", out_csv)

