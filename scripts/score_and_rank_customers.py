"""
Score + rank customers by predicted CLV
────────────────────────────────────────
1. Loads model_config.yaml for champion windows + segment bins
2. Builds features for selected snapshot date
3. Predicts CLV using clv_model_latest.joblib
4. Adds rank + percentile + segment labels
5. Writes outputs/clv_ranked_predictions_<YYYYMMDD>_pred<X>m.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path

import pandas as pd

# ───────────────────────── setup ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.scoring import score_customers
from src.abs_rank import add_absolute_rank

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

snapshot = pd.Timestamp(args.cutoff) if args.cutoff else pd.Timestamp.today()
log.info("Scoring + ranking snapshot → %s", snapshot.date())


# ───────────────────────── Config + paths ─────────────────────────
cfg_path = PROJECT_ROOT / "config" / "model_config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

raw_csv = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
model_dir = PROJECT_ROOT / "models"
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

# ───────────────────────── Score customers ─────────────────────────
df_scores = score_customers(
    raw_csv_path=raw_csv,
    cutoff=snapshot,
    cfg=cfg,
    model_dir=model_dir,
)

# --- save the raw predictions so every other script shares the same filename
pred_m  = cfg["training"]["pred_months"]
base_csv = output_dir / f"clv_predictions_{snapshot:%Y%m%d}_pred{pred_m}m.csv"
df_scores.to_csv(base_csv, index=False)
log.info("Base predictions saved → %s", base_csv)

if df_scores.empty:
    raise ValueError("Scoring returned an empty DataFrame — check cutoff window or data input.")

# ───────────────────────── Rank & segment ─────────────────────────
df_ranked = add_absolute_rank(df_scores, cfg)

pred_m = cfg["training"]["pred_months"]
out_path = output_dir / f"clv_ranked_predictions_{snapshot:%Y%m%d}_pred{pred_m}m.csv"

df_ranked.to_csv(out_path, index=False)
log.info("Ranked predictions saved → %s", out_path)

# ─── persist cutoff for downstream jobs ───
cfg.setdefault("run", {})["last_score_cutoff"] = snapshot.strftime("%Y-%m-%d")

with open(cfg_path, "w") as f:
     yaml.safe_dump(cfg, f, sort_keys=False)
log.info("Config updated with last_score_cutoff → %s", snapshot.date())