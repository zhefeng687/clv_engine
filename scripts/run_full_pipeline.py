#!/usr/bin/env python
"""
End-to-end driver  —  runs every 2-weeks (early) or monthly
───────────────────────────────────────────────────────────
1. score + rank + cluster for the current cutoff
2. advance YAML run.last_score_cutoff
3. auto-detect any predictions whose forward window is now closed
   but not yet labelled → merges actuals and runs drift monitor
No retraining; grid-search & training stay manual.
"""

from __future__ import annotations
import argparse, logging, subprocess, sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[1]
CFG_YML = ROOT / "config" / "model_config.yaml"
OUT    = ROOT / "outputs"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ── helper to run subprocess and show live output ─────────
def run(cmd: list[str]) -> None:
    log.info("▶ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ── CLI ---------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--cutoff", help="Override cutoff YYYY-MM-DD")
args = p.parse_args()

# ── 1. Resolve cutoff date --------------------------------
cfg = yaml.safe_load(open(CFG_YML))
cutoff_str = args.cutoff or cfg.get("run", {}).get("last_score_cutoff")
if not cutoff_str:
    log.error("No cutoff set; first run must pass --cutoff")
    sys.exit(1)

cutoff = pd.Timestamp(cutoff_str).normalize()
pred_m = cfg["training"]["pred_months"]
log.info("Current scoring cutoff → %s (pred %sm)", cutoff.date(), pred_m)

# ── 2. Score + rank + cluster -----------------------------
run(["python", "scripts/score_and_rank_customers.py", "--cutoff", str(cutoff.date())])
run(["python", "scripts/cluster_customers.py",        "--cutoff", str(cutoff.date())])
run(["python", "scripts/cluster_rank_customers.py",   "--cutoff", str(cutoff.date())])

# ── 3. Advance cutoff (14 d first 12 weeks, else 30 d) ----
history_start = pd.Timestamp(cfg["training"]["cutoff_date"]).normalize()
weeks_live    = (cutoff - history_start).days // 7
step_days     = 14 if weeks_live < 12 else 30
next_cutoff   = cutoff + timedelta(days=step_days)
cfg.setdefault("run", {})["last_score_cutoff"] = str(next_cutoff.date())
with open(CFG_YML, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
log.info("Next scheduled cutoff written → %s (step %s days)",
         next_cutoff.date(), step_days)

# ── 4. Auto-merge & monitor any matured windows -----------
def window_finished(pred_date: pd.Timestamp) -> bool:
    return pd.Timestamp.today() >= pred_date + relativedelta(months=pred_m)

pred_pattern = f"clv_predictions_*_pred{pred_m}m.csv"
for pred_file in OUT.glob(pred_pattern):
    date_str = pred_file.stem.split("_")[2]  # clv_predictions_YYYYMMDD_pred…
    pred_date = pd.Timestamp(date_str)
    actual_file = pred_file.with_name(
        pred_file.name.replace("clv_predictions_", "clv_predictions_actual_")
    )
    if actual_file.exists():
        continue  # already merged
    if not window_finished(pred_date):
        continue  # labels not ready yet

    # Run merge + monitor for this matured window
    run(["python", "scripts/merge_actual_clv.py", "--cutoff", str(pred_date.date())])
    run(["python", "scripts/monitor_drift_simple.py", "--cutoff", str(pred_date.date())])

log.info("Pipeline run complete.")
