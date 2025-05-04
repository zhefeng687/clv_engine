#!/usr/bin/env python
"""
scripts/monitor_drift_simple.py
────────────────────────────────
Smoke-alarm monitor for the CLV model.

• Reads baseline RMSE & R² from models/clv_model_latest.metrics.json
• Loads the latest prediction-vs-actual CSV
  └ default path is driven by   run.last_score_cutoff   in model_config.yaml
• Compares deltas against YAML thresholds
• Logs a 🚨  line and writes a short Markdown report if drift exceeds thresholds
• Makes **no** attempt to retrain — human review only
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import root_mean_squared_error, r2_score

# ── repo paths ───────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
CONFIG_YML  = ROOT / "config" / "model_config.yaml"
MODEL_DIR   = ROOT / "models"
OUTPUT_DIR  = ROOT / "outputs"

PRED_COL   = "predicted_clv"
ACTUAL_COL = "actual_clv"          # change if your merge script uses another name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────
def load_baseline(path: Path) -> dict[str, float]:
    if not path.exists():
        log.error("Baseline metrics not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

def pick_prediction_file(cfg: dict, cutoff: str | None) -> Path:
    if cutoff:
        pm   = cfg["training"]["pred_months"]
        name = f"clv_predictions_actual_{pd.Timestamp(cutoff):%Y%m%d}_pred{pm}m.csv"
        path = OUTPUT_DIR / name
        if not path.exists():
            log.error("Expected file not found: %s", path)
            sys.exit(1)
        return path

    # fallback: YAML hand-off
    raw = cfg.get("run", {}).get("last_score_cutoff")
    if not raw:
        log.error("No cutoff specified and YAML run.last_score_cutoff is empty.")
        sys.exit(1)
    pm = cfg["training"]["pred_months"]
    return OUTPUT_DIR / f"clv_predictions_actual_{pd.Timestamp(raw):%Y%m%d}_pred{pm}m.csv"

# ── main ─────────────────────────────────────────────────────────────
def main():
    # ─ CLI: 0-or-1 arg (CSV path **or** --cutoff date) ───────────────
    csv_or_date = sys.argv[1] if len(sys.argv) == 2 else None
    csv_path    = Path(csv_or_date) if csv_or_date and csv_or_date.endswith(".csv") else None
    cutoff_arg  = csv_or_date if csv_or_date and csv_or_date.endswith(".csv") is False else None

    cfg        = yaml.safe_load(open(CONFIG_YML))
    baseline   = load_baseline(MODEL_DIR / "clv_model_latest.metrics.json")
    pred_file  = csv_path or pick_prediction_file(cfg, cutoff_arg)

    log.info("Monitoring file: %s", pred_file.name)
    df = pd.read_csv(pred_file)

    # ─ sanity columns ────────────────────────────────────────────────
    missing = {PRED_COL, ACTUAL_COL} - set(df.columns)
    if missing:
        log.error("Missing columns in %s : %s", pred_file.name, missing)
        sys.exit(1)

    # ─ compute current metrics ───────────────────────────────────────
    rmse_now = root_mean_squared_error(df[ACTUAL_COL], df[PRED_COL])
    r2_now   = r2_score(df[ACTUAL_COL], df[PRED_COL])

    rmse_delta = (rmse_now - baseline["rmse"]) / baseline["rmse"]
    r2_delta   = (baseline["r2"] - r2_now) / baseline["r2"]

    log.info("Baseline RMSE %.4f → %.4f  (Δ %.1f%%)",
             baseline["rmse"], rmse_now, 100 * rmse_delta)
    log.info("Baseline R²   %.4f → %.4f  (Δ %.1f%%)",
             baseline["r2"],  r2_now,   100 * r2_delta)

    # ─ thresholds from YAML ──────────────────────────────────────────
    rmse_thr = cfg["monitoring"]["drift_threshold_rmse"]   # e.g. 0.20
    r2_thr   = cfg["monitoring"]["drift_threshold_r2"]     # e.g. 0.15

    alerts = []
    if rmse_delta > rmse_thr:
        alerts.append(f"RMSE drift {rmse_delta:.1%} > {rmse_thr:.0%}")
    if r2_delta > r2_thr:
        alerts.append(f"R² drift {r2_delta:.1%} > {r2_thr:.0%}")

    # ─ plain-language report if needed ───────────────────────────────
    if alerts:
        md_path = OUTPUT_DIR / f"drift_alert_{pred_file.stem}.md"
        with open(md_path, "w") as md:
            md.write(f"### 📣 CLV Model Health Alert — {pred_file.stem[-8:]}\n\n")
            md.write("**Status:** ⚠ **Attention needed**\n\n")
            md.write("| Metric | Baseline | Latest | Change | Threshold |\n")
            md.write("|--------|----------|--------|--------|-----------|\n")
            md.write(f"| RMSE | {baseline['rmse']:.2f} | {rmse_now:.2f} | "
                     f"**{rmse_delta:+.0%}** | {rmse_thr:.0%} |\n")
            md.write(f"| R²   | {baseline['r2']:.3f} | {r2_now:.3f} | "
                     f"**{r2_delta:+.0%}** | {r2_thr:.0%} |\n\n")
            md.write("#### What this means\n")
            md.write("* Predictions are **less precise** than when the model was trained.\n")
            md.write("* Marketing actions based on CLV scores may target customers less effectively.\n\n")
            md.write("#### Suggested next steps\n")
            md.write("1. **Data Check** – look for one-off promos or data glitches.\n")
            md.write("2. **Model Review** – if drift persists, schedule retraining.\n\n")
            md.write("_(No automatic retrain has been triggered — human review required.)_\n")
        log.warning("🚨 DRIFT ALERT — human review required: " + " | ".join(alerts))
        log.info("Markdown report saved → %s", md_path)
    else:
        log.info("✅ Drift within thresholds — no action needed.")

if __name__ == "__main__":
    main()
