"""
src/modeling.py
────────────────────────────────────────────────────────────
Utility functions for persisting and retrieving CLV models.

Key entry-points
----------------
save_model(model, model_dir=Path("models"), metrics=None)
    • Saves timestamped artefact   clv_model_<TS>.joblib
    • Updates   clv_model_latest.joblib      → symlink (copy on Windows)
    • If *metrics* dict provided, writes a JSON side-car and updates
      clv_model_latest.metrics.json
    • Returns the Path to the timestamped model file.

load_latest_model(model_dir=Path("models"))
    • Loads clv_model_latest.joblib and returns the fitted estimator.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from git import Repo
import yaml

import joblib

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)


def _safe_symlink(src: Path, dst: Path) -> None:
    """Create or replace a symlink; fall back to full copy on Windows."""
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.name)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        logger.warning("Symlink failed on this OS; full copy made for %s", dst.name)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def save_model(
    model: Any,
    *,
    model_dir: str | Path = "models",
    metrics: Dict[str, float] | None = None,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> Path:
    """
    Persist a fitted estimator and (optionally) its eval metrics.

    Returns
    -------
    Path to the timestamped .joblib artefact.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    ts   = _timestamp(timestamp_fmt)
    ts_path   = model_dir / f"clv_model_{ts}.joblib"
    latest_sy = model_dir / "clv_model_latest.joblib"

    joblib.dump(model, ts_path, compress=3)
    _safe_symlink(ts_path, latest_sy)

    if metrics is not None:
        metrics_path = ts_path.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        latest_metrics_sy = model_dir / "clv_model_latest.metrics.json"
        _safe_symlink(metrics_path, latest_metrics_sy)

    # ----- logging summary (always execute, even if metrics=None) -----
    logger.info(
        "Model saved: %s  |  Latest pointer updated → %s",
        ts_path.name,
        latest_sy.name,
    )
    if metrics:
        logger.info("Baseline metrics written: %s (RMSE %.4f, R² %.4f)",
                    metrics_path.name, metrics.get("rmse"), metrics.get("r2"))

    # -- persist lightweight evidence ------------------------------------
    cfg = yaml.safe_load(
        open(Path(__file__).resolve().parents[2] / "config" / "model_config.yaml")
        )
    repo_hash = Repo(Path(__file__).resolve().parents[1]).head.commit.hexsha[:7]
    model_card = {
        "model_id": ts_path.stem,                # clv_model_YYYYMMDD_HHMMSS
        "code_commit": repo_hash,
        "training_cutoff": cfg["training"]["cutoff_date"],
        "history_months": cfg["training"]["history_months"],
        "pred_months":    cfg["training"]["pred_months"],
        "hyper_params":   cfg["modeling"],       # learning_rate, depth, etc.
        "metrics": metrics or {},
    
    }
    card_path = model_dir / "cards" / f"{ts_path.stem}.json"
    card_path.parent.mkdir(exist_ok=True)
    json.dump(model_card, open(card_path, "w"), indent=2)

    manifest = model_dir / "champion_manifest.csv"
    pd.DataFrame([model_card]).to_csv(
        manifest, mode="a", index=False, header=not manifest.exists()
    )

    return ts_path

    
def load_latest_model(model_dir: str | Path = "models"):
    """
    Load the model pointed to by clv_model_latest.joblib.
    """
    path = Path(model_dir) / "clv_model_latest.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Latest model not found at {path}")
    logger.info("Loaded model %s", path)
    return joblib.load(path)
