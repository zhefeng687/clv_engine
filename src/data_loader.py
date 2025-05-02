"""
src/data_loader.py
--------------------------------------------------------------------
Robust helpers for reading and writing the project’s datasets.
• Paths are resolved relative to project root, so “cd” mishaps don’t break runs.
• CSV loader parses order_date once; no duplicate conversions downstream.
• All I/O functions accept **kwargs and pass them to pandas for full flexibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# ------------------------------------------------------------------#
# Logging setup
# ------------------------------------------------------------------#
logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # keep global config if caller already set it
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    )

# ------------------------------------------------------------------#
# Project-root resolution
# ------------------------------------------------------------------#
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _resolve(path_like: str | Path) -> Path:
    """Resolve a relative path against project root; leave absolute paths as-is."""
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ------------------------------------------------------------------#
# Public API
# ------------------------------------------------------------------#
def load_raw_data(
    path: str | Path = "data/raw/transactions.csv",
    *,
    parse_dates: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load raw transactional data.

    Parameters
    ----------
    path:
        Location of the CSV file, relative to project root by default.
    parse_dates:
        Columns to parse as dates (defaults to ["order_date"]).
    **kwargs:
        Extra keyword args forwarded to pandas.read_csv (e.g. sep=";", encoding="utf-8").

    Returns
    -------
    DataFrame (copy) so callers can mutate safely.
    """
    parse_dates = parse_dates or ["order_date"]
    full_path = _resolve(path)
    df = pd.read_csv(full_path, parse_dates=parse_dates, **kwargs)
    logger.info("Loaded raw data %s — shape=%s", full_path, df.shape)
    return df.copy()


def load_processed_data(
    path: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load any previously processed CSV or Parquet file."""
    full_path = _resolve(path)
    if full_path.suffix == ".parquet":
        df = pd.read_parquet(full_path, **kwargs)
    else:
        df = pd.read_csv(full_path, **kwargs)
    logger.info("Loaded processed data %s — shape=%s", full_path, df.shape)
    return df.copy()


def save_processed_data(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Save a DataFrame to disk, creating parent folders if needed.

    The format is inferred from the file extension (.csv or .parquet).

    Returns
    -------
    Path to the file written.
    """
    full_path = _resolve(path)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if full_path.suffix == ".parquet":
        df.to_parquet(full_path, index=index, **kwargs)
    else:
        df.to_csv(full_path, index=index, **kwargs)

    logger.info("Saved processed data %s — shape=%s", full_path, df.shape)
    return full_path
