"""
Data Cleaner: normalizes raw OHLCV data from IB or yfinance.
Handles gaps, bad ticks, zero volumes, and OHLC integrity.

Entry points:
  clean(df, timeframe)   — full pipeline, returns cleaned DataFrame
  validate(df)           — raises ValueError on any remaining data quality issue
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# Max allowed price spike as a multiplier of recent median price
SPIKE_THRESHOLD = 0.10  # 10% move in a single bar flagged as suspicious

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def clean(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a cleaned DataFrame.

    Steps:
      1. Enforce required columns and float dtype
      2. Drop duplicate timestamps
      3. Drop bars with zero or missing volume
      4. Fix OHLC integrity (high >= all, low <= all)
      5. Remove price spikes
      6. Forward-fill short gaps (max 3 bars)
      7. Sort by datetime ascending
      8. Final validate() — raises if anything slipped through
    """
    df = df.copy()
    initial_len = len(df)

    df = _enforce_schema(df)
    df = _drop_duplicates(df)
    df = _drop_zero_volume(df)
    df = _fix_ohlc_integrity(df)
    df = _remove_spikes(df)
    df = _fill_gaps(df, timeframe)
    df = df.sort_index()

    removed = initial_len - len(df)
    logger.info(f"[cleaner] {initial_len} bars in → {len(df)} bars out ({removed} removed/fixed)")

    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """
    Strict validation pass. Raises ValueError if the DataFrame has:
      - Missing required columns
      - Any NaN values
      - Non-float dtypes
      - Unsorted index
      - OHLC integrity violations
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[cleaner] Missing columns: {missing}")

    nan_counts = df[REQUIRED_COLS].isna().sum()
    if nan_counts.any():
        raise ValueError(f"[cleaner] NaN values remain after cleaning:\n{nan_counts[nan_counts > 0]}")

    for col in REQUIRED_COLS:
        if not pd.api.types.is_float_dtype(df[col]):
            raise ValueError(f"[cleaner] Column '{col}' is not float64 (got {df[col].dtype})")

    if not df.index.is_monotonic_increasing:
        raise ValueError("[cleaner] Index is not sorted in ascending order.")

    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    bad_low  = (df["low"]  > df[["open", "close"]].min(axis=1)).sum()
    if bad_high or bad_low:
        raise ValueError(
            f"[cleaner] OHLC integrity violations: {bad_high} high errors, {bad_low} low errors."
        )


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns are present and cast to float64."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[cleaner] Raw data missing columns: {missing}")
    df = df[REQUIRED_COLS].copy()
    df = df.astype(float)
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n = df.index.duplicated().sum()
    if n:
        logger.warning(f"[cleaner] Dropping {n} duplicate timestamps")
    return df[~df.index.duplicated(keep="last")]


def _drop_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["volume"] <= 0) | df["volume"].isna()
    n = mask.sum()
    if n:
        logger.warning(f"[cleaner] Dropping {n} zero/null-volume bars")
    return df[~mask]


def _fix_ohlc_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp high up and low down to maintain OHLC consistency."""
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"]  = df[["low",  "open", "close"]].min(axis=1)
    return df


def _remove_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bars where close deviates > SPIKE_THRESHOLD from rolling median."""
    rolling_median = df["close"].rolling(20, min_periods=5).median()
    pct_change = (df["close"] - rolling_median).abs() / rolling_median
    spikes = pct_change > SPIKE_THRESHOLD
    n = spikes.sum()
    if n:
        logger.warning(
            f"[cleaner] Removing {n} price spike bars "
            f"(>{SPIKE_THRESHOLD * 100:.0f}% from rolling median)"
        )
    return df[~spikes]


def _fill_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Reindex to expected trading frequency and forward-fill up to 3 bars.
    Remaining NaNs (weekends, holidays) are dropped.
    """
    freq_map = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "1h": "1h", "4h": "4h", "1d": "1D",
    }
    freq = freq_map.get(timeframe)
    if freq is None:
        return df

    full_index = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz
    )
    df = df.reindex(full_index)

    gaps = df["close"].isna().sum()
    if gaps:
        logger.debug(f"[cleaner] Forward-filling {gaps} gap bars (max 3 consecutive)")

    df = df.ffill(limit=3)
    df = df.dropna()
    return df
