"""
Data Cleaner: normalizes raw OHLCV data from IB.
Handles gaps, bad ticks, zero volumes, and OHLC integrity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Max allowed price spike as a multiplier of recent median price
SPIKE_THRESHOLD = 0.10  # 10% move in a single bar flagged as suspicious


def clean(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a cleaned DataFrame.

    Steps:
      1. Drop duplicate timestamps
      2. Drop bars with zero or missing volume
      3. Fix OHLC integrity (high >= all, low <= all)
      4. Remove price spikes
      5. Forward-fill gaps up to max_gap_bars
      6. Sort by datetime ascending
    """
    df = df.copy()
    initial_len = len(df)

    df = _drop_duplicates(df)
    df = _drop_zero_volume(df)
    df = _fix_ohlc_integrity(df)
    df = _remove_spikes(df)
    df = _fill_gaps(df, timeframe)
    df = df.sort_index()

    removed = initial_len - len(df)
    print(f"[cleaner] {initial_len} bars in → {len(df)} bars out ({removed} removed/fixed)")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n = df.index.duplicated().sum()
    if n:
        print(f"[cleaner] Dropping {n} duplicate timestamps")
    return df[~df.index.duplicated(keep="last")]


def _drop_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["volume"] <= 0) | df["volume"].isna()
    n = mask.sum()
    if n:
        print(f"[cleaner] Dropping {n} zero/null-volume bars")
    return df[~mask]


def _fix_ohlc_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure high >= open,close and low <= open,close."""
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"]  = df[["low",  "open", "close"]].min(axis=1)
    return df


def _remove_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and remove bars where close moves > SPIKE_THRESHOLD vs rolling median."""
    rolling_median = df["close"].rolling(20, min_periods=5).median()
    pct_change = (df["close"] - rolling_median).abs() / rolling_median
    spikes = pct_change > SPIKE_THRESHOLD
    n = spikes.sum()
    if n:
        print(f"[cleaner] Removing {n} price spike bars (>{SPIKE_THRESHOLD*100:.0f}% from rolling median)")
    return df[~spikes]


def _fill_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Reindex to expected frequency and forward-fill up to 3 bars.
    Remaining NaNs (e.g. weekends, holidays) are dropped.
    """
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(timeframe)
    if freq is None:
        return df

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_index)

    gaps = df["close"].isna().sum()
    if gaps:
        print(f"[cleaner] Forward-filling {gaps} gap bars (max 3 consecutive)")

    df = df.ffill(limit=3)
    df = df.dropna()
    return df
