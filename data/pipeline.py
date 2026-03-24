"""
Data Pipeline: orchestrates fetch → clean → cache.

Saves cleaned data to data/processed/ as Parquet files.
On subsequent runs, loads from cache unless refresh=True.

Usage:
    from data.pipeline import DataPipeline
    pipe = DataPipeline(config)
    df_1h = pipe.get("MNQ", "1h")
    df_1d = pipe.get("MNQ", "1d")
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .fetcher import fetch_historical
from .cleaner import clean


PROCESSED_DIR = Path(__file__).parent / "processed"


class DataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.symbol = config["strategy"]["instrument"]
        self.lookback_days = config["data"]["lookback_days"]
        self.ib_host = config["ib"]["host"]
        self.ib_port = config["ib"]["port"]
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return PROCESSED_DIR / f"{symbol}_{timeframe}.parquet"

    def get(self, symbol: str = None, timeframe: str = "1h", refresh: bool = False) -> pd.DataFrame:
        """
        Return cleaned OHLCV DataFrame for symbol/timeframe.
        Loads from cache if available, unless refresh=True.
        """
        symbol = symbol or self.symbol
        cache = self._cache_path(symbol, timeframe)

        if cache.exists() and not refresh:
            print(f"[pipeline] Loading cached data: {cache.name}")
            df = pd.read_parquet(cache)
            print(f"[pipeline] {len(df)} bars | {df.index[0]} → {df.index[-1]}")
            return df

        print(f"[pipeline] Fetching fresh data for {symbol} {timeframe}...")
        raw = fetch_historical(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=self.lookback_days,
            host=self.ib_host,
            port=self.ib_port,
        )
        df = clean(raw, timeframe=timeframe)
        df.to_parquet(cache)
        print(f"[pipeline] Saved to {cache}")
        return df

    def refresh_all(self):
        """Re-fetch and re-cache all configured timeframes."""
        for tf in self.config["data"]["timeframes"]:
            print(f"\n--- Refreshing {self.symbol} {tf} ---")
            self.get(timeframe=tf, refresh=True)
