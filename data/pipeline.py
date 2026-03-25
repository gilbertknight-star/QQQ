"""
data/pipeline.py — Fetch → Clean → Cache orchestrator.

Supports two data sources transparently:
  "ib"       — Interactive Brokers (live/paper trading)
  "yfinance" — yfinance fallback (backtesting, no IB required)

Cleaned DataFrames are cached as Parquet files. Subsequent calls
load from cache unless refresh=True or the cache is stale.

Usage:
    from config import config
    from data.pipeline import DataPipeline

    pipe = DataPipeline(config)

    # Backtest (yfinance, no IB needed)
    df_1h = pipe.get("NQ", "1h", source="yfinance")
    df_1d = pipe.get("NQ", "1d", source="yfinance")
    macro  = pipe.get_macro()

    # Live (IB)
    df_5m = pipe.get("NQ", "5m", source="ib")
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from .fetcher import DataFetcher, fetch_yfinance, _fetch_macro_yfinance
from .cleaner import clean


class DataPipeline:
    """
    Orchestrates data fetching, cleaning, and Parquet caching.

    Args:
        config: top-level Config dataclass from config.py
    """

    def __init__(self, config):
        self._cfg = config
        self._fetcher: DataFetcher | None = None
        self._cache_dir = Path(config.data.parquet_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # IB connection (lazy — only opened when source="ib")
    # ------------------------------------------------------------------

    def _get_fetcher(self) -> DataFetcher:
        if self._fetcher is None:
            self._fetcher = DataFetcher(self._cfg)
            ok = self._fetcher.connect()
            if not ok:
                raise RuntimeError(
                    "[pipeline] Could not connect to IB. "
                    "Is TWS / IB Gateway running? Try source='yfinance' for backtesting."
                )
        return self._fetcher

    def disconnect(self) -> None:
        if self._fetcher:
            self._fetcher.disconnect()
            self._fetcher = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str, timeframe: str, source: str) -> Path:
        return self._cache_dir / f"{symbol}_{timeframe}_{source}.parquet"

    def _is_stale(self, path: Path, max_age_hours: float = 4.0) -> bool:
        """Return True if the cache file is older than max_age_hours."""
        if not path.exists():
            return True
        age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        return age > max_age_hours * 3600

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get(
        self,
        symbol: str | None = None,
        timeframe: str = "1h",
        source: str | None = None,
        lookback_days: int | None = None,
        refresh: bool = False,
        max_cache_age_hours: float = 4.0,
    ) -> pd.DataFrame:
        """
        Return a clean OHLCV DataFrame for the given symbol and timeframe.

        Args:
            symbol:             "NQ" or "MNQ". Defaults to config.data.primary_symbol.
            timeframe:          "1m", "5m", "15m", "1h", "4h", "1d".
            source:             "ib" or "yfinance". Defaults to config.data.data_source.
            lookback_days:      Override lookback. Defaults conservatively per source/timeframe.
            refresh:            Force re-fetch even if cache is fresh.
            max_cache_age_hours: Cache is considered stale after this many hours.

        Returns:
            DataFrame indexed by UTC datetime with columns
            [open, high, low, close, volume], all float64.
        """
        symbol = symbol or self._cfg.data.primary_symbol
        source = source or self._cfg.data.data_source

        cache = self._cache_path(symbol, timeframe, source)

        if not refresh and not self._is_stale(cache, max_cache_age_hours):
            logger.info(f"[pipeline] Cache hit: {cache.name}")
            df = pd.read_parquet(cache)
            logger.info(f"[pipeline] {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")
            return df

        logger.info(f"[pipeline] Fetching {symbol} {timeframe} via {source}...")

        if source == "yfinance":
            days = lookback_days or self._default_lookback(timeframe, "yfinance")
            raw = fetch_yfinance(symbol=symbol, timeframe=timeframe, lookback_days=days)
        elif source == "ib":
            days = lookback_days or self._default_lookback(timeframe, "ib")
            fetcher = self._get_fetcher()
            raw = fetcher.get_historical_bars(
                symbol=symbol, timeframe=timeframe, lookback_days=days
            )
        else:
            raise ValueError(f"[pipeline] Unknown source '{source}'. Use 'ib' or 'yfinance'.")

        df = clean(raw, timeframe=timeframe)
        df.to_parquet(cache)
        logger.info(f"[pipeline] Cached → {cache}")
        return df

    # ------------------------------------------------------------------
    # Macro data
    # ------------------------------------------------------------------

    def get_macro(
        self,
        lookback_days: int = 252,           # Min 64 trading days needed for 63d momentum
        refresh: bool = False,
        max_cache_age_hours: float = 23.0,  # Macro is daily — refresh once per day
    ) -> pd.DataFrame:
        """
        Return macro indicator daily closes (VIX, TIP, IEF, SHY, GLD, GSG, UUP).
        Prior-day data only — no look-ahead.
        """
        cache = self._cache_dir / "macro_daily.parquet"

        if not refresh and not self._is_stale(cache, max_cache_age_hours):
            logger.info("[pipeline] Cache hit: macro_daily.parquet")
            return pd.read_parquet(cache)

        logger.info("[pipeline] Fetching macro data via yfinance...")
        df = _fetch_macro_yfinance(lookback_days=lookback_days)
        df.to_parquet(cache)
        logger.info(f"[pipeline] Macro cached → {cache}")
        return df

    # ------------------------------------------------------------------
    # Live bar (IB only)
    # ------------------------------------------------------------------

    def get_live_bar(self, symbol: str | None = None) -> pd.Series:
        """Return the current incomplete bar from IB (used in 5s loop)."""
        symbol = symbol or self._cfg.data.primary_symbol
        return self._get_fetcher().get_live_bar(symbol=symbol)

    # ------------------------------------------------------------------
    # Options chain (IB only)
    # ------------------------------------------------------------------

    def get_options_chain(
        self,
        symbol: str | None = None,
        expiry: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (calls_df, puts_df) from IB options chain."""
        symbol = symbol or self._cfg.data.primary_symbol
        return self._get_fetcher().get_options_chain(symbol=symbol, expiry=expiry)

    # ------------------------------------------------------------------
    # Convenience: pre-load all timeframes for backtesting
    # ------------------------------------------------------------------

    def load_backtest_data(
        self,
        symbol: str | None = None,
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Load all data needed for the walk-forward backtest via yfinance.

        Returns dict with keys: "1h", "1d", "macro"
        """
        symbol = symbol or self._cfg.data.primary_symbol
        logger.info(f"[pipeline] Loading backtest data for {symbol}...")

        return {
            "1d":    self.get(symbol, "1d",    source="yfinance", refresh=refresh),
            "1h":    self.get(symbol, "1h",    source="yfinance", refresh=refresh),
            "macro": self.get_macro(refresh=refresh),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _default_lookback(timeframe: str, source: str) -> int:
        """Conservative default lookback that won't exceed source limits."""
        if source == "yfinance":
            from .fetcher import YF_MAX_LOOKBACK_DAYS
            return YF_MAX_LOOKBACK_DAYS.get(timeframe, 60)
        else:
            from .fetcher import IB_MAX_LOOKBACK_DAYS
            return IB_MAX_LOOKBACK_DAYS.get(timeframe, 365)
