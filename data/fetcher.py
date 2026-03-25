"""
data/fetcher.py — Market data fetcher for the NQ Regime Bot.

Two fetch paths:
  1. DataFetcher (class) — Interactive Brokers via ib_insync.
     Used for live and paper trading.
  2. fetch_yfinance() — standalone yfinance path for backtesting.
     No IB connection required.

DataFetcher API:
  connect()                       — connect to TWS/Gateway with retry
  disconnect()                    — clean shutdown
  get_historical_bars(...)        — OHLCV DataFrame from IB history
  get_live_bar()                  — current incomplete bar (5s updates)
  get_options_chain(symbol, exp)  — calls + puts DataFrames
  get_macro_data()                — VIX/TIP/IEF/SHY/GLD/GSG/UUP via yfinance
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

# IB import is deferred so the module loads even without ib_insync installed
try:
    from ib_insync import IB, Future, Stock, util
    _IB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IB_AVAILABLE = False
    logger.warning("ib_insync not installed — IB data unavailable, yfinance only.")


# ---------------------------------------------------------------------------
# Contract / timeframe mappings
# ---------------------------------------------------------------------------

CONTRACT_SPECS = {
    "MNQ": {"exchange": "CME", "currency": "USD", "multiplier": "2"},
    "NQ":  {"exchange": "CME", "currency": "USD", "multiplier": "20"},
}

# IB bar size strings
IB_BAR_SIZE = {
    "1m":  "1 min",
    "5m":  "5 mins",
    "15m": "15 mins",
    "1h":  "1 hour",
    "4h":  "4 hours",
    "1d":  "1 day",
}

# IB hard limits on intraday history depth
IB_MAX_LOOKBACK_DAYS = {
    "1m":  30,
    "5m":  182,
    "15m": 365,
    "1h":  730,
    "4h":  730,
    "1d":  3_650,
}

# yfinance tickers for NQ futures (continuous front-month)
YF_SYMBOL_MAP = {
    "NQ":  "NQ=F",
    "MNQ": "MNQ=F",
}

# yfinance interval strings
YF_INTERVAL = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# yfinance max lookback days per interval
YF_MAX_LOOKBACK_DAYS = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "1h":  730,
    "4h":  60,
    "1d":  9_999,   # effectively unlimited
}

# Macro instruments fetched via yfinance
MACRO_TICKERS = {
    "VIX": "^VIX",
    "TIP": "TIP",
    "IEF": "IEF",
    "SHY": "SHY",
    "GLD": "GLD",
    "GSG": "GSG",
    "UUP": "UUP",
}

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_front_month_expiry() -> str:
    """Return nearest quarterly expiry (Mar/Jun/Sep/Dec) as 'YYYYMM'."""
    today = datetime.today()
    for month in [3, 6, 9, 12]:
        expiry = datetime(today.year, month, 1)
        if expiry > today + timedelta(days=10):
            return expiry.strftime("%Y%m")
    return datetime(today.year + 1, 3, 1).strftime("%Y%m")


def _ib_duration(days: int) -> str:
    if days <= 365:
        return f"{days} D"
    return f"{round(days / 365)} Y"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, enforce float dtype, ensure UTC index."""
    col_map = {c: c.lower() for c in df.columns}
    col_map.update({"Date": "datetime", "Datetime": "datetime", "date": "datetime"})
    df = df.rename(columns=col_map)

    if "datetime" in df.columns:
        df = df.set_index("datetime")

    df.index.name = "datetime"
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[[c for c in OHLCV_COLS if c in df.columns]]
    df = df.astype(float)
    return df


# ---------------------------------------------------------------------------
# DataFetcher — IB-backed class
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    Fetches market data from Interactive Brokers.

    Falls back to yfinance for historical and macro data when IB is
    unavailable or when explicitly requested (use_yf=True).
    """

    def __init__(self, config):
        self._cfg = config
        self._ib: Optional["IB"] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, max_retries: int = 5, retry_delay: float = 5.0) -> bool:
        """
        Establish connection to TWS / IB Gateway with retry logic.

        Returns True on success, False if all retries exhausted.
        """
        if not _IB_AVAILABLE:
            logger.error("ib_insync not installed — cannot connect to IB.")
            return False

        ib_cfg = self._cfg.ib
        for attempt in range(1, max_retries + 1):
            try:
                self._ib = IB()
                self._ib.connect(
                    ib_cfg.host,
                    ib_cfg.port,
                    clientId=ib_cfg.client_id,
                    timeout=ib_cfg.timeout,
                    readonly=ib_cfg.readonly,
                )
                self._connected = True
                logger.info(
                    f"[fetcher] Connected to IB at {ib_cfg.host}:{ib_cfg.port} "
                    f"(clientId={ib_cfg.client_id})"
                )
                return True
            except Exception as exc:
                logger.warning(
                    f"[fetcher] IB connect attempt {attempt}/{max_retries} failed: {exc}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)

        logger.error("[fetcher] All IB connection attempts failed.")
        self._connected = False
        return False

    def disconnect(self) -> None:
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("[fetcher] Disconnected from IB.")

    def _ensure_connected(self) -> None:
        """Reconnect if the IB socket has dropped."""
        if not self._connected or (self._ib and not self._ib.isConnected()):
            logger.warning("[fetcher] IB connection lost — attempting reconnect.")
            self.connect()

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str = "NQ",
        timeframe: str = "1h",
        lookback_days: int = 730,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars from IB.

        Returns DataFrame indexed by UTC datetime with columns
        [open, high, low, close, volume].
        Logs every fetch with timestamp.
        """
        self._ensure_connected()

        max_days = IB_MAX_LOOKBACK_DAYS.get(timeframe, 730)
        if lookback_days > max_days:
            logger.warning(
                f"[fetcher] {timeframe} clamped: {lookback_days}d → {max_days}d (IB limit)"
            )
            lookback_days = max_days

        spec = CONTRACT_SPECS[symbol]
        expiry = _get_front_month_expiry()
        contract = Future(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            exchange=spec["exchange"],
            currency=spec["currency"],
            multiplier=spec["multiplier"],
        )
        self._ib.qualifyContracts(contract)

        logger.info(
            f"[fetcher] {datetime.now(timezone.utc).isoformat()} — "
            f"IB historical: {symbol} {timeframe} {lookback_days}d"
        )

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=_ib_duration(lookback_days),
            barSizeSetting=IB_BAR_SIZE[timeframe],
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            raise RuntimeError(
                f"[fetcher] No data returned for {symbol} {timeframe}. "
                "Is market data subscribed in TWS?"
            )

        df = util.df(bars)
        df = df.rename(columns={"date": "datetime"})
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[OHLCV_COLS].astype(float)

        logger.info(
            f"[fetcher] Retrieved {len(df)} bars: "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df

    # ------------------------------------------------------------------
    # Live bar (current incomplete bar, updated every 5s)
    # ------------------------------------------------------------------

    def get_live_bar(self, symbol: str = "NQ") -> pd.Series:
        """
        Return the current incomplete bar as a Series with index
        [open, high, low, close, volume].

        Uses reqHistoricalData with keepUpToDate=True, then reads
        the last (live) bar. Called in the 5-second bot loop.
        """
        self._ensure_connected()

        spec = CONTRACT_SPECS[symbol]
        expiry = _get_front_month_expiry()
        contract = Future(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            exchange=spec["exchange"],
            currency=spec["currency"],
            multiplier=spec["multiplier"],
        )
        self._ib.qualifyContracts(contract)

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
            keepUpToDate=True,
        )

        if not bars:
            raise RuntimeError(f"[fetcher] No live bar data for {symbol}.")

        last = bars[-1]
        logger.debug(
            f"[fetcher] Live bar {symbol}: "
            f"O={last.open} H={last.high} L={last.low} C={last.close} V={last.volume}"
        )
        return pd.Series(
            {
                "open": float(last.open),
                "high": float(last.high),
                "low":  float(last.low),
                "close": float(last.close),
                "volume": float(last.volume),
            },
            name=pd.Timestamp(last.date, tz="UTC"),
        )

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    def get_options_chain(
        self,
        symbol: str = "NQ",
        expiry: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch options chain from IB for the given symbol and expiry.

        Returns (calls_df, puts_df), each with columns:
        [strike, bid, ask, last, impliedVolatility, volume, openInterest]

        Falls back to empty DataFrames if IB options data unavailable.
        """
        self._ensure_connected()

        opt_cols = ["strike", "bid", "ask", "last", "impliedVolatility", "volume", "openInterest"]
        empty = pd.DataFrame(columns=opt_cols)

        if expiry is None:
            expiry = _get_front_month_expiry()

        try:
            # Get the underlying contract to resolve conId
            spec = CONTRACT_SPECS.get(symbol, CONTRACT_SPECS["NQ"])
            underlying = Future(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                exchange=spec["exchange"],
                currency=spec["currency"],
            )
            self._ib.qualifyContracts(underlying)

            # Request option chain details
            chains = self._ib.reqSecDefOptParams(
                underlying.symbol, "", underlying.secType, underlying.conId
            )
            if not chains:
                logger.warning("[fetcher] No option chain returned from IB.")
                return empty.copy(), empty.copy()

            chain = next((c for c in chains if c.exchange == "CME"), chains[0])
            strikes = sorted(chain.strikes)

            # Build option contracts for each strike
            from ib_insync import Option
            opt_contracts = [
                Option(symbol, expiry, strike, right, "CME")
                for right in ["C", "P"]
                for strike in strikes
            ]
            self._ib.qualifyContracts(*opt_contracts)

            tickers = self._ib.reqTickers(*opt_contracts)
            self._ib.sleep(2)  # Wait for market data

            rows = []
            for ticker in tickers:
                c = ticker.contract
                rows.append({
                    "right":            c.right,
                    "strike":           float(c.strike),
                    "bid":              ticker.bid if ticker.bid != -1 else float("nan"),
                    "ask":              ticker.ask if ticker.ask != -1 else float("nan"),
                    "last":             ticker.last if ticker.last != -1 else float("nan"),
                    "impliedVolatility": ticker.modelGreeks.impliedVol if ticker.modelGreeks else float("nan"),
                    "volume":           ticker.volume if ticker.volume != -1 else 0,
                    "openInterest":     ticker.openInterest if ticker.openInterest != -1 else 0,
                })

            df = pd.DataFrame(rows)
            calls = df[df["right"] == "C"][opt_cols].reset_index(drop=True)
            puts  = df[df["right"] == "P"][opt_cols].reset_index(drop=True)

            logger.info(
                f"[fetcher] {datetime.now(timezone.utc).isoformat()} — "
                f"Options chain: {len(calls)} calls, {len(puts)} puts"
            )
            return calls, puts

        except Exception as exc:
            logger.error(f"[fetcher] Options chain fetch failed: {exc}")
            return empty.copy(), empty.copy()

    # ------------------------------------------------------------------
    # Macro data (always via yfinance — prior-day closes)
    # ------------------------------------------------------------------

    def get_macro_data(self, lookback_days: int = 126) -> pd.DataFrame:
        """
        Pull macro indicator daily closes via yfinance.

        Returns DataFrame with columns [VIX, TIP, IEF, SHY, GLD, GSG, UUP]
        indexed by date, using prior-day closes only (no look-ahead).
        Logs every fetch with timestamp.
        """
        return _fetch_macro_yfinance(lookback_days)


# ---------------------------------------------------------------------------
# Standalone yfinance fetch — used by backtest engine directly
# ---------------------------------------------------------------------------

def fetch_yfinance(
    symbol: str = "NQ",
    timeframe: str = "1h",
    lookback_days: int = 730,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from yfinance. No IB required.

    Returns DataFrame indexed by UTC datetime with columns
    [open, high, low, close, volume].

    Raises ValueError if lookback_days exceeds yfinance's limit
    for the given timeframe (with a warning, not a hard error).
    """
    yf_sym = YF_SYMBOL_MAP.get(symbol, symbol)
    interval = YF_INTERVAL.get(timeframe)
    if interval is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}' for yfinance.")

    max_days = YF_MAX_LOOKBACK_DAYS[timeframe]
    if lookback_days > max_days:
        logger.warning(
            f"[fetcher] yfinance {timeframe} max is {max_days}d — "
            f"clamping from {lookback_days}d."
        )
        lookback_days = max_days

    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    logger.info(
        f"[fetcher] {datetime.now(timezone.utc).isoformat()} — "
        f"yfinance: {yf_sym} {timeframe} from {start.date()}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticker = yf.Ticker(yf_sym)
        raw = ticker.history(start=start.strftime("%Y-%m-%d"), interval=interval)

    if raw.empty:
        raise RuntimeError(
            f"[fetcher] yfinance returned no data for {yf_sym} {timeframe}. "
            "Check symbol and network."
        )

    df = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[OHLCV_COLS]
    df.index.name = "datetime"

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.astype(float)

    logger.info(
        f"[fetcher] Retrieved {len(df)} bars: "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


# ---------------------------------------------------------------------------
# Macro fetch (always yfinance)
# ---------------------------------------------------------------------------

def _fetch_macro_yfinance(lookback_days: int = 126) -> pd.DataFrame:
    """
    Fetch daily closes for all macro instruments via yfinance.

    Returns DataFrame with one column per instrument, indexed by date.
    Uses prior-day closes only — today's incomplete bar is dropped.
    """
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    logger.info(
        f"[fetcher] {datetime.now(timezone.utc).isoformat()} — "
        f"Macro data via yfinance (lookback={lookback_days}d)"
    )

    closes = {}
    for name, ticker_sym in MACRO_TICKERS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.Ticker(ticker_sym).history(start=start, interval="1d")
            if raw.empty:
                logger.warning(f"[fetcher] No macro data for {ticker_sym}")
                continue
            # Normalise index to UTC date (drop intraday time component)
            idx = pd.to_datetime(raw.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC")
            else:
                idx = idx.tz_localize("UTC")
            series = raw["Close"].copy()
            series.index = idx.normalize()
            series = series[~series.index.duplicated(keep="last")]
            closes[name] = series
        except Exception as exc:
            logger.warning(f"[fetcher] Macro fetch failed for {ticker_sym}: {exc}")

    if not closes:
        raise RuntimeError("[fetcher] All macro data fetches failed.")

    # Outer join — keeps all dates; NaNs where a ticker had no data that day
    df = pd.DataFrame(closes).sort_index()

    # Drop today's incomplete bar — signals must use prior-day data only
    today = pd.Timestamp.now(tz="UTC").normalize()
    df = df[df.index < today]

    # Forward-fill up to 3 days for instruments with different trading calendars
    df = df.ffill(limit=3)
    df = df.dropna()

    logger.info(f"[fetcher] Macro data: {len(df)} days, {list(df.columns)}")
    return df
