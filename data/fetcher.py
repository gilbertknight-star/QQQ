"""
Data Fetcher: pulls historical OHLCV data from Interactive Brokers.
Supports MNQ (Micro NQ) and NQ (full) futures contracts.

Requires TWS or IB Gateway running with API connections enabled:
  TWS: File > Global Configuration > API > Settings > Enable ActiveX and Socket Clients
  Paper port: 7497 | Live port: 7496
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta
from ib_insync import IB, Future, util


# MNQ and NQ front-month contract specs
CONTRACT_SPECS = {
    "MNQ": {"exchange": "CME", "currency": "USD", "multiplier": "2"},
    "NQ":  {"exchange": "CME", "currency": "USD", "multiplier": "20"},
}

# Map timeframe strings to IB bar size settings
BAR_SIZE_MAP = {
    "1m":  "1 min",
    "5m":  "5 mins",
    "15m": "15 mins",
    "1h":  "1 hour",
    "4h":  "4 hours",
    "1d":  "1 day",
}

# Map lookback days to IB duration strings
def _duration_str(days: int) -> str:
    if days <= 365:
        return f"{days} D"
    years = round(days / 365)
    return f"{years} Y"


def _get_front_month_expiry() -> str:
    """Return the nearest quarterly expiry (Mar/Jun/Sep/Dec) as 'YYYYMM'."""
    today = datetime.today()
    quarterly_months = [3, 6, 9, 12]
    for month in quarterly_months:
        expiry = datetime(today.year, month, 1)
        if expiry > today + timedelta(days=10):
            return expiry.strftime("%Y%m")
    return datetime(today.year + 1, 3, 1).strftime("%Y%m")


def fetch_historical(
    symbol: str = "MNQ",
    timeframe: str = "1h",
    lookback_days: int = 730,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 2,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from IB for MNQ or NQ futures.

    Returns a DataFrame with columns: open, high, low, close, volume
    indexed by datetime (UTC).
    """
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"Unknown symbol '{symbol}'. Choose from: {list(CONTRACT_SPECS)}")
    if timeframe not in BAR_SIZE_MAP:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Choose from: {list(BAR_SIZE_MAP)}")

    spec = CONTRACT_SPECS[symbol]
    expiry = _get_front_month_expiry()
    bar_size = BAR_SIZE_MAP[timeframe]
    duration = _duration_str(lookback_days)

    ib = IB()
    ib.connect(host, port, clientId=client_id)
    print(f"[fetcher] Connected to IB at {host}:{port}")

    try:
        contract = Future(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            exchange=spec["exchange"],
            currency=spec["currency"],
            multiplier=spec["multiplier"],
        )
        ib.qualifyContracts(contract)
        print(f"[fetcher] Fetching {symbol} {expiry} | {timeframe} bars | {lookback_days}d lookback")

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
    finally:
        ib.disconnect()
        print("[fetcher] Disconnected from IB.")

    if not bars:
        raise RuntimeError(f"No data returned for {symbol}. Is TWS running and market data subscribed?")

    df = util.df(bars)
    df = df.rename(columns={"date": "datetime", "barCount": "bar_count", "average": "vwap"})
    df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.astype(float)

    print(f"[fetcher] Retrieved {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df
