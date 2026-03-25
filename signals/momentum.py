"""
signals/momentum.py — Intraday Directional Signal (Signal Layer 3)

Generates a directional signal from two inputs:
  1. VWAP relationship — is price above or below the session average?
  2. ATR-normalized momentum — is the move accelerating in that direction?

Both must agree. Macro bias acts as a filter (can't go long in a short-biased
macro regime and vice versa).

VWAP resets at each NYSE session open. Works on any intraday timeframe
(1h recommended for backtesting, 5m for live trading).

Usage:
    from signals.momentum import MomentumSignal
    sig = MomentumSignal(config)
    result = sig.get_signal(bars_df, macro_bias="long")
    if result.signal_valid:
        trade(result.direction)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

# US Eastern timezone — used to determine session date boundaries
_ET = "America/New_York"
_SESSION_OPEN_HOUR = 9
_SESSION_OPEN_MIN  = 30

# Normalization scales for signal strength calculation
_VWAP_FULL_STRENGTH_PCT = 0.005   # 0.5% from VWAP = maximum VWAP strength contribution
_MOM_FULL_STRENGTH      = 2.0     # ATR-normalized momentum of 2.0 = maximum contribution

DirectionType = Literal["long", "short", "flat"]
BiastType     = Literal["long", "short", "neutral"]


@dataclass
class SignalResult:
    """Output of MomentumSignal.get_signal() for a single bar."""
    direction:         DirectionType  # "long", "short", or "flat"
    strength:          float          # 0–1 composite signal strength
    vwap_distance_pct: float          # (price - vwap) / vwap  (signed)
    momentum_value:    float          # ATR-normalized ROC (signed)
    signal_valid:      bool           # True only if strength > min_strength

    def __str__(self) -> str:
        valid = "VALID" if self.signal_valid else "weak"
        return (
            f"SignalResult({valid} {self.direction} | "
            f"strength={self.strength:.2f} | "
            f"vwap_dist={self.vwap_distance_pct:+.3%} | "
            f"mom={self.momentum_value:+.3f})"
        )


class MomentumSignal:
    """
    Intraday directional signal from VWAP + ATR-normalized momentum.

    Args:
        config: top-level Config dataclass. If None, uses spec defaults.
    """

    def __init__(self, config=None):
        if config is not None:
            self._lookback_bars = config.signal.momentum_lookback_bars
            self._min_strength  = config.signal.momentum_strength_min
            self._vwap_flat_pct = config.signal.vwap_proximity_flat_pct
        else:
            self._lookback_bars = 12     # ROC lookback (spec default)
            self._min_strength  = 0.40   # Below this → signal_valid = False
            self._vwap_flat_pct = 0.001  # Within 0.1% of VWAP → flat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_signal(
        self,
        bars_df: pd.DataFrame,
        macro_bias: BiastType = "neutral",
    ) -> SignalResult:
        """
        Generate a directional signal for the latest bar in bars_df.

        Signal rules:
          LONG  — price > VWAP  AND  momentum > 0  AND  macro_bias != "short"
          SHORT — price < VWAP  AND  momentum < 0  AND  macro_bias != "long"
          FLAT  — any other condition (near VWAP, opposing signals, macro block)

        Signal strength combines:
          - How far price is from VWAP (normalized to 0–1)
          - Magnitude of ATR-normalized momentum (normalized to 0–1)

        signal_valid = True only if composite strength > min_strength (0.4).

        Args:
            bars_df:    intraday OHLCV DataFrame with UTC datetime index.
                        Must have at least lookback_bars + ATR_period bars.
            macro_bias: "long", "short", or "neutral" from MacroRegimeScorer.

        Returns:
            SignalResult dataclass.
        """
        if len(bars_df) < max(self._lookback_bars, 14) + 2:
            logger.warning("[momentum] Insufficient bars for signal — returning flat.")
            return SignalResult("flat", 0.0, 0.0, 0.0, False)

        vwap   = self.calculate_vwap(bars_df)
        mom    = self.calculate_momentum(bars_df, self._lookback_bars)
        price  = float(bars_df["close"].iloc[-1])
        vwap_  = float(vwap.iloc[-1])
        mom_   = float(mom.iloc[-1])

        if pd.isna(vwap_) or pd.isna(mom_):
            logger.warning("[momentum] NaN in VWAP or momentum — returning flat.")
            return SignalResult("flat", 0.0, 0.0, 0.0, False)

        vwap_dist_pct = (price - vwap_) / vwap_  if vwap_ != 0 else 0.0

        # Direction determination
        above_vwap = vwap_dist_pct > self._vwap_flat_pct
        below_vwap = vwap_dist_pct < -self._vwap_flat_pct
        near_vwap  = not above_vwap and not below_vwap

        if near_vwap:
            direction: DirectionType = "flat"
        elif above_vwap and mom_ > 0 and macro_bias != "short":
            direction = "long"
        elif below_vwap and mom_ < 0 and macro_bias != "long":
            direction = "short"
        else:
            direction = "flat"

        # Signal strength (0–1 composite)
        vwap_strength = min(abs(vwap_dist_pct) / _VWAP_FULL_STRENGTH_PCT, 1.0)
        mom_strength  = min(abs(mom_) / _MOM_FULL_STRENGTH, 1.0)
        strength = 0.5 * vwap_strength + 0.5 * mom_strength

        signal_valid = (direction != "flat") and (strength >= self._min_strength)

        result = SignalResult(
            direction         = direction,
            strength          = round(strength, 4),
            vwap_distance_pct = round(vwap_dist_pct, 6),
            momentum_value    = round(mom_, 4),
            signal_valid      = signal_valid,
        )
        logger.debug(f"[momentum] {result} | macro_bias={macro_bias}")
        return result

    # ------------------------------------------------------------------
    # VWAP — resets at each NYSE session open
    # ------------------------------------------------------------------

    def calculate_vwap(self, bars_df: pd.DataFrame) -> pd.Series:
        """
        Compute session VWAP for every bar in bars_df.

        VWAP = cumsum(typical_price * volume) / cumsum(volume)

        Resets at each NYSE session open (9:30 AM ET). Bars before the
        first session open in the data share the same cumulative VWAP.

        Returns a Series aligned to bars_df.index.
        """
        df = bars_df.copy()
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]

        # Determine session date in ET for each bar
        if df.index.tz is None:
            idx_et = df.index.tz_localize("UTC").tz_convert(_ET)
        else:
            idx_et = df.index.tz_convert(_ET)

        # Session date = ET date, but bars before 9:30 ET belong to the
        # prior session's date group (pre-market). Assign each bar to the
        # NYSE session it belongs to.
        session_dates = _assign_session_date(idx_et)
        df["session_date"] = session_dates.values

        # Cumulative VWAP within each session
        vwap = (
            df.groupby("session_date", sort=False)["tp_vol"]
            .cumsum()
            .div(
                df.groupby("session_date", sort=False)["volume"].cumsum()
            )
        )
        vwap.index = bars_df.index
        return vwap

    # ------------------------------------------------------------------
    # Momentum — ATR-normalized rate of change
    # ------------------------------------------------------------------

    def calculate_momentum(
        self,
        bars_df: pd.DataFrame,
        lookback_bars: int = 12,
    ) -> pd.Series:
        """
        ATR-normalized rate of change of close over lookback_bars.

        momentum = (close - close[lookback]) / ATR_14

        Dividing by ATR makes the value scale-independent — a value of 1.0
        means the move equals one ATR over the lookback period.

        Returns a Series aligned to bars_df.index.
        """
        atr    = self.get_atr(bars_df, period=14)
        roc    = bars_df["close"] - bars_df["close"].shift(lookback_bars)
        mom    = roc / atr.replace(0, np.nan)
        return mom

    # ------------------------------------------------------------------
    # ATR helper
    # ------------------------------------------------------------------

    def get_atr(self, bars_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute Average True Range over `period` bars.

        True Range = max(high - low,
                         |high - prev_close|,
                         |low  - prev_close|)
        ATR = rolling mean of TR.

        Returns a Series aligned to bars_df.index.
        """
        prev_close = bars_df["close"].shift(1)
        tr = pd.concat(
            [
                bars_df["high"] - bars_df["low"],
                (bars_df["high"] - prev_close).abs(),
                (bars_df["low"]  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    # ------------------------------------------------------------------
    # Batch scoring (for backtest engine)
    # ------------------------------------------------------------------

    def score_history(
        self,
        bars_df: pd.DataFrame,
        macro_bias_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Apply get_signal() to every bar in bars_df using only past data.

        Used by the backtest engine. macro_bias_series should be a Series
        indexed by date (from MacroRegimeScorer.score_history()) — each bar
        is looked up by its session date.

        Returns DataFrame with columns:
          [direction, strength, vwap_distance_pct, momentum_value, signal_valid]
        """
        vwap = self.calculate_vwap(bars_df)
        mom  = self.calculate_momentum(bars_df, self._lookback_bars)
        atr  = self.get_atr(bars_df)

        rows = []
        for i in range(len(bars_df)):
            if i < max(self._lookback_bars, 14):
                rows.append({
                    "direction": "flat", "strength": 0.0,
                    "vwap_distance_pct": 0.0, "momentum_value": 0.0,
                    "signal_valid": False,
                })
                continue

            price   = float(bars_df["close"].iloc[i])
            vwap_   = float(vwap.iloc[i])
            mom_    = float(mom.iloc[i])

            if pd.isna(vwap_) or pd.isna(mom_):
                rows.append({
                    "direction": "flat", "strength": 0.0,
                    "vwap_distance_pct": 0.0, "momentum_value": 0.0,
                    "signal_valid": False,
                })
                continue

            # Look up macro bias for this bar's session date
            bias: BiastType = "neutral"
            if macro_bias_series is not None:
                bar_date = bars_df.index[i]
                if bar_date.tz is not None:
                    bar_date_et = bar_date.tz_convert(_ET).date()
                else:
                    bar_date_et = bar_date.date()
                # Find the prior trading day's macro bias (no look-ahead)
                prior_dates = macro_bias_series.index[
                    macro_bias_series.index < pd.Timestamp(bar_date_et, tz="UTC")
                ]
                if len(prior_dates) > 0:
                    raw_bias = macro_bias_series.iloc[
                        macro_bias_series.index.get_loc(prior_dates[-1])
                    ]
                    bias = raw_bias if raw_bias in ("long", "short", "neutral") else "neutral"

            vwap_dist_pct = (price - vwap_) / vwap_ if vwap_ != 0 else 0.0
            above_vwap = vwap_dist_pct > self._vwap_flat_pct
            below_vwap = vwap_dist_pct < -self._vwap_flat_pct

            if not above_vwap and not below_vwap:
                direction: DirectionType = "flat"
            elif above_vwap and mom_ > 0 and bias != "short":
                direction = "long"
            elif below_vwap and mom_ < 0 and bias != "long":
                direction = "short"
            else:
                direction = "flat"

            vwap_s = min(abs(vwap_dist_pct) / _VWAP_FULL_STRENGTH_PCT, 1.0)
            mom_s  = min(abs(mom_) / _MOM_FULL_STRENGTH, 1.0)
            strength = 0.5 * vwap_s + 0.5 * mom_s
            signal_valid = (direction != "flat") and (strength >= self._min_strength)

            rows.append({
                "direction":         direction,
                "strength":          round(strength, 4),
                "vwap_distance_pct": round(vwap_dist_pct, 6),
                "momentum_value":    round(mom_, 4),
                "signal_valid":      signal_valid,
            })

        return pd.DataFrame(rows, index=bars_df.index)


# ---------------------------------------------------------------------------
# Helper — assign each bar to its NYSE session date
# ---------------------------------------------------------------------------

def _assign_session_date(idx_et: pd.DatetimeIndex) -> pd.Series:
    """
    Assign each ET bar timestamp to its NYSE session date.

    Bars at or after 9:30 AM ET belong to that calendar date.
    Bars before 9:30 AM ET (pre-market / overnight) belong to the
    same date but are grouped with the prior day's session for VWAP
    continuity — this avoids VWAP resetting in the middle of a
    pre-market session.

    For simplicity (backtesting on RTH bars), we just use the ET date.
    RTH bars all fall between 9:30–16:00 ET, so the date is unambiguous.
    """
    dates = []
    for ts in idx_et:
        if ts.hour < _SESSION_OPEN_HOUR or (
            ts.hour == _SESSION_OPEN_HOUR and ts.minute < _SESSION_OPEN_MIN
        ):
            # Pre-market: group with prior calendar date
            prior = ts - pd.Timedelta(days=1)
            dates.append(prior.date())
        else:
            dates.append(ts.date())
    return pd.Series(dates, index=idx_et)
