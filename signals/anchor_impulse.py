"""
signals/anchor_impulse.py — Anchor-Impulse Mean-Reversion Signal
                             (for volatile / ranging HMM regimes)

Translated from Pine Script "MNQ Prop Anchor-Impulse" strategy.

Strategy logic:
  Bias   : price must be above BOTH VWAP AND EMA(200) for long,
            below both for short.  No conflicted signals.
  Entry  : Long  — bullish bias AND close crosses UNDER SMA(20)
                   (pullback to the mean inside an uptrend)
           Short — bearish bias AND close crosses OVER SMA(20)
                   (bounce to the mean inside a downtrend)
  Stop   : entry ± ATR(14) × sl_atr_mult  (default 2.0)
  Exit   : Trailing stop.
             • Activates after price moves trail_activate_atr × ATR in favour
             • Once active, stop trails trail_offset_atr × ATR behind HWM
             • Hard ATR stop remains until trail activates

Why this works in volatile/ranging regimes:
  • Large ATR = bigger SMA pullbacks = more room between entry and stop
  • 200 EMA prevents trading against the dominant intraday trend
  • SMA crossunder/crossover is a clear, unambiguous trigger
  • Trailing stop captures extended volatile runs; hard stop caps the loss

Usage:
    from signals.anchor_impulse import AnchorImpulse, AnchorImpulseSignal

    ai = AnchorImpulse()
    sig = ai.get_signal(bars_window_5m)   # bars_window = last N 5m bars
    if sig.signal_valid:
        enter(sig.direction, stop=sig.stop_price)

    # Each bar while in position:
    trail_stop, activated = ai.update_trailing_stop(
        direction, entry_price, bar_high, bar_low,
        trail_hw, trail_activated, sig.atr
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

DirectionType = Literal["long", "short", "flat"]

# Regime gate: AnchorImpulse only fires in these HMM regimes
ANCHOR_REGIMES = {"volatile", "ranging"}


@dataclass
class AnchorImpulseSignal:
    """Output of AnchorImpulse.get_signal() for a single bar."""
    direction:        DirectionType   # "long", "short", or "flat"
    signal_valid:     bool            # True if entry conditions fully met

    # Prices
    entry_price:      float           # current bar close (proposed entry)
    stop_price:       float           # hard stop = entry ± ATR × sl_atr_mult

    # Indicators
    atr:              float           # ATR(14) value at this bar
    ema200:           float           # EMA(200) value
    sma20:            float           # SMA(20) value
    vwap:             float           # Session VWAP value

    # Flags
    bullish_bias:     bool            # close > VWAP AND close > EMA200
    bearish_bias:     bool            # close < VWAP AND close < EMA200
    crossed_under_sma: bool           # close just crossed below SMA20 (long trigger)
    crossed_over_sma:  bool           # close just crossed above SMA20 (short trigger)

    # Trailing stop parameters (fixed at entry time)
    trail_activate_pts: float         # points price must move to activate trail
    trail_offset_pts:   float         # ATR × trail_offset_atr

    def __str__(self) -> str:
        if not self.signal_valid:
            return (
                f"AnchorImpulse(flat | "
                f"bias={'bull' if self.bullish_bias else 'bear' if self.bearish_bias else 'none'} | "
                f"cross={'under' if self.crossed_under_sma else 'over' if self.crossed_over_sma else 'none'} | "
                f"ema200={self.ema200:.1f} sma20={self.sma20:.1f} vwap={self.vwap:.1f})"
            )
        return (
            f"AnchorImpulse({self.direction.upper()} | "
            f"entry={self.entry_price:.2f} stop={self.stop_price:.2f} | "
            f"atr={self.atr:.1f} trail_act={self.trail_activate_pts:.1f} "
            f"trail_off={self.trail_offset_pts:.1f})"
        )


class AnchorImpulse:
    """
    Anchor-Impulse pullback strategy for volatile/ranging HMM regimes.

    Mirrors the Pine Script "MNQ Prop Anchor-Impulse" logic exactly.

    Args:
        config: top-level Config dataclass (optional). Uses Pine Script
                defaults if None.
    """

    def __init__(self, config=None):
        # Allow config overrides; fall back to Pine Script defaults
        if config is not None and hasattr(config, "signal"):
            self._ema_len            = getattr(config.signal, "ai_ema_len",            200)
            self._sma_len            = getattr(config.signal, "ai_sma_len",             20)
            self._atr_len            = getattr(config.signal, "ai_atr_len",             14)
            self._sl_atr_mult        = getattr(config.signal, "ai_sl_atr_mult",        2.0)
            self._trail_activate_atr = getattr(config.signal, "ai_trail_activate_atr", 10.0)
            self._trail_offset_atr   = getattr(config.signal, "ai_trail_offset_atr",    2.0)
        else:
            self._ema_len            = 200   # Anchor EMA
            self._sma_len            = 20    # Entry SMA
            self._atr_len            = 14    # ATR period
            self._sl_atr_mult        = 2.0   # Hard stop distance (ATR multiples)
            self._trail_activate_atr = 10.0  # Activate trail after 10 ATR in favour
            self._trail_offset_atr   = 2.0   # Trail stop stays 2 ATR behind HWM

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def get_signal(self, bars_window: pd.DataFrame) -> AnchorImpulseSignal:
        """
        Generate an entry signal for the latest bar.

        Args:
            bars_window: Rolling window of 5m OHLCV bars (UTC index).
                         Needs at least EMA_LEN + 2 bars (202 minimum).

        Returns:
            AnchorImpulseSignal.  signal_valid=False if no entry.
        """
        needed = self._ema_len + 2
        if len(bars_window) < needed:
            logger.debug(
                f"[anchor] Insufficient bars ({len(bars_window)} < {needed}) — flat"
            )
            return self._flat(bars_window)

        close  = bars_window["close"]
        high   = bars_window["high"]
        low    = bars_window["low"]

        # ── Indicators ────────────────────────────────────────────────
        ema200 = self._calc_ema(close, self._ema_len)
        sma20  = close.rolling(self._sma_len).mean()
        atr    = self._calc_atr(high, low, close, self._atr_len)
        vwap   = self._calc_vwap(bars_window)

        curr_close  = float(close.iloc[-1])
        curr_ema200 = float(ema200.iloc[-1])
        curr_sma20  = float(sma20.iloc[-1])
        curr_atr    = float(atr.iloc[-1])
        curr_vwap   = float(vwap.iloc[-1])
        prev_close  = float(close.iloc[-2])
        prev_sma20  = float(sma20.iloc[-2])

        if any(pd.isna(x) for x in [curr_ema200, curr_sma20, curr_atr, curr_vwap]):
            return self._flat(bars_window)

        # ── Bias ──────────────────────────────────────────────────────
        bullish_bias = curr_close > curr_vwap and curr_close > curr_ema200
        bearish_bias = curr_close < curr_vwap and curr_close < curr_ema200

        # ── SMA crossovers ────────────────────────────────────────────
        # crossunder: was above SMA last bar, now at or below
        crossed_under = (prev_close > prev_sma20) and (curr_close <= curr_sma20)
        # crossover:  was below SMA last bar, now at or above
        crossed_over  = (prev_close < prev_sma20) and (curr_close >= curr_sma20)

        # ── Entry conditions ──────────────────────────────────────────
        long_entry  = bullish_bias and crossed_under
        short_entry = bearish_bias and crossed_over

        if not long_entry and not short_entry:
            return AnchorImpulseSignal(
                direction         = "flat",
                signal_valid      = False,
                entry_price       = curr_close,
                stop_price        = 0.0,
                atr               = curr_atr,
                ema200            = curr_ema200,
                sma20             = curr_sma20,
                vwap              = curr_vwap,
                bullish_bias      = bullish_bias,
                bearish_bias      = bearish_bias,
                crossed_under_sma = crossed_under,
                crossed_over_sma  = crossed_over,
                trail_activate_pts = curr_atr * self._trail_activate_atr,
                trail_offset_pts   = curr_atr * self._trail_offset_atr,
            )

        direction: DirectionType = "long" if long_entry else "short"
        sl_dist = curr_atr * self._sl_atr_mult
        stop_price = (
            curr_close - sl_dist if direction == "long"
            else curr_close + sl_dist
        )

        sig = AnchorImpulseSignal(
            direction         = direction,
            signal_valid      = True,
            entry_price       = curr_close,
            stop_price        = round(stop_price, 2),
            atr               = round(curr_atr, 4),
            ema200            = round(curr_ema200, 2),
            sma20             = round(curr_sma20, 2),
            vwap              = round(curr_vwap, 2),
            bullish_bias      = bullish_bias,
            bearish_bias      = bearish_bias,
            crossed_under_sma = crossed_under,
            crossed_over_sma  = crossed_over,
            trail_activate_pts = round(curr_atr * self._trail_activate_atr, 2),
            trail_offset_pts   = round(curr_atr * self._trail_offset_atr, 2),
        )
        logger.debug(f"[anchor] {sig}")
        return sig

    # ------------------------------------------------------------------
    # Trailing stop management
    # ------------------------------------------------------------------

    def update_trailing_stop(
        self,
        direction:        str,
        entry_price:      float,
        bar_high:         float,
        bar_low:          float,
        trail_hw:         float,     # current high-water mark (best price reached)
        trail_activated:  bool,      # whether trail is already active
        atr:              float,     # ATR at entry (fixed for life of trade)
    ) -> tuple[float, float, bool]:
        """
        Update the trailing stop for the current bar.

        Call this every bar while in a trailing-stop position.

        Args:
            direction:       "long" or "short"
            entry_price:     Original entry price.
            bar_high:        Current bar's high.
            bar_low:         Current bar's low.
            trail_hw:        Best price reached so far (high for long, low for short).
            trail_activated: True once the activation threshold was crossed.
            atr:             ATR value locked at entry time.

        Returns:
            (new_trail_hw, trail_stop_price, trail_now_activated)
            trail_stop_price is 0.0 until trail is activated.
        """
        activate_dist = atr * self._trail_activate_atr
        offset        = atr * self._trail_offset_atr

        if direction == "long":
            new_hw = max(trail_hw, bar_high)
            move   = new_hw - entry_price
            if move >= activate_dist or trail_activated:
                trail_stop = new_hw - offset
                return new_hw, trail_stop, True
            return new_hw, 0.0, False

        else:  # short
            new_hw = min(trail_hw, bar_low)
            move   = entry_price - new_hw
            if move >= activate_dist or trail_activated:
                trail_stop = new_hw + offset
                return new_hw, trail_stop, True
            return new_hw, 0.0, False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat(self, bars_window: pd.DataFrame) -> AnchorImpulseSignal:
        """Return a flat signal with safe defaults."""
        price = float(bars_window["close"].iloc[-1]) if len(bars_window) else 0.0
        return AnchorImpulseSignal(
            direction="flat", signal_valid=False,
            entry_price=price, stop_price=0.0,
            atr=20.0, ema200=price, sma20=price, vwap=price,
            bullish_bias=False, bearish_bias=False,
            crossed_under_sma=False, crossed_over_sma=False,
            trail_activate_pts=200.0, trail_offset_pts=40.0,
        )

    @staticmethod
    def _calc_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calc_atr(high: pd.Series, low: pd.Series,
                  close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()  # Wilder smoothing

    @staticmethod
    def _calc_vwap(bars_window: pd.DataFrame) -> pd.Series:
        """Session-resetting VWAP (resets at 09:30 ET each trading day)."""
        _ET = "America/New_York"
        df = bars_window.copy()
        # Convert to ET for session grouping
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df_et = df.copy()
        df_et.index = df.index.tz_convert(_ET)
        session_date = df_et.index.date

        typical  = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_vol   = typical * df["volume"]
        vwap_val = pd.Series(index=df.index, dtype=float)

        for d in np.unique(session_date):
            mask = session_date == d
            cum_tpv = tp_vol[mask].cumsum()
            cum_vol = df["volume"][mask].cumsum().replace(0, np.nan)
            vwap_val[mask] = cum_tpv / cum_vol

        return vwap_val
