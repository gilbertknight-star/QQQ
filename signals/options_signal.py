"""
signals/options_signal.py — Options Intelligence Signal (Signal Layer 5)

Reads institutional positioning from the NQ options chain:
  - Put/Call Ratio (volume-based + OI-based composite)
  - IV Skew (downside premium vs upside premium)
  - IV Percentile (current IV vs 252-day history)

Interpretation:
  PCR < 0.70 → more calls than puts → bullish institutional lean
  PCR > 1.30 → more puts than calls → bearish institutional lean
  Skew < 0   → upside IV > downside IV → bullish (calls expensive)
  Skew > 0   → downside IV > upside IV → bearish (puts expensive)

This signal ALWAYS falls back to neutral with signal_valid=False when:
  - Options data is unavailable (no API key, IB not connected)
  - Total options volume < min_volume (illiquid chain)
  - Data has too many NaNs to compute reliable IV

Being 4/5 on signal agreement (without options) is still enough to trade
when min_signal_agreement=3. Options data is a bonus confirmation layer.

Usage:
    from signals.options_signal import OptionsSignal
    sig = OptionsSignal(config)

    # With live data (IB or Polygon):
    result = sig.get_signal(calls_df, puts_df, current_price=24300.0)

    # Backtest (no options data) — graceful neutral:
    result = sig.get_signal(None, None, current_price=24300.0)
    # result.signal_valid == False, result.directional_lean == "neutral"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger

# PCR thresholds
PCR_BULLISH_THRESHOLD = 0.70   # Below → bullish
PCR_BEARISH_THRESHOLD = 1.30   # Above → bearish
PCR_NEUTRAL_LOW       = 0.70
PCR_NEUTRAL_HIGH      = 1.30

# Minimum total options volume for a valid signal
MIN_VOLUME_DEFAULT = 500

# IV skew wing distance (NQ points away from ATM)
WING_DISTANCE_DEFAULT = 100.0

# Minimum IV for a strike to be considered reliable
MIN_IV = 0.001

DirectionalLean = Literal["long", "short", "neutral"]


@dataclass
class OptionsSignalResult:
    """Output of OptionsSignal.get_signal()."""
    directional_lean: DirectionalLean  # "long", "short", or "neutral"
    confidence:       float            # 0–1 composite confidence
    pcr_value:        float            # Composite put/call ratio
    skew_value:       float            # IV skew: downside_iv - upside_iv
    iv_percentile:    float            # Current IV percentile (0–100)
    signal_valid:     bool             # False if data unavailable or illiquid

    # Sub-components for logging
    volume_pcr:   float = 0.0
    oi_pcr:       float = 0.0
    total_volume: float = 0.0
    fallback_reason: str = ""

    def __str__(self) -> str:
        valid = "VALID" if self.signal_valid else f"invalid({self.fallback_reason})"
        return (
            f"OptionsSignal({valid} {self.directional_lean} | "
            f"conf={self.confidence:.2f} | "
            f"pcr={self.pcr_value:.2f} | "
            f"skew={self.skew_value:+.4f} | "
            f"iv_pct={self.iv_percentile:.0f})"
        )


# Cached neutral result — returned whenever data is unavailable
_NEUTRAL = OptionsSignalResult(
    directional_lean = "neutral",
    confidence       = 0.0,
    pcr_value        = 1.0,
    skew_value       = 0.0,
    iv_percentile    = 50.0,
    signal_valid     = False,
    fallback_reason  = "no_data",
)


class OptionsSignal:
    """
    Extracts directional lean from NQ options chain data.

    Args:
        config: top-level Config dataclass. If None, uses spec defaults.
    """

    def __init__(self, config=None):
        if config is not None:
            self._min_volume     = config.signal.options_min_volume
            self._pcr_bullish    = config.signal.pcr_bullish_threshold
            self._pcr_bearish    = config.signal.pcr_bearish_threshold
        else:
            self._min_volume     = MIN_VOLUME_DEFAULT
            self._pcr_bullish    = PCR_BULLISH_THRESHOLD
            self._pcr_bearish    = PCR_BEARISH_THRESHOLD

        # Tracks historical ATM IV for percentile calculation
        self._iv_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_signal(
        self,
        calls_df: Optional[pd.DataFrame],
        puts_df:  Optional[pd.DataFrame],
        current_price: float,
        historical_iv_series: Optional[pd.Series] = None,
    ) -> OptionsSignalResult:
        """
        Generate a directional lean from the options chain.

        Expected DataFrame columns:
          [strike, bid, ask, last, impliedVolatility, volume, openInterest]

        Falls back to neutral (signal_valid=False) gracefully if:
          - calls_df or puts_df is None / empty
          - total options volume < min_volume
          - IV data has too many NaNs

        Args:
            calls_df:             Call options chain DataFrame (or None)
            puts_df:              Put options chain DataFrame (or None)
            current_price:        Current NQ price
            historical_iv_series: Optional Series of prior ATM IV values
                                  for percentile calculation.

        Returns:
            OptionsSignalResult
        """
        # --- Graceful fallback gates ---
        if calls_df is None or puts_df is None:
            return _neutral("no_data")

        if calls_df.empty or puts_df.empty:
            return _neutral("empty_chain")

        # Validate required columns
        required = {"strike", "impliedVolatility", "volume", "openInterest"}
        if not required.issubset(calls_df.columns) or not required.issubset(puts_df.columns):
            logger.warning("[options] Missing required columns — returning neutral.")
            return _neutral("missing_columns")

        # Liquidity gate
        total_vol = float(calls_df["volume"].sum() + puts_df["volume"].sum())
        if total_vol < self._min_volume:
            logger.warning(
                f"[options] Total volume {total_vol:.0f} < {self._min_volume} — returning neutral."
            )
            return _neutral("low_volume", total_volume=total_vol)

        # --- Compute signals ---
        pcr_composite, vol_pcr, oi_pcr = self.calculate_put_call_ratio(calls_df, puts_df)

        atm_strike = _find_atm_strike(current_price, calls_df)
        skew = self.calculate_iv_skew(calls_df, puts_df, atm_strike, WING_DISTANCE_DEFAULT)

        atm_iv = _get_atm_iv(calls_df, puts_df, atm_strike)
        iv_pct = self.get_iv_percentile(atm_iv, historical_iv_series)

        # --- Direction from PCR ---
        if pcr_composite <= self._pcr_bullish:
            pcr_lean: DirectionalLean = "long"
        elif pcr_composite >= self._pcr_bearish:
            pcr_lean = "short"
        else:
            pcr_lean = "neutral"

        # --- Direction from skew ---
        if skew < -0.01:
            skew_lean: DirectionalLean = "long"   # calls dearer → bullish
        elif skew > 0.01:
            skew_lean = "short"                    # puts dearer → bearish
        else:
            skew_lean = "neutral"

        # --- Combined lean: both must agree (or one is neutral) ---
        lean = _combine_lean(pcr_lean, skew_lean)

        # --- Confidence ---
        confidence = _compute_confidence(pcr_composite, skew, iv_pct)

        result = OptionsSignalResult(
            directional_lean = lean,
            confidence       = round(confidence, 4),
            pcr_value        = round(pcr_composite, 4),
            skew_value       = round(skew, 6),
            iv_percentile    = round(iv_pct, 1),
            signal_valid     = lean != "neutral" and confidence > 0.3,
            volume_pcr       = round(vol_pcr, 4),
            oi_pcr           = round(oi_pcr, 4),
            total_volume     = total_vol,
        )
        logger.debug(f"[options] {result}")
        return result

    # ------------------------------------------------------------------
    # Put/Call Ratio
    # ------------------------------------------------------------------

    def calculate_put_call_ratio(
        self,
        calls_df: pd.DataFrame,
        puts_df:  pd.DataFrame,
    ) -> tuple[float, float, float]:
        """
        Compute volume-based, OI-based, and composite PCR.

        Returns:
            (composite_pcr, volume_pcr, oi_pcr)

        PCR < 1 = more calls (bullish); PCR > 1 = more puts (bearish).
        """
        call_vol = float(calls_df["volume"].fillna(0).sum())
        put_vol  = float(puts_df["volume"].fillna(0).sum())
        call_oi  = float(calls_df["openInterest"].fillna(0).sum())
        put_oi   = float(puts_df["openInterest"].fillna(0).sum())

        vol_pcr = (put_vol / call_vol) if call_vol > 0 else 1.0
        oi_pcr  = (put_oi  / call_oi)  if call_oi  > 0 else 1.0

        # Composite: average of both (equal weight)
        composite = (vol_pcr + oi_pcr) / 2.0
        return composite, vol_pcr, oi_pcr

    # ------------------------------------------------------------------
    # IV Skew
    # ------------------------------------------------------------------

    def calculate_iv_skew(
        self,
        calls_df:      pd.DataFrame,
        puts_df:       pd.DataFrame,
        atm_strike:    float,
        wing_distance: float = WING_DISTANCE_DEFAULT,
    ) -> float:
        """
        Compute 25-delta equivalent skew as (downside_iv - upside_iv).

        downside_iv = put IV at (atm_strike - wing_distance)
        upside_iv   = call IV at (atm_strike + wing_distance)

        Positive skew → puts dearer → bearish institutional lean.
        Negative skew → calls dearer → bullish institutional lean.

        Returns 0.0 if strikes not found or IV is NaN.
        """
        downside_strike = atm_strike - wing_distance
        upside_strike   = atm_strike + wing_distance

        downside_iv = _get_iv_for_strike(puts_df,  downside_strike)
        upside_iv   = _get_iv_for_strike(calls_df, upside_strike)

        if downside_iv is None or upside_iv is None:
            logger.debug(
                f"[options] IV skew: could not find wing strikes "
                f"({downside_strike} put / {upside_strike} call) — returning 0."
            )
            return 0.0

        skew = downside_iv - upside_iv
        logger.debug(
            f"[options] IV skew: down_iv={downside_iv:.4f} "
            f"up_iv={upside_iv:.4f} skew={skew:+.4f}"
        )
        return skew

    # ------------------------------------------------------------------
    # IV Percentile
    # ------------------------------------------------------------------

    def get_iv_percentile(
        self,
        current_iv: float,
        historical_iv_series: Optional[pd.Series] = None,
    ) -> float:
        """
        Return the percentile rank of current_iv vs historical values.

        Uses historical_iv_series if provided; otherwise uses the internal
        _iv_history buffer (populated by successive calls to get_signal).

        Returns 50.0 if insufficient history (< 10 observations).
        """
        if historical_iv_series is not None and len(historical_iv_series) >= 10:
            hist = historical_iv_series.dropna().values
        elif len(self._iv_history) >= 10:
            hist = np.array(self._iv_history)
        else:
            return 50.0  # Insufficient history → assume median

        below = np.sum(hist < current_iv)
        pct = float(below / len(hist)) * 100.0
        return round(pct, 1)

    def record_iv(self, iv: float) -> None:
        """Append an IV observation to the internal history buffer."""
        if not np.isnan(iv) and iv > 0:
            self._iv_history.append(iv)
            # Keep only last 252 trading days
            if len(self._iv_history) > 252:
                self._iv_history.pop(0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _neutral(
    reason: str = "no_data",
    **kwargs,
) -> OptionsSignalResult:
    """Return a neutral result with signal_valid=False."""
    result = OptionsSignalResult(
        directional_lean = "neutral",
        confidence       = 0.0,
        pcr_value        = 1.0,
        skew_value       = 0.0,
        iv_percentile    = 50.0,
        signal_valid     = False,
        fallback_reason  = reason,
        **kwargs,
    )
    logger.debug(f"[options] Neutral fallback: {reason}")
    return result


def _find_atm_strike(current_price: float, calls_df: pd.DataFrame) -> float:
    """Return the strike in calls_df closest to current_price."""
    if calls_df.empty:
        return current_price
    strikes = calls_df["strike"].dropna().values
    if len(strikes) == 0:
        return current_price
    idx = int(np.argmin(np.abs(strikes - current_price)))
    return float(strikes[idx])


def _get_atm_iv(
    calls_df:   pd.DataFrame,
    puts_df:    pd.DataFrame,
    atm_strike: float,
) -> float:
    """
    Return average ATM IV (average of call IV and put IV at the ATM strike).
    Falls back to the median of all IVs if ATM strike not found.
    """
    call_iv = _get_iv_for_strike(calls_df, atm_strike)
    put_iv  = _get_iv_for_strike(puts_df,  atm_strike)

    values = [v for v in [call_iv, put_iv] if v is not None]
    if values:
        return float(np.mean(values))

    # Fallback: median of all available IVs
    all_ivs = pd.concat([
        calls_df["impliedVolatility"].dropna(),
        puts_df["impliedVolatility"].dropna(),
    ])
    valid = all_ivs[all_ivs > MIN_IV]
    return float(valid.median()) if not valid.empty else 0.20


def _get_iv_for_strike(
    chain_df: pd.DataFrame,
    target_strike: float,
    tolerance_pct: float = 0.005,
) -> Optional[float]:
    """
    Return the IV for the strike in chain_df closest to target_strike.
    Returns None if no strike is within tolerance_pct or if IV is NaN.
    """
    if chain_df.empty:
        return None

    strikes = chain_df["strike"].values
    diffs   = np.abs(strikes - target_strike)
    idx     = int(np.argmin(diffs))
    closest = float(strikes[idx])

    if abs(closest - target_strike) / target_strike > tolerance_pct:
        return None  # Nearest strike is too far away

    iv_val = float(chain_df.iloc[idx]["impliedVolatility"])
    if np.isnan(iv_val) or iv_val < MIN_IV:
        return None
    return iv_val


def _combine_lean(
    pcr_lean:  DirectionalLean,
    skew_lean: DirectionalLean,
) -> DirectionalLean:
    """
    Combine PCR and skew lean:
      - Both agree → use that direction
      - One is neutral → use the other
      - They disagree → neutral (conflicted)
    """
    if pcr_lean == skew_lean:
        return pcr_lean
    if pcr_lean == "neutral":
        return skew_lean
    if skew_lean == "neutral":
        return pcr_lean
    return "neutral"  # Disagreement → neutral


def _compute_confidence(
    pcr: float,
    skew: float,
    iv_percentile: float,
) -> float:
    """
    Compute composite confidence (0–1).

    Components:
      PCR score:  how far PCR is from neutral (0.7–1.3 range)
      Skew score: magnitude of IV skew (normalized)
      IV context: high IV percentile (>70) boosts confidence (move more likely)

    All components are bounded [0, 1] before averaging.
    """
    # PCR: neutral zone is 0.7–1.3, so max deviation from midpoint (1.0) is 0.3
    pcr_mid = (PCR_BULLISH_THRESHOLD + PCR_BEARISH_THRESHOLD) / 2  # 1.0
    pcr_range = (PCR_BEARISH_THRESHOLD - PCR_BULLISH_THRESHOLD) / 2  # 0.3
    pcr_score = min(abs(pcr - pcr_mid) / pcr_range, 1.0)

    # Skew: ±5% IV differential is considered max
    skew_score = min(abs(skew) / 0.05, 1.0)

    # IV context: high IV percentile (market moving) gives more signal reliability
    iv_score = min(iv_percentile / 100.0, 1.0)

    # Weighted composite
    confidence = 0.45 * pcr_score + 0.45 * skew_score + 0.10 * iv_score
    return float(np.clip(confidence, 0.0, 1.0))
