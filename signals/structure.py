"""
signals/structure.py — Market Structure Detector (Signal Layer 4)

Identifies two types of significant price levels:

  OrderBlock  — the last opposing candle before a 3+ ATR impulse move.
                Bullish OB: last bearish candle before a strong up impulse.
                Bearish OB: last bullish candle before a strong down impulse.
                These are institutional demand/supply zones.

  KeyLevel    — swing highs and swing lows from the lookback window.
                Levels with multiple touches are stronger.

The structure signal answers three questions:
  1. Is price currently at a significant level?
  2. How far is price from the nearest level?
  3. What is the invalidation price (stop loss) for the current direction?

Order blocks are NOT just any high/low — only those preceded by a confirmed
impulse move of 3+ ATR qualify. This keeps the level count meaningful.

Usage:
    from signals.structure import StructureDetector
    det = StructureDetector(config)
    obs  = det.find_order_blocks(daily_df, lookback=20)
    keys = det.find_key_levels(daily_df)
    sig  = det.get_structure_signal(current_price, "long", obs, keys)
    if sig.at_structure:
        stop = sig.invalidation_level
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Impulse threshold: move must be >= this many ATRs to qualify
IMPULSE_ATR_THRESHOLD = 3.0
# How many bars forward to measure the impulse from the candidate bar
IMPULSE_LOOKAHEAD = 5
# How many bars on each side to require for a swing pivot
SWING_LOOKBACK_BARS = 3
# Strength floor / ceiling for normalization
STRENGTH_MIN = 0.20
STRENGTH_MAX = 1.00


@dataclass
class OrderBlock:
    """A price level where institutional activity originated a strong impulse."""
    price_level:    float               # Midpoint of the OB candle: (open+close)/2
    high:           float               # High of the OB candle (used for stop loss)
    low:            float               # Low of the OB candle  (used for stop loss)
    direction:      Literal["bullish", "bearish"]
    strength:       float               # 0–1, proportional to impulse size
    created_date:   Optional[date] = None
    impulse_atr:    float = 0.0         # Impulse size in ATR units (for logging)


@dataclass
class KeyLevel:
    """A swing high or swing low with measured touch count."""
    price:      float
    level_type: Literal["support", "resistance"]
    touches:    int     = 1
    strength:   float   = 0.5


@dataclass
class StructureSignal:
    """Output of StructureDetector.get_structure_signal()."""
    at_structure:       bool    # True if price is within proximity_pct of any level
    nearest_level_pct:  float   # Signed distance to nearest level: (price-level)/level
    invalidation_level: float   # Stop loss price for the given direction
    nearest_level_price: float  = 0.0  # Actual price of nearest level (for logging)
    nearest_level_type:  str    = ""   # Type label for logging


class StructureDetector:
    """
    Detects order blocks and key levels on daily OHLCV data.

    Args:
        config: top-level Config dataclass. If None, uses spec defaults.
    """

    def __init__(self, config=None):
        if config is not None:
            self._proximity_pct = config.signal.structure_proximity_pct
        else:
            self._proximity_pct = 0.003   # 0.3% of price

    # ------------------------------------------------------------------
    # Order blocks
    # ------------------------------------------------------------------

    def find_order_blocks(
        self,
        daily_bars_df: pd.DataFrame,
        lookback: int = 20,
    ) -> list[OrderBlock]:
        """
        Find significant order blocks in the last `lookback` bars.

        Algorithm:
          For each bar i in the lookback window:
            1. Look ahead IMPULSE_LOOKAHEAD bars and measure the max move.
            2. If that move is >= IMPULSE_ATR_THRESHOLD × ATR → impulse detected.
            3. Walk back from i to find the last candle opposing the impulse.
            4. That candle is the order block. Record its price, high, low, strength.

        Order blocks are NOT just highs/lows — the impulse requirement filters
        out noise and retests. Only institutional-quality moves qualify.

        Returns:
            List of OrderBlock, deduplicated and sorted by strength descending.
        """
        df = daily_bars_df.copy()
        atr = _atr_series(df, period=14)
        obs: list[OrderBlock] = []

        # Work within the lookback window but need ATR context from full df
        start_idx = max(14, len(df) - lookback - IMPULSE_LOOKAHEAD)

        for i in range(start_idx, len(df) - IMPULSE_LOOKAHEAD):
            atr_val = float(atr.iloc[i])
            if atr_val <= 0:
                continue

            # Measure the impulse starting from bar i+1
            lookahead = df.iloc[i + 1 : i + 1 + IMPULSE_LOOKAHEAD]
            anchor_close = float(df["close"].iloc[i])

            up_move   = (lookahead["high"].max() - anchor_close) / atr_val
            down_move = (anchor_close - lookahead["low"].min())  / atr_val

            if up_move >= IMPULSE_ATR_THRESHOLD and up_move >= down_move:
                # Bullish impulse → look back for last bearish (red) candle
                ob_idx = _last_opposing_candle(df, i, impulse_direction="up")
                if ob_idx is None:
                    continue
                row = df.iloc[ob_idx]
                strength = _normalise_strength(up_move)
                created  = row.name.date() if hasattr(row.name, "date") else None
                obs.append(OrderBlock(
                    price_level  = float((row["open"] + row["close"]) / 2),
                    high         = float(row["high"]),
                    low          = float(row["low"]),
                    direction    = "bullish",
                    strength     = strength,
                    created_date = created,
                    impulse_atr  = round(up_move, 2),
                ))

            elif down_move >= IMPULSE_ATR_THRESHOLD and down_move > up_move:
                # Bearish impulse → look back for last bullish (green) candle
                ob_idx = _last_opposing_candle(df, i, impulse_direction="down")
                if ob_idx is None:
                    continue
                row = df.iloc[ob_idx]
                strength = _normalise_strength(down_move)
                created  = row.name.date() if hasattr(row.name, "date") else None
                obs.append(OrderBlock(
                    price_level  = float((row["open"] + row["close"]) / 2),
                    high         = float(row["high"]),
                    low          = float(row["low"]),
                    direction    = "bearish",
                    strength     = strength,
                    created_date = created,
                    impulse_atr  = round(down_move, 2),
                ))

        # Deduplicate: merge OBs whose price_levels are within 0.1% of each other
        obs = _dedup_order_blocks(obs, merge_pct=0.001)
        obs.sort(key=lambda x: x.strength, reverse=True)

        logger.debug(
            f"[structure] find_order_blocks: {len(obs)} OBs found | "
            f"bullish={sum(1 for o in obs if o.direction=='bullish')} "
            f"bearish={sum(1 for o in obs if o.direction=='bearish')}"
        )
        return obs

    # ------------------------------------------------------------------
    # Key levels (swing highs / lows)
    # ------------------------------------------------------------------

    def find_key_levels(
        self,
        daily_bars_df: pd.DataFrame,
        lookback: int = 20,
        swing_bars: int = SWING_LOOKBACK_BARS,
    ) -> list[KeyLevel]:
        """
        Identify swing highs and swing lows from the last `lookback` bars.

        A swing high: bar whose high is the highest within swing_bars on each side.
        A swing low:  bar whose low is the lowest within swing_bars on each side.

        Strength is based on:
          - Recency (more recent = stronger)
          - Touch count (price returns to the level)

        Returns:
            List of KeyLevel sorted by strength descending.
        """
        df = daily_bars_df.tail(lookback + swing_bars * 2).copy()
        levels: list[KeyLevel] = []
        n = len(df)

        for i in range(swing_bars, n - swing_bars):
            left_high  = df["high"].iloc[i - swing_bars : i].max()
            right_high = df["high"].iloc[i + 1 : i + swing_bars + 1].max()
            left_low   = df["low"].iloc[i - swing_bars : i].min()
            right_low  = df["low"].iloc[i + 1 : i + swing_bars + 1].min()

            if df["high"].iloc[i] > left_high and df["high"].iloc[i] > right_high:
                price    = float(df["high"].iloc[i])
                recency  = (i - swing_bars) / max(n - 2 * swing_bars, 1)
                touches  = _count_touches(df["high"], price, proximity_pct=0.002)
                strength = _level_strength(recency, touches)
                levels.append(KeyLevel(
                    price      = price,
                    level_type = "resistance",
                    touches    = touches,
                    strength   = strength,
                ))

            if df["low"].iloc[i] < left_low and df["low"].iloc[i] < right_low:
                price    = float(df["low"].iloc[i])
                recency  = (i - swing_bars) / max(n - 2 * swing_bars, 1)
                touches  = _count_touches(df["low"], price, proximity_pct=0.002)
                strength = _level_strength(recency, touches)
                levels.append(KeyLevel(
                    price      = price,
                    level_type = "support",
                    touches    = touches,
                    strength   = strength,
                ))

        # Deduplicate levels within 0.2% of each other
        levels = _dedup_key_levels(levels, merge_pct=0.002)
        levels.sort(key=lambda x: x.strength, reverse=True)

        logger.debug(
            f"[structure] find_key_levels: {len(levels)} levels | "
            f"support={sum(1 for l in levels if l.level_type=='support')} "
            f"resistance={sum(1 for l in levels if l.level_type=='resistance')}"
        )
        return levels

    # ------------------------------------------------------------------
    # Level proximity queries
    # ------------------------------------------------------------------

    def get_nearest_levels(
        self,
        current_price: float,
        order_blocks:  list[OrderBlock],
        key_levels:    list[KeyLevel],
        proximity_pct: float = 0.003,
    ) -> list[dict]:
        """
        Return all levels within proximity_pct of current_price.

        Returns a list of dicts with keys:
          price, level_type, strength, invalidation_price, distance_pct
        sorted by absolute distance ascending.
        """
        results = []

        for ob in order_blocks:
            dist = (current_price - ob.price_level) / ob.price_level
            if abs(dist) <= proximity_pct:
                inv = ob.low  if ob.direction == "bullish" else ob.high
                results.append({
                    "price":              ob.price_level,
                    "level_type":         f"order_block_{ob.direction}",
                    "strength":           ob.strength,
                    "invalidation_price": inv,
                    "distance_pct":       dist,
                })

        for kl in key_levels:
            dist = (current_price - kl.price) / kl.price
            if abs(dist) <= proximity_pct:
                # Invalidation: slightly beyond the level
                buf = kl.price * 0.0005
                inv = (kl.price - buf) if kl.level_type == "support" else (kl.price + buf)
                results.append({
                    "price":              kl.price,
                    "level_type":         kl.level_type,
                    "strength":           kl.strength,
                    "invalidation_price": inv,
                    "distance_pct":       dist,
                })

        results.sort(key=lambda x: abs(x["distance_pct"]))
        return results

    def is_at_structure(
        self,
        current_price: float,
        levels: list[dict],
        proximity_pct: float = 0.003,
    ) -> bool:
        """
        Return True if price is within proximity_pct of any level in levels.
        levels should be the output of get_nearest_levels().
        """
        return any(abs(lv["distance_pct"]) <= proximity_pct for lv in levels)

    # ------------------------------------------------------------------
    # Structure signal (used by signal combiner)
    # ------------------------------------------------------------------

    def get_structure_signal(
        self,
        current_price: float,
        direction:     Literal["long", "short", "flat"],
        order_blocks:  list[OrderBlock],
        key_levels:    list[KeyLevel],
    ) -> StructureSignal:
        """
        Determine if price is at a significant level and compute the stop.

        Invalidation level (stop loss price) selection:
          LONG:  nearest bullish OB's low  OR nearest support level
          SHORT: nearest bearish OB's high OR nearest resistance level
          Priority: order blocks > key levels (OBs have harder invalidation)

        Returns:
            StructureSignal with at_structure, nearest_level_pct,
            invalidation_level, and level metadata for logging.
        """
        nearby = self.get_nearest_levels(
            current_price, order_blocks, key_levels, self._proximity_pct
        )

        # Nearest level (for distance reporting)
        if nearby:
            nearest = nearby[0]
            nearest_dist_pct = nearest["distance_pct"]
            nearest_price    = nearest["price"]
            nearest_type     = nearest["level_type"]
        else:
            # Find absolute nearest even if outside proximity
            all_levels = self.get_nearest_levels(
                current_price, order_blocks, key_levels, proximity_pct=0.05
            )
            if all_levels:
                nearest      = all_levels[0]
                nearest_dist_pct = nearest["distance_pct"]
                nearest_price    = nearest["price"]
                nearest_type     = nearest["level_type"]
            else:
                nearest_dist_pct = 0.0
                nearest_price    = current_price
                nearest_type     = "none"

        at_structure = bool(nearby)

        # Invalidation level
        invalidation = _select_invalidation(
            current_price, direction, nearby, order_blocks, key_levels
        )

        result = StructureSignal(
            at_structure        = at_structure,
            nearest_level_pct   = round(nearest_dist_pct, 6),
            invalidation_level  = round(invalidation, 2),
            nearest_level_price = round(nearest_price, 2),
            nearest_level_type  = nearest_type,
        )
        logger.debug(
            f"[structure] price={current_price:.2f} | "
            f"at_struct={at_structure} | "
            f"nearest={nearest_type} @ {nearest_price:.2f} "
            f"({nearest_dist_pct:+.3%}) | "
            f"invalidation={invalidation:.2f}"
        )
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _last_opposing_candle(
    df: pd.DataFrame,
    i: int,
    impulse_direction: Literal["up", "down"],
    lookback: int = 4,
) -> Optional[int]:
    """
    Walk back from bar i to find the last candle opposing the impulse.
    'up' impulse → look for last bearish (close < open) candle.
    'down' impulse → look for last bullish (close > open) candle.
    Returns the index position (int) or None.
    """
    for j in range(i, max(i - lookback, -1), -1):
        is_bearish = df["close"].iloc[j] < df["open"].iloc[j]
        is_bullish = df["close"].iloc[j] > df["open"].iloc[j]
        if impulse_direction == "up" and is_bearish:
            return j
        if impulse_direction == "down" and is_bullish:
            return j
    return None


def _normalise_strength(impulse_atr: float) -> float:
    """
    Map impulse size (in ATR multiples) to a 0–1 strength score.
    3 ATR = 0.33, 6 ATR = 0.67, 9+ ATR = 1.0
    """
    raw = (impulse_atr - IMPULSE_ATR_THRESHOLD) / (IMPULSE_ATR_THRESHOLD * 2) + 0.33
    return round(float(np.clip(raw, STRENGTH_MIN, STRENGTH_MAX)), 3)


def _count_touches(series: pd.Series, price: float, proximity_pct: float = 0.002) -> int:
    """Count how many bars in `series` came within proximity_pct of price."""
    diff = (series - price).abs() / price
    return int((diff <= proximity_pct).sum())


def _level_strength(recency: float, touches: int) -> float:
    """
    Compute key level strength.
    recency: 0 = oldest, 1 = most recent
    touches: number of times price revisited the level
    """
    touch_score   = min(touches / 5.0, 1.0)         # 5+ touches = max
    recency_score = recency
    raw = 0.4 * recency_score + 0.6 * touch_score
    return round(float(np.clip(raw, STRENGTH_MIN, STRENGTH_MAX)), 3)


def _dedup_order_blocks(
    obs: list[OrderBlock], merge_pct: float = 0.001
) -> list[OrderBlock]:
    """Merge order blocks whose price_levels are within merge_pct of each other."""
    if not obs:
        return obs
    merged: list[OrderBlock] = []
    for ob in obs:
        # Check if a similar OB already exists
        duplicate = False
        for existing in merged:
            if (
                existing.direction == ob.direction
                and abs(existing.price_level - ob.price_level) / existing.price_level
                <= merge_pct
            ):
                # Keep the stronger one
                if ob.strength > existing.strength:
                    merged.remove(existing)
                    merged.append(ob)
                duplicate = True
                break
        if not duplicate:
            merged.append(ob)
    return merged


def _dedup_key_levels(
    levels: list[KeyLevel], merge_pct: float = 0.002
) -> list[KeyLevel]:
    """Merge key levels within merge_pct of each other."""
    if not levels:
        return levels
    merged: list[KeyLevel] = []
    for lv in levels:
        duplicate = False
        for existing in merged:
            if (
                existing.level_type == lv.level_type
                and abs(existing.price - lv.price) / existing.price <= merge_pct
            ):
                if lv.strength > existing.strength:
                    merged.remove(existing)
                    merged.append(lv)
                duplicate = True
                break
        if not duplicate:
            merged.append(lv)
    return merged


def _select_invalidation(
    current_price: float,
    direction: str,
    nearby: list[dict],
    order_blocks: list[OrderBlock],
    key_levels: list[KeyLevel],
) -> float:
    """
    Choose the stop loss (invalidation) price for the given direction.

    Priority:
      1. Nearby order block with matching direction alignment
      2. Nearby key level
      3. Fallback: order block in wider search
      4. Last resort: ATR-based default (1.5% of price)
    """
    # Try nearby levels first (already sorted by proximity)
    for lv in nearby:
        if direction == "long" and "bullish" in lv["level_type"]:
            return lv["invalidation_price"]
        if direction == "short" and "bearish" in lv["level_type"]:
            return lv["invalidation_price"]

    for lv in nearby:
        if direction == "long" and lv["level_type"] == "support":
            return lv["invalidation_price"]
        if direction == "short" and lv["level_type"] == "resistance":
            return lv["invalidation_price"]

    # Any nearby level
    if nearby:
        return nearby[0]["invalidation_price"]

    # Wider search: nearest OB in the correct direction
    for ob in sorted(order_blocks, key=lambda o: abs(o.price_level - current_price)):
        if direction == "long" and ob.direction == "bullish":
            return ob.low
        if direction == "short" and ob.direction == "bearish":
            return ob.high

    # Nearest key level below (long) or above (short)
    if direction == "long":
        supports = [kl for kl in key_levels if kl.price < current_price]
        if supports:
            nearest_sup = max(supports, key=lambda k: k.price)
            return nearest_sup.price * (1 - 0.0005)
    elif direction == "short":
        resistances = [kl for kl in key_levels if kl.price > current_price]
        if resistances:
            nearest_res = min(resistances, key=lambda k: k.price)
            return nearest_res.price * (1 + 0.0005)

    # Last resort: 1.5% beyond current price
    default_offset = current_price * 0.015
    return (current_price - default_offset) if direction == "long" else (current_price + default_offset)
