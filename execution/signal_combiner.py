"""
execution/signal_combiner.py — Signal Combiner (Layer 5 Gate)

Merges all five signal layers into a single trade decision.
This is the final gate before any capital is deployed.

Agreement scoring (max 5 points):
  +1  Momentum signal is valid AND direction matches
  +1  Price is at a structural level (order block or key level)
  +1  Options signal is valid AND direction matches
  +1  Macro directional bias matches direction
  +1  Regime is "breakout" (bonus conviction — breakout adds force)

Hard blocks (prevent any trade regardless of score):
  - Regime is "ranging" or "volatile"
  - Regime confidence < min_regime_confidence (default 0.65)
  - Momentum direction is "flat" (no intraday signal)
  - Stop distance is zero or negative (invalid structure)

should_trade = True only if score >= min_signal_agreement (default 3).

Usage:
    from execution.signal_combiner import SignalCombiner
    combiner = SignalCombiner(config)
    decision = combiner.combine(
        regime_result, macro_score, momentum_signal,
        structure_signal, options_signal,
        current_price=24300.0,
    )
    if decision.should_trade:
        execute(decision.direction, decision.stop_price, decision.target_price)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd
from loguru import logger

# Imported types (for type hints only — no circular dependency)
# These come from the respective signal modules at runtime.


NON_TRADEABLE_REGIMES = {"ranging", "volatile"}
RR_RATIO_DEFAULT = 2.0


@dataclass
class TradeDecision:
    """Output of SignalCombiner.combine()."""
    should_trade:     bool
    direction:        Literal["long", "short", "flat"]
    entry_price:      float
    stop_price:       float
    target_price:     float
    conviction_score: int      # 0–5 agreement points
    reason:           str      # Human-readable explanation for logging

    # Per-layer agreement flags (for dashboard / debugging)
    regime_ok:      bool  = False
    momentum_ok:    bool  = False
    structure_ok:   bool  = False
    options_ok:     bool  = False
    macro_ok:       bool  = False
    breakout_bonus: bool  = False

    def __str__(self) -> str:
        if not self.should_trade:
            return f"TradeDecision(NO TRADE | {self.reason})"
        return (
            f"TradeDecision({self.direction.upper()} | "
            f"score={self.conviction_score}/5 | "
            f"entry={self.entry_price:.2f} "
            f"stop={self.stop_price:.2f} "
            f"target={self.target_price:.2f} | "
            f"{self.reason})"
        )

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def reward_distance(self) -> float:
        return abs(self.target_price - self.entry_price)

    @property
    def rr_ratio(self) -> float:
        if self.stop_distance == 0:
            return 0.0
        return self.reward_distance / self.stop_distance


class SignalCombiner:
    """
    Combines all five signal layers into a single TradeDecision.

    Args:
        config: top-level Config dataclass. If None, uses spec defaults.
    """

    def __init__(self, config=None):
        if config is not None:
            self._min_confidence = config.signal.min_regime_confidence
            self._min_agreement  = config.signal.min_signal_agreement
            self._rr_ratio       = config.signal.rr_ratio_min
        else:
            self._min_confidence = 0.65
            self._min_agreement  = 3
            self._rr_ratio       = RR_RATIO_DEFAULT

    # ------------------------------------------------------------------
    # Main combiner
    # ------------------------------------------------------------------

    def combine(
        self,
        regime_result:    pd.DataFrame,
        macro_score,                    # MacroScore dataclass
        momentum_signal,                # SignalResult dataclass
        structure_signal,               # StructureSignal dataclass
        options_signal,                 # OptionsSignalResult dataclass
        current_price:    float = 0.0,
    ) -> TradeDecision:
        """
        Merge all five signal layers into a single TradeDecision.

        Args:
            regime_result:    DataFrame from HMMRegimeDetector.predict()
                              (uses last row).
            macro_score:      MacroScore from MacroRegimeScorer.score().
            momentum_signal:  SignalResult from MomentumSignal.get_signal().
            structure_signal: StructureSignal from StructureDetector.get_structure_signal().
            options_signal:   OptionsSignalResult from OptionsSignal.get_signal().
            current_price:    Current NQ price (used for entry / fallback stop).

        Returns:
            TradeDecision dataclass.
        """
        # ── Extract latest regime state ──────────────────────────────────
        last_regime = regime_result.iloc[-1]
        regime      = str(last_regime["regime"])
        confidence  = float(last_regime["regime_confidence"])

        # ── Hard block 1: non-tradeable regime ───────────────────────────
        if regime in NON_TRADEABLE_REGIMES:
            return _no_trade(
                f"Regime={regime} (non-tradeable; only trending/breakout allowed)",
                current_price,
            )

        # ── Hard block 2: low confidence ─────────────────────────────────
        if confidence < self._min_confidence:
            return _no_trade(
                f"Regime confidence {confidence:.1%} < {self._min_confidence:.0%} threshold",
                current_price,
            )

        # ── Hard block 3: no intraday momentum direction ──────────────────
        direction = momentum_signal.direction
        if direction == "flat":
            return _no_trade(
                f"Momentum flat (near VWAP or opposing signals; "
                f"vwap_dist={momentum_signal.vwap_distance_pct:+.3%} "
                f"mom={momentum_signal.momentum_value:+.3f})",
                current_price,
            )

        # ── Agreement scoring ─────────────────────────────────────────────
        score = 0
        flags = {}

        # +1 Momentum: valid signal that agrees with direction
        mom_ok = momentum_signal.signal_valid and momentum_signal.direction == direction
        if mom_ok:
            score += 1
        flags["momentum_ok"] = mom_ok

        # +1 Structure: price at or near a significant level
        struct_ok = structure_signal.at_structure
        if struct_ok:
            score += 1
        flags["structure_ok"] = struct_ok

        # +1 Options: valid signal that agrees with direction
        opts_ok = (
            options_signal.signal_valid
            and options_signal.directional_lean == direction
        )
        if opts_ok:
            score += 1
        flags["options_ok"] = opts_ok

        # +1 Macro: directional bias matches (neutral counts as not matching)
        macro_ok = macro_score.directional_bias == direction
        if macro_ok:
            score += 1
        flags["macro_ok"] = macro_ok

        # +1 Breakout bonus (regime conviction boost)
        breakout = regime == "breakout"
        if breakout:
            score += 1
        flags["breakout_bonus"] = breakout

        # ── Hard block 4: insufficient agreement ─────────────────────────
        if score < self._min_agreement:
            layers = _format_layers(flags, direction, momentum_signal,
                                    macro_score, options_signal)
            return _no_trade(
                f"Score {score}/{self._min_agreement} — insufficient agreement. "
                f"{layers}",
                current_price,
            )

        # ── Entry / stop / target ─────────────────────────────────────────
        entry = current_price if current_price > 0 else float(
            last_regime.name.timestamp() if hasattr(last_regime.name, "timestamp") else 0
        )

        stop = structure_signal.invalidation_level

        # Hard block 5: degenerate stop
        stop_dist = abs(entry - stop)
        if stop_dist <= 0 or entry <= 0:
            return _no_trade(
                f"Degenerate stop: entry={entry:.2f} stop={stop:.2f}",
                current_price,
            )

        if direction == "long":
            target = entry + stop_dist * self._rr_ratio
        else:
            target = entry - stop_dist * self._rr_ratio

        # ── Build reason string ──────────────────────────────────────────
        reason = _build_reason(
            direction, regime, confidence, score, flags,
            momentum_signal, macro_score, options_signal,
            structure_signal, entry, stop, target, self._rr_ratio,
        )

        decision = TradeDecision(
            should_trade     = True,
            direction        = direction,
            entry_price      = round(entry, 2),
            stop_price       = round(stop, 2),
            target_price     = round(target, 2),
            conviction_score = score,
            reason           = reason,
            regime_ok        = True,
            momentum_ok      = flags["momentum_ok"],
            structure_ok     = flags["structure_ok"],
            options_ok       = flags["options_ok"],
            macro_ok         = flags["macro_ok"],
            breakout_bonus   = flags["breakout_bonus"],
        )
        logger.info(f"[combiner] {decision}")
        return decision

    # ------------------------------------------------------------------
    # Partial exit helper
    # ------------------------------------------------------------------

    def get_partial_exit_price(
        self,
        entry:    float,
        stop:     float,
        rr_ratio: float = 1.0,
    ) -> float:
        """
        Return the price for a partial position exit at rr_ratio R:R.

        Default rr_ratio=1.0 gives a 1:1 partial exit (first scale-out).
        Used to lock in partial profits and move stop to break-even.

        Args:
            entry:    Trade entry price.
            stop:     Initial stop price.
            rr_ratio: R:R multiple for the partial exit (default 1.0 = 1:1).

        Returns:
            Partial exit price.
        """
        risk = abs(entry - stop)
        if risk <= 0:
            return entry
        # Direction inferred from stop position
        if stop < entry:   # long
            return entry + risk * rr_ratio
        else:              # short
            return entry - risk * rr_ratio


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _no_trade(reason: str, current_price: float) -> TradeDecision:
    """Return a no-trade decision with explanation."""
    logger.debug(f"[combiner] NO TRADE: {reason}")
    return TradeDecision(
        should_trade     = False,
        direction        = "flat",
        entry_price      = current_price,
        stop_price       = 0.0,
        target_price     = 0.0,
        conviction_score = 0,
        reason           = reason,
    )


def _format_layers(
    flags:          dict,
    direction:      str,
    momentum_signal,
    macro_score,
    options_signal,
) -> str:
    """Build a concise per-layer status string for the reason field."""
    parts = []
    parts.append(f"mom={'Y' if flags['momentum_ok'] else 'N'}({momentum_signal.strength:.2f})")
    parts.append(f"struct={'Y' if flags['structure_ok'] else 'N'}")
    parts.append(f"opts={'Y' if flags['options_ok'] else 'N'}")
    parts.append(f"macro={'Y' if flags['macro_ok'] else 'N'}({macro_score.directional_bias})")
    if flags["breakout_bonus"]:
        parts.append("breakout=Y")
    return " | ".join(parts)


def _build_reason(
    direction:       str,
    regime:          str,
    confidence:      float,
    score:           int,
    flags:           dict,
    momentum_signal,
    macro_score,
    options_signal,
    structure_signal,
    entry:           float,
    stop:            float,
    target:          float,
    rr_ratio:        float,
) -> str:
    """Build a detailed reason string covering every factor."""
    parts = [
        f"TRADE {direction.upper()} | score={score}/5",
        f"regime={regime}({confidence:.0%})",
        f"mom={'✓' if flags['momentum_ok'] else '✗'}"
        f"(str={momentum_signal.strength:.2f} "
        f"vwap={momentum_signal.vwap_distance_pct:+.3%})",
        f"struct={'✓' if flags['structure_ok'] else '✗'}"
        f"({structure_signal.nearest_level_pct:+.3%})",
        f"opts={'✓' if flags['options_ok'] else '✗'}"
        f"(pcr={options_signal.pcr_value:.2f} "
        f"skew={options_signal.skew_value:+.3f})" if options_signal.signal_valid
        else f"opts=✗(no data)",
        f"macro={'✓' if flags['macro_ok'] else '✗'}"
        f"({macro_score.directional_bias}/{macro_score.dominant_regime})",
        f"entry={entry:.2f} stop={stop:.2f} tgt={target:.2f} R={rr_ratio:.1f}:1",
    ]
    if flags["breakout_bonus"]:
        parts.append("breakout_bonus=✓")
    return " | ".join(parts)
