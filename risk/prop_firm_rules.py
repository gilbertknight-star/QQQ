"""
risk/prop_firm_rules.py — Prop Firm Rule Enforcement

The most critical safety component. Enforces all Topstep / Apex constraints
before any order is placed. If ANY check fails, the bot must not trade.

Rules enforced:

  1. Daily loss limit
       current_daily_loss >= daily_loss_limit → HALT (stop trading for the day)

  2. Trailing drawdown (two flavors):
       "eod"      — high-water mark advances only at end of day (Topstep default)
       "intraday" — high-water mark advances tick-by-tick (Apex aggressive)
       current_balance <= (high_water_mark - max_drawdown) → HALT IMMEDIATELY

  3. Profit target reached
       current_balance >= (starting_balance + profit_target) → STOP (pass phase)

  4. Consistency rule (if enabled, e.g. Apex 50%)
       No single day's profit may exceed `consistency_rule` × total_net_profit.
       If today's profit would exceed that cap → reduce size or skip trade.

  5. News / time blackout (optional — delegated to NewsFilter)
       PropFirmRules itself only checks trading-hours gate via the config.

  6. Max position size gate
       Trade size may not exceed config.position.max_contracts.

State is held in PropFirmState (a mutable dataclass). The caller must update
state after every fill and at every EOD settlement.

Usage:
    from risk.prop_firm_rules import PropFirmRules, PropFirmState

    state = PropFirmState(
        starting_balance = 100_000.0,
        current_balance  = 100_000.0,
        high_water_mark  = 100_000.0,
        daily_pnl        = 0.0,
        total_net_profit = 0.0,
    )
    rules = PropFirmRules(config)

    check = rules.check(state, proposed_contracts=1, stop_distance_points=12.5)
    if not check.can_trade:
        logger.warning(check.reason)
        return  # abort order

    # After fill and trade close:
    state = rules.update_after_trade(state, trade_pnl=250.0, is_eod=False)

    # At end of day:
    state = rules.end_of_day(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

HaltReason = Literal[
    "daily_loss_limit",
    "trailing_drawdown",
    "profit_target_reached",
    "consistency_rule",
    "max_contracts",
    "ok",
]

NQ_POINT_VALUE = 20.0


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class PropFirmState:
    """
    Mutable snapshot of account state used by PropFirmRules.

    The caller is responsible for keeping this up to date:
      - after every trade closes: call update_after_trade()
      - at end of every trading day: call end_of_day()
    """
    starting_balance:  float          # Balance at start of the eval phase
    current_balance:   float          # Current mark-to-market equity
    high_water_mark:   float          # Highest balance ever reached (for trailing DD)
    daily_pnl:         float          # Net P&L for today (resets at EOD)
    total_net_profit:  float          # Cumulative net profit since phase start

    # Derived / convenience
    is_halted:         bool  = False  # True if a hard rule has triggered
    halt_reason:       str   = ""     # Human-readable reason for halt
    phase_passed:      bool  = False  # True if profit target was reached
    trades_today:      int   = 0      # Trade count for the current session


@dataclass
class RuleCheckResult:
    """Output of PropFirmRules.check()."""
    can_trade:  bool
    reason:     HaltReason
    detail:     str           # Human-readable detail for logging
    safe_contracts: int       # Adjusted contract count (may be reduced for consistency)

    def __str__(self) -> str:
        status = "OK" if self.can_trade else f"BLOCKED({self.reason})"
        return f"RuleCheck({status} | {self.detail} | safe_contracts={self.safe_contracts})"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PropFirmRules:
    """
    Enforces prop firm trading rules before every order.

    Args:
        config: top-level Config dataclass. If None, uses Topstep 100k defaults.
    """

    def __init__(self, config=None):
        if config is not None:
            pf = config.prop_firm
            self._daily_loss_limit     = pf.daily_loss_limit
            self._max_drawdown         = pf.max_drawdown
            self._profit_target        = pf.profit_target
            self._consistency_rule     = pf.consistency_rule    # e.g. 0.40 or 0.50
            self._trailing_type        = pf.trailing_drawdown_type  # "eod" | "intraday"
            self._max_contracts        = config.position.max_contracts
            self._point_value          = config.position.nq_point_value
        else:
            # Topstep 100k defaults
            self._daily_loss_limit     = 2_000.0
            self._max_drawdown         = 3_000.0
            self._profit_target        = 6_000.0
            self._consistency_rule     = 0.40
            self._trailing_type        = "eod"
            self._max_contracts        = 3
            self._point_value          = NQ_POINT_VALUE

    # ------------------------------------------------------------------
    # Primary gate — call before every order
    # ------------------------------------------------------------------

    def check(
        self,
        state:                 PropFirmState,
        proposed_contracts:    int   = 1,
        stop_distance_points:  float = 0.0,
        point_value:           float | None = None,
    ) -> RuleCheckResult:
        """
        Run all rule checks for the proposed trade.

        Args:
            state:                Current PropFirmState.
            proposed_contracts:   Number of contracts about to be placed.
            stop_distance_points: Stop distance in points (for worst-case calc).
            point_value:          Dollar value per point. Defaults to NQ ($20).
                                  Pass MNQ_POINT_VALUE ($2) when trading MNQ.

        Returns:
            RuleCheckResult. Trade must be aborted if can_trade is False.
        """
        pv = point_value if point_value is not None else self._point_value
        # ── Already halted ───────────────────────────────────────────
        if state.is_halted:
            return _blocked("daily_loss_limit", f"Account already halted: {state.halt_reason}", 0)

        # ── Phase already passed ─────────────────────────────────────
        if state.phase_passed:
            return _blocked(
                "profit_target_reached",
                f"Profit target ${self._profit_target:,.0f} already reached — stop trading",
                0,
            )

        # ── Rule 1: Daily loss limit ──────────────────────────────────
        daily_loss = -min(state.daily_pnl, 0.0)   # positive = how much lost today
        if daily_loss >= self._daily_loss_limit:
            detail = (
                f"Daily loss ${daily_loss:,.2f} >= limit ${self._daily_loss_limit:,.2f}"
            )
            return _blocked("daily_loss_limit", detail, 0)

        # ── Rule 2: Trailing drawdown ─────────────────────────────────
        drawdown_floor = state.high_water_mark - self._max_drawdown
        if state.current_balance <= drawdown_floor:
            detail = (
                f"Balance ${state.current_balance:,.2f} <= "
                f"drawdown floor ${drawdown_floor:,.2f} "
                f"(HWM=${state.high_water_mark:,.2f} - ${self._max_drawdown:,.2f})"
            )
            return _blocked("trailing_drawdown", detail, 0)

        # ── Rule 3: Would next loss breach daily limit? ───────────────
        # Pre-check: if this trade hits its stop, will that breach the daily limit?
        worst_case_loss = proposed_contracts * stop_distance_points * pv
        if stop_distance_points > 0 and (daily_loss + worst_case_loss) > self._daily_loss_limit:
            # Reduce contracts to the largest count that fits within remaining budget
            remaining_budget = self._daily_loss_limit - daily_loss
            max_safe = int(remaining_budget / (stop_distance_points * pv))
            if max_safe <= 0:
                detail = (
                    f"No room for even 1 contract: daily_loss=${daily_loss:,.2f}, "
                    f"remaining_budget=${remaining_budget:,.2f}, "
                    f"stop_cost=${stop_distance_points * pv:,.2f}"
                )
                return _blocked("daily_loss_limit", detail, 0)
            # Reduce and continue
            proposed_contracts = min(proposed_contracts, max_safe)
            logger.warning(
                f"[rules] Contracts reduced to {proposed_contracts} to stay within "
                f"daily loss limit (remaining ${remaining_budget:,.0f})"
            )

        # ── Rule 4: Would next loss breach trailing drawdown floor? ───
        if stop_distance_points > 0:
            worst_case_balance = state.current_balance - worst_case_loss
            if worst_case_balance <= drawdown_floor:
                # Reduce contracts to the largest count that keeps worst-case above floor
                headroom = state.current_balance - drawdown_floor
                max_safe_dd = int(headroom / (stop_distance_points * pv))
                if max_safe_dd <= 0:
                    detail = (
                        f"No room: balance ${state.current_balance:,.2f}, "
                        f"drawdown floor ${drawdown_floor:,.2f}, "
                        f"headroom ${headroom:,.2f}"
                    )
                    return _blocked("trailing_drawdown", detail, 0)
                proposed_contracts = min(proposed_contracts, max_safe_dd)
                logger.warning(
                    f"[rules] Contracts reduced to {proposed_contracts} to stay above "
                    f"drawdown floor (headroom ${headroom:,.0f})"
                )

        # ── Rule 5: Profit target ─────────────────────────────────────
        profit_so_far = state.current_balance - state.starting_balance
        if profit_so_far >= self._profit_target:
            detail = (
                f"Profit target reached: ${profit_so_far:,.2f} >= ${self._profit_target:,.2f}"
            )
            return _blocked("profit_target_reached", detail, 0)

        # ── Rule 6: Consistency rule ──────────────────────────────────
        safe_contracts = self._apply_consistency_rule(
            state, proposed_contracts, stop_distance_points
        )

        # ── Rule 7: Max contracts cap ─────────────────────────────────
        safe_contracts = min(safe_contracts, self._max_contracts)
        if safe_contracts <= 0:
            return _blocked("consistency_rule", "Consistency cap prevents any new position", 0)

        detail = (
            f"daily_loss=${daily_loss:,.2f}/{self._daily_loss_limit:,.2f} | "
            f"balance=${state.current_balance:,.2f} | "
            f"floor=${drawdown_floor:,.2f} | "
            f"profit=${profit_so_far:,.2f}/{self._profit_target:,.2f} | "
            f"contracts={safe_contracts}"
        )
        result = RuleCheckResult(can_trade=True, reason="ok", detail=detail,
                                 safe_contracts=safe_contracts)
        logger.debug(f"[rules] {result}")
        return result

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------

    def update_after_trade(
        self,
        state:    PropFirmState,
        trade_pnl: float,
        is_eod:   bool = False,
    ) -> PropFirmState:
        """
        Update state after a trade closes.

        Args:
            state:     Current PropFirmState.
            trade_pnl: Realised P&L of the completed trade (signed, $).
            is_eod:    True if this update is at end-of-day settlement.

        Returns:
            Updated PropFirmState (modifies in place and returns it).
        """
        state.current_balance  += trade_pnl
        state.daily_pnl        += trade_pnl
        state.total_net_profit += trade_pnl
        state.trades_today     += 1

        # Intraday trailing: advance HWM immediately on every profitable tick
        if self._trailing_type == "intraday":
            if state.current_balance > state.high_water_mark:
                old_hwm = state.high_water_mark
                state.high_water_mark = state.current_balance
                logger.debug(
                    f"[rules] Intraday HWM advanced: ${old_hwm:,.2f} → ${state.high_water_mark:,.2f}"
                )

        # Check if profit target now reached
        profit_so_far = state.current_balance - state.starting_balance
        if profit_so_far >= self._profit_target:
            state.phase_passed = True
            logger.info(
                f"[rules] PROFIT TARGET REACHED: ${profit_so_far:,.2f} >= "
                f"${self._profit_target:,.2f} — phase complete!"
            )

        # Check trailing drawdown breach
        drawdown_floor = state.high_water_mark - self._max_drawdown
        if state.current_balance <= drawdown_floor:
            state.is_halted   = True
            state.halt_reason = (
                f"Trailing drawdown breached: balance=${state.current_balance:,.2f} "
                f"<= floor=${drawdown_floor:,.2f}"
            )
            logger.error(f"[rules] HALT — {state.halt_reason}")

        # Check daily loss limit breach
        daily_loss = -min(state.daily_pnl, 0.0)
        if daily_loss >= self._daily_loss_limit:
            state.is_halted   = True
            state.halt_reason = (
                f"Daily loss limit breached: loss=${daily_loss:,.2f} "
                f">= limit=${self._daily_loss_limit:,.2f}"
            )
            logger.error(f"[rules] HALT — {state.halt_reason}")

        if is_eod:
            state = self.end_of_day(state)

        return state

    def end_of_day(self, state: PropFirmState) -> PropFirmState:
        """
        Perform end-of-day settlement:
          - EOD-type: advance high-water mark to current balance if higher
          - Reset daily_pnl and trades_today
          - Clear intraday halt if it was a daily-limit halt (not a drawdown halt)

        Call this once per trading day after the session close.

        Args:
            state: Current PropFirmState.

        Returns:
            Updated PropFirmState.
        """
        # EOD trailing: advance HWM only now (Topstep behavior)
        if self._trailing_type == "eod":
            if state.current_balance > state.high_water_mark:
                old_hwm = state.high_water_mark
                state.high_water_mark = state.current_balance
                logger.info(
                    f"[rules] EOD HWM advanced: ${old_hwm:,.2f} → ${state.high_water_mark:,.2f}"
                )

        # Reset daily counters
        state.daily_pnl    = 0.0
        state.trades_today = 0

        # Lift daily-loss halt for next day (drawdown halts are permanent)
        if state.is_halted and "Daily loss" in state.halt_reason:
            logger.info(
                f"[rules] Daily loss halt lifted at EOD — resuming next session. "
                f"Previous halt: {state.halt_reason}"
            )
            state.is_halted   = False
            state.halt_reason = ""

        return state

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def current_drawdown(self, state: PropFirmState) -> float:
        """Return current drawdown from high-water mark (positive = in drawdown)."""
        return max(0.0, state.high_water_mark - state.current_balance)

    def drawdown_pct_used(self, state: PropFirmState) -> float:
        """Return fraction of max_drawdown consumed (0–1)."""
        dd = self.current_drawdown(state)
        return dd / self._max_drawdown if self._max_drawdown > 0 else 0.0

    def daily_loss_pct_used(self, state: PropFirmState) -> float:
        """Return fraction of daily_loss_limit consumed (0–1)."""
        loss = -min(state.daily_pnl, 0.0)
        return loss / self._daily_loss_limit if self._daily_loss_limit > 0 else 0.0

    def is_near_limits(self, state: PropFirmState, threshold: float = 0.75) -> bool:
        """
        Return True if either drawdown or daily loss is above `threshold` (default 75%).
        Useful for reducing aggression before hitting hard limits.
        """
        return (
            self.drawdown_pct_used(state) >= threshold
            or self.daily_loss_pct_used(state) >= threshold
        )

    # ------------------------------------------------------------------
    # Consistency rule helper
    # ------------------------------------------------------------------

    def _apply_consistency_rule(
        self,
        state:               PropFirmState,
        proposed_contracts:  int,
        stop_distance_points: float,
    ) -> int:
        """
        Enforce consistency rule: no single day's profit > rule_pct × profit_target.

        Topstep / most prop firms define the consistency rule as:
          "No single day may contribute more than X% of the PROFIT TARGET."

        Using profit_target (fixed) rather than total_net_profit (rolling) avoids
        the pathological case where the first profitable day blocks all further
        same-day trades because 40% of a small running total < today's first winner.

        The cap is only active when today's realised profit already exceeds the
        daily cap — any future trades that same day are blocked.

        Returns safe contract count (>= 0).
        """
        if self._consistency_rule <= 0:
            return proposed_contracts

        # Cap = X% of the absolute profit target (fixed denominator)
        max_today_profit    = self._consistency_rule * self._profit_target
        today_profit_so_far = max(state.daily_pnl, 0.0)

        if today_profit_so_far >= max_today_profit:
            logger.warning(
                f"[rules] Consistency cap: today's profit ${today_profit_so_far:,.2f} "
                f">= cap ${max_today_profit:,.2f} "
                f"({self._consistency_rule:.0%} x ${self._profit_target:,.0f}) "
                f"— no new trades."
            )
            return 0

        # Under the cap — allow trade
        return proposed_contracts


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _blocked(reason: HaltReason, detail: str, safe_contracts: int) -> RuleCheckResult:
    """Return a RuleCheckResult that blocks trading."""
    logger.warning(f"[rules] BLOCKED({reason}): {detail}")
    return RuleCheckResult(
        can_trade       = False,
        reason          = reason,
        detail          = detail,
        safe_contracts  = safe_contracts,
    )
