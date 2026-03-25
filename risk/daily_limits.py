"""
risk/daily_limits.py — Daily Loss Lock + Profit Lock-In

Two complementary daily limits:

  1. Loss lock — stop trading when daily loss reaches the limit.
     This is enforced by PropFirmRules but DailyLimits adds a
     tiered early-warning system so the bot can reduce size
     before hitting the hard wall.

  2. Profit lock — once daily profit reaches a configurable
     fraction of the daily loss limit, switch to "protect mode":
     - No new entries unless they would not risk more than
       lock_buffer_pct of the locked profit.
     - The bot locks in gains rather than gambling them back.

     Example (Topstep 100k):
       daily_loss_limit = $2,000
       profit_lock_pct  = 0.50  → lock at $1,000 profit
       lock_buffer_pct  = 0.20  → allow trades risking ≤ $200
                                   (20% of locked profit)

Loss-limit tiers (configurable):
  NORMAL  — daily loss < 50% of limit
  WARNING — daily loss >= 50% of limit  (reduce size)
  ALERT   — daily loss >= 75% of limit  (stop new setups)
  LOCKED  — daily loss >= 100% of limit (hard stop — no trading)

Profit-lock tiers:
  FREE    — daily profit < lock_threshold
  LOCKED  — daily profit >= lock_threshold
              (only trades within buffer are allowed)

Usage:
    from risk.daily_limits import DailyLimits, DailyLimitsState

    state = DailyLimitsState(daily_loss_limit=2_000.0)
    limits = DailyLimits(config)

    # After each trade:
    state = limits.update(state, trade_pnl=350.0)

    # Before each trade:
    check = limits.check(state, risk_dollars=200.0)
    if not check.can_trade:
        return  # skip

    # At EOD:
    state = limits.end_of_day(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

LossLimitTier = Literal["NORMAL", "WARNING", "ALERT", "LOCKED"]
ProfitLockTier = Literal["FREE", "LOCKED"]

# Loss tier thresholds (fraction of daily_loss_limit)
_LOSS_TIERS: list[tuple[float, LossLimitTier]] = [
    (1.00, "LOCKED"),
    (0.75, "ALERT"),
    (0.50, "WARNING"),
    (0.00, "NORMAL"),
]

# Size multipliers per loss tier
_LOSS_TIER_SIZE_MULT: dict[LossLimitTier, float] = {
    "NORMAL":  1.00,
    "WARNING": 0.75,
    "ALERT":   0.50,
    "LOCKED":  0.00,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class DailyLimitsState:
    """Mutable daily limits tracking state."""
    daily_loss_limit:  float          # Prop firm daily loss limit ($)

    # Updated by DailyLimits.update()
    daily_pnl:         float = 0.0    # Net P&L for today (signed)
    peak_daily_profit: float = 0.0    # Highest daily profit reached today
    trades_today:      int   = 0

    # Thresholds (set from config or defaults)
    profit_lock_pct:   float = 0.50   # Lock profit at this fraction of loss limit
    lock_buffer_pct:   float = 0.20   # Allow trades risking <= this fraction of locked profit

    @property
    def daily_loss(self) -> float:
        """Positive dollar amount lost today (0 if profitable)."""
        return max(0.0, -self.daily_pnl)

    @property
    def daily_profit(self) -> float:
        """Positive dollar amount gained today (0 if losing)."""
        return max(0.0, self.daily_pnl)

    @property
    def loss_pct_used(self) -> float:
        """Fraction of daily loss limit consumed (0–1)."""
        if self.daily_loss_limit <= 0:
            return 0.0
        return self.daily_loss / self.daily_loss_limit

    @property
    def profit_lock_threshold(self) -> float:
        """Dollar profit level that triggers profit lock."""
        return self.daily_loss_limit * self.profit_lock_pct

    @property
    def is_profit_locked(self) -> bool:
        """True if daily profit has reached the lock threshold."""
        return self.daily_pnl >= self.profit_lock_threshold

    @property
    def lock_buffer_dollars(self) -> float:
        """Max risk allowed per trade while in profit-lock mode."""
        if not self.is_profit_locked:
            return float("inf")
        return self.daily_pnl * self.lock_buffer_pct


@dataclass
class DailyLimitsCheck:
    """Output of DailyLimits.check()."""
    can_trade:       bool
    loss_tier:       LossLimitTier
    profit_tier:     ProfitLockTier
    size_multiplier: float          # Suggested size reduction (0–1)
    reason:          str
    detail:          str

    def __str__(self) -> str:
        status = "OK" if self.can_trade else "BLOCKED"
        return (
            f"DailyLimitsCheck({status} | "
            f"loss={self.loss_tier} | profit={self.profit_tier} | "
            f"size_mult={self.size_multiplier:.2f} | {self.reason})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DailyLimits:
    """
    Daily loss lock and profit lock-in management.

    Args:
        config: top-level Config dataclass. If None, uses Topstep 100k defaults.
    """

    def __init__(self, config=None):
        if config is not None:
            pf = config.prop_firm
            self._daily_loss_limit  = pf.daily_loss_limit
            self._profit_lock_pct   = getattr(pf, "profit_lock_pct",   0.50)
            self._lock_buffer_pct   = getattr(pf, "lock_buffer_pct",   0.20)
        else:
            self._daily_loss_limit  = 2_000.0
            self._profit_lock_pct   = 0.50
            self._lock_buffer_pct   = 0.20

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def check(
        self,
        state:        DailyLimitsState,
        risk_dollars: float = 0.0,
    ) -> DailyLimitsCheck:
        """
        Check whether a new trade is allowed given current daily limits.

        Args:
            state:        Current DailyLimitsState.
            risk_dollars: Worst-case dollar risk for the proposed trade
                          (contracts × stop_pts × point_value).

        Returns:
            DailyLimitsCheck. Trade must be skipped if can_trade is False.
        """
        loss_tier   = self._get_loss_tier(state)
        profit_tier = self._get_profit_tier(state)

        # ── Hard block: loss limit ───────────────────────────────────
        if loss_tier == "LOCKED":
            return DailyLimitsCheck(
                can_trade       = False,
                loss_tier       = loss_tier,
                profit_tier     = profit_tier,
                size_multiplier = 0.0,
                reason          = "daily_loss_locked",
                detail          = (
                    f"Daily loss ${state.daily_loss:,.2f} >= "
                    f"limit ${state.daily_loss_limit:,.2f}"
                ),
            )

        # ── Profit lock: reject trades exceeding buffer ──────────────
        if profit_tier == "LOCKED" and risk_dollars > state.lock_buffer_dollars:
            return DailyLimitsCheck(
                can_trade       = False,
                loss_tier       = loss_tier,
                profit_tier     = profit_tier,
                size_multiplier = 0.0,
                reason          = "profit_locked",
                detail          = (
                    f"Profit locked at ${state.daily_pnl:,.2f} "
                    f"(threshold=${state.profit_lock_threshold:,.2f}). "
                    f"Trade risk ${risk_dollars:,.2f} > "
                    f"buffer ${state.lock_buffer_dollars:,.2f}"
                ),
            )

        # ── Compute size multiplier ───────────────────────────────────
        size_mult = _LOSS_TIER_SIZE_MULT[loss_tier]

        # In profit-lock mode, also apply a size reduction
        if profit_tier == "LOCKED":
            size_mult = min(size_mult, 0.50)

        detail = (
            f"daily_pnl=${state.daily_pnl:,.2f} | "
            f"loss_used={state.loss_pct_used:.0%} | "
            f"profit_locked={state.is_profit_locked} | "
            f"lock_threshold=${state.profit_lock_threshold:,.2f} | "
            f"size_mult={size_mult:.2f}"
        )

        result = DailyLimitsCheck(
            can_trade       = True,
            loss_tier       = loss_tier,
            profit_tier     = profit_tier,
            size_multiplier = size_mult,
            reason          = "ok",
            detail          = detail,
        )
        logger.debug(f"[daily] {result}")
        return result

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update(
        self,
        state:     DailyLimitsState,
        trade_pnl: float,
    ) -> DailyLimitsState:
        """
        Update state after a trade closes.

        Args:
            state:     Current DailyLimitsState (mutated in place).
            trade_pnl: Realised trade P&L (signed, $).

        Returns:
            Updated DailyLimitsState.
        """
        state.daily_pnl    += trade_pnl
        state.trades_today += 1

        # Track peak daily profit for reporting
        if state.daily_pnl > state.peak_daily_profit:
            state.peak_daily_profit = state.daily_pnl

        tier = self._get_loss_tier(state)
        if tier in ("ALERT", "LOCKED"):
            logger.warning(
                f"[daily] Loss tier={tier}: "
                f"daily_loss=${state.daily_loss:,.2f} / ${state.daily_loss_limit:,.2f}"
            )

        if state.is_profit_locked and trade_pnl != 0:
            logger.info(
                f"[daily] Profit locked: ${state.daily_pnl:,.2f} >= "
                f"${state.profit_lock_threshold:,.2f}"
            )

        return state

    def end_of_day(self, state: DailyLimitsState) -> DailyLimitsState:
        """
        Reset all daily counters at session close.

        Preserves: daily_loss_limit, profit_lock_pct, lock_buffer_pct
        Resets: daily_pnl, peak_daily_profit, trades_today
        """
        logger.debug(
            f"[daily] EOD reset — "
            f"daily_pnl=${state.daily_pnl:,.2f} | "
            f"peak_profit=${state.peak_daily_profit:,.2f} | "
            f"trades={state.trades_today}"
        )
        state.daily_pnl         = 0.0
        state.peak_daily_profit = 0.0
        state.trades_today      = 0
        return state

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def remaining_loss_budget(self, state: DailyLimitsState) -> float:
        """Dollar amount that can still be lost today before hitting the limit."""
        return max(0.0, state.daily_loss_limit - state.daily_loss)

    def max_contracts_for_stop(
        self,
        state:                DailyLimitsState,
        stop_distance_points: float,
        point_value:          float = 20.0,
    ) -> int:
        """
        Return the maximum number of contracts that fit within the remaining
        loss budget (1 contract minimum).

        Args:
            state:                Current DailyLimitsState.
            stop_distance_points: Stop distance in NQ points.
            point_value:          NQ point value (default $20).

        Returns:
            Max contracts (>= 1).
        """
        if stop_distance_points <= 0:
            return 1
        budget = self.remaining_loss_budget(state)
        cost_per_contract = stop_distance_points * point_value
        return max(1, int(budget / cost_per_contract))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_loss_tier(self, state: DailyLimitsState) -> LossLimitTier:
        for threshold, tier in _LOSS_TIERS:
            if state.loss_pct_used >= threshold:
                return tier
        return "NORMAL"

    def _get_profit_tier(self, state: DailyLimitsState) -> ProfitLockTier:
        return "LOCKED" if state.is_profit_locked else "FREE"
