"""
risk/drawdown_monitor.py — Real-Time Drawdown Monitor

Tracks drawdown in real time and emits tiered alerts before hard prop-firm
limits are reached. Works alongside PropFirmRules — the monitor warns early
so the bot can reduce size; PropFirmRules enforces hard stops.

Alert tiers (configurable, defaults match spec):
  WATCH    — drawdown > 50% of max_drawdown  (start monitoring closely)
  CAUTION  — drawdown > 65% of max_drawdown  (reduce position size)
  DANGER   — drawdown > 80% of max_drawdown  (stop new entries)
  CRITICAL — drawdown > 90% of max_drawdown  (flatten all positions)

The monitor also tracks:
  - Peak intraday drawdown (max loss reached at any point today)
  - Consecutive losing trades (streak counter)
  - Daily drawdown velocity (how fast drawdown is accumulating)

DrawdownState is separate from PropFirmState — the monitor only cares
about drawdown-related metrics, not daily P&L or phase tracking.

Usage:
    from risk.drawdown_monitor import DrawdownMonitor, DrawdownState

    monitor = DrawdownMonitor(config)
    state = DrawdownState(
        account_balance  = 100_000.0,
        high_water_mark  = 100_000.0,
        max_drawdown     = 3_000.0,
    )

    # After each bar or trade update:
    state = monitor.update(state, current_equity=99_200.0)
    alert = monitor.get_alert(state)

    if alert.level == "DANGER":
        # Skip new entries; let existing trades run to target/stop
        pass
    elif alert.level == "CRITICAL":
        # Flatten everything immediately
        flatten_all_positions()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

AlertLevel = Literal["OK", "WATCH", "CAUTION", "DANGER", "CRITICAL"]

# Default tier thresholds as fraction of max_drawdown
_DEFAULT_TIERS: dict[AlertLevel, float] = {
    "WATCH":    0.50,
    "CAUTION":  0.65,
    "DANGER":   0.80,
    "CRITICAL": 0.90,
}

# Action recommendations per tier
_TIER_ACTIONS: dict[AlertLevel, str] = {
    "OK":       "Normal trading",
    "WATCH":    "Monitor closely — drawdown above 50%",
    "CAUTION":  "Reduce position size — drawdown above 65%",
    "DANGER":   "Stop new entries — drawdown above 80%",
    "CRITICAL": "Flatten all positions — drawdown above 90%",
}


# ---------------------------------------------------------------------------
# State and alert dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DrawdownState:
    """
    Mutable drawdown tracking state.

    Kept separate from PropFirmState to allow independent monitoring
    (e.g., the monitor can be used without the full prop firm rules engine).
    """
    account_balance:        float          # Current mark-to-market equity
    high_water_mark:        float          # Highest equity ever reached
    max_drawdown:           float          # Prop firm max allowed drawdown ($)

    # Running statistics (updated by DrawdownMonitor.update())
    current_drawdown:       float  = 0.0  # HWM - current_balance (positive)
    peak_intraday_drawdown: float  = 0.0  # Worst drawdown seen today
    consecutive_losses:     int    = 0    # Number of consecutive losing trades
    trades_today:           int    = 0    # Total trades today
    losses_today:           int    = 0    # Losing trades today
    last_alert_level:       str    = "OK"
    equity_history:         list   = field(default_factory=list)  # For velocity calc

    @property
    def drawdown_pct(self) -> float:
        """Fraction of max_drawdown consumed (0–1)."""
        if self.max_drawdown <= 0:
            return 0.0
        return self.current_drawdown / self.max_drawdown

    @property
    def drawdown_floor(self) -> float:
        """Absolute price level that would trigger the drawdown halt."""
        return self.high_water_mark - self.max_drawdown

    @property
    def headroom(self) -> float:
        """Dollars remaining before hitting the drawdown floor."""
        return max(0.0, self.account_balance - self.drawdown_floor)


@dataclass
class DrawdownAlert:
    """Alert emitted by DrawdownMonitor.get_alert()."""
    level:            AlertLevel
    drawdown_dollars: float    # Current drawdown in dollars
    drawdown_pct:     float    # Fraction of max_drawdown (0–1)
    headroom:         float    # Dollars before hard stop
    action:           str      # Recommended action string
    detail:           str      # Full detail for logging

    def __str__(self) -> str:
        return (
            f"DrawdownAlert({self.level} | "
            f"dd=${self.drawdown_dollars:,.0f} ({self.drawdown_pct:.0%}) | "
            f"headroom=${self.headroom:,.0f} | "
            f"{self.action})"
        )

    @property
    def should_stop_entries(self) -> bool:
        """True if new entries should be blocked (DANGER or CRITICAL)."""
        return self.level in ("DANGER", "CRITICAL")

    @property
    def should_flatten(self) -> bool:
        """True if all positions should be closed immediately (CRITICAL)."""
        return self.level == "CRITICAL"

    @property
    def size_multiplier(self) -> float:
        """
        Suggested position size multiplier for this alert level.
        Callers may use this to scale down contracts before calling PositionSizer.
        """
        return {
            "OK":       1.00,
            "WATCH":    0.75,
            "CAUTION":  0.50,
            "DANGER":   0.00,   # No new entries
            "CRITICAL": 0.00,
        }.get(self.level, 1.00)


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class DrawdownMonitor:
    """
    Real-time drawdown monitor with tiered alerts.

    Args:
        config:       top-level Config dataclass. If None, uses spec defaults.
        custom_tiers: Override default alert thresholds {level: fraction}.
                      Example: {"WATCH": 0.40, "CAUTION": 0.60, ...}
    """

    def __init__(self, config=None, custom_tiers: dict | None = None):
        if config is not None:
            self._max_consecutive_losses = getattr(
                config.signal, "max_consecutive_losses", 3
            )
        else:
            self._max_consecutive_losses = 3

        self._tiers = {**_DEFAULT_TIERS, **(custom_tiers or {})}

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def update(
        self,
        state:          DrawdownState,
        current_equity: float,
        trade_pnl:      float | None = None,
    ) -> DrawdownState:
        """
        Update drawdown state with the latest equity value.

        Call after every bar update (for live monitoring) or after every
        trade close (for trade-level monitoring).

        Args:
            state:          Current DrawdownState (mutated in place).
            current_equity: Latest account equity ($).
            trade_pnl:      Realised P&L if a trade just closed (None for bar updates).

        Returns:
            Updated DrawdownState.
        """
        state.account_balance = current_equity

        # Advance HWM if equity rose
        if current_equity > state.high_water_mark:
            old_hwm = state.high_water_mark
            state.high_water_mark = current_equity
            logger.debug(
                f"[ddmon] HWM advanced: ${old_hwm:,.2f} → ${state.high_water_mark:,.2f}"
            )

        # Recalculate drawdown
        state.current_drawdown = max(0.0, state.high_water_mark - current_equity)

        # Track peak intraday drawdown
        if state.current_drawdown > state.peak_intraday_drawdown:
            state.peak_intraday_drawdown = state.current_drawdown

        # Track equity history (last 20 observations for velocity)
        state.equity_history.append(current_equity)
        if len(state.equity_history) > 20:
            state.equity_history.pop(0)

        # Update trade stats if a trade just closed
        if trade_pnl is not None:
            state.trades_today += 1
            if trade_pnl < 0:
                state.losses_today += 1
                state.consecutive_losses += 1
            else:
                state.consecutive_losses = 0  # Reset streak on any win

        # Log alert transitions
        alert = self.get_alert(state)
        if alert.level != state.last_alert_level:
            if alert.level in ("DANGER", "CRITICAL"):
                logger.error(f"[ddmon] {alert}")
            elif alert.level in ("WATCH", "CAUTION"):
                logger.warning(f"[ddmon] {alert}")
            else:
                logger.info(f"[ddmon] Alert cleared: {state.last_alert_level} → OK")
            state.last_alert_level = alert.level

        return state

    def get_alert(self, state: DrawdownState) -> DrawdownAlert:
        """
        Return the current alert level and recommended action.

        Args:
            state: Current DrawdownState.

        Returns:
            DrawdownAlert dataclass.
        """
        level = self._classify(state.drawdown_pct)
        action = _TIER_ACTIONS[level]

        detail = (
            f"dd=${state.current_drawdown:,.2f} ({state.drawdown_pct:.1%}) | "
            f"hwm=${state.high_water_mark:,.2f} | "
            f"floor=${state.drawdown_floor:,.2f} | "
            f"headroom=${state.headroom:,.2f} | "
            f"peak_today=${state.peak_intraday_drawdown:,.2f} | "
            f"consec_losses={state.consecutive_losses}"
        )

        return DrawdownAlert(
            level            = level,
            drawdown_dollars = round(state.current_drawdown, 2),
            drawdown_pct     = round(state.drawdown_pct, 4),
            headroom         = round(state.headroom, 2),
            action           = action,
            detail           = detail,
        )

    def is_losing_streak(self, state: DrawdownState) -> bool:
        """Return True if consecutive losses exceed the configured threshold."""
        return state.consecutive_losses >= self._max_consecutive_losses

    def drawdown_velocity(self, state: DrawdownState) -> float:
        """
        Return average equity change per observation over recent history.

        Negative value = equity declining (drawdown accelerating).
        Uses last min(20, len(history)) equity observations.
        Returns 0.0 if fewer than 2 observations.
        """
        hist = state.equity_history
        if len(hist) < 2:
            return 0.0
        return (hist[-1] - hist[0]) / (len(hist) - 1)

    # ------------------------------------------------------------------
    # EOD reset
    # ------------------------------------------------------------------

    def end_of_day(self, state: DrawdownState) -> DrawdownState:
        """
        Reset intraday counters at session close.

        Preserves: account_balance, high_water_mark, current_drawdown,
                   consecutive_losses (carries across days)
        Resets: peak_intraday_drawdown, trades_today, losses_today,
                equity_history, last_alert_level
        """
        state.peak_intraday_drawdown = 0.0
        state.trades_today           = 0
        state.losses_today           = 0
        state.equity_history         = []
        state.last_alert_level       = "OK"
        logger.debug(
            f"[ddmon] EOD reset — balance=${state.account_balance:,.2f} "
            f"hwm=${state.high_water_mark:,.2f} dd=${state.current_drawdown:,.2f}"
        )
        return state

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _classify(self, drawdown_pct: float) -> AlertLevel:
        """Map drawdown fraction to alert level using configured tiers."""
        # Check from most severe to least severe
        for level in ("CRITICAL", "DANGER", "CAUTION", "WATCH"):
            if drawdown_pct > self._tiers[level]:
                return level
        return "OK"
