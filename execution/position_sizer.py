"""
execution/position_sizer.py — Position Sizer (Layer 5 Sizing)

Volatility-adjusted, prop-firm-aware contract sizing for NQ/MNQ futures.

Sizing algorithm:
  1. Base contracts = (account_balance × max_risk_pct) / (stop_distance_points × point_value)
  2. Drawdown multiplier (how much of max_drawdown is consumed):
       0–25%  → 1.00x  (full size)
       25–50% → 0.75x  (reduce as drawdown builds)
       50–75% → 0.50x  (half size — danger zone)
       75–100%→ 0.25x  (minimum — near limit)
  3. Regime multiplier:
       trending  → 1.00x  (highest confidence)
       breakout  → 0.75x  (reduce for choppier entries)
       other     → 1.00x  (shouldn't reach here — combiner blocks ranging/volatile)
  4. Final = int(base × dd_mult × regime_mult)
       Clamped to [1, config.position.max_contracts]

Micro equivalent:
  MNQ = NQ × 10  (MNQ is 1/10 the size of NQ)
  Use MNQ when account < $25,000 (prop firm micro tier).

Validation:
  Worst-case loss = contracts × stop_distance_points × point_value
  Valid if worst-case_loss <= daily_loss_limit × 0.30

  The 30% guard means one trade can consume at most 30% of the daily loss
  limit — leaving room for up to 3 full-size losers per day before hitting
  the daily limit (matching spec safety requirement).

Usage:
    from execution.position_sizer import PositionSizer
    sizer = PositionSizer(config)
    n_contracts = sizer.calculate_size(
        account_balance=50_000,
        stop_distance_points=12.5,
        current_drawdown=500.0,
        max_drawdown=2_000.0,
        regime="trending",
    )
    if not sizer.validate_size(n_contracts, daily_loss_limit=1_000, stop_distance_points=12.5):
        n_contracts = 1  # force minimum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger

RegimeType = Literal["trending", "breakout", "ranging", "volatile"]

# Point value per contract (NQ full-size = $20/point)
NQ_POINT_VALUE  = 20.0
MNQ_POINT_VALUE =  2.0   # MNQ = 1/10 of NQ

# Drawdown buckets: (upper_fraction, multiplier)
_DD_BUCKETS: list[tuple[float, float]] = [
    (0.25, 1.00),
    (0.50, 0.75),
    (0.75, 0.50),
    (1.00, 0.25),
]

# Regime multipliers
_REGIME_MULT: dict[str, float] = {
    "trending": 1.00,
    "breakout": 0.75,
}

# Accounts below this threshold should trade MNQ
_MICRO_ACCOUNT_THRESHOLD = 25_000.0

# Max fraction of daily loss limit a single trade may risk
_SINGLE_TRADE_DAILY_LIMIT_FRACTION = 0.30


@dataclass
class SizingResult:
    """Detailed output of PositionSizer.calculate_size()."""
    contracts:        int     # Final contract count (NQ or MNQ)
    instrument:       str     # "NQ" or "MNQ"
    base_contracts:   float   # Pre-rounding base value
    dd_multiplier:    float   # Drawdown tier multiplier applied
    regime_multiplier:float   # Regime multiplier applied
    risk_per_trade:   float   # Dollar risk at this size + stop
    dd_pct_used:      float   # Fraction of max_drawdown consumed (0–1)
    valid:            bool    # Passed validate_size()
    reason:           str     # Human-readable sizing log

    def __str__(self) -> str:
        return (
            f"SizingResult({self.contracts}x{self.instrument} | "
            f"risk=${self.risk_per_trade:,.0f} | "
            f"dd_used={self.dd_pct_used:.0%} | "
            f"dd_mult={self.dd_multiplier:.2f} | "
            f"regime_mult={self.regime_multiplier:.2f} | "
            f"{'VALID' if self.valid else 'INVALID'} | "
            f"{self.reason})"
        )


class PositionSizer:
    """
    Volatility-adjusted, prop-firm-aware contract sizer for NQ/MNQ.

    Args:
        config: top-level Config dataclass. If None, uses spec defaults.
    """

    def __init__(self, config=None):
        if config is not None:
            self._max_risk_pct   = config.position.max_risk_per_trade_pct
            self._max_contracts  = config.position.max_contracts
            self._point_value    = config.position.point_value
        else:
            self._max_risk_pct   = 0.01     # 1% of account per trade
            self._max_contracts  = 3        # never hold more than 3 NQ
            self._point_value    = NQ_POINT_VALUE

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def calculate_size(
        self,
        account_balance:      float,
        stop_distance_points: float,
        current_drawdown:     float,
        max_drawdown:         float,
        regime:               str,
        point_value:          float | None = None,
        max_risk_pct:         float | None = None,
    ) -> int:
        """
        Calculate number of contracts to trade.

        Args:
            account_balance:      Current account equity ($).
            stop_distance_points: Distance from entry to stop in NQ points.
            current_drawdown:     Current unrealised + realised drawdown ($, positive).
            max_drawdown:         Prop firm max allowed drawdown ($, positive).
            regime:               HMM regime string ("trending", "breakout", …).
            point_value:          Override NQ point value (default $20).
            max_risk_pct:         Override max risk % (default 1%).

        Returns:
            Number of contracts (minimum 1, maximum config.max_contracts).
        """
        result = self.calculate_size_detailed(
            account_balance, stop_distance_points, current_drawdown,
            max_drawdown, regime, point_value, max_risk_pct,
        )
        return result.contracts

    def calculate_size_detailed(
        self,
        account_balance:      float,
        stop_distance_points: float,
        current_drawdown:     float,
        max_drawdown:         float,
        regime:               str,
        point_value:          float | None = None,
        max_risk_pct:         float | None = None,
    ) -> SizingResult:
        """
        Full sizing calculation returning a SizingResult with all components.

        Same parameters as calculate_size().
        """
        pv          = point_value   if point_value   is not None else self._point_value
        risk_pct    = max_risk_pct  if max_risk_pct  is not None else self._max_risk_pct

        # ── Guard: degenerate inputs ──────────────────────────────────
        if stop_distance_points <= 0 or account_balance <= 0:
            logger.warning(
                f"[sizer] Degenerate input: "
                f"balance={account_balance:.2f} stop={stop_distance_points:.2f} — sizing to 1"
            )
            return self._build_result(
                1, account_balance, 0.0, 1.0, 1.0, 0.0,
                stop_distance_points, pv, regime, "degenerate_input"
            )

        # ── Step 1: Base contracts ────────────────────────────────────
        dollar_risk = account_balance * risk_pct
        base = dollar_risk / (stop_distance_points * pv)

        # ── Step 2: Drawdown multiplier ───────────────────────────────
        dd_pct = _drawdown_fraction(current_drawdown, max_drawdown)
        dd_mult = _get_dd_multiplier(dd_pct)

        # ── Step 3: Regime multiplier ─────────────────────────────────
        reg_mult = _REGIME_MULT.get(regime, 1.0)

        # ── Step 4: Final contracts ───────────────────────────────────
        raw = base * dd_mult * reg_mult
        contracts = max(1, min(int(raw), self._max_contracts))

        reason = (
            f"base={base:.2f} × dd_mult={dd_mult:.2f} × reg_mult={reg_mult:.2f} "
            f"= {raw:.2f} → {contracts} (cap={self._max_contracts})"
        )
        result = self._build_result(
            contracts, account_balance, dd_pct, dd_mult, reg_mult,
            base, stop_distance_points, pv, regime, reason
        )
        logger.info(f"[sizer] {result}")
        return result

    # ------------------------------------------------------------------
    # Micro equivalent
    # ------------------------------------------------------------------

    def get_micro_equivalent(self, nq_contracts: int) -> int:
        """
        Return the MNQ contract count equivalent to nq_contracts NQ.

        MNQ is 1/10 the size of NQ, so multiply by 10.
        Used when account_balance < $25,000 (prop firm micro tier).

        Args:
            nq_contracts: Number of NQ contracts.

        Returns:
            Number of MNQ contracts (nq_contracts × 10).
        """
        mnq = nq_contracts * 10
        logger.debug(f"[sizer] Micro equivalent: {nq_contracts} NQ → {mnq} MNQ")
        return mnq

    def should_use_micro(self, account_balance: float) -> bool:
        """Return True if the account should trade MNQ instead of NQ."""
        return account_balance < _MICRO_ACCOUNT_THRESHOLD

    def get_instrument(self, account_balance: float) -> str:
        """Return 'MNQ' or 'NQ' based on account size."""
        return "MNQ" if self.should_use_micro(account_balance) else "NQ"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_size(
        self,
        contracts:            int,
        daily_loss_limit:     float,
        stop_distance_points: float,
        point_value:          float | None = None,
    ) -> bool:
        """
        Validate that worst-case loss for this trade fits within risk budget.

        Rule: contracts × stop_distance_points × point_value
              must be ≤ daily_loss_limit × 30%

        This ensures one trade can consume at most 30% of the daily loss
        allowance — leaving room for 3 full-sized losers.

        Args:
            contracts:            Proposed contract count.
            daily_loss_limit:     Prop firm daily loss limit ($, positive).
            stop_distance_points: Stop distance in NQ points.
            point_value:          Override point value (default $20).

        Returns:
            True if size is within the 30% single-trade limit.
        """
        pv = point_value if point_value is not None else self._point_value
        worst_case = contracts * stop_distance_points * pv
        budget     = daily_loss_limit * _SINGLE_TRADE_DAILY_LIMIT_FRACTION

        valid = worst_case <= budget
        logger.debug(
            f"[sizer] validate_size: {contracts}c × {stop_distance_points}pts × ${pv} "
            f"= ${worst_case:,.0f} vs budget=${budget:,.0f} ({_SINGLE_TRADE_DAILY_LIMIT_FRACTION:.0%} "
            f"of ${daily_loss_limit:,.0f}) → {'OK' if valid else 'FAIL'}"
        )
        if not valid:
            logger.warning(
                f"[sizer] Size REJECTED: worst-case ${worst_case:,.0f} "
                f"> 30% daily limit (${budget:,.0f})"
            )
        return valid

    def safe_size(
        self,
        contracts:            int,
        daily_loss_limit:     float,
        stop_distance_points: float,
        point_value:          float | None = None,
    ) -> int:
        """
        Return contracts if it passes validate_size(), else reduce to the
        largest valid count (minimum 1).

        Args:
            contracts:            Proposed contract count.
            daily_loss_limit:     Prop firm daily loss limit ($, positive).
            stop_distance_points: Stop distance in NQ points.
            point_value:          Override point value (default $20).

        Returns:
            Validated contract count (≥ 1).
        """
        pv = point_value if point_value is not None else self._point_value
        if self.validate_size(contracts, daily_loss_limit, stop_distance_points, pv):
            return contracts

        # Walk down until valid (at minimum return 1)
        budget = daily_loss_limit * _SINGLE_TRADE_DAILY_LIMIT_FRACTION
        max_allowed = int(budget / (stop_distance_points * pv))
        result = max(1, min(contracts, max_allowed))
        logger.warning(
            f"[sizer] safe_size: reduced {contracts} → {result} to fit daily limit"
        )
        return result

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _build_result(
        self,
        contracts:    int,
        balance:      float,
        dd_pct:       float,
        dd_mult:      float,
        reg_mult:     float,
        base:         float,
        stop_pts:     float,
        pv:           float,
        regime:       str,
        reason:       str,
    ) -> SizingResult:
        instrument = self.get_instrument(balance)
        risk       = contracts * stop_pts * pv
        return SizingResult(
            contracts         = contracts,
            instrument        = instrument,
            base_contracts    = round(base, 4),
            dd_multiplier     = dd_mult,
            regime_multiplier = reg_mult,
            risk_per_trade    = round(risk, 2),
            dd_pct_used       = round(dd_pct, 4),
            valid             = True,  # caller can override via validate_size()
            reason            = reason,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _drawdown_fraction(current_drawdown: float, max_drawdown: float) -> float:
    """Return fraction of max_drawdown consumed (clamped 0–1)."""
    if max_drawdown <= 0:
        return 0.0
    return max(0.0, min(current_drawdown / max_drawdown, 1.0))


def _get_dd_multiplier(dd_fraction: float) -> float:
    """Return the drawdown tier multiplier for dd_fraction ∈ [0, 1]."""
    for upper, mult in _DD_BUCKETS:
        if dd_fraction <= upper:
            return mult
    return _DD_BUCKETS[-1][1]  # Fully consumed → minimum multiplier
