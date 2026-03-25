"""
tests/test_daily_limits.py — Unit tests for DailyLimits
"""

import pytest
from risk.daily_limits import DailyLimits, DailyLimitsState, DailyLimitsCheck


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def limits():
    """Default: daily_loss_limit=$2000, profit_lock at 50% ($1000), buffer 20%."""
    return DailyLimits()


@pytest.fixture
def state():
    return DailyLimitsState(
        daily_loss_limit = 2_000.0,
        profit_lock_pct  = 0.50,
        lock_buffer_pct  = 0.20,
    )


# ---------------------------------------------------------------------------
# 1. DailyLimitsState properties
# ---------------------------------------------------------------------------

class TestStateProps:

    def test_daily_loss_zero(self, state):
        assert state.daily_loss == 0.0

    def test_daily_loss_from_negative_pnl(self, state):
        state.daily_pnl = -800.0
        assert state.daily_loss == 800.0

    def test_daily_profit_zero_when_losing(self, state):
        state.daily_pnl = -500.0
        assert state.daily_profit == 0.0

    def test_daily_profit_positive(self, state):
        state.daily_pnl = 400.0
        assert state.daily_profit == 400.0

    def test_loss_pct_used(self, state):
        state.daily_pnl = -1_000.0
        assert state.loss_pct_used == pytest.approx(0.50)

    def test_profit_lock_threshold(self, state):
        # 50% of $2000 = $1000
        assert state.profit_lock_threshold == pytest.approx(1_000.0)

    def test_not_profit_locked_below_threshold(self, state):
        state.daily_pnl = 999.0
        assert state.is_profit_locked is False

    def test_profit_locked_at_threshold(self, state):
        state.daily_pnl = 1_000.0
        assert state.is_profit_locked is True

    def test_lock_buffer_inf_when_not_locked(self, state):
        state.daily_pnl = 500.0
        assert state.lock_buffer_dollars == float("inf")

    def test_lock_buffer_when_locked(self, state):
        state.daily_pnl = 1_200.0  # locked; 20% of $1200 = $240
        assert state.lock_buffer_dollars == pytest.approx(240.0)


# ---------------------------------------------------------------------------
# 2. check() — loss tiers
# ---------------------------------------------------------------------------

class TestCheckLossTiers:

    def test_normal_tier_can_trade(self, limits, state):
        state.daily_pnl = -500.0  # 25% used
        result = limits.check(state, risk_dollars=200.0)
        assert result.can_trade is True
        assert result.loss_tier == "NORMAL"
        assert result.size_multiplier == 1.0

    def test_warning_tier_reduces_size(self, limits, state):
        state.daily_pnl = -1_100.0  # 55% used
        result = limits.check(state, risk_dollars=200.0)
        assert result.can_trade is True
        assert result.loss_tier == "WARNING"
        assert result.size_multiplier == 0.75

    def test_alert_tier_reduces_size(self, limits, state):
        state.daily_pnl = -1_600.0  # 80% used
        result = limits.check(state, risk_dollars=200.0)
        assert result.can_trade is True
        assert result.loss_tier == "ALERT"
        assert result.size_multiplier == 0.50

    def test_locked_tier_blocks_trade(self, limits, state):
        state.daily_pnl = -2_000.0  # 100% used
        result = limits.check(state, risk_dollars=200.0)
        assert result.can_trade is False
        assert result.loss_tier == "LOCKED"
        assert result.size_multiplier == 0.0

    def test_above_locked_tier_blocks_trade(self, limits, state):
        state.daily_pnl = -2_500.0
        result = limits.check(state, risk_dollars=100.0)
        assert result.can_trade is False


# ---------------------------------------------------------------------------
# 3. check() — profit lock
# ---------------------------------------------------------------------------

class TestCheckProfitLock:

    def test_below_profit_lock_free(self, limits, state):
        state.daily_pnl = 800.0   # < $1000 threshold
        result = limits.check(state, risk_dollars=500.0)
        assert result.profit_tier == "FREE"
        assert result.can_trade is True

    def test_profit_locked_small_risk_allowed(self, limits, state):
        state.daily_pnl = 1_200.0   # locked; buffer = 20% × $1200 = $240
        result = limits.check(state, risk_dollars=200.0)
        assert result.profit_tier == "LOCKED"
        assert result.can_trade is True   # $200 <= $240 buffer

    def test_profit_locked_large_risk_blocked(self, limits, state):
        state.daily_pnl = 1_200.0   # buffer = $240
        result = limits.check(state, risk_dollars=300.0)
        assert result.profit_tier == "LOCKED"
        assert result.can_trade is False   # $300 > $240

    def test_profit_locked_size_mult_capped(self, limits, state):
        """In profit-lock mode, size multiplier should be ≤ 0.50."""
        state.daily_pnl = 1_500.0   # locked
        result = limits.check(state, risk_dollars=100.0)
        assert result.can_trade is True
        assert result.size_multiplier <= 0.50

    def test_profit_lock_exact_boundary(self, limits, state):
        state.daily_pnl = 1_000.0   # exactly at threshold → locked
        result = limits.check(state, risk_dollars=100.0)
        assert result.profit_tier == "LOCKED"

    def test_profit_lock_zero_risk_always_allowed(self, limits, state):
        """risk_dollars=0 should always pass profit lock check."""
        state.daily_pnl = 2_000.0   # well into locked territory
        result = limits.check(state, risk_dollars=0.0)
        assert result.can_trade is True


# ---------------------------------------------------------------------------
# 4. check() result fields
# ---------------------------------------------------------------------------

class TestCheckResultFields:

    def test_result_str_no_error(self, limits, state):
        result = limits.check(state, risk_dollars=200.0)
        s = str(result)
        assert "OK" in s or "BLOCKED" in s

    def test_result_detail_populated(self, limits, state):
        result = limits.check(state, risk_dollars=100.0)
        assert isinstance(result.detail, str) and len(result.detail) > 0


# ---------------------------------------------------------------------------
# 5. update()
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_loss_updates_daily_pnl(self, limits, state):
        state = limits.update(state, trade_pnl=-400.0)
        assert state.daily_pnl == -400.0
        assert state.trades_today == 1

    def test_profit_updates_daily_pnl(self, limits, state):
        state = limits.update(state, trade_pnl=600.0)
        assert state.daily_pnl == 600.0

    def test_peak_daily_profit_tracked(self, limits, state):
        limits.update(state, trade_pnl=700.0)
        limits.update(state, trade_pnl=-100.0)
        assert state.peak_daily_profit == 700.0  # peak was $700 before the -$100

    def test_peak_daily_profit_not_negative(self, limits, state):
        limits.update(state, trade_pnl=-500.0)
        assert state.peak_daily_profit == 0.0

    def test_multiple_trades_accumulate(self, limits, state):
        limits.update(state, trade_pnl=300.0)
        limits.update(state, trade_pnl=-150.0)
        limits.update(state, trade_pnl=200.0)
        assert state.daily_pnl == pytest.approx(350.0)
        assert state.trades_today == 3


# ---------------------------------------------------------------------------
# 6. end_of_day()
# ---------------------------------------------------------------------------

class TestEndOfDay:

    def test_eod_resets_pnl(self, limits, state):
        state.daily_pnl = -800.0
        state = limits.end_of_day(state)
        assert state.daily_pnl == 0.0

    def test_eod_resets_peak_profit(self, limits, state):
        state.peak_daily_profit = 1_200.0
        state = limits.end_of_day(state)
        assert state.peak_daily_profit == 0.0

    def test_eod_resets_trades(self, limits, state):
        state.trades_today = 7
        state = limits.end_of_day(state)
        assert state.trades_today == 0

    def test_eod_preserves_limits(self, limits, state):
        state = limits.end_of_day(state)
        assert state.daily_loss_limit  == 2_000.0
        assert state.profit_lock_pct   == 0.50
        assert state.lock_buffer_pct   == 0.20

    def test_eod_allows_trading_next_day_after_loss(self, limits, state):
        """After losing day, EOD reset should unblock trading."""
        state.daily_pnl = -2_100.0   # over limit
        assert limits.check(state, risk_dollars=100.0).can_trade is False
        state = limits.end_of_day(state)
        assert limits.check(state, risk_dollars=100.0).can_trade is True


# ---------------------------------------------------------------------------
# 7. Convenience helpers
# ---------------------------------------------------------------------------

class TestConvenienceHelpers:

    def test_remaining_budget_full(self, limits, state):
        assert limits.remaining_loss_budget(state) == 2_000.0

    def test_remaining_budget_partial(self, limits, state):
        state.daily_pnl = -700.0
        assert limits.remaining_loss_budget(state) == pytest.approx(1_300.0)

    def test_remaining_budget_zero_when_locked(self, limits, state):
        state.daily_pnl = -2_100.0
        assert limits.remaining_loss_budget(state) == 0.0

    def test_max_contracts_for_stop(self, limits, state):
        # Budget $2000, stop 10pts × $20 = $200/contract → 10 contracts
        assert limits.max_contracts_for_stop(state, 10.0) == 10

    def test_max_contracts_for_stop_partial_budget(self, limits, state):
        state.daily_pnl = -1_600.0   # $400 remaining; 10pts×$20=$200 → 2c
        assert limits.max_contracts_for_stop(state, 10.0) == 2

    def test_max_contracts_for_stop_minimum_1(self, limits, state):
        state.daily_pnl = -1_999.0   # $1 remaining — can't fit any, floor to 1
        assert limits.max_contracts_for_stop(state, 10.0) == 1

    def test_max_contracts_for_stop_zero_stop(self, limits, state):
        assert limits.max_contracts_for_stop(state, 0.0) == 1
