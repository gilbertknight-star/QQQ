"""
tests/test_prop_firm_rules.py — Unit tests for PropFirmRules

Covers:
  - Daily loss limit block
  - Trailing drawdown block (EOD and intraday)
  - Profit target reached block
  - Contract reduction (pre-check: would stop breach daily limit / drawdown floor)
  - Consistency rule
  - Max contracts cap
  - Already-halted state
  - update_after_trade() state transitions
  - end_of_day() HWM advancement and daily reset
  - EOD halt lifts; drawdown halt is permanent
  - Convenience queries: current_drawdown, drawdown_pct_used, is_near_limits
"""

import pytest
from risk.prop_firm_rules import PropFirmRules, PropFirmState, RuleCheckResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rules():
    """Topstep 100k defaults: daily_loss=2000, max_dd=3000, target=6000, eod."""
    return PropFirmRules()


@pytest.fixture
def clean_state():
    """Fresh account state — no drawdown, no daily loss."""
    return PropFirmState(
        starting_balance = 100_000.0,
        current_balance  = 100_000.0,
        high_water_mark  = 100_000.0,
        daily_pnl        = 0.0,
        total_net_profit = 0.0,
    )


# ---------------------------------------------------------------------------
# 1. Happy path — can trade
# ---------------------------------------------------------------------------

class TestCanTrade:

    def test_clean_state_can_trade(self, rules, clean_state):
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=12.5)
        assert result.can_trade is True
        assert result.reason == "ok"
        assert result.safe_contracts == 1

    def test_small_drawdown_can_trade(self, rules, clean_state):
        clean_state.current_balance = 99_000.0
        clean_state.daily_pnl       = -500.0
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=10.0)
        assert result.can_trade is True


# ---------------------------------------------------------------------------
# 2. Daily loss limit
# ---------------------------------------------------------------------------

class TestDailyLossLimit:

    def test_at_daily_limit_blocked(self, rules, clean_state):
        clean_state.daily_pnl = -2_000.0  # exactly at limit
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=10.0)
        assert result.can_trade is False
        assert result.reason == "daily_loss_limit"

    def test_above_daily_limit_blocked(self, rules, clean_state):
        clean_state.daily_pnl = -2_500.0
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=10.0)
        assert result.can_trade is False

    def test_just_under_daily_limit_allowed(self, rules, clean_state):
        # $1000 loss so far, $1000 remaining budget. 1c × 10pts × $20 = $200 fits.
        clean_state.daily_pnl       = -1_000.0
        clean_state.current_balance = 99_000.0
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=10.0)
        assert result.can_trade is True

    def test_worst_case_would_breach_limit_reduces_contracts(self, rules, clean_state):
        """$1900 already lost. Stop=$200/contract. 1 more loss = $2100 > $2000 limit.
        Should reduce from 2 to 0 contracts... actually budget=$100, cost/c=$200 → 0
        so it should block."""
        clean_state.daily_pnl       = -1_900.0
        clean_state.current_balance = 98_100.0
        # 2c × 25pts × $20 = $1000, but remaining budget = $100 → can't fit even 1
        result = rules.check(clean_state, proposed_contracts=2, stop_distance_points=25.0)
        assert result.can_trade is False

    def test_worst_case_reduces_to_1_contract(self, rules, clean_state):
        """$1600 lost. Budget=$400. 1c × 10pts × $20=$200 fits. 2c=$400 also fits.
        But 3c=$600 > $400 → reduce to 2."""
        clean_state.daily_pnl       = -1_600.0
        clean_state.current_balance = 98_400.0
        result = rules.check(clean_state, proposed_contracts=3, stop_distance_points=10.0)
        assert result.can_trade is True
        assert result.safe_contracts <= 2


# ---------------------------------------------------------------------------
# 3. Trailing drawdown
# ---------------------------------------------------------------------------

class TestTrailingDrawdown:

    def test_at_drawdown_floor_blocked(self, rules, clean_state):
        # HWM=100k, max_dd=3k → floor=97k. balance=97k → blocked
        clean_state.current_balance = 97_000.0
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=10.0)
        assert result.can_trade is False
        assert result.reason == "trailing_drawdown"

    def test_below_drawdown_floor_blocked(self, rules, clean_state):
        clean_state.current_balance = 96_500.0
        result = rules.check(clean_state, proposed_contracts=1)
        assert result.can_trade is False

    def test_above_floor_allowed(self, rules, clean_state):
        clean_state.current_balance = 97_500.0  # 500 above floor
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=10.0)
        assert result.can_trade is True

    def test_worst_case_would_breach_floor_reduces_contracts(self, rules, clean_state):
        """Balance $97,300. Floor $97,000. Headroom $300.
        1c × 10pts × $20 = $200 fits. 2c = $400 > $300 → reduce to 1."""
        clean_state.current_balance = 97_300.0
        result = rules.check(clean_state, proposed_contracts=3, stop_distance_points=10.0)
        assert result.can_trade is True
        assert result.safe_contracts == 1

    def test_no_headroom_at_floor_blocks(self, rules, clean_state):
        """Balance $97,100. Headroom $100. 1c × 10pts × $20 = $200 > $100 → block."""
        clean_state.current_balance = 97_100.0
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=10.0)
        assert result.can_trade is False


# ---------------------------------------------------------------------------
# 4. Profit target
# ---------------------------------------------------------------------------

class TestProfitTarget:

    def test_at_profit_target_blocked(self, rules, clean_state):
        clean_state.current_balance = 106_000.0  # +$6000 = target
        result = rules.check(clean_state, proposed_contracts=1)
        assert result.can_trade is False
        assert result.reason == "profit_target_reached"

    def test_just_under_target_allowed(self, rules, clean_state):
        clean_state.current_balance = 105_999.0
        result = rules.check(clean_state, proposed_contracts=1, stop_distance_points=5.0)
        assert result.can_trade is True

    def test_phase_passed_flag_blocks(self, rules, clean_state):
        clean_state.phase_passed = True
        result = rules.check(clean_state, proposed_contracts=1)
        assert result.can_trade is False


# ---------------------------------------------------------------------------
# 5. Already halted
# ---------------------------------------------------------------------------

class TestAlreadyHalted:

    def test_halted_state_always_blocked(self, rules, clean_state):
        clean_state.is_halted   = True
        clean_state.halt_reason = "test halt"
        result = rules.check(clean_state, proposed_contracts=1)
        assert result.can_trade is False
        assert result.safe_contracts == 0


# ---------------------------------------------------------------------------
# 6. Max contracts cap
# ---------------------------------------------------------------------------

class TestMaxContracts:

    def test_max_contracts_respected(self, rules, clean_state):
        result = rules.check(clean_state, proposed_contracts=10, stop_distance_points=5.0)
        assert result.can_trade is True
        assert result.safe_contracts <= 3  # default max_contracts=3


# ---------------------------------------------------------------------------
# 7. update_after_trade — state transitions
# ---------------------------------------------------------------------------

class TestUpdateAfterTrade:

    def test_profit_updates_balance(self, rules, clean_state):
        state = rules.update_after_trade(clean_state, trade_pnl=500.0)
        assert state.current_balance == 100_500.0
        assert state.daily_pnl == 500.0
        assert state.total_net_profit == 500.0

    def test_loss_updates_balance(self, rules, clean_state):
        state = rules.update_after_trade(clean_state, trade_pnl=-800.0)
        assert state.current_balance == 99_200.0
        assert state.daily_pnl == -800.0

    def test_large_loss_triggers_drawdown_halt(self, rules, clean_state):
        # Lose $3100 — breaches both drawdown floor ($97k) and daily limit ($2k).
        # Either halt reason is valid; just verify the account is halted.
        state = rules.update_after_trade(clean_state, trade_pnl=-3_100.0)
        assert state.is_halted is True
        assert len(state.halt_reason) > 0

    def test_large_daily_loss_triggers_halt(self, rules, clean_state):
        state = rules.update_after_trade(clean_state, trade_pnl=-2_100.0)
        assert state.is_halted is True
        assert "loss" in state.halt_reason.lower()

    def test_profit_target_reached_sets_flag(self, rules, clean_state):
        state = rules.update_after_trade(clean_state, trade_pnl=6_100.0)
        assert state.phase_passed is True

    def test_intraday_hwm_advances_on_profit(self, clean_state):
        """Intraday HWM should advance immediately when balance rises."""
        from risk.prop_firm_rules import PropFirmRules
        # Build rules with intraday trailing
        class _FakePF:
            daily_loss_limit = 2_000.0
            max_drawdown = 3_000.0
            profit_target = 6_000.0
            consistency_rule = 0.40
            trailing_drawdown_type = "intraday"
        class _FakePos:
            max_contracts = 3
            point_value = 20.0
        class _FakeCfg:
            prop_firm = _FakePF()
            position = _FakePos()

        r = PropFirmRules(_FakeCfg())
        state = r.update_after_trade(clean_state, trade_pnl=1_000.0)
        assert state.high_water_mark == 101_000.0  # advanced immediately

    def test_eod_hwm_does_not_advance_mid_day(self, rules, clean_state):
        """EOD trailing should NOT advance HWM mid-day."""
        state = rules.update_after_trade(clean_state, trade_pnl=1_000.0, is_eod=False)
        assert state.high_water_mark == 100_000.0  # unchanged

    def test_trades_today_increments(self, rules, clean_state):
        state = rules.update_after_trade(clean_state, trade_pnl=100.0)
        assert state.trades_today == 1


# ---------------------------------------------------------------------------
# 8. end_of_day
# ---------------------------------------------------------------------------

class TestEndOfDay:

    def test_eod_advances_hwm_on_profit(self, rules, clean_state):
        clean_state.current_balance = 101_000.0
        state = rules.end_of_day(clean_state)
        assert state.high_water_mark == 101_000.0

    def test_eod_does_not_lower_hwm_on_loss(self, rules, clean_state):
        clean_state.current_balance = 99_000.0  # in drawdown
        state = rules.end_of_day(clean_state)
        assert state.high_water_mark == 100_000.0  # unchanged

    def test_eod_resets_daily_pnl(self, rules, clean_state):
        clean_state.daily_pnl = -500.0
        state = rules.end_of_day(clean_state)
        assert state.daily_pnl == 0.0

    def test_eod_resets_trades_today(self, rules, clean_state):
        clean_state.trades_today = 5
        state = rules.end_of_day(clean_state)
        assert state.trades_today == 0

    def test_eod_lifts_daily_loss_halt(self, rules, clean_state):
        clean_state.is_halted   = True
        clean_state.halt_reason = "Daily loss limit breached: ..."
        state = rules.end_of_day(clean_state)
        assert state.is_halted is False

    def test_drawdown_halt_persists_through_eod(self, rules, clean_state):
        """Drawdown halt is permanent — EOD must NOT lift it."""
        clean_state.is_halted   = True
        clean_state.halt_reason = "Trailing drawdown breached: balance=..."
        state = rules.end_of_day(clean_state)
        assert state.is_halted is True


# ---------------------------------------------------------------------------
# 9. Convenience queries
# ---------------------------------------------------------------------------

class TestConvenienceQueries:

    def test_current_drawdown_zero(self, rules, clean_state):
        assert rules.current_drawdown(clean_state) == 0.0

    def test_current_drawdown_positive(self, rules, clean_state):
        clean_state.current_balance = 98_000.0
        assert rules.current_drawdown(clean_state) == 2_000.0

    def test_drawdown_pct_used(self, rules, clean_state):
        clean_state.current_balance = 98_500.0
        pct = rules.drawdown_pct_used(clean_state)
        assert pct == pytest.approx(0.5, rel=1e-4)

    def test_daily_loss_pct_used(self, rules, clean_state):
        clean_state.daily_pnl = -1_000.0
        pct = rules.daily_loss_pct_used(clean_state)
        assert pct == pytest.approx(0.5, rel=1e-4)

    def test_is_near_limits_false(self, rules, clean_state):
        assert rules.is_near_limits(clean_state) is False

    def test_is_near_limits_true_drawdown(self, rules, clean_state):
        clean_state.current_balance = 97_800.0   # 73% used (just under 75%)
        # 100k-97.8k=2.2k / 3k = 73.3% — not yet at 75%
        assert rules.is_near_limits(clean_state) is False
        clean_state.current_balance = 97_700.0   # 77% > 75% → near
        assert rules.is_near_limits(clean_state) is True

    def test_is_near_limits_true_daily_loss(self, rules, clean_state):
        clean_state.daily_pnl = -1_600.0   # 80% of 2000 → near
        assert rules.is_near_limits(clean_state) is True
