"""
tests/test_drawdown_monitor.py — Unit tests for DrawdownMonitor
"""

import pytest
from risk.drawdown_monitor import DrawdownMonitor, DrawdownState, DrawdownAlert


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monitor():
    return DrawdownMonitor()


@pytest.fixture
def state():
    return DrawdownState(
        account_balance = 100_000.0,
        high_water_mark = 100_000.0,
        max_drawdown    = 3_000.0,
    )


# ---------------------------------------------------------------------------
# 1. DrawdownState properties
# ---------------------------------------------------------------------------

class TestDrawdownStateProps:

    def test_drawdown_pct_zero(self, state):
        assert state.drawdown_pct == 0.0

    def test_drawdown_pct_half(self, state):
        state.current_drawdown = 1_500.0
        assert state.drawdown_pct == pytest.approx(0.5)

    def test_drawdown_floor(self, state):
        assert state.drawdown_floor == 97_000.0

    def test_headroom_full(self, state):
        assert state.headroom == 3_000.0

    def test_headroom_partial(self, state):
        state.account_balance  = 98_500.0
        state.current_drawdown = 1_500.0
        assert state.headroom == pytest.approx(1_500.0)

    def test_headroom_at_floor(self, state):
        state.account_balance  = 97_000.0
        state.current_drawdown = 3_000.0
        assert state.headroom == 0.0


# ---------------------------------------------------------------------------
# 2. update() — equity tracking
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_equity_rise_advances_hwm(self, monitor, state):
        state = monitor.update(state, current_equity=101_000.0)
        assert state.high_water_mark == 101_000.0
        assert state.current_drawdown == 0.0

    def test_equity_drop_sets_drawdown(self, monitor, state):
        state = monitor.update(state, current_equity=98_500.0)
        assert state.current_drawdown == pytest.approx(1_500.0)

    def test_peak_intraday_drawdown_tracked(self, monitor, state):
        monitor.update(state, current_equity=98_000.0)
        monitor.update(state, current_equity=98_500.0)   # partial recovery
        assert state.peak_intraday_drawdown == pytest.approx(2_000.0)

    def test_equity_history_appended(self, monitor, state):
        monitor.update(state, current_equity=99_000.0)
        monitor.update(state, current_equity=99_500.0)
        assert len(state.equity_history) == 2

    def test_equity_history_capped_at_20(self, monitor, state):
        for i in range(25):
            monitor.update(state, current_equity=100_000.0 - i * 10)
        assert len(state.equity_history) == 20

    def test_trade_pnl_loss_increments_streak(self, monitor, state):
        monitor.update(state, current_equity=99_500.0, trade_pnl=-500.0)
        assert state.consecutive_losses == 1
        assert state.losses_today == 1
        assert state.trades_today == 1

    def test_trade_pnl_win_resets_streak(self, monitor, state):
        state.consecutive_losses = 2
        monitor.update(state, current_equity=100_200.0, trade_pnl=200.0)
        assert state.consecutive_losses == 0

    def test_bar_update_no_trade_pnl_no_streak_change(self, monitor, state):
        state.consecutive_losses = 2
        monitor.update(state, current_equity=99_000.0, trade_pnl=None)
        assert state.consecutive_losses == 2  # unchanged


# ---------------------------------------------------------------------------
# 3. get_alert() — tier classification
# ---------------------------------------------------------------------------

class TestGetAlert:

    def test_ok_at_zero_drawdown(self, monitor, state):
        alert = monitor.get_alert(state)
        assert alert.level == "OK"
        assert alert.should_stop_entries is False
        assert alert.should_flatten is False
        assert alert.size_multiplier == 1.0

    def test_watch_above_50_pct(self, monitor, state):
        state.current_drawdown = 1_600.0  # 53.3%
        alert = monitor.get_alert(state)
        assert alert.level == "WATCH"
        assert alert.size_multiplier == 0.75

    def test_caution_above_65_pct(self, monitor, state):
        state.current_drawdown = 2_000.0  # 66.7%
        alert = monitor.get_alert(state)
        assert alert.level == "CAUTION"
        assert alert.size_multiplier == 0.50

    def test_danger_above_80_pct(self, monitor, state):
        state.current_drawdown = 2_500.0  # 83.3%
        alert = monitor.get_alert(state)
        assert alert.level == "DANGER"
        assert alert.should_stop_entries is True
        assert alert.size_multiplier == 0.0

    def test_critical_above_90_pct(self, monitor, state):
        state.current_drawdown = 2_800.0  # 93.3%
        alert = monitor.get_alert(state)
        assert alert.level == "CRITICAL"
        assert alert.should_flatten is True
        assert alert.size_multiplier == 0.0

    def test_alert_fields_populated(self, monitor, state):
        state.current_drawdown = 1_000.0
        alert = monitor.get_alert(state)
        assert alert.drawdown_dollars == 1_000.0
        assert alert.drawdown_pct == pytest.approx(1_000.0 / 3_000.0, rel=1e-4)
        assert alert.headroom > 0
        assert isinstance(alert.detail, str) and len(alert.detail) > 0

    def test_alert_str_no_error(self, monitor, state):
        state.current_drawdown = 500.0
        alert = monitor.get_alert(state)
        s = str(alert)
        assert "OK" in s or "WATCH" in s


# ---------------------------------------------------------------------------
# 4. Custom tiers
# ---------------------------------------------------------------------------

class TestCustomTiers:

    def test_custom_tiers_override_defaults(self):
        m = DrawdownMonitor(custom_tiers={"WATCH": 0.30, "CAUTION": 0.50,
                                           "DANGER": 0.70, "CRITICAL": 0.85})
        s = DrawdownState(account_balance=100_000.0,
                          high_water_mark=100_000.0,
                          max_drawdown=3_000.0)
        s.current_drawdown = 1_000.0  # 33.3% > 30% WATCH threshold
        alert = m.get_alert(s)
        assert alert.level == "WATCH"

    def test_custom_tiers_more_aggressive(self):
        """Tighter thresholds → alerts fire sooner."""
        m = DrawdownMonitor(custom_tiers={"WATCH": 0.10})
        s = DrawdownState(account_balance=100_000.0,
                          high_water_mark=100_000.0,
                          max_drawdown=3_000.0)
        s.current_drawdown = 400.0   # 13.3% > 10% → WATCH
        alert = m.get_alert(s)
        assert alert.level in ("WATCH", "CAUTION", "DANGER", "CRITICAL")


# ---------------------------------------------------------------------------
# 5. Losing streak
# ---------------------------------------------------------------------------

class TestLosingStreak:

    def test_no_streak(self, monitor, state):
        assert monitor.is_losing_streak(state) is False

    def test_at_threshold(self, monitor, state):
        state.consecutive_losses = 3  # default threshold = 3
        assert monitor.is_losing_streak(state) is True

    def test_below_threshold(self, monitor, state):
        state.consecutive_losses = 2
        assert monitor.is_losing_streak(state) is False


# ---------------------------------------------------------------------------
# 6. Drawdown velocity
# ---------------------------------------------------------------------------

class TestDrawdownVelocity:

    def test_velocity_no_history(self, monitor, state):
        assert monitor.drawdown_velocity(state) == 0.0

    def test_velocity_one_observation(self, monitor, state):
        state.equity_history = [100_000.0]
        assert monitor.drawdown_velocity(state) == 0.0

    def test_velocity_declining(self, monitor, state):
        # Equity falling $100 per bar over 5 bars
        state.equity_history = [100_000, 99_900, 99_800, 99_700, 99_600]
        v = monitor.drawdown_velocity(state)
        assert v == pytest.approx(-100.0)

    def test_velocity_rising(self, monitor, state):
        state.equity_history = [100_000, 100_200, 100_400]
        v = monitor.drawdown_velocity(state)
        assert v == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# 7. end_of_day reset
# ---------------------------------------------------------------------------

class TestEndOfDay:

    def test_eod_resets_intraday_stats(self, monitor, state):
        state.peak_intraday_drawdown = 1_000.0
        state.trades_today           = 5
        state.losses_today           = 2
        state.equity_history         = [100_000.0, 99_000.0]
        state.last_alert_level       = "CAUTION"

        state = monitor.end_of_day(state)
        assert state.peak_intraday_drawdown == 0.0
        assert state.trades_today == 0
        assert state.losses_today == 0
        assert state.equity_history == []
        assert state.last_alert_level == "OK"

    def test_eod_preserves_balance_and_hwm(self, monitor, state):
        state.account_balance  = 99_000.0
        state.high_water_mark  = 100_500.0
        state.current_drawdown = 1_500.0

        state = monitor.end_of_day(state)
        assert state.account_balance  == 99_000.0
        assert state.high_water_mark  == 100_500.0
        assert state.current_drawdown == 1_500.0

    def test_eod_preserves_consecutive_losses(self, monitor, state):
        """Losing streak carries across days — not reset at EOD."""
        state.consecutive_losses = 3
        state = monitor.end_of_day(state)
        assert state.consecutive_losses == 3
