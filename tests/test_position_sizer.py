"""
tests/test_position_sizer.py — Unit tests for PositionSizer

Covers:
  - Base contract calculation
  - Drawdown tier multipliers (all 4 buckets)
  - Regime multipliers (trending vs breakout)
  - Contract clamping (min 1, max config.max_contracts)
  - Micro equivalent (MNQ = NQ × 10)
  - validate_size() — 30% daily limit guard
  - safe_size() — auto-reduction to valid count
  - Degenerate input guard (zero stop, zero balance)
  - SizingResult dataclass values
"""

import pytest
from execution.position_sizer import (
    PositionSizer,
    SizingResult,
    _drawdown_fraction,
    _get_dd_multiplier,
    NQ_POINT_VALUE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sizer():
    """Default sizer: 1% risk, max 3 contracts, $20/point."""
    return PositionSizer()


# ---------------------------------------------------------------------------
# 1. Drawdown fraction helpers
# ---------------------------------------------------------------------------

class TestDrawdownFraction:

    def test_zero_drawdown(self):
        assert _drawdown_fraction(0.0, 2000.0) == 0.0

    def test_full_drawdown(self):
        assert _drawdown_fraction(2000.0, 2000.0) == 1.0

    def test_half_drawdown(self):
        assert _drawdown_fraction(1000.0, 2000.0) == pytest.approx(0.5)

    def test_clamps_above_1(self):
        assert _drawdown_fraction(3000.0, 2000.0) == 1.0

    def test_zero_max_drawdown(self):
        assert _drawdown_fraction(500.0, 0.0) == 0.0


class TestDDMultiplier:

    def test_bucket_0_to_25(self):
        assert _get_dd_multiplier(0.00) == 1.00
        assert _get_dd_multiplier(0.10) == 1.00
        assert _get_dd_multiplier(0.25) == 1.00

    def test_bucket_25_to_50(self):
        assert _get_dd_multiplier(0.26) == 0.75
        assert _get_dd_multiplier(0.40) == 0.75
        assert _get_dd_multiplier(0.50) == 0.75

    def test_bucket_50_to_75(self):
        assert _get_dd_multiplier(0.51) == 0.50
        assert _get_dd_multiplier(0.65) == 0.50
        assert _get_dd_multiplier(0.75) == 0.50

    def test_bucket_75_to_100(self):
        assert _get_dd_multiplier(0.76) == 0.25
        assert _get_dd_multiplier(1.00) == 0.25


# ---------------------------------------------------------------------------
# 2. Base contract calculation
# ---------------------------------------------------------------------------

class TestBaseCalc:

    def test_standard_case(self, sizer):
        """50k account, 1% risk = $500. Stop 25pts × $20 = $500. → 1 NQ."""
        n = sizer.calculate_size(
            account_balance=50_000,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=2_000.0,
            regime="trending",
        )
        assert n == 1

    def test_larger_account(self, sizer):
        """100k account, 1% risk = $1000. Stop 25pts × $20 = $500. → 2 NQ."""
        n = sizer.calculate_size(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert n == 2

    def test_max_contracts_cap(self, sizer):
        """Very large account should be capped at max_contracts=3."""
        n = sizer.calculate_size(
            account_balance=500_000,
            stop_distance_points=10.0,
            current_drawdown=0.0,
            max_drawdown=10_000.0,
            regime="trending",
        )
        assert n == 3

    def test_minimum_1_contract(self, sizer):
        """Tiny account or huge stop should still return 1."""
        n = sizer.calculate_size(
            account_balance=5_000,
            stop_distance_points=100.0,
            current_drawdown=0.0,
            max_drawdown=2_000.0,
            regime="trending",
        )
        assert n == 1


# ---------------------------------------------------------------------------
# 3. Drawdown multiplier integration
# ---------------------------------------------------------------------------

class TestDrawdownMultiplier:

    def test_no_drawdown_full_size(self, sizer):
        """0% drawdown → 1.0x multiplier."""
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert result.dd_multiplier == 1.0

    def test_20pct_drawdown_full_size(self, sizer):
        """20% of max_dd used → still 1.0x (first bucket ≤25%)."""
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=600.0,   # 20% of 3000
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert result.dd_multiplier == 1.0

    def test_40pct_drawdown_reduced(self, sizer):
        """40% of max_dd used → 0.75x multiplier."""
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=1_200.0,  # 40% of 3000
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert result.dd_multiplier == 0.75

    def test_60pct_drawdown_halved(self, sizer):
        """60% of max_dd used → 0.50x multiplier."""
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=1_800.0,  # 60% of 3000
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert result.dd_multiplier == 0.50

    def test_80pct_drawdown_minimum(self, sizer):
        """80% of max_dd used → 0.25x multiplier."""
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=2_400.0,  # 80% of 3000
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert result.dd_multiplier == 0.25


# ---------------------------------------------------------------------------
# 4. Regime multiplier
# ---------------------------------------------------------------------------

class TestRegimeMultiplier:

    def test_trending_multiplier(self, sizer):
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert result.regime_multiplier == 1.0

    def test_breakout_multiplier(self, sizer):
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=3_000.0,
            regime="breakout",
        )
        assert result.regime_multiplier == 0.75

    def test_breakout_produces_smaller_size_than_trending(self, sizer):
        common = dict(
            account_balance=200_000,
            stop_distance_points=15.0,
            current_drawdown=0.0,
            max_drawdown=5_000.0,
        )
        trending = sizer.calculate_size(**common, regime="trending")
        breakout = sizer.calculate_size(**common, regime="breakout")
        assert breakout <= trending


# ---------------------------------------------------------------------------
# 5. Micro equivalent
# ---------------------------------------------------------------------------

class TestMicroEquivalent:

    def test_micro_conversion(self, sizer):
        assert sizer.get_micro_equivalent(1) == 10
        assert sizer.get_micro_equivalent(2) == 20
        assert sizer.get_micro_equivalent(3) == 30

    def test_should_use_micro_below_threshold(self, sizer):
        assert sizer.should_use_micro(24_999) is True
        assert sizer.should_use_micro(25_000) is False
        assert sizer.should_use_micro(50_000) is False

    def test_instrument_label(self, sizer):
        assert sizer.get_instrument(10_000) == "MNQ"
        assert sizer.get_instrument(50_000) == "NQ"


# ---------------------------------------------------------------------------
# 6. validate_size
# ---------------------------------------------------------------------------

class TestValidateSize:

    def test_valid_within_30_pct(self, sizer):
        """1 contract × 10pts × $20 = $200 ≤ 30% of $1000 ($300) → valid."""
        assert sizer.validate_size(
            contracts=1,
            daily_loss_limit=1_000.0,
            stop_distance_points=10.0,
        ) is True

    def test_invalid_exceeds_30_pct(self, sizer):
        """3 contracts × 25pts × $20 = $1500 > 30% of $1000 ($300) → invalid."""
        assert sizer.validate_size(
            contracts=3,
            daily_loss_limit=1_000.0,
            stop_distance_points=25.0,
        ) is False

    def test_exactly_at_30_pct(self, sizer):
        """Exactly at the 30% limit should be valid."""
        # budget = 1000 × 0.30 = 300
        # 1c × 15pts × $20 = $300 → valid (≤)
        assert sizer.validate_size(
            contracts=1,
            daily_loss_limit=1_000.0,
            stop_distance_points=15.0,
        ) is True

    def test_custom_point_value(self, sizer):
        """MNQ point value $2 — 1c × 50pts × $2 = $100 ≤ $300 → valid."""
        assert sizer.validate_size(
            contracts=1,
            daily_loss_limit=1_000.0,
            stop_distance_points=50.0,
            point_value=2.0,
        ) is True


# ---------------------------------------------------------------------------
# 7. safe_size
# ---------------------------------------------------------------------------

class TestSafeSize:

    def test_safe_size_no_change_needed(self, sizer):
        """1 contract × 10pts × $20 = $200 < $300 budget → unchanged."""
        result = sizer.safe_size(
            contracts=1,
            daily_loss_limit=1_000.0,
            stop_distance_points=10.0,
        )
        assert result == 1

    def test_safe_size_reduces(self, sizer):
        """3 contracts is too many; should be reduced to fit budget."""
        result = sizer.safe_size(
            contracts=3,
            daily_loss_limit=1_000.0,
            stop_distance_points=25.0,
        )
        # budget = 300; 1c × 25 × 20 = 500 > 300 → forced to 1 (minimum)
        assert result == 1

    def test_safe_size_minimum_1(self, sizer):
        """Even if the math says 0, must return at least 1."""
        result = sizer.safe_size(
            contracts=5,
            daily_loss_limit=100.0,
            stop_distance_points=200.0,
        )
        assert result >= 1


# ---------------------------------------------------------------------------
# 8. Degenerate inputs
# ---------------------------------------------------------------------------

class TestDegenerateInputs:

    def test_zero_stop_returns_1(self, sizer):
        n = sizer.calculate_size(
            account_balance=50_000,
            stop_distance_points=0.0,
            current_drawdown=0.0,
            max_drawdown=2_000.0,
            regime="trending",
        )
        assert n == 1

    def test_zero_balance_returns_1(self, sizer):
        n = sizer.calculate_size(
            account_balance=0.0,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=2_000.0,
            regime="trending",
        )
        assert n == 1


# ---------------------------------------------------------------------------
# 9. SizingResult fields
# ---------------------------------------------------------------------------

class TestSizingResult:

    def test_result_fields_populated(self, sizer):
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=3_000.0,
            regime="trending",
        )
        assert isinstance(result, SizingResult)
        assert result.contracts >= 1
        assert result.instrument in ("NQ", "MNQ")
        assert result.base_contracts > 0
        assert result.dd_multiplier > 0
        assert result.regime_multiplier > 0
        assert result.risk_per_trade > 0
        assert 0.0 <= result.dd_pct_used <= 1.0
        assert isinstance(result.reason, str) and len(result.reason) > 0

    def test_result_str_no_error(self, sizer):
        result = sizer.calculate_size_detailed(
            account_balance=50_000,
            stop_distance_points=12.5,
            current_drawdown=200.0,
            max_drawdown=2_000.0,
            regime="breakout",
        )
        s = str(result)
        assert "NQ" in s or "MNQ" in s

    def test_risk_per_trade_calculation(self, sizer):
        """risk_per_trade = contracts × stop_pts × point_value."""
        result = sizer.calculate_size_detailed(
            account_balance=100_000,
            stop_distance_points=25.0,
            current_drawdown=0.0,
            max_drawdown=3_000.0,
            regime="trending",
        )
        expected_risk = result.contracts * 25.0 * NQ_POINT_VALUE
        assert result.risk_per_trade == pytest.approx(expected_risk, rel=1e-4)
