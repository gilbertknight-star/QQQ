"""
Layer 5: Risk Manager
Non-negotiable hard stops. All parameters are enforced without exception.
Position sizing, daily loss limits, drawdown controls.
"""


class RiskManager:
    def __init__(self, config: dict):
        cfg = config["risk"]
        self.max_risk_per_trade = cfg["max_risk_per_trade"]       # 1%
        self.daily_loss_limit = cfg["daily_loss_limit"]            # 3%
        self.drawdown_reduce_threshold = cfg["drawdown_reduce_threshold"]  # 5%
        self.max_drawdown_halt = cfg["max_drawdown_halt"]          # 10%
        self.max_concurrent = cfg["max_concurrent_positions"]      # 2

        self._daily_pnl = 0.0
        self._peak_equity = None
        self._halted = False

    def reset_daily(self):
        self._daily_pnl = 0.0

    def update_equity(self, equity: float):
        if self._peak_equity is None:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    @property
    def current_drawdown(self) -> float:
        if self._peak_equity is None or self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def is_halted(self) -> bool:
        return self._halted

    def check_halt(self, equity: float) -> bool:
        """Returns True if trading should be halted."""
        self.update_equity(equity)
        self._current_equity = equity

        dd = self.current_drawdown
        if dd >= self.max_drawdown_halt:
            self._halted = True

        if self._daily_pnl <= -self.daily_loss_limit:
            self._halted = True

        return self._halted

    def position_size(self, account_equity: float, stop_distance: float, tick_value: float) -> int:
        """
        Calculate number of contracts based on volatility-adjusted risk.
        stop_distance: price distance to stop loss
        tick_value: dollar value per point (MNQ = $2/point)
        """
        if self._halted:
            return 0

        risk_dollars = account_equity * self.max_risk_per_trade

        # Apply 50% size reduction if in drawdown zone
        dd = self.current_drawdown if hasattr(self, "_current_equity") else 0
        if dd >= self.drawdown_reduce_threshold:
            risk_dollars *= 0.5

        if stop_distance <= 0 or tick_value <= 0:
            return 0

        contracts = int(risk_dollars / (stop_distance * tick_value))
        return max(0, contracts)

    def record_trade_pnl(self, pnl: float):
        self._daily_pnl += pnl
