"""
Backtesting engine — placeholder for Phase 5.
Will run walk-forward validation using the 5-layer signal pipeline.
"""


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config

    def run(self, df, initial_equity: float = 10_000.0) -> dict:
        raise NotImplementedError("Backtest engine coming in Phase 5.")
