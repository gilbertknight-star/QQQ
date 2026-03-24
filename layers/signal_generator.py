"""
Layer 4: Signal Generator
Combines all three layers. A trade signal is only generated when
Kalman filter, HMM regime, and market structure simultaneously agree.
"""

import pandas as pd
from .kalman_filter import KalmanFilter
from .hmm_regime import HMMRegimeDetector
from .market_structure import MarketStructure


TRADEABLE_REGIMES = {"trending", "breakout"}


class SignalGenerator:
    def __init__(self, config: dict):
        self.kalman = KalmanFilter(
            process_noise=config["kalman_filter"]["process_noise"],
            measurement_noise=config["kalman_filter"]["measurement_noise"],
        )
        self.hmm = HMMRegimeDetector(
            n_states=config["hmm"]["n_states"],
            n_iter=config["hmm"]["n_iter"],
        )
        self.structure = MarketStructure(
            order_block_lookback=config["market_structure"]["order_block_lookback"],
            volume_threshold=config["market_structure"]["volume_threshold"],
        )
        self.min_rr = config["signal"]["min_rr_ratio"]

    def train(self, df: pd.DataFrame):
        """Train HMM on historical data."""
        self.hmm.fit(df)
        return self

    def generate(self, df: pd.DataFrame) -> dict:
        """
        Generate a signal for the latest bar.
        Returns dict with signal direction ('long', 'short', or None) and metadata.
        """
        # Layer 1: Kalman
        kalman_result = self.kalman.run(df["close"])
        trend_dir = kalman_result["trend_direction"].iloc[-1]
        velocity = kalman_result["velocity"].iloc[-1]

        # Layer 2: HMM regime
        regime = self.hmm.current_regime(df)

        # Layer 3: Structure
        current_price = df["close"].iloc[-1]
        structure_score = self.structure.score(df, current_price)

        # All layers must agree
        regime_ok = regime in TRADEABLE_REGIMES
        trend_ok = abs(velocity) > 0  # non-zero velocity
        structure_ok = structure_score["distance_to_node"] < current_price * 0.005  # within 0.5%

        direction = None
        if regime_ok and trend_ok and structure_ok:
            direction = "long" if trend_dir > 0 else "short"

        return {
            "direction": direction,
            "regime": regime,
            "trend_direction": trend_dir,
            "velocity": velocity,
            "filtered_price": kalman_result["filtered_price"].iloc[-1],
            "structure": structure_score,
        }
