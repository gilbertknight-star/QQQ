"""
Layer 1: Kalman Filter
Filters price noise to reveal true underlying trend and momentum.
Analogous to radar signal processing in weather forecasting.
"""

import numpy as np
import pandas as pd


class KalmanFilter:
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self.process_noise = process_noise      # Q
        self.measurement_noise = measurement_noise  # R

        # State: [price, velocity]
        self.F = np.array([[1, 1], [0, 1]])     # State transition
        self.H = np.array([[1, 0]])             # Observation model
        self.Q = np.eye(2) * self.process_noise
        self.R = np.array([[self.measurement_noise]])

        self._state = None
        self._cov = None

    def reset(self, initial_price: float):
        self._state = np.array([[initial_price], [0.0]])
        self._cov = np.eye(2) * 1.0

    def update(self, price: float) -> dict:
        """Single step update. Returns filtered estimate and confidence."""
        if self._state is None:
            self.reset(price)

        # Predict
        state_pred = self.F @ self._state
        cov_pred = self.F @ self._cov @ self.F.T + self.Q

        # Update
        innovation = price - (self.H @ state_pred)[0, 0]
        S = self.H @ cov_pred @ self.H.T + self.R
        K = cov_pred @ self.H.T @ np.linalg.inv(S)

        self._state = state_pred + K * innovation
        self._cov = (np.eye(2) - K @ self.H) @ cov_pred

        return {
            "filtered_price": self._state[0, 0],
            "velocity": self._state[1, 0],
            "uncertainty": np.sqrt(self._cov[0, 0]),
        }

    def run(self, prices: pd.Series) -> pd.DataFrame:
        """Run filter over a full price series. Returns DataFrame with results."""
        self.reset(prices.iloc[0])
        results = [self.update(p) for p in prices]
        df = pd.DataFrame(results, index=prices.index)
        df["trend_direction"] = np.sign(df["velocity"])
        return df
