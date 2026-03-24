"""
Layer 2: Hidden Markov Model Regime Detector
Classifies market into 4 regimes: trending, ranging, volatile, breakout.
Analogous to identifying high/low pressure systems in weather forecasting.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm


REGIMES = {0: "trending", 1: "ranging", 2: "volatile", 3: "breakout"}


class HMMRegimeDetector:
    def __init__(self, n_states: int = 4, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self._regime_labels = REGIMES

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from OHLCV data."""
        log_returns = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        volatility = log_returns.rolling(20).std().fillna(0)
        volume_ratio = (df["volume"] / df["volume"].rolling(20).mean()).fillna(1)
        return np.column_stack([log_returns, volatility, volume_ratio])

    def fit(self, df: pd.DataFrame):
        """Train HMM on historical OHLCV data."""
        features = self._build_features(df)
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
        )
        self.model.fit(features)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return regime label series for given OHLCV data."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        features = self._build_features(df)
        state_sequence = self.model.predict(features)
        return pd.Series(state_sequence, index=df.index).map(self._regime_labels)

    def current_regime(self, df: pd.DataFrame) -> str:
        """Return the most recent regime label."""
        return self.predict(df).iloc[-1]
