"""
signals/hmm_regime.py — HMM Regime Detector (Core Signal Layer 1)

Classifies NQ market into four regimes using a Gaussian HMM trained on
log returns, rolling volatility, and relative volume.

Regimes:
  trending  — strong directional move, moderate volatility
  breakout  — elevated volume + elevated vol (institutional impulse)
  ranging   — low volatility, near-zero return (choppy/sideways)
  volatile  — highest volatility (news events, erratic)

Only trending and breakout are tradeable per the spec.

Usage:
    from signals.hmm_regime import HMMRegimeDetector
    from config import config

    det = HMMRegimeDetector(config)
    det.train(df_daily)                  # or det.load()
    result = det.predict(df_daily)
    tradeable = det.is_tradeable(result["regime"].iloc[-1],
                                 result["regime_confidence"].iloc[-1])
"""

from __future__ import annotations

import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger
from sklearn.preprocessing import StandardScaler

TRADEABLE_REGIMES = {"trending", "breakout"}
ALL_REGIMES = {"trending", "breakout", "ranging", "volatile"}


class HMMRegimeDetector:
    """
    Gaussian HMM regime detector for NQ futures.

    Train on daily OHLCV data (at least 6 months recommended; 18 months ideal).
    Predict regime + confidence on any OHLCV DataFrame.
    """

    def __init__(self, config=None):
        # Pull settings from config dataclass or use spec defaults
        if config is not None:
            cfg = config.hmm
            self._n_states       = cfg.n_states
            self._cov_type       = cfg.covariance_type
            self._n_iter         = cfg.n_iter
            self._random_state   = cfg.random_state
            self._vol_window     = cfg.vol_window
            self._vol_ratio_win  = cfg.volume_ratio_window
            self._retrain_days   = cfg.retrain_days
            self._model_path     = Path(cfg.model_path)
            self._scaler_path    = Path(cfg.scaler_path)
        else:
            self._n_states       = 4
            self._cov_type       = "full"
            self._n_iter         = 1_000
            self._random_state   = 42
            self._vol_window     = 20
            self._vol_ratio_win  = 20
            self._retrain_days   = 60
            self._model_path     = Path("models/hmm_model.pkl")
            self._scaler_path    = Path("models/hmm_scaler.pkl")

        self._model: Optional[hmm.GaussianHMM] = None
        self._scaler: Optional[StandardScaler] = None
        self._regime_map: dict[int, str] = {}   # state_int → regime_name
        self._last_trained: Optional[date] = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from OHLCV DataFrame.

        Returns DataFrame with columns:
          log_return         — log(close_t / close_{t-1})
          rolling_volatility — rolling std of log returns (window=vol_window)
          volume_ratio       — volume / rolling mean volume (window=vol_ratio_win)

        All NaN rows are dropped (first ~vol_window bars will be NaN).
        """
        feat = pd.DataFrame(index=df.index)

        feat["log_return"] = np.log(
            df["close"] / df["close"].shift(1)
        )
        feat["rolling_volatility"] = (
            feat["log_return"]
            .rolling(self._vol_window, min_periods=max(5, self._vol_window // 2))
            .std()
        )
        vol_mean = df["volume"].rolling(
            self._vol_ratio_win, min_periods=max(5, self._vol_ratio_win // 2)
        ).mean()
        feat["volume_ratio"] = df["volume"] / vol_mean.replace(0, np.nan)

        feat = feat.dropna()
        logger.debug(
            f"[hmm] build_features: {len(df)} bars in, "
            f"{len(feat)} usable rows ({len(df) - len(feat)} dropped for NaN)"
        )
        return feat

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit GaussianHMM on OHLCV DataFrame (use daily bars for best results).

        Saves model and scaler to disk. Updates _last_trained to today.

        Args:
            df: clean OHLCV DataFrame with columns [open, high, low, close, volume]

        Returns:
            self (for chaining)
        """
        features_df = self.build_features(df)
        X = features_df.values  # (n_samples, 3)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        logger.info(
            f"[hmm] Training GaussianHMM: n_states={self._n_states}, "
            f"n_iter={self._n_iter}, samples={len(X_scaled)}"
        )

        self._model = hmm.GaussianHMM(
            n_components=self._n_states,
            covariance_type=self._cov_type,
            n_iter=self._n_iter,
            random_state=self._random_state,
            verbose=False,
        )
        self._model.fit(X_scaled)

        self._regime_map = self.auto_label_regimes(self._model, X_scaled)
        self._last_trained = date.today()

        logger.info(
            f"[hmm] Training complete. Regime map: "
            + ", ".join(f"{v}={k}" for k, v in self._regime_map.items())
        )

        self._save()
        return self

    # ------------------------------------------------------------------
    # Auto-labelling
    # ------------------------------------------------------------------

    def auto_label_regimes(
        self,
        model: hmm.GaussianHMM,
        features_scaled: np.ndarray,
    ) -> dict[int, str]:
        """
        Infer which HMM state corresponds to which regime using learned means.

        Strategy (deterministic, order-independent):
          1. volatile  → state with the highest volatility mean (feature 1)
          2. breakout  → among remaining, highest volume_ratio mean (feature 2)
          3. ranging   → among remaining, lowest volatility mean (feature 1)
          4. trending  → the last remaining state

        Returns:
            dict mapping state_int → regime_name
        """
        means = model.means_  # shape (n_states, 3): [log_ret, vol, vol_ratio]
        remaining = set(range(model.n_components))
        labels: dict[int, str] = {}

        # 1. Volatile: highest volatility
        volatile_state = max(remaining, key=lambda s: means[s, 1])
        labels[volatile_state] = "volatile"
        remaining.remove(volatile_state)

        # 2. Breakout: highest volume ratio (institutional impulse)
        breakout_state = max(remaining, key=lambda s: means[s, 2])
        labels[breakout_state] = "breakout"
        remaining.remove(breakout_state)

        # 3. Ranging: lowest volatility (sideways / choppy)
        ranging_state = min(remaining, key=lambda s: means[s, 1])
        labels[ranging_state] = "ranging"
        remaining.remove(ranging_state)

        # 4. Trending: whatever is left
        trending_state = remaining.pop()
        labels[trending_state] = "trending"

        # Sanity check: log regime means for inspection
        for state_int, regime_name in sorted(labels.items()):
            m = means[state_int]
            logger.debug(
                f"[hmm]   state {state_int} → {regime_name:10s} | "
                f"log_ret={m[0]:.5f}  vol={m[1]:.5f}  vol_ratio={m[2]:.3f}"
            )

        return labels

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime and confidence for each bar in df.

        Returns DataFrame aligned to df's index with columns:
          regime             — string label (trending/breakout/ranging/volatile)
          regime_confidence  — probability of the predicted state (0–1)
          state_int          — raw HMM state integer

        Bars dropped during feature building (first ~vol_window) will not
        appear in the output — callers should be aware of this offset.
        """
        self._require_trained()

        features_df = self.build_features(df)
        X = features_df.values
        X_scaled = self._scaler.transform(X)

        state_sequence = self._model.predict(X_scaled)          # (n_samples,)
        state_proba = self._model.predict_proba(X_scaled)       # (n_samples, n_states)

        # Confidence = probability of the predicted state
        confidence = state_proba[np.arange(len(state_sequence)), state_sequence]

        result = pd.DataFrame(
            {
                "regime":            [self._regime_map[s] for s in state_sequence],
                "regime_confidence": confidence,
                "state_int":         state_sequence,
            },
            index=features_df.index,
        )

        logger.debug(
            f"[hmm] predict: {len(result)} bars | "
            f"latest={result['regime'].iloc[-1]} "
            f"({result['regime_confidence'].iloc[-1]:.2%})"
        )
        return result

    # ------------------------------------------------------------------
    # Tradeable check
    # ------------------------------------------------------------------

    def is_tradeable(
        self,
        regime: str,
        confidence: float,
        min_confidence: float = 0.65,
    ) -> bool:
        """
        Returns True only if:
          - regime is 'trending' or 'breakout'
          - confidence >= min_confidence

        This is the gate that prevents trading in choppy or erratic markets.
        """
        return regime in TRADEABLE_REGIMES and confidence >= min_confidence

    # ------------------------------------------------------------------
    # Retrain logic
    # ------------------------------------------------------------------

    def needs_retrain(
        self,
        last_trained_date: Optional[date] = None,
        retrain_days: Optional[int] = None,
    ) -> bool:
        """
        Returns True if the model is untrained or older than retrain_days.

        Args:
            last_trained_date: override; defaults to self._last_trained
            retrain_days:      override; defaults to self._retrain_days
        """
        check_date = last_trained_date or self._last_trained
        if check_date is None:
            return True
        days = retrain_days or self._retrain_days
        return (date.today() - check_date).days >= days

    def retrain_if_needed(self, df: pd.DataFrame) -> bool:
        """
        Retrain if model is missing or stale. Returns True if retrain occurred.

        Tries to load from disk first before training from scratch.
        """
        if self._model is None:
            loaded = self.load()
            if not loaded:
                logger.info("[hmm] No saved model found — training from scratch.")
                self.train(df)
                return True

        if self.needs_retrain():
            logger.info(
                f"[hmm] Model is {(date.today() - self._last_trained).days}d old "
                f"(retrain threshold: {self._retrain_days}d) — retraining."
            )
            self.train(df)
            return True

        logger.info(
            f"[hmm] Model is current (trained {self._last_trained}). No retrain needed."
        )
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(
                {
                    "model":        self._model,
                    "regime_map":   self._regime_map,
                    "last_trained": self._last_trained,
                },
                f,
            )
        with open(self._scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(f"[hmm] Model saved to {self._model_path}")

    def load(self) -> bool:
        """
        Load model and scaler from disk.

        Returns True on success, False if files don't exist.
        """
        if not self._model_path.exists() or not self._scaler_path.exists():
            logger.debug("[hmm] No saved model files found.")
            return False

        with open(self._model_path, "rb") as f:
            bundle = pickle.load(f)
        self._model        = bundle["model"]
        self._regime_map   = bundle["regime_map"]
        self._last_trained = bundle.get("last_trained")

        with open(self._scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        logger.info(
            f"[hmm] Loaded model from {self._model_path} "
            f"(trained {self._last_trained})"
        )
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if self._model is None or self._scaler is None:
            raise RuntimeError(
                "[hmm] Model not trained. Call train(df) or load() first."
            )

    def regime_distribution(self, predict_result: pd.DataFrame) -> pd.Series:
        """Return fraction of bars in each regime — useful for sanity checks."""
        return predict_result["regime"].value_counts(normalize=True).sort_index()

    @property
    def is_trained(self) -> bool:
        return self._model is not None and self._scaler is not None
