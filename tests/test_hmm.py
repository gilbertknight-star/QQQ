"""
tests/test_hmm.py — Unit tests for HMMRegimeDetector

Tests: train, predict, auto_label_regimes, is_tradeable,
       needs_retrain, save/load — on real NQ daily data via yfinance.

Run with:
    pytest tests/test_hmm.py -v
"""

from __future__ import annotations

import pickle
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import Config
from data.pipeline import DataPipeline
from signals.hmm_regime import HMMRegimeDetector, TRADEABLE_REGIMES, ALL_REGIMES


# ---------------------------------------------------------------------------
# Shared fixture: 2 years of NQ daily bars (cached after first run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def daily_df() -> pd.DataFrame:
    """Load 2yr NQ daily bars from yfinance (uses Parquet cache on repeat runs)."""
    cfg = Config()
    pipe = DataPipeline(cfg)
    df = pipe.get("NQ", "1d", source="yfinance", lookback_days=730)
    assert len(df) >= 200, "Need at least 200 daily bars for meaningful HMM tests"
    return df


@pytest.fixture(scope="module")
def trained_detector(daily_df) -> HMMRegimeDetector:
    """Detector trained on the shared daily DataFrame."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config()
        cfg.hmm.model_path  = str(Path(tmp) / "hmm_model.pkl")
        cfg.hmm.scaler_path = str(Path(tmp) / "hmm_scaler.pkl")
        det = HMMRegimeDetector(cfg)
        det.train(daily_df)
    return det


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_output_columns(self, trained_detector, daily_df):
        feat = trained_detector.build_features(daily_df)
        assert list(feat.columns) == ["log_return", "rolling_volatility", "volume_ratio"]

    def test_no_nans(self, trained_detector, daily_df):
        feat = trained_detector.build_features(daily_df)
        assert feat.isna().sum().sum() == 0, "Feature matrix must have no NaNs"

    def test_row_count_reduced(self, trained_detector, daily_df):
        feat = trained_detector.build_features(daily_df)
        # First ~vol_window rows dropped due to rolling calculations
        assert len(feat) < len(daily_df)
        assert len(feat) >= len(daily_df) - 25  # should only lose ~vol_window rows

    def test_log_returns_bounded(self, trained_detector, daily_df):
        feat = trained_detector.build_features(daily_df)
        # NQ log returns should be within ±10% per day (extremely generous bound)
        assert (feat["log_return"].abs() <= 0.10).all(), "Suspicious log returns"

    def test_volume_ratio_positive(self, trained_detector, daily_df):
        feat = trained_detector.build_features(daily_df)
        assert (feat["volume_ratio"] > 0).all()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTrain:
    def test_is_trained_after_train(self, trained_detector):
        assert trained_detector.is_trained

    def test_regime_map_has_all_four(self, trained_detector):
        assert set(trained_detector._regime_map.values()) == ALL_REGIMES

    def test_regime_map_keys_are_valid_states(self, trained_detector):
        n = trained_detector._n_states
        assert set(trained_detector._regime_map.keys()) == set(range(n))

    def test_last_trained_set(self, trained_detector):
        assert trained_detector._last_trained is not None
        assert trained_detector._last_trained == date.today()


# ---------------------------------------------------------------------------
# Auto-label
# ---------------------------------------------------------------------------

class TestAutoLabel:
    def test_exactly_four_labels(self, trained_detector, daily_df):
        feat = trained_detector.build_features(daily_df)
        regime_map = trained_detector.auto_label_regimes(
            trained_detector._model, trained_detector._scaler.transform(feat.values)
        )
        assert set(regime_map.values()) == ALL_REGIMES

    def test_no_duplicate_labels(self, trained_detector, daily_df):
        feat = trained_detector.build_features(daily_df)
        regime_map = trained_detector.auto_label_regimes(
            trained_detector._model, trained_detector._scaler.transform(feat.values)
        )
        labels = list(regime_map.values())
        assert len(labels) == len(set(labels)), "Each regime must map to exactly one state"

    def test_volatile_has_highest_volatility(self, trained_detector):
        """The state labelled 'volatile' should have the highest volatility mean."""
        means = trained_detector._model.means_
        rm = trained_detector._regime_map
        volatile_state = next(k for k, v in rm.items() if v == "volatile")
        other_states = [k for k in rm if k != volatile_state]
        assert all(
            means[volatile_state, 1] >= means[s, 1] for s in other_states
        ), "Volatile state must have the highest volatility mean"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_output_columns(self, trained_detector, daily_df):
        result = trained_detector.predict(daily_df)
        assert set(result.columns) >= {"regime", "regime_confidence", "state_int"}

    def test_all_regimes_are_valid(self, trained_detector, daily_df):
        result = trained_detector.predict(daily_df)
        assert result["regime"].isin(ALL_REGIMES).all()

    def test_confidence_in_range(self, trained_detector, daily_df):
        result = trained_detector.predict(daily_df)
        assert (result["regime_confidence"] >= 0).all()
        assert (result["regime_confidence"] <= 1).all()

    def test_no_single_regime_dominates(self, trained_detector, daily_df):
        """No regime should account for more than 60% of bars."""
        result = trained_detector.predict(daily_df)
        dist = result["regime"].value_counts(normalize=True)
        assert (dist <= 0.60).all(), (
            f"A regime dominates (>60%):\n{dist}\n"
            "Check HMM vol_window or n_iter — model may not have converged."
        )

    def test_all_four_regimes_detected(self, trained_detector, daily_df):
        """All four regimes must appear in predictions over 2 years of data."""
        result = trained_detector.predict(daily_df)
        detected = set(result["regime"].unique())
        assert detected == ALL_REGIMES, (
            f"Not all regimes detected: {ALL_REGIMES - detected} missing\n"
            "Distribution:\n" + str(result["regime"].value_counts())
        )

    def test_index_aligned_to_features(self, trained_detector, daily_df):
        """Predict output index must be a subset of (and aligned with) input index."""
        result = trained_detector.predict(daily_df)
        assert result.index.isin(daily_df.index).all()

    def test_latest_bar_accessible(self, trained_detector, daily_df):
        result = trained_detector.predict(daily_df)
        last = result.iloc[-1]
        assert last["regime"] in ALL_REGIMES
        assert 0 <= last["regime_confidence"] <= 1


# ---------------------------------------------------------------------------
# is_tradeable
# ---------------------------------------------------------------------------

class TestIsTradeable:
    @pytest.mark.parametrize("regime,conf,expected", [
        ("trending",  0.80, True),
        ("breakout",  0.70, True),
        ("trending",  0.65, True),   # exactly at threshold
        ("trending",  0.64, False),  # just below threshold
        ("ranging",   0.90, False),
        ("volatile",  0.95, False),
        ("trending",  0.00, False),
    ])
    def test_tradeable_cases(self, trained_detector, regime, conf, expected):
        assert trained_detector.is_tradeable(regime, conf) == expected

    def test_custom_confidence_threshold(self, trained_detector):
        assert trained_detector.is_tradeable("trending", 0.50, min_confidence=0.50)
        assert not trained_detector.is_tradeable("trending", 0.49, min_confidence=0.50)


# ---------------------------------------------------------------------------
# needs_retrain
# ---------------------------------------------------------------------------

class TestNeedsRetrain:
    def test_needs_retrain_if_untrained(self):
        det = HMMRegimeDetector()
        assert det.needs_retrain() is True

    def test_no_retrain_if_fresh(self, trained_detector):
        assert trained_detector.needs_retrain() is False

    def test_needs_retrain_after_threshold(self, trained_detector):
        old_date = date.today() - timedelta(days=61)
        assert trained_detector.needs_retrain(last_trained_date=old_date, retrain_days=60)

    def test_no_retrain_at_threshold_minus_one(self, trained_detector):
        recent = date.today() - timedelta(days=59)
        assert not trained_detector.needs_retrain(last_trained_date=recent, retrain_days=60)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, daily_df):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.hmm.model_path  = str(Path(tmp) / "hmm_model.pkl")
            cfg.hmm.scaler_path = str(Path(tmp) / "hmm_scaler.pkl")

            det1 = HMMRegimeDetector(cfg)
            det1.train(daily_df)
            map1 = det1._regime_map.copy()
            pred1 = det1.predict(daily_df)

            # Load into a fresh instance
            det2 = HMMRegimeDetector(cfg)
            ok = det2.load()
            assert ok, "load() should return True when files exist"
            assert det2._regime_map == map1

            pred2 = det2.predict(daily_df)
            pd.testing.assert_frame_equal(pred1, pred2)

    def test_load_returns_false_when_missing(self):
        det = HMMRegimeDetector()
        det._model_path  = Path("/nonexistent/model.pkl")
        det._scaler_path = Path("/nonexistent/scaler.pkl")
        assert det.load() is False


# ---------------------------------------------------------------------------
# retrain_if_needed
# ---------------------------------------------------------------------------

class TestRetrainIfNeeded:
    def test_trains_when_no_model(self, daily_df):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.hmm.model_path  = str(Path(tmp) / "hmm_model.pkl")
            cfg.hmm.scaler_path = str(Path(tmp) / "hmm_scaler.pkl")
            det = HMMRegimeDetector(cfg)
            retrained = det.retrain_if_needed(daily_df)
            assert retrained is True
            assert det.is_trained

    def test_no_retrain_when_fresh(self, trained_detector, daily_df):
        retrained = trained_detector.retrain_if_needed(daily_df)
        assert retrained is False


# ---------------------------------------------------------------------------
# regime_distribution helper
# ---------------------------------------------------------------------------

class TestRegimeDistribution:
    def test_returns_series_with_four_regimes(self, trained_detector, daily_df):
        result = trained_detector.predict(daily_df)
        dist = trained_detector.regime_distribution(result)
        assert set(dist.index) == ALL_REGIMES

    def test_sums_to_one(self, trained_detector, daily_df):
        result = trained_detector.predict(daily_df)
        dist = trained_detector.regime_distribution(result)
        assert abs(dist.sum() - 1.0) < 1e-9
