"""
signals/macro_score.py — Macro Regime Scoring Model (Signal Layer 2)

Scores the macro environment using 8 indicators to determine:
  - Whether the market is in deflation panic or inflation stress
  - Directional bias for NQ trades (long / short / neutral)
  - Hedge destination (ZB / GC / ZN / FLAT)

All inputs must be prior-day closes — no intraday data, no look-ahead.

Scoring rules (exact per spec):
  Inflation stress (0–5 points):
    +1  TIP/IEF 63-day momentum positive  (combined rule: both positive)
    +1  VIX between 20–30                 (fear without panic)
    +1  GSG 63-day momentum positive      (commodities rising)
    +1  UUP 63-day momentum negative      (dollar weakening)
    +1  GLD 21-day momentum positive      (gold rising)

  Deflation panic (0–4 points):
    +2  VIX > 30                          (extreme fear)
    +1  IEF 21-day momentum positive      (bonds rallying = safety bid)
    +1  Yield curve (SHY/IEF) 21-day momentum negative  (curve flattening/inverting)

  Note: VIX contributes to ONLY ONE bucket per day (>30 → panic, 20-30 → stress).

Regime classification:
  deflation_panic   — panic >= 3
  inflation_stress  — stress >= 3 AND VIX < 25
  cash              — both panic >= 3 AND stress >= 3 (conflicted market)
  neutral           — neither threshold met

Directional bias:
  short   — deflation_panic >= 3 (risk-off)
  long    — inflation_stress >= 3 AND VIX < 25 (risk-on)
  neutral — otherwise

Usage:
    from signals.macro_score import MacroRegimeScorer
    scorer = MacroRegimeScorer()
    result = scorer.score(macro_df)         # score latest row
    history = scorer.score_history(macro_df) # score every row (for backtest)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

# Thresholds
VIX_PANIC_THRESHOLD  = 30.0   # VIX > 30  → deflation panic (+2)
VIX_STRESS_THRESHOLD = 20.0   # VIX 20-30 → inflation stress (+1)
VIX_LONG_BLOCK       = 25.0   # VIX >= 25 blocks long bias even if stress >= 3

PANIC_TRADE_THRESHOLD  = 3    # deflation_panic_score >= this → short bias
STRESS_TRADE_THRESHOLD = 3    # inflation_stress_score >= this → long bias

MOM_SHORT = 21    # Short momentum lookback (days)
MOM_LONG  = 63    # Long momentum lookback (days)

DominantRegime  = Literal["deflation_panic", "inflation_stress", "neutral", "cash"]
DirectionalBias = Literal["long", "short", "neutral"]
HedgeTarget     = Literal["ZB", "GC", "ZN", "FLAT"]


@dataclass
class MacroScore:
    """Output of MacroRegimeScorer.score() for a single day."""
    deflation_panic_score:  int              # 0–4
    inflation_stress_score: int              # 0–5
    dominant_regime:        DominantRegime
    directional_bias:       DirectionalBias
    hedge_destination:      HedgeTarget

    # Raw values used in scoring (for logging / debugging)
    vix:                    float = 0.0
    tip_mom_63:             float = 0.0
    ief_mom_63:             float = 0.0
    ief_mom_21:             float = 0.0
    shy_ief_mom_21:         float = 0.0
    gsg_mom_63:             float = 0.0
    uup_mom_63:             float = 0.0
    gld_mom_21:             float = 0.0

    def __str__(self) -> str:
        return (
            f"MacroScore("
            f"regime={self.dominant_regime}, "
            f"bias={self.directional_bias}, "
            f"panic={self.deflation_panic_score}, "
            f"stress={self.inflation_stress_score}, "
            f"hedge={self.hedge_destination})"
        )


class MacroRegimeScorer:
    """
    Scores the macro environment from daily closes of 7 instruments.

    Expected DataFrame columns: [VIX, TIP, IEF, SHY, GLD, GSG, UUP]
    All inputs must be prior-day closes (no look-ahead).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, macro_df: pd.DataFrame) -> MacroScore:
        """
        Score the macro environment using the most recent complete row.

        Args:
            macro_df: DataFrame with daily closes, columns
                      [VIX, TIP, IEF, SHY, GLD, GSG, UUP].
                      Must have at least 64 rows for all momentum windows.

        Returns:
            MacroScore dataclass.
        """
        self._validate_columns(macro_df)

        if len(macro_df) < MOM_LONG + 1:
            logger.warning(
                f"[macro] Only {len(macro_df)} rows — need {MOM_LONG + 1} for "
                f"63-day momentum. Some scores may be zero."
            )

        result = self._score_row(macro_df)
        logger.info(f"[macro] {result}")
        return result

    def score_history(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scoring to every row in macro_df (using only past data at each step).
        Returns a DataFrame aligned to macro_df's index.

        Used by the backtest engine — each row sees only data up to that date.

        Columns:
          deflation_panic_score, inflation_stress_score,
          dominant_regime, directional_bias, hedge_destination,
          vix, tip_mom_63, ief_mom_63, ief_mom_21,
          shy_ief_mom_21, gsg_mom_63, uup_mom_63, gld_mom_21
        """
        self._validate_columns(macro_df)
        rows = []

        for i in range(len(macro_df)):
            # Slice up to and including row i — no future data
            window = macro_df.iloc[: i + 1]
            s = self._score_row(window)
            rows.append({
                "deflation_panic_score":  s.deflation_panic_score,
                "inflation_stress_score": s.inflation_stress_score,
                "dominant_regime":        s.dominant_regime,
                "directional_bias":       s.directional_bias,
                "hedge_destination":      s.hedge_destination,
                "vix":                    s.vix,
                "tip_mom_63":             s.tip_mom_63,
                "ief_mom_63":             s.ief_mom_63,
                "ief_mom_21":             s.ief_mom_21,
                "shy_ief_mom_21":         s.shy_ief_mom_21,
                "gsg_mom_63":             s.gsg_mom_63,
                "uup_mom_63":             s.uup_mom_63,
                "gld_mom_21":             s.gld_mom_21,
            })

        df = pd.DataFrame(rows, index=macro_df.index)
        logger.info(
            f"[macro] score_history: {len(df)} days | "
            f"bias distribution: {df['directional_bias'].value_counts().to_dict()}"
        )
        return df

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_row(self, df: pd.DataFrame) -> MacroScore:
        """Score using the last row of df, with momentum computed over full df."""
        vix = float(df["VIX"].iloc[-1])

        # Momentum helpers
        tip_mom_63    = _momentum(df["TIP"],              MOM_LONG)
        ief_mom_63    = _momentum(df["IEF"],              MOM_LONG)
        ief_mom_21    = _momentum(df["IEF"],              MOM_SHORT)
        gsg_mom_63    = _momentum(df["GSG"],              MOM_LONG)
        uup_mom_63    = _momentum(df["UUP"],              MOM_LONG)
        gld_mom_21    = _momentum(df["GLD"],              MOM_SHORT)
        # Yield curve: SHY/IEF ratio — negative momentum = curve flattening
        shy_ief_ratio = df["SHY"] / df["IEF"]
        shy_ief_mom_21 = _momentum(shy_ief_ratio,         MOM_SHORT)

        # ── Inflation stress (0–5) ──────────────────────────────────────
        stress = 0
        # Rule 1: TIP AND IEF both positive 63-day momentum → +1
        if tip_mom_63 > 0 and ief_mom_63 > 0:
            stress += 1
        # Rule 2: VIX between 20–30 → +1 (fear without systemic panic)
        if VIX_STRESS_THRESHOLD <= vix < VIX_PANIC_THRESHOLD:
            stress += 1
        # Rule 3: Commodities (GSG) rising 63-day → +1
        if gsg_mom_63 > 0:
            stress += 1
        # Rule 4: Dollar (UUP) weakening 63-day → +1
        if uup_mom_63 < 0:
            stress += 1
        # Rule 5: Gold (GLD) rising 21-day → +1
        if gld_mom_21 > 0:
            stress += 1

        # ── Deflation panic (0–4) ───────────────────────────────────────
        panic = 0
        # Rule 1: VIX > 30 → +2 (extreme fear / systemic risk)
        if vix > VIX_PANIC_THRESHOLD:
            panic += 2
        # Rule 2: Bonds (IEF) rallying 21-day → +1 (flight to safety)
        if ief_mom_21 > 0:
            panic += 1
        # Rule 3: Yield curve flattening/inverting 21-day → +1
        if shy_ief_mom_21 < 0:
            panic += 1

        # ── Dominant regime ─────────────────────────────────────────────
        if panic >= PANIC_TRADE_THRESHOLD and stress >= STRESS_TRADE_THRESHOLD:
            regime: DominantRegime = "cash"          # conflicted — stay out
        elif panic >= PANIC_TRADE_THRESHOLD:
            regime = "deflation_panic"
        elif stress >= STRESS_TRADE_THRESHOLD and vix < VIX_LONG_BLOCK:
            regime = "inflation_stress"
        else:
            regime = "neutral"

        # ── Directional bias ────────────────────────────────────────────
        if regime == "deflation_panic":
            bias: DirectionalBias = "short"
        elif regime == "inflation_stress":
            bias = "long"
        else:
            bias = "neutral"

        # ── Hedge destination ───────────────────────────────────────────
        if regime == "deflation_panic":
            hedge: HedgeTarget = "ZB"    # 30yr Treasury (max duration in flight-to-safety)
        elif regime == "inflation_stress":
            hedge = "GC"                 # Gold (classic inflation hedge)
        elif regime == "cash":
            hedge = "ZN"                 # 10yr Treasury (mild safety without full duration)
        else:
            hedge = "FLAT"               # No hedge needed in neutral

        return MacroScore(
            deflation_panic_score  = panic,
            inflation_stress_score = stress,
            dominant_regime        = regime,
            directional_bias       = bias,
            hedge_destination      = hedge,
            vix            = vix,
            tip_mom_63     = tip_mom_63,
            ief_mom_63     = ief_mom_63,
            ief_mom_21     = ief_mom_21,
            shy_ief_mom_21 = shy_ief_mom_21,
            gsg_mom_63     = gsg_mom_63,
            uup_mom_63     = uup_mom_63,
            gld_mom_21     = gld_mom_21,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = {"VIX", "TIP", "IEF", "SHY", "GLD", "GSG", "UUP"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[macro] Missing columns: {missing}")
        if df.empty:
            raise ValueError("[macro] macro_df is empty.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _momentum(series: pd.Series, days: int) -> float:
    """
    Rate of change: (current - N_days_ago) / N_days_ago

    Returns 0.0 if there is insufficient history.
    Uses prior-day data only (iloc[-1] is the last complete day).
    """
    s = series.dropna()
    if len(s) < days + 1:
        return 0.0
    current    = float(s.iloc[-1])
    prior      = float(s.iloc[-(days + 1)])
    if prior == 0:
        return 0.0
    return (current - prior) / abs(prior)
