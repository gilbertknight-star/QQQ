"""
config.py — All configurable parameters for the NQ Regime Bot.

Load with:
    from config import Config, get_prop_firm_config
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prop Firm Configuration
# ---------------------------------------------------------------------------

TrailingDrawdownType = Literal["eod", "intraday"]


@dataclass
class PropFirmConfig:
    """Rules for a single prop firm account."""

    firm_name: str
    account_size: float               # Nominal account size in USD
    max_drawdown: float               # Maximum trailing drawdown in USD
    daily_loss_limit: float           # Max daily loss allowed in USD (0 = no limit)
    profit_target: float              # Profit target to pass evaluation in USD
    consistency_rule: float           # Max fraction of total profit in one day (0 = no rule)
    trailing_drawdown_type: TrailingDrawdownType  # "eod" or "intraday"
    time_limit_days: int = 0          # Evaluation time limit in days (0 = no limit)
    profit_split: float = 0.90        # Trader's share of profits (e.g. 0.90 = 90%)

    # Convenience helpers
    @property
    def daily_loss_limit_active(self) -> bool:
        return self.daily_loss_limit > 0

    @property
    def consistency_rule_active(self) -> bool:
        return self.consistency_rule > 0


# ---------------------------------------------------------------------------
# Prop Firm Presets  (Section 6 — verified March 2026)
# ---------------------------------------------------------------------------

PROP_FIRM_CONFIGS: dict[str, PropFirmConfig] = {
    # ── Topstep ──────────────────────────────────────────────────────────
    # Trailing drawdown updates on EOD balance only; intraday breach is OK
    # if you recover by close.
    "topstep_50k": PropFirmConfig(
        firm_name="Topstep",
        account_size=50_000,
        max_drawdown=2_000,
        daily_loss_limit=1_000,
        profit_target=3_000,
        consistency_rule=0.40,
        trailing_drawdown_type="eod",
        time_limit_days=0,
        profit_split=0.90,
    ),
    "topstep_100k": PropFirmConfig(
        firm_name="Topstep",
        account_size=100_000,
        max_drawdown=3_000,
        daily_loss_limit=2_000,
        profit_target=6_000,
        consistency_rule=0.40,
        trailing_drawdown_type="eod",
        time_limit_days=0,
        profit_split=0.90,
    ),
    "topstep_150k": PropFirmConfig(
        firm_name="Topstep",
        account_size=150_000,
        max_drawdown=4_500,
        daily_loss_limit=3_000,
        profit_target=9_000,
        consistency_rule=0.40,
        trailing_drawdown_type="eod",
        time_limit_days=0,
        profit_split=0.90,
    ),
    # ── Apex Trader Funding  (March 2026 4.0 rules) ───────────────────────
    # EOD accounts: trailing drawdown updates at market close only.
    # Consistency rule applies in PA (funded) phase only.
    "apex_100k_eod": PropFirmConfig(
        firm_name="Apex",
        account_size=100_000,
        max_drawdown=3_000,
        daily_loss_limit=1_500,       # EOD-only enforcement
        profit_target=6_000,
        consistency_rule=0.50,        # PA phase only
        trailing_drawdown_type="eod",
        time_limit_days=30,
        profit_split=0.90,
    ),
    # Intraday account: NO daily loss limit; trailing drawdown ticks every bar.
    # Safer for systematic bots if you use the EOD variant above.
    "apex_150k_intraday": PropFirmConfig(
        firm_name="Apex",
        account_size=150_000,
        max_drawdown=5_000,
        daily_loss_limit=0,           # No daily loss limit on intraday account
        profit_target=9_000,
        consistency_rule=0.50,        # PA phase only
        trailing_drawdown_type="intraday",
        time_limit_days=30,
        profit_split=0.90,
    ),
}


def get_prop_firm_config(account_key: str) -> PropFirmConfig:
    """Return the PropFirmConfig for the given account key.

    Args:
        account_key: One of the keys in PROP_FIRM_CONFIGS
                     (e.g. "topstep_100k", "apex_100k_eod").

    Raises:
        KeyError: If the account key is not found.
    """
    if account_key not in PROP_FIRM_CONFIGS:
        valid = ", ".join(PROP_FIRM_CONFIGS.keys())
        raise KeyError(f"Unknown account '{account_key}'. Valid options: {valid}")
    return PROP_FIRM_CONFIGS[account_key]


# ---------------------------------------------------------------------------
# Main Bot Configuration
# ---------------------------------------------------------------------------

@dataclass
class IBConfig:
    """Interactive Brokers TWS / Gateway connection settings."""
    host: str = "127.0.0.1"
    port: int = 7497           # 7497 = paper trading; 7496 = live
    client_id: int = 1
    timeout: int = 10          # Connection timeout in seconds
    readonly: bool = False


@dataclass
class HMMConfig:
    """Hidden Markov Model regime detector settings."""
    n_states: int = 4          # trending, breakout, ranging, volatile
    covariance_type: str = "full"
    n_iter: int = 1_000
    random_state: int = 42
    vol_window: int = 20       # Rolling window for volatility feature
    volume_ratio_window: int = 20
    retrain_days: int = 60     # Retrain HMM every N calendar days
    model_path: str = "models/hmm_model.pkl"
    scaler_path: str = "models/hmm_scaler.pkl"


@dataclass
class SignalConfig:
    """Signal generation thresholds."""
    min_regime_confidence: float = 0.65   # Below this → no trade
    min_signal_agreement: int = 3         # Out of 5 layers; must be >= this to trade
    momentum_lookback_bars: int = 12      # Rate-of-change lookback
    momentum_strength_min: float = 0.40  # Minimum momentum signal strength
    vwap_proximity_flat_pct: float = 0.001  # Within 0.1% of VWAP → flat signal
    structure_proximity_pct: float = 0.003  # Within 0.3% of level → at structure
    options_min_volume: int = 500         # Min options volume for valid signal
    pcr_bullish_threshold: float = 0.70   # PCR below this → bullish lean
    pcr_bearish_threshold: float = 1.30   # PCR above this → bearish lean
    rr_ratio_min: float = 2.0             # Minimum reward-to-risk ratio


@dataclass
class PositionConfig:
    """Position sizing parameters."""
    max_risk_per_trade_pct: float = 0.01  # 1% of account balance per trade
    max_contracts: int = 3                # Hard cap: never exceed this many NQ lots
    nq_point_value: float = 20.0          # $20 per point for NQ
    mnq_point_value: float = 2.0          # $2 per point for MNQ
    micro_threshold_balance: float = 25_000  # Below this → use MNQ automatically
    # Drawdown-based size multipliers (fraction of max_drawdown consumed → multiplier)
    drawdown_multipliers: dict = field(default_factory=lambda: {
        0.00: 1.00,   # 0–25% of max_dd used → full size
        0.25: 0.75,   # 25–50% used → 75%
        0.50: 0.50,   # 50–75% used → 50%
        0.75: 0.25,   # 75–100% used → 25%
    })
    # Regime-based size multipliers
    regime_multipliers: dict = field(default_factory=lambda: {
        "trending": 1.00,
        "breakout": 0.75,
        "ranging": 0.00,   # No trades in ranging
        "volatile": 0.00,  # No trades in volatile
    })
    # Daily loss limit guard: worst-case loss must fit within this fraction
    daily_limit_guard_pct: float = 0.30


@dataclass
class DataConfig:
    """Data source and symbol settings."""
    primary_symbol: str = "NQ"           # Full NQ futures
    micro_symbol: str = "MNQ"            # Micro NQ futures
    exchange: str = "CME"
    currency: str = "USD"
    data_source: str = "ib"              # "ib" | "yfinance" (fallback)
    # Macro instruments fetched via yfinance
    macro_symbols: list = field(default_factory=lambda: [
        "^VIX",   # CBOE Volatility Index
        "TIP",    # iShares TIPS Bond ETF (inflation)
        "IEF",    # iShares 7-10yr Treasury (rates)
        "SHY",    # iShares 1-3yr Treasury (short rates)
        "GLD",    # SPDR Gold (gold)
        "GSG",    # iShares S&P GSCI Commodity (commodities)
        "UUP",    # Invesco DB US Dollar Index (dollar)
    ])
    macro_momentum_short: int = 21       # Days for short momentum calculation
    macro_momentum_long: int = 63        # Days for long momentum calculation
    cache_dir: str = "data/cache"
    parquet_dir: str = "data/parquet"


@dataclass
class NewsConfig:
    """News/economic event filter settings."""
    calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    minutes_before_event: int = 5        # Flatten positions this many min before
    minutes_after_event: int = 10        # No new entries this many min after
    high_impact_events: list = field(default_factory=lambda: [
        "FOMC Rate Decision",
        "Non-Farm Payrolls",
        "CPI",
        "PPI",
        "GDP",
        "Fed Chair Speech",
        "ISM Manufacturing",
        "Federal Funds Rate",
        "NFP",
        "Core CPI",
    ])
    # Pre-programmed known event times (ET) as a fallback when API is unreachable
    # Format: (month, day, hour, minute) — updated manually each quarter
    fomc_times: list = field(default_factory=lambda: [
        # 2026 FOMC decision times at 2:00 PM ET
        (1, 29, 14, 0),
        (3, 19, 14, 0),
        (5, 7, 14, 0),
        (6, 18, 14, 0),
        (7, 30, 14, 0),
        (9, 17, 14, 0),
        (11, 5, 14, 0),
        (12, 17, 14, 0),
    ])


@dataclass
class TradingHoursConfig:
    """Session and trading window settings (all times US Eastern)."""
    pre_market_regime_time: time = field(default_factory=lambda: time(6, 0))
    market_open: time = field(default_factory=lambda: time(9, 30))
    first_hour_end: time = field(default_factory=lambda: time(10, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    # Do not trade after this time (avoid low-liquidity late session)
    cutoff_time: time = field(default_factory=lambda: time(15, 45))
    # Flatten all positions at this time regardless
    eod_flatten_time: time = field(default_factory=lambda: time(15, 55))
    # Loop interval for live/paper mode (seconds)
    loop_interval_seconds: int = 5


@dataclass
class BacktestConfig:
    """Walk-forward backtesting parameters."""
    train_months: int = 18
    test_months: int = 6
    step_months: int = 3
    min_windows: int = 4               # Minimum complete train/test cycles required
    slippage_points: int = 3           # Applied against direction per round trip
    commission_per_rt: float = 4.50    # IB NQ commission per round trip
    data_start_year: int = 2021        # Must include 2022 bear market
    results_dir: str = "backtest_results"


@dataclass
class MonitoringConfig:
    """Logging, alerting, and dashboard settings."""
    log_dir: str = "logs"
    trade_log_path: str = "logs/trades.csv"
    signal_log_path: str = "logs/signals.csv"
    dashboard_refresh_seconds: int = 5
    # Email / SMS alerts (populated via .env)
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = 587
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    alert_email: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL", ""))


# ---------------------------------------------------------------------------
# Top-Level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Master configuration object for the NQ Regime Bot."""

    # Sub-configs
    ib: IBConfig = field(default_factory=IBConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    hours: TradingHoursConfig = field(default_factory=TradingHoursConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Active prop firm account (set via --account CLI flag or directly)
    account_key: str = "topstep_100k"

    # API keys (loaded from .env)
    polygon_api_key: str = field(
        default_factory=lambda: os.getenv("POLYGON_API_KEY", "")
    )
    ib_account_id: str = field(
        default_factory=lambda: os.getenv("IB_ACCOUNT_ID", "")
    )

    @property
    def prop_firm(self) -> PropFirmConfig:
        """Return the active prop firm config."""
        return get_prop_firm_config(self.account_key)

    def validate(self) -> None:
        """Raise ValueError if any critical setting is misconfigured."""
        if self.signal.min_regime_confidence <= 0 or self.signal.min_regime_confidence >= 1:
            raise ValueError("min_regime_confidence must be between 0 and 1 exclusive.")
        if self.signal.min_signal_agreement < 1 or self.signal.min_signal_agreement > 5:
            raise ValueError("min_signal_agreement must be between 1 and 5.")
        if self.position.max_risk_per_trade_pct <= 0 or self.position.max_risk_per_trade_pct > 0.05:
            raise ValueError("max_risk_per_trade_pct must be > 0 and <= 5%.")
        if self.position.max_contracts < 1:
            raise ValueError("max_contracts must be at least 1.")
        if not self.polygon_api_key:
            import warnings
            warnings.warn("POLYGON_API_KEY not set — options data will be unavailable.")


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------

# Import and use this in all modules:
#   from config import config
config = Config()
