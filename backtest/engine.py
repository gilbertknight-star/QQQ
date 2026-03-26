"""
backtest/engine.py — Walk-Forward Backtesting Engine (5-minute bars)

Runs the complete 5-layer signal pipeline bar-by-bar on 5-minute NQ data
sourced from yfinance (up to 60 calendar days of history).

Architecture per bar:
  Layer 1 — HMM Regime        (daily bars, refreshed once per calendar day)
  Layer 2 — Macro Score        (prior-day macro ETFs, refreshed once per day)
  Layer 3 — Momentum / VWAP   (rolling 5m window, every bar)
  Layer 4 — Market Structure   (daily OBs + key levels, refreshed once per day)
  Layer 5 — Options            (always neutral in backtest — no chain data)
  Gate    — SignalCombiner     (5 layers → TradeDecision)

Position management:
  - One NQ (or MNQ) contract position at a time
  - Entry at close of signal bar
  - Stop loss at structure invalidation level
  - Target at entry ± stop_dist × 2.0 (2:1 R:R)
  - Partial exit (50%) at 1:1 R — stop moves to break-even
  - EOD flatten at 15:55 ET
  - Prop firm rules enforced on every entry attempt

Usage:
    from backtest.engine import BacktestEngine
    from config import config

    engine = BacktestEngine(config)
    result = engine.run()
    result.print_summary()

    # Or run standalone:
    python -m backtest.engine
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Signal imports
# ---------------------------------------------------------------------------
from signals.anchor_impulse import AnchorImpulse, AnchorImpulseSignal, ANCHOR_REGIMES
from signals.hmm_regime import HMMRegimeDetector
from signals.macro_score import MacroRegimeScorer, MacroScore
from signals.momentum import MomentumSignal, SignalResult
from signals.structure import StructureDetector, StructureSignal, OrderBlock, KeyLevel
from signals.options_signal import OptionsSignal, OptionsSignalResult
from execution.signal_combiner import SignalCombiner, TradeDecision
from execution.position_sizer import PositionSizer
from risk.prop_firm_rules import PropFirmRules, PropFirmState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ET              = "America/New_York"
_RTH_OPEN        = time(9, 30)
_RTH_CLOSE       = time(16, 0)
_EOD_FLATTEN     = time(15, 55)   # flatten any open trade at this bar's close
_MIN_SIGNAL_BARS = 30             # warm-up bars before first trade attempt
_MOM_WINDOW      = 200            # rolling 5m bars fed to MomentumSignal (~1 day)
_STRUCT_LOOKBACK = 40             # daily bars for order-block / key-level scan

# Intraday stop cap: if daily-structure stop is wider than this many 5m ATRs,
# clamp it to entry ± (ATR_5M_CAP × 5m_ATR).  This keeps stops realistic for
# 5-minute bar trading while still respecting the structural direction.
_ATR_STOP_CAP_MULT = 3.0          # max stop = 3× the current 5m ATR
_ATR_PERIOD        = 14           # ATR period for 5m bars

NQ_POINT_VALUE  = 20.0
MNQ_POINT_VALUE =  2.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single completed trade."""
    direction:           str       # "long" | "short"
    entry_time:          datetime
    exit_time:           datetime
    entry_price:         float
    stop_price_initial:  float
    target_price:        float
    exit_price:          float
    contracts:           int
    instrument:          str       # "NQ" | "MNQ"
    point_value:         float
    pnl_dollars:         float
    exit_reason:         str       # "stop"|"target"|"trail_stop"|"eod"|"partial->stop"|"partial->target"|"partial->eod"
    conviction_score:    int
    regime:              str
    macro_bias:          str
    partial_executed:    bool = False

    @property
    def risk_dollars(self) -> float:
        return abs(self.entry_price - self.stop_price_initial) * self.contracts * self.point_value

    @property
    def r_multiple(self) -> float:
        if self.risk_dollars == 0:
            return 0.0
        return self.pnl_dollars / self.risk_dollars


@dataclass
class BacktestResult:
    """Full result of a completed backtest run."""
    trades:            list[Trade]
    equity_curve:      pd.Series       # datetime → account balance
    daily_pnl:         pd.Series       # date → daily P&L
    metrics:           dict
    prop_firm_result:  dict
    config_summary:    dict

    def print_summary(self) -> None:
        """Print a formatted summary to stdout (ASCII-safe for all terminals)."""
        m   = self.metrics
        pf  = self.prop_firm_result
        sep = "-" * 60

        print(f"\n{sep}")
        print(f"  NQ REGIME BOT -- BACKTEST RESULTS (5-minute bars)")
        print(sep)
        print(f"  Period      : {m.get('start_date','?')} to {m.get('end_date','?')}")
        print(f"  Bars traded : {m.get('total_bars','?'):,}")
        print(f"  Trades      : {m.get('total_trades', 0)}")
        print(f"  Win rate    : {m.get('win_rate', 0):.1%}")
        print(f"  Prof factor : {m.get('profit_factor', 0):.2f}")
        print(f"  Avg winner  : ${m.get('avg_winner', 0):,.2f}")
        print(f"  Avg loser   : ${m.get('avg_loser', 0):,.2f}")
        print(f"  Avg R       : {m.get('avg_r', 0):.2f}R")
        print(f"  Sharpe      : {m.get('sharpe', 0):.2f}")
        print(f"  Max DD      : ${m.get('max_drawdown_dollars', 0):,.2f}")
        print(f"  Net P&L     : ${m.get('net_pnl', 0):,.2f}")
        print(sep)
        print(f"  Prop Firm   : {pf.get('firm','?')}  ({pf.get('account_key','?')})")
        print(f"  Result      : {'PASSED' if pf.get('passed') else 'FAILED'}")
        print(f"  Reason      : {pf.get('reason','')}")
        print(f"  Max DD hit  : ${pf.get('max_dd_hit', 0):,.2f} / ${pf.get('max_dd_limit', 0):,.2f}")
        print(f"  Daily limits breached: {pf.get('daily_limit_breaches', 0)}")
        print(sep)
        if self.trades:
            print("\n  Last 5 trades:")
            for t in self.trades[-5:]:
                sign = "+" if t.pnl_dollars >= 0 else ""
                print(
                    f"    {t.entry_time.strftime('%m-%d %H:%M')} "
                    f"{t.direction.upper():5s} "
                    f"entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                    f"P&L={sign}${t.pnl_dollars:,.0f} ({t.exit_reason})"
                )
        print(sep + "\n")


# ---------------------------------------------------------------------------
# Internal position tracker
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    direction:           str
    entry_time:          datetime
    entry_price:         float
    stop_price:          float     # moves to BE after partial (fixed-target) or hard stop (trailing)
    stop_price_initial:  float     # never changes (for R calculation)
    target_price:        float     # fixed target (fixed-target style) or 0 (trailing)
    partial_exit_price:  float
    contracts_total:     int
    contracts_remaining: int
    instrument:          str
    point_value:         float
    conviction_score:    int
    regime:              str
    macro_bias:          str
    partial_executed:    bool  = False
    partial_pnl:         float = 0.0
    # ── Trailing stop fields (Anchor-Impulse only) ──────────────────
    exit_style:          str   = "fixed_target"  # "fixed_target" | "trailing"
    trail_hw:            float = 0.0   # best price reached since entry
    trail_activated:     bool  = False # True once activation threshold crossed
    trail_stop:          float = 0.0   # current trailing stop (0 = not yet active)
    trail_activate_pts:  float = 0.0   # price move needed to activate
    trail_offset_pts:    float = 0.0   # ATR offset behind HWM
    atr_at_entry:        float = 0.0   # ATR locked at entry (for trail calc)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Simulate the full 5-layer signal strategy on 5-minute NQ bars.

    Args:
        config: Config dataclass from config.py. Uses default_config if None.
    """

    def __init__(self, config=None):
        if config is None:
            from config import config as default_config
            config = default_config

        self._cfg = config
        self._pf  = config.prop_firm

        # Instantiate all signal layer objects
        self._hmm      = HMMRegimeDetector(config)
        self._macro    = MacroRegimeScorer()
        self._mom      = MomentumSignal(config)
        self._struct   = StructureDetector(config)
        self._opts     = OptionsSignal(config)
        self._combiner = SignalCombiner(config)
        self._anchor   = AnchorImpulse(config)   # volatile/ranging strategy
        self._sizer    = PositionSizer(config)
        self._rules    = PropFirmRules(config)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, refresh_data: bool = False) -> BacktestResult:
        """
        Fetch data, build signals, simulate trades, return BacktestResult.

        Args:
            refresh_data: Force re-fetch from yfinance even if cache is fresh.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        logger.info("[backtest] Starting 5-minute bar backtest ...")

        # ── 1. Fetch data ──────────────────────────────────────────────
        from data.pipeline import DataPipeline
        pipe    = DataPipeline(self._cfg)
        df_5m   = pipe.get("NQ", "5m",  source="yfinance", lookback_days=60,  refresh=refresh_data)
        df_1d   = pipe.get("NQ", "1d",  source="yfinance", lookback_days=730, refresh=refresh_data)
        df_macro = pipe.get_macro(lookback_days=252, refresh=refresh_data)

        logger.info(
            f"[backtest] Data loaded: "
            f"5m={len(df_5m)} bars, 1d={len(df_1d)} bars, "
            f"macro={len(df_macro)} rows"
        )

        # ── 2. Train / load HMM ────────────────────────────────────────
        if not self._hmm.load():
            logger.info("[backtest] No saved HMM — training on daily bars ...")
            self._hmm.train(df_1d)
        else:
            logger.info("[backtest] HMM loaded from disk.")

        # ── 3. Pre-compute macro score history (no look-ahead) ─────────
        macro_history = self._macro.score_history(df_macro)
        logger.info(f"[backtest] Macro history: {len(macro_history)} rows")

        # ── 4. Pre-compute daily regime predictions ────────────────────
        regime_history = self._hmm.predict(df_1d)
        logger.info(f"[backtest] Regime history: {len(regime_history)} rows")

        # ── 5. Convert 5m bars to ET for session logic ─────────────────
        df_5m_et = df_5m.copy()
        df_5m_et.index = df_5m_et.index.tz_convert(_ET)

        # ── 6. Initialise prop firm state ──────────────────────────────
        acct = self._pf.account_size
        pf_state = PropFirmState(
            starting_balance = acct,
            current_balance  = acct,
            high_water_mark  = acct,
            daily_pnl        = 0.0,
            total_net_profit = 0.0,
        )

        # ── 7. Main simulation loop ────────────────────────────────────
        trades:         list[Trade]       = []
        equity_points:  list[tuple]       = []   # (datetime, balance)
        position:       _Position | None  = None
        last_trade_date: date | None      = None
        daily_limit_breaches:   int       = 0
        max_dd_hit:             float     = 0.0
        bars_seen:              int       = 0

        for idx, bar in df_5m_et.iterrows():
            bar_et: datetime = idx
            bar_date: date   = bar_et.date()
            bar_time: time   = bar_et.time()

            # ── Skip pre/post market ───────────────────────────────────
            if bar_time < _RTH_OPEN or bar_time >= _RTH_CLOSE:
                continue

            bars_seen += 1

            # ── EOD flatten ────────────────────────────────────────────
            if position is not None and bar_time >= _EOD_FLATTEN:
                exit_price = float(bar["close"])
                trade, pf_state = self._close_position(
                    position, exit_price, bar_et, "eod", pf_state
                )
                trades.append(trade)
                equity_points.append((bar_et, pf_state.current_balance))
                position = None
                continue

            # ── EOD reset ──────────────────────────────────────────────
            if last_trade_date is not None and bar_date != last_trade_date:
                pf_state = self._rules.end_of_day(pf_state)
                if pf_state.is_halted:
                    logger.warning(f"[backtest] Account HALTED: {pf_state.halt_reason}")
                    break
                if pf_state.phase_passed:
                    logger.success("[backtest] Profit target REACHED — evaluation passed!")
                    break
            last_trade_date = bar_date

            # ── Check position exit conditions ─────────────────────────
            if position is not None:
                bar_high = float(bar["high"])
                bar_low  = float(bar["low"])
                pos = position

                # ── TRAILING STOP (Anchor-Impulse) ─────────────────────
                if pos.exit_style == "trailing":
                    new_hw, new_trail_stop, now_active = self._anchor.update_trailing_stop(
                        pos.direction, pos.entry_price,
                        bar_high, bar_low,
                        pos.trail_hw, pos.trail_activated,
                        pos.atr_at_entry,
                    )
                    pos.trail_hw        = new_hw
                    pos.trail_activated = now_active
                    pos.trail_stop      = new_trail_stop

                    if pos.direction == "long":
                        # Trailing stop hit (once activated)
                        if now_active and bar_low <= new_trail_stop:
                            trade, pf_state = self._close_position(
                                pos, new_trail_stop, bar_et, "trail_stop", pf_state
                            )
                            trades.append(trade)
                            equity_points.append((bar_et, pf_state.current_balance))
                            position = None
                            max_dd_hit = max(max_dd_hit,
                                            pf_state.high_water_mark - pf_state.current_balance)
                            continue
                        # Hard stop (before trail activates)
                        if bar_low <= pos.stop_price:
                            trade, pf_state = self._close_position(
                                pos, pos.stop_price, bar_et, "stop", pf_state
                            )
                            trades.append(trade)
                            equity_points.append((bar_et, pf_state.current_balance))
                            position = None
                            max_dd_hit = max(max_dd_hit,
                                            pf_state.high_water_mark - pf_state.current_balance)
                            continue
                    else:  # short trailing
                        if now_active and bar_high >= new_trail_stop:
                            trade, pf_state = self._close_position(
                                pos, new_trail_stop, bar_et, "trail_stop", pf_state
                            )
                            trades.append(trade)
                            equity_points.append((bar_et, pf_state.current_balance))
                            position = None
                            max_dd_hit = max(max_dd_hit,
                                            pf_state.high_water_mark - pf_state.current_balance)
                            continue
                        if bar_high >= pos.stop_price:
                            trade, pf_state = self._close_position(
                                pos, pos.stop_price, bar_et, "stop", pf_state
                            )
                            trades.append(trade)
                            equity_points.append((bar_et, pf_state.current_balance))
                            position = None
                            max_dd_hit = max(max_dd_hit,
                                            pf_state.high_water_mark - pf_state.current_balance)
                            continue

                    # Still in trailing-stop position
                    equity_points.append((bar_et, pf_state.current_balance))
                    continue

                # ── FIXED TARGET (5-layer regime bot) ─────────────────
                if pos.direction == "long":
                    # Partial exit check (before stop/target)
                    if not pos.partial_executed and bar_high >= pos.partial_exit_price:
                        pos, pf_state = self._partial_exit(pos, bar_et, pf_state)

                    # Stop: check AFTER potential partial (stop may now be BE)
                    if bar_low <= pos.stop_price:
                        exit_reason = "partial->stop" if pos.partial_executed else "stop"
                        trade, pf_state = self._close_position(
                            pos, pos.stop_price, bar_et, exit_reason, pf_state
                        )
                        trades.append(trade)
                        equity_points.append((bar_et, pf_state.current_balance))
                        position = None
                        max_dd_hit = max(max_dd_hit,
                                        pf_state.high_water_mark - pf_state.current_balance)
                        continue

                    # Target
                    if bar_high >= pos.target_price:
                        exit_reason = "partial->target" if pos.partial_executed else "target"
                        trade, pf_state = self._close_position(
                            pos, pos.target_price, bar_et, exit_reason, pf_state
                        )
                        trades.append(trade)
                        equity_points.append((bar_et, pf_state.current_balance))
                        position = None
                        continue

                else:  # short fixed-target
                    if not pos.partial_executed and bar_low <= pos.partial_exit_price:
                        pos, pf_state = self._partial_exit(pos, bar_et, pf_state)

                    if bar_high >= pos.stop_price:
                        exit_reason = "partial->stop" if pos.partial_executed else "stop"
                        trade, pf_state = self._close_position(
                            pos, pos.stop_price, bar_et, exit_reason, pf_state
                        )
                        trades.append(trade)
                        equity_points.append((bar_et, pf_state.current_balance))
                        position = None
                        max_dd_hit = max(max_dd_hit,
                                        pf_state.high_water_mark - pf_state.current_balance)
                        continue

                    if bar_low <= pos.target_price:
                        exit_reason = "partial->target" if pos.partial_executed else "target"
                        trade, pf_state = self._close_position(
                            pos, pos.target_price, bar_et, exit_reason, pf_state
                        )
                        trades.append(trade)
                        equity_points.append((bar_et, pf_state.current_balance))
                        position = None
                        continue

                # Still in fixed-target position
                equity_points.append((bar_et, pf_state.current_balance))
                continue

            # ── Not in position — look for entry ───────────────────────

            # Need warmup bars
            if bars_seen < _MIN_SIGNAL_BARS:
                equity_points.append((bar_et, pf_state.current_balance))
                continue

            # Skip if halted
            if pf_state.is_halted or pf_state.phase_passed:
                break

            # ── Build 5m rolling window for momentum ──────────────────
            # Convert bar_et back to UTC to slice df_5m
            bar_utc = bar_et.tz_convert("UTC")
            # Get the integer position of this bar in the original UTC index
            try:
                loc = df_5m.index.get_loc(bar_utc)
            except KeyError:
                equity_points.append((bar_et, pf_state.current_balance))
                continue

            start_loc = max(0, loc - _MOM_WINDOW + 1)
            bars_window = df_5m.iloc[start_loc: loc + 1]

            # ── Layer 2: Macro score (prior trading day) ───────────────
            macro_score = self._get_macro_score(macro_history, bar_date)

            # ── Layer 1: HMM Regime (prior trading day) ────────────────
            regime_row = self._get_regime_row(regime_history, bar_date)
            if regime_row is None:
                equity_points.append((bar_et, pf_state.current_balance))
                continue

            # ── Determine which strategy handles this regime ────────────
            current_regime = str(regime_row.iloc[-1]["regime"])
            use_anchor     = current_regime in ANCHOR_REGIMES

            if use_anchor:
                # ════════════════════════════════════════════════════════
                # ANCHOR-IMPULSE  (volatile / ranging regimes)
                # EMA200 + SMA20 pullback with ATR trailing stop
                # ════════════════════════════════════════════════════════
                ai_sig = self._anchor.get_signal(bars_window)
                if not ai_sig.signal_valid:
                    equity_points.append((bar_et, pf_state.current_balance))
                    continue

                entry_price   = ai_sig.entry_price
                stop_price    = ai_sig.stop_price
                stop_dist_pts = abs(entry_price - stop_price)
                target_price  = 0.0  # trailing stop — no fixed target
                direction     = ai_sig.direction
                conviction    = 2    # anchor-impulse has no 0-5 score; use 2

            else:
                # ════════════════════════════════════════════════════════
                # 5-LAYER REGIME BOT  (trending / breakout regimes)
                # ════════════════════════════════════════════════════════

                # Layer 3: Momentum / VWAP
                mom_signal = self._mom.get_signal(bars_window, macro_score.directional_bias)

                # Layer 4: Market Structure (prior daily bars)
                daily_prior = df_1d[
                    df_1d.index.normalize() < pd.Timestamp(bar_date).tz_localize("UTC")
                ]
                if len(daily_prior) < 5:
                    equity_points.append((bar_et, pf_state.current_balance))
                    continue
                obs  = self._struct.find_order_blocks(daily_prior, lookback=_STRUCT_LOOKBACK)
                keys = self._struct.find_key_levels(daily_prior)
                struct_signal = self._struct.get_structure_signal(
                    float(bar["close"]), mom_signal.direction, obs, keys,
                )

                # Layer 5: Options (always neutral in backtest)
                opts_signal = self._opts.get_signal(None, None, float(bar["close"]))

                # Gate: combine all layers
                decision = self._combiner.combine(
                    regime_row, macro_score, mom_signal,
                    struct_signal, opts_signal,
                    current_price=float(bar["close"]),
                )
                if not decision.should_trade:
                    equity_points.append((bar_et, pf_state.current_balance))
                    continue

                entry_price = float(bar["close"])
                stop_price  = _cap_stop(
                    decision.direction, entry_price, decision.stop_price, bars_window,
                )

                # Validate stop side
                if decision.direction == "long" and stop_price >= entry_price:
                    atr_fb     = _atr_estimate(bars_window)
                    stop_price = entry_price - max(atr_fb * 1.5, 5.0)
                elif decision.direction == "short" and stop_price <= entry_price:
                    atr_fb     = _atr_estimate(bars_window)
                    stop_price = entry_price + max(atr_fb * 1.5, 5.0)

                stop_dist_pts = abs(entry_price - stop_price)
                direction     = decision.direction
                conviction    = decision.conviction_score
                if decision.direction == "long":
                    target_price = entry_price + stop_dist_pts * self._cfg.signal.rr_ratio_min
                else:
                    target_price = entry_price - stop_dist_pts * self._cfg.signal.rr_ratio_min

            # ── Instrument selection (shared) ──────────────────────────
            nq_worst_case = 1 * stop_dist_pts * NQ_POINT_VALUE
            if (nq_worst_case > self._pf.daily_loss_limit
                    or self._sizer.should_use_micro(pf_state.current_balance)):
                instrument  = "MNQ"
                point_value = MNQ_POINT_VALUE
            else:
                instrument  = "NQ"
                point_value = NQ_POINT_VALUE

            # ── Size the trade (shared) ────────────────────────────────
            current_dd = pf_state.high_water_mark - pf_state.current_balance
            contracts  = self._sizer.calculate_size(
                account_balance      = pf_state.current_balance,
                stop_distance_points = stop_dist_pts,
                current_drawdown     = current_dd,
                max_drawdown         = self._pf.max_drawdown,
                regime               = current_regime,
                point_value          = point_value,
            )
            contracts = max(1, contracts)

            # ── Prop firm pre-trade check (shared) ─────────────────────
            check = self._rules.check(
                pf_state, contracts, stop_dist_pts, point_value=point_value
            )
            if not check.can_trade:
                logger.debug(f"[backtest] Trade blocked: {check.detail}")
                if check.reason == "daily_loss_limit":
                    daily_limit_breaches += 1
                equity_points.append((bar_et, pf_state.current_balance))
                continue
            contracts = check.safe_contracts

            # ── Build position (exit style differs by strategy) ────────
            if use_anchor:
                partial_price = 0.0   # no partial exit for trailing-stop strategy
                position = _Position(
                    direction           = direction,
                    entry_time          = bar_et,
                    entry_price         = entry_price,
                    stop_price          = stop_price,
                    stop_price_initial  = stop_price,
                    target_price        = 0.0,
                    partial_exit_price  = 0.0,
                    contracts_total     = contracts,
                    contracts_remaining = contracts,
                    instrument          = instrument,
                    point_value         = point_value,
                    conviction_score    = conviction,
                    regime              = current_regime,
                    macro_bias          = macro_score.directional_bias,
                    exit_style          = "trailing",
                    trail_hw            = entry_price,
                    trail_activated     = False,
                    trail_stop          = 0.0,
                    trail_activate_pts  = ai_sig.trail_activate_pts,
                    trail_offset_pts    = ai_sig.trail_offset_pts,
                    atr_at_entry        = ai_sig.atr,
                )
            else:
                partial_price = self._combiner.get_partial_exit_price(
                    entry_price, stop_price, rr_ratio=1.0
                )
                position = _Position(
                    direction           = direction,
                    entry_time          = bar_et,
                    entry_price         = entry_price,
                    stop_price          = stop_price,
                    stop_price_initial  = stop_price,
                    target_price        = target_price,
                    partial_exit_price  = partial_price,
                    contracts_total     = contracts,
                    contracts_remaining = contracts,
                    instrument          = instrument,
                    point_value         = point_value,
                    conviction_score    = conviction,
                    regime              = current_regime,
                    macro_bias          = macro_score.directional_bias,
                    exit_style          = "fixed_target",
                )
            strat_tag = "ANCHOR" if use_anchor else "REGIME"
            tgt_str   = "trail" if use_anchor else f"{target_price:.2f}"
            logger.debug(
                f"[backtest] [{strat_tag}] ENTER {direction.upper()} @ {entry_price:.2f} "
                f"stop={stop_price:.2f} tgt={tgt_str} "
                f"score={conviction} contracts={contracts}{instrument} regime={current_regime}"
            )
            equity_points.append((bar_et, pf_state.current_balance))

        # ── Force-close any open position at end of data ────────────────
        if position is not None and len(df_5m_et) > 0:
            last_close = float(df_5m_et["close"].iloc[-1])
            last_time  = df_5m_et.index[-1]
            trade, pf_state = self._close_position(
                position, last_close, last_time, "eod", pf_state
            )
            trades.append(trade)

        # ── 8. Build equity curve ──────────────────────────────────────
        if equity_points:
            eq_idx = [p[0] for p in equity_points]
            eq_val = [p[1] for p in equity_points]
            equity_curve = pd.Series(eq_val, index=eq_idx)
        else:
            equity_curve = pd.Series(dtype=float)

        # ── 9. Daily P&L series ────────────────────────────────────────
        if trades:
            trade_df = pd.DataFrame([
                {"date": t.exit_time.date(), "pnl": t.pnl_dollars} for t in trades
            ])
            daily_pnl = trade_df.groupby("date")["pnl"].sum()
        else:
            daily_pnl = pd.Series(dtype=float)

        # ── 10. Metrics ────────────────────────────────────────────────
        metrics = _compute_metrics(trades, equity_curve, daily_pnl, df_5m_et)

        # ── 11. Prop firm summary ──────────────────────────────────────
        net_pnl      = pf_state.current_balance - acct
        phase_passed = pf_state.phase_passed or (net_pnl >= self._pf.profit_target)
        pf_result = {
            "firm":                  self._pf.firm_name,
            "account_key":           self._cfg.account_key,
            "passed":                phase_passed,
            "reason":                (
                "Profit target reached" if phase_passed
                else pf_state.halt_reason if pf_state.is_halted
                else "Backtest ended without hitting target or being halted"
            ),
            "max_dd_hit":            max_dd_hit,
            "max_dd_limit":          self._pf.max_drawdown,
            "daily_limit_breaches":  daily_limit_breaches,
            "final_balance":         pf_state.current_balance,
            "net_pnl":               net_pnl,
        }

        config_summary = {
            "account_key":     self._cfg.account_key,
            "timeframe":       "5m",
            "signal_min":      self._cfg.signal.min_signal_agreement,
            "rr_ratio":        self._cfg.signal.rr_ratio_min,
            "max_contracts":   self._cfg.position.max_contracts,
            "max_risk_pct":    self._cfg.position.max_risk_per_trade_pct,
        }

        result = BacktestResult(
            trades           = trades,
            equity_curve     = equity_curve,
            daily_pnl        = daily_pnl,
            metrics          = metrics,
            prop_firm_result = pf_result,
            config_summary   = config_summary,
        )

        result.print_summary()
        logger.info(
            f"[backtest] Complete. {len(trades)} trades. "
            f"Net P&L: ${net_pnl:,.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_macro_score(self, macro_history: pd.DataFrame, bar_date: date) -> MacroScore:
        """Return the macro score for the prior trading day (no look-ahead)."""
        if macro_history is None or macro_history.empty:
            return _neutral_macro()
        try:
            prior = macro_history[
                macro_history.index.normalize() < pd.Timestamp(bar_date).tz_localize("UTC")
            ]
            if prior.empty:
                return _neutral_macro()
            row = prior.iloc[-1]
            return MacroScore(
                deflation_panic_score  = int(row.get("deflation_panic_score", 0)),
                inflation_stress_score = int(row.get("inflation_stress_score", 0)),
                dominant_regime        = str(row.get("dominant_regime", "neutral")),
                directional_bias       = str(row.get("directional_bias", "neutral")),
                hedge_destination      = str(row.get("hedge_destination", "FLAT")),
                vix                    = float(row.get("vix", 0)),
                tip_mom_63             = float(row.get("tip_mom_63", 0)),
                ief_mom_63             = float(row.get("ief_mom_63", 0)),
                ief_mom_21             = float(row.get("ief_mom_21", 0)),
                gsg_mom_63             = float(row.get("gsg_mom_63", 0)),
                uup_mom_63             = float(row.get("uup_mom_63", 0)),
                gld_mom_21             = float(row.get("gld_mom_21", 0)),
                shy_ief_mom_21         = float(row.get("shy_ief_mom_21", 0)),
            )
        except Exception as e:
            logger.warning(f"[backtest] Macro lookup failed ({e}) — using neutral")
            return _neutral_macro()

    def _get_regime_row(
        self, regime_history: pd.DataFrame, bar_date: date
    ) -> pd.DataFrame | None:
        """Return a 1-row DataFrame of regime for the prior trading day."""
        if regime_history is None or regime_history.empty:
            return None
        try:
            prior = regime_history[
                regime_history.index.normalize() < pd.Timestamp(bar_date).tz_localize("UTC")
            ]
            if prior.empty:
                return None
            return prior.iloc[[-1]]  # 1-row DataFrame (combiner expects .iloc[-1])
        except Exception as e:
            logger.warning(f"[backtest] Regime lookup failed ({e})")
            return None

    def _partial_exit(
        self, pos: _Position, bar_et: datetime, pf_state: PropFirmState
    ) -> tuple[_Position, PropFirmState]:
        """Execute a 50% partial exit and move stop to break-even."""
        half = max(1, pos.contracts_remaining // 2)
        if pos.direction == "long":
            partial_pnl = (pos.partial_exit_price - pos.entry_price) * half * pos.point_value
        else:
            partial_pnl = (pos.entry_price - pos.partial_exit_price) * half * pos.point_value

        pf_state = self._rules.update_after_trade(pf_state, partial_pnl, is_eod=False)
        pos.partial_pnl         = partial_pnl
        pos.contracts_remaining = pos.contracts_remaining - half
        pos.stop_price          = pos.entry_price      # move stop to BE
        pos.partial_executed    = True

        logger.debug(
            f"[backtest] PARTIAL EXIT: {half}x{pos.instrument} @ {pos.partial_exit_price:.2f} "
            f"pnl=${partial_pnl:,.0f} | stop → BE={pos.entry_price:.2f}"
        )
        return pos, pf_state

    def _close_position(
        self,
        pos: _Position,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        pf_state: PropFirmState,
    ) -> tuple[Trade, PropFirmState]:
        """Close remaining contracts and return a completed Trade."""
        c = pos.contracts_remaining
        if pos.direction == "long":
            close_pnl = (exit_price - pos.entry_price) * c * pos.point_value
        else:
            close_pnl = (pos.entry_price - exit_price) * c * pos.point_value

        total_pnl = close_pnl + pos.partial_pnl
        pf_state  = self._rules.update_after_trade(pf_state, close_pnl, is_eod=False)

        trade = Trade(
            direction          = pos.direction,
            entry_time         = pos.entry_time,
            exit_time          = exit_time,
            entry_price        = pos.entry_price,
            stop_price_initial = pos.stop_price_initial,
            target_price       = pos.target_price,
            exit_price         = exit_price,
            contracts          = pos.contracts_total,
            instrument         = pos.instrument,
            point_value        = pos.point_value,
            pnl_dollars        = round(total_pnl, 2),
            exit_reason        = reason,
            conviction_score   = pos.conviction_score,
            regime             = pos.regime,
            macro_bias         = pos.macro_bias,
            partial_executed   = pos.partial_executed,
        )

        sign = "+" if trade.pnl_dollars >= 0 else ""
        logger.info(
            f"[backtest] CLOSE {pos.direction.upper()} @ {exit_price:.2f} "
            f"({reason}) pnl={sign}${trade.pnl_dollars:,.0f} "
            f"bal=${pf_state.current_balance:,.0f}"
        )
        return trade, pf_state


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    trades: list[Trade],
    equity_curve: pd.Series,
    daily_pnl: pd.Series,
    df_5m_et: pd.DataFrame,
) -> dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_winner": 0.0, "avg_loser": 0.0, "avg_r": 0.0,
            "sharpe": 0.0, "max_drawdown_dollars": 0.0, "net_pnl": 0.0,
            "total_bars": 0,
            "start_date": str(df_5m_et.index[0].date()) if len(df_5m_et) else "?",
            "end_date":   str(df_5m_et.index[-1].date()) if len(df_5m_et) else "?",
        }

    winners = [t.pnl_dollars for t in trades if t.pnl_dollars > 0]
    losers  = [t.pnl_dollars for t in trades if t.pnl_dollars <= 0]
    rs      = [t.r_multiple  for t in trades]

    gross_win  = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe from daily P&L
    if len(daily_pnl) > 1:
        mean_d  = daily_pnl.mean()
        std_d   = daily_pnl.std(ddof=1)
        sharpe  = (mean_d / std_d) * (252 ** 0.5) if std_d > 0 else 0.0
    else:
        sharpe  = 0.0

    # Max drawdown from equity curve
    if len(equity_curve) > 1:
        roll_max = equity_curve.cummax()
        dd       = (roll_max - equity_curve).max()
    else:
        dd = 0.0

    net_pnl = sum(t.pnl_dollars for t in trades)

    return {
        "total_trades":          len(trades),
        "wins":                  len(winners),
        "losses":                len(losers),
        "win_rate":              len(winners) / len(trades),
        "profit_factor":         round(pf, 2),
        "avg_winner":            round(sum(winners) / len(winners), 2) if winners else 0.0,
        "avg_loser":             round(sum(losers)  / len(losers),  2) if losers  else 0.0,
        "avg_r":                 round(sum(rs) / len(rs), 2),
        "sharpe":                round(sharpe, 2),
        "max_drawdown_dollars":  round(dd, 2),
        "net_pnl":               round(net_pnl, 2),
        "total_bars":            len(df_5m_et),
        "start_date":            str(df_5m_et.index[0].date())  if len(df_5m_et) else "?",
        "end_date":              str(df_5m_et.index[-1].date()) if len(df_5m_et) else "?",
        "by_exit_reason":        _count_by_exit(trades),
        "by_regime":             _count_by_regime(trades),
    }


def _atr_estimate(bars_window: pd.DataFrame, period: int = _ATR_PERIOD) -> float:
    """Return the current ATR from a bars window. Returns 20.0 as fallback."""
    try:
        if len(bars_window) < period + 2:
            return 20.0
        high = bars_window["high"]
        low  = bars_window["low"]
        prev = bars_window["close"].shift(1)
        tr   = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()],
                         axis=1).max(axis=1)
        val  = tr.rolling(period).mean().iloc[-1]
        return float(val) if not pd.isna(val) and val > 0 else 20.0
    except Exception:
        return 20.0


def _cap_stop(
    direction: str,
    entry_price: float,
    structure_stop: float,
    bars_window: pd.DataFrame,
    atr_mult: float = _ATR_STOP_CAP_MULT,
    atr_period: int = _ATR_PERIOD,
) -> float:
    """
    Cap the structure-derived stop to at most atr_mult × 5m-ATR from entry.

    Daily order-block invalidation levels can be hundreds of points away —
    far too wide for intraday 5m position sizing.  We keep the DIRECTION of
    the structural stop but clamp its distance to a sensible intraday range.

    Returns the capped stop price (may equal structure_stop if already tight).
    """
    try:
        if len(bars_window) < atr_period + 2:
            return structure_stop

        high = bars_window["high"]
        low  = bars_window["low"]
        prev_close = bars_window["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean().iloc[-1]

        if pd.isna(atr) or atr <= 0:
            return structure_stop

        max_stop_dist = atr_mult * atr
        if direction == "long":
            capped = entry_price - max_stop_dist
            # Use the tighter (higher) of the two stops for a long
            return max(structure_stop, capped)
        else:
            capped = entry_price + max_stop_dist
            # Use the tighter (lower) of the two stops for a short
            return min(structure_stop, capped)
    except Exception:
        return structure_stop


def _count_by_exit(trades: list[Trade]) -> dict:
    counts: dict[str, dict] = {}
    for t in trades:
        r = t.exit_reason
        if r not in counts:
            counts[r] = {"count": 0, "pnl": 0.0}
        counts[r]["count"] += 1
        counts[r]["pnl"]   += t.pnl_dollars
    return counts


def _count_by_regime(trades: list[Trade]) -> dict:
    counts: dict[str, dict] = {}
    for t in trades:
        r = t.regime
        if r not in counts:
            counts[r] = {"count": 0, "pnl": 0.0}
        counts[r]["count"] += 1
        counts[r]["pnl"]   += t.pnl_dollars
    return counts


# ---------------------------------------------------------------------------
# Neutral MacroScore fallback
# ---------------------------------------------------------------------------

def _neutral_macro() -> "MacroScore":
    return MacroScore(
        deflation_panic_score  = 0,
        inflation_stress_score = 0,
        dominant_regime        = "neutral",
        directional_bias       = "neutral",
        hedge_destination      = "FLAT",
        vix                    = 0.0,
        tip_mom_63             = 0.0,
        ief_mom_63             = 0.0,
        ief_mom_21             = 0.0,
        gsg_mom_63             = 0.0,
        uup_mom_63             = 0.0,
        gld_mom_21             = 0.0,
        shy_ief_mom_21         = 0.0,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_backtest(account_key: str = "topstep_100k", refresh: bool = False) -> BacktestResult:
    """
    Convenience function. Import or call from main.py or CLI.

    Args:
        account_key: Prop firm preset (e.g. "topstep_100k", "apex_100k_eod").
        refresh:     Force data re-fetch from yfinance.

    Returns:
        BacktestResult
    """
    from config import Config
    cfg = Config(account_key=account_key)
    engine = BacktestEngine(cfg)
    return engine.run(refresh_data=refresh)


if __name__ == "__main__":
    # Allow:  python -m backtest.engine [account_key] [--refresh]
    import argparse
    parser = argparse.ArgumentParser(description="NQ Regime Bot — 5-minute backtest")
    parser.add_argument("--account", default="topstep_100k",
                        help="Prop firm preset key (default: topstep_100k)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch data from yfinance")
    args = parser.parse_args()

    # Configure loguru output
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    run_backtest(account_key=args.account, refresh=args.refresh)
