"""
main.py — NQ Regime Bot Entry Point

Usage:
    # Run a backtest on 5-minute bars (yfinance, last 60 days)
    python main.py --mode backtest

    # Run backtest for a specific prop firm account
    python main.py --mode backtest --account topstep_50k

    # Force re-fetch data from yfinance (bypass cache)
    python main.py --mode backtest --refresh

    # Paper / live trading (IB required — coming in Phase 6)
    python main.py --mode paper
    python main.py --mode live

Available account keys:
    topstep_50k, topstep_100k, topstep_150k,
    apex_100k_eod, apex_150k_intraday
"""

import argparse
import sys
from loguru import logger


def _configure_logging(verbose: bool = False) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/nq_bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="14 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )


def run_backtest(account_key: str, refresh: bool) -> None:
    from backtest.engine import BacktestEngine
    from config import Config

    logger.info(f"[main] Backtest mode | account={account_key}")
    cfg    = Config(account_key=account_key)
    engine = BacktestEngine(cfg)
    result = engine.run(refresh_data=refresh)

    # Save trade log to CSV
    if result.trades:
        import pandas as pd
        from pathlib import Path
        Path("results").mkdir(exist_ok=True)
        rows = []
        for t in result.trades:
            rows.append({
                "direction":        t.direction,
                "entry_time":       t.entry_time,
                "exit_time":        t.exit_time,
                "entry_price":      t.entry_price,
                "stop_initial":     t.stop_price_initial,
                "target":           t.target_price,
                "exit_price":       t.exit_price,
                "contracts":        t.contracts,
                "instrument":       t.instrument,
                "pnl_dollars":      t.pnl_dollars,
                "r_multiple":       round(t.r_multiple, 2),
                "exit_reason":      t.exit_reason,
                "conviction_score": t.conviction_score,
                "regime":           t.regime,
                "macro_bias":       t.macro_bias,
                "partial_executed": t.partial_executed,
            })
        df = pd.DataFrame(rows)
        fname = f"results/backtest_{account_key}.csv"
        df.to_csv(fname, index=False)
        logger.info(f"[main] Trade log saved: {fname}")

        # Save equity curve
        eq_fname = f"results/equity_{account_key}.csv"
        result.equity_curve.to_csv(eq_fname, header=["balance"])
        logger.info(f"[main] Equity curve saved: {eq_fname}")


def run_paper(account_key: str) -> None:
    logger.warning("[main] Paper trading mode not yet implemented (Phase 6)")
    logger.info("[main] Tip: run --mode backtest first to validate the strategy")


def run_live(account_key: str) -> None:
    logger.warning("[main] Live trading mode not yet implemented (Phase 6)")
    logger.warning("[main] NEVER run live without completing backtest validation first!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NQ Regime Bot — 5-layer signal trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Execution mode (default: backtest)",
    )
    parser.add_argument(
        "--account",
        default="topstep_100k",
        help="Prop firm account preset (default: topstep_100k)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch data from yfinance (bypass cache)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    logger.info(
        f"[main] NQ Regime Bot | mode={args.mode} | account={args.account}"
    )

    if args.mode == "backtest":
        run_backtest(args.account, args.refresh)
    elif args.mode == "paper":
        run_paper(args.account)
    elif args.mode == "live":
        run_live(args.account)


if __name__ == "__main__":
    main()
