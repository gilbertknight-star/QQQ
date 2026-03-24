"""
Entry point for the NQ/MNQ Weather-Pattern Trading Bot.
"""

from utils import load_config, get_logger
from data import DataPipeline

logger = get_logger("nq_bot")


def main():
    config = load_config()
    logger.info(f"Starting {config['strategy']['name']}")
    logger.info(f"Instrument: {config['strategy']['instrument']} | Mode: paper")

    # Phase 1: Data pipeline
    pipe = DataPipeline(config)
    df_5m = pipe.get(timeframe="5m")
    df_1h = pipe.get(timeframe="1h")
    df_1d = pipe.get(timeframe="1d")
    logger.info(f"5m bars: {len(df_5m)} | 1h bars: {len(df_1h)} | 1d bars: {len(df_1d)}")

    # Phase 2: Kalman filter
    # Phase 3: HMM regime detection
    # Phase 4: Market structure + signal generation
    # Phase 5: Backtesting
    # Phase 6: Live trading


if __name__ == "__main__":
    main()
