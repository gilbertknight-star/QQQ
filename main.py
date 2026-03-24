"""
Entry point for the NQ/MNQ Weather-Pattern Trading Bot.
"""

from utils import load_config, get_logger

logger = get_logger("nq_bot")


def main():
    config = load_config()
    logger.info(f"Starting {config['strategy']['name']}")
    logger.info(f"Instrument: {config['strategy']['instrument']} | Mode: paper")

    # Phase 1: Data pipeline
    # Phase 2: Kalman filter
    # Phase 3: HMM regime detection
    # Phase 4: Market structure + signal generation
    # Phase 5: Backtesting
    # Phase 6: Live trading


if __name__ == "__main__":
    main()
