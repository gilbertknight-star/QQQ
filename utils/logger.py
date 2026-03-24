from loguru import logger
from pathlib import Path


def get_logger(name: str = "nq_bot"):
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / f"{name}.log", rotation="1 day", retention="30 days", level="INFO")
    return logger
