"""Logging setup. Author: Pradeep Kumar Verma"""
import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")


def setup_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_DIR / "rag_system.log")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger
