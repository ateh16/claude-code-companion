"""Log to a file and the screen, with timestamps in New York (market) time."""
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from common.settings import ROOT

NY = ZoneInfo("America/New_York")


class _NYFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, NY).strftime("%Y-%m-%d %H:%M:%S ET")


def get_logger(name: str, logfile: str) -> logging.Logger:
    path = Path(logfile)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = _NYFormatter("%(asctime)s | %(levelname)-7s | %(message)s")
    fh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
