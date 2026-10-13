"""Loads API credentials from .env and strategy settings from .conf files."""
import configparser
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Credentials:
    api_key: str
    secret_key: str
    paper: bool


def load_credentials() -> Credentials:
    """Read Alpaca keys from .env (local) or environment variables (cloud).

    Paper trading is the default. Real money requires ALPACA_PAPER=false on purpose.
    """
    load_dotenv(ROOT / ".env")
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise SystemExit(
            "Missing Alpaca keys. Copy .env.example to .env and fill in "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY (or set them as GitHub secrets)."
        )
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
    return Credentials(key, secret, paper)


def load_conf(path) -> configparser.ConfigParser:
    """Read an INI-style .conf file. Comments may start with # or ;."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cp.read(path)
    return cp
