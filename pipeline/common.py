"""Shared paths for the Nextbell pipeline."""
import os
from pathlib import Path

# NEXTBELL_ROOT lets tests write somewhere other than the project folder.
ROOT = Path(os.environ.get("NEXTBELL_ROOT") or Path(__file__).resolve().parent.parent)
DATA = ROOT / "data"
WEB_DATA = ROOT / "web" / "data"
DATA.mkdir(parents=True, exist_ok=True)
WEB_DATA.mkdir(parents=True, exist_ok=True)

UNIVERSE = DATA / "universe.csv"      # ticker, name, sector
PRICES = DATA / "prices.parquet"      # date, ticker, open, high, low, close, volume
MARKET = DATA / "market.parquet"      # date, spy_close, vix_close
OUT_JSON = WEB_DATA / "predictions.json"
OUT_JS = WEB_DATA / "predictions.js"
