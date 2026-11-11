"""Step 1: build the list of US stocks (Russell 3000) with sectors.

Primary source : iShares Russell 3000 ETF (IWV) holdings file.
Fallback       : S&P 500 + 400 + 600 from Wikipedia (about 1,500 stocks).
Manual         : --file my_tickers.csv  (columns: ticker, name, sector)

If the download fails and a universe.csv from an earlier run exists, that file
is kept (a daily job should not silently shrink from 3,000 to 1,500 stocks).
Use --source sp1500 to force the Wikipedia list.

Output: data/universe.csv
"""
import argparse
import io
import sys

import pandas as pd
import requests

from common import UNIVERSE

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
IWV_URL = (
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)


def from_ishares() -> pd.DataFrame:
    r = requests.get(IWV_URL, headers=UA, timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Ticker,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df = df[df["Asset Class"].astype(str).str.strip() == "Equity"]
    out = pd.DataFrame(
        {
            "ticker": df["Ticker"].astype(str).str.strip(),
            "name": df["Name"].astype(str).str.strip(),
            "sector": df["Sector"].astype(str).str.strip(),
        }
    )
    return out


def from_wikipedia() -> pd.DataFrame:
    pages = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    ]
    frames = []
    for url in pages:
        html = requests.get(url, headers=UA, timeout=60).text
        for t in pd.read_html(io.StringIO(html)):
            cols = {c: str(c) for c in t.columns}
            sym = next((c for c, s in cols.items() if "Symbol" in s or "Ticker" in s), None)
            nam = next((c for c, s in cols.items() if s in ("Security", "Company")), None)
            sec = next((c for c, s in cols.items() if "Sector" in s), None)
            if sym is not None and nam is not None and sec is not None:
                frames.append(
                    pd.DataFrame({"ticker": t[sym], "name": t[nam], "sector": t[sec]})
                )
                break
    if not frames:
        raise RuntimeError("Could not parse the Wikipedia index tables")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    df = df[df["ticker"].str.fullmatch(r"[A-Z]{1,5}(-[A-Z])?")]
    df["sector"] = df["sector"].replace({"nan": "Other", "-": "Other"}).fillna("Other")
    return df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="CSV with columns ticker,name,sector")
    ap.add_argument("--source", choices=["auto", "ishares", "sp1500"], default="auto")
    args = ap.parse_args()

    if args.file:
        df = pd.read_csv(args.file)
    elif args.source in ("auto", "ishares"):
        try:
            df = from_ishares()
            print("Source: iShares Russell 3000 (IWV) holdings")
        except Exception as e:  # noqa: BLE001
            if args.source == "ishares":
                sys.exit(f"iShares download failed: {e}")
            if UNIVERSE.exists():
                n = len(pd.read_csv(UNIVERSE))
                print(f"iShares download failed ({e}); keeping the existing universe.csv ({n:,} tickers)")
                return
            print(f"iShares download failed ({e}); falling back to S&P 1500 from Wikipedia")
            df = from_wikipedia()
    else:
        df = from_wikipedia()

    df = clean(df)
    df.to_csv(UNIVERSE, index=False)
    print(f"Saved {len(df):,} tickers -> {UNIVERSE}")
    print(df["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
