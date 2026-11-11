"""Step 2: download daily prices for every ticker in data/universe.csv.

Uses yfinance (free Yahoo Finance data). Prices are adjusted for splits and
dividends. Also downloads SPY and the VIX as market-wide features.

Output: data/prices.parquet, data/market.parquet

Modes
-----
default        download the full history (--years, default 6) for every ticker.
--incremental  keep the cached history and only download the last ~10 days,
               then splice them on. Used for the daily automatic update.
               Splits and dividends re-adjust old prices at Yahoo, so the cached
               history is rescaled to match the fresh bars at the splice point.
               New tickers get a full download; tickers that failed before are
               retried on Mondays.

Run this on your own computer or on GitHub Actions; Yahoo blocks many other
cloud servers.
"""
import argparse
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from common import MARKET, PRICES, UNIVERSE

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def fetch(batch, start, retries=3):
    for attempt in range(retries):
        try:
            return yf.download(
                batch,
                start=start,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}/{retries}: {e}")
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def split_batch(raw: pd.DataFrame, batch) -> list[pd.DataFrame]:
    out = []
    if raw.empty:
        return out
    for t in batch:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub[FIELDS].dropna(subset=["Close"])
        except KeyError:
            continue
        if sub.empty:
            continue
        sub = sub.rename(columns=str.lower)
        sub.index = pd.to_datetime(sub.index).tz_localize(None)
        sub.index.name = "date"
        sub.insert(0, "ticker", t)
        out.append(sub.reset_index())
    return out


def download(tickers, start, batch_size):
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        got = split_batch(fetch(batch, start), batch)
        frames.extend(got)
        print(f"  {min(i + batch_size, len(tickers)):>5}/{len(tickers)}  ok in batch: {len(got)}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_incremental(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Splice fresh bars onto cached history.

    Yahoo re-adjusts past prices after a split or dividend. For every ticker we
    compare old and new close on the first overlapping day and rescale the older
    cached rows by that ratio, so returns across the splice point stay correct.
    Volume is only rescaled when the ratio looks like a split (more than 30%).
    """
    key = ["ticker", "date"]
    old = old.copy()
    new = new.copy()
    old["date"] = pd.to_datetime(old["date"])
    new["date"] = pd.to_datetime(new["date"])
    ov = old[key + ["close"]].merge(new[key + ["close"]], on=key, suffixes=("_old", "_new"))
    if not ov.empty:
        first = ov.sort_values("date").groupby("ticker").first()
        first["factor"] = first["close_new"] / first["close_old"]
        first = first.rename(columns={"date": "d0"})[["d0", "factor"]]
        old = old.join(first, on="ticker")
        hit = old["factor"].notna() & (old["date"] < old["d0"]) & ((old["factor"] - 1).abs() > 1e-4)
        for c in ["open", "high", "low", "close"]:
            old.loc[hit, c] = old.loc[hit, c] * old.loc[hit, "factor"]
        split = hit & (np.log(old["factor"].where(hit, 1.0)).abs() > np.log(1.3))
        old.loc[split, "volume"] = old.loc[split, "volume"] / old.loc[split, "factor"]
        old = old.drop(columns=["d0", "factor"])
    out = pd.concat([old, new[old.columns]], ignore_index=True)
    out = out.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=6.0, help="history length")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tickers (testing)")
    ap.add_argument("--incremental", action="store_true", help="only fetch recent days and splice onto the cache")
    ap.add_argument("--retry-missing", action="store_true", help="in incremental mode, retry tickers that have no cached data now")
    ap.add_argument("--min-coverage", type=float, default=0.6,
                    help="abort if fewer than this share of tickers have the latest date (guards against a half-blocked Yahoo)")
    args = ap.parse_args()

    uni = pd.read_csv(UNIVERSE)
    tickers = uni["ticker"].tolist()
    if args.limit:
        tickers = tickers[: args.limit]
    start = (date.today() - timedelta(days=int(args.years * 365.25))).isoformat()

    if args.incremental and PRICES.exists():
        old = pd.read_parquet(PRICES)
        old["date"] = pd.to_datetime(old["date"])
        old = old[old["ticker"].isin(tickers)]
        last = old["date"].max()
        recent_start = (last - timedelta(days=10)).date().isoformat()
        have = set(old["ticker"].unique())
        todo = [t for t in tickers if t in have]
        missing = [t for t in tickers if t not in have]
        print(f"Incremental: {len(todo):,} cached tickers from {recent_start} (cache ends {last.date()})")
        new = download(todo, recent_start, args.batch)
        if new.empty:
            raise SystemExit("No recent data downloaded. Yahoo may be blocking this machine; keeping the old files.")
        px = merge_incremental(old, new)
        if missing and (args.retry_missing or date.today().weekday() == 0):
            print(f"Retrying {len(missing):,} tickers that have no cached history")
            full = download(missing, start, args.batch)
            if not full.empty:
                px = pd.concat([px, full], ignore_index=True)
        cutoff = pd.Timestamp(start)
        px = px[px["date"] >= cutoff]
    else:
        print(f"{len(tickers):,} tickers from {start}")
        px = download(tickers, start, args.batch)
        if px.empty:
            raise SystemExit("No data downloaded. Check your internet connection.")

    counts = px.groupby("ticker")["close"].count()
    keep = counts[counts >= 400].index
    px = px[px["ticker"].isin(keep)].copy()
    latest = px["date"].max()
    coverage = px.loc[px["date"] == latest, "ticker"].nunique() / max(1, px["ticker"].nunique())
    print(f"Latest date {pd.Timestamp(latest).date()}, {coverage:.0%} of tickers have it")
    if coverage < args.min_coverage:
        raise SystemExit(
            f"Only {coverage:.0%} of tickers have the latest date (need {args.min_coverage:.0%}). "
            "Not saving, so the previous data stays in place."
        )
    for c in ["open", "high", "low", "close", "volume"]:
        px[c] = px[c].astype("float32")
    px.to_parquet(PRICES, index=False)
    print(f"Saved {px['ticker'].nunique():,} tickers, {len(px):,} rows -> {PRICES}")

    mk = fetch(["SPY", "^VIX"], start)
    spy = mk["SPY"]["Close"].rename("spy_close")
    vix = mk["^VIX"]["Close"].rename("vix_close")
    m = pd.concat([spy, vix], axis=1).dropna(how="all")
    m.index = pd.to_datetime(m.index).tz_localize(None)
    m.index.name = "date"
    m.reset_index().to_parquet(MARKET, index=False)
    print(f"Saved market series -> {MARKET}")


if __name__ == "__main__":
    main()
