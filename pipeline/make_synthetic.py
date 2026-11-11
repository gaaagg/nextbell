"""Generate FAKE price data in the same format as step 2.

Used to test the pipeline end to end and to produce the demo shown on the site.
Tickers contain a digit (for example KTR4) so they can never be mistaken for
real companies. Returns are a random walk with a market factor, sector factors
and, optionally, a tiny short-term reversal signal (--signal).
With --signal 0 there is nothing to find, so any accuracy meaningfully above
the base rate would mean the pipeline leaks future data.
"""
import argparse

import numpy as np
import pandas as pd

from common import MARKET, PRICES, UNIVERSE

SECTORS = [
    "Information Technology", "Health Care", "Financials", "Industrials",
    "Consumer Discretionary", "Communication", "Consumer Staples",
    "Energy", "Real Estate", "Utilities", "Materials",
]
WEIGHTS = np.array([17, 14, 13, 12, 11, 7, 6, 5, 5, 4, 4], dtype=float)
WORDS = ["Alta", "Brio", "Cobalt", "Delta", "Ember", "Fjord", "Gantry", "Helix", "Ionic",
         "Juniper", "Kestrel", "Lumen", "Meridian", "Nimbus", "Orbit", "Pylon", "Quartz",
         "Radial", "Summit", "Tidal", "Umbra", "Vertex", "Willow", "Xenon", "Yarrow", "Zenith"]
SUFFIX = ["Systems", "Holdings", "Labs", "Partners", "Group", "Energy", "Bancorp", "Foods",
          "Therapeutics", "Networks", "Industries", "Realty", "Materials", "Media"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=3000)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--signal", type=float, default=0.03, help="strength of planted reversal (0 = none)")
    ap.add_argument("--end", default="2026-11-10")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    nt, nd = args.tickers, args.days

    dates = pd.bdate_range(end=args.end, periods=nd)
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    tickers = set()
    while len(tickers) < nt:
        tickers.add("".join(rng.choice(letters, 3)) + str(rng.integers(1, 10)))
    tickers = sorted(tickers)

    sec_idx = rng.choice(len(SECTORS), size=nt, p=WEIGHTS / WEIGHTS.sum())
    names = [f"{rng.choice(WORDS)} {rng.choice(SUFFIX)}" for _ in range(nt)]
    pd.DataFrame(
        {"ticker": tickers, "name": names, "sector": [SECTORS[i] for i in sec_idx]}
    ).to_csv(UNIVERSE, index=False)

    # return model
    mkt = rng.normal(0.0004, 0.010, nd)
    sec_f = rng.normal(0, 0.006, (nd, len(SECTORS)))
    beta = rng.normal(1.0, 0.3, nt)
    sigma = np.exp(rng.normal(np.log(0.015), 0.35, nt))
    mu = rng.normal(0.0002, 0.0002, nt)

    ret = np.zeros((nd, nt), dtype="float64")
    for t in range(nd):
        base = (
            beta * mkt[t] + sec_f[t, sec_idx] + mu + sigma * rng.standard_normal(nt)
        )
        if args.signal and t >= 5:
            r5 = ret[t - 5 : t].sum(axis=0)
            base = base - args.signal * (r5 - r5.mean())   # reversal against the crowd, not the market
        ret[t] = base
    ret = np.clip(ret, -0.4, 0.4)

    start = np.exp(rng.normal(3.5, 0.8, nt))
    close = start * np.exp(np.cumsum(ret, axis=0))
    prev = np.vstack([start[None, :], close[:-1]])
    openp = prev * (1 + rng.normal(0, 0.004, (nd, nt)))
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.005, (nd, nt))))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.005, (nd, nt))))
    vol = np.exp(rng.normal(13.5, 1.0, nt))[None, :] * (1 + 6 * np.abs(ret)) * rng.lognormal(0, 0.3, (nd, nt))

    df = pd.DataFrame(
        {
            "date": np.repeat(dates.values, nt),
            "ticker": np.tile(np.array(tickers), nd),
            "open": openp.ravel().astype("float32"),
            "high": high.ravel().astype("float32"),
            "low": low.ravel().astype("float32"),
            "close": close.ravel().astype("float32"),
            "volume": vol.ravel().astype("float32"),
        }
    )
    df.to_parquet(PRICES, index=False)

    spy = 400 * np.exp(np.cumsum(mkt))
    vix = 15 + 900 * pd.Series(mkt).rolling(10, min_periods=1).std().fillna(0.01).to_numpy()
    pd.DataFrame({"date": dates, "spy_close": spy, "vix_close": vix}).to_parquet(MARKET, index=False)
    print(f"Synthetic data: {nt} tickers x {nd} days, signal={args.signal}")


if __name__ == "__main__":
    main()
