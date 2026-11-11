"""Step 3: train the model, grade it honestly, and predict the next session.

What this script does
---------------------
1. Builds ~35 features per stock per day from past prices only.
2. Target: does the stock close higher on the next trading day (1) or not (0).
3. Walk-forward backtest: the model is retrained every `--step` days using only
   data that was available at that time, then scored on the days that follow.
   Every reported accuracy figure comes from these out-of-sample days.
4. Trains a final model on everything and predicts the latest day.
5. Writes web/data/predictions.json and predictions.js for the website.

Rules that keep the numbers honest
----------------------------------
* Features at day t use data up to the close of day t. Nothing later.
* Training rows stop `embargo` days before the first test day.
* Hyper-parameters are fixed up front. They are never tuned on test days.
* Confidence intervals are computed over days, not over stocks, because
  stocks move together on the same day.
* Market direction is treated as unpredictable. Each day's probabilities are
  centred on the trailing one-year share of stocks that rose, so the model only
  chooses which stocks lean up or down relative to the crowd.
"""
import argparse
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from common import MARKET, OUT_JS, OUT_JSON, PRICES, UNIVERSE


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def wide(px: pd.DataFrame, col: str) -> pd.DataFrame:
    return px.pivot(index="date", columns="ticker", values=col).sort_index()


def build_panel(px: pd.DataFrame, mkt: pd.DataFrame, min_dv: float, min_price: float):
    C, O, H, L, V = (wide(px, c) for c in ["close", "open", "high", "low", "volume"])
    dates, tickers = C.index, C.columns
    nd, nt = len(dates), len(tickers)

    r1 = C.pct_change(fill_method=None)
    F: dict[str, pd.DataFrame | np.ndarray] = {}

    for n in (1, 2, 3, 5, 10, 21, 63, 126):
        F[f"ret_{n}"] = C.pct_change(n, fill_method=None)
    for n in (5, 21, 63):
        F[f"vol_{n}"] = r1.rolling(n).std()
    for n in (21, 63, 126):
        F[f"ma_gap_{n}"] = C / C.rolling(n).mean() - 1
    gain = r1.clip(lower=0).rolling(14).mean()
    loss = (-r1.clip(upper=0)).rolling(14).mean()
    F["rsi_14"] = gain / (gain + loss)
    F["gap"] = O / C.shift(1) - 1
    F["intraday"] = C / O - 1
    rng = (H - L) / C
    F["range_rel"] = rng / rng.rolling(21).mean()
    F["vol_ratio"] = V / V.rolling(21).mean().replace(0, np.nan)
    dv = (C * V).rolling(21).mean()
    F["log_dv"] = np.log1p(dv)
    F["dist_high_126"] = C / C.rolling(126).max() - 1
    F["dist_low_126"] = C / C.rolling(126).min() - 1

    # market-wide features (same value for every stock on a given day)
    m = mkt.set_index("date").reindex(dates).ffill()
    spy = m["spy_close"]
    vix = m["vix_close"]
    spy_r1 = spy.pct_change()
    market = {
        "mkt_ret_1": spy_r1,
        "mkt_ret_5": spy.pct_change(5),
        "mkt_ret_21": spy.pct_change(21),
        "mkt_vol_21": spy_r1.rolling(21).std(),
        "vix": vix,
        "vix_chg_5": vix.pct_change(5),
    }
    for k, s in market.items():
        F[k] = np.repeat(s.to_numpy(dtype="float32")[:, None], nt, axis=1)
    F["rel_5"] = F["ret_5"].to_numpy() - F["mkt_ret_5"]
    F["rel_21"] = F["ret_21"].to_numpy() - F["mkt_ret_21"]

    # cross-sectional ranks (where the stock sits among all stocks that day)
    for k in ("ret_1", "ret_5", "ret_21", "vol_21"):
        F[f"rank_{k}"] = F[k].rank(axis=1, pct=True)

    nxt_close = C.shift(-1)
    nxt_ret = nxt_close / C - 1
    y = (nxt_close > C).astype("float32").where(nxt_close.notna())

    flat = {k: np.asarray(v, dtype="float32").ravel() for k, v in F.items()}
    X = pd.DataFrame(flat)
    di = np.repeat(np.arange(nd, dtype=np.int32), nt)
    ti = np.tile(np.arange(nt, dtype=np.int32), nd)
    close = C.to_numpy(dtype="float32").ravel()
    meta = pd.DataFrame(
        {
            "di": di,
            "ti": ti,
            "y": y.to_numpy(dtype="float32").ravel(),
            "nxt": nxt_ret.to_numpy(dtype="float32").ravel(),
            "close": close,
            "dv": dv.to_numpy(dtype="float32").ravel(),
        }
    )
    ok = (
        np.isfinite(close)
        & (close >= min_price)
        & (meta["dv"].to_numpy() >= min_dv)
        & np.isfinite(X["ret_126"].to_numpy())
        & np.isfinite(X["vol_21"].to_numpy())
    )
    X = X[ok].reset_index(drop=True)
    meta = meta[ok].reset_index(drop=True)
    return X, meta, dates, list(tickers)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def fit_model(X: pd.DataFrame, y: np.ndarray, iters: int, seed: int = 0):
    model = HistGradientBoostingClassifier(
        max_iter=iters,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=1000,
        l2_regularization=2.0,
        max_bins=64,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def trailing_base(meta, n_days: int, window: int = 252) -> np.ndarray:
    """Average share of stocks that rose, over the previous `window` days.

    The value for day d only uses labels of days up to d-1, which are known
    at the close of day d.
    """
    up = meta.groupby("di")["y"].mean().reindex(range(n_days))
    base = up.shift(1).rolling(window, min_periods=20).mean()
    return base.fillna(0.5).to_numpy()


def center_on_base(p: np.ndarray, di: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Market direction is treated as unpredictable.

    Each day's probabilities are shifted so their average equals the trailing
    up-rate. The model only decides which stocks lean up or down versus the crowd.
    """
    day_mean = pd.Series(p).groupby(di).transform("mean").to_numpy()
    return np.clip(p - day_mean + base[di], 0.02, 0.98).astype("float32")


def train_slice(meta, lo_day: int, hi_day: int, max_rows: int, rng):
    """Row positions for training days lo_day..hi_day (inclusive) that have labels."""
    di = meta["di"].to_numpy()
    a = np.searchsorted(di, lo_day, side="left")
    b = np.searchsorted(di, hi_day, side="right")
    idx = np.arange(a, b)
    idx = idx[np.isfinite(meta["y"].to_numpy()[idx])]
    if len(idx) > max_rows:
        idx = np.sort(rng.choice(idx, size=max_rows, replace=False))
    return idx


def walk_forward(X, meta, n_days, args):
    rng = np.random.default_rng(0)
    base = trailing_base(meta, n_days)
    first_day = int(meta["di"].min())
    start = first_day + args.min_train
    parts = []
    folds = 0
    s = start
    while s < n_days - 1:
        e = min(s + args.step, n_days - 1)           # last day has no label
        hi = s - 1 - args.embargo
        lo = max(first_day, hi - args.window + 1) if args.window else first_day
        tr = train_slice(meta, lo, hi, args.max_train_rows, rng)
        di = meta["di"].to_numpy()
        te = np.arange(np.searchsorted(di, s, "left"), np.searchsorted(di, e, "left"))
        te = te[np.isfinite(meta["y"].to_numpy()[te])]
        t0 = time.time()
        model = fit_model(X.iloc[tr], meta["y"].to_numpy()[tr], args.iters)
        p = center_on_base(model.predict_proba(X.iloc[te])[:, 1], di[te], base)
        parts.append(
            pd.DataFrame(
                {
                    "di": di[te],
                    "ti": meta["ti"].to_numpy()[te],
                    "p": p.astype("float32"),
                    "y": meta["y"].to_numpy()[te],
                    "nxt": meta["nxt"].to_numpy()[te],
                }
            )
        )
        folds += 1
        print(f"  fold {folds}: train {len(tr):,} rows, test days {s}-{e - 1}, {time.time() - t0:.0f}s")
        s = e
    return pd.concat(parts, ignore_index=True), folds


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def evaluate(oos: pd.DataFrame, dates: pd.DatetimeIndex):
    oos = oos.copy()
    oos["ok"] = ((oos["p"] > 0.5) == (oos["y"] > 0.5)).astype("float32")
    g = oos.groupby("di")
    daily = g["ok"].mean()
    up_day = g["y"].mean()
    nd = len(daily)
    acc = daily.mean()
    ci = 1.96 * daily.std(ddof=1) / np.sqrt(nd)
    diff = daily - up_day
    edge = diff.mean()
    edge_ci = 1.96 * diff.std(ddof=1) / np.sqrt(nd)

    auc = roc_auc_score(oos["y"], oos["p"])
    brier = brier_score_loss(oos["y"], oos["p"])

    # cross-sectional rank correlation between prediction and next-day return
    rp = g["p"].rank(pct=True)
    rn = g["nxt"].rank(pct=True)
    tmp = pd.DataFrame({"di": oos["di"].to_numpy(), "rp": rp.to_numpy(), "rn": rn.to_numpy()})
    tmp = tmp[tmp.groupby("di")["rp"].transform("size") >= 30]   # skip days with too few stocks
    with np.errstate(invalid="ignore", divide="ignore"):   # a day with identical predictions has no correlation
        ic = tmp.groupby("di").apply(lambda x: x["rp"].corr(x["rn"])).dropna()
    ic_t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))

    # top-decile minus bottom-decile next-day return, before any costs
    oos["rp"] = rp.to_numpy()
    top = oos[oos["rp"] > 0.9].groupby("di")["nxt"].mean()
    bot = oos[oos["rp"] <= 0.1].groupby("di")["nxt"].mean()
    spread = (top - bot).dropna()
    spread_bps = spread.mean() * 1e4
    spread_t = spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))

    # accuracy by confidence decile (graded per day, then averaged over days)
    oos["conf"] = (oos["p"] - 0.5).abs()
    oos["dec"] = pd.qcut(oos["conf"], 10, labels=False, duplicates="drop")
    per_day = oos.groupby(["dec", "di"])["ok"].mean()
    dec_rows = []
    for d, grp in oos.groupby("dec"):
        pdv = per_day.loc[d]
        dec_rows.append(
            {
                "d": int(d) + 1,
                "conf": r(grp["conf"].mean() * 100, 2),
                "acc": r(pdv.mean() * 100, 2),
                "ci": r(1.96 * pdv.std(ddof=1) / np.sqrt(len(pdv)) * 100, 2),
                "up": r(grp["y"].mean() * 100, 2),
                "pred_up": r((grp["p"] > 0.5).mean() * 100, 1),
                "n": int(len(grp)),
            }
        )

    # calibration: predicted probability vs how often the stock really rose
    oos["pbin"] = pd.qcut(oos["p"], 10, labels=False, duplicates="drop")
    cal = [
        {"p": r(gp["p"].mean() * 100, 2), "hit": r(gp["y"].mean() * 100, 2), "n": int(len(gp))}
        for _, gp in oos.groupby("pbin")
    ]

    # rolling 63-day accuracy
    roll = daily.rolling(63).mean().dropna()
    roll_b = up_day.rolling(63).mean().reindex(roll.index)
    rolling = {
        "d": [dates[i].strftime("%Y-%m-%d") for i in roll.index],
        "a": [r(v * 100, 2) for v in roll.values],
        "b": [r(v * 100, 2) for v in roll_b.values],
    }

    metrics = {
        "accuracy": r(acc * 100, 2),
        "ci95": r(ci * 100, 2),
        "base_up": r(up_day.mean() * 100, 2),
        "edge_vs_up": r(edge * 100, 2),
        "edge_ci95": r(edge_ci * 100, 2),
        "auc": r(auc, 4),
        "brier": r(brier, 4),
        "ic": r(ic.mean(), 4),
        "ic_t": r(ic_t, 2),
        "spread_bps": r(spread_bps, 2),
        "spread_t": r(spread_t, 2),
        "n_days": int(nd),
        "n_preds": int(len(oos)),
        "oos_start": dates[int(daily.index.min())].strftime("%Y-%m-%d"),
        "oos_end": dates[int(daily.index.max())].strftime("%Y-%m-%d"),
    }
    return metrics, dec_rows, cal, rolling


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "demo"], default="live",
                    help="'demo' marks the output as synthetic so the site shows a DEMO banner")
    ap.add_argument("--min-train", type=int, default=504, help="first training window, in trading days")
    ap.add_argument("--step", type=int, default=63, help="retrain every N trading days")
    ap.add_argument("--window", type=int, default=756, help="rolling training window in days (0 = expanding)")
    ap.add_argument("--embargo", type=int, default=1, help="days skipped between train and test")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--max-train-rows", type=int, default=1_500_000)
    ap.add_argument("--min-dv", type=float, default=1_000_000, help="min 21-day average dollar volume")
    ap.add_argument("--min-price", type=float, default=1.0)
    ap.add_argument("--eval-cache", default="",
                    help="JSON file that stores the last walk-forward result (used with --eval-every)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="re-run the slow walk-forward backtest only every N days; in between, reuse the cached "
                         "grading and just refresh the predictions (0 = always re-run)")
    args = ap.parse_args()

    t_start = time.time()
    px = pd.read_parquet(PRICES)
    mkt = pd.read_parquet(MARKET)
    uni = pd.read_csv(UNIVERSE).set_index("ticker")
    px["date"] = pd.to_datetime(px["date"])
    mkt["date"] = pd.to_datetime(mkt["date"])

    print("Building features ...")
    X, meta, dates, tickers = build_panel(px, mkt, args.min_dv, args.min_price)
    n_days = len(dates)
    print(f"  {len(X):,} rows, {X.shape[1]} features, {len(tickers):,} tickers, {n_days} days")

    params = {k: getattr(args, k) for k in
              ["min_train", "step", "window", "embargo", "iters", "max_train_rows", "min_dv", "min_price", "mode"]}
    params["n_features"] = int(X.shape[1])
    cache = Path(args.eval_cache) if args.eval_cache else None
    reuse = None
    if cache is not None and args.eval_every > 0 and cache.exists():
        try:
            c = json.loads(cache.read_text(encoding="utf-8"))
            age = (date.today() - date.fromisoformat(c["computed_on"])).days
            if c.get("params") == params and 0 <= age < args.eval_every:
                reuse = c
        except Exception as e:  # noqa: BLE001
            print(f"  could not use eval cache ({e}); re-running the backtest")

    if reuse is not None:
        metrics, deciles, calibration, rolling, folds = (
            reuse["metrics"], reuse["deciles"], reuse["calibration"], reuse["rolling"], reuse["folds"])
        eval_computed = reuse["computed_on"]
        print(f"Walk-forward backtest: reusing the result from {eval_computed} (refreshed every {args.eval_every} days)")
    else:
        print("Walk-forward backtest ...")
        oos, folds = walk_forward(X, meta, n_days, args)
        metrics, deciles, calibration, rolling = evaluate(oos, dates)
        eval_computed = date.today().isoformat()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "computed_on": eval_computed, "params": params, "metrics": metrics, "deciles": deciles,
                "calibration": calibration, "rolling": rolling, "folds": folds,
            }, allow_nan=False), encoding="utf-8")
    print(
        f"  out-of-sample accuracy {metrics['accuracy']}% +/- {metrics['ci95']} "
        f"(always-up baseline {metrics['base_up']}%, AUC {metrics['auc']})"
    )

    print("Final model and next-session predictions ...")
    rng = np.random.default_rng(1)
    tr = train_slice(meta, max(0, n_days - 2 - (args.window or n_days) + 1), n_days - 2, args.max_train_rows, rng)
    final = fit_model(X.iloc[tr], meta["y"].to_numpy()[tr], args.iters)
    last = np.where(meta["di"].to_numpy() == n_days - 1)[0]
    p_last = center_on_base(
        final.predict_proba(X.iloc[last])[:, 1], meta["di"].to_numpy()[last], trailing_base(meta, n_days)
    )

    stocks = []
    for k, row in enumerate(last):
        t = tickers[int(meta["ti"].iat[row])]
        info = uni.loc[t] if t in uni.index else None
        stocks.append(
            {
                "t": t,
                "n": str(info["name"]) if info is not None else t,
                "s": str(info["sector"]) if info is not None else "Other",
                "p": r(p_last[k], 4),
                "c": r(meta["close"].iat[row], 2),
                "r1": r(X["ret_1"].iat[row] * 100, 2),
                "r5": r(X["ret_5"].iat[row] * 100, 2),
                "r21": r(X["ret_21"].iat[row] * 100, 2),
                "v": r(X["vol_21"].iat[row] * np.sqrt(252) * 100, 1),
            }
        )
    stocks.sort(key=lambda s: -s["p"])

    out = {
        "meta": {
            "mode": args.mode,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "as_of": dates[-1].strftime("%Y-%m-%d"),
            "universe": len(stocks),
            "model": "Gradient-boosted trees (scikit-learn HistGradientBoosting)",
            "n_features": int(X.shape[1]),
            "folds": folds,
            "eval_computed": eval_computed,
            "retrain_every_days": args.step,
            "train_window_days": args.window,
            "min_dollar_volume": args.min_dv,
            "survivorship_bias": True,
        },
        "metrics": metrics,
        "deciles": deciles,
        "calibration": calibration,
        "rolling": rolling,
        "stocks": stocks,
    }
    payload = json.dumps(out, separators=(",", ":"), allow_nan=False)
    OUT_JSON.write_text(payload, encoding="utf-8")
    OUT_JS.write_text("window.NEXTBELL=" + payload + ";\n", encoding="utf-8")
    print(f"Wrote {OUT_JS} ({len(payload) / 1e6:.2f} MB) in {(time.time() - t_start) / 60:.1f} min")
    print("Open web/index.html in your browser.")


if __name__ == "__main__":
    main()
