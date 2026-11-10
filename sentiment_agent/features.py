"""Builds the point-in-time feature panel.

The rule everything here follows: bar n's features may only contain what was public before bar n+1 opened.
Posts are bucketed by that cutoff, so overnight posts inform the first bar of the next day, and all rolling
statistics look strictly backwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

GROUPS = ("forum", "chat", "news")
PRICE_FEATURES = ["r1", "r3", "r6", "vol", "rsi", "range_pos", "mkt_r1", "first_bar"]
SENTIMENT_FEATURES = [f"ewm_{g}" for g in GROUPS] + [f"attn_{g}" for g in GROUPS] + ["ewm_social", "gap_social_news"]
ALL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES


def post_weight(posts: pd.DataFrame) -> pd.Series:
    """Engagement weight. Downvotes discount a post, and a post naming N tickers counts 1/N for each."""
    num = lambda c: pd.to_numeric(posts[c], errors="coerce").fillna(0).clip(lower=0).astype(float)
    likes, dislikes = num("likes"), num("dislikes")
    engagement = likes + num("reposts") + num("replies")
    agreement = (likes + 3) / (likes + dislikes + 3)
    size = 1 + 0.5 * np.log1p(engagement) + 0.1 * np.log1p(num("views")) + 0.1 * np.log1p(num("author_followers"))
    n_tickers = pd.to_numeric(posts["n_tickers"], errors="coerce").fillna(1).clip(lower=1)
    weight = size * agreement / n_tickers
    return weight.where(posts["group"] != "news", 3.0)  # articles have no engagement data; fixed editorial weight


def price_features(bars: pd.DataFrame, index_bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    log_close = np.log(df["close"])
    df["r1"] = log_close.diff()
    df["r3"] = log_close.diff(3)
    df["r6"] = log_close.diff(6)
    df["vol"] = df["r1"].rolling(12, min_periods=6).std()
    gain = df["r1"].clip(lower=0).ewm(alpha=1 / 14, min_periods=6).mean()
    loss = (-df["r1"].clip(upper=0)).ewm(alpha=1 / 14, min_periods=6).mean()
    df["rsi"] = (gain / (gain + loss + 1e-12)) - 0.5
    low, high = df["low"].rolling(20, min_periods=8).min(), df["high"].rolling(20, min_periods=8).max()
    df["range_pos"] = (df["close"] - low) / (high - low + 1e-12) - 0.5
    idx = index_bars[["end", "close"]].assign(mkt_r1=lambda d: np.log(d["close"]).diff())[["end", "mkt_r1"]]
    df = pd.merge_asof(df.sort_values("end"), idx.sort_values("end"), on="end", direction="backward")
    df["first_bar"] = (df["start"].dt.date != df["start"].shift(1).dt.date).astype(float)
    df["y"] = log_close.shift(-1) - log_close
    df["oc_next"] = np.log(df["close"].shift(-1) / df["open"].shift(-1))
    df["close_next"] = df["close"].shift(-1)
    df["target_end"] = df["end"].shift(-1)
    return df


def _attention(rate: np.ndarray, covered: np.ndarray, min_periods: int = 6) -> np.ndarray:
    """Expanding z-score of posting rate, using only earlier bars."""
    out = np.zeros(len(rate))
    seen: list[float] = []
    for i, value in enumerate(rate):
        if not covered[i]:
            continue
        if len(seen) >= min_periods:
            std = float(np.std(seen))
            out[i] = np.clip((value - np.mean(seen)) / std, -4, 4) if std > 0 else 0.0
        seen.append(value)
    return out


def _group_series(cutoffs: np.ndarray, hours: np.ndarray, posts: pd.DataFrame, halflife: float, prior: float,
                  latency: pd.Timedelta, coverage_start: pd.Timestamp | None) -> dict[str, np.ndarray]:
    m = len(cutoffs)
    n, wsum, wscore = np.zeros(m), np.zeros(m), np.zeros(m)
    if len(posts):
        usable_at = (posts["ts_utc"] + latency).astype("datetime64[ns, UTC]").to_numpy(dtype="datetime64[ns]")
        bucket = np.searchsorted(cutoffs, usable_at, side="left")
        mask = bucket < m  # later posts are "pending": not known by the last cutoff
        idx, w = bucket[mask], posts["weight"].to_numpy(float)[mask]
        s = posts["score"].to_numpy(float)[mask]
        n = np.bincount(idx, minlength=m).astype(float)
        wsum = np.bincount(idx, weights=w, minlength=m)
        wscore = np.bincount(idx, weights=w * s, minlength=m)
    covered = np.ones(m, dtype=bool)
    if coverage_start is not None:
        covered = cutoffs >= np.datetime64(coverage_start.tz_convert("UTC").tz_localize(None))
    decay = 0.5 ** (1 / halflife)
    num, den, ewm = 0.0, 0.0, np.zeros(m)
    for i in range(m):
        num, den = decay * num + wscore[i], decay * den + wsum[i]
        ewm[i] = num / (den + prior)
    rate = np.log1p(n / hours)
    return {"n": n, "ewm": ewm, "attn": _attention(rate, covered), "wsum": wsum, "wscore": wscore}


def build_panel(cfg: Config, bars: dict[str, pd.DataFrame], index_bars: dict[str, pd.DataFrame], posts: pd.DataFrame,
                cov: pd.DataFrame, window_start: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """One row per ticker per bar inside the sentiment window. `posts` needs score and weight columns."""
    latency = pd.Timedelta(minutes=cfg.post_latency_minutes)
    halflife, prior = cfg.sentiment["ewm_halflife_bars"], cfg.sentiment["prior_weight"]
    frames = []
    for ticker in cfg.tickers:
        if ticker.symbol not in bars or bars[ticker.symbol].empty:
            continue
        ex = cfg.exchanges[ticker.exchange]
        df = price_features(bars[ticker.symbol], index_bars[ex.key])
        df = df[df["end"] >= window_start].reset_index(drop=True)
        if df.empty:
            continue
        cutoff = df["next_start"].where(df["next_start"] <= now, now)
        df["cutoff"] = cutoff.astype("datetime64[ns, UTC]")
        # The next bar opens a new session (overnight or weekend), so its return includes the gap since the close.
        df["overnight"] = (df["next_start"] - df["end"]) > pd.Timedelta(hours=3)
        cuts = df["cutoff"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        previous = df["cutoff"].shift(1).fillna(window_start)
        hours = ((df["cutoff"] - previous).dt.total_seconds() / 3600).clip(lower=1 / 6).to_numpy()
        df["symbol"], df["exchange"] = ticker.symbol, ticker.exchange
        mine = posts[posts["symbol"] == ticker.symbol] if len(posts) else posts
        social_num, social_den = np.zeros(len(df)), np.zeros(len(df))
        for group in GROUPS:
            subset = mine[mine["group"] == group] if len(mine) else mine
            start = None
            if group == "forum" and len(cov):
                row = cov[(cov["symbol"] == ticker.symbol) & (cov["source"].isin(["lihkg", "naver"])) & cov["truncated"]]
                start = row["coverage_start"].max() if len(row) else None
            series = _group_series(cuts, hours, subset, halflife, prior, latency, start)
            df[f"n_{group}"], df[f"ewm_{group}"], df[f"attn_{group}"] = series["n"], series["ewm"], series["attn"]
            if group != "news":
                social_num += series["wscore"]
                social_den += series["wsum"]
        decay, num, den, ewm = 0.5 ** (1 / halflife), 0.0, 0.0, np.zeros(len(df))
        for i in range(len(df)):
            num, den = decay * num + social_num[i], decay * den + social_den[i]
            ewm[i] = num / (den + prior)
        df["ewm_social"] = ewm
        df["gap_social_news"] = df["ewm_social"] - df["ewm_news"]
        frames.append(df)
    if not frames:
        raise RuntimeError("no price bars available inside the sentiment window")
    panel = pd.concat(frames, ignore_index=True)
    panel[PRICE_FEATURES] = panel[PRICE_FEATURES].fillna(0.0)
    return panel.sort_values(["cutoff", "symbol"]).reset_index(drop=True)
