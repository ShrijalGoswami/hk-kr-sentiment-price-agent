"""Turns scored posts, levels and forecasts into the per-ticker record the brief asks for."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Config, Ticker
from .features import GROUPS
from .levels import find_levels
from .model import ModelResult


def _weighted(frame: pd.DataFrame, prior: float) -> float:
    if frame.empty:
        return 0.0
    return float((frame["weight"] * frame["score"]).sum() / (frame["weight"].sum() + prior))


def direction(score: float, band_pts: float) -> str:
    return "Bullish" if score > band_pts else "Bearish" if score < -band_pts else "Neutral"


def position_plan(price: float, side: str, levels: dict) -> dict:
    """Entry at the last price, stop just beyond the nearest level against the trade, target at the nearest level for it."""
    atr_value = levels["atr"]
    if side == "Long":
        stop = levels["support"][0]["price"] - 0.25 * atr_value
        target = levels["resistance"][0]["price"]
    else:
        stop = levels["resistance"][0]["price"] + 0.25 * atr_value
        target = levels["support"][0]["price"]
    risk, reward = abs(price - stop), abs(target - price)
    return {"entry": round(price, 2), "stop": round(stop, 2), "target": round(target, 2),
            "reward_risk": round(reward / risk, 2) if risk > 0 else None}


def decide_position(cfg: Config, ticker: Ticker, score: float, forecast: pd.Series, levels: dict, gate: dict) -> dict:
    """Sentiment and forecast must agree to create a bias; the bias only becomes a position if the model has
    shown a statistically credible edge out of sample for this exchange."""
    band_pts = cfg.sentiment["neutral_band"] * 100
    hurdle = cfg.exchanges[ticker.exchange].round_trip_cost_bps / 2 / 1e4
    pred_ret = float(forecast["pred_ret"])
    if score > band_pts and pred_ret > 0:
        bias = "Long"
    elif score < -band_pts and pred_ret < 0:
        bias = "Short"
    else:
        return {"side": "Flat", "bias": "Neutral", "reason": "Sentiment and price forecast do not agree"}
    plan = position_plan(float(forecast["base_close"]), bias, levels)
    if not gate["passed"]:
        return {"side": "Flat", "bias": bias, "plan": plan,
                "reason": "Bias only: the model has not beaten a random walk out of sample on this exchange"}
    if abs(pred_ret) < hurdle:
        return {"side": "Flat", "bias": bias, "plan": plan, "reason": "Forecast move is smaller than trading costs"}
    if (plan["reward_risk"] or 0) < 1.2:
        return {"side": "Flat", "bias": bias, "plan": plan, "reason": "Reward/risk to the nearest levels is below 1.2"}
    return {"side": bias, "bias": bias, "plan": plan, "reason": "Sentiment, forecast and out-of-sample gate all agree"}


def _samples(frame: pd.DataFrame, n: int) -> list[dict]:
    out = []
    for _, r in frame.sort_values("weight", ascending=False).head(n).iterrows():
        text = " ".join(str(x) for x in [r["title"], r["text"]] if isinstance(x, str) and x).strip()
        out.append({"source": r["source"], "text": text[:180], "score": round(float(r["score"]) * 100), "url": r["url"],
                    "time": r["ts_utc"].strftime("%b %d %H:%M UTC")})
    return out


def estimated_mentions(symbol: str, by_source: dict[str, int], cov: pd.DataFrame, now: pd.Timestamp, window_days: int) -> tuple[int, bool]:
    """A source that hit its row cap undercounts the busiest stocks. Its observed posting rate is scaled to the full
    window so the ranking still reflects attention. Returns (estimate, whether anything was extrapolated)."""
    total, extrapolated = 0.0, False
    for source, count in by_source.items():
        row = cov[(cov["symbol"] == symbol) & (cov["source"] == source)] if len(cov) else cov
        if len(row) and bool(row["truncated"].iloc[0]):
            days = max((now - row["first_post"].iloc[0]).total_seconds() / 86400, 0.02)
            total += count * max(window_days / days, 1.0)
            extrapolated = True
        else:
            total += count
    if extrapolated and total > 0:  # two significant figures: this is an order of magnitude, not a count
        total = round(total, 1 - int(math.floor(math.log10(total))))
    return int(round(total)), extrapolated


def build_cards(cfg: Config, posts: pd.DataFrame, panel: pd.DataFrame, result: ModelResult,
                daily: dict[str, pd.DataFrame], cov: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    prior = cfg.sentiment["prior_weight"]
    band_pts = cfg.sentiment["neutral_band"] * 100
    cards = []
    for ticker in cfg.tickers:
        rows = panel[panel["symbol"] == ticker.symbol].sort_values("end")
        forecast = result.forecasts[result.forecasts["symbol"] == ticker.symbol]
        if rows.empty or forecast.empty or ticker.symbol not in daily:
            continue
        forecast = forecast.iloc[0]
        latest = rows.iloc[-1]
        ex = cfg.exchanges[ticker.exchange]
        mine = posts[posts["symbol"] == ticker.symbol] if len(posts) else posts
        by_group = {g: mine[mine["group"] == g] if len(mine) else mine for g in GROUPS}
        social = mine[mine["group"] != "news"] if len(mine) else mine
        social_score, news_score = _weighted(social, prior) * 100, _weighted(by_group["news"], prior) * 100
        if len(social) and len(by_group["news"]):
            score = 0.7 * social_score + 0.3 * news_score
        else:
            score = social_score if len(social) else news_score

        signs = {np.sign(_weighted(f, prior)) for f in by_group.values() if len(f)}
        points = int(len(social) >= 50) + int(len(signs) == 1 and len(by_group["news"]) > 0) + int(len(by_group["news"]) >= 3)
        confidence = ["Low", "Low", "Medium", "High"][points]

        price = float(forecast["base_close"])
        levels = find_levels(daily[ticker.symbol], price)
        stances = social["stance"] if len(social) else pd.Series(dtype=float)
        stated = int((stances != 0).sum())
        holders = social[social["verified_holder"] == True] if len(social) else social  # noqa: E712 - nullable column

        pos = decide_position(cfg, ticker, score, forecast, levels, result.gate[ticker.exchange])
        tail = rows.tail(60)
        by_source = social["source"].value_counts().to_dict() if len(social) else {}
        mentions_est, extrapolated = estimated_mentions(ticker.symbol, by_source, cov, now, cfg.window_days)
        cards.append({
            "symbol": ticker.symbol, "code": ticker.code, "name": ticker.name, "local_name": ticker.local_name,
            "exchange": ticker.exchange, "currency": ex.currency,
            "price": round(price, 2), "as_of": forecast["base_end"].isoformat(),
            "mentions": int(len(social)), "mentions_est": mentions_est, "mentions_extrapolated": extrapolated,
            "mentions_by_source": by_source,
            "news_count": int(len(by_group["news"])),
            "score": round(score, 1), "social_score": round(social_score, 1), "news_score": round(news_score, 1),
            "direction": direction(score, band_pts), "confidence": confidence,
            "attention_z": round(max(float(latest["attn_forum"]), float(latest["attn_chat"])), 2),
            "momentum": round(float(latest["ewm_social"]) * 100 - social_score, 1),
            "social_vs_news": round(social_score - news_score, 1) if len(social) and len(by_group["news"]) else None,
            "crowd": {"stated": stated,
                      "long_pct": round(float((stances > 0).sum()) / stated * 100) if stated else None,
                      "exit_pct": round(float((stances < 0).sum()) / stated * 100) if stated else None},
            "verified_holder_score": round(_weighted(holders, prior) * 100, 1) if len(holders) else None,
            "levels": levels,
            "forecast": {"price": round(float(forecast["pred_price"]), 2), "ret_bps": round(float(forecast["pred_ret"]) * 1e4, 1),
                         "low": round(float(forecast["band_low"]), 2), "high": round(float(forecast["band_high"]), 2),
                         "direction": "Up" if forecast["pred_ret"] > 0 else "Down"},
            "position": pos,
            "series": {"t": [t.isoformat() for t in tail["end"]], "close": [round(float(x), 3) for x in tail["close"]],
                       "sentiment": [round(float(x) * 100, 1) for x in tail["ewm_social"]],
                       "mentions": [int(a + b) for a, b in zip(tail["n_forum"], tail["n_chat"])]},
            "top_posts": _samples(social, 3) if len(social) else [],
            "top_news": _samples(by_group["news"], 3) if len(by_group["news"]) else [],
        })
    return sorted(cards, key=lambda c: (c["exchange"], -c["mentions_est"], -c["score"]))
