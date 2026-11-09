"""Support and resistance from clustered swing highs and lows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(daily: pd.DataFrame, period: int = 14) -> float:
    prev = daily["close"].shift(1)
    true_range = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev).abs(), (daily["low"] - prev).abs()], axis=1).max(axis=1)
    return float(true_range.rolling(period, min_periods=3).mean().iloc[-1])


def swings(daily: pd.DataFrame, span: int = 3) -> list[tuple[float, int]]:
    """(price, bar index) of every local extreme with `span` bars on each side."""
    highs, lows = daily["high"].to_numpy(), daily["low"].to_numpy()
    out = []
    for i in range(span, len(daily) - span):
        if highs[i] == highs[i - span:i + span + 1].max():
            out.append((float(highs[i]), i))
        if lows[i] == lows[i - span:i + span + 1].min():
            out.append((float(lows[i]), i))
    return out


def cluster(points: list[tuple[float, int]], tolerance: float, n_bars: int) -> list[dict]:
    """Merges nearby swings into levels. Strength is touch count plus up to one extra point for recency."""
    clusters: list[list[tuple[float, int]]] = []
    for point in sorted(points):
        if clusters and point[0] - np.mean([p for p, _ in clusters[-1]]) <= tolerance:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    return [{
        "price": float(np.mean([p for p, _ in c])),
        "touches": len(c),
        "strength": len(c) + max(i for _, i in c) / max(n_bars, 1),
    } for c in clusters]


def find_levels(daily: pd.DataFrame, price: float, per_side: int = 2) -> dict:
    """Nearest levels on each side, keeping the stronger of crowded ones. Falls back to ATR steps when price is
    in open space, such as at new highs where no swing exists above, and labels those levels as projected."""
    current_atr = atr(daily)
    levels = cluster(swings(daily), 0.5 * current_atr, len(daily))
    min_gap = 0.25 * current_atr

    def pick(candidates: list[dict], sign: int) -> list[dict]:
        ordered = sorted(candidates, key=lambda lv: sign * (lv["price"] - price))
        chosen = [{"price": round(lv["price"], 4), "touches": lv["touches"], "method": "swing"} for lv in ordered[:per_side]]
        while len(chosen) < per_side:
            base = chosen[-1]["price"] if chosen else price
            chosen.append({"price": round(base + sign * current_atr, 4), "touches": 0, "method": "atr"})
        return chosen

    return {
        "atr": round(current_atr, 4),
        "support": pick([lv for lv in levels if lv["price"] < price - min_gap], -1),
        "resistance": pick([lv for lv in levels if lv["price"] > price + min_gap], 1),
    }
