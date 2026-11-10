"""Self-contained visual HTML report. Charts are inline SVG, so the file needs no network access to view."""
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .config import Config
from .model import ModelResult

TEMPLATES = Path(__file__).parent / "templates"
MODEL_LABELS = {"naive": "Random walk", "ridge_price": "Ridge, price only", "ridge_full": "Ridge, price + sentiment",
                "gbm_full": "Boosting, price + sentiment"}
SOURCE_LABELS = {"lihkg": "LIHKG", "naver": "Naver 종목토론실", "telegram": "Telegram", "exa": "Exa news"}


def _tip(text: str) -> str:
    return html.escape(text, quote=True)


def price_fmt(value: float | None, currency: str = "HKD") -> str:
    if value is None:
        return "–"
    return f"{value:,.0f}" if currency == "KRW" else f"{value:,.2f}"


def safe_url(url: Any) -> str | None:
    """Scraped links are untrusted; only plain web links may become href values."""
    return url if isinstance(url, str) and url.startswith(("http://", "https://")) else None


def pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 1) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def signed(value: float | None, digits: int = 1) -> str:
    """Signed number that never prints as -0.0."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    rounded = round(float(value), digits)
    return f"{rounded + 0.0:+.{digits}f}"


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = max(high - low, 1e-9)
    magnitude = 10 ** math.floor(math.log10(span / count))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if span / (m * magnitude) <= count)
    first = math.ceil(low / step) * step
    return [round(first + i * step, 6) for i in range(int((high - first) / step) + 1)]


def polarity_class(score: float, band: float = 10.0) -> str:
    return "bull" if score > band else "bear" if score < -band else "flat"


def scatter_chart(cards: list[dict]) -> Markup:
    """Attention (x) against sentiment (y) for one exchange. Every dot is named, so colour never carries identity."""
    if not cards:
        return Markup("")
    w, h, left, right, top, bottom = 520, 270, 46, 24, 14, 40
    counts = [max(c["mentions_est"], 1) for c in cards]
    x_low, x_high = math.log10(min(counts) * 0.7), math.log10(max(counts) * 1.5)
    y_span = max(25.0, max(abs(c["score"]) for c in cards) * 1.3)
    px = lambda v: left + (math.log10(v) - x_low) / (x_high - x_low) * (w - left - right)
    py = lambda v: top + (y_span - v) / (2 * y_span) * (h - top - bottom)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Social mentions against sentiment score">']
    for v in nice_ticks(-y_span, y_span, 4):
        cls = "axis" if v == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{left}" x2="{w - right}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{signed(v, 0)}</text>')
    candidates = sorted({m * 10 ** k for k in range(0, 7) for m in (1, 2, 5)})
    ticks = [v for v in candidates if 10 ** x_low <= v <= 10 ** x_high]
    for v in ticks[:: max(1, len(ticks) // 5)]:
        parts.append(f'<text class="tick" x="{px(v):.1f}" y="{h - bottom + 18}" text-anchor="middle">{v:,}</text>')
    parts.append(f'<text class="tick" x="{(left + w - right) / 2:.0f}" y="{h - 4}" text-anchor="middle">Social mentions, last 10 days (log scale)</text>')

    dots = [(px(n), py(c["score"])) for c, n in zip(cards, counts)]
    placed: list[tuple[float, float, float, float]] = [(x - 8, y - 8, x + 8, y + 8) for x, y in dots]
    for c, (cx, cy) in sorted(zip(cards, dots), key=lambda t: t[1][1]):
        tip = _tip(f'{c["name"]}: {c["mentions_est"]:,} mentions, sentiment {c["score"]:+.1f} ({c["direction"]})')
        parts.append(f'<g class="dot {polarity_class(c["score"])}" data-tip="{tip}" tabindex="0">'
                     f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="14"/><circle class="mark" cx="{cx:.1f}" cy="{cy:.1f}" r="6"/></g>')
        label = c["name"].split()[0]
        width = 7.2 * len(label)
        options = [(cx + 11, cy + 4), (cx - 11 - width, cy + 4), (cx - width / 2, cy - 12), (cx - width / 2, cy + 20),
                   (cx + 11, cy + 18), (cx + 11, cy - 10), (cx - 11 - width, cy + 18), (cx - 11 - width, cy - 10)]
        lx, ly = options[0]
        for option_x, option_y in options:
            box = (option_x, option_y - 11, option_x + width, option_y + 2)
            clear = all(box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3] for o in placed)
            if clear and option_x >= left and option_x + width <= w:
                lx, ly = option_x, option_y
                break
        placed.append((lx, ly - 11, lx + width, ly + 2))
        parts.append(f'<text class="dlabel" x="{lx:.1f}" y="{ly:.1f}">{html.escape(label)}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def ticker_chart(card: dict) -> Markup:
    """Price with nearest levels and the n+1 forecast, above a separate sentiment strip (two charts, never two axes)."""
    s = card["series"]
    if len(s["close"]) < 2:
        return Markup("")
    fc, lv = card["forecast"], card["levels"]
    w, h, left, right, top, bottom, strip = 380, 196, 8, 62, 10, 18, 44
    support, resistance = lv["support"][0]["price"], lv["resistance"][0]["price"]
    values = s["close"] + [fc["price"], fc["low"], fc["high"], support, resistance]
    low, high = min(values), max(values)
    pad = (high - low) * 0.06 or 1
    low, high = low - pad, high + pad
    n = len(s["close"])
    plot_w, price_h = w - left - right, h - top - bottom - strip - 10
    px = lambda i: left + i / n * plot_w
    py = lambda v: top + (high - v) / (high - low) * price_h
    points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(s["close"]))
    aria = f'{card["name"]} price with support, resistance and forecast'
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_tip(aria)}">']
    for level, name, cls in ((support, "S", "sup"), (resistance, "R", "res")):
        y = py(level)
        parts.append(f'<line class="level {cls}" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{w - right + 6}" y="{y + 4:.1f}">{name} {price_fmt(level, card["currency"])}</text>')
    parts.append(f'<polyline class="price" points="{points}"/>')
    last_x, last_y, fx = px(n - 1), py(s["close"][-1]), px(n)
    parts.append(f'<line class="band" x1="{fx:.1f}" x2="{fx:.1f}" y1="{py(fc["low"]):.1f}" y2="{py(fc["high"]):.1f}"/>')
    parts.append(f'<line class="forecast" x1="{last_x:.1f}" x2="{fx:.1f}" y1="{last_y:.1f}" y2="{py(fc["price"]):.1f}"/>')
    tip = _tip(f'n+1 forecast {price_fmt(fc["price"], card["currency"])} (80% band {price_fmt(fc["low"], card["currency"])} to {price_fmt(fc["high"], card["currency"])})')
    parts.append(f'<g data-tip="{tip}" tabindex="0"><circle class="hit" cx="{fx:.1f}" cy="{py(fc["price"]):.1f}" r="12"/>'
                 f'<circle class="fc-dot" cx="{fx:.1f}" cy="{py(fc["price"]):.1f}" r="4.5"/></g>')
    base = h - bottom - strip / 2
    span = max(10.0, max(abs(v) for v in s["sentiment"]))
    step = plot_w / n
    parts.append(f'<line class="axis" x1="{left}" x2="{w - right}" y1="{base:.1f}" y2="{base:.1f}"/>')
    parts.append(f'<text class="tick" x="{w - right + 6}" y="{base + 4:.1f}">Sentiment</text>')
    for i, v in enumerate(s["sentiment"]):
        length = abs(v) / span * (strip / 2 - 2)
        y = base - length if v >= 0 else base
        time = pd.Timestamp(s["t"][i]).strftime("%b %d %H:%M UTC")
        tip = _tip(f'{time}: close {price_fmt(s["close"][i], card["currency"])}, sentiment {v:+.1f}, {s["mentions"][i]} mentions')
        parts.append(f'<rect class="sbar {"bull" if v >= 0 else "bear"}" x="{px(i) + 0.5:.1f}" y="{y:.1f}" width="{max(step - 1.2, 1):.1f}" height="{max(length, 0.5):.1f}"/>')
        parts.append(f'<rect class="hover" data-tip="{tip}" x="{px(i):.1f}" y="{top}" width="{step:.1f}" height="{h - top - bottom}"/>')
    first, last = pd.Timestamp(s["t"][0]).strftime("%b %d"), pd.Timestamp(s["t"][-1]).strftime("%b %d")
    parts.append(f'<text class="tick" x="{left}" y="{h - 4}">{first}</text><text class="tick" x="{w - right}" y="{h - 4}" text-anchor="end">{last}</text></svg>')
    return Markup("".join(parts))


def advantage_chart(result: ModelResult, cfg: Config) -> Markup:
    """Cumulative error saved against the random walk; a rising line means the model is beating it."""
    champion = cfg.model["champion"]
    scored = result.oos.dropna(subset=[champion]).sort_values("cutoff")
    if scored.empty:
        return Markup("<p class='muted'>Not enough history for walk-forward predictions yet.</p>")
    names = [("ridge_full", "s1"), ("ridge_price", "s2"), ("gbm_full", "s3")]
    series = {}
    for name, _ in names:
        step = (scored["y"].abs() - (scored["y"] - scored[name]).abs()).groupby(scored["cutoff"]).mean() * 1e4
        series[name] = step.cumsum()
    index = series["ridge_full"].index
    w, h, left, right, top, bottom = 1080, 300, 54, 190, 12, 26
    values = np.concatenate([s.to_numpy() for s in series.values()] + [np.array([0.0])])
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.1, 1.0)
    low, high = low - pad, high + pad
    px = lambda i: left + i / max(len(index) - 1, 1) * (w - left - right)
    py = lambda v: top + (high - v) / (high - low) * (h - top - bottom)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Cumulative error saved against the random walk">']
    for v in nice_ticks(low, high, 5):
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="tick" x="{left - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{signed(v, 0)}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{w - right}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>')
    ends = []
    for name, cls in names:
        data = series[name].to_numpy()
        path = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(data))
        parts.append(f'<polyline class="line {cls}" points="{path}"/>')
        ends.append([py(data[-1]) + 4, cls, f"{MODEL_LABELS[name].split(',')[0]} {'+ sent.' if 'full' in name else 'price'}: {data[-1]:+.0f}"])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for y, cls, label in ends:
        parts.append(f'<circle class="key {cls}" cx="{w - right + 10}" cy="{y - 4:.1f}" r="4"/><text class="dlabel" x="{w - right + 19}" y="{y:.1f}">{html.escape(label)}</text>')
    for i, t in enumerate(index):
        tip = " | ".join(f"{MODEL_LABELS[n].split(',')[0]}{' + sent.' if 'full' in n else ''}: {series[n].iloc[i]:+.1f} bps" for n, _ in names)
        width = (w - left - right) / max(len(index), 1)
        parts.append(f'<rect class="hover" data-tip="{_tip(t.strftime("%b %d %H:%M UTC") + " | " + tip)}" x="{px(i) - width / 2:.1f}" y="{top}" width="{width:.1f}" height="{h - top - bottom}"/>')
    parts.append(f'<text class="tick" x="{left}" y="{h - 6}">{index[0].strftime("%b %d")}</text>'
                 f'<text class="tick" x="{w - right}" y="{h - 6}" text-anchor="end">{index[-1].strftime("%b %d")}</text></svg>')
    return Markup("".join(parts))


def verdict(result: ModelResult, cfg: Config) -> dict:
    """Plain-language conclusion generated from the numbers, so the text can't drift away from them."""
    champion = cfg.model["champion"]
    row = result.scores[(result.scores["scope"] == "ALL") & (result.scores["model"] == champion)]
    if row.empty or not row["n"].iloc[0]:
        return {"headline": "Not enough history to validate the model yet.", "detail": "", "skill": None, "hit": None, "n": 0}
    r = row.iloc[0]
    skill, p, hit = r["skill_mae"], r["dm_p"], r.get("hit_rate")
    if pd.notna(p) and skill > 0 and p < cfg.model["gate_max_p_value"]:
        headline = "The model beat a random walk out of sample, and the difference is statistically credible."
    elif skill > 0:
        headline = "The model was slightly more accurate than a random walk, but the sample is too small to call that real."
    else:
        headline = "The model did not beat a random walk out of sample. Forecasts should be read as indicative only."
    perm = result.permutation
    sentiment_line = ""
    if perm:
        sentiment_line = (f" Compared with the price-only model, sentiment changed average error by {perm['observed_lift_bps']:+.2f} bps "
                          f"(permutation p = {perm['p_value']:.2f}; positive means sentiment helped).")
    detail = (f"Over {int(r['n'])} out-of-sample forecasts, mean error was {r['mae_bps']:.1f} bps, a {skill * 100:+.1f}% change in skill against "
              f"the random walk (Diebold-Mariano p = {num(p, 2)}). Direction was right {pct(hit)} of the time "
              f"(95% CI {pct(r.get('hit_low'))} to {pct(r.get('hit_high'))}).{sentiment_line}")
    return {"headline": headline, "detail": detail, "skill": skill, "hit": hit, "n": int(r["n"])}


def build_context(cfg: Config, run_id: str, now: pd.Timestamp, window_start: pd.Timestamp, cards: list[dict], result: ModelResult,
                  status: dict, cov: pd.DataFrame, scored: pd.DataFrame, warnings: list[str], ledger_state: pd.DataFrame,
                  ledger_summary: dict, backend: str, demo: bool, rebuilt: str | None = None) -> dict:
    sections = []
    for key, ex in cfg.exchanges.items():
        mine = [c for c in cards if c["exchange"] == key]
        sources_here = [s for s in ex.social_sources]
        max_mentions = max([c["mentions"] for c in mine] + [1])
        for c in mine:
            c["chart"] = ticker_chart(c)
            c["bar"] = [{"source": s, "label": SOURCE_LABELS[s], "count": c["mentions_by_source"].get(s, 0),
                         "width": c["mentions_by_source"].get(s, 0) / max_mentions * 100} for s in sources_here]
        sections.append({"key": key, "label": ex.label, "cards": mine, "scatter": scatter_chart(mine),
                         "sources": [{"key": s, "label": SOURCE_LABELS[s]} for s in sources_here], "gate": result.gate[key],
                         "strategy": result.strategy.get(key, {})})
    social = scored[scored["group"] != "news"]
    scores = [{**r, "label": MODEL_LABELS[r["model"]], "champion": r["model"] == cfg.model["champion"]}
              for r in result.scores.replace({np.nan: None}).to_dict("records")]
    max_coef = float(result.coefficients["coef_bps"].abs().max() or 1)
    coefs = [{"feature": r["feature"], "value": r["coef_bps"], "width": abs(r["coef_bps"]) / max_coef * 100,
              "kind": "sentiment" if r["feature"].startswith(("ewm", "attn", "gap")) else "price"}
             for r in result.coefficients.head(10).to_dict("records")]
    source_rows = []
    for source, label in SOURCE_LABELS.items():
        rows = scored[scored["source"] == source]
        if len(rows):
            source_rows.append({"label": label, "posts": len(rows), "tickers": rows["symbol"].nunique(),
                                "first": rows["ts_utc"].min().strftime("%b %d %H:%M"), "last": rows["ts_utc"].max().strftime("%b %d %H:%M"),
                                "actor": cfg.actors.get(source, "Exa /search")})
    truncated = cov[cov["truncated"]].to_dict("records") if len(cov) else []
    return {
        "demo": demo, "rebuilt": rebuilt, "band": result.band, "run_id": run_id, "generated": now.strftime("%Y-%m-%d %H:%M UTC"), "window_start": window_start.strftime("%Y-%m-%d %H:%M UTC"),
        "interval": cfg.interval, "backend": backend, "sections": sections, "n_social": len(social), "n_news": int((scored["group"] == "news").sum()),
        "verdict": verdict(result, cfg), "scores": scores, "advantage": advantage_chart(result, cfg), "permutation": result.permutation, "coefs": coefs,
        "positions": sum(1 for c in cards if c["position"]["side"] != "Flat"), "n_tickers": len(cards), "sources": source_rows, "truncated": truncated,
        "warnings": warnings, "status": status, "ledger_summary": ledger_summary,
        "ledger": ledger_state.tail(30).replace({np.nan: None}).to_dict("records") if len(ledger_state) else [],
        "champion": MODEL_LABELS[cfg.model["champion"]], "gate_p": cfg.model["gate_max_p_value"], "min_train": cfg.model["min_train_rows"],
    }


def render(context: dict) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters.update(price=price_fmt, pct=pct, num=num, polarity=polarity_class, safe_url=safe_url, signed=signed)
    return env.get_template("report.html").render(**context)


def tickers_table(cards: list[dict]) -> pd.DataFrame:
    """Flat per-ticker output for spreadsheets."""
    return pd.DataFrame([{
        "exchange": c["exchange"], "ticker": c["symbol"], "asset_name": c["name"], "local_name": c["local_name"], "current_price": c["price"],
        "currency": c["currency"], "as_of_utc": c["as_of"], "social_mentions": c["mentions"], "social_mentions_10d_est": c["mentions_est"], "mentions_extrapolated": c["mentions_extrapolated"], "news_articles": c["news_count"], "sentiment_score": c["score"],
        "direction": c["direction"], "confidence": c["confidence"], "social_score": c["social_score"], "news_score": c["news_score"],
        "attention_z": c["attention_z"], "position": c["position"]["side"], "bias": c["position"]["bias"], "position_reason": c["position"]["reason"],
        "support_1": c["levels"]["support"][0]["price"], "support_2": c["levels"]["support"][1]["price"],
        "resistance_1": c["levels"]["resistance"][0]["price"], "resistance_2": c["levels"]["resistance"][1]["price"],
        "predicted_price_n1": c["forecast"]["price"], "predicted_return_bps": c["forecast"]["ret_bps"],
        "band_low": c["forecast"]["low"], "band_high": c["forecast"]["high"],
    } for c in cards])
