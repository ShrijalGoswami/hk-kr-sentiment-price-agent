"""Scrapes each social source through Apify and maps every actor's output to one post schema.

Not every platform exposes every field the brief asks for; unavailable values stay null
(see the field-availability table in the README) rather than being invented.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .apify import ActorResult, ApifyClient, ApifyError
from .config import Config, Exchange, Ticker

log = logging.getLogger(__name__)

POST_COLUMNS = [
    "post_id", "exchange", "symbol", "source", "group", "url", "ts_utc", "title", "text",
    "author", "author_followers", "likes", "dislikes", "reposts", "replies", "views",
    "verified_holder", "n_tickers",
]
# "group" lets the model pool sources across exchanges: LIHKG and Naver are both retail forums.
GROUP = {"lihkg": "forum", "naver": "forum", "telegram": "chat"}

Runner = Callable[[str, dict[str, Any], float | None], ActorResult]


def alias_pattern(aliases: tuple[str, ...]) -> re.Pattern[str]:
    """Latin/digit aliases need token boundaries ('HSBC', '00700'); CJK and Hangul have no spaces to bound on."""
    parts = []
    for alias in sorted(aliases, key=len, reverse=True):
        escaped = re.escape(alias)
        parts.append(rf"(?<![0-9A-Za-z]){escaped}(?![0-9A-Za-z])" if alias.isascii() else escaped)
    return re.compile("|".join(parts), re.IGNORECASE)


def _parse_utc(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _post(**kw: Any) -> dict[str, Any]:
    row = {c: None for c in POST_COLUMNS}
    row["n_tickers"] = 1
    row.update(kw)
    row["group"] = GROUP[row["source"]]
    return row


def _focus_on_ticker(text: str, pattern: re.Pattern[str]) -> str:
    """In multi-stock wraps, score only the lines about this ticker so one stock's news doesn't tag another."""
    lines = [s.strip() for s in re.split(r"[\n。！？!?]+", text) if s.strip()]
    hits = [s for s in lines if pattern.search(s)]
    return " ".join(hits) if hits else text


# --- normalisers: one per actor, pure functions so they can be tested against documented output ------------

def normalise_naver(items: list[dict[str, Any]], tickers: tuple[Ticker, ...]) -> tuple[pd.DataFrame, dict[str, int]]:
    by_code = {t.code: t for t in tickers}
    rows, dropped = [], {"notice": 0, "bot_or_price_feed": 0, "cleanbot": 0, "no_timestamp": 0}
    for it in items:
        if it.get("type") != "post":
            dropped["notice"] += 1
            continue
        ticker = by_code.get(str(it.get("stockCode")))
        ts = _parse_utc(it.get("writtenAt"))
        if ticker is None or ts is None:
            dropped["no_timestamp"] += 1
            continue
        if it.get("postType") != "normal":
            dropped["bot_or_price_feed"] += 1
            continue
        if it.get("isCleanbotPassed") is False:
            dropped["cleanbot"] += 1
            continue
        rows.append(_post(
            post_id=f"naver:{it.get('postId')}", exchange=ticker.exchange, symbol=ticker.symbol, source="naver",
            url=it.get("url"), ts_utc=ts, title=it.get("title") or "", text=it.get("text") or "",
            author=it.get("authorProfileId") or it.get("author"), likes=it.get("likes"), dislikes=it.get("dislikes"),
            views=it.get("views"), verified_holder=bool(it.get("isHolderVerified")),
        ))
    return pd.DataFrame(rows, columns=POST_COLUMNS), dropped


def normalise_lihkg_threads(items: list[dict[str, Any]], tickers: tuple[Ticker, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Search rows carry the title and counters only. Returns posts plus a thread_id -> symbol map for reply fetching."""
    keyword_to_ticker = {k: t for t in tickers for k in t.search_keywords}
    rows, thread_map = [], {}
    for it in items:
        ticker = keyword_to_ticker.get(str(it.get("source_query")))
        title = it.get("title") or ""
        ts = _parse_utc(it.get("created_at"))
        if ticker is None or ts is None or not alias_pattern(ticker.aliases).search(title):
            continue  # LIHKG search is fuzzy; require the stock to appear in the title
        thread_id = str(it.get("thread_id"))
        thread_map[thread_id] = ticker.symbol
        rows.append(_post(
            post_id=f"lihkg:t{thread_id}:{ticker.symbol}", exchange=ticker.exchange, symbol=ticker.symbol, source="lihkg",
            url=it.get("thread_url"), ts_utc=ts, title=title, text=it.get("own_text") or it.get("content") or "",
            author=it.get("author"), likes=it.get("like_count"), dislikes=it.get("dislike_count"),
            replies=it.get("reply_count"),
        ))
    return pd.DataFrame(rows, columns=POST_COLUMNS), thread_map


def normalise_lihkg_replies(items: list[dict[str, Any]], thread_map: dict[str, str],
                            tickers: tuple[Ticker, ...]) -> pd.DataFrame:
    by_symbol = {t.symbol: t for t in tickers}
    rows = []
    for it in items:
        if it.get("reply_id") is None:
            continue
        symbol = thread_map.get(str(it.get("thread_id")))
        ts = _parse_utc(it.get("created_at"))
        text = (it.get("own_text") or "").strip()  # own_text excludes quoted posts, so quotes aren't double-counted
        if symbol is None or ts is None or not text:
            continue
        rows.append(_post(
            post_id=f"lihkg:r{it['reply_id']}:{symbol}", exchange=by_symbol[symbol].exchange, symbol=symbol, source="lihkg",
            url=f"https://lihkg.com/thread/{it.get('thread_id')}/page/1", ts_utc=ts, text=text, author=it.get("author"),
            likes=it.get("like_count"), dislikes=it.get("dislike_count"),
        ))
    return pd.DataFrame(rows, columns=POST_COLUMNS)


def normalise_telegram(items: list[dict[str, Any]], tickers: tuple[Ticker, ...]) -> pd.DataFrame:
    """Channels aren't per-stock, so tickers are found in the text. A post naming N tickers gets weight 1/N later."""
    followers = {str(it.get("channelUsername") or it.get("username") or "").lower(): it.get("subscribers")
                 for it in items if it.get("rowType") == "profile"}
    patterns = [(t, alias_pattern(t.aliases)) for t in tickers]
    rows = []
    for it in items:
        if it.get("rowType") != "message":
            continue
        text = it.get("text") or ""
        ts = _parse_utc(it.get("date"))
        hits = [(t, p) for t, p in patterns if p.search(text)]
        if ts is None or not hits:
            continue
        channel = str(it.get("channelUsername") or "")
        for ticker, pattern in hits:
            rows.append(_post(
                post_id=f"telegram:{channel}:{it.get('messageId')}:{ticker.symbol}", exchange=ticker.exchange,
                symbol=ticker.symbol, source="telegram", url=it.get("url"), ts_utc=ts,
                text=_focus_on_ticker(text, pattern), author=channel, author_followers=followers.get(channel.lower()),
                likes=it.get("totalReactions"), replies=it.get("commentCount"), views=it.get("views"),
                reposts=None, n_tickers=len(hits),
            ))
    return pd.DataFrame(rows, columns=POST_COLUMNS)


def clean_posts(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int]]:
    """Window filter plus de-duplication of forwards and copy-paste spam within each ticker."""
    before = len(df)
    df = df[(df["ts_utc"] >= start) & (df["ts_utc"] <= end)].copy()
    outside = before - len(df)
    body = (df["title"].fillna("") + " " + df["text"].fillna("")).str.replace(r"\s+", "", regex=True)
    df["_fingerprint"] = body.str.lower()
    df = df[df["_fingerprint"].str.len() >= 4]
    too_short = before - outside - len(df)
    df = df.sort_values("ts_utc").drop_duplicates(["symbol", "_fingerprint"], keep="first")
    duplicates = before - outside - too_short - len(df)
    return df.drop(columns="_fingerprint").reset_index(drop=True), {
        "outside_window": outside, "too_short": too_short, "duplicates": duplicates}


# --- orchestration -------------------------------------------------------------------------------------------

class RawStore:
    """Keeps every actor response on disk so re-runs and model changes never pay for scraping twice."""

    def __init__(self, root: Path, live: bool):
        self.root = root
        self.live = live
        root.mkdir(parents=True, exist_ok=True)

    def fetch(self, name: str, actor: str, run_input: dict[str, Any], runner: Runner | None,
              max_usd: float | None) -> list[dict[str, Any]]:
        digest = hashlib.sha1(json.dumps([actor, run_input], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
        path = self.root / f"{name}_{digest}.json"
        if path.exists():
            log.info("%s: using cached response %s", name, path.name)
            return json.loads(path.read_text(encoding="utf-8"))["items"]
        if not self.live or runner is None:
            raise ApifyError(f"{name}: no cached data and live scraping is off")
        log.info("%s: starting actor %s", name, actor)
        result = runner(actor, run_input, max_usd)
        path.write_text(json.dumps({
            "actor": actor, "run_id": result.run_id, "status": result.status, "usage_usd": result.usage_usd,
            "fetched_at": datetime.now(timezone.utc).isoformat(), "input": run_input, "items": result.items,
        }, ensure_ascii=False), encoding="utf-8")
        log.info("%s: %d rows, status %s, cost $%s", name, len(result.items), result.status, result.usage_usd)
        return result.items


def build_inputs(cfg: Config, ex: Exchange, start: datetime) -> dict[str, dict[str, Any]]:
    lim = cfg.limits
    inputs: dict[str, dict[str, Any]] = {}
    if "lihkg" in ex.social_sources:
        inputs["lihkg_search"] = {
            "mode": "search", "keywords": [k for t in ex.tickers for k in t.search_keywords],
            "order": "desc_create_time", "maxItems": lim["lihkg_search_per_keyword"], "dateFrom": f"{cfg.window_days}d",
        }
    if "naver" in ex.social_sources:
        inputs["naver"] = {
            "stocks": [t.code for t in ex.tickers], "dateFrom": start.strftime("%Y-%m-%d"),
            "maxResults": lim["naver_posts_per_ticker"],
        }
    if "telegram" in ex.social_sources:
        inputs["telegram"] = {
            "channels": list(ex.telegram_channels), "maxMessages": lim["telegram_messages_per_channel"],
            "lastNDays": cfg.window_days, "keywords": sorted({a for t in ex.tickers for a in t.aliases}),
            "includeChannelInfo": True,
        }
    return inputs


def worst_case_usd(cfg: Config) -> dict[str, float]:
    """Upper bound from the actors' published per-event prices; the real bill is normally far below it."""
    lim, out = cfg.limits, {}
    for ex in cfg.exchanges.values():
        cost = 0.0
        if "lihkg" in ex.social_sources:
            n_kw = sum(len(t.search_keywords) for t in ex.tickers)
            cost += 0.005 * 2 + n_kw * lim["lihkg_search_per_keyword"] * 0.002
            n_threads = len(ex.tickers) * lim["lihkg_threads_per_ticker"]
            cost += n_threads * (0.008 + lim["lihkg_replies_per_thread"] * 0.003)
        if "naver" in ex.social_sources:
            cost += len(ex.tickers) * lim["naver_posts_per_ticker"] * 0.0025  # observed Free-plan price
        if "telegram" in ex.social_sources:
            n = len(ex.telegram_channels)
            cost += 0.005 + n * 0.002 + n * lim["telegram_messages_per_channel"] * 0.001
        out[ex.key] = round(cost, 2)
    return out


def scrape_exchange(cfg: Config, ex: Exchange, store: RawStore, client: ApifyClient | None,
                    start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Runs every source for one exchange. One failing source is recorded, not fatal, and shows up in the report."""
    runner: Runner | None = None
    if client is not None:
        runner = lambda actor, run_input, max_usd: client.run(actor, run_input, max_usd)
    max_usd = cfg.limits["max_usd_per_actor_run"]
    inputs = build_inputs(cfg, ex, start.to_pydatetime() - timedelta(days=1))  # actor dates are local; clean_posts trims
    frames: list[pd.DataFrame] = []
    status: dict[str, Any] = {}

    def attempt(name: str, fn: Callable[[], pd.DataFrame]) -> None:
        try:
            frame = fn()
            frames.append(frame)
            status[name] = {"ok": True, "rows": len(frame)}
        except (ApifyError, OSError, ValueError, KeyError) as exc:
            log.error("%s/%s failed: %s", ex.key, name, exc)
            status[name] = {"ok": False, "rows": 0, "error": str(exc)[:300]}

    if "naver" in inputs:
        def naver() -> pd.DataFrame:
            items = store.fetch(f"{ex.key}_naver", cfg.actors["naver"], inputs["naver"], runner, max_usd)
            frame, dropped = normalise_naver(items, ex.tickers)
            status["naver_dropped"] = dropped
            return frame
        attempt("naver", naver)

    if "lihkg_search" in inputs:
        def lihkg() -> pd.DataFrame:
            items = store.fetch(f"{ex.key}_lihkg_search", cfg.actors["lihkg"], inputs["lihkg_search"], runner, max_usd)
            threads, thread_map = normalise_lihkg_threads(items, ex.tickers)
            if threads.empty:
                return threads
            top = (threads.sort_values("replies", ascending=False)
                   .groupby("symbol").head(cfg.limits["lihkg_threads_per_ticker"]))
            ids = sorted(p.split(":")[1][1:] for p in top["post_id"])
            replies_input = {"mode": "thread", "threadIds": ids, "maxReplies": cfg.limits["lihkg_replies_per_thread"]}
            reply_items = store.fetch(f"{ex.key}_lihkg_replies", cfg.actors["lihkg"], replies_input, runner, max_usd)
            return pd.concat([threads, normalise_lihkg_replies(reply_items, thread_map, ex.tickers)], ignore_index=True)
        attempt("lihkg", lihkg)

    if "telegram" in inputs:
        def telegram() -> pd.DataFrame:
            items = store.fetch(f"{ex.key}_telegram", cfg.actors["telegram"], inputs["telegram"], runner, max_usd)
            frame = normalise_telegram(items, ex.tickers)
            status["telegram_channels"] = frame.groupby("author").size().to_dict() if not frame.empty else {}
            return frame
        attempt("telegram", telegram)

    usable = [f for f in frames if not f.empty]
    posts = pd.concat(usable, ignore_index=True) if usable else pd.DataFrame(columns=POST_COLUMNS)
    if posts.empty:
        return posts, status
    posts, removed = clean_posts(posts, start, end)
    status["removed"] = removed
    return posts, status


def coverage(posts: pd.DataFrame, window_start: pd.Timestamp) -> pd.DataFrame:
    """Marks sources that hit their row cap: older bars then have unknown attention, which must not read as zero."""
    if posts.empty:
        return pd.DataFrame(columns=["symbol", "source", "posts", "first_post", "last_post", "truncated", "coverage_start"])
    rows = []
    for (symbol, source), g in posts.groupby(["symbol", "source"]):
        first = g["ts_utc"].min()
        # Naver returns newest first, so history that starts well after the window opens means a cap cut it off.
        # Counting rows isn't enough: cleaning removes some, and a spending limit can stop a stock below its cap.
        truncated = source == "naver" and first > window_start + pd.Timedelta(hours=12)
        rows.append({"symbol": symbol, "source": source, "posts": len(g), "first_post": first,
                     "last_post": g["ts_utc"].max(), "truncated": bool(truncated),
                     "coverage_start": first if truncated else window_start})
    return pd.DataFrame(rows)
