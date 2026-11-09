"""Recent company news from Exa, normalised into the same schema as social posts."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import Config, Ticker
from .sources import POST_COLUMNS, alias_pattern

log = logging.getLogger(__name__)
EXA_SEARCH = "https://api.exa.ai/search"


def available_at(published: Any) -> pd.Timestamp | None:
    """Exa often gives a date with no real time. Midnight would look like the article predates that day's trading,
    leaking it into earlier bars, so date-only stamps are treated as available at the end of that UTC day."""
    ts = pd.to_datetime(published, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    if ts == ts.normalize():
        return ts + pd.Timedelta(hours=23, minutes=59)
    return ts


def normalise_exa(results: list[dict[str, Any]], ticker: Ticker) -> pd.DataFrame:
    pattern = alias_pattern(ticker.aliases)
    rows, seen = [], set()
    for r in results:
        url = r.get("url")
        title = r.get("title") or ""
        highlights = " ".join(r.get("highlights") or [])
        body = highlights or (r.get("text") or "")[:1500]
        ts = available_at(r.get("publishedDate"))
        if not url or url in seen or ts is None or not pattern.search(f"{title} {body}"):
            continue
        seen.add(url)
        row = {c: None for c in POST_COLUMNS}
        row.update(post_id=f"exa:{hashlib.sha1(url.encode()).hexdigest()[:12]}:{ticker.symbol}", exchange=ticker.exchange,
                   symbol=ticker.symbol, source="exa", group="news", url=url, ts_utc=ts, title=title, text=body,
                   author=r.get("author"), n_tickers=1)
        rows.append(row)
    return pd.DataFrame(rows, columns=POST_COLUMNS)


def search(api_key: str, query: str, start: pd.Timestamp, end: pd.Timestamp, n: int,
           session: requests.Session | None = None) -> list[dict[str, Any]]:
    body = {
        "query": query, "type": "auto", "category": "news", "numResults": n,
        "startPublishedDate": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "endPublishedDate": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "contents": {"highlights": {"maxCharacters": 900}, "text": {"maxCharacters": 1500}},
    }
    resp = (session or requests).post(EXA_SEARCH, json=body, headers={"x-api-key": api_key}, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Exa {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("results", [])


def fetch_news(cfg: Config, api_key: str | None, cache_dir: Path, start: pd.Timestamp,
               end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, errors = [], []
    for ticker in cfg.tickers:
        for query in ticker.news_queries:
            digest = hashlib.sha1(f"{query}|{start.date()}|{end.date()}".encode()).hexdigest()[:10]
            path = cache_dir / f"exa_{ticker.code}_{digest}.json"
            try:
                if path.exists():
                    results = json.loads(path.read_text(encoding="utf-8"))
                elif api_key:
                    results = search(api_key, query, start, end, cfg.limits["exa_results_per_query"])
                    path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
                else:
                    raise RuntimeError("EXA_API_KEY is not set and there is no cached response")
                frames.append(normalise_exa(results, ticker))
            except (RuntimeError, OSError, requests.RequestException) as exc:
                log.error("Exa failed for %r: %s", query, exc)
                errors.append(f"{query}: {str(exc)[:160]}")
    usable = [f for f in frames if not f.empty]
    news = pd.concat(usable, ignore_index=True) if usable else pd.DataFrame(columns=POST_COLUMNS)
    if not news.empty:
        news = news.drop_duplicates("post_id").sort_values("ts_utc").reset_index(drop=True)
    return news, {"ok": not errors, "rows": len(news), "errors": errors[:5]}
