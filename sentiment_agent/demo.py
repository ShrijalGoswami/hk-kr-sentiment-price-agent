"""Synthetic stand-ins for the Apify actors and Exa, for trying the pipeline without API tokens.

Rows follow each service's documented output format, so the demo exercises the real normalisers and alignment.
Everything downstream of a demo run is labelled synthetic in the report. By default sentiment is random and
carries no information about returns; `signal` > 0 plants a known effect, which the tests use to prove the
validation can tell the two cases apart.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .apify import ActorResult
from .config import Config, Ticker

POSITIVE = {"ko": ["{a} 오늘 떡상 간다 풀매수", "{a} 호재 나왔다 상한가 가즈아", "{a} 저평가 구간 줍줍"],
            "zh": ["{a} 業績勝預期 繼續入貨", "{a} 爆升 大行唱好", "{a} 今次升穿阻力 加倉"]}
NEGATIVE = {"ko": ["{a} 물렸다 손절해야 하나", "{a} 악재 떡락 곡소리", "{a} 설거지 당했다 망했네"],
            "zh": ["{a} 插水 蝕到入肉", "{a} 跌穿支持 走貨算", "{a} 大行唱淡 利淡"]}
NEUTRAL = {"ko": ["{a} 내일 공시 언제 나오나요", "{a} 오늘 거래량 어떤가요"], "zh": ["{a} 聽日除淨", "{a} 今日成交幾多"]}
NEWS_POS = ["{n} shares surge as analysts upgrade outlook", "{n} beats estimates and announces buyback"]
NEWS_NEG = ["{n} shares slump after downgrade", "{n} misses estimates, faces regulatory probe"]


class SyntheticWorld:
    def __init__(self, cfg: Config, bars: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp,
                 seed: int = 3, signal: float = 0.0) -> None:
        self.cfg, self.bars, self.start, self.end = cfg, bars, start, end
        self.rng = np.random.default_rng(seed)
        self.signal = signal
        self.counter = 0

    def _sentiment_at(self, ticker: Ticker, ts: pd.Timestamp) -> float:
        """Latent mood in [-1, 1]. With `signal` it leans toward the next bar's return, otherwise it is pure noise."""
        noise = float(self.rng.normal(0, 0.7))
        if self.signal and ticker.symbol in self.bars:
            frame = self.bars[ticker.symbol]
            later = frame[frame["start"] > ts]
            if len(later) >= 1:
                bar = later.iloc[0]
                previous = frame[frame["start"] <= ts]
                if len(previous):
                    ret = float(np.log(bar["close"] / previous["close"].iloc[-1]))
                    vol = max(float(frame["close"].pipe(np.log).diff().std()), 1e-4)
                    noise = self.signal * ret / vol + (1 - min(self.signal, 0.9)) * noise
        return float(np.clip(noise, -1, 1))

    def _timestamps(self, per_ticker: int) -> list[pd.Timestamp]:
        seconds = (self.end - self.start).total_seconds()
        return sorted(self.start + pd.Timedelta(seconds=float(s)) for s in self.rng.uniform(0, seconds, per_ticker))

    def _text(self, ticker: Ticker, mood: float, lang: str) -> str:
        book = POSITIVE if mood > 0.2 else NEGATIVE if mood < -0.2 else NEUTRAL
        self.counter += 1
        return f"{self.rng.choice(book[lang]).format(a=ticker.local_name)} #{self.counter}"

    def naver(self, run_input: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for code in run_input["stocks"]:
            ticker = next(t for t in self.cfg.tickers if t.code == code)
            for ts in self._timestamps(int(self.rng.integers(120, 220))):
                mood = self._sentiment_at(ticker, ts)
                rows.append({
                    "type": "post", "stockCode": code, "stockName": ticker.local_name, "market": "KOSPI",
                    "postId": f"{code}{self.counter}", "url": "https://example.invalid/naver",
                    "title": self._text(ticker, mood, "ko"), "text": "",
                    "writtenAt": ts.tz_convert("Asia/Seoul").isoformat(), "author": "synthetic", "authorProfileId": "synthetic",
                    "isHolderVerified": bool(self.rng.random() < 0.2), "views": int(self.rng.integers(5, 800)),
                    "likes": int(self.rng.integers(0, 12)), "dislikes": int(self.rng.integers(0, 4)), "imageCount": 0,
                    "imageUrl": None, "postType": "normal", "isCleanbotPassed": True, "scrapedAt": self.end.isoformat(),
                })
        return rows

    def lihkg(self, run_input: dict[str, Any]) -> list[dict[str, Any]]:
        if run_input["mode"] == "search":
            rows = []
            for keyword in run_input["keywords"]:
                ticker = next(t for t in self.cfg.tickers if keyword in t.search_keywords)
                for ts in self._timestamps(int(self.rng.integers(6, 16))):
                    self.counter += 1
                    rows.append({"thread_id": 900000 + self.counter, "title": self._text(ticker, self._sentiment_at(ticker, ts), "zh"),
                                 "content": None, "own_text": None, "author": "synthetic", "category": "stock",
                                 "reply_count": int(self.rng.integers(1, 60)), "like_count": int(self.rng.integers(0, 40)),
                                 "dislike_count": int(self.rng.integers(0, 15)), "created_at": ts.isoformat(), "last_reply_at": ts.isoformat(),
                                 "thread_url": "https://example.invalid/lihkg", "source_mode": "search", "source_query": keyword,
                                 "scraped_at": self.end.isoformat()})
            return rows
        rows = []
        tickers = list(self.cfg.exchanges["HKEX"].tickers)
        for thread_id in run_input["threadIds"]:
            ticker = tickers[int(thread_id) % len(tickers)]
            for floor, ts in enumerate(self._timestamps(run_input["maxReplies"])):
                self.counter += 1
                rows.append({"reply_id": self.counter, "thread_id": thread_id, "floor": floor + 1, "author": "synthetic",
                             "content": "", "own_text": self._text(ticker, self._sentiment_at(ticker, ts), "zh"),
                             "like_count": int(self.rng.integers(0, 10)), "dislike_count": int(self.rng.integers(0, 4)),
                             "created_at": ts.isoformat(), "scraped_at": self.end.isoformat()})
        return rows

    def telegram(self, run_input: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        channels = run_input["channels"]
        is_hk = channels[0] in self.cfg.exchanges["HKEX"].telegram_channels
        tickers, lang = (self.cfg.exchanges["HKEX"].tickers, "zh") if is_hk else (self.cfg.exchanges["KRX"].tickers, "ko")
        for channel in channels:
            rows.append({"rowType": "profile", "channelUsername": channel, "subscribers": int(self.rng.integers(1000, 90000))})
            for ts in self._timestamps(int(self.rng.integers(20, 50))):
                ticker = tickers[int(self.rng.integers(0, len(tickers)))]
                self.counter += 1
                rows.append({"rowType": "message", "messageId": self.counter, "channelUsername": channel,
                             "url": "https://example.invalid/telegram", "date": ts.isoformat(),
                             "text": self._text(ticker, self._sentiment_at(ticker, ts), lang),
                             "views": int(self.rng.integers(100, 20000)), "totalReactions": int(self.rng.integers(0, 30)),
                             "commentCount": None})
        return rows

    def news(self, ticker: Ticker) -> list[dict[str, Any]]:
        rows = []
        for ts in self._timestamps(int(self.rng.integers(3, 9))):
            mood = self._sentiment_at(ticker, ts)
            title = str(self.rng.choice(NEWS_POS if mood >= 0 else NEWS_NEG)).format(n=ticker.name)
            self.counter += 1
            rows.append({"title": title, "url": f"https://example.invalid/news/{self.counter}", "publishedDate": ts.isoformat(),
                         "author": "synthetic", "highlights": [title], "text": title})
        return rows


class SyntheticApify:
    """Stands in for ApifyClient and dispatches on actor id."""

    def __init__(self, world: SyntheticWorld) -> None:
        self.world = world

    def run(self, actor_id: str, run_input: dict[str, Any], max_usd: float | None = None) -> ActorResult:
        if "naver" in actor_id:
            items = self.world.naver(run_input)
        elif "lihkg" in actor_id:
            items = self.world.lihkg(run_input)
        else:
            items = self.world.telegram(run_input)
        return ActorResult(actor_id, "synthetic", "SUCCEEDED", items, 0.0)
