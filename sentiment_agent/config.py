"""Typed access to config/universe.yaml and the two API tokens."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "universe.yaml"


@dataclass(frozen=True)
class Ticker:
    symbol: str
    code: str
    name: str
    local_name: str
    search_keywords: tuple[str, ...]
    aliases: tuple[str, ...]
    news_queries: tuple[str, ...]
    exchange: str


@dataclass(frozen=True)
class Exchange:
    key: str
    label: str
    tz: str
    index: str
    currency: str
    sessions: tuple[tuple[str, str], ...]
    closing_auction_minutes: int
    last_bar_label: str
    round_trip_cost_bps: float
    social_sources: tuple[str, ...]
    telegram_channels: tuple[str, ...]
    tickers: tuple[Ticker, ...]


@dataclass(frozen=True)
class Config:
    window_days: int
    interval: str
    post_latency_minutes: int
    actors: dict[str, str]
    limits: dict[str, Any]
    sentiment: dict[str, Any]
    model: dict[str, Any]
    exchanges: dict[str, Exchange]

    @property
    def tickers(self) -> list[Ticker]:
        return [t for ex in self.exchanges.values() for t in ex.tickers]


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    exchanges: dict[str, Exchange] = {}
    for key, ex in raw["exchanges"].items():
        tickers = tuple(
            Ticker(
                symbol=t["symbol"],
                code=t["code"],
                name=t["name"],
                local_name=t["local_name"],
                search_keywords=tuple(t["search_keywords"]),
                aliases=tuple(t["aliases"]),
                news_queries=tuple(t["news_queries"]),
                exchange=key,
            )
            for t in ex["tickers"]
        )
        exchanges[key] = Exchange(
            key=key,
            label=ex["label"],
            tz=ex["tz"],
            index=ex["index"],
            currency=ex["currency"],
            sessions=tuple((a, b) for a, b in ex["sessions"]),
            closing_auction_minutes=ex["closing_auction_minutes"],
            last_bar_label=ex["last_bar_label"],
            round_trip_cost_bps=float(ex["round_trip_cost_bps"]),
            social_sources=tuple(ex["social_sources"]),
            telegram_channels=tuple(ex["telegram_channels"]),
            tickers=tickers,
        )
    return Config(
        window_days=raw["window_days"],
        interval=raw["interval"],
        post_latency_minutes=raw["post_latency_minutes"],
        actors=raw["actors"],
        limits=raw["limits"],
        sentiment=raw["sentiment"],
        model=raw["model"],
        exchanges=exchanges,
    )


def tokens() -> tuple[str | None, str | None]:
    load_dotenv(ROOT / ".env")
    return (os.getenv("APIFY_TOKEN") or None, os.getenv("EXA_API_KEY") or None)
