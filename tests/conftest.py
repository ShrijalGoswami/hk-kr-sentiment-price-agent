from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentiment_agent import prices
from sentiment_agent.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_raw(ex, days: int = 40, seed: int = 1, end_day: str = "2026-11-06") -> pd.DataFrame:
    """Random-walk OHLCV in Yahoo's shape, using the exchange's real bar labels."""
    rng = np.random.default_rng(seed)
    business_days = pd.bdate_range(end=end_day, periods=days, tz=ex.tz)
    labels = [lbl for d in business_days for lbl in prices.schedule_labels(d, ex, "60m")]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, len(labels))))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002, "Low": np.minimum(open_, close) * 0.998,
                         "Close": close, "Volume": 1000}, index=pd.DatetimeIndex(labels))


@pytest.fixture
def now():
    return pd.Timestamp("2026-11-08 12:00", tz="UTC")


@pytest.fixture
def market(cfg, now):
    """(bars, index_bars) of synthetic prices for the whole universe."""
    bars, index_bars = {}, {}
    for i, ticker in enumerate(cfg.tickers):
        ex = cfg.exchanges[ticker.exchange]
        bars[ticker.symbol] = prices.annotate(make_raw(ex, seed=i + 10), ex, "60m", now)
    for j, ex in enumerate(cfg.exchanges.values()):
        index_bars[ex.key] = prices.annotate(make_raw(ex, seed=j + 100), ex, "60m", now)
    return bars, index_bars
