"""Price bars with true open/close times.

Yahoo labels bars by nominal start, which misleads around the session structure: the HK "11:30" bar stops at the
12:00 lunch break and its "12:30" bar really starts at 13:00. Aligning posts against nominal times would leak
information, so each bar gets its real start and end here.

Yahoo's hourly KRX series also stops near 15:00, about 1% away from the official close on recent days. The last bar
of each finished session therefore takes the official daily close, so it genuinely ends at the session close.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from .config import Exchange

log = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)


def _sessions(day: pd.Timestamp, ex: Exchange) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    base = day.normalize()
    out = []
    for open_, close in ex.sessions:
        oh, om = map(int, open_.split(":"))
        ch, cm = map(int, close.split(":"))
        out.append((base + pd.Timedelta(hours=oh, minutes=om), base + pd.Timedelta(hours=ch, minutes=cm)))
    return out


def bar_window(label: pd.Timestamp, ex: Exchange, interval: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Real (start, end) in exchange-local time for a Yahoo bar label."""
    sessions = _sessions(label, ex)
    auction = pd.Timedelta(minutes=ex.closing_auction_minutes)
    if interval == "1d":
        return sessions[0][0], sessions[-1][1] + auction
    start, session_close = label, sessions[-1][1]
    for open_, close in sessions:
        if label < close:
            start, session_close = max(label, open_), close
            break
    end = min(label + HOUR, session_close)
    if label.strftime("%H:%M") >= ex.last_bar_label:
        end = sessions[-1][1] + auction
    return start, end


def schedule_labels(day: pd.Timestamp, ex: Exchange, interval: str) -> list[pd.Timestamp]:
    if interval == "1d":
        return [day.normalize()]
    label = _sessions(day, ex)[0][0]
    h, m = map(int, ex.last_bar_label.split(":"))
    last = day.normalize() + pd.Timedelta(hours=h, minutes=m)
    out = []
    while label <= last:
        out.append(label)
        label += HOUR
    return out


def next_bar(after_utc: pd.Timestamp, ex: Exchange, interval: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) in UTC of the first scheduled bar starting after a moment. Weekends are skipped; holidays are not
    known here, so the ledger checks forecasts against whichever real bar comes next."""
    local = after_utc.tz_convert(ex.tz)
    for offset in range(0, 12):
        day = local.normalize() + pd.Timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        for label in schedule_labels(day, ex, interval):
            start, end = bar_window(label, ex, interval)
            if start > local:
                return start.tz_convert("UTC"), end.tz_convert("UTC")
    raise RuntimeError(f"no upcoming bar found for {ex.key}")


def annotate(raw: pd.DataFrame, ex: Exchange, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    """Lower-cases OHLCV, adds real UTC start/end and next_start, and drops any bar not finished at `now`."""
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"]).copy()
    labels = df.index.tz_convert(ex.tz) if df.index.tz is not None else df.index.tz_localize(ex.tz)
    windows = [bar_window(lbl, ex, interval) for lbl in labels]
    df["start"] = [s.tz_convert("UTC") for s, _ in windows]
    df["end"] = [e.tz_convert("UTC") for _, e in windows]
    df["final_bar"] = [lbl.strftime("%H:%M") >= ex.last_bar_label for lbl in labels] if interval != "1d" else True
    df = df[df["end"] <= now].reset_index(drop=True)
    if df.empty:
        return df
    df["next_start"] = df["start"].shift(-1)
    df.loc[df.index[-1], "next_start"] = next_bar(df["end"].iloc[-1], ex, interval)[0]
    return df


def _history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, prepost=False)


def apply_official_close(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Sets each finished session's last hourly bar to that day's official close."""
    if bars.empty or daily.empty:
        return bars
    closes = dict(zip(daily["end"], daily["close"]))
    out = bars.copy()
    official = out["end"].map(closes).where(out["final_bar"])
    out["close"] = official.fillna(out["close"])
    out["high"] = out[["high", "close"]].max(axis=1)
    out["low"] = out[["low", "close"]].min(axis=1)
    return out


def fetch(symbol: str, ex: Exchange, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    if interval == "1d":
        return annotate(_history(symbol, "1y", "1d"), ex, "1d", now)
    return apply_official_close(annotate(_history(symbol, "60d", interval), ex, interval, now), fetch_daily(symbol, ex, now))


def fetch_daily(symbol: str, ex: Exchange, now: pd.Timestamp) -> pd.DataFrame:
    return annotate(_history(symbol, "1y", "1d"), ex, "1d", now)
