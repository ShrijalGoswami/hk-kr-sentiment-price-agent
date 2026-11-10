"""The tests concentrate on what would silently invalidate results: lookahead, leakage and mis-parsed fields."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sentiment_agent import features, ledger, levels, model, news, pipeline, prices, report, sentiment, sources
from sentiment_agent.apify import ApifyClient, ApifyError
from sentiment_agent.sources import POST_COLUMNS, alias_pattern

UTC = "UTC"


def ts(text, tz="Asia/Hong_Kong"):
    return pd.Timestamp(text, tz=tz)


# --- session calendar ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("label,start,end", [
    ("2026-11-05 09:30", "09:30", "10:30"),
    ("2026-11-05 11:30", "11:30", "12:00"),   # stops at lunch
    ("2026-11-05 12:30", "13:00", "13:30"),   # label sits inside lunch; data starts at 13:00
    ("2026-11-05 15:30", "15:30", "16:10"),   # includes the closing auction
])
def test_hk_bar_windows(cfg, label, start, end):
    s, e = prices.bar_window(ts(label), cfg.exchanges["HKEX"], "60m")
    assert s == ts(f"2026-11-05 {start}") and e == ts(f"2026-11-05 {end}")


def test_krx_last_bar_runs_to_close(cfg):
    _, e = prices.bar_window(ts("2026-11-05 14:00", "Asia/Seoul"), cfg.exchanges["KRX"], "60m")
    assert e == ts("2026-11-05 15:30", "Asia/Seoul")


def test_unfinished_bar_is_dropped(cfg):
    ex = cfg.exchanges["HKEX"]
    raw = pd.DataFrame({"Open": [1.0, 1.0], "High": [1.0, 1.0], "Low": [1.0, 1.0], "Close": [1.0, 1.0], "Volume": [1, 1]},
                       index=pd.DatetimeIndex([ts("2026-11-05 09:30"), ts("2026-11-05 10:30")]))
    bars = prices.annotate(raw, ex, "60m", ts("2026-11-05 11:00").tz_convert(UTC))
    assert len(bars) == 1


def test_next_bar_skips_weekend(cfg):
    start, _ = prices.next_bar(ts("2026-11-06 16:30").tz_convert(UTC), cfg.exchanges["HKEX"], "60m")
    assert start == ts("2026-11-09 09:30")


def test_official_close_replaces_last_bar_only(cfg, now):
    ex = cfg.exchanges["KRX"]
    bars = prices.annotate(make_krx_day(), ex, "60m", now)
    daily = pd.DataFrame({"end": [bars["end"].iloc[-1]], "close": [188700.0]})
    patched = prices.apply_official_close(bars, daily)
    assert patched["close"].iloc[-1] == 188700.0
    assert patched["close"].iloc[0] == bars["close"].iloc[0]
    assert patched["low"].iloc[-1] <= 188700.0 <= patched["high"].iloc[-1]


def make_krx_day():
    labels = pd.DatetimeIndex([ts(f"2026-11-06 {h:02d}:00", "Asia/Seoul") for h in range(9, 15)])
    return pd.DataFrame({"Open": 190000.0, "High": 191000.0, "Low": 189000.0, "Close": 190400.0, "Volume": 1}, index=labels)


# --- normalisers against each actor's documented output --------------------------------------------------------

def test_alias_boundaries():
    digits = alias_pattern(("00700", "HSBC", "騰訊"))
    assert digits.search("買 00700 好")
    assert not digits.search("order 1007001")
    assert digits.search("hsbc results")
    assert digits.search("今日騰訊升")


def test_naver_filters_and_converts_to_utc(cfg):
    tickers = cfg.exchanges["KRX"].tickers
    base = {"type": "post", "stockCode": "005930", "postId": "1", "title": "삼전", "text": "좋아요", "writtenAt": "2026-11-05T09:00:00+09:00",
            "views": 10, "likes": 2, "dislikes": 1, "postType": "normal", "isCleanbotPassed": True, "isHolderVerified": True}
    items = [base, {**base, "postId": "2", "postType": "itemNewsPrice"}, {**base, "postId": "3", "isCleanbotPassed": False},
             {"type": "notice", "code": "no_results"}]
    frame, dropped = sources.normalise_naver(items, tickers)
    assert len(frame) == 1
    assert frame["ts_utc"].iloc[0] == pd.Timestamp("2026-11-05 00:00", tz=UTC)
    assert bool(frame["verified_holder"].iloc[0]) and frame["author_followers"].isna().all()
    assert dropped == {"notice": 1, "bot_or_price_feed": 1, "cleanbot": 1, "no_timestamp": 0}


def test_lihkg_requires_stock_in_title_and_prefers_own_text(cfg):
    tickers = cfg.exchanges["HKEX"].tickers
    row = {"thread_id": 7, "created_at": "2026-11-05T02:00:00Z", "like_count": 3, "dislike_count": 1, "reply_count": 9, "source_query": "騰訊"}
    frame, thread_map = sources.normalise_lihkg_threads([{**row, "title": "騰訊今日點睇"}, {**row, "thread_id": 8, "title": "食咩好"}], tickers)
    assert len(frame) == 1 and thread_map == {"7": "0700.HK"}
    reply = {"reply_id": 1, "thread_id": 7, "created_at": "2026-11-05T03:00:00Z", "own_text": "繼續揸", "content": "quoted 繼續揸", "like_count": 0, "dislike_count": 0}
    replies = sources.normalise_lihkg_replies([reply, {**reply, "reply_id": 2, "own_text": ""}], thread_map, tickers)
    assert replies["text"].tolist() == ["繼續揸"]


def test_telegram_maps_tickers_and_followers(cfg):
    tickers = cfg.exchanges["HKEX"].tickers
    items = [{"rowType": "profile", "channelUsername": "chan", "subscribers": 5000},
             {"rowType": "message", "channelUsername": "chan", "messageId": 1, "date": "2026-11-05T01:00:00+00:00", "views": 100, "totalReactions": 4,
              "text": "騰訊大升\n阿里插水\n今日天氣好"},
             {"rowType": "message", "channelUsername": "chan", "messageId": 2, "date": "2026-11-05T01:00:00+00:00", "text": "no stock here"}]
    frame = sources.normalise_telegram(items, tickers)
    assert sorted(frame["symbol"]) == ["0700.HK", "9988.HK"]
    assert (frame["n_tickers"] == 2).all() and (frame["author_followers"] == 5000).all()
    assert frame.set_index("symbol").loc["0700.HK", "text"] == "騰訊大升"  # the other stock's line isn't scored for it


def test_clean_posts_removes_duplicates_and_out_of_window():
    def row(pid, text, when):
        return {**{c: None for c in POST_COLUMNS}, "post_id": pid, "symbol": "X", "source": "naver", "group": "forum", "title": "", "text": text, "ts_utc": pd.Timestamp(when, tz=UTC), "n_tickers": 1}
    frame = pd.DataFrame([row("1", "같은 글입니다", "2026-11-05"), row("2", "같은  글입니다", "2026-11-06"), row("3", "old old old", "2026-10-01"), row("4", "hi", "2026-11-05")])
    cleaned, removed = sources.clean_posts(frame, pd.Timestamp("2026-11-01", tz=UTC), pd.Timestamp("2026-11-08", tz=UTC))
    assert cleaned["post_id"].tolist() == ["1"]
    assert removed == {"outside_window": 1, "too_short": 1, "duplicates": 1}


def test_row_cap_scales_mentions_to_the_full_window(cfg):
    from sentiment_agent import signals
    now = pd.Timestamp("2026-11-08 12:00", tz=UTC)
    cov = pd.DataFrame([{"symbol": "S", "source": "naver", "truncated": True, "first_post": now - pd.Timedelta(days=2)},
                        {"symbol": "S", "source": "telegram", "truncated": False, "first_post": now - pd.Timedelta(days=9)}])
    assert signals.estimated_mentions("S", {"naver": 250, "telegram": 30}, cov, now, 10) == (1300, True)
    assert signals.estimated_mentions("S", {"telegram": 30}, cov, now, 10) == (30, False)


def test_truncated_source_is_flagged(cfg):
    start = pd.Timestamp("2026-10-29", tz=UTC)
    stamps = pd.date_range("2026-11-07", periods=300, freq="min", tz=UTC)
    frame = pd.DataFrame({"symbol": "005930.KS", "source": "naver", "ts_utc": stamps})
    cov = sources.coverage(frame, start)
    assert cov["truncated"].iloc[0] and cov["coverage_start"].iloc[0] == stamps[0]


def test_short_naver_history_counts_as_truncated_even_below_the_row_cap():
    # A spending limit stopped one real run at 97 posts for a stock, well under its 150 cap.
    start = pd.Timestamp("2026-10-29", tz=UTC)
    frame = pd.DataFrame({"symbol": "005380.KS", "source": "naver", "ts_utc": pd.date_range("2026-11-09", periods=97, freq="min", tz=UTC)})
    assert bool(sources.coverage(frame, start)["truncated"].iloc[0])


def test_telegram_with_late_first_post_is_not_truncated():
    start = pd.Timestamp("2026-10-29", tz=UTC)
    frame = pd.DataFrame({"symbol": "X", "source": "telegram", "ts_utc": pd.date_range("2026-11-09", periods=5, freq="h", tz=UTC)})
    assert not bool(sources.coverage(frame, start)["truncated"].iloc[0])


# --- news -------------------------------------------------------------------------------------------------------

def test_date_only_news_is_not_backdated():
    assert news.available_at("2026-11-05T00:00:00.000Z") == pd.Timestamp("2026-11-05 23:59", tz=UTC)
    assert news.available_at("2026-11-05T07:31:00.000Z") == pd.Timestamp("2026-11-05 07:31", tz=UTC)
    assert news.available_at(None) is None


def test_exa_requires_relevance_and_dedupes(cfg):
    ticker = cfg.exchanges["HKEX"].tickers[0]
    good = {"title": "Tencent beats estimates", "url": "https://a.test/1", "publishedDate": "2026-11-05T07:31:00.000Z", "highlights": ["profit"]}
    frame = news.normalise_exa([good, good, {**good, "url": "https://a.test/2", "title": "Unrelated", "highlights": ["x"]}], ticker)
    assert len(frame) == 1 and frame["group"].iloc[0] == "news"


# --- sentiment --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,sign", [
    ("삼전 떡상 간다 풀매수", 1), ("하이닉스 물렸다 손절각 망했네", -1), ("騰訊勝預期 繼續入貨", 1), ("阿里插水 蝕到入肉", -1),
    ("Tencent beats estimates", 1), ("Samsung shares slump", -1),
])
def test_lexicon_polarity(text, sign):
    assert np.sign(sentiment.lexicon_score(text)[0]) == sign


def test_lexicon_negation_and_longest_match():
    assert sentiment.lexicon_score("唔會升")[0] < 0
    assert sentiment.lexicon_score("大跌")[1] == 1


def test_language_and_stance():
    assert sentiment.detect_language("삼성전자 좋다") == "ko" and sentiment.detect_language("騰訊") == "zh" and sentiment.detect_language("Tencent") == "en"
    assert sentiment.stance("어제 풀매수 함") == 1 and sentiment.stance("어제 손절했다") == -1 and sentiment.stance("뉴스 봤어") == 0


# --- features: the lookahead tests ----------------------------------------------------------------------------

def post_frame(symbol, group, source, stamps, scores):
    rows = [{**{c: None for c in POST_COLUMNS}, "post_id": str(i), "symbol": symbol, "group": group, "source": source, "ts_utc": t, "score": s, "weight": 1.0, "n_tickers": 1}
            for i, (t, s) in enumerate(zip(stamps, scores))]
    return pd.DataFrame(rows)


def empty_cov():
    return pd.DataFrame(columns=["symbol", "source", "truncated", "coverage_start"])


def test_future_posts_cannot_change_past_features(cfg, market, now):
    bars, index_bars = market
    window = now - pd.Timedelta(days=10)
    rng = np.random.default_rng(0)
    stamps = sorted(window + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 14000, 400))
    base = post_frame("0700.HK", "forum", "lihkg", stamps, rng.uniform(-1, 1, 400))
    p1 = features.build_panel(cfg, bars, index_bars, base, empty_cov(), window, now)

    mine = p1[p1["symbol"] == "0700.HK"].reset_index(drop=True)
    split = mine["cutoff"].iloc[len(mine) // 2]
    tampered = base.copy()
    # Only posts usable strictly after the split cutoff are altered.
    tampered.loc[tampered["ts_utc"] + pd.Timedelta(minutes=cfg.post_latency_minutes) > split, "score"] = 1.0
    tampered.loc[tampered["ts_utc"] + pd.Timedelta(minutes=cfg.post_latency_minutes) > split, "weight"] = 50.0
    p2 = features.build_panel(cfg, bars, index_bars, tampered, empty_cov(), window, now)
    a, b = (p[(p["symbol"] == "0700.HK") & (p["cutoff"] <= split)] for p in (p1, p2))
    pd.testing.assert_frame_equal(a[features.SENTIMENT_FEATURES], b[features.SENTIMENT_FEATURES])
    later = p1[(p1["symbol"] == "0700.HK") & (p1["cutoff"] > split)]["ewm_forum"].to_numpy()
    assert not np.allclose(later, p2[(p2["symbol"] == "0700.HK") & (p2["cutoff"] > split)]["ewm_forum"].to_numpy())


def test_overnight_post_reaches_the_bar_before_next_open_and_no_earlier(cfg, market, now):
    bars, index_bars = market
    window = now - pd.Timedelta(days=10)
    post_time = ts("2026-11-05 20:00").tz_convert(UTC)   # after the Nov 5 close, before the Nov 6 open
    panel = features.build_panel(cfg, bars, index_bars, post_frame("0700.HK", "forum", "lihkg", [post_time], [1.0]), empty_cov(), window, now)
    mine = panel[panel["symbol"] == "0700.HK"]
    hit = mine[mine["n_forum"] == 1]
    assert len(hit) == 1
    assert hit["end"].iloc[0] == ts("2026-11-05 16:10").tz_convert(UTC)
    assert hit["cutoff"].iloc[0] == ts("2026-11-06 09:30").tz_convert(UTC)


def test_truncated_history_is_not_read_as_silence(cfg, market, now):
    bars, index_bars = market
    window = now - pd.Timedelta(days=10)
    stamps = [now - pd.Timedelta(days=1, minutes=i) for i in range(40)]
    posts = post_frame("005930.KS", "forum", "naver", stamps, [0.5] * 40)
    cov = pd.DataFrame([{"symbol": "005930.KS", "source": "naver", "truncated": True, "coverage_start": min(stamps)}])
    panel = features.build_panel(cfg, bars, index_bars, posts, cov, window, now)
    before = panel[(panel["symbol"] == "005930.KS") & (panel["cutoff"] < min(stamps))]
    assert len(before) > 0 and (before["attn_forum"] == 0).all()


def test_weights_discount_downvotes_and_multi_ticker_posts():
    frame = pd.DataFrame({"likes": [5, 5, 5], "dislikes": [0, 50, 0], "reposts": None, "replies": None, "views": None, "author_followers": None,
                          "n_tickers": [1, 1, 4], "group": ["forum", "forum", "forum"]})
    w = features.post_weight(frame)
    assert w[1] < w[0] and w[2] == pytest.approx(w[0] / 4)


# --- walk-forward validation ---------------------------------------------------------------------------------

def synthetic_panel(cfg, effect: float, seed: int = 5, n_steps: int = 140, tickers: int = 6) -> pd.DataFrame:
    """Panel where next-bar return depends on sentiment with strength `effect`."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-10-01 00:00", tz=UTC)
    rows = []
    for t in range(tickers):
        price = 100.0
        for i in range(n_steps):
            row = {name: float(rng.normal()) for name in features.ALL_FEATURES}
            ret = effect * 0.01 * row["ewm_social"] + (1 - effect) * float(rng.normal(0, 0.01))
            close_next = price * np.exp(ret)
            bar_end = start + pd.Timedelta(hours=i)
            row.update(symbol=f"T{t}", exchange="HKEX" if t < tickers / 2 else "KRX", start=bar_end - pd.Timedelta(hours=1), end=bar_end,
                       cutoff=bar_end, target_end=bar_end + pd.Timedelta(hours=1), overnight=False, close=price, close_next=close_next, y=ret, oc_next=ret, vol=0.01)
            rows.append(row)
            price = close_next
    return pd.DataFrame(rows).sort_values(["cutoff", "symbol"]).reset_index(drop=True)


def quick(cfg, permutations=30):
    return replace(cfg, model={**cfg.model, "permutations": permutations, "min_train_rows": 200})


def test_walk_forward_never_uses_future_targets(cfg):
    c = quick(cfg)
    panel = synthetic_panel(c, 0.5)
    baseline = model.walk_forward(panel, features.ALL_FEATURES, model.fit_ridge, c)
    cut = panel["cutoff"].sort_values().iloc[int(len(panel) * 0.7)]
    tampered = panel.copy()
    tampered.loc[tampered["cutoff"] >= cut, "y"] = 5.0
    changed = model.walk_forward(tampered, features.ALL_FEATURES, model.fit_ridge, c)
    earlier = panel["cutoff"] <= cut - pd.Timedelta(hours=4)
    pd.testing.assert_series_equal(baseline[earlier].dropna(), changed[earlier].dropna())


def test_validation_detects_a_real_effect(cfg):
    c = quick(cfg)
    result = model.run(synthetic_panel(c, 0.6), c)
    row = result.scores.query("scope == 'ALL' and model == 'ridge_full'").iloc[0]
    assert row["skill_mae"] > 0.1 and row["dm_p"] < 0.01
    assert result.permutation["p_value"] < 0.05
    assert all(g["passed"] for g in result.gate.values())


def test_validation_rejects_pure_noise(cfg):
    c = quick(cfg)
    passed = 0
    for seed in range(4):
        result = model.run(synthetic_panel(c, 0.0, seed=seed), c)
        row = result.scores.query("scope == 'ALL' and model == 'ridge_full'").iloc[0]
        assert row["skill_mae"] < 0.03
        passed += sum(g["passed"] for g in result.gate.values())
    assert passed <= 1  # 8 gate decisions at the 10% level: one chance pass is tolerable, a pattern is not


def test_gate_blocks_position_without_edge(cfg):
    from sentiment_agent import signals
    ticker = cfg.exchanges["HKEX"].tickers[0]
    fc = pd.Series({"pred_ret": 0.01, "base_close": 100.0})
    lv = {"atr": 2.0, "support": [{"price": 96.0}], "resistance": [{"price": 110.0}]}
    blocked = signals.decide_position(cfg, ticker, 40.0, fc, lv, {"passed": False})
    allowed = signals.decide_position(cfg, ticker, 40.0, fc, lv, {"passed": True})
    disagree = signals.decide_position(cfg, ticker, -40.0, fc, lv, {"passed": True})
    assert blocked["side"] == "Flat" and blocked["bias"] == "Long"
    assert allowed["side"] == "Long" and allowed["plan"]["stop"] < 96.0 and allowed["plan"]["target"] == 110.0
    assert disagree["side"] == "Flat" and disagree["bias"] == "Neutral"


def error_frame(seed=3, days=40):
    """Hourly forecast errors where every 7th target opens a new session and carries a gap 8x larger."""
    rng = np.random.default_rng(seed)
    rows, t = [], pd.Timestamp("2026-10-01", tz=UTC)
    for i in range(days * 7):
        overnight = i % 7 == 6
        t += pd.Timedelta(hours=16 if overnight else 1)
        rows.append({"symbol": "X", "exchange": "HKEX", "cutoff": t, "target_end": t + pd.Timedelta(hours=1), "overnight": overnight,
                     "y": float(rng.normal(0, 0.016 if overnight else 0.002)), "ridge_full": 0.0})
    return pd.DataFrame(rows)


def test_new_session_band_is_wider_than_intraday_band():
    frame = error_frame()
    night = model.band_offsets(frame, "ridge_full", "HKEX", True)
    day = model.band_offsets(frame, "ridge_full", "HKEX", False)
    assert night[1] - night[0] > 4 * (day[1] - day[0])


def test_split_band_fixes_new_session_coverage():
    frame = error_frame()
    split, pooled = model.band_coverage(frame, "ridge_full", True), model.band_coverage(frame, "ridge_full", False)
    assert pooled["overnight"] < 0.5 < 0.65 < split["overnight"]
    assert abs(split["intraday"] - 0.8) < 0.1 and split["n_overnight"] > 15


def test_thin_error_history_falls_back_to_returns_of_the_same_kind():
    frame = error_frame()
    frame.loc[frame.index[:-20], "ridge_full"] = np.nan  # only 20 recent predictions: about 3 of them overnight
    night = model.band_offsets(frame, "ridge_full", "HKEX", True)
    day = model.band_offsets(frame, "ridge_full", "HKEX", False)
    assert night[1] - night[0] > 4 * (day[1] - day[0])


def test_band_coverage_only_uses_errors_known_at_forecast_time():
    frame = error_frame()
    base = model.band_coverage(frame, "ridge_full", True)
    late = frame["cutoff"] > frame["cutoff"].iloc[len(frame) // 2]
    first_half = model.band_coverage(frame[~late], "ridge_full", True)
    tampered = frame.copy()
    tampered.loc[late, "y"] *= 50
    changed = model.band_coverage(tampered[~late], "ridge_full", True)
    assert first_half == changed and base["n"] > first_half["n"]


def test_as_of_requires_saved_data(cfg):
    with pytest.raises(SystemExit):
        pipeline.run(cfg, pipeline.RunOptions(as_of=pd.Timestamp("2026-11-09 10:43:28", tz=UTC)))


def test_wilson_interval():
    low, high = model.wilson(50, 100)
    assert low < 0.5 < high and model.wilson(0, 0)[0] != model.wilson(0, 0)[0]


# --- levels and ledger ------------------------------------------------------------------------------------------

def daily_frame(values):
    a = np.array(values, dtype=float)
    return pd.DataFrame({"high": a * 1.01, "low": a * 0.99, "close": a})


def test_levels_sit_on_the_correct_side():
    wave = 100 + 8 * np.sin(np.linspace(0, 12 * np.pi, 200))
    found = levels.find_levels(daily_frame(wave), 100.0)
    assert all(lv["price"] < 100 for lv in found["support"]) and all(lv["price"] > 100 for lv in found["resistance"])
    assert found["resistance"][0]["method"] == "swing" and found["resistance"][0]["touches"] >= 2


def test_levels_project_from_atr_at_new_highs():
    found = levels.find_levels(daily_frame(np.linspace(50, 100, 80)), 101.0)
    assert [lv["method"] for lv in found["resistance"]] == ["atr", "atr"]
    assert found["resistance"][0]["price"] > 101


def forecast_frame(base_end, pred=101.0):
    return pd.DataFrame([{"symbol": "X", "exchange": "HKEX", "base_end": base_end, "base_close": 100.0, "pred_ret": 0.01, "pred_price": pred, "band_low": 99.0, "band_high": 103.0}])


def test_ledger_keeps_first_forecast_and_scores_it(tmp_path):
    path = tmp_path / "ledger.csv"
    base = pd.Timestamp("2026-11-05 08:00", tz=UTC)
    ledger.append(path, forecast_frame(base), "r1", base)
    ledger.append(path, forecast_frame(base, pred=120.0), "r2", base)
    assert len(ledger.load(path)) == 1 and ledger.load(path)["pred_price"].iloc[0] == 101.0
    bars = {"X": pd.DataFrame({"end": [base, base + pd.Timedelta(hours=1)], "close": [100.0, 102.0]})}
    state = ledger.settle(path, bars)
    row = state.iloc[0]
    assert row["status"] == "settled" and row["actual_close"] == 102.0
    assert row["direction_hit"] == 1 and row["beat_naive"] == 1 and row["in_band"] == 1
    assert ledger.summary(state)["settled"] == 1


def test_ledger_waits_for_the_target_bar(tmp_path):
    path = tmp_path / "ledger.csv"
    base = pd.Timestamp("2026-11-05 08:00", tz=UTC)
    ledger.append(path, forecast_frame(base), "r1", base)
    state = ledger.settle(path, {"X": pd.DataFrame({"end": [base], "close": [100.0]})})
    assert state["status"].iloc[0] == "pending"


# --- Apify client, end to end, report safety --------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.ok, self.text = payload, status, status < 400, json.dumps(payload)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, statuses, pages):
        self.headers, self.statuses, self.pages, self.calls = {}, list(statuses), list(pages), []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if method == "POST":
            return FakeResponse({"data": {"id": "run1", "status": "RUNNING"}})
        if "actor-runs" in url:
            return FakeResponse({"data": {"id": "run1", "status": self.statuses.pop(0), "defaultDatasetId": "ds", "usageTotalUsd": 0.12}})
        return FakeResponse(self.pages.pop(0))


def test_apify_runs_polls_and_pages():
    session = FakeSession(["RUNNING", "SUCCEEDED"], [[{"a": i} for i in range(1000)], [{"a": 1}]])
    result = ApifyClient("tok", session, poll_seconds=0).run("user/actor", {"x": 1}, max_usd=1.5)
    assert len(result.items) == 1001 and result.usage_usd == 0.12
    method, url, kw = session.calls[0]
    assert method == "POST" and url.endswith("/acts/user~actor/runs") and kw["params"] == {"maxTotalChargeUsd": 1.5}
    assert session.headers["Authorization"] == "Bearer tok" and "tok" not in url


def test_apify_failed_run_raises():
    with pytest.raises(ApifyError):
        ApifyClient("tok", FakeSession(["FAILED"], []), poll_seconds=0).run("user/actor", {})


def test_one_failing_source_does_not_stop_the_exchange(cfg, tmp_path):
    ex = cfg.exchanges["KRX"]

    class Flaky:
        def run(self, actor, run_input, max_usd=None):
            if "naver" in actor:
                raise ApifyError("boom")
            from sentiment_agent.apify import ActorResult
            return ActorResult(actor, "r", "SUCCEEDED", [{"rowType": "message", "channelUsername": "c", "messageId": 1, "date": "2026-11-05T01:00:00+00:00", "text": "삼성전자 떡상 가즈아"}], 0.0)

    store = sources.RawStore(tmp_path, live=True)
    posts, status = sources.scrape_exchange(cfg, ex, store, Flaky(), pd.Timestamp("2026-11-01", tz=UTC), pd.Timestamp("2026-11-08", tz=UTC))
    assert status["naver"]["ok"] is False and status["telegram"]["rows"] == 1 and len(posts) == 1


def test_cached_scrape_is_reused_without_spending(cfg, tmp_path):
    calls = []

    class Counting:
        def run(self, actor, run_input, max_usd=None):
            from sentiment_agent.apify import ActorResult
            calls.append(actor)
            return ActorResult(actor, "r", "SUCCEEDED", [], 0.0)

    store = sources.RawStore(tmp_path, live=True)
    runner = lambda a, i, m: Counting().run(a, i, m)
    store.fetch("n", "a/b", {"k": 1}, runner, 1.0)
    store.fetch("n", "a/b", {"k": 1}, runner, 1.0)
    assert len(calls) == 1


def test_two_full_runs_fit_apify_free_tier(cfg):
    assert 2 * sum(sources.worst_case_usd(cfg).values()) <= 5.0


def test_demo_end_to_end_is_labelled_and_escapes_scraped_html(cfg, market, now, tmp_path, monkeypatch):
    bars, index_bars = market
    daily = {}
    for symbol, frame in bars.items():
        grouped = frame.groupby(frame["end"].dt.date).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), end=("end", "last"))
        daily[symbol] = grouped.reset_index(drop=True)
    monkeypatch.setattr(pipeline, "_load_prices", lambda c, n: (bars, index_bars, daily))
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)

    original = sources.normalise_naver

    def inject(items, tickers):
        frame, dropped = original(items, tickers)
        frame.loc[frame.index[0], "title"] = "<script>alert(1)</script> 떡상"
        return frame, dropped

    monkeypatch.setattr(sources, "normalise_naver", inject)
    fast = replace(cfg, model={**cfg.model, "permutations": 2})
    path = pipeline.run(fast, pipeline.RunOptions(demo=True, backend="lexicon"))
    page = path.read_text(encoding="utf-8")
    assert "DEMO RUN" in page and "<script>alert(1)" not in page
    table = pd.read_csv(path.parent / "tickers.csv", encoding="utf-8-sig")
    assert len(table) == 10 and set(table["exchange"]) == {"HKEX", "KRX"}
    needed = {"ticker", "asset_name", "current_price", "sentiment_score", "direction", "position", "support_1", "resistance_1", "predicted_price_n1"}
    assert needed <= set(table.columns) and table[list(needed)].notna().all().all()
    assert not (tmp_path / "data" / "ledger.csv").exists()  # synthetic runs must never touch the real forecast ledger


def test_unsafe_links_are_dropped():
    assert report.safe_url("javascript:alert(1)") is None and report.safe_url("https://a.test") == "https://a.test" and report.safe_url(None) is None
