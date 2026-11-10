"""End-to-end orchestration: scrape -> score -> align -> model -> report -> ledger."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import features, ledger, model, news, prices, report, signals, sources
from .apify import ApifyClient
from .config import ROOT, Config, tokens
from .demo import SyntheticApify, SyntheticWorld
from .sentiment import Scorer

log = logging.getLogger(__name__)
LEDGER_PATH = ROOT / "data" / "ledger.csv"


@dataclass
class RunOptions:
    demo: bool = False
    from_cache: bool = False
    backend: str | None = None
    signal: float = 0.0  # demo only: plant a known effect
    as_of: pd.Timestamp | None = None  # with from_cache: rebuild a past run exactly, from its saved scrape


def _load_prices(cfg: Config, now: pd.Timestamp) -> tuple[dict, dict, dict]:
    bars, index_bars, daily = {}, {}, {}
    for ex in cfg.exchanges.values():
        index_bars[ex.key] = prices.fetch(ex.index, ex, cfg.interval, now)
        for ticker in ex.tickers:
            frame = prices.fetch(ticker.symbol, ex, cfg.interval, now)
            if frame.empty:
                log.error("no bars returned for %s", ticker.symbol)
                continue
            bars[ticker.symbol] = frame
            daily[ticker.symbol] = prices.fetch_daily(ticker.symbol, ex, now)
    return bars, index_bars, daily


def source_warnings(cfg: Config, posts: pd.DataFrame) -> list[str]:
    """Returns warnings for any exchange that ended with fewer than the two social sources the brief requires."""
    warnings = []
    for key, ex in cfg.exchanges.items():
        got = posts[(posts["exchange"] == key) & (posts["group"] != "news")]["source"].nunique() if len(posts) else 0
        if got < 2:
            warnings.append(f"{key}: only {got} of {len(ex.social_sources)} social sources returned usable posts")
        for ticker in ex.tickers:
            n = int((posts["symbol"] == ticker.symbol).sum()) if len(posts) else 0
            if n == 0:
                warnings.append(f"{ticker.symbol}: no posts at all")
    return warnings


def run(cfg: Config, options: RunOptions) -> Path:
    if options.as_of is not None and not options.from_cache:
        raise SystemExit("--as-of rebuilds a past run from its saved scrape, so it needs --from-cache.")
    today = pd.Timestamp.now(tz="UTC").floor("s")
    now = options.as_of if options.as_of is not None else today
    rebuilt = (f"Rebuilt on {today:%Y-%m-%d} from the scrape saved on {now:%Y-%m-%d}. No new posts or news were collected; "
               f"the live ledger below includes forecasts scored after that date.") if options.as_of is not None else None
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    window_start = now - pd.Timedelta(days=cfg.window_days)
    out_dir = ROOT / "reports" / ("demo" if options.demo else "latest")
    raw_dir = ROOT / "data" / "raw" / ("demo" if options.demo else now.strftime("%Y-%m-%d"))
    if options.demo and raw_dir.exists():
        shutil.rmtree(raw_dir)

    apify_token, exa_key = tokens()
    if not options.demo and not options.from_cache and not (apify_token and exa_key):
        raise SystemExit("APIFY_TOKEN and EXA_API_KEY must be set in .env (or use --demo to try it with synthetic data).")

    log.info("loading price bars")
    bars, index_bars, daily = _load_prices(cfg, now)

    client = None
    world = None
    if options.demo:
        world = SyntheticWorld(cfg, bars, window_start, now, signal=options.signal)
        client = SyntheticApify(world)
    elif apify_token and not options.from_cache:
        client = ApifyClient(apify_token)

    store = sources.RawStore(raw_dir, live=client is not None)
    frames, status = [], {}
    for ex in cfg.exchanges.values():
        log.info("collecting social posts: %s", ex.key)
        frame, status[ex.key] = sources.scrape_exchange(cfg, ex, store, client, window_start, now)
        frames.append(frame)
    usable = [f for f in frames if not f.empty]
    social = pd.concat(usable, ignore_index=True) if usable else pd.DataFrame(columns=sources.POST_COLUMNS)
    cov = sources.coverage(social, window_start)

    log.info("collecting news")
    if options.demo:
        rows = [news.normalise_exa(world.news(t), t) for t in cfg.tickers]
        news_frame, status["news"] = pd.concat(rows, ignore_index=True), {"ok": True, "rows": sum(map(len, rows)), "errors": []}
    else:
        news_frame, status["news"] = news.fetch_news(cfg, exa_key if not options.from_cache else None, raw_dir, window_start, now)

    all_posts = pd.concat([f for f in (social, news_frame) if not f.empty], ignore_index=True) if len(social) or len(news_frame) else social
    if all_posts.empty:
        raise SystemExit("No posts or news were collected; see the errors above.")
    all_posts["ts_utc"] = pd.to_datetime(all_posts["ts_utc"], utc=True)

    backend = options.backend or ("lexicon" if options.demo else cfg.sentiment["backend"])
    scorer = Scorer(backend, ROOT / "data" / "cache" / "sentiment.json")
    log.info("scoring %d posts (%s)", len(all_posts), scorer.used)
    scored = scorer.score_frame(all_posts)
    scored["weight"] = features.post_weight(scored)
    warnings = source_warnings(cfg, scored)
    for w in warnings:
        log.warning(w)

    log.info("building point-in-time panel")
    panel = features.build_panel(cfg, bars, index_bars, scored, cov, window_start, now)

    interval_hours = 24.0 if cfg.interval == "1d" else 1.0
    log.info("running walk-forward validation on %d rows", len(panel))
    result = model.run(panel, cfg, interval_hours)
    cards = signals.build_cards(cfg, scored, panel, result, daily, cov, now)

    if options.demo:
        ledger_state, ledger_summary = pd.DataFrame(columns=ledger.COLUMNS), {"settled": 0, "pending": 0}
    else:
        ledger.settle(LEDGER_PATH, bars)
        ledger_state = ledger.append(LEDGER_PATH, result.forecasts, run_id, now)
        ledger_summary = ledger.summary(ledger_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    context = report.build_context(cfg, run_id, now, window_start, cards, result, status, cov, scored, warnings, ledger_state,
                                   ledger_summary, scorer.used, options.demo, rebuilt)
    (out_dir / "report.html").write_text(report.render(context), encoding="utf-8")
    (out_dir / "tickers.json").write_text(json.dumps(cards, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    report.tickers_table(cards).to_csv(out_dir / "tickers.csv", index=False, encoding="utf-8-sig")
    result.scores.round(4).to_csv(out_dir / "model_scores.csv", index=False)
    result.oos.dropna(subset=[cfg.model["champion"]]).round(6).to_csv(out_dir / "walk_forward_predictions.csv", index=False)
    log.info("report written to %s", out_dir / "report.html")
    return out_dir / "report.html"


def check(cfg: Config) -> dict:
    """Scores any forecasts whose target bar has since closed."""
    now = pd.Timestamp.now(tz="UTC").floor("s")
    bars = {t.symbol: prices.fetch(t.symbol, cfg.exchanges[t.exchange], cfg.interval, now) for t in cfg.tickers}
    return ledger.summary(ledger.settle(LEDGER_PATH, bars))
