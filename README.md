# Retail Sentiment Price Prediction Agent

A Python pipeline that scrapes retail investor chatter and news for 5 stocks on each of two Asian exchanges, scores it in Korean, Cantonese/Chinese and English, and forecasts the price at the next bar (n+1). Each forecast is validated out of sample against a random walk.

Built for the CrowdWisdomTrading AI agents developer assessment.

## Results from the run of 2026-11-09

The model **did not beat a random walk** on real data. On a 10-day sample that is the expected outcome, so every ticker stayed Flat.

| Out-of-sample result (220 forecasts) | Value |
|---|---|
| Mean error, champion model | 123.4 bps |
| Mean error, random walk | 120.3 bps |
| Skill against the random walk | -2.5% (Diebold-Mariano p = 0.94) |
| Direction correct | 47.1% (95% CI 40.5% to 53.9%) |
| Sentiment added beyond price | No (permutation p = 0.76) |
| Positions taken | 0 of 10 |

The run covered 770 social posts and 230 news articles. The full output is in [`reports/latest/`](reports/latest/): download `report.html` and open it in a browser.

### Live n+1 check (2026-11-10)

The 10 forecasts made on 9 November were scored against the first bar of the next session with `python -m sentiment_agent check`.

| Live result (10 forecasts) | Value |
|---|---|
| Mean price error, model | 2.57% |
| Mean price error, random walk | 2.56% |
| Direction correct | 2 of 9 (Celltrion didn't move) |
| Beat the random walk | 2 of 10 |
| Actual price inside the 80% band | 3 of 10 |

The model forecast moves of about 0.01%, while Korean stocks opened sharply higher (SK hynix +6.6%, Samsung Electronics +4.7%). All ten outcomes come from one market-wide move, so this is effectively a single observation.

The 80% band missed because it was sized mostly from intraday hourly errors, while these targets include the overnight gap. Sizing the band separately for the first bar of each session is the next fix.

The scored forecasts are in [`data/ledger.csv`](data/ledger.csv). `report.html` was generated before this check, so its ledger section still says pending.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # then add APIFY_TOKEN and EXA_API_KEY

python -m sentiment_agent estimate   # worst-case Apify cost before spending anything
python -m sentiment_agent run        # scrape, score, validate, forecast, report
python -m sentiment_agent check      # later: score the forecasts against bars that have since closed
```

Open `reports/latest/report.html`.

| Option | Effect |
|---|---|
| `run --demo` | Tries the whole pipeline with no tokens, using synthetic posts and real prices. Output goes to `reports/demo/` and is labelled synthetic on the page. |
| `run --from-cache` | Re-runs modelling and reporting on today's saved scrape, with no new API spend. |
| `run --sentiment lexicon` | Skips the transformer models (about 1.6 GB of downloads on first use). |

Tickers, actors, row caps, Telegram channels and model settings are all in [`config/universe.yaml`](config/universe.yaml). Setting `interval: 1d` there switches to daily bars, but ten days then gives only about 70 rows, too few for a stable fit. That is why hourly is the default.

## Data sources

The brief requires two different social sources per exchange, plus Exa news.

| Exchange | Tickers | Social source 1 | Social source 2 |
|---|---|---|---|
| HKEX | 0700 Tencent, 9988 Alibaba, 3690 Meituan, 0005 HSBC, 1810 Xiaomi | LIHKG | Telegram |
| KRX | 005930 Samsung Electronics, 000660 SK hynix, 005380 Hyundai Motor, 373220 LG Energy Solution, 068270 Celltrion | Naver 종목토론실 | Telegram |

| Purpose | Service | ID |
|---|---|---|
| LIHKG threads and replies | Apify | `claude_code_reviewer/lihkg-scraper-en` |
| Naver discussion boards | Apify | `hgservices/naver-stock-discussion-scraper` |
| Telegram channels | Apify | `automation-lab/telegram-scraper` |
| News | Exa | `POST /search`, `category: news` |
| Price bars | Yahoo Finance | `yfinance` |

Prices come from Yahoo Finance rather than Apify because it provides hourly bars for both exchanges and the benchmark indices with no extra cost or token.

A search of the Apify store found no actors for Futu/Moomoo community posts, AAStocks/ETNet comment boards, Daum forums or Blind, so those sources were not used.

### Which fields each platform exposes

The brief asks for timestamps, text, engagement and author follower counts. Platforms that don't publish a field leave it null; nothing is imputed.

| Field | LIHKG | Naver | Telegram | Exa |
|---|---|---|---|---|
| Timestamp (UTC) | yes | yes (KST converted) | yes | yes (see below) |
| Text | title and reply | title and body | message | title and highlights |
| Likes | yes | yes | reactions | – |
| Dislikes | yes | yes | – | – |
| Reposts | – | – | – | – |
| Replies | thread rows | – | – | – |
| Views | – | yes | yes | – |
| Author followers | – | – | channel subscribers | – |
| Verified holder flag | – | yes | – | – |

## What you get for each ticker

| Brief item | Field in `tickers.csv` / `tickers.json` |
|---|---|
| Ticker, name, current price | `ticker`, `asset_name`, `current_price` |
| Sentiment and direction | `sentiment_score` (-100 to +100), `direction`, `confidence` |
| Position, if any | `position` (Long, Short or Flat), `bias`, `position_reason`; entry, stop and target in the JSON |
| Support and resistance | `support_1/2`, `resistance_1/2` |
| n+1 price | `predicted_price_n1`, `predicted_return_bps`, 80% band |

Additional signals:

- **Crowd positioning:** the share of posts where people say they are buying or holding versus selling, separate from how they feel.
- **Verified-holder sentiment:** Naver's holder badge separates people who own the stock from those who only talk about it.
- **Social vs news gap:** how far retail mood is from the tone of the news.
- **Attention shock:** posting rate as a z-score against the stock's own recent history.
- **Slang-aware scoring:** a lexicon for terms generic models misread, such as 떡상, 물렸다, 插水 and 入貨.
- **Downvote discounting:** a heavily disliked post counts for less, even if it is popular.

The HTML report ranks each exchange by social mentions, with sortable columns, an attention-versus-sentiment map, a source breakdown, and a card for each ticker.

## How the forecast works

```
Apify actors ─┐
              ├─ normalise ─ de-duplicate ─ score sentiment ─┐
Exa news ─────┘                                              ├─ point-in-time panel ─ walk-forward ─ report + ledger
Yahoo bars ─── true session open/close times ────────────────┘
```

**n+1** is the next bar (next hour by default). The champion model is a ridge regression on price and sentiment features, predicting the log return to the next bar's close. It was named in the config before any results existed, so the headline isn't the best of several attempts. Price-only ridge, gradient boosting and a random walk are reported next to it.

### Keeping future information out

| Risk | Handling |
|---|---|
| A post leaking into an earlier bar | A post only counts for bar n if it was usable before bar n+1 opened, with a 5-minute publication delay. |
| Yahoo's nominal bar times | Each bar gets its real open and close, including the HK lunch break and closing auction. |
| KRX hourly data stopping near 15:00 | The last bar takes the official daily close. |
| Exa dates with no time of day | Treated as available at the end of that day. |
| Unfinished bars | Dropped. |
| Training on unrealised targets | A refit only uses rows whose target had closed, plus a one-bar embargo. |
| Scaling and tuning on test data | Fitted inside each training window. |

### Deciding whether a forecast is credible

- **Benchmark:** every model is compared with a random walk.
- **Diebold-Mariano test:** losses are averaged per timestamp first, because stocks on the same bar aren't independent.
- **Permutation test:** each stock's sentiment history is randomly time-shifted to check whether sentiment adds anything beyond price.
- **Trading test:** next open to close, after each market's costs.
- **Position gate:** a position is only taken when sentiment and forecast agree, the move clears costs, reward/risk is at least 1.2, and the model has beaten the random walk on that exchange. Otherwise the ticker stays Flat and shows any bias.
- **Live ledger:** forecasts go into `data/ledger.csv` and are scored by `check` once the bar closes. The first forecast for a bar can't be overwritten.

## Limitations

- **Ten days is a very small sample.** It gives roughly 400 stock-hours, heavily overlapping across correlated stocks. A failure to beat the random walk is the expected result for a sample this short, and any apparent success needs a much longer test.
- **Row caps let two full runs fit the $5 Apify free tier**, so a reviewer can reproduce the pipeline on the same account. The busiest Naver boards can exceed 75 posts within minutes. Those mention counts are scaled from the observed rate and marked `~`, and earlier bars are treated as unknown attention rather than zero.
- **Naver covers minutes to hours, not ten days.** The only available actor returns newest posts first with no end date, and the observed Free-plan price is about $2.50 per 1,000 posts. On the busiest boards 75 posts span only minutes, so for Korea the ten-day history comes from Telegram and news, while Naver shows the current mood.
- **LIHKG is thin and the actor is unreliable.** On 2026-11-09 the search step reported failed requests on both attempts, and Alibaba and Meituan returned no LIHKG posts. Each exchange still has two sources, but those two tickers rely on Telegram and news.
- **Telegram channels here are mostly commentary and news feeds**, not retail chatrooms, and the list needs reviewing periodically. The report shows how many posts each source contributed.
- **Cantonese is the weakest language for the models.** The Chinese model is trained mainly on Mandarin, and the lexicon compensates only partly.
- **Exchange holidays aren't in the calendar**, so the expected time of the next bar can be wrong around them. Ledger scoring uses whichever real bar comes next.

## Tests

```bash
python -m pytest
```

The suite focuses on what would silently invalidate results: future posts cannot change past features, future targets cannot change earlier predictions, a planted effect is detected, pure noise is rejected, and each normaliser matches its actor's documented output.

## Project layout

```
config/universe.yaml     tickers, actors, caps, model settings
sentiment_agent/
  apify.py               Apify REST client with per-run spending cap
  sources.py             one normaliser per actor, de-duplication, coverage tracking
  news.py                Exa search and normalisation
  prices.py              bars with real session times
  sentiment.py           transformer + slang lexicon, crowd stance
  features.py            point-in-time panel
  levels.py              support and resistance
  model.py               walk-forward validation and statistics
  signals.py             per-ticker record and position rules
  ledger.py              live forecast tracking
  report.py              HTML report
  demo.py                synthetic data for token-free runs
  pipeline.py            orchestration
tests/                   test suite
```

Tokens are read from `.env`, which is git-ignored, and are never written to the saved scrape files.

This is research output for an assessment, not investment advice.
