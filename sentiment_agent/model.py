"""Walk-forward n+1 forecasting and the statistics that decide whether a forecast deserves any trust.

Three protections against fooling ourselves on a short sample:
  * every refit trains only on rows whose target had already been realised, plus an embargo;
  * the champion model is named in config before results exist, so the headline isn't the best of several tries;
  * every model is compared with a random walk (price(n+1) = price(n)), with significance tests, not just a score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import ALL_FEATURES, PRICE_FEATURES, SENTIMENT_FEATURES

log = logging.getLogger(__name__)
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
Fitter = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def fit_ridge(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Ridge with alpha picked on the last quarter of the training window only."""
    cut = int(len(X) * 0.75)
    best, best_err = ALPHAS[-1], np.inf
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[:cut], y[:cut])
        err = float(np.abs(model.predict(X[cut:]) - y[cut:]).mean())
        if err < best_err:
            best, best_err = alpha, err
    return make_pipeline(StandardScaler(), Ridge(alpha=best)).fit(X, y).predict


def fit_gbm(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=80, min_samples_leaf=20,
                                         l2_regularization=5.0, random_state=7).fit(X, y).predict


def _winsorise(y: np.ndarray) -> np.ndarray:
    limit = 3 * float(np.std(y))
    return np.clip(y, -limit, limit) if limit > 0 else y


MODELS: dict[str, tuple[list[str], Fitter]] = {
    "ridge_price": (PRICE_FEATURES, fit_ridge),
    "ridge_full": (ALL_FEATURES, fit_ridge),
    "gbm_full": (ALL_FEATURES, fit_gbm),
}


def walk_forward(panel: pd.DataFrame, features: list[str], fitter: Fitter, cfg: Config,
                 interval_hours: float = 1.0) -> pd.Series:
    """Out-of-sample prediction per row, NaN where there wasn't yet enough history to train."""
    m = cfg.model
    data = panel[panel["y"].notna()].sort_values("cutoff")
    embargo = pd.Timedelta(hours=interval_hours * m["embargo_bars"])
    preds = pd.Series(np.nan, index=data.index, dtype=float)
    cut_values = data["cutoff"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    target_end = data["target_end"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    steps = np.unique(cut_values)
    X, y = data[features].to_numpy(float), data["y"].to_numpy(float)
    refit_times = steps[::m["refit_every_steps"]]
    min_rows = min(m["min_train_rows"], int(0.4 * len(data)))  # daily bars give far fewer rows than hourly
    for i, t in enumerate(refit_times):
        until = refit_times[i + 1] if i + 1 < len(refit_times) else None
        train = target_end + embargo.to_timedelta64() <= t
        test = (cut_values >= t) & ((cut_values < until) if until is not None else True)
        if train.sum() < min_rows or not test.any():
            continue
        predict = fitter(X[train], _winsorise(y[train]))
        preds.iloc[np.where(test)[0]] = predict(X[test])
    return preds


def diebold_mariano(frame: pd.DataFrame, pred: str) -> float:
    """One-sided p-value that absolute error beats a random walk.
    Losses are averaged per timestamp first because stocks on the same bar are not independent observations."""
    loss_diff = (frame["y"].abs() - (frame["y"] - frame[pred]).abs()).groupby(frame["cutoff"]).mean().to_numpy()
    T = len(loss_diff)
    if T < 8 or np.allclose(loss_diff, 0):
        return float("nan")
    mean, x = loss_diff.mean(), loss_diff - loss_diff.mean()
    lags = int(T ** (1 / 3))
    lrv = float(x @ x) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * float(x[k:] @ x[:-k]) / T
    if lrv <= 0:
        return float("nan")
    return float(1 - stats.t.cdf(mean / np.sqrt(lrv / T), df=T - 1))


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, denom = hits / n, 1 + z * z / n
    centre, spread = (p + z * z / (2 * n)) / denom, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - spread), float(centre + spread)


def score(frame: pd.DataFrame, pred: str) -> dict:
    """Error is in basis points of price; skill is relative to the random walk (above 0 means better)."""
    f = frame.dropna(subset=[pred, "y"])
    if f.empty:
        return {"n": 0}
    err = f["y"] - f[pred]
    naive_mae, mae = float(f["y"].abs().mean()), float(err.abs().mean())
    out = {
        "n": int(len(f)),
        "mae_bps": mae * 1e4,
        "rmse_bps": float(np.sqrt((err ** 2).mean())) * 1e4,
        "mape_pct": float((f["close_next"] - f["close"] * np.exp(f[pred])).abs().div(f["close_next"]).mean()) * 100,
        "skill_mae": 1 - mae / naive_mae if naive_mae else float("nan"),
        "dm_p": diebold_mariano(f, pred),
    }
    if (f[pred] != 0).any():
        active = f[(f[pred] != 0) & (f["y"] != 0)]
        hits = int((np.sign(active[pred]) == np.sign(active["y"])).sum())
        low, high = wilson(hits, len(active))
        ic = stats.spearmanr(f[pred], f["y"]).statistic if f[pred].nunique() > 1 else float("nan")
        out.update(hit_rate=hits / len(active) if len(active) else float("nan"), hit_low=low, hit_high=high, ic=float(ic))
    return out


def strategy(frame: pd.DataFrame, pred: str, cfg: Config) -> dict:
    """Enter at the next bar's open, exit at its close, only when the forecast clears half the round-trip cost."""
    out: dict[str, dict] = {}
    for key, ex in cfg.exchanges.items():
        f = frame[(frame["exchange"] == key)].dropna(subset=[pred, "oc_next"])
        cost = ex.round_trip_cost_bps / 1e4
        active = f[f[pred].abs() >= cost / 2]
        if active.empty:
            out[key] = {"trades": 0, "gross_bps": 0.0, "net_bps": 0.0, "avg_net_bps": 0.0}
            continue
        gross = np.sign(active[pred]) * active["oc_next"]
        out[key] = {"trades": int(len(active)), "gross_bps": float(gross.sum()) * 1e4,
                    "net_bps": float((gross - cost).sum()) * 1e4, "avg_net_bps": float((gross - cost).mean()) * 1e4}
    return out


def permutation_test(panel: pd.DataFrame, oos_rows: pd.Index, price_pred: pd.Series, cfg: Config,
                     seed: int = 11) -> dict:
    """Does sentiment add anything? Each ticker's sentiment history is shifted by a random offset, which keeps its
    autocorrelation but breaks its link to returns, then the walk-forward is repeated."""
    n_perm = cfg.model["permutations"]
    base = panel.loc[oos_rows]
    price_err = (base["y"] - price_pred.loc[oos_rows]).abs().mean()

    def lift(preds: pd.Series) -> float:
        return float(price_err - (base["y"] - preds.loc[oos_rows]).abs().mean()) * 1e4

    observed = lift(walk_forward(panel, ALL_FEATURES, fit_ridge, cfg))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        shuffled = panel.copy()
        for _, idx in panel.groupby("symbol").groups.items():
            positions = np.asarray(idx)
            offset = rng.integers(1, max(len(positions), 2))
            shuffled.loc[positions, SENTIMENT_FEATURES] = np.roll(
                panel.loc[positions, SENTIMENT_FEATURES].to_numpy(float), offset, axis=0)
        null.append(lift(walk_forward(shuffled, ALL_FEATURES, fit_ridge, cfg)))
    arr = np.asarray(null)
    return {"observed_lift_bps": observed, "null_mean_bps": float(arr.mean()), "null_p95_bps": float(np.percentile(arr, 95)),
            "p_value": float((1 + (arr >= observed).sum()) / (1 + n_perm)), "n_permutations": n_perm}


BAND = (0.1, 0.9)
MIN_BAND_SAMPLE = 15


def band_offsets(history: pd.DataFrame, pred: str, exchange: str, overnight: bool,
                 split: bool = True) -> tuple[float, float] | None:
    """10th and 90th percentile of earlier misses for the same kind of target.

    `history` holds realised rows, with the out-of-sample prediction where one existed. A target that opens a new
    session carries the whole overnight gap and misses far more widely than an hour-to-hour target. One pooled band
    was too narrow exactly there: on the first live check only 3 of 10 next-morning prices fell inside it. When a
    kind has too few model errors, the spread of its realised returns is used instead, never the other kind."""
    errors = history.dropna(subset=[pred])
    miss = errors["y"] - errors[pred]
    same_exchange = errors["exchange"] == exchange
    if split:
        same_kind = errors["overnight"] == overnight
        realised = history.loc[(history["exchange"] == exchange) & (history["overnight"] == overnight), "y"]
        candidates = [miss[same_exchange & same_kind], miss[same_kind], realised]
    else:
        candidates = [miss[same_exchange]]
    for sample in candidates:
        if len(sample) >= MIN_BAND_SAMPLE:
            low, high = sample.quantile(BAND).to_numpy()
            return float(low), float(high)
    return None


def band_coverage(oos: pd.DataFrame, pred: str, split: bool) -> dict:
    """Share of out-of-sample targets that landed inside the 80% band, building every band only from data that
    had already been realised when that forecast was made."""
    history = oos.sort_values("cutoff")
    inside: dict[bool, list[bool]] = {True: [], False: []}
    for _, row in history.dropna(subset=[pred]).iterrows():
        past = history[history["target_end"] <= row["cutoff"]]
        offsets = band_offsets(past, pred, row["exchange"], bool(row["overnight"]), split) if len(past) else None
        if offsets:
            inside[bool(row["overnight"])].append(offsets[0] <= row["y"] - row[pred] <= offsets[1])
    both = inside[True] + inside[False]
    share = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {"n": len(both), "n_overnight": len(inside[True]), "overall": share(both),
            "overnight": share(inside[True]), "intraday": share(inside[False])}


@dataclass
class ModelResult:
    oos: pd.DataFrame
    scores: pd.DataFrame
    strategy: dict
    permutation: dict
    forecasts: pd.DataFrame
    coefficients: pd.DataFrame
    gate: dict[str, dict]
    band: dict


def forecast_next(panel: pd.DataFrame, oos: pd.DataFrame, cfg: Config, fitter: Fitter, features: list[str]) -> pd.DataFrame:
    """Fits on everything realised and predicts the bar after each ticker's latest bar, with an empirical 80% band."""
    train = panel[panel["y"].notna()]
    predict = fitter(train[features].to_numpy(float), _winsorise(train["y"].to_numpy(float)))
    latest = panel.sort_values("end").groupby("symbol").tail(1).copy()
    latest["pred_ret"] = predict(latest[features].to_numpy(float))
    champion = cfg.model["champion"]
    rows = []
    for _, r in latest.iterrows():
        offsets = band_offsets(oos, champion, r["exchange"], bool(r["overnight"]))
        if offsets is None:
            log.warning("%s: too few past errors for an empirical band; using recent volatility", r["symbol"])
        low_q, high_q = offsets or (-1.28 * r["vol"], 1.28 * r["vol"])
        rows.append({"symbol": r["symbol"], "exchange": r["exchange"], "base_end": r["end"], "base_close": float(r["close"]),
                     "pred_ret": float(r["pred_ret"]), "pred_price": float(r["close"] * np.exp(r["pred_ret"])),
                     "band_low": float(r["close"] * np.exp(r["pred_ret"] + low_q)),
                     "band_high": float(r["close"] * np.exp(r["pred_ret"] + high_q))})
    return pd.DataFrame(rows)


def run(panel: pd.DataFrame, cfg: Config, interval_hours: float = 1.0) -> ModelResult:
    champion = cfg.model["champion"]
    known = panel[panel["y"].notna()]
    oos = known[["symbol", "exchange", "start", "end", "cutoff", "target_end", "overnight", "close", "close_next", "y", "oc_next"]].copy()
    oos["naive"] = 0.0
    for name, (features, fitter) in MODELS.items():
        log.info("walk-forward: %s", name)
        oos[name] = walk_forward(panel, features, fitter, cfg, interval_hours)
    scored = oos.dropna(subset=[champion])

    scope_rows = []
    for scope, frame in [("ALL", scored)] + [(k, scored[scored["exchange"] == k]) for k in cfg.exchanges]:
        for name in ["naive", *MODELS]:
            scope_rows.append({"scope": scope, "model": name, **score(frame, name)})
    scores = pd.DataFrame(scope_rows)

    if len(scored) and cfg.model["permutations"] > 0:
        log.info("permutation test: %d runs", cfg.model["permutations"])
        permutation = permutation_test(panel, scored.index, oos["ridge_price"], cfg)
    else:
        permutation = {}

    gate = {}
    for key in cfg.exchanges:
        row = scores[(scores["scope"] == key) & (scores["model"] == champion)]
        skill = float(row["skill_mae"].iloc[0]) if len(row) and "skill_mae" in row and pd.notna(row["skill_mae"].iloc[0]) else float("nan")
        p = float(row["dm_p"].iloc[0]) if len(row) and "dm_p" in row and pd.notna(row["dm_p"].iloc[0]) else float("nan")
        gate[key] = {"skill_mae": skill, "dm_p": p,
                     "passed": bool(skill > 0 and p < cfg.model["gate_max_p_value"])}

    features, fitter = MODELS[champion]
    forecasts = forecast_next(panel, oos, cfg, fitter, features)
    full = known[features].to_numpy(float)
    y = _winsorise(known["y"].to_numpy(float))
    scaler = StandardScaler().fit(full)
    ridge = Ridge(alpha=100.0).fit(scaler.transform(full), y)
    coefficients = pd.DataFrame({"feature": features, "coef_bps": ridge.coef_ * 1e4}).sort_values("coef_bps", key=np.abs, ascending=False)
    band = {"split": band_coverage(oos, champion, split=True), "pooled": band_coverage(oos, champion, split=False)}
    return ModelResult(oos, scores, strategy(scored, champion, cfg), permutation, forecasts, coefficients, gate, band)
