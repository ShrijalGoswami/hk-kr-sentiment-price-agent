"""Append-only record of live n+1 forecasts, scored against the real bar once it has closed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["run_id", "created_utc", "symbol", "exchange", "base_end", "base_close", "pred_ret", "pred_price", "band_low",
           "band_high", "status", "target_end", "actual_close", "error_pct", "direction_hit", "beat_naive", "in_band"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"symbol": str})


def append(path: Path, forecasts: pd.DataFrame, run_id: str, created: pd.Timestamp) -> pd.DataFrame:
    """Adds this run's forecasts. A base bar that was already forecast keeps its first prediction, so re-runs
    can't quietly replace a forecast that later turned out wrong."""
    ledger = load(path)
    new = forecasts.assign(run_id=run_id, created_utc=created.isoformat(), status="pending",
                           base_end=forecasts["base_end"].map(lambda t: t.isoformat()))
    for col in COLUMNS:
        if col not in new:
            new[col] = np.nan
    existing = set(zip(ledger["symbol"], ledger["base_end"].astype(str)))
    new = new[[(s, b) not in existing for s, b in zip(new["symbol"], new["base_end"])]][COLUMNS]
    if new.empty:
        return ledger
    merged = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def settle(path: Path, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Scores every pending forecast whose next real bar has finished."""
    ledger = load(path)
    if ledger.empty:
        return ledger
    for col in ("status", "target_end"):
        ledger[col] = ledger[col].astype(object)
    changed = False
    for i, row in ledger[ledger["status"] == "pending"].iterrows():
        frame = bars.get(row["symbol"])
        if frame is None or frame.empty:
            continue
        later = frame[frame["end"] > pd.Timestamp(row["base_end"])]
        if later.empty:
            continue
        bar = later.iloc[0]
        actual, pred, base = float(bar["close"]), float(row["pred_price"]), float(row["base_close"])
        ledger.loc[i, ["status", "target_end"]] = ["settled", bar["end"].isoformat()]
        ledger.loc[i, "actual_close"] = actual
        ledger.loc[i, "error_pct"] = (pred - actual) / actual * 100
        ledger.loc[i, "direction_hit"] = float(np.sign(pred - base) == np.sign(actual - base)) if actual != base else np.nan
        ledger.loc[i, "beat_naive"] = float(abs(pred - actual) < abs(base - actual))
        ledger.loc[i, "in_band"] = float(row["band_low"] <= actual <= row["band_high"])
        changed = True
    if changed:
        ledger.to_csv(path, index=False)
    return ledger


def summary(ledger: pd.DataFrame) -> dict:
    settled = ledger[ledger["status"] == "settled"] if len(ledger) else ledger
    if not len(settled):
        return {"settled": 0, "pending": int(len(ledger))}
    return {
        "settled": int(len(settled)), "pending": int(len(ledger) - len(settled)),
        "mean_abs_error_pct": float(settled["error_pct"].abs().mean()),
        "direction_hit_rate": float(pd.to_numeric(settled["direction_hit"]).mean()),
        "beat_naive_rate": float(pd.to_numeric(settled["beat_naive"]).mean()),
        "band_coverage": float(pd.to_numeric(settled["in_band"]).mean()),
    }
