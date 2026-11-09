"""Minimal Apify REST client: start an actor, wait for it, page through its dataset."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

API = "https://api.apify.com/v2"
TERMINAL = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
USABLE = {"SUCCEEDED", "TIMED-OUT"}  # a timed-out run still holds the rows scraped so far


class ApifyError(RuntimeError):
    pass


@dataclass
class ActorResult:
    actor_id: str
    run_id: str
    status: str
    items: list[dict[str, Any]]
    usage_usd: float | None


class ApifyClient:
    def __init__(self, token: str, session: requests.Session | None = None, poll_seconds: float = 5.0):
        self.session = session or requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.poll_seconds = poll_seconds

    def _json(self, method: str, url: str, **kw: Any) -> Any:
        for attempt in range(4):
            resp = self.session.request(method, url, timeout=60, **kw)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if not resp.ok:
                raise ApifyError(f"{method} {url} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise ApifyError(f"{method} {url} kept failing after retries")

    def run(self, actor_id: str, run_input: dict[str, Any], max_usd: float | None = None,
            timeout_seconds: int = 1800) -> ActorResult:
        params = {"maxTotalChargeUsd": max_usd} if max_usd else {}
        path = actor_id.replace("/", "~")
        run = self._json("POST", f"{API}/acts/{path}/runs", json=run_input, params=params)["data"]
        deadline = time.monotonic() + timeout_seconds
        while run["status"] not in TERMINAL:
            if time.monotonic() > deadline:
                raise ApifyError(f"{actor_id}: gave up waiting for run {run['id']}")
            time.sleep(self.poll_seconds)
            run = self._json("GET", f"{API}/actor-runs/{run['id']}")["data"]
        if run["status"] not in USABLE:
            raise ApifyError(f"{actor_id}: run {run['id']} ended {run['status']}")
        return ActorResult(actor_id, run["id"], run["status"],
                           self._items(run["defaultDatasetId"]), run.get("usageTotalUsd"))

    def _items(self, dataset_id: str, page: int = 1000) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while True:
            batch = self._json("GET", f"{API}/datasets/{dataset_id}/items",
                               params={"clean": "true", "format": "json", "limit": page, "offset": len(items)})
            items.extend(batch)
            if len(batch) < page:
                return items
