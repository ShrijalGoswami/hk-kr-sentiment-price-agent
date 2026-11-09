"""Command line: python -m sentiment_agent {run,check,estimate}"""
from __future__ import annotations

import argparse
import json
import logging

from . import pipeline, sources
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="sentiment_agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="scrape, score, model and write the report")
    run.add_argument("--demo", action="store_true", help="synthetic posts and news, real prices, no tokens needed")
    run.add_argument("--from-cache", action="store_true", help="re-use today's saved scrape instead of calling Apify and Exa")
    run.add_argument("--sentiment", choices=["auto", "transformer", "lexicon"], help="override the sentiment backend")
    sub.add_parser("check", help="score earlier forecasts against bars that have since closed")
    sub.add_parser("estimate", help="print the worst-case Apify cost of a full run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()

    if args.command == "estimate":
        costs = sources.worst_case_usd(cfg)
        print(json.dumps({"worst_case_usd_by_exchange": costs, "total": round(sum(costs.values()), 2)}, indent=1))
    elif args.command == "check":
        print(json.dumps(pipeline.check(cfg), indent=1))
    else:
        path = pipeline.run(cfg, pipeline.RunOptions(demo=args.demo, from_cache=args.from_cache, backend=args.sentiment))
        print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
