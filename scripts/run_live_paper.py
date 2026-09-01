"""run_live_paper.py — Live paper trading entry point.

Starts the 24/7 continuous trading loop:
  - Equity Options module: evaluated during market hours (9:30 - 16:00 ET).
  - Crypto Spot module: evaluated 24/7.
  - Daily midnight UTC: cointegration rechecks and nightly trade reflections.

Prerequisites:
  - `.env` file populated with ALPACA_PAPER=true and paper API keys.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.broker import create_broker
from src.persistence.db import create_database
from src.pipeline import StatArbPipeline, load_config

# Configure structured console and file logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/system.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("live_runner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Paper Trading Runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--interval", type=int, default=None, help="Override poll interval in seconds")
    parser.add_argument("--single-cycle", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--force-open", action="store_true", help="Force run equity module even if market closed")
    args = parser.parse_args()

    # 1. Load environment variables
    load_dotenv()

    # 2. Hard Startup Assertion: Paper Trading Only
    paper_env = os.environ.get("ALPACA_PAPER", "").strip().lower()
    if paper_env != "true":
        logger.critical(
            "STARTUP REFUSED: ALPACA_PAPER is not set to 'true'. "
            "This system is strictly paper-trading only. Check your .env file."
        )
        sys.exit(1)

    # 3. Load config and initialize pipeline
    config = load_config(args.config)
    poll_interval = args.interval or int(config.get("poll_interval_seconds", 900))

    logger.info("Initializing StatArb Pipeline (Paper Trading Mode)...")
    try:
        broker = create_broker()
        db = create_database(config.get("db_path", "data/trades.db"))
        pipeline = StatArbPipeline(broker=broker, db=db, config=config)
    except Exception as exc:
        logger.critical("Failed to initialize pipeline: %s", exc, exc_info=True)
        sys.exit(1)

    account = broker.get_account()
    logger.info(
        "Alpaca Paper Account Connected | Equity: $%.2f | Buying Power: $%.2f | Cash: $%.2f",
        account.equity,
        account.buying_power,
        account.cash,
    )

    # 4. Handle Graceful Shutdown
    shutdown_requested = False

    def handle_signal(sig, frame):
        nonlocal shutdown_requested
        logger.info("Shutdown signal received. Exiting gracefully after cycle...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("=== Stat-Arb Live Paper Trading Loop Started (Interval: %ds) ===", poll_interval)

    # 5. Main Polling Loop
    cycle_num = 1
    while not shutdown_requested:
        now_utc = datetime.now(timezone.utc)
        logger.info("--- Starting Cycle #%d [%s UTC] ---", cycle_num, now_utc.strftime("%Y-%m-%d %H:%M:%S"))

        try:
            results = pipeline.run_cycle(
                now=now_utc,
                force_market_open=args.force_open,
            )

            # Print concise console summary
            signals_count = len(results["signals"])
            approved_count = sum(1 for d in results["decisions"] if d.approved)
            entries_count = len(results["executed_entries"])
            exits_count = len(results["executed_exits"])

            logger.info(
                "Cycle #%d Complete: %d pairs evaluated | %d approved | %d entries | %d exits",
                cycle_num,
                signals_count,
                approved_count,
                entries_count,
                exits_count,
            )

            if results["skipped_reasons"]:
                logger.info("Notes: %s", ", ".join(results["skipped_reasons"]))

        except Exception as exc:
            logger.error("Unexpected error in cycle #%d: %s", cycle_num, exc, exc_info=True)

        if args.single_cycle:
            logger.info("Single cycle completed. Exiting.")
            break

        cycle_num += 1
        time.sleep(poll_interval)

    logger.info("Stat-Arb Runner shut down cleanly.")


if __name__ == "__main__":
    main()
