#!/usr/bin/env python3
"""Show aggregate progress for parallel LIBERO-Para workers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_progress(workers_dir: Path) -> list[dict]:
    records = []
    for path in sorted(workers_dir.glob("rank_*/progress.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.total < 1:
        parser.error("--total must be >= 1")
    if args.interval <= 0:
        parser.error("--interval must be > 0")

    start_time = time.monotonic()
    start_completed = None
    last_counters = None
    try:
        while True:
            records = load_progress(args.workers_dir)
            completed = sum(int(item.get("completed", 0)) for item in records)
            successes = sum(int(item.get("successes", 0)) for item in records)
            counters = (completed, successes, len(records))
            if start_completed is None:
                start_completed = completed

            if counters != last_counters:
                success_rate = 100.0 * successes / completed if completed else 0.0
                percent = 100.0 * completed / args.total
                elapsed = max(time.monotonic() - start_time, 1e-6)
                measured = max(completed - start_completed, 0)
                tasks_per_minute = measured / elapsed * 60.0
                remaining = max(args.total - completed, 0)
                eta = (
                    f"{remaining / tasks_per_minute / 60.0:.2f}h"
                    if tasks_per_minute > 0
                    else "--"
                )
                print(
                    f"[global] {completed}/{args.total} ({percent:.2f}%) | "
                    f"success={successes}/{completed} | SR={success_rate:.2f}% | "
                    f"workers={len(records)} | {tasks_per_minute:.2f} tasks/min | "
                    f"ETA={eta}",
                    flush=True,
                )
                last_counters = counters

            if completed >= args.total:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
