#!/usr/bin/env python3
"""Monitor aggregate success rate from parallel LIBERO-Para workers."""

import argparse
import json
import time
from pathlib import Path


def load_progress(workers_dir: Path):
    records = []
    for path in sorted(workers_dir.glob("rank_*/progress.json")):
        try:
            with path.open("r") as handle:
                records.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            # Each worker uses atomic replacement, but tolerate transient I/O.
            continue
    return records


def main():
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
                sr = 100.0 * successes / completed if completed else 0.0
                percent = 100.0 * completed / args.total
                elapsed = max(time.monotonic() - start_time, 1e-6)
                measured = max(completed - start_completed, 0)
                tasks_per_minute = measured / elapsed * 60.0
                remaining = max(args.total - completed, 0)
                if tasks_per_minute > 0:
                    eta_hours = remaining / tasks_per_minute / 60.0
                    eta_text = f"{eta_hours:.2f}h"
                else:
                    eta_text = "--"

                print(
                    f"[global] {completed}/{args.total} ({percent:.2f}%) | "
                    f"success={successes}/{completed} | SR={sr:.2f}% | "
                    f"workers={len(records)} | {tasks_per_minute:.2f} tasks/min | "
                    f"ETA={eta_text}",
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
