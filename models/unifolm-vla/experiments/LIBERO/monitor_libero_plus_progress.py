#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from tqdm import tqdm


def _count_results(result_dir: Path) -> tuple[int, int]:
    successes = 0
    episodes = 0
    for path in sorted(glob.glob(str(result_dir / "rank_*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episodes += 1
                successes += int(bool(record.get("success")))
    return successes, episodes


def _target_episodes(classification_path: Path, task_suites: list[str], num_trials_per_task: int, limit_tasks: int) -> int:
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)
    total_tasks = sum(len(classification[suite]) for suite in task_suites)
    if limit_tasks > 0:
        total_tasks = min(total_tasks, limit_tasks)
    return total_tasks * num_trials_per_task


def _rate(successes: int, total: int) -> float:
    return 100.0 * successes / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a clean aggregate progress bar for UnifoLM-VLA LIBERO-plus eval.")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--classification-path", required=True, type=Path)
    parser.add_argument("--task-suites", nargs="+", required=True)
    parser.add_argument("--num-trials-per-task", type=int, required=True)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    total = _target_episodes(args.classification_path, args.task_suites, args.num_trials_per_task, args.limit_tasks)
    with tqdm(total=total, desc="Episodes", dynamic_ncols=True) as pbar:
        while True:
            successes, episodes = _count_results(args.result_dir)
            pbar.n = min(episodes, total)
            pbar.set_postfix(success=f"{successes}/{episodes}", rate=f"{_rate(successes, episodes):.1f}%")
            pbar.refresh()
            if episodes >= total:
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
