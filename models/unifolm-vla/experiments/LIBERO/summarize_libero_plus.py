#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict


CATEGORY_TO_COLUMN = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
COLUMNS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]


def _iter_records(patterns: list[str]):
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def _rate(successes: int, total: int) -> float:
    return 100.0 * successes / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize UnifoLM-VLA LIBERO-plus JSONL results.")
    parser.add_argument("patterns", nargs="+", help="JSONL files or glob patterns.")
    parser.add_argument("--model", default="UnifoLM-VLA")
    args = parser.parse_args()

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total_successes = 0
    total_episodes = 0
    missing_category = 0

    for record in _iter_records(args.patterns):
        column = record.get("column") or CATEGORY_TO_COLUMN.get(record.get("category"))
        if column is None:
            missing_category += 1
            continue

        if "successes" in record and "episodes" in record:
            successes = int(record["successes"])
            episodes = int(record["episodes"])
        else:
            successes = int(bool(record.get("success")))
            episodes = 1

        stats[column][0] += successes
        stats[column][1] += episodes
        total_successes += successes
        total_episodes += episodes

    print("| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |")
    print("|-------|--------|-------|----------|-------|------------|-------|--------|-------|")
    values = [_rate(*stats[column]) for column in COLUMNS]
    total = _rate(total_successes, total_episodes)
    print("| " + " | ".join([args.model, *[f"{value:.1f}" for value in values], f"{total:.1f}"]) + " |")

    print("\nCounts:")
    for column in COLUMNS:
        successes, episodes = stats[column]
        print(f"{column}: {successes}/{episodes}")
    print(f"Total: {total_successes}/{total_episodes}")
    if missing_category:
        print(f"Warning: skipped {missing_category} records without LIBERO-plus category.")


if __name__ == "__main__":
    main()
