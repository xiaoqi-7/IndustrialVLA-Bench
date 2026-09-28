#!/usr/bin/env python3
"""Merge sharded UnifoLM-VLA LIBERO-Para results."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def category_key(episode: dict[str, Any]) -> str:
    paraphrase_type = episode.get("paraphrase_type", "unknown")
    categories = episode.get("categories", [])
    subcategories = episode.get("subcategories", [])
    if paraphrase_type == "comp":
        return f"comp_{'+'.join(categories)}_{'+'.join(subcategories)}"
    if categories and subcategories:
        return f"{paraphrase_type}_{categories[0]}_{subcategories[0]}"
    return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.expected_total < 1:
        parser.error("--expected-total must be >= 1")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    return args


def main() -> None:
    args = _parse_args()
    worker_dirs = sorted(
        path for path in args.workers_dir.glob("rank_*") if path.is_dir()
    )
    if len(worker_dirs) != args.num_shards:
        raise SystemExit(
            f"Expected {args.num_shards} worker directories, found {len(worker_dirs)}"
        )

    metas: list[dict[str, Any]] = []
    seen_task_ids: set[int] = set()
    eval_basenames: set[str] = set()
    total_episodes = 0
    for worker_dir in worker_dirs:
        meta_path = worker_dir / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"Worker metadata is missing: {meta_path}")
        metas.append(load_json(meta_path))

        for path in worker_dir.glob("eval*.json"):
            eval_basenames.add(path.name)
            for episode in load_json(path).get("episodes", []):
                task_id = int(episode["task_id"])
                if task_id in seen_task_ids:
                    raise SystemExit(
                        f"Duplicate global task_id across shards: {task_id}"
                    )
                seen_task_ids.add(task_id)
                total_episodes += 1

    if total_episodes != args.expected_total:
        raise SystemExit(
            f"Expected {args.expected_total} episodes, found {total_episodes}; "
            "refusing a partial merge"
        )
    expected_ids = set(range(args.expected_total))
    if seen_task_ids != expected_ids:
        missing = sorted(expected_ids - seen_task_ids)
        unexpected = sorted(seen_task_ids - expected_ids)
        raise SystemExit(
            "Global task IDs are incomplete or invalid; "
            f"first missing={missing[:20]}, first unexpected={unexpected[:20]}"
        )
    if not eval_basenames:
        raise SystemExit("No eval*.json files were produced by the workers")

    per_eval: dict[str, dict[str, Any]] = {}
    per_category: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "successes": 0}
    )
    total_successes = 0

    for basename in sorted(eval_basenames):
        episodes: list[dict[str, Any]] = []
        eval_id = -1
        original_instruction = None
        for worker_dir in worker_dirs:
            path = worker_dir / basename
            if not path.is_file():
                continue
            data = load_json(path)
            eval_id = int(data.get("eval_id", eval_id))
            original_instruction = (
                data.get("original_instruction") or original_instruction
            )
            episodes.extend(data.get("episodes", []))

        episodes.sort(key=lambda item: int(item["task_id"]))
        successes = sum(bool(episode.get("success")) for episode in episodes)
        total_successes += successes
        per_eval[f"eval{eval_id}"] = {
            "total": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
        }
        for episode in episodes:
            stats = per_category[category_key(episode)]
            stats["total"] += 1
            stats["successes"] += int(bool(episode.get("success")))

        atomic_dump(
            args.output_dir / basename,
            {
                "eval_id": eval_id,
                "original_instruction": original_instruction,
                "episodes": episodes,
            },
        )

    per_category_with_rates: dict[str, dict[str, Any]] = {}
    for key, stats in per_category.items():
        per_category_with_rates[key] = {
            **stats,
            "success_rate": stats["successes"] / stats["total"],
        }

    merged_meta = dict(metas[0])
    merged_meta.update(
        {
            "total_tasks": total_episodes,
            "total_tasks_global": total_episodes,
            "num_shards": args.num_shards,
            "shard_index": None,
            "worker_dirs": [str(path) for path in worker_dirs],
            "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    summary = {
        "overall_success_rate": total_successes / total_episodes,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "per_eval": per_eval,
        "per_category": per_category_with_rates,
    }
    atomic_dump(args.output_dir / "meta.json", merged_meta)
    atomic_dump(args.output_dir / "summary.json", summary)

    print(
        f"Merged {total_episodes} episodes from {args.num_shards} shards: "
        f"success={total_successes}/{total_episodes}, "
        f"SR={100.0 * total_successes / total_episodes:.2f}%"
    )
    print(f"Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
