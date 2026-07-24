#!/usr/bin/env python3
"""Merge sharded LIBERO-Para JSON logs into the official single-run layout."""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def atomic_dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def category_key(episode):
    paraphrase_type = episode.get("paraphrase_type", "unknown")
    categories = episode.get("categories", [])
    subcategories = episode.get("subcategories", [])
    if paraphrase_type == "comp":
        return f"comp_{'+'.join(categories)}_{'+'.join(subcategories)}"
    if categories and subcategories:
        return f"{paraphrase_type}_{categories[0]}_{subcategories[0]}"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    worker_dirs = sorted(path for path in args.workers_dir.glob("rank_*") if path.is_dir())
    if len(worker_dirs) != args.num_shards:
        raise SystemExit(
            f"Expected {args.num_shards} worker directories, found {len(worker_dirs)}"
        )

    # Validate completeness and global task uniqueness before publishing output.
    seen_task_ids = set()
    total_episodes = 0
    eval_basenames = set()
    for worker_dir in worker_dirs:
        for path in worker_dir.glob("eval*.json"):
            eval_basenames.add(path.name)
            data = load_json(path)
            for episode in data.get("episodes", []):
                task_id = int(episode["task_id"])
                if task_id in seen_task_ids:
                    raise SystemExit(f"Duplicate global task_id across shards: {task_id}")
                seen_task_ids.add(task_id)
                total_episodes += 1

    if total_episodes != args.expected_total:
        raise SystemExit(
            f"Expected {args.expected_total} episodes, found {total_episodes}; refusing partial merge"
        )
    if seen_task_ids != set(range(args.expected_total)):
        missing = sorted(set(range(args.expected_total)) - seen_task_ids)
        raise SystemExit(f"Global task IDs are incomplete; first missing IDs: {missing[:20]}")

    per_eval = {}
    per_category = defaultdict(lambda: {"total": 0, "successes": 0})
    total_successes = 0

    for basename in sorted(eval_basenames):
        episodes = []
        eval_id = -1
        original_instruction = None
        for worker_dir in worker_dirs:
            path = worker_dir / basename
            if not path.is_file():
                continue
            data = load_json(path)
            eval_id = int(data.get("eval_id", eval_id))
            original_instruction = data.get("original_instruction") or original_instruction
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

    for stats in per_category.values():
        stats["success_rate"] = stats["successes"] / stats["total"]

    metas = [load_json(path / "meta.json") for path in worker_dirs]
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
        "per_category": dict(per_category),
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
