#!/usr/bin/env python3
"""Validate and merge sharded GR00T LIBERO-Para logs into one seed result."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
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
    paraphrase_type = str(episode.get("paraphrase_type", "unknown"))
    categories = episode.get("categories", [])
    subcategories = episode.get("subcategories", [])
    if paraphrase_type == "comp":
        return f"comp_{'+'.join(categories)}_{'+'.join(subcategories)}"
    if categories and subcategories:
        return f"{paraphrase_type}_{categories[0]}_{subcategories[0]}"
    return "unknown"


def require_equal(metas: list[dict[str, Any]], key: str) -> Any:
    values = [meta.get(key) for meta in metas]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Worker metadata disagree on {key}: {values}")
    return values[0]


def eval_sort_key(name: str) -> int:
    return int(Path(name).stem.removeprefix("eval"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.expected_total < 1:
        parser.error("--expected-total must be >= 1")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")

    worker_dirs = [args.workers_dir / f"rank_{rank}" for rank in range(args.num_shards)]
    missing_worker_dirs = [str(path) for path in worker_dirs if not path.is_dir()]
    if missing_worker_dirs:
        raise FileNotFoundError(f"Missing worker directories: {missing_worker_dirs}")

    metas = [load_json(path / "meta.json") for path in worker_dirs]
    summaries = [load_json(path / "summary.json") for path in worker_dirs]
    progresses = [load_json(path / "progress.json") for path in worker_dirs]

    for key in (
        "model_name",
        "model_family",
        "global_seed",
        "num_shards",
        "total_tasks_global",
        "max_steps",
        "n_action_steps",
        "num_steps_wait",
        "initial_state_index",
        "bddl_dir",
    ):
        require_equal(metas, key)

    if int(metas[0]["global_seed"]) != args.seed:
        raise ValueError(
            f"Worker global seed {metas[0]['global_seed']} does not match requested {args.seed}"
        )
    if int(metas[0]["num_shards"]) != args.num_shards:
        raise ValueError("Worker num_shards does not match the merge configuration")
    if int(metas[0]["total_tasks_global"]) != args.expected_total:
        raise ValueError("Worker total_tasks_global does not match --expected-total")

    seen_task_ids: set[int] = set()
    seen_bddl_files: set[str] = set()
    eval_basenames: set[str] = set()

    for rank, (worker_dir, meta, summary, progress) in enumerate(
        zip(worker_dirs, metas, summaries, progresses, strict=True)
    ):
        expected_local = len(range(rank, args.expected_total, args.num_shards))
        if int(meta["shard_index"]) != rank:
            raise ValueError(
                f"{worker_dir} records shard_index={meta['shard_index']}, expected {rank}"
            )
        if int(meta["total_tasks_local"]) != expected_local:
            raise ValueError(
                f"{worker_dir} records {meta['total_tasks_local']} local tasks, "
                f"expected {expected_local}"
            )
        if int(summary["total_episodes"]) != expected_local:
            raise ValueError(f"Incomplete summary in {worker_dir}")
        if int(progress["completed"]) != expected_local:
            raise ValueError(f"Incomplete progress in {worker_dir}")

        worker_episode_count = 0
        for path in worker_dir.glob("eval*.json"):
            eval_basenames.add(path.name)
            data = load_json(path)
            eval_id = int(data["eval_id"])
            if path.name != f"eval{eval_id}.json":
                raise ValueError(f"Evaluation ID/file mismatch in {path}")
            for episode in data.get("episodes", []):
                task_id = int(episode["task_id"])
                if not 0 <= task_id < args.expected_total:
                    raise ValueError(f"Out-of-range global task_id in {path}: {task_id}")
                if task_id % args.num_shards != rank:
                    raise ValueError(
                        f"Task {task_id} is in rank {rank}, violating round-robin sharding"
                    )
                if int(episode.get("shard_index", -1)) != rank:
                    raise ValueError(f"Episode shard metadata mismatch for task {task_id}")
                if int(episode.get("seed", args.seed)) != args.seed:
                    raise ValueError(f"Episode seed mismatch for task {task_id}")
                if task_id in seen_task_ids:
                    raise ValueError(f"Duplicate global task_id across shards: {task_id}")
                bddl_file = str(episode["bddl_file"])
                if bddl_file in seen_bddl_files:
                    raise ValueError(f"Duplicate BDDL across shards: {bddl_file}")
                seen_task_ids.add(task_id)
                seen_bddl_files.add(bddl_file)
                worker_episode_count += 1

        if worker_episode_count != expected_local:
            raise ValueError(
                f"Expected {expected_local} episode records in {worker_dir}, "
                f"found {worker_episode_count}"
            )

    expected_ids = set(range(args.expected_total))
    if seen_task_ids != expected_ids:
        missing = sorted(expected_ids - seen_task_ids)
        extra = sorted(seen_task_ids - expected_ids)
        raise ValueError(
            f"Global task IDs are incomplete: missing={missing[:20]}, extra={extra[:20]}"
        )

    per_eval: dict[str, dict[str, int | float]] = {}
    per_category: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "successes": 0}
    )
    total_successes = 0

    for basename in sorted(eval_basenames, key=eval_sort_key):
        episodes: list[dict[str, Any]] = []
        eval_id: int | None = None
        original_instruction: str | None = None
        for worker_dir in worker_dirs:
            path = worker_dir / basename
            if not path.is_file():
                continue
            data = load_json(path)
            current_eval_id = int(data["eval_id"])
            if eval_id is not None and current_eval_id != eval_id:
                raise ValueError(f"Conflicting eval IDs while merging {basename}")
            eval_id = current_eval_id
            original_instruction = data.get("original_instruction") or original_instruction
            episodes.extend(data.get("episodes", []))

        if eval_id is None or not episodes:
            raise ValueError(f"No episodes found while merging {basename}")
        episodes.sort(key=lambda item: int(item["task_id"]))
        successes = sum(int(bool(episode.get("success"))) for episode in episodes)
        total_successes += successes
        per_eval[f"eval{eval_id}"] = {
            "total": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes),
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

    server_seeds = [int(meta["server_seed"]) for meta in metas]
    merged_meta = dict(metas[0])
    merged_meta.update(
        {
            "seed": args.seed,
            "global_seed": args.seed,
            "server_seed": None,
            "server_seeds": server_seeds,
            "server_seed_by_shard": {str(rank): seed for rank, seed in enumerate(server_seeds)},
            "ports": [int(meta["port"]) for meta in metas],
            "total_tasks": args.expected_total,
            "total_tasks_local": None,
            "total_tasks_global": args.expected_total,
            "num_shards": args.num_shards,
            "shard_index": None,
            "worker_dirs": [str(path) for path in worker_dirs],
            "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    summary = {
        "overall_success_rate": total_successes / args.expected_total,
        "total_episodes": args.expected_total,
        "total_successes": total_successes,
        "per_eval": per_eval,
        "per_category": dict(per_category),
    }
    progress = {
        "completed": args.expected_total,
        "successes": total_successes,
        "success_rate": total_successes / args.expected_total,
        "total": args.expected_total,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_dump(args.output_dir / "meta.json", merged_meta)
    atomic_dump(args.output_dir / "summary.json", summary)
    atomic_dump(args.output_dir / "progress.json", progress)

    print(
        f"Merged {args.expected_total} episodes from {args.num_shards} shards: "
        f"success={total_successes}/{args.expected_total}, "
        f"SR={100.0 * total_successes / args.expected_total:.2f}%",
        flush=True,
    )
    print(f"Summary: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
