#!/usr/bin/env python3
"""Aggregate completed Isaac GR00T LIBERO-Para seed runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics


def atomic_dump(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    seed_results = []
    for seed in args.seeds:
        seed_dir = args.run_root / f"seed{seed}"
        with (seed_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        with (seed_dir / "meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if int(meta["seed"]) != seed:
            raise ValueError(f"Seed metadata mismatch in {seed_dir}: {meta['seed']} != {seed}")
        if int(meta.get("global_seed", meta["seed"])) != seed:
            raise ValueError(f"Global seed metadata mismatch in {seed_dir}")
        seed_results.append(
            {
                "seed": seed,
                "num_shards": int(meta.get("num_shards", 1)),
                "server_seeds": meta.get("server_seeds", [meta.get("server_seed", seed)]),
                "successes": int(summary["total_successes"]),
                "episodes": int(summary["total_episodes"]),
                "success_rate": float(summary["overall_success_rate"]),
                "result_dir": str(seed_dir),
            }
        )

    episode_counts = {result["episodes"] for result in seed_results}
    if len(episode_counts) != 1:
        raise ValueError(f"Seed runs have different episode counts: {sorted(episode_counts)}")
    shard_counts = {result["num_shards"] for result in seed_results}
    if len(shard_counts) != 1:
        raise ValueError(f"Seed runs have different shard counts: {sorted(shard_counts)}")

    rates = [result["success_rate"] for result in seed_results]
    total_successes = sum(result["successes"] for result in seed_results)
    total_episodes = sum(result["episodes"] for result in seed_results)
    aggregate = {
        "model_name": "Isaac-GR00T-N1.7-LIBERO-goal",
        "seeds": args.seeds,
        "seed_results": seed_results,
        "mean_success_rate": statistics.fmean(rates),
        "std_success_rate": statistics.pstdev(rates),
        "pooled_success_rate": total_successes / total_episodes,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
    }
    atomic_dump(args.run_root / "three_seed_summary.json", aggregate)

    lines = [
        "# Isaac GR00T LIBERO-Para three-seed summary",
        "",
        "| Seed | GPU workers | Successes | Episodes | Success rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in seed_results:
        lines.append(
            f"| {result['seed']} | {result['num_shards']} | {result['successes']} | "
            f"{result['episodes']} | "
            f"{100.0 * result['success_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Mean: {100.0 * aggregate['mean_success_rate']:.2f}%",
            f"Population std: {100.0 * aggregate['std_success_rate']:.2f}%",
            f"Pooled: {100.0 * aggregate['pooled_success_rate']:.2f}%",
            "",
        ]
    )
    (args.run_root / "three_seed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
