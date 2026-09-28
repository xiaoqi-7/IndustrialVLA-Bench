#!/usr/bin/env python3
"""Write a compact success-rate report for the four standard LIBERO suites.

The evaluator keeps a detailed ``summary.json`` below every seed directory.
This utility reduces those files to a small report that is easy to inspect or
feed into a results table.  By default all requested seeds must be complete;
``--allow-partial`` is useful while a long-running evaluation is still in
progress and records which seeds were available at the time of the report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEEDS = (1, 7, 42)
DEFAULT_SUBTASKS = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_seed_summaries(
    run_root: Path,
    seeds: Iterable[int],
    *,
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    results: list[dict[str, Any]] = []
    missing: list[int] = []
    for raw_seed in seeds:
        seed = int(raw_seed)
        path = run_root / f"seed{seed}" / "summary.json"
        if not path.is_file():
            missing.append(seed)
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read seed summary {path}: {exc}") from exc
        if int(result.get("seed", seed)) != seed:
            raise ValueError(f"Seed mismatch in {path}: {result.get('seed')} != {seed}")
        if not isinstance(result.get("suites"), dict):
            raise ValueError(f"Seed summary has no suite statistics: {path}")
        results.append(result)

    if missing and not allow_partial:
        paths = ", ".join(str(run_root / f"seed{seed}" / "summary.json") for seed in missing)
        raise FileNotFoundError(f"Missing seed summary file(s): {paths}")
    if not results:
        raise ValueError(f"No completed seed summaries found below {run_root}")
    return results, missing


def build_summary(
    seed_results: list[dict[str, Any]],
    *,
    expected_seeds: list[int],
    missing_seeds: list[int],
) -> dict[str, Any]:
    included_seeds = [int(result["seed"]) for result in seed_results]
    # Keep the canonical LIBERO order in the report; append any explicitly
    # evaluated non-standard suites deterministically after it.
    extra_suites: set[str] = set()
    for result in seed_results:
        extra_suites.update(str(name) for name in result.get("suites", {}))
    suite_names = [*DEFAULT_SUBTASKS, *sorted(extra_suites - set(DEFAULT_SUBTASKS))]

    subtasks: dict[str, dict[str, Any]] = {}
    for suite in suite_names:
        entries = [result["suites"][suite] for result in seed_results if suite in result.get("suites", {})]
        if not entries:
            # Keep the standard four names visible, but do not fabricate a
            # zero rate for a suite that was not evaluated at all.
            subtasks[suite] = {
                "seed_rates": {},
                "mean_success_rate": None,
                "pooled_successes": 0,
                "pooled_episodes": 0,
                "pooled_success_rate": None,
                "status": "not_available",
            }
            continue

        successes = sum(int(entry.get("successes", 0)) for entry in entries)
        episodes = sum(int(entry.get("episodes", 0)) for entry in entries)
        seed_rates = {}
        rates = []
        for result in seed_results:
            entry = result.get("suites", {}).get(suite)
            if entry is None:
                continue
            entry_successes = int(entry.get("successes", 0))
            entry_episodes = int(entry.get("episodes", 0))
            rate = entry_successes / entry_episodes if entry_episodes else 0.0
            seed_rates[str(result["seed"])] = rate
            rates.append(rate)
        subtasks[suite] = {
            "seed_rates": seed_rates,
            "mean_success_rate": sum(rates) / len(rates),
            "pooled_successes": successes,
            "pooled_episodes": episodes,
            "pooled_success_rate": successes / episodes if episodes else 0.0,
            "status": "complete" if len(entries) == len(expected_seeds) else "partial",
        }

    total_successes = sum(int(result.get("successes", 0)) for result in seed_results)
    total_episodes = sum(int(result.get("episodes", 0)) for result in seed_results)
    seed_rates = []
    for result in seed_results:
        successes = int(result.get("successes", 0))
        episodes = int(result.get("episodes", 0))
        seed_rates.append(successes / episodes if episodes else 0.0)
    return {
        "format": "fastwam-libero-four-task-summary-v1",
        "status": "complete" if not missing_seeds else "partial",
        "expected_seeds": expected_seeds,
        "included_seeds": included_seeds,
        "missing_seeds": missing_seeds,
        "aggregation_policy": "Only complete seed summary.json files are included; partial worker outputs are excluded.",
        "overall": {
            "mean_success_rate": sum(seed_rates) / len(seed_rates),
            "pooled_successes": total_successes,
            "pooled_episodes": total_episodes,
            "pooled_success_rate": total_successes / total_episodes if total_episodes else 0.0,
        },
        "subtasks": subtasks,
        # Keep the evaluator's existing terminology available to consumers
        # that already read ``summary.json`` files.
        "suites": subtasks,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# FastWAM LIBERO four-subtask summary",
        "",
        f"Status: **{summary['status']}**",
        f"Included seeds: {', '.join(str(seed) for seed in summary['included_seeds'])}",
    ]
    if summary["missing_seeds"]:
        lines.append(f"Missing seeds: {', '.join(str(seed) for seed in summary['missing_seeds'])}")
    lines.extend(
        [
            "",
            "| Subtask | Pooled successes | Pooled episodes | Pooled rate | Mean seed rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, entry in summary["subtasks"].items():
        pooled_rate = entry["pooled_success_rate"]
        mean_rate = entry["mean_success_rate"]
        pooled_text = "n/a" if pooled_rate is None else f"{100.0 * pooled_rate:.2f}%"
        mean_text = "n/a" if mean_rate is None else f"{100.0 * mean_rate:.2f}%"
        lines.append(
            f"| `{name}` | {entry['pooled_successes']} | {entry['pooled_episodes']} | "
            f"{pooled_text} | {mean_text} |"
        )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            f"Overall pooled rate: **{100.0 * overall['pooled_success_rate']:.2f}%** "
            f"({overall['pooled_successes']}/{overall['pooled_episodes']})",
            f"Overall arithmetic mean seed rate: **{100.0 * overall['mean_success_rate']:.2f}%**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing seed*/summary.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a partial report when one or more requested seed summaries are missing",
    )
    args = parser.parse_args()

    expected_seeds = [int(seed) for seed in args.seeds]
    try:
        seed_results, missing = _load_seed_summaries(
            args.run_root,
            expected_seeds,
            allow_partial=args.allow_partial,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    summary = build_summary(seed_results, expected_seeds=expected_seeds, missing_seeds=missing)
    _atomic_json_dump(args.run_root / "four_task_summary.json", summary)
    _write_markdown(args.run_root / "four_task_summary.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
