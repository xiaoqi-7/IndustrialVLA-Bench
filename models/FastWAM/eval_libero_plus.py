#!/usr/bin/env python3
"""FastWAM multi-GPU worker and result aggregator for LIBERO-Plus.

LIBERO-Plus extends the four standard LIBERO suites with 10,030 perturbation
tasks spanning camera, robot state, language, light, background, sensor-noise,
and object-layout robustness.  Tasks are loaded in the exact order specified
by ``task_classification.json`` so task IDs, categories, and difficulty levels
remain auditable in the result files.

The launcher starts one process per selected GPU.  A worker loads FastWAM once
and evaluates its deterministic round-robin task shard.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval_libero import (  # noqa: E402
    SUITE_MAX_STEPS,
    _add_project_paths,
    _atomic_json_dump,
    _build_model,
    _configure_libero_paths,
    _json_default,
    _load_normalizers,
    _run_task,
    _set_seed,
)

LOG = logging.getLogger("fastwam_libero_plus")

SEEDS = (1, 7, 42)
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_PLUS_ROOT = Path("/mnt/afs/zhengmingkai/raozf/benchmark/LIBERO-plus")
DEFAULT_FASTWAM_ROOT = HERE

CATEGORY_TO_COLUMN = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
COLUMNS = ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")


@dataclass(frozen=True)
class PlusTask:
    global_index: int
    suite: str
    task_id: int
    classification_id: int
    task_name: str
    category: str
    column: str
    difficulty_level: int | None


def _parse_suites(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = [part for item in value for part in str(item).replace(",", " ").split()]
    suites = [item.strip() for item in raw if item.strip()]
    if not suites:
        raise ValueError("At least one LIBERO-Plus suite is required")
    unknown = sorted(set(suites) - set(SUITE_MAX_STEPS))
    if unknown:
        raise ValueError(f"Unknown LIBERO suite(s): {', '.join(unknown)}")
    duplicates = [name for name, count in Counter(suites).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate task suite(s): {', '.join(sorted(duplicates))}")
    return suites


def _classification_path(plus_root: Path, explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return plus_root / "libero" / "libero" / "benchmark" / "task_classification.json"


def _load_tasks(classification_path: Path, suites: list[str], max_tasks: int = -1) -> list[PlusTask]:
    if not classification_path.is_file():
        raise FileNotFoundError(f"LIBERO-Plus task classification not found: {classification_path}")
    classification = json.loads(classification_path.read_text(encoding="utf-8"))

    tasks: list[PlusTask] = []
    for suite in suites:
        if suite not in classification:
            raise KeyError(
                f"Suite {suite!r} is not in {classification_path}; available={sorted(classification)}"
            )
        for item in classification[suite]:
            category = str(item.get("category", ""))
            column = CATEGORY_TO_COLUMN.get(category)
            if column is None:
                raise ValueError(f"Unknown LIBERO-Plus category {category!r} for {suite}/{item.get('name')}")
            classification_id = int(item["id"])
            if classification_id <= 0:
                raise ValueError(f"Classification IDs must be one-based and positive: {item}")
            difficulty = item.get("difficulty_level")
            tasks.append(
                PlusTask(
                    global_index=len(tasks),
                    suite=suite,
                    task_id=classification_id - 1,
                    classification_id=classification_id,
                    task_name=str(item["name"]),
                    category=category,
                    column=column,
                    difficulty_level=None if difficulty is None else int(difficulty),
                )
            )
    if max_tasks >= 0:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError("The selected LIBERO-Plus task list is empty")
    return tasks


def _build_suite_objects(tasks: list[PlusTask]) -> dict[str, Any]:
    """Import LIBERO-Plus and construct only suites needed by this worker."""

    from libero.libero import benchmark

    available = benchmark.get_benchmark_dict()
    result: dict[str, Any] = {}
    for suite_name in sorted({task.suite for task in tasks}):
        if suite_name not in available:
            raise KeyError(f"LIBERO-Plus does not register suite {suite_name!r}")
        # Upstream prints the entire 2k-task order at construction time.  It is
        # deterministic order 0, but the giant line makes worker logs unusable.
        with contextlib.redirect_stdout(io.StringIO()):
            result[suite_name] = available[suite_name]()
    return result


def _actual_bddl_path(plus_root: Path, task: Any) -> Path:
    path = plus_root / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
    # Camera/robot/noise variants are encoded in a virtual filename.  The
    # LIBERO-Plus wrapper strips this suffix before opening the underlying BDDL.
    path_text = str(path)
    if "_view_" in path_text and "_initstate_" in path_text:
        path = Path(path_text.split("_view_", 1)[0] + ".bddl")
    return path


@lru_cache(maxsize=None)
def _read_bddl_language(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Underlying LIBERO-Plus BDDL not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("(:language"):
            return stripped[len("(:language") :].rstrip(")").strip()
    raise ValueError(f"No (:language ...) entry found in {path}")


def _read_task_language(plus_root: Path, task: Any) -> str:
    return _read_bddl_language(str(_actual_bddl_path(plus_root, task)))


def _standard_instruction_from_name(task_info: PlusTask) -> str:
    """Recover the unperturbed LIBERO instruction from a Plus task name."""

    name = task_info.task_name
    if task_info.category == "Background Textures":
        markers = ("_table_", "_tb_")
    elif task_info.category in {"Camera Viewpoints", "Robot Initial States", "Sensor Noise"}:
        markers = ("_view_",)
    elif task_info.category == "Light Conditions":
        markers = ("_light_",)
    elif task_info.category == "Objects Layout":
        markers = ("_add_", "_level")
    else:
        markers = ()

    if task_info.category == "Objects Layout" and "_moved_level" in name:
        positions = [name.rfind("_moved_level")]
    else:
        positions = [name.rfind(marker) for marker in markers if name.rfind(marker) >= 0]
    if not positions:
        raise ValueError(f"Could not strip perturbation suffix from {task_info.category}: {name}")
    base = name[: max(positions)]
    if base and base[0].isupper():
        scene = re.search(r"SCENE\d+_", base)
        if scene is None:
            raise ValueError(f"Cannot parse LIBERO-10 scene prefix: {base}")
        base = base[scene.end() :]
    return " ".join(base.split("_"))


def _with_eval_language(plus_root: Path, task: Any, task_info: PlusTask) -> Any:
    """Use rewritten BDDL text only for Language; strip IDs otherwise."""

    if task_info.category == "Language Instructions":
        language = _read_task_language(plus_root, task)
    else:
        language = _standard_instruction_from_name(task_info)
    replacer = getattr(task, "_replace", None)
    if replacer is None:
        raise TypeError(f"LIBERO task object does not support language replacement: {type(task)}")
    return replacer(language=language)


def _failure_result(task_info: PlusTask, num_trials: int, error: BaseException) -> dict[str, Any]:
    return {
        "suite": task_info.suite,
        "task_id": int(task_info.task_id),
        "task_description": "",
        "successes": 0,
        "episodes": int(num_trials),
        "success_episodes": [],
        "failure_episodes": list(range(num_trials)),
        "error": repr(error),
    }


def _annotate_task_result(
    result: dict[str, Any], task_info: PlusTask, task_rng_seed: int
) -> dict[str, Any]:
    result.update(
        {
            "global_index": int(task_info.global_index),
            "classification_id": int(task_info.classification_id),
            "task_name": task_info.task_name,
            "category": task_info.category,
            "column": task_info.column,
            "difficulty_level": task_info.difficulty_level,
            "task_rng_seed": int(task_rng_seed),
        }
    )
    return result


def evaluate_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_seed(args.seed)
    plus_root = Path(args.libero_plus_root).expanduser().resolve()
    suites = _parse_suites(args.task_suites)
    classification_path = _classification_path(plus_root, args.classification_path)
    tasks = _load_tasks(classification_path, suites, args.max_tasks)
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank/world_size must satisfy 0 <= rank < world_size; got {args.rank}/{args.world_size}")
    assigned = tasks[args.rank :: args.world_size]
    if not assigned:
        raise ValueError(f"Worker rank {args.rank} received no LIBERO-Plus tasks")

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

    # Validate suite registration and classification IDs before spending time
    # loading the 25-GB policy snapshot.
    suite_objects = _build_suite_objects(assigned)
    for task_info in assigned:
        suite = suite_objects[task_info.suite]
        if not 0 <= task_info.task_id < int(suite.n_tasks):
            raise IndexError(
                f"Classification task ID outside suite: {task_info.suite}/{task_info.task_id}, n={suite.n_tasks}"
            )
        registered_name = str(suite.get_task(task_info.task_id).name)
        if registered_name != task_info.task_name:
            raise ValueError(
                f"Classification/benchmark mismatch for {task_info.suite}/{task_info.task_id}: "
                f"{task_info.task_name!r} != {registered_name!r}"
            )

    model_dir = Path(args.model_path).expanduser().resolve()
    LOG.info(
        "worker rank=%s/%s physical_gpu=%s device=%s tasks=%s/%s model=%s",
        args.rank,
        args.world_size,
        args.gpu_id,
        device,
        len(assigned),
        len(tasks),
        model_dir,
    )
    model, model_config, dtype = _build_model(model_dir, device)
    state_stats, action_stats = _load_normalizers(model_dir)

    output_dir = Path(args.output_dir)
    run_token = os.environ.get("FASTWAM_RUN_TOKEN", str(output_dir.resolve()))
    task_results: list[dict[str, Any]] = []
    _record_worker_progress(
        output_dir,
        run_token=run_token,
        seed=int(args.seed),
        rank=int(args.rank),
        world_size=int(args.world_size),
        expected_tasks=len(tasks),
        task_results=task_results,
    )
    started = time.time()
    for task_info in assigned:
        # Sensor-noise transformations use NumPy's global RNG.  Deriving a
        # stable per-task RNG makes results independent of GPU/world-size
        # sharding while retaining the requested run seed.
        task_rng_seed = (int(args.seed) + (task_info.global_index + 1) * 1_000_003) % (2**31 - 1)
        _set_seed(task_rng_seed)
        suite = suite_objects[task_info.suite]
        try:
            task = _with_eval_language(plus_root, suite.get_task(task_info.task_id), task_info)
            initial_states = suite.get_task_init_states(task_info.task_id)
            result = _run_task(
                model,
                task,
                initial_states,
                suite_name=task_info.suite,
                task_id=task_info.task_id,
                seed=args.seed,
                num_trials=args.num_trials,
                num_steps_wait=args.num_steps_wait,
                replan_steps=args.replan_steps,
                action_horizon=args.action_horizon or int(model_config.get("action_horizon", 32)),
                num_inference_steps=args.num_inference_steps
                or int(model_config.get("num_inference_steps", 10)),
                state_stats=state_stats,
                action_stats=action_stats,
                device=device,
                dtype=dtype,
                binarize_gripper=args.binarize_gripper,
                max_steps_override=args.max_steps,
            )
        except Exception as exc:
            LOG.exception(
                "Task failed: seed=%s suite=%s task=%s category=%s",
                args.seed,
                task_info.suite,
                task_info.task_name,
                task_info.category,
            )
            if args.fail_fast:
                raise
            result = _failure_result(task_info, args.num_trials, exc)
        task_results.append(_annotate_task_result(result, task_info, task_rng_seed))
        _record_worker_progress(
            output_dir,
            run_token=run_token,
            seed=int(args.seed),
            rank=int(args.rank),
            world_size=int(args.world_size),
            expected_tasks=len(tasks),
            task_results=task_results,
        )

    successes = sum(int(item["successes"]) for item in task_results)
    episodes = sum(int(item["episodes"]) for item in task_results)
    errors = sum(1 for item in task_results if item.get("error"))
    result = {
        "format": "fastwam-libero-plus-worker-v1",
        "seed": int(args.seed),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "gpu_id": str(args.gpu_id),
        "model_path": str(model_dir),
        "libero_plus_root": str(plus_root),
        "classification_path": str(classification_path),
        "task_suites": suites,
        "num_tasks_total": len(tasks),
        "num_tasks_assigned": len(assigned),
        "num_trials_per_task": int(args.num_trials),
        "successes": successes,
        "episodes": episodes,
        "errors": errors,
        "success_rate": successes / episodes if episodes else 0.0,
        "duration_s": time.time() - started,
        "tasks": task_results,
    }
    output = output_dir / f"worker_{args.rank:03d}.json"
    _atomic_json_dump(output, result)
    LOG.info(
        "Wrote %s: %s/%s (%.4f), task_errors=%s",
        output,
        successes,
        episodes,
        result["success_rate"],
        errors,
    )
    return result


def _add_group_value(group: dict[str, dict[str, Any]], key: str, task: dict[str, Any]) -> None:
    entry = group.setdefault(str(key), {"successes": 0, "episodes": 0, "tasks": 0, "errors": 0})
    entry["successes"] += int(task["successes"])
    entry["episodes"] += int(task["episodes"])
    entry["tasks"] += 1
    entry["errors"] += int(bool(task.get("error")))


def _finish_group(group: dict[str, dict[str, Any]]) -> None:
    for entry in group.values():
        entry["success_rate"] = entry["successes"] / entry["episodes"] if entry["episodes"] else 0.0


def _progress_payload(
    *,
    run_token: str,
    seed: int,
    rank: int,
    world_size: int,
    expected_tasks: int,
    task_results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_column: dict[str, dict[str, Any]] = {}
    for task in task_results:
        _add_group_value(by_column, str(task["column"]), task)
    _finish_group(by_column)
    successes = sum(int(task["successes"]) for task in task_results)
    episodes = sum(int(task["episodes"]) for task in task_results)
    errors = sum(int(bool(task.get("error"))) for task in task_results)
    return {
        "format": "fastwam-libero-plus-progress-v1",
        "run_token": run_token,
        "seed": int(seed),
        "rank": int(rank),
        "world_size": int(world_size),
        "expected_tasks": int(expected_tasks),
        "completed_tasks": len(task_results),
        "successes": successes,
        "episodes": episodes,
        "errors": errors,
        "success_rate": successes / episodes if episodes else 0.0,
        "by_column": by_column,
    }


def _append_total_progress(output_dir: Path, payload: dict[str, Any]) -> None:
    """Append one de-duplicated, globally aggregated progress line."""

    import fcntl

    progress_dir = output_dir / "progress"
    lock_path = progress_dir / "total.lock"
    state_path = progress_dir / "last_total.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        snapshots = []
        for path in sorted(progress_dir.glob("worker_*.json")):
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                snapshot.get("run_token") == payload["run_token"]
                and int(snapshot.get("seed", -1)) == int(payload["seed"])
                and int(snapshot.get("world_size", -1)) == int(payload["world_size"])
            ):
                snapshots.append(snapshot)

        completed_tasks = sum(int(item.get("completed_tasks", 0)) for item in snapshots)
        if completed_tasks <= 0:
            return
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        same_run = previous.get("run_token") == payload["run_token"]
        previous_completed = int(previous.get("completed_tasks", 0)) if same_run else 0
        if previous_completed >= completed_tasks:
            return

        log_every = max(1, int(os.environ.get("PROGRESS_LOG_EVERY", "1")))
        expected_tasks = int(payload["expected_tasks"])
        if completed_tasks != expected_tasks and completed_tasks - previous_completed < log_every:
            return

        successes = sum(int(item.get("successes", 0)) for item in snapshots)
        episodes = sum(int(item.get("episodes", 0)) for item in snapshots)
        errors = sum(int(item.get("errors", 0)) for item in snapshots)
        fields = []
        for column in COLUMNS:
            entries = [item.get("by_column", {}).get(column, {}) for item in snapshots]
            column_successes = sum(int(entry.get("successes", 0)) for entry in entries)
            column_episodes = sum(int(entry.get("episodes", 0)) for entry in entries)
            if column_episodes:
                fields.append(f"{column}={100.0 * column_successes / column_episodes:.2f}%")
            else:
                fields.append(f"{column}=n/a")
        total_rate = successes / episodes if episodes else 0.0
        progress = 100.0 * completed_tasks / expected_tasks if expected_tasks else 0.0
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        line = (
            f"[progress] {timestamp} seed={payload['seed']} "
            f"completed_tasks={completed_tasks}/{expected_tasks} ({progress:.2f}%) "
            + " ".join(fields)
            + f" Total={100.0 * total_rate:.2f}% ({successes}/{episodes}) task_errors={errors}\n"
        )
        with (output_dir.parent / "total.log").open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
        _atomic_json_dump(
            state_path,
            {
                "run_token": payload["run_token"],
                "seed": payload["seed"],
                "completed_tasks": completed_tasks,
            },
        )


def _record_worker_progress(
    output_dir: Path,
    *,
    run_token: str,
    seed: int,
    rank: int,
    world_size: int,
    expected_tasks: int,
    task_results: list[dict[str, Any]],
) -> None:
    try:
        payload = _progress_payload(
            run_token=run_token,
            seed=seed,
            rank=rank,
            world_size=world_size,
            expected_tasks=expected_tasks,
            task_results=task_results,
        )
        progress_dir = output_dir / "progress"
        _atomic_json_dump(progress_dir / f"worker_{rank:03d}.json", payload)
        _append_total_progress(output_dir, payload)
    except Exception:
        # Progress logging is diagnostic and must never invalidate an otherwise
        # successful benchmark episode.
        LOG.exception("Could not update shared LIBERO-Plus progress log")


def _entry_success_rate(entry: dict[str, Any]) -> float:
    successes = int(entry.get("successes", 0))
    episodes = int(entry.get("episodes", 0))
    return successes / episodes if episodes else 0.0


def _atomic_text_dump(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _evaluated_task_count(result: dict[str, Any]) -> int:
    if "evaluated_tasks" in result:
        return int(result["evaluated_tasks"])
    return sum(int(entry.get("tasks", 0)) for entry in result.get("by_column", {}).values())


def _write_seed_markdown_summary(path: Path, result: dict[str, Any]) -> None:
    column_entries = [result.get("by_column", {}).get(column, {}) for column in COLUMNS]
    values = [100.0 * _entry_success_rate(entry) for entry in column_entries]
    total = 100.0 * _entry_success_rate(result)
    lines = [
        f"# FastWAM LIBERO-Plus seed {result['seed']} summary",
        "",
        "| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |",
        "|-------|--------|-------|----------|-------|------------|-------|--------|-------|",
        "| " + " | ".join(["FastWAM", *[f"{value:.1f}" for value in values], f"{total:.1f}"]) + " |",
        "",
        "Counts:",
    ]
    for column, entry in zip(COLUMNS, column_entries, strict=True):
        lines.append(f"{column}: {int(entry.get('successes', 0))}/{int(entry.get('episodes', 0))}")
    lines.extend(
        [
            f"Total: {int(result['successes'])}/{int(result['episodes'])}",
            "",
            f"Evaluated tasks: {_evaluated_task_count(result)}",
            f"Task errors: {int(result.get('errors', 0))}",
        ]
    )
    _atomic_text_dump(path, "\n".join(lines) + "\n")


def aggregate_seed(output_dir: Path, *, expected_workers: int, seed: int) -> dict[str, Any]:
    if expected_workers <= 0:
        raise ValueError("expected_workers must be positive")
    workers = []
    for rank in range(expected_workers):
        path = output_dir / f"worker_{rank:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing worker result for rank {rank}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("seed")) != int(seed):
            raise ValueError(f"Seed mismatch in {path}: {payload.get('seed')} != {seed}")
        workers.append(payload)

    tasks: list[dict[str, Any]] = []
    for worker in workers:
        tasks.extend(worker.get("tasks", []))
    tasks.sort(key=lambda item: int(item["global_index"]))
    successes = sum(int(item["successes"]) for item in tasks)
    episodes = sum(int(item["episodes"]) for item in tasks)
    errors = sum(int(bool(item.get("error"))) for item in tasks)
    expected_task_counts = {
        int(worker["num_tasks_total"])
        for worker in workers
        if worker.get("num_tasks_total") is not None
    }
    expected_tasks = next(iter(expected_task_counts)) if len(expected_task_counts) == 1 else len(tasks)

    by_suite: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    by_column: dict[str, dict[str, Any]] = {}
    by_difficulty: dict[str, dict[str, Any]] = {}
    for task in tasks:
        _add_group_value(by_suite, str(task["suite"]), task)
        _add_group_value(by_category, str(task["category"]), task)
        _add_group_value(by_column, str(task["column"]), task)
        difficulty = "unclassified" if task.get("difficulty_level") is None else str(task["difficulty_level"])
        _add_group_value(by_difficulty, difficulty, task)
    for group in (by_suite, by_category, by_column, by_difficulty):
        _finish_group(group)

    result = {
        "format": "fastwam-libero-plus-seed-v1",
        "seed": int(seed),
        "workers": int(expected_workers),
        "successes": successes,
        "episodes": episodes,
        "errors": errors,
        "evaluated_tasks": len(tasks),
        "expected_tasks": expected_tasks,
        "success_rate": successes / episodes if episodes else 0.0,
        "by_suite": by_suite,
        "by_category": by_category,
        "by_column": by_column,
        "by_difficulty": by_difficulty,
        "tasks": tasks,
    }
    _atomic_json_dump(output_dir / "summary.json", result)
    _write_seed_markdown_summary(output_dir / "summary.md", result)
    return result


def _aggregate_group(seed_results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    names = sorted({name for result in seed_results for name in result.get(key, {})})
    output: dict[str, Any] = {}
    for name in names:
        entries = [result[key][name] for result in seed_results if name in result.get(key, {})]
        rates = [_entry_success_rate(entry) for entry in entries]
        successes = sum(int(entry["successes"]) for entry in entries)
        episodes = sum(int(entry["episodes"]) for entry in entries)
        output[name] = {
            "seed_rates": {
                str(result["seed"]): _entry_success_rate(result[key][name])
                for result in seed_results
                if name in result.get(key, {})
            },
            "mean_success_rate": sum(rates) / len(rates),
            "pooled_successes": successes,
            "pooled_episodes": episodes,
            "pooled_success_rate": successes / episodes if episodes else 0.0,
            "errors": sum(int(entry.get("errors", 0)) for entry in entries),
        }
    return output


def _write_three_seed_markdown_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# FastWAM LIBERO-Plus three-seed summary",
        "",
        "| Seed | Camera | Robot | Language | Light | Background | Noise | Layout | Total | Evaluated tasks |",
        "|-----:|-------:|------:|---------:|------:|-----------:|------:|-------:|------:|----------------:|",
    ]
    for seed_result in result["seed_results"]:
        values = [
            100.0 * _entry_success_rate(seed_result.get("by_column", {}).get(column, {}))
            for column in COLUMNS
        ]
        total = 100.0 * _entry_success_rate(seed_result)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed_result["seed"]),
                    *[f"{value:.1f}" for value in values],
                    f"{total:.1f}",
                    str(_evaluated_task_count(seed_result)),
                ]
            )
            + " |"
        )

    mean_values = [
        100.0 * float(result.get("by_column", {}).get(column, {}).get("mean_success_rate", 0.0))
        for column in COLUMNS
    ]
    mean_total = 100.0 * float(result["mean_success_rate"])
    lines.extend(
        [
            "| "
            + " | ".join(
                ["**Mean**", *[f"**{value:.1f}**" for value in mean_values], f"**{mean_total:.1f}**", "—"]
            )
            + " |",
            "",
            "Counts:",
        ]
    )
    for seed_result in result["seed_results"]:
        successes = int(seed_result["successes"])
        episodes = int(seed_result["episodes"])
        errors = int(seed_result.get("errors", 0))
        lines.append(
            f"Seed {seed_result['seed']}: {successes}/{episodes} "
            f"({episodes - successes} failures; {errors} task errors)"
        )
    lines.extend(
        [
            "",
            "The mean is the arithmetic mean of the three unrounded per-seed success rates, "
            "giving each seed equal weight.",
            f"Pooled: {result['pooled_successes']}/{result['pooled_episodes']} "
            f"({100.0 * float(result['pooled_success_rate']):.1f}%)",
        ]
    )
    _atomic_text_dump(path, "\n".join(lines) + "\n")


def aggregate_three_seed(output_root: Path, seeds: Iterable[int]) -> dict[str, Any]:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("At least one seed is required")
    seed_results = []
    for seed in seed_list:
        path = output_root / f"seed{seed}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed summary: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if int(result.get("seed")) != seed:
            raise ValueError(f"Seed mismatch in {path}")
        seed_results.append(result)
        # Backfill/refresh the per-seed Markdown when aggregating an existing
        # run produced before seed-level tables were introduced.
        _write_seed_markdown_summary(path.with_suffix(".md"), result)

    rates = [_entry_success_rate(result) for result in seed_results]
    pooled_successes = sum(int(result["successes"]) for result in seed_results)
    pooled_episodes = sum(int(result["episodes"]) for result in seed_results)
    result = {
        "format": "fastwam-libero-plus-three-seed-v1",
        "seeds": seed_list,
        "seed_results": [{key: value for key, value in item.items() if key != "tasks"} for item in seed_results],
        "mean_success_rate": sum(rates) / len(rates),
        "pooled_successes": pooled_successes,
        "pooled_episodes": pooled_episodes,
        "pooled_success_rate": pooled_successes / pooled_episodes if pooled_episodes else 0.0,
        "errors": sum(int(item.get("errors", 0)) for item in seed_results),
        "by_suite": _aggregate_group(seed_results, "by_suite"),
        "by_category": _aggregate_group(seed_results, "by_category"),
        "by_column": _aggregate_group(seed_results, "by_column"),
        "by_difficulty": _aggregate_group(seed_results, "by_difficulty"),
    }
    _atomic_json_dump(output_root / "three_seed_summary.json", result)
    _write_three_seed_markdown_summary(output_root / "summary.md", result)
    # Preserve the original filename for downstream consumers while exposing
    # the reference-compatible ``summary.md`` requested by the benchmark.
    _write_three_seed_markdown_summary(output_root / "three_seed_summary.md", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--libero-plus-root", default=str(DEFAULT_PLUS_ROOT))
    parser.add_argument("--classification-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--task-suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-binarize-gripper", dest="binarize_gripper", action="store_false")
    parser.set_defaults(binarize_gripper=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-workers", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-workers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--aggregate-three-seed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS), help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    plus_root = Path(args.libero_plus_root).expanduser().resolve()
    fastwam_root = Path(os.environ.get("FASTWAM_SOURCE_ROOT", str(DEFAULT_FASTWAM_ROOT))).expanduser()
    _add_project_paths(fastwam_root, plus_root)
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _configure_libero_paths(plus_root, Path(args.output_dir).expanduser().resolve())

    if args.aggregate_three_seed:
        root = Path(args.output_root or args.output_dir).expanduser().resolve()
        print(json.dumps(aggregate_three_seed(root, args.seeds), indent=2, ensure_ascii=False, default=_json_default))
        return
    if args.aggregate_workers:
        seed_dir = Path(args.output_dir).expanduser().resolve()
        print(
            json.dumps(
                aggregate_seed(seed_dir, expected_workers=int(args.expected_workers), seed=int(args.seed)),
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
        )
        return

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if args.num_trials <= 0 or args.world_size <= 0:
        raise ValueError("--num-trials and --world-size must be positive")
    if args.max_tasks == 0 or args.max_tasks < -1:
        raise ValueError("--max-tasks must be -1 or positive")
    if args.num_steps_wait < 0 or args.replan_steps <= 0 or args.max_steps < 0:
        raise ValueError("wait/max steps must be non-negative and replan steps must be positive")

    suites = _parse_suites(args.task_suites)
    classification_path = _classification_path(plus_root, args.classification_path)
    tasks = _load_tasks(classification_path, suites, args.max_tasks)
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank must be in [0, {args.world_size}), got {args.rank}")
    assigned = tasks[args.rank :: args.world_size]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "rank": args.rank,
                    "world_size": args.world_size,
                    "tasks_total": len(tasks),
                    "tasks_assigned": len(assigned),
                    "task_suites": suites,
                    "suite_counts": dict(Counter(task.suite for task in tasks)),
                    "column_counts": dict(Counter(task.column for task in tasks)),
                    "model_path": str(model_path),
                    "libero_plus_root": str(plus_root),
                    "classification_path": str(classification_path),
                },
                separators=(",", ":"),
            )
        )
        return
    evaluate_worker(args)


if __name__ == "__main__":
    main()
