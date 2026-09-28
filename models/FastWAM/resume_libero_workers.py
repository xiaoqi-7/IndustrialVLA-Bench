#!/usr/bin/env python3
"""Resume interrupted standard LIBERO worker shards.

This utility is intentionally separate from ``run_fastwam_libero.sh``.  It
reads the episode records already present in a worker log, evaluates only the
missing episodes, appends new records to that same log, and writes a complete
``worker_<rank>.json``.  It is useful when a long evaluation was interrupted
after some workers had finished their shards.

The default invocation targets the interrupted seed-42 workers 6 and 7 from
the result directory supplied in the benchmark request.  Use ``--dry-run``
first to inspect the resume plan without loading the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = HERE / "result" / "libero_3seeds" / "20260716_194717"
DEFAULT_MODEL_PATH = Path("/mnt/afs/zhengmingkai/raozf/models/fastwam_libero")
DEFAULT_LIBERO_ROOT = HERE.parent / "LIBERO"
DEFAULT_FASTWAM_ROOT = HERE
DEFAULT_SEED = 42
DEFAULT_WORLD_SIZE = 8
DEFAULT_RANKS = (6, 7)
DEFAULT_GPUS = (6, 7)
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_TRIALS = 50

EPISODE_RE = re.compile(
    r"seed=(?P<seed>-?\d+)\s+suite=(?P<suite>\S+)\s+"
    r"task=(?P<task>\d+)\s+episode=(?P<episode>\d+)\s+"
    r"success=(?P<success>True|False)"
)


def _load_eval_module():
    """Import the standard evaluator only after this script has parsed args."""

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import eval_libero

    return eval_libero


def _parse_log(path: Path, seed: int) -> dict[tuple[str, int], dict[int, bool]]:
    records: dict[tuple[str, int], dict[int, bool]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Worker log not found: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = EPISODE_RE.search(line)
        if match is None:
            continue
        record_seed = int(match.group("seed"))
        if record_seed != seed:
            continue
        key = (match.group("suite"), int(match.group("task")))
        episode = int(match.group("episode"))
        success = match.group("success") == "True"
        task_records = records.setdefault(key, {})
        previous = task_records.get(episode)
        if previous is not None and previous != success:
            raise ValueError(
                f"Conflicting results in {path}:{line_number} for {key} episode {episode}: "
                f"{previous} vs {success}"
            )
        task_records[episode] = success
    return records


def _worker_log_path(seed_dir: Path, rank: int, gpu: int) -> Path:
    requested = seed_dir / "logs" / f"worker_{rank}_gpu_{gpu}.log"
    if requested.is_file():
        return requested
    existing = sorted((seed_dir / "logs").glob(f"worker_{rank}_gpu_*.log"))
    if len(existing) == 1:
        return existing[0]
    if not existing:
        return requested
    raise ValueError(f"Multiple existing logs found for worker rank {rank}: {existing}")


def _task_description(benchmark: Any, suite_name: str, task_id: int) -> str:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    return str(suite.get_task(task_id).language)


def _build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    evaluator = _load_eval_module()
    fastwam_root = Path(args.fastwam_source_root).expanduser().resolve()
    libero_root = Path(args.libero_root).expanduser().resolve()
    seed_dir = Path(args.run_root).expanduser().resolve() / f"seed{args.seed}"
    evaluator._add_project_paths(fastwam_root, libero_root)
    evaluator._configure_libero_paths(libero_root, seed_dir)

    suites = evaluator._parse_suites(args.task_suites)
    all_tasks = evaluator._task_list(suites, args.max_tasks)
    assigned = all_tasks[args.rank :: args.world_size]
    log_path = _worker_log_path(seed_dir, args.rank, args.gpu)
    observed = _parse_log(log_path, args.seed)

    from libero.libero import benchmark

    available = benchmark.get_benchmark_dict()
    plan: list[dict[str, Any]] = []
    for suite_name, task_id in assigned:
        key = (suite_name, int(task_id))
        episodes = observed.get(key, {})
        invalid = sorted(index for index in episodes if index < 0 or index >= args.num_trials)
        if invalid:
            raise ValueError(f"Invalid episode indices for {key} in {log_path}: {invalid}")
        pending = [index for index in range(args.num_trials) if index not in episodes]
        suite = available[suite_name]()
        task = suite.get_task(task_id)
        plan.append(
            {
                "suite": suite_name,
                "task_id": int(task_id),
                "task_description": str(task.language),
                "observed": {str(index): bool(success) for index, success in sorted(episodes.items())},
                "pending": pending,
            }
        )
    return plan


def _print_plan(args: argparse.Namespace, plan: list[dict[str, Any]]) -> None:
    print(
        json.dumps(
            {
                "seed": args.seed,
                "rank": args.rank,
                "gpu": args.gpu,
                "world_size": args.world_size,
                "tasks": [
                    {
                        "suite": item["suite"],
                        "task_id": item["task_id"],
                        "completed": len(item["observed"]),
                        "remaining": len(item["pending"]),
                        "first_remaining_episode": item["pending"][0] if item["pending"] else None,
                    }
                    for item in plan
                ],
                "remaining_episodes": sum(len(item["pending"]) for item in plan),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _configure_worker_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("fastwam_libero")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _merge_task_result(
    item: dict[str, Any],
    resumed: dict[str, Any] | None,
    *,
    num_trials: int,
) -> dict[str, Any]:
    observed = {int(index): bool(success) for index, success in item["observed"].items()}
    if resumed is not None:
        for index in resumed.get("success_episodes", []):
            if int(index) in observed:
                raise ValueError(f"Resumed episode overlaps an existing episode: {item['suite']}/{item['task_id']}/{index}")
            observed[int(index)] = True
        for index in resumed.get("failure_episodes", []):
            if int(index) in observed:
                raise ValueError(f"Resumed episode overlaps an existing episode: {item['suite']}/{item['task_id']}/{index}")
            observed[int(index)] = False
    missing = sorted(set(range(num_trials)) - set(observed))
    if missing:
        raise ValueError(f"Task {item['suite']}/{item['task_id']} still has missing episodes: {missing}")
    success_episodes = sorted(index for index, success in observed.items() if success)
    failure_episodes = sorted(index for index, success in observed.items() if not success)
    return {
        "suite": item["suite"],
        "task_id": int(item["task_id"]),
        "task_description": item["task_description"],
        "successes": len(success_episodes),
        "episodes": int(num_trials),
        "success_episodes": success_episodes,
        "failure_episodes": failure_episodes,
    }


def _run_rank(args: argparse.Namespace, plan: list[dict[str, Any]]) -> None:
    evaluator = _load_eval_module()
    seed_dir = Path(args.run_root).expanduser().resolve() / f"seed{args.seed}"
    log_path = _worker_log_path(seed_dir, args.rank, args.gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = _configure_worker_logging(log_path)
    log.info(
        "=== RESUME START seed=%s rank=%s/%s gpu=%s remaining_episodes=%s ===",
        args.seed,
        args.rank,
        args.world_size,
        args.gpu,
        sum(len(item["pending"]) for item in plan),
    )

    fastwam_root = Path(args.fastwam_source_root).expanduser().resolve()
    libero_root = Path(args.libero_root).expanduser().resolve()
    evaluator._add_project_paths(fastwam_root, libero_root)
    evaluator._configure_libero_paths(libero_root, seed_dir)
    evaluator._set_seed(args.seed)

    import torch
    from libero.libero import benchmark

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(0)
    model_dir = Path(args.model_path).expanduser().resolve()
    model, model_config, dtype = evaluator._build_model(model_dir, device)
    state_stats, action_stats = evaluator._load_normalizers(model_dir)
    available = benchmark.get_benchmark_dict()
    started = time.time()
    task_results: list[dict[str, Any]] = []

    for item in plan:
        resumed_result = None
        if item["pending"]:
            suite = available[item["suite"]]()
            task = suite.get_task(int(item["task_id"]))
            initial_states = suite.get_task_init_states(int(item["task_id"]))
            log.info(
                "Resuming seed=%s suite=%s task=%s episodes=%s",
                args.seed,
                item["suite"],
                item["task_id"],
                item["pending"],
            )
            resumed_result = evaluator._run_task(
                model,
                task,
                initial_states,
                suite_name=item["suite"],
                task_id=int(item["task_id"]),
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
                episode_indices=item["pending"],
            )
        task_results.append(_merge_task_result(item, resumed_result, num_trials=args.num_trials))

    successes = sum(int(item["successes"]) for item in task_results)
    episodes = sum(int(item["episodes"]) for item in task_results)
    result = {
        "format": "fastwam-libero-worker-v1",
        "seed": int(args.seed),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "gpu_id": str(args.gpu),
        "model_path": str(model_dir),
        "task_suites": list(args.task_suites),
        "num_trials_per_task": int(args.num_trials),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "duration_s": time.time() - started,
        "tasks": task_results,
        "resumed": True,
    }
    output = seed_dir / f"worker_{args.rank:03d}.json"
    evaluator._atomic_json_dump(output, result)
    log.info(
        "=== RESUME COMPLETE wrote=%s successes=%s/%s rate=%.4f ===",
        output,
        successes,
        episodes,
        result["success_rate"],
    )


def _child_args(args: argparse.Namespace, rank: int, gpu: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--run-root",
        str(args.run_root),
        "--model-path",
        str(args.model_path),
        "--libero-root",
        str(args.libero_root),
        "--fastwam-source-root",
        str(args.fastwam_source_root),
        "--seed",
        str(args.seed),
        "--rank",
        str(rank),
        "--gpu",
        str(gpu),
        "--world-size",
        str(args.world_size),
        "--num-trials",
        str(args.num_trials),
        "--num-steps-wait",
        str(args.num_steps_wait),
        "--replan-steps",
        str(args.replan_steps),
        "--action-horizon",
        str(args.action_horizon),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--max-steps",
        str(args.max_steps),
        "--max-tasks",
        str(args.max_tasks),
        "--task-suites",
        *args.task_suites,
    ]
    if not args.binarize_gripper:
        command.append("--no-binarize-gripper")
    return command


def _refresh_summary_files(args: argparse.Namespace) -> None:
    evaluator = _load_eval_module()
    run_root = Path(args.run_root).expanduser().resolve()
    seed_dir = run_root / f"seed{args.seed}"
    evaluator.aggregate_seed(seed_dir, expected_workers=args.world_size, seed=args.seed)
    evaluator.aggregate_three_seed(run_root, [1, 7, 42])
    summarizer = HERE / "summarize_four_task_results.py"
    if summarizer.is_file():
        subprocess.run(
            [
                sys.executable,
                str(summarizer),
                "--run-root",
                str(run_root),
                "--seeds",
                "1",
                "7",
                "42",
            ],
            check=True,
        )
    print(f"Resume complete; updated {seed_dir / 'summary.json'} and three-seed summaries", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--fastwam-source-root", type=Path, default=DEFAULT_FASTWAM_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(DEFAULT_GPUS))
    parser.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--task-suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--num-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--no-binarize-gripper", dest="binarize_gripper", action="store_false")
    parser.set_defaults(binarize_gripper=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the resume plan without starting workers")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild seed42 and three-seed summaries from existing worker JSON files",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing worker JSON")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    # Match run_fastwam_libero.sh's headless renderer.  Without this, a
    # resumed process may select EGL even though the original workers used
    # OSMesa successfully.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    if len(args.ranks) != len(args.gpus):
        parser.error("--ranks and --gpus must have the same length")
    if any(rank < 0 or rank >= args.world_size for rank in args.ranks):
        parser.error(f"Ranks must be in [0, {args.world_size})")
    if args.num_trials <= 0:
        parser.error("--num-trials must be positive")

    if args.aggregate_only:
        _refresh_summary_files(args)
        return

    if args.child:
        if args.rank < 0 or args.gpu < 0:
            parser.error("child mode requires --rank and --gpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        plan = _build_plan(args)
        _run_rank(args, plan)
        return

    plans = []
    for rank, gpu in zip(args.ranks, args.gpus, strict=True):
        child_args = argparse.Namespace(**vars(args))
        child_args.rank = rank
        child_args.gpu = gpu
        plan = _build_plan(child_args)
        plans.append((rank, gpu, plan))
        _print_plan(child_args, plan)

    # A dry run is also useful after a resume has completed: it should report
    # zero missing episodes instead of failing merely because the final worker
    # JSON files now exist.
    if args.dry_run:
        return

    for rank, _gpu, _plan in plans:
        worker_path = Path(args.run_root).expanduser().resolve() / f"seed{args.seed}" / f"worker_{rank:03d}.json"
        if worker_path.is_file() and not args.force:
            parser.error(f"Worker output already exists: {worker_path}; use --force to overwrite")

    processes: list[tuple[int, int, subprocess.Popen[Any]]] = []
    for rank, gpu, _plan in plans:
        child = subprocess.Popen(
            _child_args(args, rank, gpu),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )
        processes.append((rank, gpu, child))

    failed = []
    for rank, gpu, process in processes:
        status = process.wait()
        print(f"resume worker rank={rank} gpu={gpu} exit={status}", flush=True)
        if status != 0:
            failed.append((rank, status))
    if failed:
        raise SystemExit(f"Resume failed: {failed}")

    _refresh_summary_files(args)


if __name__ == "__main__":
    main()
