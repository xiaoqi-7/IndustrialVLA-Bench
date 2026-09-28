#!/usr/bin/env python3
"""FastWAM evaluator for the LIBERO-Para paraphrase benchmark.

LIBERO-Para contains one BDDL file per paraphrased instruction.  The files are
grouped by ``eval0`` ... ``eval9`` and share the ten original LIBERO-Goal
scenes.  This evaluator scans those files directly (rather than importing the
large 4k-entry LIBERO benchmark map), uses the matching original scene for the
environment, and feeds the paraphrased BDDL language to FastWAM.

The model/checkpoint compatibility loader and image/action preprocessing are
shared with ``eval_libero.py``.  A worker loads the policy once and evaluates a
round-robin task shard on one visible GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the tested FastWAM checkpoint loader and LIBERO observation pipeline.
from eval_libero import (  # noqa: E402
    _add_project_paths,
    _atomic_json_dump,
    _build_model,
    _configure_libero_paths,
    _dummy_action,
    _env_step,
    _json_default,
    _load_normalizers,
    _predict_action,
    _set_seed,
)

LOG = logging.getLogger("fastwam_libero_para")

SEEDS = (1, 7, 42)
DEFAULT_MODEL_PATH = Path("/mnt/afs/zhengmingkai/raozf/models/fastwam_libero")
DEFAULT_PARA_ROOT = Path("/mnt/afs/zhengmingkai/raozf/benchmark/LIBERO-Para")
DEFAULT_FASTWAM_ROOT = HERE
DEFAULT_MAX_STEPS = 300
DEFAULT_RESOLUTION = 360
KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}


@dataclass(frozen=True)
class ParaTask:
    """One paraphrased instruction/task file."""

    index: int
    path: Path
    instruction: str
    eval_id: int
    variant_id: int
    paraphrase_type: str
    categories: tuple[str, ...]
    subcategories: tuple[str, ...]


def _read_bddl_language(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("(:language"):
            return stripped[len("(:language") :].rstrip(")").strip()
    raise ValueError(f"No (:language ...) entry found in {path}")


def _parse_filename_metadata(path: Path) -> tuple[int, int, str, tuple[str, ...], tuple[str, ...]]:
    stem = path.stem
    match = re.search(r"_eval(\d+)_ver(\d+)$", stem)
    if match is None:
        raise ValueError(f"LIBERO-Para filename does not contain _evalN_verM: {path.name}")
    eval_id = int(match.group(1))
    variant_id = int(match.group(2))
    prefix = stem[: match.start()]

    if prefix.startswith("act_"):
        paraphrase_type = "act"
        body = prefix[len("act_") :]
    elif prefix.startswith("obj_"):
        paraphrase_type = "obj"
        body = prefix[len("obj_") :]
    elif prefix.startswith("comp_"):
        paraphrase_type = "comp"
        body = prefix[len("comp_") :]
    else:
        paraphrase_type = "unknown"
        body = prefix

    categories: list[str] = []
    subcategories: list[str] = []
    if paraphrase_type in {"act", "obj"}:
        for category in KNOWN_CATEGORIES:
            marker = category + "_"
            if body.startswith(marker):
                categories = [category]
                subcategories = [body[len(marker) :]]
                break
        if not categories:
            categories = [body]
    elif paraphrase_type == "comp":
        # A compositional name has the form
        # comp_<cat1>+<cat2>_<subcat1>+<subcat2>.
        first_plus = body.find("+")
        if first_plus >= 0:
            left = body[:first_plus]
            remainder = body[first_plus + 1 :]
            # The second category is explicitly one of the known category
            # names, so split there before separating its two subcategories.
            second_category = next(
                (category for category in KNOWN_CATEGORIES if remainder.startswith(category + "_")),
                None,
            )
            if second_category is not None:
                after_category = remainder[len(second_category) + 1 :]
                subcat1, subcat2 = after_category.rsplit("+", 1)
                categories = [left, second_category]
                subcategories = [subcat1, subcat2]
            else:
                categories = [left, remainder]
        else:
            categories = [body]

    return eval_id, variant_id, paraphrase_type, tuple(categories), tuple(subcategories)


def _normalised_bddl_structure(path: Path) -> str:
    """Ignore the language line when matching Para files to Goal scenes."""

    text = path.read_text(encoding="utf-8")
    return re.sub(r"\(:language\s+.*?\)", "(:language <instruction>)", text, count=1, flags=re.S)


def _resolve_scene_bddls(para_root: Path, tasks: list[ParaTask]) -> dict[int, Path]:
    """Map eval IDs to original LIBERO-Goal BDDL files.

    The repository currently orders these ten files by eval ID, but matching
    normalized BDDL content is safer than relying on filename order.
    """

    goal_dir = para_root / "libero" / "libero" / "bddl_files" / "libero_goal"
    goal_files = sorted(goal_dir.glob("*.bddl"))
    if not goal_files:
        raise FileNotFoundError(f"No LIBERO-Goal BDDL files found in {goal_dir}")

    first_by_eval: dict[int, Path] = {}
    for task in tasks:
        first_by_eval.setdefault(task.eval_id, task.path)

    result: dict[int, Path] = {}
    normalized_goals = {path: _normalised_bddl_structure(path) for path in goal_files}
    for eval_id, para_path in first_by_eval.items():
        normalized_para = _normalised_bddl_structure(para_path)
        matches = [path for path, text in normalized_goals.items() if text == normalized_para]
        if len(matches) == 1:
            result[eval_id] = matches[0]
        elif eval_id < len(goal_files):
            # Keep compatibility with the upstream repository if a future
            # release changes whitespace in one BDDL file.
            result[eval_id] = goal_files[eval_id]
        else:
            raise ValueError(f"Could not map eval{eval_id} to a Goal BDDL scene")
    return result


def _scan_tasks(para_root: Path, max_tasks: int = -1) -> tuple[list[ParaTask], dict[int, Path]]:
    bddl_dir = para_root / "libero" / "libero" / "bddl_files" / "libero_para"
    init_dir = para_root / "libero" / "libero" / "init_files" / "libero_para"
    if not bddl_dir.is_dir():
        raise FileNotFoundError(f"LIBERO-Para BDDL directory not found: {bddl_dir}")
    if not init_dir.is_dir():
        raise FileNotFoundError(f"LIBERO-Para init directory not found: {init_dir}")

    files = sorted(bddl_dir.glob("*.bddl"))
    if max_tasks >= 0:
        files = files[:max_tasks]
    if not files:
        raise ValueError(f"No LIBERO-Para BDDL files found in {bddl_dir}")

    tasks: list[ParaTask] = []
    for index, path in enumerate(files):
        eval_id, variant_id, para_type, categories, subcategories = _parse_filename_metadata(path)
        init_path = init_dir / f"eval{eval_id}.pruned_init"
        if not init_path.is_file():
            raise FileNotFoundError(f"Missing fixed initial states for eval{eval_id}: {init_path}")
        tasks.append(
            ParaTask(
                index=index,
                path=path,
                instruction=_read_bddl_language(path),
                eval_id=eval_id,
                variant_id=variant_id,
                paraphrase_type=para_type,
                categories=categories,
                subcategories=subcategories,
            )
        )
    return tasks, _resolve_scene_bddls(para_root, tasks)


def _load_para_initial_states(para_root: Path, eval_ids: Iterable[int]) -> dict[int, Any]:
    import torch

    result: dict[int, Any] = {}
    for eval_id in sorted(set(int(value) for value in eval_ids)):
        path = para_root / "libero" / "libero" / "init_files" / "libero_para" / f"eval{eval_id}.pruned_init"
        try:
            states = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:  # torch versions before the weights_only argument
            states = torch.load(str(path), map_location="cpu")
        if len(states) == 0:
            raise ValueError(f"No initial states in {path}")
        result[eval_id] = states
    return result


class ParaEnvPool:
    """Lazily create one reusable environment for each original eval scene."""

    def __init__(self, scene_bddls: dict[int, Path], seed: int, resolution: int):
        from libero.libero.envs import OffScreenRenderEnv

        self._offscreen_env = OffScreenRenderEnv
        self._scene_bddls = scene_bddls
        self._seed = int(seed)
        self._resolution = int(resolution)
        self._envs: dict[int, Any] = {}

    def get(self, eval_id: int):
        eval_id = int(eval_id)
        if eval_id not in self._envs:
            path = self._scene_bddls[eval_id]
            env = self._offscreen_env(
                bddl_file_name=path,
                camera_heights=self._resolution,
                camera_widths=self._resolution,
            )
            env.seed(self._seed)
            self._envs[eval_id] = env
        return self._envs[eval_id]

    def close(self) -> None:
        for env in self._envs.values():
            close = getattr(env, "close", None)
            if close is not None:
                close()
        self._envs.clear()


def _check_success(env: Any) -> bool:
    checker = getattr(env, "check_success", None)
    return bool(checker()) if checker is not None else False


def _run_task(
    model: Any,
    task: ParaTask,
    env: Any,
    initial_states: Any,
    *,
    seed: int,
    num_trials: int,
    num_steps_wait: int,
    replan_steps: int,
    action_horizon: int,
    num_inference_steps: int,
    state_stats: Any,
    action_stats: Any,
    device: str,
    dtype: Any,
    binarize_gripper: bool,
    max_steps: int,
) -> dict[str, Any]:
    successes: list[int] = []
    failures: list[int] = []
    for episode_idx in range(num_trials):
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
        pending: list[list[float]] = []
        success = _check_success(env)
        done = False
        t = 0
        while not success and not done and t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = _env_step(env, _dummy_action())
                success = _check_success(env)
                t += 1
                continue

            if not pending:
                chunk = _predict_action(
                    model,
                    obs,
                    task.instruction,
                    state_stats,
                    action_stats,
                    seed=seed,
                    action_horizon=action_horizon,
                    num_inference_steps=num_inference_steps,
                    device=device,
                    dtype=dtype,
                    binarize_gripper=binarize_gripper,
                )
                pending = chunk[:replan_steps].tolist()

            obs, _, done, _ = _env_step(env, pending.pop(0))
            success = _check_success(env)
            t += 1

        if success:
            successes.append(episode_idx)
        else:
            failures.append(episode_idx)
        LOG.info(
            "seed=%s task=%s eval=%s variant=%s episode=%s success=%s",
            seed,
            task.index,
            task.eval_id,
            task.variant_id,
            episode_idx,
            success,
        )

    return {
        "task_index": int(task.index),
        "bddl_file": task.path.name,
        "instruction": task.instruction,
        "eval_id": int(task.eval_id),
        "variant_id": int(task.variant_id),
        "paraphrase_type": task.paraphrase_type,
        "categories": list(task.categories),
        "subcategories": list(task.subcategories),
        "successes": len(successes),
        "episodes": int(num_trials),
        "success_episodes": successes,
        "failure_episodes": failures,
    }


def evaluate_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_seed(args.seed)
    para_root = Path(args.libero_para_root).expanduser().resolve()
    tasks, scene_bddls = _scan_tasks(para_root, args.max_tasks)
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank/world_size must satisfy 0 <= rank < world_size; got {args.rank}/{args.world_size}")
    assigned = tasks[args.rank :: args.world_size]
    if not assigned:
        raise ValueError(f"Worker rank {args.rank} received no LIBERO-Para tasks")

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

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
    initial_states = _load_para_initial_states(para_root, (task.eval_id for task in assigned))
    env_pool = ParaEnvPool(
        {eval_id: scene_bddls[eval_id] for eval_id in initial_states},
        seed=args.seed,
        resolution=args.resolution,
    )

    task_results: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task in assigned:
            task_results.append(
                _run_task(
                    model,
                    task,
                    env_pool.get(task.eval_id),
                    initial_states[task.eval_id],
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
                    max_steps=args.max_steps,
                )
            )
    finally:
        env_pool.close()

    successes = sum(int(item["successes"]) for item in task_results)
    episodes = sum(int(item["episodes"]) for item in task_results)
    result = {
        "format": "fastwam-libero-para-worker-v1",
        "seed": int(args.seed),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "gpu_id": str(args.gpu_id),
        "model_path": str(model_dir),
        "libero_para_root": str(para_root),
        "num_tasks_total": len(tasks),
        "num_tasks_assigned": len(assigned),
        "num_trials_per_task": int(args.num_trials),
        "max_steps": int(args.max_steps),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "duration_s": time.time() - started,
        "tasks": task_results,
    }
    output = Path(args.output_dir) / f"worker_{args.rank:03d}.json"
    _atomic_json_dump(output, result)
    LOG.info("Wrote %s: %s/%s (%.4f)", output, successes, episodes, result["success_rate"])
    return result


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
    tasks.sort(key=lambda item: int(item["task_index"]))
    successes = sum(int(item["successes"]) for item in tasks)
    episodes = sum(int(item["episodes"]) for item in tasks)

    by_eval: dict[str, dict[str, Any]] = {}
    by_type: dict[str, dict[str, Any]] = {}
    for item in tasks:
        eval_key = str(item["eval_id"])
        eval_entry = by_eval.setdefault(eval_key, {"successes": 0, "episodes": 0, "tasks": 0})
        eval_entry["successes"] += int(item["successes"])
        eval_entry["episodes"] += int(item["episodes"])
        eval_entry["tasks"] += 1
        type_key = str(item.get("paraphrase_type", "unknown"))
        type_entry = by_type.setdefault(type_key, {"successes": 0, "episodes": 0, "tasks": 0})
        type_entry["successes"] += int(item["successes"])
        type_entry["episodes"] += int(item["episodes"])
        type_entry["tasks"] += 1
    for entry in [*by_eval.values(), *by_type.values()]:
        entry["success_rate"] = entry["successes"] / entry["episodes"] if entry["episodes"] else 0.0

    result = {
        "format": "fastwam-libero-para-seed-v1",
        "seed": int(seed),
        "workers": int(expected_workers),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "by_eval_id": by_eval,
        "by_paraphrase_type": by_type,
        "tasks": tasks,
    }
    _atomic_json_dump(output_dir / "summary.json", result)
    return result


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

    rates = [float(result["success_rate"]) for result in seed_results]
    pooled_successes = sum(int(result["successes"]) for result in seed_results)
    pooled_episodes = sum(int(result["episodes"]) for result in seed_results)

    def aggregate_group(key: str) -> dict[str, Any]:
        names = sorted({name for result in seed_results for name in result.get(key, {})})
        output: dict[str, Any] = {}
        for name in names:
            entries = [result[key][name] for result in seed_results if name in result.get(key, {})]
            entry_rates = [float(entry["success_rate"]) for entry in entries]
            output[str(name)] = {
                "seed_rates": {
                    str(result["seed"]): float(result[key][name]["success_rate"])
                    for result in seed_results
                    if name in result.get(key, {})
                },
                "mean_success_rate": sum(entry_rates) / len(entry_rates),
                "pooled_successes": sum(int(entry["successes"]) for entry in entries),
                "pooled_episodes": sum(int(entry["episodes"]) for entry in entries),
            }
        return output

    result = {
        "format": "fastwam-libero-para-three-seed-v1",
        "seeds": seed_list,
        "seed_results": seed_results,
        "mean_success_rate": sum(rates) / len(rates),
        "pooled_successes": pooled_successes,
        "pooled_episodes": pooled_episodes,
        "pooled_success_rate": pooled_successes / pooled_episodes if pooled_episodes else 0.0,
        "by_eval_id": aggregate_group("by_eval_id"),
        "by_paraphrase_type": aggregate_group("by_paraphrase_type"),
    }
    _atomic_json_dump(output_root / "three_seed_summary.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--libero-para-root", default=str(DEFAULT_PARA_ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-binarize-gripper", dest="binarize_gripper", action="store_false")
    parser.set_defaults(binarize_gripper=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-workers", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-workers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--aggregate-three-seed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS), help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    para_root = Path(args.libero_para_root).expanduser().resolve()
    fastwam_root = Path(os.environ.get("FASTWAM_SOURCE_ROOT", str(DEFAULT_FASTWAM_ROOT))).expanduser()
    _add_project_paths(fastwam_root, para_root)
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _configure_libero_paths(para_root, Path(args.output_dir).expanduser().resolve())

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
    if args.num_steps_wait < 0 or args.replan_steps <= 0 or args.max_steps <= 0:
        raise ValueError("wait must be non-negative; replan and max steps must be positive")
    tasks, scene_bddls = _scan_tasks(para_root, args.max_tasks)
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
                    "eval_ids": sorted({task.eval_id for task in tasks}),
                    "scene_bddls": {str(key): str(value) for key, value in scene_bddls.items()},
                    "model_path": str(model_path),
                    "libero_para_root": str(para_root),
                },
                separators=(",", ":"),
            )
        )
        return
    evaluate_worker(args)


if __name__ == "__main__":
    main()
