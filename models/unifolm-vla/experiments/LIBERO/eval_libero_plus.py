#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import logging
import os
import pathlib
import re
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=r".*Overriding environment.*already in registry.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium\.envs\.registration")

from experiments.LIBERO.eval_libero import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    get_action_state,
    prepare_observation,
    process_action,
)
from experiments.LIBERO.unifolm_vla_inference import Unifolm_VLA_Inference  # noqa: E402


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
DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
UNNORM_KEY_BY_SUITE = {
    suite: f"{suite}_no_noops" for suite in MAX_STEPS_BY_SUITE
}

benchmark = None


@dataclasses.dataclass(frozen=True)
class LiberoPlusTask:
    suite: str
    task_id: int
    task_name: str
    category: str
    difficulty_level: int | None
    classification_id: int | None


def _rate(successes: int, total: int) -> float:
    return 100.0 * successes / total if total else 0.0


def _import_libero(libero_plus_root: Path | None) -> None:
    global benchmark
    if libero_plus_root is not None:
        libero_plus_root = libero_plus_root.resolve()
        if str(libero_plus_root) not in sys.path:
            sys.path.insert(0, str(libero_plus_root))

    warnings.filterwarnings("ignore", message=r".*Overriding environment.*already in registry.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium\.envs\.registration")

    from libero.libero import benchmark as benchmark_module

    benchmark = benchmark_module


def _make_suite(suite_name: str):
    assert benchmark is not None
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise KeyError(f"Unknown suite {suite_name!r}. Available: {sorted(benchmark_dict)}")
    with contextlib.redirect_stdout(io.StringIO()):
        return benchmark_dict[suite_name]()


def _load_classification_tasks(
    classification_path: Path,
    task_suites: list[str],
    suite_objects: dict[str, Any],
) -> list[LiberoPlusTask]:
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)

    tasks: list[LiberoPlusTask] = []
    for suite_name in task_suites:
        if suite_name not in classification:
            raise KeyError(f"Suite {suite_name!r} not found in {classification_path}. Available: {sorted(classification)}")

        suite = suite_objects[suite_name]
        name_to_task_id = {suite.get_task(i).name: i for i in range(suite.n_tasks)}
        for item in classification[suite_name]:
            task_name = item["name"]
            task_id = name_to_task_id.get(task_name)
            if task_id is None:
                candidate = int(item["id"]) - 1 if item.get("id") is not None else -1
                if not (0 <= candidate < suite.n_tasks and suite.get_task(candidate).name == task_name):
                    raise KeyError(f"Task {task_name!r} from {classification_path} is not in {suite_name} benchmark.")
                task_id = candidate

            tasks.append(
                LiberoPlusTask(
                    suite=suite_name,
                    task_id=task_id,
                    task_name=task_name,
                    category=item.get("category", ""),
                    difficulty_level=item.get("difficulty_level"),
                    classification_id=item.get("id"),
                )
            )
    return tasks


def _safe_path_segment(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "task"


def _unnorm_key_for_task(task_info: LiberoPlusTask, unnorm_key: str) -> str:
    if unnorm_key.lower() == "auto":
        return UNNORM_KEY_BY_SUITE[task_info.suite]
    return unnorm_key


def _initial_unnorm_key(assigned_tasks: list[LiberoPlusTask], task_suites: list[str], unnorm_key: str) -> str:
    if unnorm_key.lower() != "auto":
        return unnorm_key
    if assigned_tasks:
        return _unnorm_key_for_task(assigned_tasks[0], unnorm_key)
    return UNNORM_KEY_BY_SUITE[task_suites[0]]


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_failure_video(video_out_path: Path, task_info: LiberoPlusTask, episode_idx: int, replay_images: list[np.ndarray]) -> str | None:
    if not replay_images:
        return None
    task_dir = video_out_path / task_info.suite / _safe_path_segment(task_info.category)
    task_dir.mkdir(parents=True, exist_ok=True)
    video_path = task_dir / f"{task_info.task_id:05d}_{_safe_path_segment(task_info.task_name, 96)}_ep{episode_idx}_failure.mp4"
    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
    return str(video_path)


def _run_episode(
    *,
    suite_objects: dict[str, Any],
    task_info: LiberoPlusTask,
    episode_idx: int,
    model: Unifolm_VLA_Inference,
    args: argparse.Namespace,
) -> tuple[bool, int, str | None, str | None]:
    env = None
    replay_images: list[np.ndarray] = []
    t = 0
    try:
        suite = suite_objects[task_info.suite]
        task = suite.get_task(task_info.task_id)
        initial_states = suite.get_task_init_states(task_info.task_id)
        state_index = episode_idx % len(initial_states)
        max_steps = args.max_steps if args.max_steps > 0 else MAX_STEPS_BY_SUITE[task_info.suite]

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + task_info.task_id * 1000 + episode_idx)
        model.reset(task_description=task_description)
        env.reset()
        obs = env.set_init_state(initial_states[state_index])

        action_queue = deque()
        obs_queue = deque(maxlen=args.window_size)
        success = False

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            while len(obs_queue) < args.window_size:
                observation, img = prepare_observation(obs, resize_size=224)
                obs_queue.append(observation)

            if args.save_failure_video:
                replay_images.append(img)

            if not action_queue:
                actions = get_action_state(obs_queue, task_description, model)
                action_queue.extend(actions)
            obs_queue.popleft()

            action = process_action(action_queue.popleft())
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

        video_path = None
        if args.save_failure_video and not success and args.video_out_path is not None:
            video_path = _save_failure_video(Path(args.video_out_path), task_info, episode_idx, replay_images)
        return success, max(0, t - args.num_steps_wait), video_path, None
    except Exception as exc:
        logging.exception("Episode failed: suite=%s task=%s episode=%s", task_info.suite, task_info.task_name, episode_idx)
        return False, max(0, t - args.num_steps_wait), None, repr(exc)
    finally:
        if env is not None:
            env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UnifoLM-VLA on LIBERO-plus tasks.")
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--vlm-pretrained-path", required=True)
    parser.add_argument("--unnorm-key", default="auto", help="Use 'auto' to select the matching <suite>_no_noops stats per task.")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--task-suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--classification-path", required=True, type=Path)
    parser.add_argument("--result-jsonl", required=True, type=Path)
    parser.add_argument("--libero-plus-root", type=Path, default=None)
    parser.add_argument("--eval-rank", type=int, default=0)
    parser.add_argument("--eval-world-size", type=int, default=1)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--video-out-path", type=Path, default=None)
    parser.add_argument("--save-failure-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only-first-task", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.eval_world_size < 1:
        raise ValueError("--eval-world-size must be >= 1")
    if not (0 <= args.eval_rank < args.eval_world_size):
        raise ValueError("--eval-rank must be in [0, eval_world_size)")
    if args.num_trials_per_task < 1:
        raise ValueError("--num-trials-per-task must be >= 1")

    _import_libero(args.libero_plus_root)
    np.random.seed(args.seed + args.eval_rank)

    suite_objects = {suite_name: _make_suite(suite_name) for suite_name in args.task_suites}
    all_tasks = _load_classification_tasks(args.classification_path, args.task_suites, suite_objects)
    if args.only_first_task:
        all_tasks = all_tasks[:1]
    if args.limit_tasks > 0:
        all_tasks = all_tasks[: args.limit_tasks]
    assigned_tasks = all_tasks[args.eval_rank :: args.eval_world_size]

    args.result_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.result_jsonl.write_text("", encoding="utf-8")
    if args.save_failure_video and args.video_out_path is not None:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] rank={args.eval_rank}/{args.eval_world_size} tasks={len(assigned_tasks)}/{len(all_tasks)} "
        f"episodes={len(assigned_tasks) * args.num_trials_per_task}",
        flush=True,
    )

    initial_unnorm_key = _initial_unnorm_key(assigned_tasks, args.task_suites, args.unnorm_key)
    print(f"[INFO] initial unnorm_key={initial_unnorm_key} requested={args.unnorm_key}", flush=True)

    model = Unifolm_VLA_Inference(
        policy_ckpt_path=args.pretrained_path,
        image_size=[224, 224],
        unnorm_key=initial_unnorm_key,
        vlm_pretrained_path=args.vlm_pretrained_path,
    )

    total_successes = 0
    total_episodes = 0
    total_episode_target = len(assigned_tasks) * args.num_trials_per_task
    progress = tqdm.tqdm(total=total_episode_target, desc=f"Episodes rank {args.eval_rank}", dynamic_ncols=True)
    try:
        for task_index, task_info in enumerate(assigned_tasks):
            task_unnorm_key = _unnorm_key_for_task(task_info, args.unnorm_key)
            model.set_unnorm_key(task_unnorm_key, policy_ckpt_path=args.pretrained_path)
            task_successes = 0
            for episode_idx in range(args.num_trials_per_task):
                success, episode_length, video_path, error = _run_episode(
                    suite_objects=suite_objects,
                    task_info=task_info,
                    episode_idx=episode_idx,
                    model=model,
                    args=args,
                )
                total_episodes += 1
                total_successes += int(success)
                task_successes += int(success)

                record = {
                    "suite": task_info.suite,
                    "task_id": task_info.task_id,
                    "classification_id": task_info.classification_id,
                    "task_name": task_info.task_name,
                    "category": task_info.category,
                    "column": CATEGORY_TO_COLUMN.get(task_info.category),
                    "difficulty_level": task_info.difficulty_level,
                    "unnorm_key": task_unnorm_key,
                    "rank": args.eval_rank,
                    "episode_idx": episode_idx,
                    "success": bool(success),
                    "episode_length": int(episode_length),
                    "video_path": video_path,
                    "error": error,
                }
                _write_jsonl(args.result_jsonl, record)

                progress.update(1)
                progress.set_postfix(success=f"{total_successes}/{total_episodes}", rate=f"{_rate(total_successes, total_episodes):.1f}%")

            print(
                f"[RESULT] rank={args.eval_rank} ({task_index + 1}/{len(assigned_tasks)}) "
                f"suite={task_info.suite} category={task_info.category} task={task_info.task_name} "
                f"success={task_successes}/{args.num_trials_per_task} overall={total_successes}/{total_episodes} "
                f"rate={_rate(total_successes, total_episodes):.1f}",
                flush=True,
            )
    finally:
        progress.close()

    print(
        f"[DONE] rank={args.eval_rank} success={total_successes}/{total_episodes} "
        f"rate={_rate(total_successes, total_episodes):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
