#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import re
import socket
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

XIAOMI_ROOT = Path(__file__).resolve().parents[1]
if str(XIAOMI_ROOT) not in sys.path:
    sys.path.insert(0, str(XIAOMI_ROOT))

from deploy.client import Client


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
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

benchmark = None
get_libero_path = None
OffScreenRenderEnv = None


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
    global benchmark, get_libero_path, OffScreenRenderEnv
    if libero_plus_root is not None:
        libero_plus_root = libero_plus_root.resolve()
        if str(libero_plus_root) not in sys.path:
            sys.path.insert(0, str(libero_plus_root))

    warnings.filterwarnings("ignore", message=r".*Overriding environment.*already in registry.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium\.envs\.registration")

    from libero.libero import benchmark as benchmark_module
    from libero.libero import get_libero_path as libero_path_getter
    from libero.libero.envs import OffScreenRenderEnv as LiberoOffScreenRenderEnv

    benchmark = benchmark_module
    get_libero_path = libero_path_getter
    OffScreenRenderEnv = LiberoOffScreenRenderEnv


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__type__": "numpy", "dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tobytes().hex()}
    if isinstance(obj, Image.Image):
        img_hash = hashlib.md5(obj.tobytes()).hexdigest()
        return {"__type__": "PIL.Image", "mode": obj.mode, "size": obj.size, "content_hash": img_hash}
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


def hash_data_to_seed(data: dict[str, Any], max_bytes: int = 4) -> int:
    json_str = json.dumps(
        data,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    seed_int = int(hashlib.sha256(json_str.encode("utf-8")).hexdigest(), 16)
    if max_bytes > 0:
        seed_int %= 2 ** (8 * max_bytes)
    return seed_int


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


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
) -> list[LiberoPlusTask]:
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)

    tasks: list[LiberoPlusTask] = []
    for suite_name in task_suites:
        if suite_name not in classification:
            raise KeyError(f"Suite {suite_name!r} not found in {classification_path}. Available: {sorted(classification)}")

        for item in classification[suite_name]:
            task_name = item["name"]
            task_id = int(item["id"]) - 1

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


def _make_env(task: Any, resolution: int, seed: int):
    assert get_libero_path is not None and OffScreenRenderEnv is not None
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def _safe_path_segment(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "task"


def _save_video(video_out_path: Path, task_info: LiberoPlusTask, episode_idx: int, success: bool, replay_images: list[np.ndarray]) -> str | None:
    if not replay_images:
        return None
    import imageio

    task_dir = video_out_path / task_info.suite / _safe_path_segment(task_info.category)
    task_dir.mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    filename = f"{task_info.task_id:05d}_{_safe_path_segment(task_info.task_name, 96)}_ep{episode_idx}_{suffix}.mp4"
    video_path = task_dir / filename
    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
    return str(video_path)


def _run_episode(
    *,
    suite_objects: dict[str, Any],
    task_info: LiberoPlusTask,
    episode_idx: int,
    client: Client,
    args: argparse.Namespace,
) -> tuple[bool, int, str | None, str | None]:
    env = None
    replay_images: list[np.ndarray] = []
    t = 0
    try:
        if task_info.suite not in suite_objects:
            suite_objects[task_info.suite] = _make_suite(task_info.suite)
        suite = suite_objects[task_info.suite]
        task = suite.get_task(task_info.task_id)
        initial_states = suite.get_task_init_states(task_info.task_id)
        state_index = episode_idx % len(initial_states)

        env_seed = args.seed + task_info.task_id * 1000 + episode_idx
        env = _make_env(task, LIBERO_ENV_RESOLUTION, env_seed)
        env.reset()
        obs = env.set_init_state(initial_states[state_index])

        action_plan: collections.deque = collections.deque()
        done = False
        max_steps = args.max_steps if args.max_steps > 0 else MAX_STEPS_BY_SUITE[task_info.suite]

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = convert_to_uint8(img)
            wrist_img = convert_to_uint8(wrist_img)
            if args.save_video:
                replay_images.append(img)

            if not action_plan:
                state = np.concatenate(
                    [
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                        np.array([0.0] * 24),
                    ]
                )
                instruction = str(task.language).capitalize()
                model_inputs = {
                    "task_id": "libero_all",
                    "state": state,
                    "base": Image.fromarray(img),
                    "wrist_left": Image.fromarray(wrist_img),
                    "language": instruction,
                }
                model_inputs["seed"] = hash_data_to_seed(model_inputs)
                model_inputs["language"] = model_inputs["language"] + "."
                action_chunk = client(**model_inputs)[0, :, :-1]
                action_plan.extend(action_chunk[0 : args.replan_steps, 0:7])

            action = action_plan.popleft().tolist()
            obs, _, done, _ = env.step(action)
            if done:
                break
            t += 1

        success = bool(done)
        video_path = None
        if args.save_video and args.video_out_path is not None:
            video_path = _save_video(Path(args.video_out_path), task_info, episode_idx, success, replay_images)
        return success, max(0, t - args.num_steps_wait), video_path, None
    except Exception as exc:  # Keep long multi-task sweeps running and record failures.
        logging.exception("Episode failed: suite=%s task=%s episode=%s", task_info.suite, task_info.task_name, episode_idx)
        return False, max(0, t - args.num_steps_wait), None, repr(exc)
    finally:
        if env is not None:
            env.close()


def _write_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: OSError | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for model server at {host}:{port}: {last_error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Xiaomi-Robotics-0 on all LIBERO-plus perturbation tasks.")
    parser.add_argument("--task-suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--classification-path", required=True, type=Path)
    parser.add_argument("--result-jsonl", required=True, type=Path)
    parser.add_argument("--libero-plus-root", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--eval-rank", type=int, default=0)
    parser.add_argument("--eval-world-size", type=int, default=1)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--server-wait-timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Override max episode steps; 0 uses per-suite defaults.")
    parser.add_argument("--video-out-path", type=Path, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--worker-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only-first-task", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=28)
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

    suite_objects: dict[str, Any] = {}
    all_tasks = _load_classification_tasks(args.classification_path, args.task_suites)
    if args.only_first_task:
        all_tasks = all_tasks[:1]
    if args.limit_tasks > 0:
        all_tasks = all_tasks[: args.limit_tasks]
    assigned_tasks = all_tasks[args.eval_rank :: args.eval_world_size]

    args.result_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.result_jsonl.write_text("", encoding="utf-8")
    if args.save_video and args.video_out_path is not None:
        args.video_out_path.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] rank={args.eval_rank}/{args.eval_world_size} tasks={len(assigned_tasks)}/{len(all_tasks)} "
        f"episodes={len(assigned_tasks) * args.num_trials_per_task} port={args.port}",
        flush=True,
    )

    _wait_for_port(args.host, args.port, args.server_wait_timeout_s)
    client = Client(args.host, args.port)
    total_successes = 0
    total_episodes = 0
    total_episode_target = len(assigned_tasks) * args.num_trials_per_task

    progress = tqdm(total=total_episode_target, desc=f"Episodes rank {args.eval_rank}", dynamic_ncols=True, disable=not args.worker_progress)
    try:
        with args.result_jsonl.open("a", encoding="utf-8") as result_file:
            for task_index, task_info in enumerate(assigned_tasks):
                task_successes = 0
                for episode_idx in range(args.num_trials_per_task):
                    success, episode_length, video_path, error = _run_episode(
                        suite_objects=suite_objects,
                        task_info=task_info,
                        episode_idx=episode_idx,
                        client=client,
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
                        "rank": args.eval_rank,
                        "episode_idx": episode_idx,
                        "success": bool(success),
                        "episode_length": int(episode_length),
                        "video_path": video_path,
                        "error": error,
                    }
                    _write_record(result_file, record)

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
        client.close()

    print(
        f"[DONE] rank={args.eval_rank} success={total_successes}/{total_episodes} "
        f"rate={_rate(total_successes, total_episodes):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
