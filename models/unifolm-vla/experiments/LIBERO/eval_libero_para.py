#!/usr/bin/env python3
"""Evaluate UnifoLM-VLA on LIBERO-Para with deterministic task sharding.

The evaluation protocol follows the structured Xiaomi-Robotics-0 evaluator in
the local LIBERO-Para checkout: every paraphrase is one episode, ``eval0`` to
``eval9`` select the ten LIBERO-Goal environments, and the first pruned initial
state is reused for every paraphrase of the same base task.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from experiments.LIBERO.eval_libero import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    get_action_state,
    prepare_observation,
    process_action,
)
from experiments.LIBERO.libero_utils import quat2axisangle  # noqa: E402
from experiments.LIBERO.unifolm_vla_inference import (  # noqa: E402
    Unifolm_VLA_Inference,
)
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}


def parse_bddl_instruction(bddl_path: Path) -> str:
    """Read the language instruction from a BDDL file."""
    with bddl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("(:language"):
                return line.removeprefix("(:language").rstrip(")").strip()
    raise ValueError(f"No (:language ...) entry found in {bddl_path}")


def _split_category_subcategory(body: str) -> tuple[str, str]:
    for category in KNOWN_CATEGORIES:
        prefix = f"{category}_"
        if body.startswith(prefix):
            return category, body[len(prefix) :]
    return body, ""


def parse_bddl_filename(filename: str) -> dict[str, Any]:
    """Extract LIBERO-Para taxonomy metadata from a BDDL filename."""
    basename = Path(filename).stem
    if "_eval" not in basename:
        raise ValueError(f"Cannot parse LIBERO-Para filename: {filename}")

    prefix_part, eval_ver_part = basename.rsplit("_eval", 1)
    try:
        eval_text, variant_text = eval_ver_part.split("_ver", 1)
        eval_id = int(eval_text)
        variant_id = int(variant_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot parse eval/variant IDs from {filename}") from exc

    if prefix_part.startswith("comp_"):
        paraphrase_type = "comp"
        body = prefix_part.removeprefix("comp_")
        first_category, separator, remainder = body.partition("+")
        if not separator:
            raise ValueError(f"Cannot parse compositional filename: {filename}")

        second_category = ""
        subcategory_text = ""
        for category in KNOWN_CATEGORIES:
            marker = f"{category}_"
            if remainder.startswith(marker):
                second_category = category
                subcategory_text = remainder[len(marker) :]
                break
        if not second_category or "+" not in subcategory_text:
            raise ValueError(f"Cannot parse compositional filename: {filename}")
        subcategory_a, subcategory_b = subcategory_text.rsplit("+", 1)
        categories = [first_category, second_category]
        subcategories = [subcategory_a, subcategory_b]
    elif prefix_part.startswith(("act_", "obj_")):
        paraphrase_type, body = prefix_part.split("_", 1)
        category, subcategory = _split_category_subcategory(body)
        categories = [category]
        subcategories = [subcategory]
    else:
        raise ValueError(f"Unknown LIBERO-Para filename prefix: {filename}")

    return {
        "paraphrase_type": paraphrase_type,
        "categories": categories,
        "subcategories": subcategories,
        "eval_id": eval_id,
        "variant_id": variant_id,
    }


def select_task_shard(
    bddl_files: list[Path], max_tasks: int, num_shards: int, shard_index: int
) -> tuple[list[tuple[int, Path]], int]:
    """Select a round-robin shard while preserving global task IDs."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    if max_tasks == 0 or max_tasks < -1:
        raise ValueError(f"max_tasks must be -1 or a positive integer, got {max_tasks}")

    selected = bddl_files if max_tasks < 0 else bddl_files[:max_tasks]
    global_count = len(selected)
    indexed_files = list(enumerate(selected))[shard_index::num_shards]
    if not indexed_files:
        raise ValueError(
            f"Shard {shard_index}/{num_shards} has no tasks out of {global_count}"
        )
    return indexed_files, global_count


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


class StructuredLogger:
    """Atomically maintain the official per-eval JSON result layout."""

    def __init__(self, output_dir: Path, original_instructions: dict[int, str]):
        self.output_dir = output_dir
        self.original_instructions = original_instructions
        self.cache: dict[int, dict[str, Any]] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, eval_id: int) -> Path:
        return self.output_dir / f"eval{eval_id}.json"

    def _get(self, eval_id: int) -> dict[str, Any]:
        if eval_id not in self.cache:
            path = self._path(eval_id)
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    self.cache[eval_id] = json.load(handle)
            else:
                self.cache[eval_id] = {
                    "eval_id": eval_id,
                    "original_instruction": self.original_instructions.get(eval_id),
                    "episodes": [],
                }
        return self.cache[eval_id]

    def log_episode(self, episode: dict[str, Any]) -> None:
        eval_id = int(episode["eval_id"])
        record = dict(episode)
        record.pop("eval_id")
        data = self._get(eval_id)
        data["episodes"].append(record)
        atomic_dump(self._path(eval_id), data)

    def save_meta(self, meta: dict[str, Any]) -> None:
        atomic_dump(self.output_dir / "meta.json", meta)

    def save_progress(
        self,
        *,
        completed: int,
        successes: int,
        shard_total: int,
        global_total: int,
        shard_index: int,
        num_shards: int,
    ) -> None:
        atomic_dump(
            self.output_dir / "progress.json",
            {
                "completed": completed,
                "successes": successes,
                "success_rate": successes / completed if completed else 0.0,
                "shard_total": shard_total,
                "global_total": global_total,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def save_summary(self, total_episodes: int, total_successes: int) -> None:
        per_eval: dict[str, dict[str, Any]] = {}
        per_category: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successes": 0}
        )
        for eval_id, data in self.cache.items():
            episodes = data["episodes"]
            successes = sum(bool(item.get("success")) for item in episodes)
            per_eval[f"eval{eval_id}"] = {
                "total": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes) if episodes else 0.0,
            }
            for episode in episodes:
                stats = per_category[category_key(episode)]
                stats["total"] += 1
                stats["successes"] += int(bool(episode.get("success")))

        per_category_with_rates: dict[str, dict[str, Any]] = {}
        for key, stats in per_category.items():
            per_category_with_rates[key] = {
                **stats,
                "success_rate": stats["successes"] / stats["total"],
            }
        atomic_dump(
            self.output_dir / "summary.json",
            {
                "overall_success_rate": (
                    total_successes / total_episodes if total_episodes else 0.0
                ),
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "per_eval": per_eval,
                "per_category": per_category_with_rates,
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate UnifoLM-VLA on LIBERO-Para."
    )
    parser.add_argument("--pretrained-path", type=Path, required=True)
    parser.add_argument("--vlm-pretrained-path", type=Path, required=True)
    parser.add_argument("--unnorm-key", default="libero_goal_no_noops")
    parser.add_argument("--bddl-dir", type=Path, required=True)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--goal-bddl-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    required_files = [args.pretrained_path]
    required_dirs = [
        args.vlm_pretrained_path,
        args.bddl_dir,
        args.init_dir,
        args.goal_bddl_dir,
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Required directory not found: {path}")
    for name in ("max_steps", "window_size", "replan_steps", "num_shards"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.num_steps_wait < 0:
        raise ValueError("--num-steps-wait must be >= 0")


def _to_numpy_actions(actions: Any) -> np.ndarray:
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] < 7:
        raise ValueError(f"Expected an action chunk shaped [N, >=7], got {array.shape}")
    return array[:, :7]


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    existing_results = list(args.output_dir.glob("eval*.json"))
    if existing_results or (args.output_dir / "meta.json").exists():
        raise FileExistsError(
            f"Output already contains evaluation data: {args.output_dir}"
        )

    bddl_files = sorted(args.bddl_dir.glob("*.bddl"), key=lambda path: path.name)
    if not bddl_files:
        raise FileNotFoundError(f"No BDDL files found under {args.bddl_dir}")
    task_entries, global_total = select_task_shard(
        bddl_files, args.max_tasks, args.num_shards, args.shard_index
    )

    tasks: list[dict[str, Any]] = []
    for global_task_id, bddl_path in task_entries:
        metadata = parse_bddl_filename(bddl_path.name)
        tasks.append(
            {
                "task_id": global_task_id,
                "bddl_file": bddl_path.name,
                "instruction": parse_bddl_instruction(bddl_path),
                **metadata,
            }
        )

    goal_bddl_files = sorted(
        args.goal_bddl_dir.glob("*.bddl"), key=lambda path: path.name
    )
    if len(goal_bddl_files) != 10:
        raise ValueError(
            f"Expected 10 LIBERO-Goal BDDLs, found {len(goal_bddl_files)}"
        )
    original_instructions = {
        eval_id: parse_bddl_instruction(path)
        for eval_id, path in enumerate(goal_bddl_files)
    }

    eval_ids = sorted({int(task["eval_id"]) for task in tasks})
    if any(eval_id < 0 or eval_id >= len(goal_bddl_files) for eval_id in eval_ids):
        raise ValueError(f"Invalid eval IDs in task shard: {eval_ids}")

    init_states: dict[int, Any] = {}
    envs: dict[int, OffScreenRenderEnv] = {}
    try:
        for eval_id in eval_ids:
            init_path = args.init_dir / f"eval{eval_id}.pruned_init"
            if not init_path.is_file():
                raise FileNotFoundError(f"Initial states not found: {init_path}")
            states = torch.load(init_path, map_location="cpu", weights_only=False)
            if len(states) < 1:
                raise ValueError(f"Initial-state file is empty: {init_path}")
            init_states[eval_id] = states

            env = OffScreenRenderEnv(
                bddl_file_name=str(goal_bddl_files[eval_id]),
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(args.seed)
            env.reset()
            envs[eval_id] = env

        logging.info(
            "Shard %d/%d: %d of %d tasks; eval IDs=%s",
            args.shard_index + 1,
            args.num_shards,
            len(tasks),
            global_total,
            eval_ids,
        )
        logging.info("Loading UnifoLM-VLA checkpoint: %s", args.pretrained_path)
        model = Unifolm_VLA_Inference(
            policy_ckpt_path=str(args.pretrained_path),
            image_size=[224, 224],
            unnorm_key=args.unnorm_key,
            vlm_pretrained_path=str(args.vlm_pretrained_path),
        )

        result_logger = StructuredLogger(args.output_dir, original_instructions)
        result_logger.save_meta(
            {
                "model_name": "UnifoLM-VLA",
                "model_family": "unifolm_vla",
                "pretrained_path": str(args.pretrained_path),
                "vlm_pretrained_path": str(args.vlm_pretrained_path),
                "unnorm_key": args.unnorm_key,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "num_steps_wait": args.num_steps_wait,
                "window_size": args.window_size,
                "replan_steps": args.replan_steps,
                "bddl_dir": str(args.bddl_dir),
                "init_dir": str(args.init_dir),
                "goal_bddl_dir": str(args.goal_bddl_dir),
                "total_tasks": len(tasks),
                "total_tasks_global": global_total,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        result_logger.save_progress(
            completed=0,
            successes=0,
            shard_total=len(tasks),
            global_total=global_total,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )

        total_successes = 0
        start_time = time.monotonic()
        progress = tqdm(tasks, desc=f"Eval rank {args.shard_index}", dynamic_ncols=True)
        for local_index, task in enumerate(progress, start=1):
            eval_id = int(task["eval_id"])
            env = envs[eval_id]
            instruction = str(task["instruction"])

            model.reset(task_description=instruction)
            obs = env.reset()
            obs = env.set_init_state(init_states[eval_id][0])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            action_plan: collections.deque[np.ndarray] = collections.deque()
            observation_window: collections.deque[dict[str, Any]] = collections.deque(
                maxlen=args.window_size
            )
            all_actions: list[list[float]] = []
            all_proprio_states: list[list[float]] = []
            all_rewards: list[float] = []
            action_chunk_boundaries: list[int] = []
            replay_images: list[np.ndarray] = []
            success = False

            for step in range(args.max_steps):
                while len(observation_window) < args.window_size:
                    prepared_observation, image = prepare_observation(
                        obs, resize_size=224
                    )
                    observation_window.append(prepared_observation)

                if args.save_video:
                    replay_images.append(np.asarray(image).copy())

                proprio = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"].copy()),
                        obs["robot0_gripper_qpos"],
                    )
                )
                all_proprio_states.append(proprio.tolist())

                if not action_plan:
                    action_chunk_boundaries.append(step)
                    with torch.inference_mode():
                        predicted_actions = _to_numpy_actions(
                            get_action_state(observation_window, instruction, model)
                        )
                    if len(predicted_actions) < args.replan_steps:
                        raise ValueError(
                            "Model returned "
                            f"{len(predicted_actions)} actions, fewer than "
                            f"--replan-steps={args.replan_steps}"
                        )
                    action_plan.extend(predicted_actions[: args.replan_steps])

                observation_window.popleft()
                action = process_action(np.asarray(action_plan.popleft()).copy())
                action = np.asarray(action, dtype=np.float32)[:7]
                all_actions.append(action.tolist())

                obs, reward, done, _ = env.step(action.tolist())
                all_rewards.append(float(reward))
                if done or env.check_success():
                    success = True
                    break

            total_successes += int(success)
            completed = local_index
            video_path = ""
            if args.save_video and replay_images:
                video_dir = args.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                suffix = "success" if success else "failure"
                video_file = (
                    video_dir
                    / f"task_{int(task['task_id']):04d}_eval{eval_id}_{suffix}.mp4"
                )
                imageio.mimwrite(video_file, replay_images, fps=10)
                video_path = str(video_file)

            result_logger.log_episode(
                {
                    "eval_id": eval_id,
                    "task_id": int(task["task_id"]),
                    "bddl_file": str(task["bddl_file"]),
                    "paraphrase_type": task["paraphrase_type"],
                    "categories": task["categories"],
                    "subcategories": task["subcategories"],
                    "variant_id": int(task["variant_id"]),
                    "paraphrased_instruction": instruction,
                    "success": success,
                    "num_steps": len(all_actions),
                    "actions": all_actions,
                    "proprio_states": all_proprio_states,
                    "action_chunk_boundaries": action_chunk_boundaries,
                    "rewards": all_rewards,
                    "replay_video_path": video_path,
                    "initial_state_idx": 0,
                    "episode_idx": 0,
                }
            )
            result_logger.save_progress(
                completed=completed,
                successes=total_successes,
                shard_total=len(tasks),
                global_total=global_total,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )

            success_rate = 100.0 * total_successes / completed
            progress.set_postfix(
                success=f"{total_successes}/{completed}", SR=f"{success_rate:.2f}%"
            )
            tqdm.write(
                f"Task {completed}/{len(tasks)} "
                f"(global {int(task['task_id']) + 1}/{global_total}) "
                f"[eval{eval_id}]: {'success' if success else 'failure'} | "
                f"SR={success_rate:.2f}% | {instruction[:72]}"
            )

        result_logger.save_summary(len(tasks), total_successes)
        elapsed = time.monotonic() - start_time
        final_rate = 100.0 * total_successes / len(tasks)
        print(
            f"[DONE] rank={args.shard_index}/{args.num_shards} "
            f"success={total_successes}/{len(tasks)} SR={final_rate:.2f}% "
            f"elapsed={elapsed / 3600:.2f}h output={args.output_dir}",
            flush=True,
        )
    finally:
        for env in envs.values():
            env.close()


if __name__ == "__main__":
    main()
