#!/usr/bin/env python3
"""Evaluate one Isaac GR00T N1.7 libero_goal checkpoint on LIBERO-Para."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def parse_bddl_instruction(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("(:language"):
                return line.removeprefix("(:language").rstrip(")").strip()
    raise ValueError(f"No (:language ...) instruction in {path}")


def split_category_subcategory(body: str) -> tuple[str, str]:
    for category in KNOWN_CATEGORIES:
        if body.startswith(f"{category}_"):
            return category, body[len(category) + 1 :]
    return body, ""


def parse_bddl_filename(filename: str) -> dict[str, Any]:
    basename = Path(filename).stem
    if "_eval" not in basename:
        raise ValueError(f"Cannot parse LIBERO-Para BDDL filename: {filename}")

    prefix, eval_variant = basename.rsplit("_eval", 1)
    eval_text, variant_text = eval_variant.split("_ver", 1)
    eval_id = int(eval_text)
    variant_id = int(variant_text)

    if prefix.startswith("comp_"):
        paraphrase_type = "comp"
        body = prefix[len("comp_") :]
        first_plus = body.index("+")
        category_1 = body[:first_plus]
        remainder = body[first_plus + 1 :]
        category_2 = None
        subcategory_1 = None
        subcategory_2 = None
        for known_category in KNOWN_CATEGORIES:
            if remainder.startswith(f"{known_category}_"):
                category_2 = known_category
                after_category = remainder[len(known_category) + 1 :]
                subcategory_1, subcategory_2 = after_category.rsplit("+", 1)
                break
        if category_2 is None:
            categories = [body]
            subcategories = [body]
        else:
            categories = [category_1, category_2]
            subcategories = [subcategory_1, subcategory_2]
    elif prefix.startswith("act_"):
        paraphrase_type = "act"
        category, subcategory = split_category_subcategory(prefix[len("act_") :])
        categories = [category]
        subcategories = [subcategory]
    elif prefix.startswith("obj_"):
        paraphrase_type = "obj"
        category, subcategory = split_category_subcategory(prefix[len("obj_") :])
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


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    result = quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator
    return result.astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image, dtype=np.uint8)


def build_model_observation(raw_observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    position = np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32)
    rotation = quaternion_to_axis_angle(raw_observation["robot0_eef_quat"])
    gripper = np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32)
    base_image = to_uint8(raw_observation["agentview_image"][::-1, ::-1])
    wrist_image = to_uint8(raw_observation["robot0_eye_in_hand_image"][::-1, ::-1])

    observation: dict[str, Any] = {
        "video.image": base_image[None, None, ...],
        "video.wrist_image": wrist_image[None, None, ...],
        "annotation.human.action.task_description": [instruction],
    }
    state_values = {
        "x": position[0:1],
        "y": position[1:2],
        "z": position[2:3],
        "roll": rotation[0:1],
        "pitch": rotation[1:2],
        "yaw": rotation[2:3],
        "gripper": gripper,
    }
    for key, value in state_values.items():
        observation[f"state.{key}"] = np.asarray(value, dtype=np.float32)[None, None, ...]
    return observation


def get_action_horizon(actions: dict[str, Any]) -> int:
    horizons = set()
    for key in ACTION_KEYS:
        action_key = f"action.{key}"
        if action_key not in actions:
            raise KeyError(f"GR00T response is missing {action_key}; keys={sorted(actions)}")
        value = np.asarray(actions[action_key])
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f"Unexpected {action_key} shape: {value.shape}, expected (1, T, D)")
        horizons.add(int(value.shape[1]))
    if len(horizons) != 1:
        raise ValueError(f"GR00T action horizons do not match: {sorted(horizons)}")
    return horizons.pop()


def action_vector_at(actions: dict[str, Any], index: int) -> np.ndarray:
    components = [
        np.asarray(actions[f"action.{key}"], dtype=np.float32)[0, index].reshape(-1)
        for key in ACTION_KEYS
    ]
    action = np.concatenate(components).astype(np.float32)
    if action.shape != (7,):
        raise ValueError(f"Expected a 7D LIBERO action, got shape {action.shape}")
    # Match gr00t.eval.sim.LIBERO.libero_env.LiberoEnv.step exactly.
    action[-1] = np.sign(2.0 * action[-1] - 1.0)
    action[-1] *= -1.0
    return action


def proprioception(raw_observation: dict[str, Any]) -> list[float]:
    return np.concatenate(
        [
            np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32),
            quaternion_to_axis_angle(raw_observation["robot0_eef_quat"]),
            np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).tolist()


def atomic_json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class StructuredLogger:
    """Append episode records efficiently and publish official LIBERO-Para JSON files."""

    def __init__(self, output_dir: Path, original_instructions: dict[int, str]):
        self.output_dir = output_dir
        self.episode_dir = output_dir / "episodes"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.original_instructions = original_instructions
        self.per_eval: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "successes": 0})
        self.per_category: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successes": 0}
        )

    @staticmethod
    def category_key(record: dict[str, Any]) -> str:
        if record["paraphrase_type"] == "comp":
            return f"comp_{'+'.join(record['categories'])}_{'+'.join(record['subcategories'])}"
        if record["categories"]:
            return (
                f"{record['paraphrase_type']}_{record['categories'][0]}_"
                f"{record['subcategories'][0]}"
            )
        return "unknown"

    def save_meta(self, data: dict[str, Any]) -> None:
        atomic_json_dump(self.output_dir / "meta.json", data)

    def append_episode(self, record: dict[str, Any]) -> None:
        eval_id = int(record["eval_id"])
        path = self.episode_dir / f"eval{eval_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        self.per_eval[eval_id]["total"] += 1
        self.per_eval[eval_id]["successes"] += int(record["success"])
        category = self.category_key(record)
        self.per_category[category]["total"] += 1
        self.per_category[category]["successes"] += int(record["success"])

    def save_progress(
        self,
        completed: int,
        successes: int,
        total: int,
        seed: int,
        shard_index: int,
        num_shards: int,
        total_global: int,
    ) -> None:
        atomic_json_dump(
            self.output_dir / "progress.json",
            {
                "completed": completed,
                "successes": successes,
                "success_rate": successes / completed if completed else 0.0,
                "total": total,
                "total_global": total_global,
                "seed": seed,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def finalize(self, total_episodes: int, total_successes: int) -> None:
        for eval_id in sorted(self.per_eval):
            source = self.episode_dir / f"eval{eval_id}.jsonl"
            destination = self.output_dir / f"eval{eval_id}.json"
            temporary = destination.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as output:
                output.write(
                    json.dumps(
                        {
                            "eval_id": eval_id,
                            "original_instruction": self.original_instructions.get(eval_id),
                        },
                        ensure_ascii=False,
                    )[:-1]
                )
                output.write(', "episodes": [\n')
                first = True
                with source.open("r", encoding="utf-8") as episodes:
                    for line in episodes:
                        if not first:
                            output.write(",\n")
                        output.write(line.strip())
                        first = False
                output.write("\n]}\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)

        per_eval_summary = {}
        for eval_id, stats in self.per_eval.items():
            per_eval_summary[f"eval{eval_id}"] = {
                **stats,
                "success_rate": stats["successes"] / stats["total"],
            }
        per_category_summary = {}
        for category, stats in self.per_category.items():
            per_category_summary[category] = {
                **stats,
                "success_rate": stats["successes"] / stats["total"],
            }
        atomic_json_dump(
            self.output_dir / "summary.json",
            {
                "overall_success_rate": (
                    total_successes / total_episodes if total_episodes else 0.0
                ),
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "per_eval": per_eval_summary,
                "per_category": per_category_summary,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    para_root = repo_root.parent / "LIBERO-Para"
    internal_root = para_root / "libero" / "libero"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl-dir", type=Path, default=internal_root / "bddl_files" / "libero_para"
    )
    parser.add_argument(
        "--init-dir", type=Path, default=internal_root / "init_files" / "libero_para"
    )
    parser.add_argument(
        "--goal-bddl-dir", type=Path, default=internal_root / "bddl_files" / "libero_goal"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--server-seed",
        type=int,
        default=None,
        help="Model-server seed recorded for reproducibility (defaults to --seed).",
    )
    parser.add_argument(
        "--num-shards",
        "--num_shards",
        dest="num_shards",
        type=int,
        default=1,
        help="Number of deterministic round-robin task shards.",
    )
    parser.add_argument(
        "--shard-index",
        "--shard_index",
        dest="shard_index",
        type=int,
        default=0,
        help="Zero-based shard index for this worker.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--client-timeout-ms", type=int, default=120_000)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--log-trajectories", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.n_action_steps < 1:
        raise ValueError("--n-action-steps must be >= 1")
    if args.num_steps_wait < 0:
        raise ValueError("--num-steps-wait must be >= 0")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    server_seed = args.seed if args.server_seed is None else args.server_seed
    evaluation_artifacts = [
        args.output_dir / "meta.json",
        args.output_dir / "progress.json",
        args.output_dir / "summary.json",
        args.output_dir / "episodes",
    ]
    evaluation_artifacts.extend(args.output_dir.glob("eval*.json"))
    if any(path.exists() for path in evaluation_artifacts):
        raise FileExistsError(f"Output directory already has evaluation data: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from gr00t.policy.server_client import PolicyClient
    from gr00t.utils.determinism import seed_everything
    from libero.libero.envs import OffScreenRenderEnv

    seed_everything(args.seed)
    logging.basicConfig(level=logging.INFO)

    bddl_files = sorted(args.bddl_dir.glob("*.bddl"))
    logging.info("Found %d LIBERO-Para BDDL files", len(bddl_files))
    if args.max_tasks > 0:
        bddl_files = bddl_files[: args.max_tasks]
        logging.info("Limited evaluation to %d tasks", len(bddl_files))
    if not bddl_files:
        raise FileNotFoundError(f"No BDDL files in {args.bddl_dir}")

    total_tasks_global = len(bddl_files)
    indexed_bddl_files = list(enumerate(bddl_files))
    shard_bddl_files = indexed_bddl_files[args.shard_index :: args.num_shards]
    if not shard_bddl_files:
        raise ValueError(
            f"Shard {args.shard_index}/{args.num_shards} has no tasks; "
            f"only {total_tasks_global} global tasks were selected"
        )
    logging.info(
        "Shard %d/%d selected %d of %d global tasks",
        args.shard_index,
        args.num_shards,
        len(shard_bddl_files),
        total_tasks_global,
    )

    tasks = []
    for task_id, path in shard_bddl_files:
        metadata = parse_bddl_filename(path.name)
        tasks.append(
            {
                "task_id": task_id,
                "bddl_file": path.name,
                "instruction": parse_bddl_instruction(path),
                **metadata,
            }
        )

    eval_ids = sorted({int(task["eval_id"]) for task in tasks})
    init_states = {}
    for eval_id in eval_ids:
        path = args.init_dir / f"eval{eval_id}.pruned_init"
        init_states[eval_id] = torch.load(path, map_location="cpu", weights_only=False)
        if not 0 <= args.initial_state_index < len(init_states[eval_id]):
            raise IndexError(
                f"initial-state-index {args.initial_state_index} is invalid for {path} "
                f"({len(init_states[eval_id])} states)"
            )

    goal_bddl_files = sorted(args.goal_bddl_dir.glob("*.bddl"))
    if len(goal_bddl_files) != 10:
        raise ValueError(f"Expected 10 libero_goal BDDLs, found {len(goal_bddl_files)}")
    original_instructions = {
        eval_id: parse_bddl_instruction(goal_bddl_files[eval_id]) for eval_id in eval_ids
    }

    environments = {}
    for eval_id in eval_ids:
        environment = OffScreenRenderEnv(
            bddl_file_name=str(goal_bddl_files[eval_id]),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
            ignore_done=True,
        )
        environment.seed(args.seed)
        environment.reset()
        environments[eval_id] = environment
        logging.info("Created env %d: %s", eval_id, goal_bddl_files[eval_id].name)

    client = PolicyClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.client_timeout_ms,
        strict=False,
    )
    if not client.ping():
        raise ConnectionError(f"GR00T policy server did not respond at {args.host}:{args.port}")
    logging.info("Connected to GR00T policy server at %s:%d", args.host, args.port)

    logger = StructuredLogger(args.output_dir, original_instructions)
    logger.save_meta(
        {
            "model_name": "Isaac-GR00T-N1.7-LIBERO-goal",
            "model_family": "gr00t_n1d7",
            "seed": args.seed,
            "global_seed": args.seed,
            "server_seed": server_seed,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "host": args.host,
            "port": args.port,
            "max_steps": args.max_steps,
            "n_action_steps": args.n_action_steps,
            "num_steps_wait": args.num_steps_wait,
            "initial_state_index": args.initial_state_index,
            "bddl_dir": str(args.bddl_dir),
            "total_tasks": len(tasks),
            "total_tasks_local": len(tasks),
            "total_tasks_global": total_tasks_global,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    logger.save_progress(
        0,
        0,
        len(tasks),
        args.seed,
        args.shard_index,
        args.num_shards,
        total_tasks_global,
    )

    total_successes = 0
    start_time = time.time()
    progress = tqdm(
        total=len(tasks),
        desc=f"GR00T seed={args.seed} rank={args.shard_index}",
        dynamic_ncols=True,
    )

    try:
        for local_index, task in enumerate(tasks):
            eval_id = int(task["eval_id"])
            environment = environments[eval_id]
            environment.seed(args.seed)
            raw_observation = environment.reset()
            raw_observation = environment.set_init_state(
                init_states[eval_id][args.initial_state_index]
            )
            for _ in range(args.num_steps_wait):
                raw_observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION)

            client.reset()
            success = bool(environment.check_success())
            step = 0
            actions_log: list[list[float]] = []
            proprio_log: list[list[float]] = []
            rewards_log: list[float] = []
            chunk_boundaries: list[int] = []
            inference_seconds: list[float] = []
            replay_images: list[np.ndarray] = []

            while step < args.max_steps and not success:
                if args.save_video:
                    replay_images.append(to_uint8(raw_observation["agentview_image"][::-1, ::-1]))

                model_observation = build_model_observation(
                    raw_observation, str(task["instruction"])
                )
                inference_start = time.perf_counter()
                action_chunk, _ = client.get_action(model_observation)
                inference_seconds.append(time.perf_counter() - inference_start)
                horizon = get_action_horizon(action_chunk)
                execution_horizon = min(args.n_action_steps, horizon)
                chunk_boundaries.append(step)

                for chunk_index in range(execution_horizon):
                    if step >= args.max_steps or success:
                        break
                    action = action_vector_at(action_chunk, chunk_index)
                    if args.log_trajectories:
                        proprio_log.append(proprioception(raw_observation))
                        actions_log.append(action.tolist())
                    raw_observation, reward, done, _ = environment.step(action)
                    rewards_log.append(float(reward))
                    step += 1
                    success = bool(done or environment.check_success())

            total_successes += int(success)
            video_path = ""
            if args.save_video and replay_images:
                video_dir = args.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                suffix = "success" if success else "failure"
                video_path = str(video_dir / f"task_{task['task_id']:04d}_{suffix}.mp4")
                imageio.mimwrite(video_path, replay_images, fps=10)

            episode_record = {
                "task_id": int(task["task_id"]),
                "bddl_file": str(task["bddl_file"]),
                "paraphrase_type": str(task["paraphrase_type"]),
                "categories": task["categories"],
                "subcategories": task["subcategories"],
                "eval_id": eval_id,
                "variant_id": int(task["variant_id"]),
                "paraphrased_instruction": str(task["instruction"]),
                "success": success,
                "num_steps": step,
                "actions": actions_log,
                "proprio_states": proprio_log,
                "action_chunk_boundaries": chunk_boundaries,
                "inference_seconds": inference_seconds,
                "rewards": rewards_log,
                "replay_video_path": video_path,
                "initial_state_idx": args.initial_state_index,
                "episode_idx": 0,
                "seed": args.seed,
                "server_seed": server_seed,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            logger.append_episode(episode_record)
            completed = local_index + 1
            logger.save_progress(
                completed,
                total_successes,
                len(tasks),
                args.seed,
                args.shard_index,
                args.num_shards,
                total_tasks_global,
            )

            success_rate = 100.0 * total_successes / completed
            progress.update(1)
            progress.set_postfix(
                success=f"{total_successes}/{completed}", SR=f"{success_rate:.2f}%"
            )
            tqdm.write(
                f"Rank {args.shard_index} task {completed}/{len(tasks)} "
                f"[global={task['task_id']}/{total_tasks_global - 1}, eval{eval_id}] "
                f"{'✓' if success else '✗'} | success={total_successes}/{completed} | "
                f"SR={success_rate:.2f}% | seed={args.seed} | "
                f"{str(task['instruction'])[:70]}",
                file=sys.stderr,
            )
            sys.stderr.flush()
    finally:
        progress.close()
        for environment in environments.values():
            environment.close()

    logger.finalize(len(tasks), total_successes)
    elapsed = time.time() - start_time
    print(
        f"GR00T LIBERO-Para seed={args.seed} rank={args.shard_index}/{args.num_shards}: "
        f"success={total_successes}/{len(tasks)} "
        f"SR={100.0 * total_successes / len(tasks):.2f}% elapsed={elapsed / 3600:.2f}h",
        flush=True,
    )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
