#!/usr/bin/env python3
"""Evaluate an OpenPI pi0.5 PyTorch policy server on a LIBERO-Para task shard."""

from __future__ import annotations

import argparse
import collections
from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import imageio.v2 as imageio
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import torch
from tqdm import tqdm
import websockets.sync.client

KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_ACTION_DIM = 7


def parse_bddl_instruction(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
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
        body = prefix.removeprefix("comp_")
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
        category, subcategory = split_category_subcategory(prefix.removeprefix("act_"))
        categories = [category]
        subcategories = [subcategory]
    elif prefix.startswith("obj_"):
        paraphrase_type = "obj"
        category, subcategory = split_category_subcategory(prefix.removeprefix("obj_"))
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


def build_policy_observation(raw_observation: dict[str, Any], instruction: str, resize_size: int) -> dict[str, Any]:
    base_image = to_uint8(raw_observation["agentview_image"][::-1, ::-1])
    wrist_image = to_uint8(raw_observation["robot0_eye_in_hand_image"][::-1, ::-1])
    base_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    state = np.concatenate(
        (
            np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32),
            quaternion_to_axis_angle(raw_observation["robot0_eef_quat"]),
            np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": instruction,
    }


def validate_action_chunk(response: dict[str, Any]) -> np.ndarray:
    if "actions" not in response:
        raise KeyError(f"OpenPI response has no actions field; keys={sorted(response)}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != LIBERO_ACTION_DIM:
        raise ValueError(f"Expected OpenPI actions with shape (T, 7), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("OpenPI returned non-finite actions")
    return actions


def atomic_json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class StructuredLogger:
    """Append episode records and publish per-eval LIBERO-Para JSON files."""

    def __init__(self, output_dir: Path, original_instructions: dict[int, str]):
        self.output_dir = output_dir
        self.episode_dir = output_dir / "episodes"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.original_instructions = original_instructions
        self.per_eval: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "successes": 0})
        self.per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "successes": 0})

    @staticmethod
    def category_key(record: dict[str, Any]) -> str:
        if record["paraphrase_type"] == "comp":
            return f"comp_{'+'.join(record['categories'])}_{'+'.join(record['subcategories'])}"
        if record["categories"]:
            return f"{record['paraphrase_type']}_{record['categories'][0]}_{record['subcategories'][0]}"
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

        per_eval_summary = {
            f"eval{eval_id}": {
                **stats,
                "success_rate": stats["successes"] / stats["total"],
            }
            for eval_id, stats in self.per_eval.items()
        }
        per_category_summary = {
            category: {
                **stats,
                "success_rate": stats["successes"] / stats["total"],
            }
            for category, stats in self.per_category.items()
        }
        atomic_json_dump(
            self.output_dir / "summary.json",
            {
                "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "per_eval": per_eval_summary,
                "per_category": per_category_summary,
            },
        )


class ResilientPolicyClient:
    def __init__(self, host: str, port: int, retries: int):
        self.host = host
        self.port = port
        self.retries = retries
        self.client = self._connect()

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        logging.info("Connecting to OpenPI policy server at %s:%d", self.host, self.port)
        return websocket_client_policy.WebsocketClientPolicy(self.host, self.port)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self.client.infer(observation)
            except Exception as error:
                last_error = error
                logging.warning(
                    "OpenPI inference failed, attempt=%d/%d: %r",
                    attempt + 1,
                    self.retries + 1,
                    error,
                )
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
                    self.client = self._connect()
        assert last_error is not None
        raise last_error


def wait_for_server(host: str, port: int, timeout_s: float, poll_s: float) -> None:
    uri = f"ws://{host}:{port}"
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with websockets.sync.client.connect(
                uri,
                open_timeout=3.0,
                close_timeout=1.0,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
            ):
                logging.info("OpenPI policy server is ready at %s", uri)
                return
        except Exception as error:
            last_error = error
            time.sleep(poll_s)
    raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for {uri}: {last_error}")


def build_parser() -> argparse.ArgumentParser:
    openpi_root = Path(__file__).resolve().parents[1]
    para_root = openpi_root.parent / "LIBERO-Para"
    internal_root = para_root / "libero" / "libero"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl-dir", type=Path, default=internal_root / "bddl_files" / "libero_para")
    parser.add_argument("--init-dir", type=Path, default=internal_root / "init_files" / "libero_para")
    parser.add_argument("--goal-bddl-dir", type=Path, default=internal_root / "bddl_files" / "libero_goal")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--server-seed", type=int, default=None)
    parser.add_argument("--num-shards", "--num_shards", dest="num_shards", type=int, default=1)
    parser.add_argument("--shard-index", "--shard_index", dest="shard_index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--server-wait-timeout-s", type=float, default=1800.0)
    parser.add_argument("--server-wait-poll-s", type=float, default=2.0)
    parser.add_argument("--client-retries", type=int, default=2)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.num_steps_wait < 0:
        raise ValueError("--num-steps-wait must be >= 0")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be >= 1")
    if args.resize_size < 1:
        raise ValueError("--resize-size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    if args.initial_state_index < 0:
        raise ValueError("--initial-state-index must be >= 0")
    if args.client_retries < 0:
        raise ValueError("--client-retries must be >= 0")
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

    random.seed(args.seed + args.shard_index)
    np.random.seed((args.seed + args.shard_index) % (2**32))
    logging.basicConfig(level=logging.INFO, force=True)

    bddl_files = sorted(args.bddl_dir.glob("*.bddl"))
    logging.info("Found %d LIBERO-Para BDDL files", len(bddl_files))
    if args.max_tasks > 0:
        bddl_files = bddl_files[: args.max_tasks]
        logging.info("Limited evaluation to %d global tasks", len(bddl_files))
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
        tasks.append(
            {
                "task_id": task_id,
                "bddl_file": path.name,
                "instruction": parse_bddl_instruction(path),
                **parse_bddl_filename(path.name),
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
    original_instructions = {eval_id: parse_bddl_instruction(goal_bddl_files[eval_id]) for eval_id in eval_ids}

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

    wait_for_server(args.host, args.port, args.server_wait_timeout_s, args.server_wait_poll_s)
    client = ResilientPolicyClient(args.host, args.port, args.client_retries)

    logger = StructuredLogger(args.output_dir, original_instructions)
    logger.save_meta(
        {
            "model_name": "OpenPI-pi0.5-LIBERO-PyTorch",
            "model_family": "openpi_pi05",
            "checkpoint_format": "pytorch_safetensors",
            "checkpoint_dir": str(args.checkpoint_dir),
            "policy_config": args.policy_config,
            "seed": args.seed,
            "global_seed": args.seed,
            "server_seed": server_seed,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "host": args.host,
            "port": args.port,
            "max_steps": args.max_steps,
            "replan_steps": args.replan_steps,
            "resize_size": args.resize_size,
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
        desc=f"OpenPI seed={args.seed} rank={args.shard_index}",
        dynamic_ncols=True,
    )

    try:
        for local_index, task in enumerate(tasks):
            eval_id = int(task["eval_id"])
            environment = environments[eval_id]
            environment.seed(args.seed)
            raw_observation = environment.reset()
            raw_observation = environment.set_init_state(init_states[eval_id][args.initial_state_index])
            success = bool(environment.check_success())
            for _ in range(args.num_steps_wait):
                if success:
                    break
                raw_observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION)
                success = bool(environment.check_success())

            step = 0
            action_plan: collections.deque[np.ndarray] = collections.deque()
            inference_seconds: list[float] = []
            policy_infer_ms: list[float] = []
            action_chunk_horizons: list[int] = []
            replay_images: list[np.ndarray] = []

            while step < args.max_steps and not success:
                if args.save_video:
                    replay_images.append(to_uint8(raw_observation["agentview_image"][::-1, ::-1]))

                if not action_plan:
                    policy_observation = build_policy_observation(
                        raw_observation,
                        str(task["instruction"]),
                        args.resize_size,
                    )
                    inference_start = time.perf_counter()
                    response = client.infer(policy_observation)
                    inference_seconds.append(time.perf_counter() - inference_start)
                    actions = validate_action_chunk(response)
                    action_chunk_horizons.append(len(actions))
                    execution_horizon = min(args.replan_steps, len(actions))
                    action_plan.extend(actions[index] for index in range(execution_horizon))
                    timing = response.get("policy_timing")
                    if isinstance(timing, dict) and "infer_ms" in timing:
                        policy_infer_ms.append(float(timing["infer_ms"]))

                action = action_plan.popleft()
                raw_observation, _, done, _ = environment.step(action.tolist())
                step += 1
                success = bool(done or environment.check_success())

            total_successes += int(success)
            video_path = ""
            if args.save_video and replay_images:
                video_dir = args.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                suffix = "success" if success else "failure"
                destination = video_dir / f"task_{task['task_id']:04d}_{suffix}.mp4"
                imageio.mimwrite(destination, replay_images, fps=10)
                video_path = str(destination)

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
                "num_inferences": len(inference_seconds),
                "inference_seconds": inference_seconds,
                "policy_infer_ms": policy_infer_ms,
                "action_chunk_horizons": action_chunk_horizons,
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
            progress.set_postfix(success=f"{total_successes}/{completed}", SR=f"{success_rate:.2f}%")
            tqdm.write(
                f"Rank {args.shard_index} task {completed}/{len(tasks)} "
                f"[global={task['task_id']}/{total_tasks_global - 1}, eval{eval_id}] "
                f"{'✓' if success else '✗'} | success={total_successes}/{completed} | "
                f"SR={success_rate:.2f}% | seed={args.seed} | {str(task['instruction'])[:70]}",
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
        f"OpenPI LIBERO-Para seed={args.seed} rank={args.shard_index}/{args.num_shards}: "
        f"success={total_successes}/{len(tasks)} "
        f"SR={100.0 * total_successes / len(tasks):.2f}% elapsed={elapsed / 3600:.2f}h",
        flush=True,
    )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
