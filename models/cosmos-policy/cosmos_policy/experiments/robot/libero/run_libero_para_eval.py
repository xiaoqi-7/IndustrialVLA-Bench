# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos Policy evaluator for the LIBERO-Para benchmark.

LIBERO-Para is not registered as a normal LIBERO task suite.  It contains
thousands of BDDL files whose only changing field is the natural-language
instruction.  This module therefore evaluates the BDDL files directly while
using the corresponding ``libero_goal`` environment and ``eval{id}.pruned_init``
state file.

The module has two modes:

* ``--prepare-t5`` computes a persistent embedding cache for all paraphrased
  instructions using the local Cosmos T5 encoder.
* The normal mode evaluates one deterministic task shard.  The shell launcher
  starts one process per GPU and aggregates the shard progress.

The worker deliberately passes a single instruction embedding to
``get_action`` instead of initializing Cosmos' global T5 cache.  This keeps the
several-gigabyte Para cache in host memory and transfers only the current
instruction to the policy GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import re
import sys
import tempfile
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}
# LIBERO-Para's reference evaluators use a 300-action horizon for every base
# task.  Override --max-steps when reproducing the regular Cosmos LIBERO-10
# (520-step) protocol instead.
DEFAULT_MAX_STEPS = 300
DEFAULT_WAIT_STEPS = 10
DEFAULT_ENV_RESOLUTION = 256
DEFAULT_CHUNK_SIZE = 16
DEFAULT_DENOISING_STEPS = 5
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def parse_bddl_instruction(bddl_path: str | Path) -> str:
    """Read the single ``(:language ...)`` line from a BDDL file."""

    with open(bddl_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("(:language"):
                return stripped[len("(:language") :].rstrip(")").strip()
    raise ValueError(f"No (:language ...) instruction found in {bddl_path}")


def parse_bddl_filename(filename: str) -> dict[str, Any]:
    """Return the Para metadata encoded in a BDDL filename."""

    basename = os.path.basename(filename).removesuffix(".bddl")
    match = re.search(r"_eval(\d+)_ver(\d+)$", basename)
    if match is None:
        return {
            "paraphrase_type": "unknown",
            "categories": [],
            "subcategories": [],
            "eval_id": -1,
            "variant_id": -1,
        }

    eval_id, variant_id = int(match.group(1)), int(match.group(2))
    prefix = basename[: match.start()]
    if prefix.startswith("comp_"):
        body = prefix[5:]
        if "+" in body:
            first, remainder = body.split("+", 1)
            second = None
            for category in KNOWN_CATEGORIES:
                marker = category + "_"
                if remainder.startswith(marker):
                    second = category
                    remainder = remainder[len(marker) :]
                    break
            if second is not None and "+" in remainder:
                sub1, sub2 = remainder.rsplit("+", 1)
                categories = [first, second]
                subcategories = [sub1, sub2]
            else:
                categories = [body]
                subcategories = [body]
        else:
            categories = [body]
            subcategories = [body]
        paraphrase_type = "comp"
    elif prefix.startswith("act_"):
        paraphrase_type = "act"
        body = prefix[4:]
        categories, subcategories = _split_category_subcategory(body)
        categories, subcategories = [categories], [subcategories]
    elif prefix.startswith("obj_"):
        paraphrase_type = "obj"
        body = prefix[4:]
        categories, subcategories = _split_category_subcategory(body)
        categories, subcategories = [categories], [subcategories]
    else:
        paraphrase_type = "unknown"
        categories, subcategories = [], []

    return {
        "paraphrase_type": paraphrase_type,
        "categories": categories,
        "subcategories": subcategories,
        "eval_id": eval_id,
        "variant_id": variant_id,
    }


def _split_category_subcategory(body: str) -> tuple[str, str]:
    for category in KNOWN_CATEGORIES:
        marker = category + "_"
        if body.startswith(marker):
            return category, body[len(marker) :]
    return body, ""


def extract_eval_id(filename: str) -> int:
    match = re.search(r"_eval(\d+)(?:_ver\d+)?", os.path.basename(filename))
    if match is None:
        raise ValueError(f"Cannot extract eval id from {filename}")
    return int(match.group(1))


def _torch_load_compat(path: str | Path):
    """Load trusted LIBERO init states on both old and new PyTorch versions."""

    import torch

    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _scan_tasks(bddl_dir: str | Path, max_tasks: int = -1) -> list[dict[str, Any]]:
    files = sorted(Path(bddl_dir).glob("*.bddl"))
    if max_tasks > 0:
        files = files[:max_tasks]
    if not files:
        raise FileNotFoundError(f"No .bddl files found in {bddl_dir}")

    tasks: list[dict[str, Any]] = []
    for task_id, path in enumerate(files):
        metadata = parse_bddl_filename(path.name)
        if metadata["eval_id"] < 0:
            raise ValueError(f"Invalid LIBERO-Para filename: {path.name}")
        tasks.append(
            {
                "task_id": task_id,
                "bddl_file": path.name,
                "bddl_path": str(path),
                "instruction": parse_bddl_instruction(path),
                "eval_id": metadata["eval_id"],
                "variant_id": metadata["variant_id"],
                "paraphrase_type": metadata["paraphrase_type"],
                "categories": metadata["categories"],
                "subcategories": metadata["subcategories"],
            }
        )
    return tasks


def _as_cpu_embedding(value: Any):
    """Normalize a serialized embedding to a CPU bfloat16 tensor."""

    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    # A batched encoder result is indexed one item at a time during cache
    # creation, so accept both (512, 1024) and (1, 512, 1024) forms.
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tuple(tensor.shape[-2:]) != (512, 1024):
        raise ValueError(f"Unexpected T5 embedding shape: {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.bfloat16).contiguous()


def _load_embedding_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"T5 embedding file is not a dictionary: {path}")
    return raw


def _write_embedding_pickle(path: Path, embeddings: dict[str, Any]) -> None:
    """Atomically write a CPU-only embedding dictionary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a lock beside the cache so two independently started launchers do
    # not overwrite one another while extending the same cache.
    try:
        from filelock import FileLock

        lock_context = FileLock(str(path) + ".lock", timeout=1800)
    except ImportError:  # pragma: no cover - filelock is a Cosmos dependency.
        lock_context = None

    if lock_context is None:
        _write_embedding_pickle_unlocked(path, embeddings)
    else:
        with lock_context:
            _write_embedding_pickle_unlocked(path, embeddings)


def _write_embedding_pickle_unlocked(path: Path, embeddings: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _load_local_t5_encoder(model_dir: Path, tokenizer_dir: Path, device: str):
    """Load the split Cosmos T5 files (weights and tokenizer live separately)."""

    import torch
    from transformers import T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device)
    text_encoder.eval()

    @torch.inference_mode()
    def encode(prompts: list[str]):
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            return_length=True,
            return_offsets_mapping=False,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        output = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1).cpu()
        for batch_id, length in enumerate(lengths):
            output[batch_id][int(length) :] = 0
        return output

    return tokenizer, text_encoder, encode


def prepare_t5_embeddings(args: argparse.Namespace) -> int:
    """Compute missing Para instruction embeddings using the local T5 model."""

    import torch

    tasks = _scan_tasks(args.bddl_dir, args.max_tasks)
    instructions = sorted({task["instruction"] for task in tasks})
    output_path = Path(args.t5_cache)
    embeddings: dict[str, Any] = {}

    # Prefer a previously generated Para cache, then seed it with the regular
    # LIBERO cache (which contains the ten original instructions).
    for candidate in (output_path, Path(args.base_t5_cache) if args.base_t5_cache else None):
        if candidate is None or not candidate.is_file() or (candidate == output_path and embeddings):
            continue
        try:
            loaded = _load_embedding_pickle(candidate)
            for key, value in loaded.items():
                if key not in embeddings:
                    embeddings[key] = _as_cpu_embedding(value)
        except Exception as exc:
            if candidate == output_path:
                raise RuntimeError(f"Cannot read existing T5 cache {candidate}: {exc}") from exc
            print(f"[t5] ignoring unusable seed cache {candidate}: {exc}", flush=True)

    missing = [instruction for instruction in instructions if instruction not in embeddings]
    print(
        f"[t5] instructions={len(instructions)}, cached={len(instructions) - len(missing)}, "
        f"missing={len(missing)}",
        flush=True,
    )
    if missing:
        if not torch.cuda.is_available() and str(args.t5_device).startswith("cuda"):
            raise RuntimeError("CUDA is required to build the 11B T5 cache; torch.cuda.is_available() is false")
        if str(args.t5_device).startswith("cuda"):
            device_count = torch.cuda.device_count()
            if device_count < 1:
                raise RuntimeError("No CUDA device is visible for T5 precomputation")
        model_dir = Path(args.t5_model_dir)
        tokenizer_dir = Path(args.t5_tokenizer_dir)
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Local T5 config not found: {model_dir / 'config.json'}")
        if not (tokenizer_dir / "spiece.model").is_file():
            raise FileNotFoundError(f"Local T5 tokenizer not found: {tokenizer_dir / 'spiece.model'}")
        print(
            f"[t5] loading local encoder from {model_dir} and tokenizer from "
            f"{tokenizer_dir} on {args.t5_device}",
            flush=True,
        )
        _, encoder, encode_prompts = _load_local_t5_encoder(
            model_dir, tokenizer_dir, args.t5_device
        )

        batch_size = max(1, int(args.t5_batch_size))
        index = 0
        while index < len(missing):
            current = min(batch_size, len(missing) - index)
            batch = missing[index : index + current]
            while True:
                try:
                    with torch.inference_mode():
                        result = encode_prompts(batch)
                    break
                except RuntimeError as exc:
                    if current <= 1 or "out of memory" not in str(exc).lower():
                        raise
                    current = max(1, current // 2)
                    batch = missing[index : index + current]
                    print(f"[t5] CUDA OOM; retrying with batch_size={current}", flush=True)
                    torch.cuda.empty_cache()
            for instruction, value in zip(batch, result):
                embeddings[instruction] = _as_cpu_embedding(value)
            del result
            index += current
            if index == len(missing) or index % max(1, int(args.t5_log_every)) == 0:
                print(f"[t5] encoded {index}/{len(missing)} new instructions", flush=True)

        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Keep only the instructions needed by this benchmark plus any original
    # entries already present; retaining the originals makes the cache reusable
    # by the normal LIBERO evaluator as well.
    embeddings = {key: _as_cpu_embedding(value) for key, value in embeddings.items()}
    _write_embedding_pickle(output_path, embeddings)
    print(f"[t5] wrote {len(embeddings)} embeddings to {output_path}", flush=True)
    return 0


def _load_task_embeddings(cache_path: Path, instructions: Iterable[str]) -> dict[str, np.ndarray]:
    """Load only the current worker's instruction embeddings into NumPy."""

    required = list(dict.fromkeys(instructions))
    raw = _load_embedding_pickle(cache_path)
    missing = [instruction for instruction in required if instruction not in raw]
    if missing:
        preview = "; ".join(repr(item) for item in missing[:3])
        raise KeyError(
            f"{len(missing)} Para instructions are absent from {cache_path}; "
            f"run the launcher with PRECOMPUTE_T5=1 (examples: {preview})"
        )
    result: dict[str, np.ndarray] = {}
    for instruction in required:
        tensor = _as_cpu_embedding(raw[instruction])
        result[instruction] = tensor.float().numpy()
    del raw
    gc.collect()
    return result


def _make_cfg(args):
    """Construct the regular Cosmos LIBERO config for direct get_action use."""

    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

    return PolicyEvalConfig(
        suite="libero",
        model_family="cosmos",
        config=args.config,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        use_third_person_image=True,
        num_third_person_images=1,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        flip_images=args.flip_images,
        use_variance_scale=False,
        use_jpeg_compression=True,
        ar_future_prediction=False,
        ar_value_prediction=False,
        ar_qvalue_prediction=False,
        num_denoising_steps_action=args.num_denoising_steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        unnormalize_actions=True,
        normalize_proprio=True,
        dataset_stats_path=args.dataset_stats,
        # The custom worker does not initialize this global path cache.  The
        # embedding is passed to get_action as a NumPy array instead.
        t5_text_embeddings_path="",
        trained_with_image_aug=True,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.open_loop_steps,
        deterministic=True,
        deterministic_reset=False,
        task_suite_name="libero_10",
        num_trials_per_task=1,
        env_img_res=args.env_img_res,
        seed=args.seed,
        randomize_seed=False,
        data_collection=False,
    )


def _run_one_task(
    args,
    task: dict[str, Any],
    env,
    initial_state,
    embedding: np.ndarray,
    cfg,
    model,
    dataset_stats: dict[str, np.ndarray],
) -> tuple[bool, int, str | None]:
    """Roll one paraphrase episode and return (success, action_steps, error)."""

    import torch

    # Derive a task-specific environment seed so results do not depend on the
    # number of ranks used to shard the BDDL list.
    task_seed = int(args.seed) + int(task["task_id"])
    try:
        from cosmos_policy.utils.utils import set_seed_everywhere

        set_seed_everywhere(task_seed)
    except Exception:
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)

    try:
        env.seed(task_seed)
    except Exception:
        pass
    env.reset()
    obs = env.set_init_state(initial_state)
    if obs is None:
        obs = env.get_observation()

    # Settle the simulator after restoring the saved state, matching the
    # official LIBERO-Para reference evaluators.
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    action_queue: deque[np.ndarray] = deque()
    success = False
    steps = 0
    error: str | None = None
    try:
        from cosmos_policy.experiments.robot.cosmos_utils import get_action
        from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation

        while steps < args.max_steps:
            if not action_queue:
                observation = prepare_observation(obs, args.env_img_res, cfg.flip_images)
                result = get_action(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    embedding,
                    seed=args.seed,
                    randomize_seed=False,
                    num_denoising_steps_action=args.num_denoising_steps,
                    # Future image/value decoding is not needed to evaluate
                    # success and substantially increases the rollout cost.
                    generate_future_state_and_value_in_parallel=False,
                )
                actions = result["actions"]
                del result
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()
                if len(actions) < args.open_loop_steps:
                    raise RuntimeError(
                        f"Policy returned {len(actions)} actions, need {args.open_loop_steps}"
                    )
                action_queue.extend(np.asarray(action) for action in actions[: args.open_loop_steps])

            action = action_queue.popleft()
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float64).tolist())
            steps += 1
            success = bool(done)
            try:
                success = success or bool(env.check_success())
            except Exception:
                # Older LIBERO builds expose only the ``done`` signal.
                pass
            if success:
                break
    except Exception as exc:  # A failed rollout counts as one failed task.
        error = f"{type(exc).__name__}: {exc}"
        logging.exception("Task %s failed", task["task_id"])

    return success, steps, error


def _write_progress(path: Path, **values: Any) -> None:
    _atomic_json(path, values)


def run_worker(args: argparse.Namespace) -> int:
    """Evaluate this rank's task shard."""

    if args.eval_world_size < 1 or not (0 <= args.eval_rank < args.eval_world_size):
        raise ValueError("eval_rank must satisfy 0 <= eval_rank < eval_world_size")

    output_dir = Path(args.output_dir)
    progress_path = Path(args.progress_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"rank{args.eval_rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {args.eval_rank}] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,
    )

    tasks_all = _scan_tasks(args.bddl_dir, args.max_tasks)
    tasks = tasks_all[args.eval_rank :: args.eval_world_size]
    total_for_rank = len(tasks)
    started_at = time.time()
    _write_progress(
        progress_path,
        seed=args.seed,
        rank=args.eval_rank,
        world_size=args.eval_world_size,
        total=total_for_rank,
        completed=0,
        successes=0,
        success_rate=0.0,
        status="starting",
        started_at=started_at,
        updated_at=started_at,
    )

    if not tasks:
        _write_progress(
            progress_path,
            seed=args.seed,
            rank=args.eval_rank,
            world_size=args.eval_world_size,
            total=0,
            completed=0,
            successes=0,
            success_rate=0.0,
            status="done",
            started_at=started_at,
            updated_at=time.time(),
        )
        _atomic_json(
            output_dir / f"rank{args.eval_rank}.json",
            {"seed": args.seed, "rank": args.eval_rank, "total": 0, "successes": 0, "records": []},
        )
        return 0

    # Import LIBERO and the Cosmos model only after the caller has supplied
    # LIBERO_CONFIG_PATH/CUDA_VISIBLE_DEVICES in the environment.
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from cosmos_policy.experiments.robot.cosmos_utils import get_model, load_dataset_stats

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; a CUDA GPU is required")
    torch.cuda.set_device(0)

    needed_eval_ids = sorted({int(task["eval_id"]) for task in tasks})
    goal_files = sorted(Path(args.goal_bddl_dir).glob("*.bddl"))
    if not goal_files:
        raise FileNotFoundError(f"No goal BDDL files found in {args.goal_bddl_dir}")
    if max(needed_eval_ids) >= len(goal_files):
        raise ValueError(
            f"Need goal BDDL index {max(needed_eval_ids)}, but only {len(goal_files)} files exist"
        )

    initial_states: dict[int, Any] = {}
    envs: dict[int, Any] = {}
    try:
        for eval_id in needed_eval_ids:
            init_path = Path(args.init_dir) / f"eval{eval_id}.pruned_init"
            if not init_path.is_file():
                raise FileNotFoundError(f"Missing init state file: {init_path}")
            states = _torch_load_compat(init_path)
            if len(states) == 0:
                raise ValueError(f"No initial states in {init_path}")
            # The official Para evaluators use state 0 for every paraphrase of
            # a base task; the file still contains 50 states for compatibility.
            initial_states[eval_id] = states[0]

            goal_path = goal_files[eval_id]
            env = OffScreenRenderEnv(
                bddl_file_name=str(goal_path),
                camera_heights=args.env_img_res,
                camera_widths=args.env_img_res,
            )
            env.seed(args.seed)
            env.reset()
            envs[eval_id] = env
            logging.info("eval%d -> %s (%d tasks on this rank)", eval_id, goal_path.name, sum(1 for task in tasks if task["eval_id"] == eval_id))

        embedding_map = _load_task_embeddings(
            Path(args.t5_cache), (task["instruction"] for task in tasks)
        )
        dataset_stats = load_dataset_stats(args.dataset_stats)
        cfg = _make_cfg(args)
        model, cosmos_config = get_model(cfg)
        expected_chunk = getattr(getattr(cosmos_config, "dataloader_train", None), "dataset", None)
        expected_chunk = getattr(expected_chunk, "chunk_size", args.chunk_size)
        if int(expected_chunk) != int(args.chunk_size):
            raise ValueError(
                f"chunk_size mismatch: model was trained with {expected_chunk}, requested {args.chunk_size}"
            )
        model.eval()
        _write_progress(
            progress_path,
            seed=args.seed,
            rank=args.eval_rank,
            world_size=args.eval_world_size,
            total=total_for_rank,
            completed=0,
            successes=0,
            success_rate=0.0,
            status="running",
            started_at=started_at,
            updated_at=time.time(),
        )

        records: list[dict[str, Any]] = []
        successes = 0
        completed = 0
        jsonl_path = output_dir / f"rank{args.eval_rank}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            for task_index, task in enumerate(tasks, start=1):
                success, num_steps, error = _run_one_task(
                    args,
                    task,
                    envs[int(task["eval_id"])],
                    initial_states[int(task["eval_id"])],
                    embedding_map[task["instruction"]],
                    cfg,
                    model,
                    dataset_stats,
                )
                completed += 1
                successes += int(success)
                record = {
                    "task_id": task["task_id"],
                    "bddl_file": task["bddl_file"],
                    "eval_id": task["eval_id"],
                    "variant_id": task["variant_id"],
                    "paraphrase_type": task["paraphrase_type"],
                    "categories": task["categories"],
                    "subcategories": task["subcategories"],
                    "paraphrased_instruction": task["instruction"],
                    "success": bool(success),
                    "num_steps": int(num_steps),
                    "error": error,
                    "initial_state_idx": 0,
                    "episode_idx": 0,
                }
                records.append(record)
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()
                rate = successes / completed if completed else 0.0
                now = time.time()
                elapsed = now - started_at
                eta = elapsed / completed * (total_for_rank - completed) if completed else 0.0
                _write_progress(
                    progress_path,
                    seed=args.seed,
                    rank=args.eval_rank,
                    world_size=args.eval_world_size,
                    total=total_for_rank,
                    completed=completed,
                    successes=successes,
                    success_rate=rate,
                    last_task=task["bddl_file"],
                    last_success=bool(success),
                    updated_at=now,
                    elapsed_seconds=elapsed,
                    eta_seconds=max(0.0, eta),
                    status="running",
                )
                print(
                    f"[seed={args.seed} rank={args.eval_rank}] "
                    f"[{task_index}/{total_for_rank}] "
                    f"{'OK' if success else 'FAIL'} "
                    f"eval{task['eval_id']} {task['bddl_file']} | SR={rate * 100:.2f}%",
                    flush=True,
                )

        final_rate = successes / completed if completed else 0.0
        result = {
            "seed": args.seed,
            "rank": args.eval_rank,
            "world_size": args.eval_world_size,
            "total": completed,
            "successes": successes,
            "success_rate": final_rate,
            "records_file": str(jsonl_path),
            "records": records,
            "started_at": started_at,
            "finished_at": time.time(),
        }
        _atomic_json(output_dir / f"rank{args.eval_rank}.json", result)
        _write_progress(
            progress_path,
            seed=args.seed,
            rank=args.eval_rank,
            world_size=args.eval_world_size,
            total=total_for_rank,
            completed=completed,
            successes=successes,
            success_rate=final_rate,
            status="done",
            updated_at=time.time(),
        )
        print(
            f"[seed={args.seed} rank={args.eval_rank}] finished: "
            f"{successes}/{completed} ({final_rate * 100:.2f}%)",
            flush=True,
        )
        return 0
    finally:
        for env in envs.values():
            try:
                env.close()
            except Exception:
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-t5", action="store_true", help="Build the Para T5 cache and exit")
    parser.add_argument("--bddl-dir", default=os.environ.get("LIBERO_PARA_BDDL_DIR", ""))
    parser.add_argument("--init-dir", default=os.environ.get("LIBERO_PARA_INIT_DIR", ""))
    parser.add_argument("--goal-bddl-dir", default=os.environ.get("LIBERO_PARA_GOAL_BDDL_DIR", ""))
    parser.add_argument("--t5-cache", default=os.environ.get("LIBERO_PARA_T5_CACHE", ""))
    parser.add_argument("--base-t5-cache", default=os.environ.get("LIBERO_T5_EMBEDDINGS", ""))
    parser.add_argument("--t5-model-dir", default=os.environ.get("COSMOS_T5_MODEL_DIR", ""))
    parser.add_argument("--t5-tokenizer-dir", default=os.environ.get("COSMOS_T5_TOKENIZER_DIR", ""))
    parser.add_argument("--t5-device", default="cuda:0")
    parser.add_argument("--t5-batch-size", type=int, default=int(os.environ.get("T5_BATCH_SIZE", "4")))
    parser.add_argument("--t5-log-every", type=int, default=100)
    parser.add_argument("--max-tasks", type=int, default=int(os.environ.get("MAX_TASKS", "-1")))

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-rank", type=int, default=0)
    parser.add_argument("--eval-world-size", type=int, default=1)
    parser.add_argument("--output-dir", default="./results/libero_para")
    parser.add_argument("--progress-file", default="")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH", ""))
    parser.add_argument("--config", default="cosmos_predict2_2b_480p_libero__inference_only")
    parser.add_argument("--ckpt-path", default=os.environ.get("COSMOS_POLICY_CHECKPOINT", ""))
    parser.add_argument("--config-file", default="cosmos_policy/config/config.py")
    parser.add_argument("--dataset-stats", default=os.environ.get("COSMOS_DATASET_STATS", ""))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--open-loop-steps", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--num-denoising-steps", type=int, default=DEFAULT_DENOISING_STEPS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--num-steps-wait", type=int, default=DEFAULT_WAIT_STEPS)
    parser.add_argument("--env-img-res", type=int, default=DEFAULT_ENV_RESOLUTION)
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.prepare_t5:
        required = {
            "bddl_dir": args.bddl_dir,
            "t5_cache": args.t5_cache,
            "t5_model_dir": args.t5_model_dir,
            "t5_tokenizer_dir": args.t5_tokenizer_dir,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing arguments for --prepare-t5: {', '.join(missing)}")
        return prepare_t5_embeddings(args)

    required = {
        "bddl_dir": args.bddl_dir,
        "init_dir": args.init_dir,
        "goal_bddl_dir": args.goal_bddl_dir,
        "t5_cache": args.t5_cache,
        "ckpt_path": args.ckpt_path,
        "dataset_stats": args.dataset_stats,
        "libero_config_path": args.libero_config_path,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing worker arguments: {', '.join(missing)}")
    if not args.progress_file:
        args.progress_file = str(Path(args.output_dir) / "progress.json")
    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("DETERMINISTIC", "True")
    try:
        return run_worker(args)
    except Exception as exc:
        if args.progress_file:
            try:
                _write_progress(
                    Path(args.progress_file),
                    seed=args.seed,
                    rank=args.eval_rank,
                    world_size=args.eval_world_size,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    updated_at=time.time(),
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(1) from exc
