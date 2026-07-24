# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Multi-process Cosmos Policy evaluator for the LIBERO-plus benchmark.

The official LIBERO-plus task map contains 10,030 one-trial tasks across four
LIBERO suites.  This module builds one global task list and evaluates the
deterministic shard ``tasks[eval_rank::eval_world_size]``.  It is intentionally
not DDP: every process owns one complete policy model and one physical GPU.

The module also provides launcher-facing utility modes for manifest
validation, local T5 embedding preparation, and Markdown result summaries.
LIBERO-plus has almost one unique prompt per task, so embeddings are stored as
individually compressed rows in SQLite.  Workers read only the embedding for
the current task instead of loading a roughly 10 GiB pickle on every GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import os
import pickle
import sqlite3
import sys
import tempfile
import time
import traceback
import zlib
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
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
COLUMN_TO_CATEGORY = {column: category for category, column in CATEGORY_TO_COLUMN.items()}
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
EMBEDDING_SHAPE = (1, 512, 1024)
EMBEDDING_RAW_BYTES = int(np.prod(EMBEDDING_SHAPE)) * 2
T5_CACHE_SCHEMA_VERSION = "1"


def _split_words(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return value.replace(",", " ").split()
    return [str(item) for item in value]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _classification_path(explicit_path: str = "") -> Path:
    if explicit_path:
        return Path(explicit_path)
    from libero.libero import get_libero_path

    return Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"


def _build_manifest(
    task_suite_names: Iterable[str],
    classification_path: str = "",
    max_tasks: int = -1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the official ordered global task list and instantiated suites."""

    from libero.libero import benchmark

    names = _split_words(task_suite_names)
    if not names:
        raise ValueError("At least one LIBERO-plus task suite is required")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate task suite in {names}")

    path = _classification_path(classification_path)
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO-plus classification file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        classification = json.load(f)

    registry = benchmark.get_benchmark_dict()
    manifest: list[dict[str, Any]] = []
    suites: dict[str, Any] = {}
    for suite_name in names:
        if suite_name not in registry:
            raise ValueError(f"Unknown LIBERO suite {suite_name!r}; available={sorted(registry)}")
        if suite_name not in classification:
            raise ValueError(f"No LIBERO-plus classification entries for suite {suite_name!r}")

        # LIBERO-plus prints the complete task order (thousands of integers)
        # from _make_benchmark().  Suppress only that noisy informational line.
        with contextlib.redirect_stdout(io.StringIO()):
            suite = registry[suite_name]()
        suites[suite_name] = suite

        entries = classification[suite_name]
        if len(entries) != suite.n_tasks:
            raise ValueError(
                f"{suite_name}: classification has {len(entries)} tasks, suite has {suite.n_tasks}"
            )
        by_name: dict[str, dict[str, Any]] = {}
        for entry in entries:
            task_name = str(entry["name"])
            if task_name in by_name:
                raise ValueError(f"{suite_name}: duplicate classified task name {task_name!r}")
            by_name[task_name] = entry

        for task_id in range(suite.n_tasks):
            if max_tasks > 0 and len(manifest) >= max_tasks:
                break
            task = suite.get_task(task_id)
            if task.name not in by_name:
                raise ValueError(f"{suite_name}: task {task.name!r} has no category label")
            entry = by_name[task.name]
            category = str(entry["category"])
            if category not in CATEGORY_TO_COLUMN:
                raise ValueError(f"{suite_name}/{task.name}: unknown category {category!r}")
            manifest.append(
                {
                    "global_task_id": len(manifest),
                    "task_suite": suite_name,
                    "task_id": task_id,
                    "task_name": task.name,
                    # Match the official evaluator, which passes task.language
                    # to the policy (including LIBERO-plus variant suffixes).
                    "instruction": str(task.language),
                    "bddl_file": task.bddl_file,
                    "category": category,
                    "column": CATEGORY_TO_COLUMN[category],
                    "difficulty_level": entry.get("difficulty_level"),
                    "max_steps": SUITE_MAX_STEPS.get(suite_name),
                }
            )
        if max_tasks > 0 and len(manifest) >= max_tasks:
            break

    if not manifest:
        raise ValueError("The LIBERO-plus manifest is empty")
    return manifest, suites


def _manifest_summary(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    per_suite = Counter(record["task_suite"] for record in manifest)
    per_category = Counter(record["category"] for record in manifest)
    return {
        "total_tasks": len(manifest),
        "unique_instructions": len({record["instruction"] for record in manifest}),
        "per_suite": dict(per_suite),
        "per_category": dict(per_category),
    }


def _normalize_embedding(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != EMBEDDING_SHAPE:
        raise ValueError(f"Unexpected T5 embedding shape {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.bfloat16).contiguous()


def _serialize_embedding(value: Any) -> bytes:
    tensor = _normalize_embedding(value)
    raw = tensor.view(__import__("torch").uint16).numpy().tobytes(order="C")
    if len(raw) != EMBEDDING_RAW_BYTES:
        raise ValueError(f"Unexpected raw embedding size {len(raw)}")
    return zlib.compress(raw, level=1)


def _deserialize_embedding(payload: bytes) -> np.ndarray:
    import torch

    raw = zlib.decompress(payload)
    if len(raw) != EMBEDDING_RAW_BYTES:
        raise ValueError(
            f"Corrupt T5 cache row: decompressed {len(raw)} bytes, expected {EMBEDDING_RAW_BYTES}"
        )
    # NumPy has no portable bfloat16 dtype.  Recreate the tensor from its
    # uint16 bit representation, then expose float32 to get_action().
    bits = np.frombuffer(raw, dtype=np.uint16).copy().reshape(EMBEDDING_SHAPE)
    tensor = torch.from_numpy(bits).view(torch.bfloat16)
    return tensor.float().numpy()


def _open_t5_cache(path: Path, *, create: bool = False, read_only: bool = False) -> sqlite3.Connection:
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60.0)
    else:
        connection = sqlite3.connect(str(path), timeout=60.0)
    if create:
        # DELETE journal mode is reliable on shared filesystems and is only
        # used by the single cache-builder process.  Evaluation is read-only.
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (prompt TEXT PRIMARY KEY, payload BLOB NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('schema_version', ?)",
            (T5_CACHE_SCHEMA_VERSION,),
        )
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('dtype', 'bfloat16')"
        )
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('shape', '1,512,1024')"
        )
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('compression', 'zlib-1')"
        )
        connection.commit()
    _validate_t5_cache_schema(connection, path)
    return connection


def _validate_t5_cache_schema(connection: sqlite3.Connection, path: Path) -> None:
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Invalid T5 SQLite cache {path}: {exc}") from exc
    expected = {
        "schema_version": T5_CACHE_SCHEMA_VERSION,
        "dtype": "bfloat16",
        "shape": "1,512,1024",
        "compression": "zlib-1",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Incompatible T5 cache {path}: metadata[{key!r}]={metadata.get(key)!r}, expected {value!r}"
            )


def _load_local_t5_encoder(model_dir: Path, tokenizer_dir: Path, device: str):
    """Load Cosmos' split T5 model and return a dynamically padded encoder."""

    import torch
    from transformers import T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device)
    encoder.eval()

    @torch.inference_mode()
    def encode(prompts: list[str]):
        # Dynamic padding is substantially faster than running the 11B encoder
        # over 512 tokens for every short LIBERO instruction.  The returned
        # tensor is padded back to Cosmos' required (B, 512, 1024) shape.
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=False,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        short_output = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        short_output = short_output.cpu()
        lengths = attention_mask.sum(dim=1).cpu().tolist()
        output = torch.zeros(
            (len(prompts), EMBEDDING_SHAPE[1], EMBEDDING_SHAPE[2]), dtype=torch.bfloat16
        )
        for index, length in enumerate(lengths):
            output[index, : int(length)] = short_output[index, : int(length)]
        return output

    return tokenizer, encoder, encode


def _seed_cache_from_pickle(
    connection: sqlite3.Connection,
    pickle_path: Path,
    required_prompts: set[str],
) -> int:
    if not pickle_path.is_file():
        return 0
    with pickle_path.open("rb") as f:
        values = pickle.load(f)
    if not isinstance(values, dict):
        raise ValueError(f"Base T5 cache is not a dictionary: {pickle_path}")
    existing = {row[0] for row in connection.execute("SELECT prompt FROM embeddings")}
    rows = []
    for prompt, value in values.items():
        if prompt in required_prompts and prompt not in existing:
            rows.append((prompt, sqlite3.Binary(_serialize_embedding(value))))
    if rows:
        connection.executemany(
            "INSERT OR REPLACE INTO embeddings(prompt, payload) VALUES (?, ?)", rows
        )
        connection.commit()
    del values
    gc.collect()
    return len(rows)


def prepare_t5_cache(args: argparse.Namespace) -> int:
    import torch
    from filelock import FileLock

    manifest, _ = _build_manifest(args.task_suites, args.classification_path, args.max_tasks)
    instructions = {record["instruction"] for record in manifest}
    cache_path = Path(args.t5_cache)
    lock = FileLock(str(cache_path) + ".lock", timeout=args.t5_lock_timeout)
    with lock:
        connection = _open_t5_cache(cache_path, create=True)
        try:
            seeded = 0
            if args.base_t5_cache:
                seeded = _seed_cache_from_pickle(
                    connection, Path(args.base_t5_cache), instructions
                )
            existing = {row[0] for row in connection.execute("SELECT prompt FROM embeddings")}
            missing = sorted(instructions - existing, key=lambda prompt: (len(prompt), prompt))
            print(
                f"[t5] tasks={len(manifest)}, instructions={len(instructions)}, "
                f"cached={len(instructions) - len(missing)}, seeded={seeded}, missing={len(missing)}",
                flush=True,
            )
            if not missing:
                return 0
            if str(args.t5_device).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is required to build the local T5 cache")

            model_dir = Path(args.t5_model_dir)
            tokenizer_dir = Path(args.t5_tokenizer_dir)
            if not (model_dir / "config.json").is_file():
                raise FileNotFoundError(f"T5 config not found: {model_dir / 'config.json'}")
            if not (tokenizer_dir / "spiece.model").is_file():
                raise FileNotFoundError(f"T5 tokenizer not found: {tokenizer_dir / 'spiece.model'}")
            print(
                f"[t5] loading encoder={model_dir}, tokenizer={tokenizer_dir}, device={args.t5_device}",
                flush=True,
            )
            tokenizer, encoder, encode = _load_local_t5_encoder(
                model_dir, tokenizer_dir, args.t5_device
            )

            index = 0
            pending_since_commit = 0
            requested_batch_size = max(1, int(args.t5_batch_size))
            while index < len(missing):
                current = min(requested_batch_size, len(missing) - index)
                batch = missing[index : index + current]
                while True:
                    try:
                        encoded = encode(batch)
                        break
                    except RuntimeError as exc:
                        if current <= 1 or "out of memory" not in str(exc).lower():
                            raise
                        current = max(1, current // 2)
                        batch = missing[index : index + current]
                        print(f"[t5] CUDA OOM; retrying batch_size={current}", flush=True)
                        torch.cuda.empty_cache()

                rows = [
                    (prompt, sqlite3.Binary(_serialize_embedding(value)))
                    for prompt, value in zip(batch, encoded)
                ]
                connection.executemany(
                    "INSERT OR REPLACE INTO embeddings(prompt, payload) VALUES (?, ?)", rows
                )
                index += current
                pending_since_commit += current
                del encoded, rows
                if pending_since_commit >= args.t5_commit_every or index == len(missing):
                    connection.commit()
                    pending_since_commit = 0
                if index == len(missing) or index % max(1, args.t5_log_every) < current:
                    print(f"[t5] encoded {index}/{len(missing)} missing instructions", flush=True)

            del encoder, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            final_count = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            print(f"[t5] cache ready: {cache_path} ({final_count} rows)", flush=True)
            return 0
        finally:
            connection.close()


def check_t5_cache(args: argparse.Namespace) -> int:
    manifest, _ = _build_manifest(args.task_suites, args.classification_path, args.max_tasks)
    required = {record["instruction"] for record in manifest}
    path = Path(args.t5_cache)
    if not path.is_file():
        raise FileNotFoundError(f"T5 cache not found: {path}")
    connection = _open_t5_cache(path, read_only=True)
    try:
        existing = {row[0] for row in connection.execute("SELECT prompt FROM embeddings")}
    finally:
        connection.close()
    missing = required - existing
    if missing:
        preview = "; ".join(repr(item) for item in sorted(missing)[:3])
        raise RuntimeError(
            f"T5 cache is missing {len(missing)}/{len(required)} instructions; "
            f"run with PRECOMPUTE_T5=1. Examples: {preview}"
        )
    print(f"[t5] cache complete: {path} ({len(required)} required instructions)", flush=True)
    return 0


class T5EmbeddingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"T5 cache not found: {self.path}")
        self.connection = _open_t5_cache(self.path, read_only=True)

    def get(self, prompt: str) -> np.ndarray:
        row = self.connection.execute(
            "SELECT payload FROM embeddings WHERE prompt = ?", (prompt,)
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Instruction absent from {self.path}: {prompt!r}; run launcher with PRECOMPUTE_T5=1"
            )
        return _deserialize_embedding(row[0])

    def close(self) -> None:
        self.connection.close()


def _make_cfg(args: argparse.Namespace):
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


def _run_one_attempt(
    args: argparse.Namespace,
    record: dict[str, Any],
    suite: Any,
    embedding: np.ndarray,
    cfg: Any,
    model: Any,
    dataset_stats: dict[str, np.ndarray],
) -> tuple[bool, int, str | None]:
    """Run one episode attempt, always closing its simulator environment."""

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    env = None
    steps = 0
    try:
        task = suite.get_task(int(record["task_id"]))
        initial_states = suite.get_task_init_states(int(record["task_id"]))
        if len(initial_states) < 1:
            raise ValueError(f"No initial state for {record['task_suite']}/{record['task_id']}")
        initial_state = initial_states[0]

        bddl_path = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=args.env_img_res,
            camera_widths=args.env_img_res,
        )
        # Keep simulator initialization fixed across task sharding.  The three
        # requested seeds control policy sampling; ENVIRONMENT_SEED remains an
        # explicit override for experiments that also vary the simulator.
        env.seed(args.environment_seed)
        env.reset()
        obs = env.set_init_state(initial_state)
        if obs is None and hasattr(env, "get_observation"):
            obs = env.get_observation()

        success = False
        for _ in range(args.num_steps_wait):
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            success = success or bool(done)
        try:
            success = success or bool(env.check_success())
        except Exception:
            pass

        action_queue: deque[np.ndarray] = deque()
        max_steps = args.max_steps if args.max_steps > 0 else int(record["max_steps"])
        from cosmos_policy.experiments.robot.cosmos_utils import get_action
        from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation

        while not success and steps < max_steps:
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
                    generate_future_state_and_value_in_parallel=False,
                )
                actions = result["actions"]
                del result
                if hasattr(actions, "detach"):
                    actions = actions.detach().cpu().numpy()
                actions = np.asarray(actions)
                if len(actions) < args.open_loop_steps:
                    raise RuntimeError(
                        f"Policy returned {len(actions)} actions, need {args.open_loop_steps}"
                    )
                action_queue.extend(
                    np.asarray(action) for action in actions[: args.open_loop_steps]
                )

            action = action_queue.popleft()
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float64).tolist())
            steps += 1
            success = bool(done)
            try:
                success = success or bool(env.check_success())
            except Exception:
                pass
        return bool(success), steps, None
    except Exception as exc:
        logging.exception(
            "Task failed: global=%s suite=%s local=%s",
            record["global_task_id"],
            record["task_suite"],
            record["task_id"],
        )
        return False, steps, f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _evaluate_task(
    args: argparse.Namespace,
    record: dict[str, Any],
    suite: Any,
    embedding: np.ndarray,
    cfg: Any,
    model: Any,
    dataset_stats: dict[str, np.ndarray],
) -> tuple[bool, int, str | None, int]:
    last: tuple[bool, int, str | None] = (False, 0, "not started")
    for attempt in range(args.task_retries + 1):
        last = _run_one_attempt(args, record, suite, embedding, cfg, model, dataset_stats)
        if last[2] is None:
            return (*last, attempt + 1)
        if "out of memory" in last[2].lower():
            break
        if attempt < args.task_retries:
            logging.warning(
                "Retrying global task %s (%d/%d)",
                record["global_task_id"],
                attempt + 1,
                args.task_retries,
            )
    return (*last, min(args.task_retries + 1, attempt + 1))


def _category_progress(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats = {category: {"completed": 0, "successes": 0} for category in CATEGORY_TO_COLUMN}
    for record in records:
        category = record["category"]
        stats[category]["completed"] += 1
        stats[category]["successes"] += int(bool(record.get("success")))
    return stats


def _write_progress(path: Path, **values: Any) -> None:
    _atomic_json(path, values)


def _load_resume_records(
    jsonl_path: Path,
    task_ids: set[int],
    seed: int,
    rank: int,
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not jsonl_path.is_file():
        return records
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed resume JSONL {jsonl_path}:{line_number}: {exc}") from exc
            global_task_id = int(record["global_task_id"])
            if global_task_id not in task_ids:
                raise ValueError(
                    f"Resume record task {global_task_id} does not belong to rank {rank}; "
                    "WORLD_SIZE/TASK_SUITES/MAX_TASKS changed"
                )
            if int(record.get("seed", seed)) != seed:
                raise ValueError(f"Resume record seed mismatch in {jsonl_path}:{line_number}")
            records[global_task_id] = record
    return records


def _finalize_rank(
    args: argparse.Namespace,
    output_dir: Path,
    progress_path: Path,
    jsonl_path: Path,
    records_by_id: dict[int, dict[str, Any]],
    total_for_rank: int,
    started_at: float,
) -> int:
    records = [records_by_id[key] for key in sorted(records_by_id)]
    completed = len(records)
    successes = sum(int(bool(record.get("success"))) for record in records)
    errors = sum(int(record.get("error") is not None) for record in records)
    if completed != total_for_rank:
        raise RuntimeError(f"Rank {args.eval_rank}: completed {completed}, expected {total_for_rank}")
    if not jsonl_path.exists():
        _atomic_text(jsonl_path, "")
    result = {
        "seed": args.seed,
        "rank": args.eval_rank,
        "world_size": args.eval_world_size,
        "total": total_for_rank,
        "completed": completed,
        "successes": successes,
        "task_errors": errors,
        "success_rate": successes / completed if completed else 0.0,
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
        task_errors=errors,
        success_rate=successes / completed if completed else 0.0,
        category_stats=_category_progress(records),
        status="done",
        started_at=started_at,
        updated_at=time.time(),
    )
    print(
        f"[seed={args.seed} rank={args.eval_rank}] finished: "
        f"{successes}/{completed} ({(successes / completed * 100) if completed else 0.0:.2f}%), "
        f"task_errors={errors}",
        flush=True,
    )
    return 0


def run_worker(args: argparse.Namespace) -> int:
    if args.eval_world_size < 1 or not (0 <= args.eval_rank < args.eval_world_size):
        raise ValueError("eval_rank must satisfy 0 <= eval_rank < eval_world_size")
    if args.open_loop_steps > args.chunk_size:
        raise ValueError("open_loop_steps must be <= chunk_size")

    output_dir = Path(args.output_dir)
    progress_path = Path(args.progress_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {args.eval_rank}] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / f"rank{args.eval_rank}.log"),
        ],
        force=True,
    )

    manifest, suites = _build_manifest(
        args.task_suites, args.classification_path, args.max_tasks
    )
    tasks = manifest[args.eval_rank :: args.eval_world_size]
    total_for_rank = len(tasks)
    task_id_set = {int(record["global_task_id"]) for record in tasks}
    jsonl_path = output_dir / f"rank{args.eval_rank}.jsonl"
    records_by_id = (
        _load_resume_records(jsonl_path, task_id_set, args.seed, args.eval_rank)
        if args.resume
        else {}
    )
    if not args.resume and jsonl_path.exists():
        _atomic_text(jsonl_path, "")

    started_at = time.time()
    completed = len(records_by_id)
    resumed_tasks = completed
    successes = sum(int(bool(record.get("success"))) for record in records_by_id.values())
    task_errors = sum(
        int(record.get("error") is not None) for record in records_by_id.values()
    )
    _write_progress(
        progress_path,
        seed=args.seed,
        rank=args.eval_rank,
        world_size=args.eval_world_size,
        total=total_for_rank,
        completed=completed,
        successes=successes,
        task_errors=task_errors,
        success_rate=successes / completed if completed else 0.0,
        category_stats=_category_progress(records_by_id.values()),
        status="starting",
        resumed_tasks=resumed_tasks,
        started_at=started_at,
        updated_at=started_at,
    )
    if completed == total_for_rank:
        return _finalize_rank(
            args,
            output_dir,
            progress_path,
            jsonl_path,
            records_by_id,
            total_for_rank,
            started_at,
        )
    if args.max_task_errors > 0 and task_errors >= args.max_task_errors:
        raise RuntimeError(
            f"Resume data already has {task_errors} task errors (limit={args.max_task_errors})"
        )

    import torch
    from cosmos_policy.experiments.robot.cosmos_utils import get_model, load_dataset_stats
    from cosmos_policy.utils.utils import set_seed_everywhere

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; a CUDA GPU is required")
    torch.cuda.set_device(0)
    set_seed_everywhere(args.seed)
    dataset_stats = load_dataset_stats(args.dataset_stats)
    cfg = _make_cfg(args)
    model, cosmos_config = get_model(cfg)
    expected_chunk = getattr(getattr(cosmos_config, "dataloader_train", None), "dataset", None)
    expected_chunk = getattr(expected_chunk, "chunk_size", args.chunk_size)
    if int(expected_chunk) != int(args.chunk_size):
        raise ValueError(
            f"chunk_size mismatch: checkpoint expects {expected_chunk}, requested {args.chunk_size}"
        )
    model.eval()
    embedding_store = T5EmbeddingStore(args.t5_cache)

    _write_progress(
        progress_path,
        seed=args.seed,
        rank=args.eval_rank,
        world_size=args.eval_world_size,
        total=total_for_rank,
        completed=completed,
        successes=successes,
        task_errors=task_errors,
        success_rate=successes / completed if completed else 0.0,
        category_stats=_category_progress(records_by_id.values()),
        status="running",
        resumed_tasks=resumed_tasks,
        started_at=started_at,
        updated_at=time.time(),
    )

    try:
        with jsonl_path.open("a", encoding="utf-8") as jsonl:
            for shard_index, task_record in enumerate(tasks, start=1):
                global_task_id = int(task_record["global_task_id"])
                if global_task_id in records_by_id:
                    continue
                embedding = embedding_store.get(task_record["instruction"])
                success, num_steps, error, attempts = _evaluate_task(
                    args,
                    task_record,
                    suites[task_record["task_suite"]],
                    embedding,
                    cfg,
                    model,
                    dataset_stats,
                )
                del embedding

                record = {
                    **task_record,
                    "seed": args.seed,
                    "eval_rank": args.eval_rank,
                    "eval_world_size": args.eval_world_size,
                    "episode_idx": 0,
                    "initial_state_idx": 0,
                    "environment_seed": args.environment_seed,
                    "success": bool(success),
                    "num_steps": int(num_steps),
                    "attempts": int(attempts),
                    "error": error,
                }
                records_by_id[global_task_id] = record
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()

                completed += 1
                successes += int(success)
                task_errors += int(error is not None)
                rate = successes / completed
                _write_progress(
                    progress_path,
                    seed=args.seed,
                    rank=args.eval_rank,
                    world_size=args.eval_world_size,
                    total=total_for_rank,
                    completed=completed,
                    successes=successes,
                    task_errors=task_errors,
                    success_rate=rate,
                    category_stats=_category_progress(records_by_id.values()),
                    last_global_task_id=global_task_id,
                    last_task_suite=task_record["task_suite"],
                    last_task=task_record["task_name"],
                    last_success=bool(success),
                    status="running",
                    resumed_tasks=resumed_tasks,
                    started_at=started_at,
                    updated_at=time.time(),
                )
                print(
                    f"[seed={args.seed} rank={args.eval_rank}] "
                    f"[{shard_index}/{total_for_rank}] "
                    f"{'OK' if success else 'FAIL'} {task_record['task_suite']}/"
                    f"{task_record['task_id']} {task_record['column']} | SR={rate * 100:.2f}%",
                    flush=True,
                )

                if error is not None and "out of memory" in error.lower():
                    raise RuntimeError(f"CUDA out of memory on global task {global_task_id}: {error}")
                if args.max_task_errors > 0 and task_errors >= args.max_task_errors:
                    raise RuntimeError(
                        f"Reached MAX_TASK_ERRORS={args.max_task_errors}; latest error={error}"
                    )
    finally:
        embedding_store.close()

    return _finalize_rank(
        args,
        output_dir,
        progress_path,
        jsonl_path,
        records_by_id,
        total_for_rank,
        started_at,
    )


def _rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def _seed_markdown(seed: int, summary: dict[str, Any]) -> str:
    values = [summary["categories"][column]["success_rate"] * 100 for column in COLUMNS]
    total_rate = summary["success_rate"] * 100
    row = [str(seed), *[f"{value:.1f}" for value in values], f"{total_rate:.1f}", str(summary["total_tasks"])]
    lines = [
        f"# Cosmos Policy LIBERO-plus seed {seed} summary",
        "",
        "| Seed | Camera | Robot | Language | Light | Background | Noise | Layout | Total | Evaluated tasks |",
        "|-----:|-------:|------:|---------:|------:|-----------:|------:|-------:|------:|----------------:|",
        "| " + " | ".join(row) + " |",
        "",
        "Counts:",
    ]
    for column in COLUMNS:
        item = summary["categories"][column]
        lines.append(f"{column}: {item['successes']}/{item['total']}")
    lines.append(f"Total: {summary['successes']}/{summary['total_tasks']}")
    lines.append(f"Task errors counted as failures: {summary['task_errors']}")
    lines.append("")
    return "\n".join(lines)


def summarize_seed(args: argparse.Namespace) -> int:
    seed_dir = Path(args.output_dir)
    rank_files = sorted(seed_dir.glob("rank*.json"))
    if not rank_files:
        raise FileNotFoundError(f"No rank result files under {seed_dir}")
    if args.eval_world_size > 0 and len(rank_files) != args.eval_world_size:
        raise RuntimeError(
            f"Seed {args.seed}: found {len(rank_files)} rank files, expected {args.eval_world_size}"
        )

    records: list[dict[str, Any]] = []
    ranks: set[int] = set()
    for path in rank_files:
        with path.open("r", encoding="utf-8") as f:
            rank_result = json.load(f)
        if int(rank_result["seed"]) != args.seed:
            raise ValueError(f"Seed mismatch in {path}")
        rank = int(rank_result["rank"])
        if rank in ranks:
            raise ValueError(f"Duplicate rank {rank}")
        ranks.add(rank)
        records.extend(rank_result.get("records", []))

    by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        global_task_id = int(record["global_task_id"])
        if global_task_id in by_id:
            raise ValueError(f"Duplicate global task id {global_task_id}")
        by_id[global_task_id] = record
    records = [by_id[key] for key in sorted(by_id)]
    if args.expected_tasks > 0 and len(records) != args.expected_tasks:
        raise RuntimeError(
            f"Seed {args.seed}: got {len(records)} task records, expected {args.expected_tasks}"
        )

    category_counts = {column: [0, 0] for column in COLUMNS}
    suite_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    successes = 0
    task_errors = 0
    for record in records:
        column = CATEGORY_TO_COLUMN.get(record.get("category"))
        if column is None:
            raise ValueError(f"Unknown category in result: {record.get('category')!r}")
        success = int(bool(record.get("success")))
        category_counts[column][0] += success
        category_counts[column][1] += 1
        suite_counts[record["task_suite"]][0] += success
        suite_counts[record["task_suite"]][1] += 1
        successes += success
        task_errors += int(record.get("error") is not None)

    categories = {
        column: {
            "category": COLUMN_TO_CATEGORY[column],
            "successes": category_counts[column][0],
            "total": category_counts[column][1],
            "success_rate": _rate(category_counts[column][0], category_counts[column][1]),
        }
        for column in COLUMNS
    }
    per_suite = {
        suite: {
            "successes": values[0],
            "total": values[1],
            "success_rate": _rate(values[0], values[1]),
        }
        for suite, values in sorted(suite_counts.items())
    }
    summary = {
        "model": "Cosmos Policy LIBERO Predict2 2B",
        "benchmark": "LIBERO-plus",
        "seed": args.seed,
        "total_tasks": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "task_errors": task_errors,
        "success_rate": _rate(successes, len(records)),
        "success_rate_percent": _rate(successes, len(records)) * 100,
        "categories": categories,
        "per_suite": per_suite,
        "rank_files": [str(path) for path in rank_files],
    }
    _atomic_json(seed_dir / "summary.json", summary)
    markdown = _seed_markdown(args.seed, summary)
    _atomic_text(seed_dir / "summary.md", markdown)
    _atomic_text(
        seed_dir / "all_results.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    print(markdown, end="", flush=True)
    return 0


def _final_markdown(seeds: list[int], summaries: dict[int, dict[str, Any]], final: dict[str, Any]) -> str:
    lines = [
        "# Cosmos Policy LIBERO-plus three-seed summary",
        "",
        "| Seed | Camera | Robot | Language | Light | Background | Noise | Layout | Total | Evaluated tasks |",
        "|-----:|-------:|------:|---------:|------:|-----------:|------:|-------:|------:|----------------:|",
    ]
    for seed in seeds:
        item = summaries[seed]
        values = [item["categories"][column]["success_rate"] * 100 for column in COLUMNS]
        row = [
            str(seed),
            *[f"{value:.1f}" for value in values],
            f"{item['success_rate'] * 100:.1f}",
            str(item["total_tasks"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    mean_values = [final["category_means"][column] * 100 for column in COLUMNS]
    mean_row = [
        "**Mean**",
        *[f"**{value:.1f}**" for value in mean_values],
        f"**{final['three_seed_average'] * 100:.1f}**",
        "—",
    ]
    lines.append("| " + " | ".join(mean_row) + " |")
    lines.extend(["", "Counts:"])
    for seed in seeds:
        item = summaries[seed]
        lines.append(
            f"Seed {seed}: {item['successes']}/{item['total_tasks']} "
            f"({item['failures']} failures; {item['task_errors']} task errors)"
        )
    lines.extend(
        [
            "",
            "The mean is the arithmetic mean of the unrounded per-seed success rates, giving each seed equal weight.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_final(args: argparse.Namespace) -> int:
    root = Path(args.result_root)
    seeds = [int(seed) for seed in _split_words(args.seeds)]
    if not seeds:
        raise ValueError("No seeds provided")
    summaries: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = root / f"seed{seed}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed summary: {path}")
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        if int(item["seed"]) != seed:
            raise ValueError(f"Seed mismatch in {path}")
        if args.expected_tasks > 0 and int(item["total_tasks"]) != args.expected_tasks:
            raise RuntimeError(
                f"Seed {seed}: {item['total_tasks']} tasks, expected {args.expected_tasks}"
            )
        summaries[seed] = item

    category_means = {
        column: sum(item["categories"][column]["success_rate"] for item in summaries.values())
        / len(summaries)
        for column in COLUMNS
    }
    mean_rate = sum(item["success_rate"] for item in summaries.values()) / len(summaries)
    pooled_successes = sum(int(item["successes"]) for item in summaries.values())
    pooled_tasks = sum(int(item["total_tasks"]) for item in summaries.values())
    final = {
        "model": "Cosmos Policy LIBERO Predict2 2B",
        "benchmark": "LIBERO-plus",
        "result_root": str(root),
        "seeds": seeds,
        "tasks_per_seed": args.expected_tasks,
        "per_seed": {str(seed): summaries[seed] for seed in seeds},
        "category_means": category_means,
        "three_seed_average": mean_rate,
        "three_seed_average_percent": mean_rate * 100,
        "pooled": {
            "successes": pooled_successes,
            "tasks": pooled_tasks,
            "success_rate": _rate(pooled_successes, pooled_tasks),
            "success_rate_percent": _rate(pooled_successes, pooled_tasks) * 100,
        },
    }
    markdown = _final_markdown(seeds, summaries, final)
    _atomic_json(root / "summary.json", final)
    _atomic_text(root / "summary.md", markdown)
    if args.publish_summary:
        publish_path = Path(args.publish_summary)
        if publish_path.resolve() != (root / "summary.md").resolve():
            _atomic_text(publish_path, markdown)
    print(markdown, end="", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manifest-json", action="store_true")
    mode.add_argument("--prepare-t5", action="store_true")
    mode.add_argument("--check-t5-cache", action="store_true")
    mode.add_argument("--summarize-seed", action="store_true")
    mode.add_argument("--summarize-final", action="store_true")

    parser.add_argument("--task-suites", default=" ".join(DEFAULT_TASK_SUITES))
    parser.add_argument("--classification-path", default="")
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH", ""))

    parser.add_argument("--t5-cache", default=os.environ.get("LIBERO_PLUS_T5_CACHE", ""))
    parser.add_argument("--base-t5-cache", default=os.environ.get("LIBERO_T5_EMBEDDINGS", ""))
    parser.add_argument("--t5-model-dir", default=os.environ.get("COSMOS_T5_MODEL_DIR", ""))
    parser.add_argument("--t5-tokenizer-dir", default=os.environ.get("COSMOS_T5_TOKENIZER_DIR", ""))
    parser.add_argument("--t5-device", default="cuda:0")
    parser.add_argument("--t5-batch-size", type=int, default=16)
    parser.add_argument("--t5-log-every", type=int, default=100)
    parser.add_argument("--t5-commit-every", type=int, default=100)
    parser.add_argument("--t5-lock-timeout", type=float, default=86400.0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", default="1 7 42")
    parser.add_argument("--eval-rank", type=int, default=0)
    parser.add_argument("--eval-world-size", type=int, default=1)
    parser.add_argument("--expected-tasks", type=int, default=-1)
    parser.add_argument("--output-dir", default="./results/libero_plus")
    parser.add_argument("--result-root", default="./results/libero_plus")
    parser.add_argument("--progress-file", default="")
    parser.add_argument("--publish-summary", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--config", default="cosmos_predict2_2b_480p_libero__inference_only")
    parser.add_argument("--ckpt-path", default=os.environ.get("COSMOS_POLICY_CHECKPOINT", ""))
    parser.add_argument("--config-file", default="cosmos_policy/config/config.py")
    parser.add_argument("--dataset-stats", default=os.environ.get("COSMOS_DATASET_STATS", ""))
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--open-loop-steps", type=int, default=16)
    parser.add_argument("--num-denoising-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--max-task-errors", type=int, default=20)
    parser.add_argument("--flip-images", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.task_suites = _split_words(args.task_suites)
    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("DETERMINISTIC", "True")

    if args.summarize_seed:
        return summarize_seed(args)
    if args.summarize_final:
        return summarize_final(args)
    if not args.libero_config_path:
        raise ValueError("--libero-config-path (or LIBERO_CONFIG_PATH) is required")
    if args.max_tasks == 0 or args.max_tasks < -1:
        raise ValueError("--max-tasks must be -1 or a positive integer")
    if args.manifest_json:
        manifest, _ = _build_manifest(
            args.task_suites, args.classification_path, args.max_tasks
        )
        print(json.dumps(_manifest_summary(manifest), ensure_ascii=False, sort_keys=True))
        return 0
    if args.prepare_t5:
        required = {
            "t5_cache": args.t5_cache,
            "t5_model_dir": args.t5_model_dir,
            "t5_tokenizer_dir": args.t5_tokenizer_dir,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing T5 preparation arguments: {', '.join(missing)}")
        return prepare_t5_cache(args)
    if args.check_t5_cache:
        if not args.t5_cache:
            raise ValueError("--t5-cache is required")
        return check_t5_cache(args)

    required = {
        "t5_cache": args.t5_cache,
        "ckpt_path": args.ckpt_path,
        "dataset_stats": args.dataset_stats,
        "progress_file": args.progress_file,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing worker arguments: {', '.join(missing)}")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or a positive integer")
    if args.task_retries < 0 or args.max_task_errors < 0:
        raise ValueError("task retries/errors must be non-negative")
    try:
        return run_worker(args)
    except Exception as exc:
        try:
            progress_path = Path(args.progress_file)
            current: dict[str, Any] = {}
            if progress_path.is_file():
                try:
                    current = json.loads(progress_path.read_text(encoding="utf-8"))
                except Exception:
                    current = {}
            current.update(
                {
                    "seed": args.seed,
                    "rank": args.eval_rank,
                    "world_size": args.eval_world_size,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": time.time(),
                }
            )
            _atomic_json(progress_path, current)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(1) from exc
