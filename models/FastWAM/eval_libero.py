#!/usr/bin/env python3
"""FastWAM LIBERO evaluator.

This evaluator is intentionally independent of Hydra and of the LeRobot CLI.  The
checkpoint supplied for this benchmark is a LeRobot ``model.safetensors`` file,
while the checkout in ``FastWAM/src`` contains the reference FastWAM model.  The
small compatibility loader below bridges those two formats and loads the Wan
sidecar files from the same model directory.

The script is normally started by ``run_fastwam_libero.sh``.  One process is
started per selected GPU and each process loads the policy once, then evaluates
its deterministic round-robin task shard.  It can also be run directly for a
single worker (useful for debugging).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Keep imports of the fairly large model package out of the module import path
# until after command-line validation.  These paths are also used when this file
# is launched from a directory other than benchmark/FastWAM.
HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = HERE.parent
DEFAULT_FASTWAM_ROOT = BENCHMARK_ROOT / "FastWAM"
DEFAULT_LIBERO_ROOT = BENCHMARK_ROOT / "LIBERO"

LOG = logging.getLogger("fastwam_libero")

SEEDS = (1, 7, 42)
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}


def _add_project_paths(fastwam_root: Path, libero_root: Path) -> None:
    """Make the local reference FastWAM and LIBERO checkouts importable."""

    for path in (fastwam_root / "src", fastwam_root, libero_root):
        path = path.resolve()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # The copied LeRobot archive is useful to callers that want to inspect the
    # policy files, but older snapshots do not contain the fastwam policy.  Add
    # it only as a fallback import location; the reference implementation above
    # remains the source of truth for this local checkpoint.
    lerobot_zip = os.environ.get("LEROBOT_ZIP")
    if lerobot_zip and Path(lerobot_zip).is_file() and lerobot_zip not in sys.path:
        sys.path.append(lerobot_zip)


def _configure_libero_paths(libero_root: Path, output_dir: Path) -> None:
    """Point LIBERO's global path resolver at the requested checkout.

    LIBERO reads ``$LIBERO_CONFIG_PATH/config.yaml`` at import time.  A copied
    environment often has a stale ``~/.libero/config.yaml`` from another
    benchmark, so relying on that default can silently evaluate the wrong BDDL
    and initial-state files.  Write a small run-local config before importing
    ``libero.libero``.
    """

    import tempfile

    root = libero_root.resolve() / "libero" / "libero"
    datasets_dir = output_dir.resolve() / "libero_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "assets": root / "assets",
        "datasets": datasets_dir,
    }
    missing = [str(path) for path in (required["bddl_files"], required["init_states"], required["assets"]) if not path.is_dir()]
    if missing:
        raise FileNotFoundError("LIBERO checkout is incomplete; missing: " + ", ".join(missing))

    config_dir = Path(
        os.environ.get("FASTWAM_LIBERO_CONFIG_PATH", str(output_dir / "libero_config"))
    ).expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    # Avoid rewriting a config another worker has just written; all workers use
    # the same absolute paths for a given run.
    if not config_path.is_file():
        lines = [f"{key}: {path}\n" for key, path in required.items()]
        fd, temporary = tempfile.mkstemp(prefix="config.", suffix=".tmp", dir=config_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, config_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import numpy as np
    import torch

    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _parse_suites(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    else:
        values = [str(item) for item in value]
    suites = [item.strip() for item in values if item.strip()]
    if not suites:
        raise ValueError("At least one LIBERO task suite is required")
    unknown = sorted(set(suites) - set(SUITE_MAX_STEPS))
    if unknown:
        raise ValueError(f"Unknown LIBERO suite(s): {', '.join(unknown)}")
    return suites


def _task_list(suites: list[str], max_tasks: int = -1) -> list[tuple[str, int]]:
    # Import lazily so --dry-run and --help do not initialize MuJoCo.
    from libero.libero import benchmark

    result: list[tuple[str, int]] = []
    available = benchmark.get_benchmark_dict()
    for suite_name in suites:
        if suite_name not in available:
            raise ValueError(f"LIBERO benchmark does not provide suite {suite_name!r}")
        suite = available[suite_name]()
        result.extend((suite_name, task_id) for task_id in range(int(suite.n_tasks)))
    if max_tasks >= 0:
        result = result[:max_tasks]
    if not result:
        raise ValueError("The selected task list is empty")
    return result


def _dtype_from_name(name: str):
    import torch

    key = str(name).lower().replace("torch.", "")
    table = {"float32": torch.float32, "float": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if key not in table:
        raise ValueError(f"Unsupported model torch_dtype {name!r}; expected float32, float16 or bfloat16")
    return table[key]


def _filter_constructor_kwargs(cls: type, config: dict[str, Any]) -> dict[str, Any]:
    """Drop keys introduced by newer LeRobot configs but absent in reference code."""

    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    accepted.discard("self")
    return {key: value for key, value in config.items() if key in accepted}


def _stream_load_safetensors(
    module: Any,
    path: Path,
    *,
    key_transform=None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a safetensors file one tensor at a time.

    ``safetensors.torch.load_file`` temporarily keeps a complete CPU copy of a
    12-GB checkpoint.  With one model copy per GPU that is needlessly expensive,
    so this loader uses ``safe_open`` and copies each tensor directly to its
    already allocated destination.
    """

    from safetensors import safe_open

    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    targets = {**parameters, **buffers}
    seen: set[str] = set()
    unexpected: list[str] = []
    mismatched: list[str] = []

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for source_key in handle.keys():
            target_key = key_transform(source_key) if key_transform is not None else source_key
            # A single LeRobot checkpoint contains both ``model.mot.*`` and
            # ``model.proprio_encoder.*``.  Callers can return ``None`` for the
            # portion that belongs to another module.
            if target_key is None:
                continue
            if target_key not in targets:
                unexpected.append(source_key)
                continue
            target = targets[target_key]
            shape = tuple(handle.get_slice(source_key).get_shape())
            if tuple(target.shape) != shape:
                mismatched.append(f"{source_key}: checkpoint={shape}, model={tuple(target.shape)}")
                continue
            tensor = handle.get_tensor(source_key)
            with __import__("torch").no_grad():
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            seen.add(target_key)

    missing = sorted(set(targets) - seen)
    if mismatched:
        raise RuntimeError(f"Shape mismatch while loading {path}: {'; '.join(mismatched[:8])}")
    if strict and (missing or unexpected):
        # Some reference modules expose non-persistent helper buffers.  They are
        # harmless, but a missing trainable parameter is not.
        missing_parameters = sorted(set(parameters) - seen)
        if missing_parameters or unexpected:
            details = []
            if missing_parameters:
                details.append(f"missing={missing_parameters[:8]}")
            if unexpected:
                details.append(f"unexpected={unexpected[:8]}")
            raise RuntimeError(f"Could not load {path}: " + ", ".join(details))
    return missing, unexpected


def _checkpoint_path(model_dir: Path) -> Path:
    if model_dir.is_file():
        return model_dir
    path = model_dir / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {path}")
    return path


def _build_model(model_dir: Path, device: str):
    """Build the reference FastWAM model and load the local LeRobot snapshot."""

    import torch
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38

    if model_dir.is_file():
        model_dir = model_dir.parent
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"FastWAM config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dtype = _dtype_from_name(config.get("torch_dtype", "bfloat16"))

    video_cfg = _filter_constructor_kwargs(WanVideoDiT, dict(config["video_dit_config"]))
    action_cfg = _filter_constructor_kwargs(ActionDiT, dict(config["action_dit_config"]))
    mot_checkpoint = bool(config.get("mot_checkpoint_mixed_attn", False))

    # Construct directly in the checkpoint dtype.  Constructing a 5B module in
    # fp32 and converting it afterwards can briefly require another ~20 GB RAM.
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        video_expert = WanVideoDiT(**video_cfg)
        action_expert = ActionDiT(**action_cfg)
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint,
        )
        text_encoder = WanTextEncoder(
            vocab=256384,
            dim=int(video_cfg.get("text_dim", 4096)),
            dim_attn=int(video_cfg.get("text_dim", 4096)),
        )
        vae = WanVideoVAE38()
    finally:
        torch.set_default_dtype(old_dtype)

    tokenizer_path = model_dir / "google" / "umt5-xxl"
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Local UMT5 tokenizer not found: {tokenizer_path}")
    tokenizer = HuggingfaceTokenizer(
        name=str(tokenizer_path),
        seq_len=int(config.get("tokenizer_max_len", 128)),
        clean="whitespace",
    )

    model = FastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_dim=int(video_cfg.get("text_dim", 4096)),
        proprio_dim=int(config.get("proprio_dim", 8)) if config.get("proprio_dim") is not None else None,
        device=device,
        torch_dtype=dtype,
        video_train_shift=float(config.get("video_scheduler", {}).get("train_shift", 5.0)),
        video_infer_shift=float(config.get("video_scheduler", {}).get("infer_shift", 5.0)),
        video_num_train_timesteps=int(config.get("video_scheduler", {}).get("num_train_timesteps", 1000)),
        action_train_shift=float(config.get("action_scheduler", {}).get("train_shift", 5.0)),
        action_infer_shift=float(config.get("action_scheduler", {}).get("infer_shift", 5.0)),
        action_num_train_timesteps=int(config.get("action_scheduler", {}).get("num_train_timesteps", 1000)),
        loss_lambda_video=float(config.get("loss", {}).get("lambda_video", 1.0)),
        loss_lambda_action=float(config.get("loss", {}).get("lambda_action", 1.0)),
    ).eval()

    text_path = model_dir / "models_t5_umt5-xxl-enc-bf16.safetensors"
    vae_path = model_dir / "Wan2.2_VAE.safetensors"
    if not text_path.is_file() or not vae_path.is_file():
        raise FileNotFoundError(
            "The local checkpoint must include models_t5_umt5-xxl-enc-bf16.safetensors "
            "and Wan2.2_VAE.safetensors beside model.safetensors."
        )

    LOG.info("Loading UMT5 sidecar: %s", text_path)
    _stream_load_safetensors(text_encoder, text_path, strict=True)
    LOG.info("Loading Wan VAE sidecar: %s", vae_path)
    _stream_load_safetensors(
        vae,
        vae_path,
        key_transform=lambda key: key if key.startswith("model.") else "model." + key,
        strict=True,
    )

    checkpoint = _checkpoint_path(model_dir)
    if checkpoint.suffix == ".safetensors":
        LOG.info("Loading policy weights: %s", checkpoint)
        # The LeRobot wrapper stores the trainable core under ``model.mot`` and
        # ``model.proprio_encoder``.  The reference FastWAM object exposes those
        # two modules without the leading ``model.``.
        _stream_load_safetensors(
            model.mot,
            checkpoint,
            key_transform=lambda key: key.removeprefix("model.mot.")
            if key.startswith("model.mot.")
            else None,
            strict=True,
        )
        if model.proprio_encoder is not None:
            _stream_load_safetensors(
                model.proprio_encoder,
                checkpoint,
                key_transform=lambda key: key.removeprefix("model.proprio_encoder.")
                if key.startswith("model.proprio_encoder.")
                else None,
                strict=True,
            )
    else:
        # Backward-compatible path for the original FastWAM .pt release file.
        model.load_checkpoint(str(checkpoint))

    model.eval()
    return model, config, dtype


class MinMaxStats:
    """The min/max normalizer used by the FastWAM LeRobot processor."""

    def __init__(self, minimum, maximum):
        import torch

        self.minimum = torch.as_tensor(minimum, dtype=torch.float32)
        self.maximum = torch.as_tensor(maximum, dtype=torch.float32)
        self.range = self.maximum - self.minimum
        self.ignore = self.range < 1.0e-4

    def forward(self, value):
        import torch

        value = torch.as_tensor(value, dtype=torch.float32)
        scale = torch.where(self.ignore, torch.ones_like(self.range), 2.0 / self.range)
        # This matches ``SingleFieldLinearNormalizer`` used by the training
        # processor: constant dimensions are translated to zero rather than
        # being left in their original coordinate system.
        offset = torch.where(self.ignore, -self.minimum, -1.0 - scale * self.minimum)
        return torch.clamp(value * scale + offset, -5.0, 5.0)

    def backward(self, value):
        import torch

        value = torch.as_tensor(value, dtype=torch.float32)
        raw = (value + 1.0) * 0.5 * self.range + self.minimum
        return torch.where(self.ignore, value + self.minimum, raw)


def _load_normalizers(model_dir: Path) -> tuple[MinMaxStats, MinMaxStats]:
    if model_dir.is_file():
        model_dir = model_dir.parent
    stats_path = model_dir / "libero_uncond_2cam224_dataset_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Dataset statistics not found: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    def bounds(kind: str):
        entry = stats[kind].get("default", stats[kind])
        minimum = entry.get("global_min", entry.get("min"))
        maximum = entry.get("global_max", entry.get("max"))
        if minimum is None or maximum is None:
            raise ValueError(f"No global min/max statistics for {kind} in {stats_path}")
        # The dataset writer may retain a leading singleton time dimension.
        if isinstance(minimum, list) and len(minimum) == 1 and isinstance(minimum[0], list):
            minimum = minimum[0]
        if isinstance(maximum, list) and len(maximum) == 1 and isinstance(maximum[0], list):
            maximum = maximum[0]
        return minimum, maximum

    state_min, state_max = bounds("state")
    action_min, action_max = bounds("action")
    return MinMaxStats(state_min, state_max), MinMaxStats(action_min, action_max)


def _center_crop_resize(image, width: int, height: int):
    from PIL import Image
    import numpy as np

    array = np.asarray(image)
    # LIBERO-Plus sensor corruptions may return float RGB arrays.  Its values
    # are normally either [0, 1] or [0, 255], while standard LIBERO returns
    # uint8.  Normalize all three cases without destroying a [0, 1] image by
    # casting it directly to zeros and ones.
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0 + 1.0e-6:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(array)
    src_w, src_h = pil.size
    scale = max(width / src_w, height / src_h)
    resized = pil.resize((round(src_w * scale), round(src_h * scale)), Image.Resampling.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def _quat2axisangle(quat):
    import math
    import numpy as np

    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3])))) / den


def _extract_state(obs: dict[str, Any]):
    import numpy as np

    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


def _images_to_tensor(obs: dict[str, Any], device: str, dtype):
    import numpy as np
    import torch

    # LIBERO renders upside down relative to the training recordings.
    primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    primary = _center_crop_resize(primary, 224, 224)
    wrist = _center_crop_resize(wrist, 224, 224)
    image = np.concatenate((primary, wrist), axis=1)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # The LeRobot visual normalizer has mean=.5 and std=.5.  The reference
    # FastWAM model consumes the equivalent [-1, 1] representation directly.
    return (tensor.to(device=device, dtype=dtype) / 255.0) * 2.0 - 1.0


def _predict_action(
    model,
    obs: dict[str, Any],
    task_description: str,
    state_stats: MinMaxStats,
    action_stats: MinMaxStats,
    *,
    seed: int,
    action_horizon: int,
    num_inference_steps: int,
    device: str,
    dtype,
    binarize_gripper: bool,
):
    import numpy as np
    import torch

    state = state_stats.forward(_extract_state(obs)).unsqueeze(0).to(device=device)
    image = _images_to_tensor(obs, device=device, dtype=dtype)
    prompt = "A video recorded from a robot's point of view executing the following instruction: " + str(
        task_description
    )
    with torch.inference_mode():
        prediction = model.infer_action(
            prompt=prompt,
            input_image=image,
            action_horizon=action_horizon,
            proprio=state,
            negative_prompt="",
            text_cfg_scale=1.0,
            num_inference_steps=num_inference_steps,
            sigma_shift=None,
            seed=int(seed),
            rand_device="cpu",
            tiled=False,
        )["action"]

    # Model output is normalized action.  Undo the LeRobot min/max transform,
    # convert gripper [0,1] to LIBERO [-1,1], then restore LIBERO's sign.
    action = action_stats.backward(prediction).cpu().numpy().astype(np.float32)
    action[..., -1] = action[..., -1] * 2.0 - 1.0
    action[..., -1] *= -1.0
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _make_env(task, seed: int, resolution: int = 256):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(int(seed))
    return env, task.language


def _env_step(env, action):
    result = env.step(action)
    if len(result) == 4:
        return result
    # Newer Gym-style wrappers return terminated/truncated separately.
    obs, reward, terminated, truncated, info = result
    return obs, reward, bool(terminated or truncated), info


def _dummy_action():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _run_task(
    model,
    task,
    initial_states,
    *,
    suite_name: str,
    task_id: int,
    seed: int,
    num_trials: int,
    num_steps_wait: int,
    replan_steps: int,
    action_horizon: int,
    num_inference_steps: int,
    state_stats: MinMaxStats,
    action_stats: MinMaxStats,
    device: str,
    dtype,
    binarize_gripper: bool,
    max_steps_override: int | None,
    episode_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    max_steps = int(max_steps_override or SUITE_MAX_STEPS[suite_name])
    env, task_description = _make_env(task, seed)
    successes: list[int] = []
    failures: list[int] = []
    selected_episodes = list(range(num_trials)) if episode_indices is None else [int(index) for index in episode_indices]
    if any(index < 0 or index >= num_trials for index in selected_episodes):
        raise ValueError(f"Episode indices must be in [0, {num_trials}); got {selected_episodes}")
    try:
        states = list(initial_states)
        if not states:
            raise RuntimeError(f"Task {suite_name}/{task_id} has no initial states")
        for episode_idx in selected_episodes:
            env.reset()
            obs = env.set_init_state(states[episode_idx % len(states)])
            pending: list[list[float]] = []
            done = False
            t = 0
            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    obs, _, done, _ = _env_step(env, _dummy_action())
                    t += 1
                    if done:
                        break
                    continue

                if not pending:
                    chunk = _predict_action(
                        model,
                        obs,
                        task_description,
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
                t += 1
                if done:
                    break
            if done:
                successes.append(episode_idx)
            else:
                failures.append(episode_idx)
            LOG.info(
                "seed=%s suite=%s task=%s episode=%s success=%s",
                seed,
                suite_name,
                task_id,
                episode_idx,
                bool(done),
            )
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()

    return {
        "suite": suite_name,
        "task_id": int(task_id),
        "task_description": str(task_description),
        "successes": len(successes),
        "episodes": len(selected_episodes),
        "success_episodes": successes,
        "failure_episodes": failures,
    }


def evaluate_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from libero.libero import benchmark

    _set_seed(args.seed)
    suites = _parse_suites(args.task_suites)
    tasks = _task_list(suites, args.max_tasks)
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank must be in [0, {args.world_size}), got {args.rank}")
    assigned = tasks[args.rank :: args.world_size]
    if not assigned:
        raise ValueError(f"Worker rank {args.rank} received no tasks")

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
        "worker rank=%s/%s physical_gpu=%s device=%s tasks=%s model=%s",
        args.rank,
        args.world_size,
        args.gpu_id,
        device,
        len(assigned),
        model_dir,
    )
    model, model_config, dtype = _build_model(model_dir, device)
    state_stats, action_stats = _load_normalizers(model_dir)

    available = benchmark.get_benchmark_dict()
    task_results = []
    started = time.time()
    for suite_name, task_id in assigned:
        suite = available[suite_name]()
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        task_results.append(
            _run_task(
                model,
                task,
                initial_states,
                suite_name=suite_name,
                task_id=task_id,
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
        )

    successes = sum(int(item["successes"]) for item in task_results)
    episodes = sum(int(item["episodes"]) for item in task_results)
    result = {
        "format": "fastwam-libero-worker-v1",
        "seed": int(args.seed),
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "gpu_id": str(args.gpu_id),
        "model_path": str(model_dir),
        "task_suites": suites,
        "num_trials_per_task": int(args.num_trials),
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
    workers = []
    for rank in range(expected_workers):
        path = output_dir / f"worker_{rank:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing worker result for rank {rank}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("seed")) != seed:
            raise ValueError(f"Seed mismatch in {path}: {payload.get('seed')} != {seed}")
        workers.append(payload)

    tasks: list[dict[str, Any]] = []
    for worker in workers:
        tasks.extend(worker.get("tasks", []))
    tasks.sort(key=lambda item: (str(item["suite"]), int(item["task_id"])))
    successes = sum(int(item["successes"]) for item in tasks)
    episodes = sum(int(item["episodes"]) for item in tasks)
    suites: dict[str, dict[str, Any]] = {}
    for item in tasks:
        suite = str(item["suite"])
        aggregate = suites.setdefault(suite, {"successes": 0, "episodes": 0, "tasks": 0})
        aggregate["successes"] += int(item["successes"])
        aggregate["episodes"] += int(item["episodes"])
        aggregate["tasks"] += 1
    for aggregate in suites.values():
        aggregate["success_rate"] = aggregate["successes"] / aggregate["episodes"] if aggregate["episodes"] else 0.0

    result = {
        "format": "fastwam-libero-seed-v1",
        "seed": int(seed),
        "workers": expected_workers,
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "suites": suites,
        "tasks": tasks,
    }
    _atomic_json_dump(output_dir / "summary.json", result)
    return result


def aggregate_three_seed(output_root: Path, seeds: Iterable[int]) -> dict[str, Any]:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("At least one seed summary is required")
    seed_results = []
    for seed in seed_list:
        path = output_root / f"seed{seed}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed summary: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if int(result.get("seed")) != seed:
            raise ValueError(f"Seed mismatch in {path}")
        seed_results.append(result)

    pooled_successes = sum(int(result["successes"]) for result in seed_results)
    pooled_episodes = sum(int(result["episodes"]) for result in seed_results)
    rates = [
        int(result["successes"]) / int(result["episodes"]) if int(result["episodes"]) else 0.0
        for result in seed_results
    ]
    suite_names = sorted({name for result in seed_results for name in result.get("suites", {})})
    suite_summary = {}
    for suite in suite_names:
        suite_results = [result["suites"][suite] for result in seed_results if suite in result.get("suites", {})]
        suite_rates = [
            int(item["successes"]) / int(item["episodes"]) if int(item["episodes"]) else 0.0
            for item in suite_results
        ]
        suite_pooled_successes = sum(int(item["successes"]) for item in suite_results)
        suite_pooled_episodes = sum(int(item["episodes"]) for item in suite_results)
        suite_summary[suite] = {
            "seed_rates": {
                str(result["seed"]): (
                    int(result["suites"][suite]["successes"])
                    / int(result["suites"][suite]["episodes"])
                    if int(result["suites"][suite]["episodes"])
                    else 0.0
                )
                for result in seed_results
                if suite in result.get("suites", {})
            },
            "mean_success_rate": sum(suite_rates) / len(suite_rates),
            "pooled_successes": suite_pooled_successes,
            "pooled_episodes": suite_pooled_episodes,
            "pooled_success_rate": suite_pooled_successes / suite_pooled_episodes if suite_pooled_episodes else 0.0,
            "status": "complete",
        }

    result = {
        "format": "fastwam-libero-three-seed-v1",
        "seeds": seed_list,
        "seed_results": seed_results,
        # Requested metric: arithmetic mean of the three independent seed rates.
        "mean_success_rate": sum(rates) / len(rates),
        # Also report the episode-weighted pooled rate for auditability.
        "pooled_successes": pooled_successes,
        "pooled_episodes": pooled_episodes,
        "pooled_success_rate": pooled_successes / pooled_episodes if pooled_episodes else 0.0,
        "suites": suite_summary,
    }
    _atomic_json_dump(output_root / "three_seed_summary.json", result)

    # Keep a compact, human-facing summary of the four LIBERO subtasks next to
    # the full aggregate.  ``three_seed_summary.json`` intentionally retains
    # the detailed per-seed payload for auditing; this file is the convenient
    # table-level result requested by the evaluation workflow.
    four_task_summary = {
        "format": "fastwam-libero-four-task-summary-v1",
        "status": "complete",
        "expected_seeds": seed_list,
        "included_seeds": seed_list,
        "missing_seeds": [],
        "seeds": seed_list,
        "aggregation_policy": "Only complete seed summary.json files are included; partial worker outputs are excluded.",
        "overall": {
            "mean_success_rate": sum(rates) / len(rates),
            "pooled_successes": pooled_successes,
            "pooled_episodes": pooled_episodes,
            "pooled_success_rate": pooled_successes / pooled_episodes if pooled_episodes else 0.0,
        },
        "subtasks": suite_summary,
        # ``suites`` mirrors the key used by each seed's summary.json.
        "suites": suite_summary,
    }
    _atomic_json_dump(output_root / "four_task_summary.json", four_task_summary)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="FastWAM LeRobot checkpoint directory")
    parser.add_argument("--libero-root", required=True, help="LIBERO checkout containing libero/libero")
    parser.add_argument("--output-dir", required=True, help="Worker/seed output directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--task-suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0 (worker sees one visible GPU)")
    parser.add_argument("--no-binarize-gripper", dest="binarize_gripper", action="store_false")
    parser.set_defaults(binarize_gripper=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print task sharding without loading weights")
    parser.add_argument("--aggregate-workers", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-workers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--aggregate-three-seed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS), help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    fastwam_root = Path(os.environ.get("FASTWAM_SOURCE_ROOT", DEFAULT_FASTWAM_ROOT)).expanduser()
    libero_root = Path(args.libero_root).expanduser().resolve()
    _add_project_paths(fastwam_root, libero_root)
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _configure_libero_paths(libero_root, Path(args.output_dir).expanduser().resolve())

    if args.aggregate_three_seed:
        root = Path(args.output_root or args.output_dir).expanduser().resolve()
        result = aggregate_three_seed(root, args.seeds)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
        return
    if args.aggregate_workers:
        seed_dir = Path(args.output_dir).expanduser().resolve()
        result = aggregate_seed(seed_dir, expected_workers=int(args.expected_workers), seed=int(args.seed))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
        return

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if args.num_trials <= 0 or args.world_size <= 0:
        raise ValueError("--num-trials and --world-size must be positive")
    suites = _parse_suites(args.task_suites)
    tasks = _task_list(suites, args.max_tasks)
    assigned = tasks[args.rank :: args.world_size]
    if args.dry_run:
        print(json.dumps({
            "seed": args.seed,
            "rank": args.rank,
            "world_size": args.world_size,
            "tasks_total": len(tasks),
            "tasks_assigned": len(assigned),
            "task_suites": suites,
            "model_path": str(model_path),
            "libero_root": str(libero_root),
        }, separators=(",", ":")))
        return
    evaluate_worker(args)


if __name__ == "__main__":
    main()
