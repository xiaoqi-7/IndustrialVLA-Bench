#!/usr/bin/env python3
"""Measure OpenPI pi0.5 on one real LIBERO observation.

The timed region is the same policy endpoint used by the LIBERO evaluator:
observation transforms, tokenization, the ten-step flow sampler, and output
transforms are included; environment reset and env.step are not.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def quaternion_to_axis_angle(quaternion):
    import math
    import numpy as np

    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    result = quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator
    return result.astype(np.float32)


def build_policy_observation(raw_observation, instruction: str, resize_size: int):
    """Match ``scripts/eval_libero_para.py`` without importing that script."""
    import numpy as np
    from openpi_client import image_tools

    def to_uint8(image):
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image, dtype=np.uint8)

    base_image = to_uint8(raw_observation["agentview_image"][::-1, ::-1])
    wrist_image = to_uint8(raw_observation["robot0_eye_in_hand_image"][::-1, ::-1])
    base_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(base_image, resize_size, resize_size)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
    )
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


def make_config_dir(output: Path, libero_root: Path) -> Path:
    config_dir = output.resolve().parent / "openpi_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    root = libero_root.resolve() / "libero" / "libero"
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"assets: {root / 'assets'}",
                f"datasets: {output.resolve().parent / 'openpi_libero_datasets'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output.resolve().parent / "openpi_libero_datasets").mkdir(exist_ok=True)
    return config_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/mnt/afs/zhengmingkai/raozf/models/pi05_libero_pytorch")
    parser.add_argument("--libero-root", default="/mnt/afs/zhengmingkai/raozf/benchmark/LIBERO-plus")
    parser.add_argument("--output", default="latency_benchmark/openpi.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    args = parser.parse_args()

    # Set this before importing torch/JAX/LIBERO so each process sees one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    # The OpenPI environment contains an optional TensorFlow install whose
    # binary is incompatible with this MetaX runtime; Transformers does not
    # need it for the PyTorch checkpoint.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")

    import numpy as np
    import torch
    print("[openpi] imported numpy/torch", flush=True)

    output = Path(args.output)
    config_dir = make_config_dir(output, Path(args.libero_root))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.policies import policy_config
    from openpi.training import config as train_config
    print("[openpi] imported LIBERO/openpi modules", flush=True)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl_path = suite.get_task_bddl_file_path(0)
    init_states = suite.get_task_init_states(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
    )
    env.seed(1)
    env.reset()
    raw_observation = env.set_init_state(init_states[0])
    print("[openpi] created/reset LIBERO env", flush=True)
    # Use the exact LIBERO evaluator observation construction, including image
    # flips/padding and the 7-D eef state.
    policy_observation = build_policy_observation(raw_observation, task.language, 224)
    print("[openpi] built observation", flush=True)

    cfg = train_config.get_config("pi05_libero")
    # MetaX does not benefit from TorchDynamo compilation and compilation can
    # introduce a large one-time graph-capture cost into a microbenchmark.
    model_cfg = dataclasses.replace(cfg.model, pytorch_compile_mode=None)
    cfg = dataclasses.replace(cfg, model=model_cfg)
    print("[openpi] loading policy", flush=True)
    policy = policy_config.create_trained_policy(
        cfg,
        args.model_dir,
        pytorch_device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print("[openpi] policy loaded", flush=True)

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        sync()
        policy.infer(policy_observation)
        sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    samples: list[float] = []
    action_shape = None
    policy_model_ms: list[float] = []
    for _ in range(args.samples):
        sync()
        start = time.perf_counter()
        result = policy.infer(policy_observation)
        sync()
        samples.append(time.perf_counter() - start)
        action_shape = list(np.asarray(result["actions"]).shape)
        if isinstance(result.get("policy_timing"), dict):
            policy_model_ms.append(float(result["policy_timing"]["infer_ms"]))

    payload = {
        "model": "OpenPI π0.5",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": str(args.gpu),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "task_description": task.language,
        "warmup": args.warmup,
        "samples": len(samples),
        "action_horizon": int(cfg.model.action_horizon),
        "replan_steps": args.replan_steps,
        "num_inference_steps": args.num_inference_steps,
        "action_shape": action_shape,
        "call_seconds": samples,
        "call_mean_ms": statistics.mean(samples) * 1000,
        "call_median_ms": statistics.median(samples) * 1000,
        "call_p95_ms": percentile(samples, 0.95) * 1000,
        "model_infer_mean_ms": statistics.mean(policy_model_ms) if policy_model_ms else None,
        "latency_per_executed_step_ms": statistics.mean(samples) * 1000 / args.replan_steps,
        "latency_per_executed_step_median_ms": statistics.median(samples) * 1000 / args.replan_steps,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None,
    }
    env.close()
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
