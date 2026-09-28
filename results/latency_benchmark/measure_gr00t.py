#!/usr/bin/env python3
"""Measure the GR00T LIBERO policy endpoint on a real LIBERO observation."""

from __future__ import annotations

import argparse
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/mnt/afs/zhengmingkai/raozf/models/Gr00t-N1.7-libero/libero_spatial")
    p.add_argument("--libero-root", default="/mnt/afs/zhengmingkai/raozf/benchmark/LIBERO-plus")
    p.add_argument("--output", default="latency_benchmark/gr00t.json")
    p.add_argument("--gpu", default="0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--replan-steps", type=int, default=8)
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config_dir = Path(args.output).resolve().parent / "gr00t_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.libero_root).resolve() / "libero" / "libero"
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"assets: {root / 'assets'}",
                f"datasets: {Path(args.output).resolve().parent / 'gr00t_libero_datasets'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (Path(args.output).resolve().parent / "gr00t_libero_datasets").mkdir(exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    import numpy as np
    import torch
    import gymnasium as gym

    # Importing the local GR00T package registers the model class with HF.
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    env_name = "libero_sim/" + "libero_spatial"  # replaced below by the first registered task
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    env_name = "libero_sim/" + task.name
    env = gym.make(env_name)
    policy = Gr00tSimPolicyWrapper(
        Gr00tPolicy(
            embodiment_tag="LIBERO_PANDA",
            model_path=args.model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            strict=True,
        )
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    try:
        obs, _ = env.reset(seed=1)
        # Keep the timed calls on this fixed valid observation; no reset/step
        # is performed in the timed region.  The wrapper's public endpoint
        # expects a batch dimension.
        flat = {}
        for key, value in obs.items():
            if key.startswith("video."):
                flat[key] = np.asarray(value)[None, None, ...]
            elif key.startswith("state."):
                flat[key] = np.asarray(value, dtype=np.float32)[None, None, ...]
            else:
                flat[key] = [value]

        def sync() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        for _ in range(args.warmup):
            sync()
            policy.get_action(flat)
            sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        samples = []
        action_shape = None
        for _ in range(args.samples):
            sync()
            start = time.perf_counter()
            action, _ = policy.get_action(flat)
            sync()
            samples.append(time.perf_counter() - start)
            if action:
                action_shape = {k: list(v.shape) for k, v in action.items()}

        actual_action_horizon = (
            next(iter(action_shape.values()))[1] if action_shape else None
        )
        payload = {
            "model": "GR00T N1.7",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": str(args.gpu),
            "device": device,
            "task_suite": "libero_spatial",
            "task_id": 0,
            "warmup": args.warmup,
            "samples": len(samples),
            # The checkpoint's generic max horizon is 40, but the LIBERO
            # embodiment processor emits 16 actions per policy call.
            "action_horizon": actual_action_horizon,
            "configured_max_action_horizon": 40,
            "replan_steps": args.replan_steps,
            "num_inference_steps": 4,
            "action_shape": action_shape,
            "call_seconds": samples,
            "call_mean_ms": statistics.mean(samples) * 1000,
            "call_median_ms": statistics.median(samples) * 1000,
            "call_p95_ms": percentile(samples, 0.95) * 1000,
            "latency_per_executed_step_ms": statistics.mean(samples) * 1000 / args.replan_steps,
            "latency_per_executed_step_median_ms": statistics.median(samples) * 1000 / args.replan_steps,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None,
        }
    finally:
        env.close()
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
