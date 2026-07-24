#!/usr/bin/env python3
"""Measure FastWAM LIBERO policy-call latency on one real LIBERO observation.

The benchmark deliberately does not call ``env.reset`` or ``env.step`` inside
the timed region.  It uses the same model loader and action adapter as the
repository evaluator, then reports both call latency and latency amortized over
the configured replan interval.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/afs/zhengmingkai/raozf/models/fastwam_libero")
    parser.add_argument("--output", default="latency_benchmark/fastwam.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    from FastWAM import eval_libero as ev

    out_dir = Path(args.output).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ev._add_project_paths(ev.DEFAULT_FASTWAM_ROOT, ev.DEFAULT_LIBERO_ROOT)
    ev._configure_libero_paths(ev.DEFAULT_LIBERO_ROOT, out_dir)

    # Imports that depend on LIBERO_CONFIG_PATH must happen after the config is
    # installed.  The evaluator itself follows the same ordering.
    from libero.libero import benchmark

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(0)
    model_dir = Path(args.model_path).resolve()
    model, config, dtype = ev._build_model(model_dir, device)
    state_stats, action_stats = ev._load_normalizers(model_dir)

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    env, task_description = ev._make_env(task, seed=1)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[0])

        def sync() -> None:
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        # Force allocator/model kernels to settle before collecting samples.
        for i in range(args.warmup):
            sync()
            ev._predict_action(
                model,
                obs,
                task_description,
                state_stats,
                action_stats,
                seed=1000 + i,
                action_horizon=int(config.get("action_horizon", 32)),
                num_inference_steps=int(config.get("num_inference_steps", 10)),
                device=device,
                dtype=dtype,
                binarize_gripper=True,
            )
            sync()

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        samples: list[float] = []
        for i in range(args.samples):
            sync()
            start = time.perf_counter()
            action = ev._predict_action(
                model,
                obs,
                task_description,
                state_stats,
                action_stats,
                seed=2000 + i,
                action_horizon=int(config.get("action_horizon", 32)),
                num_inference_steps=int(config.get("num_inference_steps", 10)),
                device=device,
                dtype=dtype,
                binarize_gripper=True,
            )
            sync()
            samples.append(time.perf_counter() - start)

        payload = {
            "model": "FastWAM",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": str(args.gpu),
            "device": device,
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "warmup": args.warmup,
            "samples": len(samples),
            "action_horizon": int(config.get("action_horizon", 32)),
            "replan_steps": args.replan_steps,
            "num_inference_steps": int(config.get("num_inference_steps", 10)),
            "dtype": str(dtype),
            "task_description": str(task_description),
            "action_shape": list(action.shape),
            "call_seconds": samples,
            "call_mean_ms": statistics.mean(samples) * 1000,
            "call_median_ms": statistics.median(samples) * 1000,
            "call_p95_ms": percentile(samples, 0.95) * 1000,
            "latency_per_executed_step_ms": statistics.mean(samples) * 1000 / args.replan_steps,
            "latency_per_executed_step_median_ms": statistics.median(samples) * 1000 / args.replan_steps,
            "torch_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
            ),
            "torch_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20 if device.startswith("cuda") else None
            ),
        }
    finally:
        env.close()
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
