#!/usr/bin/env python3
"""Measure Xiaomi-Robotics-0 on one real LIBERO observation."""

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
    p.add_argument("--model-path", default="/mnt/afs/zhengmingkai/raozf/models/Xiaomi-Robotics-0-LIBERO")
    p.add_argument("--libero-root", default="/mnt/afs/zhengmingkai/raozf/benchmark/LIBERO")
    p.add_argument("--output", default="latency_benchmark/xiaomi.json")
    p.add_argument("--gpu", default="2")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--replan-steps", type=int, default=10)
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    import sys

    repo = Path(__file__).resolve().parents[1] / "Xiaomi-Robotics-0"
    sys.path.insert(0, str(repo))
    from deploy.server import _load_processor

    # Configure LIBERO before importing its path resolver.
    root = Path(args.libero_root).resolve() / "libero" / "libero"
    config_dir = Path(args.output).resolve().parent / "xiaomi_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"assets: {root / 'assets'}",
                f"datasets: {Path(args.output).resolve().parent / 'xiaomi_libero_datasets'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (Path(args.output).resolve().parent / "xiaomi_libero_datasets").mkdir(exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    model_path = Path(args.model_path).resolve()
    processor = _load_processor(str(model_path))
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).cuda().to(torch.bfloat16).eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(1)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        base = Image.fromarray(img.astype(np.uint8))
        wrist_left = Image.fromarray(wrist.astype(np.uint8))
        q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        qw = float(np.clip(q[3], -1.0, 1.0))
        den = float(np.sqrt(max(1e-12, 1.0 - qw * qw)))
        rpy = np.zeros(3, dtype=np.float64) if abs(qw) >= 0.999999 else q[:3] * (2.0 * np.arccos(qw)) / den
        state = np.concatenate([obs["robot0_eef_pos"], rpy, obs["robot0_gripper_qpos"], np.zeros(24)])
        language = str(task.language).capitalize() + "."
        robot_type = "libero_all"
        instruction = (
            "<|im_start|>user\nThe following observations are captured from multiple views.\n"
            "# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n"
            "# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n"
            f"Generate robot actions for the task:\n{language} /no_cot<|im_end|>\n"
            "<|im_start|>assistant\n<cot></cot><|im_end|>\n"
        )

        def run_once() -> np.ndarray:
            vl_inputs = processor(
                text=[instruction], images=[base, wrist_left], videos=None, padding=True, return_tensors="pt"
            ).to(model.device)
            data = dict(vl_inputs)
            data["task_id"] = robot_type
            data["seed"] = 42
            data["action_mask"] = processor.get_action_mask(robot_type).to(model.device, model.dtype)
            data["state"] = torch.from_numpy(state).to(model.device, model.dtype).view(1, 1, -1)
            with torch.inference_mode():
                outputs = model(**data)
            return processor.decode_action(outputs.actions, robot_type=robot_type).detach().cpu().numpy()

        for _ in range(args.warmup):
            torch.cuda.synchronize()
            run_once()
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        shape = None
        for _ in range(args.samples):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = run_once()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
            shape = list(out.shape)

        payload = {
            "model": "Xiaomi-Robotics-0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": str(args.gpu),
            "device": device,
            "task_suite": "libero_spatial",
            "task_id": 0,
            "warmup": args.warmup,
            "samples": len(samples),
            # The LIBERO processor's action_config for ``libero_all`` is
            # (10, 32), so the evaluator actually receives 10 actions even
            # though the generic config.json advertises action_length=30.
            "action_horizon": int(shape[-2]) if shape and len(shape) >= 2 else None,
            "configured_action_length": 30,
            "replan_steps": args.replan_steps,
            "num_inference_steps": 5,
            "action_shape": shape,
            "call_seconds": samples,
            "call_mean_ms": statistics.mean(samples) * 1000,
            "call_median_ms": statistics.median(samples) * 1000,
            "call_p95_ms": percentile(samples, 0.95) * 1000,
            "latency_per_executed_step_ms": statistics.mean(samples) * 1000 / args.replan_steps,
            "latency_per_executed_step_median_ms": statistics.median(samples) * 1000 / args.replan_steps,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
    finally:
        env.close()
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
