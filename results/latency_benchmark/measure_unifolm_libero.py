#!/usr/bin/env python3
"""Measure UnifoLM-VLA's actual LIBERO policy call latency."""

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


def make_config_dir(output: Path, libero_root: Path) -> Path:
    config_dir = output.resolve().parent / "unifolm_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    root = libero_root.resolve() / "libero" / "libero"
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"assets: {root / 'assets'}",
                f"datasets: {output.resolve().parent / 'unifolm_libero_datasets'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output.resolve().parent / "unifolm_libero_datasets").mkdir(exist_ok=True)
    return config_dir


def build_qwen_inputs(observations, task_description, model):
    """Build the same two-frame Qwen input as experiments/LIBERO/eval_libero.py."""
    import numpy as np
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    all_images = []
    for observation in observations:
        all_images.append(observation["full_image"])
    for observation in observations:
        all_images.append(observation["wrist_image"])
    all_images = [Image.fromarray(np.asarray(image)).convert("RGB") for image in all_images]
    text = (
        'You are a robot using the joint control. The task is "'
        + task_description.lower()
        + '". Please predict up to 10 key trajectory points to complete the task. '
        + "Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], "
        + "where each tuple contains the x and y coordinates of a point."
    )
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in all_images],
                {"type": "text", "text": text},
            ],
        }
    ]
    text = model.vla.qwen_vl_interface.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    qwen_inputs = model.vla.qwen_vl_interface.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    qwen_inputs["state"] = np.stack([obs["state"] for obs in observations], axis=0)
    return qwen_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/mnt/afs/zhengmingkai/raozf/models/UnifoLM/UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt",
    )
    parser.add_argument(
        "--vlm-path",
        default="/mnt/afs/zhengmingkai/raozf/models/UnifoLM/UnifoLM-VLM-Base",
    )
    parser.add_argument("--libero-root", default="/mnt/afs/zhengmingkai/raozf/benchmark/LIBERO-plus")
    parser.add_argument("--output", default="latency_benchmark/unifolm.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output = Path(args.output)
    config_dir = make_config_dir(output, Path(args.libero_root))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    import numpy as np
    import torch

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from unifolm_vla.rlds_dataloader.constants import NUM_ACTIONS_CHUNK
    from unifolm_vla.model.framework.share_tools import read_mode_config
    from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
    from unifolm_vla.model.modules.vlm.QWen2_5 import _QWen_VL_Interface

    # Reuse the evaluator's image conventions, but avoid importing its CLI.
    import cv2 as cv
    from PIL import Image

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(0),
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
    )
    env.seed(1)
    env.reset()
    raw = env.set_init_state(suite.get_task_init_states(0)[0])

    def resize_image(image):
        return cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)

    def make_observation(raw_observation):
        base = raw_observation["agentview_image"][::-1, ::-1]
        wrist = raw_observation["robot0_eye_in_hand_image"][::-1, ::-1]
        return {
            "full_image": resize_image(base),
            "wrist_image": resize_image(wrist),
            "state": np.concatenate(
                (
                    np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32),
                    quat2axisangle(raw_observation["robot0_eef_quat"]),
                    np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32),
                )
            ),
        }

    def quat2axisangle(quat):
        import math

        quat = np.asarray(quat, dtype=np.float64).copy()
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(float(den), 0.0):
            return np.zeros(3, dtype=np.float32)
        return ((quat[:3] * 2.0 * math.acos(float(quat[3])) / den)).astype(np.float32)

    observations = [make_observation(raw) for _ in range(args.window_size)]
    model = __import__(
        "experiments.LIBERO.unifolm_vla_inference",
        fromlist=["Unifolm_VLA_Inference"],
    ).Unifolm_VLA_Inference(
        policy_ckpt_path=args.model_path,
        unnorm_key=args.unnorm_key,
        image_size=[224, 224],
        use_bf16=True,
        vlm_pretrained_path=args.vlm_path,
    )
    model.reset(task.language)

    def infer_once():
        qwen_inputs = build_qwen_inputs(observations, task.language, model)
        return model.step(qwen_inputs)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        sync()
        infer_once()
        sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    samples = []
    action_shape = None
    for _ in range(args.samples):
        sync()
        start = time.perf_counter()
        actions = infer_once()
        sync()
        samples.append(time.perf_counter() - start)
        action_shape = list(np.asarray(actions).shape)

    payload = {
        "model": "UnifoLM-VLA",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": str(args.gpu),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "task_description": task.language,
        "warmup": args.warmup,
        "samples": len(samples),
        # ``predict_action`` returns (T, D) after removing the batch axis.
        "action_horizon": int(action_shape[0]) if action_shape else int(NUM_ACTIONS_CHUNK),
        "replan_steps": args.replan_steps,
        "window_size": args.window_size,
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
    env.close()
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
