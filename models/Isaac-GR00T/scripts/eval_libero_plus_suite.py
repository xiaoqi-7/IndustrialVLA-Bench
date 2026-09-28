#!/usr/bin/env python3
"""Evaluate one GR00T LIBERO-plus suite and summarize the 7 robustness columns."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CATEGORY_TO_COLUMN = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
COLUMNS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
SUITE_ALIASES = {
    "10": "libero_10",
    "libero_10": "libero_10",
    "goal": "libero_goal",
    "libero_goal": "libero_goal",
    "object": "libero_object",
    "libero_object": "libero_object",
    "spatial": "libero_spatial",
    "libero_spatial": "libero_spatial",
}


def _rate(successes: int, total: int) -> float:
    return 100.0 * successes / total if total else 0.0


def _load_tasks(classification_path: Path, suite: str) -> list[dict[str, Any]]:
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)
    if suite not in classification:
        raise KeyError(f"Suite {suite!r} not in {classification_path}. Available: {sorted(classification)}")
    return classification[suite]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summarize(records: list[dict[str, Any]], suite: str, model_name: str, summary_path: Path) -> None:
    stats = {column: [0, 0] for column in COLUMNS}
    total_successes = 0
    total_episodes = 0
    missing_category = 0

    for record in records:
        column = CATEGORY_TO_COLUMN.get(record.get("category"))
        if column is None:
            missing_category += 1
            continue
        successes = int(record["successes"])
        episodes = int(record["episodes"])
        stats[column][0] += successes
        stats[column][1] += episodes
        total_successes += successes
        total_episodes += episodes

    lines = [
        "| Model | Suite | Camera | Robot | Language | Light | Background | Noise | Layout | Total |",
        "|-------|-------|--------|-------|----------|-------|------------|-------|--------|-------|",
    ]
    values = [_rate(*stats[column]) for column in COLUMNS]
    total = _rate(total_successes, total_episodes)
    lines.append(
        "| "
        + " | ".join([model_name, suite, *[f"{value:.1f}" for value in values], f"{total:.1f}"])
        + " |"
    )
    lines.append("")
    lines.append("Counts:")
    for column in COLUMNS:
        successes, episodes = stats[column]
        lines.append(f"{column}: {successes}/{episodes}")
    lines.append(f"Total: {total_successes}/{total_episodes}")
    if missing_category:
        lines.append(f"Warning: skipped {missing_category} records without LIBERO-plus category.")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=sorted(SUITE_ALIASES), help="One suite only.")
    parser.add_argument("--classification-path", required=True, type=Path)
    parser.add_argument("--result-jsonl", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--model-name", default="GR00T")
    parser.add_argument("--eval-rank", type=int, default=0)
    parser.add_argument("--eval-world-size", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only-first-task", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=8)
    return parser.parse_args()


def _run_gr00t_sim_policy(
    *,
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    policy_client_host: str,
    policy_client_port: int,
    n_envs: int,
    n_action_steps: int,
    video_dir: str | None,
    seed: int | None,
):
    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        VideoConfig,
        WrapperConfigs,
        create_gr00t_sim_policy,
        run_rollout_gymnasium_policy,
    )
    from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
    from gr00t.utils.determinism import seed_everything

    seed = seed_everything(seed)
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=video_dir, max_episode_steps=max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    policy = create_gr00t_sim_policy(
        model_path="",
        embodiment_tag=embodiment_tag,
        policy_client_host=policy_client_host,
        policy_client_port=policy_client_port,
    )
    return run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
    )


def main() -> None:
    args = _parse_args()
    suite = SUITE_ALIASES[args.suite]

    if args.summarize_only:
        _summarize(_read_jsonl(args.result_jsonl), suite=suite, model_name=args.model_name, summary_path=args.summary_path)
        return

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    tasks = _load_tasks(args.classification_path, suite)
    if args.only_first_task:
        tasks = tasks[:1]
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    tasks = tasks[args.eval_rank :: args.eval_world_size]

    args.result_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.result_jsonl.write_text("", encoding="utf-8")

    records: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        task_name = task["name"]
        env_name = f"libero_sim/{task_name}"
        task_video_dir = None
        if args.save_video and args.video_dir is not None:
            task_video_dir = str(args.video_dir / suite / f"{task_index:05d}_{task_name[:120]}")

        print(
            f"[INFO] rank={args.eval_rank}/{args.eval_world_size} ({task_index + 1}/{len(tasks)}) "
            f"suite={suite} category={task['category']} env={env_name}",
            flush=True,
        )
        _, episode_successes, episode_infos = _run_gr00t_sim_policy(
            env_name=env_name,
            n_episodes=args.n_episodes,
            max_episode_steps=args.max_episode_steps,
            policy_client_host=args.policy_client_host,
            policy_client_port=args.policy_client_port,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            video_dir=task_video_dir,
            seed=args.seed,
        )

        successes = int(np.sum(episode_successes))
        episodes = int(len(episode_successes))
        record = {
            "suite": suite,
            "task_id": task.get("id"),
            "task_index": task_index,
            "task_name": task_name,
            "env_name": env_name,
            "category": task.get("category"),
            "column": CATEGORY_TO_COLUMN.get(task.get("category")),
            "difficulty_level": task.get("difficulty_level"),
            "successes": successes,
            "episodes": episodes,
            "success_rate": _rate(successes, episodes),
            "episode_successes": [bool(value) for value in episode_successes],
            "episode_lengths": [int(value) for value in episode_infos.get("episode_lengths", [])],
            "episode_rewards": [float(value) for value in episode_infos.get("episode_rewards", [])],
        }
        _write_jsonl(args.result_jsonl, record)
        records.append(record)
        print(
            f"[RESULT] task={task_name} category={task['category']} success={successes}/{episodes} "
            f"rate={record['success_rate']:.1f}",
            flush=True,
        )

    if not args.skip_summary:
        _summarize(records, suite=suite, model_name=args.model_name, summary_path=args.summary_path)


if __name__ == "__main__":
    main()
