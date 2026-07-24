import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import socket
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # Multi-process evaluation parameters
    #################################################################################################################
    # Each eval process only evaluates a slice of LIBERO tasks:
    # task_ids = all_task_ids[eval_rank::eval_world_size]
    eval_rank: int = _env_int("RANK", _env_int("LOCAL_RANK", 0))
    eval_world_size: int = _env_int("WORLD_SIZE", 1)

    # If True, rank k connects to port + k. This matches one policy server per GPU.
    port_offset_by_rank: bool = True

    # Allows starting main.py before serve_policy.py. Each eval process waits until its server port is ready.
    server_wait_timeout_s: float = 1800.0
    server_wait_poll_s: float = 2.0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_video: bool = True
    result_out_path: str | None = None  # Optional JSONL path for per-episode results

    seed: int = 1  # Random Seed (for reproducibility)


def _wait_for_server(host: str, port: int, timeout_s: float, poll_s: float) -> None:
    """Wait until the websocket policy server is reachable.

    Use a real websocket handshake instead of a raw TCP connection. A raw TCP
    probe makes the server print "did not receive a valid HTTP request".
    """
    import websockets.sync.client

    uri = f"ws://{host}:{port}"
    deadline = time.time() + timeout_s
    last_error: Exception | None = None

    logging.info("Waiting for policy server at %s ...", uri)
    while time.time() < deadline:
        try:
            with websockets.sync.client.connect(
                uri,
                open_timeout=3.0,
                close_timeout=1.0,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
            ):
                logging.info("Policy server is reachable at %s", uri)
                return
        except Exception as exc:
            last_error = exc
            time.sleep(poll_s)

    raise TimeoutError(
        f"Timed out after {timeout_s:.1f}s waiting for policy server at {uri}. "
        f"Last error: {last_error}"
    )


def eval_libero(args: Args) -> None:
    # Set random seed. Offset by eval rank so each process is deterministic but distinct.
    np.random.seed(args.seed + args.eval_rank)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    category_by_task_name = _load_libero_plus_categories(args.task_suite_name)

    video_out_path = pathlib.Path(args.video_out_path) / f"rank_{args.eval_rank}"
    video_out_path.mkdir(parents=True, exist_ok=True)

    result_file = None
    if args.result_out_path:
        result_out_path = pathlib.Path(args.result_out_path)
        result_out_path.parent.mkdir(parents=True, exist_ok=True)
        result_file = result_out_path.open("a", encoding="utf-8")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    elif args.task_suite_name in {"libero_100", "libero_mix"}:
        max_steps = 520  # LIBERO-plus suites contain LIBERO-10-style long-horizon tasks
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_port = args.port + args.eval_rank if args.port_offset_by_rank else args.port

    all_task_ids = list(range(num_tasks_in_suite))
    task_ids = all_task_ids[args.eval_rank :: args.eval_world_size]

    logging.info(
        "[Eval Split] rank=%s, world_size=%s, client=%s:%s, task_ids=%s",
        args.eval_rank,
        args.eval_world_size,
        args.host,
        client_port,
        task_ids,
    )

    if len(task_ids) == 0:
        logging.info("[Eval Split] rank=%s has no tasks. Exiting.", args.eval_rank)
        return

    _wait_for_server(args.host, client_port, args.server_wait_timeout_s, args.server_wait_poll_s)

    def make_client():
        logging.info("[Rank %s] Connecting to policy server %s:%s", args.eval_rank, args.host, client_port)
        return _websocket_client_policy.WebsocketClientPolicy(args.host, client_port)

    client = make_client()

    def safe_infer(element, max_retries: int = 2):
        """Run one policy inference with reconnect.

        The original code keeps a single WebSocket connection for the whole
        evaluation. If that connection is closed once (for example, keepalive
        ping timeout), every following episode reuses the broken client and
        becomes False. This function drops the broken connection, reconnects,
        and retries the current inference.
        """
        nonlocal client
        last_error: Exception | None = None

        for retry_idx in range(max_retries + 1):
            try:
                return client.infer(element)
            except Exception as exc:
                last_error = exc
                logging.warning(
                    "[Rank %s] client.infer failed, retry=%s/%s: %r",
                    args.eval_rank,
                    retry_idx,
                    max_retries,
                    exc,
                )

                # The WebSocket object is already broken after ConnectionClosedError.
                # Close it if possible, then build a fresh client.
                try:
                    if hasattr(client, "_ws"):
                        client._ws.close()
                except Exception:
                    pass

                if retry_idx < max_retries:
                    time.sleep(1.0 + retry_idx)
                    client = make_client()

        raise last_error

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids, desc=f"rank{args.eval_rank}: tasks"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + args.eval_rank)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task),
            desc=f"rank{args.eval_rank}: task{task_id}",
            leave=False,
        ):
            logging.info("\nTask %s: %s", task_id, task_description)

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []

            logging.info("Starting task=%s episode=%s ...", task_id, task_episodes + 1)
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall.
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing.
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk.
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = safe_infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.exception("Caught exception on rank=%s task=%s episode=%s: %s", args.eval_rank, task_id, episode_idx, e)
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if args.save_video and replay_images:
                suffix = "success" if done else "failure"
                task_segment = _safe_filename(task_description)
                imageio.mimwrite(
                    video_out_path / f"task{task_id}_episode{episode_idx}_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info("Success: %s", done)
            logging.info("# episodes completed on this rank so far: %s", total_episodes)
            logging.info("# successes on this rank: %s (%.1f%%)", total_successes, total_successes / total_episodes * 100)

            if result_file is not None:
                result_file.write(
                    json.dumps(
                        {
                            "task_suite": args.task_suite_name,
                            "eval_rank": args.eval_rank,
                            "task_id": task_id,
                            "task_name": task.name,
                            "category": category_by_task_name.get(task.name),
                            "episode_idx": episode_idx,
                            "success": bool(done),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                result_file.flush()

        # Log task results
        logging.info("Current task success rate on rank %s: %s", args.eval_rank, float(task_successes) / float(task_episodes))
        logging.info("Current rank success rate: %s", float(total_successes) / float(total_episodes))

        # Close env if supported by the installed LIBERO/robosuite version.
        if hasattr(env, "close"):
            env.close()

    logging.info("Rank %s total success rate: %s", args.eval_rank, float(total_successes) / float(total_episodes))
    logging.info("Rank %s total episodes: %s", args.eval_rank, total_episodes)

    if result_file is not None:
        result_file.close()


def _load_libero_plus_categories(task_suite_name: str) -> dict[str, str]:
    """Loads LIBERO-plus perturbation category labels when available."""
    classification_path = pathlib.Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    if not classification_path.exists():
        return {}

    with classification_path.open("r", encoding="utf-8") as f:
        task_classification = json.load(f)

    return {
        item["name"]: item["category"]
        for item in task_classification.get(task_suite_name, [])
    }


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is close to a zero-degree rotation, immediately return.
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _safe_filename(text: str, max_len: int = 120) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.replace(" ", "_"))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe[:max_len].strip("_")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    eval_libero(tyro.cli(Args))
