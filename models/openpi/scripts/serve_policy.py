import dataclasses
import enum
import logging
import os
import random
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str

    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO

    # If provided, will be used in case the "prompt" key is not present in the data,
    # or if the model doesn't have a default prompt.
    default_prompt: str | None = None

    # Base port to serve the policy on.
    port: int = 8000

    # Host to bind. 0.0.0.0 is useful when client and server are in different containers.
    host: str = "0.0.0.0"

    # Multi-server parameters. server_rank k uses port + k when port_offset_by_rank=True.
    server_rank: int = _env_int("RANK", _env_int("LOCAL_RANK", 0))
    server_world_size: int = _env_int("WORLD_SIZE", 1)
    port_offset_by_rank: bool = True

    # Explicit PyTorch device. Use cuda:0, cuda:1, ... for multi-GPU.
    # If omitted, this script will set it to cuda:<server_rank> when CUDA/MACA is available.
    pytorch_device: str | None = None

    # Fail fast if CUDA/MACA is unavailable, instead of silently running on CPU.
    require_cuda: bool = True

    # Seed used by PyTorch flow-matching noise sampling.
    seed: int = 1

    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        # Local checkpoint path. Make sure this directory contains assets/physical-intelligence/libero/norm_stats.json.
        dir="/mnt/afs/zhengmingkai/raozf/models/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                pytorch_device=args.pytorch_device,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                pytorch_device=args.pytorch_device,
            )


def _setup_pytorch_device(args: Args) -> None:
    """Set and verify the PyTorch device before loading the model."""
    try:
        import torch  # noqa: PLC0415 - torch is an optional dependency for JAX-only servers.
    except ImportError as exc:
        if args.require_cuda:
            raise RuntimeError("torch is not installed, cannot run policy on GPU.") from exc
        args.pytorch_device = args.pytorch_device or "cpu"
        return

    if os.environ.get("TORCH_COMPILE_DISABLE", "1") == "1":
        # MetaX supports PyTorch eager CUDA-compatible kernels, but not NVIDIA Triton.
        torch._dynamo.config.disable = True  # noqa: SLF001
        logging.info("torch.compile disabled; using PyTorch eager GPU inference")

    logging.info("torch version: %s", torch.__version__)
    logging.info("torch file: %s", torch.__file__)
    logging.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    logging.info("torch.cuda.device_count(): %s", torch.cuda.device_count())

    if args.pytorch_device is None:
        if torch.cuda.is_available():
            # For 8 independent server processes, rank k uses cuda:k.
            args.pytorch_device = f"cuda:{args.server_rank}"
        else:
            args.pytorch_device = "cpu"

    logging.info("Using pytorch_device=%s", args.pytorch_device)

    if args.require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA/MACA is not available in this Python environment. "
                "The policy server would run on CPU. Install the correct MetaX mcPytorch "
                "or fix MACA runtime environment variables."
            )
        if not str(args.pytorch_device).startswith("cuda"):
            raise RuntimeError(f"GPU is required, but pytorch_device={args.pytorch_device!r}.")
        if args.pytorch_device.startswith("cuda:"):
            device_index = int(args.pytorch_device.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {args.pytorch_device}, but torch only sees {torch.cuda.device_count()} device(s)."
                )

    if str(args.pytorch_device).startswith("cuda"):
        # Force initialization early so failure happens before the websocket server starts.
        device = torch.device(args.pytorch_device)
        torch.zeros(1, device=device)
        logging.info("CUDA/MACA device initialized successfully: %s", device)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without enabling slower deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch  # noqa: PLC0415 - preserve support for JAX-only environments.
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: Args) -> None:
    server_port = args.port + args.server_rank if args.port_offset_by_rank else args.port

    logging.info(
        "[Policy Server] rank=%s, world_size=%s, bind=%s:%s",
        args.server_rank,
        args.server_world_size,
        args.host,
        server_port,
    )

    logging.info("Policy sampling seed: %s", args.seed)
    _seed_everything(args.seed)
    _setup_pytorch_device(args)

    policy = create_policy(args)
    # Model construction may consume the global RNG. Reset it so the first
    # flow-matching noise sample is a deterministic function of --seed.
    _seed_everything(args.seed)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        record_dir = f"policy_records/rank_{args.server_rank}"
        policy = _policy.PolicyRecorder(policy, record_dir)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"

    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=server_port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
