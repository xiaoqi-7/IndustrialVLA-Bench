# Third-Party Notices

This repository vendors the source code of several third-party projects so that the
evaluations in the IndustrialVLA-Bench paper are reproducible from one archive. Each
vendored component **retains its own license**, which governs that component regardless
of the repository's top-level [MIT license](LICENSE). Model checkpoints and simulation
assets are **not** redistributed here and remain subject to their original terms.

| Component | Path | Upstream | License (as vendored) |
|---|---|---|---|
| LIBERO | `LIBERO/` | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT (`LIBERO/LICENSE`, © 2023 Lifelong Robot Learning) |
| LIBERO-plus | `LIBERO-plus/` | https://github.com/HuangJanuary/LIBERO-plus | **No license file included in the vendored copy** — derived from LIBERO (MIT); consult the upstream repository for its terms |
| LIBERO-Para | `LIBERO-para/` | https://github.com/CAU-HAI-Lab/LIBERO-Para | MIT (`LIBERO-para/LICENSE`, © 2023 Lifelong Robot Learning) |
| OpenPI (π₀.₅) | `models/openpi/` | https://github.com/Physical-Intelligence/openpi | Apache-2.0 (`models/openpi/LICENSE`); Gemma model terms in `models/openpi/LICENSE_GEMMA.txt` |
| UnifoLM-VLA | `models/unifolm-vla/` | https://github.com/unitreerobotics/unifolm-vla | **No license file included in the vendored copy** — consult the upstream repository for its terms |
| Xiaomi-Robotics-0 | `models/Xiaomi-Robotics-0/` | https://github.com/XiaomiRobotics/Xiaomi-Robotics-0 | Apache-2.0 (`models/Xiaomi-Robotics-0/LICENSE`) |
| Isaac-GR00T | `models/Isaac-GR00T/` | https://github.com/NVIDIA/Isaac-GR00T | Apache-2.0 (`models/Isaac-GR00T/LICENSE`) |
| FastWAM | `models/FastWAM/` | (upstream release) | MIT (`models/FastWAM/LICENSE`, © 2026 The FastWAM Authors) |
| Cosmos Policy | `models/cosmos-policy/` | https://github.com/nvidia-cosmos | Apache-2.0 (`models/cosmos-policy/LICENSE`) |
| OpenVLA | `models/openvla/` | https://github.com/openvla/openvla | MIT (`models/openvla/LICENSE`, © 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti) |

Additional notes:

- The two components without a vendored license file (LIBERO-plus, UnifoLM-VLA) are
  redistributed here solely for reproducibility of the published evaluation. If you are
  an author of either project and object to this redistribution, please open an issue.
- Evaluation outputs under `results/` were produced by executing the above components
  and the officially released checkpoints of the respective models; the checkpoints
  themselves are not included and retain their original licenses and use restrictions.
- Dependency licenses (PyTorch, MuJoCo, robosuite, transformers, etc.) are not vendored
  and are governed by the packages installed via the materials under `envs/`.
