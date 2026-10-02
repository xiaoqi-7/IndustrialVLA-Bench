<div align="center">

<h1>IndustrialVLA-Bench</h1>
<h3>A Traceable Multi-Axis Evaluation of Open Robot Policy Models</h3>

<p>
Yiqi Wang<sup>*</sup>, Zhifeng Rao<sup>*</sup>, Jiaqi Zhang, Xiaoyang Li, Zhangkai Wu,<br>
Yiqun Duan, Mingkai Zheng, Fei Wang, Shan You, Taotao Cai
</p>
<p><sup>*</sup>Equal contribution.</p>

<p>
  <a href="https://arxiv.org/abs/2609.25562">
    <img src="https://img.shields.io/badge/arXiv-2609.25562-B31B1B?logo=arxiv&logoColor=white" alt="arXiv:2609.25562">
  </a>
  <img src="https://img.shields.io/badge/Topic-VLA%20%26%20WAM%20Evaluation-1f6f6f" alt="Topic">
  <img src="https://img.shields.io/badge/Suites-LIBERO%20%7C%20Plus%20%7C%20Para-2B6CB0" alt="Suites">
  <img src="https://img.shields.io/badge/Models-6%20Open%20Policies-27ae60" alt="Models">
</p>

<p><i>Comparable, reproducible, and deployment-aware evaluation of open<br>
Vision-Language-Action (VLA) and World-Action Models (WAMs).</i></p>

</div>

---

## News

- **[2026-09]** Paper *"IndustrialVLA-Bench: A Traceable Multi-Axis Evaluation of Open Robot Policy Models"* ([arXiv:2609.25562](https://arxiv.org/abs/2609.25562)).
- **[2026-07]** Reproducible experiment archive released: raw logs, per-seed results, latency benchmarks, and standardized launch scripts for all six evaluated models.

## Contents

- [Overview](#overview)
- [Why an Official-First Benchmark?](#why-an-official-first-benchmark)
- [Core Concepts](#core-concepts)
- [Evaluated Models](#evaluated-models)
- [Benchmark Tracks](#benchmark-tracks)
- [Results](#results)
- [Deployability and hardware](#deployability-and-hardware)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [What Is Not Included](#what-is-not-included)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

<div align="center">
<img src="figs/Overview3.png" width="850">
<p><i><b>Figure 1.</b> IndustrialVLA-Bench evaluates open robot policy models through an official-first
harness, organizes evidence into tiers, and reports results along capability, robustness,
language grounding, and deployability axes — each backed by raw logs and traceable metadata.</i></p>
</div>

Open VLA and WAM systems are advancing rapidly, yet their reported results are hard to compare: checkpoints, evaluation protocols, action interfaces, and deployment settings differ across releases. **IndustrialVLA-Bench** evaluates six publicly executable robot policy models under a unified protocol:

- **LIBERO** for clean manipulation capability,
- **LIBERO-Plus** for robustness under controlled perturbations,
- **LIBERO-Para** for sensitivity to paraphrased instructions,
- a **deployability harness** for latency, peak VRAM, runtime mode, and setup burden.

Every model-benchmark configuration uses three fixed run-level seeds `{1, 7, 42}` with the checkpoint and inference configuration held fixed. Reported uncertainty is the population standard deviation across the three run-level results. Raw logs, per-seed summaries, and launch commands are preserved in this repository.

## Why an Official-First Benchmark?

**1. Fragmented evaluation.** Even within the same benchmark family, results can differ because of checkpoints, prompt formats, observation-action interfaces, action normalization, inference scripts, trial counts, and aggregation rules. Apparent gains may therefore reflect evaluation assumptions rather than stronger policies.

**2. Clean success is not reliability.** A model performing well under standard conditions may still fail under changes to cameras, robots, backgrounds, layouts, or instruction wording. Clean success alone cannot expose these differences.

**3. Deployability is rarely reported consistently.** VLA and WAM models operate inside robotic execution loops. Policy-call latency, amortized action latency, peak VRAM, runtime architecture, action chunking, and replanning configuration materially affect practical reuse.

IndustrialVLA-Bench treats robot-policy evaluation as a **multi-axis diagnostic and reproducibility problem**, not a single success-rate ranking problem. Capability, robustness, language grounding, and deployability are reported as separate evidence views.

## Core Concepts

| Concept | Meaning |
| --- | --- |
| **Official-first execution** | For each model, the official repository, checkpoint, inference path, and evaluation script are prioritized. Wrappers are limited to engineering functions and must not alter benchmark semantics. |
| **Protocol-faithful (PF)** | A reproduced path that preserves task definitions, observation/action semantics, success rules, episode horizon, trial counts, and aggregation procedures. |
| **Three-seed protocol** | Each model-benchmark configuration is evaluated with fixed run-level seeds `{1, 7, 42}`; tables report mean ± population standard deviation across the three runs. |
| **Multi-axis profile** | Capability, robustness, language grounding, and deployability are reported jointly but are not collapsed into one scalar score. |
| **Traceability** | Results are linked to launch commands, environments, raw logs, per-seed summaries, and aggregation rules preserved in the artifact. |

## Evaluated Models

Six publicly executable systems spanning two design families are included. Following the rebuttal-stage verification audit, **all six evaluated systems now satisfy the protocol-faithful checklist**.

| Model | Family | Params | Representative design | LIBERO | LIBERO-Plus | LIBERO-Para | Evidence status |
| --- | :---: | :---: | --- | :---: | :---: | :---: | :---: |
| π<sub>0.5</sub> / OpenPI | VLA | 3.6B | Unified single-policy VLA | ✅ | ✅ | ✅ | PF |
| UnifoLM-VLA-0 | VLA | 8.9B | Industrial VLA, model-specific action generation | ✅ | ✅ | ✅ | PF |
| Xiaomi-Robotics-0 | VLA | 4.7B | Industrial VLA, diffusion-based action prediction | ✅ | ✅ | ✅ | PF |
| GR00T-N1.7 | VLA | 3.4B | Modular vision-language + action architecture | ✅ | ✅ | ✅ | PF |
| FastWAM | WAM | 12.4B | Video-latent-conditioned action prediction | ✅ | ✅ | ✅ | PF |
| Cosmos Policy | WAM | 2.0B | Multi-step video-diffusion world-action prediction | ✅ | ✅ | ✅ | PF |

<sub>The submission-time NR/PV labels reflected incomplete verification evidence. The rebuttal-stage audit completed checkpoint provenance, task-definition, observation/action-semantics, inference-configuration, termination, success-criterion, and aggregation verification without changing checkpoints, inference settings, or mean success rates.</sub>

## Benchmark Tracks

| Track | Question | Suite | Episodes / seed | Metric |
| --- | --- | --- | :---: | --- |
| Capability | Can the model complete clean manipulation tasks? | LIBERO (Spatial / Object / Goal / Long) | 2,000 (50/task) | Success rate |
| Robustness | Does performance survive visual and embodiment shifts? | LIBERO-Plus | 10,030 | Success rate, drop, retention |
| Language grounding | Does the model follow paraphrased instructions? | LIBERO-Para | 4,092 | Success rate, clean-to-Para drop |
| Deployability | Can the model run practically? | Evaluation harness | — | Policy-call latency, amortized action latency, effective control frequency, peak VRAM |

## Results

Unless otherwise stated, values are **mean ± population standard deviation over the three run-level seeds `{1, 7, 42}`**. The values below incorporate the rebuttal-stage corrections to the affected OpenPI SD entries.

### Clean capability — LIBERO

| Model | Spatial | Object | Goal | Long | **Average** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Xiaomi-Robotics-0 | 98.67 ± 0.09 | 100.00 ± 0.00 | 97.93 ± 0.94 | 95.80 ± 0.91 | **98.10 ± 0.43** |
| UnifoLM-VLA-0 | 98.67 ± 0.66 | 100.00 ± 0.00 | 97.73 ± 0.25 | 95.27 ± 0.41 | **97.92 ± 0.06** |
| GR00T-N1.7 | 97.60 ± 1.50 | 99.30 ± 0.50 | 99.30 ± 0.50 | 95.30 ± 1.00 | **97.88 ± 0.65** |
| Cosmos Policy | 96.40 ± 0.33 | 99.60 ± 0.33 | 97.93 ± 0.34 | 96.60 ± 0.28 | **97.63 ± 0.23** |
| FastWAM | 97.07 ± 0.25 | 99.13 ± 0.09 | 96.47 ± 0.50 | 93.53 ± 0.41 | **96.55 ± 0.16** |
| π<sub>0.5</sub> / OpenPI | 98.33 ± 0.09 | 98.73 ± 0.52 | 97.60 ± 0.43 | 91.40 ± 0.65 | **96.52 ± 0.22** |

<sub>6,000 episodes per model (50 trials/task × 3 seeds). Average is the unweighted mean of the four suite-level success rates. The OpenPI average SD is corrected from 0.62 to 0.22 after rechecking the released per-seed records using the Appendix-D population-SD definition.</sub>

### Robustness — LIBERO-Plus

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | **Robust Avg.** | Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| π<sub>0.5</sub> / OpenPI | 70.63 | 75.12 | 85.97 | 96.76 | 95.82 | 87.05 | 86.47 | **85.31 ± 0.32** | 11.21 |
| Cosmos Policy | 72.65 | 52.56 | 89.07 | 98.48 | 84.94 | 90.21 | 84.00 | **80.47 ± 0.23** | 17.16 |
| UnifoLM-VLA-0 | 57.03 | 68.37 | 91.09 | 93.70 | 95.14 | 79.26 | 79.19 | **78.78 ± 0.30** | 19.14 |
| GR00T-N1.7 | 64.40 | 38.40 | 83.88 | 94.87 | 93.72 | 84.36 | 74.94 | **75.12 ± 0.28** | 22.76 |
| Xiaomi-Robotics-0 | 40.22 | 55.63 | 89.02 | 94.51 | 90.43 | 86.78 | 75.87 | **73.91 ± 0.13** | 24.19 |
| FastWAM | 44.65 | 72.04 | 66.99 | 94.22 | 66.57 | 67.17 | 79.45 | **70.69 ± 0.52** | 25.86 |

<sub>Robust Avg. is the unweighted mean over the six non-linguistic perturbation dimensions; the LIBERO-Plus language condition is shown separately and excluded from Robust Avg. and Drop.</sub>

### Language grounding — LIBERO-Para

| Model | Para success (%) | Clean-to-Para drop (pp) |
| --- | ---: | ---: |
| UnifoLM-VLA-0 | **82.24 ± 0.48** | 15.67 ± 0.53 |
| Xiaomi-Robotics-0 | **75.73 ± 0.16** | 22.37 ± 0.48 |
| GR00T-N1.7 | **74.26 ± 1.52** | 23.62 ± 1.52 |
| π<sub>0.5</sub> / OpenPI | **71.33 ± 0.05** | **25.19 ± 0.18** |
| Cosmos Policy | **70.50 ± 0.26** | 27.13 ± 0.18 |
| FastWAM | **51.16 ± 0.38** | 45.39 ± 0.32 |

<sub>Same checkpoints as the clean LIBERO evaluation; no model is adapted or fine-tuned on LIBERO-Para. The OpenPI clean-to-Para drop SD is corrected from 0.12 to 0.18.</sub>

### Key findings

1. **Clean LIBERO is close to saturation** — averages span only **1.58 pp** (96.52–98.10%).
2. **Robustness separates models much more strongly** — LIBERO-Plus robustness spans **14.62 pp**.
3. **Language sensitivity is even more discriminative** — LIBERO-Para spans **31.08 pp**.
4. **Clean ranking does not predict robustness** — the clean leader does not lead under perturbations.
5. **Language sensitivity is partially decoupled from clean capability** — models with similar clean scores can diverge sharply under paraphrases.
6. **All six systems now satisfy the PF checklist**, so the reported six-system comparisons no longer mix evidence tiers.

## Deployability and hardware

All six models were measured on a **single MetaX C500 GPU with batch size 1**. Because dtype, action chunk length, and replanning configuration differ across models, these measurements are **observational deployment profiles**, not a controlled intrinsic-efficiency ranking.

| Model | Device | Peak VRAM | Dtype | Batch size | Action chunk | Replan / open-loop steps | Latency / call | Amortized policy latency / action | Effective control frequency |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GR00T N1.7 | MetaX C500 | 7,810 MiB | BF16 | 1 | 16 | 8 | 153.03 ms | 19.13 ms | 52.3 Hz |
| Cosmos Policy | MetaX C500 | 8,654 MiB | BF16 autocast | 1 | 16 | 16 | 340.16 ms | 21.26 ms | 47.0 Hz |
| Xiaomi-Robotics-0 | MetaX C500 | 10,294 MiB | BF16 | 1 | 10 | 10 | 254.35 ms | 25.43 ms | 39.3 Hz |
| OpenPI π0.5 | MetaX C500 | 14,550 MiB | BF16 | 1 | 10 | 5 | 294.02 ms | 58.80 ms | 17.0 Hz |
| UnifoLM-VLA-0 | MetaX C500 | 18,332 MiB | BF16/FP32 mixed | 1 | 8 | 8 | 158.54 ms | 19.82 ms | 50.5 Hz |
| FastWAM | MetaX C500 | 26,902 MiB | BF16 | 1 | 32 | 10 | 453.33 ms | 45.33 ms | 22.1 Hz |

**Latency definitions**
- **Latency / call**: wall-clock latency of one policy invocation.
- **Amortized policy latency / action**: policy-call latency divided by the number of actions executed before replanning.
- **Effective control frequency**: inverse of the amortized policy-side latency per executed action.

Peak VRAM is measured with `nvidia-smi` during evaluation-time inference after warm-up. Environment reset time is excluded.

### Runtime and reproducibility summary

| Model | Params | Runtime mode | Evidence status |
| --- | ---: | --- | :---: |
| GR00T-N1.7 | 3.4B | Deployment-oriented | PF |
| Cosmos Policy | 2.0B | Deployment-oriented | PF |
| Xiaomi-Robotics-0 | 4.7B | Model-specific | PF |
| π<sub>0.5</sub> / OpenPI | 3.6B | Local / server-client | PF |
| UnifoLM-VLA-0 | 8.9B | Local / model-specific | PF |
| FastWAM | 12.4B | Deployment-oriented | PF |

## Getting Started

This repository is a **reproducible experiment archive**. To re-run evaluations on a new machine:

**1. Install benchmark environments**

```bash
bash envs/libero/install.sh
bash envs/libero-plus/install.sh
bash envs/libero-para/install.sh
```

**2. Prepare simulation assets**

```bash
ARCHIVE=/path/to/assets.zip bash LIBERO-plus/prepare_assets.sh
```

**3. Install the model environment** following the per-model materials in `envs/<model>/`.

**4. Launch an evaluation**

```bash
MODEL_PATH=/path/to/checkpoint \
LIBERO_PLUS_ROOT=/path/to/LIBERO-plus \
bash scripts/openpi/eval_openpi_libero_plus.sh
```

Unified entry points write outputs to `results/<model>/` by default. Local source and benchmark paths can be overridden through the documented environment variables.

## Repository Structure

```text
IndustrialVLA-Bench/
├── LIBERO/
├── LIBERO-plus/
├── LIBERO-para/
├── models/
├── results/
│   ├── <model>/
│   └── latency_benchmark/
├── scripts/
│   └── <model>/original/
├── envs/
└── figs/
```

- `results/` contains raw logs, per-seed outputs, and summary records.
- `scripts/` contains standardized launchers and preserved model-specific launch paths.
- `envs/` contains per-model and per-benchmark environment materials.
- `results/latency_benchmark/` contains deployment measurements.

## What Is Not Included

To keep the archive lightweight, the following are not redistributed:

- model checkpoints,
- large simulation assets,
- evaluation rollout videos and demo GIFs.

These should be obtained from the corresponding official releases. The repository preserves the evaluation code, launch paths, logs, summaries, and deployment measurements used by the benchmark.

## Citation

```bibtex
@article{industrialvlabench2026,
  title   = {IndustrialVLA-Bench: A Traceable Multi-Axis Evaluation of Open Robot Policy Models},
  author  = {Wang, Yiqi and Rao, Zhifeng and Zhang, Jiaqi and Li, Xiaoyang and Wu, Zhangkai and Duan, Yiqun and Zheng, Mingkai and Wang, Fei and You, Shan and Cai, Taotao},
  journal = {arXiv preprint arXiv:2609.25562},
  year    = {2026}
}
```

## Acknowledgements

This benchmark builds on LIBERO, LIBERO-Plus, and LIBERO-Para and evaluates the released OpenPI, UnifoLM-VLA-0, Xiaomi-Robotics-0, GR00T-N1.7, FastWAM, and Cosmos Policy systems.

## Contact

For questions about the benchmark, results, or reproduction, please open an issue in this repository or contact the authors listed on the arXiv page.
