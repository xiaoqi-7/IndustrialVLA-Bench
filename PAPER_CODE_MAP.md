# IndustrialVLA-Bench — Paper ↔ Code Reference Map

**Purpose.** This document organizes the details of this repository according to the paper
*"IndustrialVLA-Bench: A Traceable Multi-Axis Evaluation of Open Robot Policy Models"*
(`VLA_benchmark/main.tex`, revision of 2026-07-27). **The paper is the authoritative source**;
where the code or archived results deviate from it, the deviation is flagged here rather than
reinterpreted. Repository state audited: commit `bd6ec3b` ("Add cross-axis profiles figure").

**Legend**

| Mark | Meaning |
|---|---|
| ✅ | Code/artifact matches the paper claim |
| ⚠️ | Mismatch — the paper statement is not (fully) supported by the repository |
| ● | Code-only detail the paper does not specify (useful implementation fact) |

---

## 1. Overview: paper axes ↔ repository layout

The paper evaluates **six released VLA/WAM checkpoints** on **three LIBERO-family tracks**
through **four coordinated evidence axes** (§2, Fig. 1): *Capability*, *Robustness*,
*Language sensitivity*, and *Deployability & reproducibility* (execution/reproduction evidence).

| Paper concept | Repository location |
|---|---|
| Clean capability track — **LIBERO** (§3.3, Table 2) | `LIBERO/` (vendored fork of the upstream benchmark) |
| Robustness track — **LIBERO-Plus** (§3.3, Table 3) | `LIBERO-plus/` (drop-in fork; all perturbations live here) |
| Language track — **LIBERO-Para** (§3.3, Table 4) | `LIBERO-para/` (paraphrase fork) |
| Model integrations (§3.2) | `models/<model>/` — 7 vendored model repos, no weights |
| Standardized launch scripts (App C.1 "wrappers") | `scripts/<model>/eval_<model>_<suite>.sh` + `scripts/<model>/original/` |
| Environment isolation (App C.2) | `envs/` — copied install materials only (requirements, Dockerfiles, `install*.sh`); **no simulation code** |
| Raw logs / per-seed results / summaries (App C.3) | `results/<model>/…` |
| Deployability measurements (§4.5, Table 5) | `results/latency_benchmark/` |
| Figures 1–3 | `figs/Overview3.png`, `figs/Pipeline3.png`, `figs/cross_axis_profiles.png` (Fig. 3 data hardcoded in `VLA_benchmark/figs/plot_cross_axis_profiles.py`) |

⚠️ The repo `README.md` still carries the **old paper title** ("An Official-First Benchmark for
Open Robot Policy Models"), an arXiv placeholder, and a broken image link (`figs/overview.png`;
the file on disk is `figs/Overview3.png`). See §11.

---

## 2. Model scope (paper §3.2, evidence statuses from App D)

Paper's evidence assignment (App D, "Evidence status"): **PF** = OpenPI, GR00T-N1.7, FastWAM;
**NR** = UnifoLM-VLA-0; **PV** = Xiaomi-Robotics-0, Cosmos Policy. Only PF rows support strict
comparisons.

| Paper name (status) | Family / design label (§3.2) | Code dir | Runtime | Checkpoint loading |
|---|---|---|---|---|
| $\pi_{0.5}$ / OpenPI (PF) | VLA, single-stack | `models/openpi/` | server/client | `scripts/serve_policy.py … policy:checkpoint --policy.config pi05_libero --policy.dir $TORCH_CHECKPOINT_DIR`; needs `model.safetensors` + `assets/physical-intelligence/libero/norm_stats.json`; JAX→PyTorch conversion via `examples/convert_jax_model_to_pytorch.py` (bfloat16) ● |
| UnifoLM-VLA-0 (NR) | VLA, model-specific action generation | `models/unifolm-vla/` | in-process | `--pretrained-path …/pytorch_model.pt` + `--vlm-pretrained-path` (UnifoLM-VLM-Base); `UNNORM_KEY` per track (`libero_object_no_noops` clean / `auto` plus / `libero_goal_no_noops` para); `WINDOW_SIZE=2` ● |
| Xiaomi-Robotics-0 (PV) | VLA, diffusion-based action prediction | `models/Xiaomi-Robotics-0/` | server/client | servers via `scripts/deploy.sh`, clients via `deploy.client.Client`; `BASE_PORT` 10086 (clean/para) / 9999 (plus) ● |
| GR00T-N1.7 (PF) | VLA, separate vision-language + action module | `models/Isaac-GR00T/` | server/client | `gr00t.eval.rollout_policy.create_gr00t_sim_policy(...)`; **one checkpoint per suite** (`$MODEL_ROOT/{libero_10,libero_goal,libero_object,libero_spatial}`); `PATCH_COSMOS_PATH=1` rewrites `nvidia/Cosmos-Reason2-2B` to a local path ● |
| FastWAM (PF) | WAM, video-latent-conditioned (no video diffusion at inference) | `models/FastWAM/` | in-process | `_build_model()` in `eval_libero.py`: `WanVideoDiT` + `ActionDiT` inside `MoT`, `WanTextEncoder` (umt5-xxl), `WanVideoVAE38`; bfloat16 ● |
| Cosmos Policy (PV) | WAM, multi-step video-diffusion | `models/cosmos-policy/` | in-process | `--ckpt-path Cosmos-Policy-LIBERO-Predict2-2B.pt --config cosmos_predict2_2b_480p_libero__inference_only`; needs base Cosmos-Predict2-2B-Video2World + T5 embedding cache ● |

Launch wrappers: `scripts/<model>/eval_<model>_{libero,libero_plus,libero-para}.sh` — thin
(≈10-line) wrappers that export path variables and `exec` the model's own launcher under
`models/`, consistent with the paper's allowed-wrapper policy (App C.1). ✅

- **OpenVLA**: code and clean-LIBERO scripts are kept (`models/openvla/`, `scripts/openvla/`) but
  it is **excluded from the six-model comparison**; its plus/para wrappers are explicit `exit 2`
  stubs. Consistent with the paper's bounded six-model scope. ✅
- **DreamZero**: excluded in the paper (no publicly executable checkpoint); no code in the repo. ✅

---

## 3. Track 1 — Clean capability on LIBERO (§3.3, §4.2, Table 2)

**Suite name mapping** (paper header ↔ code suite key):

| Paper | Code key | Tasks | Episode horizon ● (`MAX_STEPS_BY_SUITE`) |
|---|---|---:|---:|
| LIBERO-Spatial ("Spatial") | `libero_spatial` | 10 | 220 |
| LIBERO-Object ("Object") | `libero_object` | 10 | 280 |
| LIBERO-Goal ("Goal") | `libero_goal` | 10 | 300 |
| LIBERO-Long ("Long") | **`libero_10`** | 10 | 520 |
| — (not evaluated) | `libero_90` | 90 | 400 |

Note the naming trap: the paper's "Long" is `libero_10` everywhere in code (the "10" means
10 long-horizon tasks, not LIBERO-10-suite-of-90). Suites are defined in
`<fork>/libero/libero/benchmark/libero_suite_task_map.py` and registered via
`@register_benchmark` into `BENCHMARK_MAPPING` (`benchmark/__init__.py`).

- Protocol: 50 trials/task → 2,000 episodes/seed → 6,000 over three seeds (§4.1). Code default
  `NUM_TRIALS_PER_TASK=50` in the clean launchers. ✅
- Suite selection ●: `TASK_SUITES` env var (canonical value
  `libero_spatial libero_object libero_goal libero_10`); GR00T is one-suite-per-invocation
  (`SUITE` env var with aliases `spatial|object|goal|10`); Xiaomi clean uses a fixed task-ID order
  `TASK_IDS=(0 1 8 3 4 5 6 7 2 9)`.
- ⚠️ `scripts/openpi/original/eval_libero.sh` and `eval_libero_plus.sh` **default to only two
  suites** (`libero_10 libero_object`); a full four-suite run requires setting `TASK_SUITES`
  explicitly. The paper protocol implies all four suites in every run.
- ⚠️ All 130 clean-LIBERO `.bddl` files in `LIBERO/libero/libero/bddl_files/` are **zero-byte**
  in this repository (asset-pruned checkout with no restore script for the clean fork), so the
  published clean track is not directly runnable. See §11.

---

## 4. Track 2 — Robustness on LIBERO-Plus (§3.3, §4.3, Table 3)

The paper evaluates six non-language perturbation dimensions (+ a Language condition reported
for completeness but excluded from *Robust Avg.*). LIBERO-Plus is cited as `fei2026liberoplus`. ✅

**The seven dimensions — paper term ↔ code term ↔ implementation:**

| Paper column | Paper prose (§3.3) | Code `category` string | Task-name token | Implementation (all under `LIBERO-plus/libero/`) |
|---|---|---|---|---|
| Camera | camera viewpoint | `Camera Viewpoints` | `_view_<h>_<v>_<scale>_<rot>_<vert>` | helpers at top of `libero/envs/problems/*.py`: `rotate_around_y/z()`, `scale_distance_from_pivot()`; scale 1.00–2.00, h 0–359°, v ∈ {0,15} ● |
| Robot | robot appearance or embodiment | `Robot Initial States` | `_initstate_<K>` (K ≤ 500) | 501 subclasses each in `libero/envs/robots/{mounted_panda,on_the_ground_panda}.py`, overriding 7-DoF `init_qpos` ● |
| Language | (excluded from Robust Avg.) | `Language Instructions` | `_language_<L>` (L 1–50) | LLM-rewritten `(:language …)` blocks materialized as separate bddl files |
| Light | illumination | `Light Conditions` | `_light_<G>` (G 1–50) | problem classes selecting `scenes/lights/tabletop_light_sync_modified_<0..1023>.xml` ● |
| Background | background | `Background Textures` | `_table_<T>` (1–28), `_tb_<T>` (1–23) | per-texture problem classes; texture vocab in `libero/envs/arenas/style.py` (`FLOOR_STYLE` 26, `WALL_STYLE` 25) ● |
| Noise | visual noise | `Sensor Noise` | `_noise_<N>` (N 1–50) | applied to rendered frames in `libero/envs/env_wrapper.py` `step()`/`reset()`: 5 corruption types × 10 severities (motion/gaussian/zoom blur, fog, glass) ● |
| Layout | object layout | `Objects Layout` | `_add_<A>` (1–30) or `_level<V>_sample<S>` (V 1–5, S 1–4) | distractors from 868 `CustomObjects` classes (`libero/envs/objects/custom_objects.py`) + region-centroid jitter; programmatic API `libero/randomizer/bddl_operators.py::perturb_region_info()` ● |

The canonical short-name mapping is replicated verbatim as `CATEGORY_TO_COLUMN` in six places —
`models/openpi/scripts/summarize_libero_plus.py:11`,
`models/Isaac-GR00T/scripts/eval_libero_plus_suite.py`,
`models/Xiaomi-Robotics-0/eval_libero/{eval,summarize}_libero_plus.py`,
`models/unifolm-vla/experiments/LIBERO/{eval,summarize}_libero_plus.py`,
`models/FastWAM/eval_libero_plus.py`, and (as `category_to_short`)
`models/cosmos-policy/run_libero_plus.sh` / `run_libero_plus_eval.py`. ✅

**Task inventory** — perturbations are **pre-baked static variants**, not runtime randomization.
The manifest `LIBERO-plus/libero/libero/benchmark/task_classification.json` maps every task to
`{id, name, category, difficulty_level}`:

| Suite | Perturbed tasks |
|---|---:|
| `libero_spatial` | 2,402 |
| `libero_object` | 2,518 |
| `libero_goal` | 2,591 |
| `libero_10` (Long) | 2,519 |
| **Total** | **10,030** ✅ (= paper's 10,030 episodes/seed at 1 trial/config) |

`scripts/cosmos-policy/original/run_libero_plus.sh` hard-asserts `TOTAL_TASKS == 10030` for a
full run. ✅ Camera/robot/noise variants create no new bddl file — they are encoded in the task
*name* and stripped at env construction (`env_wrapper.py`, `ControlEnv.__init__` splits on
`_view_`); texture/light/language/layout variants are materialized bddl files. ●

**Details the paper does not specify** ●:
- `difficulty_level` 1–5 per task (Light has 121 `null`); per-category totals: Noise 1,601,
  Camera 1,599, Robot 1,550, Language 1,537, Layout 1,525, Light 1,142, Background 1,076.
- The seven categories are **unbalanced per suite**, so *Robust Avg.* (unweighted mean over six
  category rates, per the paper) ≠ a pooled episode-level average. Code follows the paper's
  definition. ✅
- LIBERO-plus keeps `name="libero"` in `setup.py` — it is a drop-in replacement for the LIBERO
  package; the harness's only change is `num_trials_per_task: 50 → 1`.
- Known fork bugs (do not affect the evaluated suites): `LIBERO_MIX` is registered but
  `libero_mix` is missing from `libero_task_map` → instantiating it raises `KeyError`; the
  benchmark's task-order shuffle for `task_order_index > 0` is unseeded at import time
  (all launchers use the default order 0).
- Assets are stripped: restore with `ARCHIVE=/path/to/assets.zip bash LIBERO-plus/prepare_assets.sh`
  (see `LIBERO-plus/PREPARE_ASSETS.md`); ⚠️ even so, 9,112 of 16,330 plus-bddl files are
  zero-byte in the repo (see §11).

---

## 5. Track 3 — Language sensitivity on LIBERO-Para (§3.3, §4.4, Table 4)

- **4,092 paraphrases over the 10 LIBERO-Goal tasks** (`eval0`–`eval9`): bddl in
  `LIBERO-para/libero/libero/bddl_files/libero_para/`, init states
  `init_files/libero_para/eval{0..9}.pruned_init`; environments are built from the *libero_goal*
  bddl with the paraphrased instruction swapped in. ✅ (matches the paper's 4,092 episodes/seed,
  1 per configuration)
- Suite key `libero_para`, registered as `LIBERO_PARA`; the fork's only env-code change vs LIBERO
  is `get_init_state()` in `env_wrapper.py`. ●
- Eval entry points: `LIBERO-para/eval_scripts/eval_template.py` + per-model examples; model-side
  launchers `models/<model>/…eval_libero_para*` with `--mode {auto,para,original}` and
  `--num_shards/--shard_index` task sharding. GR00T-Para always serves the single `libero_goal`
  checkpoint (paraphrases target the 10 goal tasks). ●
- Paraphrase taxonomy ● (`LIBERO-para/metrics/libero_para_metadata.csv`, 4,092 rows):
  `high` ∈ {`obj` 259, `act` 870, `comp` 2,963} (object-referring / action-referring /
  compositional), `mid` ∈ lexical / structural / pragmatical (+ combinations), plus
  keyword/structural similarity columns.
- **PRIDE metric** ● — the fork defines `PRIDE(α) = Σ(success·PD)/Σ(PD)·100` with
  `PD = 1 − (α·S_K + (1−α)·S_T)`, α = 0.5 (`LIBERO-para/metrics/README.md`,
  `analyze_results.py`). **The paper reports plain success rate only and does not use PRIDE.**
- The paper deliberately does **not pool** the LIBERO-Plus Language condition with LIBERO-Para
  (§4.4: Spearman ρ = 0.60 between the two orderings, mean offset 13.47 pp). Nothing in the code
  pools them either. ✅

---

## 6. Evaluation protocol (§4.1, App D)

| Paper requirement | Code reality |
|---|---|
| Run-level seeds {1, 7, 42}; a run seed initializes simulator-reset + task-order RNGs (+ policy sampling when stochastic) | `SEEDS="1 7 42"` default in the clean and plus launchers ✅ — **but** para launcher defaults are `SEED=7` (GR00T), `SEED=43` (Xiaomi), `SEED=8` (unifolm multi-GPU) ⚠️, and archived para runs record other seeds (GR00T `[1, 0, 8]`, Xiaomi `42/7/44` — §11) ⚠️ |
| Clean: 50 trials/task = 2,000 eps/seed; Plus: 10,030 eps/seed; Para: 4,092 eps/seed | `NUM_TRIALS_PER_TASK` 50 / 1 / 1 ✅; cosmos asserts 10,030 ✅; task inventories verified (§4, §5) ✅ — but archived *clean* coverage is incomplete for OpenPI and GR00T ⚠️ (§11) |
| Fixed checkpoint & inference config across seeds and tracks; no adaptation/fine-tuning on Plus/Para | Same checkpoint paths across track launchers per model ✅ (GR00T uses per-suite clean checkpoints, incl. `libero_goal` for Para — a documented official-path property ●) |
| Aggregation: run-level mean + population std over 3 runs | Implemented per model: `models/openpi/scripts/summarize_libero_plus.py`, `summarize_libero_para_seeds.py`, `models/unifolm-vla/experiments/LIBERO/summarize_libero_plus.py`, `models/Xiaomi-Robotics-0/eval_libero/summarize_libero_plus.py`, `models/Isaac-GR00T/scripts/summarize_libero_para_seeds.py`, `models/FastWAM/summarize_four_task_results.py`, cosmos `--summarize-final --seeds "1 7 42"`; cross-seed numbers land in per-model `three_seed_average.md` / `three_seed_summary.json` files ✅ |
| Official success rules / horizons / termination unchanged | Launchers use suite-default horizons via `MAX_STEPS=-1` → 220/280/300/520 ✅; success comes from the LIBERO env's own success check ✅ |
| Seeds vary replay nondeterminism over a **fixed** configuration set | Matches the static-manifest design of Plus/Para (1 episode per pre-baked config) ✅ |

● Common launcher vocabulary: `SEEDS`, `GPUS`/`RANKS`, `EVAL_WORLD_SIZE`/`NUM_GPUS`,
`NUM_TRIALS_PER_TASK`, `MAX_TASKS=-1`, `MAX_STEPS`, `NUM_STEPS_WAIT=10`, `REPLAN_STEPS`,
`MUJOCO_GL=osmesa|egl`, `LIBERO_CONFIG_PATH`, `RESUME`, `SAVE_VIDEO`, `DRY_RUN`,
`PREFLIGHT_ONLY`; path roots `LIBERO_ROOT` / `LIBERO_PLUS_ROOT` / `LIBERO_PARA_ROOT`,
`RESULT_ROOT`, `LOG_ROOT`.

---

## 7. Metrics (§3.4)

| Paper metric | Definition (paper) | Where computed |
|---|---|---|
| SR (capability) | mean of episode success indicators, per suite + unweighted 4-suite Average | per-model summarizers; e.g. openpi clean rates in `results/openpi/results/three_seed_average.md` |
| Robust Avg. | unweighted mean over the six non-language dimensions | `CATEGORY_TO_COLUMN`-based summarizers (§4); Language excluded ✅ |
| Δ_robust (Drop) | SR_clean − SR_pert (pp) | computed at table level from the two averages (per Table 3 note, from rounded values); not a separate script ● |
| Retention ratio | SR_pert / SR_clean ("corresponding retention ratio", §3.4) | ⚠️ defined in the paper but reported in **no table and no repo artifact** |
| Δ_para | SR_clean − SR_para (pp) | table-level; per-seed para rates from `summarize_libero_para_seeds.py` / `merge_libero_para_results.py` |
| Execution evidence | latency/call, peak VRAM, runtime mode, single-GPU feasibility, setup burden, provenance, verification status | `results/latency_benchmark/` + Table 5 (see §8 for provenance issues ⚠️) |

● The FastWAM plus summary uses a versioned format `"fastwam-libero-plus-seed-v1"` with
`by_suite` / `by_category` / `by_column` / `by_difficulty` breakdowns — the richest per-model
result schema in the repo.

---

## 8. Deployability (§4.5, Table 5, App D "Runtime measurements")

Measurement code: `results/latency_benchmark/measure_{openpi,gr00t,fastwam,unifolm_libero,xiaomi}.py`
producing `{openpi,gr00t,fastwam,unifolm,xiaomi}.json` with fields `warmup`, `samples`,
`action_horizon`, `replan_steps`, `num_inference_steps`, `call_mean_ms`, `call_median_ms`,
`call_p95_ms`, `model_infer_mean_ms`, `latency_per_executed_step_ms`,
`torch_peak_allocated_mib`, `torch_peak_reserved_mib`. ●

**Latency column** — Table 5 header says **"Latency (ms/call)"** and App D says the value is
per policy invocation and *"not amortized per executed action"*. ⚠️ The archived JSONs show the
table values are exactly `latency_per_executed_step_ms` — i.e. **amortized per executed step**
(call time ÷ replan steps), not per call:

| Model | Table 5 | JSON `latency_per_executed_step_ms` | JSON `call_mean_ms` (true ms/call) |
|---|---:|---:|---:|
| OpenPI | 58.80 | 58.80 | 294.0 |
| UnifoLM-VLA-0 | 19.82 | 19.82 | 158.5 |
| Xiaomi-Robotics-0 | 25.44 | 25.43 | 254.3 |
| GR00T-N1.7 | 19.13 | 19.13 | 153.0 |
| FastWAM | 45.33 | 45.33 | 453.3 |
| Cosmos Policy | 21.26 | **no measurement file exists** ⚠️ | — |

Either the Table 5 caption/App D wording or the numbers need to change (per-call data already
exists in the JSONs, so both variants are reportable).

**Peak VRAM column** — Table 5 values match **no field** in the archived JSONs ⚠️
(e.g. OpenPI 14,550 vs `torch_peak_allocated_mib` 7,347 — plausibly server+client or
nvidia-smi process totals, but no artifact records this, and App C.2 promises recorded
GPU/driver/hardware metadata):

| Model | Table 5 VRAM (MiB) | JSON `torch_peak_allocated_mib` | JSON `torch_peak_reserved_mib` |
|---|---:|---:|---:|
| OpenPI | 14,550 | 7,347 | 7,416 |
| UnifoLM-VLA-0 | 18,332 | 17,324 | 17,420 |
| Xiaomi-Robotics-0 | 10,294 | 9,269 | 9,366 |
| GR00T-N1.7 | 7,810 | 6,160 | 6,252 |
| FastWAM | 26,902 | 23,878 | 25,996 |
| Cosmos Policy | 8,654 | — ⚠️ | — |

● Diffusion-steps column matches JSON `num_inference_steps` where available (OpenPI 10,
UnifoLM 4, Xiaomi 5, GR00T 4, FastWAM 10). ✅
● Parameter counts (3.6B / 8.9B / 4.7B / 3.4B / 12.4B / 2.0B) are not derivable from any repo
artifact (no weights vendored).

---

## 9. Harness and evidence policy (App B, App C)

| Paper mechanism | Code reality |
|---|---|
| **Official-first execution** (App C.1): wrappers limited to path forwarding, logging, env isolation, device placement, result formatting, deployment measurement | `scripts/<model>/eval_*.sh` wrappers only export paths and `exec` the model's own launcher; `scripts/<model>/original/` preserves byte-identical copies of the launchers ✅ |
| **Environment isolation** (App C.2): per-model conda/Docker envs; harness records package versions, CUDA, GPU, OS, driver, commit, checkpoint id, benchmark version | `envs/<model|suite>/` holds copied requirements/pyproject/Dockerfiles/install scripts ✅ (isolation materials) — ⚠️ but **no recorded runtime metadata** (GPU/driver/versions/commit/checkpoint hashes) exists anywhere in `results/` |
| **Command manifests** (App C.3, 15-field schema: model, repo_commit, checkpoint_id, benchmark, task_suite, entry_point, command, seed, num_trials, raw_log, success_rate, latency, peak_vram, runtime_mode, status) | ⚠️ **Not implemented.** The only file named `manifest.json` in `results/` (`results/cosmos-policy/results/libero_plus_3seeds/cosmos_libero_plus_3seeds_20260719_082355/manifest.json`) is **zero-byte** |
| **Normalized result schema** (App C.3: one row per model–benchmark–suite–track) | ⚠️ **Not implemented** as a machine-readable artifact. Closest equivalents: per-model `summary.json` / `three_seed_summary.json` / Chinese-language `three_seed_average.md` files with heterogeneous schemas |
| **Result promotion checklist** (App C.4) → PF/NR/PV assignment | No checklist artifact in the repo; the PF/NR/PV states exist only in the paper. ⚠️ Two archived summaries actively contradict a PF promotion (GR00T clean, OpenPI clean — §11) |
| **Evidence taxonomy** (App A: Main/Secondary/Diagnostic/Appendix; App B views) | Descriptive framework only; nothing to map in code (as intended) |

---

## 10. Results artifacts — where each paper table's numbers live

Layout: `results/<model>/results/<track>/<seed>/…` with per-model naming drift ●:

| Model | Clean | Plus | Para |
|---|---|---|---|
| OpenPI | `results/openpi/results/libero/seed_{1,7,42}/<suite>/eval_rank_N.log` | `results/openpi/results/libero-plus/seed_*/results/<suite>/rank_N.jsonl` | `results/openpi/results/libero_para/seed{1,7,42}/` |
| UnifoLM-VLA-0 | `results/unifolm-vla/results/LIBERO/…` | `…/LIBERO-plus/…` | `…/libero_para/…` |
| Xiaomi-Robotics-0 | `results/Xiaomi-Robotics-0/results/LIBERO/…` | `…/LIBERO-plus/…` | `…/libero_para/…` |
| GR00T-N1.7 | `results/Isaac-GR00T/results/libero/libero_<suite>_<timestamp>/` | `…/libero-plus/…` | `…/libero_para/seed{1,7,42}/` |
| FastWAM | `results/FastWAM/result/seed{1,7,42}` | `…/results_plus/…` | `…/results_para/…` |
| Cosmos Policy | `results/cosmos-policy/results/libero_3seeds/<run>/` | `…/libero_plus_3seeds/<run>/` | `…/libero_para_3seeds/<run>/` |

- LIBERO-plus JSONL record schema ●:
  `{"task_suite", "eval_rank", "task_id", "task_name", "category", "episode_idx", "success"}`
  (GR00T/FastWAM add `column`, `difficulty_level`, per-task aggregates).
- Para results: `eval0.json … eval9.json` + `meta.json` + `summary.json` per seed.
- Cross-seed summaries: `three_seed_average.md` / `three_seed_summary.json` per model+track;
  repo-wide there is **no** aggregator (each model has its own summarizer).
- `results/latency_benchmark/` — §8.
- ⚠️ `results/README.md` advertises `model_efficiency_comparison.md`,
  `model_efficiency_table.{md,xlsx}`, `revised_leaderboard_tables.tex` — none exist.

**Table-by-table provenance check** (paper value ↔ archive):

| Paper table | Public-archive support |
|---|---|
| Table 2 (clean) | **Partial ⚠️** — UnifoLM, Xiaomi, FastWAM, Cosmos rows have full per-seed archives; OpenPI and GR00T rows are **not supported** (see §11 A1/A2) |
| Table 3 (Plus) | **Supported ✅** — spot-verified: OpenPI archive row (70.63/75.12/85.97/96.76/95.82/87.05/86.47) matches the paper exactly |
| Table 4 (Para) | **Supported for values ✅ / seed claim ⚠️** — means/stds match (e.g. GR00T 74.26±1.52, Xiaomi 75.72±0.16 ≈ paper 75.73), but recorded seeds deviate from {1,7,42} (§11 A3) |
| Table 5 (deployability) | **Partial ⚠️** — 5 of 6 latency values reproducible (as per-step, not per-call); VRAM values unreproducible; Cosmos entirely unmeasured (§8) |
| Fig. 3 | `VLA_benchmark/figs/plot_cross_axis_profiles.py` hardcodes the same numbers as Tables 2–5 ✅ (consistency, not provenance) |

---

## 11. Discrepancy & traceability audit (paper is ground truth)

An external review of the public repo concluded that *"the paper's artifact claims are not yet
supported by the public materials."* Every item below has been re-verified against this checkout.

### A. Critical — paper claims not supported by the public artifact

1. **GR00T clean LIBERO (Table 2, PF, 97.88, 6,000 episodes).**
   `results/Isaac-GR00T/results/libero/three_seed_average.md` states outright that a three-seed
   mean **cannot be reliably computed**: the four run dirs are single-task
   (`ONLY_FIRST_TASK=1`), single-run, ~150-episode evaluations (recorded rates 95.33 / 99.33 /
   99.34 / 92.00), with no seed metadata. Three of the four paper values (Long 95.30, Goal
   99.30, Object 99.30) coincide with these single-task rates to one decimal; Spatial (97.60)
   matches nothing in the archive. The PF status contradicts the repo's own promotion evidence.
2. **OpenPI clean LIBERO (Table 2, PF).**
   `results/openpi/results/three_seed_average.md` documents **incomplete coverage** — LIBERO-10
   `928/1500` (2 seeds only), Object `934/1500`, Goal `1300/1500`, Spatial `1500/1500` — and
   reports Long 93.13 / Object 97.89 / Overall 96.74, which **disagrees with the paper's**
   91.40 / 98.73 / 96.52. (OpenPI's Plus and Para archives match the paper exactly.)
3. **LIBERO-Para seeds ≠ {1, 7, 42}.**
   GR00T: `results/Isaac-GR00T/results/libero_para/three_seed_summary.json` records
   `"seeds": [1, 0, 8]`; its own md admits directory names (`seed1/seed7/seed42`) don't match
   recorded seeds. Xiaomi: `results/Xiaomi-Robotics-0/results/libero_para/three_seed_average.md`
   records actual seeds **42 / 7 / 44** under dirs `seed_1/seed_7/seed_42`. The reported means
   match Table 4, but the fixed-seed protocol claim (§4.1, App D) does not hold for these runs.
4. **Zero-byte task definitions.** Verified counts of empty `.bddl` files:
   clean LIBERO **130/130**, LIBERO-plus **9,112/16,330**, LIBERO-para **4,164/4,222**.
   Only LIBERO-plus ships a restore path (`prepare_assets.sh` + external `ARCHIVE` zip); the
   clean and Para tracks are not runnable from the public repo as-is.
5. **Empty result records and the empty manifest.** Multiple `meta.json` / `summary.json` /
   `progress.json` files (e.g. under GR00T Para seeds) are zero-byte; the **only**
   `manifest.json` in `results/` (cosmos Plus run) is zero-byte — while App C.3 promises a
   15-field command manifest and normalized records per run.
6. **Cosmos Policy deployability row (Table 5).** No measurement script and no JSON exist for
   Cosmos in `results/latency_benchmark/`; its 8,654 MiB / 21.26 ms values have no public source.
7. **VRAM values unreproducible.** Table 5 VRAM matches no archived field (§8 table), and no
   hardware/env metadata exists to explain the gap, despite App C.2's recording promise.
8. **Latency semantics.** Table 5 says "ms/call", App D says "not amortized" — the archived
   values are amortized per-executed-step (`latency_per_executed_step_ms`); true per-call means
   are 153–453 ms (§8).
9. **Missing provenance scaffolding.** No root `LICENSE`, no third-party license notices, no
   pinned repo commits or checkpoint hashes anywhere; 18 `scripts/*/original/*.sh` files and
   archived result JSONs reference private `/mnt/afs/...` paths as defaults.

### B. Protocol deviations in code defaults

- Para launcher default seeds: GR00T `SEED=7`, Xiaomi `SEED=43`, unifolm `SEED=8`
  (paper: {1, 7, 42}).
- OpenPI clean/plus scripts default to 2 of 4 suites (`libero_10 libero_object`).
- Command manifest / normalized result schema (App C.3) unimplemented — see §9.

### C. Minor / cosmetic

- `README.md`: old paper title, arXiv placeholder `XXXX.XXXXX`, broken `figs/overview.png` link;
  `results/README.md` lists four files that don't exist.
- Naming drift: `libero_plus` vs `libero-plus` vs `LIBERO-plus`; `libero-para` (script names) vs
  `libero_para` (results dirs); per-model results-dir casing inconsistent (§10).
- `LIBERO_MIX` registration bug and unseeded task-order shuffle in the LIBERO-plus fork
  (harmless for the evaluated configuration).
- OpenVLA plus/para wrappers are `exit 2` stubs (consistent with its exclusion).

### D. Already fixed by the 2026-07-27 paper revision

- LIBERO-Plus is now cited (`fei2026liberoplus`).
- Evidence statuses are unified as PF/NR/PV across all tables (the earlier
  Table 2-vs-Table 6 contradiction is gone, and the old coverage table was removed).

### E. Remediated on branch `docs/paper-code-map` (repo-side fixes; no result data touched)

- **README refresh**: new paper title, arXiv placeholder replaced with "ID pending",
  broken `figs/overview.png` link fixed, PF/NR/PV evidence statuses added, stale
  key-findings numbers aligned to the paper (1.58 pp; 58.26/60.35/95.42), latency
  footnote now states the per-executed-step semantics, citation updated;
  `results/README.md` no longer claims nonexistent summary files.
- **Licensing**: root `LICENSE` (MIT, repo's own materials) and `THIRD_PARTY_NOTICES.md`
  (per-component licenses; notes that LIBERO-plus and UnifoLM-VLA ship no license file).
- **Script defaults** (executable launchers under `models/` only — the as-run archival
  copies under `scripts/*/original/` are intentionally untouched):
  - Para single-seed launchers now default to `SEED=1` with a "paper protocol: 1, 7, 42"
    note (previously 7/43/44/8); FastWAM LIBERO-Plus `SEEDS` default fixed from `7 42`
    to `1 7 42`.
  - openpi clean/plus launchers now default to all four suites.
  - All private `/mnt/afs/...` checkpoint defaults in the wired launchers replaced with
    empty, documented placeholders; benchmark-fork roots now default to the in-repo
    `LIBERO/`, `LIBERO-plus/`, `LIBERO-para/` copies (matching the wrappers).
- **Still open** (requires internal data or author decisions): zero-byte bddl/result
  files, missing Cosmos latency/VRAM measurements, VRAM provenance, command-manifest /
  normalized-schema implementation, pinned commits & checkpoint hashes, `/mnt/afs`
  defaults inside vendored Python internals (e.g. GR00T `qwen3_backbone.py`) and
  vendored per-model READMEs, and the paper-side Table 5 "ms/call" caption fix.

---

*Generated from `VLA_benchmark/main.tex` (2026-07-27) and repository commit `bd6ec3b`.
All file paths, counts, and numeric cross-checks in this document were verified against the
checkout on disk.*
