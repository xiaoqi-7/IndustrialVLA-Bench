# IndustrialVLA-Bench

这个目录是从 benchmark 工作区整理出的可复现实验归档。原始目录仍保留在
`/mnt/afs/zhengmingkai/raozf/benchmark`；本目录中的内容均为复制件。

## 目录约定

| 目录 | 内容 |
| --- | --- |
| `LIBERO/` | 官方 LIBERO 代码和任务定义（大型仿真资源未包含） |
| `LIBERO-plus/` | 官方 LIBERO-plus 代码和任务定义（大型仿真资源未包含） |
| `LIBERO-para/` | 官方 LIBERO-Para 代码和任务定义（大型仿真资源未包含；源目录名为 `LIBERO-Para`） |
| `models/` | 七个模型的源代码、配置和模型专用评测代码 |
| `results/` | 六个有评测结果的模型、延迟基准和汇总表 |
| `scripts/` | 按模型归档、统一命名的启动脚本 |
| `envs/` | 模型环境与三个 LIBERO 环境的安装材料 |

`models/` 中的七个模型为：FastWAM、Isaac-GR00T、Xiaomi-Robotics-0、
openpi（OpenPI π0.5）、openvla、unifolm-vla（UnifoLM-VLA）和
cosmos-policy（Cosmos Policy）。

`results/` 中的六个模型目录与效率汇总表一致；OpenVLA 目前没有纳入六模型
对比结果，因此只保留在 `models/openvla/`。

启动脚本默认使用源仓库中的模型检查点路径。迁移到另一台机器时，请先按
`envs/` 中的说明安装依赖，再通过环境变量（例如 `MODEL_PATH`、`PYTHON`、
`LIBERO_ROOT`、`LIBERO_PLUS_ROOT` 或 `LIBERO_PARA_ROOT`）覆盖本机路径。

## 轻量归档说明

为控制归档体积，本目录不包含下载得到的 LIBERO 仿真 assets、模型仓库中的
重复 assets，也不包含评测 rollout 视频或仓库中的演示 GIF。六个模型的 JSON、
JSONL、日志、文本汇总及 latency benchmark 均保留。

运行评测前需要按照各项目官方说明重新下载资源。LIBERO-plus 可将官方
`assets.zip` 下载到任意位置，再通过 `ARCHIVE=/path/to/assets.zip bash
LIBERO-plus/prepare_assets.sh` 安装；具体说明见
`LIBERO-plus/PREPARE_ASSETS.md`。
