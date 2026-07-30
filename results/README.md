# Results

结果按模型分目录保存：

- `FastWAM/`
- `Isaac-GR00T/`
- `Xiaomi-Robotics-0/`
- `openpi/`
- `unifolm-vla/`
- `cosmos-policy/`
- `latency_benchmark/`（延迟 / 显存测量脚本与 JSON）

每个模型目录尽量保留原始的 seed、suite、日志和 summary 层级。跨 seed 汇总保存在
各模型目录下的 `three_seed_average.md` / `three_seed_summary.json`；本仓库没有
统一的跨模型汇总文件。各结果文件与论文表格的对应关系见根目录的
[`PAPER_CODE_MAP.md`](../PAPER_CODE_MAP.md)（第 10 节）。
