# Standardized launch scripts

脚本按模型分组，文件名统一为 `eval_<model>_<suite>.sh`，其中 suite 为
`libero`、`libero_plus` 或 `libero-para`。统一入口会设置本归档中的模型、
基准和结果路径，然后调用 `models/` 中对应的原启动脚本；没有改写模型评测
逻辑。每个模型的 `original/` 目录还保留了原启动脚本的独立复制件。

统一入口默认把新运行的输出写入本归档的 `results/<model>/`。如需写入其他
位置，请按各脚本支持的变量设置 `RESULT_ROOT`、`OUTPUT_DIR`、`LOG_ROOT` 或
`LOG_BASE`。所有源代码和基准路径都可用相应的 `*_ROOT` 环境变量覆盖。

OpenVLA 当前仅有标准 LIBERO 启动实现；其 `libero_plus` 和 `libero-para` 文件
会明确提示尚未提供对应评测实现，而不会误把标准 LIBERO 当作另外两套基准。
