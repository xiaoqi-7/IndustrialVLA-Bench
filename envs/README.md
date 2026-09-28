# Environment installation materials

每个子目录对应一个模型或一个官方基准环境，文件从原仓库复制而来；其中的
`README`、`requirements`、`pyproject`、`setup`、Dockerfile 和 `install*.sh`
可作为安装依据。推荐先安装 `libero/`、`libero-plus/` 或 `libero-para/`，再
安装模型对应的环境。

三个 LIBERO 子目录中的 `install.sh` 是本归档提供的便捷入口，会使用对应的
官方 `requirements.txt`，并从本目录的一级 LIBERO 源码执行 editable install。
LIBERO-plus 还需按 `../LIBERO-plus/PREPARE_ASSETS.md` 展开完整资源。

脚本中的绝对路径是原机器上的默认值，执行前请用环境变量覆盖路径。这里仅
整理安装材料，不会自动创建或修改 Conda 环境。
