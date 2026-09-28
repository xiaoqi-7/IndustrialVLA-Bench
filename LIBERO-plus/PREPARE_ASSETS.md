# LIBERO-plus assets

为控制 IndustrialVLA-Bench 的归档体积，官方 `assets.zip` 和已展开的 assets
均未保留。运行评测前请从官方页面下载：

<https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main>

下载完成后可把压缩包放在任意位置，并通过 `ARCHIVE` 指定：

```bash
ARCHIVE=/path/to/assets.zip bash prepare_assets.sh
```

脚本会把压缩包直接展开到一个新目录，成功后原子切换 `assets/`，并保留旧
目录作为带时间戳的备份。需要约 9 GB 的额外可用空间，并可能因小文件数量
较多而运行较久。
