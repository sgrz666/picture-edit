# evidence.zip —— 实验证据包（不纳入 Git）

本目录下的 `evidence.zip`（约 202 MB）是便于移交的实验证据包，**超过 GitHub 单文件 100 MB 限制，不纳入仓库**，可在本地/服务器自行打包生成。

## 打包方式

```bash
python -m vram_lab.package \
  --root results/review_20260908_v2 \
  --output results/review_20260908_v2/evidence.zip
```

## 包含内容

实验目录下除 `.pt / .npy / .zip / .tmp` 之外的全部文件：

- 每个运行的输入/输出/控制图 PNG、config.json、summary.json、validation.json
- 50ms NVML 显存采样 `gpu_samples.csv`、阶段事件 `events.jsonl`、更新记录 `updates.json`
- 各阶段 manifest、汇总 JSON/CSV、报告 Markdown、可视化 `index.html`
- 运行日志与源码快照

## 不包含内容

- `adapter.pt` 适配器检查点（单个约 500 MB）
- `control.npy` 等张量缓存
- 大型条件缓存（`cache/` 下按哈希命名的 latent）

这些大文件只保留在服务器完整实验目录（`/home/shangguanrz/project/pic-edit/experiments/review_20260908_v2/`）中，可直接用于加载与复测。
