# deepgen/ —— DeepGen 上游官方仓库（不纳入 Git）

本目录是 DeepGen 官方代码仓库的克隆，属于第三方上游代码，不纳入本项目仓库，需要时自行克隆。

## 获取方式

```bash
git clone https://github.com/deepgenteam/deepgen.git deepgen
```

包含官方的 `src/`、`scripts/`、`configs/`、训练/推理文档（TRAIN.md / INFERENCE.md / EVAL.md / DATA.md）等。

## 与本项目的关系

- 本项目实际加载的是 **diffusers 格式权重** `deepgenteam/DeepGen-1.0-diffusers`（下载说明见 `../models/README.md`），该版本自带 `deepgen_pipeline.py`，**无需克隆本目录即可推理**。
- 本目录用于查阅官方训练框架、模型结构定义和数据格式；项目自身的适配代码在上级目录的 `src/`（外挂姿态适配器与控制信号生成）和 `vram_lab/`（显存实验）中。
