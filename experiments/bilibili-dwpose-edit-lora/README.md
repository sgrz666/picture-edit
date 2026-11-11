# Bilibili 双图 Edit LoRA 首轮探索

验证 Qwen-Image-2.1 的纯 Edit LoRA 能否从 A 的 RGB、B 的原始 DWPose 骨架和统一短指令生成 B。冻结基座、Qwen3-VL 和 VAE，只训练 attention Q/K/V/Out 的 LoRA。2026-11-11 已完成 300 次真实更新、100 步断点恢复、原生 LoRA 重载及 40 张对照/诊断图。

首轮结论：训练接口已跑通，尚未看到稳定的生成质量收益。10 对验证样本的 PCK@0.02 从 0.9147 到 0.9202，PCK@0.05 均为 0.9750，Face ID cosine 从 0.5106 到 0.5026；原始 Qwen 本身已能响应换骨架。小样本结果不能证明 LoRA 改善了姿态与身份的综合表现。

训练阶段本进程 CUDA reserved 峰值约 15.65 GiB；所有阶段最低空余显存 44.46 GiB、最低可用系统内存 86.54 GiB。冻结编码阶段 RSS 峰值约 18.44 GiB。完整指标和资源表见 [`report/REPORT.md`](report/REPORT.md)，另有[验证集对照图](report/validation_overview.png)与[训练曲线](report/training_curves.png)。

`scripts/` 是本实验专用的数据、缓存、训练、推理和评测入口；`configs/pilot.json` 固定 384×672 画布、rank/alpha=16、seed=42、300 次更新。`tests/` 检查数据和条件协议。本地 `source/diffusers/` 是官方代码快照，服务器对应位置为 `/data/shangguanrz/LoRA/source/diffusers/`；字节级 Git blob 身份记录在 `provenance.json`。服务器新产物位于 `/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora/`，原媒体仍在原位置。

输入图按 `[A, B 的骨架]` 排列；只有 B 的 RGB 生成区域计算 Flow Matching Loss。DWPose 渲染采用数据中已有的 `valid_keypoint_mask`，不把原始 SimCC 分数当概率重新筛点。B 和骨架使用同一等比缩放和 padding。

本轮使用原 train/val/test 的 9,520/1,490/1,146 对索引，从 train 固定取 8 个视频各 4 对同 A 多 B，从 val 取 5 个视频各 2 对；test 不参与调参。上游是现有 Bilibili 原始配对和 DWPose JSON；下游为独立纯 LoRA 实验，尚未接入 Native V5。

本地 CPU 契约复核：

```powershell
python -m unittest discover -s experiments/bilibili-dwpose-edit-lora/tests -v
```

## 复跑与使用

服务器沿用 `qwen21`，未安装或升级依赖。运行前使用服务器已有的进程级 NVML 兼容库；下面是新一轮完整命令，每次自动生成新名字，避免覆盖本轮产物：

```bash
ssh sg
cd /data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora
export LD_LIBRARY_PATH=/data/shangguanrz/仪表盘/compat:${LD_LIBRARY_PATH:-}
QPY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
ROOT=/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora
TAG=$(date +%Y%m%dT%H%M%S)
"$QPY" -m unittest discover -s tests -v
"$QPY" scripts/run.py prepare-data --name "data_$TAG"
"$QPY" scripts/run.py cache --name "cache_$TAG" --data "$ROOT/data/data_$TAG"
"$QPY" -u scripts/campaign.py --config configs/pilot.json \
  --data "$ROOT/data/data_$TAG" --cache "$ROOT/cache/cache_$TAG" --tag "$TAG"
"$QPY" scripts/build_report.py --config configs/pilot.json --tag "$TAG"
```

`campaign.py` 会封存运行脚本，按顺序做 1 步真实反向、LoRA 原生重载/推理、100 步训练、恢复至 300 步、同输入 baseline/LoRA 推理及 CPU 评测。任何阶段失败会记录并停止；不会修改并行 V5 任务。阶段记录在 `output/campaign_<TAG>/events.jsonl`，详细日志与运行源码快照同目录。

本轮已准备的数据与缓存可直接复用：

```bash
DATA=$ROOT/data/data_20261111_v1
CACHE=$ROOT/cache/cache_20261111_v1
# 只做原始 Qwen 推理；给新 NAME，不带 --adapter。
NAME=baseline_$(date +%Y%m%dT%H%M%S)
"$QPY" scripts/run.py infer --name "$NAME" --data "$DATA" --cache "$CACHE"
# 加载本轮最终 LoRA，自动生成相同输入的 baseline 和 LoRA 对照。
NAME=compare_$(date +%Y%m%dT%H%M%S)
"$QPY" scripts/run.py infer --name "$NAME" --data "$DATA" --cache "$CACHE" \
  --adapter "$ROOT/output/pilot_20261111_v1/checkpoint-000300"
"$QPY" scripts/run.py evaluate --name "eval_$NAME" --inference "$ROOT/output/$NAME"
```

`infer --limit 1` 可缩小到一组真实推理检查。默认推理覆盖 10 个验证配对、8 个训练视频各一个配对，并增加同 A 换骨架、空白骨架诊断。两种诊断都会同时改变 Qwen3-VL 与 VAE 的输入，Prompt、seed 和采样参数不变。

每个 `checkpoint-XXXXXX/` 同时包含可由 `QwenImage21Pipeline.load_lora_weights()` 加载的 `pytorch_lora_weights.safetensors`，以及本实验用的 `training_state.pt`。恢复训练须使用同一个 `--name`、配置和缓存，并指定 `--resume`；先检查 `checkpoint.json`，不从未完成的目录恢复。配置或模型改变会触发身份检查失败，需新建对应数据/缓存，不能原地改写。

本轮最终权重为 `$ROOT/output/pilot_20261111_v1/checkpoint-000300/pytorch_lora_weights.safetensors`。服务器完整报告位于 `$ROOT/report/pilot_report_20261111_v1/`，本地 `report/` 保留轻量副本。对照图的列顺序为 A、B 骨架、B RGB、原始 Qwen、LoRA。

完整性复核会检查源数据清单、43 份缓存、100/200/300 检查点、40 张图片及资源边界；报告名每次使用新值：

```bash
"$QPY" scripts/audit.py --config configs/pilot.json --tag 20261111_v1 \
  --report-name "integrity_$(date +%Y%m%dT%H%M%S).json"
```

完整性审计结果见 [`report/integrity_final_v2.json`](report/integrity_final_v2.json)：300 次更新、43 份缓存、40 张图片和全部检查点校验通过。最终代码的 16 项 CPU 契约在本地与服务器均通过。实际运行源码封存在服务器 `output/campaign_20261111_v1/source_runtime/`；后续补充的版本/输入文件检查和审计工具不会改写该快照。

服务器保存大文件；Git 仅保留代码、配置、官方源码快照和轻量报告。
