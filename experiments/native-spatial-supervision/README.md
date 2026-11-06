# Native Adapter 离线空间监督

目的：对 GT 的穿衣人体、有效身体关节、可信可见手部生成离线软区域，让 Native V5 可配置地使用空间加权 Flow Matching。先验收 20 张，再处理 Bilibili 全量 1,960 帧；不启动训练、不筛掉原训练样本。

入口为仓库 `scripts/data/build_native_spatial_weights.py`，纯几何逻辑在 `src/pose_control/qwen21_full/native_v5/spatial_masks.py`。服务器产物根目录为 `/data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1`，`pilot20_v2/` 与 `full/` 分开保存。每帧有原图分辨率 Body Mask、Canvas Joint/Hand 图、分量 NPZ、权重热图、latent 权重图及 7 栏对照图；index 以原图 SHA256 和 Canvas 指纹索引。`frames/<frame_id>/ready.json` 声明每张产物的绝对路径和 SHA；重试保留独立的 `attempt_<uuid>/`，不覆盖中断产物。

已完成 20 张 CUDA 预览，6 张启用可信手部区域，生成阶段耗时 35.33 秒（不含加载模型），GT／组件／Canvas 审计通过，Qwen64 零系数误差与 Padding 权重均为 0。已人工查看总览及手部近图：区域与 GT 对齐，手部增强在 latent 网格中保留；少数花纹服饰有分割碎孔，Body Mask 是弱区域权重，不能当精确分割标注。全量正在生成，尚不能宣称全量完成或姿态指标改善。默认预览系数 Body=.5、Joint=.5、Hand=1，像素权重最高 2.5，真实背景基础权重 1，Padding 为 0。手部遵守原 loss_mask 与 hand_raw_pose_usable，两只手独立判断。

上游是 `/data/shangguanrz/datasets/bilibili/frames/frames.jsonl` 及已有 SAM2.1 权重；保留当前 Native 数据选择的 `sapiens_v1` 版本标签，实际区域来自 GT 分割和已有 2D 关键点，不重新拟合 SMPL-X。下游是 Native V5 Training 的可选 Source 空间缓存与 Loss 系数；旧 face/hand 区域项与新空间项互斥，零系数保持旧训练路径。此实验不替代数据划分、原有训练准备或姿态评测。

## 使用与验证

在服务器使用已有 deepgen 环境做 SAM 推理，不安装任何包。以下输出目录必须尚不存在；只在源代码、输入和参数完全一致时用 `--resume` 恢复，修改配置应更换目录。

```bash
cd /home/shangguanrz/project/pic-edit
/home/shangguanrz/miniconda3/envs/deepgen/bin/python scripts/data/build_native_spatial_weights.py \
  --frames /data/shangguanrz/datasets/bilibili/frames/frames.jsonl \
  --output-root /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v2/full \
  --limit 0 --canvas-long-side 512 --device cuda --torch-threads 4
```

`--limit 20` 为预览；`--limit 0` 保留全量。`--canvas-long-side 512` 与当前 Bilibili Native 装配一致，GT 1080×1920 对应 288×512 Canvas 和 18×32 latent；该网格不适用于采用其他 Canvas 的训练，读取器会拒绝错位。

审计本轮全量缓存（审计报告必须用新文件名）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/audit/audit_native_spatial_weights.py \
  --index /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/full/index.json \
  --frames /data/shangguanrz/datasets/bilibili/frames/frames.jsonl \
  --output /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/full/audit_qwen64_repeat.json
```

在既有训练配置的每个所选训练／验证 Source 加 `spatial_weights: {path: 索引绝对路径, sha256: 实测索引SHA}`，对应 route 的 loss 使用 `spatial_body=.5`、`spatial_joint=.5`、`spatial_hand=1`、`spatial_max=2.5`，原 `face`、`hand` 局部 velocity 项设为 0。调三个空间系数可直接复用 NPZ，无需重跑 SAM；训练时不加载分割／关键点模型。完整配置与 preflight 命令见 `src/pose_control/qwen21_full/native_v5/training/README.md`。不得将本轮缓存数值验收当成真实 Qwen 前反向或训练通过报告。

```bash
python -m pytest tests/test_native_spatial_masks.py tests/test_native_spatial_weights.py tests/test_native_spatial_audit.py -q
```
