# 小红书双人 Native V7 SFT

在小红书同场景照片和视频中，训练 Native Full V7 的双人姿态编辑。Qwen-Image-2.1 保持冻结。4Ref 输入是双人 Source、目标合并骨架，以及按 Source 左右身份排序的两张目标单人骨架。模型训练用目标 RGB 作监督。

## 数据和训练

- 数据位于 `/data/shangguanrz/datasets/xiaohongshu(two)`。照片和视频从当前清单回接到数据集配对目录。每张 RGB 的 SHA256 必须与几何帧索引一致。配对使用 `bg_sim >= 0.83`、`l1_bg <= 26`、`person_col_sim >= 0.52`，两位人物重新检测并要求跨帧身份余弦 `>= 0.38`、相对另一人物领先 `>= 0.05`。按 note、视频和重复 RGB 的连通组划分 train/validation/test，保留拒绝原因。
- Canvas 保持原宽高比，最长边上限 768，按 32 对齐。Native 条件使用已完成的 SDPose 引导 SMPL-X 拟合，经原模型全网格 FK 核验。Source 使用 glintr100 身份、DINOv2 局部特征和带人物归属的细节 Mask。冻结 SDPose 条件以 Wholebody + YOLO11x、置信度 0.30 为口径。当前拟合缺少有效表情和明确的身体接触标签，面部表情参数和 Contact FM 权重关闭。
- 初始化用 `bilibili20k-v7-dual192new.pt`，SHA256 `17b23faec7c39c1c8e0b99f4bd05b82bbfe646eb3695a9336459d44796c72331`。它继承 Bilibili S3 20,000 步 `d1a655a8…` 的 1,054 个共享张量和双人 180+12 步 `eca8f5ec…` 的 89 个 V7-only 张量。新训练建立独立 optimizer 和计数。Qwen model manifest SHA256 为 `a8c34c15ce9632313dec2f1d96ff2629315d6a8d94b657b4539c196b2f6bcebb`，身份 evaluator SHA256 为 `bf059ec89614024682639ef3eaffecd3ef0933aae9c114df4097636d800269d6`。
- 正式预算 3,000 个 optimizer updates，S0/S1/S2/S3 为 300/400/800/1,500，梯度累积 4。用 V7 默认逐人物 FM 和身份对比，加 Source ID 0.2，Source/Target Canvas 可对齐时加背景保持 0.1。每阶段检查实际数据、GPU 梯度和已提交父权重，每 100 updates 保存并跑 validation loss probe。独立 test 仅供训练后评估。Qwen 编码、Native 装配分卡并行，现有训练入口是单 GPU、batch 1。

## 进度及怎么使用

服务器根目录为 `/data/shangguanrz/pic-edit-experiments/two/qwen21_hi4d_dual/xiaohongshu-dual-sft-20261127-v1`。

`scripts/select_pairs.py` 生成 `data/selection.json`。 `scripts/qwen_cache.py` 准备 4Ref 提示词、骨架与冻结编码， `scripts/native_assets.py` 构建训练格式及监督。 `scripts/run_pipeline.py` 监管准备、 `scripts/train_sft.py` 调用现有 V7 Trainer。源码快照在 `source/main-c18924f4-dual-canvas-v1`，DINO 提取复用 Bilibili 实验的 `scripts/appearance.py`。验证这轮新增的 Canvas 修复可运行 `python -m pytest tests/test_qwen21_native_dual_v3.py tests/test_native_v7_runtime.py tests/test_native_v7_training.py -q`。

```bash
# 登录 sg 后执行
ROOT=/data/shangguanrz/pic-edit-experiments/two/qwen21_hi4d_dual/xiaohongshu-dual-sft-20261127-v1
tail -n 8 "$ROOT/logs/select-pairs-v3.log"
tail -n 8 "$ROOT/logs/production-supervisor.log"
find "$ROOT/production/output/checkpoints" -name '*.commit.json' | sort | tail -n 5
```

生产数据和梯度报告在 `production/sft/`，每阶段的 `S0-committed.json` 等记录权重路径和 SHA。拿到 commit 后，可在一张空闲卡上用同一份验证清单和相同种子对照初始化与训练后权重。替换 `CHECKPOINT` 为实际已提交文件路径，每次出图使用新输出目录。

```bash
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
SRC="$ROOT/source/main-c18924f4-dual-canvas-v1"
CUDA_VISIBLE_DEVICES=2 "$PY" "$SRC/scripts/run/run_qwen21_native_full_v7.py" \
  --checkpoint "$CHECKPOINT" --manifest "$ROOT/production/validation.jsonl" \
  --model /data/shangguanrz/pic-edit-models/Qwen-Image-2.1 --data-profile full_smplx \
  --pair-index 0 --seed 42 --steps 20 --output "$ROOT/production/output/validation-review-v1"
```

## 起步证据和已有实验关系

双人 192 步贡献 V7 新模块，见上一轮 `native-v7-source-preservation-20261121`。原图缩到训练 Canvas 时，曾触发双人 Relation Raster 的深度有效掩码不一致。修复后两对真实照片通过 Native 资产和 4Ref 缓存装配、真实数据和 GPU 梯度门禁，相关本地回归 21 项通过。小规模 S0 完成并提交第 1 个 optimizer update，梯度范数 0.17551，权重 SHA256 `2e3956a4280fb59268de230af0a89093232b239f5ba08d4c7b99f4efee700c96`，凭据在 `smoke-v2/sft/complete.json`。正式训练从相同迁移初始化独立起跑，效果以生产 checkpoint 的验证出图为准。
