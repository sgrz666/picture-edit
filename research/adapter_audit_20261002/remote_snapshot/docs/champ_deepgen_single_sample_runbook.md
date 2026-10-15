# CHAMP + DeepGen PoseAdapter 单样本运行手册

## 数据契约

一个样本目录必须包含：

```text
real_src_frame_0.png   # 图片 A，RGB
real_tgt_frame_60.png  # 与控制图同帧的真实目标 RGB
ctrl_normal.png        # Normal，RGB 3 通道
ctrl_depth.png         # Depth，灰度 1 通道
ctrl_openpose.png      # Pose/OpenPose，RGB 3 通道
ctrl_semantic.png      # Semantic/Mask，灰度 1 通道
```

Adapter 输入通道顺序固定为 `Normal RGB + Depth + Pose RGB + Semantic`。

## 本地无模型校验

```powershell
cd D:\codeplus\pictureedit
python run_champ_single_overfit.py `
  --data_dir inputs\champ_sample `
  --validate_only
```

成功时会输出 `[1, 8, 512, 512]`、控制值域和目标 RGB 图像统计，并以 `CHAMP sample validation: OK` 结束。若目标文件是 mask、控制文件缺失或通道不合法，脚本会在加载 DeepGen 前退出。

## DeepGen 接口补丁

远端目标文件：

```text
/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/deepgen_pipeline.py
```

应用补丁前先保存原文件：

```bash
cd /home/shangguanrz/project/pic-edit
test ! -e models/DeepGen-1.0-diffusers/deepgen_pipeline.py.bak.pre-control
cp -p models/DeepGen-1.0-diffusers/deepgen_pipeline.py \
  models/DeepGen-1.0-diffusers/deepgen_pipeline.py.bak.pre-control
sed -i 's/\r$//' patches/deepgen_pose_control.patch
patch --dry-run -p1 < patches/deepgen_pose_control.patch
patch -p1 < patches/deepgen_pose_control.patch
```

补丁为外层 `DeepGenPipeline` 和内层 `_SD3Pipeline` 增加：

```python
block_controlnet_hidden_states: Optional[list[torch.Tensor]] = None
control_scale: float = 1.0
```

内层 Pipeline 负责 CFG batch 复制和一次性 `control_scale`，调用端负责目标/参考 token 补零。

### 恢复原 Pipeline

仅在确认需要回滚时执行：

```bash
cd /home/shangguanrz/project/pic-edit
cp -p models/DeepGen-1.0-diffusers/deepgen_pipeline.py.bak.pre-control \
  models/DeepGen-1.0-diffusers/deepgen_pipeline.py
```

## 远端单样本训练

先检查共享 GPU：

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

确认显存足够后运行：

```bash
cd /home/shangguanrz/project/pic-edit
export DEEPGEN_PROJECT=/home/shangguanrz/project/pic-edit
/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  run_champ_single_overfit.py \
  --data_dir inputs/champ_sample \
  --output_dir experiments/champ_deepgen_formal_v1 \
  --max_steps 120 \
  --learning_rate 5e-5 \
  --save_every 30 \
  --inference_steps 30 \
  --guidance_scale 4.5 \
  --control_scale 1.0 \
  --seed 42
```

## DeepGen 无控制消融

该消融保持源图、文本、分辨率、推理步数、CFG 和 seed 与受控实验一致，但不创建或加载 Adapter，也不向 DeepGen 传入控制参数：

```bash
cd /home/shangguanrz/project/pic-edit
export DEEPGEN_PROJECT=/home/shangguanrz/project/pic-edit
/home/shangguanrz/miniconda3/envs/deepgen/bin/python \
  run_deepgen_no_control_ablation.py \
  --data_dir inputs/champ_sample \
  --controlled_dir experiments/champ_deepgen_formal_v1 \
  --output_dir experiments/deepgen_no_control_ablation_v1 \
  --inference_steps 30 \
  --guidance_scale 4.5 \
  --seed 42
```

`config.json` 中的 `adapter_created` 和 `control_arguments_passed` 必须均为 `false`。输出文件：

```text
01_INPUT_SOURCE_IMAGE.png          图片 A
02_GROUND_TRUTH_TARGET_IMAGE.png   对齐的真实目标，仅用于评估
03_DEEPGEN_NO_CONTROL.png          纯 DeepGen 输出
04_ABLATION_COMPARISON_GRID.png    源图/目标/无控制/Step 90/Step 120 对比
diagnostics.json                   三组输出的有效性、MAE、MSE 和 PSNR
summary.json                       耗时、有效性和最高 PSNR 组
```

## 输出解释

```text
01_INPUT_SOURCE_IMAGE.png          图片 A
02_GROUND_TRUTH_TARGET_IMAGE.png   训练标签，仅训练使用
03_INPUT_CONTROL_*.png             Pose/Normal/Depth/Semantic
04_MODEL_OUTPUT_STEP_000_*.png     零初始化 Adapter 基线
05_MODEL_OUTPUT_STEP_*.png         各保存点生成结果
adapter_step_*.pt                  Adapter checkpoint
loss_history.json                  loss、梯度与残差统计
diagnostics.json                   每张输出的塌缩检查
summary.json                       运行总结
06_EXPERIMENT_COMPARISON_GRID.png  完整横向对比图
```

数值有效条件：无 NaN/Inf、像素标准差不低于 `5.0`、近白和近黑比例均不高于 `0.95`。数值有效不等于视觉合格，仍需比较目标姿态、身份、服装和背景。

## 回归验证

```powershell
cd D:\codeplus\pictureedit
python -m unittest discover -s tests -v
python -m vram_lab.acceptance --root results\review_20260908_v2
```

预期所有单元测试通过，历史实验验收保持 `111/111 PASS`。
