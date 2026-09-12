# DeepGen 无动作约束消融实验设计

## 目标

在已经完成的 CHAMP-style PoseConditionAdapter + DeepGen 单样本实验基础上，增加一个严格的纯 DeepGen 对照组。对照组保持源图、文本指令、随机种子、CFG、分辨率和推理步数不变，只移除 Adapter 及其动作控制残差，用于判断当前心形手势有多少来自 DeepGen 自身的文本理解，又有多少来自 Normal、Depth、Pose 和 Semantic 约束。

## 实验定义

对照组采用纯 DeepGen 路径：

```text
源图 A + 相同文本指令
        │
        ▼
     DeepGen
        │
        ▼
   无控制输出
```

运行期间不创建 `PoseConditionAdapter`，不加载 Adapter checkpoint，也不向 Pipeline 传入 `block_controlnet_hidden_states` 或 `control_scale`。这比传入零残差或设置 `control_scale=0` 更容易审计，能够保证动作控制分支没有参与推理。

## 固定变量

- 数据目录：`inputs/champ_sample`
- 源图：`real_src_frame_0.png`
- 目标真值：`real_tgt_frame_60.png`，仅用于对比和指标计算
- 文本指令：与正式受控实验完全相同
- 分辨率：512 × 512
- 推理步数：30
- CFG：4.5
- Seed：42
- DeepGen 模型与受控实验相同

唯一改变的变量是是否注入 Adapter 控制残差。

## 实现边界

新增独立运行脚本 `run_deepgen_no_control_ablation.py`，避免在正式训练脚本中加入容易混淆的模式分支。新增一个可单元测试的纯 DeepGen 调用帮助函数：它拒绝任何控制相关参数，然后原样调用 Pipeline。

脚本加载源图和目标真值，执行一次纯 DeepGen 推理，并读取现有受控实验的 Step 90 和 Step 120 输出生成横向对比。若受控输出、模型目录或样本文件缺失，脚本应给出明确错误并停止，不能静默省略对比组。

## 输出

远端输出目录：

```text
/home/shangguanrz/project/pic-edit/experiments/deepgen_no_control_ablation_v1/
```

本地审阅副本：

```text
results/deepgen_no_control_ablation_v1/
```

输出文件包括：

- `01_INPUT_SOURCE_IMAGE.png`
- `02_GROUND_TRUTH_TARGET_IMAGE.png`
- `03_DEEPGEN_NO_CONTROL.png`
- `04_ABLATION_COMPARISON_GRID.png`
- `config.json`
- `diagnostics.json`
- `summary.json`

对比图列顺序固定为：源图、目标真值、纯 DeepGen、受控 Step 90、受控 Step 120。

## 指标与判定

记录纯 DeepGen、Step 90 和 Step 120 相对目标真值的 MAE、MSE、PSNR，并沿用现有输出塌缩检查，包括像素方差、近白比例、近黑比例和有限值检查。

数值指标只用于辅助判断。最终视觉审阅重点是：

1. 是否形成目标心形手势；
2. 手臂和双手位置是否匹配目标姿态；
3. 身份、服装和背景是否保持；
4. 有无白屏、肢体破损或多手指等退化。

## 测试策略

先编写失败测试，证明纯 DeepGen 调用帮助函数会拒绝 `block_controlnet_hidden_states` 和 `control_scale`，且正常调用中不会向 Pipeline 发送这两个参数。实现后运行新增单元测试、项目完整测试、远端实际推理以及历史 111 项验收。

最终交付必须同时包含可复现命令、对比图、诊断 JSON 和远端输出路径。
