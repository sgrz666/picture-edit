# CHAMP 控制条件与 DeepGen PoseAdapter 单样本联合设计

## 目标

建立一条可复现、可测试的单样本闭环：输入人物图片 A、语言编辑指令，以及从 CHAMP 目标帧得到的 Normal、Pose、Depth、Semantic 控制图，通过 PoseConditionAdapter 将结构残差注入 DeepGen DiT，生成保持 A 的身份、服装和背景、同时匹配目标姿态的输出图。

本阶段只证明单样本过拟合与接口正确，不宣称跨人物、跨动作泛化能力。

## 当前基线与问题

- DeepGen Transformer 已支持 `block_controlnet_hidden_states`，但内层 `_SD3Pipeline.__call__` 和外层 DeepGen Pipeline 的公开调用接口未传递该参数。
- 当前 `run_champ_single_overfit.py` 临时替换 `transformer.forward` 注入控制残差，训练与推理分别准备条件，存在行为漂移风险。
- 旧样本曾把 mask 当成目标 RGB；新一轮已改为真实源帧、真实目标帧与配套控制图，但缺少自动完整性校验。
- 60 步实验的最终输出能恢复人物与目标手势，但中间输出破坏严重，loss 波动较大，尚不能作为稳定接口验收结果。

## 选定方案

采用正式 Pipeline 参数方案，不继续依赖运行时 monkeypatch，也不在本阶段重构为完整 Diffusers ControlNet。

具体做法：

1. 在 DeepGen 内层采样 Pipeline 增加 `block_controlnet_hidden_states` 与 `control_scale` 参数，并在每个去噪步显式传给 Transformer。
2. 在 DeepGen 外层 Pipeline 暴露相同参数，负责将控制残差传入内层采样器。
3. 将控制图读取、8 通道拼接、Adapter 输出、目标/参考 token 对齐和 CFG 批次对齐放入可单测的集成模块。
4. 训练与推理共用同一控制预处理和残差对齐函数；训练仍冻结 DeepGen、VAE、LLM 与 Connector，只优化 PoseConditionAdapter。
5. 保留 Step 0 基线、半程输出和最终输出，但增加数值诊断，避免仅凭 loss 或单张图判断成功。

## 组件边界

### CHAMP 样本加载与校验

新增一个独立的 CHAMP 单样本加载器，输入为样本目录，输出：

- `source_image`: RGB 图片 A。
- `target_image`: 与控制图同帧的真实 RGB 训练标签。
- `control_tensor`: `[B, 8, H, W]`、`float32`、值域 `[0, 1]`。
- `prompt`: 描述动作变化但不覆盖人物身份与服装的语言指令。
- `metadata`: 原始路径、尺寸、帧号与通道定义。

8 通道顺序固定为：

```text
normal RGB (3) + depth (1) + pose RGB (3) + semantic/mask (1)
```

加载时必须拒绝以下情况：目标 RGB 是近似二值 mask、任一控制图缺失、尺寸或目标帧号不一致、通道数错误、出现 NaN/Inf。

### Adapter 残差适配

PoseConditionAdapter 接收 `[B, 8, H, W]`，输出 6 组 `[B, N_target, 1536]` 残差。

集成模块负责：

- 验证 `N_target = (H / 16) * (W / 16)`。
- 将参考图 token 对应区域补零，形成 `[B, N_target + N_reference, 1536]`。
- CFG 开启时，将残差按无条件/有条件批次复制到 `2B`。
- 应用显式 `control_scale`，默认 `1.0`。
- 在进入 Transformer 前验证层数、batch、token 数、hidden size、dtype 与 device。

零初始化 Adapter 的所有注入残差必须严格为 0，从而保证 Step 0 等价于原生 DeepGen 基线。

### DeepGen Pipeline 接口

外层公开接口增加：

```python
block_controlnet_hidden_states: Optional[list[torch.Tensor]] = None
control_scale: float = 1.0
```

内层 `_SD3Pipeline.__call__` 接收同样的参数，在 CFG 扩展完成后，将已对齐残差传给每一步 Transformer。Pipeline 不负责读取 CHAMP 文件，也不持有 Adapter；它只消费已经验证和对齐的残差，以保持职责清晰。

### 单样本训练与推理

训练流程：

1. 加载并校验同帧 CHAMP 样本。
2. 预编码图片 A、真实目标 RGB、文本条件。
3. 使用目标 latent 与噪声构造 flow-matching 训练样本。
4. Adapter 产生控制残差，通过正式 Transformer 参数参与预测。
5. 保存 loss、梯度范数、残差 RMS/最大值、checkpoint 与固定 seed 的阶段输出。

推理流程使用外层 DeepGen Pipeline 正式参数，不替换 `transformer.forward`。训练和推理必须调用同一个控制残差对齐函数。

## 错误处理与诊断

- 输入校验失败时，在加载模型前报出具体文件和原因，避免浪费 GPU 时间。
- 控制残差与目标/参考 token 不匹配时立即失败，错误信息包含期望与实际 shape。
- 每个保存点记录 loss、梯度范数、残差 RMS、输出均值/标准差和近白像素比例。
- 若输出近白、近黑、包含 NaN/Inf，标记该 checkpoint 为失败，不能作为最佳结果。
- 最佳 checkpoint 由数值有效性、目标姿态一致性和身份保持的组合结果选择，不直接使用最后一步。

## 测试策略

先写失败测试，再实现接口：

1. CHAMP 加载器能按固定顺序组成 8 通道，并拒绝 mask 冒充目标 RGB。
2. 零初始化 Adapter 的 6 层输出严格为 0。
3. 残差对齐正确补齐参考 token，参考区保持为 0。
4. CFG 时控制残差从 `B` 正确扩展为 `2B`。
5. 内层 Pipeline 确实在每个去噪步向 Transformer 传递控制残差。
6. `control_scale=0` 与无控制调用等价。
7. 现有 17 项单元测试与 111 项实验验收继续通过。

远端真实验收使用固定样本与 seed，产出：输入 A、Prompt、Pose、Normal、Depth、Semantic、真实目标、Step 0、最佳 checkpoint、最终 checkpoint、loss/残差诊断 JSON 和横向对比图。

## 验收标准

- 所有接口和回归测试通过。
- 运行脚本中不再修改 `transformer.forward`。
- 输入检查确认目标文件是真实 RGB，且控制图与目标动作同帧。
- Step 0 与原生 DeepGen 在同 seed 下结果一致。
- 训练后的有效 checkpoint 不是近白/近黑图，人物身份、服装和背景相对图片 A 保持，姿态明显向控制图靠拢。
- 输出目录包含可复现命令、环境信息、checkpoint、诊断 JSON 和完整对比图。

## 非目标

- 本阶段不训练通用 CHAMP 数据集。
- 不实现双人交互、FaceID、手部精修或跨数据集泛化。
- 不把 Qwen-Image-Edit 切换为生成主干。
- 不修改 CHAMP 本身的网络结构；CHAMP 在本阶段作为真实配对样本与几何控制来源。

## 代码与资产策略

- 可复现的集成代码、测试和补丁说明进入 Git。
- DeepGen 大模型权重、CHAMP 原始数据、训练 checkpoint 和大体积实验输出不进入 Git。
- 由于 `models/DeepGen-1.0-diffusers` 不在仓库中，正式接口改动同时保存为可重复应用的补丁或安装脚本，避免只存在于服务器副本。
