# src/ —— 人物图像姿态编辑系统的全部可训练与推理代码

`src/` 是 pic-edit 的 Python 源码根目录，承载项目的全部核心逻辑（仓库根只放配置、模型权重与 `experiments/`，两阶段基座流水线在仓库其他位置的 `workflow/`）。

项目要解的题是：给定源人物 A 的图像与参考人物 B 的 SMPL-X 姿态，保持 A 的身份（Face ID）、服饰材质与背景不变，把 B 的姿态与表情重定向到 A 上。`src/` 内的代码按三条技术线组织：

1. **DeepGen-1.0-diffusers 线**：基座是 2.47B 参数的 MMDiT 双流模型，推理时 frozen；在其上挂载自研 **Unified SMPL-X Adapter V6**（trainable）。条件编码 → 循环控制核心 → 残差注入（block 0–22，第 23 层不注入）；后续演进为 V6.4 detail 分支、V6.5 face 分支、V6.6 hand 分支。
2. **Qwen-Image-2.1 迁移线**：基座换成 7B 单流 DiT。其中 `pose_control/qwen21/` 是简化实现（body/face/dual 三个独立控制头），`pose_control/qwen21_full/` 是把 V6 全量迁移到 Qwen 的完整实现（32 层注入、面部监督、双人 Token 绑定）。
3. **workflow/ 基座流水线**（不在 `src/` 内）：无 adapter/LoRA 的两阶段方案，A 保外观、B 供动作，SMPLest-X 估计 + DWPose-134 骨架 + PE-I2I 提示词编译。

## 里面有什么

- `pose_control/` —— 上述前两线的控制适配器核心（**训练侧**），详见 [pose_control/README.md](pose_control/README.md)。
- `pose_edit/` —— 两条 workflow 线（单人 `workflow/`、双人 `workflow-multiple/`）共用的**推理侧共享层**：缓存指纹与阶段溯源、SMPLest-X 估计器封装、DWPose-134 关节表、geometry 定标、外观相似度、提示词契约。与 `pose_control/` 的区别是「推理侧流水线共享层 vs 训练侧适配器」，二者互不依赖。详见 [pose_edit/README.md](pose_edit/README.md)。
- `adapter/` —— 早期/独立的姿态条件适配器（`pose_adapter.py`），ControlNet 式条件编码雏形。
- `controls/` —— 骨架可视化与生成辅助（`generator.py`）。
- `data/` —— iPER / TikTok / Hi4D 数据集读取与几何/letterbox 预处理。
- `evaluation/` —— 四臂评测（身份/姿态指标）、无 GT 评测指标集与手部合理性判断、DWPose-134 ONNX 推理封装。
- `integration/` —— CHAMP / DeepGen 样本校验与残差对齐。

## 与上下层的关系

- 上游：仓库根的模型权重（DeepGen、Qwen-Image-2.1）、`third_party/`（v65_face.xdyna、visual_persona 等锁定依赖）、`datasets/`。
- 下游：训练脚本（`pose_control/qwen21_full/execution.py`、`joint_training.py` 等）与推理入口调用 `pose_control/` 下导出的适配器与 pipeline；评测脚本调用 `evaluation/`；两条 workflow 流水线（单人 `workflow/`、双人 `workflow-multiple/`）`import src.pose_edit.*` 复用推理侧共享层。
- 仓库根 README（人工正稿）不在本目录覆盖范围内，整体导航见 [../README.md](../README.md)。

## 注意与坑

- 多处直接 `import third_party.*`（如 `third_party.v65_face.xdyna`、`third_party.v65_face.stableanimator`），这是该目录锁定的外部实现，缺包时相关模块无法导入。
