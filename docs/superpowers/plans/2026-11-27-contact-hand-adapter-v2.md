# Contact Hand Adapter V2 实现计划

**目标**　实现可独立加载、训练和推理的 V2。V1 规格继续作为数据契约与评测背景。

**架构**　四手槽 → 层次化静态 token、对齐空间特征和多尺度接触图 → 每次 DiT forward 重置的动态手部状态 → 原生 Query 驱动的独立 K/V 分支。全图与联合 ROI 共用模型和权重。依赖 PyTorch、仓库 CanvasSpec 与 Qwen-Image 2.1 diffusers 接口。

- [x] 在 `tests/test_contact_hand_adapter.py` 先锁定三态接触、同侧不同人、可选条件、32/64 表面 token、非方形画布映射和全空旁路。实现 `src/pose_control/contact_hand_adapter/conditions.py`、`geometry.py` 和 `model.py`。
- [x] 用真实的小型 Qwen-Image 2.1 验证零初始化、零强度、ROI 写入、跨层状态与非重入重算。`integration.py` 保留常规与 Flex 原生处理器并读取其 Query。
- [x] `runtime.py` 及 `scripts/run/run_contact_hand_adapter.py` 接通全图、局部和全图接局部。测试联合 ROI 等比往返、贴回区域和独立 checkpoint。
- [x] `training.py` 和 `scripts/train/train_contact_hand_adapter.py` 接通区域 FM、CPU 预检、权重身份检查与中间 checkpoint。真实小型冻结 VAE 验证 RGB 辅助项的有效梯度及未知标签跳过机制。
- [x] 参数、消融、条件包与使用命令写入 [V2 README](../../../src/pose_control/contact_hand_adapter/README.md)。Contact Hand、Native V7、Canvas、SDPose 合计 48 项测试通过。

原生 Attention 的整个分支继续由 diffusers 计算。Hand branch 借鉴 OminiControl 的条件交互，使用分开归一化的概率分布和独立 K/V。 [OminiControl 原文](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_OminiControl_Minimal_and_Universal_Control_for_Diffusion_Transformer_ICCV_2025_paper.html)
