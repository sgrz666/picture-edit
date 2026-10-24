# src/pose_control/qwen21_full/ —— V6 全量迁移到 Qwen-Image-2.1

`qwen21_full/` 是 V6 控制架构向 Qwen-Image-2.1（7B 单流 DiT）的**完整迁移**：把 `v6/` 的全部共享文件（conditions / control_core / condition_injector / checkpoint / deepgen_adapter / controlled_pipeline / experts / geometry_encoder / token_encoders，外加 detail/face/hand/interface/reasoning 五个分支）**原样复制**过来，内部仍保持 1536 维表示；只在新文件里适配 Qwen 的 latent/text 输入坐标、残差输出坐标与官方目标流注入，并把残差注入扩展到 Qwen 的全部 32 层（6 组），补上面部监督、双人 Token 绑定（`PersonTokenBinder`）与训练基础设施。

## 里面有什么

- `native_geometry.py` —— 方案C的独立原生几何前端：从已有Full继承编码器/reasoner及高低尺度融合，输出默认512维token；不含旧SD3控制核或Union，尚未替换主干控制。使用与验证见 [NATIVE_GEOMETRY.md](NATIVE_GEOMETRY.md)。
- `native_body.py` / `native_integration.py` —— 方案C第二步：主干当层hidden驱动的512维控制分支、target-only写入及checkpoint安全的pipeline条件绑定；显式启用，原默认入口仍走旧Full。见 [NATIVE_BODY.md](NATIVE_BODY.md)。

与 `v6/` 同名的迁移文件（逐文件对应，见 [v6/README.md](../v6/README.md)），外加 qwen21_full 独有件：

- `qwen_adapter.py` —— `Qwen21FullAdapter` / `Qwen21FullControlContext`：`BRANCH_NAMES=("geometry","interaction","detail","face","hand")`，`LAYER_GROUPS` 把 32 层均分为 6 组注入；`validate_face_group_gains` 为运行时消融增益（非可训练参数）。
- `qwen_runtime.py` —— 原生 Qwen I/O（复制自 qwen21/runtime.py，支持 `native` 分辨率与 `resize_v1` 预处理）。
- `execution.py` —— 完整 V6 控制器接 Qwen21 的训练工具：原生 I/O、优化器选择、分支执行；固定 `BODY_CHECKPOINT`/`FACE_CHECKPOINT`/`MODEL`（Qwen-Image-2.1）路径，`ARCHITECTURE="qwen21_full_v1"`。
- `joint_training.py` —— 在不改继承架构的前提下做 Body/Face 联合优化（`freeze_evaluator`、`assert_exact_state`）。
- `frozen_bfs_lora.py` —— 加载未融合的 BFS 先验同时**冻结 Qwen Transformer**（仅 Transformer 参数冻结，不装 no-grad 前向包裹，梯度仍可流向外部 Adapter）。
- `data_runtime.py` —— `FaceConditionStore`/`HandConditionStore`/`QwenFullDataset`/`build_condition_bundle`：源身份 + 目标几何（沿用 V6.5 ArcFace/DINO 格式；目标帧可仅几何记录，其外观向量从不读取）。
- `face_supervision.py` / `face_loss_alignment.py` / `face_coordinates.py` —— 离线的目标脸合格性（布尔损失 mask，非嵌入/图像）、可微 ArcFace 对齐、以及把 RGB letterbox 变换应用到归一化脸坐标。

## 与上下层的关系

- 上游：基座 Qwen-Image-2.1（`models/Qwen-Image-2.1`）、`third_party.v65_face.*`、data 层与 `data_runtime.py`。
- 下游：由 `execution.py` 训练；推理经 `Qwen21FullAdapter` + `Qwen21FullControlContext` 包裹 Qwen transformer（经 `block_controlnet_hidden_states` 式注入）。
- 简化版见 [../qwen21/README.md](../qwen21/README.md)；共享的 V6 控制结构见 [../v6/README.md](../v6/README.md)。

## 注意与坑

- 顶层共享文件是从 `v6/` **复制**而来（非 import），修改通用控制逻辑时务必确认你改的是 qwen21_full 这一份还是 v6 那一份；五个分支目录同样逐文件一致。
- `execution.py` 中 hand 默认 `hand_strength=0, hand_mode='off'`——即**默认关闭手分支**，要用手控制需显式开启。
- `frozen_bfs_lora.py` 只冻结 Transformer 参数；若要彻底阻断 Qwen 梯度，需调用方自行处理，本模块不提供 no-grad 包裹。
- `smplx_global` 仍为 26 维（betas10 + root rot6d6 + transl3 + cam7），不含关节姿态，与 v6 口径一致。

## Native Depth / CHAMP (2026-10-23)

独立Depth输入契约、CHAMP CNN/空间attention、512维残差与新checkpoint已接入Native。见 [NATIVE_DEPTH.md](NATIVE_DEPTH.md)。旧Full默认入口保留。

## Native Full V4（架构交付，尚未验收）

V4独立Runtime已接通单/双人共享Body、Depth、Interaction、Identity和Native Face/Hand；4096→512稀疏Query、六个零残差头、pixel owner与Static对照。见 [NATIVE_FULL_V4.md](NATIVE_FULL_V4.md)。本轮仅语法/AST/SHA核对，未运行测试、模型、迁移或训练；SMPL-X资产待补齐。

## Native Full V4.1 Hi4D Hybrid

Hi4D新增`hybrid_hi4d`数据Profile：保留V3原生SMPL Body21／Depth／Interaction／Contact，使用可选原图DWPose Face／Hand；不要求Hi4D补齐完整SMPL-X。共用V4网络与严格checkpoint接口，不新增迁移，原完整SMPL-X路径保留。三个Hybrid模块、离线CLI、独立缺失模态和来源校验见[NATIVE_FULL_V41_HYBRID.md](NATIVE_FULL_V41_HYBRID.md)。本轮只完成静态交付，真实数据和运行时尚未验收。
