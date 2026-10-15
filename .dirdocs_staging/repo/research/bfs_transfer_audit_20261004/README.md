# research/bfs_transfer_audit_20261004/ —— BFS 双人控制能否顶替 Body adapter 的探针审计

> 工作区路径：`/home/shangguanrz/project/pic-edit/research/bfs_transfer_audit_20261004`　|　本级：叶子目录，7 个实证文件（`目录说明.md`、`README.md` 为 10-11 生成件，不计）　|　体积：472 KB（du 实测）

## 一、这一层在干什么

2026-10-04 的问题：**已训好的 BFS head LoRA（`bfs_head_v1.1_qwen_2.1.safetensors`）能否替代 qwen21 的 Body adapter 做姿态控制，768 分辨率探针结果是否可信**。做法是在 iPER face 验证对（`/home/shangguanrz/nihaochen/qwen21_face/face_val_pairs.jsonl` 取 6 对，mostly 两个身份）上跑 512 探针（30 步、seed 42、body checkpoint 为 `experiments/qwen21_iper_body/full_20k/checkpoint_last.pt`），再对已有的 768 探针 PNG 做 CPU 复评。归属 qwen21 单人控制线，下游是 10-08 的 BFS body-face 联合设计调研。

## 二、层级地图

叶子目录，无子目录。7 个文件两两配套：3 组 JSON（生成指标）+ 1 组复评 JSON + 1 份汇总 + 1 张对比面板图 `panel_004.png`（414 KB）。

## 三、关键文件与实证内容

- `results.json`（14 KB）：512 探针四臂对照，n=6。Body adapter `pck_mean=1.0`、identity_mean 0.4169；stage2 无 LoRA / bfs_head / bfs_head_rev 三臂同为 pck 0.9902、identity 0.327~0.334；`source_target_ceiling.identity_mean=0.6272`（源-目标身份余弦的经验参照）。每对含 `pose_pck_002/005`、17 关节匹配、耗时约 4.5~4.9 s。
- `existing_probe_summary.json`：核心横向结论——`bfs_768_id_A_mean=0.4920` > `nolora_768=0.4476` > `bfs_512=0.3281`；`bfs_768_better_than_body_512=5`（6 对中 5 对）；`limitations` 数组明确 6 对太少、768 探针未记录完整采样设置。
- `probe768.json` / `probe768_nolora.json`：逐对 768 与 512 的 identity 对比、无 LoRA 768 的逐对分数（注意其中 `pck` 全为 0.0——是当时坐标系错的原始记录，不要直接引用）。
- `corrected_768_pose.json`：CPU 复评修正版——把 DWPose 检测按实际宽高归一化回 512 坐标系后，`bfs_head_768 pck_mean=0.9706`、`nolora_768 pck_mean=0.9902`，6 对逐对列出；这是 PCK 的权威版本。
- `cpu_compatibility_verification.json`：BFS 权重映射到已安装 qwen21（diffusers 0.41.0.dev0 / peft 0.20.0 / torch 2.8.0）的 CPU 兼容性核验，352→384 张量键、192 目标模块、rank 64、`split_projection_max_absolute_error=0.0`，passed=true。

## 四、注意事项 / 调用路径

- 采纳/否决结论：**768 的 BFS 探针在身份保持上优于 512 Body（5/6 对），但样本只有 6 对且两臂 PCK 近乎持平（0.97 vs 0.99），不足以支持"BFS 顶替 Body adapter"**；该审计的价值在于固化了 PCK 复评方法（`corrected_768_pose.json`）和权重转换等价性证据。
- 引用 PCK 时一律用 `corrected_768_pose.json`，勿用 `probe768_nolora.json` 的原始 pck。
- 权重路径 `/home/shangguanrz/models/bfs/*.safetensors` 与 `nihaochen/qwen21_face/` 为外部依赖，清理时不在本目录。
