# Bilibili：新区域目标 + SDPose + GRPO pilot

## 目的

用真实 Bilibili 数据验证 `uniform` 与 `region_balanced_v1` 的训练接入，并比较 RN SFT 后继续 SFT/GRPO。训练骨架和生成图评价均使用 SDPose，保留全部生成图。第一轮是视频均衡的有限预算集成 pilot，不称作全量、多 seed 的收益结论。

## 里面有什么

- `scripts/prepare.py`：从旧 Native manifest 只读选择每个 train/val 视频各 2 对，创建新的 SDPose Canvas 骨架条件、独立逐手区域与评价清单；不覆盖原 RGB、标注或缓存。
- `scripts/pack_runtime.py`：生成带逐文件 SHA 的源码包，不含模型、数据和 checkpoint。服务器在新运行根的 `source/runtime/` 解包。
- 共享目标在 `src/pose_control/qwen21_full/region_objective.py`，sidecar 在 `region_supervision.py`；默认 legacy 配置签名保持兼容。
- SFT 每臂预算为 Body 100、Head 50、Full 100、Joint 50，共 300 个 optimizer updates；B1、accumulation=4、seed42。两臂初始化、数据和阶段 ID 一致，仅区域模式不同。
- RL 从 RN SFT 末权重出发，另有相同 optimizer-update 的继续 SFT 对照。实际 RL 预算、采样与 Reward 参数写入运行记录后执行，不把额外候选采样成本称作与 SFT 等算力。

## 关键状态与复测

共享 loss/配置/sidecar 的 CPU 数值、绑定与旧行为回归已通过；相关 Native V7 集成回归 94 项通过。真实数据预检也已通过：78 train / 10 val，覆盖 39/5 个原视频，88 个独立区域侧车；10 对验证输入面板已核对。

服务器运行根为 `/data/shangguanrz/pic-edit-experiments/bilibili-region-sdpose-sft-rl-20261120-v1`。用户选择等待现有 Hi4D 作业完成，本实验已由 PID 2526964 排队，等待 PID 2370979 退出。源码固定在 `source/runtime-v3`，实际状态看 `logs/events.jsonl` 和进程，不能把排队视作训练完成。GPU/训练/SDPose/GRPO 结果以该目录报告为准。

`scripts/run.py` 串行运行两臂 SFT、从同一 RN 检查点继续 SFT 10 updates 和 GRPO 10 updates。`scripts/stage.py` 复用共享 Trainer，保存阶段生成及逐图 SHA；`scripts/evaluate.py` 在原 SDPose 环境预测，再在 qwen21 的 CPU 身份环境计算身份/背景。`scripts/rl.py` 存真实随机转移、候选图、组内奖励、重放 gate、固定 SFT reference 和独立 RL 快照，未接入 LoRA 或修改 qwen21 包。

```powershell
python -m pytest tests/test_region_objective.py tests/test_region_supervision.py tests/test_region_rewards.py tests/test_native_region_rl.py tests/test_sdpose_backend_binding.py tests/test_bilibili_region_prepare.py -q
```

## 与已有实验的关系

复用 Bilibili 原视频划分、Native 几何与冻结模型。新 SDPose 2D 骨架进入新的 Qwen 条件版本；已有 SMPL-X/Sapiens 三维条件仍如实保留，SDPose 不伪造三维参数。原 LoRA Pose/ID GRPO 提供随机流、重放与固定 SFT reference 思路，本轮只优化 Native Adapter；原 LoRA 权重和环境不改动。
