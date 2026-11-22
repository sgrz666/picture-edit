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

服务器运行根为 `/data/shangguanrz/pic-edit-experiments/bilibili-region-sdpose-sft-rl-20261120-v1`。用户选择等待现有 Hi4D 作业完成。2026-11-21 04:47 UTC 核对：本实验等待进程 PID 2554458，等待 Hi4D PID 2370979 退出；尚未开始 Bilibili GPU 缓存或训练。源码固定在 `source/runtime-v4`，源码包 SHA 为 `6704dc8b126a7166bd34f68a2b21ec7b87da1ae1cc12b024876faf163e1d215b`。实际状态看 `logs/events.jsonl`、`logs/pilot-all-v2.log` 和进程，不能把排队视作训练完成。

V4 修复了抽出预测接口时遗留的局部变量引用；回归确认预测不需要 GT latent。本地和服务器的 50 项针对性回归通过，服务器 921 个源文件 SHA 校验通过，本实验脚本与共享源码编译通过。V3 尚未开始 GPU 工作，旧等待日志与源码均保留。

`scripts/run.py` 串行运行两臂 SFT、从同一 RN 检查点继续 SFT 10 updates 和 GRPO 10 updates。`scripts/stage.py` 复用共享 Trainer，保存阶段生成及逐图 SHA。`scripts/evaluate.py` 用 SDPose 预测姿态，在 qwen21 环境用 CPU 计算身份和像素指标。`scripts/rl.py` 复用共享 Native RL 的采样、Replay、Guarded-GDPO 和更新函数，保存随机转移、候选图、固定 SFT reference 和独立 RL 快照。四个计数器为 `optimizer_updates`、`rollout_batches`、`generated_images`、`replayed_transitions`，历史快照按原脚本和 SHA 复现。

区域 RL 的 Background 权重为 `0.15`，Body、Hands 与 ID 按原比例分享剩余 `0.85`。复用 Native 前景，或读取 `data/evaluation.json` 中带 SHA 的原图 `source_mask`。Source A 到输出使用恒等映射或同宽高比的内容框等比重采样，前景并集外扩 16 px，剩余支持至少占有效画布的 10%。在该区域计算 `exp(-Masked L1 / 0.05)`。前景不合格或画布无法对齐时 `background_reward` 写 `null` 并标注原因。`valid_mse` 保留作 Target GT 诊断指标。

自动执行完后，原始图保存在 `output/generation/<分组>/`，RL 候选图在 `output/grpo/rollouts/`。`report/evaluation/` 保存逐图指标；`report/comparison-ready.json` 指向最终指标表与逐对 7 列面板（Source、SDPose、GT、uniform、regional、继续 SFT、GRPO）。汇总前会逐阶段核对两臂的 route/source/index/sigma/LR 轨迹，必须全部一致。汇总脚本已用合成图片和真实格式的已提交 checkpoint 做过流程冒烟验证，该验证不代表训练收益。

查看实际进度（PowerShell）：

```powershell
ssh sg "tail -n 12 /data/shangguanrz/pic-edit-experiments/bilibili-region-sdpose-sft-rl-20261120-v1/logs/events.jsonl"
```

本地复测本次目标与预测接口：

```powershell
python -m pytest tests/test_native_region_backend.py tests/test_region_objective.py tests/test_region_supervision.py tests/test_region_rewards.py tests/test_native_region_rl.py tests/test_sdpose_backend_binding.py tests/test_bilibili_region_prepare.py tests/test_bilibili_region_rl_driver.py tests/test_prompt_bank_formats.py tests/test_native_v7_training.py -q
```

## 与已有实验的关系

复用 Bilibili 原视频划分、Native 几何与冻结模型。新 SDPose 2D 骨架进入新的 Qwen 条件版本；已有 SMPL-X/Sapiens 三维条件仍如实保留，SDPose 不伪造三维参数。原 LoRA Pose/ID GRPO 提供随机流、重放与固定 SFT reference 思路，本轮只优化 Native Adapter；原 LoRA 权重和环境不改动。
