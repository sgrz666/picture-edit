# Native V6.2 / V7 在线 RL

从当前架构已提交的 Native SFT checkpoint 开始训练。入口按配置 schema 装配 V6.2 或 V7 Backend，V7 运行时仍按权重记录使用逐人物几何、Contact 和身份调制。每组候选共享同一任务的 Source、Prompt、人物绑定和 Canvas。默认每轮抽 4 个任务，每组生成 8 张图，10 步采样中第 4 步使用 Flow-SDE。Replay 以实际保存的 BF16 状态和动作重新计算 Native 特征和转移概率，首次 Replay 的 `|ratio - 1|` 上限为 `1e-4`。每轮最多回放两次，第二次起 PPO clipping 可产生约束。

训练范围由 `ParameterCatalog` 和所选 route 决定，包括启用的 Body、PoseLink 输出头，以及启用的双人 Interaction 输出头。学习率默认 `1e-5`，梯度裁剪 `1.0`，KL 系数 `0.001`。参考策略共享冻结的 Qwen 与 Native 编码器，单独保留 SFT 输出头；以相同状态、同方差的转移分布计算 KL。Native ODE 和 RL 的 velocity 走同一 V6.2 绑定入口，沿用 Qwen 原生 scheduler、VAE 和画布。

## 奖励与对照

- 单人默认 Pose / ID 各 `0.5`。Pose 用 SDPose-Wholebody + YOLO11x，以 `geometry` 序映射 COCO17 计算 OKS，同时记录 PCK@0.02 和关键点召回。预先按置信度 `0.30` 在目标图建立可见点分母，候选漏点计零。ID 用 SCRFD + glintr100，Source 和目标都能绑定人脸时启用，候选漏脸计零。
- 双人按 Native 槽位和目标几何固定匹配，用 `0.30 / 0.25 / 0.20 / 0.15 / 0.10` 融合 Pose、ID、Binding、Layout、Contact。Pose 给较弱一人额外权重。Binding 比较正确与交叉的 ID 相似度。Contact 只用 V6.2 已确认的 Part14 接触对和可见 COCO 关节代理，采用画布对角线归一化的宽容距离。
- `--pose-only` 跑 Pose GRPO。`--aggregation sum` 先加权聚合全部适用的原始 Reward 再作组内标准化。`--aggregation gdpo` 使用 Flow-Factory v0.5.0 的逐项组归一化、权重和与采集批标准化。默认 `gdpo-guarded` 在标准化前抑制 std ≤ `0.001` 的奖励，并按当前适用的权重归一。整个批次无有效优势时跳过 optimizer。区域实验复用同一个 Flow-SDE、优势和更新内核。 [Flow-Factory v0.5.0](https://github.com/X-GenGroup/Flow-Factory/tree/ced2cd9b0465a2c9d733b9b8aed4d4351f950a22)

## 运行

从仓库根目录执行，`CFG` 指向 Native V6.2 或 V7 SFT JSON。它要绑定完整 SFT `initialization` 及 SHA256、原视频或序列分组、Qwen model manifest 和冻结 latent cache。一个 run 使用一种条件 profile，可选 `single_2ref`、`dual_4ref` 或 `dual_source_5ref`。多 route 时用 `--route` 指定要继承的条件开关。`--models` 指定 AntelopeV2 的模型父目录。SDPose 走统一权重和环境，`--pose-device` 默认 `cpu`，有独立显存预算时可以选 `cuda:0`。

```bash
PY=/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python
CFG=/absolute/path/to/native-v7-sft.json
RUN=/data/shangguanrz/pic-edit-experiments/native-rl-$(date -u +%Y%m%dT%H%M%SZ)

"$PY" scripts/train/train_qwen21_native_v62_rl.py check-data --config "$CFG" --groups-per-update 1
"$PY" scripts/train/train_qwen21_native_v62_rl.py preflight --config "$CFG" \
  --groups-per-update 1 --group-size 2 --steps 3 --sde-step 1 --val-limit 1 --output "$RUN/preflight"
"$PY" scripts/train/train_qwen21_native_v62_rl.py train --config "$CFG" \
  --groups-per-update 1 --group-size 2 --steps 3 --sde-step 1 --updates 1 --output "$RUN/pilot"
```

正式训练需要 `experiment_mode=standard` 和按视频或序列隔离的 validation。启动时打印 SFT 路径、SHA256、阶段和父链。先用固定 seed 生成 SFT 的 ODE 基线，再按 `--validate-every` 个 optimizer updates 和结尾复测。Pose 或 ID 相对基线下降超过默认 `0.02` 时淘汰该候选，连续两次触发会停止。`summary.json` 中的 `selected` 指向门禁内评分最高且超过基线的 RL 权重，或 SFT 基线。`update-*.commit.json` 记录 RL 权重和 SHA256，可用相应架构的 loader 推理。通用入口当前从 SFT 启动新 run，区域入口提供其自身训练状态的断点恢复。

`--updates`（也可写 `--optimizer-updates`）按实际 `optimizer.step()` 计数，旧版命令的预算需按新语义重算。`rollout_batches` 包括零优势批，`generated_images` 统计训练候选，`replayed_transitions` 统计参与成功更新的转移。`--max-rollout-batches` 默认是更新预算的 5 倍，用于限制无有效优势时的生成成本。

`run.json` 记录模型、数据、人物、评估器 SHA 和固定资格。`rollout-*/rollout.json` 记录候选分数、PCK、ID 缺失、Swap、接触适用率、优势和样本图。`rollout-*/replay.json` 有 ratio、clip fraction、KL、梯度范数和显存。需要离线审计单步状态时加 `--keep-replay`。固定 ODE 图和逐样本指标在 `ode-*/` 与对应 JSON。

## 自动验证

```bash
NATIVE_RL_TEST_DEVICE=cuda "$PY" -m pytest -q tests/test_native_v62_rl.py
```

该测试使用小型真实 Qwen21 transformer 和 scheduler、V6.2/V7 注入和轻量 decoder，检查 ODE 一致性、BF16 Replay、冻结参数与 KL。真实 Bilibili 和 Hi4D 的效果对照需要分别使用各自的 SFT 基线和独立的留出集。
