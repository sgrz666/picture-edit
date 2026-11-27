# SpatialReward-8B 实验整理（2026-11-24）

## 1. 这份材料回答什么

把仓库里与 **SpatialReward-8B** 有关的代码、验证、计划和旁证实验收在一处，分清三件事：

1. **已经跑出数字的**：只有离线冒烟校准（1 对图 / 2 候选）；
2. **已经写好但还没开跑的**：固定权重消融（0.10 / 0.25 / 0.40）；
3. **容易被误当成"空间奖励收益"的**：区域目标 SFT、Identity Reward、beach14 复合打分——它们是旁证，不是 SpatialReward 的训练收益。

配套图：

- [`fig1-scoring.png`](fig1-scoring.png) —— 评分说明图（显式编辑区域定位 vs 无定位整体打分），bbox / reasoning / 子分均为模型真实输出；
- [`fig2-gains.png`](fig2-gains.png) —— 收益整理图（区域目标 SFT 实测增量 + SpatialReward 冒烟结果 + 实验状态表）。

## 2. 里面有什么

```
docs/spatial-reward-20261124/
├── README.md              # 本文
├── fig1-scoring.html/.png # 评分说明图（1600×1260）
├── fig2-gains.html/.png   # 收益整理图（1600×1000）
├── index.html             # 两图并排预览
└── assets/                # 真实样本图：Source A、initial、regional_S3
```

相关代码与入口（不在本目录，列出来方便追）：

| 位置 | 作用 |
|---|---|
| `src/pose_control/qwen21_full/native_rl/spatial_reward.py` | SC / PQ prompt、融合模式、解析与 `spatial_target()` 条件构建 |
| `src/pose_control/qwen21_full/native_rl/spatial_worker.py` | 持久隔离子进程；按 rollout group 交候选与 SHA，整组返回 |
| `scripts/eval/eval_native_spatialreward.py` | 离线校准入口（manifest → 逐候选子分） |
| `src/pose_control/qwen21_full/native_rl/README.md` §SpatialReward-8B | 协议、模式、权重与门禁的权威说明 |
| 模型 | `sg:/data/shangguanrz/models/rl/SpatialReward-8B`，上游代码 `0222484b` |

## 3. 关键结论

### 3.1 接入（已完成）

- 提交：`05dca81`（隔离子进程评分）、`dd28549`（固定消融参数 + Hands42 + PQ 命名字段解析 + AdaFace）。
- 评分协议：**SC** 走「有序参考 + 指令 + 候选」，要求模型**先输出 `edit_region`（bbox_2d，0–1000）再打分**，返回 IF / CON；**PQ** 只看候选，返回 NAT / ART；四项各 0–25。
- 融合：`visual-linear`（默认）= `0.40·CON/25 + 0.30·NAT/25 + 0.30·ART/25`，IF 仅诊断；`official` = `SC^0.8 × PQ^0.2`，`SC=(0.6·IF+0.4·CON)/25`；`complementary` 用 `CON/25` 替换 SC。
- 训练接入：默认权重 `0.25`，其余奖励同比缩放；Guarded-GDPO 的临时方差门 `max(0.005, --std-floor)`；整组评分齐全才计算优势，解析失败/超时/ID 异常写 `evaluation-failed.json` 并终止该批。
- 模型身份：16 个文件 SHA 固定，score_version 由 `spatialreward_native_v1` 升到 `v2`（PQ 命名字段）。

### 3.2 冒烟实测（唯一真实评分数据）

样本：小红书 `pair_039_6a61e37b`，seed42，同条件两候选 `initial` / `regional_S3`；CPU bfloat16，`max_pixels=65536`，`max_new_tokens=512`。

| 参考模式 | 候选 | IF | CON | NAT | ART | reward（official） |
|---|---|---|---|---|---|---|
| rgb | initial | 25 | 25 | 18 | 15 | **0.9203** |
| rgb | regional_S3 | 22 | 23 | 18 | 15 | **0.8429** |
| guides | 两者 | 0 | 0 | 18 | 15 | **0.0000** |

- **rgb 可用**：排序方向与"初始图更贴源图"一致；模型给出的 bbox 为 `[356,308,788,998]`（initial）与 `[333,308,777,998]`（regional_S3），reasoning 明确引用 `<|bbox_1|>` / `<|global|>`。
- **guides 失效**：骨架图作为参考时两候选 IF=CON=0、SC=0、reward=0，模型不认骨架输入。`--spatial-references guides` 目前不可用于训练。
- **batch 敏感**：同一对图 `batch=1` 复跑得到 CON 23/23（reward 0.897 / 0.843），与 `batch=2` 的 25/23（0.920 / 0.843）有差异；`regional_S3` 稳定，`initial` 的 CON 抖 2 分。
- **PQ 解析 bug**：模型返回 `{"naturalness":18,"artifacts":15}`（无 `score` 字段），旧解析抛 `ValueError: requires two finite numeric scores in [0,25]`；`dd28549` 已按命名字段兼容。
- **样本量**：n=1 对、2 候选。只能证明"可运行 + 边界"，不能证明任何训练收益。

### 3.3 尚未开跑：固定权重消融

- 计划：固定 `rgb + visual-linear`，三组只改 `--spatial-weight 0.10 / 0.25 / 0.40`；锁定同一 SFT 父 SHA、奖励模型、阈值、rollout 配置、seed、更新预算、候选上限与留出集；各自使用新输出目录。
- 报告项：有效更新数、实际生成图数、ODE 门禁、固定留出图的独立人工盲评；另需同批图对照不同 `--batch-size`，记录 `repeat_reward_std` 并与人工配对排序对齐。
- 现状：小红书 GRPO 于 2026-11-24 22:01 启动、22:12 报 `torch.OutOfMemoryError`（GPU 被 beach arms 7/8/9 占用），且该命令**未带** `--spatial-model`，因此 SpatialReward 还没进入任何训练循环。

### 3.4 旁证：区域目标 SFT 的实测增量

小红书 expanded run，val 40 对，seed42，20-step ODE，SDPose-Wholebody + YOLO11x（阈值 0.30）：

| 臂 | Body OKS | PCK@0.02 | Hands | ID cos | BG reward |
|---|---|---|---|---|---|
| initial | 0.8114 | 83.61% | 0.7901 | 0.2274 | 0.1157 |
| uniform_S3 | 0.8487 | 86.84% | 0.8313 | 0.2306 | 0.1055 |
| **regional_S3** | **0.8643** | **89.19%** | **0.8330** | **0.2347** | 0.1060 |
| continue_sft | 0.8599 | 88.62% | 0.8289 | 0.2307 | 0.1079 |

- initial → regional_S3：Body **+0.0529**、PCK **+5.58 pp**、Hands **+0.0429**、ID **+0.0073**；背景奖励下降 0.0097（口径为 Source A 前景外 L1）。
- regional 四项全面优于 uniform；continue_sft 相对 regional_S3 轻微回退（Body −0.0044、PCK −0.57 pp）。
- **口径提醒**：这是"空间加权监督"的收益，不是 SpatialReward-8B 作为 RL 奖励的收益。两者不能合并宣称。

### 3.5 其他旁证（不同方法，勿混）

- `experiments/face-identity-reward-20261123`：AdaFace-KPRPE AUC **0.9906** vs glintr100 0.9830（+0.0076，2948 对 tier-1 clean）。
- `experiments/beach14-rl-reward-scoring-20261122`：beach14 四臂 × 14 模板复合分，#4 with_b+skel+Depth **0.7014** 居首（12/14 可打分）。

## 4. 复现冒烟评分

```bash
PY="$HOME/nihaochen/envs/qwen21/bin/python"
ROOT=/data/shangguanrz/code-snapshots/native-rl-spatialreward-20261124T091945Z
"$PY" "$ROOT/scripts/eval/eval_native_spatialreward.py" \
  --model /data/shangguanrz/models/rl/SpatialReward-8B \
  --manifest "$ROOT/smoke-manifest.json" --references rgb --mode official \
  --batch-size 2 --output "$ROOT/release/recheck-rgb.json"
```

输出目录已存在会拒绝覆盖；换 `--references guides`、`--batch-size 1` 或 `--mode visual-linear` 即得到对应口径。

## 5. 与其他实验的关系

- **上游**：`src/pose_control/qwen21_full/native_rl/`（通用 Native RL 内核）与 `experiments/xiaohongshu-region-sdpose-sft-rl/`（区域入口，`rl.py` / `run.py` / `pipeline.py` 接受同一组 `--spatial-*` 参数）。
- **下游**：小红书 expanded run 的 GRPO 50 updates（父权重锁定 `regional_S3` `630b71f8…`）是第一个候选接入点；接之前必须先解决 guides 失效与 batch 方差，否则奖励噪声会直接进入优势。
- **独立线**：`face-identity-reward-20261123`（ID 后端）与 `beach14-rl-reward-scoring-20261122`（复合打分基线）为奖励系统的其他部件，与本模型并列而非从属。
