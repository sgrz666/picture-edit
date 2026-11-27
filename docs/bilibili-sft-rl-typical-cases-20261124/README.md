# Bilibili 单人姿态迁移 SFT→RL 典型生成样本（2026-11-24）

## 目的

从已完成实验「Bilibili Pose / ID GRPO」的 50 对固定验证集里挑出可直接贴进文档的典型样本，做成「SFT 9520（RL 前）→ RL 200（RL 后）」的并排对比图。回答一个问题：**追加 300 步 RL 后，生成图到底哪里变了。**

## 里面有什么

- `fig1_typical.png` —— 4 个典型收益样本（#37 / #31 / #19 / #05），五列：A 源图｜B 目标骨架｜B 真值｜SFT 9520｜RL 200，带逐样本指标与一行变化说明
- `fig2_boundary.png` —— 2 个边界样本（#20 取舍、#36 唯一反例），版式同上，用来说明平均收益的适用边界
- `fig3_checkpoint_curves.png` —— 原报告的四联曲线（Pose Reward / ID Reward / PCK@0.02 / NME 随 update 变化），未修改
- `case_metrics.json` —— 入选 6 个样本的完整指标（Pose / PCK@0.02 / PCK@0.05 / glintr / buffalo / NME）
- `per_pair_metrics_50.csv` —— 原报告 50 对逐样本指标，供换样本重做
- `scripts/pack_cases.py` —— 远端按 case_no 打包图片与指标（服务器执行）
- `scripts/compose_k5.py` —— 本地拼图（PIL，5 列 × N 行）

## 关键结论

**整体（50 对固定验证集，RL 200 vs SFT 9520）**

| 指标 | SFT 9520 | RL 200 | 增量 |
|---|---:|---:|---:|
| Pose 连续 Reward | 0.7349 | 0.7722 | +0.0373（+5.07%） |
| PCK@0.02 | 91.13% | 94.48% | +3.35pp |
| glintr100 cosine（33 例） | 0.6127 | 0.6626 | +0.0499 |
| NME（匹配关节） | 0.00852 | 0.00671 | −21.25% |

**入选样本（都能在图上直接看出改动方向）**

| # | clip | 类型 | Pose | PCK@0.02 | ID cos | 肉眼看到的变化 |
|---|---|---|---:|---:|---:|---|
| 37 | BV1XssX6AEsR | 双升 | 0.473→0.589（+0.116） | 64.7→76.5（+11.8pp） | 0.516→0.639（+0.123） | 抬腿偏低/脚尖下垂 → 膝盖抬高、小腿回收 |
| 31 | BV1XssX6AEsR | 双升 | 0.571→0.760（+0.190） | 70.6→94.1（+23.5pp） | 0.575→0.600（+0.025） | 双脚落地、双臂外张 → 真值的单腿交叉站立 |
| 19 | BV1Aynt6zEQd | 双升 | 0.631→0.667（+0.036） | 80.0→86.7（+6.7pp） | 0.783→0.844（+0.060） | 双手都抬到脑后 → 只抬右手、左手放下 |
| 05 | BV19vxY6mE8j | 仅姿态可评 | 0.452→0.571（+0.119） | 47.1→82.4（+35.3pp） | 不适用 | 手没到脸 → 双手收到面前；下半身仍未跪下 |
| 20 | BV1Aynt6zEQd | 取舍 | 0.896→0.823（−0.074） | 100→100（0） | 0.788→0.872（+0.084） | 手臂略外张、Pose 降，但脸更像 A |
| 36 | BV1XssX6AEsR | 反例 | 0.745→0.591（−0.154） | 94.1→82.4（−11.8pp） | 0.528→0.516（−0.012） | 左手从贴脸落到胸前 |

**不要过度解读**

- 50 对里 Pose 38 升 12 降，ID 30 升 3 降；#36 是唯一两项都降的反例，说明均值提升不排除个别退化。
- 提升集中在局部肢体与手部位置；衣袖、披肩、手指仍有问题，分数涨不等于整图都更忠实于 A（#05 的下半身就是例子）。
- 只有 5 个 Pose 视频 / 4 个 ID 视频，50/33 张图不是 50/33 个独立实验；RL200 还是在同一个 val 上从 7 个 checkpoint 里选出来的，存在选点偏差。
- 本实验没有等算力的「继续 SFT」对照，只能讲「相对现有 SFT 基线的观测收益」，不能讲 RL 训练范式更优。

## 与其他实验的关系

- 上游：`/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora` 的 `output/train_full_20261111_v1/checkpoint-009520`（SFT 9520 步 LoRA）
- 本实验：`/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo`（300 次 optimizer update，Pose:ID Reward = 1:1，固定 50 对 val）
- 原始报告：`.../bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1/`（`report.html`、`panels_200/`、`analysis.json`）
- 与 Native V7 系（`experiments/hi4d_qwen21_dual/`）无关，本实验只训 LoRA，不加载 Native Adapter，背景不进 Reward

## 复现

```bash
# 1) 远端按 case_no 打包（脚本里的 CASES 可改）
scp scripts/pack_cases.py sg:/tmp/pack_k5.py && ssh sg "python3 /tmp/pack_k5.py"
scp sg:/tmp/k5_cases.tar.gz . && tar xzf k5_cases.tar.gz
scp sg:/tmp/k5_cases/meta.json k5_cases/meta.json

# 2) 本地拼图
python scripts/compose_k5.py
```
